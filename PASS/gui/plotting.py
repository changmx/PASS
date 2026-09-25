"""Numeric result preview with bounded display work and full-resolution data."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QThread, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPainterPath, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton, QSpinBox, QStackedWidget, QVBoxLayout, QWidget


def _line_indices(x, y, valid, width, x_limits, monotonic):
    """Keep extrema in source order; missing samples still break each line."""
    start, end = 0, len(x)
    if monotonic and end:
        start = max(0, int(np.searchsorted(x, x_limits[0], side="left")) - 1)
        end = min(end, int(np.searchsorted(x, x_limits[1], side="right")) + 1)
    n = end - start
    if n <= max(1, width) * 4:
        return np.flatnonzero(valid[start:end]) + start
    # Contiguous bins retain temporal order, unlike sorting a waveform by X.
    block = max(1, int(np.ceil(n / max(1, width))))
    bins = int(np.ceil(n / block))
    padded_y = np.full(bins * block, np.nan)
    padded_y[:n] = np.where(valid[start:end], y[start:end], np.nan)
    rows = padded_y.reshape(bins, block)
    row_valid = np.isfinite(rows)
    active = np.flatnonzero(row_valid.any(axis=1))
    if not len(active):
        return np.empty(0, dtype=np.intp)
    selected = rows[active]
    good = row_valid[active]
    offsets = [
        good.argmax(axis=1), block - 1 - good[:, ::-1].argmax(axis=1),
        np.where(good, selected, np.inf).argmin(axis=1),
        np.where(good, selected, -np.inf).argmax(axis=1)
    ]
    if not monotonic:
        padded_x = np.full(bins * block, np.nan)
        padded_x[:n] = x[start:end]
        selected_x = padded_x.reshape(bins, block)[active]
        offsets.extend((np.where(good, selected_x, np.inf).argmin(axis=1), np.where(good, selected_x, -np.inf).argmax(axis=1)))
    return np.unique(np.concatenate([start + active * block + offset for offset in offsets]))


class PlotCanvas(QWidget):
    """Qt plot using arrays, extrema reduction, and cached batched painting."""

    def __init__(self) -> None:
        super().__init__()
        self._x_values = np.empty(0)
        self._values = np.empty(0)
        self._valid = np.empty(0, dtype=bool)
        self._missing_prefix = np.zeros(1, dtype=np.int64)
        self._monotonic = False
        self._x_limits = self._y_limits = None
        self._geometry_key = None
        self._geometry = None
        self.mode = "line"
        self.empty_message = "选择 X / Y 数值列以绘图"
        self.x_label = self.y_label = ""
        self.displayed_points = 0
        # Leave room for controls and reference metadata in the 1000x650 window.
        # A larger child minimum can extend beyond QStackedWidget and hide X labels.
        self.setMinimumHeight(220)

    @property
    def x_values(self):
        """List compatibility for callers inspecting the displayed series."""
        return self._x_values.tolist()

    @property
    def values(self):
        return self._values.tolist()

    def set_series(self, x_values, values, *, mode="line", empty_message=None, x_label="", y_label="") -> None:
        self._x_values = np.asarray(x_values, dtype=float)
        self._values = np.asarray(values, dtype=float)
        if self._x_values.ndim != 1 or self._values.ndim != 1 or len(self._x_values) != len(self._values):
            raise ValueError("X / Y 数据必须是一维且长度一致。")
        self._valid = np.isfinite(self._x_values) & np.isfinite(self._values)
        self._missing_prefix = np.concatenate(([0], np.cumsum(~self._valid)))
        self._monotonic = bool(np.isfinite(self._x_values).all() and np.all(self._x_values[1:] >= self._x_values[:-1]))
        self.mode = mode
        self.x_label, self.y_label = x_label, y_label
        self.empty_message = empty_message or "当前 X / Y 列没有可绘制的有限值"
        self._geometry_key = None
        self._geometry = None
        self.fit_view()

    def fit_view(self) -> None:
        if self._valid.any():
            x = self._x_values[self._valid]
            y = self._values[self._valid]
            self._x_limits = (float(x.min()), float(x.max()))
            self._y_limits = (float(y.min()), float(y.max()))
        else:
            self._x_limits = self._y_limits = None
        self.update()

    def set_values(self, values) -> None:
        self.set_series(np.arange(len(values)), values)

    def _plot_area(self):
        return self.rect().adjusted(68, 30, -24, -48)

    @staticmethod
    def _expanded_range(low, high):
        if low == high:
            padding = abs(low) * 0.05 or 1.0
            return low - padding, high + padding
        return low, high

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._x_limits is None or self._y_limits is None:
            event.ignore()
            return
        area = self._plot_area()
        if not area.contains(event.position().toPoint()):
            event.ignore()
            return
        factor = 0.8 if event.angleDelta().y() > 0 else 1.25
        x_ratio = (event.position().x() - area.left()) / max(area.width(), 1)
        y_ratio = 1.0 - (event.position().y() - area.top()) / max(area.height(), 1)
        x_low, x_high = self._expanded_range(*self._x_limits)
        y_low, y_high = self._expanded_range(*self._y_limits)
        x_anchor = x_low + (x_high - x_low) * x_ratio
        y_anchor = y_low + (y_high - y_low) * y_ratio
        self._x_limits = (x_anchor - (x_anchor - x_low) * factor, x_anchor + (x_high - x_anchor) * factor)
        self._y_limits = (y_anchor - (y_anchor - y_low) * factor, y_anchor + (y_high - y_anchor) * factor)
        self.update()
        event.accept()

    def _prepare_geometry(self, area):
        x_low, x_high = self._expanded_range(*self._x_limits)
        y_low, y_high = self._expanded_range(*self._y_limits)
        color = self.palette().color(QPalette.Link)
        key = (area.width(), area.height(), x_low, x_high, y_low, y_high, self.mode, color.rgba())
        if key == self._geometry_key:
            return self._geometry
        width, height = max(1, area.width()), max(1, area.height())
        if self.mode == "scatter":
            visible = self._valid & (self._x_values >= x_low) & (self._x_values <= x_high)
            visible &= (self._values >= y_low) & (self._values <= y_high)
            x = np.rint((self._x_values[visible] - x_low) / (x_high - x_low) * (width - 1)).astype(np.intp)
            y = np.rint((y_high - self._values[visible]) / (y_high - y_low) * (height - 1)).astype(np.intp)
            pixels = np.zeros((height, width, 4), dtype=np.uint8)
            # Pixel occupancy retains every visible particle without connecting
            # unordered rows or constructing a Python object for each particle.
            pixels[y, x] = color.getRgb()
            pixels[np.minimum(y + 1, height - 1), x] = color.getRgb()
            pixels[y, np.minimum(x + 1, width - 1)] = color.getRgb()
            self.displayed_points = int(visible.sum())
            geometry = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format_RGBA8888).copy()
        else:
            indices = _line_indices(self._x_values, self._values, self._valid, width, (x_low, x_high), self._monotonic)
            x = (self._x_values[indices] - x_low) / (x_high - x_low) * width
            y = (y_high - self._values[indices]) / (y_high - y_low) * height
            breaks = np.ones(len(indices), dtype=bool)
            if len(indices) > 1:
                breaks[1:] = self._missing_prefix[indices[1:]] != self._missing_prefix[indices[:-1] + 1]
            path = QPainterPath()
            singletons = QPolygonF()
            for offset, (px, py) in enumerate(zip(x, y)):
                point = QPointF(float(np.clip(px, -1e6, 1e6)), float(np.clip(py, -1e6, 1e6)))
                if breaks[offset]:
                    path.moveTo(point)
                    if offset == len(indices) - 1 or breaks[offset + 1]:
                        singletons.append(point)
                else:
                    path.lineTo(point)
            self.displayed_points = len(indices)
            geometry = path, singletons
        self._geometry_key, self._geometry = key, geometry
        return geometry

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(QPalette.Base))
        painter.setRenderHint(QPainter.Antialiasing)
        area = self._plot_area()
        painter.setPen(QPen(self.palette().color(QPalette.Mid), 1))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = area.bottom() - fraction * area.height()
            painter.drawLine(area.left(), int(y), area.right(), int(y))
        if self._x_limits is None or self._y_limits is None:
            painter.setPen(self.palette().color(QPalette.PlaceholderText))
            painter.drawText(area, Qt.AlignCenter, self.empty_message)
            return
        geometry = self._prepare_geometry(area)
        painter.save()
        painter.setClipRect(area)
        painter.translate(area.left(), area.top())
        painter.setPen(QPen(self.palette().color(QPalette.Link), 1.5))
        if self.mode == "scatter":
            painter.drawImage(QRectF(0, 0, area.width(), area.height()), geometry)
        else:
            path, singletons = geometry
            painter.drawPath(path)
            painter.setPen(QPen(self.palette().color(QPalette.Link), 5, Qt.SolidLine, Qt.RoundCap))
            painter.drawPoints(singletons)
        painter.restore()
        x_low, x_high = self._expanded_range(*self._x_limits)
        low, high = self._expanded_range(*self._y_limits)
        painter.setPen(self.palette().color(QPalette.PlaceholderText))
        painter.drawText(5, area.top() + 5, f"{high:.5g}")
        painter.drawText(5, area.bottom(), f"{low:.5g}")
        painter.drawText(area.left(), self.height() - 24, f"{x_low:.5g}")
        painter.drawText(area.right() - 65, self.height() - 24, f"{x_high:.5g}")
        painter.drawText(area.left(), 19, self.y_label)
        painter.drawText(area.left(), self.height() - 6, self.x_label)
        if self.mode == "line" and self.displayed_points < int(self._valid.sum()):
            painter.drawText(area.adjusted(0, 3, -3, 0), Qt.AlignRight | Qt.AlignTop, "按视图保留极值显示 · 原始数据完整保留")


class _ResultReader(QThread):
    progress = Signal(int, int, int, str)
    completed = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, token, paths):
        super().__init__()
        self.token, self.paths = token, paths

    def run(self):
        from PASS.gui.plot_analysis import read_plot_result

        loaded = {}
        try:
            for index, path in enumerate(self.paths):
                if self.isInterruptionRequested():
                    return
                loaded[path] = read_plot_result(path,
                                                cancelled=self.isInterruptionRequested,
                                                progress=lambda message, i=index: self.progress.emit(self.token, i, len(self.paths), message))
            if not self.isInterruptionRequested():
                self.completed.emit(self.token, loaded)
        except InterruptedError:
            pass
        except Exception as exc:
            self.failed.emit(self.token, str(exc))


class PlotPage(QWidget):
    shutdown_finished = Signal()
    load_finished = Signal()
    load_failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.columns = {}
        self.result_files = {}
        self._active_path = None
        self._preferences = {}
        self._reader = None
        self._export_dialog = None
        self._pending_load = None
        self._load_token = 0
        self._closing = False
        self._analysis_canvas = None
        self._analysis_toolbar = None
        self._figure_spec = None
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        header = QHBoxLayout()
        header.addWidget(QLabel("绘图"))
        header.addStretch()
        load = QPushButton("加载 CSV / TFS / HDF5")
        load.clicked.connect(self.load_data)
        header.addWidget(load)
        self.unload_button = QPushButton("关闭当前数据")
        self.unload_button.clicked.connect(self.unload_current)
        self.unload_button.setEnabled(False)
        header.addWidget(self.unload_button)
        self.export_data_button = QPushButton("导出数据")
        self.export_data_button.clicked.connect(self._choose_data_export)
        header.addWidget(self.export_data_button)
        self.export_image_button = QPushButton("导出图像")
        self.export_image_button.clicked.connect(self._choose_image_export)
        header.addWidget(self.export_image_button)
        root.addLayout(header)
        loading = QHBoxLayout()
        self.load_status = QLabel("")
        self.load_status.setWordWrap(True)
        loading.addWidget(self.load_status, 1)
        self.load_progress = QProgressBar()
        self.load_progress.setMaximumWidth(140)
        self.load_progress.hide()
        loading.addWidget(self.load_progress)
        self.cancel_load_button = QPushButton("取消读取")
        self.cancel_load_button.clicked.connect(self.cancel_load)
        self.cancel_load_button.hide()
        loading.addWidget(self.cancel_load_button)
        root.addLayout(loading)
        self.result_selector = QComboBox()
        self.result_selector.currentIndexChanged.connect(self._select_result)
        root.addWidget(self.result_selector)
        self.canvas = PlotCanvas()
        self.table_controls = QWidget()
        table_layout = QVBoxLayout(self.table_controls)
        table_layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("X 列"))
        self.x_column_box = QComboBox()
        self.x_column_box.currentTextChanged.connect(self._select_series)
        controls.addWidget(self.x_column_box, 1)
        controls.addWidget(QLabel("Y 列"))
        self.column_box = QComboBox()
        self.column_box.currentTextChanged.connect(self._select_series)
        controls.addWidget(self.column_box, 1)
        self.mode_box = QComboBox()
        for title, value in (("自动", "auto"), ("折线", "line"), ("散点", "scatter"), ("密度与投影", "density")):
            self.mode_box.addItem(title, value)
        self.mode_box.setToolTip("自动模式：带 tag 或 particle_id 的粒子快照使用散点，其余数据使用折线。")
        self.mode_box.currentIndexChanged.connect(self._select_series)
        controls.addWidget(self.mode_box)
        fit = QPushButton("适配视图")
        fit.setToolTip("恢复到当前数据的完整范围")
        fit.clicked.connect(self.fit_view)
        controls.addWidget(fit)
        table_layout.addLayout(controls)
        filters = QHBoxLayout()
        self.status_filter = QComboBox()
        for title, value in (("全部粒子", "all"), ("存活粒子", "alive"), ("已损失粒子", "lost")):
            self.status_filter.addItem(title, value)
        self.batch_filter = QComboBox()
        self.batch_filter.addItem("全部注入批次", None)
        for field in (self.status_filter, self.batch_filter):
            field.currentIndexChanged.connect(self._select_series)
            filters.addWidget(field)
        self.view_hint = QLabel("滚轮缩放，适配视图恢复完整范围")
        filters.addWidget(self.view_hint)
        filters.addStretch()
        table_layout.addLayout(filters)
        analysis = QHBoxLayout()
        self.preset_box = QComboBox()
        for title, values in (("相空间预设", None), ("水平 x–px", ("x", "px")), ("垂直 y–py", ("y", "py")), ("纵向 z–dp", ("z", "dp")), ("横截面 x–y", ("x", "y")),
                              ("到达时间–dp", ("arrival_time_s", "dp"))):
            self.preset_box.addItem(title, values)
        self.preset_box.currentIndexChanged.connect(self._apply_preset)
        analysis.addWidget(self.preset_box)
        self.compare_mode = QComboBox()
        for title, value in (("单文件", "single"), ("与基线叠加", "overlay"), ("差值 A−B", "difference"), ("相对差 (A−B)/B", "relative")):
            self.compare_mode.addItem(title, value)
        self.compare_mode.currentIndexChanged.connect(self._select_series)
        analysis.addWidget(self.compare_mode)
        self.baseline_box = QComboBox()
        self.baseline_box.setToolTip("基线 B；比较要求 X 和行顺序完全一致，不进行插值。")
        self.baseline_box.currentIndexChanged.connect(self._select_series)
        analysis.addWidget(self.baseline_box, 1)
        analysis.addWidget(QLabel("密度分箱"))
        self.density_bins = QSpinBox()
        self.density_bins.setRange(8, 256)
        self.density_bins.setValue(80)
        self.density_bins.valueChanged.connect(self._select_series)
        analysis.addWidget(self.density_bins)
        table_layout.addLayout(analysis)
        root.addWidget(self.table_controls)
        self.field_controls = QWidget()
        field_layout = QHBoxLayout(self.field_controls)
        field_layout.setContentsMargins(0, 0, 0, 0)
        self.field_box, self.slice_box = QComboBox(), QComboBox()
        self.average_field = QCheckBox("除以 Δz：显示切片平均量")
        self.aperture_overlay = QCheckBox("显示孔径")
        self.aperture_overlay.setChecked(True)
        for item in (QLabel("SC 场"), self.field_box, QLabel("切片"), self.slice_box, self.average_field, self.aperture_overlay):
            field_layout.addWidget(item)
        for field in (self.field_box, self.slice_box):
            field.currentIndexChanged.connect(self._select_series)
        for field in (self.average_field, self.aperture_overlay):
            field.toggled.connect(self._select_series)
        self.field_controls.hide()
        root.addWidget(self.field_controls)
        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.canvas)
        self.analysis_view = QWidget()
        self.analysis_layout = QVBoxLayout(self.analysis_view)
        self.analysis_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_stack.addWidget(self.analysis_view)
        root.addWidget(self.plot_stack, 1)
        self.info = QLabel("未加载数据")
        self.info.setObjectName("muted")
        self.info.setWordWrap(True)
        root.addWidget(self.info)
        QApplication.instance().aboutToQuit.connect(self._wait_reader)

    def load_data(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "加载结果文件", "", "Data files (*.csv *.tfs *.h5 *.hdf5);;All files (*)")
        if not paths:
            return
        self.load_paths_async(paths)

    def load_paths(self, paths):
        from PASS.gui.plot_analysis import read_plot_result

        paths = list(paths)
        if not paths:
            return
        self.cancel_load()
        loaded = {str(path): read_plot_result(path) for path in paths}
        self.result_files.update(loaded)
        self._refresh_selector(str(paths[0]))

    def load_paths_async(self, paths):
        paths = [str(path) for path in paths]
        if not paths or self._closing:
            return
        self._load_token += 1
        self._pending_load = self._load_token, paths
        self.load_status.setText("准备读取…")
        self.load_progress.show()
        self.load_progress.setRange(0, len(paths))
        self.load_progress.setValue(0)
        self.cancel_load_button.show()
        if self._reader is not None:
            self._reader.requestInterruption()
            self.load_status.setText("等待上一次读取释放资源…")
        else:
            self._start_reader()

    def _start_reader(self):
        if self._pending_load is None or self._closing:
            return
        token, paths = self._pending_load
        self._pending_load = None
        reader = self._reader = _ResultReader(token, paths)
        reader.progress.connect(self._read_progress)
        reader.completed.connect(self._read_complete)
        reader.failed.connect(self._read_failed)
        reader.finished.connect(self._reader_finished)
        reader.start()

    def _read_progress(self, token, current, total, message):
        if token == self._load_token and not self._closing:
            self.load_status.setText(message)
            self.load_progress.setRange(0, total)
            self.load_progress.setValue(current)

    def _read_complete(self, token, loaded):
        if token != self._load_token or self._closing:
            return
        self.result_files.update(loaded)
        self._refresh_selector(next(iter(loaded)))
        self.load_status.setText(f"已加载 {len(loaded)} 份数据")
        self.load_finished.emit()

    def _read_failed(self, token, message):
        if token == self._load_token and not self._closing:
            self.load_status.setText("读取失败：" + message)
            self.load_failed.emit(message)

    def _reader_finished(self):
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.deleteLater()
        if self._closing:
            if not self.busy:
                self.shutdown_finished.emit()
        elif self._pending_load is not None:
            self._start_reader()
        else:
            self.load_progress.hide()
            self.cancel_load_button.hide()
            if self.load_status.text().startswith("正在取消"):
                self.load_status.setText("读取已取消")

    def cancel_load(self):
        self._load_token += 1
        self._pending_load = None
        if self._reader is not None:
            self._reader.requestInterruption()
            self.load_status.setText("正在取消；当前文件读取结束后释放资源…")

    def shutdown(self):
        self._closing = True
        self.cancel_load()
        if self._export_dialog is not None:
            self._export_dialog.cancel()
        return not self.busy

    @property
    def busy(self):
        # The dialog remains owned until its nested event loop has returned.
        return self._reader is not None or self._export_dialog is not None

    def _wait_reader(self):
        if self._reader is not None:
            self._reader.requestInterruption()
            self._reader.wait()
        if self._export_dialog is not None:
            self._export_dialog.worker.context.cancelled.set()
            self._export_dialog.worker.wait()

    def _refresh_selector(self, selected=None):
        from PASS.gui.plot_analysis import FieldResult

        baseline = self.baseline_box.currentData()
        self.result_selector.blockSignals(True)
        self.result_selector.clear()
        for path, result in self.result_files.items():
            identity = " · ".join(f"{k}={result.metadata[k]}" for k in ("BeamId", "BunchId", "Turn") if k in result.metadata)
            self.result_selector.addItem(Path(path).name + (" · " + identity if identity else ""), path)
        index = self.result_selector.findData(selected)
        self.result_selector.setCurrentIndex(max(0, index) if self.result_files else -1)
        self.result_selector.blockSignals(False)
        self.baseline_box.blockSignals(True)
        self.baseline_box.clear()
        for path, result in self.result_files.items():
            if not isinstance(result, FieldResult):
                self.baseline_box.addItem(Path(path).name, path)
        index = self.baseline_box.findData(baseline)
        if index < 0:
            index = next((i for i in range(self.baseline_box.count()) if self.baseline_box.itemData(i) != selected), 0)
        self.baseline_box.setCurrentIndex(index)
        self.baseline_box.blockSignals(False)
        self.unload_button.setEnabled(bool(self.result_files))
        self._select_result()

    def unload_current(self):
        path = self.result_selector.currentData()
        self.result_files.pop(path, None)
        self._preferences.pop(path, None)
        self._active_path = None
        self._refresh_selector()

    def _save_preferences(self):
        if self._active_path in self.result_files:
            self._preferences[self._active_path] = (self.x_column_box.currentText(), self.column_box.currentText(), self.mode_box.currentData(),
                                                    self.status_filter.currentData(), self.batch_filter.currentData())

    def _select_result(self, *_args):
        from PASS.gui.plot_analysis import FieldResult

        previous_axes = (self.x_column_box.currentText(), self.column_box.currentText())
        self._save_preferences()
        path = self.result_selector.currentData()
        self._active_path = path
        result = self.result_files.get(path)
        field_result = isinstance(result, FieldResult)
        self.field_controls.setVisible(field_result)
        self.table_controls.setVisible(not field_result)
        for field in (self.x_column_box, self.column_box, self.mode_box, self.preset_box, self.compare_mode, self.baseline_box, self.density_bins):
            field.setEnabled(not field_result)
        for field in (self.field_box, self.slice_box):
            field.blockSignals(True)
            field.clear()
        if field_result:
            self.field_box.addItems(list(result.fields))
            for index, identity in enumerate(result.slice_ids):
                self.slice_box.addItem(str(identity), index)
        for field in (self.field_box, self.slice_box):
            field.blockSignals(False)
        self.columns = columns = result.columns if result else {}
        preference = self._preferences.get(path, (*previous_axes, "auto", "all", None))
        fields = (self.x_column_box, self.column_box, self.batch_filter, self.status_filter, self.mode_box)
        for field in fields:
            field.blockSignals(True)
        for box, selected, fallback in ((self.x_column_box, preference[0], 0), (self.column_box, preference[1], 1)):
            box.clear()
            box.addItems(list(columns))
            index = box.findText(selected)
            box.setCurrentIndex(index if index >= 0 else min(fallback, box.count() - 1))
        self.batch_filter.clear()
        self.batch_filter.addItem("全部注入批次", None)
        batches = np.asarray(next((v for k, v in columns.items() if k.casefold() == "injection_batch"), []))
        for batch in np.unique(batches[np.isfinite(batches)]):
            self.batch_filter.addItem(str(int(batch)), int(batch))
        self.batch_filter.setEnabled(bool(batches.size))
        self.status_filter.setEnabled(any(k.casefold() == "tag" for k in columns))
        for box, selected in ((self.mode_box, preference[2]), (self.status_filter, preference[3]), (self.batch_filter, preference[4])):
            box.setCurrentIndex(max(0, box.findData(selected)))
        for field in fields:
            field.blockSignals(False)
        self._select_series()

    @staticmethod
    def _read_table(path):
        from PASS.gui.results import read_result

        return read_result(path).columns

    def _select_series(self, *_args):
        from PASS.gui.plot_analysis import FieldResult, column_label, compare_columns

        result = self.result_files.get(self._active_path)
        self._figure_spec = None
        self._clear_analysis()
        self.view_hint.setText("滚轮缩放，适配视图恢复完整范围")
        self.export_data_button.setEnabled(result is not None)
        self.export_image_button.setEnabled(result is not None)
        if result is None:
            self.plot_stack.setCurrentWidget(self.canvas)
            self.canvas.set_series([], [], empty_message="加载结果文件后选择 X / Y 数值列")
            self.info.setText("未加载数据")
            return
        if isinstance(result, FieldResult):
            self._select_field(result)
            return
        x_name, y_name = self.x_column_box.currentText(), self.column_box.currentText()
        columns = result.select_columns((x_name, y_name), self.status_filter.currentData(), self.batch_filter.currentData())
        x, y = columns.get(x_name, np.empty(0)), columns.get(y_name, np.empty(0))
        names = {name.casefold() for name in self.columns}
        mode = self.mode_box.currentData()
        if mode == "auto":
            mode = "scatter" if names & {"tag", "particle_id"} else "line"
        message = "当前筛选没有匹配数据" if len(x) == 0 else "当前 X / Y 列没有可绘制的有限值"
        x_label, y_label = column_label(result, x_name), column_label(result, y_name)
        comparison = self.compare_mode.currentData()
        explanation = ""
        try:
            if comparison != "single":
                if mode == "density":
                    raise ValueError("文件比较请先选择折线或散点；密度与投影用于当前文件。")
                baseline = self.result_files.get(self.baseline_box.currentData())
                if baseline is None:
                    raise ValueError("请选择一份已加载的基线数据。")
                x, y, reference = compare_columns(result, baseline, x_name, y_name, comparison, self.status_filter.currentData(),
                                                  self.batch_filter.currentData())
                explanation = "比较要求 X 和行顺序严格一致，未进行插值。"
                if comparison == "relative":
                    y_label = f"({y_name} A−B)/B [1]"
                    explanation += " 基线为零或无效时保留 NaN。"
                elif comparison == "difference":
                    y_label = f"Δ {y_label} (A−B)"
                if comparison == "overlay":
                    payload = {
                        "series": [(Path(self._active_path).name + " (A)", x, y, mode),
                                   (Path(self.baseline_box.currentData()).name + " (B)", x, reference, mode)],
                        "x_label":
                        x_label,
                        "y_label":
                        y_label
                    }
                    self._show_figure("series", payload)
                else:
                    self.plot_stack.setCurrentWidget(self.canvas)
                    self.canvas.set_series(x, y, mode="line", empty_message=message, x_label=x_label, y_label=y_label)
                    self._figure_spec = "series", {"series": [(comparison, x, y, "line")], "x_label": x_label, "y_label": y_label}
            elif mode == "density":
                self._show_figure("density", {"x": x, "y": y, "bins": self.density_bins.value(), "x_label": x_label, "y_label": y_label})
                explanation = "密度及边缘投影统计全部有限 X/Y 配对；计数为宏粒子行数，未按电荷加权。"
            else:
                self.plot_stack.setCurrentWidget(self.canvas)
                self.canvas.set_series(x, y, mode=mode, empty_message=message, x_label=x_label, y_label=y_label)
                self._figure_spec = "series", {"series": [(Path(self._active_path).name, x, y, mode)], "x_label": x_label, "y_label": y_label}
        except (ValueError, KeyError, TypeError) as exc:
            self.plot_stack.setCurrentWidget(self.canvas)
            self.canvas.set_series([], [], empty_message=str(exc))
            self.info.setText(str(exc))
            self.export_image_button.setEnabled(False)
            return
        self._save_preferences()
        n_rows = len(next(iter(self.columns.values()), []))
        n_valid = int(np.count_nonzero(np.isfinite(x) & np.isfinite(y)))
        memory = sum(table.nbytes if isinstance(table, FieldResult) else sum(np.asarray(value).nbytes for value in table.columns.values())
                     for table in self.result_files.values()) / 1024**2
        detail = " · ".join(f"{k}={result.metadata[k]}" for k in ("NumAlive", "NumLost", "NumPending", "ZCoordinate", "ReferenceArrivalTime",
                                                                  "ReferenceBeta", "ReferenceMomentum") if k in result.metadata)
        convention = "PASS 粒子列：px=Px/P0、py=Py/P0，不能直接当作 x′/y′；z 为束团相对时间坐标。配套参考量仅适用于存活粒子。" if "tag" in names else ""
        self.info.setText(f"{self._active_path} · {n_rows} rows · {len(self.columns)} numeric columns · 筛选 {len(x)} 行 / 有效 {n_valid} 点\n"
                          f"已加载 {len(self.result_files)} 份数据，数值数组 {memory:.1f} MiB" + (f"\n{detail}" if detail else "") +
                          (f"\n{convention}" if convention else "") + (f"\n{explanation}" if explanation else ""))

    def _apply_preset(self):
        selected = self.preset_box.currentData()
        if selected is None:
            return
        names = {name.casefold(): name for name in self.columns}
        if any(name not in names for name in selected):
            self.load_status.setText("当前文件缺少该相空间预设所需的列。")
            return
        self.x_column_box.blockSignals(True)
        self.column_box.blockSignals(True)
        self.x_column_box.setCurrentText(names[selected[0]])
        self.column_box.setCurrentText(names[selected[1]])
        self.x_column_box.blockSignals(False)
        self.column_box.blockSignals(False)
        self._select_series()

    def _show_figure(self, kind, payload):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from PASS.gui.plot_analysis import create_figure

        figure = create_figure(kind, payload)
        self._clear_analysis()
        self._analysis_canvas = FigureCanvasQTAgg(figure)
        self._analysis_toolbar = NavigationToolbar2QT(self._analysis_canvas, self.analysis_view)
        self.analysis_layout.addWidget(self._analysis_toolbar)
        self.analysis_layout.addWidget(self._analysis_canvas)
        self.plot_stack.setCurrentWidget(self.analysis_view)
        self.view_hint.setText("图形工具栏：平移 / 框选缩放 / 恢复")
        self._analysis_canvas.draw_idle()
        self._figure_spec = kind, payload

    def _clear_analysis(self):
        for widget in (self._analysis_toolbar, self._analysis_canvas):
            if widget is not None:
                self.analysis_layout.removeWidget(widget)
                widget.close()
                widget.deleteLater()
        self._analysis_toolbar = self._analysis_canvas = None

    def fit_view(self):
        if self.plot_stack.currentWidget() is self.analysis_view and self._analysis_toolbar is not None:
            self._analysis_toolbar.home()
        else:
            self.canvas.fit_view()

    def _select_field(self, result):
        name, index = self.field_box.currentText(), self.slice_box.currentData()
        if not name or index is None:
            return
        try:
            values, unit = result.plane(name, index, self.average_field.isChecked())
            self._show_figure("field", {
                "result": result,
                "field": name,
                "index": index,
                "average": self.average_field.isChecked(),
                "aperture": self.aperture_overlay.isChecked()
            })
            attrs = result.metadata
            self.info.setText(
                f"{self._active_path}\n{name} [{unit}] · (slice, y, x)={result.fields[name].shape} · Δz={result.delta_z[index]:.6g} m\n"
                f"turn={attrs.get('turn', '?')} · solver={attrs.get('solver', '?')} · aperture_role={attrs.get('aperture_role', '?')} · "
                f"potential_gauge={attrs.get('potential_gauge', '?')}\n"
                "原始积分量：ρ [C/m²]、φ [V·m]、∫E dz [V]；除以 Δz 后分别为 C/m³、V、V/m。孔径轮廓由保存的参数重建。")
        except (ValueError, KeyError, TypeError) as exc:
            self.plot_stack.setCurrentWidget(self.canvas)
            self.canvas.set_series([], [], empty_message=str(exc))
            self.info.setText(str(exc))
            self.export_image_button.setEnabled(False)

    def _capture_export(self):
        from PASS.gui.plot_analysis import ExportSelection, freeze_result

        result = self.result_files.get(self._active_path)
        if result is None:
            raise ValueError("请先加载数据。")
        return ExportSelection(source=self._active_path,
                               result=freeze_result(result),
                               x_name=self.x_column_box.currentText(),
                               y_name=self.column_box.currentText(),
                               status=self.status_filter.currentData(),
                               batch=self.batch_filter.currentData(),
                               mode=self.mode_box.currentData(),
                               comparison=self.compare_mode.currentData(),
                               baseline_path=self.baseline_box.currentData(),
                               baseline=freeze_result(self.result_files.get(self.baseline_box.currentData())),
                               bins=self.density_bins.value(),
                               field=self.field_box.currentText(),
                               index=self.slice_box.currentData(),
                               average=self.average_field.isChecked(),
                               aperture=self.aperture_overlay.isChecked())

    def export_data(self, path):
        """Synchronous compatibility API; GUI buttons use the background dialog."""
        from PASS.gui.plot_analysis import export_selection_data

        if self._export_dialog is not None:
            raise ValueError("正在导出，请等待当前任务结束。")
        export_selection_data(self._capture_export(), path)

    def export_image(self, path):
        """Synchronous compatibility API, intended for serial programmatic callers."""
        from PASS.gui.plot_analysis import export_selection_image

        if self._export_dialog is not None:
            raise ValueError("正在导出，请等待当前任务结束。")
        if self._figure_spec is None:
            raise ValueError("当前没有可导出的有效图像。")
        export_selection_image(self._capture_export(), path)

    def _run_export(self, path, kind):
        from PASS.gui.jobs import TaskCancelled, TaskDialog
        from PASS.gui.plot_analysis import export_selection_data, export_selection_image

        if self.busy or self._closing:
            self.load_status.setText("请等待当前读取或导出任务结束。")
            return False
        if kind == "image" and self._figure_spec is None:
            self.load_status.setText("当前没有可导出的有效图像。")
            return False
        try:
            selection = self._capture_export()
        except Exception as exc:
            self.load_status.setText("导出失败：" + str(exc))
            QMessageBox.warning(self, "导出失败", str(exc))
            return False

        # Capture only plain values and immutable array views in the worker closure.
        def operation(context):
            if kind == "data":
                export_selection_data(selection, path, context=context)
            else:
                export_selection_image(selection, path, context=context, isolated=True)

        title = "后台导出完整数据" if kind == "data" else "后台导出完整图像"
        dialog = self._export_dialog = TaskDialog(self, title, operation)
        self.export_data_button.setEnabled(False)
        self.export_image_button.setEnabled(False)
        success = False
        try:
            dialog.worker.start()
            dialog.exec()
            dialog.worker.wait()
            if isinstance(dialog.worker.error, TaskCancelled):
                self.load_status.setText("导出已取消，原文件保持不变。")
            elif dialog.worker.error is not None:
                self.load_status.setText("导出失败：" + str(dialog.worker.error))
                if not self._closing:
                    QMessageBox.warning(self, "导出失败", str(dialog.worker.error))
            else:
                success = True
                self.load_status.setText("已导出完整筛选数据及 .metadata.json 元数据。" if kind == "data" else "已从完整筛选数据导出图像。")
        finally:
            # Never release a TaskDialog (which owns the QThread) before it finishes.
            if dialog.worker.isRunning():
                dialog.worker.context.cancelled.set()
                dialog.worker.wait()
            self._export_dialog = None
            dialog.deleteLater()
            self.export_data_button.setEnabled(bool(self.result_files))
            self.export_image_button.setEnabled(self._figure_spec is not None)
            if self._closing and not self.busy:
                self.shutdown_finished.emit()
        return success

    def _choose_data_export(self):
        if self.busy or self._closing:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出全部筛选行与元数据", "filtered_data.csv", "CSV (*.csv)")
        if path:
            self._run_export(path, "data")

    def _choose_image_export(self):
        if self.busy or self._closing:
            return
        path, _ = QFileDialog.getSaveFileName(self, "从完整筛选数据导出图像", "result.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            self._run_export(path, "image")
