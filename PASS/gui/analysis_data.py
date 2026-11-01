"""Shared input selectors and bounded displays for independent signal analysis."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QSpinBox,
                               QTableWidget, QTableWidgetItem, QToolBar, QToolButton, QVBoxLayout, QWidget)

from PASS.gui.appearance import THEMES


def _integer(value, minimum=-1, maximum=2147483647):
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    widget.setMinimumWidth(56)
    widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    return widget


def _responsive_form(form):
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setHorizontalSpacing(8)
    for row in range(form.rowCount()):
        item = form.itemAt(row, QFormLayout.LabelRole)
        if item is not None and isinstance(item.widget(), QLabel):
            item.widget().setWordWrap(True)


def _compact_fields(parent):
    for combo in parent.findChildren(QComboBox):
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(6)
        combo.setMinimumWidth(0)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    for field in parent.findChildren(QLineEdit):
        field.setMinimumWidth(0)
        field.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)


def _pair(first, second):
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(first)
    layout.addWidget(QLabel("至"))
    layout.addWidget(second)
    return widget


class SourceControls(QGroupBox):
    inspect_requested = Signal()
    file_settings_changed = Signal()
    changed = Signal()

    def __init__(self, *, mode="generic"):
        super().__init__("数据与采样")
        self.mode = mode
        self._inventory_source = None
        self._manual_axis = -1
        self._particle_monitor = False
        form = QFormLayout(self)
        self.path = QLineEdit()
        self.path.setPlaceholderText("选择信号文件")
        self.browse = QPushButton("选择…")
        line = QWidget()
        row = QHBoxLayout(line)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.path, 1)
        row.addWidget(self.browse)
        form.addRow("文件", line)
        self.delimiter = QComboBox()
        for title, value in (("自动", None), ("逗号", ","), ("制表符", "\t"), ("空白", "whitespace"), ("分号", ";")):
            self.delimiter.addItem(title, value)
        self.header = QComboBox()
        for title, value in (("自动", "auto"), ("首行是列名", True), ("无列名", False)):
            self.header.addItem(title, value)
        self.skip_rows = _integer(0, 0)
        self.delimiter.setToolTip("仅用于文本文件：自动按文件类型选择分隔符，或明确指定逗号、空白等。修改后自动刷新列列表。")
        self.header.setToolTip("指定首行是否为列名；无列名时使用 column_0 等名称。修改后自动刷新。")
        self.skip_rows.setToolTip("读取文本前跳过的行数，可用于忽略文件开头的说明文字。")
        self.inspect_button = QPushButton("刷新列 / 数组")
        self.inspect_button.setToolTip("打开文件会自动扫描；文件内容更新后，可在这里重新扫描可用列、数组及其形状。")
        self.inspect_button.setAutoDefault(False)
        self.inspect_button.clicked.connect(self.inspect_requested)
        form.addRow(self.inspect_button)
        self.selection_x = QComboBox()
        self.selection_y = QComboBox()
        self.selection_x.setToolTip("选择需要分析的振荡信号，例如粒子 x 或束团质心 xAverage；采样坐标应在下方另行选择。")
        self.selection_y.setToolTip("选择与 X 信号具有相同粒子顺序及采样坐标的 Y 信号。")
        self.label_x = QLabel("信号列 / 数组")
        self.label_y = QLabel("Y 信号列 / 数组")
        form.addRow(self.label_x, self.selection_x)
        form.addRow(self.label_y, self.selection_y)
        self.axis = _integer(-1, -32, 31)
        self.axis.setToolTip("-1 表示最后一维；(粒子数, 圈数) 用 -1，(圈数, 粒子数) 用 0。原生 ParticleMonitor 自动固定为 0。")
        self.coordinate = QComboBox()
        self.coordinate.addItem("未指定坐标", None)
        self.coordinate.setToolTip("指定坐标时从坐标确定 Δt 并检查间隔。未指定时，ParticleMonitor 文件自动使用 turn；其他数据使用手动 Δt。")
        form.addRow("采样坐标列 / 数组", self.coordinate)
        self.alive = QComboBox()
        self.alive.addItem("自动 / 未指定", None)
        self.alive.setToolTip("ParticleMonitor 自动检查 tag；其他数据可选择正值表示存活的标记。所选区间含损失记录时会报错，不会删除后拼接。")
        self.spacing = QLineEdit("1.0")
        self.spacing.setToolTip("坐标单位为秒时频率单位为 Hz；逐圈坐标时为 cycles/turn。指定坐标列后由实际坐标检查并确定间隔。")
        form.addRow("手动采样间隔 Δt", self.spacing)
        self.unit = QComboBox()
        for title, value in (("圈（频率为 cycles/turn）", "turn"), ("秒（频率为 Hz）", "s"), ("自定义采样单位", "sample unit")):
            self.unit.addItem(title, value)
        self.unit.setToolTip("仅控制坐标和频率单位标签，不换算数据；选择秒不会把圈数自动转换为秒。")
        form.addRow("采样单位（仅标签）", self.unit)
        self.sample_start, self.sample_end = _integer(0, 0), _integer(-1)
        self.object_start, self.object_end = _integer(0, 0), _integer(-1)
        self.sample_end.setSpecialValueText("末尾")
        self.object_end.setSpecialValueText("全部")
        for widget in (self.sample_start, self.sample_end):
            widget.setToolTip("采样数组的位置索引，结束索引不包含在内；不一定等于文件中的圈号。")
        for widget in (self.object_start, self.object_end):
            widget.setToolTip("粒子或轨迹的位置索引，结束索引不包含在内；不是直接填写粒子 ID。")
        form.addRow("采样索引 [起, 止)", _pair(self.sample_start, self.sample_end))
        self.object_range = _pair(self.object_start, self.object_end)
        form.addRow("粒子索引 [起, 止)" if mode == "particle" else "对象索引 [起, 止)", self.object_range)
        self.advanced_button = QToolButton()
        self.advanced_button.setText("高级设置")
        self.advanced_button.setCheckable(True)
        self.advanced_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_button.setArrowType(Qt.RightArrow)
        self.advanced = QWidget()
        advanced_form = QFormLayout(self.advanced)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.addRow("文本分隔符", self.delimiter)
        advanced_form.addRow("文本列名", self.header)
        advanced_form.addRow("跳过开头行数", self.skip_rows)
        advanced_form.addRow("采样轴", self.axis)
        advanced_form.addRow("存活标记（大于 0）", self.alive)
        self.advanced.setVisible(False)
        self.advanced_button.toggled.connect(self._show_advanced)
        form.addRow(self.advanced_button)
        form.addRow(self.advanced)
        self.info = QLabel("支持 CSV、TSV、TXT、DAT、TFS、HDF5、NPY、NPZ；明确选择列，不自动配对粒子。")
        self.info.setWordWrap(True)
        form.addRow(self.info)
        _responsive_form(form)
        _responsive_form(advanced_form)
        _compact_fields(self)
        self.set_fma(False)
        if mode in {"statistics", "particle"}:
            self.axis.hide()
            advanced_form.labelForField(self.axis).hide()
            self.alive.hide()
            advanced_form.labelForField(self.alive).hide()
        if mode == "statistics":
            self.object_range.hide()
            form.labelForField(self.object_range).hide()
        self.coordinate.currentIndexChanged.connect(self._coordinate_changed)
        for widget in self.findChildren(QComboBox):
            widget.currentIndexChanged.connect(self.changed)
        for widget in self.findChildren(QSpinBox):
            widget.valueChanged.connect(self.changed)
        for widget in self.findChildren(QLineEdit):
            widget.textChanged.connect(self.changed)
        self.path.textChanged.connect(self.file_settings_changed)
        self.path.textChanged.connect(self._update_file_controls)
        self.delimiter.currentIndexChanged.connect(self.file_settings_changed)
        self.header.currentIndexChanged.connect(self.file_settings_changed)
        self.skip_rows.valueChanged.connect(self.file_settings_changed)

    def _update_file_controls(self):
        text_input = Path(self.path.text()).suffix.lower() in {".csv", ".tsv", ".txt", ".dat"}
        for widget in (self.delimiter, self.header, self.skip_rows):
            widget.setEnabled(text_input)

    def _show_advanced(self, enabled):
        self.advanced.setVisible(enabled)
        self.advanced_button.setArrowType(Qt.DownArrow if enabled else Qt.RightArrow)

    def set_fma(self, enabled):
        self.label_x.setText("X 信号列 / 数组" if enabled else "粒子坐标" if self.mode == "particle" else "统计量" if self.mode == "statistics" else "信号列 / 数组")
        self.label_y.setVisible(enabled)
        self.selection_y.setVisible(enabled)

    def _coordinate_changed(self):
        self.spacing.setEnabled(self.coordinate.currentData() is None)

    def file_options(self):
        return dict(delimiter=self.delimiter.currentData(), skip_rows=self.skip_rows.value(), header=self.header.currentData())

    def load_options(self):
        spacing = 1.0 if self.coordinate.currentData() is not None else float(self.spacing.text())
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("采样间隔必须是有限正数。")
        sample_end = self.sample_end.value()
        object_end = self.object_end.value()
        return dict(sample_axis=-1 if self.mode == "statistics" else self.axis.value(),
                    sample_range=(self.sample_start.value(), None if sample_end < 0 else sample_end),
                    object_range=None if self.mode == "statistics" else (self.object_start.value(), None if object_end < 0 else object_end),
                    coordinate=self.coordinate.currentData(),
                    alive_selection=self.alive.currentData() if self.mode == "generic" else None,
                    sample_spacing=spacing,
                    **self.file_options())

    def set_inventory(self, data):
        names = data["arrays"]
        metadata = data.get("metadata", {})
        monitor_name = metadata.get("parameters", {}).get("Name")
        if self.mode == "statistics":
            if monitor_name == "PASS Particle Monitor":
                raise ValueError("这是 ParticleMonitor 粒子轨迹，请切换到“PM 粒子频谱”；统计数据频谱用于 StatMonitor 或一维统计量序列。")
            if monitor_name and monitor_name.startswith("PASS ") and monitor_name != "PASS Statistic Data":
                raise ValueError("请选择 StatMonitor 统计历史或一维统计量序列；当前 PASS 文件不是统计历史。")
            signal_names = [entry for entry in names if len(entry["shape"] or ()) == 1]
            if not signal_names:
                raise ValueError("统计数据频谱需要一维统计量列 / 数组；粒子轨迹请使用“PM 粒子频谱”。")
        elif self.mode == "particle":
            if monitor_name != "PASS Particle Monitor" or metadata.get("parameters", {}).get("Layout") != "single_file":
                raise ValueError("PM 粒子频谱需要原生 ParticleMonitor HDF5/TFS 文件；StatMonitor 或普通统计量请使用“统计数据频谱”。")
            signal_names = [entry for entry in names if entry["name"].lstrip("/") in {"x", "px", "y", "py", "z", "dp"}]
            if not signal_names:
                raise ValueError("ParticleMonitor 文件没有可用的 x/px/y/py/z/dp 轨迹。")
        else:
            signal_names = names
        source = metadata.get("source")
        same_source = source is not None and source == self._inventory_source
        self._inventory_source = source
        previous = ([self.selection_x.currentData(),
                     self.selection_y.currentData(),
                     self.coordinate.currentData(),
                     self.alive.currentData()] if same_source else [None] * 4)
        monitor = monitor_name == "PASS Particle Monitor"
        statistics = monitor_name == "PASS Statistic Data"
        lookup = {entry["name"].lstrip("/").casefold(): entry["name"] for entry in names}
        defaults = [lookup.get("x" if monitor else "xaverage"), lookup.get("y" if monitor else "yaverage"), None, None]
        if monitor or statistics:
            defaults[2] = lookup.get("turn")
        else:
            defaults[:2] = [None, None]
        excluded = {"turn", "time", "tag", "particle_id", "referencetime", "referencebeta", "referencemomentum", "zcenter"}
        candidates = [entry["name"] for entry in signal_names if entry["name"].rsplit("/", 1)[-1].casefold() not in excluded]
        for index, combo in enumerate((self.selection_x, self.selection_y, self.coordinate, self.alive)):
            combo.blockSignals(True)
            combo.clear()
            if index < 2:
                combo.addItem("请选择信号", None)
            elif index == 2:
                combo.addItem("未指定坐标", None)
            elif index == 3:
                combo.addItem("自动 / 未指定", None)
            for entry in signal_names if index < 2 else names:
                combo.addItem(f"{entry['name']}  {tuple(entry['shape'] or ())}", entry["name"])
                combo.setItemData(combo.count() - 1, f"{entry['name']}  {tuple(entry['shape'] or ())}", Qt.ToolTipRole)
            selection = previous[index] if same_source and combo.findData(previous[index]) >= 0 else defaults[index]
            if index < 2 and selection is None and candidates:
                selection = candidates[min(index, len(candidates) - 1)]
            combo.setCurrentIndex(max(0, combo.findData(selection)))
            combo.blockSignals(False)
        native_monitor = monitor and metadata.get("parameters", {}).get("Layout") == "single_file"
        if native_monitor and not self._particle_monitor:
            self._manual_axis = self.axis.value()
        elif not native_monitor and self._particle_monitor:
            self.axis.setValue(self._manual_axis)
        self._particle_monitor = native_monitor
        if native_monitor:
            self.axis.setValue(0)
        elif not same_source:
            selected = next((entry for entry in names if entry["name"] == self.selection_x.currentData()), None)
            if selected is not None and len(selected["shape"]) == 1:
                self.axis.setValue(-1)
        self.axis.setEnabled(not native_monitor)
        self._update_file_controls()
        self.info.setText(f"{len(names)} 个数值列 / 数组；采样轴、坐标及范围应用于选定信号。")
        if self.mode == "statistics":
            self.info.setText("统计数据频谱：每个统计量是一条随圈数或时间变化的序列，无需选择粒子。")
        if monitor:
            self.info.setText("ParticleMonitor 文件：从 turn 确定采样间隔，并自动检查 tag 存活标记。")
        if native_monitor:
            self.info.setText("ParticleMonitor 文件：采样轴固定为 0，从 turn 确定间隔，并自动检查 tag 存活标记。")
        if self.mode == "particle":
            self.info.setText("PM 粒子频谱：选择坐标和粒子索引范围；每个粒子单独计算，图示范围内首个粒子，结果保留粒子 ID。")
        if (monitor or statistics) and not same_source:
            self.unit.setCurrentIndex(self.unit.findData("turn"))
        self._coordinate_changed()


class MethodControls(QGroupBox):
    changed = Signal()

    def __init__(self, *, fma=False):
        super().__init__("频率估计方法")
        form = QFormLayout(self)
        self.method = QComboBox()
        self.method.addItem("基础 FFT", "fft")
        self.method.addItem("精细 FFT", "refined_fft")
        self.method.setCurrentIndex(1)
        self.method.setToolTip("基础 FFT 给出离散频谱；精细 FFT 增加加窗、补零和峰值估计。")
        form.addRow("方法", self.method)
        self.remove_mean = QCheckBox("减去信号均值")
        self.remove_mean.setChecked(fma)
        self.remove_mean.setToolTip("从每条信号减去其平均值，去除偏置产生的直流分量；需要保留均值时不要勾选。")
        form.addRow(self.remove_mean)
        self.refined = QWidget()
        fields = QFormLayout(self.refined)
        fields.setContentsMargins(0, 0, 0, 0)
        self.window = QComboBox()
        for title, value in (("Hann", "hann"), ("矩形", "rectangle"), ("Hamming", "hamming"), ("Blackman", "blackman")):
            self.window.addItem(title, value)
        self.window.setToolTip("控制有限记录截断造成的谱泄漏；矩形相当于不加窗，不同窗改变主峰宽度与旁瓣。")
        self.padding = _integer(8, 1, 1024)
        self.padding.setToolTip("FFT 长度为样本数乘以此倍数；仅细化频率网格，不增加观测时长或真实分辨能力。")
        self.interpolation = QComboBox()
        self.interpolation.addItem("抛物线峰值插值", "parabolic")
        self.interpolation.addItem("不插值", "none")
        self.interpolation.setToolTip("利用峰附近三个频点的对数幅值估计网格之间的峰位置；不插值时直接使用网格频率。")
        fields.addRow("窗函数", self.window)
        fields.addRow("补零倍数", self.padding)
        fields.addRow("峰值插值", self.interpolation)
        form.addRow(self.refined)
        self.range_x = QLineEdit()
        self.range_x.setPlaceholderText("留空为全频段；例如 0.1, 0.4")
        self.range_y = QLineEdit()
        self.range_y.setPlaceholderText("留空为全频段")
        for field in (self.range_x, self.range_y):
            field.setToolTip("仅限制找峰的频率范围，不裁剪完整频谱；单位与采样间隔一致，例如 cycles/turn 或 Hz。")
        self.range_label = QLabel("X 频段" if fma else "峰值搜索频段")
        form.addRow(self.range_label, self.range_x)
        if fma:
            form.addRow("Y 频段", self.range_y)
        self.n_peaks = _integer(1, 1, 100)
        self.n_peaks.setToolTip("按幅值选择的局部谱峰数量；不会自动识别这些峰的物理模式。")
        if not fma:
            fields.addRow("峰值数", self.n_peaks)
        _responsive_form(form)
        _responsive_form(fields)
        _compact_fields(self)
        self._fma = fma
        self.method.currentIndexChanged.connect(self._update_method)
        self._update_method()
        for widget in self.findChildren(QComboBox):
            widget.currentIndexChanged.connect(self.changed)
        for widget in self.findChildren(QSpinBox):
            widget.valueChanged.connect(self.changed)
        for widget in self.findChildren(QCheckBox):
            widget.toggled.connect(self.changed)
        for widget in self.findChildren(QLineEdit):
            widget.textChanged.connect(self.changed)

    def _update_method(self):
        refined = self.method.currentData() == "refined_fft"
        self.refined.setEnabled(refined)
        self.range_x.setEnabled(refined or self._fma)
        self.range_label.setEnabled(refined or self._fma)

    @staticmethod
    def _range(text):
        if not text.strip():
            return None
        values = tuple(float(value.strip()) for value in text.split(","))
        if len(values) != 2 or not all(np.isfinite(values)) or values[0] >= values[1]:
            raise ValueError("频段应为递增的两个有限数，例如 0.1, 0.4。")
        return values

    def parameters(self):
        method = self.method.currentData()
        kwargs = dict(remove_mean=self.remove_mean.isChecked())
        if method == "refined_fft":
            kwargs.update(window=self.window.currentData(), padding_factor=self.padding.value(), interpolation=self.interpolation.currentData())
        if self._fma:
            kwargs.update(method=method, frequency_range_x=self._range(self.range_x.text()), frequency_range_y=self._range(self.range_y.text()))
        elif method == "refined_fft":
            kwargs.update(frequency_range=self._range(self.range_x.text()), n_peaks=self.n_peaks.value())
        return method, kwargs


def style_figure(figure, theme):
    colors = THEMES[theme]
    figure.set_facecolor(colors["bg"])
    for ax in figure.axes:
        ax.set_facecolor(colors["input"])
        ax.tick_params(colors=colors["text"])
        ax.grid(True, color=colors["line"], alpha=.35)
        for spine in ax.spines.values():
            spine.set_color(colors["line"])
        for label in (ax.xaxis.label, ax.yaxis.label, ax.title, *ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_color(colors["text"])
            label.set_fontfamily(["Microsoft YaHei", "DejaVu Sans"])
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(colors["input"])
            legend.get_frame().set_edgecolor(colors["line"])
            for label in (legend.get_title(), *legend.get_texts()):
                label.set_color(colors["text"])
                label.set_fontfamily(["Microsoft YaHei", "DejaVu Sans"])


def bounded_indices(size, limit=10000):
    return np.arange(size) if size <= limit else np.linspace(0, size - 1, limit, dtype=np.intp)


def bounded_line_indices(values, limit=10000):
    """Preserve narrow extrema in contiguous bins instead of skipping FFT peaks."""
    values = np.asarray(values)
    if values.size <= limit:
        return np.arange(values.size)
    block = int(np.ceil(values.size / max(1, (limit - 4) // 2)))
    full = values.size // block
    rows = values[:full * block].reshape(full, block)
    start = np.arange(full) * block
    selected = np.concatenate((start + rows.argmin(axis=1), start + rows.argmax(axis=1), [0, values.size - 1]))
    if full * block < values.size:
        tail = values[full * block:]
        selected = np.concatenate((selected, [full * block + tail.argmin(), full * block + tail.argmax()]))
    return np.unique(selected)


def object_labels(loaded, count):
    identities = loaded.get("object_ids")
    if identities is not None and np.asarray(identities).size == count:
        return np.asarray(identities).reshape(-1)
    return np.arange(count)


class AnalysisResultView(QWidget):
    """Create Matplotlib only when analysis is first activated."""

    def __init__(self):
        super().__init__()
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure
        from PASS.gui.data_cursor import FigureDataCursor

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(7, 5), dpi=100, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(250)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.cursor_toolbar = QToolBar(self)
        self.cursor_toolbar.setMovable(False)
        self.cursor = FigureDataCursor(self.canvas, toolbar=self.toolbar, controls_toolbar=self.cursor_toolbar)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.cursor_toolbar)
        layout.addWidget(self.canvas, 3)
        self.summary = QLabel("选择数据后运行分析。图形最多显示 10,000 个点，表格最多显示 200 行；数据导出保留完整结果。")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMaximumHeight(180)
        layout.addWidget(self.table, 1)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.details)

    def heightForWidth(self, width):
        # Qt's minimum layout height only includes one line per wrapping label.
        # Use the readable minimum, not Matplotlib's larger preferred size.
        height = self.layout().minimumSize().height()
        for label in (self.summary, self.details):
            height += max(0, label.heightForWidth(width) - label.minimumSizeHint().height())
        return height

    def set_rows(self, columns):
        names = list(columns)
        n = min(200, len(next(iter(columns.values())))) if columns else 0
        self.table.setColumnCount(len(names))
        labels = dict(object="对象索引",
                      peak="峰",
                      frequency="频率",
                      amplitude="幅值",
                      phase_rad="相位 (rad)",
                      valid="有效",
                      quality="质量标记",
                      qx_first="窗口一 Qx",
                      qy_first="窗口一 Qy",
                      qx_second="窗口二 Qx",
                      qy_second="窗口二 Qy",
                      delta_qx="ΔQx",
                      delta_qy="ΔQy",
                      drift="频率漂移",
                      diffusion_log10="log10 漂移")
        self.table.setHorizontalHeaderLabels([labels.get(name, name) for name in names])
        self.table.setRowCount(n)
        for j, name in enumerate(names):
            for i, value in enumerate(columns[name][:n]):
                text = f"{value:.9g}" if isinstance(value, (float, np.floating)) else str(value)
                self.table.setItem(i, j, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
