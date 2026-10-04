"""Asynchronous dynamic-aperture result analysis inside the Analysis workspace."""

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from PASS.gui.analysis import AnalysisWorker
from PASS.gui.widgets import file_dialog_directory


def _integer(value, maximum=2147483647, minimum=1):
    field = QSpinBox()
    field.setRange(minimum, maximum)
    field.setValue(value)
    return field


class DynamicAperturePage(QWidget):
    """Read, plot and export existing particle-monitor results without editing input."""

    shutdown_finished = Signal()
    analysis_finished = Signal(object)

    def __init__(self):
        super().__init__()
        self.theme = "dark"
        self._activated = False
        self._closing = False
        self._worker = None
        self.result = None
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(16, 12, 16, 12)

    def activate(self):
        if self._activated or self._closing:
            return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self._activated = True
        self.title = QLabel("动力学孔径")
        self.title.setObjectName("formTitle")
        self._root.addWidget(self.title)
        splitter = QSplitter()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        controls = QWidget()
        layout = QVBoxLayout(controls)
        self.analysis_controls = QWidget()
        analyse_layout = QVBoxLayout(self.analysis_controls)
        self.source = QLineEdit()
        self.requested_turn = _integer(-1, minimum=-1)
        self.requested_turn.setToolTip("按文件中的圈数填写，从 0 开始，在所选 ParticleMonitor 位置判定粒子状态。"
                                       "手动输入超过最后记录圈时自动调整并提示；-1 使用计划末圈检查记录是否完整。")
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.addRow("ParticleMonitor 文件", self._path_row(self.source))
        form.addRow("终点圈数（-1 自动）", self.requested_turn)
        analyse_layout.addLayout(form)
        self.turn_notice = QLabel()
        self.turn_notice.setWordWrap(True)
        self.turn_notice.hide()
        analyse_layout.addWidget(self.turn_notice)
        self.analyse_button = QPushButton("读取并分析")
        self.analyse_button.clicked.connect(self.analyse)
        analyse_layout.addWidget(self.analyse_button)
        self.plot_mode = QComboBox()
        self.plot_mode.addItem("单个 dp：散点与边界", "single")
        self.plot_mode.addItem("多个 dp：边界叠加", "overlay")
        self.plot_mode.currentIndexChanged.connect(self._change_plot_mode)
        plot_choices = QFormLayout()
        plot_choices.setRowWrapPolicy(QFormLayout.WrapLongRows)
        plot_choices.addRow("绘图方式", self.plot_mode)
        analyse_layout.addLayout(plot_choices)
        self.dp = QComboBox()
        self.dp.currentIndexChanged.connect(self.draw_result)
        self.mode = QComboBox()
        self.mode.addItem("存活状态", "status")
        self.mode.addItem("丢失圈数", "loss_turn")
        self.mode.currentIndexChanged.connect(self.draw_result)
        self.boundary = QCheckBox("显示孔径边界")
        self.boundary.setChecked(True)
        self.boundary.setToolTip("在固定 px、py、z 的完整 x-y 网格上，根据相邻存活点与丢失点估计边界；保留稳定岛、孔洞和未知区域。")
        self.boundary.toggled.connect(self.draw_result)
        self.single_controls = QWidget()
        choices = QFormLayout(self.single_controls)
        choices.setContentsMargins(0, 0, 0, 0)
        choices.setRowWrapPolicy(QFormLayout.WrapLongRows)
        choices.addRow("初始 dp", self.dp)
        choices.addRow("颜色", self.mode)
        choices.addRow(self.boundary)
        analyse_layout.addWidget(self.single_controls)
        self.overlay_controls = QWidget()
        overlay_layout = QVBoxLayout(self.overlay_controls)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.addWidget(QLabel("叠加的初始 dp"))
        self.dp_groups = QListWidget()
        self.dp_groups.setSelectionMode(QAbstractItemView.NoSelection)
        self.dp_groups.setMinimumHeight(84)
        self.dp_groups.setMaximumHeight(148)
        self.dp_groups.itemChanged.connect(self.draw_result)
        overlay_layout.addWidget(self.dp_groups)
        selection = QHBoxLayout()
        self.select_all_dp = QPushButton("全选")
        self.select_all_dp.clicked.connect(lambda: self._select_dp_groups(True))
        self.clear_dp = QPushButton("清空")
        self.clear_dp.clicked.connect(lambda: self._select_dp_groups(False))
        selection.addWidget(self.select_all_dp)
        selection.addWidget(self.clear_dp)
        overlay_layout.addLayout(selection)
        self.dp_selection_note = QLabel()
        self.dp_selection_note.setWordWrap(True)
        overlay_layout.addWidget(self.dp_selection_note)
        analyse_layout.addWidget(self.overlay_controls)
        self.overlay_controls.hide()
        numeric = QPushButton("导出分析数据 CSV / NPZ…")
        numeric.setToolTip("导出所有 dp 下的粒子初始坐标、存活状态、丢失圈和分析元数据，便于再次分析；不导出逐圈轨迹。")
        numeric.clicked.connect(self.export_result)
        analyse_layout.addWidget(numeric)
        picture = QPushButton("导出当前图 PNG / PDF / SVG…")
        picture.setToolTip("保存右侧当前显示的 Matplotlib 图，包括当前 dp 选择和边界；导出图片或矢量图，不生成 Python 脚本。")
        picture.clicked.connect(self.export_image)
        analyse_layout.addWidget(picture)
        help_text = QLabel("终点圈数按文件记录填写，例如记录 0 至 999 圈时，终点填 999；手动输入超出末圈时自动调整并提示。"
                           "-1 自动选择文件的计划末圈。"
                           "在该监测位置判定粒子状态，缺失初值或终点圈的采样不会自动判为存活。"
                           "边界线为当前圈数、dp 和网格下的孔径边界估计；扫描区域以外尚未验证，半平面扫描可在 y=0 保留开放边界。")
        help_text.setWordWrap(True)
        analyse_layout.addWidget(help_text)
        analyse_layout.addStretch()
        layout.addWidget(self.analysis_controls)
        self.cancel_button = QPushButton("取消后台操作")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        layout.addWidget(self.cancel_button)
        self.status = QLabel("在 Injection 中配置扫描网格；这里读取已有 ParticleMonitor 文件。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        scroll.setWidget(controls)
        splitter.addWidget(scroll)
        plot = QWidget()
        plot_layout = QVBoxLayout(plot)
        self.figure = Figure(figsize=(7, 5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        plot_layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot)
        splitter.setSizes([360, 750])
        self._root.addWidget(splitter, 1)
        for combo in (self.dp, self.mode, self.plot_mode):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(6)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _path_row(self, field):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field, 1)
        browse = QPushButton("文件…")
        browse.clicked.connect(lambda: self._browse(field))
        layout.addWidget(browse)
        return row

    def _browse(self, field):
        start = file_dialog_directory(self)
        value, _ = QFileDialog.getOpenFileName(self, "选择 ParticleMonitor 文件", start, "PASS data (*.h5 *.hdf5 *.tfs);;All files (*)")
        if value:
            field.setText(value)

    def analyse(self):
        path = self.source.text().strip()
        requested = self.requested_turn.value()

        def operation(check):
            from PASS.analysis.dynamic_aperture import read_dynamic_aperture

            check()
            if not path:
                raise ValueError("请选择 ParticleMonitor 文件。")
            result = read_dynamic_aperture(path, requested_turn=None if requested < 0 else requested, cancel=check, clamp_to_available=True)
            check()
            return result

        self._start(operation, "analysis")

    def _start(self, operation, kind):
        if self.busy or self._closing:
            return
        if kind == "analysis":
            self.turn_notice.clear()
            self.turn_notice.hide()
        self.analysis_controls.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText("正在后台处理…")
        worker = AnalysisWorker(operation, self)
        self._worker = worker
        worker.finished.connect(lambda: self._finish(worker, kind))
        worker.start()

    def _finish(self, worker, kind):
        self._worker = None
        worker.deleteLater()
        if self._closing:
            self.shutdown_finished.emit()
            return
        self.analysis_controls.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if isinstance(worker.error, InterruptedError) or (worker.isInterruptionRequested() and not (kind == "export" and worker.result is not None)):
            self.status.setText("操作已取消。")
        elif worker.error is not None:
            self.status.setText(str(worker.error))
        elif kind == "analysis":
            self.result = worker.result
            metadata = self.result["metadata"]
            requested = metadata.get("requested_turn_input")
            effective = metadata["requested_turn"]
            if requested is not None and requested != effective:
                self.requested_turn.setValue(effective)
                self.turn_notice.setText(f"终点圈数 {requested} 超出文件最后记录圈 {effective}，已自动调整为 {effective}。")
                self.turn_notice.show()
            initial = np.asarray(self.result["initial_coordinates"])
            self._set_dp_groups()
            states, counts = np.unique(self.result["status"], return_counts=True)
            summary = "，".join(f"{state}: {count}" for state, count in zip(states, counts))
            self.status.setText(f"已分析 {len(initial):,} 个粒子。{summary}。存活与丢失状态以该位置的实际采样记录为准。")
            try:
                self.draw_result()
            except (ValueError, TypeError, KeyError) as exc:
                self.status.setText("结果已读取，但无法绘图：" + str(exc))
            self.analysis_finished.emit(self.result)
        else:
            self.status.setText("已导出：" + str(worker.result))

    def _set_dp_groups(self):
        self.dp.blockSignals(True)
        self.dp_groups.blockSignals(True)
        self.dp.clear()
        self.dp_groups.clear()
        values = self.result["dp_values"]
        labels = [f"{value:.9g}" for value in values]
        _, label_indices, label_counts = np.unique(labels, return_inverse=True, return_counts=True)
        for index in np.flatnonzero(label_counts[label_indices] > 1):
            labels[index] = f"{values[index]:.17g}"
        for value, label in zip(values, labels):
            self.dp.addItem(label, float(value))
            self.dp.setItemData(self.dp.count() - 1, f"{value:.17g}", Qt.ToolTipRole)
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, float(value))
            item.setToolTip(f"{value:.17g}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.dp_groups.addItem(item)
        self.dp.blockSignals(False)
        self.dp_groups.blockSignals(False)

    def _change_plot_mode(self, *_args):
        overlay = self.plot_mode.currentData() == "overlay"
        self.single_controls.setVisible(not overlay)
        self.overlay_controls.setVisible(overlay)
        self.draw_result()

    def _select_dp_groups(self, checked):
        self.dp_groups.blockSignals(True)
        for index in range(self.dp_groups.count()):
            self.dp_groups.item(index).setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.dp_groups.blockSignals(False)
        self.draw_result()

    def draw_result(self, *_args):
        if self.result is None:
            return
        self.figure.clear()
        if self.dp.currentIndex() < 0:
            self._style()
            return
        from PASS.plot.plot_dynamic_aperture import plot_dynamic_aperture, plot_dynamic_aperture_boundaries

        ax = self.figure.add_subplot()
        if self.plot_mode.currentData() == "overlay":
            selected = [
                self.dp_groups.item(index).data(Qt.UserRole) for index in range(self.dp_groups.count())
                if self.dp_groups.item(index).checkState() == Qt.Checked
            ]
            self.dp_selection_note.setText("按注入时的 dp 分组；勾选需要叠加的边界。" if selected else "请至少勾选一个初始 dp，也可点击“全选”。")
            if selected:
                plot_dynamic_aperture_boundaries(self.result, dp_values=selected, ax=ax)
            else:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "Select at least one initial dp.", transform=ax.transAxes, ha="center", va="center")
        else:
            plot_dynamic_aperture(self.result, dp=self.dp.currentData(), mode=self.mode.currentData(), boundary=self.boundary.isChecked(), ax=ax)
        self._style()

    def export_result(self):
        if self.result is None:
            self.status.setText("请先读取并分析结果。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出动力学孔径分析数据", file_dialog_directory(self), "NumPy (*.npz);;CSV (*.csv)")
        if not path:
            return
        result = self.result

        def operation(check):
            from PASS.analysis.dynamic_aperture import export_dynamic_aperture

            check()
            export_dynamic_aperture(result, path)
            return path

        self._start(operation, "export")

    def export_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出图形", file_dialog_directory(self), "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            try:
                self.figure.savefig(path, dpi=180)
                self.status.setText("已导出：" + path)
            except OSError as exc:
                self.status.setText(str(exc))

    def _style(self):
        from PASS.gui.analysis_data import style_figure
        from PASS.gui.appearance import THEMES

        style_figure(self.figure, self.theme)
        colors = THEMES[self.theme]
        for axes in self.figure.axes:
            for annotation in axes.texts:
                annotation.set_color(colors["muted"])
            for artist in axes.collections:
                if getattr(artist, "_pass_da_boundary", False) and not getattr(artist, "_pass_da_overlay", False):
                    artist.set_edgecolor(colors["text"])
                    artist.set_facecolor("none")
            legend = axes.get_legend()
            lines = [*axes.lines, *(legend.get_lines() if legend is not None else [])]
            for line in lines:
                if getattr(line, "_pass_da_boundary", False) and not getattr(line, "_pass_da_overlay", False):
                    line.set_color(colors["text"])
        self.canvas.draw_idle()

    @property
    def busy(self):
        return self._worker is not None

    def cancel(self):
        if self._worker is not None:
            self._worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.status.setText("正在取消；等待当前文件操作完成…")

    def shutdown(self):
        self._closing = True
        self.cancel()
        return not self.busy

    def set_theme(self, theme):
        self.theme = theme
        if self._activated:
            self._style()
