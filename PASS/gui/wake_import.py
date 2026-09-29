"""Explicit physical wake import with isolated preview and conversion jobs."""
import json
import math
import os
from pathlib import Path
import sys

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QIODevice, QProcess, QSaveFile, QTemporaryDir, Qt, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from PASS.gui.widgets import file_dialog_directory


class WakeImportDialog(QDialog):
    """Import numeric wakes or inspect standard TFS; keep sources immutable."""
    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导入尾场 → 标准 TFS")
        self.resize(1180, 850)
        self.process = None
        self._job_directory = None
        self._preview_options = None
        self._preview_sha256 = None
        self._cancelled = False
        self._buffer = self._stderr = ""
        self._job_result = None
        self._updating = False
        self._unit_order = None
        self._component_config = None
        self._data_kind = None
        self._preset_active = False
        self._delimiter_explicit = False
        root = QVBoxLayout(self)
        files = QHBoxLayout()
        self.source = QLineEdit()
        self.source.setReadOnly(True)
        self.source.setPlaceholderText("选择 CSV / TXT / HEADTAIL，或打开标准尾场 TFS 复核")
        self.open_button = QPushButton("选择文件…")
        self.open_button.clicked.connect(self.choose_file)
        files.addWidget(self.source, 1)
        files.addWidget(self.open_button)
        root.addLayout(files)
        presets = QHBoxLayout()
        self.preset_name = QLineEdit()
        self.preset_name.setPlaceholderText("导入预设名称（如 CST 纵向 ns-V/pC）")
        self.load_preset_button = QPushButton("加载预设…")
        self.load_preset_button.clicked.connect(self.load_preset)
        self.save_preset_button = QPushButton("保存预设…")
        self.save_preset_button.clicked.connect(self.save_preset)
        presets.addWidget(self.preset_name, 1)
        presets.addWidget(self.load_preset_button)
        presets.addWidget(self.save_preset_button)
        root.addLayout(presets)
        note = QLabel("每次导入一个分量；默认前两列为横坐标与尾场值。请按源文件声明单位和符号。\n"
                      "因果表从零开始并填写 W(0⁺)，程序处理半自作用。有限束长 wake potential 需先去卷积，不能直接作为点电荷尾场。")
        note.setWordWrap(True)
        root.addWidget(note)
        splitter = QSplitter(Qt.Horizontal)
        self.options_panel = QWidget()
        form = QFormLayout(self.options_panel)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.format = QComboBox()
        self.format.addItem("CSV / TXT 数值表", "table")
        self.format.addItem("HEADTAIL（ns；V/pC 或 V/(pC·mm)）", "headtail")
        self.format.addItem("标准尾场 TFS（只读约定）", "tfs")
        form.addRow("输入约定", self.format)
        self.delimiter = QComboBox()
        for title, value in (("逗号", ","), ("空白分隔", None), ("制表符", "\t"), ("分号", ";")):
            self.delimiter.addItem(title, value)
        form.addRow("分隔符", self.delimiter)
        self.skiprows = QSpinBox()
        self.skiprows.setRange(0, 1000000000)
        self.skiprows.setToolTip("例如有一行普通表头时填写 1；# 开头的注释行可自动忽略。")
        form.addRow("跳过开头行数", self.skiprows)
        self.axis_column = QSpinBox()
        self.value_column = QSpinBox()
        for widget in (self.axis_column, self.value_column):
            widget.setRange(0, 1000000)
        self.value_column.setValue(1)
        form.addRow("横坐标列（从 0 开始）", self.axis_column)
        form.addRow("尾场值列（从 0 开始）", self.value_column)
        self.component = QComboBox()
        for name in ("longitudinal", "constant_x", "constant_y", "dipolar_x", "dipolar_y", "dipolar_xy", "dipolar_yx", "quadrupolar_x",
                     "quadrupolar_y", "quadrupolar_xy", "quadrupolar_yx", "custom"):
            self.component.addItem(name, name)
        self.component.model().item(self.component.findData("custom")).setEnabled(False)
        form.addRow("尾场分量", self.component)
        self.axis = QComboBox()
        self.axis.addItem("时间延迟 τ", "time")
        self.axis.addItem("距离", "distance")
        self.axis.addItem("频率（TFS 阻抗）", "frequency")
        self.axis.model().item(self.axis.findData("frequency")).setEnabled(False)
        form.addRow("横坐标含义", self.axis)
        self.distance_convention = QComboBox()
        self.distance_convention.addItem("s = βcτ（按参考速度）", "beta_c_tau")
        self.distance_convention.addItem("s = cτ（按光速）", "c_tau")
        form.addRow("距离轴定义", self.distance_convention)
        self.axis_unit = QComboBox()
        form.addRow("横坐标单位", self.axis_unit)
        self.value_unit = QComboBox()
        self.value_unit.setEditable(True)
        form.addRow("尾场单位", self.value_unit)
        self.reference_beta = QLineEdit("1")
        self.reference_beta.setToolTip("产生该响应的参考 β，范围 (0, 1]；距离换算使用上方明确选择的距离轴定义。")
        form.addRow("参考 β", self.reference_beta)
        self.integrated = QCheckBox("已沿元件长度积分")
        self.integrated.setChecked(True)
        form.addRow("长度归一化", self.integrated)
        self.length = QLineEdit()
        self.length.setPlaceholderText("单位长度数据需要正长度")
        form.addRow("元件长度（m）", self.length)
        self.positive_trailing = QCheckBox("横坐标正方向表示尾随")
        self.positive_trailing.setChecked(True)
        form.addRow("横坐标方向", self.positive_trailing)
        self.positive_loss = QCheckBox("纵向正值表示能量损失")
        self.positive_loss.setChecked(True)
        form.addRow("纵向符号", self.positive_loss)
        self.causal = QCheckBox("因果表（起点 τ = 0）")
        self.causal.setChecked(True)
        form.addRow("支持区间", self.causal)
        self.reconstruction = QComboBox()
        self.reconstruction.addItem("two_sided（保留双侧响应）", "two_sided")
        self.reconstruction.addItem("causal_projection（因果投影）", "causal_projection")
        form.addRow("阻抗重构（仿真设置）", self.reconstruction)
        self.unit_notice = QLabel()
        self.unit_notice.setWordWrap(True)
        form.addRow(self.unit_notice)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.options_panel)
        scroll.setMinimumWidth(355)
        splitter.addWidget(scroll)
        self.tabs = QTabWidget()
        plot = QWidget()
        plot_layout = QVBoxLayout(plot)
        self.figure = Figure(figsize=(6, 4), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel("Delay τ [s]")
        self.axes.set_ylabel("W [SI]")
        self.diagnostics = QLabel("全表诊断将在预览后显示。")
        self.diagnostics.setWordWrap(True)
        self.diagnostics.setTextInteractionFlags(Qt.TextSelectableByMouse)
        plot_layout.addWidget(self.diagnostics)
        plot_layout.addWidget(self.canvas, 1)
        plot_layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        self.tabs.addTab(plot, "SI 曲线预览")
        self.metadata = QPlainTextEdit()
        self.metadata.setReadOnly(True)
        self.tabs.addTab(self.metadata, "元数据与诊断")
        self.configuration = QPlainTextEdit()
        self.configuration.setReadOnly(True)
        configuration_panel = QWidget()
        configuration_layout = QVBoxLayout(configuration_panel)
        self.configuration_hint = QLabel("预览标准 TFS 或导出成功后，可复制此分量配置。")
        self.configuration_hint.setWordWrap(True)
        configuration_layout.addWidget(self.configuration_hint)
        configuration_layout.addWidget(self.configuration, 1)
        self.tabs.addTab(configuration_panel, "分量配置")
        splitter.addWidget(self.tabs)
        splitter.setSizes([390, 710])
        root.addWidget(splitter, 1)
        self.status = QLabel("保留原始采样点；默认不拟合、不平滑、不重新采样。先预览，再导出标准尾场 TFS。")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.status)
        actions = QHBoxLayout()
        self.preview_button = QPushButton("预览并校验")
        self.preview_button.clicked.connect(self.preview)
        self.export_button = QPushButton("导出尾场 TFS…")
        self.export_button.clicked.connect(self.export)
        self.copy_config_button = QPushButton("复制分量 JSON")
        self.copy_config_button.setToolTip("复制已验证 TFS 的组件配置；求解器、历史和边界仍需在仿真中配置。")
        self.copy_config_button.clicked.connect(self.copy_component_config)
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.clicked.connect(self.cancel_job)
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.reject)
        for widget in (self.preview_button, self.export_button, self.copy_config_button, self.cancel_button, self.close_button):
            actions.addWidget(widget)
        root.addLayout(actions)
        self.format.currentIndexChanged.connect(self._set_format)
        self.component.currentIndexChanged.connect(self._set_units)
        self.integrated.toggled.connect(self._set_units)
        self.axis.currentIndexChanged.connect(self._set_axis_units)
        for widget in (self.delimiter, self.axis_unit, self.value_unit, self.distance_convention, self.reconstruction):
            widget.currentTextChanged.connect(self._invalidate_preview)
        self.delimiter.currentIndexChanged.connect(self._remember_delimiter)
        for widget in (self.skiprows, self.axis_column, self.value_column):
            widget.valueChanged.connect(self._invalidate_preview)
        for widget in (self.reference_beta, self.length, self.source):
            widget.textChanged.connect(self._invalidate_preview)
        for widget in (self.positive_trailing, self.positive_loss, self.causal):
            widget.toggled.connect(self._invalidate_preview)
        self._set_axis_units()
        self._set_units()
        self._set_busy(False)

    @property
    def busy(self):
        return self.process is not None

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择尾场文件", file_dialog_directory(self), "尾场数据 (*.tfs *.csv *.txt *.dat);;所有文件 (*)")
        if path:
            self.open_path(path)

    def open_path(self, path):
        if self.busy:
            return
        is_tfs = Path(path).suffix.lower() == ".tfs"
        if not is_tfs and self.axis.currentData() == "frequency":
            self._unit_order = None
        self._data_kind = None
        self.source.setText(str(Path(path).resolve()))
        if is_tfs:
            self.format.setCurrentIndex(self.format.findData("tfs"))
            self.preview()
        else:
            if self.format.currentData() == "tfs":
                self.format.setCurrentIndex(self.format.findData("table"))
            if not self._preset_active and not self._delimiter_explicit:
                self._updating = True
                try:
                    self.delimiter.setCurrentIndex(0 if Path(path).suffix.lower() == ".csv" else 1)
                finally:
                    self._updating = False

    def _remember_delimiter(self, *_args):
        if not self._updating:
            self._delimiter_explicit = True

    def _set_axis_units(self, *_args):
        if self._updating:
            return
        previous = self.axis_unit.currentText()
        self.axis_unit.clear()
        choices = {"time": ["s", "ms", "us", "ns", "ps"], "distance": ["m", "cm", "mm"], "frequency": ["Hz", "kHz", "MHz", "GHz"]}
        self.axis_unit.addItems(choices[self.axis.currentData()])
        self.axis_unit.setCurrentText(previous if previous in choices[self.axis.currentData()] else "ns" if self.axis.currentData() ==
                                      "time" else choices[self.axis.currentData()][0])
        self._update_controls()
        self._invalidate_preview()

    def _set_units(self, *_args):
        if self._updating or self.format.currentData() == "tfs":
            return
        order = int(self.component.currentData().startswith(("dipolar", "quadrupolar")))
        headtail = self.format.currentData() == "headtail"
        order += int(not self.integrated.isChecked())
        previous = self.value_unit.currentText()
        previous_order = self._unit_order
        self._updating = True
        self.value_unit.clear()
        if headtail:
            self.value_unit.addItem("V/pC/mm" if order else "V/pC")
        else:
            suffix = "" if order == 0 else "/m" if order == 1 else f"/m^{order}"
            self.value_unit.addItems([f"V/{charge}{suffix}" for charge in ("C", "nC", "pC")])
            if previous and previous_order == order:
                self.value_unit.setCurrentText(previous)
        self._updating = False
        self._unit_order = order
        if previous and previous != self.value_unit.currentText():
            self.unit_notice.setText(f"单位要求已改变：{previous} → {self.value_unit.currentText()}。请按源文件重新核对单位；数据值尚未转换。")
        elif previous_order == order:
            self.unit_notice.setText(f"已保留单位 {self.value_unit.currentText()}；请确认与该分量的源数据一致。")
        self._update_controls()
        self._invalidate_preview()

    def _set_format(self, *_args):
        if self._updating:
            return
        kind = self.format.currentData()
        if kind == "tfs":
            self.unit_notice.setText("标准 TFS 的分量、单位、符号与 β 由文件头读取，物理约定只读。")
            self._update_controls()
            self._invalidate_preview()
            return
        if self.component.currentData() == "custom":
            self.component.setCurrentIndex(0)
        if self.axis.currentData() == "frequency":
            self.axis.setCurrentIndex(0)
        self._set_axis_units()
        if self._data_kind == "impedance":
            self._unit_order = None
        self._data_kind = "wake_function"
        headtail = self.format.currentData() == "headtail"
        if headtail:
            self.axis.setCurrentIndex(0)
            self.axis_unit.setCurrentText("ns")
            self.integrated.setChecked(True)
            self.delimiter.setCurrentIndex(1)
        self._set_units()

    def _update_controls(self):
        kind = self.format.currentData()
        readonly = kind == "tfs"
        for widget in (self.delimiter, self.skiprows, self.axis_column, self.value_column, self.component, self.reference_beta,
                       self.positive_trailing, self.positive_loss, self.causal):
            widget.setEnabled(not readonly)
        for widget in (self.axis, self.axis_unit, self.integrated, self.value_unit):
            widget.setEnabled(kind == "table")
        self.distance_convention.setEnabled(not readonly and self.axis.currentData() == "distance")
        self.length.setEnabled(not readonly and not self.integrated.isChecked())
        self.positive_loss.setEnabled(not readonly and self.component.currentData() == "longitudinal")
        self.reconstruction.setEnabled(readonly and self._data_kind == "impedance")

    def options(self, *, require_source=True):
        if require_source and not self.source.text():
            raise ValueError("请先选择源文件。")
        if self.format.currentData() == "tfs":
            return {"format": "tfs", "reconstruction": self.reconstruction.currentData()}
        return {
            "format": self.format.currentData(),
            "component": self.component.currentData(),
            "axis_column": self.axis_column.value(),
            "value_column": self.value_column.value(),
            "delimiter": self.delimiter.currentData(),
            "skiprows": self.skiprows.value(),
            "causal": self.causal.isChecked(),
            "length": None if self.integrated.isChecked() else float(self.length.text()),
            "convention": {
                "data_kind": "wake_function",
                "axis": self.axis.currentData(),
                "axis_unit": self.axis_unit.currentText(),
                "distance_convention": self.distance_convention.currentData(),
                "value_unit": self.value_unit.currentText().strip(),
                "positive_trailing": self.positive_trailing.isChecked(),
                "longitudinal_positive_loss": self.positive_loss.isChecked(),
                "fourier_exponent": -1,
                "transverse_impedance_factor": "i",
                "shunt_impedance_convention": "not_applicable",
                "integrated": self.integrated.isChecked(),
                "reference_beta": float(self.reference_beta.text())
            }
        }

    def _invalidate_preview(self, *_args):
        if self._updating:
            return
        self._preview_options = None
        self._preview_sha256 = None
        self._set_component_config(None)
        self.diagnostics.setText("文件或约定已改变，请重新预览以更新全表诊断。")
        self.export_button.setEnabled(False)
        self.preview_button.setEnabled(bool(self.source.text()) and not self.busy)
        self.status.setText("文件或约定已改变，请重新预览并校验后导出。")

    def _validate_preset(self, document):
        from PASS.para.schema.wake_field import WakeFileConvention
        if (not isinstance(document, dict) or set(document) != {"kind", "version", "name", "options"} or document["kind"] != "pass_wake_import_preset"
                or type(document["version"]) is not int or document["version"] != 1):
            raise ValueError("请选择版本 1 的 PASS 尾场导入预设 JSON。")
        if not isinstance(document["name"], str) or not document["name"].strip():
            raise ValueError("导入预设需要非空名称。")
        options = document["options"]
        if not isinstance(options, dict):
            raise ValueError("预设 options 必须为对象。")
        if options.get("format") == "tfs":
            if set(options) != {"format", "reconstruction"} or options["reconstruction"] not in {"two_sided", "causal_projection"}:
                raise ValueError("TFS 预设只保存格式和阻抗重构选择，物理约定必须来自文件头。")
            return document
        required = {"format", "component", "axis_column", "value_column", "delimiter", "skiprows", "causal", "length", "convention"}
        if set(options) != required or not isinstance(options["format"], str) or options["format"] not in {"table", "headtail"}:
            raise ValueError("数值导入预设缺少必要选项或包含不支持的选项。")
        named = {self.component.itemData(index) for index in range(self.component.count())} - {"custom"}
        if not isinstance(options["component"], str) or options["component"] not in named:
            raise ValueError("数值导入预设需要命名分量；自定义分量请使用转换器 API/CLI。")
        if any(
                type(options[name]) is not int or not 0 <= options[name] <= maximum
                for name, maximum in (("axis_column", 1000000), ("value_column", 1000000), ("skiprows", 1000000000))):
            raise ValueError("预设列号和跳过行数必须为 GUI 范围内的非负整数。")
        delimiter = options["delimiter"]
        if (options["axis_column"] == options["value_column"] or delimiter is not None and not isinstance(delimiter, str)
                or delimiter not in {None, ",", "\t", ";"} or type(options["causal"]) is not bool):
            raise ValueError("预设必须指定不同的数据列、受支持的分隔符和布尔因果选项。")
        convention = WakeFileConvention.model_validate(options["convention"]).model_dump()
        units = {"time": {"s", "ms", "us", "ns", "ps"}, "distance": {"m", "cm", "mm"}}
        if convention["data_kind"] != "wake_function" or convention["axis_unit"] not in units.get(convention["axis"], set()):
            raise ValueError("GUI 数值导入预设只支持时域点电荷尾场；阻抗请转换为标准 TFS 后复核。")
        if (convention["fourier_exponent"] != -1 or convention["transverse_impedance_factor"] != "i"
                or convention["shunt_impedance_convention"] != "not_applicable"):
            raise ValueError("时域 GUI 预设不支持额外的阻抗约定。")
        length = options["length"]
        if convention["integrated"]:
            if length is not None:
                raise ValueError("已积分预设不应另设长度。")
        elif isinstance(length, bool) or not isinstance(length, (int, float)) or not math.isfinite(length) or length <= 0:
            raise ValueError("单位长度预设需要正的有限元件长度。")
        if options["format"] == "headtail":
            order = int(options["component"].startswith(("dipolar", "quadrupolar")))
            if (convention["axis"] != "time" or convention["axis_unit"] != "ns" or not convention["integrated"]
                    or convention["value_unit"] != ("V/pC/mm" if order else "V/pC")):
                raise ValueError("HEADTAIL 预设必须使用 ns 与已积分 V/pC 或 V/pC/mm。")
        return {**document, "options": {**options, "convention": convention}}

    def save_preset_file(self, path, *, overwrite=False):
        """Save only declared input options; never save a source path or digest."""
        destination = Path(path).resolve()
        if self.source.text():
            source = Path(self.source.text()).resolve()
            if destination == source or destination.exists() and source.exists() and os.path.samefile(destination, source):
                raise ValueError("预设文件必须与源数据文件不同。")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"预设已存在：{destination}")
        document = self._validate_preset({
            "kind": "pass_wake_import_preset",
            "version": 1,
            "name": self.preset_name.text().strip() or destination.stem,
            "options": self.options(require_source=False)
        })
        payload = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        stream = QSaveFile(str(destination))
        if not stream.open(QIODevice.WriteOnly):
            raise OSError(stream.errorString())
        if stream.write(payload) != len(payload) or not stream.commit():
            raise OSError(stream.errorString())
        self.preset_name.setText(document["name"])
        return document

    def load_preset_file(self, path):
        """Validate the complete preset before changing any GUI fields."""
        source = Path(path)
        if source.stat().st_size > 1048576:
            raise ValueError("预设 JSON 不应超过 1 MiB；请确认未选择数值数据文件。")
        document = self._validate_preset(json.loads(source.read_text(encoding="utf-8-sig")))
        options = document["options"]
        self._updating = True
        try:
            self.format.setCurrentIndex(self.format.findData(options["format"]))
            if options["format"] == "tfs":
                self.reconstruction.setCurrentIndex(self.reconstruction.findData(options["reconstruction"]))
                self._data_kind = None
            else:
                convention = options["convention"]
                self.component.setCurrentIndex(self.component.findData(options["component"]))
                self.axis.setCurrentIndex(self.axis.findData(convention["axis"]))
                self.axis_unit.clear()
                self.axis_unit.addItems(["s", "ms", "us", "ns", "ps"] if convention["axis"] == "time" else ["m", "cm", "mm"])
                self.axis_unit.setCurrentText(convention["axis_unit"])
                self.distance_convention.setCurrentIndex(self.distance_convention.findData(convention["distance_convention"]))
                self.value_unit.setCurrentText(convention["value_unit"])
                self.reference_beta.setText(str(convention["reference_beta"]))
                self.integrated.setChecked(convention["integrated"])
                self.length.setText("" if options["length"] is None else str(options["length"]))
                self.positive_trailing.setChecked(convention["positive_trailing"])
                self.positive_loss.setChecked(convention["longitudinal_positive_loss"])
                self.delimiter.setCurrentIndex(self.delimiter.findData(options["delimiter"]))
                self.axis_column.setValue(options["axis_column"])
                self.value_column.setValue(options["value_column"])
                self.skiprows.setValue(options["skiprows"])
                self.causal.setChecked(options["causal"])
                self._unit_order = int(options["component"].startswith(("dipolar", "quadrupolar"))) + int(not convention["integrated"])
                self._data_kind = "wake_function"
            self.preset_name.setText(document["name"])
            self._preset_active = True
        finally:
            self._updating = False
        self._update_controls()
        self._invalidate_preview()
        self.unit_notice.setText("已加载完整导入约定，请核对与所选源文件一致；源路径和内容哈希不属于预设。")
        self.status.setText(f"已加载预设：{document['name']}。请选择对应文件并重新预览。")
        return document

    def save_preset(self):
        path, _ = QFileDialog.getSaveFileName(self,
                                              "保存尾场导入预设",
                                              file_dialog_directory(self),
                                              "导入预设 (*.json)",
                                              options=QFileDialog.DontConfirmOverwrite)
        if not path:
            return
        destination = Path(path).with_suffix(".json")
        overwrite = destination.exists()
        if overwrite and QMessageBox.question(self, "覆盖已有预设", f"是否覆盖预设文件？\n{destination}", QMessageBox.Yes | QMessageBox.Cancel,
                                              QMessageBox.Cancel) != QMessageBox.Yes:
            return
        try:
            self.save_preset_file(destination, overwrite=overwrite)
            self.status.setText(f"已保存预设：{destination}")
        except (ValueError, TypeError, OSError) as exc:
            QMessageBox.warning(self, "无法保存预设", str(exc))

    def load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载尾场导入预设", file_dialog_directory(self), "导入预设 (*.json)")
        if not path:
            return
        try:
            self.load_preset_file(path)
        except (ValueError, TypeError, OSError) as exc:
            QMessageBox.warning(self, "无法加载预设", str(exc))

    def preview(self):
        try:
            options = self.options()
        except ValueError as exc:
            QMessageBox.warning(self, "输入无效", str(exc))
            return
        self._invalidate_preview()
        self._start_job("wake_preview", options=options)

    def export(self):
        if self._preview_options is None or self.busy or self.format.currentData() == "tfs":
            return
        source = Path(self.source.text())
        suggested = source.with_name(f"{source.stem}_{self.component.currentData()}.tfs")
        path, _ = QFileDialog.getSaveFileName(self, "导出标准尾场 TFS", str(suggested), "尾场 TFS (*.tfs)", options=QFileDialog.DontConfirmOverwrite)
        if not path:
            return
        destination = Path(path).with_suffix(".tfs")
        if destination.resolve() == source.resolve():
            QMessageBox.warning(self, "目标无效", "输出文件必须与源文件不同。")
            return
        overwrite = destination.exists()
        if overwrite:
            answer = QMessageBox.question(self, "覆盖已有文件", f"已存在：\n{destination}\n\n是否用当前已预览的尾场覆盖？", QMessageBox.Yes | QMessageBox.Cancel,
                                          QMessageBox.Cancel)
            if answer != QMessageBox.Yes:
                return
        self._start_job("wake_convert",
                        options=self._preview_options,
                        destination=str(destination),
                        expected_sha256=self._preview_sha256,
                        overwrite=overwrite)

    def _set_busy(self, busy):
        self.options_panel.setEnabled(not busy)
        self.open_button.setEnabled(not busy)
        self.load_preset_button.setEnabled(not busy)
        self.save_preset_button.setEnabled(not busy)
        self.preset_name.setEnabled(not busy)
        self.preview_button.setEnabled(not busy and bool(self.source.text()))
        self.export_button.setEnabled(not busy and self._preview_options is not None and self.format.currentData() != "tfs")
        self.copy_config_button.setEnabled(not busy and self._component_config is not None)
        self.cancel_button.setEnabled(busy)
        self.busy_changed.emit(busy)

    def _start_job(self, action, **kwargs):
        if self.busy:
            return
        directory = QTemporaryDir()
        if not directory.isValid():
            QMessageBox.warning(self, "无法创建任务", "无法创建临时工作目录。")
            return
        self._job_directory = directory
        self._request = {"action": action, "source": self.source.text(), **kwargs}
        request_path = Path(directory.path()) / "request.json"
        request_path.write_text(json.dumps(self._request, ensure_ascii=False), encoding="utf-8")
        self._job_result = None
        self._buffer = self._stderr = ""
        self._cancelled = False
        process = QProcess(self)
        self.process = process
        process.setProgram(sys.executable)
        process.setArguments(["-u", "-X", "utf8", "-m", "PASS.gui.conversion_worker", str(request_path)])
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[2]))
        process.readyReadStandardOutput.connect(self._read_output)
        process.readyReadStandardError.connect(self._read_error)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self._set_busy(True)
        self.status.setText("正在转换并校验尾场…" if action == "wake_preview" else "正在导出标准尾场 TFS…")
        process.start()

    def _read_output(self):
        self._buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            try:
                self._job_result = json.loads(line)
            except ValueError:
                self._stderr = (self._stderr + line + "\n")[-12000:]

    def _read_error(self):
        self._stderr = (self._stderr + bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace"))[-12000:]

    def _process_error(self, error):
        if error == QProcess.FailedToStart:
            self._finished(-1, QProcess.CrashExit)

    def _finished(self, code, _exit_status):
        if self.process is None:
            return
        self._read_output()
        self._read_error()
        self.process.deleteLater()
        self.process = None
        self._job_directory = None
        self._set_busy(False)
        message = self._job_result or {}
        if self._cancelled and "result" not in message:
            self.status.setText("任务已取消。")
            return
        if code != 0 or "error" in message or "result" not in message:
            error = message.get("error", self._stderr or f"任务失败（退出码 {code}）")
            self._invalidate_preview()
            self.status.setText(error)
            QMessageBox.warning(self, "尾场导入失败", error)
            return
        result = message["result"]
        if self._request["action"] == "wake_preview":
            self._show_preview(result)
        else:
            self.status.setText(f"已导出：{self._request['destination']}")
            self.metadata.setPlainText(json.dumps(result, ensure_ascii=False, indent=2))
            self._set_component_config(result.get("component_config"), result)

    def _show_preview(self, result):
        self._data_kind = result["data_kind"]
        if self.format.currentData() == "tfs":
            self._show_tfs_convention(result)
        self._preview_options = self._request["options"]
        self._preview_sha256 = result["sha256"]
        self.axes.clear()
        spectrum = result["data_kind"] == "impedance"
        if spectrum:
            self.axes.plot(result["columns"]["FREQUENCY"], result["columns"]["REAL"], linewidth=1.1, label="Re Z")
            self.axes.plot(result["columns"]["FREQUENCY"], result["columns"]["IMAG"], linewidth=1.1, label="Im Z")
            self.axes.set_xlabel("Frequency [Hz]")
            self.axes.set_ylabel(f"Z [{result['units']['REAL']}]")
            self.axes.legend()
        else:
            self.axes.plot(result["columns"]["TAU"], result["columns"]["W"], linewidth=1.1)
            self.axes.set_xlabel("Delay τ [s]")
            self.axes.set_ylabel(f"W [{result['units']['W']}]")
        self.axes.set_title(result["component"])
        self.axes.grid(True, alpha=.25)
        self.canvas.draw_idle()
        self.metadata.setPlainText(json.dumps({k: v for k, v in result.items() if k != "columns"}, ensure_ascii=False, indent=2))
        diagnostics = result["diagnostics"]
        axis_unit = result["units"]["FREQUENCY" if spectrum else "TAU"]
        value_unit = result["units"]["REAL" if spectrum else "W"]
        symbol = "Z" if spectrum else "W"
        self.diagnostics.setText(f"全表 |{symbol}| 最大值：{diagnostics['max_abs_value']:.6g} {value_unit}；"
                                 f"末点 / 峰值：{diagnostics['tail_relative_amplitude']:.3%}\n"
                                 f"范围：[{diagnostics['axis_min']:.6g}, {diagnostics['axis_max']:.6g}] {axis_unit}；"
                                 f"采样间隔：{diagnostics['min_spacing']:.6g}–{diagnostics['max_spacing']:.6g} {axis_unit}")
        self._set_component_config(result.get("component_config"), result)
        notices = "\n".join(result.get("notices", []))
        details = "标准 TFS 只读复核，可复制分量配置。" if self.format.currentData() == "tfs" else "导出保留全部原始采样点。"
        self.status.setText(f"已校验 {result['rows']:,} 个采样点，曲线显示 {result['preview_rows']:,} 点。{details}" + ("\n" + notices if notices else ""))
        self.tabs.setCurrentIndex(0)
        self.export_button.setEnabled(self.format.currentData() != "tfs")

    def _show_tfs_convention(self, result):
        convention = result["metadata"]["convention"]
        self._updating = True
        try:
            self.component.setCurrentIndex(self.component.findData(result["component"]))
            self.axis.setCurrentIndex(self.axis.findData(convention["axis"]))
            self.axis_unit.clear()
            self.axis_unit.addItem(convention["axis_unit"])
            self.value_unit.clear()
            self.value_unit.addItem(convention["value_unit"])
            self.reference_beta.setText(str(convention["reference_beta"]))
            self.integrated.setChecked(convention["integrated"])
            self.positive_trailing.setChecked(convention["positive_trailing"])
            self.positive_loss.setChecked(convention["longitudinal_positive_loss"])
            self.causal.setChecked(result["metadata"].get("causal", False))
            self.length.clear()
            spatial = result["metadata"]["spatial"]
            self._unit_order = sum(spatial["source_powers"]) + sum(spatial["test_powers"])
        finally:
            self._updating = False
        self.unit_notice.setText("标准 TFS 的物理约定已由文件头填入并锁定。" + ("自定义空间幂次详见元数据与分量配置。" if result["component"] == "custom" else ""))
        self._update_controls()

    def _set_component_config(self, config, result=None):
        self._component_config = config
        self.copy_config_button.setEnabled(config is not None and not self.busy)
        self.configuration.setPlainText(json.dumps(config, ensure_ascii=False, indent=2) if config else "")
        if config is None:
            self.configuration_hint.setText("预览标准 TFS 或导出成功后，可复制此分量配置。")
            return
        notices = (result or {}).get("configuration_notices", [])
        message = "此模板仅定义一个分量，使用文件参考 β 的 fixed 速度模型。请在仿真中选择 Solver、History 和 Boundary。"
        metadata = (result or {}).get("metadata", {})
        if metadata.get("convention", {}).get("data_kind") == "impedance":
            message += " 阻抗的 Reconstruction 已采用当前选择；请核对频带及边界适用性。"
        if metadata.get("causal") is False:
            message += " 双侧尾场需要 isolated 或 periodic 边界，不能使用因果历史。"
        self.configuration_hint.setText(message + ("\n" + "\n".join(notices) if notices else ""))

    def copy_component_config(self):
        if self._component_config is not None and not self.busy:
            QApplication.clipboard().setText(json.dumps(self._component_config, ensure_ascii=False, indent=2))
            self.tabs.setCurrentIndex(2)
            self.status.setText("已复制分量 JSON。请粘贴到 WakeField 的 Components，并核对边界、历史和求解器设置。")

    def cancel_job(self):
        if self.process:
            self._cancelled = True
            self.process.kill()

    def shutdown(self):
        if self.process:
            self.cancel_job()
            self.process.waitForFinished(2000)

    def reject(self):
        if self.busy:
            self.cancel_job()
            self.status.setText("正在取消；任务结束后可关闭窗口。")
            return
        super().reject()

    def closeEvent(self, event):
        if self.busy:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
