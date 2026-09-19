"""Read-only file preview and explicit conversion selection."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

from PySide6.QtCore import QProcess, QTemporaryDir, Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTabWidget, QTableWidget,
                               QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QScrollArea, QSizePolicy)


class ConversionPage(QWidget):
    """Each job owns a subprocess and temporary directory; source files are read-only."""
    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.info = None
        self.process = None
        self._job_directory = None
        self._pending_export = None
        self._preview_signature = None
        self._preview_selection = None
        self._updating = False
        self._cancelled = False
        self._job_result = None
        self._buffer = ""
        self._stderr = ""
        self._column_types = {}
        self._sdds_check = None
        self._preview_metadata = {}
        root = QVBoxLayout(self)
        header = QHBoxLayout()
        self.open_button = QPushButton("打开数据文件…")
        self.open_button.clicked.connect(self.choose_file)
        header.addWidget(self.open_button)
        self.source_label = QLabel("支持拖入 OMC3 SDDS、HDF5、TFS、CSV；原文件只读")
        self.source_label.setWordWrap(True)
        header.addWidget(self.source_label, 1)
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.clicked.connect(self.cancel_job)
        self.cancel_button.setEnabled(False)
        header.addWidget(self.cancel_button)
        root.addLayout(header)
        splitter = QSplitter(Qt.Horizontal)
        self.structure = QTreeWidget()
        self.structure.setHeaderLabels(["数据 / 字段", "类型 / 形状"])
        self.structure.setMinimumWidth(235)
        self.structure.setColumnWidth(0, 155)
        self.structure.itemChanged.connect(self._selection_changed)
        self.structure.currentItemChanged.connect(self._show_metadata)
        source_panel = QWidget()
        source_layout = QVBoxLayout(source_panel)
        source_layout.setContentsMargins(0, 0, 0, 0)
        self.bpm_controls = QWidget()
        bpm_layout = QVBoxLayout(self.bpm_controls)
        bpm_layout.setContentsMargins(0, 0, 0, 0)
        self.bpm_search = QLineEdit()
        self.bpm_search.setPlaceholderText("搜索 BPM 名称…")
        self.bpm_search.setClearButtonEnabled(True)
        self.bpm_search.textChanged.connect(self._filter_bpms)
        bpm_layout.addWidget(self.bpm_search)
        bpm_actions = QHBoxLayout()
        self.bpm_select_all = QPushButton("全选可见")
        self.bpm_select_none = QPushButton("取消可见")
        self.bpm_select_all.clicked.connect(lambda: self._select_visible_bpms(True))
        self.bpm_select_none.clicked.connect(lambda: self._select_visible_bpms(False))
        for widget in (self.bpm_select_all, self.bpm_select_none):
            bpm_actions.addWidget(widget)
        bpm_layout.addLayout(bpm_actions)
        self.bpm_count = QLabel()
        self.bpm_count.setWordWrap(True)
        bpm_layout.addWidget(self.bpm_count)
        source_layout.addWidget(self.bpm_controls)
        source_layout.addWidget(self.structure, 1)
        splitter.addWidget(source_panel)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.selection_summary = QLabel("打开文件后显示选择范围和单位。")
        self.selection_summary.setWordWrap(True)
        self.selection_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right_layout.addWidget(self.selection_summary)
        self.tabs = QTabWidget()
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tabs.addTab(self.table, "数据预览（最多 500 行）")
        self.metadata = QPlainTextEdit()
        self.metadata.setReadOnly(True)
        self.tabs.addTab(self.metadata, "属性与说明")
        right_layout.addWidget(self.tabs, 1)
        self.preview_status = QLabel("选择数据后点击预览；预览范围不限制实际导出范围。")
        self.preview_status.setWordWrap(True)
        right_layout.addWidget(self.preview_status)
        splitter.addWidget(right)
        options = QWidget()
        options_layout = QVBoxLayout(options)
        options_layout.setContentsMargins(0, 0, 0, 0)
        form_panel = QWidget()
        form = QFormLayout(form_panel)
        options_layout.addWidget(form_panel, 0, Qt.AlignTop)
        options_layout.addStretch(1)
        self.options_form = form
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFormAlignment(Qt.AlignTop)
        self.advanced_options = QWidget()
        advanced_form = QFormLayout(self.advanced_options)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.mode = QComboBox()
        for title, value in (("表格列 / 逐圈 BPM", "columns"), ("二维矩阵 / 选定平面", "matrix"), ("网格点长表", "long")):
            self.mode.addItem(title, value)
        form.addRow("转换方式", self.mode)
        self.bunches = QListWidget()
        self.bunches.setMaximumHeight(90)
        form.addRow("束团（左侧勾选 BPM）", self.bunches)
        self.turn_start = QLineEdit("0")
        self.turn_stop = QLineEdit()
        form.addRow("起始圈（从 0 开始）", self.turn_start)
        form.addRow("结束圈（不含，空=末尾）", self.turn_stop)
        self.planes = QWidget()
        self.planes.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        plane_layout = QHBoxLayout(self.planes)
        plane_layout.setContentsMargins(0, 0, 0, 0)
        self.plane_x = QCheckBox("X")
        self.plane_y = QCheckBox("Y")
        for widget in (self.plane_x, self.plane_y):
            widget.setChecked(True)
            widget.toggled.connect(self._clear_output_fields)
            widget.toggled.connect(self._selection_changed)
            plane_layout.addWidget(widget)
        form.addRow("位置平面", self.planes)
        self.row_start = QLineEdit("0")
        self.row_stop = QLineEdit()
        self.row_step = QLineEdit("1")
        advanced_form.addRow("起始行（从 0 开始）", self.row_start)
        advanced_form.addRow("结束行（不含，空=末尾）", self.row_stop)
        advanced_form.addRow("行步长", self.row_step)
        self.indices = QLineEdit()
        self.indices.setPlaceholderText("如 0,:,:；固定第一维，保留后两维")
        form.addRow("数组切片", self.indices)
        self.axes = QLineEdit()
        self.axes.setPlaceholderText("按维度顺序，如 /slice_id,/y,/x")
        form.addRow("长表坐标路径", self.axes)
        self.filter_column = QLineEdit()
        self.filter_column.setPlaceholderText("列名；空=不筛选")
        self.filter_operator = QComboBox()
        self.filter_operator.addItems(["==", "!=", "<", "<=", ">", ">="])
        self.filter_value = QLineEdit()
        advanced_form.addRow("筛选列", self.filter_column)
        advanced_form.addRow("比较", self.filter_operator)
        advanced_form.addRow("筛选值", self.filter_value)
        self.output_columns = QLineEdit()
        self.output_columns.setPlaceholderText("空=全部；用逗号分隔预览中的列名")
        advanced_form.addRow("输出列", self.output_columns)
        self.output_fields = QListWidget()
        self.output_fields.setMaximumHeight(105)
        self.output_fields.setToolTip("首次预览后，可勾选实际输出列，再次预览确认。")
        advanced_form.addRow("勾选输出列", self.output_fields)
        self.type_column = QComboBox()
        self.type_choice = QComboBox()
        self.type_choice.addItems(["保留原类型", "string", "int64", "uint64", "float64", "bool"])
        advanced_form.addRow("类型映射：列", self.type_column)
        advanced_form.addRow("输出类型", self.type_choice)
        self.type_column.currentTextChanged.connect(self._show_type)
        self.type_choice.currentTextChanged.connect(self._set_type)
        self.parameters = QListWidget()
        self.parameters.setMaximumHeight(115)
        advanced_form.addRow("保留参数 / 属性", self.parameters)
        self.format = QComboBox()
        for title, value in (("CSV", "csv"), ("TFS", "tfs"), ("HDF5", "hdf5"), ("OMC3 SDDS（逐圈 BPM）", "sdds")):
            self.format.addItem(title, value)
        form.addRow("输出格式", self.format)
        self.tbt_columns = {}
        for name in ("BPM", "BUNCH", "TURN", "X", "Y"):
            widget = QComboBox()
            widget.currentIndexChanged.connect(self._selection_changed)
            self.tbt_columns[name] = widget
            form.addRow(f"OMC3 {name} 对应列", widget)
        self.csv_metadata = QCheckBox("同时保存 CSV 元数据文件")
        self.csv_metadata.setChecked(True)
        self.csv_metadata.toggled.connect(self._refresh_summary)
        form.addRow(self.csv_metadata)
        self.advanced_toggle = QPushButton("高级选项（展开）")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        form.addRow(self.advanced_toggle)
        form.addRow(self.advanced_options)
        self.advanced_options.hide()
        self.preview_button = QPushButton("预览所选数据")
        self.preview_button.clicked.connect(self.preview)
        self.export_button = QPushButton("另存为…")
        self.export_button.clicked.connect(self.export)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(options)
        scroll.setMinimumWidth(290)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(scroll, 1)
        actions = QHBoxLayout()
        actions.addWidget(self.preview_button)
        actions.addWidget(self.export_button)
        controls_layout.addLayout(actions)
        splitter.addWidget(controls)
        splitter.setSizes([250, 520, 310])
        root.addWidget(splitter, 1)
        self.status = QLabel("打开文件后显示数据结构。多维数组需要明确选择轴；不自动转换单位或坐标。")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.mode.currentIndexChanged.connect(self._selection_changed)
        self.mode.currentIndexChanged.connect(self._update_controls)
        self.format.currentIndexChanged.connect(self._update_controls)
        self.format.currentIndexChanged.connect(self._selection_changed)
        for widget in (self.turn_start, self.turn_stop, self.row_start, self.row_stop, self.row_step, self.indices, self.axes, self.filter_column,
                       self.filter_value, self.output_columns):
            widget.textChanged.connect(self._selection_changed)
        self.filter_operator.currentIndexChanged.connect(self._selection_changed)
        self.parameters.itemChanged.connect(self._selection_changed)
        self.bunches.itemChanged.connect(self._selection_changed)
        self.output_fields.itemChanged.connect(self._selection_changed)
        self.structure.itemChanged.connect(self._clear_output_fields)
        self.mode.currentIndexChanged.connect(self._clear_output_fields)
        self._set_busy(False)

    @property
    def busy(self):
        return self.process is not None

    def set_theme(self, theme):
        pass

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开数据文件", "", "数据文件 (*.sdds *.h5 *.hdf5 *.tfs *.csv);;所有文件 (*)")
        if path:
            self.open_path(path)

    def open_path(self, path):
        if self.busy:
            QMessageBox.information(self, "任务正在进行", "请先取消当前读取或转换任务，再打开文件。")
            return
        self.source = str(Path(path).resolve())
        self.info = None
        self._preview_signature = None
        self._preview_selection = None
        self._sdds_check = None
        self._preview_metadata = {}
        self.structure.clear()
        self.parameters.clear()
        self.bunches.clear()
        for widget in self.tbt_columns.values():
            widget.clear()
        self.output_fields.clear()
        self._column_types.clear()
        self.type_column.clear()
        self.table.clear()
        self.table.setRowCount(0)
        self.bpm_search.clear()
        self.selection_summary.setText("正在读取文件…")
        self.source_label.setText(Path(self.source).name)
        self.source_label.setToolTip(self.source)
        self._start_job("inspect")

    def _set_busy(self, busy):
        self.open_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.preview_button.setEnabled(not busy and self.info is not None)
        self.export_button.setEnabled(not busy and self._can_export())
        for widget in (self.structure, self.parameters, self.mode, self.bunches, self.turn_start, self.turn_stop, self.row_start, self.row_stop,
                       self.row_step, self.indices, self.axes, self.filter_column, self.filter_operator, self.filter_value, self.output_columns,
                       self.output_fields, self.format, self.csv_metadata, self.type_column, self.type_choice, self.planes, self.bpm_controls,
                       self.advanced_toggle, *self.tbt_columns.values()):
            widget.setEnabled(not busy)
        self.busy_changed.emit(busy)
        self._update_controls()

    def _update_controls(self, *_args):
        kind = self.info["format"] if self.info else None
        mode = self.mode.currentData()
        self.bpm_controls.setVisible(kind == "sdds")
        self.options_form.setRowVisible(self.mode, kind == "hdf5")
        for widget in (self.bunches, self.turn_start, self.turn_stop, self.planes):
            self.options_form.setRowVisible(widget, kind == "sdds")
        self.options_form.setRowVisible(self.indices, kind == "hdf5" and mode in {"matrix", "long"})
        self.options_form.setRowVisible(self.axes, kind == "hdf5" and mode == "long")
        for widget in self.tbt_columns.values():
            self.options_form.setRowVisible(widget, self.format.currentData() == "sdds")
        self.options_form.setRowVisible(self.csv_metadata, self.format.currentData() == "csv")
        for index in range(self.mode.count()):
            value = self.mode.itemData(index)
            enabled = value == "columns" or (kind == "hdf5" and value in {"matrix", "long"})
            self.mode.model().item(index).setEnabled(enabled)

    def _selection_changed(self, *_args):
        if self._updating:
            return
        self._preview_selection = None
        self._sdds_check = None
        self._preview_metadata = {}
        self.export_button.setEnabled(False)
        self.preview_status.setText("选择已改变，请重新预览后导出。预览仅展示最多 500 行。")
        self._refresh_summary()

    def _clear_output_fields(self, *_args):
        self.output_fields.clear()
        self._column_types.clear()
        self.type_column.clear()

    def _can_export(self):
        if self._preview_selection is None:
            return False
        return self.format.currentData() != "sdds" or bool(self._sdds_check and self._sdds_check["compatible"])

    def _toggle_advanced(self, checked):
        self.advanced_options.setVisible(checked)
        self._refresh_summary()

    def _bpm_items(self):

        def visit(item):
            field = item.data(0, Qt.UserRole) or {}
            if field.get("kind") == "bpm":
                yield item
            for index in range(item.childCount()):
                yield from visit(item.child(index))

        for index in range(self.structure.topLevelItemCount()):
            yield from visit(self.structure.topLevelItem(index))

    def _filter_bpms(self, *_args):
        query = self.bpm_search.text().strip().casefold()
        for item in self._bpm_items():
            name = item.data(0, Qt.UserRole)["bpm_name"]
            item.setHidden(query not in name.casefold())
        self._refresh_summary()

    def _select_visible_bpms(self, checked):
        self.structure.blockSignals(True)
        try:
            for item in self._bpm_items():
                if not item.isHidden():
                    item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
        finally:
            self.structure.blockSignals(False)
        self._clear_output_fields()
        self._selection_changed()

    def _refresh_summary(self, *_args):
        if not self.info:
            return
        advanced_active = (self.row_start.text().strip() != "0" or bool(self.row_stop.text().strip()) or self.row_step.text().strip() != "1"
                           or bool(self.filter_column.text().strip()) or bool(self.output_columns.text().strip()) or bool(self._column_types)
                           or any(self.output_fields.item(i).checkState() != Qt.Checked for i in range(self.output_fields.count()))
                           or any(self.parameters.item(i).checkState() != Qt.Checked for i in range(self.parameters.count())))
        self.advanced_toggle.setText(("高级选项 · 已启用" if advanced_active else "高级选项") + ("（收起）" if self.advanced_toggle.isChecked() else "（展开）"))
        items = list(self._bpm_items())
        selected = sum(item.checkState(0) == Qt.Checked for item in items)
        shown = sum(not item.isHidden() for item in items)
        self.bpm_count.setText(f"已选 {selected} / {len(items)}；显示 {shown}。搜索不改变已有选择。")
        metadata = self._preview_metadata or self.info.get("metadata", {})
        definitions = metadata.get("column_definitions", {})
        mapping = {name: name for name in ("X", "Y")}
        if self.format.currentData() == "sdds":
            mapping = {name: self.tbt_columns[name].currentData() for name in ("X", "Y")}
        lines = []
        check = self._sdds_check
        if check and "bpm_count" in check:
            lines.append(f"实际选择：{check['bpm_count']} BPM × {check['bunch_count']} 束团 × {check['turn_count']} 圈；"
                         f"{check['row_count']:,} 行")
        elif self.info["format"] == "sdds":
            bunches = sum(self.bunches.item(i).checkState() == Qt.Checked for i in range(self.bunches.count()))
            try:
                start = int(self.turn_start.text())
                stop = int(self.turn_stop.text()) if self.turn_stop.text().strip() else self.info["n_turns"]
                if not 0 <= start < stop <= self.info["n_turns"]:
                    raise ValueError
                lines.append(f"范围选择：{selected} BPM × {bunches} 束团 × {stop - start} 圈；圈区间 [{start}, {stop})")
            except ValueError:
                lines.append(f"范围选择：{selected} BPM × {bunches} 束团；请填写有效圈区间。")
        elif check:
            lines.append(f"实际选择：{check['row_count']:,} 行")
        field_names = {field["name"] for field in self.info["fields"] if field["kind"] == "column"}
        tbt_context = (self.info["format"] == "sdds" or self.format.currentData() == "sdds" or {"BPM", "BUNCH", "TURN"}.issubset(field_names)
                       or "omc3_tbt" in metadata)
        if tbt_context:
            units = "；".join(f"{plane}：{definitions.get(column, {}).get('units') or '未声明'}" for plane, column in mapping.items())
            lines.append(f"位置单位：{units}（数值不缩放）")
            parameters = metadata.get("parameters", {})
            kept = {self.parameters.item(i).text() for i in range(self.parameters.count()) if self.parameters.item(i).checkState() == Qt.Checked}
            stamp = parameters.get("acqStamp") if "acqStamp" in kept else None
            if self.format.currentData() == "csv" and not self.csv_metadata.isChecked():
                lines.append("CSV 元数据未启用；单位和采集时间不会随表格保存。")
            elif stamp is None or stamp == 0:
                lines.append("采集时间：未指定；转回 SDDS 时 acqStamp=0。")
            else:
                lines.append(f"采集时间：保留 {stamp} ns。")
            if check:
                if check["compatible"]:
                    lines.append("完整选择已检查：可转回 OMC3 SDDS。")
                    start, stop = check["turn_start"], check["turn_stop"]
                    if start:
                        lines.append(f"转回时圈号 [{start}, {stop}) → [0, {check['turn_count']})。")
                else:
                    reasons = {
                        "mapping": "请将 BPM、BUNCH、TURN、X、Y 映射到五个不同的源列。",
                        "missing_columns": "需要 BPM、BUNCH、TURN、X、Y 五列，两个位置平面均不可缺少。",
                        "duplicate": "存在重复的 BPM／束团／圈采样。",
                        "turn_gaps": "所选圈号不连续，不能自动补齐缺失圈。",
                        "incomplete_grid": "采样不完整，各 BPM 和束团必须覆盖相同的连续圈。",
                        "empty": "没有选中的采样。"
                    }
                    lines.append("无法转回 OMC3 SDDS：" + reasons.get(check.get("code"), check["reason"]))
            else:
                lines.append("转回 SDDS 的完整性将在预览时检查。")
        elif definitions:
            lines.append("单位：" + "；".join(f"{name}：{value.get('units') or '未声明'}" for name, value in definitions.items()))
        if advanced_active:
            lines.append("高级选择已启用：行范围、筛选、列或参数选择仍生效。")
        self.selection_summary.setText("\n".join(lines) or "选择数据后预览。")
        self.export_button.setEnabled(not self.busy and self._can_export())

    def _show_type(self, name):
        self.type_choice.blockSignals(True)
        self.type_choice.setCurrentText(self._column_types.get(name, "保留原类型"))
        self.type_choice.blockSignals(False)

    def _set_type(self, dtype):
        name = self.type_column.currentText()
        if not name:
            return
        if dtype == "保留原类型":
            self._column_types.pop(name, None)
        else:
            self._column_types[name] = dtype
        self._selection_changed()

    def _show_metadata(self, item, _previous=None):
        if item:
            self.metadata.setPlainText(json.dumps(item.data(0, Qt.UserRole) or {}, ensure_ascii=False, indent=2))

    def _checked_fields(self):
        result = []

        def visit(item):
            info = item.data(0, Qt.UserRole)
            if info and item.checkState(0) == Qt.Checked:
                result.append(info)
            for index in range(item.childCount()):
                visit(item.child(index))

        for index in range(self.structure.topLevelItemCount()):
            visit(self.structure.topLevelItem(index))
        return result

    def selection(self):
        from PASS.tool.data_conversion import DataSelection
        checked = self._checked_fields()
        mode = self.mode.currentData()
        datasets = [f["name"] for f in checked if f["kind"] == "dataset"]
        columns = None
        if self.info["format"] in {"csv", "tfs", "sdds"}:
            columns = [f["name"] for f in checked if f["kind"] == "column"]
        if self.info["format"] == "sdds":
            columns = ["BPM", "BUNCH", "TURN"] + [name for name, widget in (("X", self.plane_x), ("Y", self.plane_y)) if widget.isChecked()]
        if self.output_columns.text().strip():
            columns = [c.strip() for c in self.output_columns.text().split(",")]
        elif self.output_fields.count():
            columns = [
                self.output_fields.item(i).text() for i in range(self.output_fields.count()) if self.output_fields.item(i).checkState() == Qt.Checked
            ]
        params = [self.parameters.item(i).text() for i in range(self.parameters.count()) if self.parameters.item(i).checkState() == Qt.Checked]
        bpms, bunch_ids, turns = None, None, (0, None)
        if self.info["format"] == "sdds":
            bpms = [f["bpm_name"] for f in checked if f["kind"] == "bpm"]
            bunch_ids = [
                self.bunches.item(i).data(Qt.UserRole) for i in range(self.bunches.count()) if self.bunches.item(i).checkState() == Qt.Checked
            ]
            turns = (int(self.turn_start.text()), int(self.turn_stop.text()) if self.turn_stop.text().strip() else None)
        mapping = {}
        if self.format.currentData() == "sdds":
            mapping = {name: widget.currentData() or "" for name, widget in self.tbt_columns.items()}
        indices = [None if v.strip() == ":" else int(v.strip()) for v in self.indices.text().split(",")] if self.indices.text().strip() else []
        axes = {str(i): value.strip() for i, value in enumerate(self.axes.text().split(",")) if value.strip()}
        filters = [(self.filter_column.text().strip(), self.filter_operator.currentText(),
                    self.filter_value.text())] if self.filter_column.text().strip() else []
        return DataSelection(datasets=datasets,
                             columns=columns,
                             parameters=params,
                             bpms=bpms,
                             bunch_ids=bunch_ids,
                             turns=turns,
                             tbt_columns=mapping,
                             mode=mode,
                             indices=indices,
                             axes=axes,
                             rows=(int(self.row_start.text()), int(self.row_stop.text()) if self.row_stop.text().strip() else None,
                                   int(self.row_step.text())),
                             filters=filters,
                             column_types=dict(self._column_types))

    def preview(self):
        try:
            selection = asdict(self.selection())
            self._start_job("preview", selection=selection)
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "选择无效", str(exc))

    def export(self):
        if not self._can_export():
            return
        kind = self.format.currentData()
        extension = "h5" if kind == "hdf5" else kind
        path, _ = QFileDialog.getSaveFileName(self, "另存所选数据", str(Path(self.source).with_name(Path(self.source).stem + "_converted." + extension)),
                                              f"{kind.upper()} (*.{extension})")
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != "." + extension:
            destination = destination.with_suffix("." + extension)
        self._pending_export = {
            "destination": str(destination),
            "selection": self._preview_selection,
            "signature": self._preview_signature,
            "csv_metadata": self.csv_metadata.isChecked()
        }
        self._start_job("plan", **self._pending_export)

    def _start_job(self, action, **kwargs):
        if self.busy or not self.source:
            return
        parent = Path(kwargs["destination"]).parent if action == "convert" else None
        if parent:
            parent.mkdir(parents=True, exist_ok=True)
        directory = QTemporaryDir(str(parent / ".pass-conversion-XXXXXX")) if parent else QTemporaryDir()
        if not directory.isValid():
            QMessageBox.warning(self, "无法创建任务", "无法创建临时工作目录。")
            return
        self._job_directory = directory
        request = {"source": self.source, "action": action, **kwargs}
        if action == "convert":
            request["staging_directory"] = directory.path()
        request_path = Path(directory.path()) / "request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        self._request = request
        self._job_result = None
        self._cancelled = False
        self._buffer = self._stderr = ""
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
        self.status.setText({"inspect": "正在读取文件结构…", "preview": "正在读取所选数据…", "plan": "正在检查导出范围和目标文件…", "convert": "正在转换，可随时取消…"}[action])
        process.start()

    def _read_output(self):
        self._buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            try:
                message = json.loads(line)
            except ValueError:
                self._stderr += line + "\n"
                continue
            if "progress" in message:
                self.status.setText("正在转换：" + message["progress"])
            else:
                self._job_result = message

    def _read_error(self):
        self._stderr = (self._stderr + bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace"))[-12000:]

    def _process_error(self, error):
        if error == QProcess.FailedToStart:
            self._finished(-1, QProcess.CrashExit)

    def _finished(self, code, exit_status):
        if self.process is None:
            return
        self._read_output()
        self._read_error()
        self.process.deleteLater()
        self.process = None
        self._job_directory = None
        self._set_busy(False)
        if self._cancelled:
            self.status.setText("任务已取消。原文件保持不变。")
            return
        message = self._job_result or {}
        if code != 0 or "error" in message or "result" not in message:
            error = message.get("error", self._stderr or f"任务失败（退出码 {code}）")
            self.status.setText(error)
            QMessageBox.warning(self, "读取 / 转换失败", error)
            return
        result = message["result"]
        action = self._request["action"]
        if action == "inspect":
            self._show_structure(result)
        elif action == "preview":
            self._show_preview(result)
        elif action == "plan":
            details = "\n".join(result["paths"][:15])
            if len(result["paths"]) > 15:
                details += f"\n… 共 {len(result['paths'])} 个文件"
            selection = self._pending_export["selection"]
            names = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
            summary = (f"按当前选择导出，预览 500 行上限不限制导出。\n"
                       f"输出列：{', '.join(names)}\n"
                       f"行范围（从 0 开始，结束行不含）：{selection['rows']}\n筛选：{selection['filters'] or '无'}\n\n{details}\n\n")
            summary += self._preview_notices
            summary += "\n\n" + self.selection_summary.text()
            if result["existing"]:
                summary += f"\n\n将覆盖 {len(result['existing'])} 个已有文件。"
            answer = QMessageBox.question(self, "确认导出范围", summary, QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
            if answer == QMessageBox.Yes:
                self._start_job("convert", **self._pending_export, overwrite=bool(result["existing"]))
        else:
            self.status.setText(f"已导出 {sum(result['row_counts']):,} 行，{len(result['outputs'])} 个文件。")
            self.metadata.setPlainText("输出文件：\n" + "\n".join(result["outputs"]) + "\n\n说明：\n" + "\n".join(result["notices"]))
            self.tabs.setCurrentIndex(1)

    def _show_structure(self, info):
        self._updating = True
        self.info = info
        self.structure.clear()
        self.parameters.clear()
        self.bunches.clear()
        parents = {}
        for field in info["fields"]:
            name = field["name"]
            parent_path = name.rsplit("/", 1)[0] if "/" in name else ""
            item = QTreeWidgetItem([name.rsplit("/", 1)[-1] or name, f"{field.get('dtype', '')} {field.get('shape', '')}"])
            item.setData(0, Qt.UserRole, field)
            item.setToolTip(0, name)
            item.setToolTip(1, item.text(1))
            if parent_path in parents:
                parents[parent_path].addChild(item)
            else:
                self.structure.addTopLevelItem(item)
            if field["kind"] == "group":
                parents[name] = item
            elif field["kind"] in {"column", "dataset", "bpm"}:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                # HDF5 datasets require explicit selection, even when lengths match.
                item.setCheckState(0, Qt.Unchecked if field["kind"] == "dataset" else Qt.Checked)
                if field.get("required"):
                    item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                if info["format"] == "sdds" and field["kind"] == "column":
                    item.setHidden(True)
            if field["kind"] == "parameter":
                p = QListWidgetItem(name, self.parameters)
                p.setCheckState(Qt.Checked)
        for name in info.get("metadata", {}).get("parameters", {}):
            p = QListWidgetItem(name, self.parameters)
            p.setCheckState(Qt.Checked)
        self.structure.expandAll()
        self.mode.setCurrentIndex(0)
        self.plane_x.setChecked(True)
        self.plane_y.setChecked(True)
        self.advanced_toggle.setChecked(False)
        for bunch_id in info.get("bunch_ids", []):
            item = QListWidgetItem(str(bunch_id), self.bunches)
            item.setData(Qt.UserRole, bunch_id)
            item.setCheckState(Qt.Checked)
        source_columns = [field["name"] for field in info["fields"] if field["kind"] == "column"]
        for name, widget in self.tbt_columns.items():
            widget.clear()
            widget.addItem("请选择源数据列", "")
            for column in source_columns:
                widget.addItem(column, column)
                if column.casefold() == name.casefold():
                    widget.setCurrentIndex(widget.count() - 1)
        self.turn_start.setText("0")
        self.turn_stop.setPlaceholderText(str(info.get("n_turns", "")))
        for widget in (self.indices, self.axes, self.output_columns, self.filter_column, self.filter_value, self.row_stop, self.turn_stop):
            widget.clear()
        self.row_start.setText("0")
        self.row_step.setText("1")
        self.format.setCurrentIndex(0)
        for i in range(self.format.count()):
            self.format.model().item(i).setEnabled(info["format"] in {"csv", "tfs"} or self.format.itemData(i) in {"csv", "tfs"})
        self.metadata.setPlainText(json.dumps(info.get("metadata", {}), ensure_ascii=False, indent=2))
        self.status.setText("；".join(info.get("notices", [])) or "已读取结构，请选择数据并预览。")
        self._updating = False
        self._set_busy(False)
        self._filter_bpms()

    def _show_preview(self, result):
        if not result["tables"]:
            self.status.setText("没有可预览的数据，请检查选择范围。")
            self._preview_selection = None
            self.export_button.setEnabled(False)
            return
        table = result["tables"][0]
        if self.output_fields.count() == 0:
            self._updating = True
            for name in table["columns"]:
                item = QListWidgetItem(name, self.output_fields)
                item.setCheckState(Qt.Checked)
                item.setToolTip(table["dtypes"][name])
                if self.info["format"] == "sdds" and name in {"BPM", "BUNCH", "TURN"}:
                    item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
            self.type_column.clear()
            self.type_column.addItems([name for name in table["columns"] if self.info["format"] != "sdds" or name in {"X", "Y"}])
            self._updating = False
        self.table.clear()
        self.table.setColumnCount(len(table["columns"]))
        self.table.setHorizontalHeaderLabels(table["columns"])
        self.table.setRowCount(len(table["rows"]))
        for row, values in enumerate(table["rows"]):
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.metadata.setPlainText(json.dumps(table["metadata"], ensure_ascii=False, indent=2))
        self._preview_signature = result["signature"]
        self._preview_selection = self._request["selection"]
        self._sdds_check = result.get("sdds_check")
        self._preview_metadata = table["metadata"]
        self._preview_notices = "\n".join(table["notices"])
        self.preview_status.setText(f"{table['label']}：预览 {len(table['rows'])} 行，{len(table['columns'])} 列。导出使用完整选择范围。")
        self.status.setText(self._preview_notices or "预览完成，可另存为。")
        self.tabs.setCurrentIndex(0)
        self._refresh_summary()

    def cancel_job(self):
        if self.process:
            self._cancelled = True
            self.process.kill()

    def shutdown(self):
        if self.process:
            self.cancel_job()
            self.process.waitForFinished(2000)
