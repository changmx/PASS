"""Import and generate normalized magnet strength programs without changing inputs."""

from pathlib import Path
import re

import numpy as np
from PySide6.QtCore import QEvent, QSignalBlocker, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QSplitter, QTabWidget, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from PASS.gui.plotting import PlotCanvas
from PASS.gui.widgets import file_dialog_directory


class MagnetRampingDialog(QDialog):
    """Map source columns or edit breakpoints, then export a standard TFS program."""

    def __init__(self, parent=None, *, element=None):
        super().__init__(parent)
        self._element = element
        self._element_order = {"Quadrupole": 1, "Sextupole": 2, "Octupole": 3}.get((element or {}).get("Command"))
        self.exported_path = None
        self.setWindowTitle("磁铁 ramping → 标准 TFS")
        self.resize(1050, 820)
        self._source_data = None
        self._source_signature = None
        self._source_headers = {}
        self._preview = None
        root = QVBoxLayout(self)
        note = QLabel("适用于 Quadrupole / Sextupole / Octupole / Multipole。输入为绝对归一化强度；Kₙ 的单位为 m⁻⁽ⁿ⁺¹⁾，KₙL 为 m⁻ⁿ。\n"
                      "时间统一导出为秒；断点之间线性插值，区间外保持端点值。未提供的分量保留元件静态值。")
        note.setWordWrap(True)
        root.addWidget(note)
        splitter = QSplitter(Qt.Vertical)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_import_tab(), "导入 CSV / TXT / TFS")
        self.tabs.addTab(self._create_generator_tab(), "按时间断点生成")
        splitter.addWidget(self.tabs)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        controls = QHBoxLayout()
        self.preview_button = QPushButton("预览并校验")
        self.preview_button.clicked.connect(self.preview)
        self.preview_component = QComboBox()
        self.preview_component.currentTextChanged.connect(self._plot_component)
        controls.addWidget(self.preview_button)
        controls.addWidget(QLabel("曲线分量"))
        controls.addWidget(self.preview_component, 1)
        preview_layout.addLayout(controls)
        self.plot = PlotCanvas()
        self.plot.empty_message = "配置导入映射或编辑断点后，点击预览并校验"
        preview_layout.addWidget(self.plot, 1)
        splitter.addWidget(preview)
        splitter.setSizes([420, 250])
        root.addWidget(splitter, 1)
        self.status = QLabel(
            "成功导出后回填当前元件的 Ramping file 并启用 Is ramping；原分量文件字段将清空。仍需应用或插入元件。" if element else "导出后，在磁铁参数中勾选 Is ramping 并选择 Ramping file。不会自动修改当前仿真配置。")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.status)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.export_button = QPushButton("导出并用于当前元件…" if element else "导出 ramping TFS…")
        self.export_button.clicked.connect(self.export)
        self.export_button.setEnabled(False)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.reject)
        actions.addWidget(self.export_button)
        actions.addWidget(close_button)
        root.addLayout(actions)
        self.tabs.currentChanged.connect(self._invalidate_preview)
        if element:
            self._initialize_element(element)

    def _create_import_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        files = QHBoxLayout()
        self.source = QLineEdit()
        self.source.setReadOnly(True)
        self.source.setPlaceholderText("源文件保持只读")
        open_button = QPushButton("选择文件…")
        open_button.clicked.connect(self.choose_file)
        self.load_button = QPushButton("读取列")
        self.load_button.clicked.connect(self.load_source)
        files.addWidget(self.source, 1)
        files.addWidget(open_button)
        files.addWidget(self.load_button)
        layout.addLayout(files)
        options = QHBoxLayout()
        self.delimiter = QComboBox()
        for label, value in (("逗号", ","), ("空白", r"\s+"), ("制表符", "\t"), ("分号", ";")):
            self.delimiter.addItem(label, value)
        self.has_header = QCheckBox("首行为列名")
        self.has_header.setChecked(True)
        self.skiprows = QSpinBox()
        self.skiprows.setRange(0, 1000000)
        options.addWidget(QLabel("分隔符"))
        options.addWidget(self.delimiter)
        options.addWidget(self.has_header)
        options.addWidget(QLabel("跳过开头行数"))
        options.addWidget(self.skiprows)
        options.addStretch(1)
        layout.addLayout(options)
        time_options = QHBoxLayout()
        self.time_column = QComboBox()
        self.time_unit = QComboBox()
        for label, value in (("s", 1.), ("ms", 1e-3), ("μs", 1e-6), ("ns", 1e-9)):
            self.time_unit.addItem(label, value)
        time_options.addWidget(QLabel("时间列"))
        time_options.addWidget(self.time_column, 1)
        time_options.addWidget(QLabel("源时间单位"))
        time_options.addWidget(self.time_unit)
        layout.addLayout(time_options)
        self.mapping = QTableWidget(0, 2)
        self.mapping.setHorizontalHeaderLabels(["源强度列", "输出分量（空=忽略）"])
        self.mapping.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mapping.setToolTip("例如 K1L / K1SL，或 K2 / K2S。可直接输入任意非负阶数；同一分量不能同时提供 K 和 KL。")
        layout.addWidget(self.mapping, 1)
        notice = QLabel("源强度必须已经是归一化 K 或 KL（SI 单位）；不从磁场、电流或圈数推算强度与时间。TFS 自动读取列名。")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        for widget in (self.delimiter, self.time_column, self.time_unit):
            widget.currentIndexChanged.connect(self._invalidate_preview)
        self.has_header.toggled.connect(self._invalidate_preview)
        self.skiprows.valueChanged.connect(self._invalidate_preview)
        return panel

    def _create_generator_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        components = QHBoxLayout()
        self.order = QSpinBox()
        self.order.setRange(0, 1000)
        self.order.setValue(1)
        self.kind = QComboBox()
        self.kind.addItems(["normal", "skew"])
        self.strength = QComboBox()
        self.strength.addItems(["KL（积分强度）", "K（单位长度强度）"])
        add_component = QPushButton("增加分量")
        add_component.clicked.connect(self.add_component)
        remove_component = QPushButton("删除选中分量")
        remove_component.clicked.connect(self.remove_component)
        for widget in (QLabel("阶数 n"), self.order, self.kind, self.strength, add_component, remove_component):
            components.addWidget(widget)
        layout.addLayout(components)
        template = QGroupBox("上升–平台–下降模板（应用于选中的强度列，保留其他分量及原有时间断点）")
        template_layout = QGridLayout(template)
        self.template_fields = {}
        definitions = (("start", "开始时间 (s)", "0"), ("rise", "上升时长 (s)", "0.1"), ("hold", "平台时长 (s)", "0.2"), ("fall", "下降时长 (s)", "0.1"),
                       ("initial", "起始强度", "0"), ("plateau", "平台强度", "0.2"), ("final", "结束强度", "0"))
        for index, (key, label, value) in enumerate(definitions):
            row, column = divmod(index, 4)
            field = QLineEdit(value)
            field.setMinimumWidth(65)
            self.template_fields[key] = field
            template_layout.addWidget(QLabel(label), row * 2, column)
            template_layout.addWidget(field, row * 2 + 1, column)
        self.template_button = QPushButton("应用到选中分量")
        self.template_button.clicked.connect(self.apply_template)
        template_layout.addWidget(self.template_button, 3, 3)
        layout.addWidget(template)
        self.breakpoints = QTableWidget(2, 2)
        self.breakpoints.setHorizontalHeaderLabels(["TIME (s)", "K1L"])
        self.breakpoints.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.breakpoints.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.breakpoints.installEventFilter(self)
        for row, values in enumerate((("0", "0"), ("1", "0.2"))):
            for column, value in enumerate(values):
                self.breakpoints.setItem(row, column, QTableWidgetItem(value))
        self.breakpoints.setCurrentCell(0, 1)
        self.breakpoints.itemChanged.connect(self._invalidate_preview)
        layout.addWidget(self.breakpoints, 1)
        rows = QHBoxLayout()
        add_row = QPushButton("增加断点")
        add_row.clicked.connect(self.add_breakpoint)
        remove_rows = QPushButton("删除选中行")
        remove_rows.clicked.connect(self.remove_breakpoints)
        rows.addWidget(add_row)
        rows.addWidget(remove_rows)
        self.paste_button = QPushButton("粘贴单元格 (Ctrl+V)")
        self.paste_button.clicked.connect(self.paste_cells)
        rows.addWidget(self.paste_button)
        rows.addStretch(1)
        layout.addLayout(rows)
        notice = QLabel("时间必须严格递增；单个断点表示常数。K 需要正元件长度。粘贴 Excel/TSV 数值矩形时自动增加行；请先添加所需强度列。")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        return panel

    def _initialize_element(self, element):
        """Start from the current strength draft without changing the active element."""
        if self._element_order is not None:
            self.order.setValue(self._element_order)
            self.order.setEnabled(False)
            names = [f"K{self._element_order}L", f"K{self._element_order}SL"]
            columns = {name: [float(element.get(name, 0.))] * 2 for name in names}
        else:
            columns = {}
            for key, suffix in (("KiL", "L"), ("KiSL", "SL")):
                for order, value in enumerate(element.get(key, [])):
                    if value:
                        columns[f"K{order}{suffix}"] = [float(value)] * 2
            if not columns:
                columns = {"K1L": [0., 0.]}
        self._set_breakpoints([0., 1.], columns)
        initial = float(next(iter(columns.values()))[0])
        for key, value in (("initial", initial), ("plateau", initial * 1.2 if initial else .2), ("final", initial)):
            self.template_fields[key].setText(str(float(value)))
        self.tabs.setCurrentIndex(1)
        self.breakpoints.setCurrentCell(0, 1)

    def _set_breakpoints(self, times, columns):
        with QSignalBlocker(self.breakpoints):
            self.breakpoints.setRowCount(len(times))
            self.breakpoints.setColumnCount(len(columns) + 1)
            self.breakpoints.setHorizontalHeaderLabels(["TIME (s)", *columns])
            for column, values in enumerate((times, *columns.values())):
                for row, value in enumerate(values):
                    self.breakpoints.setItem(row, column, QTableWidgetItem(str(float(value))))
        self._invalidate_preview()

    def eventFilter(self, watched, event):
        if watched is self.breakpoints and event.type() == QEvent.KeyPress and event.matches(QKeySequence.Paste):
            self.paste_cells()
            return True
        return super().eventFilter(watched, event)

    def paste_cells(self):
        """Paste a numeric rectangle atomically; column meanings remain explicit."""
        try:
            text = QApplication.clipboard().text().strip("\r\n")
            if not text:
                return
            rows = [line.split("\t") for line in text.splitlines()]
            width = len(rows[0])
            if any(len(row) != width for row in rows):
                raise ValueError("请粘贴以制表符分隔的矩形数值区域，不含表头。")
            values = np.asarray(rows, dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError("粘贴区域必须仅包含有限数值。")
            selected = self.breakpoints.selectedIndexes()
            start_row = min((index.row() for index in selected), default=max(0, self.breakpoints.currentRow()))
            start_column = min((index.column() for index in selected), default=max(0, self.breakpoints.currentColumn()))
            if start_column + width > self.breakpoints.columnCount():
                raise ValueError("粘贴区域超出已有列；请先增加并命名所需强度分量。")
            with QSignalBlocker(self.breakpoints):
                self.breakpoints.setRowCount(max(self.breakpoints.rowCount(), start_row + len(rows)))
                for row, items in enumerate(rows):
                    for column, value in enumerate(items):
                        self.breakpoints.setItem(start_row + row, start_column + column, QTableWidgetItem(value.strip()))
            self._invalidate_preview()
        except (ValueError, TypeError) as error:
            QMessageBox.warning(self, "无法粘贴断点", str(error))

    def apply_template(self):
        try:
            values = {key: float(field.text()) for key, field in self.template_fields.items()}
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError("模板参数必须是有限数值。")
            if values["rise"] <= 0 or values["fall"] <= 0 or values["hold"] < 0:
                raise ValueError("上升和下降时长必须大于零；平台时长不能为负。")
            times, columns = self._read_program()
            column = self.breakpoints.currentColumn()
            if column < 1:
                raise ValueError("请先在表格中选中要生成模板的强度列。")
            name = self.breakpoints.horizontalHeaderItem(column).text()
            start = values["start"]
            top = start + values["rise"]
            down = top + values["hold"]
            end = down + values["fall"]
            if not np.isfinite(end) or top <= start or end <= down or (values["hold"] > 0 and down <= top):
                raise ValueError("模板时间精度不足或溢出，请调整时间起点和时长。")
            knots = [start, top, down, end] if values["hold"] else [start, top, end]
            amplitudes = [values["initial"], values["plateau"], values["plateau"], values["final"]
                          ] if values["hold"] else [values["initial"], values["plateau"], values["final"]]
            output_times = np.union1d(times, knots)
            output = {key: np.interp(output_times, times, samples) for key, samples in columns.items()}
            output[name] = np.interp(output_times, knots, amplitudes)
            self._set_breakpoints(output_times, output)
            self.breakpoints.setCurrentCell(0, column)
            self.preview()
            self.preview_component.setCurrentText(name)
        except (ValueError, TypeError) as error:
            QMessageBox.warning(self, "无法生成模板", str(error))

    def _invalidate_preview(self, *_args):
        self._preview = None
        if hasattr(self, "export_button"):
            self.export_button.setEnabled(False)
            self.status.setText("设置已修改，请重新预览并校验。")
            self.plot.set_series([], [], empty_message="设置已修改，请重新预览并校验")

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择磁铁 ramping 源文件", file_dialog_directory(self), "数值表 (*.csv *.txt *.dat *.tfs);;所有文件 (*)")
        if path:
            self.source.setText(path)
            if Path(path).suffix.lower() in {".txt", ".dat"}:
                self.delimiter.setCurrentIndex(1)
            elif Path(path).suffix.lower() == ".csv":
                self.delimiter.setCurrentIndex(0)
            self.load_source()

    def _import_signature(self):
        return self.source.text(), self.delimiter.currentData(), self.has_header.isChecked(), self.skiprows.value()

    def load_source(self):
        self._invalidate_preview()
        try:
            from PASS.para.tools.ramping import read_magnet_ramping_source
            path = Path(self.source.text())
            if not path.is_file():
                raise ValueError("请选择存在的源文件。")
            is_tfs = path.suffix.lower() == ".tfs"
            data = read_magnet_ramping_source(path,
                                              delimiter=self.delimiter.currentData(),
                                              skiprows=self.skiprows.value(),
                                              header=0 if self.has_header.isChecked() else None)
            if len(data.columns) < 2 or len(data) == 0:
                raise ValueError("至少需要时间列、一列强度和一个数据点。")
            labels = [str(name) if self.has_header.isChecked() or is_tfs else f"第 {index + 1} 列" for index, name in enumerate(data.columns)]
            self._source_data = data
            self._source_signature = self._import_signature()
            self._source_headers = {str(key).upper(): value for key, value in data.headers.items()}
            self.time_column.clear()
            for index, label in enumerate(labels):
                self.time_column.addItem(label, index)
            time_index = next((index for index, label in enumerate(labels) if label.upper() in {"TIME", "TIME_S"}), 0)
            self.time_column.setCurrentIndex(time_index)
            self.mapping.setRowCount(len(labels))
            for index, label in enumerate(labels):
                item = QTableWidgetItem(label)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.mapping.setItem(index, 0, item)
                target = QComboBox()
                target.setEditable(True)
                target.addItem("")
                target.addItems([f"K{order}{suffix}" for order in range(4) for suffix in ("L", "SL", "", "S")])
                if re.fullmatch(r"K\d+S?L?", label.upper()):
                    target.setCurrentText(label.upper())
                target.currentTextChanged.connect(self._invalidate_preview)
                self.mapping.setCellWidget(index, 1, target)
            if is_tfs:
                unit = str(self._source_headers.get("TIME_UNIT", "s")).lower()
                unit = "μs" if unit in {"us", "µs"} else unit
                unit_index = self.time_unit.findText(unit)
                if unit_index < 0:
                    raise ValueError(f"TFS TIME_UNIT={unit!r} 不受支持；请先转换为物理时间。")
                convention = str(self._source_headers.get("STRENGTH_CONVENTION", "normalized")).lower()
                if convention != "normalized":
                    raise ValueError("源 TFS 不是 normalized 强度；请先转换为归一化 K 或 KL。")
                self.time_unit.setCurrentIndex(unit_index)
            self.time_unit.setEnabled("TIME_UNIT" not in self._source_headers)
            self.time_unit.setToolTip("源 TFS 已声明 TIME_UNIT，按该单位转换。" if "TIME_UNIT" in self._source_headers else "明确选择源时间列单位。")
            for widget in (self.delimiter, self.has_header, self.skiprows):
                widget.setEnabled(not is_tfs)
            self.status.setText(f"已读取 {len(data)} 行、{len(labels)} 列。请确认时间单位与强度列映射，再预览。")
        except Exception as error:
            self._source_data = None
            self._source_signature = None
            QMessageBox.warning(self, "读取失败", str(error))

    def add_component(self):
        name = f"K{self.order.value()}{'S' if self.kind.currentText() == 'skew' else ''}{'L' if self.strength.currentIndex() == 0 else ''}"
        names = [self.breakpoints.horizontalHeaderItem(column).text() for column in range(1, self.breakpoints.columnCount())]
        if any(existing.removesuffix("L") == name.removesuffix("L") for existing in names):
            QMessageBox.warning(self, "分量已存在", "同一 normal/skew 分量只能使用 K 或 KL 中的一种。")
            return
        column = self.breakpoints.columnCount()
        self.breakpoints.insertColumn(column)
        self.breakpoints.setHorizontalHeaderItem(column, QTableWidgetItem(name))
        for row in range(self.breakpoints.rowCount()):
            self.breakpoints.setItem(row, column, QTableWidgetItem("0"))
        self._invalidate_preview()

    def remove_component(self):
        column = self.breakpoints.currentColumn()
        if column > 0:
            self.breakpoints.removeColumn(column)
            self._invalidate_preview()

    def add_breakpoint(self):
        row = self.breakpoints.rowCount()
        time = 0.
        if row:
            try:
                time = float(self.breakpoints.item(row - 1, 0).text()) + 1.
            except (AttributeError, ValueError):
                time = float(row)
        self.breakpoints.insertRow(row)
        for column in range(self.breakpoints.columnCount()):
            previous = self.breakpoints.item(row - 1, column) if row else None
            value = f"{time:.12g}" if column == 0 else previous.text() if previous else "0"
            self.breakpoints.setItem(row, column, QTableWidgetItem(value))
        self._invalidate_preview()

    def remove_breakpoints(self):
        rows = sorted({index.row() for index in self.breakpoints.selectedIndexes()}, reverse=True)
        for row in rows:
            self.breakpoints.removeRow(row)
        self._invalidate_preview()

    def _read_program(self):
        headers = {}
        if self.tabs.currentIndex() == 0:
            if self._source_data is None or self._source_signature != self._import_signature():
                raise ValueError("请先按当前设置重新读取源文件列。")
            data = self._source_data
            time_column = self.time_column.currentData()
            times = data.iloc[:, time_column].to_numpy()
            columns = {}
            for index in range(self.mapping.rowCount()):
                target = self.mapping.cellWidget(index, 1).currentText().strip().upper()
                if not target:
                    continue
                if index == time_column:
                    raise ValueError("时间列不能同时映射为磁铁强度。")
                if target in columns:
                    raise ValueError(f"多个源列映射到同一分量 {target}。")
                columns[target] = data.iloc[:, index].to_numpy()
                unit_key = f"{str(data.columns[index]).upper()}_UNIT"
                if unit_key in self._source_headers:
                    headers[f"{target}_UNIT"] = self._source_headers[unit_key]
        else:
            data = np.empty((self.breakpoints.rowCount(), self.breakpoints.columnCount()))
            for row in range(data.shape[0]):
                for column in range(data.shape[1]):
                    item = self.breakpoints.item(row, column)
                    if item is None or not item.text().strip():
                        raise ValueError(f"第 {row + 1} 行第 {column + 1} 列不能为空。")
                    data[row, column] = float(item.text())
            times = data[:, 0]
            columns = {self.breakpoints.horizontalHeaderItem(column).text(): data[:, column] for column in range(1, data.shape[1])}
        from PASS.para.tools.ramping import validate_magnet_ramping
        if self.tabs.currentIndex() == 0:
            times, columns = validate_magnet_ramping(times, columns, headers)
            times *= self.time_unit.currentData()
        times, columns = validate_magnet_ramping(times, columns, headers)
        if self._element:
            from PASS.utils.constants import const
            for name in columns:
                order = int(re.search(r"\d+", name).group())
                if self._element_order is not None and order != self._element_order:
                    raise ValueError(f"当前 {self._element['Command']} 只能使用 n={self._element_order} 的强度分量，不能使用 {name}。")
                if not name.endswith("L") and self._element["Length (m)"] <= const.eps:
                    raise ValueError("当前元件为薄磁铁；必须使用 KL 积分强度列。")
        return times, columns

    def preview(self):
        try:
            self._preview = self._read_program()
            times, columns = self._preview
            self.preview_component.clear()
            self.preview_component.addItems(columns)
            self._plot_component()
            self.status.setText(f"校验通过：{len(times)} 个断点；时间 {times[0]:.8g} — {times[-1]:.8g} s；分量 {', '.join(columns)}。")
            self.export_button.setEnabled(True)
        except Exception as error:
            self._invalidate_preview()
            QMessageBox.warning(self, "ramping 数据无效", str(error))

    def _plot_component(self, *_args):
        if self._preview is None:
            return
        times, columns = self._preview
        name = self.preview_component.currentText()
        if name in columns:
            order = int(re.search(r"\d+", name).group())
            exponent = order if name.endswith("L") else order + 1
            self.plot.set_series(times, columns[name], x_label="Time [s]", y_label=f"{name} [m^(-{exponent})]")

    def export(self):
        try:
            times, columns = self._read_program()
        except Exception as error:
            QMessageBox.warning(self, "ramping 数据无效", str(error))
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出磁铁 ramping", str(Path(file_dialog_directory(self)) / "magnet_ramping.tfs"), "TFS (*.tfs)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".tfs"
            if Path(path).exists():
                answer = QMessageBox.question(self, "覆盖文件", f"{path}\n已存在，是否覆盖？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if answer != QMessageBox.Yes:
                    return
        try:
            if self.source.text():
                source, output_path = Path(self.source.text()), Path(path)
                same_source = output_path.resolve() == source.resolve()
                if output_path.exists() and source.exists():
                    same_source = same_source or output_path.samefile(source)
                if same_source:
                    QMessageBox.warning(self, "保留源文件", "请使用不同的输出文件名，源文件保持只读。")
                    return
            from PASS.para.tools.ramping import write_magnet_ramping
            output = write_magnet_ramping(path, times, columns)
            self.exported_path = str(Path(output).resolve())
            self.status.setText(f"已导出 {output}。在磁铁参数中启用 Is ramping 并选择此 Ramping file。")
            if self._element:
                self.accept()
        except Exception as error:
            QMessageBox.warning(self, "导出失败", str(error))
