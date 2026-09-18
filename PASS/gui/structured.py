"""Domain-specific property editors. JSON syntax never appears in these controls."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import csv
import io
import math
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QValidator
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSizePolicy,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ScientificSpinBox(QDoubleSpinBox):
    """Scientific notation with round-trip float precision, including tiny values."""

    def __init__(self, value=0.0, parent=None):
        super().__init__(parent)
        self.setDecimals(323)
        self.setRange(-1e100, 1e100)
        self.setSingleStep(0.001)
        self.setValue(float(value))
        self.setMinimumWidth(70)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setKeyboardTracking(False)

    def textFromValue(self, value):
        return str(int(value)) if value.is_integer() and abs(value) < 1e16 else repr(value)

    def valueFromText(self, text):
        return float(text.strip())

    def validate(self, text, position):
        try:
            value = float(text)
            state = QValidator.Acceptable if math.isfinite(value) and self.minimum() <= value <= self.maximum() else QValidator.Invalid
        except ValueError:
            state = QValidator.Intermediate if re.fullmatch(r"[+-]?(?:\d*\.?\d*)?(?:[eE][+-]?\d*)?", text) else QValidator.Invalid
        return state, text, position

    def wheelEvent(self, event):
        event.ignore()


class IntegerSpinBox(QSpinBox):

    def __init__(self, value=0, minimum=0, maximum=2147483647, parent=None):
        super().__init__(parent)
        self.setRange(minimum, maximum)
        self.setValue(value)
        self.setKeyboardTracking(False)

    def wheelEvent(self, event):
        event.ignore()


class StructuredField(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("structuredField")
        self.setMinimumWidth(100)


@dataclass(frozen=True)
class Column:
    title: str
    integer: bool = False
    minimum: float = -1e100
    maximum: float = 1e100

    def parse(self, value):
        if isinstance(value, bool):
            raise ValueError(f"{self.title} 必须是数值")
        if self.integer:
            # Do not round fractional indices or silently accept 1.0 as a turn.
            if not re.fullmatch(r"[+-]?\d+", str(value).strip()):
                raise ValueError(f"{self.title} 必须是整数")
            number = int(value)
        else:
            number = float(value)
        if not math.isfinite(number) or not self.minimum <= number <= self.maximum:
            raise ValueError(f"{self.title} 超出有效数值范围")
        return number


class NumberDelegate(QStyledItemDelegate):

    def __init__(self, columns, parent, owner):
        super().__init__(parent)
        self.columns = columns
        self.owner = owner
        self.editor = None
        self.index = None
        self.loading = False

    def createEditor(self, parent, option, index):
        column = self.columns[index.column()]
        editor = (IntegerSpinBox(minimum=int(column.minimum), maximum=int(column.maximum), parent=parent) if column.integer else ScientificSpinBox(
            parent=parent))
        editor.setRange(column.minimum, column.maximum)
        self.editor, self.index = editor, index
        editor.destroyed.connect(lambda: self._forget_editor(editor))
        editor.lineEdit().textEdited.connect(lambda *_: self.owner.changed.emit())
        editor.valueChanged.connect(lambda *_: None if self.loading else self.owner.changed.emit())
        return editor

    def _forget_editor(self, editor):
        if self.editor is editor:
            self.editor = self.index = None

    def setEditorData(self, editor, index):
        self.loading = True
        editor.setValue(index.data(Qt.EditRole))
        editor.selectAll()
        self.loading = False

    def setModelData(self, editor, model, index):
        editor.interpretText()
        model.setData(index, editor.value(), Qt.EditRole)

    def flush(self):
        if self.editor is not None and self.index.isValid():
            state, _, _ = self.editor.validate(self.editor.lineEdit().text(), 0)
            if state != QValidator.Acceptable:
                raise ValueError("当前单元格需要完整且有效的数值")
            self.setModelData(self.editor, self.index.model(), self.index)


class NumericTable(StructuredField):
    """Typed cells, atomic spreadsheet paste, and a larger optional editing window."""

    def __init__(self, columns, rows, hint="", default_row=None, index_origin=1):
        super().__init__()
        self.columns = columns
        self.default_row = default_row or [0] * len(columns)
        self.index_origin = index_origin
        self.load_error = ""
        self.large = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(5)
        self.hint = QLabel(hint)
        self.hint.setWordWrap(True)
        self.hint.setObjectName("muted")
        root.addWidget(self.hint)
        self.table = QTableWidget(0, len(columns))
        self.table.setHorizontalHeaderLabels([column.title for column in columns])
        self.table.setItemDelegate(NumberDelegate(columns, self.table, self))
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed | QAbstractItemView.AnyKeyPressed)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setMinimumSectionSize(55)
        for index in range(len(columns)):
            self.table.setColumnWidth(index, 78 if len(columns) == 3 else 90)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._menu)
        paste = QAction("粘贴表格", self.table)
        paste.setShortcut(QKeySequence.Paste)
        paste.setShortcutContext(Qt.WidgetShortcut)
        paste.triggered.connect(self._paste_clipboard)
        self.table.addAction(paste)
        self.table.itemChanged.connect(self._edited)
        root.addWidget(self.table)
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.add_button = QPushButton("添加")
        self.add_button.clicked.connect(lambda: self.append_row())
        self.remove_button = QPushButton("删除")
        self.remove_button.clicked.connect(self.remove_selected)
        self.expand_button = QPushButton("表格窗口…")
        self.expand_button.clicked.connect(self.expand)
        for item in (self.add_button, self.remove_button, self.expand_button):
            bar.addWidget(item)
        root.addLayout(bar)
        self.count = QLabel()
        self.count.setObjectName("muted")
        root.addWidget(self.count)
        try:
            self.set_rows(rows)
        except (ValueError, TypeError) as exc:
            self.set_rows([])
            self.load_error = str(exc)
            self.count.setText("原数据无效：" + self.load_error + "；请清空后重新填写。")

    def _validated_rows(self, rows):
        result = []
        for index, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) != len(self.columns):
                raise ValueError(f"第 {index + 1} 行需要 {len(self.columns)} 个数值")
            result.append([column.parse(value) for column, value in zip(self.columns, row)])
        return result

    def set_rows(self, rows):
        rows = self._validated_rows(rows)
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            self.table.setVerticalHeaderItem(row_index, QTableWidgetItem(str(row_index + self.index_origin)))
            for column, value in enumerate(row):
                item = QTableWidgetItem()
                item.setData(Qt.EditRole, value)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row_index, column, item)
        self.table.blockSignals(False)
        self.load_error = ""
        self._edited()

    def rows(self):
        if self.load_error:
            raise ValueError(self.load_error)
        self.table.itemDelegate().flush()
        return self._validated_rows([[self.table.item(r, c).data(Qt.EditRole) for c in range(len(self.columns))]
                                     for r in range(self.table.rowCount())])

    def get_value(self):
        return self.serialize(self.rows())

    def serialize(self, rows):
        return rows

    def new_row(self, rows):
        return list(self.default_row)

    def append_row(self, values=None):
        rows = self.rows()
        self.set_rows(rows + [list(self.new_row(rows) if values is None else values)])
        self.table.setCurrentCell(self.table.rowCount() - 1, 0)

    def remove_selected(self):
        selected = {index.row() for index in self.table.selectedIndexes()}
        if selected:
            self.set_rows([row for index, row in enumerate(self.rows()) if index not in selected])

    def _edited(self, *_args):
        self.count.setText(f"{self.table.rowCount()} 行 · 双击单元格填写数值；右键可粘贴、排序或清空。")
        if not self.large:
            self.table.setFixedHeight(34 + 26 * max(1, min(4, self.table.rowCount())) + 10)
        self.changed.emit()

    def paste(self, text):
        lines = text.strip().splitlines()
        if not lines:
            return
        delimiter = "\t" if "\t" in lines[0] else "," if "," in lines[0] else None
        rows = list(csv.reader(io.StringIO(text.strip()), delimiter=delimiter)) if delimiter else [line.split() for line in lines]
        # Accept a matching header copied from the expanded editor / spreadsheet.
        if rows and rows[0] == [column.title for column in self.columns]:
            rows = rows[1:]
        self.set_rows(self.rows() + self._validated_rows(rows))

    def _paste_clipboard(self):
        try:
            self.paste(QApplication.clipboard().text())
        except (ValueError, TypeError, csv.Error) as exc:
            QMessageBox.warning(self, "无法粘贴表格", str(exc) + "。原有数据保持不变。")

    def move_row(self, delta):
        row = self.table.currentRow()
        values = self.rows()
        if 0 <= row < len(values) and 0 <= row + delta < len(values):
            values[row], values[row + delta] = values[row + delta], values[row]
            self.set_rows(values)
            self.table.setCurrentCell(row + delta, 0)

    def _menu(self, point):
        menu = QMenu(self)
        menu.addAction("粘贴表格行", self._paste_clipboard)
        menu.addAction("复制表格", lambda: QApplication.clipboard().setText("\n".join("\t".join(map(str, row)) for row in self.rows())))
        menu.addAction("上移当前行", lambda: self.move_row(-1))
        menu.addAction("下移当前行", lambda: self.move_row(1))
        menu.addAction("清空", lambda: self.set_rows([]))
        menu.exec(self.table.viewport().mapToGlobal(point))

    def expand(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("参数表格")
        dialog.resize(820, 520)
        layout = QVBoxLayout(dialog)
        editor = NumericTable(self.columns, self.rows(), self.hint.text(), self.default_row, self.index_origin)
        editor.new_row = self.new_row
        editor.expand_button.hide()
        editor.large = True
        editor.table.setMinimumHeight(320)
        editor.table.setMaximumHeight(16777215)
        editor.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(editor)
        bar = QHBoxLayout()
        amount = IntegerSpinBox(1, 1, 10000)
        add = QPushButton("添加多行")

        def append_many():
            rows = editor.rows()
            for _ in range(amount.value()):
                rows.append(editor.new_row(rows))
            editor.set_rows(rows)

        add.clicked.connect(append_many)
        paste = QPushButton("粘贴表格")
        paste.clicked.connect(editor._paste_clipboard)
        bar.addWidget(amount)
        bar.addWidget(add)
        bar.addWidget(paste)
        layout.addLayout(bar)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("应用表格")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")

        def accept():
            try:
                rows = editor.rows()
                self.serialize(rows)
                self.set_rows(rows)
                dialog.accept()
            except ValueError as exc:
                QMessageBox.warning(dialog, "参数无效", str(exc))

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()


TURN = Column("起始圈", True, 0, 2147483647)


class TurnsEditor(NumericTable):

    def __init__(self, value, total_turns, analysis=False):
        self.analysis = analysis
        self.total_turns = total_turns
        rows = []
        for item in value if isinstance(value, list) else [] if value in (0, None) else [value]:
            if analysis:
                rows.append(item)
            elif isinstance(item, int):
                rows.append([item, item, 1])
            elif isinstance(item, (list, tuple)) and len(item) == 1:
                rows.append([item[0], item[0], 1])
            else:
                rows.append(item)
        columns = [TURN, Column("结束圈（不含）" if analysis else "结束圈（含）", True, 0, 2147483647)]
        if not analysis:
            columns.append(Column("步长", True, 1, 2147483647))
        hint = ("从第 0 圈计数；分析区间不含结束圈，至少包含两圈。空表不分析。" if analysis else "从第 0 圈计数；包含结束圈。起止相同表示单圈，空表不保存。")
        super().__init__(columns, rows, hint, [0, 2] if analysis else [0, 0, 1])
        shortcuts = QHBoxLayout()
        interval = QPushButton("添加区间")
        interval.clicked.connect(lambda: self.append_row([0, max(2, self.total_turns)] if analysis else [0, max(0, self.total_turns - 1), 1]))
        all_turns = QPushButton("全部圈")
        all_turns.clicked.connect(lambda: self.set_rows([[0, max(2, self.total_turns)]] if analysis else [[0, max(0, self.total_turns - 1), 1]]))
        clear = QPushButton("不分析" if analysis else "不保存")
        clear.clicked.connect(lambda: self.set_rows([]))
        for widget in (interval, all_turns, clear):
            shortcuts.addWidget(widget)
        self.layout().addLayout(shortcuts)
        self.add_button.setText("添加区间" if analysis else "添加单圈")

    def serialize(self, rows):
        for row in rows:
            if row[1] < row[0] + (2 if self.analysis else 0):
                raise ValueError("分析区间至少包含两圈" if self.analysis else "结束圈不能小于起始圈")
        if self.analysis:
            return rows or 0
        return [[start] if start == end else [start, end, step] for start, end, step in rows]


class ParticleEditor(NumericTable):

    def __init__(self, value):
        super().__init__([Column(title) for title in ("x / m", "px / rad", "y / m", "py / rad", "z_rel / m", "dp/p")], value,
                         "每行一个粒子，顺序为 x、px、y、py、束团相对 z、dp/p。数量由行数确定；这些粒子包含在宏粒子总数内。")

    def _edited(self, *_args):
        super()._edited()
        self.count.setText(f"手动粒子数：{self.table.rowCount()} · 表格窗口支持批量添加和粘贴。")


class CoefficientsEditor(NumericTable):

    def __init__(self, value, key):
        self.coefficient_key = key
        super().__init__([Column("阶次 n", True, 0, 1000), Column(key)], list(enumerate(value)), "n=0 二极，n=1 四极，n=2 六极；数值为积分强度，单位 m⁻ⁿ。未列出的阶次取 0。")

    def new_row(self, rows):
        return [max((row[0] for row in rows), default=-1) + 1, 0.0]

    def serialize(self, rows):
        orders = [row[0] for row in rows]
        if len(set(orders)) != len(orders):
            raise ValueError("多极场阶次不能重复")
        values = [0.0] * (max(orders, default=-1) + 1)
        for order, value in rows:
            values[order] = value
        return values


class DevicesEditor(NumericTable):

    def __init__(self, value):
        super().__init__([Column("GPU 编号", True, 0, 2147483647)], [[v] for v in value], "GPU 编号从 0 开始，每行一个设备；设备数量自动计算。")

    def serialize(self, rows):
        values = [row[0] for row in rows]
        if len(set(values)) != len(values):
            raise ValueError("GPU 编号不能重复")
        return values


APERTURES = {
    "off": ((), ()),
    "default": ((), ()),
    "circle": (("半径 R / m", ), (0.01, )),
    "rectangle": (("水平半宽 / m", "垂直半高 / m"), (0.01, 0.01)),
    "ellipse": (("水平半轴 a / m", "垂直半轴 b / m"), (0.01, 0.01)),
    "rectcircle": (("矩形半宽 W / m", "矩形半高 H / m", "圆半径 R / m"), (0.01, 0.01, 0.012)),
    "rectellipse": (("矩形半宽 W / m", "矩形半高 H / m", "椭圆半轴 a / m", "椭圆半轴 b / m"), (0.01, 0.01, 0.012, 0.012)),
    "racetrack": (("中央矩形半宽 W / m", "中央矩形半高 H / m", "端部水平半轴 a / m", "端部垂直半轴 b / m"), (0.01, 0.01, 0.005, 0.01)),
    "octagon": (("水平半宽 W / m", "垂直半高 H / m", "切角长度 D / m"), (0.01, 0.01, 0.002)),
    "polygon": ((), ((-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01))),
}


class ApertureEditor(StructuredField):

    def __init__(self, kind, value):
        super().__init__()
        self.kind = ""
        self.cache = {}
        self.fields = []
        self.polygon = None
        self.load_error = ""
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.set_kind(kind, value)

    def set_kind(self, kind, initial=None):
        kind = str(kind).lower()
        if self.kind:
            try:
                self.cache[self.kind] = self._raw_value()
            except ValueError:
                pass
        while self.form.rowCount():
            self.form.removeRow(0)
        self.kind, self.fields, self.polygon = kind, [], None
        self.load_error = ""
        labels, defaults = APERTURES.get(kind, ((), ()))
        value = initial if initial is not None else self.cache.get(kind, defaults)
        if not isinstance(value, (list, tuple)):
            self.load_error = "孔径参数必须是一组尺寸；请重新填写。"
            value = defaults
        if kind in ("off", "default"):
            hint = QLabel("不限制横向孔径。" if kind == "off" else "采用默认孔径：SC 点跟随网格，内部 SC 跟随元件。")
            hint.setWordWrap(True)
            self.form.addRow(hint)
        elif kind == "polygon":
            self.polygon = NumericTable([Column("顶点 x / m"), Column("顶点 y / m")], value, "按边界顺序排列至少 3 个顶点，末点自动连接首点。", [0.0, 0.0])
            self.polygon.changed.connect(self.changed)
            self.form.addRow(self.polygon)
        else:
            if len(value) != len(labels):
                self.load_error = f"此形状需要 {len(labels)} 个尺寸；请重新确认各项数值。"
            for index, label in enumerate(labels):
                try:
                    number = Column(label).parse(value[index]) if index < len(value) else defaults[index]
                except (ValueError, TypeError):
                    number = defaults[index]
                    self.load_error = "孔径尺寸需要有限数值；请重新填写。"
                field = ScientificSpinBox(number)
                field.valueChanged.connect(self._dimension_changed)
                self.fields.append(field)
                self.form.addRow(label, field)
        self.error_label = QLabel(self.load_error)
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(bool(self.load_error))
        self.form.addRow(self.error_label)
        self.changed.emit()

    def _dimension_changed(self):
        self.load_error = ""
        self.error_label.hide()
        self.changed.emit()

    def _raw_value(self):
        if self.polygon:
            return self.polygon.rows()
        return [field.value() for field in self.fields]

    def get_value(self):
        if self.load_error:
            raise ValueError(self.load_error)
        value = self._raw_value()
        from PASS.para.schema.space_charge import validate_loss_aperture
        return validate_loss_aperture(self.kind, value)


class RangeEditor(StructuredField):

    def __init__(self, value, explicit=False):
        super().__init__()
        self.explicit = explicit
        self.active = value is not None
        root = QFormLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.enabled_box = QCheckBox("自定义动量范围")
        self.enabled_box.setChecked(self.active)
        if not explicit:
            root.addRow(self.enabled_box)
        pair = [value.get("z min", -0.1), value.get("z max", 0.1)] if explicit and isinstance(
            value, dict) else value if isinstance(value, list) and len(value) == 2 else [-0.1, 0.1] if explicit else [-1.0, 1.0]
        self.lower = ScientificSpinBox(pair[0])
        self.upper = ScientificSpinBox(pair[1])
        root.addRow("z 最小值 / m" if explicit else "最小 dp/p", self.lower)
        root.addRow("z 最大值 / m" if explicit else "最大 dp/p", self.upper)
        hint = QLabel("范围使用所选 Coordinate。arrival_phase 为 [-C, 0]；自动模式不使用显式范围。" if explicit else "取消自定义时使用引擎默认范围 −1 至 1。")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        root.addRow(hint)
        self.enabled_box.toggled.connect(self.set_active)
        self.lower.valueChanged.connect(self.changed)
        self.upper.valueChanged.connect(self.changed)
        self.set_active(self.active)

    def set_active(self, active):
        self.active = bool(active)
        self.lower.setEnabled(self.active)
        self.upper.setEnabled(self.active)
        self.changed.emit()

    def get_value(self):
        if not self.active:
            return None
        low, high = self.lower.value(), self.upper.value()
        if not low < high:
            raise ValueError("范围下限必须小于上限")
        return {"z min": low, "z max": high} if self.explicit else [low, high]


class InternalSpaceChargeEditor(StructuredField):

    def __init__(self, value, configurations, total_turns):
        super().__init__()
        from PASS.para.schema.space_charge import ElementSpaceCharge
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.enabled_box = QCheckBox("启用元件内部空间电荷")
        self.enabled_box.setChecked(isinstance(value, dict))
        root.addWidget(self.enabled_box)
        self.body = QWidget()
        form = QFormLayout(self.body)
        form.setContentsMargins(0, 0, 0, 0)
        initial = ElementSpaceCharge(Configuration=next(iter(configurations), "default")).model_dump(by_alias=True)
        if isinstance(value, dict):
            initial.update(value)
        self.configuration = QComboBox()
        self.configuration.addItems(list(dict.fromkeys([*configurations, initial["Configuration"]])))
        self.configuration.setCurrentText(initial["Configuration"])
        self.kicks = IntegerSpinBox(initial["Num kicks"], 1)
        self.kicks.setToolTip("内部 SC 节点数量；与父元件 Num slices 的输运切分不同。")
        form.addRow("计算配置", self.configuration)
        form.addRow("内部 kick 数", self.kicks)
        self.aperture_type = QComboBox()
        self.aperture_type.addItems(list(APERTURES))
        self.aperture_type.setCurrentText(initial["Aperture type"])
        self.aperture = ApertureEditor(initial["Aperture type"], initial["Aperture value"])
        form.addRow("孔径类型", self.aperture_type)
        form.addRow(self.aperture)
        hint = QLabel("内部 SC 始终继承父元件孔径；这里的旧显式孔径若冲突会被引擎覆盖。default 表示继承。")
        hint.setWordWrap(True)
        form.addRow(hint)
        self.flags = {}
        for key, title in (("Save field", "保存电场"), ("Save potential", "保存电势"), ("Save density", "保存电荷密度")):
            check = QCheckBox(title)
            check.setChecked(initial[key])
            check.toggled.connect(self.changed)
            self.flags[key] = check
            form.addRow(check)
        self.turns = TurnsEditor(initial["Save turns"], total_turns)
        form.addRow(QLabel("保存圈数"))
        form.addRow(self.turns)
        root.addWidget(self.body)
        self.enabled_box.toggled.connect(self.body.setEnabled)
        self.body.setEnabled(self.enabled_box.isChecked())
        self.enabled_box.toggled.connect(self.changed)
        self.configuration.currentTextChanged.connect(self.changed)

        def output_modes():
            resource = configurations.get(self.configuration.currentText(), {}) if isinstance(configurations, dict) else {}
            potential = self.flags["Save potential"]
            pic = resource.get("Method", "pic") == "pic"
            potential.setEnabled(pic or potential.isChecked())
            potential.setToolTip("解析求解器不提供电势；已有选项须取消，可改为保存电场或密度。" if not pic else "保存网格电势")

        self.configuration.currentTextChanged.connect(output_modes)
        self.flags["Save potential"].toggled.connect(output_modes)
        output_modes()
        self.kicks.valueChanged.connect(self.changed)
        self.aperture_type.currentTextChanged.connect(self.aperture.set_kind)
        self.aperture.changed.connect(self.changed)
        self.turns.changed.connect(self.changed)

    def get_value(self):
        if not self.enabled_box.isChecked():
            return None
        from PASS.para.schema.space_charge import ElementSpaceCharge
        value = {
            "Configuration": self.configuration.currentText(),
            "Num kicks": self.kicks.value(),
            "Aperture type": self.aperture_type.currentText(),
            "Aperture value": self.aperture.get_value(),
            "Save turns": self.turns.get_value(),
            **{
                key: item.isChecked()
                for key, item in self.flags.items()
            }
        }
        return ElementSpaceCharge.model_validate(value).model_dump(by_alias=True)


class ObjectEditor(StructuredField):
    """Expandable data is represented as permanent labeled child fields."""

    def __init__(self, value, factory, reader):
        super().__init__()
        self.original = deepcopy(value)
        self.reader = reader
        self.fields = {}
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        for key, item in value.items():
            field = factory(key, item)
            self.fields[key] = field
            form.addRow(str(key), field)

    def get_value(self):
        return {key: self.reader(key, field, self.original[key]) for key, field in self.fields.items()}


class ListEditor(NumericTable):

    def __init__(self, value):
        super().__init__([Column("数值")], [[v] for v in value], "每行一个数值，按行序保存。")

    def serialize(self, rows):
        return [row[0] for row in rows]


class ObjectListEditor(StructuredField):
    """An ordered list of labeled parameter objects, with explicit add/remove."""

    def __init__(self, value, factory, reader, default):
        super().__init__()
        self.factory, self.reader, self.default = factory, reader, deepcopy(default)
        self.entries = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.rows = QVBoxLayout()
        layout.addLayout(self.rows)
        add = QPushButton("添加分量")
        add.clicked.connect(lambda: self.add_entry(self.default))
        layout.addWidget(add)
        for item in value:
            self.add_entry(item)

    def add_entry(self, value):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        editor = ObjectEditor(value, self.factory, self.reader)
        layout.addWidget(editor)
        remove = QPushButton("移除此分量")
        layout.addWidget(remove)
        self.entries.append((panel, editor))
        self.rows.addWidget(panel)
        remove.clicked.connect(lambda: self.remove_entry(panel, editor))
        self.changed.emit()

    def remove_entry(self, panel, editor):
        self.entries.remove((panel, editor))
        self.rows.removeWidget(panel)
        panel.deleteLater()
        self.changed.emit()

    def get_value(self):
        return [editor.get_value() for _, editor in self.entries]
