"""Schema-aware editors for nested physics parameters and time programs.

Drafts may contain unfilled required fields; only get_value validates them.
The public Pydantic models own all physical and cross-field constraints.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import re
import types
from typing import Annotated, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QSizePolicy,
)

from PASS.gui.structured import Column, NumericTable, StructuredField
from PASS.gui.widgets import Choice
from PASS.validation.files import INPUT_FILE_FIELDS


class IntegerValidator(QValidator):
    """Python integers, without Qt's signed 32-bit input limit."""

    def validate(self, text, position):
        state = (QValidator.Acceptable
                 if re.fullmatch(r"[+-]?\d+", text.strip()) else QValidator.Intermediate if text.strip() in {"", "+", "-"} else QValidator.Invalid)
        return state, text, position


def bare(annotation):
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def nullable(annotation):
    annotation = bare(annotation)
    return get_origin(annotation) in (Union, types.UnionType) and type(None) in get_args(annotation)


def model_draft(model, supplied=None):
    """Expose all fields without inventing required physical quantities."""
    result = {}
    for name, field in model.model_fields.items():
        value = None if field.is_required() else field.get_default(call_default_factory=True)
        if isinstance(value, BaseModel):
            value = value.model_dump(by_alias=True, mode="json")
        result[field.alias or name] = deepcopy(value)
    if supplied:
        aliases = {str(field.alias or name).casefold(): field.alias or name for name, field in model.model_fields.items()}
        aliases.update({name.casefold(): field.alias or name for name, field in model.model_fields.items()})
        for key, value in supplied.items():
            result[aliases.get(str(key).casefold(), key)] = deepcopy(value)
    return result


class ScalarField(StructuredField):

    def __init__(self, annotation, value, label, base_dir):
        super().__init__()
        self.annotation, self.label = bare(annotation), label
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        if get_origin(self.annotation) is Literal:
            self.input = Choice()
            choices = get_args(self.annotation)
            if value is None:
                self.input.addItem("请选择", None)
            elif value not in choices:
                self.input.addItem(str(value), value)
            for item in choices:
                self.input.addItem(str(item), item)
            self.input.setCurrentIndex(max(0, self.input.findData(value)))
            if len(choices) == 1:
                self.input.setCurrentIndex(self.input.findData(choices[0]))
                self.input.setEnabled(False)
            self.input.currentIndexChanged.connect(self.changed)
        elif self.annotation is bool:
            self.input = QCheckBox()
            if value is None:
                self.input.setTristate(True)
                from PySide6.QtCore import Qt
                self.input.setCheckState(Qt.PartiallyChecked)
            else:
                self.input.setChecked(value)
            self.input.stateChanged.connect(self.changed)
        else:
            self.input = QLineEdit("" if value is None else str(value))
            self.input.setPlaceholderText("必填" if value is None else "")
            if self.annotation is int:
                self.input.setValidator(IntegerValidator(self.input))
            self.input.textChanged.connect(self.changed)
            if label.casefold() in INPUT_FILE_FIELDS:
                choose = QPushButton("浏览…")
                choose.clicked.connect(lambda: self._browse(base_dir))
                root.addWidget(choose)
        root.insertWidget(0, self.input, 1)

    def _browse(self, base_dir):
        path, _ = QFileDialog.getOpenFileName(self, self.label, str(base_dir), "All files (*)")
        if path:
            self.input.setText(path)

    def get_value(self):
        if isinstance(self.input, QComboBox):
            value = self.input.currentData()
            if value is None:
                raise ValueError(f"{self.label}：请选择一个值")
            return value
        if isinstance(self.input, QCheckBox):
            from PySide6.QtCore import Qt
            if self.input.checkState() == Qt.PartiallyChecked:
                raise ValueError(f"{self.label}：请明确选择是或否")
            return self.input.isChecked()
        text = self.input.text().strip()
        if self.label.casefold() in INPUT_FILE_FIELDS and not text:
            raise ValueError(f"{self.label}：请选择文件，或关闭文件输入模式")
        if self.annotation is int:
            if not re.fullmatch(r"[+-]?\d+", text):
                raise ValueError(f"{self.label}：需要整数")
            return int(text)
        if self.annotation is float:
            try:
                value = float(text)
            except ValueError as exc:
                self.input.setFocus()
                raise ValueError(f"{self.label}：请填写数值（支持小数和科学计数法）") from exc
            if not math.isfinite(value):
                raise ValueError(f"{self.label}：需要有限数值")
            return value
        return text


class OptionalField(StructuredField):

    def __init__(self, annotation, value, label, base_dir):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.enabled_box = QCheckBox("自定义（关闭使用默认时钟）" if label == "Reference clock" else "指定此项")
        self.enabled_box.setChecked(value is not None)
        root.addWidget(self.enabled_box)
        args = tuple(a for a in get_args(bare(annotation)) if a is not type(None))
        self.editor = make_editor(args[0] if len(args) == 1 else Union[args], value, label, base_dir)
        root.addWidget(self.editor)
        self.editor.setEnabled(value is not None)
        self.editor.setVisible(value is not None)
        self.enabled_box.toggled.connect(self.editor.setEnabled)
        self.enabled_box.toggled.connect(self.editor.setVisible)
        self.enabled_box.toggled.connect(self.changed)
        self.editor.changed.connect(self.changed)

    def get_value(self):
        return self.editor.get_value() if self.enabled_box.isChecked() else None


class ArrayField(NumericTable):

    def __init__(self, annotation, value, label):
        self.annotation = bare(annotation)
        item_type = bare(get_args(self.annotation)[0])
        self.vector = get_origin(self.annotation) is tuple
        self.matrix = get_origin(item_type) in (tuple, list)
        width = len(get_args(self.annotation)) if self.vector else len(get_args(item_type)) if self.matrix else 1
        names = (["实部", "虚部"] if "poles" in label.casefold() or label == "Residues" else
                 ["x 次数", "y 次数"] if label.endswith("powers") else [label] if width == 1 else [f"{label} · {i+1}" for i in range(width)])
        integer = (bare(get_args(item_type)[0]) if self.matrix else item_type) is int
        columns = [Column(name, integer, -2147483647 if integer else -1e100, 2147483647 if integer else 1e100) for name in names]
        rows = [list(value)] if self.vector and value is not None else value if self.matrix else [[v] for v in value or []]
        super().__init__(columns, rows or [], "按行填写；可粘贴 CSV/制表符数据。")

    def serialize(self, rows):
        if self.vector:
            if len(rows) != 1:
                raise ValueError("需要一行坐标分量")
            return tuple(rows[0])
        return rows if self.matrix else [row[0] for row in rows]


class ActiveStack(QWidget):
    """An unused file-model form must not set a scalar editor's minimum size."""

    def __init__(self):
        super().__init__()
        self.pages, self.index = [], 0
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(0, 0, 0, 0)

    def addWidget(self, widget):
        self.pages.append(widget)
        self.body.addWidget(widget)
        widget.setVisible(len(self.pages) == 1)

    def setCurrentIndex(self, index):
        self.index = index
        for i, widget in enumerate(self.pages):
            widget.setVisible(i == index)
        self.body.invalidate()
        self.updateGeometry()

    def currentWidget(self):
        return self.pages[self.index]


class UnionField(StructuredField):
    """Keep drafts for each scalar/table or discriminated model alternative."""

    def __init__(self, annotation, value, label, base_dir):
        super().__init__()
        self.types = get_args(bare(annotation))
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.mode = Choice()
        self.stack = ActiveStack()
        self.editors = []
        selected = 0
        matched = False
        for index, typ in enumerate(self.types):
            typ = bare(typ)
            is_model = isinstance(typ, type) and issubclass(typ, BaseModel)
            kind = typ.model_fields.get("kind") if is_model else None
            title = str(kind.default) if kind else "时间表" if get_origin(typ) is list else "固定值"
            matches = (isinstance(value, dict) and kind and value.get("Kind", value.get("kind")) == kind.default
                       if is_model else isinstance(value, list) if get_origin(typ) is list else not isinstance(value, (list, dict)))
            if matches:
                selected = index
                matched = True
            self.mode.addItem(title)
            editor = make_editor(typ, value if matches else None, label, base_dir)
            editor.changed.connect(self.changed)
            self.editors.append(editor)
            self.stack.addWidget(editor)
        root.addWidget(self.mode)
        root.addWidget(self.stack)
        self.mode.setCurrentIndex(selected)
        self.stack.setCurrentIndex(selected)
        if isinstance(value, dict) and not matched:
            self.mode.addItem("未知模型：请明确选择")
            self.mode.setCurrentIndex(len(self.editors))
        self.mode.currentIndexChanged.connect(self._select)
        self.mode.currentIndexChanged.connect(self.changed)
        self._select(self.mode.currentIndex())

    def _select(self, index):
        if index < len(self.editors):
            self.stack.setCurrentIndex(index)
        for i, editor in enumerate(self.editors):
            editor.setSizePolicy(QSizePolicy.Expanding if i == index else QSizePolicy.Ignored,
                                 QSizePolicy.Preferred if i == index else QSizePolicy.Ignored)
        self.stack.setVisible(index < len(self.editors))
        self.stack.updateGeometry()

    def get_value(self):
        if self.mode.currentIndex() >= len(self.editors):
            raise ValueError("未知模型类型；请选择受支持的模型并明确填写参数")
        return self.editors[self.mode.currentIndex()].get_value()


class ModelListEditor(StructuredField):

    def __init__(self, model, value, label, base_dir):
        super().__init__()
        self.model, self.base_dir, self.label = model, base_dir, label
        self.entries = []
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.rows = QVBoxLayout()
        root.addLayout(self.rows)
        add = QPushButton("添加" + label)
        add.clicked.connect(lambda: self.add_entry())
        root.addWidget(add)
        for item in value or []:
            self.add_entry(item)

    def add_entry(self, value=None):
        panel = QGroupBox(f"{self.label} {len(self.entries)+1}")
        layout = QVBoxLayout(panel)
        editor = make_editor(self.model, value, self.label, self.base_dir)
        editor.changed.connect(self.changed)
        layout.addWidget(editor)
        buttons = QHBoxLayout()
        duplicate, remove = QPushButton("复制"), QPushButton("移除")
        buttons.addWidget(duplicate)
        buttons.addWidget(remove)
        layout.addLayout(buttons)
        self.entries.append((panel, editor))
        self.rows.addWidget(panel)
        duplicate.clicked.connect(lambda: self._duplicate(editor))
        remove.clicked.connect(lambda: self.remove_entry(panel, editor))
        self.changed.emit()

    def _duplicate(self, editor):
        from PySide6.QtWidgets import QMessageBox
        try:
            value = editor.get_value()
            if "Name" in value:
                used = {e.fields["Name"].input.text() for _, e in self.entries if isinstance(e, SchemaEditor) and "Name" in e.fields}
                from PASS.gui.project import unique_name
                value["Name"] = unique_name(value["Name"], used)
            self.add_entry(value)
        except ValueError as exc:
            QMessageBox.warning(self, "请先完成参数", str(exc))

    def remove_entry(self, panel, editor):
        self.entries.remove((panel, editor))
        self.rows.removeWidget(panel)
        panel.deleteLater()
        for index, (box, _) in enumerate(self.entries):
            box.setTitle(f"{self.label} {index+1}")
        self.changed.emit()

    def get_value(self):
        values = []
        for index, (_, editor) in enumerate(self.entries):
            try:
                values.append(editor.get_value())
            except ValueError as exc:
                raise ValueError(f"{self.label}[{index}]：{exc}") from exc
        return values


def make_editor(annotation, value, label, base_dir):
    typ = bare(annotation)
    if nullable(typ):
        return OptionalField(typ, value, label, base_dir)
    if get_origin(typ) in (Union, types.UnionType):
        return UnionField(typ, value, label, base_dir)
    if isinstance(typ, type) and issubclass(typ, BaseModel):
        return SchemaEditor(typ, value, base_dir)
    if get_origin(typ) in (list, tuple):
        item_type = bare(get_args(typ)[0])
        if isinstance(item_type, type) and issubclass(item_type, BaseModel):
            return ModelListEditor(item_type, value, label, base_dir)
        return ArrayField(typ, value, label)
    return ScalarField(typ, value, label, base_dir)


class SchemaEditor(StructuredField):

    def __init__(self, model, value=None, base_dir=Path(".")):
        super().__init__()
        self.model = model
        self.original = model_draft(model, value)
        self.fields, self.labels, self.inactive = {}, {}, {}
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        for name, info in model.model_fields.items():
            key = info.alias or name
            field = make_editor(info.annotation, self.original[key], key, base_dir)
            field.setToolTip(info.description or key)
            self.fields[key] = field
            label = QLabel(key + ("\N{NO-BREAK SPACE}*" if info.is_required() else ""))
            label.setWordWrap(True)
            self.labels[key] = label
            label.setMaximumWidth(150)
            if isinstance(field, ScalarField):
                self.form.addRow(label, field)
            else:
                self.form.addRow(label)
                self.form.addRow(field)
            field.changed.connect(self.changed)
        self._configure_modes()

    def _choice_value(self, key):
        field = self.fields.get(key)
        return field.input.currentData() if isinstance(field, ScalarField) and isinstance(field.input, QComboBox) else None

    def _active(self, key, enabled, fallback=None):
        field = self.fields[key]
        field.setEnabled(enabled)
        self.form.setRowVisible(field, enabled)
        self.form.setRowVisible(self.labels[key], enabled)
        if enabled:
            self.inactive.pop(key, None)
        else:
            self.inactive[key] = fallback

    def _configure_modes(self):
        name = self.model.__name__
        if name == "RFComponent":
            self.source_mode, self.frequency_mode = Choice(), Choice()
            self.source_mode.addItems(["内联参数", "波形文件"])
            self.frequency_mode.addItems(["谐波 × 规定时钟", "直接频率 Hz"])
            self.source_mode.setCurrentIndex(int(self.original.get("Program file") is not None))
            self.frequency_mode.setCurrentIndex(int(self.original.get("Harmonic") is None))
            self.form.insertRow(0, "输入来源", self.source_mode)
            self.form.insertRow(1, "频率定义", self.frequency_mode)
            self.source_mode.currentIndexChanged.connect(self._rf_modes)
            self.frequency_mode.currentIndexChanged.connect(self._rf_modes)
            hint = QLabel("V/相位/频率可选择固定值或时间表；所有时间表共用 Time (s)。文件列：TIME, VOLTAGE, PHASE；直接频率还需 FREQUENCY。频率按物理时间积分。")
            hint.setWordWrap(True)
            self.form.addRow(hint)
            self._rf_modes()
        elif name == "WakeSolverGroup":
            self.fields["Solver"].input.currentIndexChanged.connect(self._solver_changed)
            for key in ("History", "Boundary"):
                self.fields[key].input.currentIndexChanged.connect(self._wake_modes)
            self._wake_modes()
        elif name == "WakeVelocity":
            self.fields["Kind"].input.currentIndexChanged.connect(self._velocity_modes)
            self._velocity_modes()
        elif name == "WakeComponentConfig":
            self.fields["Component"].input.currentIndexChanged.connect(self._component_modes)
            self._component_modes()

    def _rf_modes(self):
        file = self.source_mode.currentIndex() == 1
        harmonic = self.frequency_mode.currentIndex() == 0
        for key in ("Voltage (V)", "Phase (rad)"):
            self._active(key, not file, 0.)
        self._active("Time (s)", not file)
        self._active("Program file", file)
        self._active("Frequency (Hz)", not file and not harmonic)
        self._active("Harmonic", harmonic)
        for key, enabled in (("Program file", file), ("Harmonic", harmonic), ("Frequency (Hz)", not file and not harmonic)):
            self.fields[key].enabled_box.setChecked(enabled)
            self.fields[key].enabled_box.hide()  # Required whenever the selected RF mode uses it.
        self.changed.emit()

    def _solver_changed(self):
        solver = self._choice_value("Solver")
        history = ("partitioned" if solver in {"partitioned_fft", "time_fft"} else "state" if solver in {"recursive", "modal"} else "none")
        field = self.fields["History"].input
        field.setCurrentIndex(field.findData(history))
        for key, enabled in (("Convolution grid", solver == "partitioned_fft"), ("Time grid", solver == "time_fft"),
                             ("Memory turns", solver == "partitioned_fft"), ("Memory time (s)", solver == "time_fft")):
            self.fields[key].enabled_box.setChecked(enabled)
        self._wake_modes()

    def _wake_modes(self):
        solver, history, boundary = (self._choice_value(k) for k in ("Solver", "History", "Boundary"))
        active = {
            "Convolution grid": solver == "partitioned_fft",
            "Time grid": solver == "time_fft",
            "Memory turns": history in {"direct", "partitioned"} and solver != "time_fft",
            "Memory time (s)": history in {"direct", "partitioned"} and solver != "partitioned_fft",
            "Partition": solver in {"partitioned_fft", "time_fft"},
            "Max workspace (MiB)": solver in {"partitioned_fft", "time_fft"},
            "Period (s)": boundary == "periodic",
            "Periodic images": boundary == "periodic"
        }
        for key, enabled in active.items():
            self._active(key, enabled)
        self.changed.emit()

    def _velocity_modes(self):
        kind = self._choice_value("Kind")
        self._active("Beta", kind == "fixed")
        for key in ("Betas", "Source", "Witness"):
            self._active(key, kind == "factorized", [])
        self.changed.emit()

    def _component_modes(self):
        self._active("Spatial", self._choice_value("Component") == "custom")
        self.changed.emit()

    def get_value(self):
        values = {key: deepcopy(value) for key, value in self.original.items() if key not in self.fields}
        for key, field in self.fields.items():
            try:
                values[key] = deepcopy(self.inactive[key]) if key in self.inactive else field.get_value()
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{key}：{exc}") from exc
        return self.model.model_validate(values).model_dump(by_alias=True, mode="json")


def focus_parameter(field, path):
    """Navigate nested model/list/union editors using a schema error path."""
    if isinstance(field, OptionalField):
        return focus_parameter(field.editor, path)
    if isinstance(field, UnionField):
        if field.mode.currentIndex() >= len(field.editors):
            field.mode.setFocus()
            return
        return focus_parameter(field.editors[field.mode.currentIndex()], path)
    if isinstance(field, SchemaEditor) and path:
        if path[0] in field.fields:
            return focus_parameter(field.fields[path[0]], path[1:])
    if isinstance(field, ModelListEditor) and path and isinstance(path[0], int):
        if 0 <= path[0] < len(field.entries):
            return focus_parameter(field.entries[path[0]][1], path[1:])
    if isinstance(field, NumericTable) and path and isinstance(path[0], int):
        row = path[0]
        col = path[1] if len(path) > 1 and isinstance(path[1], int) else 0
        if row < field.table.rowCount() and col < field.table.columnCount():
            field.table.setCurrentCell(row, col)
            field.table.scrollToItem(field.table.item(row, col))
    (field.input if isinstance(field, ScalarField) else field).setFocus()
