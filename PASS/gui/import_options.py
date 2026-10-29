"""Typed property overrides for newly imported lattice elements."""

from copy import deepcopy
from typing import Literal

from PySide6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from PASS.gui.parameters import make_editor, model_draft
from PASS.gui.structured import APERTURES, ApertureEditor, InternalSpaceChargeEditor, StructuredField
from PASS.gui.widgets import PropertyComboBox
from PASS.para.schema.elements import ELEMENT_REGISTRY


def _element_parameter_options(command, key):
    if key == "Model":
        if command == "SBend":
            return ("adaptive", "drift-kick-drift-exact", "rot-kick-rot")
        if command == "Quadrupole":
            return ("adaptive", "drift-kick-drift-exact", "mat-kick-mat")
    if key == "Integrator":
        return ("adaptive", "uniform", "yoshida4")
    if key == "Aperture type":
        return tuple(APERTURES)
    return ()


class ElementImportOverrides(StructuredField):
    """Only explicitly added properties replace the corresponding source values."""

    def __init__(self, base_dir, context=None):
        super().__init__()
        self.base_dir = base_dir
        self.context = context if context is not None else {}
        self.models = {model.model_fields["command"].default: model for model in ELEMENT_REGISTRY.values()}
        self.rules = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.enabled = QCheckBox("按元件类型批量设置属性")
        root.addWidget(self.enabled)
        self.body = QWidget()
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("仅覆盖本次导入中匹配类型的所选属性；位置和长度仍使用来源数据。未添加的属性保持原值。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.command = PropertyComboBox()
        self.command.addItems(("SBend", "Quadrupole", "Sextupole", "Octupole", "Multipole", "Solenoid", "Kicker", "Drift", "Marker"))
        self.parameter = PropertyComboBox()
        form.addRow("元件类型", self.command)
        form.addRow("要统一设置的属性", self.parameter)
        layout.addLayout(form)
        self.add_button = QPushButton("添加属性设置")
        self.add_button.clicked.connect(self._add_selected)
        layout.addWidget(self.add_button)
        self.rows = QVBoxLayout()
        layout.addLayout(self.rows)
        root.addWidget(self.body)
        self.enabled.toggled.connect(self.body.setVisible)
        self.enabled.toggled.connect(self.changed)
        self.body.setVisible(False)
        self.command.currentTextChanged.connect(self._refresh_parameters)
        self._refresh_parameters()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

    def _specs(self, command):
        model = self.models[command]
        return {field.alias or name: field for name, field in model.model_fields.items() if name not in {"s", "length", "command"}}

    def _refresh_parameters(self):
        previous = self.parameter.currentText()
        self.parameter.clear()
        command = self.command.currentText()
        specs = self._specs(command)
        preferred = ("Model", "Integrator", "Num slices", "Order", "Aperture type", "Aperture value")
        available = [key for key in (*preferred, *specs) if key in specs and (command, key) not in self.rules]
        if (command, "Aperture type") not in self.rules:
            available = [key for key in available if key != "Aperture value"]
        self.parameter.addItems(dict.fromkeys(available))
        if previous in available:
            self.parameter.setCurrentText(previous)
        self.add_button.setEnabled(bool(available))

    def _add_selected(self):
        if self.parameter.currentText():
            self.add_rule(self.command.currentText(), self.parameter.currentText())

    def add_rule(self, command, key, value=None, *, supplied=False):
        identifier = command, key
        if identifier in self.rules or key not in self._specs(command):
            return
        spec = self._specs(command)[key]
        annotation = spec.annotation
        options = _element_parameter_options(command, key)
        if options:
            annotation = Literal[options]
        elif annotation is list:
            annotation = list[float]
        if not supplied:
            value = model_draft(self.models[command])[key]
        group = QGroupBox(f"{command} · {key}")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        if key == "Aperture value":
            editor = ApertureEditor(self._aperture_kind(command), deepcopy(value) if supplied else None)
            editor.setParent(group)
        elif key == "Space charge":
            configurations = self.context.get("Space charge", {}).get("Configurations", {})
            editor = InternalSpaceChargeEditor(deepcopy(value), configurations, int(self.context.get("Number of turns", 100)))
            editor.setParent(group)
        else:
            editor = make_editor(annotation, deepcopy(value), key, self.base_dir, group)
        layout.addWidget(editor)
        remove = QPushButton("移除此设置")
        remove.clicked.connect(lambda: self.remove_rule(identifier))
        layout.addWidget(remove)
        self.rows.addWidget(group)
        self.rules[identifier] = group, editor
        editor.changed.connect(self.changed)
        if key == "Aperture type":
            editor.changed.connect(lambda: self._refresh_aperture(command))
            self._refresh_aperture(command)
        self._refresh_parameters()
        self.changed.emit()

    def _aperture_kind(self, command):
        rule = self.rules.get((command, "Aperture type"))
        return rule[1].get_value() if rule is not None else "off"

    def _refresh_aperture(self, command):
        rule = self.rules.get((command, "Aperture value"))
        if rule is not None:
            rule[1].set_kind(self._aperture_kind(command))

    def remove_rule(self, identifier):
        group, _editor = self.rules.pop(identifier)
        self.rows.removeWidget(group)
        group.deleteLater()
        if identifier[1] == "Aperture type":
            self._refresh_aperture(identifier[0])
        self._refresh_parameters()
        self.changed.emit()

    def get_value(self):
        if not self.enabled.isChecked():
            return {}
        result = {}
        for (command, key), (_group, editor) in self.rules.items():
            try:
                result.setdefault(command, {})[key] = editor.get_value()
            except ValueError as exc:
                raise ValueError(f"{command} · {key}: {exc}") from exc
        return result

    def set_value(self, values):
        for identifier in tuple(self.rules):
            self.remove_rule(identifier)
        for command, parameters in values.items():
            for key in sorted(parameters, key=lambda key: key != "Aperture type"):
                self.add_rule(command, key, parameters[key], supplied=True)
        self.enabled.setChecked(bool(values))

    def capture_draft(self):
        from PASS.gui.property_state import capture_field

        return {
            "enabled": self.enabled.isChecked(),
            "command": self.command.currentText(),
            "parameter": self.parameter.currentText(),
            "rules": [{
                "command": command,
                "parameter": key,
                "state": capture_field(editor)
            } for (command, key), (_group, editor) in self.rules.items()],
        }

    def restore_draft(self, state):
        from PASS.gui.property_state import restore_field

        for identifier in tuple(self.rules):
            self.remove_rule(identifier)
        for row in state.get("rules", []):
            identifier = row["command"], row["parameter"]
            self.add_rule(*identifier)
            restore_field(self.rules[identifier][1], row["state"])
        self.command.setCurrentText(state.get("command", "SBend"))
        self.parameter.setCurrentText(state.get("parameter", ""))
        self.enabled.setChecked(state.get("enabled", False))

    def apply_to(self, items, names, circumference, context=None):
        overrides = self.get_value()
        if not overrides:
            return items

        from PASS.para.schema.main import MainConfig
        from PASS.utils.command_order import sort_commands
        from PASS.validation import ValidationReport
        from PASS.validation.rules import Validator

        data = MainConfig().model_dump(by_alias=True)
        data.update(deepcopy(context or {}))
        data["Sequence"] = {}
        data["Circumference (m)"] = circumference
        validator = Validator(data, self.base_dir, ValidationReport(full=False), False)
        validator.globals()
        result = []
        for name, item in zip(names, items):
            parameters = overrides.get(item.command)
            if parameters:
                values = item.model_dump(by_alias=True)
                values.update(deepcopy(parameters))
                validator.command(name, values)
                if validator.report.errors:
                    raise ValueError(f"批量属性设置无效：{validator.report.errors[0]}")
                item = type(item).model_validate(values)
            result.append(item)
        if any("Order" in parameters for parameters in overrides.values()):
            # Same-position commands require complete, distinct explicit orders.
            sort_commands(result)
        return result
