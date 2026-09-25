"""Lossless property drafts and plain-data comparisons for editor tools."""
from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QPlainTextEdit, QSpinBox, QTableWidgetItem, QWidget

from PASS.gui.parameters import ModelListEditor, OptionalField, ScalarField, SchemaEditor, UnionField
from PASS.gui.structured import ApertureEditor, NumericTable, ObjectEditor, ObjectListEditor


def capture_field(field):
    """Capture incomplete text without flushing a numeric delegate or validating."""
    from PASS.gui.wake_configuration import WakeConfigurationEditor
    if isinstance(field, WakeConfigurationEditor):
        return {
            "kind": "wake",
            "resources": deepcopy(field.resources),
            "references": deepcopy(field.references),
            "active": field.active,
            "enabled": field.enabled.isChecked(),
            "name": field.name.text(),
            "editor": capture_field(field.editor) if field.editor else None
        }
    if isinstance(field, NumericTable):
        rows = [[field.table.item(row, column).text() if field.table.item(row, column) else "" for column in range(field.table.columnCount())]
                for row in range(field.table.rowCount())]
        delegate = field.table.itemDelegate()
        if delegate.editor is not None and delegate.index.isValid():
            rows[delegate.index.row()][delegate.index.column()] = delegate.editor.lineEdit().text()
        return {"kind": "table", "rows": rows, "load_error": field.load_error}
    if isinstance(field, (ModelListEditor, ObjectListEditor)):
        return {"kind": "entries", "entries": [capture_field(editor) for _, editor in field.entries]}
    if isinstance(field, SchemaEditor):
        modes = {name: capture_field(getattr(field, name)) for name in ("source_mode", "frequency_mode") if hasattr(field, name)}
        return {"kind": "schema", "fields": {key: capture_field(item) for key, item in field.fields.items()}, "modes": modes}
    if isinstance(field, ObjectEditor):
        return {"kind": "schema", "fields": {key: capture_field(item) for key, item in field.fields.items()}}
    if isinstance(field, ScalarField):
        return {"kind": "scalar", "input": capture_field(field.input)}
    if isinstance(field, OptionalField):
        return {"kind": "optional", "enabled": field.enabled_box.isChecked(), "editor": capture_field(field.editor)}
    if isinstance(field, UnionField):
        return {"kind": "union", "index": field.mode.currentIndex(), "editors": [capture_field(editor) for editor in field.editors]}
    if isinstance(field, ApertureEditor):
        return {
            "kind": "aperture",
            "shape": field.kind,
            "fields": [capture_field(item) for item in field.fields],
            "polygon": capture_field(field.polygon) if field.polygon else None,
            "load_error": field.load_error
        }
    if isinstance(field, QComboBox):
        return {"kind": "choice", "text": field.currentText(), "data": field.currentData()}
    if isinstance(field, QCheckBox):
        return {"kind": "check", "state": field.checkState().value}
    if isinstance(field, (QSpinBox, QDoubleSpinBox)):
        return {"kind": "spin", "text": field.lineEdit().text()}
    if isinstance(field, QLineEdit):
        return {"kind": "text", "text": field.text()}
    if isinstance(field, QPlainTextEdit):
        return {"kind": "text", "text": field.toPlainText()}
    # Domain editors use stable child widgets; tables and dynamic lists above
    # need explicit handling because their delegate/item children are transient.
    children = [(index, capture_field(child)) for index, child in enumerate(field.children()) if isinstance(child, QWidget)]
    return {"kind": "container", "children": [[index, state] for index, state in children if state.get("kind") != "container" or state["children"]]}


def restore_field(field, state):
    kind = state.get("kind")
    if kind == "wake":
        field.resources = deepcopy(state["resources"])
        field.references = deepcopy(state["references"])
        field.enabled.setChecked(state["enabled"])
        field._show(state["active"])
        field.name.setText(state["name"])
        if field.editor and state.get("editor"):
            restore_field(field.editor, state["editor"])
    elif kind == "table" and isinstance(field, NumericTable):
        rows = state["rows"]
        field.table.blockSignals(True)
        field.table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            field.table.setVerticalHeaderItem(row, QTableWidgetItem(str(row + field.index_origin)))
            for column, value in enumerate(values[:field.table.columnCount()]):
                item = QTableWidgetItem()
                try:
                    item.setData(Qt.EditRole, field.columns[column].parse(value))
                except (ValueError, TypeError):
                    item.setText(value)
                field.table.setItem(row, column, item)
        field.table.blockSignals(False)
        field.load_error = state.get("load_error", "")
        field._edited()
    elif kind == "entries" and isinstance(field, (ModelListEditor, ObjectListEditor)):
        while field.entries:
            field.remove_entry(*field.entries[-1])
        for item in state["entries"]:
            field.add_entry(deepcopy(getattr(field, "default", None)))
            restore_field(field.entries[-1][1], item)
    elif kind == "schema" and isinstance(field, (SchemaEditor, ObjectEditor)):
        for name, value in state.get("modes", {}).items():
            if hasattr(field, name):
                restore_field(getattr(field, name), value)
        for name, value in state["fields"].items():
            if name in field.fields:
                restore_field(field.fields[name], value)
    elif kind == "scalar" and isinstance(field, ScalarField):
        restore_field(field.input, state["input"])
    elif kind == "optional" and isinstance(field, OptionalField):
        field.enabled_box.setChecked(state["enabled"])
        restore_field(field.editor, state["editor"])
    elif kind == "union" and isinstance(field, UnionField):
        field.mode.setCurrentIndex(state["index"])
        for editor, value in zip(field.editors, state["editors"]):
            restore_field(editor, value)
    elif kind == "aperture" and isinstance(field, ApertureEditor):
        field.set_kind(state["shape"])
        for item, value in zip(field.fields, state["fields"]):
            restore_field(item, value)
        if field.polygon and state.get("polygon"):
            restore_field(field.polygon, state["polygon"])
        field.load_error = state.get("load_error", "")
    elif kind == "choice" and isinstance(field, QComboBox):
        index = field.findText(state["text"])
        if index < 0:
            field.addItem(state["text"], state.get("data"))
            index = field.count() - 1
        field.setCurrentIndex(index)
    elif kind == "check" and isinstance(field, QCheckBox):
        field.setCheckState(Qt.CheckState(state["state"]))
    elif kind == "spin" and isinstance(field, (QSpinBox, QDoubleSpinBox)):
        text = state["text"]
        if field.validate(text, len(text))[0] == QValidator.Acceptable:
            field.setValue(int(text) if isinstance(field, QSpinBox) else float(text))
        else:
            field.lineEdit().setText(text)
    elif kind == "text" and isinstance(field, QLineEdit):
        field.setText(state["text"])
    elif kind == "text" and isinstance(field, QPlainTextEdit):
        field.setPlainText(state["text"])
    elif kind == "container":
        for index, value in state.get("children", []):
            children = field.children()
            if index < len(children) and isinstance(children[index], QWidget):
                restore_field(children[index], value)


def draft_value(field, reader, key, previous):
    """Show invalid input explicitly in a diff instead of dropping the draft."""
    from PASS.gui.wake_configuration import WakeConfigurationEditor
    detached = None
    if isinstance(field, WakeConfigurationEditor):
        state = capture_field(field)
        detached = WakeConfigurationEditor({"Configurations": state["resources"]}, {}, field.base_dir)
        restore_field(detached, state)
    try:
        return reader(key, detached or field, previous)
    except (ValueError, TypeError):
        state = capture_field(field)
        if "text" in state:
            return {"未完成输入": state["text"]}
        if "rows" in state:
            return {"未完成表格": state["rows"]}
        return {"未完成参数": state}
    finally:
        if detached is not None:
            detached.deleteLater()


def value_changes(before, after, path=()):
    """Return readable paths, preserving explicit additions and removals."""
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in dict.fromkeys([*before, *after]):
            if key not in before:
                result.append(((*path, str(key)), "（未设置）", after[key]))
            elif key not in after:
                result.append(((*path, str(key)), before[key], "（已删除）"))
            else:
                result.extend(value_changes(before[key], after[key], (*path, str(key))))
        return result
    return [] if before == after else [(path, before, after)]


def merge_draft(base, draft, current, path=()):
    """Apply a draft delta without overwriting newer, unrelated document edits."""
    if draft == base:
        return deepcopy(current)
    if current == base or current == draft:
        return deepcopy(draft)
    if isinstance(base, dict) and isinstance(draft, dict) and isinstance(current, dict):
        result = deepcopy(current)
        for key in dict.fromkeys([*base, *draft]):
            if key not in base:
                if key in current and current[key] != draft[key]:
                    raise ValueError("配置在编辑期间发生冲突：" + " / ".join((*path, str(key))))
                result[key] = deepcopy(draft[key])
            elif key not in draft:
                if key in current and current[key] != base[key]:
                    raise ValueError("配置在编辑期间发生冲突：" + " / ".join((*path, str(key))))
                result.pop(key, None)
            elif base[key] != draft[key]:
                if key not in current:
                    raise ValueError("配置在编辑期间被移除：" + " / ".join((*path, str(key))))
                result[key] = merge_draft(base[key], draft[key], current[key], (*path, str(key)))
        return result
    raise ValueError("配置在编辑期间发生冲突：" + " / ".join(path))


def named_fields(field, path):
    """Index nested editors by labels without changing their visibility."""
    yield path, field
    if isinstance(field, (SchemaEditor, ObjectEditor)):
        for key, child in field.fields.items():
            yield from named_fields(child, (*path, key))
    elif isinstance(field, (ModelListEditor, ObjectListEditor)):
        for index, (_, child) in enumerate(field.entries):
            yield from named_fields(child, (*path, str(index)))
    elif isinstance(field, OptionalField):
        yield from named_fields(field.editor, (*path, "详细参数"))
    elif isinstance(field, UnionField) and field.mode.currentIndex() < len(field.editors):
        yield from named_fields(field.editors[field.mode.currentIndex()], (*path, field.mode.currentText()))
    elif not isinstance(field, (ScalarField, NumericTable)):
        for form in field.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                label_item = form.itemAt(row, QFormLayout.LabelRole)
                field_item = form.itemAt(row, QFormLayout.FieldRole)
                if label_item and field_item and isinstance(label_item.widget(), QLabel) and field_item.widget():
                    yield from named_fields(field_item.widget(), (*path, label_item.widget().text()))
