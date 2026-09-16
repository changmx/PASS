"""Named wake settings editor; physical history belongs to individual points."""
from copy import deepcopy

from PySide6.QtWidgets import QCheckBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout

from PASS.gui.parameters import Choice, SchemaEditor
from PASS.gui.structured import StructuredField
from PASS.para.schema.wake_field import WakeFieldConfig, WakeResourceConfig


class WakeConfigurationEditor(StructuredField):
    def __init__(self, block, sequence, base_dir):
        super().__init__()
        self.base_dir = base_dir
        self.resources = deepcopy(block.get("Configurations", {}))
        self.references = {name: point["Configuration"] for name, point in sequence.items()
                           if isinstance(point, dict) and point.get("Command") == "WakeField"
                           and point.get("Configuration") is not None}
        self.active = None
        self.editor = None
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 0, 0, 0)
        self.enabled = QCheckBox("启用尾场")
        self.enabled.setChecked(block.get("Enabled", True))
        self.enabled.setToolTip("总开关与每个尾场点的 Is enabled 同时开启，该点才执行。")
        self.enabled.toggled.connect(self.changed)
        self.root.addWidget(self.enabled)
        row = QHBoxLayout()
        row.addWidget(QLabel("当前配置"))
        self.selector = Choice()
        self.selector.currentTextChanged.connect(self.select)
        row.addWidget(self.selector, 1)
        for label, action in (("添加", self.add), ("复制", self.copy), ("删除", self.remove)):
            button = QPushButton(label)
            button.clicked.connect(action)
            row.addWidget(button)
        self.root.addLayout(row)
        self.name = QLineEdit()
        self.name.textChanged.connect(self.changed)
        form = QFormLayout()
        form.addRow("配置名称", self.name)
        self.root.addLayout(form)
        self._show(next(iter(self.resources), None))

    def _show(self, name):
        if self.editor is not None:
            self.root.removeWidget(self.editor)
            self.editor.setParent(None)
            self.editor.deleteLater()
        self.active = name
        self.selector.blockSignals(True)
        self.selector.clear()
        self.selector.addItems(list(self.resources))
        self.selector.setCurrentText(name or "")
        self.selector.blockSignals(False)
        self.name.setText(name or "")
        self.name.setEnabled(name is not None)
        self.editor = None
        if name is not None:
            self.editor = SchemaEditor(WakeResourceConfig, self.resources[name], self.base_dir)
            self.editor.changed.connect(self.changed)
            self.root.addWidget(self.editor)

    def _commit(self):
        if self.active is None:
            return
        name = self.name.text().strip()
        if not name:
            raise ValueError("尾场配置名称不能为空")
        if name != self.active and name in self.resources:
            raise ValueError(f"尾场配置名称 {name!r} 已存在")
        value = self.editor.get_value()
        self.resources = {name if k == self.active else k: value if k == self.active else v
                          for k, v in self.resources.items()}
        self.references = {point: name if ref == self.active else ref for point, ref in self.references.items()}
        self.active = name

    def _try_commit(self):
        try:
            self._commit()
            return True
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "尾场配置无效", str(exc))
            return False

    def select(self, name):
        if name == self.active:
            return
        if self._try_commit():
            self._show(name)
        else:
            self.selector.blockSignals(True)
            self.selector.setCurrentText(self.active or "")
            self.selector.blockSignals(False)

    def _unique(self, base):
        name, index = base, 2
        while name in self.resources:
            name, index = f"{base}_{index}", index + 1
        return name

    def add(self):
        if not self._try_commit():
            return
        name = self._unique("wake")
        self.resources[name] = {"Groups": []}
        self._show(name)
        self.changed.emit()

    def copy(self):
        if self.active is None or not self._try_commit():
            return
        name = self._unique(self.active + "_copy")
        self.resources[name] = deepcopy(self.resources[self.active])
        self._show(name)
        self.changed.emit()

    def remove(self):
        if self.active is None:
            return
        used = [point for point, ref in self.references.items() if ref == self.active]
        if used:
            QMessageBox.warning(self, "配置正在使用", "请先修改这些尾场点的引用：" + "、".join(used))
            return
        del self.resources[self.active]
        self._show(next(iter(self.resources), None))
        self.changed.emit()

    def get_value(self):
        self._commit()
        return WakeFieldConfig.model_validate({"Enabled": self.enabled.isChecked(),
            "Configurations": self.resources}).model_dump(by_alias=True, mode="json")
