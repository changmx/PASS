"""Structured editing of shared beam-beam configurations and source profiles."""

from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QTabWidget, QVBoxLayout

from PASS.commands.collision.config import BeamBeamConfig, BeamBeamConfiguration, BeamBeamLuminosityConfig, BeamBeamSourceConfig, FrozenParameters
from PASS.gui.parameters import Choice, SchemaEditor, make_editor, model_draft
from PASS.gui.structured import Column, NumericTable, StructuredField


class FrozenProfileEditor(SchemaEditor):

    def __init__(self, value, base_dir, solver):
        super().__init__(FrozenParameters, value, base_dir)
        self.set_solver(solver)

    def set_solver(self, solver):
        gaussian, round_profile = solver.startswith("gaussian_"), "_round_" in solver
        required = ({"Sigma (m)"} if round_profile else {"Sigma X (m)", "Sigma Y (m)"}) if gaussian else (
            {"Radius (m)"} if round_profile else {"Semi-axis A (m)", "Semi-axis B (m)"})
        for key in ("Sigma (m)", "Sigma X (m)", "Sigma Y (m)", "Radius (m)", "Semi-axis A (m)", "Semi-axis B (m)"):
            self._active(key, key in required)
            field = self.fields[key]
            field.enabled_box.setChecked(key in required)
            field.enabled_box.hide()
        self._active("Angle (rad)", not round_profile, 0.)


class SliceProfilesEditor(StructuredField):

    def __init__(self, value, base_dir, solver):
        super().__init__()
        self.base_dir, self.solver, self.entries = base_dir, solver, []
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 0, 0, 0)
        add = QPushButton("添加切片参数覆盖")
        add.clicked.connect(lambda: self.add())
        self.root.addWidget(add)
        for index, profile in (value or {}).items():
            self.add(str(index), profile)

    def add(self, index=None, value=None):
        group = QGroupBox("切片参数")
        layout = QVBoxLayout(group)
        row = QHBoxLayout()
        row.addWidget(QLabel("切片编号（从 0 开始）"))
        key = QLineEdit(str(len(self.entries)) if index is None else str(index))
        row.addWidget(key)
        remove = QPushButton("删除")
        row.addWidget(remove)
        layout.addLayout(row)
        profile = FrozenProfileEditor(value, self.base_dir, self.solver)
        layout.addWidget(profile)
        entry = (group, key, profile)
        self.entries.append(entry)
        self.root.addWidget(group)
        key.textChanged.connect(self.changed)
        profile.changed.connect(self.changed)
        remove.clicked.connect(lambda: self.remove(entry))
        self.changed.emit()

    def remove(self, entry):
        self.entries.remove(entry)
        self.root.removeWidget(entry[0])
        entry[0].deleteLater()
        self.changed.emit()

    def set_solver(self, solver):
        self.solver = solver
        for _, _, profile in self.entries:
            profile.set_solver(solver)

    def get_value(self):
        result = {}
        for _, field, profile in self.entries:
            key = field.text().strip()
            if not key.isdecimal() or str(int(key)) in result:
                raise ValueError("切片编号必须是互不重复的非负整数")
            result[str(int(key))] = profile.get_value()
        return result

    def capture_draft(self):
        from PASS.gui.property_state import capture_field
        return [{"index": key.text(), "profile": capture_field(profile)} for _, key, profile in self.entries]

    def restore_draft(self, state):
        from PASS.gui.property_state import restore_field
        for entry in list(self.entries):
            self.remove(entry)
        for item in state:
            self.add(item["index"])
            restore_field(self.entries[-1][2], item["profile"])


class CollisionSourceEditor(StructuredField):

    def __init__(self, value, base_dir):
        super().__init__()
        self.fields, self.rows = {}, {}
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        values = model_draft(BeamBeamSourceConfig, value)
        self.method, self.solver = Choice(), Choice()
        for title, method in (("仅目标粒子", None), ("PIC", "pic"), ("固定解析源 frozen", "frozen"), ("随粒子矩更新 quasi-frozen", "quasi-frozen")):
            self.method.addItem(title, method)
        self.method.setCurrentIndex(max(0, self.method.findData(values["Method"])))
        self.form.addRow("源模型", self.method)
        self.form.addRow("求解器", self.solver)
        for name, info in BeamBeamSourceConfig.model_fields.items():
            key = info.alias or name
            if key in {"Method", "Solver"}:
                continue
            if key == "Frozen parameters":
                field = FrozenProfileEditor(values[key], base_dir, values["Solver"] or "gaussian_ellipse_free_space")
            elif key == "Slice parameters":
                field = SliceProfilesEditor(values[key], base_dir, values["Solver"] or "gaussian_ellipse_free_space")
            else:
                field = make_editor(info.annotation, values[key], key, base_dir)
            self.fields[key] = field
            label = QLabel(key)
            label.setWordWrap(True)
            if key == "Propagation step (m)":
                label.setText(key + " *")
                field.setToolTip("PIC 必填：沿碰撞距离 S 采样源势的正有限步长，单位 m；不是实际碰撞距离 S。程序不自动设置。")
                field.editor.input.setPlaceholderText("必填：沿 S 的采样步长（m）")
            self.rows[key] = label
            if key in {"Frozen parameters", "Slice parameters"}:
                self.form.addRow(label)
                self.form.addRow(field)
            else:
                self.form.addRow(label, field)
            field.changed.connect(self.changed)
        self.method.currentIndexChanged.connect(self._method_changed)
        self.solver.currentIndexChanged.connect(self._solver_changed)
        self._method_changed(preferred=values["Solver"])

    def _method_changed(self, *_args, preferred=None):
        previous = preferred or self.solver.currentData()
        method = self.method.currentData()
        solvers = (["fft_free_space"] if method == "pic" else
                   [f"{profile}_{shape}_free_space" for profile in ("gaussian", "uniform", "parabolic") for shape in ("ellipse", "round")])
        self.solver.blockSignals(True)
        self.solver.clear()
        for solver in solvers:
            self.solver.addItem(solver, solver)
        self.solver.setCurrentIndex(max(0, self.solver.findData(previous)))
        self.solver.blockSignals(False)
        self._solver_changed()

    def _solver_changed(self, *_args):
        method = self.method.currentData()
        active = {"Slice set"}
        if method:
            active.add("Statistics precision")
        if method == "pic":
            active.update({"Nx", "Ny", "Grid Half Width X (m)", "Grid Half Width Y (m)", "Particle Deposition Method", "Propagation step (m)"})
            for key in ("Grid Half Width X (m)", "Grid Half Width Y (m)", "Propagation step (m)"):
                self.fields[key].enabled_box.setChecked(True)
                self.fields[key].enabled_box.hide()
                self.fields[key].enabled_box.setEnabled(False)
        if method == "frozen":
            active.update({"Frozen parameters", "Slice parameters", "Frozen optics reference", "Source center slopes"})
            for key in ("Frozen parameters", "Slice parameters"):
                self.fields[key].set_solver(self.solver.currentData())
        self.active_fields = active
        self.solver.setEnabled(method is not None)
        for key, field in self.fields.items():
            self.form.setRowVisible(field, key in active)
            self.form.setRowVisible(self.rows[key], key in active)
        self.changed.emit()

    def get_value(self):
        values = {key: self.fields[key].get_value() for key in self.active_fields}
        method = self.method.currentData()
        if method:
            values.update({"Method": method, "Solver": self.solver.currentData()})
        return BeamBeamSourceConfig.model_validate(values).model_dump(by_alias=True, mode="json")

    def capture_draft(self):
        from PASS.gui.property_state import capture_field
        return {
            "method": self.method.currentData(),
            "solver": self.solver.currentData(),
            "fields": {
                key: capture_field(field)
                for key, field in self.fields.items()
            }
        }

    def restore_draft(self, state):
        from PASS.gui.property_state import restore_field
        self.method.setCurrentIndex(self.method.findData(state["method"]))
        self.solver.setCurrentIndex(max(0, self.solver.findData(state["solver"])))
        for key, value in state["fields"].items():
            restore_field(self.fields[key], value)
        # An old optional-field draft must not disable a now-required PIC field.
        self._solver_changed()


class CollisionSettingsEditor(StructuredField):

    def __init__(self, value, base_dir):
        super().__init__()
        self.original = model_draft(BeamBeamConfiguration, value)
        self.fields = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)
        general = StructuredField()
        form = QFormLayout(general)
        self.general_form = form
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.addRow(QLabel("两个输入按 beam0、beam1 的顺序运行；两束引用 IP 的次序与次数必须一致。"))
        for name, info in BeamBeamConfiguration.model_fields.items():
            key = info.alias or name
            if key in {"Beams", "Sources", "Luminosity"}:
                continue
            if key == "Bunch pairs":
                self.automatic_pairs = QCheckBox("按相同 bunch ID 自动配对")
                self.automatic_pairs.setChecked(self.original[key] is None)
                form.addRow(self.automatic_pairs)
                field = NumericTable([Column("beam0 bunch ID", True, 0), Column("beam1 bunch ID", True, 0)], self.original[key] or [])
                self.automatic_pairs.toggled.connect(lambda enabled, field=field: self.general_form.setRowVisible(field, not enabled))
                self.automatic_pairs.toggled.connect(self.changed)
            else:
                field = make_editor(info.annotation, self.original[key], key, base_dir)
            self.fields[key] = field
            form.addRow(key, field)
            if key == "Bunch pairs":
                form.setRowVisible(field, self.original[key] is not None)
            field.changed.connect(self.changed)
        self.tabs.addTab(general, "碰撞")
        self.sources = {}
        for side in ("0", "1"):
            self.sources[side] = CollisionSourceEditor((self.original["Sources"] or {}).get(side, {"Slice set": "collision"}), base_dir)
            self.tabs.addTab(self.sources[side], "beam" + side + " 源")
            self.sources[side].changed.connect(self.changed)
        self.luminosity = SchemaEditor(BeamBeamLuminosityConfig, self.original["Luminosity"], base_dir)
        for label in self.luminosity.labels.values():
            label.setMinimumWidth(190)
            label.setMaximumWidth(300)
        self.tabs.addTab(self.luminosity, "亮度")
        self.luminosity.changed.connect(self.changed)
        self.fields["Mode"].input.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    def _mode_changed(self, *_args):
        field = self.fields["Weak beam"]
        active = self.fields["Mode"].input.currentData() == "weak-strong"
        field.enabled_box.setChecked(active)
        field.enabled_box.hide()
        self.general_form.setRowVisible(field, active)
        if active and not field.editor.input.text():
            field.editor.input.setText("0")
        self.changed.emit()

    def get_value(self):
        values = {key: field.get_value() for key, field in self.fields.items() if key != "Bunch pairs"}
        values["Bunch pairs"] = None if self.automatic_pairs.isChecked() else self.fields["Bunch pairs"].get_value()
        values["Sources"] = {side: field.get_value() for side, field in self.sources.items()}
        luminosity = self.luminosity.get_value()
        if luminosity["Enabled"] or self.original["Luminosity"] is not None:
            values["Luminosity"] = luminosity
        return BeamBeamConfiguration.model_validate(values).model_dump(by_alias=True, mode="json")

    def capture_draft(self):
        from PASS.gui.property_state import capture_field
        return {
            "fields": {
                key: capture_field(field)
                for key, field in self.fields.items()
            },
            "automatic_pairs": self.automatic_pairs.isChecked(),
            "sources": {
                side: capture_field(field)
                for side, field in self.sources.items()
            },
            "luminosity": capture_field(self.luminosity),
            "tab": self.tabs.currentIndex()
        }

    def restore_draft(self, state):
        from PASS.gui.property_state import restore_field
        for key, value in state["fields"].items():
            restore_field(self.fields[key], value)
        self.automatic_pairs.setChecked(state["automatic_pairs"])
        for side, value in state["sources"].items():
            restore_field(self.sources[side], value)
        restore_field(self.luminosity, state["luminosity"])
        self.tabs.setCurrentIndex(state["tab"])


class BeamBeamConfigurationEditor(StructuredField):

    def __init__(self, block, sequence, base_dir):
        super().__init__()
        self.base_dir, self.resources = base_dir, deepcopy(block.get("Configurations", {}))
        self.references = {
            name: point["Configuration"]
            for name, point in sequence.items() if isinstance(point, dict) and point.get("Configuration") is not None and (
                point.get("Command") in {"BeamBeam", "CrossingAngle"} or point.get("Command") == "Slicer" and point.get("Purpose") == "beam_beam")
        }
        self.active, self.editor = None, None
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 0, 0, 0)
        self.enabled = QCheckBox("启用束束效应（半选：由另一输入声明）")
        self.enabled.setTristate(True)
        self.enabled.setCheckState(Qt.PartiallyChecked if "Enabled" not in block else Qt.Checked if block["Enabled"] else Qt.Unchecked)
        self.enabled.stateChanged.connect(self.changed)
        self.root.addWidget(self.enabled)
        hint = QLabel("共享 IP 配置只在两个输入文件之一声明；另一输入只引用名称。两束显式开关必须一致。重命名后请同步另一输入的引用。")
        hint.setWordWrap(True)
        self.root.addWidget(hint)
        row = QHBoxLayout()
        row.addWidget(QLabel("当前 IP"))
        self.selector = Choice()
        self.selector.currentTextChanged.connect(self.select)
        row.addWidget(self.selector, 1)
        for title, action in (("添加", self.add), ("复制", self.copy), ("删除", self.remove)):
            button = QPushButton(title)
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
            self.editor = CollisionSettingsEditor(self.resources[name], self.base_dir)
            self.editor.changed.connect(self.changed)
            self.root.addWidget(self.editor)

    def _commit(self):
        if self.active is None:
            return
        name = self.name.text().strip()
        if not name or name != self.active and name in self.resources:
            raise ValueError("IP 配置名称必须非空且不能重复")
        value = self.editor.get_value()
        self.resources = {name if key == self.active else key: value if key == self.active else item for key, item in self.resources.items()}
        self.references = {point: name if reference == self.active else reference for point, reference in self.references.items()}
        self.active = name

    def _try_commit(self):
        try:
            self._commit()
            return True
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "束束配置无效", str(exc))
            return False

    def select(self, name):
        if name != self.active:
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
        name = self._unique("IP1")
        self.resources[name] = {
            "Mode": "strong-strong",
            "Sources": {
                str(side): {
                    "Slice set": "collision",
                    "Method": "quasi-frozen",
                    "Solver": "gaussian_ellipse_free_space"
                }
                for side in (0, 1)
            }
        }
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
        used = [point for point, reference in self.references.items() if reference == self.active]
        if used:
            QMessageBox.warning(self, "配置正在使用", "请先修改这些命令的引用：" + "、".join(used))
            return
        del self.resources[self.active]
        self._show(next(iter(self.resources), None))
        self.changed.emit()

    def get_value(self):
        self._commit()
        value = {"Configurations": self.resources}
        if self.enabled.checkState() != Qt.PartiallyChecked:
            value["Enabled"] = self.enabled.isChecked()
        return BeamBeamConfig.model_validate(value).model_dump(by_alias=True, mode="json")

    def capture_draft(self):
        from PASS.gui.property_state import capture_field
        return {
            "resources": deepcopy(self.resources),
            "references": deepcopy(self.references),
            "active": self.active,
            "enabled": self.enabled.checkState().value,
            "name": self.name.text(),
            "editor": capture_field(self.editor) if self.editor else None
        }

    def restore_draft(self, state):
        from PASS.gui.property_state import restore_field
        self.resources, self.references = deepcopy(state["resources"]), deepcopy(state["references"])
        self.enabled.setCheckState(Qt.CheckState(state["enabled"]))
        self._show(state["active"])
        self.name.setText(state["name"])
        if self.editor and state.get("editor"):
            restore_field(self.editor, state["editor"])

    def clone_for_preview(self):
        clone = type(self)({"Configurations": self.resources}, {}, self.base_dir)
        clone.restore_draft(self.capture_draft())
        return clone
