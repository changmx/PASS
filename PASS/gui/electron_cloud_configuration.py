"""Named electron-cloud resources with lossless mode and geometry drafts."""

from copy import deepcopy

from pydantic import TypeAdapter
from PySide6.QtWidgets import QCheckBox, QFormLayout, QHBoxLayout, QLabel, QLayout, QLineEdit, QMessageBox, QPushButton, QTabWidget, QVBoxLayout

from PASS.gui.parameters import Choice, SchemaEditor, make_editor, model_draft
from PASS.gui.structured import APERTURES, Column, NumericTable, StructuredField
from PASS.para.schema.electron_cloud import ElectronCloudBuildUpConfiguration, ElectronCloudConfig, ElectronCloudConfiguration
from PASS.para.schema.space_charge import validate_loss_aperture


def _field_help():
    return {
        "Electron density (1/m^3)": ("电子数密度 / m⁻³", "非负真实电子数密度；动态模式中描述初始圆盘，可取零。"),
        "Radius (m)": ("电子云圆盘半径 / m", "必须为正；动态模式要求整个初始圆盘严格位于圆形真空室内。"),
        "Center X (m)": ("圆盘中心 x / m", "初始电子云圆盘相对束流轴的水平偏移。"),
        "Center Y (m)": ("圆盘中心 y / m", "初始电子云圆盘相对束流轴的垂直偏移。"),
        "Number of macro electrons": ("初始宏电子数", "用于电子云采样的正整数样本数；每个宏电子携带相应真实电子权重。"),
        "Random seed": ("随机种子", "不指定时写入 JSON null，保持非确定性；指定整数时可复现初始采样和发射抽样。"),
        "Nx": ("水平网格点数 Nx", "一般至少 3；coupled 要求不小于 5 的奇数。"),
        "Ny": ("垂直网格点数 Ny", "一般至少 3；coupled 要求不小于 5 的奇数。"),
        "Grid Width X (m)": ("网格水平全宽 / m", "正的全宽。coupled 必须覆盖真空室直径，dx = 全宽 / (Nx − 1) ≤ 真空室半径 / 2。"),
        "Grid Width Y (m)": ("网格垂直全宽 / m", "正的全宽。coupled 必须覆盖真空室直径，dy = 全宽 / (Ny − 1) ≤ 真空室半径 / 2。"),
        "Particle Deposition Method": ("粒子电荷沉积方法", "CIC 为线性权重，TSC 为二次权重；同时影响粒子到网格的沉积。"),
        "Aperture type": ("边界形状", "Dirichlet 模式使用接地导体边界；自由空间求解器只允许 default/off，网格不代表导体壁面。"),
        "Aperture value": ("边界尺寸 / m", "尺寸采用半宽、半轴或半径；polygon 顶点按边界顺序填写。"),
        "Chamber radius (m)": ("圆形真空室半径 / m", "必须为正，圆心位于束流轴；初始圆盘中心距加半径必须严格小于此值。"),
        "Beam sigma (m)": ("束流高斯 sigma / m", "build_up 使用的固定圆高斯横向 sigma，必须为正。coupled 使用实际切片粒子，此值不参与计算，但当前输入格式仍要求填写正值。"),
        "Max time step (s)": ("最大积分时间步长 / s", "必须为正；电子推进器可根据动力学限制进一步缩短。"),
        "Magnetic field (T)": ("均匀磁场 [Bx, By, Bz] / T", "按束流局部坐标 x、y、z 顺序填写 3 个分量。"),
        "Magnetic gradient (T/m)": ("正规四极磁场梯度 / (T/m)", "带符号的有限正规四极磁场梯度。"),
        "Initial electron energy (eV)": ("初始电子动能 / eV", "每电子的非负初始动能；初始方向在三维空间中各向同性。"),
        "Primary electrons per beam particle (1/m)": ("每束流粒子每米的初级电子产额 / m⁻¹", "非负预设初级电子源强度，按真实束流粒子数计。"),
        "Primary macro electrons": ("每次初级发射的宏电子数", "每个非零初级发射事件产生的正整数宏电子样本数。"),
        "Secondary yield max": ("二次电子产额峰值", "非负的未截断真二次产额曲线峰值；零表示吸收壁。"),
        "Secondary peak energy (eV)": ("二次产额峰值入射能量 / eV", "未截断产额曲线达到峰值时的正入射动能。"),
        "Secondary shape": ("二次产额曲线形状参数", "必须严格大于 1。"),
        "Emission energy (eV)": ("发射能量 / eV", "正的初级发射能量及名义二次发射能量。"),
        "Max macro electrons": ("最大宏电子数", "正整数资源上限；超限时报错，不通过丢弃电荷继续运行。"),
        "Max steps": ("每时间区间的最大积分步数", "每个物理时间区间内允许的积分步数，必须为正整数。"),
        "Max wall hits per step": ("单步最大壁面碰撞次数", "每个电子在单个积分步内允许的壁面事件数，必须为正整数。"),
    }


class ElectronCloudApertureEditor(StructuredField):
    """Keep incomplete dimensions for every shape without supplying a chamber size."""

    def __init__(self, kind, value, base_dir):
        super().__init__()
        self.base_dir = base_dir
        self.kind, self.fields, self.polygon, self.drafts = None, [], None, {}
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.set_kind(kind, value)

    def _current_draft(self):
        from PASS.gui.property_state import capture_field
        return {"fields": [capture_field(field) for field in self.fields], "polygon": capture_field(self.polygon) if self.polygon else None}

    def set_kind(self, kind, value=None):
        from PASS.gui.property_state import restore_field
        if self.kind == kind and value is None:
            return
        if self.kind is not None:
            self.drafts[self.kind] = self._current_draft()
        while self.form.rowCount():
            self.form.removeRow(0)
        self.kind, self.fields, self.polygon = kind, [], None
        if kind == "polygon":
            self.polygon = NumericTable([Column("顶点 x / m"), Column("顶点 y / m")], value or [], "按边界顺序填写至少 3 个顶点；最后一点自动连接第一点。")
            self.polygon.changed.connect(self.changed)
            self.form.addRow(self.polygon)
        elif kind in {"default", "off"}:
            label = QLabel("边界尺寸由当前模型确定；default/off 不填写尺寸。")
            label.setWordWrap(True)
            self.form.addRow(label)
        else:
            for index, label in enumerate(APERTURES.get(kind, ((), ()))[0]):
                number = value[index] if isinstance(value, list) and index < len(value) else None
                field = make_editor(float, number, label, self.base_dir)
                field.changed.connect(self.changed)
                self.fields.append(field)
                self.form.addRow(label, field)
        if value is None and kind in self.drafts:
            state = self.drafts[kind]
            for field, draft in zip(self.fields, state["fields"]):
                restore_field(field, draft)
            if self.polygon and state.get("polygon"):
                restore_field(self.polygon, state["polygon"])
        self.changed.emit()

    def get_value(self):
        value = self.polygon.get_value() if self.polygon else [field.get_value() for field in self.fields]
        return validate_loss_aperture(self.kind, value)

    def capture_draft(self):
        drafts = deepcopy(self.drafts)
        drafts[self.kind] = self._current_draft()
        return {"kind": self.kind, "drafts": drafts}

    def restore_draft(self, state):
        self.kind = None
        self.drafts = deepcopy(state["drafts"])
        self.set_kind(state["kind"])


class ElectronCloudBuildUpEditor(SchemaEditor):

    def __init__(self, value, base_dir):
        super().__init__(ElectronCloudBuildUpConfiguration, value, base_dir)
        # Stacked rows keep long physical labels readable in the narrow property pane.
        while self.form.rowCount():
            self.form.takeRow(0)
        self.form.setSizeConstraint(QLayout.SetMinimumSize)
        for key, field in self.fields.items():
            label, help_text = _field_help()[key]
            self.labels[key].setText(label + (" *" if key in {"Chamber radius (m)", "Beam sigma (m)", "Max time step (s)"} else ""))
            self.labels[key].setMaximumWidth(16777215)
            self.labels[key].setToolTip(help_text)
            field.setToolTip(help_text)
            self.form.addRow(self.labels[key])
            self.form.addRow(field)


class ElectronCloudSettingsEditor(StructuredField):
    """All cloud parameters; hidden drafts survive mode and solver changes."""

    def __init__(self, value, base_dir):
        super().__init__()
        self.base_dir = base_dir
        self.original = model_draft(ElectronCloudConfiguration, value if isinstance(value, dict) else None)
        self.fields, self.labels, self.forms = {}, {}, {}
        self._selection = None
        self._solver_preferences, self._aperture_preferences = {}, {}
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.mode, self.solver = Choice(), Choice()
        for label, mode in (("冻结电子云 frozen", "frozen"), ("电子云积累 build_up", "build_up"), ("束流与电子云耦合 coupled", "coupled")):
            self.mode.addItem(label, mode)
        self.mode.setCurrentIndex(max(0, self.mode.findData(self.original["Mode"])))
        self.fields.update({"Mode": self.mode, "Solver": self.solver})
        form.addRow("模式", self.mode)
        form.addRow("求解器", self.solver)
        root.addLayout(form)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        root.addWidget(self.hint)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)
        source, grid = StructuredField(), StructuredField()
        source_form, grid_form = QFormLayout(source), QFormLayout(grid)
        for layout in (source_form, grid_form):
            layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
            layout.setSizeConstraint(QLayout.SetMinimumSize)
        self.tabs.addTab(source, "云与采样")
        self.tabs.addTab(grid, "网格/边界")
        self.buildup = ElectronCloudBuildUpEditor(self.original["Build up"], base_dir)
        self.fields["Build up"] = self.buildup
        self.tabs.addTab(self.buildup, "运动/发射")
        for index, title in enumerate(("电子云与初始采样", "网格与边界", "电子运动与壁面发射")):
            self.tabs.setTabToolTip(index, title)
        self.buildup.changed.connect(self.changed)
        source_keys = {"Electron density (1/m^3)", "Radius (m)", "Center X (m)", "Center Y (m)", "Number of macro electrons", "Random seed"}
        for name, info in ElectronCloudConfiguration.model_fields.items():
            key = info.alias or name
            if key in self.fields:
                continue
            if key == "Aperture type":
                field = Choice()
            elif key == "Aperture value":
                field = ElectronCloudApertureEditor(self.original["Aperture type"], self.original[key], base_dir)
            else:
                field = make_editor(info.annotation, self.original[key], key, base_dir)
            title, help_text = _field_help()[key]
            label = QLabel(title + (" *" if info.is_required() else ""))
            label.setWordWrap(True)
            label.setToolTip(help_text)
            field.setToolTip(help_text)
            layout = source_form if key in source_keys else grid_form
            layout.addRow(label, field)
            self.fields[key], self.labels[key], self.forms[key] = field, label, layout
            if key != "Aperture type":
                field.changed.connect(self.changed)
        self.fields["Random seed"].enabled_box.setText("指定整数种子（关闭写入 null）")
        self._mode_changed(preferred_solver=self.original["Solver"], preferred_aperture=self.original["Aperture type"])
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.solver.currentIndexChanged.connect(self._solver_changed)
        self.fields["Aperture type"].currentIndexChanged.connect(self._aperture_changed)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        policy = QFormLayout.WrapAllRows if event.size().width() < 420 else QFormLayout.WrapLongRows
        for layout in self.forms.values():
            if layout.rowWrapPolicy() != policy:
                layout.setRowWrapPolicy(policy)

    def _mode_changed(self, *_args, preferred_solver=None, preferred_aperture=None):
        mode = self.mode.currentData()
        solvers = (["uniform_round_free_space", "fd_dirichlet", "dst_dirichlet", "fft_free_space"]
                   if mode == "frozen" else ["round_gaussian_beam"] if mode == "build_up" else ["fd_dirichlet"])
        if self._selection:
            self._solver_preferences[self._selection[0]] = self._selection[1]
        preferred = preferred_solver or self._solver_preferences.get(mode, solvers[0])
        self.solver.blockSignals(True)
        self.solver.clear()
        self.solver.addItems(solvers)
        for index, solver in enumerate(solvers):
            self.solver.setItemData(index, solver)
        if preferred not in solvers:
            self.solver.addItem(preferred + "（不适用于此模式）", preferred)
        self.solver.setCurrentIndex(self.solver.findData(preferred))
        self.solver.setEnabled(len(solvers) > 1 or preferred not in solvers)
        self.solver.blockSignals(False)
        self._solver_changed(preferred_aperture=preferred_aperture)

    def _solver_changed(self, *_args, preferred_aperture=None):
        mode, solver = self.mode.currentData(), self.solver.currentData()
        aperture = self.fields["Aperture type"]
        if self._selection:
            self._aperture_preferences[self._selection] = aperture.currentData()
        self._selection = (mode, solver)
        shapes = (["default", "off"] if mode != "frozen" or solver in {"uniform_round_free_space", "fft_free_space"} else
                  ["default", "rectangle"] if solver == "dst_dirichlet" else [kind for kind in APERTURES if kind != "off"])
        preferred = preferred_aperture or self._aperture_preferences.get(self._selection, "default")
        aperture.blockSignals(True)
        aperture.clear()
        for shape in shapes:
            aperture.addItem(shape, shape)
        if preferred not in shapes:
            aperture.addItem(preferred + "（不适用于此模型）", preferred)
        aperture.setCurrentIndex(aperture.findData(preferred))
        aperture.blockSignals(False)
        self._aperture_changed()
        active = {"Mode", "Solver", "Electron density (1/m^3)", "Radius (m)", "Center X (m)", "Center Y (m)", "Aperture type", "Aperture value"}
        numerical = mode == "coupled" or mode == "frozen" and solver != "uniform_round_free_space"
        if mode != "frozen" or numerical:
            active.update({"Number of macro electrons", "Random seed"})
        if mode == "frozen" or numerical:
            active.update({"Nx", "Ny", "Grid Width X (m)", "Grid Width Y (m)"})
        if numerical:
            active.add("Particle Deposition Method")
        if mode != "frozen":
            active.add("Build up")
        self.active_fields = active
        for key, layout in self.forms.items():
            layout.setRowVisible(self.fields[key], key in active)
        self.tabs.setTabVisible(2, mode != "frozen")
        analytic = mode == "frozen" and solver == "uniform_round_free_space"
        self.tabs.setTabText(1, "诊断网格" if analytic else "网格/边界")
        hint = {
            "frozen": "冻结模式只施加给定电子云的场；解析均匀圆云不需要宏电子采样。FD 支持有限导体边界；DST 限定为整个网格矩形；FFT 和解析模型为自由空间。",
            "build_up": "积累模式由固定圆高斯束流驱动电子运动与壁面发射，不反踢束流，也不计电子自场。边界采用 Build up 中的圆形真空室。",
            "coupled": "耦合模式计算电子云 PIC 自场与横向束流踢角。Nx/Ny 必须为 ≥5 的奇数，网格全宽 ≥ 真空室直径，网格间距 ≤ 真空室半径的一半。边界采用圆形真空室。",
        }[mode]
        if analytic:
            hint += " Nx/Ny 和网格全宽仅控制 Save fields 的诊断采样，不影响解析踢角。"
        if mode != "frozen":
            hint += " 动态模式需要同圈同位置、覆盖全部存活粒子的非周期 z_rel Slicer。"
        self.hint.setText(hint)
        self.changed.emit()

    def _aperture_changed(self, *_args):
        self.fields["Aperture value"].set_kind(self.fields["Aperture type"].currentData())
        self.changed.emit()

    def get_value(self):
        values = deepcopy(self.original)
        for name, info in ElectronCloudConfiguration.model_fields.items():
            key = info.alias or name
            field = self.fields[key]
            if key == "Build up" and self.mode.currentData() == "frozen":
                values[key] = None
            elif key in self.active_fields:
                values[key] = field.currentData() if isinstance(field, Choice) else field.get_value()
            else:
                # An inactive incomplete draft must not block another physical mode.
                try:
                    values[key] = TypeAdapter(info.rebuild_annotation()).validate_python(field.get_value())
                except (ValueError, TypeError):
                    pass
        return ElectronCloudConfiguration.model_validate(values).model_dump(by_alias=True, mode="json")

    def capture_draft(self):
        from PASS.gui.property_state import capture_field
        return {
            "fields": {
                key: capture_field(field)
                for key, field in self.fields.items()
            },
            "solver_preferences": deepcopy(self._solver_preferences),
            "aperture_preferences": [[list(key), value] for key, value in self._aperture_preferences.items()],
            "tab": self.tabs.currentIndex(),
        }

    def restore_draft(self, state):
        from PASS.gui.property_state import restore_field
        self.mode.blockSignals(True)
        restore_field(self.mode, state["fields"]["Mode"])
        self.mode.blockSignals(False)
        self._selection = None
        self._solver_preferences = deepcopy(state.get("solver_preferences", {}))
        self._aperture_preferences = {tuple(key): value for key, value in state.get("aperture_preferences", [])}
        self._mode_changed(preferred_solver=state["fields"]["Solver"]["data"], preferred_aperture=state["fields"]["Aperture type"]["data"])
        for key, value in state["fields"].items():
            if key not in {"Mode", "Solver", "Aperture type"}:
                restore_field(self.fields[key], value)
        self.tabs.setCurrentIndex(state.get("tab", 0))


class ElectronCloudConfigurationEditor(StructuredField):
    """Edit shared parameters; each referencing command retains its own cloud state."""

    def __init__(self, block, sequence, base_dir):
        super().__init__()
        normalized = ElectronCloudConfig._normalize_fields(block)
        self.base_dir, self.resources = base_dir, deepcopy(normalized.get("configurations", {}))
        self.references = {
            name: point["Configuration"]
            for name, point in sequence.items()
            if isinstance(point, dict) and point.get("Command") == "ElectronCloud" and point.get("Configuration") is not None
        }
        self.active, self.editor, self._baseline = None, None, None
        self._drafts = {}
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 0, 0, 0)
        self.enabled = QCheckBox("启用电子云")
        self.enabled.setChecked(normalized.get("enabled", False))
        self.enabled.toggled.connect(self.changed)
        self.root.addWidget(self.enabled)
        hint = QLabel("命名配置共享参数；各 ElectronCloud 点的电子云状态独立。关闭开关会保留配置，重新启用时检查完整参数。标有 * 的物理量必须填写。")
        hint.setWordWrap(True)
        self.root.addWidget(hint)
        row = QHBoxLayout()
        row.addWidget(QLabel("当前配置"))
        self.selector = Choice()
        self.selector.currentTextChanged.connect(self.select)
        row.addWidget(self.selector, 1)
        self.root.addLayout(row)
        row = QHBoxLayout()
        for title, action in (("添加", self.add), ("复制", self.copy), ("删除", self.remove)):
            button = QPushButton(title)
            button.clicked.connect(action)
            row.addWidget(button)
        self.root.addLayout(row)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.textChanged.connect(self.changed)
        form.addRow("配置名称", self.name)
        self.root.addLayout(form)
        self._show(next(iter(self.resources), None))

    def _show(self, name):
        if self.editor is not None:
            self.root.removeWidget(self.editor)
            self.editor.setParent(None)
            self.editor.deleteLater()
        self.active, self.editor = name, None
        self.selector.blockSignals(True)
        self.selector.clear()
        self.selector.addItems(list(self.resources))
        self.selector.setCurrentText(name or "")
        self.selector.blockSignals(False)
        self.name.setText(name or "")
        self.name.setEnabled(name is not None)
        if name is not None:
            self.editor = ElectronCloudSettingsEditor(self.resources[name], self.base_dir)
            if name in self._drafts:
                self.editor.restore_draft(self._drafts[name])
            self.editor.changed.connect(self.changed)
            self.root.addWidget(self.editor)
            self._baseline = self.editor.capture_draft()

    def _commit(self):
        if self.active is None:
            return
        name = self.name.text().strip()
        if not name or name != self.active and name in self.resources:
            raise ValueError("电子云配置名称必须非空且不能重复")
        draft = self.editor.capture_draft()
        # Disabled blocks can contain ignored legacy data; preserve untouched input.
        value = self.editor.get_value() if draft["fields"] != self._baseline["fields"] else deepcopy(self.resources[self.active])
        old_name = self.active
        self.resources = {name if key == old_name else key: value if key == old_name else item for key, item in self.resources.items()}
        self.references = {point: name if reference == old_name else reference for point, reference in self.references.items()}
        self._drafts.pop(old_name, None)
        self._drafts[name] = draft
        self.active, self._baseline = name, draft

    def _try_commit(self):
        try:
            self._commit()
            return True
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "电子云配置无效", str(exc))
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
        name = self._unique("cloud")
        self.resources[name] = {"Mode": "frozen", "Solver": "uniform_round_free_space"}
        self._show(name)
        self.changed.emit()

    def copy(self):
        if self.active is None or not self._try_commit():
            return
        name = self._unique(self.active + "_copy")
        self.resources[name] = deepcopy(self.resources[self.active])
        if self.active in self._drafts:
            self._drafts[name] = deepcopy(self._drafts[self.active])
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
        self._drafts.pop(self.active, None)
        self._show(next(iter(self.resources), None))
        self.changed.emit()

    def get_value(self):
        self._commit()
        value = {"Enabled": self.enabled.isChecked(), "Configurations": deepcopy(self.resources)}
        if value["Enabled"]:
            return ElectronCloudConfig.model_validate(value).model_dump(by_alias=True, mode="json")
        return value

    def capture_draft(self):
        return {
            "resources": deepcopy(self.resources),
            "references": deepcopy(self.references),
            "drafts": deepcopy(self._drafts),
            "active": self.active,
            "enabled": self.enabled.isChecked(),
            "name": self.name.text(),
            "baseline": deepcopy(self._baseline),
            "editor": self.editor.capture_draft() if self.editor else None,
        }

    def restore_draft(self, state):
        self.resources, self.references = deepcopy(state["resources"]), deepcopy(state["references"])
        self._drafts = deepcopy(state.get("drafts", {}))
        self.enabled.setChecked(state["enabled"])
        self._show(state["active"])
        self.name.setText(state["name"])
        if self.editor and state.get("editor"):
            self.editor.restore_draft(state["editor"])
        self._baseline = deepcopy(state.get("baseline", self._baseline))

    def clone_for_preview(self):
        clone = type(self)({"Configurations": self.resources}, {}, self.base_dir)
        clone.restore_draft(self.capture_draft())
        return clone
