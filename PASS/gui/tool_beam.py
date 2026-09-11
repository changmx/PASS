"""Independent reference-particle and beam-intensity calculator page."""
import math
import re

from PySide6.QtCore import QSignalBlocker, QStringListModel, Qt, Signal
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QCompleter,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget)

from PASS.gui.structured import IntegerSpinBox, ScientificSpinBox
from PASS.gui.tool_formulas import BEAM_FORMULAS, FormulaDialog
from PASS.gui.beam_calculator import (beam_current, beam_power, circulating_beam,
    particle_rate, pulsed_beam, solve_kinematics)
from PASS.tool.particles import ParticleSpec, SPECIAL_PARTICLES, resolve_particle, search_particles
from PASS.utils.constants import const


def number(value=0, minimum=0):
    widget = ScientificSpinBox(value)
    widget.setMinimum(minimum)
    widget.setButtonSymbols(QAbstractSpinBox.NoButtons)
    return widget


def output():
    widget = QLineEdit("—")
    widget.setReadOnly(True)
    widget.setObjectName("readonlyField")
    widget.setMinimumWidth(80)
    return widget


def format_number(value):
    if value is None:
        return "无有限值"
    if not math.isfinite(value):
        return "—"
    return f"{value:.12g}"


def hint(text=""):
    widget = QLabel(text)
    widget.setObjectName("muted")
    widget.setWordWrap(True)
    return widget


class ParticleEditor(QGroupBox):
    """Shared species/ion inputs; search selection is preview-only."""
    changed = Signal()

    def __init__(self, *, searchable=True, title="粒子信息"):
        super().__init__(title)
        self._updating = False
        self._ion_values = (12, 6, 6)
        grid = QGridLayout(self)
        self.mass_number = IntegerSpinBox(12, 0, 1000)
        self.mass_number.setToolTip("A = Z + N；同一核素改变电荷态时 A 不变，实际质量比 μ 会变化。")
        self.charge_state = IntegerSpinBox(6, -1000, 118)
        self.atomic_number = IntegerSpinBox(6, 0, 118)
        self.atomic_number.setSpecialValueText("未指定")
        self.species = QComboBox()
        self.species.addItem("离子 / 原子（A、q、Z）", "ion")
        for key, entry in SPECIAL_PARTICLES.items():
            self.species.addItem(entry[0], key)
        for col, (label, widget) in enumerate((("质量数 A", self.mass_number), ("电荷态 q", self.charge_state),
                                             ("质子数 Z", self.atomic_number), ("粒子类型", self.species))):
            grid.addWidget(QLabel(label), 0, col)
            grid.addWidget(widget, 1, col)
        for widget in (self.mass_number, self.charge_state, self.atomic_number):
            widget.setButtonSymbols(QAbstractSpinBox.NoButtons)
            widget.valueChanged.connect(self.refresh)
        grid.setColumnStretch(3, 1)
        self.identity = hint()
        grid.addWidget(self.identity, 2, 0, 1, 4)
        self.species.currentIndexChanged.connect(self.change_species)
        if searchable:
            self.search = QLineEdit()
            self.search.setPlaceholderText("搜索：carbon / 238U35+ / 电子 / muon / pi+ / 中子")
            self.search_button = QPushButton("使用搜索结果")
            row = QHBoxLayout()
            row.addWidget(self.search, 1)
            row.addWidget(self.search_button)
            grid.addLayout(row, 3, 0, 1, 4)
            self.search_model = QStringListModel(search_particles(""), self)
            self.completer = QCompleter(self.search_model, self)
            self.completer.setCaseSensitivity(Qt.CaseInsensitive)
            self.completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
            self.search.setCompleter(self.completer)
            self.search.textEdited.connect(self.search_changed)
            self.completer.activated[str].connect(self.preview_search)
            self.search.returnPressed.connect(lambda: self.preview_search(self.search.text()))
            self.search_button.clicked.connect(lambda: self.apply_search(self.search.text()))
            self.search_status = hint("候选只预览；点击“使用搜索结果”替换粒子类型、A、q、Z。")
            grid.addWidget(self.search_status, 4, 0, 1, 4)
        self.refresh()

    def current(self):
        key = self.species.currentData()
        if key != "ion":
            return ParticleSpec.from_species(key)
        return ParticleSpec(self.mass_number.value(), self.charge_state.value(), self.atomic_number.value() or None)

    def set_particle(self, particle):
        self._updating = True
        with QSignalBlocker(self.species):
            self.species.setCurrentIndex(self.species.findData(particle.species))
        for widget, value in zip((self.mass_number, self.charge_state, self.atomic_number),
                                 (particle.mass_number, particle.charge_state, particle.atomic_number or 0)):
            with QSignalBlocker(widget):
                widget.setValue(value)
            widget.setEnabled(particle.species == "ion")
        self._updating = False
        self.refresh()

    def change_species(self):
        key = self.species.currentData()
        if key == "ion":
            particle = ParticleSpec(*self._ion_values)
        else:
            if self.mass_number.isEnabled():
                self._ion_values = (self.mass_number.value(), self.charge_state.value(), self.atomic_number.value() or None)
            particle = ParticleSpec.from_species(key)
        self.set_particle(particle)

    def refresh(self):
        if self._updating:
            return
        try:
            particle = self.current()
            if particle.species != self.species.currentData():
                self.set_particle(particle)
                return
            self.identity.setText(particle.label + "；" + particle.mass_note)
            self.identity.setToolTip(particle.mass_record.reference)
        except (ValueError, OverflowError) as exc:
            self.identity.setText(str(exc))
        self.changed.emit()

    def search_changed(self, text):
        self.search_model.setStringList(search_particles(text))
        if text.strip():
            self.completer.complete()

    @staticmethod
    def search_spec(text):
        try:
            return resolve_particle(text)
        except ValueError:
            choices = search_particles(text) if re.fullmatch(r"[^\d+\-]+|\d+", text.strip()) else []
            if not choices:
                raise
            return resolve_particle(choices[0])

    def preview_search(self, text):
        self.search.setText(text)
        try:
            p = self.search_spec(text)
            self.search_status.setText(f"待应用：{p.label}。点击“使用搜索结果”应用。")
        except ValueError as exc:
            self.search_status.setText(str(exc))

    def apply_search(self, text):
        try:
            particle = self.search_spec(text)
        except ValueError as exc:
            self.search_status.setText(str(exc))
            return
        self.set_particle(particle)
        self.search.setText(particle.label)
        self.search_status.setText(f"已全部替换：{particle.label}，A={particle.mass_number}，q={particle.charge_state:+d}，Z={particle.atomic_number or 0}。")


class BeamCalculatorPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.kinematics = None
        self._updating = False
        self.formula_dialog = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        heading = QHBoxLayout()
        title = QLabel("束流参数计算器")
        title.setObjectName("formTitle")
        heading.addWidget(title)
        heading.addStretch()
        self.formula_button = QPushButton("详细公式…")
        self.formula_button.clicked.connect(self.show_formulas)
        heading.addWidget(self.formula_button)
        self.copy_button = QPushButton("复制结果")
        self.copy_button.clicked.connect(self.copy_results)
        heading.addWidget(self.copy_button)
        outer.addLayout(heading)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        self.particle_editor = ParticleEditor()
        for name in ("mass_number", "charge_state", "atomic_number", "species", "identity", "search", "search_button",
                     "search_model", "completer", "search_status"):
            setattr(self, name, getattr(self.particle_editor, name))
        self.mass_description = self.particle_editor.identity
        layout.addWidget(self.particle_editor)
        kinematics = QGroupBox("运动学与磁刚度")
        kg = QGridLayout(kinematics)
        self.known = QComboBox()
        for label, key in (("每核子动能 Ek", "kinetic_energy"), ("每核子总能量 E/A", "total_energy"),
                           ("每核子动量 p/A", "momentum"), ("磁刚度 Bρ", "brho"),
                           ("速度比 β", "beta"), ("相对论因子 γ", "gamma")):
            self.known.addItem(label, key)
        self.known_value = number(100)
        self.known_unit = QLabel("(AMeV)")
        known_row = QHBoxLayout()
        known_row.addWidget(QLabel("已知量"))
        known_row.addWidget(self.known, 1)
        known_row.addWidget(self.known_value, 1)
        known_row.addWidget(self.known_unit)
        kg.addLayout(known_row, 0, 0, 1, 4)
        self.results = {}
        fields = (("brho", "Bρ (T·m)"), ("gamma", "γ"), ("beta", "β"),
                  ("beta_gamma", "βγ"), ("velocity", "v (m/s)"), ("momentum_ev_c", "p/A (GeV/c)"),
                  ("ek_per_nucleon_ev", "Ek (AMeV)"), ("total_energy_ev", "E/A (MeV)"),
                  ("rest_energy", "静止质量 m₀/A (MeV/c²)"), ("mass_ratio", "质量比 μ = m₀/u"))
        self.result_labels = dict(fields)
        self.result_label_widgets = {}
        self.ek_label = None
        for index, (key, label) in enumerate(fields):
            row, column = 1 + index // 2, (index % 2) * 2
            field_label = QLabel(label)
            self.result_label_widgets[key] = field_label
            kg.addWidget(field_label, row, column)
            value = output()
            self.results[key] = value
            if key == "mass_ratio":
                value.setToolTip("无量纲，数值等于该粒子的静止质量以 u 表示时的数值；不是整数质量数 A。")
            kg.addWidget(value, row, column + 1)
            if key == "ek_per_nucleon_ev":
                self.ek_label = field_label
        kg.setColumnStretch(1, 1)
        kg.setColumnStretch(3, 1)
        self.kin_error = hint()
        kg.addWidget(self.kin_error, 6, 0, 1, 4)
        layout.addWidget(kinematics)
        power = QGroupBox("流强、功率与环内储能")
        pl = QVBoxLayout(power)
        self.power_mode = QComboBox()
        for label, key in (("流强 ↔ 功率", "direct"), ("脉冲束流", "pulse"), ("环内储能与循环流强", "ring")):
            self.power_mode.addItem(label, key)
        pl.addWidget(self.power_mode)
        self.power_inputs = QStackedWidget()
        direct = QWidget()
        dl = QFormLayout(direct)
        self.power_known = QComboBox()
        for label, key in (("电流", "current"), ("束流功率", "power"), ("粒子率", "rate")):
            self.power_known.addItem(label, key)
        self.time_basis = QComboBox()
        self.time_basis.addItem("平均值", "average")
        self.time_basis.addItem("瞬时值 / 已知峰值", "instant")
        dl.addRow("已知量", self.power_known)
        dl.addRow("时间口径", self.time_basis)
        self.direct_value = number(0)
        self.direct_label = QLabel("平均电流 (mA)")
        dl.addRow(self.direct_label, self.direct_value)
        self.power_inputs.addWidget(direct)
        pulse = QWidget()
        pul = QGridLayout(pulse)
        self.pulse_count = number(0)
        self.repetition_frequency = number(1)
        self.use_duration = QCheckBox("提供脉宽 (μs)")
        self.pulse_duration = number(1)
        self.pulse_duration.setEnabled(False)
        pul.addWidget(QLabel("每脉冲真实粒子数"), 0, 0)
        pul.addWidget(self.pulse_count, 0, 1)
        pul.addWidget(QLabel("重复频率 (Hz)"), 0, 2)
        pul.addWidget(self.repetition_frequency, 0, 3)
        pul.addWidget(self.use_duration, 1, 0)
        pul.addWidget(self.pulse_duration, 1, 1)
        self.power_inputs.addWidget(pulse)
        ring = QWidget()
        rl = QGridLayout(ring)
        self.ring_count = number(0)
        self.circumference = number(100)
        rl.addWidget(QLabel("环内真实粒子总数"), 0, 0)
        rl.addWidget(self.ring_count, 0, 1)
        rl.addWidget(QLabel("环周长 C (m)"), 0, 2)
        rl.addWidget(self.circumference, 0, 3)
        self.power_inputs.addWidget(ring)
        pl.addWidget(self.power_inputs)
        self.power_note = hint()
        pl.addWidget(self.power_note)
        self.power_outputs = QWidget()
        self.power_result_layout = QGridLayout(self.power_outputs)
        self.power_result_layout.setContentsMargins(0, 0, 0, 0)
        self.power_results = {}
        self.power_labels = {}
        for index, key in enumerate(("current", "power", "rate", "frequency", "period", "energy", "pulse_current", "pulse_power")):
            label, value = QLabel(), output()
            self.power_labels[key] = label
            self.power_results[key] = value
            self.power_result_layout.addWidget(label, index // 2, index % 2 * 2)
            self.power_result_layout.addWidget(value, index // 2, index % 2 * 2 + 1)
        self.power_result_layout.setColumnStretch(1, 1)
        self.power_result_layout.setColumnStretch(3, 1)
        pl.addWidget(self.power_outputs)
        self.power_error = hint()
        pl.addWidget(self.power_error)
        layout.addWidget(power)
        layout.addStretch()
        self.known_value.valueChanged.connect(self.recalculate)
        self.known.currentIndexChanged.connect(self.change_known)
        self.particle_editor.changed.connect(self.recalculate)
        self.power_mode.currentIndexChanged.connect(self.change_power_mode)
        self.power_known.currentIndexChanged.connect(self.change_power_mode)
        self.time_basis.currentIndexChanged.connect(self.change_power_mode)
        for widget in (self.direct_value, self.pulse_count, self.repetition_frequency,
                       self.pulse_duration, self.ring_count, self.circumference):
            widget.valueChanged.connect(self.recalculate_power)
        self.use_duration.toggled.connect(self.pulse_duration.setEnabled)
        self.use_duration.toggled.connect(self.recalculate_power)
        self.change_power_mode()
        self.recalculate()

    def _search_changed(self, text):
        self.particle_editor.search_changed(text)

    def preview_search(self, text):
        self.particle_editor.preview_search(text)

    def apply_search(self, text):
        self.particle_editor.apply_search(text)

    def _spec(self):
        return self.particle_editor.current()

    def _scale(self, key):
        return 1e6 if key in ("kinetic_energy", "total_energy") else 1e9 if key == "momentum" else 1.

    def change_known(self):
        key = self.known.currentData()
        if self.kinematics is not None:
            names = {"kinetic_energy": "ek_per_nucleon_ev", "total_energy": "total_energy_ev", "momentum": "momentum_ev_c"}
            raw = getattr(self.kinematics, names.get(key, key))
            if key in ("total_energy", "momentum"):
                raw /= self.kinematics.particle.energy_divisor
            value = (raw or 0.) / self._scale(key)
            with QSignalBlocker(self.known_value):
                self.known_value.setValue(value)
        self.recalculate()

    def recalculate(self):
        if self._updating:
            return
        key = self.known.currentData()
        try:
            spec = self._spec()
            self.update_units(spec)
            value = self.known_value.value() * self._scale(key)
            if key in ("total_energy", "momentum"):
                value *= spec.energy_divisor
            self.kinematics = solve_kinematics(spec, key, value)
            for name, widget in self.results.items():
                if name == "rest_energy":
                    value = spec.normalized_rest_mass_ev_c2
                elif name == "mass_ratio":
                    value = spec.mass_in_u
                else:
                    value = getattr(self.kinematics, name)
                if name in ("momentum_ev_c", "total_energy_ev"):
                    value /= spec.energy_divisor
                if name in ("ek_per_nucleon_ev", "kinetic_energy_ev", "total_energy_ev", "rest_energy"):
                    value /= 1e6
                if name == "momentum_ev_c":
                    value /= 1e9
                widget.setText(format_number(value))
            self.kin_error.clear()
        except (ValueError, OverflowError) as exc:
            self.kinematics = None
            self.identity.setText("请检查粒子与输入参数")
            self.kin_error.setText(str(exc))
            for widget in self.results.values():
                widget.setText("—")
        self.recalculate_power()

    def update_units(self, particle):
        per_nucleon = particle.uses_nucleon_units
        suffix = "/A" if per_nucleon else ""
        prefix = "每核子" if per_nucleon else "单粒子"
        with QSignalBlocker(self.known):
            for key, label in (("kinetic_energy", f"{prefix}动能 Ek"),
                               ("total_energy", f"{prefix}总能量 E{suffix}"),
                               ("momentum", f"{prefix}动量 p{suffix}")):
                self.known.setItemText(self.known.findData(key), label)
        labels = {"ek_per_nucleon_ev": f"Ek ({particle.kinetic_energy_unit})",
                  "total_energy_ev": f"E{suffix} (MeV)",
                  "momentum_ev_c": f"p{suffix} (GeV/c)",
                  "rest_energy": f"静止质量 m₀{suffix} (MeV/c²)"}
        self.result_labels.update(labels)
        for key, label in labels.items():
            self.result_label_widgets[key].setText(label)
        unit = {"kinetic_energy": particle.kinetic_energy_unit, "total_energy": "MeV",
                "momentum": "GeV/c", "brho": "T·m", "gamma": "", "beta": ""}[self.known.currentData()]
        self.known_unit.setText(f"({unit})" if unit else "")

    def change_power_mode(self):
        mode = self.power_mode.currentData()
        self.power_inputs.setCurrentIndex({"direct": 0, "pulse": 1, "ring": 2}[mode])
        self.power_inputs.setFixedHeight(max(44, self.power_inputs.currentWidget().sizeHint().height()))
        prefix = "瞬时" if self.time_basis.currentData() == "instant" else "平均"
        self.direct_label.setText({"current": f"{prefix}电流 (mA)", "power": f"{prefix}束流功率 (kW)",
                                   "rate": f"{prefix}粒子率 (1/s)"}[self.power_known.currentData()])
        self.power_note.setText({"direct": "电流、功率与 Ek 采用同一时间口径；不需要周长。仅有平均值不能确定峰值。",
            "pulse": "重复频率为实际脉冲/引出频率；脉冲内平均值只有平顶波形才等于峰值。",
            "ring": "周长用于求回旋频率。环内储能与循环电流不能当作靶上引出功率。"}[mode])
        self.recalculate_power()

    def recalculate_power(self):
        for widget in self.power_results.values():
            widget.setText("—")
        mode = self.power_mode.currentData()
        prefix = "瞬时" if self.time_basis.currentData() == "instant" else "平均"
        labels = {"current": f"{prefix}电流 (mA)", "power": f"{prefix}束流功率 (kW)", "rate": f"{prefix}粒子率 (1/s)"}
        if mode == "ring":
            labels = {"current": "循环平均电流 (mA)", "frequency": "回旋频率 (MHz)", "period": "回旋周期 (μs)", "energy": "束流动能储量 (J)"}
        elif mode == "pulse":
            labels = {"current": "平均电流 (mA)", "power": "平均功率 (kW)", "energy": "每脉冲动能 (J)"}
            if self.use_duration.isChecked():
                labels.update(pulse_current="脉冲内平均电流 (mA)", pulse_power="脉冲内平均功率 (kW)")
        else:
            labels.pop(self.power_known.currentData())  # Do not echo the editable known quantity.
        for key in self.power_results:
            self.power_labels[key].setVisible(key in labels)
            self.power_results[key].setVisible(key in labels)
        for index, (key, text) in enumerate(labels.items()):
            self.power_labels[key].setText(text)
            self.power_result_layout.addWidget(self.power_labels[key], index // 2, index % 2 * 2)
            self.power_result_layout.addWidget(self.power_results[key], index // 2, index % 2 * 2 + 1)
        try:
            k = self.kinematics
            if k is None:
                raise ValueError("请先填写有效的粒子与运动学参数。")
            if mode == "direct":
                known = self.power_known.currentData()
                if known == "rate":
                    rate = self.direct_value.value()
                    current = rate * abs(k.particle.charge_state) * const.e
                    power = rate * k.kinetic_energy_ev * const.e
                else:
                    current = self.direct_value.value()*1e-3 if known == "current" else beam_current(k, self.direct_value.value()*1e3)
                    power, rate = beam_power(k, current), particle_rate(k, current)
                values = dict(current=current*1e3, power=power/1e3, rate=rate)
            elif mode == "pulse":
                r = pulsed_beam(k, self.pulse_count.value(), self.repetition_frequency.value(), self.pulse_duration.value()*1e-6 if self.use_duration.isChecked() else None)
                values = dict(current=r.current_a*1e3, power=r.power_w/1e3, energy=r.pulse_energy_j)
                if r.pulse_current_a is not None:
                    values.update(pulse_current=r.pulse_current_a*1e3, pulse_power=r.pulse_power_w/1e3)
            else:
                r = circulating_beam(k, self.ring_count.value(), self.circumference.value())
                values = dict(current=r.current_a*1e3, frequency=r.revolution_frequency/1e6,
                              period=None if r.revolution_period is None else r.revolution_period*1e6, energy=r.stored_energy_j)
            for key, value in values.items():
                self.power_results[key].setText(format_number(value))
            self.power_error.clear()
        except (ValueError, OverflowError) as exc:
            self.power_error.setText(str(exc))

    def show_formulas(self):
        if self.formula_dialog is None:
            self.formula_dialog = FormulaDialog("束流参数计算器 · 详细公式", BEAM_FORMULAS, self)
        self.formula_dialog.show()
        self.formula_dialog.raise_()

    def copy_results(self):
        lines = [self.identity.text()]
        if self.kinematics is not None:
            lines.extend(f"{self.result_labels[key]}: {widget.text()}" for key, widget in self.results.items())
        lines.append(self.power_mode.currentText())
        if self.power_mode.currentData() == "direct":
            lines.append(f"{self.direct_label.text()}: {self.direct_value.value()}")
        if self.power_error.text():
            lines.append(self.power_error.text())
        else:
            lines.extend(f"{self.power_labels[key].text()}: {widget.text()}" for key, widget in self.power_results.items() if not widget.isHidden())
        QApplication.clipboard().setText("\n".join(lines))
