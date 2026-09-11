"""RF bucket page, single harmonic with frozen beam and cavity parameters."""
import math
import numpy as np
from PySide6.QtWidgets import QComboBox, QFormLayout, QGroupBox
from PySide6.QtCore import QSignalBlocker

from PASS.gui.tool_beam import hint, number
from PASS.gui.tool_optics_formulas import RF_FORMULAS
from PASS.gui.tool_physics_common import PhysicsToolPage, ResultFields, integer
from PASS.gui.rf_bucket import calculate_bucket


class RFBucketPage(PhysicsToolPage):
    def __init__(self, source=None):
        super().__init__("RF bucket绘制", RF_FORMULAS, source, plot=True)
        self.result = None
        box = QGroupBox("环与 RF 参数")
        form = QFormLayout(box)
        self.voltage = self.bind(number(100))
        self.harmonic = self.bind(integer(4, 1))
        self.circumference = self.bind(number(100))
        self.phase = self.bind(number(0, -1e9))
        for label, widget in (("RF 幅值 V (kV)", self.voltage), ("RF 谐波数 h", self.harmonic),
                              ("环周长 C (m)", self.circumference), ("同步相位 φs (deg)", self.phase)):
            form.addRow(label, widget)
        self.eta_mode = QComboBox()
        self.eta_mode.addItem("由跃迁 γt 计算", "gamma_t")
        self.eta_mode.addItem("直接输入滑移因子 η", "eta")
        form.addRow("滑移因子", self.eta_mode)
        self.gamma_t = self.bind(number(6))
        self.eta = self.bind(number(-.1, -1e100))
        form.addRow("跃迁 γt", self.gamma_t)
        form.addRow("滑移因子 η", self.eta)
        self.eta_mode.currentIndexChanged.connect(lambda: self._slip_mode(form))
        self._slip_mode(form)
        self.controls.addWidget(box)
        display = QGroupBox("绘图")
        df = QFormLayout(display)
        self.x_axis = QComboBox()
        self.x_axis.addItem("RF 相位 (deg)", "phase")
        self.x_axis.addItem("z_rel (m)", "z")
        self.y_axis = QComboBox()
        self.y_axis.addItem("δ = Δp/p (%)", "delta")
        self.y_axis.addItem("ΔE/A (MeV)", "energy")
        self.orbits = self.bind(integer(5, 0, 20))
        df.addRow("横轴", self.bind(self.x_axis))
        df.addRow("纵轴", self.bind(self.y_axis))
        df.addRow("桶内轨道数", self.orbits)
        self.controls.addWidget(display)
        self.controls.addWidget(hint("φs 是束团中心的有效相位。单谐波、固定参数、小 δ 近似；详细数值见“计算结果”。"))
        self.controls.addStretch()
        self.results = ResultFields([
            ("eta", "滑移因子 η", 1), ("delta_max", "δ 半高 (%)", 100),
            ("energy_half_height_ev", "ΔE/A 半高 (MeV)", 1e-6),
            ("phase_width", "相位全宽 (deg)", 180/math.pi), ("length_width", "z 全宽 (m)", 1),
            ("time_width", "时间全宽 (μs)", 1e6), ("area_ev_s", "桶面积 (eV·s/核子)", 1),
            ("synchrotron_tune", "同步振荡 Qs", 1), ("synchrotron_frequency", "同步频率 fs (Hz)", 1),
            ("revolution_frequency", "回旋频率 (MHz)", 1e-6), ("rf_frequency", "RF 频率 (MHz)", 1e-6),
            ("energy_gain_ev", "每核子每圈增能 (MeV)", 1e-6)])
        self.result_layout.addWidget(self.results)
        self.result_layout.addStretch()
        self.recalculate()

    def _slip_mode(self, form):
        direct = self.eta_mode.currentData() == "eta"
        form.setRowVisible(self.gamma_t, not direct)
        form.setRowVisible(self.eta, direct)
        self.schedule()

    def recalculate(self):
        self.timer.stop()
        self.result = None
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                k = self.reference.current()
                per_nucleon = k.particle.uses_nucleon_units
                energy_name = "ΔE/A" if per_nucleon else "ΔE"
                with QSignalBlocker(self.y_axis):
                    self.y_axis.setItemText(self.y_axis.findData("energy"), f"{energy_name} (MeV)")
                self.results.labels["energy_half_height_ev"].setText(f"{energy_name} 半高 (MeV)")
                self.results.labels["area_ev_s"].setText("桶面积 (eV·s/核子)" if per_nucleon else "桶面积 (eV·s)")
                self.results.labels["energy_gain_ev"].setText("每核子每圈增能 (MeV)" if per_nucleon else "每圈增能 (MeV)")
                r = calculate_bucket(k, self.voltage.value()*1e3, self.harmonic.value(),
                    self.circumference.value(), self.phase.value(),
                    **({"eta": self.eta.value()} if self.eta_mode.currentData() == "eta" else {"gamma_t": self.gamma_t.value()}))
                colors = self.prepare_plot()
                for fraction in [1., *np.linspace(.12, .9, self.orbits.value())]:
                    offsets, delta = r.contour(float(fraction))
                    x = np.degrees(offsets+r.phase_s) if self.x_axis.currentData() == "phase" else -offsets*r.circumference/(2*np.pi*r.harmonic)
                    y = delta*100 if self.y_axis.currentData() == "delta" else delta*k.beta**2*k.normalized_total_energy_ev/1e6
                    if fraction == 1:
                        self.ax.fill_between(x, -y, y, color=colors["accent"], alpha=.08)
                    self.ax.plot(x, y, color=colors["accent"] if fraction == 1 else colors["muted"],
                                 lw=1.6 if fraction == 1 else .8, label="Separatrix" if fraction == 1 else None)
                    self.ax.plot(x, -y, color=colors["accent"] if fraction == 1 else colors["muted"], lw=1.6 if fraction == 1 else .8)
                center = math.degrees(r.phase_s) if self.x_axis.currentData() == "phase" else 0
                self.ax.plot(center, 0, "o", color=colors["accent"], label="Synchronous particle", ms=5)
                self.ax.set_xlabel("RF phase (deg)" if self.x_axis.currentData() == "phase" else "$z_{rel}$ (m)", color=colors["text"])
                energy_label = r"$\Delta E/A$ (MeV)" if per_nucleon else r"$\Delta E$ (MeV)"
                self.ax.set_ylabel(r"$\delta=\Delta p/p$ (%)" if self.y_axis.currentData() == "delta" else energy_label, color=colors["text"])
                self.finish_plot()
                values = dict(r.__dict__)
                for key in ("energy_half_height_ev", "energy_gain_ev", "area_ev_s"):
                    values[key] /= k.particle.energy_divisor
                self.results.set_values(values)
                self.summary.setText(f"δ 半高 {r.delta_max*100:.5g}% · Qs={r.synchrotron_tune:.6g} · fs={r.synchrotron_frequency:.6g} Hz")
                self.result = r
                self.set_valid({"inputs": {"V_kV": self.voltage.value(), "h": self.harmonic.value(), "C_m": self.circumference.value(),
                    "phi_s_deg": self.phase.value(), "eta_mode": self.eta_mode.currentData(), "eta": r.eta}, "results": self.results.snapshot()})
        except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError) as exc:
            self.set_error(exc)

    def export_rows(self):
        r = self.result
        offsets, delta = r.contour()
        z = -offsets*r.circumference/(2*np.pi*r.harmonic)
        de = delta*r.kinematics.beta**2*r.kinematics.normalized_total_energy_ev
        unit = "eV_per_nucleon" if r.kinematics.particle.uses_nucleon_units else "eV_per_particle"
        return ("phase_deg", "z_rel_m", "delta_upper", "delta_lower", f"delta_E_upper_{unit}", f"delta_E_lower_{unit}"), zip(np.degrees(offsets+r.phase_s), z, delta, -delta, de, -de)
