"""RMS emittance, projected beam size and covariance ellipses."""
import math
import numpy as np
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QComboBox, QFormLayout, QGroupBox, QLabel

from PASS.gui.tool_beam import hint, number
from PASS.gui.tool_optics_formulas import EMITTANCE_FORMULAS
from PASS.gui.tool_physics_common import PhysicsToolPage, ResultFields
from PASS.tool.optics_calculator import emittance_from, finite_number, twiss_from


class EmittancePage(PhysicsToolPage):
    def __init__(self, source=None):
        super().__init__("发射度与束斑换算", EMITTANCE_FORMULAS, source, plot=True, stacked=True)
        self.result = None
        box = QGroupBox("单平面 RMS 参数")
        form = QFormLayout(box)
        self.known = QComboBox()
        for label, key in (("几何 rms 发射度", "geometric"), ("归一化 rms 发射度", "normalized"), ("投影 rms 束斑 σx", "sigma")):
            self.known.addItem(label, key)
        form.addRow("已知量", self.known)
        self.value = self.bind(number(1))
        self.value_label = QLabel("ε (π·mm·mrad)")
        form.addRow(self.value_label, self.value)
        self.beta = self.bind(number(10))
        self.alpha = self.bind(number(0, -1e100))
        self.gamma = self.bind(number(.1))
        self.twiss_mode = QComboBox()
        self.twiss_mode.addItem("α、β → γ", "alpha")
        self.twiss_mode.addItem("β、γ → α", "gamma")
        self.alpha_sign = QComboBox()
        self.alpha_sign.addItem("α ≥ 0", 1)
        self.alpha_sign.addItem("α ≤ 0", -1)
        form.addRow("Twiss 已知量", self.twiss_mode)
        self.alpha_sign_label = QLabel("α 分支")
        self.dispersion = self.bind(number(0, -1e100))
        self.dispersion_prime = self.bind(number(0, -1e100))
        self.sigma_delta = self.bind(number(.1))
        self.n_sigma = self.bind(number(1))
        for label, widget in (("Twiss β (m)", self.beta), ("Twiss α", self.alpha), ("Twiss γ (1/m)", self.gamma)):
            form.addRow(label, widget)
        form.addRow(self.alpha_sign_label, self.alpha_sign)
        self.twiss_mode.currentIndexChanged.connect(self.change_twiss_mode)
        self.bind(self.alpha_sign)
        for label, widget in (("色散 D (m)", self.dispersion),
            ("色散 D′", self.dispersion_prime), ("σδ = rms Δp/p (%)", self.sigma_delta), ("椭圆倍数 n", self.n_sigma)):
            form.addRow(label, widget)
        self.known.currentIndexChanged.connect(self.change_known)
        self.controls.addWidget(box)
        self.controls.addWidget(hint("ε=1 π·mm·mrad 的计算值为 10⁻⁶ m·rad；π 表示椭圆面积约定，不额外相乘。"))
        self.results = ResultFields([
            ("geometric", "几何 ε (π·mm·mrad)", 1e6), ("normalized", "归一化 εn (π·mm·mrad)", 1e6),
            ("projected", "含色散 ε (π·mm·mrad)", 1e6),
            ("sigma_betatron", "betatron σx (mm)", 1e3), ("sigma_x", "投影 σx (mm)", 1e3),
            ("sigma_xp", "投影 σx′ (mrad)", 1e3), ("covariance", "Cov(x,x′) (mm·mrad)", 1e6),
            ("correlation", "相关系数 r", 1)])
        self.result_layout.addWidget(self.results)
        self.result_layout.setContentsMargins(0, 0, 0, 0)
        self.result_layout.addWidget(hint("投影包含不同动量粒子的色散位移；D=D′=0 时与 betatron 椭圆相同。Cov 衡量 x 与 x′ 的共同变化，r=Cov/(σxσx′) 是归一化相关系数。"))
        self.result_layout.addWidget(hint("二维高斯 n=1 椭圆包含约 39.35%；几何 ε 为 RMS 发射度。"))
        self.controls.addWidget(self.result_page)
        self.controls.addStretch()
        self.change_twiss_mode()
        self.recalculate()

    def change_twiss_mode(self):
        inverse = self.twiss_mode.currentData() == "gamma"
        self.alpha.setReadOnly(inverse)
        self.gamma.setReadOnly(not inverse)
        self.alpha_sign.setVisible(inverse)
        self.alpha_sign_label.setVisible(inverse)
        with QSignalBlocker(self.alpha_sign):
            self.alpha_sign.setCurrentIndex(1 if self.alpha.value() < 0 else 0)
        self.recalculate()

    def change_known(self):
        key = self.known.currentData()
        if self.result is not None:
            value = getattr(self.result, "sigma_x" if key == "sigma" else key)
            with QSignalBlocker(self.value):
                self.value.setValue(value * (1e3 if key == "sigma" else 1e6))
        self.value_label.setText({"geometric": "ε (π·mm·mrad)", "normalized": "εn (π·mm·mrad)", "sigma": "σx (mm)"}[key])
        self.recalculate()

    def recalculate(self):
        self.timer.stop()
        self.result = None
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                n = finite_number(self.n_sigma.value(), "椭圆倍数", positive=True)
                inverse = self.twiss_mode.currentData() == "gamma"
                alpha, gamma = twiss_from(self.beta.value(), **({"gamma": self.gamma.value(), "alpha_sign": self.alpha_sign.currentData()}
                                         if inverse else {"alpha": self.alpha.value()}))
                dependent = self.alpha if inverse else self.gamma
                with QSignalBlocker(dependent):
                    dependent.setValue(alpha if inverse else gamma)
                r = emittance_from(self.reference.current(), self.known.currentData(),
                    self.value.value() * (1e-3 if self.known.currentData() == "sigma" else 1e-6), self.beta.value(),
                    alpha, self.dispersion.value(), self.dispersion_prime.value(), self.sigma_delta.value()*.01)
                colors = self.prepare_plot()
                for projected, label, color in ((True, "Projected (with dispersion)", "#c98a42"), (False, "Betatron", colors["accent"])):
                    xy = r.ellipse(n, projected=projected)*1e3
                    self.ax.plot(*xy, color=color, lw=1.6, label=label)
                self.ax.set_xlabel("x (mm)", color=colors["text"])
                self.ax.set_ylabel("x′ (mrad)", color=colors["text"])
                if r.sigma_x == 0 and r.sigma_xp == 0:
                    self.ax.plot(0, 0, "o", color=colors["accent"])
                self.finish_plot()
                self.results.set_values(r.__dict__)
                self.summary.setText(f"ε={r.geometric*1e6:.9g} π·mm·mrad · σx={r.sigma_x*1e3:.9g} mm · n={n:g}")
                self.result = r
                self.set_valid({"inputs": {"known": self.known.currentData(), "value": self.value.value(),
                    "value_unit": "mm" if self.known.currentData() == "sigma" else "pi*mm*mrad (area convention, no extra pi)",
                    "beta_m": self.beta.value(), "alpha": alpha, "gamma_per_m": gamma,
                    "D_m": self.dispersion.value(), "Dprime": self.dispersion_prime.value(), "sigma_delta_percent": self.sigma_delta.value(), "n": n},
                    "results": self.results.snapshot()})
        except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError) as exc:
            self.set_error(exc)

    def export_rows(self):
        intrinsic = self.result.ellipse(self.n_sigma.value())
        projected = self.result.ellipse(self.n_sigma.value(), projected=True)
        return ("betatron_x_m", "betatron_xprime_rad", "projected_x_m", "projected_xprime_rad"), zip(*intrinsic, *projected)
