"""Independent RMS parameter pages overlaid on a shared phase-space plot."""
import math

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from PASS.gui.appearance import THEMES
from PASS.gui.tool_beam import hint, number
from PASS.gui.tool_optics_formulas import EMITTANCE_FORMULAS
from PASS.gui.tool_physics_common import PhysicsToolPage, ResultFields
from PASS.gui.optics_calculator import emittance_from, emittance_from_rms, finite_number, twiss_from


CALCULATION_ERRORS = (ValueError, OverflowError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError)
PALETTE = ("#5399df", "#e18b45", "#4eaf87", "#bd7ad5", "#d96578", "#57aeb6", "#aaa142")

class EmittanceInput(QWidget):
    """One independently validated set of centered statistics and plot settings."""
    numeric_fields = ("value", "beta", "alpha", "gamma", "dispersion", "dispersion_prime",
                      "sigma_delta", "n_sigma", "rms_x", "rms_xp", "correlation", "covariance", "center_x", "center_xp")
    combo_fields = ("known", "twiss_mode", "alpha_sign", "correlation_mode")
    check_fields = ("draw_phase", "draw_projected")
    text_fields = ("legend_name", "projected_name")

    def __init__(self, owner, page_id):
        super().__init__()
        self.owner, self.page_id = owner, page_id
        self.page_name = f"第{page_id}页"
        self.result = self.payload = None
        self.curves = None
        self.color = (PALETTE[page_id-1] if page_id <= len(PALETTE) else
                      QColor.fromHsv(int(page_id * 137.508) % 360, 150, 195).name())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        display = QGroupBox("绘图设置")
        form = QFormLayout(display)
        self.draw_phase = QCheckBox("绘制相空间")
        self.draw_phase.setChecked(True)
        self.draw_projected = QCheckBox("绘制投影椭圆（含色散）")
        form.addRow(self.draw_phase)
        form.addRow(self.draw_projected)
        self.legend_name = QLineEdit(f"{self.page_name} Betatron")
        self.projected_name = QLineEdit(f"{self.page_name} 投影")
        for widget in (self.legend_name, self.projected_name):
            widget.setPlaceholderText("留空则不显示该曲线图例")
        form.addRow("相空间图例名称", self.legend_name)
        form.addRow("投影图例名称", self.projected_name)
        self.color_button = QPushButton()
        self.color_button.clicked.connect(self.choose_color)
        form.addRow("曲线颜色", self.color_button)
        self.update_color_button()
        self.center_x, self.center_xp = number(0, -1e100), number(0, -1e100)
        self.n_sigma = number(1)
        form.addRow("中心 x₀ (mm)", self.center_x)
        form.addRow("中心 x′₀ (mrad)", self.center_xp)
        form.addRow("椭圆倍数 n", self.n_sigma)
        layout.addWidget(display)

        box = QGroupBox("单平面 RMS 参数")
        self.form = form = QFormLayout(box)
        self.known = QComboBox()
        for label, key in (("几何 rms 发射度", "geometric"), ("归一化 rms 发射度", "normalized"),
                           ("投影 rms 束斑 σx", "sigma"), ("投影 RMS 与相关性 → ε、Twiss", "rms")):
            self.known.addItem(label, key)
        form.addRow("已知量", self.known)
        self.value, self.value_label = number(1), QLabel("ε (π·mm·mrad)")
        form.addRow(self.value_label, self.value)
        self.rms_x, self.rms_xp = number(math.sqrt(10)), number(math.sqrt(.1))
        self.correlation_mode = QComboBox()
        self.correlation_mode.addItem("相关系数 r", "correlation")
        self.correlation_mode.addItem("协方差 Cov(x,x′)", "covariance")
        self.correlation, self.covariance = number(0, -1e100), number(0, -1e100)
        # Keep out-of-range r editable so invalid input receives an explicit error.
        for label, widget in (("投影 σx (mm)", self.rms_x), ("投影 σx′ (mrad)", self.rms_xp),
                              ("相关性已知量", self.correlation_mode), ("相关系数 r", self.correlation),
                              ("Cov (mm·mrad)", self.covariance)):
            form.addRow(label, widget)
        self.beta, self.alpha, self.gamma = number(10), number(0, -1e100), number(.1)
        self.twiss_mode = QComboBox()
        self.twiss_mode.addItem("α、β → γ", "alpha")
        self.twiss_mode.addItem("β、γ → α", "gamma")
        self.alpha_sign = QComboBox()
        self.alpha_sign.addItem("α ≥ 0", 1)
        self.alpha_sign.addItem("α ≤ 0", -1)
        for label, widget in (("Twiss 已知量", self.twiss_mode), ("Twiss β (m)", self.beta),
                              ("Twiss α", self.alpha), ("Twiss γ (1/m)", self.gamma), ("α 分支", self.alpha_sign)):
            form.addRow(label, widget)
        self.dispersion, self.dispersion_prime = number(0, -1e100), number(0, -1e100)
        self.sigma_delta = number(.1)
        for label, widget in (("色散 D (m)", self.dispersion), ("色散 D′", self.dispersion_prime),
                              ("σδ = rms Δp/p (%)", self.sigma_delta)):
            form.addRow(label, widget)
        layout.addWidget(box)
        layout.addWidget(hint("RMS 输入均为含色散的中心统计量；反算先扣除色散贡献。质心偏移仅平移曲线。"))
        self.error = hint()
        layout.addWidget(self.error)
        self.results = ResultFields([
            ("geometric", "Betatron ε (π·mm·mrad)", 1e6), ("normalized", "归一化 εn (π·mm·mrad)", 1e6),
            ("projected", "投影 ε (π·mm·mrad)", 1e6),
            ("sigma_betatron", "Betatron σx (mm)", 1e3), ("sigma_betatron_xp", "Betatron σx′ (mrad)", 1e3),
            ("sigma_x", "投影 σx (mm)", 1e3), ("sigma_xp", "投影 σx′ (mrad)", 1e3),
            ("covariance", "投影 Cov (mm·mrad)", 1e6), ("correlation", "投影相关系数 r", 1),
            ("beta", "Betatron Twiss β (m)", 1), ("alpha", "Betatron Twiss α", 1),
            ("twiss_gamma", "Betatron Twiss γ (1/m)", 1)])
        layout.addWidget(self.results)
        layout.addWidget(hint("ε=1 π·mm·mrad 按 10⁻⁶ m·rad 计算，不额外乘 π。二维高斯 n=1 椭圆包含约 39.35%。零发射度反算的 Twiss 未定义。"))
        layout.addStretch()
        self._known_key = "geometric"
        self._correlation_key = "correlation"
        self.update_modes()
        for name in self.numeric_fields:
            getattr(self, name).valueChanged.connect(owner.schedule)
        for name in self.check_fields:
            getattr(self, name).toggled.connect(owner.schedule)
        for name in self.text_fields:
            getattr(self, name).textChanged.connect(owner.schedule)
        self.known.currentIndexChanged.connect(self.change_known)
        self.correlation_mode.currentIndexChanged.connect(self.change_correlation)
        self.twiss_mode.currentIndexChanged.connect(self.change_twiss_mode)
        self.alpha_sign.currentIndexChanged.connect(owner.schedule)

    def update_color_button(self):
        self.color_button.setText(self.color)
        color = QColor(self.color)
        foreground = "#000000" if color.lightness() > 150 else "#ffffff"
        self.color_button.setStyleSheet(f"background-color: {self.color}; color: {foreground};")

    def choose_color(self):
        color = QColorDialog.getColor(QColor(self.color), self, "选择曲线颜色")
        if color.isValid():
            self.color = color.name()
            self.update_color_button()
            self.owner.schedule()

    def update_modes(self):
        key = self.known.currentData()
        rms = key == "rms"
        self.form.setRowVisible(self.value, not rms)
        for widget in (self.rms_x, self.rms_xp, self.correlation_mode):
            self.form.setRowVisible(widget, rms)
        self.form.setRowVisible(self.correlation, rms and self.correlation_mode.currentData() == "correlation")
        self.form.setRowVisible(self.covariance, rms and self.correlation_mode.currentData() == "covariance")
        for widget in (self.beta, self.alpha, self.gamma, self.twiss_mode):
            self.form.setRowVisible(widget, not rms)
        inverse = self.twiss_mode.currentData() == "gamma"
        self.form.setRowVisible(self.alpha_sign, not rms and inverse)
        self.alpha.setReadOnly(inverse)
        self.gamma.setReadOnly(not inverse)
        if not rms:
            self.value_label.setText({"geometric": "ε (π·mm·mrad)", "normalized": "εn (π·mm·mrad)", "sigma": "σx (mm)"}[key])

    def set_number(self, name, value):
        with QSignalBlocker(getattr(self, name)):
            getattr(self, name).setValue(value)

    def change_known(self):
        # Recompute the old mode first, including edits pending in the debounce timer.
        try:
            self.calculate(self.owner.reference.current(), known=self._known_key)
        except CALCULATION_ERRORS:
            self.result = None
        r = self.result
        key = self.known.currentData()
        if r is not None:
            if key == "rms":
                self.set_number("rms_x", r.sigma_x * 1e3)
                self.set_number("rms_xp", r.sigma_xp * 1e3)
                self.set_number("covariance", r.covariance * 1e6)
                self.set_number("correlation", r.correlation or 0.)
                if r.correlation is None:
                    with QSignalBlocker(self.correlation_mode):
                        self.correlation_mode.setCurrentIndex(1)
                    self._correlation_key = "covariance"
            else:
                self.set_number("value", getattr(r, "sigma_x" if key == "sigma" else key) * (1e3 if key == "sigma" else 1e6))
                if r.beta is not None:
                    for name, value in (("beta", r.beta), ("alpha", r.alpha), ("gamma", r.twiss_gamma)):
                        self.set_number(name, value)
                    with QSignalBlocker(self.alpha_sign):
                        self.alpha_sign.setCurrentIndex(1 if r.alpha < 0 else 0)
        self._known_key = key
        self.update_modes()
        self.owner.recalculate()

    def change_correlation(self):
        product = self.rms_x.value() * self.rms_xp.value()
        if self._correlation_key == "correlation":
            self.set_number("covariance", self.correlation.value() * product)
        elif product:
            self.set_number("correlation", self.covariance.value() / product)
        self._correlation_key = self.correlation_mode.currentData()
        self.update_modes()
        self.owner.recalculate()

    def change_twiss_mode(self):
        with QSignalBlocker(self.alpha_sign):
            self.alpha_sign.setCurrentIndex(1 if self.alpha.value() < 0 else 0)
        self.update_modes()
        self.owner.recalculate()

    def settings(self):
        values = {name: getattr(self, name).value() for name in self.numeric_fields}
        values.update({name: getattr(self, name).currentData() for name in self.combo_fields})
        values.update({name: getattr(self, name).isChecked() for name in self.check_fields})
        values.update({name: getattr(self, name).text() for name in self.text_fields})
        values["color"] = self.color
        return values

    def restore_copy(self, settings):
        for name in self.numeric_fields + self.combo_fields + self.check_fields + self.text_fields:
            widget = getattr(self, name)
            value = settings[name]
            with QSignalBlocker(widget):
                if name in self.numeric_fields:
                    widget.setValue(value)
                elif name in self.combo_fields:
                    widget.setCurrentIndex(widget.findData(value))
                elif name in self.check_fields:
                    widget.setChecked(value)
                else:
                    widget.setText(value + " 副本" if value.strip() else value)
        self._known_key = self.known.currentData()
        self._correlation_key = self.correlation_mode.currentData()
        self.update_modes()

    def calculate(self, reference, *, known=None):
        self.result = self.payload = self.curves = None
        n = finite_number(self.n_sigma.value(), "椭圆倍数", positive=True)
        key = known or self.known.currentData()
        d, dp, spread = self.dispersion.value(), self.dispersion_prime.value(), self.sigma_delta.value() * .01
        if key == "rms":
            mode = self.correlation_mode.currentData()
            r = emittance_from_rms(reference, self.rms_x.value()*1e-3, self.rms_xp.value()*1e-3,
                **{mode: self.correlation.value() if mode == "correlation" else self.covariance.value()*1e-6},
                dispersion=d, dispersion_prime=dp, sigma_delta=spread)
        else:
            inverse = self.twiss_mode.currentData() == "gamma"
            alpha, gamma = twiss_from(self.beta.value(), **({"gamma": self.gamma.value(), "alpha_sign": self.alpha_sign.currentData()}
                                     if inverse else {"alpha": self.alpha.value()}))
            self.set_number("alpha" if inverse else "gamma", alpha if inverse else gamma)
            r = emittance_from(reference, key, self.value.value()*(1e-3 if key == "sigma" else 1e-6),
                              self.beta.value(), alpha, d, dp, spread)
        center = np.array([[finite_number(self.center_x.value(), "中心 x₀")],
                           [finite_number(self.center_xp.value(), "中心 x′₀")]]) * 1e-3
        curves = tuple(r.ellipse(n, projected=projected) + center for projected in (False, True))
        if not all(np.isfinite(curve).all() for curve in curves):
            raise ValueError("曲线超出有限数值范围。")
        values = dict(r.__dict__)
        values["sigma_betatron_xp"] = (math.sqrt(r.betatron_covariance[1][1]) if r.betatron_covariance is not None
                                        else math.sqrt(r.geometric * r.twiss_gamma))
        self.results.set_values(values)
        for field in ("beta", "alpha", "twiss_gamma", "correlation"):
            if values[field] is None:
                self.results.outputs[field].setText("未定义")
        self.error.clear()
        self.result, self.curves = r, curves
        self.payload = {"page_id": self.page_id, "page_name": self.page_name, "status": "valid",
                        "inputs": self.settings(), "results": self.results.snapshot()}

    def invalidate(self, message):
        self.result = self.curves = self.payload = None
        self.results.clear()
        self.error.setText(str(message))


class EmittancePage(PhysicsToolPage):
    def __init__(self, source=None):
        super().__init__("相空间绘制及发射度计算", EMITTANCE_FORMULAS, source, plot=True, stacked=True)
        self.sections = []
        self.next_page_id = 1
        old = self.tabs.widget(0)
        self.tabs.removeTab(0)
        old.deleteLater()
        self.tabs.tabBar().show()
        self.tabs.setMaximumWidth(470)
        self.tabs.setMinimumWidth(390)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.remove_page)
        self.add_button = QPushButton("＋")
        self.add_button.setToolTip("新增参数页")
        self.add_button.clicked.connect(lambda: self.add_page())
        self.copy_page_button = QPushButton("复制当前页")
        self.copy_page_button.clicked.connect(self.copy_current_page)
        corner = QWidget()
        row = QHBoxLayout(corner)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.add_button)
        row.addWidget(self.copy_page_button)
        self.tabs.setCornerWidget(corner, Qt.TopRightCorner)
        self.show_legend = QCheckBox("显示图例")
        self.show_legend.setChecked(True)
        self.show_legend.toggled.connect(self.schedule)
        self.toolbar.addWidget(self.show_legend)
        self.add_page()

    def add_page(self, settings=None):
        section = EmittanceInput(self, self.next_page_id)
        self.next_page_id += 1
        if settings is not None:
            section.restore_copy(settings)
        self.sections.append(section)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(section)
        self.tabs.addTab(scroll, section.page_name)
        self.tabs.setCurrentIndex(len(self.sections)-1)
        self.tabs.setTabsClosable(len(self.sections) > 1)
        self.recalculate()
        return section

    def copy_current_page(self):
        return self.add_page(self.sections[self.tabs.currentIndex()].settings())

    def remove_page(self, index):
        if len(self.sections) <= 1 or not 0 <= index < len(self.sections):
            return
        scroll = self.tabs.widget(index)
        self.tabs.removeTab(index)
        self.sections.pop(index)
        scroll.deleteLater()
        self.tabs.setTabsClosable(len(self.sections) > 1)
        self.recalculate()

    def recalculate(self):
        self.timer.stop()
        colors = self.prepare_plot()
        handles, labels, payloads, errors = [], [], [], []
        valid_count = 0
        try:
            reference = self.reference.current()
            reference_error = None
        except CALCULATION_ERRORS as exc:
            reference, reference_error = None, str(exc)
        for index, section in enumerate(self.sections):
            try:
                if reference_error:
                    raise ValueError(reference_error)
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    section.calculate(reference)
                valid_count += 1
                payloads.append(section.payload)
                self.tabs.setTabText(index, section.page_name)
                self.tabs.setTabToolTip(index, section.page_name)
                if section.draw_phase.isChecked():
                    for projected, curve in enumerate(section.curves):
                        if projected and not section.draw_projected.isChecked():
                            continue
                        label = (section.projected_name if projected else section.legend_name).text().strip()
                        xy = curve * 1e3
                        point = np.all(xy == xy[:, :1])
                        line, = self.ax.plot(*xy, color=section.color, lw=1.6,
                            linestyle="--" if projected else "-", marker="o" if point else None,
                            markevery=[0] if point else None)
                        if label:
                            handles.append(line)
                            labels.append(label)
            except CALCULATION_ERRORS as exc:
                section.invalidate(exc)
                self.tabs.setTabText(index, section.page_name + " ⚠")
                self.tabs.setTabToolTip(index, str(exc))
                errors.append(f"{section.page_name}：{exc}")
                payloads.append({"page_id": section.page_id, "page_name": section.page_name,
                                 "status": "invalid", "inputs": section.settings(), "error": str(exc)})
        self.ax.set_xlabel("x (mm)", color=colors["text"])
        self.ax.set_ylabel("x′ (mrad)", color=colors["text"])
        if self.show_legend.isChecked() and handles:
            # Assign text explicitly: names starting with '_' are user labels too.
            legend = self.ax.legend(handles, [f"curve{i}" for i in range(len(handles))], loc="upper right", fontsize=8,
                facecolor=THEMES[self.theme]["panel"], edgecolor=colors["line"], labelcolor=colors["text"])
            for text, label in zip(legend.get_texts(), labels):
                text.set_text(label)
                text.set_fontfamily(["Microsoft YaHei", "DejaVu Sans"])
        self.toolbar.update()
        self.canvas.draw_idle()
        self.summary.setText(f"{valid_count}/{len(self.sections)} 页计算有效 · 已绘制 {len(self.ax.lines)} 条曲线 · 各页独立计算，不合并束流")
        if valid_count:
            self.set_valid({"show_legend": self.show_legend.isChecked(), "pages": payloads,
                "input_units": {"value": "mm for sigma; pi*mm*mrad otherwise (no extra pi)",
                    "rms_x": "mm", "rms_xp": "mrad", "covariance": "mm*mrad", "correlation": "1",
                    "beta": "m", "alpha": "1", "gamma": "1/m", "dispersion": "m", "dispersion_prime": "1",
                    "sigma_delta": "%", "center_x": "mm", "center_xp": "mrad", "n_sigma": "1"}})
        else:
            self.copy_payload = None
            for button in (self.copy_button, self.data_button, self.export_button):
                button.setEnabled(False)
        self.error.setText("\n".join(errors))

    def export_rows(self):
        if self.timer.isActive():
            self.recalculate()
        columns = ("page_id", "page_name", "betatron_label", "projected_label", "draw_betatron", "draw_projected",
                   "betatron_x_m", "betatron_xprime_rad", "projected_x_m", "projected_xprime_rad")
        rows = []
        for section in self.sections:
            if section.result is None:
                continue
            settings = section.payload["inputs"]
            metadata = (section.page_id, section.page_name, settings["legend_name"], settings["projected_name"],
                        settings["draw_phase"], settings["draw_phase"] and settings["draw_projected"])
            rows.extend((*metadata, *point) for point in zip(*section.curves[0], *section.curves[1]))
        return columns, rows
