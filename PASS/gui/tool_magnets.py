"""Signed rigidity, dipole and quadrupole conversion controls."""
import math
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QTabWidget, QVBoxLayout, QWidget

from PASS.gui.tool_beam import format_number, hint, number, output
from PASS.gui.tool_optics_formulas import MAGNET_FORMULAS
from PASS.gui.tool_physics_common import PhysicsToolPage, ResultFields
from PASS.tool.optics_calculator import dipole_from, quadrupole_from, signed_rigidity, finite_number, multipole_from, solenoid_from


class MagnetSection(QWidget):
    def __init__(self, kind, owner):
        super().__init__()
        self.kind, self.owner, self.result = kind, owner, None
        layout = QVBoxLayout(self)
        box = QGroupBox("磁铁输入")
        form = QFormLayout(box)
        self.length = owner.bind(number(1 if kind == "dipole" else .5))
        form.addRow("有效磁长 L (m)", self.length)
        self.known = QComboBox()
        choices = {
            "dipole": (("磁场 B (T)", "field"), ("弯转半径 ρ (m)", "radius"), ("弯转角 θ (deg)", "angle"),
                       ("积分场 ∫B dl (T·m)", "integrated_field")),
            "quadrupole": (("梯度 G (T/m)", "gradient"), ("归一化 k1 (1/m²)", "k1"),
                           ("积分强度 K1L (1/m)", "k1l"), ("薄透镜 fx (m)", "focal_x"), ("极面场 Bp (Gauss)", "pole_field")),
            "sextupole": (("二阶梯度 G2 (T/m²)", "derivative"), ("归一化 k2 (1/m³)", "kn"),
                          ("积分强度 K2L (1/m²)", "knl"), ("积分梯度 (T/m)", "integrated_derivative"), ("极面场 Bp (Gauss)", "pole_field")),
            "octupole": (("三阶梯度 G3 (T/m³)", "derivative"), ("归一化 k3 (1/m⁴)", "kn"),
                         ("积分强度 K3L (1/m³)", "knl"), ("积分梯度 (T/m²)", "integrated_derivative"), ("极面场 Bp (Gauss)", "pole_field")),
            "solenoid": (("轴向场 Bz (Gauss)", "field"), ("归一化 Ks (1/m)", "ks"),
                         ("Larmor 参数 θ (deg)", "angle"), ("积分场 ∫Bz dl (T·m)", "integrated_field")),
        }[kind]
        for label, key in choices:
            self.known.addItem(label, key)
        default = {"dipole": "field", "quadrupole": "k1", "sextupole": "kn", "octupole": "kn", "solenoid": "ks"}[kind]
        self.known.setCurrentIndex(self.known.findData(default))
        form.addRow("已知量", self.known)
        self.value = owner.bind(number(1 if kind == "dipole" else 5, -1e100))
        self.value_label = QLabel(self.known.currentText())
        form.addRow(self.value_label, self.value)
        self.known.currentIndexChanged.connect(self.change_known)
        layout.addWidget(box)
        fields = {
            "dipole": [("field", "磁场 B (T)", 1), ("radius", "弯转半径 ρ (m)", 1),
                ("curvature", "曲率 k0 (1/m)", 1), ("angle", "K0L = θ (rad)", 1),
                ("angle_deg", "弯转角 θ (deg)", 1), ("integrated_field", "积分场 ∫B dl (T·m)", 1)],
            "quadrupole": [("gradient", "梯度 G (T/m)", 1), ("k1", "归一化 k1 (1/m²)", 1),
                ("k1l", "积分强度 K1L (1/m)", 1), ("integrated_gradient", "积分梯度 ∫G dl (T)", 1),
                ("focal_x", "薄透镜 fx (m)", 1), ("focal_y", "薄透镜 fy (m)", 1), ("pole_field", "极面场 Bp (Gauss)", 1e4)],
            "solenoid": [("field", "轴向场 Bz (Gauss)", 1e4), ("ks", "PASS Ks (1/m)", 1),
                ("kappa", "κ = Ks/2 (1/m)", 1), ("angle", "Larmor 参数 θ (rad)", 1),
                ("integrated_field", "积分场 (T·m)", 1), ("focusing", "聚焦系数 κ² (1/m²)", 1),
                ("focal", "弱透镜焦距 f (m)", 1)]}.get(kind)
        if kind in ("sextupole", "octupole"):
            fields = [(key, label, 1e4 if key == "pole_field" else 1) for label, key in choices]
        if kind in ("quadrupole", "sextupole", "octupole"):
            self.radius = owner.bind(number(30))
            form.addRow("磁极半径 r (mm)", self.radius)
        self.results = ResultFields(fields, columns=2)
        layout.addWidget(self.results)
        notes = {"dipole": "L 为参考轨道的有效弧长；零磁场时半径无有限值。",
            "quadrupole": "理想极面场 Bp=G r；1 T=10000 Gauss，场值保留极性。fx、fy 为薄透镜焦距；K1L > 0 时 x 聚焦、y 散焦。",
            "sextupole": "G2=∂²By/∂x²；理想极面场 Bp=G2 r²/2，保留极性。K2L 的定义与 PASS 一致，不省略 2!。",
            "octupole": "G3=∂³By/∂x³；理想极面场 Bp=G3 r³/6，保留极性。K3L 的定义与 PASS 一致，不省略 3!。",
            "solenoid": "Ks=Bz/(Bρ)，κ=Ks/2，θ=κL；f≈1/(κ²L) 仅适用于弱透镜。显示的是轴向场，铁芯极面场还需要磁路和几何信息，不能由孔径半径与 Ks 推出。"}
        layout.addWidget(hint(notes[kind]))
        layout.addStretch()

    def change_known(self):
        key = self.known.currentData()
        if self.result is not None:
            value = getattr(self.result, key)
            if value is not None:
                with QSignalBlocker(self.value):
                    self.value.setValue(value / self.input_scale())
        self.value_label.setText(self.known.currentText())
        self.owner.recalculate()

    def input_scale(self):
        key = self.known.currentData()
        if key == "angle":
            return math.pi / 180
        return 1e-4 if key == "pole_field" or (self.kind == "solenoid" and key == "field") else 1

    def calculate(self, brho):
        self.result = None
        self.results.clear()
        fn = {"dipole": dipole_from, "quadrupole": quadrupole_from, "solenoid": solenoid_from}.get(self.kind)
        value = self.value.value() * self.input_scale()
        if fn is None:
            result = multipole_from(2 if self.kind == "sextupole" else 3, brho, self.length.value(),
                                    self.known.currentData(), value, self.radius.value()*1e-3)
        else:
            result = fn(brho, self.length.value(), self.known.currentData(), value,
                        **({"radius": self.radius.value()*1e-3} if self.kind == "quadrupole" else {}))
        data = dict(result.__dict__)
        if self.kind == "dipole":
            data["angle_deg"] = math.degrees(result.angle)
        self.results.set_values(data)
        self.result = result
        return {"magnet": self.kind, "inputs": {"L_m": self.length.value(), "known": self.known.currentText(),
                 "value": self.value.value(), "signed_brho_Tm": brho,
                 "reference_radius_mm": self.radius.value() if hasattr(self, "radius") else None}, "results": self.results.snapshot()}


class MagnetPage(PhysicsToolPage):
    def __init__(self, source=None):
        super().__init__("磁铁参数换算", MAGNET_FORMULAS, source)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        rigidity = QGroupBox("磁刚度")
        rl = QHBoxLayout(rigidity)
        self.rigidity_mode = QComboBox()
        self.rigidity_mode.addItem("由参考粒子和 Ek 计算", "particle")
        self.rigidity_mode.addItem("直接输入带符号 Bρ", "direct")
        rl.addWidget(self.rigidity_mode)
        rl.addWidget(QLabel("Bρ (T·m)"))
        self.direct_rigidity = self.bind(number(3, -1e100))
        rl.addWidget(self.direct_rigidity)
        self.rigidity_result = output()
        rl.addWidget(self.rigidity_result)
        self.rigidity_mode.currentIndexChanged.connect(self.recalculate)
        layout.addWidget(rigidity)
        layout.addWidget(hint("这里使用带符号 Bρ=p/(qe)；负电荷对应负 Bρ。场强、梯度和转换出的 K 值均保留符号。"))
        self.tabs = QTabWidget()
        self.sections = [MagnetSection(kind, self) for kind in ("dipole", "quadrupole", "sextupole", "octupole", "solenoid")]
        for section, title in zip(self.sections, ("二极铁", "四极铁", "六极铁", "八极铁", "螺线管")):
            self.tabs.addTab(section, title)
        self.tabs.currentChanged.connect(self.recalculate)
        layout.addWidget(self.tabs, 1)
        self.layout.insertWidget(self.layout.count()-1, content, 1)
        self.recalculate()

    def recalculate(self):
        self.timer.stop()
        direct = self.rigidity_mode.currentData() == "direct"
        self.reference.setEnabled(not direct)
        self.direct_rigidity.setVisible(direct)
        self.rigidity_result.setVisible(not direct)
        section = self.sections[self.tabs.currentIndex()]
        try:
            brho = self.direct_rigidity.value() if direct else signed_rigidity(self.reference.current())
            finite_number(brho, "带符号磁刚度", nonzero=True)
            self.rigidity_result.setText(format_number(brho))
            self.set_valid(section.calculate(brho))
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            section.result = None
            section.results.clear()
            self.rigidity_result.setText("—")
            self.set_error(exc)
