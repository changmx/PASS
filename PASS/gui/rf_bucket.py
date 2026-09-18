"""GUI RF bucket calculation backend using PASS's phase and slip conventions.

phi = phi_s - 2*pi*h*z_rel/C; dphi/dturn = 2*pi*h*eta*delta;
ddelta/dturn = |q|*V/(beta^2*E) * (sin(phi)-sin(phi_s)).
This is the small-delta smooth Hamiltonian, not full turn-by-turn tracking.
The existing calc_bucket module's public entry points remain independent.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from PASS.gui.beam_calculator import Kinematics
from PASS.gui.optics_calculator import finite_number


@dataclass(frozen=True)
class RFBucket:
    kinematics: Kinematics
    voltage: float
    harmonic: int
    circumference: float
    eta: float
    phase_s: float
    left: float  # phase offset from synchronous particle, rad
    right: float
    potential_scale: float
    kinetic_scale: float
    separatrix_energy: float
    delta_max: float
    energy_half_height_ev: float
    phase_width: float
    length_width: float
    time_width: float
    area_ev_s: float
    revolution_frequency: float
    rf_frequency: float
    synchrotron_tune: float
    synchrotron_frequency: float
    energy_gain_ev: float

    def potential(self, offset):
        return self.potential_scale * (-2 * math.cos(self.phase_s) * np.sin(np.asarray(offset) / 2)**2 + math.sin(self.phase_s) *
                                       (np.asarray(offset) - np.sin(offset)))

    def contour(self, fraction=1., samples=601):
        """Return phase offsets and positive delta branch at H/H_sep=fraction."""
        fraction = finite_number(fraction, "轨道能量比例", positive=True)
        if fraction > 1:
            raise ValueError("桶内轨道能量比例不能超过 1。")
        level = self.separatrix_energy * fraction
        if fraction == 1:
            left, right = self.left, self.right
        else:
            left = brentq(lambda t: self.potential(t) - level, self.left, 0., xtol=1e-13)
            right = brentq(lambda t: self.potential(t) - level, 0., self.right, xtol=1e-13)
        offsets = np.linspace(left, right, samples)
        delta = np.sqrt(np.maximum(0., (level - self.potential(offsets)) / self.kinetic_scale))
        delta[[0, -1]] = 0.
        return offsets, delta


def calculate_bucket(kinematics: Kinematics,
                     voltage: float,
                     harmonic: int,
                     circumference: float,
                     phase_s_deg=0.,
                     *,
                     eta=None,
                     gamma_t=None) -> RFBucket:
    voltage = finite_number(voltage, "RF 电压幅值", positive=True)
    circumference = finite_number(circumference, "环周长", positive=True)
    if type(harmonic) is not int or harmonic < 1:
        raise ValueError("RF 谐波数 h 必须是正整数。")
    if kinematics.beta == 0:
        raise ValueError("RF 桶计算需要 Ek > 0。")
    if not kinematics.particle.charge_state:
        raise ValueError("中性粒子不能形成 RF bucket。")
    if (eta is None) == (gamma_t is None):
        raise ValueError("请选择直接输入 η 或输入 γt，两者必须且只能提供一个。")
    if gamma_t is not None:
        gamma_t = finite_number(gamma_t, "跃迁 γt", positive=True)
        eta = 1 / gamma_t**2 - 1 / kinematics.gamma**2
    eta = finite_number(eta, "滑移因子 η", nonzero=True)
    phase_deg = finite_number(phase_s_deg, "同步相位")
    phase_s = math.radians(math.remainder(phase_deg, 360.))
    cos_s = math.cos(phase_s)
    if abs(cos_s) < 1e-9 or eta * cos_s >= 0:
        raise ValueError("无稳定 RF 桶：PASS 的正弦电压约定要求 η·cos(φs) < 0，且不能处于 90°/270° 退化相位。")
    a = 2 * math.pi * harmonic * eta
    b = abs(kinematics.particle.charge_state) * voltage / (kinematics.beta**2 * kinematics.total_energy_ev)
    potential_scale = math.copysign(b, eta)
    kinetic_scale = abs(a) / 2

    def potential(offset):
        return potential_scale * (-2 * cos_s * math.sin(offset / 2)**2 + math.sin(phase_s) * (offset - math.sin(offset)))

    saddles = [math.pi - 2 * phase_s + 2 * math.pi * n for n in range(-3, 4)]
    left_saddle = max(t for t in saddles if t < 0)
    right_saddle = min(t for t in saddles if t > 0)
    left_height, right_height = potential(left_saddle), potential(right_saddle)
    level = min(left_height, right_height)
    if level <= 0 or not math.isfinite(level):
        raise ValueError("RF 桶退化或参数超出数值范围。")
    left = left_saddle if left_height <= level * (1 + 1e-12) else brentq(lambda t: potential(t) - level, left_saddle, 0., xtol=1e-13)
    right = right_saddle if right_height <= level * (1 + 1e-12) else brentq(lambda t: potential(t) - level, 0., right_saddle, xtol=1e-13)
    delta_max = math.sqrt(level / kinetic_scale)
    if delta_max >= 1:
        raise ValueError("桶高达到 |δ|≥1，超出此小动量偏差模型的适用范围。")
    energy_scale = kinematics.beta**2 * kinematics.total_energy_ev
    f_rev = kinematics.velocity / circumference
    f_rf = harmonic * f_rev
    qs = math.sqrt(-a * b * cos_s) / (2 * math.pi)
    # Integrate normalized height so quadrature tolerance does not depend on V.
    normalized_area = quad(lambda t: math.sqrt(max(0., (level - potential(t)) / level)), left, right, epsabs=1e-10, epsrel=1e-10)[0]
    area = 2 * delta_max * normalized_area * energy_scale / (2 * math.pi * f_rf)
    width = right - left
    result = RFBucket(kinematics, voltage, harmonic, circumference, eta, phase_s, left, right, potential_scale, kinetic_scale, level, delta_max,
                      delta_max * energy_scale, width, width * circumference / (2 * math.pi * harmonic), width / (2 * math.pi * f_rf), area, f_rev,
                      f_rf, qs, qs * f_rev,
                      abs(kinematics.particle.charge_state) * voltage * math.sin(phase_s))
    for key in result.__dataclass_fields__:
        if key != "kinematics":
            finite_number(getattr(result, key), "计算结果")
    return result
