"""GUI calculation backend for reference kinematics and beam current/power.

Ek is eV per nucleon for A > 0 (K=A*Ek), and eV per particle for A=0.
Evaluated, charge-dependent rest masses are used without the m0=A*u approximation.
Total-particle energies remain eV and momentum eV/c internally; current is A,
power W and time s.
"""
from dataclasses import dataclass
import math

from PASS.tool.particles import ParticleSpec
from PASS.utils.constants import const


def _finite(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是数值。")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} 必须是有限{'正' if positive else '非负'}数。")
    return value


@dataclass(frozen=True)
class Kinematics:
    """Complete-particle results with explicit normalized display properties.

    The historical field ek_per_nucleon_ev holds per-particle Ek when A=0.
    The *_per_u accessors still mean division by m0/u; Tools do not use them.
    """
    particle: ParticleSpec
    ek_per_nucleon_ev: float
    kinetic_energy_ev: float
    total_energy_ev: float
    momentum_ev_c: float
    gamma: float
    beta: float
    beta_gamma: float
    velocity: float
    brho: float | None

    @property
    def ek_ev(self):
        """Input Ek: eV per nucleon (A > 0), or per particle (A=0)."""
        return self.ek_per_nucleon_ev

    @property
    def normalized_total_energy_ev(self):
        return self.total_energy_ev / self.particle.energy_divisor

    @property
    def normalized_momentum_ev_c(self):
        return self.momentum_ev_c / self.particle.energy_divisor

    @property
    def ek_ev_per_u(self):
        """K/(m0/u), for explicit per-u conversions outside the Tools UI."""
        return self.kinetic_energy_ev / self.particle.mass_in_u

    @property
    def total_energy_ev_per_u(self):
        return self.total_energy_ev / self.particle.mass_in_u

    @property
    def momentum_ev_c_per_u(self):
        return self.momentum_ev_c / self.particle.mass_in_u


def solve_kinematics(particle: ParticleSpec, known: str, value: float) -> Kinematics:
    """Solve from normalized kinetic_energy, full total_energy/momentum, or β/γ/Bρ.

    Energy inputs are in eV, momentum in eV/c, and rigidity magnitude in T·m.
    Only kinetic_energy is divided by A (or one for A=0) in this numerical API.
    """
    value = _finite(value, "已知量")
    e0 = particle.rest_energy_ev
    if known == "kinetic_energy":
        kinetic = value * particle.energy_divisor
    elif known == "total_energy":
        if value < e0:
            raise ValueError("总能量不能小于静止能量。")
        kinetic = value - e0
    elif known in ("momentum", "brho"):
        if known == "brho" and particle.charge_state == 0:
            raise ValueError("中性粒子没有磁刚度，不能由 Bρ 反算动能。")
        momentum = value if known == "momentum" else value * abs(particle.charge_state) * const.c
        kinetic = momentum * (momentum / (math.hypot(momentum, e0) + e0))
    elif known == "gamma":
        if value < 1:
            raise ValueError("γ 必须大于或等于 1。")
        kinetic = (value - 1) * e0
    elif known == "beta":
        if value >= 1:
            raise ValueError("β 的范围为 0 ≤ β < 1。")
        gamma = 1 / math.sqrt((1 - value) * (1 + value))
        momentum = gamma * e0 * value
        kinetic = momentum * (momentum / (math.hypot(momentum, e0) + e0))
    else:
        raise ValueError(f"未知的运动学输入量：{known}")
    total = kinetic + e0
    momentum = math.sqrt(kinetic) * math.sqrt(kinetic + 2 * e0)
    beta = momentum / total
    gamma = 1 + kinetic / e0
    brho = momentum / (abs(particle.charge_state) * const.c) if particle.charge_state else None
    result = Kinematics(particle, kinetic / particle.energy_divisor, kinetic, total,
                        momentum, gamma, beta, momentum / e0, beta * const.c, brho)
    if not all(math.isfinite(getattr(result, name)) for name in result.__dataclass_fields__
               if name != "particle" and getattr(result, name) is not None):
        raise ValueError("输入过大，计算结果超出浮点数范围。")
    return result


def _result(value):
    if not math.isfinite(value):
        raise ValueError("计算结果超出浮点数范围。")
    return value


def beam_power(kinematics: Kinematics, current_a: float) -> float:
    _charged(kinematics)
    return _result(_finite(current_a, "电流") * (kinematics.kinetic_energy_ev / abs(kinematics.particle.charge_state)))


def beam_current(kinematics: Kinematics, power_w: float) -> float:
    _charged(kinematics)
    power = _finite(power_w, "功率")
    if kinematics.kinetic_energy_ev == 0:
        raise ValueError("Ek=0 时不能由功率唯一确定流强。" if power == 0 else "Ek=0 不能对应非零动能输运功率。")
    return _result(power * (abs(kinematics.particle.charge_state) / kinematics.kinetic_energy_ev))


def particle_rate(kinematics: Kinematics, current_a: float) -> float:
    _charged(kinematics)
    return _result(_finite(current_a, "电流") / (abs(kinematics.particle.charge_state) * const.e))


def _charged(kinematics):
    if not kinematics.particle.charge_state:
        raise ValueError("中性束流不能用电流换算功率；请选择粒子率或粒子数。")


@dataclass(frozen=True)
class CirculatingBeam:
    revolution_frequency: float
    revolution_period: float | None
    current_a: float
    stored_energy_j: float


def circulating_beam(kinematics: Kinematics, num_particles: float, circumference: float) -> CirculatingBeam:
    count = _finite(num_particles, "环内真实粒子总数")
    length = _finite(circumference, "环周长", positive=True)
    frequency = _result(kinematics.velocity / length)
    period = _result(1 / frequency) if frequency else None
    return CirculatingBeam(frequency, period,
                           _result(count * abs(kinematics.particle.charge_state) * const.e * frequency),
                           _result(count * (kinematics.kinetic_energy_ev * const.e)))


@dataclass(frozen=True)
class PulsedBeam:
    current_a: float
    power_w: float
    pulse_energy_j: float
    pulse_current_a: float | None
    pulse_power_w: float | None


def pulsed_beam(kinematics: Kinematics, num_particles: float, repetition_frequency: float,
                pulse_duration: float | None = None) -> PulsedBeam:
    count = _finite(num_particles, "每脉冲真实粒子数")
    frequency = _finite(repetition_frequency, "重复频率")
    duration = None if pulse_duration is None else _finite(pulse_duration, "脉宽", positive=True)
    if duration is not None and duration * frequency > 1 + 1e-12:
        raise ValueError("脉宽 × 重复频率不能大于 1。")
    charge = _result(count * abs(kinematics.particle.charge_state) * const.e)
    energy = _result(count * (kinematics.kinetic_energy_ev * const.e))
    return PulsedBeam(_result(charge * frequency), _result(energy * frequency), energy,
                      _result(charge / duration) if duration else None,
                      _result(energy / duration) if duration else None)
