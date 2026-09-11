"""GUI calculation backend for RMS optics and signed magnets; SI units throughout.

The magnet coefficients follow PASS's normalized inputs: theta = K0L and
the thin quadrupole kick is dx' = -K1L*x. Physical field conversion uses
signed p/(q e); the beam calculator's ``brho`` is its magnitude.
"""
from dataclasses import dataclass
import math

import numpy as np

from PASS.gui.beam_calculator import Kinematics


def finite_number(value, name, *, minimum=None, positive=False, nonzero=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是数值。")
    value = float(value)
    if (not math.isfinite(value) or (minimum is not None and value < minimum)
            or (positive and value <= 0) or (nonzero and value == 0)):
        condition = "有限正数" if positive else "有限非零数" if nonzero else f"不小于 {minimum} 的有限数" if minimum is not None else "有限数"
        raise ValueError(f"{name} 必须是{condition}。")
    return value


def twiss_from(beta, *, alpha=None, gamma=None, alpha_sign=1):
    """Solve beta*gamma-alpha^2=1; gamma does not fix the sign of alpha."""
    beta = finite_number(beta, "Twiss β", positive=True)
    if (alpha is None) == (gamma is None):
        raise ValueError("Twiss 参数需要 α 或 γ 中的一个已知量。")
    if alpha is not None:
        alpha = finite_number(alpha, "Twiss α")
        gamma = (1 + alpha**2) / beta
    else:
        gamma = finite_number(gamma, "Twiss γ", positive=True)
        if alpha_sign not in (-1, 1):
            raise ValueError("α 的分支符号必须为 +1 或 -1。")
        square = beta*gamma - 1
        if square < -1e-12:
            raise ValueError("Twiss βγ 必须不小于 1，才能得到实数 α。")
        alpha = alpha_sign * math.sqrt(max(0., square))
    return finite_number(alpha, "Twiss α"), finite_number(gamma, "Twiss γ")


@dataclass(frozen=True)
class EmittanceResult:
    geometric: float
    normalized: float
    projected: float
    twiss_gamma: float
    sigma_betatron: float
    sigma_x: float
    sigma_xp: float
    covariance: float
    correlation: float | None
    beta: float
    alpha: float

    def ellipse(self, n_sigma=1., *, projected=False, samples=361):
        """Covariance ellipse in (m, rad), not a 1D Gaussian confidence band."""
        n = finite_number(n_sigma, "椭圆倍数", positive=True)
        phase = np.linspace(0, 2 * np.pi, samples)
        if projected:
            covariance = np.array([[self.sigma_x**2, self.covariance],
                                   [self.covariance, self.sigma_xp**2]])
            values, vectors = np.linalg.eigh(covariance)
            return n * (vectors * np.sqrt(np.maximum(values, 0))) @ np.array([np.cos(phase), np.sin(phase)])
        x = math.sqrt(self.geometric * self.beta) * np.cos(phase)
        xp = -math.sqrt(self.geometric / self.beta) * (self.alpha * np.cos(phase) + np.sin(phase))
        return n * np.array([x, xp])


def emittance_from(kinematics: Kinematics, known: str, value: float, beta: float, alpha=0.,
                   dispersion=0., dispersion_prime=0., sigma_delta=0.) -> EmittanceResult:
    """Uncorrelated betatron and momentum coordinates, one transverse plane.

    ``known``: geometric/normalized RMS emittance (m rad) or sigma (m).
    Dispersion is m, dispersion_prime is rad, sigma_delta is relative dp/p.
    No implicit pi or 4-rms factor is applied.
    """
    value = finite_number(value, "已知量", minimum=0)
    beta = finite_number(beta, "Twiss β", positive=True)
    alpha = finite_number(alpha, "Twiss α")
    d = finite_number(dispersion, "色散 D")
    dp = finite_number(dispersion_prime, "色散 D′")
    spread = finite_number(sigma_delta, "σδ", minimum=0)
    if known == "geometric":
        emit = value
    elif known == "normalized":
        if kinematics.beta_gamma == 0:
            raise ValueError("Ek=0 时不能由归一化发射度唯一反算几何发射度。")
        emit = value / kinematics.beta_gamma
    elif known == "sigma":
        variance = (value - abs(d * spread)) * (value + abs(d * spread))
        if variance < -1e-12 * max(value**2, (d * spread)**2, 1e-300):
            raise ValueError("输入束斑小于色散贡献 |D|σδ，不能得到非负发射度。")
        emit = max(variance, 0.) / beta
    else:
        raise ValueError(f"未知发射度输入量：{known}")
    gamma = (1 + alpha**2) / beta
    sigma_x = math.hypot(math.sqrt(beta * emit), d * spread)
    sigma_xp = math.hypot(math.sqrt(gamma * emit), dp * spread)
    cov = -alpha * emit + d * dp * spread**2
    # Expanded determinant avoids cancellation for strongly tilted ellipses.
    dispersion_term = (math.sqrt(beta) * dp + alpha * d / math.sqrt(beta))**2 + d*d/beta
    projected = math.sqrt(emit) * math.sqrt(emit + spread**2 * dispersion_term)
    correlation = cov / (sigma_x * sigma_xp) if sigma_x and sigma_xp else None
    result = EmittanceResult(emit, emit * kinematics.beta_gamma, projected, gamma,
                            math.sqrt(beta * emit), sigma_x, sigma_xp, cov,
                            correlation, beta, alpha)
    for name in result.__dataclass_fields__:
        number = getattr(result, name)
        if number is not None:
            finite_number(number, "计算结果")
    return result


def signed_rigidity(kinematics: Kinematics):
    if kinematics.brho is None:
        raise ValueError("中性粒子没有磁刚度，不能进行磁铁参数换算。")
    return math.copysign(kinematics.brho, kinematics.particle.charge_state)


@dataclass(frozen=True)
class DipoleResult:
    field: float
    radius: float | None
    curvature: float
    angle: float
    integrated_field: float


def dipole_from(brho: float, length: float, known: str, value: float) -> DipoleResult:
    """B [T], signed rho [m], theta [rad] or integral B dl [T m].

    L is effective arc length, not chord length; fields are uniform/effective.
    """
    brho = finite_number(brho, "带符号磁刚度", nonzero=True)
    length = finite_number(length, "有效磁长", positive=True)
    value = finite_number(value, "已知量")
    if known == "field":
        curvature = value / brho
    elif known == "radius":
        curvature = 1 / finite_number(value, "弯转半径", nonzero=True)
    elif known == "angle":
        curvature = value / length
    elif known == "integrated_field":
        curvature = value / brho / length
    else:
        raise ValueError(f"未知二极铁输入量：{known}")
    result = DipoleResult(curvature * brho, 1 / curvature if curvature else None,
                          curvature, curvature * length, curvature * brho * length)
    for item in result.__dict__.values():
        if item is not None:
            finite_number(item, "计算结果")
    return result


@dataclass(frozen=True)
class QuadrupoleResult:
    gradient: float
    k1: float
    k1l: float
    integrated_gradient: float
    focal_x: float | None
    focal_y: float | None
    pole_field: float


def quadrupole_from(brho: float, length: float, known: str, value: float, radius=.03) -> QuadrupoleResult:
    """G [T/m], K1 [1/m^2], K1L [1/m], fx [m], or pole field [T]."""
    brho = finite_number(brho, "带符号磁刚度", nonzero=True)
    length = finite_number(length, "有效磁长", positive=True)
    radius = finite_number(radius, "磁极半径", positive=True)
    value = finite_number(value, "已知量")
    if known == "gradient":
        k1 = value / brho
    elif known == "k1":
        k1 = value
    elif known == "k1l":
        k1 = value / length
    elif known == "focal_x":
        k1 = 1 / finite_number(value, "薄透镜焦距", nonzero=True) / length
    elif known == "pole_field":
        k1 = value / (radius * brho)
    else:
        raise ValueError(f"未知四极铁输入量：{known}")
    k1l = k1 * length
    result = QuadrupoleResult(k1 * brho, k1, k1l, k1l * brho,
                              1 / k1l if k1l else None, -1 / k1l if k1l else None, k1*brho*radius)
    for item in result.__dict__.values():
        if item is not None:
            finite_number(item, "计算结果")
    return result


@dataclass(frozen=True)
class MultipoleResult:
    derivative: float
    kn: float
    knl: float
    integrated_derivative: float
    pole_field: float


def multipole_from(order, brho, length, known, value, radius=.03):
    """Normal sextupole/octupole: G_n=d^n By/dx^n, k_n=G_n/(B rho).

    The field at (x=r,y=0) is G_n*r^n/n!, consistent with PASS's kicks.
    """
    if order not in (2, 3):
        raise ValueError("此换算支持六极 n=2 和八极 n=3。")
    brho = finite_number(brho, "带符号磁刚度", nonzero=True)
    length = finite_number(length, "有效磁长", positive=True)
    radius = finite_number(radius, "参考半径", positive=True)
    value = finite_number(value, "已知量")
    if known == "derivative":
        derivative = value
    elif known == "kn":
        derivative = value * brho
    elif known == "knl":
        derivative = value * brho / length
    elif known == "integrated_derivative":
        derivative = value / length
    elif known == "pole_field":
        derivative = value * math.factorial(order) / radius**order
    else:
        raise ValueError(f"未知多极铁输入量：{known}")
    kn = derivative / brho
    result = MultipoleResult(derivative, kn, kn*length, derivative*length,
                             derivative*radius**order / math.factorial(order))
    for item in result.__dict__.values():
        finite_number(item, "计算结果")
    return result


@dataclass(frozen=True)
class SolenoidResult:
    field: float
    ks: float
    kappa: float
    angle: float
    integrated_field: float
    focusing: float
    focal: float | None


def solenoid_from(brho, length, known, value):
    """PASS Ks=Bz/(B rho); paraxial reference Larmor parameter theta=Ks*L/2.

    Focal length is the weak thin-lens estimate 1/(kappa^2 L), not an exact
    end-face focal length of the finite solenoid tracking map.
    """
    brho = finite_number(brho, "带符号磁刚度", nonzero=True)
    length = finite_number(length, "有效磁长", positive=True)
    value = finite_number(value, "已知量")
    if known == "field":
        ks = value / brho
    elif known == "ks":
        ks = value
    elif known == "angle":
        ks = 2 * value / length
    elif known == "integrated_field":
        ks = value / (brho*length)
    else:
        raise ValueError(f"未知螺线管输入量：{known}")
    kappa = ks / 2
    result = SolenoidResult(ks*brho, ks, kappa, kappa*length, ks*brho*length,
                           kappa**2, 1/(kappa*kappa*length) if kappa else None)
    for item in result.__dict__.values():
        if item is not None:
            finite_number(item, "计算结果")
    return result
