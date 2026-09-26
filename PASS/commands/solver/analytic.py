"""Particle-local free-space fields with fixed or per-slice transverse moments.

All fields are longitudinally integrated (V); charges are signed Coulombs.
The caller supplies slice membership. No slicing history or particle z is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .formula_gaussian_round import gaussian_round_field
from .formula_gaussian_ellipse import gaussian_elliptic_field
from .formula_uniform_round import uniform_round_field
from .formula_uniform_ellipse import uniform_elliptic_field
from .formula_parabolic import parabolic_round_field, parabolic_elliptic_field, PARABOLIC_CUDA
from .pic import PICResult
from PASS.utils.constants import const


class PotentialQuadrature:
    """Reusable confocal Green quadrature, including an independent error estimate."""

    def __init__(self, xp=np, dtype="float64", order=64):
        self.xp, self.dtype = xp, np.dtype(dtype)
        self.rules = []
        for count in (order, order // 2):
            nodes, weights = np.polynomial.legendre.leggauss(count)
            self.rules.append((xp.asarray((nodes + 1) / 2, dtype=dtype), xp.asarray(weights / 2, dtype=dtype)))
        self.error_flags = xp.zeros((), dtype=xp.int32)


def evaluate_potential_jet(x,
                           y,
                           charge,
                           center,
                           covariance,
                           center_derivative,
                           covariance_derivative,
                           solver,
                           quadrature,
                           reference_length=1.0,
                           compute_potential=False):
    """Return one potential-consistent jet, with a Gaussian closed-form path.

    Potential requests and conservatively rejected Gaussian points retain the
    two-rule Green integral. Covariance derivatives include the mixed entry.
    """
    if not solver.startswith("gaussian_") or compute_potential:
        return _evaluate_potential_jet_quadrature(x, y, charge, center, covariance, center_derivative, covariance_derivative, solver, quadrature,
                                                  reference_length, compute_potential)
    xp, dtype = quadrature.xp, quadrature.dtype
    x, y = xp.broadcast_arrays(xp.asarray(x, dtype=dtype), xp.asarray(y, dtype=dtype))
    shape = x.shape
    arrays = [xp.ascontiguousarray(value).reshape(-1) for value in (x, y)]
    for value, width in ((center, 2), (covariance, 3), (center_derivative, 2), (covariance_derivative, 3)):
        arrays.append(xp.ascontiguousarray(xp.broadcast_to(xp.asarray(value, dtype=dtype), shape + (width, ))).reshape(-1, width))
    outputs = [xp.zeros(x.size, dtype=dtype) for _ in range(4)]
    if xp is np:
        accepted, values = _gaussian_jet_cpu(*arrays, charge)
        for output, value in zip(outputs[1:], values):
            output[:] = value
            accepted &= np.isfinite(output)
    else:
        accepted = xp.empty(x.size, dtype=xp.bool_)
        if x.size:
            module = _analytic_gpu_module(dtype.str, xp.cuda.runtime.getDevice())
            module.get_function("gaussian_covariance_jet")(((x.size + 255) // 256, ), (256, ),
                                                           (*arrays, np.float64(charge), *outputs[1:], accepted, np.int64(x.size)))
    fallback = xp.flatnonzero(~accepted)
    if fallback.size:
        selected = [array[fallback] for array in arrays]
        values = _evaluate_potential_jet_quadrature(*selected[:2], charge, *selected[2:], solver, quadrature, reference_length, False)
        for output, value in zip(outputs, values):
            output[fallback] = value
    if xp is np and bool(quadrature.error_flags):
        raise ValueError("analytic potential covariance or quadrature convergence check failed")
    return tuple(value.reshape(shape) for value in outputs)


def _gaussian_jet_cpu(x, y, center, covariance, center_derivative, covariance_derivative, charge):
    """Gaussian heat identity: dPsi = E.dmu + Hessian(Psi):dC/2."""
    from scipy.special import wofz

    arrays = [np.asarray(value, dtype=np.float64) for value in (x, y, center, covariance, center_derivative, covariance_derivative)]
    x, y, center, covariance, center_derivative, covariance_derivative = arrays
    xx, xy, yy = covariance.T
    determinant = xx * yy - xy * xy
    gap = np.hypot(xx - yy, 2 * xy)
    major = (xx + yy + gap) / 2
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        minor = determinant / major
        angle = .5 * np.arctan2(2 * xy, xx - yy)
        cosine, sine = np.cos(angle), np.sin(angle)
        rx, ry = x - center[:, 0], y - center[:, 1]
        u, v = cosine * rx + sine * ry, -sine * rx + cosine * ry
        radius_squared = u * u + v * v
        exact_round = (xx == yy) & (xy == 0)
        accepted = ((major > 0) & (minor > 0) & np.isfinite(major) & np.isfinite(minor)
                    & (exact_round | ((gap >= .02 * major) & (minor >= 1e-8 * major) & (radius_squared <= 100 * major))))
        accepted &= np.isfinite(x) & np.isfinite(y) & np.isfinite(center).all(axis=1)
        accepted &= np.isfinite(center_derivative).all(axis=1) & np.isfinite(covariance_derivative).all(axis=1) & np.isfinite(charge)
        # Rejected entries are not evaluated, avoiding cancellation near a=b.
        indices = np.flatnonzero(accepted)
        ex, ey, derivative = (np.zeros(x.size) for _ in range(3))
        if not indices.size:
            return accepted, (ex, ey, derivative)
        a, b, d = major[indices], minor[indices], gap[indices]
        u, v, r2 = u[indices], v[indices], radius_squared[indices]
        cs, sn = cosine[indices], sine[indices]
        round_mask = exact_round[indices]
        eu, ev, ga, gb, huv = (np.empty(indices.size) for _ in range(5))
        strength = charge / (2 * const.pi * const.epsilon0)
        if np.any(round_mask):
            ar, ur, vr, rr = (value[round_mask] for value in (a, u, v, r2))
            z = rr / (2 * ar)
            small = z < 1e-3
            safe = np.where(small, 1., z)
            f = -np.expm1(-z) / (2 * ar * safe)
            slope = (np.exp(-z) * (1 + z) - 1) / (2 * ar * ar * safe * safe)
            f = np.where(small, (1 - z / 2 + z**2 / 6 - z**3 / 24 + z**4 / 120) / (2 * ar), f)
            slope = np.where(small, (-.5 + z / 3 - z**2 / 8 + z**3 / 30 - z**4 / 144) / (2 * ar * ar), slope)
            eu[round_mask], ev[round_mask] = strength * f * ur, strength * f * vr
            ga[round_mask] = -strength * (f + slope * ur * ur) / 2
            gb[round_mask] = -strength * (f + slope * vr * vr) / 2
            huv[round_mask] = -strength * slope * ur * vr
        if np.any(~round_mask):
            selected = ~round_mask
            aa, bb, dd, uu, vv = (value[selected] for value in (a, b, d, u, v))
            sx, sy = np.sqrt(aa), np.sqrt(bb)
            den = np.sqrt(2 * dd)
            gaussian = np.exp(-(uu * uu / aa + vv * vv / bb) / 2)
            w1 = wofz((np.abs(uu) + 1j * np.abs(vv)) / den)
            w2 = wofz((np.abs(uu) * sy / sx + 1j * np.abs(vv) * sx / sy) / den)
            field = strength * np.sqrt(np.pi) / den * (w1 - gaussian * w2)
            ue, ve = field.imag * np.sign(uu), field.real * np.sign(vv)
            normalized_u, normalized_v = uu / sx, vv / sy
            near_center = normalized_u**2 + normalized_v**2 <= 1e-6
            total = sx + sy
            local_u = strength / total * normalized_u * (1 - normalized_u**2 * (2 * sx + sy) / (6 * total) - normalized_v**2 * sy / (2 * total))
            local_v = strength / total * normalized_v * (1 - normalized_v**2 * (2 * sy + sx) / (6 * total) - normalized_u**2 * sx / (2 * total))
            ue, ve = np.where(near_center, local_u, ue), np.where(near_center, local_v, ve)
            radial_work = uu * ue + vv * ve
            eu[selected], ev[selected] = ue, ve
            ga[selected] = (radial_work + strength * (sy / sx * gaussian - 1)) / (2 * dd)
            gb[selected] = -(radial_work + strength * (sx / sy * gaussian - 1)) / (2 * dd)
            huv[selected] = (uu * ve - vv * ue) / dd
        dm = center_derivative[indices]
        dc = covariance_derivative[indices]
        dmu, dmv = cs * dm[:, 0] + sn * dm[:, 1], -sn * dm[:, 0] + cs * dm[:, 1]
        duu = cs * cs * dc[:, 0] + 2 * cs * sn * dc[:, 1] + sn * sn * dc[:, 2]
        dvv = sn * sn * dc[:, 0] - 2 * cs * sn * dc[:, 1] + cs * cs * dc[:, 2]
        duv = cs * sn * (dc[:, 2] - dc[:, 0]) + (cs * cs - sn * sn) * dc[:, 1]
        ex[indices], ey[indices] = cs * eu - sn * ev, sn * eu + cs * ev
        derivative[indices] = eu * dmu + ev * dmv + ga * duu + gb * dvv + huv * duv
        accepted &= np.isfinite(ex) & np.isfinite(ey) & np.isfinite(derivative)
    return accepted, (ex, ey, derivative)


def _evaluate_potential_jet_quadrature(x,
                                       y,
                                       charge,
                                       center,
                                       covariance,
                                       center_derivative,
                                       covariance_derivative,
                                       solver,
                                       quadrature,
                                       reference_length=1.0,
                                       compute_potential=False):
    """Return (Psi, Ex, Ey, dPsi/dS) from the same free-space Green integral.

    Covariance and its derivative store (xx, xy, yy); center stores (x, y).
    Gaussian uses its covariance; uniform/parabolic support matrices are
    respectively 4/6 times covariance. Psi uses the physical log(r/r_ref)
    convention. Both rules split the integration at the ellipse scales so
    highly flat beams do not need a near-round or principal-angle difference.
    Errors remain device resident until the caller checks error_flags.
    """
    xp, dtype = quadrature.xp, quadrature.dtype
    scalar = dtype.type
    x, y = xp.broadcast_arrays(xp.asarray(x, dtype=dtype), xp.asarray(y, dtype=dtype))
    shape = x.shape
    center = xp.broadcast_to(xp.asarray(center, dtype=dtype), shape + (2, ))
    covariance = xp.broadcast_to(xp.asarray(covariance, dtype=dtype), shape + (3, ))
    center_derivative = xp.broadcast_to(xp.asarray(center_derivative, dtype=dtype), shape + (2, ))
    covariance_derivative = xp.broadcast_to(xp.asarray(covariance_derivative, dtype=dtype), shape + (3, ))
    profile = solver.split("_")[0]
    if profile not in {"gaussian", "uniform", "parabolic"}:
        raise ValueError(f"unsupported potential profile {solver!r}")
    factor = scalar(1 if profile == "gaussian" else 4 if profile == "uniform" else 6)
    power = 1 if profile == "uniform" else 2
    normalization = scalar(charge / (4 * const.pi * const.epsilon0))
    outputs = [xp.empty(shape, dtype=dtype) for _ in range(4)]
    for start in range(0, x.size, 1024):
        end = min(start + 1024, x.size)
        r = xp.stack((x.ravel()[start:end], y.ravel()[start:end]), axis=-1) - center.reshape(-1, 2)[start:end]
        c = covariance.reshape(-1, 3)[start:end] * factor
        dc = covariance_derivative.reshape(-1, 3)[start:end] * factor
        dm = center_derivative.reshape(-1, 2)[start:end]
        xx, xy, yy = (c[:, i] for i in range(3))
        dxx, dxy, dyy = (dc[:, i, None] for i in range(3))
        determinant = xx * yy - xy * xy
        invalid = xp.any((determinant <= 0) | (xx <= 0) | (yy <= 0) | ~xp.isfinite(determinant))
        quadrature.error_flags[...] |= invalid.astype(xp.int32)
        scale = xp.sqrt(determinant)
        rx, ry = r[:, 0], r[:, 1]
        lower = xp.zeros_like(rx)
        if profile != "gaussian":
            linear = xx + yy - rx * rx - ry * ry
            constant = determinant - yy * rx * rx - xx * ry * ry + scalar(2) * xy * rx * ry
            root = xp.sqrt(xp.maximum(scalar(0), linear * linear - scalar(4) * constant))
            denominator = linear + root
            stable = -scalar(2) * constant / xp.where(denominator != 0, denominator, scalar(1))
            lower = xp.where(constant < 0, xp.maximum(scalar(0), xp.where(linear >= 0, stable, (root - linear) / scalar(2))), scalar(0))
        major = (xx + yy + xp.hypot(xx - yy, scalar(2) * xy)) / scalar(2)
        minor = determinant / major
        # A remote Gaussian observation samples t of order r^2. Both rules
        # otherwise miss that peak together and falsely report convergence.
        tail_scale = xp.maximum(major, rx * rx + ry * ry) if profile == "gaussian" else major
        first_scale, last_scale = (lower + minor)[:, None], (lower + tail_scale)[:, None]
        if xp is not np and profile == "gaussian" and not compute_potential:
            dm = xp.ascontiguousarray(dm)
        estimates = []
        for nodes, weights in quadrature.rules:
            if xp is not np and profile == "gaussian" and not compute_potential:
                # Retain the same rules and batch error check, but avoid hundreds
                # of tiny array kernels for the rare rejected Gaussian points.
                values = xp.empty((3, end - start), dtype=dtype)
                module = _analytic_gpu_module(dtype.str, xp.cuda.runtime.getDevice())
                module.get_function("gaussian_jet_quadrature")(
                    ((end - start + 127) // 128, ), (128, ),
                    (r, c, dc, dm, first_scale, last_scale, nodes, weights, normalization, values, np.int64(end - start), np.int32(nodes.size)))
                estimates.append((xp.zeros_like(values[0]), values[0], values[1], values[2]))
                continue
            u = nodes[None, :]
            log_ratio = xp.log(last_scale / first_scale) / scalar(2)
            middle_scale = xp.concatenate((first_scale * xp.exp(u * log_ratio), first_scale * xp.exp((scalar(1) + u) * log_ratio)), axis=1)
            t = lower[:, None] + xp.concatenate((first_scale * u, middle_scale, last_scale / (scalar(1) - u)), axis=1)
            measure = xp.concatenate(
                (weights[None, :] * first_scale, xp.tile(weights[None, :],
                                                         (1, 2)) * middle_scale * log_ratio, weights[None, :] * last_scale / (scalar(1) - u)**2),
                axis=1)
            ax, ay = xx[:, None] + t, yy[:, None] + t
            determinant_t = ax * ay - xy[:, None]**2
            inverse_root = scalar(1) / xp.sqrt(determinant_t)
            vx = (ay * rx[:, None] - xy[:, None] * ry[:, None]) / determinant_t
            vy = (ax * ry[:, None] - xy[:, None] * rx[:, None]) / determinant_t
            radius = rx[:, None] * vx + ry[:, None] * vy
            if profile == "gaussian":
                density_integral = xp.exp(-radius / scalar(2))
                radial_derivative = -density_integral / scalar(2)
                additive = scalar(np.euler_gamma - np.log(2))
            else:
                remaining = xp.maximum(scalar(0), scalar(1) - radius)
                density_integral = remaining**power
                radial_derivative = -scalar(power) * remaining**(power - 1)
                additive = scalar(1 if power == 1 else 1.5)
            weighted = measure * inverse_root
            ex = -scalar(2) * normalization * xp.sum(weighted * radial_derivative * vx, axis=-1)
            ey = -scalar(2) * normalization * xp.sum(weighted * radial_derivative * vy, axis=-1)
            trace = (ay * dxx - scalar(2) * xy[:, None] * dxy + ax * dyy) / determinant_t
            dradius = -scalar(2) * (vx * dm[:, 0, None] + vy * dm[:, 1, None]) - (dxx * vx * vx + scalar(2) * dxy * vx * vy + dyy * vy * vy)
            derivative = normalization * xp.sum(weighted * (radial_derivative * dradius - density_integral * trace / scalar(2)), axis=-1)
            if compute_potential:
                integral = xp.sum(measure * (density_integral * inverse_root - scalar(1) / (t + scale[:, None])), axis=-1)
                potential = normalization * (integral + xp.log(scalar(reference_length)**2 / (scale + lower)) + additive)
            else:
                potential = xp.zeros_like(ex)
            estimates.append((potential, ex, ey, derivative))
        tolerance = scalar(2e-9 if dtype.itemsize == 8 else 8e-5)
        natural_field = scalar(2) * abs(normalization) / xp.sqrt(xp.min(minor))
        natural_derivative = natural_field * (xp.max(xp.abs(dm)) + xp.max(xp.abs(dc)) / xp.sqrt(xp.min(minor)))
        scales = (abs(normalization), natural_field, natural_field, natural_derivative)
        for output, high, low, natural in zip(outputs, estimates[0], estimates[1], scales):
            failed = xp.any(~xp.isfinite(high)) | (xp.max(xp.abs(high - low)) > tolerance * (xp.max(xp.abs(high)) + scalar(.001) * natural))
            quadrature.error_flags[...] |= failed.astype(xp.int32) * xp.int32(2)
            output.ravel()[start:end] = high
    if xp is np and bool(quadrature.error_flags):
        raise ValueError("analytic potential covariance or quadrature convergence check failed")
    return tuple(outputs)


@dataclass
class AnalyticResult:
    integrated_ex: np.ndarray
    integrated_ey: np.ndarray
    slice_charge: np.ndarray
    macro_count: np.ndarray
    # Columns: center_x, center_y, size_x, size_y, angle; sizes are Gaussian
    # principal RMS widths or uniform/parabolic semi-axes. Empty slices use NaN parameters.
    parameters: np.ndarray


def evaluate_profile(x, y, charge, parameters, solver):
    """Return lab-axis fields and local coordinates for one source profile."""
    center_x, center_y, size_x, size_y, angle = parameters
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    dx, dy = x - center_x, y - center_y
    u, v = cos_angle * dx + sin_angle * dy, -sin_angle * dx + cos_angle * dy
    if solver == "gaussian_round_free_space":
        eu, ev = gaussian_round_field(u, v, charge, size_x)
    elif solver == "gaussian_ellipse_free_space":
        eu, ev = gaussian_elliptic_field(u, v, charge, size_x, size_y)
    elif solver == "uniform_round_free_space":
        eu, ev = uniform_round_field(u, v, charge, size_x)
    elif solver == "uniform_ellipse_free_space":
        eu, ev = uniform_elliptic_field(u, v, charge, size_x, size_y)
    elif solver == "parabolic_round_free_space":
        eu, ev = parabolic_round_field(u, v, charge, size_x)
    elif solver == "parabolic_ellipse_free_space":
        eu, ev = parabolic_elliptic_field(u, v, charge, size_x, size_y)
    else:
        raise ValueError(f"unsupported analytic solver {solver!r}")
    return cos_angle * eu - sin_angle * ev, sin_angle * eu + cos_angle * ev, u, v


def solve_analytic(x, y, slice_id, valid, num_slices, charge_per_macro, configuration):
    """Evaluate each nonempty slice; population moments use denominator N."""
    ex, ey = np.zeros_like(x, dtype=float), np.zeros_like(y, dtype=float)
    counts = np.bincount(slice_id[valid], minlength=num_slices)
    charges = counts * charge_per_macro
    parameters = np.full((num_slices, 5), np.nan)
    solver = configuration.solver
    round_profile = "_round_" in solver
    gaussian = solver.startswith("gaussian_")
    size_factor = 1.0 if gaussian else np.sqrt(6.0) if solver.startswith("parabolic_") else 2.0
    # Group once, rather than scanning all particles for every slice.
    active = np.flatnonzero(valid)
    order = active[np.argsort(slice_id[active], kind="stable")]
    offsets = np.r_[0, np.cumsum(counts)]
    for slice_index in np.flatnonzero(counts):
        indices = order[offsets[slice_index]:offsets[slice_index + 1]]
        if configuration.method == "frozen":
            center_x, center_y = configuration.center_x or 0.0, configuration.center_y or 0.0
            angle = configuration.angle or 0.0
            if round_profile:
                size_x = size_y = configuration.sigma if gaussian else configuration.radius
            elif gaussian:
                size_x, size_y = configuration.sigma_x, configuration.sigma_y
            else:
                size_x, size_y = configuration.a, configuration.b
        else:
            minimum = 2 if round_profile else 3
            if indices.size < minimum:
                raise ValueError(f"quasi-frozen slice {slice_index}: macro_count={indices.size}, requires at least {minimum}")
            center_x, center_y = float(np.mean(x[indices])), float(np.mean(y[indices]))
            dx, dy = x[indices] - center_x, y[indices] - center_y
            covariance = np.array([[np.mean(dx * dx), np.mean(dx * dy)], [np.mean(dx * dy), np.mean(dy * dy)]])
            if not np.all(np.isfinite(covariance)):
                raise ValueError(f"quasi-frozen slice {slice_index}: non-finite covariance")
            if round_profile:
                variance = np.trace(covariance) / 2.0
                if variance <= 0:
                    raise ValueError(f"quasi-frozen slice {slice_index}: zero transverse size")
                size_x = size_y = np.sqrt(variance) * size_factor
                angle = 0.0
            else:
                eigenvalues, axes = np.linalg.eigh(covariance)
                if eigenvalues[0] <= 64 * np.finfo(float).eps * eigenvalues[1]:
                    raise ValueError(f"quasi-frozen slice {slice_index}: degenerate covariance, eigenvalues={eigenvalues}")
                size_y, size_x = np.sqrt(eigenvalues) * size_factor
                angle = (np.arctan2(axes[1, 1], axes[0, 1]) + np.pi / 2) % np.pi - np.pi / 2
        parameters[slice_index] = center_x, center_y, size_x, size_y, angle
        ex[indices], ey[indices], _, _ = evaluate_profile(x[indices], y[indices], charges[slice_index], parameters[slice_index], solver)
    return AnalyticResult(ex, ey, charges, counts, parameters)


def sample_analytic_grid(result, configuration, geometry):
    """Diagnostic sampling only; no grid interpolation enters tracking."""
    x, y = np.meshgrid(geometry.x, geometry.y)
    shape = (result.slice_charge.size, geometry.ny, geometry.nx)
    density, ex, ey = (np.zeros(shape) for _ in range(3))
    gaussian = configuration.solver.startswith("gaussian_")
    for slice_index in np.flatnonzero(result.macro_count):
        charge = result.slice_charge[slice_index]
        ex[slice_index], ey[slice_index], u, v = evaluate_profile(x, y, charge, result.parameters[slice_index], configuration.solver)
        size_x, size_y = result.parameters[slice_index, 2:4]
        radius2 = (u / size_x)**2 + (v / size_y)**2
        if gaussian:
            density[slice_index] = charge / (2 * np.pi * size_x * size_y) * np.exp(-0.5 * radius2)
        elif configuration.solver.startswith("parabolic_"):
            density[slice_index] = 2 * charge / (np.pi * size_x * size_y) * np.maximum(1 - radius2, 0)
        else:
            density[slice_index] = charge / (np.pi * size_x * size_y) * (radius2 <= 1)
    return PICResult(density, None, ex, ey, geometry, result.slice_charge)


_ANALYTIC_PROFILES = (
    "gaussian_round_free_space",
    "gaussian_ellipse_free_space",
    "uniform_round_free_space",
    "uniform_ellipse_free_space",
    "parabolic_round_free_space",
    "parabolic_ellipse_free_space",
)


@lru_cache(maxsize=None)
def _analytic_gpu_module(dtype, device):
    """Compile Weideman's N=40 Faddeeva expansion (doi:10.1137/0731077).

    Coefficients are derived by a host FFT once per precision/device.
    Centered moments and special functions use FP64 intermediates; returned
    fields follow the particle dtype.
    """
    import cupy as cp

    n = 40
    m = 2 * n
    length = np.sqrt(n / np.sqrt(2.0))
    k = np.arange(-m + 1, m)
    t = length * np.tan(k * np.pi / (2 * m))
    f = np.r_[0, np.exp(-t * t) * (length * length + t * t)]
    a = (np.fft.fft(np.fft.fftshift(f)).real / (2 * m))[1:n + 1][::-1]
    preamble = ("#define REAL " + ("float" if np.dtype(dtype).itemsize == 4 else "double") + "\n" + f"#define WOFZ_L {length:.17g}\n" +
                "__constant__ double WOFZ_A[40]={" + ",".join(f"{v:.17g}" for v in a) + "};\n")
    with cp.cuda.Device(device):
        return cp.RawModule(
            code=preamble + PARABOLIC_CUDA + _ANALYTIC_CUDA,
            options=("--std=c++17", ),
        )


def solve_analytic_gpu(x, y, slice_id, valid, num_slices, charge_per_macro, configuration):
    """GPU analytical solve with centered two-pass per-slice statistics."""
    import cupy as cp

    x = cp.asarray(x, order="C")
    y = cp.asarray(y, dtype=x.dtype, order="C")
    if x.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("GPU analytical fields require float32 or float64 particles")
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be matching one-dimensional arrays")
    slice_indices = cp.ascontiguousarray(slice_id, dtype=cp.int64)
    valid = cp.ascontiguousarray(valid, dtype=cp.bool_)
    if slice_indices.shape != x.shape or valid.shape != x.shape:
        raise ValueError("slice_id and valid must match particle shape")
    if not np.isfinite(charge_per_macro):
        raise ValueError("charge_per_macro must be finite")
    n_slices = int(num_slices)
    if n_slices < 1 or n_slices != num_slices:
        raise ValueError("num_slices must be a positive integer")
    if configuration.solver not in _ANALYTIC_PROFILES:
        raise ValueError("unsupported analytical solver")
    kind = _ANALYTIC_PROFILES.index(configuration.solver)
    module = _analytic_gpu_module(x.dtype.str, cp.cuda.runtime.getDevice())
    n_particles = x.size
    blocks = (n_particles + 255) // 256
    params = cp.full((n_slices, 5), cp.nan, dtype=cp.float64)
    # For very large slice counts, split the slice range to stay within shared
    # memory limits. Ordinary PIC workloads (100 slices) take size_x single batch.
    sums = cp.zeros((n_slices, 3), cp.float64)

    def moments(centered):
        for first in range(0, n_slices, 512):
            count = min(512, n_slices - first)
            partial = cp.empty((blocks, count, 3), cp.float64)
            if n_particles:
                module.get_function("moments")(
                    (blocks, ),
                    (256, ),
                    (
                        x,
                        y,
                        slice_indices - first,
                        valid,
                        params[first:],
                        partial,
                        np.int32(n_particles),
                        np.int32(count),
                        np.int32(centered),
                    ),
                    shared_mem=count * 3 * 8,
                )
            sums[first:first + count] = partial.sum(axis=0)
        return sums

    moments(False)
    counts = sums[:, 0].astype(cp.int64)
    charges = counts.astype(cp.float64) * charge_per_macro
    if configuration.method == "frozen":
        config = configuration
        size_x = (config.sigma if kind == 0 else config.sigma_x if kind == 1 else config.radius if kind in (2, 4) else config.a)
        size_y = size_x if kind in (0, 2, 4) else config.sigma_y if kind == 1 else config.b
        params[:] = cp.asarray([config.center_x or 0.0, config.center_y or 0.0, size_x, size_y, config.angle or 0.0], cp.float64)
        params[counts == 0] = cp.nan
    else:
        module.get_function("centers")(((n_slices + 255) // 256, ), (256, ), (sums, params, np.int32(n_slices)))
        first_sums = sums.copy()
        moments(True)
        errors = cp.zeros(n_slices, cp.int32)
        module.get_function("sizes")(
            ((n_slices + 255) // 256, ),
            (256, ),
            (
                first_sums,
                sums,
                params,
                errors,
                np.int32(n_slices),
                np.int32(kind in (0, 2, 4)),
                np.float64(1.0 if kind < 2 else np.sqrt(6.0) if kind >= 4 else 2.0),
            ),
        )
        errors_host = errors.get()
        if np.any(errors_host):
            s = int(np.flatnonzero(errors_host)[0])
            raise ValueError(f"quasi-frozen slice {s}: " + ("too few particles" if errors_host[s] == 1 else "degenerate or non-finite covariance"))
    ex, ey = cp.empty_like(x), cp.empty_like(y)
    if n_particles:
        module.get_function("analytic_fields")(
            (blocks, ),
            (256, ),
            (
                x,
                y,
                slice_indices,
                valid,
                params,
                charges,
                ex,
                ey,
                np.int32(n_particles),
                np.int32(n_slices),
                np.int32(kind),
            ),
        )
    return AnalyticResult(ex, ey, charges, counts, params)


_ANALYTIC_CUDA = r"""
typedef REAL T;
struct Z {
    double r, i;
};
__device__ Z add(
    Z a,
    Z b
) {
    return {a.r + b.r, a.i + b.i};
}
__device__ Z mul(
    Z a,
    Z b
) {
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
__device__ Z divz(
    Z a,
    Z b
) {
    double s = b.r * b.r + b.i * b.i;
    return {(a.r * b.r + a.i * b.i) / s, (a.i * b.r - a.r * b.i) / s};
}
// Weideman (1994), upper half plane, N=40. Coefficients are precomputed once.
__device__ Z wofz_upper(
    double x,
    double y
) {
    Z den = {WOFZ_L + y, -x};
    Z z = divz({WOFZ_L - y, x}, den), p = {0, 0};
    for (int k = 0; k < 40; ++k)
        p = add(mul(p, z), {WOFZ_A[k], 0});
    return add(divz({2 * p.r, 2 * p.i}, mul(den, den)), divz({0.56418958354775628695, 0}, den));
}
extern "C" __global__ void test_wofz(
    const double* x,
    const double* y,
    double* re,
    double* im,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Z w = wofz_upper(x[i], y[i]);
        re[i] = w.r;
        im[i] = w.i;
    }
}

// Block-private histograms bound contention; two passes keep covariance
// centered, avoiding E[x*x]-E[x]^2 cancellation for displaced beams.
extern "C" __global__ void moments(
    const T* x,
    const T* y,
    const long long* slice_indices,
    const bool* valid,
    const double* params,
    double* partial,
    int n,
    int n_slices,
    int centered
) {
    extern __shared__ double cache[];
    for (int j = threadIdx.x; j < 3 * n_slices; j += blockDim.x)
        cache[j] = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && valid[i] && slice_indices[i] >= 0 && slice_indices[i] < n_slices) {
        int s = slice_indices[i];
        double a = x[i], b = y[i];
        if (centered) {
            a -= params[5 * s];
            b -= params[5 * s + 1];
            atomicAdd(cache + 3 * s, a * a);
            atomicAdd(cache + 3 * s + 1, b * b);
            atomicAdd(cache + 3 * s + 2, a * b);
        } else {
            atomicAdd(cache + 3 * s, 1.);
            atomicAdd(cache + 3 * s + 1, a);
            atomicAdd(cache + 3 * s + 2, b);
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < 3 * n_slices; j += blockDim.x)
        partial[(long long)blockIdx.x * 3 * n_slices + j] = cache[j];
}
extern "C" __global__ void centers(
    const double* sums,
    double* params,
    int n_slices
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < n_slices && sums[3 * s] > 0) {
        params[5 * s] = sums[3 * s + 1] / sums[3 * s];
        params[5 * s + 1] = sums[3 * s + 2] / sums[3 * s];
    }
}
extern "C" __global__ void sizes(
    const double* counts,
    const double* cov,
    double* p,
    int* error,
    int n_slices,
    int round,
    double size_factor
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slices || counts[3 * s] == 0)
        return;
    double count = counts[3 * s], a = cov[3 * s] / count, b = cov[3 * s + 1] / count, c = cov[3 * s + 2] / count;
    if (count < (round ? 2 : 3)) {
        error[s] = 1;
        return;
    }
    double hi, lo, angle;
    if (round) {
        hi = lo = (a + b) / 2;
        angle = 0;
    } else {
        double d = hypot(a - b, 2 * c);
        hi = (a + b + d) / 2;
        lo = (a + b - d) / 2;
        angle = 0.5 * atan2(2 * c, a - b);
    }
    if (!isfinite(hi) || !isfinite(lo) || hi <= 0 || lo <= (round ? 0 : 64 * 2.2204460492503131e-16 * hi)) {
        error[s] = 2;
        return;
    }
    p[5 * s + 2] = sqrt(hi) * size_factor;
    p[5 * s + 3] = sqrt(lo) * size_factor;
    p[5 * s + 4] = angle;
}

__device__ void profile(
    double x,
    double y,
    double q,
    double a,
    double b,
    int kind,
    double* ex,
    double* ey
) {
    const double pi = 3.14159265358979323846, eps = 8.8541878128e-12;
    if (kind >= 4) {
        parabolic_field(x, y, q, a, b, ex, ey);
        return;
    }
    if (kind == 0 || (kind == 1 && fabs(a - b) <= 1e-10 * b)) {
        double r2 = x * x + y * y;
        double f = r2 > 0 ? q / (2 * pi * eps) * (-expm1(-r2 / (2 * a * a))) / r2 : 0;
        *ex = f * x;
        *ey = f * y;
        return;
    }
    if (kind >= 2) {
        double lambda = 0;
        if (x * x / (a * a) + y * y / (b * b) > 1) {
            double aa = a * a, bb = b * b, linear = aa + bb - x * x - y * y, constant = aa * bb - bb * x * x - aa * y * y;
            double root = sqrt(fmax(0., linear * linear - 4 * constant));
            lambda = fmax(0., linear >= 0 ? -2 * constant / (linear + root) : (root - linear) / 2);
        }
        double A = sqrt(a * a + lambda), B = sqrt(b * b + lambda), f = q / (pi * eps * (A + B));
        *ex = f * x / A;
        *ey = f * y / B;
        return;
    }
    bool swap = a < b;
    if (swap) {
        double t = a;
        a = b;
        b = t;
        t = x;
        x = y;
        y = t;
    }
    double eu, ev, rx = x / a, ry = y / b, sum = a + b;
    if (rx * rx + ry * ry <= 1e-6) {
        double f = q / (2 * pi * eps * sum);
        eu = f * rx * (1 - rx * rx * (2 * a + b) / (6 * sum) - ry * ry * b / (2 * sum));
        ev = f * ry * (1 - ry * ry * (2 * b + a) / (6 * sum) - rx * rx * a / (2 * sum));
    } else {
        double diff = (a - b) * sum, den = sqrt(2 * diff), gx = fabs(x), gy = fabs(y), gauss = exp(-(rx * rx + ry * ry) / 2);
        Z w1 = wofz_upper(gx / den, gy / den), w2 = wofz_upper(gx * b / a / den, gy * a / b / den);
        double f = q / (2 * eps * sqrt(2 * pi * diff));
        eu = f * (w1.i - gauss * w2.i) * (x > 0 ? 1 : x < 0 ? -1 : 0);
        ev = f * (w1.r - gauss * w2.r) * (y > 0 ? 1 : y < 0 ? -1 : 0);
    }
    *ex = swap ? ev : eu;
    *ey = swap ? eu : ev;
}
extern "C" __global__ void gaussian_jet_quadrature(
    const T* r,
    const T* covariance,
    const T* covariance_derivative,
    const T* center_derivative,
    const T* first_scale,
    const T* last_scale,
    const T* nodes,
    const T* weights,
    T normalization,
    T* values,
    long long n,
    int order
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    T rx = r[2 * i], ry = r[2 * i + 1];
    T xx = covariance[3 * i], xy = covariance[3 * i + 1], yy = covariance[3 * i + 2];
    T dxx = covariance_derivative[3 * i], dxy = covariance_derivative[3 * i + 1], dyy = covariance_derivative[3 * i + 2];
    T dmx = center_derivative[2 * i], dmy = center_derivative[2 * i + 1];
    T first = first_scale[i], last = last_scale[i], log_ratio = log(last / first) / T(2);
    double sum_x = 0, sum_y = 0, sum_derivative = 0;
    for (int segment = 0; segment < 4; ++segment) {
        for (int node = 0; node < order; ++node) {
            T u = nodes[node], weight = weights[node], t, measure;
            if (segment == 0) {
                t = first * u;
                measure = weight * first;
            } else if (segment < 3) {
                t = first * exp((T(segment - 1) + u) * log_ratio);
                measure = weight * t * log_ratio;
            } else {
                t = last / (T(1) - u);
                measure = weight * last / ((T(1) - u) * (T(1) - u));
            }
            T ax = xx + t, ay = yy + t, determinant = ax * ay - xy * xy;
            T vx = (ay * rx - xy * ry) / determinant, vy = (ax * ry - xy * rx) / determinant;
            T density_integral = exp(-(rx * vx + ry * vy) / T(2));
            T weighted = measure / sqrt(determinant);
            T trace = (ay * dxx - T(2) * xy * dxy + ax * dyy) / determinant;
            T dradius = -T(2) * (vx * dmx + vy * dmy) - (dxx * vx * vx + T(2) * dxy * vx * vy + dyy * vy * vy);
            sum_x += double(weighted * density_integral * vx);
            sum_y += double(weighted * density_integral * vy);
            sum_derivative += double(weighted * (-density_integral / T(2) * dradius - density_integral * trace / T(2)));
        }
    }
    values[i] = normalization * T(sum_x);
    values[n + i] = normalization * T(sum_y);
    values[2 * n + i] = normalization * T(sum_derivative);
}
extern "C" __global__ void gaussian_covariance_jet(
    const T* x,
    const T* y,
    const T* center,
    const T* covariance,
    const T* center_derivative,
    const T* covariance_derivative,
    double charge,
    T* ex,
    T* ey,
    T* derivative,
    bool* accepted,
    long long n
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    accepted[i] = false;
    ex[i] = ey[i] = derivative[i] = 0;
    double xx = covariance[3 * i], xy = covariance[3 * i + 1], yy = covariance[3 * i + 2];
    double rx = (double)x[i] - center[2 * i], ry = (double)y[i] - center[2 * i + 1];
    double dmx = center_derivative[2 * i], dmy = center_derivative[2 * i + 1];
    double dxx = covariance_derivative[3 * i], dxy = covariance_derivative[3 * i + 1], dyy = covariance_derivative[3 * i + 2];
    double gap = hypot(xx - yy, 2 * xy), a = (xx + yy + gap) / 2, b = (xx * yy - xy * xy) / a;
    bool round = xx == yy && xy == 0;
    if (!(a > 0 && b > 0) || !isfinite(a) || !isfinite(b) || !isfinite(rx) || !isfinite(ry) || !isfinite(dmx) || !isfinite(dmy) || !isfinite(dxx) ||
        !isfinite(dxy) || !isfinite(dyy) || !isfinite(charge))
        return;
    double angle = .5 * atan2(2 * xy, xx - yy), cs = cos(angle), sn = sin(angle);
    double u = cs * rx + sn * ry, v = -sn * rx + cs * ry, r2 = u * u + v * v;
    // Closed covariance derivatives divide by the eigenvalue gap. Reject
    // ill-conditioned gaps, nearly singular ellipses and remote cancellation.
    if (!round && (gap < .02 * a || b < 1e-8 * a || r2 > 100 * a))
        return;
    const double pi = 3.14159265358979323846, eps = 8.8541878128e-12;
    double strength = charge / (2 * pi * eps), eu, ev, ga, gb, huv;
    if (round) {
        double z = r2 / (2 * a), f, slope;
        if (z < 1e-3) {
            f = (1 - z / 2 + z * z / 6 - z * z * z / 24 + z * z * z * z / 120) / (2 * a);
            slope = (-.5 + z / 3 - z * z / 8 + z * z * z / 30 - z * z * z * z / 144) / (2 * a * a);
        } else {
            f = -expm1(-z) / (2 * a * z);
            slope = (exp(-z) * (1 + z) - 1) / (2 * a * a * z * z);
        }
        eu = strength * f * u;
        ev = strength * f * v;
        ga = -strength * (f + slope * u * u) / 2;
        gb = -strength * (f + slope * v * v) / 2;
        huv = -strength * slope * u * v;
    } else {
        double sx = sqrt(a), sy = sqrt(b), gaussian = exp(-(u * u / a + v * v / b) / 2);
        profile(u, v, charge, sx, sy, 1, &eu, &ev);
        double radial_work = u * eu + v * ev;
        ga = (radial_work + strength * (sy / sx * gaussian - 1)) / (2 * gap);
        gb = -(radial_work + strength * (sx / sy * gaussian - 1)) / (2 * gap);
        huv = (u * ev - v * eu) / gap;
    }
    double dmu = cs * dmx + sn * dmy, dmv = -sn * dmx + cs * dmy;
    double duu = cs * cs * dxx + 2 * cs * sn * dxy + sn * sn * dyy;
    double dvv = sn * sn * dxx - 2 * cs * sn * dxy + cs * cs * dyy;
    double duv = cs * sn * (dyy - dxx) + (cs * cs - sn * sn) * dxy;
    ex[i] = cs * eu - sn * ev;
    ey[i] = sn * eu + cs * ev;
    derivative[i] = eu * dmu + ev * dmv + ga * duu + gb * dvv + huv * duv;
    accepted[i] = isfinite(ex[i]) && isfinite(ey[i]) && isfinite(derivative[i]);
}
extern "C" __global__ void analytic_fields(
    const T* x,
    const T* y,
    const long long* slice_indices,
    const bool* valid,
    const double* params,
    const double* charges,
    T* ex,
    T* ey,
    int n,
    int n_slices,
    int kind
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    ex[i] = 0;
    ey[i] = 0;
    if (!valid[i] || slice_indices[i] < 0 || slice_indices[i] >= n_slices)
        return;
    int s = slice_indices[i];
    const double* p = params + 5 * s;
    if (charges[s] == 0)
        return;
    double c = cos(p[4]), v = sin(p[4]), dx = (double)x[i] - p[0], dy = (double)y[i] - p[1], eu, ev;
    profile(c * dx + v * dy, -v * dx + c * dy, charges[s], p[2], p[3], kind, &eu, &ev);
    ex[i] = c * eu - v * ev;
    ey[i] = v * eu + c * ev;
}
"""
