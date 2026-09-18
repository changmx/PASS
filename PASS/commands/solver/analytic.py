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
