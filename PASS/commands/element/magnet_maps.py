"""Shared CPU/GPU polynomial maps and GPU launch context, without element dispatch."""

from functools import lru_cache

import numpy as np

from PASS.utils.constants import const, _build_yoshida_cuda_constants
from PASS.utils.slicing import run_body_slices


def _apply_multipole_kick_cpu(knl, ksl, inv_fact, x, px, y, py, tag, scale=1.0):
    """Horner kick for integrated strengths; p_x=P_x/P0 and p_y=P_y/P0."""
    active = tag > 0
    # Evaluate only live particles, including when lost coordinates are nonfinite.
    active = slice(None) if np.all(active) else active
    xr, yi = x[active], y[active]
    re = np.full_like(xr, knl[-1] * inv_fact[-1] * scale)
    im = np.full_like(yi, ksl[-1] * inv_fact[-1] * scale)
    for i in range(len(knl) - 2, -1, -1):
        re, im = (re * xr - im * yi + knl[i] * inv_fact[i] * scale, re * yi + im * xr + ksl[i] * inv_fact[i] * scale)
    px[active] -= re
    py[active] += im


CUDA_REAL_PREAMBLE = _build_yoshida_cuda_constants() + f'''
#ifndef PASS_USE_FLOAT
#define PASS_USE_FLOAT 0
#endif
#if PASS_USE_FLOAT
using pass_real_t = float;
#else
using pass_real_t = double;
#endif
#define PASS_EPS ((pass_real_t){const.eps:.17g})
'''

MULTIPOLE_KERNEL_BODY = r'''
__device__ __forceinline__ bool pass_drift(
    pass_real_t& x,
    pass_real_t& px,
    pass_real_t& y,
    pass_real_t& py,
    pass_real_t& z,
    pass_real_t dp,
    int& tag,
    float* lost_position,
    int* lost_turn,
    int index,
    double L,
    pass_real_t reference_beta_gamma,
    pass_real_t inv_gamma,
    pass_real_t s_position,
    int turn
) {
    // A tolerance matches the CPU const.eps guard and avoids exact floating-point
    // equality.  ``L`` is an element/slice length, so sub-epsilon transport is
    // intentionally treated as a no-op.
    if (fabs(L) < PASS_EPS || tag <= 0)
        return tag > 0;
    pass_real_t momentum_ratio = (pass_real_t)1 + dp;
    pass_real_t pz_sq = momentum_ratio * momentum_ratio - px * px - py * py;
    if (!(pz_sq > (pass_real_t)0)) {
        tag = -abs(tag);
        lost_position[index] = (float)s_position;
        lost_turn[index] = turn;
        return false;
    }
    pass_real_t pz = sqrt(pz_sq);
    pass_real_t inv_pz = (pass_real_t)1 / pz;
    pass_real_t inv_gamma_sq = inv_gamma * inv_gamma;
    pass_real_t transverse_momentum_squared = px * px + py * py;
    pass_real_t energy_ratio = sqrt(inv_gamma_sq + ((pass_real_t)1 - inv_gamma_sq) * momentum_ratio * momentum_ratio);
    // Stable even when the physical time slip is much smaller than float epsilon.
    pass_real_t slip = (dp * ((pass_real_t)2 + dp) * inv_gamma_sq - transverse_momentum_squared) / (pz * (pz + energy_ratio));
    x += L * px * inv_pz;
    y += L * py * inv_pz;
    z += L * slip;
    return true;
}

__device__ __forceinline__ void pass_kick(
    pass_real_t& px,
    pass_real_t& py,
    pass_real_t x,
    pass_real_t y,
    const pass_real_t* __restrict__ knl,
    const pass_real_t* __restrict__ ksl,
    const pass_real_t* __restrict__ inv_fact,
    int order,
    double scale
) {
    double dpx_mul = (double)knl[order] * inv_fact[order] * scale;
    double dpy_mul = (double)ksl[order] * inv_fact[order] * scale;
    for (int n = order; n > 0; --n) {
        double zre = dpx_mul * x - dpy_mul * y;
        double zim = dpx_mul * y + dpy_mul * x;
        dpx_mul = (double)knl[n - 1] * inv_fact[n - 1] * scale + zre;
        dpy_mul = (double)ksl[n - 1] * inv_fact[n - 1] * scale + zim;
    }
    px -= dpx_mul;
    py += dpy_mul;
}

__device__ __forceinline__ bool pass_dkd_step(
    pass_real_t& x,
    pass_real_t& px,
    pass_real_t& y,
    pass_real_t& py,
    pass_real_t& z,
    pass_real_t dp,
    int& tag,
    float* lost_position,
    int* lost_turn,
    int index,
    double ds,
    pass_real_t reference_beta_gamma,
    pass_real_t inv_gamma,
    pass_real_t s_position,
    int turn,
    const pass_real_t* __restrict__ knl,
    const pass_real_t* __restrict__ ksl,
    const pass_real_t* __restrict__ inv_fact,
    int order
) {
    if (!pass_drift(x, px, y, py, z, dp, tag, lost_position, lost_turn, index, ds * (pass_real_t)0.5, reference_beta_gamma, inv_gamma, s_position,
                    turn))
        return false;
    pass_kick(px, py, x, y, knl, ksl, inv_fact, order, ds);
    return pass_drift(x, px, y, py, z, dp, tag, lost_position, lost_turn, index, ds * (pass_real_t)0.5, reference_beta_gamma, inv_gamma, s_position,
                      turn);
}

// mode: 0 = thin integrated kick, 1 = thick sliced DKD, 2 = pure drift.
// integrator: 0 = uniform second-order DKD, 1 = Yoshida fourth-order DKD.
extern "C" __global__ void track_multipole_dkd(
    pass_real_t* __restrict__ x,
    pass_real_t* __restrict__ px,
    pass_real_t* __restrict__ y,
    pass_real_t* __restrict__ py,
    pass_real_t* __restrict__ z,
    const pass_real_t* __restrict__ dp,
    int* __restrict__ tag,
    float* __restrict__ lost_position,
    int* __restrict__ lost_turn,
    int start_index,
    int end_index,
    pass_real_t reference_beta_gamma,
    pass_real_t inv_gamma,
    double L,
    pass_real_t s_position,
    int turn,
    const pass_real_t* __restrict__ knl,
    const pass_real_t* __restrict__ ksl,
    const pass_real_t* __restrict__ inv_fact,
    int order,
    int num_slice,
    int integrator,
    int mode
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (index >= end_index || tag[index] <= 0)
        return;

    pass_real_t xi = x[index], pxi = px[index];
    pass_real_t yi = y[index], pyi = py[index];
    pass_real_t zi = z[index], dpi = dp[index];
    int ti = tag[index];
    bool alive = true;

    if (mode == 0) {
        pass_kick(pxi, pyi, xi, yi, knl, ksl, inv_fact, order, (pass_real_t)1);
    } else if (mode == 2) {
        alive = pass_drift(xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn, index, L, reference_beta_gamma, inv_gamma, s_position, turn);
    } else {
        const double ds = L / (double)num_slice;
        for (int slice = 0; slice < num_slice && alive; ++slice) {
            if (integrator == 0) {
                alive = pass_dkd_step(xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn, index, ds, reference_beta_gamma, inv_gamma, s_position,
                                      turn, knl, ksl, inv_fact, order);
            } else {
                alive = pass_dkd_step(xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn, index, ds * pass_yoshida_z1, reference_beta_gamma,
                                      inv_gamma, s_position, turn, knl, ksl, inv_fact, order);
                if (alive)
                    alive = pass_dkd_step(xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn, index, ds * pass_yoshida_z0, reference_beta_gamma,
                                          inv_gamma, s_position, turn, knl, ksl, inv_fact, order);
                if (alive)
                    alive = pass_dkd_step(xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn, index, ds * pass_yoshida_z1, reference_beta_gamma,
                                          inv_gamma, s_position, turn, knl, ksl, inv_fact, order);
            }
        }
    }

    if (alive) {
        x[index] = xi;
        px[index] = pxi;
        y[index] = yi;
        py[index] = pyi;
        z[index] = zi;
    }
    tag[index] = ti;
}
'''

_GPU_STAGE_HEADER = r"""
extern "C" __global__ void internal_stage(
    pass_particle_t* x,
    pass_particle_t* px,
    pass_particle_t* y,
    pass_particle_t* py,
    pass_particle_t* z,
    const pass_particle_t* dp,
    int* tag,
    float* lp,
    int* lt,
    int start,
    int end,
    pass_real_t beta0,
    pass_real_t reference_beta_gamma,
    pass_real_t invgamma,
    double L,
    pass_real_t s0,
    int turn,
    const pass_real_t* params,
    const pass_real_t* kn,
    const pass_real_t* ks,
    const pass_real_t* inv,
    int order,
    int action
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i >= end || tag[i] <= 0)
        return;
    pass_real_t xi = x[i], pxi = px[i], yi = y[i], pyi = py[i], zi = z[i], dpi = dp[i];
    int ti = tag[i];
    bool alive = true;
"""
_GPU_STAGE_FOOTER = r"""
    x[i]=xi;px[i]=pxi;y[i]=yi;py[i]=pyi;z[i]=zi;tag[i]=ti;
}
"""
_GPU_STAGE_MULTIPOLE = r"""
if (action != 2) {
    alive = pass_drift(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i, L / 2, reference_beta_gamma, invgamma, s0, turn);
    if (alive)
        pass_kick(pxi, pyi, xi, yi, kn, ks, inv, order, L);
}
if (action != 1 && alive)
    pass_drift(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i, L / 2, reference_beta_gamma, invgamma, s0, turn);
"""


def _prepare_multipole_coefficients(element, normal, skew):
    """Combine full nominal KL and absolute errors once, returning map coefficients."""
    normal, skew = np.asarray(normal, dtype=float), np.asarray(skew, dtype=float)
    errors = element.field_errors
    signature = (element.length, element.is_thick, normal.tobytes(), skew.tobytes(), errors.active, errors.knl.tobytes(), errors.ksl.tobytes())
    cached = getattr(element, "_multipole_map_coefficients", None)
    if cached is not None and cached[0] == signature:
        return cached[1]
    kn, ks = errors.combine(normal, skew)
    if element.is_thick:
        kn, ks = kn / element.length, ks / element.length
    inv_fact = np.ones(len(kn))
    for index in range(1, len(inv_fact)):
        inv_fact[index] = inv_fact[index - 1] / index
    coefficients = (kn, ks, inv_fact)
    element._multipole_map_coefficients = (signature, coefficients)
    return coefficients


def _upload_coefficients(element, name, coefficients, dtype, device):
    """Reuse device buffers; update their contents only when coefficients change."""
    import cupy as cp

    cache = getattr(element, "_gpu_map_resources", None)
    if cache is None:
        cache = element._gpu_map_resources = {}
    key = (name, np.dtype(dtype).str, device)
    host = tuple(np.ascontiguousarray(value, dtype=dtype) for value in coefficients)
    signature = tuple((value.shape, value.tobytes()) for value in host)
    previous = cache.get(key)
    if previous is not None and previous[0] == signature:
        return previous[1]
    with cp.cuda.Device(device):
        if previous is not None and all(old.shape == new.shape for old, new in zip(previous[1], host, strict=True)):
            buffers = previous[1]
            for old, new in zip(buffers, host, strict=True):
                old.set(new)
        else:
            buffers = tuple(cp.asarray(value) for value in host)
    cache[key] = (signature, buffers)
    return buffers


@lru_cache(maxsize=None)
def _get_multipole_kernel(dtype, device):
    import cupy as cp

    with cp.cuda.Device(device):
        return cp.RawKernel(CUDA_REAL_PREAMBLE + MULTIPOLE_KERNEL_BODY,
                            "track_multipole_dkd",
                            options=("--std=c++17", f"-DPASS_USE_FLOAT={int(np.dtype(dtype).itemsize == 4)}"))


def _launch_multipole_bunch(element, beam, bunch, turn, kn, ks, inv_fact, mode):
    """Track exactly one bunch: mode 0 thin kick, 1 sliced DKD, or 2 pure drift."""
    import cupy as cp

    start, end = bunch.start_idx, bunch.end_idx
    if end <= start:
        return
    p = beam.particles
    real = p.real
    device = cp.cuda.runtime.getDevice()
    coefficients = _upload_coefficients(element, "multipole", (kn, ks, inv_fact), p.dtype, device)
    kernel = _get_multipole_kernel(np.dtype(p.dtype).str, device)
    kernel(((end - start + 255) // 256, ), (256, ),
           (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, p.lost_position, p.lost_turn, np.int32(start), np.int32(end), real(
               bunch.beta * bunch.gamma), real(1 / bunch.gamma), np.float64(element.length), real(element.s), np.int32(turn), *coefficients,
            np.int32(len(kn) - 1), np.int32(element.num_slice), np.int32(element.integrator != "uniform"), np.int32(mode)))


def _compile_stage_kernel(source, body, dtype, device, *, use_double=False):
    """Compile an element-owned stage using the common particle/launch interface."""
    import cupy as cp

    particle_type = "float" if np.dtype(dtype).itemsize == 4 else "double"
    with cp.cuda.Device(device):
        return cp.RawKernel(f"typedef {particle_type} pass_particle_t;\n" + source + _GPU_STAGE_HEADER + body + _GPU_STAGE_FOOTER,
                            "internal_stage",
                            options=("--std=c++17", f"-DPASS_USE_FLOAT={int(particle_type == 'float' and not use_double)}"))


@lru_cache(maxsize=None)
def _multipole_stage_kernel(dtype, device):
    return _compile_stage_kernel(CUDA_REAL_PREAMBLE + MULTIPOLE_KERNEL_BODY, _GPU_STAGE_MULTIPOLE, dtype, device)


class _GpuBody:
    """Bind one bunch to device maps and maintain the scheduler's loss position."""

    def __init__(self, element, beam, bunch, turn):
        import cupy as cp

        self.element, self.beam, self.bunch, self.turn = element, beam, bunch, turn
        self.p = beam.particles
        self.key = (np.dtype(self.p.dtype).str, cp.cuda.runtime.getDevice())
        self.blocks = ((bunch.end_idx - bunch.start_idx + 255) // 256, )
        self.position = element.s - element.length

    def stage(self, kernel, coefficients, *, use_double=False):
        p, bunch = self.p, self.bunch
        real = np.float64 if use_double else p.real
        cache = _upload_coefficients(self.element, "stage", coefficients, np.float64 if use_double else p.dtype, self.key[1])
        order = np.int32(len(coefficients[1]) - 1)

        def launch(length, action):
            kernel(self.blocks, (256, ), (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, p.lost_position, p.lost_turn, np.int32(
                bunch.start_idx), np.int32(bunch.end_idx), real(bunch.beta), real(bunch.beta * bunch.gamma), real(
                    1 / bunch.gamma), np.float64(length), real(self.position), np.int32(self.turn), *cache, order, np.int32(action)))

        return launch

    def multipole_stage(self, kn, ks, inv_fact):
        return self.stage(_multipole_stage_kernel(*self.key), (np.zeros(1), kn, ks, inv_fact))

    def drift(self, length):
        from PASS.commands.element.drift import _get_transfer_drift_kernel

        p, bunch = self.p, self.bunch
        real = p.real
        _get_transfer_drift_kernel(np.dtype(p.dtype).str)(
            self.blocks, (256, ), (p.x, p.y, p.z, p.px, p.py, p.dp, p.tag, p.lost_position, p.lost_turn, np.int32(
                bunch.start_idx), np.int32(bunch.end_idx), real(1 / bunch.gamma**2), real(length), real(self.position), np.int32(self.turn)))

    def run(self, transport):

        def advance(ds, on_center):
            end_position = self.position + ds
            self.position += ds / 2 if on_center is not None else ds
            callback = None
            if on_center is not None:

                def callback():
                    on_center()
                    self.position = end_position

            transport(ds, callback)
            self.position = end_position

        run_body_slices(self.element, self.beam, self.bunch, self.turn, advance, gpu=True)
