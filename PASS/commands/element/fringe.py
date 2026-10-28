"""PTC MULTIPOLE_FRINGER/FRINGE2QUADR maps (MAD-X 5.09.03), with z = -beta0 * internal_ct."""

from functools import lru_cache

import numpy as np


class _MagnetFringe:
    """Keep physical ends outside body slicing; FRNGMAX uses one-based pole order."""

    def __init__(self, kwargs, order=None):
        self.order = order
        self.fringe = kwargs.get("fringe", 0)
        self.frngmax = kwargs.get("frngmax", 2)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in (self.fringe, self.frngmax)):
            raise ValueError("FRINGE and FRNGMAX must be integers.")
        self.f1 = float(kwargs.get("f1 (m)", kwargs.get("f1", 0.))) if order == 1 else 0.
        self.f2 = float(kwargs.get("f2 (m3)", kwargs.get("f2", 0.))) if order == 1 else 0.
        kill_entrance, kill_exit = kwargs.get("kill_ent_fringe", False), kwargs.get("kill_exi_fringe", False)
        if not all(isinstance(value, (bool, np.bool_)) for value in (kill_entrance, kill_exit)):
            raise ValueError("KILL_ENT_FRINGE and KILL_EXI_FRINGE must be booleans.")
        self.kill = int(kill_entrance) + 2 * int(kill_exit)
        if not 0 <= self.fringe <= 7 or self.frngmax < 1 or not np.isfinite([self.f1, self.f2]).all():
            raise ValueError("Fringe requires FRINGE in [0, 7], FRNGMAX >= 1, and finite F1/F2.")
        self.active = self.kill != 3 and bool(self.fringe & 2 or self.fringe & 4 and (self.f1 or self.f2))

    def arguments(self, real, enabled=True):
        return (np.int32(self.fringe if self.active and enabled else 0), np.int32(min(self.frngmax,
                                                                                      np.iinfo(np.int32).max)), np.int32(self.kill), real(self.f1),
                real(self.f2))

    def enabled(self, kn, ks, inv_fact):
        """Reuse the coefficient cache, including ramp/error invalidation, for zero-field ends."""
        if not self.active or len(kn) < 2:
            return False
        previous = getattr(self, "_enabled_coefficients", None)
        if previous is not None and previous[0] is kn and previous[1] is ks and previous[2] is inv_fact:
            return previous[3]
        intrinsic = self.fringe & 2 and (np.any(kn[:self.frngmax]) or np.any(ks[:self.frngmax]))
        quadrupole = self.fringe & 4 and (self.f1 or self.f2) and (kn[1] or ks[1])
        enabled = bool(intrinsic or quadrupole)
        # inv_fact is renewed on every coefficient-cache miss, including in-place thin ramps.
        self._enabled_coefficients = (kn, ks, inv_fact, enabled)
        return enabled

    def coefficients(self, element):
        from PASS.commands.element.magnet_maps import _prepare_multipole_coefficients

        if self.order is None:
            normal, skew = element.knl, element.ksl
        else:
            normal, skew = np.zeros(self.order + 1), np.zeros(self.order + 1)
            normal[self.order] = getattr(element, f"k{self.order}l")
            skew[self.order] = getattr(element, f"k{self.order}sl")
        kn, ks, inv_fact = _prepare_multipole_coefficients(element, normal, skew)
        if not element.is_thick and self.enabled(kn, ks, inv_fact):
            raise ValueError("A nonzero straight-magnet fringe requires a positive physical length; integrated thin strengths do not define it.")
        return kn, ks, inv_fact

    def apply_cpu(self, element, beam, bunch, turn, entrance):
        if not self.active or self.kill & (1 if entrance else 2):
            return
        kn, ks, inv_fact = self.coefficients(element)
        if not self.enabled(kn, ks, inv_fact):
            return
        p = beam.particles
        start, end = bunch.start_idx, bunch.end_idx
        indices = np.flatnonzero(p.tag[start:end] > 0) + start
        if not len(indices):
            return
        x, px, y, py, z, dp = (getattr(p, name)[indices].astype(np.float64) for name in ("x", "px", "y", "py", "z", "dp"))
        momentum_ratio = 1. + dp
        energy_ratio = np.sqrt(1. + bunch.beta**2 * dp * (2. + dp))
        direction = 1. if entrance else -1.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if not entrance and self.fringe & 4 and len(kn) > 1:
                x, px, y, py, z = _quadrupole_fringe_cpu(x, px, y, py, z, momentum_ratio, energy_ratio, kn[1], ks[1], self.f1, self.f2, direction)
            if self.fringe & 2 and len(kn) > 1:
                x, px, y, py, z = _multipole_fringe_cpu(x, px, y, py, z, momentum_ratio, energy_ratio, kn, ks, inv_fact, self.frngmax, direction)
            if entrance and self.fringe & 4 and len(kn) > 1:
                x, px, y, py, z = _quadrupole_fringe_cpu(x, px, y, py, z, momentum_ratio, energy_ratio, kn[1], ks[1], self.f1, self.f2, direction)
        alive = (momentum_ratio > 0.) & (momentum_ratio**2 > px**2 + py**2) & np.isfinite(x + px + y + py + z)
        if np.dtype(p.dtype).itemsize < 8:
            # Double intermediates must also fit the particle storage before committing.
            limit = np.finfo(p.dtype).max
            for value in (x, px, y, py, z):
                alive &= np.abs(value) <= limit
        for name, value in zip(("x", "px", "y", "py", "z"), (x, px, y, py, z), strict=True):
            getattr(p, name)[indices[alive]] = value[alive]
        lost = indices[~alive]
        p.tag[lost] = -1
        p.lost_position[lost] = element.s - element.length if entrance else element.s
        p.lost_turn[lost] = turn

    def apply_gpu(self, element, beam, bunch, turn, entrance):
        """Only interrupted bodies need a separate edge launch; ordinary bodies fuse it."""
        if not self.active or self.kill & (1 if entrance else 2):
            return
        import cupy as cp

        from PASS.commands.element.magnet_maps import _upload_coefficients

        p = beam.particles
        device = cp.cuda.runtime.getDevice()
        kn, ks, inv_fact = self.coefficients(element)
        if not self.enabled(kn, ks, inv_fact):
            return
        coefficients = _upload_coefficients(element, "fringe", (kn, ks, inv_fact), p.dtype, device)
        n = bunch.end_idx - bunch.start_idx
        if n <= 0:
            return
        _get_fringe_kernel(np.dtype(p.dtype).str, device)(
            ((n + 255) // 256, ), (256, ), (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, p.lost_position, p.lost_turn, np.int32(
                bunch.start_idx), np.int32(bunch.end_idx), p.real(1. / bunch.gamma), p.real(element.s - element.length if entrance else element.s),
                                            np.int32(turn), *coefficients, np.int32(len(kn) - 1), *self.arguments(p.real), np.int32(entrance)))


def _multipole_fringe_cpu(x, px, y, py, z, momentum_ratio, energy_ratio, kn, ks, inv_fact, frngmax, direction):
    fx, fy, fxx, fxy, fyx, fyy = (np.zeros_like(x) for _ in range(6))
    rx, ix = np.ones_like(x), np.zeros_like(x)
    for j in range(1, min(len(kn), frngmax) + 1):
        drx, dix = rx, ix
        rx, ix = drx * x - dix * y, drx * y + dix * x
        normal, skew = kn[j - 1] * inv_fact[j - 1], ks[j - 1] * inv_fact[j - 1]
        if normal == 0. and skew == 0.:
            continue
        scale = -direction / (4. * (j + 1))
        u, v = scale * (normal * rx - skew * ix), scale * (normal * ix + skew * rx)
        du, dv = scale * (normal * drx - skew * dix), scale * (normal * dix + skew * drx)
        nf = (j + 2.) / j
        fx += u * x + nf * v * y
        fy += u * y - nf * v * x
        fxx += j * du * x + u + nf * y * j * dv
        fxy += -j * dv * x + nf * v + nf * y * j * du
        fyx += j * du * y - nf * v - nf * x * j * dv
        fyy += -j * dv * y + u - nf * x * j * du
    a, b, c, d = 1. - fxx / momentum_ratio, -fyx / momentum_ratio, -fxy / momentum_ratio, 1. - fyy / momentum_ratio
    det = a * d - b * c
    px, py = (d * px - b * py) / det, (a * py - c * px) / det
    z = z + energy_ratio * (px * fx + py * fy) / momentum_ratio**3
    return x - fx / momentum_ratio, px, y - fy / momentum_ratio, py, z


def _quadrupole_fringe_cpu(x, px, y, py, z, momentum_ratio, energy_ratio, k1, k1s, f1, f2, direction):
    if not (f1 or f2) or not (k1 or k1s):
        return x, px, y, py, z
    strength = np.hypot(k1, k1s)
    angle = -.5 * np.arctan2(k1s, k1)
    cosine, sine = np.cos(angle), np.sin(angle)
    x, y = cosine * x + sine * y, -sine * x + cosine * y
    px, py = cosine * px + sine * py, -sine * px + cosine * py
    a = -direction * f1 * abs(f1) * strength / (24. * momentum_ratio)
    b = f2 * strength / momentum_ratio
    exp_a, exp_minus_a = np.exp(a), np.exp(-a)
    z -= energy_ratio / momentum_ratio * ((a * x + b * (1. + a / 2.) * px / momentum_ratio * exp_minus_a) * px / momentum_ratio -
                                          (a * y + b * (1. - a / 2.) * py / momentum_ratio * exp_a) * py / momentum_ratio)
    x, y = x * exp_a + px * b / momentum_ratio, y * exp_minus_a - py * b / momentum_ratio
    px, py = px * exp_minus_a, py * exp_a
    return cosine * x - sine * y, cosine * px - sine * py, sine * x + cosine * y, sine * px + cosine * py, z


FRINGE_CUDA = r'''
#ifndef PASS_ENABLE_FRINGE
#define PASS_ENABLE_FRINGE 1
#endif
__device__ inline void pass_qsad_fringe(
    pass_real_t& x,
    pass_real_t& px,
    pass_real_t& y,
    pass_real_t& py,
    pass_real_t& z,
    double r,
    double energy_ratio,
    double k1,
    double k1s,
    double f1,
    double f2,
    double direction
) {
    if ((f1 == 0 && f2 == 0) || (k1 == 0 && k1s == 0))
        return;
    double strength = hypot(k1, k1s), angle = -0.5 * atan2(k1s, k1);
    double cosine = cos(angle), sine = sin(angle);
    double u = cosine * x + sine * y, v = -sine * x + cosine * y;
    double pu = cosine * px + sine * py, pv = -sine * px + cosine * py;
    double a = -direction * f1 * fabs(f1) * strength / (24 * r), b = f2 * strength / r;
    double ea = exp(a), ema = exp(-a);
    z -= energy_ratio / r * ((a * u + b * (1 + a / 2) * pu / r * ema) * pu / r - (a * v + b * (1 - a / 2) * pv / r * ea) * pv / r);
    u = u * ea + pu * b / r;
    v = v * ema - pv * b / r;
    pu *= ema;
    pv *= ea;
    x = cosine * u - sine * v;
    y = sine * u + cosine * v;
    px = cosine * pu - sine * pv;
    py = sine * pu + cosine * pv;
}

__device__ inline bool pass_fringe(
    pass_real_t& x,
    pass_real_t& px,
    pass_real_t& y,
    pass_real_t& py,
    pass_real_t& z,
    pass_real_t dp,
    pass_real_t inv_gamma,
    const pass_real_t* kn,
    const pass_real_t* ks,
    const pass_real_t* inv_fact,
    int order,
    int fringe,
    int frngmax,
    int kill,
    pass_real_t f1,
    pass_real_t f2,
    bool entrance
) {
#if !PASS_ENABLE_FRINGE
    return true;
#else
    if (fringe == 0 || (kill & (entrance ? 1 : 2)) || order < 1)
        return true;
    double r = 1.0 + (double)dp;
    double energy_ratio = sqrt(1.0 + (1.0 - (double)inv_gamma * inv_gamma) * dp * (2.0 + dp));
    double direction = entrance ? 1.0 : -1.0;
    if (!entrance && (fringe & 4))
        pass_qsad_fringe(x, px, y, py, z, r, energy_ratio, kn[1], ks[1], f1, f2, direction);
    if (fringe & 2) {
        double fx = 0, fy = 0, fxx = 0, fxy = 0, fyx = 0, fyy = 0, rx = 1, ix = 0;
        for (int j = 1; j <= order + 1 && j <= frngmax; ++j) {
            double drx = rx, dix = ix;
            rx = drx * x - dix * y;
            ix = drx * y + dix * x;
            double normal = (double)kn[j - 1] * inv_fact[j - 1], skew = (double)ks[j - 1] * inv_fact[j - 1];
            if (normal == 0 && skew == 0)
                continue;
            double scale = -direction / (4.0 * (j + 1)), nf = (j + 2.0) / j;
            double u = scale * (normal * rx - skew * ix), v = scale * (normal * ix + skew * rx);
            double du = scale * (normal * drx - skew * dix), dv = scale * (normal * dix + skew * drx);
            fx += u * x + nf * v * y;
            fy += u * y - nf * v * x;
            fxx += j * du * x + u + nf * y * j * dv;
            fxy += -j * dv * x + nf * v + nf * y * j * du;
            fyx += j * du * y - nf * v - nf * x * j * dv;
            fyy += -j * dv * y + u - nf * x * j * du;
        }
        double a = 1 - fxx / r, b = -fyx / r, c = -fxy / r, d = 1 - fyy / r, det = a * d - b * c;
        double pn = (d * px - b * py) / det, qn = (a * py - c * px) / det;
        x -= fx / r;
        y -= fy / r;
        px = pn;
        py = qn;
        z += energy_ratio * (pn * fx + qn * fy) / (r * r * r);
    }
    if (entrance && (fringe & 4))
        pass_qsad_fringe(x, px, y, py, z, r, energy_ratio, kn[1], ks[1], f1, f2, direction);
    return r > 0 && r * r > (double)px * px + (double)py * py && isfinite((double)x + px + y + py + z);
#endif
}
'''

_FRINGE_KERNEL = r'''
extern "C" __global__ void track_fringe(
    pass_real_t* x,
    pass_real_t* px,
    pass_real_t* y,
    pass_real_t* py,
    pass_real_t* z,
    const pass_real_t* dp,
    int* tag,
    float* lost_position,
    int* lost_turn,
    int start,
    int end,
    pass_real_t inv_gamma,
    pass_real_t position,
    int turn,
    const pass_real_t* kn,
    const pass_real_t* ks,
    const pass_real_t* inv_fact,
    int order,
    int fringe,
    int frngmax,
    int kill,
    pass_real_t f1,
    pass_real_t f2,
    int entrance
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i >= end || tag[i] <= 0)
        return;
    pass_real_t xi = x[i], pxi = px[i], yi = y[i], pyi = py[i], zi = z[i];
    if (!pass_fringe(xi, pxi, yi, pyi, zi, dp[i], inv_gamma, kn, ks, inv_fact, order, fringe, frngmax, kill, f1, f2, entrance)) {
        tag[i] = -1;
        lost_position[i] = position;
        lost_turn[i] = turn;
        return;
    }
    x[i] = xi;
    px[i] = pxi;
    y[i] = yi;
    py[i] = pyi;
    z[i] = zi;
}
'''


@lru_cache(maxsize=None)
def _get_fringe_kernel(dtype, device):
    import cupy as cp

    from PASS.commands.element.magnet_maps import CUDA_REAL_PREAMBLE

    with cp.cuda.Device(device):
        return cp.RawKernel(CUDA_REAL_PREAMBLE + FRINGE_CUDA + _FRINGE_KERNEL,
                            "track_fringe",
                            options=("--std=c++17", f"-DPASS_USE_FLOAT={int(np.dtype(dtype).itemsize == 4)}"))
