"""
PASS test framework: mock simulation objects + Xsuite pure-Python reference implementations.

This module provides:
1. MockBunch / MockBeam / MockSim — lightweight objects to call PASS elements directly
2. Xsuite pure-Python reference functions (ported from C headers)
3. Comparison utilities (max_abs_diff, max_rel_diff, assert_close)

Usage:
    from tests.support.framework import MockBunch, MockBeam, MockSim, make_particles, compare_arrays
"""
import numpy as np
import sys
import os

# Add PASS root to path
_PASS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PASS_ROOT not in sys.path:
    sys.path.insert(0, _PASS_ROOT)

from PASS.utils.constants import const

# ============================================================
# Mock simulation objects
# ============================================================

class MockBunch:
    """Lightweight BunchInfo replacement for testing."""
    def __init__(self, beta=0.999, gamma=None, circum=1000.0, sigma_z=0.01,
                 dp=1e-3, gamma_t=None, n_particles=100, t0=0.0,
                 m0=const.m_p_eV, qm_ratio=1.0, Ek=None,
                 harmonic_number=1, harmonic_id=0):
        self.beta = beta
        if gamma is None:
            gamma = 1.0 / np.sqrt(1.0 - beta**2)
        self.gamma = gamma
        self.circum = circum
        self.sigma_z = sigma_z
        self.dp = dp
        self.gamma_t = gamma_t if gamma_t is not None else gamma
        self.t0 = t0
        self.m0 = m0
        self.qm_ratio = qm_ratio

        # Compute brho
        p0_kg = gamma * (m0 * const.e / (const.c * const.c)) * beta * const.c
        self.brho = p0_kg / (qm_ratio * const.e)
        self.p0 = gamma * m0 * beta  # in eV/c

        if Ek is None:
            Ek = (gamma - 1.0) * m0
        self.Ek = Ek

        self.start_idx = 0
        self.end_idx = n_particles
        self.Nrp = n_particles
        self.Np = n_particles
        self.ratio = 1.0
        self.bunch_id = 0
        self.harmonic_number = harmonic_number
        self.harmonic_id = harmonic_id
        self.z_center = harmonic_id * circum / harmonic_number

    def t0_i(self):
        return self.t0 - self.z_center / (self.beta * const.c)


class MockBeam:
    """Lightweight Beam replacement for testing."""
    def __init__(self, n_particles, xp=np):
        from PASS.core.particle import ParticlePool
        self.particles = ParticlePool(n_particles, xp)
        self.bunches = [MockBunch(n_particles=n_particles)]


class MockSim:
    """Lightweight Simulation replacement for testing."""
    def __init__(self, beam, turn=0):
        self.beams = [beam]
        self.state = type('State', (), {'turn': turn})()


class MockState:
    def __init__(self, turn=0):
        self.turn = turn


def make_particles(n, x=None, px=None, y=None, py=None, z=None, dp=None, tag=None, xp=np):
    """Create a ParticlePool with given initial coordinates."""
    from PASS.core.particle import ParticlePool
    p = ParticlePool(n, xp)
    if x is not None: p.x[:] = x
    if px is not None: p.px[:] = px
    if y is not None: p.y[:] = y
    if py is not None: p.py[:] = py
    if z is not None: p.z[:] = z
    if dp is not None: p.dp[:] = dp
    if tag is not None: p.tag[:] = tag
    return p


def make_beam(n_particles, beta=0.999, circum=1000.0, **kw):
    """Create a MockBeam with n_particles and given bunch parameters."""
    beam = MockBeam(n_particles)
    beam.bunches = [MockBunch(beta=beta, circum=circum, n_particles=n_particles, **kw)]
    beam.particles = make_particles(n_particles)
    return beam


def make_sim(beam, turn=0):
    """Create a MockSim from a beam."""
    s = MockSim.__new__(MockSim)
    s.beams = [beam]
    s.state = MockState(turn)
    return s


# ============================================================
# Comparison utilities
# ============================================================

def max_abs_diff(a, b):
    """Maximum absolute difference between two arrays (only for alive particles)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float('inf')
    return float(np.max(np.abs(a - b)))


def max_rel_diff(a, b):
    """Maximum relative difference between two arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float('inf')
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom < 1e-30, 1e-30, denom)
    return float(np.max(np.abs(a - b) / denom))


def compare_particles(p1, p2, alive_mask=None, tol=1e-12, label=""):
    """Compare two ParticlePools, return list of (field, max_abs, max_rel, pass)."""
    results = []
    fields = ['x', 'px', 'y', 'py', 'z', 'dp']
    for f in fields:
        a = np.asarray(getattr(p1, f), dtype=np.float64)
        b = np.asarray(getattr(p2, f), dtype=np.float64)
        if alive_mask is not None:
            a = a[alive_mask]
            b = b[alive_mask]
        ma = max_abs_diff(a, b)
        mr = max_rel_diff(a, b)
        ok = ma < tol
        results.append((f, ma, mr, ok))
    return results


def print_comparison(results, label=""):
    """Print comparison results."""
    if label:
        print(f"  [{label}]")
    all_pass = True
    for f, ma, mr, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    {f:4s}: max_abs={ma:.3e}  max_rel={mr:.3e}  [{status}]")
    return all_pass


# ============================================================
# Xsuite pure-Python reference implementations
# Ported from xtrack/beam_elements/elements_src/*.h
# ============================================================

def ref_beta_from_delta(delta, beta0, gamma0=None):
    """Compute beta = v/c from delta and beta0.
    
    P/P0 = 1+delta = beta*gamma / (beta0*gamma0)
    => beta*gamma = (1+delta)*beta0*gamma0
    => beta = (beta*gamma) / sqrt(1 + (beta*gamma)^2)
    """
    if gamma0 is None:
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
    bg0 = beta0 * gamma0
    bg = (1.0 + delta) * bg0
    return bg / np.sqrt(1.0 + bg**2)


def ref_rvv(delta, beta0, gamma0=None):
    """rvv = beta / beta0"""
    return ref_beta_from_delta(delta, beta0, gamma0) / beta0


def ref_rv0v(delta, beta0, gamma0=None):
    """rv0v = 1/rvv = beta0 / beta"""
    return beta0 / ref_beta_from_delta(delta, beta0, gamma0)


def ref_exact_drift(x, px, y, py, z, dp, tag, L, beta0, gamma0=None):
    """Exact drift map (Xsuite track_drift.h / track_magnet_drift.h, drift_model=1).
    
    x  += (px / pz) * L
    y  += (py / pz) * L
    z  += L * (1 - (beta0/beta) * (1+dp) / pz)
    
    where pz = sqrt((1+dp)^2 - px^2 - py^2)
    """
    if abs(L) < const.eps:
        return x, px, y, py, z, dp, tag
    
    if gamma0 is None:
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
    
    one_plus_delta = 1.0 + dp
    pz_sq = one_plus_delta**2 - px**2 - py**2
    
    valid = (pz_sq > 0.0) & (tag > 0)
    tag_out = tag.copy()
    tag_out[~valid] = -np.abs(tag_out[~valid])
    
    pz_sq_safe = np.maximum(pz_sq, const.eps)
    pz = np.sqrt(pz_sq_safe)
    
    beta = ref_beta_from_delta(dp, beta0, gamma0)
    
    mask = (tag_out > 0).astype(np.float64)
    L_mask = L * mask
    
    x_new = x + L_mask * (px / pz)
    y_new = y + L_mask * (py / pz)
    z_new = z + L_mask * (1.0 - (beta0 / beta) * one_plus_delta / pz)
    
    return x_new, px, y_new, py, z_new, dp, tag_out


def ref_dkd_exact(x, px, y, py, z, dp, tag, L, kick_fn, beta0, gamma0=None,
                  integrator="uniform", num_slices=1):
    """DKD-exact reference: Drift(L/2) -> Kick -> Drift(L/2), sliced.
    
    integrator: "uniform" (2nd order leapfrog) or "yoshida4" (4th order Yoshida)
    kick_fn: function(x, px, y, py, z, dp, tag, L_kick) -> updated (px, py, z, ...)
    """
    YOSHIDA_Z1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    YOSHIDA_Z0 = 1.0 - 2.0 * YOSHIDA_Z1
    
    ds = L / num_slices
    
    def dkd_step(x, px, y, py, z, dp, tag, eff_ds):
        # Drift(eff_ds/2)
        x, px, y, py, z, dp, tag = ref_exact_drift(x, px, y, py, z, dp, tag, eff_ds * 0.5, beta0, gamma0)
        # Kick(eff_ds)
        x, px, y, py, z, dp, tag = kick_fn(x, px, y, py, z, dp, tag, eff_ds)
        # Drift(eff_ds/2)
        x, px, y, py, z, dp, tag = ref_exact_drift(x, px, y, py, z, dp, tag, eff_ds * 0.5, beta0, gamma0)
        return x, px, y, py, z, dp, tag
    
    for _ in range(num_slices):
        if integrator == "uniform":
            x, px, y, py, z, dp, tag = dkd_step(x, px, y, py, z, dp, tag, ds)
        elif integrator == "yoshida4":
            x, px, y, py, z, dp, tag = dkd_step(x, px, y, py, z, dp, tag, ds * YOSHIDA_Z1)
            x, px, y, py, z, dp, tag = dkd_step(x, px, y, py, z, dp, tag, ds * YOSHIDA_Z0)
            x, px, y, py, z, dp, tag = dkd_step(x, px, y, py, z, dp, tag, ds * YOSHIDA_Z1)
    
    return x, px, y, py, z, dp, tag


# ============================================================
# Xsuite multipole kick reference (track_magnet_kick.h)
# ============================================================

def ref_multipole_kick(x, px, y, py, z, dp, tag, knl_eff, ksl_eff, chi=1.0):
    """General multipole kick via Horner nested evaluation.
    
    Ported from Xsuite kick_simple_single_coordinates (track_magnet_kick.h).
    knl_eff, ksl_eff: arrays of integrated strengths (already scaled by ds for thick lens).
    
    dpx = -chi * sum_n knl[n]/n! * Re[(x+iy)^n]
    dpy =  chi * sum_n ksl[n]/n! * Im[(x+iy)^n]
    """
    knl_eff = np.asarray(knl_eff, dtype=np.float64)
    ksl_eff = np.asarray(ksl_eff, dtype=np.float64)
    
    if np.all(np.abs(knl_eff) < const.eps) and np.all(np.abs(ksl_eff) < const.eps):
        return x, px, y, py, z, dp, tag
    
    order = len(knl_eff) - 1
    inv_fact = np.ones(order + 1)
    for n in range(1, order + 1):
        inv_fact[n] = inv_fact[n - 1] / n
    
    mask = (tag > 0).astype(np.float64)
    
    index = order
    dpx_mul = chi * knl_eff[index] * inv_fact[index]
    dpy_mul = chi * ksl_eff[index] * inv_fact[index]
    
    while index > 0:
        zre = dpx_mul * x - dpy_mul * y
        zim = dpx_mul * y + dpy_mul * x
        index -= 1
        dpx_mul = chi * knl_eff[index] * inv_fact[index] + zre
        dpy_mul = chi * ksl_eff[index] * inv_fact[index] + zim
    
    dpx_mul *= mask
    dpy_mul *= mask
    
    px_new = px - dpx_mul
    py_new = py + dpy_mul
    
    return x, px_new, y, py_new, z, dp, tag


# ============================================================
# Xsuite dipole tracking reference (track_yrotation.h, track_wedge.h,
# track_dipole_fringe.h, track_magnet.h)
# ============================================================

def ref_yrotation(x, px, y, py, z, dp, tag, angle, beta0, gamma0=None):
    """YRotation: rotate reference frame by angle about y-axis.
    
    Ported from Xsuite track_yrotation.h.
    """
    if abs(angle) < const.eps:
        return x, px, y, py, z, dp, tag
    
    if gamma0 is None:
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
    
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)
    tan_angle = np.tan(angle)
    
    one_plus_delta = 1.0 + dp
    pz_sq = one_plus_delta**2 - px**2 - py**2
    
    valid = (pz_sq > 0.0) & (tag > 0)
    tag_out = tag.copy()
    tag_out[~valid] = -np.abs(tag_out[~valid])
    pz_sq_safe = np.maximum(pz_sq, const.eps)
    pz = np.sqrt(pz_sq_safe)
    
    ptt = 1.0 + tan_angle * px / pz
    ptt_safe = np.where(np.abs(ptt) < const.eps, const.eps, ptt)
    
    # time_fac = 1/beta0 + ptau = sqrt((1+delta)^2 + 1/(beta0^2*gamma0^2))
    time_fac = np.sqrt(one_plus_delta**2 + 1.0 / (beta0**2 * gamma0**2))
    
    mask = (tag_out > 0).astype(np.float64)
    
    x_new = x / (cos_angle * ptt_safe)
    px_new = cos_angle * px - sin_angle * pz
    y_new = y - tan_angle * x * py / (pz * ptt_safe)
    z_new = z + beta0 * tan_angle * x * time_fac / (pz * ptt_safe)
    
    x_out = x_new * mask + x * (1.0 - mask)
    px_out = px_new * mask + px * (1.0 - mask)
    y_out = y_new * mask + y * (1.0 - mask)
    z_out = z_new * mask + z * (1.0 - mask)
    
    return x_out, px_out, y_out, py, z_out, dp, tag_out


def ref_wedge(x, px, y, py, z, dp, tag, theta, k0, chi, beta0, gamma0=None):
    """Wedge map: rotate observation plane by theta in uniform dipole field.
    
    Ported from Xsuite track_wedge.h.
    """
    b1 = k0 * chi
    
    if abs(b1) < const.eps:
        return ref_yrotation(x, px, y, py, z, dp, tag, theta, beta0, gamma0)
    
    if gamma0 is None:
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
    
    rvv = ref_rvv(dp, beta0, gamma0)
    rv0v = 1.0 / rvv
    
    one_plus_delta = 1.0 + dp
    A = 1.0 / np.sqrt(one_plus_delta**2 - py**2)
    pz_sq = one_plus_delta**2 - px**2 - py**2
    
    valid = (pz_sq > 0.0) & (tag > 0)
    tag_out = tag.copy()
    tag_out[~valid] = -np.abs(tag_out[~valid])
    pz_sq_safe = np.maximum(pz_sq, const.eps)
    pz = np.sqrt(pz_sq_safe)
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    new_px = px * cos_t + (pz - b1 * x) * sin_t
    
    new_pz_sq = one_plus_delta**2 - new_px**2 - py**2
    new_pz_sq = np.maximum(new_pz_sq, const.eps)
    new_pz = np.sqrt(new_pz_sq)
    
    denom = new_pz + pz * cos_t - px * sin_t
    denom_safe = np.where(np.abs(denom) < const.eps, const.eps, denom)
    
    new_x = (x * cos_t
             + (x * px * np.sin(2.0 * theta)
                + sin_t**2 * (2.0 * x * pz - b1 * x**2)) / denom_safe)
    
    arg_px = np.clip(A * px, -1.0, 1.0)
    arg_new_px = np.clip(A * new_px, -1.0, 1.0)
    D = np.arcsin(arg_px) - np.arcsin(arg_new_px)
    
    b1_safe = b1 if abs(b1) > const.eps else const.eps
    delta_y = py * (theta + D) / b1_safe
    delta_ell = one_plus_delta * (theta + D) / b1_safe
    
    mask = (tag_out > 0).astype(np.float64)
    
    x_out = new_x * mask + x * (1.0 - mask)
    px_out = new_px * mask + px * (1.0 - mask)
    y_out = (y + delta_y) * mask + y * (1.0 - mask)
    z_out = (z - delta_ell / rvv) * mask + z * (1.0 - mask)
    
    return x_out, px_out, y_out, py, z_out, dp, tag_out


def ref_dipole_fringe(x, px, y, py, z, dp, tag, fint, hgap, k0, chi, beta0, gamma0=None):
    """Dipole fringe field map (PTC-compatible implementation).
    
    Ported from Xsuite track_dipole_fringe.h.
    """
    b0 = k0 * chi
    fh = hgap * fint
    fsad = 1.0 / (72.0 * fh) if fh > const.eps else 0.0
    k0w = b0
    inv_beta0 = 1.0 / beta0
    
    if gamma0 is None:
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
    
    one_plus_delta = 1.0 + dp
    dpp = one_plus_delta**2
    pz_sq = dpp - px**2 - py**2
    
    valid = (pz_sq > 0.0) & (tag > 0)
    tag_out = tag.copy()
    tag_out[~valid] = -np.abs(tag_out[~valid])
    pz_sq_safe = np.maximum(pz_sq, const.eps)
    pz = np.sqrt(pz_sq_safe)
    inv_pz = 1.0 / pz
    relp = 1.0 / np.sqrt(dpp)
    
    tfac = -np.sqrt(one_plus_delta**2 + 1.0 / (beta0**2 * gamma0**2))
    
    c2 = k0w * fh * 2.0
    c3 = k0w**2 * fsad * relp
    
    xp = px * inv_pz
    yp = py * inv_pz
    xyp = xp * yp
    yp2 = 1.0 + yp**2
    xp2 = xp**2
    inv_yp2 = 1.0 / yp2
    
    # PTC-compatible generating function: the fringe term is proportional
    # to pz (the slope derivatives below still use 1/pz).
    fi0 = np.arctan(xp * inv_yp2) - c2 * (1.0 + xp2 * (1.0 + yp2)) * pz
    cos_fi0 = np.cos(fi0)
    cos_fi0_safe = np.where(np.abs(cos_fi0) < const.eps, const.eps, cos_fi0)
    co2 = k0w / (cos_fi0_safe**2)
    co1 = co2 / (1.0 + (xp * inv_yp2)**2) * inv_yp2
    co3 = co2 * c2
    
    fi1 = co1 - co3 * 2.0 * xp * (1.0 + yp2) * pz
    fi2 = -2.0 * co1 * xyp * inv_yp2 - co3 * 2.0 * xp * xyp * pz
    fi3 = -co3 * (1.0 + xp2 * (1.0 + yp2))
    
    kx = fi1 * (1.0 + xp2) * inv_pz + fi2 * xyp * inv_pz - fi3 * xp
    ky = fi1 * xyp * inv_pz + fi2 * yp2 * inv_pz - fi3 * yp
    kz = fi1 * tfac * xp * (inv_pz**2) + fi2 * tfac * yp * (inv_pz**2) - fi3 * tfac * inv_pz
    
    discriminant = np.maximum(1.0 - 2.0 * ky * y, 0.0)
    new_y = 2.0 * y / (1.0 + np.sqrt(discriminant))
    
    new_x = x + 0.5 * kx * new_y**2
    new_py = py - 4.0 * c3 * new_y**3 - k0w * np.tan(fi0) * new_y
    new_z = z + beta0 * (0.5 * kz * new_y**2 + c3 * new_y**4 * (relp**2) * tfac)
    
    mask = (tag_out > 0).astype(np.float64)
    
    x_out = new_x * mask + x * (1.0 - mask)
    y_out = new_y * mask + y * (1.0 - mask)
    py_out = new_py * mask + py * (1.0 - mask)
    z_out = new_z * mask + z * (1.0 - mask)
    
    return x_out, px, y_out, py_out, z_out, dp, tag_out


def ref_dipole_kick(x, px, y, py, z, dp, tag, L, h, k0, chi, beta0, gamma0=None):
    """Dipole kick: curvature + main bend + weak focusing + path length.
    
    Ported from Xsuite track_magnet_kick.h.
    dpx += h*L*(1+delta) - chi*k0*L - chi*k0*h*x*L
    dzeta += -rv0v * h * L * x   where rv0v = beta0/beta
    """
    if abs(L) < const.eps:
        return x, px, y, py, z, dp, tag
    
    if gamma0 is None:
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
    
    one_plus_delta = 1.0 + dp
    mask = (tag > 0).astype(np.float64)
    L_mask = L * mask
    
    rv0v = ref_rv0v(dp, beta0, gamma0)
    
    px_new = px + L_mask * (h * one_plus_delta - chi * k0 - chi * k0 * h * x)
    z_new = z - L_mask * rv0v * h * x
    
    return x, px_new, y, py, z_new, dp, tag


def ref_dipole_edge_entry(x, px, y, py, z, dp, tag, e1, fint, hgap, k0, h, chi, beta0, gamma0=None):
    """Entry edge: YRotation(-e1) -> DipoleFringe -> Wedge(-e1, K0)"""
    has_angle = abs(e1) > const.eps
    has_fringe = abs(k0) > const.eps
    
    if not has_angle and not has_fringe:
        return x, px, y, py, z, dp, tag
    
    if has_angle:
        x, px, y, py, z, dp, tag = ref_yrotation(x, px, y, py, z, dp, tag, -e1, beta0, gamma0)
    if has_fringe:
        x, px, y, py, z, dp, tag = ref_dipole_fringe(x, px, y, py, z, dp, tag, fint, hgap, k0, chi, beta0, gamma0)
    if has_angle:
        x, px, y, py, z, dp, tag = ref_wedge(x, px, y, py, z, dp, tag, -e1, k0, chi, beta0, gamma0)
    
    return x, px, y, py, z, dp, tag


def ref_dipole_edge_exit(x, px, y, py, z, dp, tag, e2, fintx, hgap, k0, h, chi, beta0, gamma0=None):
    """Exit edge: Wedge(-e2, K0) -> DipoleFringe(-k0) -> YRotation(-e2)
    
    Xsuite track_magnet_edge.h:
    - DipoleFringe uses -k0 at exit
    - Wedge uses original k0 (NOT negated)
    """
    has_angle = abs(e2) > const.eps
    k0_fringe = -k0  # for DipoleFringe only
    has_fringe = abs(k0_fringe) > const.eps
    
    if not has_angle and not has_fringe:
        return x, px, y, py, z, dp, tag
    
    if has_angle:
        x, px, y, py, z, dp, tag = ref_wedge(x, px, y, py, z, dp, tag, -e2, k0, chi, beta0, gamma0)
    if has_fringe:
        x, px, y, py, z, dp, tag = ref_dipole_fringe(x, px, y, py, z, dp, tag, fintx, hgap, k0_fringe, chi, beta0, gamma0)
    if has_angle:
        x, px, y, py, z, dp, tag = ref_yrotation(x, px, y, py, z, dp, tag, -e2, beta0, gamma0)
    
    return x, px, y, py, z, dp, tag


def ref_sbend_track(x, px, y, py, z, dp, tag, length, k0l, e1, e2, hgap, fint, fintx,
                    num_slice, integrator, chi, beta0, gamma0=None):
    """Full SBend tracking: entry edge -> body DKD -> exit edge.
    
    Ported from Xsuite track_magnet.h.
    """
    if gamma0 is None:
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
    
    if length < const.eps:
        # Thin lens: just dipole kick
        mask = (tag > 0).astype(np.float64)
        px_new = px - chi * k0l * mask
        return x, px_new, y, py, z, dp, tag
    
    h = k0l / length
    k0 = k0l / length
    
    if fintx <= 0.0:
        fintx = fint
    
    # Entry edge
    x, px, y, py, z, dp, tag = ref_dipole_edge_entry(
        x, px, y, py, z, dp, tag, e1, fint, hgap, k0, h, chi, beta0, gamma0)
    
    # Body: DKD-exact
    def kick_fn(x, px, y, py, z, dp, tag, ds):
        return ref_dipole_kick(x, px, y, py, z, dp, tag, ds, h, k0, chi, beta0, gamma0)
    
    x, px, y, py, z, dp, tag = ref_dkd_exact(
        x, px, y, py, z, dp, tag, length, kick_fn, beta0, gamma0, integrator, num_slice)
    
    # Exit edge
    x, px, y, py, z, dp, tag = ref_dipole_edge_exit(
        x, px, y, py, z, dp, tag, e2, fintx, hgap, k0, h, chi, beta0, gamma0)
    
    # Wrap z
    circum = 1000.0  # dummy, wrapping doesn't affect comparison
    # Actually, z wrapping is done by the caller
    
    return x, px, y, py, z, dp, tag


# ============================================================
# Xsuite solenoid reference (track_legacy_solenoid.h)
# ============================================================

def ref_solenoid_exact(x, px, y, py, z, dp, tag, L, ks, beta0, gamma0=None):
    """Exact solenoid map (Xsuite track_legacy_solenoid.h / track_magnet_drift.h drift_model=6).
    
    sk = ks / 2
    pk1 = px + sk * y
    pk2 = py - sk * x
    pz = sqrt((1+delta)^2 - pk1^2 - pk2^2)
    theta = sk * L / pz
    """
    if abs(L) < const.eps:
        return x, px, y, py, z, dp, tag
    
    if abs(ks) < const.eps:
        return ref_exact_drift(x, px, y, py, z, dp, tag, L, beta0, gamma0)
    
    if gamma0 is None:
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
    
    sk = ks * 0.5
    one_plus_delta = 1.0 + dp
    
    pk1 = px + sk * y
    pk2 = py - sk * x
    ptr2 = pk1 * pk1 + pk2 * pk2
    
    pz_sq = one_plus_delta**2 - ptr2
    valid = (pz_sq > 0.0) & (tag > 0)
    tag_out = tag.copy()
    tag_out[~valid] = -np.abs(tag_out[~valid])
    pz_sq_safe = np.maximum(pz_sq, const.eps)
    pz = np.sqrt(pz_sq_safe)
    
    theta = sk * L / pz
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    si = sin_th / sk
    
    rps0 = cos_th * x + sin_th * y
    rps1 = cos_th * px + sin_th * py
    rps2 = cos_th * y - sin_th * x
    rps3 = cos_th * py - sin_th * px
    
    new_x = cos_th * rps0 + si * rps1
    new_px = cos_th * rps1 - sk * sin_th * rps0
    new_y = cos_th * rps2 + si * rps3
    new_py = cos_th * rps3 - sk * sin_th * rps2
    
    beta = ref_beta_from_delta(dp, beta0, gamma0)
    rvv = beta / beta0
    rvv_safe = np.where(np.abs(rvv) < const.eps, const.eps, rvv)
    
    add_to_z = L * (1.0 - one_plus_delta / (pz * rvv_safe))
    
    mask = (tag_out > 0).astype(np.float64)
    
    x_out = new_x * mask + x * (1.0 - mask)
    px_out = new_px * mask + px * (1.0 - mask)
    y_out = new_y * mask + y * (1.0 - mask)
    py_out = new_py * mask + py * (1.0 - mask)
    z_out = z + add_to_z * mask
    
    return x_out, px_out, y_out, py_out, z_out, dp, tag_out


# ============================================================
# Test particle generator
# ============================================================

def make_test_particles(n=100, seed=42,
                        x_range=(-0.01, 0.01), px_range=(-0.001, 0.001),
                        y_range=(-0.01, 0.01), py_range=(-0.001, 0.001),
                        z_range=(-0.05, 0.05), dp_range=(-0.05, 0.05),
                        kill_some=False):
    """Generate test particles with random coordinates in given ranges.
    
    Returns (x, px, y, py, z, dp, tag) arrays.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_range, n)
    px = rng.uniform(*px_range, n)
    y = rng.uniform(*y_range, n)
    py = rng.uniform(*py_range, n)
    z = rng.uniform(*z_range, n)
    dp = rng.uniform(*dp_range, n)
    tag = np.arange(1, n + 1, dtype=np.int32)
    
    if kill_some:
        # Kill 10% of particles
        n_kill = n // 10
        kill_idx = rng.choice(n, n_kill, replace=False)
        tag[kill_idx] = -np.abs(tag[kill_idx])
    
    return x, px, y, py, z, dp, tag


def make_grid_particles(x_vals, px_vals, y=0.0, py=0.0, z=0.0, dp=0.0):
    """Create particles on a grid of (x, px) values — useful for systematic testing."""
    xx, pxx = np.meshgrid(x_vals, px_vals)
    n = len(x_vals) * len(px_vals)
    x = xx.ravel()
    px = pxx.ravel()
    y_arr = np.full(n, y, dtype=np.float64)
    py_arr = np.full(n, py, dtype=np.float64)
    z_arr = np.full(n, z, dtype=np.float64)
    dp_arr = np.full(n, dp, dtype=np.float64)
    tag = np.arange(1, n + 1, dtype=np.int32)
    return x, px, y_arr, py_arr, z_arr, dp_arr, tag


# ============================================================
# Result reporting
# ============================================================

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.failures = []
    
    def check(self, condition, detail=""):
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(detail)
    
    def summary(self):
        total = self.passed + self.failed
        status = "ALL PASS" if self.failed == 0 else f"{self.failed} FAILED"
        print(f"\n{'='*60}")
        print(f"Test: {self.name}")
        print(f"  Passed: {self.passed}/{total}  {status}")
        if self.failures:
            print(f"  Failures:")
            for f in self.failures[:20]:
                print(f"    - {f}")
        print(f"{'='*60}")
        return self.failed == 0
