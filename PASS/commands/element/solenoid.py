from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.aperture import check_aperture_cpu

import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Yoshida 4th-order coefficients
# ============================================================
_YOSHIDA_Z1 = 1.0 / (2.0 - 2.0**(1.0/3.0))   # ≈ 1.3512071919596
_YOSHIDA_Z0 = 1.0 - 2.0 * _YOSHIDA_Z1          # ≈ -1.7024143839193


@Command.register("solenoid")
class Solenoid(Command):
    """
    Solenoid magnet with exact nonlinear tracking.

    A solenoid produces a longitudinal magnetic field Bz that couples the
    horizontal and vertical planes via Larmor rotation. Unlike quadrupoles
    and dipoles, the solenoid Hamiltonian cannot be split into a free-drift
    part and a pure-position kick part, because the Larmor rotation term
    depends on both position and momentum simultaneously.

    PASS uses the exact solenoid map derived from the Larmor-frame Hamiltonian,
    equivalent to the Xsuite implementation (track_solenoid_single_particle).
    The map is exact (zero error for a uniform solenoid), with p_z computed
    per-particle including the Larmor-transformed momenta.

    Tracking sequence:
      Thin lens (length=0):  no effect (solenoid has no thin-lens limit)
      Thick lens (length>0, no multipoles):  single exact solenoid map
      Thick lens (length>0, with multipoles):  N slices of Sol-Kick-Sol
        - uniform:   Sol(ds/2) → Kick(ds) → Sol(ds/2)  (2nd order symplectic)
        - yoshida4:  4th order Yoshida composition of SKS steps

      If ks=0 (no solenoid field), thick lens degenerates to a pure drift
      (or DKD if multipoles are present).

    Solenoid exact map (Xsuite track_magnet_drift.h, drift_model=6):
      sk = ks / 2
      pk1 = px + sk * y      (Larmor-transformed px)
      pk2 = py - sk * x      (Larmor-transformed py)
      pz = sqrt((1+δ)² - pk1² - pk2²)
      θ  = sk * L / pz       (Larmor rotation angle, per-particle)
      cos_θ, sin_θ = cos(θ), sin(θ)
      si = sin_θ / sk         (effective drift length factor)

      Rotation + drift in Larmor frame:
        rps = [cos_θ*x + sin_θ*y,  cos_θ*px + sin_θ*py,
               cos_θ*y - sin_θ*x,  cos_θ*py - sin_θ*px]
        x'  = cos_θ*rps[0] + si*rps[1]
        px' = cos_θ*rps[1] - sk*sin_θ*rps[0]
        y'  = cos_θ*rps[2] + si*rps[3]
        py' = cos_θ*rps[3] - sk*sin_θ*rps[2]
        Δζ  = L * (1 - (1+δ)/(pz * rvv))

    Coordinate convention (PASS):
      x, px, y, py, z, dp(=δ)
      px = Px/P0,  py = Py/P0,  dp = (P-P0)/P0
      z  = s - β0·c·t  (ζ coordinate)
    """

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = kwargs["length (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]

        if self.length < 0.0:
            raise ValueError(f"The length of Solenoid {self.cmd_name} is {self.length}, which should be >= 0")
        if self.length > const.eps:
            self.is_thick = True
        else:
            self.is_thick = False

        # Solenoid strength
        self.ks = kwargs.get("ks", 0.0)
        if abs(self.ks) < const.eps and not self.is_thick:
            logger.warning(f"Solenoid {self.cmd_name} has zero length and zero ks. It will act as a marker.")
        if abs(self.ks) < const.eps and self.is_thick:
            logger.warning(f"Solenoid {self.cmd_name} has zero ks. It will act as a pure drift.")

        # Multipole components (optional, for solenoid + multipole overlay)
        knl_list = kwargs.get("kil", [])
        ksl_list = kwargs.get("kisl", [])

        if not isinstance(knl_list, (list, np.ndarray)):
            raise ValueError(f"KiL of {self.cmd_name} must be a list, but got {type(knl_list)}")
        if not isinstance(ksl_list, (list, np.ndarray)):
            raise ValueError(f"KiSL of {self.cmd_name} must be a list, but got {type(ksl_list)}")

        self.knl = np.array(knl_list, dtype=np.float64)
        self.ksl = np.array(ksl_list, dtype=np.float64)

        len_n = len(self.knl)
        len_s = len(self.ksl)

        if len_n == 0 and len_s == 0:
            self.has_multipoles = False
            self.order = -1
            self.knl = np.array([0.0])
            self.ksl = np.array([0.0])
        else:
            self.order = max(len_n, len_s) - 1
            if len_n > len_s:
                self.ksl = np.pad(self.ksl, (0, len_n - len_s), mode='constant')
            elif len_n < len_s:
                self.knl = np.pad(self.knl, (0, len_s - len_n), mode='constant')
            self.has_multipoles = not (np.all(np.abs(self.knl) < const.eps) and
                                       np.all(np.abs(self.ksl) < const.eps))

        # Thick lens: compute per-unit-length multipole strength
        if self.is_thick and self.has_multipoles:
            self.kn = self.knl / self.length
            self.ksp = self.ksl / self.length
        else:
            self.kn = np.zeros_like(self.knl)
            self.ksp = np.zeros_like(self.ksl)

        # Precompute inverse factorials for multipole kick
        if self.has_multipoles:
            self.inv_fact = np.ones(self.order + 1)
            for n in range(1, self.order + 1):
                self.inv_fact[n] = self.inv_fact[n - 1] / n
        else:
            self.inv_fact = np.array([1.0])

        self.num_slice = kwargs.get("num slices", 1)
        if self.num_slice < 1:
            logger.warning(f"The number of slices of {self.cmd_name} is {self.num_slice}, which should be >= 1. It has been changed to 1 now.")
            self.num_slice = 1

        self.integrator = kwargs.get("integrator", "adaptive")
        if self.integrator not in ["adaptive", "uniform", "yoshida4"]:
            raise ValueError(f"The integrator of Solenoid {self.cmd_name} is {self.integrator}, which should be 'adaptive', 'uniform' or 'yoshida4'.")
        if self.integrator == "adaptive":
            self.integrator = "uniform"

        self.aperture_type: str = kwargs.get("aperture type", "off").lower()
        self.aperture_value: list = kwargs.get("aperture value", [])
        if not isinstance(self.aperture_value, list):
            raise ValueError(f"Aperture value of {self.cmd_name} must be a list, but got {type(self.aperture_value)}")

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, Length={self.length:.4f}, "
                    f"IsThick={self.is_thick}, Ks={self.ks:.6f}, "
                    f"HasMultipoles={self.has_multipoles}, "
                    f"KnL={np.array2string(self.knl, precision=6)}, "
                    f"KsL={np.array2string(self.ksl, precision=6)}, "
                    f"NumSlice={self.num_slice:d}, Integrator={self.integrator:s}, "
                    f"ApertureType={self.aperture_type:s}, ApertureValue={self.aperture_value}")
        set_normal_logging()

    # ============================================================
    # Main execution
    # ============================================================

    def execute_cpu(self, sim):
        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        turn = sim.state.turn

        for i, bunch in enumerate(bunches):
            self._track_solenoid_cpu(beam, bunch, turn)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)

    def execute_gpu(self, sim):
        if not self.is_thick:
            mode = 0
        elif not self.has_multipoles:
            mode = 2 if abs(self.ks) < const.eps else 1
        else:
            mode = 3
        launch_solenoid(self, sim, mode)

    # ============================================================
    # Full solenoid tracking (CPU)
    # ============================================================

    def _track_solenoid_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):
        """Track particles through the solenoid."""

        beta0 = bunch.beta
        start = bunch.start_idx
        end = bunch.end_idx

        p = beam.particles
        x = p.x[start:end]
        px = p.px[start:end]
        y = p.y[start:end]
        py = p.py[start:end]
        z = p.z[start:end]
        dp = p.dp[start:end]
        tag = p.tag[start:end]

        alive_before = tag > 0

        chi = 1.0
        mask = (tag > 0).astype(np.float64)

        if not self.is_thick:
            # Thin lens: solenoid has no thin-lens limit, no effect
            return

        if not self.has_multipoles:
            # Pure solenoid: single exact map (zero error)
            if abs(self.ks) < const.eps:
                # ks=0: pure drift
                self._drift_exact_cpu(self.length, x, px, y, py, z, dp, tag, mask, beta0)
            else:
                self._solenoid_exact_cpu(self.length, self.ks,
                                         x, px, y, py, z, dp, tag, mask, beta0)
        else:
            # Solenoid + multipoles: Sol-Kick-Sol integrator
            ds = self.length / self.num_slice
            for _ in range(self.num_slice):
                if self.integrator == "uniform":
                    self._sks_uniform_cpu(x, px, y, py, z, dp, tag, mask,
                                          ds, self.ks, self.kn, self.ksp, chi, beta0)
                elif self.integrator == "yoshida4":
                    self._sks_yoshida4_cpu(x, px, y, py, z, dp, tag, mask,
                                           ds, self.ks, self.kn, self.ksp, chi, beta0)

        # ---- Update lost particle info ----
        newly_lost = alive_before & (tag < 0)
        if np.any(newly_lost):
            lost_position = p.lost_position[start:end]
            lost_turn = p.lost_turn[start:end]
            lost_position[newly_lost] = self.s
            lost_turn[newly_lost] = turn

    # ============================================================
    # Body: Sol-Kick-Sol (uniform integrator, 2nd order symplectic)
    # ============================================================

    def _sks_uniform_cpu(self, x, px, y, py, z, dp, tag, mask,
                         ds, ks, kn, ksp, chi, beta0):
        """
        One SKS slice (uniform/leapfrog, 2nd order symplectic):

          Sol(ds/2) → Kick(ds) → Sol(ds/2)
        """
        self._solenoid_exact_cpu(ds * 0.5, ks,
                                  x, px, y, py, z, dp, tag, mask, beta0)
        self._multipole_kick_cpu(kn * ds, ksp * ds,
                                  x, px, y, py, tag, mask, chi)
        self._solenoid_exact_cpu(ds * 0.5, ks,
                                  x, px, y, py, z, dp, tag, mask, beta0)

    # ============================================================
    # Body: Sol-Kick-Sol (Yoshida 4th order)
    # ============================================================

    def _sks_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask,
                          ds, ks, kn, ksp, chi, beta0):
        """
        One Yoshida-4 slice:

          S4(ds) = S2(z1·ds) ∘ S2(z0·ds) ∘ S2(z1·ds)

        where S2 is the standard SKS (leapfrog) step.
        """
        self._sks_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z1, ks, kn, ksp, chi, beta0)
        self._sks_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z0, ks, kn, ksp, chi, beta0)
        self._sks_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z1, ks, kn, ksp, chi, beta0)

    def _sks_step_cpu(self, x, px, y, py, z, dp, tag, mask,
                      ds, ks, kn, ksp, chi, beta0):
        """Single SKS step with given effective length ds (can be negative)."""
        self._solenoid_exact_cpu(ds * 0.5, ks,
                                  x, px, y, py, z, dp, tag, mask, beta0)
        self._multipole_kick_cpu(kn * ds, ksp * ds,
                                  x, px, y, py, tag, mask, chi)
        self._solenoid_exact_cpu(ds * 0.5, ks,
                                  x, px, y, py, z, dp, tag, mask, beta0)

    # ============================================================
    # Exact solenoid map (Xsuite track_solenoid_single_particle)
    # drift_model = 6, model = -2 (sol-kick-sol)
    # ============================================================

    def _solenoid_exact_cpu(self, L, ks,
                             x, px, y, py, z, dp, tag, mask, beta0):
        """
        Exact solenoid map: Larmor rotation + focusing in the solenoid field.

        sk = ks / 2
        pk1 = px + sk * y       (Larmor-transformed px, conserved)
        pk2 = py - sk * x       (Larmor-transformed py, conserved)
        pz = sqrt((1+δ)² - pk1² - pk2²)   (per-particle)
        θ  = sk * L / pz        (Larmor rotation angle, per-particle)

        The map rotates (x,y) by θ while applying the solenoid focusing
        (equivalent to a drift of length sin(θ)/sk in the Larmor frame).
        """
        if abs(L) < const.eps:
            return

        if abs(ks) < const.eps:
            # Degenerate: ks=0 → pure drift
            self._drift_exact_cpu(L, x, px, y, py, z, dp, tag, mask, beta0)
            return

        sk = ks * 0.5

        one_plus_delta = 1.0 + dp

        # Larmor-transformed momenta (conserved quantities)
        pk1 = px + sk * y
        pk2 = py - sk * x
        ptr2 = pk1 * pk1 + pk2 * pk2

        # Per-particle longitudinal momentum
        pz_sq = one_plus_delta**2 - ptr2
        valid = pz_sq > 0.0
        alive = tag > 0
        tag[alive & ~valid] = -np.abs(tag[alive & ~valid])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)

        # Larmor rotation angle (per-particle)
        theta = sk * L / pz
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        # si = sin(θ) / sk (effective drift length in Larmor frame)
        # Protected against sk→0 (already handled above, but keep safe)
        si = sin_th / sk

        # Rotation + drift
        rps0 = cos_th * x + sin_th * y
        rps1 = cos_th * px + sin_th * py
        rps2 = cos_th * y - sin_th * x
        rps3 = cos_th * py - sin_th * px

        new_x = cos_th * rps0 + si * rps1
        new_px = cos_th * rps1 - sk * sin_th * rps0
        new_y = cos_th * rps2 + si * rps3
        new_py = cos_th * rps3 - sk * sin_th * rps2

        # Longitudinal: rvv = beta / beta0
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
        bg = beta0 * gamma0
        beta = one_plus_delta_beta(one_plus_delta=one_plus_delta, bg=bg)
        rvv = beta / beta0
        rvv_safe = np.where(np.abs(rvv) < const.eps, const.eps, rvv)

        add_to_z = L * (1.0 - one_plus_delta / (pz * rvv_safe))

        active = (alive & valid).astype(mask.dtype, copy=False)
        x[:] = new_x * active + x * (1.0 - active)
        px[:] = new_px * active + px * (1.0 - active)
        y[:] = new_y * active + y * (1.0 - active)
        py[:] = new_py * active + py * (1.0 - active)
        z += add_to_z * active

    # ============================================================
    # Exact drift map (Table 1.1, map D), Eq. 1.86-1.88
    # ============================================================

    def _drift_exact_cpu(self, L, x, px, y, py, z, dp, tag, mask, beta0):
        """
        Exact drift: free propagation in a straight, field-free region.

        x  += (px / pz) * L
        y  += (py / pz) * L
        z  += L * (1 - (beta0/beta) * (1+dp) / pz)

        where pz = sqrt((1+dp)² - px² - py²)
              beta = (1+dp)*beta0*gamma0 / sqrt(1 + ((1+dp)*beta0*gamma0)²)
        """
        if abs(L) < const.eps:
            return

        one_plus_delta = 1.0 + dp
        pz_sq = one_plus_delta**2 - px**2 - py**2

        valid = pz_sq > 0.0
        alive = tag > 0
        tag[alive & ~valid] = -np.abs(tag[alive & ~valid])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)
        inv_pz = 1.0 / pz

        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
        bg = beta0 * gamma0
        beta = one_plus_delta_beta(one_plus_delta=one_plus_delta, bg=bg)

        L_mask = L * (alive & valid)

        x += L_mask * px * inv_pz
        y += L_mask * py * inv_pz
        z += L_mask * (1.0 - (beta0 / beta) * one_plus_delta * inv_pz)

    # ============================================================
    # Multipole kick (Horner recursion, vectorized)
    # ============================================================

    def _multipole_kick_cpu(self, knl_eff, ksl_eff,
                             x, px, y, py, tag, mask, chi):
        """
        General multipole kick with integrated strengths, using Horner
        nested evaluation (equivalent to Xsuite kick_simple_single_coordinates).

        dpx = -chi * Σ_n knl_eff[n]/n! * Re[(x+iy)^n]
        dpy =  chi * Σ_n ksl_eff[n]/n! * Im[(x+iy)^n]
        """
        if np.all(np.abs(knl_eff) < const.eps) and np.all(np.abs(ksl_eff) < const.eps):
            return

        order = len(knl_eff) - 1
        inv_fact = self.inv_fact

        index = order
        dpx_mul = chi * knl_eff[index] * inv_fact[index]
        dpy_mul = chi * ksl_eff[index] * inv_fact[index]

        while index > 0:
            zre = dpx_mul * x - dpy_mul * y
            zim = dpx_mul * y + dpy_mul * x
            index -= 1
            dpx_mul = chi * knl_eff[index] * inv_fact[index] + zre
            dpy_mul = chi * ksl_eff[index] * inv_fact[index] + zim

        active = (tag > 0).astype(mask.dtype, copy=False)
        dpx_mul *= active
        dpy_mul *= active

        px -= dpx_mul
        py += dpy_mul


# ============================================================
# Helper: compute beta from (1+delta) and beta0*gamma0
# ============================================================

def one_plus_delta_beta(one_plus_delta, bg):
    """
    Compute beta = v/c given (1+delta) and beta0*gamma0.

    From: P/P0 = 1+delta = beta*gamma / (beta0*gamma0)
    => beta*gamma = (1+delta) * beta0*gamma0
    => beta = (beta*gamma) / sqrt(1 + (beta*gamma)²)
    """
    bg_new = one_plus_delta * bg
    return bg_new / np.sqrt(1.0 + bg_new**2)
CUDA_REAL_PREAMBLE = f'''
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


SOLENOID_BODY = r'''
__device__ __forceinline__ bool sol_drift(
    pass_real_t& x, pass_real_t& px, pass_real_t& y, pass_real_t& py,
    pass_real_t& z, pass_real_t dp, int& tag, float* lost_position,
    int* lost_turn, int index, pass_real_t L, pass_real_t beta0,
    pass_real_t beta_gamma, pass_real_t inv_gamma,
    pass_real_t s_position, int turn)
{
    if (fabs(L) < PASS_EPS || tag <= 0) return tag > 0;
    pass_real_t opd = (pass_real_t)1 + dp;
    pass_real_t pz2 = opd * opd - px * px - py * py;
    if (!(pz2 > (pass_real_t)0)) {
        tag = -abs(tag); lost_position[index] = (float)s_position;
        lost_turn[index] = turn; return false;
    }
    pass_real_t invpz = (pass_real_t)1 / sqrt(pz2);
    pass_real_t bg = opd * beta_gamma;
    pass_real_t beta = bg / sqrt((pass_real_t)1 + bg * bg);
    x += L * px * invpz; y += L * py * invpz;
    z += L * ((pass_real_t)1 - beta0 / beta * opd * invpz);
    return true;
}

__device__ __forceinline__ bool sol_exact(
    pass_real_t& x, pass_real_t& px, pass_real_t& y, pass_real_t& py,
    pass_real_t& z, pass_real_t dp, int& tag, float* lost_position,
    int* lost_turn, int index, pass_real_t L, pass_real_t ks,
    pass_real_t beta0, pass_real_t beta_gamma, pass_real_t s_position,
    int turn)
{
    if (fabs(L) < PASS_EPS || tag <= 0) return tag > 0;
    pass_real_t sk = ks * (pass_real_t)0.5;
    if (fabs(sk) < PASS_EPS)
        return sol_drift(x, px, y, py, z, dp, tag, lost_position,
                         lost_turn, index, L, beta0, beta_gamma,
                         (pass_real_t)1 / sqrt((pass_real_t)1 + beta_gamma * beta_gamma),
                         s_position, turn);

    pass_real_t opd = (pass_real_t)1 + dp;
    pass_real_t pk1 = px + sk * y;
    pass_real_t pk2 = py - sk * x;
    pass_real_t pz2 = opd * opd - pk1 * pk1 - pk2 * pk2;
    if (!(pz2 > (pass_real_t)0)) {
        tag = -abs(tag); lost_position[index] = (float)s_position;
        lost_turn[index] = turn; return false;
    }
    pass_real_t pz = sqrt(pz2);
    pass_real_t theta = sk * L / pz;
    pass_real_t c = cos(theta), s = sin(theta), si = s / sk;
    pass_real_t r0 = c * x + s * y;
    pass_real_t r1 = c * px + s * py;
    pass_real_t r2 = c * y - s * x;
    pass_real_t r3 = c * py - s * px;
    pass_real_t xn = c * r0 + si * r1;
    pass_real_t pxn = c * r1 - sk * s * r0;
    pass_real_t yn = c * r2 + si * r3;
    pass_real_t pyn = c * r3 - sk * s * r2;
    pass_real_t bg = opd * beta_gamma;
    pass_real_t beta = bg / sqrt((pass_real_t)1 + bg * bg);
    x = xn; px = pxn; y = yn; py = pyn;
    z += L * ((pass_real_t)1 - opd / (pz * (beta / beta0)));
    return true;
}

__device__ __forceinline__ void sol_kick(
    pass_real_t& px, pass_real_t& py, pass_real_t x, pass_real_t y,
    const pass_real_t* __restrict__ knl,
    const pass_real_t* __restrict__ ksl,
    const pass_real_t* __restrict__ inv_fact, int order,
    pass_real_t scale)
{
    pass_real_t ar = knl[order] * inv_fact[order] * scale;
    pass_real_t ai = ksl[order] * inv_fact[order] * scale;
    for (int n = order; n > 0; --n) {
        pass_real_t re = ar * x - ai * y;
        pass_real_t im = ar * y + ai * x;
        ar = knl[n - 1] * inv_fact[n - 1] * scale + re;
        ai = ksl[n - 1] * inv_fact[n - 1] * scale + im;
    }
    px -= ar; py += ai;
}

__device__ __forceinline__ bool sol_sks_step(
    pass_real_t& x, pass_real_t& px, pass_real_t& y, pass_real_t& py,
    pass_real_t& z, pass_real_t dp, int& tag, float* lost_position,
    int* lost_turn, int index, pass_real_t ds, pass_real_t ks,
    pass_real_t beta0, pass_real_t beta_gamma, pass_real_t s_position,
    int turn, const pass_real_t* __restrict__ knl,
    const pass_real_t* __restrict__ ksl,
    const pass_real_t* __restrict__ inv_fact, int order)
{
    if (!sol_exact(x, px, y, py, z, dp, tag, lost_position, lost_turn,
                   index, ds * (pass_real_t)0.5, ks, beta0, beta_gamma,
                   s_position, turn)) return false;
    sol_kick(px, py, x, y, knl, ksl, inv_fact, order, ds);
    return sol_exact(x, px, y, py, z, dp, tag, lost_position, lost_turn,
                     index, ds * (pass_real_t)0.5, ks, beta0, beta_gamma,
                     s_position, turn);
}

extern "C" __global__
void track_solenoid(
    pass_real_t* __restrict__ x, pass_real_t* __restrict__ px,
    pass_real_t* __restrict__ y, pass_real_t* __restrict__ py,
    pass_real_t* __restrict__ z, const pass_real_t* __restrict__ dp,
    int* __restrict__ tag, float* __restrict__ lost_position,
    int* __restrict__ lost_turn, int start_index, int end_index,
    pass_real_t beta0, pass_real_t beta_gamma, pass_real_t L,
    pass_real_t ks, pass_real_t s_position, int turn,
    const pass_real_t* __restrict__ knl,
    const pass_real_t* __restrict__ ksl,
    const pass_real_t* __restrict__ inv_fact, int order, int num_slice,
    int integrator, int mode)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (index >= end_index || tag[index] <= 0) return;
    if (mode == 0) return; // zero-length solenoid is a marker

    pass_real_t xi=x[index], pxi=px[index], yi=y[index], pyi=py[index];
    pass_real_t zi=z[index], dpi=dp[index]; int ti=tag[index]; bool alive=true;
    if (mode == 1) {
        alive = sol_exact(xi, pxi, yi, pyi, zi, dpi, ti, lost_position,
                          lost_turn, index, L, ks, beta0, beta_gamma,
                          s_position, turn);
    } else if (mode == 2) {
        alive = sol_drift(xi, pxi, yi, pyi, zi, dpi, ti, lost_position,
                          lost_turn, index, L, beta0, beta_gamma,
                          (pass_real_t)1 / sqrt((pass_real_t)1 + beta_gamma * beta_gamma),
                          s_position, turn);
    } else {
        pass_real_t ds = L / (pass_real_t)num_slice;
        for (int slice=0; slice<num_slice && alive; ++slice) {
            pass_real_t d = (integrator == 0)
                ? ds : ds * (pass_real_t)1.3512071919596;
            alive = sol_sks_step(xi,pxi,yi,pyi,zi,dpi,ti,lost_position,
                                 lost_turn,index,d,ks,beta0,beta_gamma,
                                 s_position,turn,knl,ksl,inv_fact,order);
            if (integrator != 0 && alive) {
                alive = sol_sks_step(xi,pxi,yi,pyi,zi,dpi,ti,lost_position,
                    lost_turn,index,ds*(pass_real_t)-1.7024143839193,ks,
                    beta0,beta_gamma,s_position,turn,knl,ksl,inv_fact,order);
                if (alive) alive = sol_sks_step(xi,pxi,yi,pyi,zi,dpi,ti,
                    lost_position,lost_turn,index,ds*(pass_real_t)1.3512071919596,
                    ks,beta0,beta_gamma,s_position,turn,knl,ksl,inv_fact,order);
            }
        }
    }
    if (alive) { x[index]=xi; px[index]=pxi; y[index]=yi; py[index]=pyi; z[index]=zi; }
    tag[index]=ti;
}
'''

_kernels = {}


def launch_solenoid(element, sim, mode):
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU solenoid tracking requires the optional 'cuda' dependencies.") from exc
    beam = sim.beams[element.beam_id]
    p = beam.particles
    key = np.dtype(p.dtype)
    if key not in _kernels:
        _kernels[key] = cp.RawKernel(
            CUDA_REAL_PREAMBLE + SOLENOID_BODY, "track_solenoid",
            options=("--std=c++14", f"-DPASS_USE_FLOAT={int(key == np.dtype(np.float32))}"),
        )
    kernel = _kernels[key]
    cache = getattr(element, "_gpu_strength_cache", None)
    if cache is None:
        cache = {}
        element._gpu_strength_cache = cache
    if np.dtype(p.dtype) not in cache:
        cache[np.dtype(p.dtype)] = (
            cp.asarray(element.kn if element.has_multipoles else [0.0], dtype=p.dtype),
            cp.asarray(element.ksp if element.has_multipoles else [0.0], dtype=p.dtype),
            cp.asarray(element.inv_fact, dtype=p.dtype),
        )
    knl, ksl, inv = cache[np.dtype(p.dtype)]
    real = p.real; threads=256; turn=sim.state.turn
    for bunch in beam.bunches:
        n=bunch.end_idx-bunch.start_idx
        if n > 0:
            blocks = (n + threads - 1) // threads
            kernel((blocks,), (threads,),
                   (p.x,p.px,p.y,p.py,p.z,p.dp,p.tag,p.lost_position,p.lost_turn,
                    np.int32(bunch.start_idx),np.int32(bunch.end_idx),real(bunch.beta),
                    real(bunch.beta*bunch.gamma),real(element.length),real(element.ks),
                    real(element.s),np.int32(turn),knl,ksl,inv,
                    np.int32(len(knl)-1),np.int32(element.num_slice),
                    np.int32(0 if element.integrator == "uniform" else 1),np.int32(mode)))
        if n > 0:
            from PASS.utils.aperture import check_aperture_gpu
            check_aperture_gpu(beam,bunch,element.aperture_type,element.aperture_value,element.s,turn)
        if abs(element.length) >= const.eps:
            bunch.t0 += element.length/(bunch.beta*const.c)
