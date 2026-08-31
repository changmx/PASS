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


@Command.register("multipole")
class Multipole(Command):
    """
    General multipole magnet with exact drift-kick-drift tracking.

    Supports arbitrary order multipoles via knl/ksl arrays.
    The kick is computed using Horner nested evaluation, equivalent to
    the Xsuite track_magnet_kick.h implementation.

    Tracking sequence:
      Thin lens (length=0):  single multipole kick
      Thick lens (length>0): N slices of drift-kick-drift-exact
        - uniform:   Drift(ds/2) → Kick(ds) → Drift(ds/2)  (2nd order symplectic)
        - yoshida4:  4th order Yoshida composition of DKD steps

      If all knl/ksl components are zero, thick lens degenerates to a pure drift.

    Multipole kick (Horner recursion, integrated strength knl_eff = knl * ds):
      dpx = -chi * Σ_n knl_eff[n]/n! * Re[(x+iy)^n]
      dpy =  chi * Σ_n ksl_eff[n]/n! * Im[(x+iy)^n]

    Drift: exact drift (Table 1.1, map D), Eq. 1.86-1.88

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
            raise ValueError(f"The length of Multipole {self.cmd_name} is {self.length}, which should be >= 0")
        if self.length > const.eps:
            self.is_thick = True
        else:
            self.is_thick = False

        # Parse knl/ksl arrays
        knl_list = kwargs.get("kil", [])
        ksl_list = kwargs.get("kisl", [])

        if not isinstance(knl_list, (list, np.ndarray)):
            raise ValueError(f"KiL of {self.cmd_name} must be a list, but got {type(knl_list)}")
        if not isinstance(ksl_list, (list, np.ndarray)):
            raise ValueError(f"KiSL of {self.cmd_name} must be a list, but got {type(ksl_list)}")

        self.knl = np.array(knl_list, dtype=np.float64)
        self.ksl = np.array(ksl_list, dtype=np.float64)

        # Order = max(len(knl), len(ksl)) - 1; pad the shorter array with zeros
        len_n = len(self.knl)
        len_s = len(self.ksl)
        if len_n == 0 and len_s == 0:
            raise ValueError(f"Multipole {self.cmd_name} has empty KiL and KiSL. At least one component is required.")

        self.order = max(len_n, len_s) - 1

        if len_n > len_s:
            self.ksl = np.pad(self.ksl, (0, len_n - len_s), mode='constant')
        elif len_n < len_s:
            self.knl = np.pad(self.knl, (0, len_s - len_n), mode='constant')

        # Thick lens: compute per-unit-length strength
        if self.is_thick:
            self.kn = self.knl / self.length
            self.ks = self.ksl / self.length
        else:
            self.kn = np.zeros_like(self.knl)
            self.ks = np.zeros_like(self.ksl)

        # Check for all-zero strengths
        all_zero = np.all(np.abs(self.knl) < const.eps) and np.all(np.abs(self.ksl) < const.eps)
        if all_zero:
            logger.warning(f"Multipole {self.cmd_name} has zero integrated strength (all knl/ksl are zero). It will act as a pure drift.")

        # Precompute inverse factorials: inv_fact[n] = 1/n!
        self.inv_fact = np.ones(self.order + 1)
        for n in range(1, self.order + 1):
            self.inv_fact[n] = self.inv_fact[n - 1] / n

        self.num_slice = kwargs.get("num slices", 1)
        if self.num_slice < 1:
            logger.warning(f"The number of slices of {self.cmd_name} is {self.num_slice}, which should be >= 1. It has been changed to 1 now.")
            self.num_slice = 1

        self.integrator = kwargs.get("integrator", "adaptive")
        if self.integrator not in ["adaptive", "uniform", "yoshida4"]:
            raise ValueError(f"The integrator of Multipole {self.cmd_name} is {self.integrator}, which should be 'adaptive', 'uniform' or 'yoshida4'.")
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
                    f"IsThick={self.is_thick}, Order={self.order:d}, "
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
            self._track_multipole_cpu(beam, bunch, turn)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)
        return True

    def execute_gpu(self, sim):
        all_zero = (np.all(np.abs(self.knl) < const.eps) and
                    np.all(np.abs(self.ksl) < const.eps))
        mode = 0 if not self.is_thick else (2 if all_zero else 1)
        launch_multipole(self, sim, self.knl if not self.is_thick else self.kn,
                         self.ksl if not self.is_thick else self.ks,
                         self.inv_fact, mode)
        return True

    # ============================================================
    # Full multipole tracking (CPU)
    # ============================================================

    def _track_multipole_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):
        """Track particles through the multipole: thin lens or sliced DKD-exact."""

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

        # chi = q/q0 * m0/m  (for same-species beam, chi = 1)
        chi = 1.0

        # mask for alive particles
        mask = (tag > 0).astype(np.float64)

        if not self.is_thick:
            # Thin lens: single multipole kick
            self._multipole_kick_cpu(self.knl, self.ksl,
                                      x, px, y, py, tag, mask, chi)
            return

        # Thick lens
        all_zero = np.all(np.abs(self.knl) < const.eps) and np.all(np.abs(self.ksl) < const.eps)
        if all_zero:
            # No field: pure drift
            self._drift_exact_cpu(self.length, x, px, y, py, z, dp, tag, mask, beta0)
        else:
            # Sliced DKD-exact
            ds = self.length / self.num_slice
            for _ in range(self.num_slice):
                if self.integrator == "uniform":
                    self._dkd_uniform_cpu(x, px, y, py, z, dp, tag, mask,
                                          ds, self.kn, self.ks, chi, beta0)
                elif self.integrator == "yoshida4":
                    self._dkd_yoshida4_cpu(x, px, y, py, z, dp, tag, mask,
                                           ds, self.kn, self.ks, chi, beta0)

        # ---- Update lost particle info ----
        newly_lost = alive_before & (tag < 0)
        if np.any(newly_lost):
            lost_position = p.lost_position[start:end]
            lost_turn = p.lost_turn[start:end]
            lost_position[newly_lost] = self.s
            lost_turn[newly_lost] = turn

    # ============================================================
    # Body: Drift-Kick-Drift exact (uniform integrator)
    # ============================================================

    def _dkd_uniform_cpu(self, x, px, y, py, z, dp, tag, mask,
                         ds, kn, ks, chi, beta0):
        """
        One DKD slice (uniform/leapfrog, 2nd order symplectic):

          Drift(ds/2) → Kick(ds) → Drift(ds/2)
        """
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._multipole_kick_cpu(kn * ds, ks * ds,
                                  x, px, y, py, tag, mask, chi)
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)

    # ============================================================
    # Body: Drift-Kick-Drift exact (Yoshida 4th order)
    # ============================================================

    def _dkd_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask,
                          ds, kn, ks, chi, beta0):
        """
        One Yoshida-4 slice:

          S4(ds) = S2(z1·ds) ∘ S2(z0·ds) ∘ S2(z1·ds)

        where S2 is the standard DKD (leapfrog) step.
        """
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z1, kn, ks, chi, beta0)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z0, kn, ks, chi, beta0)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z1, kn, ks, chi, beta0)

    def _dkd_step_cpu(self, x, px, y, py, z, dp, tag, mask,
                      ds, kn, ks, chi, beta0):
        """Single DKD step with given effective length ds (can be negative)."""
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._multipole_kick_cpu(kn * ds, ks * ds,
                                  x, px, y, py, tag, mask, chi)
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)

    # ============================================================
    # Exact drift map (Table 1.1, map D)
    # Eq. 1.86-1.88
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

        # A particle that becomes invalid at this drift exits immediately;
        # do not transport it with the stale entry mask.
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

        For thin lens mode: knl_eff = knl, ksl_eff = ksl
        For DKD mode:       knl_eff = kn * ds, ksl_eff = ks * ds

        Horner recursion (from highest order down):
          index = order
          dpx_mul = chi * knl[order] / order!
          dpy_mul = chi * ksl[order] / order!
          while index > 0:
              zre = dpx_mul * x - dpy_mul * y   # Re[(dpx+i*dpy)*(x+iy)]
              zim = dpx_mul * y + dpy_mul * x   # Im[(dpx+i*dpy)*(x+iy)]
              index -= 1
              dpx_mul = chi * knl[index] / index! + zre
              dpy_mul = chi * ksl[index] / index! + zim
          dpx = -dpx_mul
          dpy = +dpy_mul
        """
        # Skip if all strengths are zero
        if np.all(np.abs(knl_eff) < const.eps) and np.all(np.abs(ksl_eff) < const.eps):
            return

        order = len(knl_eff) - 1
        inv_fact = self.inv_fact  # precomputed 1/n! array

        # Horner recursion (vectorized over particles)
        index = order
        dpx_mul = chi * knl_eff[index] * inv_fact[index]  # scalar
        dpy_mul = chi * ksl_eff[index] * inv_fact[index]  # scalar

        while index > 0:
            zre = dpx_mul * x - dpy_mul * y   # Re[(dpx+i*dpy)*(x+iy)]  (per-particle)
            zim = dpx_mul * y + dpy_mul * x   # Im[(dpx+i*dpy)*(x+iy)]  (per-particle)
            index -= 1
            dpx_mul = chi * knl_eff[index] * inv_fact[index] + zre  # scalar + array
            dpy_mul = chi * ksl_eff[index] * inv_fact[index] + zim  # scalar + array

        # Apply mask (zero out dead particles)
        active = (tag > 0).astype(mask.dtype, copy=False)
        dpx_mul *= active
        dpy_mul *= active

        px -= dpx_mul   # sign flip on px only (rad convention)
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


MULTIPOLE_KERNEL_BODY = r'''
__device__ __forceinline__ bool pass_drift(
    pass_real_t& x, pass_real_t& px, pass_real_t& y, pass_real_t& py,
    pass_real_t& z, pass_real_t dp, int& tag, float* lost_position,
    int* lost_turn, int index, pass_real_t L, pass_real_t beta_gamma,
    pass_real_t inv_gamma, pass_real_t s_position, int turn)
{
    // A tolerance matches the CPU const.eps guard and avoids exact floating-point
    // equality.  ``L`` is an element/slice length, so sub-epsilon transport is
    // intentionally treated as a no-op.
    if (fabs(L) < PASS_EPS || tag <= 0) return tag > 0;
    pass_real_t one_plus_delta = (pass_real_t)1 + dp;
    pass_real_t pz_sq = one_plus_delta * one_plus_delta - px * px - py * py;
    if (!(pz_sq > (pass_real_t)0)) {
        tag = -abs(tag);
        lost_position[index] = (float)s_position;
        lost_turn[index] = turn;
        return false;
    }
    pass_real_t inv_pz = (pass_real_t)1 / sqrt(pz_sq);
    pass_real_t bg = one_plus_delta * beta_gamma;
    pass_real_t dzeta_factor = sqrt((pass_real_t)1 + bg * bg) * inv_pz * inv_gamma;
    x += L * px * inv_pz;
    y += L * py * inv_pz;
    z += L * ((pass_real_t)1 - dzeta_factor);
    return true;
}

__device__ __forceinline__ void pass_kick(
    pass_real_t& px, pass_real_t& py, pass_real_t x, pass_real_t y,
    const pass_real_t* __restrict__ knl,
    const pass_real_t* __restrict__ ksl,
    const pass_real_t* __restrict__ inv_fact,
    int order, pass_real_t scale)
{
    pass_real_t dpx_mul = knl[order] * inv_fact[order] * scale;
    pass_real_t dpy_mul = ksl[order] * inv_fact[order] * scale;
    for (int n = order; n > 0; --n) {
        pass_real_t zre = dpx_mul * x - dpy_mul * y;
        pass_real_t zim = dpx_mul * y + dpy_mul * x;
        dpx_mul = knl[n - 1] * inv_fact[n - 1] * scale + zre;
        dpy_mul = ksl[n - 1] * inv_fact[n - 1] * scale + zim;
    }
    px -= dpx_mul;
    py += dpy_mul;
}

__device__ __forceinline__ bool pass_dkd_step(
    pass_real_t& x, pass_real_t& px, pass_real_t& y, pass_real_t& py,
    pass_real_t& z, pass_real_t dp, int& tag, float* lost_position,
    int* lost_turn, int index, pass_real_t ds, pass_real_t beta_gamma,
    pass_real_t inv_gamma, pass_real_t s_position, int turn,
    const pass_real_t* __restrict__ knl,
    const pass_real_t* __restrict__ ksl,
    const pass_real_t* __restrict__ inv_fact, int order)
{
    if (!pass_drift(x, px, y, py, z, dp, tag, lost_position, lost_turn,
                    index, ds * (pass_real_t)0.5, beta_gamma, inv_gamma,
                    s_position, turn)) return false;
    pass_kick(px, py, x, y, knl, ksl, inv_fact, order, ds);
    return pass_drift(x, px, y, py, z, dp, tag, lost_position, lost_turn,
                      index, ds * (pass_real_t)0.5, beta_gamma, inv_gamma,
                      s_position, turn);
}

// mode: 0 = thin integrated kick, 1 = thick sliced DKD, 2 = pure drift.
// integrator: 0 = uniform second-order DKD, 1 = Yoshida fourth-order DKD.
extern "C" __global__
void track_multipole_dkd(
    pass_real_t* __restrict__ x, pass_real_t* __restrict__ px,
    pass_real_t* __restrict__ y, pass_real_t* __restrict__ py,
    pass_real_t* __restrict__ z, const pass_real_t* __restrict__ dp,
    int* __restrict__ tag, float* __restrict__ lost_position,
    int* __restrict__ lost_turn, int start_index, int end_index,
    pass_real_t beta_gamma, pass_real_t inv_gamma, pass_real_t L,
    pass_real_t s_position, int turn, const pass_real_t* __restrict__ knl,
    const pass_real_t* __restrict__ ksl,
    const pass_real_t* __restrict__ inv_fact, int order, int num_slice,
    int integrator, int mode)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (index >= end_index || tag[index] <= 0) return;

    pass_real_t xi = x[index], pxi = px[index];
    pass_real_t yi = y[index], pyi = py[index];
    pass_real_t zi = z[index], dpi = dp[index];
    int ti = tag[index];
    bool alive = true;

    if (mode == 0) {
        pass_kick(pxi, pyi, xi, yi, knl, ksl, inv_fact, order,
                  (pass_real_t)1);
    } else if (mode == 2) {
        alive = pass_drift(xi, pxi, yi, pyi, zi, dpi, ti,
                           lost_position, lost_turn, index, L, beta_gamma,
                           inv_gamma, s_position, turn);
    } else {
        pass_real_t ds = L / (pass_real_t)num_slice;
        for (int slice = 0; slice < num_slice && alive; ++slice) {
            if (integrator == 0) {
                alive = pass_dkd_step(xi, pxi, yi, pyi, zi, dpi, ti,
                                      lost_position, lost_turn, index, ds,
                                      beta_gamma, inv_gamma, s_position, turn,
                                      knl, ksl, inv_fact, order);
            } else {
                alive = pass_dkd_step(xi, pxi, yi, pyi, zi, dpi, ti,
                                      lost_position, lost_turn, index,
                                      ds * (pass_real_t)1.3512071919596,
                                      beta_gamma, inv_gamma, s_position, turn,
                                      knl, ksl, inv_fact, order);
                if (alive) alive = pass_dkd_step(
                    xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn,
                    index, ds * (pass_real_t)-1.7024143839193, beta_gamma,
                    inv_gamma, s_position, turn, knl, ksl, inv_fact, order);
                if (alive) alive = pass_dkd_step(
                    xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn,
                    index, ds * (pass_real_t)1.3512071919596, beta_gamma,
                    inv_gamma, s_position, turn, knl, ksl, inv_fact, order);
            }
        }
    }

    if (alive) {
        x[index] = xi; px[index] = pxi;
        y[index] = yi; py[index] = pyi; z[index] = zi;
    }
    tag[index] = ti;
}
'''

_kernels = {}


def get_multipole_kernel(dtype):
    """Return a dtype-specialized raw kernel, compiled lazily."""
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "GPU multipole tracking requires the optional 'cuda' dependencies."
        ) from exc
    key = np.dtype(dtype)
    if key not in _kernels:
        _kernels[key] = cp.RawKernel(
            CUDA_REAL_PREAMBLE + MULTIPOLE_KERNEL_BODY,
            "track_multipole_dkd",
            options=("--std=c++14", f"-DPASS_USE_FLOAT={int(key == np.dtype(np.float32))}"),
        )
    return _kernels[key]


def launch_multipole(element, sim, knl, ksl, inv_fact, mode):
    """Launch the fused map for every bunch of an element.

    ``mode`` is 0 for a thin kick, 1 for a thick sliced DKD map, and 2 for a
    zero-strength drift.  The kernel's ``integrator`` argument is 0 for the
    uniform second-order map and 1 for Yoshida-4.
    """
    import cupy as cp

    beam = sim.beams[element.beam_id]
    turn = sim.state.turn
    p = beam.particles
    real = p.real
    cache = getattr(element, "_gpu_strength_cache", None)
    if cache is None:
        cache = {}
        element._gpu_strength_cache = cache
    key = np.dtype(p.dtype)
    if key not in cache:
        cache[key] = (cp.asarray(knl, dtype=p.dtype),
                      cp.asarray(ksl, dtype=p.dtype),
                      cp.asarray(inv_fact, dtype=p.dtype))
    knl_gpu, ksl_gpu, inv_gpu = cache[key]
    kernel = get_multipole_kernel(p.dtype)
    threads = 256
    for bunch in beam.bunches:
        start, end = bunch.start_idx, bunch.end_idx
        n = end - start
        if n > 0:
            blocks = (n + threads - 1) // threads
            kernel((blocks,), (threads,),
                   (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag,
                    p.lost_position, p.lost_turn,
                    np.int32(start), np.int32(end),
                    real(bunch.beta * bunch.gamma), real(1.0 / bunch.gamma),
                    real(element.length), real(element.s), np.int32(turn),
                    knl_gpu, ksl_gpu, inv_gpu, np.int32(len(knl) - 1),
                    np.int32(element.num_slice),
                    np.int32(0 if element.integrator == "uniform" else 1),
                    np.int32(mode)))
        if n > 0:
            from PASS.utils.aperture import check_aperture_gpu
            check_aperture_gpu(beam, bunch, element.aperture_type,
                               element.aperture_value, element.s, turn)
        if abs(element.length) >= const.eps:
            bunch.t0 += element.length / (bunch.beta * const.c)
