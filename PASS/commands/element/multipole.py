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
import cupy as cp
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
            self._track_multipole_cpu(beam, bunch)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)

    def execute_gpu(self, sim):
        raise NotImplementedError("GPU implementation of Multipole is not yet available")

    # ============================================================
    # Full multipole tracking (CPU)
    # ============================================================

    def _track_multipole_cpu(self, beam: Beam, bunch: BunchInfo):
        """Track particles through the multipole: thin lens or sliced DKD-exact."""

        beta0 = bunch.beta
        circum = bunch.circum
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

        # ---- Wrap z into [-C/2, C/2) ----
        c_half = 0.5 * circum
        over = (z > c_half).astype(np.int64)
        under = (z < -c_half).astype(np.int64)
        z += (under - over) * circum

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

        valid = (pz_sq > 0.0) & (tag > 0)
        tag[~valid] = -np.abs(tag[~valid])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)
        inv_pz = 1.0 / pz

        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
        bg = beta0 * gamma0
        beta = one_plus_delta_beta(one_plus_delta=one_plus_delta, bg=bg)

        L_mask = L * mask

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
        dpx_mul *= mask
        dpy_mul *= mask

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
