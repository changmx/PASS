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


@Command.register("octupole")
class Octupole(Command):
    """
    Octupole magnet with exact drift-kick-drift tracking.

    Tracking sequence:
      Thin lens (length=0):  single octupole kick
      Thick lens (length>0): N slices of drift-kick-drift-exact
        - uniform:   Drift(ds/2) → Kick(ds) → Drift(ds/2)  (2nd order symplectic)
        - yoshida4:  4th order Yoshida composition of DKD steps

      If k3l=0 and k3sl=0 (no field), thick lens degenerates to a pure drift.

    Octupole kick (integrated strength k3l_eff = k3 * ds):
      dpx = -chi * k3l_eff/6 * (x³ - 3xy²) + chi * k3sl_eff/6 * (3x²y - y³)
      dpy =  chi * k3l_eff/6 * (3x²y - y³) + chi * k3sl_eff/6 * (x³ - 3xy²)

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
            raise ValueError(f"The length of Octupole {self.cmd_name} is {self.length}, which should be >= 0")
        if self.length > const.eps:
            self.is_thick = True
        else:
            self.is_thick = False

        self.k3l = kwargs.get("k3l", 0.0)
        self.k3sl = kwargs.get("k3sl", 0.0)
        if self.is_thick:
            self.k3 = self.k3l / self.length
            self.k3s = self.k3sl / self.length
        else:
            self.k3 = 0.0
            self.k3s = 0.0
        if abs(self.k3l) < const.eps and abs(self.k3sl) < const.eps:
            logger.warning(f"Octupole {self.cmd_name} has zero integrated strength (k3l=0, k3sl=0). It will act as a pure drift.")
        if abs(self.k3l) > const.eps and abs(self.k3sl) > const.eps:
            logger.warning(f"Octupole {self.cmd_name} has both normal and skew components (k3l={self.k3l}, k3sl={self.k3sl}). It will act as a combined octupole.")

        self.num_slice = kwargs.get("num slices", 1)
        if self.num_slice < 1:
            logger.warning(f"The number of slices of {self.cmd_name} is {self.num_slice}, which should be >= 1. It has been changed to 1 now.")
            self.num_slice = 1

        self.integrator = kwargs.get("integrator", "adaptive")
        if self.integrator not in ["adaptive", "uniform", "yoshida4"]:
            raise ValueError(f"The integrator of Octupole {self.cmd_name} is {self.integrator}, which should be 'adaptive', 'uniform' or 'yoshida4'.")
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
                    f"IsThick={self.is_thick}, K3L={self.k3l:.6f}, K3SL={self.k3sl:.6f}, "
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
            self._track_octupole_cpu(beam, bunch, turn)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)

    def execute_gpu(self, sim):
        raise NotImplementedError("GPU implementation of Octupole is not yet available")

    # ============================================================
    # Full octupole tracking (CPU)
    # ============================================================

    def _track_octupole_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):
        """Track particles through the octupole: thin lens or sliced DKD-exact."""

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
            # Thin lens: single octupole kick
            self._octupole_kick_cpu(self.k3l, self.k3sl,
                                     x, px, y, py, tag, mask, chi)
            return

        # Thick lens
        if abs(self.k3l) < const.eps and abs(self.k3sl) < const.eps:
            # No field: pure drift
            self._drift_exact_cpu(self.length, x, px, y, py, z, dp, tag, mask, beta0)
        else:
            # Sliced DKD-exact
            ds = self.length / self.num_slice
            for _ in range(self.num_slice):
                if self.integrator == "uniform":
                    self._dkd_uniform_cpu(x, px, y, py, z, dp, tag, mask,
                                          ds, self.k3, self.k3s, chi, beta0)
                elif self.integrator == "yoshida4":
                    self._dkd_yoshida4_cpu(x, px, y, py, z, dp, tag, mask,
                                           ds, self.k3, self.k3s, chi, beta0)

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
                         ds, k3, k3s, chi, beta0):
        """
        One DKD slice (uniform/leapfrog, 2nd order symplectic):

          Drift(ds/2) → Kick(ds) → Drift(ds/2)
        """
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._octupole_kick_cpu(k3 * ds, k3s * ds,
                                 x, px, y, py, tag, mask, chi)
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)

    # ============================================================
    # Body: Drift-Kick-Drift exact (Yoshida 4th order)
    # ============================================================

    def _dkd_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask,
                          ds, k3, k3s, chi, beta0):
        """
        One Yoshida-4 slice:

          S4(ds) = S2(z1·ds) ∘ S2(z0·ds) ∘ S2(z1·ds)

        where S2 is the standard DKD (leapfrog) step.
        """
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z1, k3, k3s, chi, beta0)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z0, k3, k3s, chi, beta0)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z1, k3, k3s, chi, beta0)

    def _dkd_step_cpu(self, x, px, y, py, z, dp, tag, mask,
                      ds, k3, k3s, chi, beta0):
        """Single DKD step with given effective length ds (can be negative)."""
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._octupole_kick_cpu(k3 * ds, k3s * ds,
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
    # Octupole kick (thin lens)
    # ============================================================

    def _octupole_kick_cpu(self, k3l_eff, k3sl_eff,
                            x, px, y, py, tag, mask, chi):
        """
        Thin octupole kick with integrated strengths.

        dpx = -chi * k3l_eff/6 * (x³ - 3xy²) + chi * k3sl_eff/6 * (3x²y - y³)
        dpy =  chi * k3l_eff/6 * (3x²y - y³) + chi * k3sl_eff/6 * (x³ - 3xy²)

        For thin lens mode: k3l_eff = k3l, k3sl_eff = k3sl
        For DKD mode:       k3l_eff = k3 * ds, k3sl_eff = k3s * ds

        Note: (x+iy)³ = (x³ - 3xy²) + i(3x²y - y³)
              Re = x*(x² - 3y²),  Im = y*(3x² - y²)
        """
        if abs(k3l_eff) < const.eps and abs(k3sl_eff) < const.eps:
            return

        k3l_mask = k3l_eff * mask

        x2 = x * x
        y2 = y * y

        # (x+iy)³ expanded: Re = x³ - 3xy², Im = 3x²y - y³
        # Optimized: Re = x*(x² - 3y²), Im = y*(3x² - y²)
        re_c3 = x * (x2 - 3.0 * y2)   # x³ - 3xy²
        im_c3 = y * (3.0 * x2 - y2)   # 3x²y - y³

        # Normal octupole
        if abs(k3l_eff) > const.eps:
            chi_k3l = chi * k3l_mask / 6.0
            px -= chi_k3l * re_c3
            py += chi_k3l * im_c3

        # Skew octupole
        if abs(k3sl_eff) > const.eps:
            k3sl_mask = k3sl_eff * mask
            chi_k3sl = chi * k3sl_mask / 6.0
            px += chi_k3sl * im_c3
            py += chi_k3sl * re_c3


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
