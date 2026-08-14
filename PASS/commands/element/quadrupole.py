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


@Command.register("quadrupole")
class Quadrupole(Command):
    """
    Quadrupole magnet with multiple tracking models.

    Tracking sequence:
      Thin lens (length=0):  single quadrupole kick
      Thick lens (length>0): model-dependent

    Models (self.model):
      'drift-kick-drift-exact' [default]:
          N slices of drift-kick-drift with exact drift.
          - uniform:   Drift(ds/2) → Kick(ds) → Drift(ds/2)  (2nd order symplectic)
          - yoshida4:  4th order Yoshida composition of DKD steps
          Preserves full nonlinear kinematics (exact pz).
          Linear chromaticity is approximate (O(1/N^2) splitting error).

      'mat-kick-mat':
          Exact linear transport matrix with chromaticity.
          k1 and k1s are diagonalized via rotation into K_eff,
          then the exact linear matrix M(L, K_eff*chi/(1+delta)) is applied.
          - If k1s != 0: rotation by theta = 0.5*arctan2(k1s, k1) diagonalizes
            the quadrupole into focusing/defocusing planes.
          - theta is delta-independent (k1 and k1s scale identically with delta).
          Linear chromaticity and R56 are exact.
          Nonlinear kinematics (pz higher-order terms) are not included.
          For pure k1 + k1s (no higher-order multipoles), this is a single
          matrix multiplication — no kick needed.

    Quadrupole kick (integrated strength k1l_eff = k1 * ds):
      dpx = -chi * k1l_eff * x + chi * k1sl_eff * y
      dpy =  chi * k1l_eff * y + chi * k1sl_eff * x

    Drift: exact drift (Table 1.1, map D), Eq. 1.86-1.88

    Coordinate convention (PASS):
      x, px, y, py, z, dp(=delta)
      px = Px/P0,  py = Py/P0,  dp = (P-P0)/P0
      z  = s - beta0*c*t  (zeta coordinate)
    """

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = kwargs["length (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]

        if self.length < 0.0:
            raise ValueError(f"The length of Quadrupole {self.cmd_name} is {self.length}, which should be >= 0")
        if self.length > const.eps:
            self.is_thick = True
        else:
            self.is_thick = False

        self.k1l = kwargs.get("k1l", 0.0)
        self.k1sl = kwargs.get("k1sl", 0.0)
        if self.is_thick:
            self.k1 = self.k1l / self.length
            self.k1s = self.k1sl / self.length
        else:
            self.k1 = 0.0
            self.k1s = 0.0
        if abs(self.k1l) < const.eps and abs(self.k1sl) < const.eps:
            logger.warning(f"Quadrupole {self.cmd_name} has zero integrated strength (k1l=0, k1sl=0). It will act as a pure drift.")
        if abs(self.k1l) > const.eps and abs(self.k1sl) > const.eps:
            logger.warning(f"Quadrupole {self.cmd_name} has both normal and skew components (k1l={self.k1l}, k1sl={self.k1sl}). It will act as a combined quadrupole.")

        # ---- Model selection ----
        self.model = kwargs.get("model", "adaptive")
        if self.model not in ["adaptive", "drift-kick-drift-exact", "mat-kick-mat"]:
            raise ValueError(f"The model of Quadrupole {self.cmd_name} is {self.model}, which should be 'adaptive', 'drift-kick-drift-exact' or 'mat-kick-mat'.")
        if self.model == "adaptive":
            self.model = "mat-kick-mat"

        self.num_slice = kwargs.get("num slices", 1)
        if self.num_slice < 1:
            logger.warning(f"The number of slices of {self.cmd_name} is {self.num_slice}, which should be >= 1. It has been changed to 1 now.")
            self.num_slice = 1

        self.integrator = kwargs.get("integrator", "adaptive")
        if self.integrator not in ["adaptive", "uniform", "yoshida4"]:
            raise ValueError(f"The integrator of Quadrupole {self.cmd_name} is {self.integrator}, which should be 'adaptive', 'uniform' or 'yoshida4'.")
        if self.integrator == "adaptive":
            self.integrator = "uniform"

        # ---- MKM precomputation: rotation diagonalization ----
        # theta = 0.5 * arctan2(k1s, k1) is delta-independent
        # (k1 and k1s both scale by chi/(1+delta), so the ratio is unchanged)
        if abs(self.k1s) > const.eps:
            self.is_skew = True
            theta = 0.5 * np.arctan2(-self.k1s, self.k1)
            self.cos_theta = np.cos(theta)
            self.sin_theta = np.sin(theta)
            self.k_eff_base = np.sqrt(self.k1**2 + self.k1s**2)
        else:
            self.is_skew = False
            self.cos_theta = 1.0
            self.sin_theta = 0.0
            self.k_eff_base = self.k1

        self.aperture_type: str = kwargs.get("aperture type", "off").lower()
        self.aperture_value: list = kwargs.get("aperture value", [])
        if not isinstance(self.aperture_value, list):
            raise ValueError(f"Aperture value of {self.cmd_name} must be a list, but got {type(self.aperture_value)}")

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, Length={self.length:.4f}, "
                    f"IsThick={self.is_thick}, K1L={self.k1l:.6f}, K1SL={self.k1sl:.6f}, "
                    f"NumSlice={self.num_slice:d}, Model={self.model:s}, Integrator={self.integrator:s}, "
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
            self._track_quadrupole_cpu(beam, bunch, turn)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)

    def execute_gpu(self, sim):
        raise NotImplementedError("GPU implementation of Quadrupole is not yet available")

    # ============================================================
    # Full quadrupole tracking (CPU)
    # ============================================================

    def _track_quadrupole_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):
        """Track particles through the quadrupole: thin lens or thick lens."""

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
            # Thin lens: single quadrupole kick
            self._quadrupole_kick_cpu(self.k1l, self.k1sl,
                                      x, px, y, py, tag, mask, chi)
            return

        # Thick lens
        if abs(self.k1l) < const.eps and abs(self.k1sl) < const.eps:
            # No field: pure drift
            self._drift_exact_cpu(self.length, x, px, y, py, z, dp, tag, mask, beta0)
        else:
            if self.model == "mat-kick-mat":
                ds = self.length / self.num_slice
                for _ in range(self.num_slice):
                    self._mat_kick_mat_cpu(x, px, y, py, z, dp, tag, mask,
                                           chi, beta0, ds)
            else:  # drift-kick-drift-exact
                ds = self.length / self.num_slice
                for _ in range(self.num_slice):
                    if self.integrator == "uniform":
                        self._dkd_uniform_cpu(x, px, y, py, z, dp, tag, mask,
                                              ds, self.k1, self.k1s, chi, beta0)
                    elif self.integrator == "yoshida4":
                        self._dkd_yoshida4_cpu(x, px, y, py, z, dp, tag, mask,
                                               ds, self.k1, self.k1s, chi, beta0)

        # ---- Update lost particle info ----
        newly_lost = alive_before & (tag < 0)
        if np.any(newly_lost):
            lost_position = p.lost_position[start:end]
            lost_turn = p.lost_turn[start:end]
            lost_position[newly_lost] = self.s
            lost_turn[newly_lost] = turn

    # ============================================================
    # Body: mat-kick-mat (exact linear transport matrix)
    # ============================================================

    def _mat_kick_mat_cpu(self, x, px, y, py, z, dp, tag, mask,
                          chi, beta0, ds):
        """
        Exact linear quadrupole transport with chromaticity (one slice).

        For k1 + k1s combined quadrupole:
          1. Rotate to principal axes by theta = 0.5*arctan2(-k1s, k1)
          2. Apply exact linear matrix with K_eff = sqrt(k1^2 + k1s^2)
             K_eff_scaled = K_eff * chi / (1 + delta)   [per-particle]
          3. Rotate back

        For pure k1 (k1s = 0):
          theta = 0, rotation is identity, directly apply matrix with K_eff = k1.

        The matrix is the exact solution of:
          u'' + K_eff * chi / (1+delta) * u = 0

        which includes exact linear chromaticity.
        Nonlinear kinematics (pz higher-order terms) are not included.

        For pure k1 + k1s (no higher-order multipoles), the matrix is exact
        for any slice length ds, so num_slice=1 is sufficient. Multiple slices
        only matter when nonlinear multipole kicks (k2, k2s, ...) are inserted
        between matrix steps (future feature).

        Longitudinal (z) update:
          Uses the linearized path length from the matrix transport,
          analogous to Xsuite's track_expanded_combined_dipole_quad.
          dzeta = ds - L_path / rvv
          where L_path is computed from the linearized trajectory.
        """
        L = ds
        one_plus_delta = 1.0 + dp

        # ---- Step 1: Rotate to principal axes (if skew) ----
        if self.is_skew:
            ct = self.cos_theta
            st = self.sin_theta
            u  =  ct * x + st * y
            pu =  ct * px + st * py
            v  = -st * x + ct * y
            pv = -st * px + ct * py
        else:
            u  = x
            pu = px
            v  = y
            pv = py

        # ---- Step 2: Apply exact linear matrix ----
        # K_eff_scaled = k_eff_base * chi / (1+delta)  [per-particle]
        K = self.k_eff_base * chi / one_plus_delta

        # u-plane (focusing for K > 0): sin/cos
        # v-plane (defocusing for K > 0): sinh/cosh
        # For K < 0, the roles swap: u uses sinh/cosh, v uses sin/cos
        # We handle this by computing based on sign of K.

        # u-plane
        K_pos_u = K > 0.0
        K_neg_u = K < 0.0
        K_zero_u = np.abs(K) < 1e-15

        sqrt_K = np.sqrt(np.abs(K))
        KL = sqrt_K * L

        # For K > 0: cos, sin/sqrt_K
        # For K < 0: cosh, sinh/sqrt_K
        Cu = np.where(K_pos_u, np.cos(KL), np.cosh(KL))
        Su = np.where(K_pos_u, np.sin(KL) / np.where(K_zero_u, 1.0, sqrt_K),
                                 np.sinh(KL) / np.where(K_zero_u, 1.0, sqrt_K))
        # Handle K ≈ 0: Cu=1, Su=L
        Su = np.where(K_zero_u, L, Su)
        Cu = np.where(K_zero_u, 1.0, Cu)

        # v-plane: opposite sign of K
        K_v = -K
        K_pos_v = K_v > 0.0
        K_neg_v = K_v < 0.0
        K_zero_v = np.abs(K_v) < 1e-15

        sqrt_Kv = np.sqrt(np.abs(K_v))
        KLv = sqrt_Kv * L

        Cv = np.where(K_pos_v, np.cos(KLv), np.cosh(KLv))
        Sv = np.where(K_pos_v, np.sin(KLv) / np.where(K_zero_v, 1.0, sqrt_Kv),
                                 np.sinh(KLv) / np.where(K_zero_v, 1.0, sqrt_Kv))
        Sv = np.where(K_zero_v, L, Sv)
        Cv = np.where(K_zero_v, 1.0, Cv)

        # Linearized slopes: xp = pu / (1+delta), yp = pv / (1+delta)
        xp = pu / one_plus_delta
        yp = pv / one_plus_delta

        # Transport: u' = u*Cu + xp*Su, pu' = (-K*u*Su + xp*Cu) * (1+delta)
        u_new  = u * Cu + xp * Su
        pu_new = (-K * u * Su + xp * Cu) * one_plus_delta

        v_new  = v * Cv + yp * Sv
        pv_new = (-K_v * v * Sv + yp * Cv) * one_plus_delta

        # ---- Step 3: Rotate back (if skew) ----
        if self.is_skew:
            x_new  =  ct * u_new - st * v_new
            px_new =  ct * pu_new - st * pv_new
            y_new  =  st * u_new + ct * v_new
            py_new =  st * pu_new + ct * pv_new
        else:
            x_new  = u_new
            px_new = pu_new
            y_new  = v_new
            py_new = pv_new

        # ---- Step 4: Longitudinal (z) update ----
        # Path length from linearized trajectory:
        # L_path = L + 0.5 * (xp^2 * L + ...) terms
        # For the quadrupole matrix, the path length correction comes from
        # the transverse motion. Using the linearized approximation:
        #   delta_ell = L * (xp^2 + yp^2) / 2  (first-order path length)
        # plus higher-order terms from the focusing.
        #
        # Following Xsuite's track_expanded_combined_dipole_quad:
        # For Kx != 0 (here K_u):
        #   L_path corrections involve A = -K*u, B = xp
        # For Ky != 0 (here K_v):
        #   L_path corrections involve C = -K_v*v, D = yp
        #
        # We use the same analytical formulas.

        A = -K * u      # = -K_eff_scaled * u
        B = xp
        C_coeff = -K_v * v
        D = yp

        L_path = L * np.ones_like(x)  # Start with nominal length

        # u-plane path length correction (Kx = K)
        # With h=0, k0=0, the first Xsuite term (h*(...)) vanishes.
        Kx_nonzero = ~K_zero_u
        K_safe_u = np.where(K_zero_u, 1.0, K)
        # The full Xsuite formula (with k0=0, h=0 simplifications):
        # length_ -= (h * ((Cx-1)*xp + Sx*A + length*(k0-h))) / Kx
        #         += 0.5 * (-(A^2*Cx*Sx)/(2*Kx) + (B^2*Cx*Sx)/2
        #                   + (A^2*length)/(2*Kx) + (B^2*length)/2
        #                   - (A*B*Cx^2)/Kx + (A*B)/Kx)
        # With h=0, k0=0: the first term vanishes, leaving:
        L_path = np.where(
            Kx_nonzero,
            L_path + 0.5 * (
                -(A**2 * Cu * Su) / (2.0 * K_safe_u)
                + (B**2 * Cu * Su) / 2.0
                + (A**2 * L) / (2.0 * K_safe_u)
                + (B**2 * L) / 2.0
                - (A * B * Cu**2) / K_safe_u
                + (A * B) / K_safe_u
            ),
            L_path
        )
        # Kx ≈ 0 case: L_path += 0.5 * B^2 * L
        L_path = np.where(
            K_zero_u,
            L_path + 0.5 * B**2 * L,
            L_path
        )

        # v-plane path length correction (Ky = K_v = -K)
        Ky_nonzero = ~K_zero_v
        K_safe_v = np.where(K_zero_v, 1.0, K_v)

        L_path = np.where(
            Ky_nonzero,
            L_path + 0.5 * (
                -(C_coeff**2 * Cv * Sv) / (2.0 * K_safe_v)
                + (D**2 * Cv * Sv) / 2.0
                + (C_coeff**2 * L) / (2.0 * K_safe_v)
                + (D**2 * L) / 2.0
                - (C_coeff * D * Cv**2) / K_safe_v
                + (C_coeff * D) / K_safe_v
            ),
            L_path
        )
        L_path = np.where(
            K_zero_v,
            L_path + 0.5 * D**2 * L,
            L_path
        )

        # dzeta = L - L_path / rvv
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
        bg = beta0 * gamma0
        bg_new = one_plus_delta * bg
        beta = bg_new / np.sqrt(1.0 + bg_new**2)
        rvv = beta / beta0

        dzeta = L - L_path / rvv

        # ---- Apply results (only alive particles) ----
        m = mask
        x[:]  = x_new * m + x * (1.0 - m)
        px[:] = px_new * m + px * (1.0 - m)
        y[:]  = y_new * m + y * (1.0 - m)
        py[:] = py_new * m + py * (1.0 - m)
        z += dzeta * m

    # ============================================================
    # Body: Drift-Kick-Drift exact (uniform integrator)
    # ============================================================

    def _dkd_uniform_cpu(self, x, px, y, py, z, dp, tag, mask,
                         ds, k1, k1s, chi, beta0):
        """
        One DKD slice (uniform/leapfrog, 2nd order symplectic):

          Drift(ds/2) → Kick(ds) → Drift(ds/2)
        """
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._quadrupole_kick_cpu(k1 * ds, k1s * ds,
                                  x, px, y, py, tag, mask, chi)
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)

    # ============================================================
    # Body: Drift-Kick-Drift exact (Yoshida 4th order)
    # ============================================================

    def _dkd_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask,
                          ds, k1, k1s, chi, beta0):
        """
        One Yoshida-4 slice:

          S4(ds) = S2(z1*ds) ∘ S2(z0*ds) ∘ S2(z1*ds)

        where S2 is the standard DKD (leapfrog) step.
        """
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z1, k1, k1s, chi, beta0)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z0, k1, k1s, chi, beta0)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z1, k1, k1s, chi, beta0)

    def _dkd_step_cpu(self, x, px, y, py, z, dp, tag, mask,
                      ds, k1, k1s, chi, beta0):
        """Single DKD step with given effective length ds (can be negative)."""
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._quadrupole_kick_cpu(k1 * ds, k1s * ds,
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

        where pz = sqrt((1+dp)^2 - px^2 - py^2)
              beta = (1+dp)*beta0*gamma0 / sqrt(1 + ((1+dp)*beta0*gamma0)^2)
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
    # Quadrupole kick (thin lens)
    # ============================================================

    def _quadrupole_kick_cpu(self, k1l_eff, k1sl_eff,
                             x, px, y, py, tag, mask, chi):
        """
        Thin quadrupole kick with integrated strengths.

        dpx = -chi * k1l_eff * x + chi * k1sl_eff * y
        dpy =  chi * k1l_eff * y + chi * k1sl_eff * x

        For thin lens mode: k1l_eff = k1l, k1sl_eff = k1sl
        For DKD mode:       k1l_eff = k1 * ds, k1sl_eff = k1s * ds
        """
        if abs(k1l_eff) < const.eps and abs(k1sl_eff) < const.eps:
            return

        k1l_mask = k1l_eff * mask

        px -= chi * k1l_mask * x
        py += chi * k1l_mask * y

        # Skew quadrupole
        if abs(k1sl_eff) > const.eps:
            k1sl_mask = k1sl_eff * mask
            px += chi * k1sl_mask * y
            py += chi * k1sl_mask * x


# ============================================================
# Helper: compute beta from (1+delta) and beta0*gamma0
# ============================================================

def one_plus_delta_beta(one_plus_delta, bg):
    """
    Compute beta = v/c given (1+delta) and beta0*gamma0.

    From: P/P0 = 1+delta = beta*gamma / (beta0*gamma0)
    => beta*gamma = (1+delta) * beta0*gamma0
    => beta = (beta*gamma) / sqrt(1 + (beta*gamma)^2)
    """
    bg_new = one_plus_delta * bg
    return bg_new / np.sqrt(1.0 + bg_new**2)
