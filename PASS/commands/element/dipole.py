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


@Command.register("sbend")
class SBend(Command):
    """
    Sector bend (dipole magnet) with nonlinear edge effects.

    Tracking sequence (entry → body → exit):

      Entry:  YRotation(-e1) → DipoleFringe → Wedge(-e1, K0)
      Body:   N slices of DKD (model-dependent)
      Exit:   Wedge(-e2, K0) → DipoleFringe(-K0) → YRotation(-e2)

    Body models (self.model):

      'drift-kick-drift-exact' (Xsuite model=5):
          drift = straight exact drift (no curvature)
          kick  = k0 + h·(1+δ) + k0·h·x + h·x(path length)
          Curvature is handled as thin-lens kicks. When num_slice is small,
          the missing (1+h·x) Jacobian in the drift causes chromaticity
          errors for d≠0 particles.

      'rot-kick-rot' (Xsuite model=3, drift_model=7) [default]:
          drift = Yoshida-4 of (polar_drift + k0_kick)
          kick  = k0·h·x weak focusing only
          k0 and h are both inside the drift step:
            - h via polar drift coordinate rotation (curvilinear)
            - k0 via interleaved k0_kick sub-steps
          The (1+h·x) Jacobian is exactly included in the polar drift,
          eliminating the chromaticity bias of the straight-drift model.
          When h=0, polar drift reduces to straight exact drift.
          Future: multipole field errors can be added to the outer kick,
          alongside the weak focusing term.

    All maps follow the Xsuite Physics Guide:
      - Drift exact (Table 1.1, map D):   Eq. 1.86-1.88
      - Polar drift: Xsuite track_magnet_drift.h:45-87
      - Dipole kick (Table 1.1, map K0):  Eq. 1.189
      - Curvature kick (Table 1.1, map h): Eq. 1.189
      - Weak focusing kick (Table 1.1, map K0h): Eq. 1.189
      - YRotation: §1.10.19.2
      - Wedge: §1.10.10, Eq. 1.196-1.201
      - DipoleFringe: §1.10.9, Eq. 1.194-1.195 (PTC-compatible implementation)

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
            raise ValueError(f"The length of SBend {self.cmd_name} is {self.length}, which should be >= 0")
        if self.length > const.eps:
            self.is_thick = True
        else:
            self.is_thick = False

        self.k0l = kwargs["k0l"]
        if self.is_thick:
            if abs(self.k0l) < const.eps:
                self.rho = 0.0
                self.h = 0.0
                self.k0 = 0.0
            else:
                self.rho = self.length / self.k0l
                self.h = self.k0l / self.length
                self.k0 = self.k0l / self.length
        else:
            self.rho = 0.0
            self.h = 0.0
            self.k0 = 0.0

        self.e1 = kwargs.get("e1 (rad)", 0.0)
        self.e2 = kwargs.get("e2 (rad)", 0.0)
        self.hgap = kwargs.get("hgap (m)", 0.0)
        self.fint = kwargs.get("fint", 0.0)
        self.fintx = kwargs.get("fintx", 0.0)
        if self.fintx <= 0.0:
            self.fintx = self.fint

        self.is_field_error = kwargs.get("is field error", False)
        self.field_err_knl = []
        self.field_err_ksl = []
        self.is_ramping = kwargs.get("is ramping", False)
        self.k0l_ramping_filepath = kwargs.get("k0l ramping filepath", None)

        self.num_slice = kwargs.get("num slices", 1)
        if self.num_slice < 1:
            logger.warning(f"The number of slices of {self.cmd_name} is {self.num_slice}, which should be >= 1. It has been changed to 1 now.")
            self.num_slice = 1

        self.model = kwargs.get("model", "adaptive")
        if self.model not in ["adaptive", "drift-kick-drift-exact", "rot-kick-rot"]:
            raise ValueError(f"The model of SBend {self.cmd_name} is {self.model}, which should be 'adaptive', 'drift-kick-drift-exact' or 'rot-kick-rot'.")
        if self.model == "adaptive":
            self.model = "rot-kick-rot"

        self.integrator = kwargs.get("integrator", "adaptive")
        if self.integrator not in ["adaptive", "uniform", "yoshida4"]:
            raise ValueError(f"The integrator of SBend {self.cmd_name} is {self.integrator}, which should be 'adaptive', 'uniform' or 'yoshida4'.")
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
                    f"IsThick={self.is_thick}, K0L={self.k0l:.4f}, E1={self.e1:.4f}, E2={self.e2:.4f}, HGap={self.hgap:.4f}, "
                    f"FInt={self.fint:.4f}, FIntX={self.fintx:.4f}, IsFieldError={self.is_field_error}, "
                    f"IsRamping={self.is_ramping}, NumSlice={self.num_slice:d}, Model={self.model:s}, Integrator={self.integrator:s}, "
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
            self._track_bend_cpu(beam, bunch, turn)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)

    def execute_gpu(self, sim):
        launch_dipole(self, sim)

    # ============================================================
    # Full bend tracking (CPU)
    # ============================================================

    def _track_bend_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):
        """Track particles through the complete bend: entry edge → body → exit edge."""

        if not self.is_thick:
            # Thin lens: only apply k0l kick (no body, no edge)
            self._thin_kick_cpu(self.k0l, beam, bunch)
            return

        beta0 = bunch.beta
        gamma0 = bunch.gamma
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
        # PASS currently tracks single-species beams, so chi = 1
        chi = 1.0

        # mask for alive particles
        mask = (tag > 0).astype(np.float64)

        # ---- Entry edge ----
        self._edge_entry_cpu(x, px, y, py, z, dp, tag, mask,
                             self.e1, self.fint, self.hgap,
                             self.k0, self.h, chi, beta0, gamma0)

        # ---- Body: sliced tracking ----
        ds = self.length / self.num_slice
        for _ in range(self.num_slice):
            if self.model == "drift-kick-drift-exact":
                if self.integrator == "uniform":
                    self._dkd_uniform_cpu(x, px, y, py, z, dp, tag, mask,
                                          ds, self.h, self.k0, chi, beta0)
                elif self.integrator == "yoshida4":
                    self._dkd_yoshida4_cpu(x, px, y, py, z, dp, tag, mask,
                                           ds, self.h, self.k0, chi, beta0)
            elif self.model == "rot-kick-rot":
                if self.integrator == "uniform":
                    self._rkr_uniform_cpu(x, px, y, py, z, dp, tag, mask,
                                          ds, self.h, self.k0, chi, beta0)
                elif self.integrator == "yoshida4":
                    self._rkr_yoshida4_cpu(x, px, y, py, z, dp, tag, mask,
                                           ds, self.h, self.k0, chi, beta0)

        # ---- Exit edge ----
        self._edge_exit_cpu(x, px, y, py, z, dp, tag, mask,
                            self.e2, self.fintx, self.hgap,
                            self.k0, self.h, chi, beta0, gamma0)

        # ---- Update lost particle info ----
        newly_lost = alive_before & (tag < 0)
        if np.any(newly_lost):
            lost_position = p.lost_position[start:end]
            lost_turn = p.lost_turn[start:end]
            lost_position[newly_lost] = self.s
            lost_turn[newly_lost] = turn

    # ============================================================
    # Entry edge: YRotation(-e1) → DipoleFringe → Wedge(-e1, K0)
    # ============================================================

    def _edge_entry_cpu(self, x, px, y, py, z, dp, tag, mask,
                        e1, fint, hgap, k0, h, chi, beta0, gamma0):
        """Entry edge effects: YRotation → Fringe → Wedge.

        Xsuite track_magnet_edge.h model=1/2 (full):
            YRotation(-e1) → DipoleFringe → Wedge(-e1, k0)

        DipoleFringe is called whenever k0 != 0, regardless of fint/hgap.
        When fint=0, the fringe map is NOT identity (it still applies
        the geometric nonlinearity atan(xp/(1+yp²))), so skipping it
        loses the vertical focusing kick.
        """

        has_angle = abs(e1) > const.eps
        has_fringe = abs(k0) > const.eps

        if not has_angle and not has_fringe:
            return

        # ---- Step 1: YRotation(-e1) ----
        if has_angle:
            self._y_rotation_cpu(x, px, y, py, z, dp, tag, mask, -e1, beta0)

        # ---- Step 2: DipoleFringe ----
        if has_fringe:
            self._dipole_fringe_cpu(x, px, y, py, z, dp, tag, mask,
                                    fint, hgap, k0, chi, beta0)

        # ---- Step 3: Wedge(-e1, K0) ----
        if has_angle:
            self._wedge_cpu(x, px, y, py, z, dp, tag, mask, -e1, k0, chi, beta0)

    # ============================================================
    # Exit edge: Wedge(-e2, K0) → DipoleFringe(-K0) → YRotation(-e2)
    # ============================================================

    def _edge_exit_cpu(self, x, px, y, py, z, dp, tag, mask,
                       e2, fintx, hgap, k0, h, chi, beta0, gamma0):
        """Exit edge effects: Wedge -> Fringe -> YRotation (mirror of entry).

        Xsuite track_magnet_edge.h:
          - Line 98:  if (is_exit) k0 = -k0;   (k0 is a local variable)
          - Line 104: DipoleFringe(... k0)     (uses negated k0)
          - Line 125: Wedge(... knorm[0])      (uses ORIGINAL knorm[0], NOT negated!)

        So at exit: DipoleFringe gets -k0, but Wedge gets +k0 (unchanged).
        Physics: Wedge describes rotation in uniform B0 field; the field
        direction does not flip at exit. Only the fringe field transition
        reverses (B0->0 vs 0->B0), hence DipoleFringe needs -k0.
        """

        has_angle = abs(e2) > const.eps
        # Xsuite: DipoleFringe uses negated k0 at exit, Wedge uses original k0
        k0_fringe = -k0   # for DipoleFringe only
        # DipoleFringe is called whenever k0 != 0, regardless of fint/hgap
        has_fringe = abs(k0_fringe) > const.eps

        if not has_angle and not has_fringe:
            return

        # ---- Step 1: Wedge(-e2, k0) ----  (k0 NOT negated, per Xsuite knorm[0])
        if has_angle:
            self._wedge_cpu(x, px, y, py, z, dp, tag, mask, -e2, k0, chi, beta0)

        # ---- Step 2: DipoleFringe(-k0) ----  (k0 negated, per Xsuite if(is_exit) k0=-k0)
        if has_fringe:
            self._dipole_fringe_cpu(x, px, y, py, z, dp, tag, mask,
                                    fintx, hgap, k0_fringe, chi, beta0)

        # ---- Step 3: YRotation(-e2) ----
        if has_angle:
            self._y_rotation_cpu(x, px, y, py, z, dp, tag, mask, -e2, beta0)

    # ============================================================
    # YRotation (pure geometric rotation about y-axis)
    # Xsuite track_yrotation.h, adapted to PASS coordinates.
    #
    # PASS uses z = ζ = s - β0·c·t,  dp = δ = P/P0 - 1.
    # Xsuite uses zeta = ζ, ptau = pτ, delta = δ.
    # Relations: pt = β0·pτ,  δ = √(1 + 2·pτ/β0 + pτ²) - 1
    #            pτ = (1/β0)·[(1+δ)² - 1] / 2  ... but for rotation
    #            we work directly in (ζ, δ) using pz.
    # ============================================================

    def _y_rotation_cpu(self, x, px, y, py, z, dp, tag, mask,
                        angle, beta0):
        """
        YRotation: rotate the reference frame by `angle` about the y-axis.

        Adapted from Xsuite track_yrotation.h.
        Xsuite variables: zeta(=ζ), ptau(=pτ), beta0.
        PASS variables:   z(=ζ),   dp(=δ),    beta0.

        Conversion:
          one_plus_delta = 1 + δ = 1 + dp
          pz = sqrt((1+δ)² - px² - py²)
          pt = β0·pτ, and (1+δ)² = (pτ + 1/β0)² - 1/(β0²·γ0²)
          For the rotation we need (1/β0 + pτ) which equals (1+δ)/β0
          (ultrarelativistic approximation used by Xsuite:
           1/β0 + pτ ≈ (1+δ)/β0  ... exact when γ0→∞)

        To stay consistent with Xsuite we use:
          time_fac = 1/β0 + ptau
          where ptau is derived from delta:
          ptau = sqrt(1 + 2*delta/beta0 + delta²) - 1/beta0
          ... but Xsuite stores ptau directly. We approximate:
          time_fac ≈ (1 + dp) / beta0   (valid for β0→1)
          For general β0, exact: time_fac = 1/β0 + ptau
          where ptau = (1/β0)*((1+dp)² - 1)/2 + dp  (from δ=β0·pτ approx)
          Actually Xsuite: delta = β0*ptau + (β - β0)/β0
          Exact: (1+delta)² = (ptau + 1/β0)² - 1/(β0²·γ0²)
          => ptau + 1/β0 = sqrt((1+delta)² + 1/(β0²·γ0²))
          => time_fac = sqrt((1+δ)² + 1/(β0²·γ0²))

        We use the exact expression.
        """
        # Direct physical angle: angle > 0 rotates frame clockwise (viewed from +y).
        # YRotation formula: px' = cos*px - sin*pz  (standard rotation)
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)
        tan_angle = np.tan(angle)

        one_plus_delta = 1.0 + dp
        pz_sq = one_plus_delta**2 - px**2 - py**2

        valid = pz_sq > 0.0
        alive = tag > 0
        tag[alive & ~valid] = -np.abs(tag[alive & ~valid])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)

        ptt = 1.0 + tan_angle * px / pz
        ptt_safe = np.where(np.abs(ptt) < const.eps, const.eps, ptt)

        # time_fac = 1/beta0 + ptau = sqrt((1+δ)² + 1/(β0²·γ0²))
        gamma0_sq = 1.0 / (1.0 - beta0**2) if beta0 < 1.0 else 1e30
        time_fac = np.sqrt(one_plus_delta**2 + 1.0 / (beta0**2 * gamma0_sq))

        x_new = x / (cos_angle * ptt_safe)
        px_new = cos_angle * px - sin_angle * pz
        y_new = y - tan_angle * x * py / (pz * ptt_safe)
        z_new = z + beta0 * tan_angle * x * time_fac / (pz * ptt_safe)

        active = (alive & valid).astype(mask.dtype, copy=False)
        x[:] = x_new * active + x * (1.0 - active)
        px[:] = px_new * active + px * (1.0 - active)
        y[:] = y_new * active + y * (1.0 - active)
        z[:] = z_new * active + z * (1.0 - active)

    # ============================================================
    # DipoleFringe (nonlinear fringe field)
    # Xsuite geometry with the PTC-compatible generating function
    # §1.10.9, Eq. 1.194-1.195
    # ============================================================

    def _dipole_fringe_cpu(self, x, px, y, py, z, dp, tag, mask,
                           fint, hgap, k0, chi, beta0):
        """
        Dipole fringe field map using the PTC-compatible generating function.

        Reference: Xsuite track_dipole_fringe.h, lines 16-84.
        Physics Guide §1.10.9, Eq. 1.194-1.195.

        This is a thin lens: position changes are O(y²) (coupling),
        momentum gets a kick proportional to y (vertical focusing).
        """
        b0 = k0 * chi  # normalized dipole strength × charge factor

        fh = hgap * fint  # hgap is half-gap, so fh = half_gap * fint
        fsad = 1.0 / (72.0 * fh) if fh > const.eps else 0.0
        k0w = b0

        inv_beta0 = 1.0 / beta0

        one_plus_delta = 1.0 + dp
        dpp = one_plus_delta**2
        pz_sq = dpp - px**2 - py**2

        valid = pz_sq > 0.0
        alive = tag > 0
        tag[alive & ~valid] = -np.abs(tag[alive & ~valid])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)
        inv_pz = 1.0 / pz
        relp = 1.0 / np.sqrt(dpp)
        # Xsuite: pt = LocalParticle_get_ptau(part); tfac = -(1/beta0 + ptau)
        # Exact: 1/beta0 + ptau = sqrt((1+delta)^2 + 1/(beta0^2 * gamma0^2))
        gamma0_sq = 1.0 / (1.0 - beta0**2) if beta0 < 1.0 else 1e30
        tfac = -np.sqrt(one_plus_delta**2 + 1.0 / (beta0**2 * gamma0_sq))

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

        # new_y: solve implicit equation y_f = 2y / (1 + sqrt(1 - 2*ky*y))
        discriminant = 1.0 - 2.0 * ky * y
        discriminant = np.maximum(discriminant, 0.0)
        new_y = 2.0 * y / (1.0 + np.sqrt(discriminant))

        new_x = x + 0.5 * kx * new_y**2
        new_py = py - 4.0 * c3 * new_y**3 - k0w * np.tan(fi0) * new_y
        new_z = z + beta0 * (0.5 * kz * new_y**2 + c3 * new_y**4 * (relp**2) * tfac)

        active = (alive & valid).astype(mask.dtype, copy=False)
        x[:] = new_x * active + x * (1.0 - active)
        y[:] = new_y * active + y * (1.0 - active)
        py[:] = new_py * active + py * (1.0 - active)
        z[:] = new_z * active + z * (1.0 - active)

    # ============================================================
    # Wedge (edge angle: geometric rotation + dipole focusing kick)
    # Xsuite track_wedge.h, Eq. 1.196-1.201
    # ============================================================

    def _wedge_cpu(self, x, px, y, py, z, dp, tag, mask,
                   theta, k0, chi, beta0):
        """
        Wedge map: rotate observation plane by `theta` in uniform dipole field.

        Reference: Xsuite track_wedge.h.
        Physics Guide §1.10.10, Eq. 1.196-1.201.

        When K0=0, this reduces to a pure YRotation.
        When K0≠0, it combines geometric rotation with dipole focusing.
        """
        b1 = k0 * chi

        # If no field, wedge degenerates to YRotation.
        # theta=-e1 is passed as-is; _y_rotation_cpu internally computes
        # sin(-e1)=-sin(e1), which matches Xsuite's convention of
        # passing (-sin_, cos_, -tan_) to YRotation.
        if abs(b1) < const.eps:
            self._y_rotation_cpu(x, px, y, py, z, dp, tag, mask, theta, beta0)
            return

        rvv = 1.0  # rvv = v/v0, for same-species beam rvv ≈ 1 (simplified)
        # More precisely: rvv = beta/beta0
        # beta = (1+dp)*beta0*gamma0 / sqrt(1 + ((1+dp)*beta0*gamma0)^2)
        # For now use the exact expression
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
        bg = beta0 * gamma0
        beta = one_plus_delta_beta(one_plus_delta=1.0 + dp, bg=bg)
        rvv = beta / beta0

        one_plus_delta = 1.0 + dp
        A = 1.0 / np.sqrt(one_plus_delta**2 - py**2)
        pz_sq = one_plus_delta**2 - px**2 - py**2

        valid = pz_sq > 0.0
        alive = tag > 0
        tag[alive & ~valid] = -np.abs(tag[alive & ~valid])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        # new_px: Eq. 1.197
        new_px = px * cos_t + (pz - b1 * x) * sin_t

        # new_pz
        new_pz_sq = one_plus_delta**2 - new_px**2 - py**2
        new_pz_sq = np.maximum(new_pz_sq, const.eps)
        new_pz = np.sqrt(new_pz_sq)

        # new_x: Eq. 1.196
        denom = new_pz + pz * cos_t - px * sin_t
        denom_safe = np.where(np.abs(denom) < const.eps, const.eps, denom)

        new_x = (x * cos_t
                 + (x * px * np.sin(2.0 * theta)
                    + sin_t**2 * (2.0 * x * pz - b1 * x**2)) / denom_safe)

        # D: stable arcsin difference, Eq. 1.105/1.131-1.136
        arg_px = A * px
        arg_new_px = A * new_px
        arg_px = np.clip(arg_px, -1.0, 1.0)
        arg_new_px = np.clip(arg_new_px, -1.0, 1.0)
        D = np.arcsin(arg_px) - np.arcsin(arg_new_px)

        # delta_y: Eq. 1.198
        b1_safe = b1 if abs(b1) > const.eps else const.eps
        delta_y = py * (theta + D) / b1_safe

        # delta_ell: Eq. 1.201
        delta_ell = one_plus_delta * (theta + D) / b1_safe

        active = (alive & valid).astype(mask.dtype, copy=False)
        x[:] = new_x * active + x * (1.0 - active)
        px[:] = new_px * active + px * (1.0 - active)
        y[:] = (y + delta_y) * active + y * (1.0 - active)
        z[:] = (z - delta_ell / rvv) * active + z * (1.0 - active)

    # ============================================================
    # Body: Drift-Kick-Drift exact (uniform integrator)
    # ============================================================

    def _dkd_uniform_cpu(self, x, px, y, py, z, dp, tag, mask,
                         ds, h, k0, chi, beta0):
        """
        One DKD slice (uniform/leapfrog, 2nd order symplectic):

          Drift(ds/2) → Kick(ds) → Drift(ds/2)

        Drift: exact drift (Table 1.1, map D), Eq. 1.86-1.88
        Kick:  dipole + curvature + weak focusing (Eq. 1.189)
        """
        # ---- Half drift ----
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)

        # ---- Kick ----
        self._dipole_kick_cpu(ds, x, px, y, py, z, dp, tag, mask,
                              h, k0, chi, beta0)

        # ---- Half drift ----
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)

    # ============================================================
    # Body: Drift-Kick-Drift exact (Yoshida 4th order)
    # ============================================================

    def _dkd_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask,
                          ds, h, k0, chi, beta0):
        """
        One Yoshida-4 slice:

          S4(ds) = S2(z1·ds) ∘ S2(z0·ds) ∘ S2(z1·ds)

        where S2 is the standard DKD (leapfrog) step.
        Each S2 uses its own drift and kick weights.
        """
        # S2(z1 * ds)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z1, h, k0, chi, beta0)
        # S2(z0 * ds)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z0, h, k0, chi, beta0)
        # S2(z1 * ds)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z1, h, k0, chi, beta0)

    def _dkd_step_cpu(self, x, px, y, py, z, dp, tag, mask,
                      ds, h, k0, chi, beta0):
        """Single DKD step with given effective length ds (can be negative)."""
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._dipole_kick_cpu(ds, x, px, y, py, z, dp, tag, mask,
                              h, k0, chi, beta0)
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

        L_mask = L * (alive & valid)

        x += L_mask * px * inv_pz
        y += L_mask * py * inv_pz
        z += L_mask * (1.0 - (beta0 / beta) * one_plus_delta * inv_pz)

    # ============================================================
    # Dipole kick (Table 1.1, maps K0 + h + K0h)
    # Eq. 1.189-1.190
    # ============================================================

    def _dipole_kick_cpu(self, L, x, px, y, py, z, dp, tag, mask,
                         h, k0, chi, beta0):
        """
        Thin dipole kick from expanded Hamiltonian (Eq. 1.189-1.190).

        Xsuite track_magnet_kick.h:
          dpx += hl * (1 + delta)
          dzeta += -rv0v * hl * x     where rv0v = 1/rvv = beta0/beta

        Terms:
          h*L*(1+dp)       -> curvature kick (reference orbit bending)
          -chi*K0*L        -> main dipole bending
          -chi*K0*h*x*L    -> weak focusing (curvature x dipole coupling)
          -(beta0/beta)*h*x*L -> path length / longitudinal effect
        """
        if abs(L) < const.eps:
            return

        one_plus_delta = 1.0 + dp
        L_mask = L * (tag > 0).astype(mask.dtype, copy=False)

        # px kick: Eq. 1.189
        px += L_mask * (h * one_plus_delta - chi * k0 - chi * k0 * h * x)

        # z (zeta) update: Xsuite uses dzeta += -rv0v * hl * x
        # where rv0v = 1/rvv = beta0/beta (NOT 1/beta0)
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
        bg = beta0 * gamma0
        beta = one_plus_delta_beta(one_plus_delta=one_plus_delta, bg=bg)
        rv0v = beta0 / beta  # = 1/rvv
        z -= L_mask * rv0v * h * x

    # ============================================================
    # Body: Rot-Kick-Rot model (Xsuite model=3, drift_model=7)
    #
    # drift = Yoshida-4 of (polar_drift + k0_kick)
    # kick  = k0·h·x weak focusing only (future: multipole field errors)
    #
    # k0 and h are BOTH inside the drift step:
    #   - h handled by polar drift coordinate rotation
    #   - k0 handled by k0_kick sub-steps interleaved with polar drift
    # The outer kick only contains k0·h·x (weak focusing correction).
    # ============================================================

    def _polar_drift_cpu(self, L, x, px, y, py, z, dp, tag, mask, beta0, h):
        """
        Polar drift in curvilinear coordinates (h ≠ 0).

        Xsuite track_magnet_drift.h:45-87, track_polar_drift_single_particle.
        Based on SUBROUTINE Sprotr in PTC and curex_drift in MAD-NG.

        When h → 0, this reduces to exact straight drift, but the caller
        should dispatch to _drift_exact_cpu for h = 0 to avoid 1/h singularity.

        The (1 + h·x) Jacobian factor is implicitly included in the
        curvilinear coordinate transformation, which is the key difference
        from _drift_exact_cpu.
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

        rho = 1.0 / h
        hs = h * L
        ca = np.cos(hs)
        sa = np.sin(hs)
        sa2 = np.sin(0.5 * hs)

        inv_pz = 1.0 / pz
        pxt = px * inv_pz
        denom = ca - sa * pxt
        denom_safe = np.where(np.abs(denom) < const.eps, const.eps, denom)
        _ptt = 1.0 / denom_safe
        pst = (x + rho) * sa * inv_pz * _ptt

        new_x = (x + rho * (2.0 * sa2**2 + sa * pxt)) * _ptt
        new_px = ca * px + sa * pz
        new_y = y + pst * py

        # delta_ell = one_plus_delta * pst (algebraically equivalent to
        #   Xsuite's one_plus_delta * (x+rho) * sa / ca / pz / (1 - px*sa/ca/pz))
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
        bg = beta0 * gamma0
        beta = one_plus_delta_beta(one_plus_delta=one_plus_delta, bg=bg)
        rvv = beta / beta0

        active = (alive & valid).astype(mask.dtype, copy=False)
        z += (L - one_plus_delta * pst / rvv) * active

        x[:] = new_x * active + x * (1.0 - active)
        px[:] = new_px * active + px * (1.0 - active)
        y[:] = new_y * active + y * (1.0 - active)

    def _rkr_drift_cpu(self, L, x, px, y, py, z, dp, tag, mask,
                       beta0, h, k0, chi):
        """
        Drift step for rot-kick-rot model (Xsuite drift_model=7).

        Yoshida-4 integrator of (polar_drift + k0_kick), with adjacent
        drifts merged. Matches Xsuite track_magnet_drift.h case 7:

          PD(z1·L) → k0k(z1) → PD(z0·L) → k0k(z0) → PD(z0·L) → k0k(z1) → PD(z1·L)

        where PD = polar_drift, k0k = px -= weight·k0·chi·L.

        Both k0 and h are handled inside this step:
          - h via polar drift coordinate rotation
          - k0 via interleaved k0_kick sub-steps

        When h = 0 (straight magnet), dispatches to _drift_exact_cpu
        (k0 is also 0 for sector bends, so no k0_kick needed).
        """
        if abs(L) < const.eps:
            return

        # h = 0 → straight drift (k0 = 0 for sector bends when h = 0)
        if abs(h) < const.eps:
            self._drift_exact_cpu(L, x, px, y, py, z, dp, tag, mask, beta0)
            return

        # Yoshida-4 with merged adjacent drifts
        # Original: PD(z1*L/2) k0k(z1) PD(z1*L/2+z0*L/2) k0k(z0)
        #           PD(z0*L/2+z1*L/2) k0k(z1) PD(z1*L/2)
        # Merged drift lengths:
        d1 = _YOSHIDA_Z1 * L * 0.5                    # ≈  0.6756 * L
        d2 = (_YOSHIDA_Z1 + _YOSHIDA_Z0) * L * 0.5     # ≈ -0.1756 * L
        # k0 kick weights:
        k1_w = _YOSHIDA_Z1 * k0 * chi * L              # ≈  1.3512 * k0 * chi * L
        k0_w = _YOSHIDA_Z0 * k0 * chi * L              # ≈ -1.7024 * k0 * chi * L

        self._polar_drift_cpu(d1, x, px, y, py, z, dp, tag, mask, beta0, h)
        px -= k1_w * (tag > 0).astype(mask.dtype, copy=False)

        self._polar_drift_cpu(d2, x, px, y, py, z, dp, tag, mask, beta0, h)
        px -= k0_w * (tag > 0).astype(mask.dtype, copy=False)

        self._polar_drift_cpu(d2, x, px, y, py, z, dp, tag, mask, beta0, h)
        px -= k1_w * (tag > 0).astype(mask.dtype, copy=False)

        self._polar_drift_cpu(d1, x, px, y, py, z, dp, tag, mask, beta0, h)

    def _rkr_kick_cpu(self, L, x, px, y, py, z, dp, tag, mask,
                      h, k0, chi, beta0):
        """
        Outer kick for rot-kick-rot model (kick_rot_frame=0).

        Only contains terms NOT handled by _rkr_drift_cpu:
          - k0·h·x  weak focusing (curvature × dipole coupling)

        Does NOT contain (handled by _rkr_drift_cpu):
          - k0 main bend       → k0_kick inside _rkr_drift_cpu
          - h·(1+δ) curvature  → polar drift coordinate rotation
          - h·x path length    → polar drift delta_ell

        Future: multipole field errors (k1, k2, knl, ksl) will be added here,
        alongside the weak focusing term, enabling per-element field errors
        within the DKD structure.
        """
        if abs(L) < const.eps:
            return

        L_mask = L * (tag > 0).astype(mask.dtype, copy=False)

        # Weak focusing: dpx = -chi * k0 * h * x * L
        px -= L_mask * chi * k0 * h * x

    def _rkr_step_cpu(self, x, px, y, py, z, dp, tag, mask,
                      ds, h, k0, chi, beta0):
        """Single DKD step for rot-kick-rot model."""
        self._rkr_drift_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask,
                            beta0, h, k0, chi)
        self._rkr_kick_cpu(ds, x, px, y, py, z, dp, tag, mask,
                           h, k0, chi, beta0)
        self._rkr_drift_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask,
                            beta0, h, k0, chi)

    def _rkr_uniform_cpu(self, x, px, y, py, z, dp, tag, mask,
                         ds, h, k0, chi, beta0):
        """
        Uniform (leapfrog, 2nd order) integrator for rot-kick-rot:

          Drift(ds/2) → Kick(ds) → Drift(ds/2)

        Drift: _rkr_drift_cpu (Yoshida-4 of polar_drift + k0_kick)
        Kick:  _rkr_kick_cpu  (k0·h·x weak focusing)
        """
        self._rkr_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds, h, k0, chi, beta0)

    def _rkr_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask,
                          ds, h, k0, chi, beta0):
        """
        Yoshida-4th order integrator for rot-kick-rot:

          S4(ds) = S2(z1·ds) ∘ S2(z0·ds) ∘ S2(z1·ds)

        where S2 is _rkr_step_cpu.
        """
        # S2(z1 * ds)
        self._rkr_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z1, h, k0, chi, beta0)
        # S2(z0 * ds)
        self._rkr_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z0, h, k0, chi, beta0)
        # S2(z1 * ds)
        self._rkr_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z1, h, k0, chi, beta0)

    # ============================================================
    # Thin lens kick (for zero-length bend)
    # ============================================================

    def _thin_kick_cpu(self, k0l, beam: Beam, bunch: BunchInfo):
        """Apply a thin dipole kick for zero-length bend."""
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

        mask = (tag > 0).astype(np.float64)
        chi = 1.0
        beta0 = bunch.beta

        px -= chi * k0l * mask
        # For thin bend with h: px += h*k0l*(1+dp) but if length=0, h is undefined
        # so we only apply the dipole kick


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
#ifndef PASS_DIPOLE_MODEL
#define PASS_DIPOLE_MODEL 0
#endif
#ifndef PASS_DIPOLE_INTEGRATOR
#define PASS_DIPOLE_INTEGRATOR 0
#endif
#ifndef PASS_DIPOLE_THIN
#define PASS_DIPOLE_THIN 0
#endif
#define PASS_DIPOLE_INLINE __forceinline__
'''


DIPOLE_BODY = r'''
__device__ PASS_DIPOLE_INLINE bool d_drift(
    pass_real_t& x, pass_real_t& px, pass_real_t& y, pass_real_t& py,
    pass_real_t& z, pass_real_t dp, int& tag, float* lp, int* lt, int i,
    pass_real_t L, pass_real_t beta_ratio, pass_real_t bg0, pass_real_t s0,
    int turn)
{
    // Match the CPU epsilon guard; exact equality is unsafe for floating-point
    // slice lengths (especially after Yoshida scaling).
    if (fabs(L) < PASS_EPS || tag <= 0) return tag > 0;
    pass_real_t one_plus_delta = (pass_real_t)1 + dp;
    pass_real_t pz_squared = one_plus_delta * one_plus_delta - px * px - py * py;
    if (!(pz_squared > (pass_real_t)0)) {
        tag = -abs(tag);
        lp[i] = (float)s0;
        lt[i] = turn;
        return false;
    }
    pass_real_t inv_pz = (pass_real_t)1 / sqrt(pz_squared);
    x += L * px * inv_pz;
    y += L * py * inv_pz;
    z += L * ((pass_real_t)1 - beta_ratio * one_plus_delta * inv_pz);
    return true;
}

__device__ PASS_DIPOLE_INLINE bool d_yrot(
    pass_real_t& x, pass_real_t& px, pass_real_t& y, pass_real_t& z,
    pass_real_t py, pass_real_t dp, int& tag, float* lp, int* lt, int i,
    pass_real_t a, pass_real_t sa, pass_real_t ca,
    pass_real_t beta0, pass_real_t time_factor,
    pass_real_t s0, int turn)
{
    if (fabs(a) < PASS_EPS || tag <= 0) return tag > 0;
    pass_real_t one_plus_delta = (pass_real_t)1 + dp;
    pass_real_t pz_squared = one_plus_delta * one_plus_delta - px * px - py * py;
    if (!(pz_squared > (pass_real_t)0)) {
        tag = -abs(tag);
        lp[i] = (float)s0;
        lt[i] = turn;
        return false;
    }
    pass_real_t pz = sqrt(pz_squared);
    pass_real_t ta = sa / ca;
    pass_real_t ptt = (pass_real_t)1 + ta * px / pz;
    if (fabs(ptt) < PASS_EPS) ptt = PASS_EPS;
    pass_real_t xold = x;
    x = xold / (ca * ptt);
    px = ca * px - sa * pz;
    y = y - ta * xold * py / (pz * ptt);
    z = z + beta0 * ta * xold * time_factor / (pz * ptt);
    return true;
}

__device__ PASS_DIPOLE_INLINE bool d_fringe(pass_real_t& x, pass_real_t& px,
    pass_real_t& y, pass_real_t& py, pass_real_t& z, pass_real_t dp,
    int& tag, float* lp, int* lt, int i, pass_real_t fint,
    pass_real_t hgap, pass_real_t k0, pass_real_t beta0, pass_real_t time_factor,
    pass_real_t s0, int turn)
{
    if (fabs(k0) < PASS_EPS || tag <= 0) return tag > 0;
    pass_real_t dipole_strength = k0;
    pass_real_t fh = hgap * fint;
    pass_real_t fsad = (fh > PASS_EPS)
        ? (pass_real_t)1 / (72 * fh) : (pass_real_t)0;
    pass_real_t one_plus_delta = (pass_real_t)1 + dp;
    pass_real_t pz_squared = one_plus_delta * one_plus_delta - px * px - py * py;
    if (!(pz_squared > (pass_real_t)0)) {
        tag = -abs(tag);
        lp[i] = (float)s0;
        lt[i] = turn;
        return false;
    }
    pass_real_t pz = sqrt(pz_squared);
    pass_real_t inv_pz = (pass_real_t)1 / pz;
    pass_real_t inv_one_plus_delta = (pass_real_t)1 / sqrt(one_plus_delta * one_plus_delta);
    time_factor = -time_factor;
    pass_real_t fringe_linear_strength = dipole_strength * fh * 2;
    pass_real_t fringe_cubic_strength = dipole_strength * dipole_strength * fsad
        * inv_one_plus_delta;
    pass_real_t normalized_px = px * inv_pz;
    pass_real_t normalized_py = py * inv_pz;
    pass_real_t normalized_px_py = normalized_px * normalized_py;
    pass_real_t one_plus_normalized_py_squared = (pass_real_t)1
        + normalized_py * normalized_py;
    pass_real_t normalized_px_squared = normalized_px * normalized_px;
    pass_real_t inv_one_plus_normalized_py_squared = (pass_real_t)1
        / one_plus_normalized_py_squared;
    pass_real_t fringe_angle = atan(normalized_px
        * inv_one_plus_normalized_py_squared)
        - fringe_linear_strength * ((pass_real_t)1 + normalized_px_squared
        * ((pass_real_t)1 + one_plus_normalized_py_squared)) * pz;
    pass_real_t sin_fringe_angle, cos_fringe_angle;
    sincos(fringe_angle, &sin_fringe_angle, &cos_fringe_angle);
    if (fabs(cos_fringe_angle) < PASS_EPS) cos_fringe_angle = PASS_EPS;
    pass_real_t fringe_second_derivative = dipole_strength
        / (cos_fringe_angle * cos_fringe_angle);
    pass_real_t fringe_first_derivative = fringe_second_derivative
        / ((pass_real_t)1 + (normalized_px
        * inv_one_plus_normalized_py_squared) * (normalized_px
        * inv_one_plus_normalized_py_squared))
        * inv_one_plus_normalized_py_squared;
    pass_real_t fringe_third_derivative = fringe_second_derivative
        * fringe_linear_strength;
    pass_real_t fringe_x_derivative = fringe_first_derivative
        - fringe_third_derivative * 2 * normalized_px
        * ((pass_real_t)1 + one_plus_normalized_py_squared) * pz;
    pass_real_t fringe_xy_derivative = -2 * fringe_first_derivative
        * normalized_px_py * inv_one_plus_normalized_py_squared
        - fringe_third_derivative * 2 * normalized_px * normalized_px_py * pz;
    pass_real_t fringe_y_derivative = -fringe_third_derivative
        * ((pass_real_t)1 + normalized_px_squared
        * ((pass_real_t)1 + one_plus_normalized_py_squared));
    pass_real_t x_kick = fringe_x_derivative
        * ((pass_real_t)1 + normalized_px_squared) * inv_pz
        + fringe_xy_derivative * normalized_px_py * inv_pz
        - fringe_y_derivative * normalized_px;
    pass_real_t y_kick = fringe_x_derivative * normalized_px_py * inv_pz
        + fringe_xy_derivative * one_plus_normalized_py_squared * inv_pz
        - fringe_y_derivative * normalized_py;
    pass_real_t z_kick = fringe_x_derivative * time_factor * normalized_px
        * inv_pz * inv_pz + fringe_xy_derivative * time_factor * normalized_py
        * inv_pz * inv_pz - fringe_y_derivative * time_factor * inv_pz;
    pass_real_t y_discriminant = (pass_real_t)1 - 2 * y_kick * y;
    if (y_discriminant < (pass_real_t)0) y_discriminant = 0;
    pass_real_t new_y = 2 * y / ((pass_real_t)1 + sqrt(y_discriminant));
    x += (pass_real_t)0.5 * x_kick * new_y * new_y;
    py -= 4 * fringe_cubic_strength * new_y * new_y * new_y
        + dipole_strength * (sin_fringe_angle / cos_fringe_angle) * new_y;
    y = new_y;
    z += beta0 * ((pass_real_t)0.5 * z_kick * new_y * new_y
        + fringe_cubic_strength * new_y * new_y * new_y * new_y
        * (inv_one_plus_delta * inv_one_plus_delta) * time_factor);
    return true;
}

__device__ PASS_DIPOLE_INLINE bool d_wedge(pass_real_t& x, pass_real_t& px,
    pass_real_t& y, pass_real_t& z, pass_real_t py, pass_real_t dp,
    int& tag, float* lp, int* lt, int i, pass_real_t theta,
    pass_real_t k0, pass_real_t sa, pass_real_t ca,
    pass_real_t beta0, pass_real_t beta_ratio,
    pass_real_t time_factor,
    pass_real_t s0, int turn)
{
    if (tag <= 0) return false;
    if (fabs(k0) < PASS_EPS)
        return d_yrot(x, px, y, z, py, dp, tag, lp, lt, i, theta,
                      sa, ca,
                      beta0, time_factor, s0, turn);
    pass_real_t one_plus_delta = (pass_real_t)1 + dp;
    pass_real_t pz_squared = one_plus_delta * one_plus_delta - px * px - py * py;
    if (!(pz_squared > (pass_real_t)0)) {
        tag = -abs(tag);
        lp[i] = (float)s0;
        lt[i] = turn;
        return false;
    }
    pass_real_t pz = sqrt(pz_squared);
    pass_real_t sin_theta = sa, cos_theta = ca;
    pass_real_t sin_2theta = (pass_real_t)2 * sin_theta * cos_theta;
    pass_real_t new_px = px * cos_theta + (pz - k0 * x) * sin_theta;
    pass_real_t new_pz_squared = one_plus_delta * one_plus_delta
        - new_px * new_px - py * py;
    if (new_pz_squared < PASS_EPS) new_pz_squared = PASS_EPS;
    pass_real_t new_pz = sqrt(new_pz_squared);
    pass_real_t denominator = new_pz + pz * cos_theta - px * sin_theta;
    if (fabs(denominator) < PASS_EPS) denominator = PASS_EPS;
    pass_real_t new_x = x * cos_theta
        + (x * px * sin_2theta + sin_theta * sin_theta
           * ((pass_real_t)2 * x * pz - k0 * x * x)) / denominator;
    pass_real_t inv_transverse_momentum = (pass_real_t)1
        / sqrt(one_plus_delta * one_plus_delta - py * py);
    pass_real_t phase_advance = asin(fmax((pass_real_t)-1,
        fmin((pass_real_t)1, inv_transverse_momentum * px)))
        - asin(fmax((pass_real_t)-1,
        fmin((pass_real_t)1, inv_transverse_momentum * new_px)));
    pass_real_t safe_strength = (fabs(k0) > PASS_EPS) ? k0 : PASS_EPS;
    x = new_x;
    px = new_px;
    y += py * (theta + phase_advance) / safe_strength;
    z -= one_plus_delta * (theta + phase_advance) / safe_strength
        * beta_ratio;
    return true;
}

__device__ PASS_DIPOLE_INLINE void d_kick(pass_real_t& px, pass_real_t& z,
    pass_real_t x, pass_real_t dp, pass_real_t L, pass_real_t h,
    pass_real_t k0, pass_real_t beta0, pass_real_t beta_ratio)
{
    pass_real_t one_plus_delta = (pass_real_t)1 + dp;
    px += L * (h * one_plus_delta - k0 - k0 * h * x);
    z -= L * beta_ratio * h * x;
}

__device__ PASS_DIPOLE_INLINE bool d_polar(pass_real_t& x, pass_real_t& px,
    pass_real_t& y, pass_real_t& z, pass_real_t py, pass_real_t dp,
    int& tag, float* lp, int* lt, int i, pass_real_t L, pass_real_t h,
    pass_real_t rho, pass_real_t sin_bend_angle, pass_real_t cos_bend_angle,
    pass_real_t beta0, pass_real_t beta_ratio, pass_real_t bg0,
    pass_real_t s0, int turn)
{
    if (fabs(L) < PASS_EPS || tag <= 0) return tag > 0;
    if (fabs(h) < PASS_EPS)
        return d_drift(x, px, y, py, z, dp, tag, lp, lt, i,
                       L, beta_ratio, bg0, s0, turn);
    pass_real_t one_plus_delta = (pass_real_t)1 + dp;
    pass_real_t pz_squared = one_plus_delta * one_plus_delta - px * px - py * py;
    if (!(pz_squared > (pass_real_t)0)) {
        tag = -abs(tag);
        lp[i] = (float)s0;
        lt[i] = turn;
        return false;
    }
    pass_real_t pz = sqrt(pz_squared);
    pass_real_t inv_pz = (pass_real_t)1 / pz;
    pass_real_t normalized_px = px * inv_pz;
    pass_real_t denominator = cos_bend_angle - sin_bend_angle * normalized_px;
    if (fabs(denominator) < PASS_EPS) denominator = PASS_EPS;
    pass_real_t path_length_factor = (pass_real_t)1 / denominator;
    pass_real_t polar_path_length = (x + rho) * sin_bend_angle
        * inv_pz * path_length_factor;
    pass_real_t new_x = (x + rho * ((pass_real_t)1 - cos_bend_angle
        + sin_bend_angle * normalized_px))
        * path_length_factor;
    pass_real_t new_px = cos_bend_angle * px + sin_bend_angle * pz;
    pass_real_t new_y = y + polar_path_length * py;
    x = new_x;
    px = new_px;
    y = new_y;
    z += L - one_plus_delta * polar_path_length * beta_ratio;
    return true;
}

__device__ PASS_DIPOLE_INLINE bool d_rkr_drift(pass_real_t& x,pass_real_t& px,
 pass_real_t& y,pass_real_t& z,pass_real_t py,pass_real_t dp,int& tag,
 float*lp,int*lt,int i,pass_real_t L,pass_real_t h,pass_real_t k0,
 pass_real_t beta0,pass_real_t beta_ratio,pass_real_t bg0,pass_real_t rho,
 pass_real_t sin_first,pass_real_t cos_first,
 pass_real_t sin_middle,pass_real_t cos_middle,
 pass_real_t s0,int turn){
  if (fabs(L) < PASS_EPS || tag <= 0) return tag > 0;
  if (fabs(h) < PASS_EPS)
      return d_drift(x, px, y, py, z, dp, tag, lp, lt, i,
                     L, beta_ratio, bg0, s0, turn);
  pass_real_t yoshida_z1 = (pass_real_t)1.3512071919596;
  pass_real_t yoshida_z0 = (pass_real_t)-1.7024143839193;
  pass_real_t first_polar_drift = yoshida_z1 * L * (pass_real_t)0.5;
  pass_real_t middle_polar_drift = (yoshida_z1 + yoshida_z0) * L * (pass_real_t)0.5;
  if (!d_polar(x, px, y, z, py, dp, tag, lp, lt, i,
               first_polar_drift, h, rho, sin_first, cos_first,
               beta0, beta_ratio, bg0, s0, turn)) return false;
  px -= yoshida_z1 * k0 * L;
  if (!d_polar(x, px, y, z, py, dp, tag, lp, lt, i,
               middle_polar_drift, h, rho, sin_middle, cos_middle,
               beta0, beta_ratio, bg0, s0, turn)) return false;
  px -= yoshida_z0 * k0 * L;
  if (!d_polar(x, px, y, z, py, dp, tag, lp, lt, i,
               middle_polar_drift, h, rho, sin_middle, cos_middle,
               beta0, beta_ratio, bg0, s0, turn)) return false;
  px -= yoshida_z1 * k0 * L;
  return d_polar(x, px, y, z, py, dp, tag, lp, lt, i,
                 first_polar_drift, h, rho, sin_first, cos_first,
                 beta0, beta_ratio, bg0, s0, turn);
}

__device__ PASS_DIPOLE_INLINE bool d_dkd(
    pass_real_t& x, pass_real_t& px, pass_real_t& y, pass_real_t& z,
    pass_real_t py, pass_real_t dp, int& tag, float* lp, int* lt, int i,
    pass_real_t L, pass_real_t h, pass_real_t k0, pass_real_t beta0,
    pass_real_t beta_ratio, pass_real_t bg0, pass_real_t s0, int turn)
{
    if (fabs(L) < PASS_EPS || tag <= 0) return tag > 0;
    if (!d_drift(x, px, y, py, z, dp, tag, lp, lt, i,
                 L * (pass_real_t)0.5, beta_ratio, bg0, s0, turn)) return false;
    d_kick(px, z, x, dp, L, h, k0, beta0, beta_ratio);
    return d_drift(x, px, y, py, z, dp, tag, lp, lt, i,
                   L * (pass_real_t)0.5, beta_ratio, bg0, s0, turn);
}

extern "C" __global__ void track_sbend(
    pass_real_t* x, pass_real_t* px, pass_real_t* y, pass_real_t* py,
    pass_real_t* z, const pass_real_t* dp, int* tag, float* lp, int* lt,
    int start, int end, pass_real_t beta0, pass_real_t bg0,
    pass_real_t time_factor_sq_const, pass_real_t rho_const, pass_real_t ds_const,
    pass_real_t L,
    pass_real_t k0l, pass_real_t h, pass_real_t k0, pass_real_t e1,
    pass_real_t e2, pass_real_t e1_s, pass_real_t e1_c,
    pass_real_t e2_s, pass_real_t e2_c,
    pass_real_t hgap, pass_real_t fint, pass_real_t fintx,
    pass_real_t s0, int turn, int slices, int integrator, int model, int thin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i >= end || tag[i] <= 0) return;
    pass_real_t xi = x[i], pxi = px[i], yi = y[i], pyi = py[i];
    pass_real_t zi = z[i], dpi = dp[i];
    int ti = tag[i];
    bool alive = true;
#if PASS_DIPOLE_THIN
    {
        pxi -= k0l;
        x[i] = xi; px[i] = pxi; y[i] = yi; py[i] = pyi;
        z[i] = zi; tag[i] = ti;
        return;
    }
#else
    pass_real_t one_plus_delta_i = (pass_real_t)1 + dpi;
    pass_real_t particle_bg = one_plus_delta_i * bg0;
    pass_real_t particle_beta = particle_bg
        / sqrt((pass_real_t)1 + particle_bg * particle_bg);
    pass_real_t beta_ratio = beta0 / particle_beta;
    pass_real_t rho = rho_const;
    pass_real_t ds = ds_const;
    pass_real_t time_factor = sqrt(one_plus_delta_i * one_plus_delta_i
        + time_factor_sq_const);
#if PASS_DIPOLE_MODEL == 1
    // RKR uses the same polar angles for both drifts around each kick.  Keep
    // these values per particle, but evaluate each distinct angle only once.
    pass_real_t rkr_sf1 = 0, rkr_cf1 = 1, rkr_sm1 = 0, rkr_cm1 = 1;
    pass_real_t rkr_sf0 = 0, rkr_cf0 = 1, rkr_sm0 = 0, rkr_cm0 = 1;
    if (fabs(h) > PASS_EPS) {
        pass_real_t rkr_base = h * ds * (pass_real_t)0.25;
        pass_real_t z1 = (pass_real_t)1.3512071919596;
        pass_real_t z0 = (pass_real_t)-1.7024143839193;
        if (PASS_DIPOLE_INTEGRATOR == 0) {
            sincos(rkr_base * z1, &rkr_sf1, &rkr_cf1);
            sincos(rkr_base * (z1 + z0), &rkr_sm1, &rkr_cm1);
        } else {
            sincos(rkr_base * z1 * z1, &rkr_sf1, &rkr_cf1);
            sincos(rkr_base * z1 * (z1 + z0), &rkr_sm1, &rkr_cm1);
            sincos(rkr_base * z0 * z1, &rkr_sf0, &rkr_cf0);
            sincos(rkr_base * z0 * (z1 + z0), &rkr_sm0, &rkr_cm0);
        }
    }
#endif
    if (fabs(e1) > PASS_EPS)
        alive = d_yrot(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
                       -e1, e1_s, e1_c, beta0, time_factor, s0, turn);
    if (alive && fabs(k0) > PASS_EPS)
        alive = d_fringe(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i,
                         fint, hgap, k0, beta0, time_factor, s0, turn);
    if (alive && fabs(e1) > PASS_EPS)
        alive = d_wedge(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
                        -e1, k0, e1_s, e1_c, beta0, beta_ratio,
                        time_factor, s0, turn);
    for (int slice_index = 0; slice_index < slices && alive; ++slice_index) {
#if PASS_DIPOLE_MODEL == 0
#if PASS_DIPOLE_INTEGRATOR == 0
        alive = d_dkd(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
                      ds, h, k0, beta0, beta_ratio, bg0, s0, turn);
#else
        alive = d_dkd(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
                      ds * (pass_real_t)1.3512071919596,
                      h, k0, beta0, beta_ratio, bg0, s0, turn);
        if (alive) alive = d_dkd(
            xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
            ds * (pass_real_t)-1.7024143839193,
            h, k0, beta0, beta_ratio, bg0, s0, turn);
        if (alive) alive = d_dkd(
            xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
            ds * (pass_real_t)1.3512071919596,
            h, k0, beta0, beta_ratio, bg0, s0, turn);
#endif
#else
#if PASS_DIPOLE_INTEGRATOR == 0
        // RKR outer step; internal polar drift is Yoshida-4.
        pass_real_t d = ds;
        if (alive) alive = d_rkr_drift(
            xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
            d * (pass_real_t)0.5, h, k0, beta0, beta_ratio, bg0, rho,
            rkr_sf1, rkr_cf1, rkr_sm1, rkr_cm1, s0, turn);
        if (alive) pxi -= d * k0 * h * xi;
        if (alive) alive = d_rkr_drift(
            xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
            d * (pass_real_t)0.5, h, k0, beta0, beta_ratio, bg0, rho,
            rkr_sf1, rkr_cf1, rkr_sm1, rkr_cm1, s0, turn);
#else
        // RKR outer step; internal polar drift is Yoshida-4.
        pass_real_t d = ds * (pass_real_t)1.3512071919596;
        if (alive) alive = d_rkr_drift(
            xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
            d * (pass_real_t)0.5, h, k0, beta0, beta_ratio, bg0, rho,
            rkr_sf1, rkr_cf1, rkr_sm1, rkr_cm1, s0, turn);
        if (alive) pxi -= d * k0 * h * xi;
        if (alive) alive = d_rkr_drift(
            xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
            d * (pass_real_t)0.5, h, k0, beta0, beta_ratio, bg0, rho,
            rkr_sf1, rkr_cf1, rkr_sm1, rkr_cm1, s0, turn);

        if (alive) {
            d = ds * (pass_real_t)-1.7024143839193;
            alive = d_rkr_drift(
                xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
                d * (pass_real_t)0.5, h, k0, beta0, beta_ratio, bg0, rho,
                rkr_sf0, rkr_cf0, rkr_sm0, rkr_cm0, s0, turn);
            if (alive) {
                pxi -= d * k0 * h * xi;
                alive = d_rkr_drift(
                    xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
                    d * (pass_real_t)0.5, h, k0, beta0, beta_ratio, bg0, rho,
                    rkr_sf0, rkr_cf0, rkr_sm0, rkr_cm0, s0, turn);
            }
            d = ds * (pass_real_t)1.3512071919596;
            if (alive) alive = d_rkr_drift(
                xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
                d * (pass_real_t)0.5, h, k0, beta0, beta_ratio, bg0, rho,
                rkr_sf1, rkr_cf1, rkr_sm1, rkr_cm1, s0, turn);
            if (alive) {
                pxi -= d * k0 * h * xi;
                alive = d_rkr_drift(
                    xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
                    d * (pass_real_t)0.5, h, k0, beta0, beta_ratio, bg0, rho,
                    rkr_sf1, rkr_cf1, rkr_sm1, rkr_cm1, s0, turn);
            }
        }
#endif
#endif
    }
    if (fabs(e2) > PASS_EPS)
        alive = d_wedge(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
                        -e2, k0, e2_s, e2_c, beta0, beta_ratio,
                        time_factor, s0, turn);
    if (alive && fabs(k0) > PASS_EPS)
        alive = d_fringe(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i,
                         fintx, hgap, -k0, beta0, time_factor, s0, turn);
    if (alive && fabs(e2) > PASS_EPS)
        alive = d_yrot(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i,
                       -e2, e2_s, e2_c, beta0, time_factor, s0, turn);
    x[i] = xi; px[i] = pxi; y[i] = yi; py[i] = pyi;
    z[i] = zi; tag[i] = ti;
#endif
}

'''

_kernels = {}


def _get_fused_kernel(dtype, model=0, integrator=0, thin=0):
    """Compile the single-launch map once per particle precision."""
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU SBend tracking requires the optional 'cuda' dependencies.") from exc
    key = (np.dtype(dtype), int(model), int(integrator), int(thin))
    if key not in _kernels:
        options = (
            "--std=c++14",
            f"-DPASS_USE_FLOAT={int(key[0] == np.dtype(np.float32))}",
            f"-DPASS_DIPOLE_MODEL={key[1]}",
            f"-DPASS_DIPOLE_INTEGRATOR={key[2]}",
            f"-DPASS_DIPOLE_THIN={key[3]}",
        )
        if key[0] == np.dtype(np.float64):
            options += ("--maxrregcount=160",)
        _kernels[key] = cp.RawKernel(
            CUDA_REAL_PREAMBLE + DIPOLE_BODY, "track_sbend", options=options
        )
    return _kernels[key]


def launch_dipole(element, sim):
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU SBend tracking requires the optional 'cuda' dependencies.") from exc
    beam=sim.beams[element.beam_id]; p=beam.particles
    real=p.real
    threads = 256
    turn=sim.state.turn
    model = 0 if element.model == "drift-kick-drift-exact" else 1
    integrator = 0 if element.integrator == "uniform" else 1
    thin = 0 if element.is_thick else 1
    kernel = _get_fused_kernel(p.dtype, model, integrator, thin)
    for b in beam.bunches:
        n=b.end_idx-b.start_idx
        if n > 0:
            blocks = (n + threads - 1) // threads
            args = (
                p.x,p.px,p.y,p.py,p.z,p.dp,p.tag,p.lost_position,p.lost_turn,
                np.int32(b.start_idx),np.int32(b.end_idx),real(b.beta),real(b.beta*b.gamma),
                real(1.0 / ((1.0 + (b.beta * b.gamma) ** 2) * b.beta ** 2)),
                real(1.0 / element.h if abs(element.h) > const.eps else 0.0),
                real(element.length / element.num_slice),
                real(element.length),real(element.k0l),real(element.h),real(element.k0),
                real(element.e1),real(element.e2),
                real(np.sin(-element.e1)),real(np.cos(-element.e1)),
                real(np.sin(-element.e2)),real(np.cos(-element.e2)),
                real(element.hgap),real(element.fint),
                real(element.fintx),real(element.s),np.int32(turn),np.int32(element.num_slice),
                np.int32(0 if element.integrator=="uniform" else 1),
                np.int32(0 if element.model=="drift-kick-drift-exact" else 1),
                np.int32(0 if element.is_thick else 1))
            kernel((blocks,), (threads,), args)
        if n > 0:
            from PASS.utils.aperture import check_aperture_gpu
            check_aperture_gpu(beam,b,element.aperture_type,element.aperture_value,element.s,turn)
        if abs(element.length) >= const.eps:
            b.t0 += element.length / (b.beta * const.c)
