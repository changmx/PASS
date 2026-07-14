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


@Command.register("sbend")
class SBend(Command):
    """
    Sector bend (dipole magnet) with nonlinear edge effects.

    Tracking sequence (entry → body → exit):

      Entry:  YRotation(-e1) → DipoleFringe → Wedge(-e1, K0)
      Body:   N slices of drift-kick-drift-exact (uniform or yoshida4)
      Exit:   Wedge(-e2, K0) → DipoleFringe → YRotation(+e2)

    All maps follow the Xsuite Physics Guide:
      - Drift exact (Table 1.1, map D):   Eq. 1.86-1.88
      - Dipole kick (Table 1.1, map K0):  Eq. 1.189
      - Curvature kick (Table 1.1, map h): Eq. 1.189
      - Weak focusing kick (Table 1.1, map K0h): Eq. 1.189
      - YRotation: §1.10.19.2
      - Wedge: §1.10.10, Eq. 1.196-1.201
      - DipoleFringe: §1.10.9, Eq. 1.194-1.195 (MAD-NG implementation)

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
        if self.model not in ["adaptive", "drift-kick-drift-exact"]:
            raise ValueError(f"The model of SBend {self.cmd_name} is {self.model}, which should be 'adaptive' or 'drift-kick-drift-exact'.")
        if self.model == "adaptive":
            self.model = "drift-kick-drift-exact"

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

    def execute_gpu(self, sim):
        raise NotImplementedError("GPU implementation of SBend is not yet available")

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

        # ---- Body: sliced DKD-exact ----
        ds = self.length / self.num_slice
        for _ in range(self.num_slice):
            if self.integrator == "uniform":
                self._dkd_uniform_cpu(x, px, y, py, z, dp, tag, mask,
                                      ds, self.h, self.k0, chi, beta0)
            elif self.integrator == "yoshida4":
                self._dkd_yoshida4_cpu(x, px, y, py, z, dp, tag, mask,
                                       ds, self.h, self.k0, chi, beta0)

        # ---- Exit edge ----
        self._edge_exit_cpu(x, px, y, py, z, dp, tag, mask,
                            self.e2, self.fintx, self.hgap,
                            self.k0, self.h, chi, beta0, gamma0)

        # ---- Wrap z into [-C/2, C/2) ----
        c_half = 0.5 * circum
        over = (z > c_half).astype(np.int64)
        under = (z < -c_half).astype(np.int64)
        z += (under - over) * circum

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
        """Entry edge effects: YRotation → Fringe → Wedge."""

        has_angle = abs(e1) > const.eps
        has_fringe = (fint > const.eps) and (hgap > const.eps) and (abs(k0) > const.eps)

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
    # Exit edge: Wedge(-e2, K0) → DipoleFringe → YRotation(+e2)
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
        has_fringe = (fintx > const.eps) and (hgap > const.eps) and (abs(k0_fringe) > const.eps)

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

        valid = (pz_sq > 0.0) & (tag > 0)
        tag[~valid] = -np.abs(tag[~valid])
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

        x[:] = x_new * mask + x * (1.0 - mask)
        px[:] = px_new * mask + px * (1.0 - mask)
        y[:] = y_new * mask + y * (1.0 - mask)
        z[:] = z_new * mask + z * (1.0 - mask)

    # ============================================================
    # DipoleFringe (nonlinear fringe field)
    # Xsuite track_dipole_fringe.h (MAD-NG implementation)
    # §1.10.9, Eq. 1.194-1.195
    # ============================================================

    def _dipole_fringe_cpu(self, x, px, y, py, z, dp, tag, mask,
                           fint, hgap, k0, chi, beta0):
        """
        Dipole fringe field map (MAD-NG / Xsuite implementation).

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

        valid = (pz_sq > 0.0) & (tag > 0)
        tag[~valid] = -np.abs(tag[~valid])
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

        x[:] = new_x * mask + x * (1.0 - mask)
        y[:] = new_y * mask + y * (1.0 - mask)
        py[:] = new_py * mask + py * (1.0 - mask)
        z[:] = new_z * mask + z * (1.0 - mask)

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

        valid = (pz_sq > 0.0) & (tag > 0)
        tag[~valid] = -np.abs(tag[~valid])
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

        x[:] = new_x * mask + x * (1.0 - mask)
        px[:] = new_px * mask + px * (1.0 - mask)
        y[:] = (y + delta_y) * mask + y * (1.0 - mask)
        z[:] = (z - delta_ell / rvv) * mask + z * (1.0 - mask)

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
        L_mask = L * mask

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
