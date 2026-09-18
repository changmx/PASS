from functools import lru_cache
import logging

import numpy as np

from PASS.commands.command import Command
from PASS.utils.slicing import print_element_slicing, configure_element_slicing, run_body_slices
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const, _build_yoshida_cuda_constants
from PASS.utils.compute_kinematics import compute_particle_beta
from PASS.utils.aperture import check_aperture_cpu

logger = logging.getLogger(__name__)


@Command.register("sbend")
class SBend(Command):
    """Track a sector bend with nonlinear entrance and exit maps.

    The body uses exact drift-kick-drift or rot-kick-rot integration. The latter
    includes curvature in the polar drift. Edge maps reverse the fringe field
    at exit while keeping the wedge field unchanged. See docs/element/dipole."""

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
            raise ValueError(
                f"The model of SBend {self.cmd_name} is {self.model}, which should be 'adaptive', 'drift-kick-drift-exact' or 'rot-kick-rot'.")
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

        configure_element_slicing(self, sim, kwargs)
        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, Length={self.length:.4f}, "
                    f"IsThick={self.is_thick}, K0L={self.k0l:.4f}, E1={self.e1:.4f}, E2={self.e2:.4f}, HGap={self.hgap:.4f}, "
                    f"FInt={self.fint:.4f}, FIntX={self.fintx:.4f}, IsFieldError={self.is_field_error}, "
                    f"IsRamping={self.is_ramping}, NumSlice={self.num_slice:d}, Model={self.model:s}, Integrator={self.integrator:s}, "
                    f"ApertureType={self.aperture_type:s}, ApertureValue={self.aperture_value}")
        print_element_slicing(self)
        set_normal_logging()

    def execute_cpu(self, sim):
        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        turn = sim.state.turn

        for i, bunch in enumerate(bunches):
            self._track_bend_cpu(beam, bunch, turn)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)
        return True

    def execute_gpu(self, sim):
        if self._sc_nodes:
            from PASS.utils.slicing import execute_internal_sc_gpu
            return execute_internal_sc_gpu(self, sim)
        launch_dipole(self, sim)
        return True

    def _track_bend_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):

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
        stored_z = p.z[start:end]
        # Static maps depend on z only through additive time-of-flight changes.
        # Sum their small increments separately before rounding the stored z once.
        z = stored_z if self._sc_nodes else np.zeros_like(stored_z)
        dp = p.dp[start:end]
        tag = p.tag[start:end]

        # RKR's internal rotations carry a large reference bending momentum.
        # Round only the completed static map, not each cancelling substep.
        precise_rkr = self.model == "rot-kick-rot" and not self._sc_nodes and p.dtype == np.float32
        stored_transverse = (x, px, y, py)
        if precise_rkr:
            x, px, y, py = (coordinate.astype(np.float64) for coordinate in stored_transverse)
            z = np.zeros_like(stored_z, dtype=np.float64)
            dp = dp.astype(np.float64)

        alive_before = tag > 0

        # chi = q/q0 * m0/m  (for same-species beam, chi = 1)
        # PASS currently tracks single-species beams, so chi = 1
        chi = 1.0

        mask = (tag > 0).astype(np.float64)

        self._edge_entry_cpu(x, px, y, py, z, dp, tag, mask, self.e1, self.fint, self.hgap, self.k0, self.h, chi, beta0, gamma0)

        if self._sc_nodes:
            entry_lost = alive_before & (tag <= 0)
            p.lost_position[start:end][entry_lost] = self.s - self.length
            p.lost_turn[start:end][entry_lost] = turn
            if self.model == "rot-kick-rot":
                step = self._rkr_uniform_cpu if self.integrator == "uniform" else self._rkr_yoshida4_cpu
            else:
                step = self._dkd_step_cpu if self.integrator == "uniform" else self._dkd_yoshida4_cpu

            def transport(ds, on_center):
                step(x, px, y, py, z, dp, tag, mask, ds, self.h, self.k0, chi, beta0, on_center=on_center)

            run_body_slices(self, beam, bunch, turn, transport)
            # Preserve body/node loss coordinates; the final block records exit losses only.
            alive_before = tag > 0
        else:
            ds = self.length / self.num_slice
            for _ in range(self.num_slice):
                if self.model == "drift-kick-drift-exact":
                    if self.integrator == "uniform":
                        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds, self.h, self.k0, chi, beta0)
                    elif self.integrator == "yoshida4":
                        self._dkd_yoshida4_cpu(x, px, y, py, z, dp, tag, mask, ds, self.h, self.k0, chi, beta0)
                elif self.model == "rot-kick-rot":
                    if self.integrator == "uniform":
                        self._rkr_uniform_cpu(x, px, y, py, z, dp, tag, mask, ds, self.h, self.k0, chi, beta0)
                    elif self.integrator == "yoshida4":
                        self._rkr_yoshida4_cpu(x, px, y, py, z, dp, tag, mask, ds, self.h, self.k0, chi, beta0)

        self._edge_exit_cpu(x, px, y, py, z, dp, tag, mask, self.e2, self.fintx, self.hgap, self.k0, self.h, chi, beta0, gamma0)

        if not self._sc_nodes:
            stored_z += z
        if precise_rkr:
            for stored, calculated in zip(stored_transverse, (x, px, y, py)):
                stored[:] = calculated

        newly_lost = alive_before & (tag < 0)
        if np.any(newly_lost):
            lost_position = p.lost_position[start:end]
            lost_turn = p.lost_turn[start:end]
            lost_position[newly_lost] = self.s
            lost_turn[newly_lost] = turn

    # Entry edge: YRotation(-e1) → DipoleFringe → Wedge(-e1, K0)

    def _edge_entry_cpu(self, x, px, y, py, z, dp, tag, mask, e1, fint, hgap, k0, h, chi, beta0, gamma0):
        """Apply entrance rotation, fringe and wedge maps in that order.

        The fringe map retains geometric focusing when fint or hgap is zero."""

        has_angle = abs(e1) > const.eps
        has_fringe = abs(k0) > const.eps

        if not has_angle and not has_fringe:
            return

        if has_angle:
            self._y_rotation_cpu(x, px, y, py, z, dp, tag, mask, -e1, beta0)

        if has_fringe:
            self._dipole_fringe_cpu(x, px, y, py, z, dp, tag, mask, fint, hgap, k0, chi, beta0)

        if has_angle:
            self._wedge_cpu(x, px, y, py, z, dp, tag, mask, -e1, k0, chi, beta0)

    # Exit edge: Wedge(-e2, K0) → DipoleFringe(-K0) → YRotation(-e2)

    def _edge_exit_cpu(self, x, px, y, py, z, dp, tag, mask, e2, fintx, hgap, k0, h, chi, beta0, gamma0):
        """Apply exit wedge, fringe and rotation maps in that order.

        Only the fringe uses -k0: the field transition reverses at exit, whereas
        the wedge still rotates through the same uniform magnetic field."""

        has_angle = abs(e2) > const.eps
        # Reverse the fringe field transition at exit.
        k0_fringe = -k0  # for DipoleFringe only
        # DipoleFringe is called whenever k0 != 0, regardless of fint/hgap
        has_fringe = abs(k0_fringe) > const.eps

        if not has_angle and not has_fringe:
            return

        # Wedge(-e2, k0)
        if has_angle:
            self._wedge_cpu(x, px, y, py, z, dp, tag, mask, -e2, k0, chi, beta0)

        # Fringe(-k0)
        if has_fringe:
            self._dipole_fringe_cpu(x, px, y, py, z, dp, tag, mask, fintx, hgap, k0_fringe, chi, beta0)

        if has_angle:
            self._y_rotation_cpu(x, px, y, py, z, dp, tag, mask, -e2, beta0)

    # Reference-frame rotation about the y axis.

    def _y_rotation_cpu(self, x, px, y, py, z, dp, tag, mask, angle, beta0):
        """Rotate the reference frame about the y axis.

        Use pz = sqrt((1+dp)**2 - px**2 - py**2) and the exact relativistic
        time factor sqrt((1+dp)**2 + 1/(beta0**2 * gamma0**2))."""
        # Direct physical angle: angle > 0 rotates frame clockwise (viewed from +y).
        # YRotation formula: px' = cos*px - sin*pz  (standard rotation)
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)
        tan_angle = np.tan(angle)

        momentum_ratio = 1.0 + dp
        pz_sq = momentum_ratio**2 - px**2 - py**2

        valid = pz_sq > 0.0
        alive = tag > 0
        tag[alive & ~valid] = -np.abs(tag[alive & ~valid])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)

        rotation_denominator = 1.0 + tan_angle * px / pz
        safe_rotation_denominator = np.where(np.abs(rotation_denominator) < const.eps, const.eps, rotation_denominator)

        # normalized_energy = 1/beta0 + ptau = sqrt((1+δ)² + 1/(β0²·γ0²))
        gamma0_sq = 1.0 / (1.0 - beta0**2) if beta0 < 1.0 else 1e30
        normalized_energy = np.sqrt(momentum_ratio**2 + 1.0 / (beta0**2 * gamma0_sq))

        x_new = x / (cos_angle * safe_rotation_denominator)
        px_new = cos_angle * px - sin_angle * pz
        y_new = y - tan_angle * x * py / (pz * safe_rotation_denominator)
        z_new = z + beta0 * tan_angle * x * normalized_energy / (pz * safe_rotation_denominator)

        active = (alive & valid).astype(mask.dtype, copy=False)
        x[:] = x_new * active + x * (1.0 - active)
        px[:] = px_new * active + px * (1.0 - active)
        y[:] = y_new * active + y * (1.0 - active)
        z[:] = z_new * active + z * (1.0 - active)

    # DipoleFringe (nonlinear fringe field)
    # Finite-gap fringe generating function
    # §1.10.9, Eq. 1.194-1.195

    def _dipole_fringe_cpu(self, x, px, y, py, z, dp, tag, mask, fint, hgap, k0, chi, beta0):
        """Apply the thin fringe map from the dipole generating function.

        Position corrections start at O(y**2); the vertical momentum kick
        includes linear focusing. See the dipole physics documentation."""
        b0 = k0 * chi  # normalized dipole strength × charge factor

        fh = hgap * fint  # hgap is half-gap, so fh = half_gap * fint
        fsad = 1.0 / (72.0 * fh) if fh > const.eps else 0.0
        k0w = b0

        inv_beta0 = 1.0 / beta0

        momentum_ratio = 1.0 + dp
        momentum_ratio_squared = momentum_ratio**2
        pz_sq = momentum_ratio_squared - px**2 - py**2

        valid = pz_sq > 0.0
        alive = tag > 0
        tag[alive & ~valid] = -np.abs(tag[alive & ~valid])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)
        inv_pz = 1.0 / pz
        inverse_abs_momentum_ratio = 1.0 / np.sqrt(momentum_ratio_squared)
        # Exact energy factor: negative_normalized_energy = -(1/beta0 + ptau)
        # Exact: 1/beta0 + ptau = sqrt((1+delta)^2 + 1/(beta0^2 * gamma0^2))
        gamma0_sq = 1.0 / (1.0 - beta0**2) if beta0 < 1.0 else 1e30
        negative_normalized_energy = -np.sqrt(momentum_ratio**2 + 1.0 / (beta0**2 * gamma0_sq))

        c2 = k0w * fh * 2.0
        c3 = k0w**2 * fsad * inverse_abs_momentum_ratio

        slope_x = px * inv_pz
        slope_y = py * inv_pz
        slope_xy = slope_x * slope_y
        one_plus_slope_y_squared = 1.0 + slope_y**2
        slope_x_squared = slope_x**2
        inv_one_plus_slope_y_squared = 1.0 / one_plus_slope_y_squared

        # PTC-compatible generating function: the fringe term is proportional
        # to pz (the slope derivatives below still use 1/pz).
        fi0 = np.arctan(slope_x * inv_one_plus_slope_y_squared) - c2 * (1.0 + slope_x_squared * (1.0 + one_plus_slope_y_squared)) * pz
        cos_fi0 = np.cos(fi0)
        cos_fi0_safe = np.where(np.abs(cos_fi0) < const.eps, const.eps, cos_fi0)
        co2 = k0w / (cos_fi0_safe**2)
        co1 = co2 / (1.0 + (slope_x * inv_one_plus_slope_y_squared)**2) * inv_one_plus_slope_y_squared
        co3 = co2 * c2

        fi1 = co1 - co3 * 2.0 * slope_x * (1.0 + one_plus_slope_y_squared) * pz
        fi2 = -2.0 * co1 * slope_xy * inv_one_plus_slope_y_squared - co3 * 2.0 * slope_x * slope_xy * pz
        fi3 = -co3 * (1.0 + slope_x_squared * (1.0 + one_plus_slope_y_squared))

        kx = fi1 * (1.0 + slope_x_squared) * inv_pz + fi2 * slope_xy * inv_pz - fi3 * slope_x
        ky = fi1 * slope_xy * inv_pz + fi2 * one_plus_slope_y_squared * inv_pz - fi3 * slope_y
        kz = fi1 * negative_normalized_energy * slope_x * (inv_pz**2) + fi2 * negative_normalized_energy * slope_y * (
            inv_pz**2) - fi3 * negative_normalized_energy * inv_pz

        # new_y: solve implicit equation y_f = 2y / (1 + sqrt(1 - 2*ky*y))
        discriminant = 1.0 - 2.0 * ky * y
        discriminant = np.maximum(discriminant, 0.0)
        new_y = 2.0 * y / (1.0 + np.sqrt(discriminant))

        new_x = x + 0.5 * kx * new_y**2
        new_py = py - 4.0 * c3 * new_y**3 - k0w * np.tan(fi0) * new_y
        new_z = z + beta0 * (0.5 * kz * new_y**2 + c3 * new_y**4 * (inverse_abs_momentum_ratio**2) * negative_normalized_energy)

        active = (alive & valid).astype(mask.dtype, copy=False)
        x[:] = new_x * active + x * (1.0 - active)
        y[:] = new_y * active + y * (1.0 - active)
        py[:] = new_py * active + py * (1.0 - active)
        z[:] = new_z * active + z * (1.0 - active)

    # Wedge (edge angle: geometric rotation + dipole focusing kick)
    # Wedge map in normalized momentum coordinates

    def _wedge_cpu(self, x, px, y, py, z, dp, tag, mask, theta, k0, chi, beta0):
        """Apply the wedge rotation in the dipole field.

        With zero field this reduces to a rotation of the reference frame."""
        b1 = k0 * chi

        # The supplied angle already contains the entrance or exit sign.
        if abs(b1) < const.eps:
            self._y_rotation_cpu(x, px, y, py, z, dp, tag, mask, theta, beta0)
            return

        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
        reference_beta_gamma = beta0 * gamma0
        particle_beta = compute_particle_beta(momentum_ratio=1.0 + dp, reference_beta_gamma=reference_beta_gamma)
        beta_over_beta0 = particle_beta / beta0

        momentum_ratio = 1.0 + dp
        A = 1.0 / np.sqrt(momentum_ratio**2 - py**2)
        pz_sq = momentum_ratio**2 - px**2 - py**2

        valid = pz_sq > 0.0
        alive = tag > 0
        tag[alive & ~valid] = -np.abs(tag[alive & ~valid])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # new_px: Eq. 1.197
        new_px = px * cos_theta + (pz - b1 * x) * sin_theta

        new_pz_sq = momentum_ratio**2 - new_px**2 - py**2
        new_pz_sq = np.maximum(new_pz_sq, const.eps)
        new_pz = np.sqrt(new_pz_sq)

        # new_x: Eq. 1.196
        denom = new_pz + pz * cos_theta - px * sin_theta
        denom_safe = np.where(np.abs(denom) < const.eps, const.eps, denom)

        new_x = (x * cos_theta + (x * px * np.sin(2.0 * theta) + sin_theta**2 * (2.0 * x * pz - b1 * x**2)) / denom_safe)

        # phase_advance: stable arcsin difference, Eq. 1.105/1.131-1.136
        arg_px = A * px
        arg_new_px = A * new_px
        arg_px = np.clip(arg_px, -1.0, 1.0)
        arg_new_px = np.clip(arg_new_px, -1.0, 1.0)
        phase_advance = np.arcsin(arg_px) - np.arcsin(arg_new_px)

        # delta_y: Eq. 1.198
        b1_safe = b1 if abs(b1) > const.eps else const.eps
        delta_y = py * (theta + phase_advance) / b1_safe

        # path_length: Eq. 1.201
        path_length = momentum_ratio * (theta + phase_advance) / b1_safe

        active = (alive & valid).astype(mask.dtype, copy=False)
        x[:] = new_x * active + x * (1.0 - active)
        px[:] = new_px * active + px * (1.0 - active)
        y[:] = (y + delta_y) * active + y * (1.0 - active)
        z[:] = (z - path_length / beta_over_beta0) * active + z * (1.0 - active)

    def _dkd_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask, ds, h, k0, chi, beta0, on_center=None):
        """Compose three drift-kick-drift steps with Yoshida coefficients."""
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, h, k0, chi, beta0)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z0, h, k0, chi, beta0, on_center=on_center)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, h, k0, chi, beta0)

    def _dkd_step_cpu(self, x, px, y, py, z, dp, tag, mask, ds, h, k0, chi, beta0, on_center=None):
        """Apply one drift-kick-drift step; Yoshida composition may use negative ds."""
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._dipole_kick_cpu(ds, x, px, y, py, z, dp, tag, mask, h, k0, chi, beta0)
        if on_center is not None:
            on_center()
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)

    def _drift_exact_cpu(self, L, x, px, y, py, z, dp, tag, mask, beta0):
        """Advance live particles in a straight, field-free region."""
        if abs(L) < const.eps:
            return

        momentum_ratio = 1.0 + dp
        pz_sq = momentum_ratio**2 - px**2 - py**2

        valid = pz_sq > 0.0
        alive = tag > 0
        tag[alive & ~valid] = -np.abs(tag[alive & ~valid])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)
        inv_pz = 1.0 / pz

        # Rationalize 1 - beta0/beta * p/pz to retain high-energy time slip.
        inv_gamma_sq = max(0.0, 1.0 - beta0**2)
        transverse_momentum_squared = px * px + py * py
        energy_ratio = np.sqrt(inv_gamma_sq + (1.0 - inv_gamma_sq) * momentum_ratio**2)
        slip = (dp * (2.0 + dp) * inv_gamma_sq - transverse_momentum_squared) / (pz * (pz + energy_ratio))

        L_mask = L * (alive & valid)

        x += L_mask * px * inv_pz
        y += L_mask * py * inv_pz
        z += L_mask * slip

    def _dipole_kick_cpu(self, L, x, px, y, py, z, dp, tag, mask, h, k0, chi, beta0):
        """Apply curvature, dipole and weak-focusing kicks.

        The longitudinal kick includes the particle-dependent beta0/particle_beta factor."""
        if abs(L) < const.eps:
            return

        momentum_ratio = 1.0 + dp
        L_mask = L * (tag > 0).astype(mask.dtype, copy=False)

        # px kick: Eq. 1.189
        px += L_mask * (h * momentum_ratio - chi * k0 - chi * k0 * h * x)

        # Path-length correction uses beta0/particle_beta.
        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
        reference_beta_gamma = beta0 * gamma0
        particle_beta = compute_particle_beta(momentum_ratio=momentum_ratio, reference_beta_gamma=reference_beta_gamma)
        beta0_over_beta = beta0 / particle_beta
        z -= L_mask * beta0_over_beta * h * x

    def _polar_drift_cpu(self, L, x, px, y, py, z, dp, tag, mask, beta0, h):
        """Drift in curved coordinates, including the (1 + h*x) Jacobian.

        The caller uses the straight exact drift when h is zero to avoid 1/h."""
        if abs(L) < const.eps:
            return

        momentum_ratio = 1.0 + dp
        pz_sq = momentum_ratio**2 - px**2 - py**2

        valid = pz_sq > 0.0
        alive = tag > 0
        tag[alive & ~valid] = -np.abs(tag[alive & ~valid])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)

        rho = 1.0 / h
        bend_angle = h * L
        cos_angle = np.cos(bend_angle)
        sin_angle = np.sin(bend_angle)
        sin_half_angle = np.sin(0.5 * bend_angle)

        inv_pz = 1.0 / pz
        pxt = px * inv_pz
        denom = cos_angle - sin_angle * pxt
        denom_safe = np.where(np.abs(denom) < const.eps, const.eps, denom)
        inverse_rotation_denominator = 1.0 / denom_safe
        pst = (x + rho) * sin_angle * inv_pz * inverse_rotation_denominator

        new_x = (x + rho * (2.0 * sin_half_angle**2 + sin_angle * pxt)) * inverse_rotation_denominator
        new_px = cos_angle * px + sin_angle * pz
        new_y = y + pst * py

        # Particle path length is (1 + dp) * pst.
        inv_gamma_sq = max(0.0, 1.0 - beta0**2)
        beta_ratio_squared_change = -inv_gamma_sq * dp * (2.0 + dp) / momentum_ratio**2
        beta0_over_beta_minus_one = beta_ratio_squared_change / (np.sqrt(1.0 + beta_ratio_squared_change) + 1.0)
        particle_path_length = momentum_ratio * pst

        active = (alive & valid).astype(mask.dtype, copy=False)
        z += ((L - particle_path_length) - particle_path_length * beta0_over_beta_minus_one) * active

        x[:] = new_x * active + x * (1.0 - active)
        px[:] = new_px * active + px * (1.0 - active)
        y[:] = new_y * active + y * (1.0 - active)

    def _rkr_drift_cpu(self, L, x, px, y, py, z, dp, tag, mask, beta0, h, k0, chi):
        """Compose polar drifts and dipole kicks with Yoshida coefficients.

        Adjacent drifts are merged. For zero curvature, use the straight drift."""
        if abs(L) < const.eps:
            return

        # h = 0 → straight drift (k0 = 0 for sector bends when h = 0)
        if abs(h) < const.eps:
            self._drift_exact_cpu(L, x, px, y, py, z, dp, tag, mask, beta0)
            return

        d1 = const.yoshida_z1 * L * 0.5
        d2 = (const.yoshida_z1 + const.yoshida_z0) * L * 0.5
        k1_w = const.yoshida_z1 * k0 * chi * L
        k0_w = const.yoshida_z0 * k0 * chi * L

        self._polar_drift_cpu(d1, x, px, y, py, z, dp, tag, mask, beta0, h)
        px -= k1_w * (tag > 0).astype(mask.dtype, copy=False)

        self._polar_drift_cpu(d2, x, px, y, py, z, dp, tag, mask, beta0, h)
        px -= k0_w * (tag > 0).astype(mask.dtype, copy=False)

        self._polar_drift_cpu(d2, x, px, y, py, z, dp, tag, mask, beta0, h)
        px -= k1_w * (tag > 0).astype(mask.dtype, copy=False)

        self._polar_drift_cpu(d1, x, px, y, py, z, dp, tag, mask, beta0, h)

    def _rkr_kick_cpu(self, L, x, px, y, py, z, dp, tag, mask, h, k0, chi, beta0):
        """Apply the outer weak-focusing kick; curvature and dipole kicks belong to the inner map."""
        if abs(L) < const.eps:
            return

        L_mask = L * (tag > 0).astype(mask.dtype, copy=False)

        # Weak focusing: dpx = -chi * k0 * h * x * L
        px -= L_mask * chi * k0 * h * x

    def _rkr_step_cpu(self, x, px, y, py, z, dp, tag, mask, ds, h, k0, chi, beta0, on_center=None):
        """Single DKD step for rot-kick-rot model."""
        self._rkr_drift_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0, h, k0, chi)
        self._rkr_kick_cpu(ds, x, px, y, py, z, dp, tag, mask, h, k0, chi, beta0)
        if on_center is not None:
            on_center()
        self._rkr_drift_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0, h, k0, chi)

    def _rkr_uniform_cpu(self, x, px, y, py, z, dp, tag, mask, ds, h, k0, chi, beta0, on_center=None):
        """Apply one symmetric rot-kick-rot slice."""
        self._rkr_step_cpu(x, px, y, py, z, dp, tag, mask, ds, h, k0, chi, beta0, on_center=on_center)

    def _rkr_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask, ds, h, k0, chi, beta0, on_center=None):
        """Compose three rot-kick-rot steps with Yoshida coefficients."""
        self._rkr_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, h, k0, chi, beta0)
        self._rkr_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z0, h, k0, chi, beta0, on_center=on_center)
        self._rkr_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, h, k0, chi, beta0)

    # Thin lens kick (for zero-length bend)

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


CUDA_REAL_PREAMBLE = _build_yoshida_cuda_constants() + f'''
#ifndef PASS_USE_FLOAT
#define PASS_USE_FLOAT 0
#endif
#if PASS_USE_FLOAT
using pass_bend_particle_t = float;
#if PASS_DIPOLE_MODEL == 1
// RKR substeps cancel reference-scale rotations; retain their small remainder.
using pass_real_t = double;
#else
using pass_real_t = float;
#endif
// Particle state remains FP32; this type is reserved for the sensitive
// polar-coordinate map where rho+x and pz-related divisions lose FP32 bits.
using pass_high_precision_t = double;
#else
using pass_bend_particle_t = double;
using pass_real_t = double;
using pass_high_precision_t = double;
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
    pass_real_t& x,
    pass_real_t& px,
    pass_real_t& y,
    pass_real_t& py,
    pass_real_t& z,
    pass_real_t dp,
    int& tag,
    float* lp,
    int* lt,
    int i,
    double L,
    pass_real_t beta0_over_beta,
    pass_real_t reference_beta_gamma,
    pass_real_t s0,
    int turn
) {
    // Match the CPU epsilon guard; exact equality is unsafe for floating-point
    // slice lengths (especially after Yoshida scaling).
    if (fabs(L) < PASS_EPS || tag <= 0)
        return tag > 0;
    pass_real_t momentum_ratio = (pass_real_t)1 + dp;
    pass_real_t pz_squared = momentum_ratio * momentum_ratio - px * px - py * py;
    if (!(pz_squared > (pass_real_t)0)) {
        tag = -abs(tag);
        lp[i] = (float)s0;
        lt[i] = turn;
        return false;
    }
    pass_real_t pz = sqrt(pz_squared);
    pass_real_t inv_pz = (pass_real_t)1 / pz;
    pass_real_t inv_gamma_sq = (pass_real_t)1 / ((pass_real_t)1 + reference_beta_gamma * reference_beta_gamma);
    pass_real_t transverse_momentum_squared = px * px + py * py;
    pass_real_t energy_ratio = sqrt(inv_gamma_sq + ((pass_real_t)1 - inv_gamma_sq) * momentum_ratio * momentum_ratio);
    // Stable even when the physical time slip is much smaller than float epsilon.
    pass_real_t slip = (dp * ((pass_real_t)2 + dp) * inv_gamma_sq - transverse_momentum_squared) / (pz * (pz + energy_ratio));
    x += L * px * inv_pz;
    y += L * py * inv_pz;
    z += L * slip;
    return true;
}

__device__ PASS_DIPOLE_INLINE bool d_yrot(
    pass_real_t& x,
    pass_real_t& px,
    pass_real_t& y,
    pass_real_t& z,
    pass_real_t py,
    pass_real_t dp,
    int& tag,
    float* lp,
    int* lt,
    int i,
    pass_real_t a,
    pass_real_t sin_angle,
    pass_real_t cos_angle,
    pass_real_t beta0,
    pass_real_t time_factor,
    pass_real_t s0,
    int turn
) {
    if (fabs(a) < PASS_EPS || tag <= 0)
        return tag > 0;
    pass_real_t momentum_ratio = (pass_real_t)1 + dp;
    pass_real_t pz_squared = momentum_ratio * momentum_ratio - px * px - py * py;
    if (!(pz_squared > (pass_real_t)0)) {
        tag = -abs(tag);
        lp[i] = (float)s0;
        lt[i] = turn;
        return false;
    }
    pass_real_t pz = sqrt(pz_squared);
    pass_real_t tan_angle = sin_angle / cos_angle;
    pass_real_t rotation_denominator = (pass_real_t)1 + tan_angle * px / pz;
    if (fabs(rotation_denominator) < PASS_EPS)
        rotation_denominator = PASS_EPS;
    pass_real_t xold = x;
    x = xold / (cos_angle * rotation_denominator);
    px = cos_angle * px - sin_angle * pz;
    y = y - tan_angle * xold * py / (pz * rotation_denominator);
    z = z + beta0 * tan_angle * xold * time_factor / (pz * rotation_denominator);
    return true;
}

__device__ PASS_DIPOLE_INLINE bool d_fringe(
    pass_real_t& x,
    pass_real_t& px,
    pass_real_t& y,
    pass_real_t& py,
    pass_real_t& z,
    pass_real_t dp,
    int& tag,
    float* lp,
    int* lt,
    int i,
    pass_real_t fint,
    pass_real_t hgap,
    pass_real_t k0,
    pass_real_t beta0,
    pass_real_t time_factor,
    pass_real_t s0,
    int turn
) {
    if (fabs(k0) < PASS_EPS || tag <= 0)
        return tag > 0;
    pass_real_t dipole_strength = k0;
    pass_real_t fh = hgap * fint;
    pass_real_t fsad = (fh > PASS_EPS) ? (pass_real_t)1 / (72 * fh) : (pass_real_t)0;
    pass_real_t momentum_ratio = (pass_real_t)1 + dp;
    pass_real_t pz_squared = momentum_ratio * momentum_ratio - px * px - py * py;
    if (!(pz_squared > (pass_real_t)0)) {
        tag = -abs(tag);
        lp[i] = (float)s0;
        lt[i] = turn;
        return false;
    }
    pass_real_t pz = sqrt(pz_squared);
    pass_real_t inv_pz = (pass_real_t)1 / pz;
    pass_real_t inverse_abs_momentum_ratio = (pass_real_t)1 / sqrt(momentum_ratio * momentum_ratio);
    time_factor = -time_factor;
    pass_real_t fringe_linear_strength = dipole_strength * fh * 2;
    pass_real_t fringe_cubic_strength = dipole_strength * dipole_strength * fsad * inverse_abs_momentum_ratio;
    pass_real_t slope_x = px * inv_pz;
    pass_real_t slope_y = py * inv_pz;
    pass_real_t slope_xy = slope_x * slope_y;
    pass_real_t one_plus_slope_y_squared = (pass_real_t)1 + slope_y * slope_y;
    pass_real_t slope_x_squared = slope_x * slope_x;
    pass_real_t inv_one_plus_slope_y_squared = (pass_real_t)1 / one_plus_slope_y_squared;
    pass_real_t fringe_angle = atan(slope_x * inv_one_plus_slope_y_squared) -
                               fringe_linear_strength * ((pass_real_t)1 + slope_x_squared * ((pass_real_t)1 + one_plus_slope_y_squared)) * pz;
    pass_real_t sin_fringe_angle, cos_fringe_angle;
    sincos(fringe_angle, &sin_fringe_angle, &cos_fringe_angle);
    if (fabs(cos_fringe_angle) < PASS_EPS)
        cos_fringe_angle = PASS_EPS;
    pass_real_t fringe_second_derivative = dipole_strength / (cos_fringe_angle * cos_fringe_angle);
    pass_real_t fringe_first_derivative = fringe_second_derivative /
                                          ((pass_real_t)1 + (slope_x * inv_one_plus_slope_y_squared) * (slope_x * inv_one_plus_slope_y_squared)) *
                                          inv_one_plus_slope_y_squared;
    pass_real_t fringe_third_derivative = fringe_second_derivative * fringe_linear_strength;
    pass_real_t fringe_x_derivative =
        fringe_first_derivative - fringe_third_derivative * 2 * slope_x * ((pass_real_t)1 + one_plus_slope_y_squared) * pz;
    pass_real_t fringe_xy_derivative =
        -2 * fringe_first_derivative * slope_xy * inv_one_plus_slope_y_squared - fringe_third_derivative * 2 * slope_x * slope_xy * pz;
    pass_real_t fringe_y_derivative = -fringe_third_derivative * ((pass_real_t)1 + slope_x_squared * ((pass_real_t)1 + one_plus_slope_y_squared));
    pass_real_t x_kick =
        fringe_x_derivative * ((pass_real_t)1 + slope_x_squared) * inv_pz + fringe_xy_derivative * slope_xy * inv_pz - fringe_y_derivative * slope_x;
    pass_real_t y_kick =
        fringe_x_derivative * slope_xy * inv_pz + fringe_xy_derivative * one_plus_slope_y_squared * inv_pz - fringe_y_derivative * slope_y;
    pass_real_t z_kick = fringe_x_derivative * time_factor * slope_x * inv_pz * inv_pz +
                         fringe_xy_derivative * time_factor * slope_y * inv_pz * inv_pz - fringe_y_derivative * time_factor * inv_pz;
    pass_real_t y_discriminant = (pass_real_t)1 - 2 * y_kick * y;
    if (y_discriminant < (pass_real_t)0)
        y_discriminant = 0;
    pass_real_t new_y = 2 * y / ((pass_real_t)1 + sqrt(y_discriminant));
    x += (pass_real_t)0.5 * x_kick * new_y * new_y;
    py -= 4 * fringe_cubic_strength * new_y * new_y * new_y + dipole_strength * (sin_fringe_angle / cos_fringe_angle) * new_y;
    y = new_y;
    z += beta0 * ((pass_real_t)0.5 * z_kick * new_y * new_y +
                  fringe_cubic_strength * new_y * new_y * new_y * new_y * (inverse_abs_momentum_ratio * inverse_abs_momentum_ratio) * time_factor);
    return true;
}

__device__ PASS_DIPOLE_INLINE bool d_wedge(
    pass_real_t& x,
    pass_real_t& px,
    pass_real_t& y,
    pass_real_t& z,
    pass_real_t py,
    pass_real_t dp,
    int& tag,
    float* lp,
    int* lt,
    int i,
    pass_real_t theta,
    pass_real_t k0,
    pass_real_t sin_angle,
    pass_real_t cos_angle,
    pass_real_t beta0,
    pass_real_t beta0_over_beta,
    pass_real_t time_factor,
    pass_real_t s0,
    int turn
) {
    if (tag <= 0)
        return false;
    if (fabs(k0) < PASS_EPS)
        return d_yrot(x, px, y, z, py, dp, tag, lp, lt, i, theta, sin_angle, cos_angle, beta0, time_factor, s0, turn);
    pass_real_t momentum_ratio = (pass_real_t)1 + dp;
    pass_real_t pz_squared = momentum_ratio * momentum_ratio - px * px - py * py;
    if (!(pz_squared > (pass_real_t)0)) {
        tag = -abs(tag);
        lp[i] = (float)s0;
        lt[i] = turn;
        return false;
    }
    pass_real_t pz = sqrt(pz_squared);
    pass_real_t sin_theta = sin_angle, cos_theta = cos_angle;
    pass_real_t sin_2theta = (pass_real_t)2 * sin_theta * cos_theta;
    pass_real_t new_px = px * cos_theta + (pz - k0 * x) * sin_theta;
    pass_real_t new_pz_squared = momentum_ratio * momentum_ratio - new_px * new_px - py * py;
    if (new_pz_squared < PASS_EPS)
        new_pz_squared = PASS_EPS;
    pass_real_t new_pz = sqrt(new_pz_squared);
    pass_real_t denominator = new_pz + pz * cos_theta - px * sin_theta;
    if (fabs(denominator) < PASS_EPS)
        denominator = PASS_EPS;
    pass_real_t new_x = x * cos_theta + (x * px * sin_2theta + sin_theta * sin_theta * ((pass_real_t)2 * x * pz - k0 * x * x)) / denominator;
    pass_real_t inv_transverse_momentum = (pass_real_t)1 / sqrt(momentum_ratio * momentum_ratio - py * py);
    pass_real_t phase_advance = asin(fmax((pass_real_t)-1, fmin((pass_real_t)1, inv_transverse_momentum * px))) -
                                asin(fmax((pass_real_t)-1, fmin((pass_real_t)1, inv_transverse_momentum * new_px)));
    pass_real_t safe_strength = (fabs(k0) > PASS_EPS) ? k0 : PASS_EPS;
    x = new_x;
    px = new_px;
    y += py * (theta + phase_advance) / safe_strength;
    z -= momentum_ratio * (theta + phase_advance) / safe_strength * beta0_over_beta;
    return true;
}

__device__ PASS_DIPOLE_INLINE void d_kick(
    pass_real_t& px,
    pass_real_t& z,
    pass_real_t x,
    pass_real_t dp,
    double L,
    pass_real_t h,
    pass_real_t k0,
    pass_real_t beta0,
    pass_real_t beta0_over_beta
) {
    pass_real_t momentum_ratio = (pass_real_t)1 + dp;
    px += L * (h * momentum_ratio - k0 - k0 * h * x);
    z -= L * beta0_over_beta * h * x;
}

__device__ PASS_DIPOLE_INLINE void d_polar_trig(
    pass_real_t angle,
    pass_real_t& sine,
    pass_real_t& cosine,
    pass_real_t& sin_half_angle
) {
    sincos(angle, &sine, &cosine);
    sin_half_angle = sin((pass_real_t)0.5 * angle);
}

__device__ PASS_DIPOLE_INLINE bool d_polar(
    pass_real_t& x,
    pass_real_t& px,
    pass_real_t& y,
    pass_real_t& z,
    pass_real_t py,
    pass_real_t dp,
    int& tag,
    float* lp,
    int* lt,
    int i,
    double L,
    pass_real_t h,
    pass_real_t rho,
    pass_real_t sin_bend_angle,
    pass_real_t cos_bend_angle,
    pass_real_t sin_half_bend_angle,
    pass_real_t beta0,
    pass_real_t beta0_over_beta,
    pass_real_t reference_beta_gamma,
    pass_real_t s0,
    int turn
) {
    if (fabs(L) < PASS_EPS || tag <= 0)
        return tag > 0;
    if (fabs(h) < PASS_EPS)
        return d_drift(x, px, y, py, z, dp, tag, lp, lt, i, L, beta0_over_beta, reference_beta_gamma, s0, turn);
    pass_high_precision_t xc = (pass_high_precision_t)x;
    pass_high_precision_t pxc = (pass_high_precision_t)px;
    pass_high_precision_t pyc = (pass_high_precision_t)py;
    pass_high_precision_t dpc = (pass_high_precision_t)dp;
    pass_high_precision_t momentum_ratio = (pass_high_precision_t)1 + dpc;
    pass_high_precision_t pz_squared = momentum_ratio * momentum_ratio - pxc * pxc - pyc * pyc;
    if (!(pz_squared > (pass_real_t)0)) {
        tag = -abs(tag);
        lp[i] = (float)s0;
        lt[i] = turn;
        return false;
    }
    pass_high_precision_t pz = sqrt(pz_squared);
    pass_high_precision_t inv_pz = (pass_high_precision_t)1 / pz;
    pass_high_precision_t slope_x = pxc * inv_pz;
    pass_high_precision_t sin_angle = (pass_high_precision_t)sin_bend_angle;
    pass_high_precision_t cos_angle = (pass_high_precision_t)cos_bend_angle;
    pass_high_precision_t rhoc = (pass_high_precision_t)rho;
    pass_high_precision_t denominator = cos_angle - sin_angle * slope_x;
    if (fabs(denominator) < (pass_high_precision_t)PASS_EPS)
        denominator = (pass_high_precision_t)PASS_EPS;
    pass_high_precision_t path_length_factor = (pass_high_precision_t)1 / denominator;
    pass_high_precision_t polar_path_length = (xc + rhoc) * sin_angle * inv_pz * path_length_factor;
    // Match the CPU half-angle identity without subtracting nearby values
    // or dividing by 1 + cos(theta). Reuse the precomputed half-angle sine.
    pass_high_precision_t sin_half_angle_high_precision = (pass_high_precision_t)sin_half_bend_angle;
    pass_high_precision_t one_minus_cos = (pass_high_precision_t)2 * sin_half_angle_high_precision * sin_half_angle_high_precision;
    pass_high_precision_t new_x = (xc + rhoc * (one_minus_cos + sin_angle * slope_x)) * path_length_factor;
    pass_high_precision_t new_px = cos_angle * pxc + sin_angle * pz;
    pass_high_precision_t new_y = (pass_high_precision_t)y + polar_path_length * pyc;
    x = (pass_real_t)new_x;
    px = (pass_real_t)new_px;
    y = (pass_real_t)new_y;
    pass_high_precision_t reference_bg = (pass_high_precision_t)reference_beta_gamma;
    pass_high_precision_t inv_gamma_sq = (pass_high_precision_t)1 / ((pass_high_precision_t)1 + reference_bg * reference_bg);
    pass_high_precision_t beta_ratio_squared_change = -inv_gamma_sq * dpc * ((pass_high_precision_t)2 + dpc) / (momentum_ratio * momentum_ratio);
    pass_high_precision_t beta0_over_beta_minus_one =
        beta_ratio_squared_change / (sqrt((pass_high_precision_t)1 + beta_ratio_squared_change) + (pass_high_precision_t)1);
    pass_high_precision_t particle_path_length = momentum_ratio * polar_path_length;
    z += (pass_real_t)(((pass_high_precision_t)L - particle_path_length) - particle_path_length * beta0_over_beta_minus_one);
    return true;
}

__device__ PASS_DIPOLE_INLINE bool d_rkr_drift(
    pass_real_t& x,
    pass_real_t& px,
    pass_real_t& y,
    pass_real_t& z,
    pass_real_t py,
    pass_real_t dp,
    int& tag,
    float* lp,
    int* lt,
    int i,
    double L,
    pass_real_t h,
    pass_real_t k0,
    pass_real_t beta0,
    pass_real_t beta0_over_beta,
    pass_real_t reference_beta_gamma,
    pass_real_t rho,
    pass_real_t sin_first,
    pass_real_t cos_first,
    pass_real_t sin_half_first,
    pass_real_t sin_middle,
    pass_real_t cos_middle,
    pass_real_t sin_half_middle,
    pass_real_t s0,
    int turn
) {
    if (fabs(L) < PASS_EPS || tag <= 0)
        return tag > 0;
    if (fabs(h) < PASS_EPS)
        return d_drift(x, px, y, py, z, dp, tag, lp, lt, i, L, beta0_over_beta, reference_beta_gamma, s0, turn);
    const double yoshida_z1 = pass_yoshida_z1;
    const double yoshida_z0 = pass_yoshida_z0;
    const double first_polar_drift = yoshida_z1 * L * (pass_real_t)0.5;
    const double middle_polar_drift = (yoshida_z1 + yoshida_z0) * L * (pass_real_t)0.5;
    if (!d_polar(x, px, y, z, py, dp, tag, lp, lt, i, first_polar_drift, h, rho, sin_first, cos_first, sin_half_first, beta0, beta0_over_beta,
                 reference_beta_gamma, s0, turn))
        return false;
    px -= yoshida_z1 * k0 * L;
    if (!d_polar(x, px, y, z, py, dp, tag, lp, lt, i, middle_polar_drift, h, rho, sin_middle, cos_middle, sin_half_middle, beta0, beta0_over_beta,
                 reference_beta_gamma, s0, turn))
        return false;
    px -= yoshida_z0 * k0 * L;
    if (!d_polar(x, px, y, z, py, dp, tag, lp, lt, i, middle_polar_drift, h, rho, sin_middle, cos_middle, sin_half_middle, beta0, beta0_over_beta,
                 reference_beta_gamma, s0, turn))
        return false;
    px -= yoshida_z1 * k0 * L;
    return d_polar(x, px, y, z, py, dp, tag, lp, lt, i, first_polar_drift, h, rho, sin_first, cos_first, sin_half_first, beta0, beta0_over_beta,
                   reference_beta_gamma, s0, turn);
}

__device__ PASS_DIPOLE_INLINE bool d_dkd(
    pass_real_t& x,
    pass_real_t& px,
    pass_real_t& y,
    pass_real_t& z,
    pass_real_t py,
    pass_real_t dp,
    int& tag,
    float* lp,
    int* lt,
    int i,
    double L,
    pass_real_t h,
    pass_real_t k0,
    pass_real_t beta0,
    pass_real_t beta0_over_beta,
    pass_real_t reference_beta_gamma,
    pass_real_t s0,
    int turn
) {
    if (fabs(L) < PASS_EPS || tag <= 0)
        return tag > 0;
    if (!d_drift(x, px, y, py, z, dp, tag, lp, lt, i, L * (pass_real_t)0.5, beta0_over_beta, reference_beta_gamma, s0, turn))
        return false;
    d_kick(px, z, x, dp, L, h, k0, beta0, beta0_over_beta);
    return d_drift(x, px, y, py, z, dp, tag, lp, lt, i, L * (pass_real_t)0.5, beta0_over_beta, reference_beta_gamma, s0, turn);
}

extern "C" __global__ void track_sbend(
    pass_bend_particle_t* x,
    pass_bend_particle_t* px,
    pass_bend_particle_t* y,
    pass_bend_particle_t* py,
    pass_bend_particle_t* z,
    const pass_bend_particle_t* dp,
    int* tag,
    float* lp,
    int* lt,
    int start,
    int end,
    pass_real_t beta0,
    pass_real_t reference_beta_gamma,
    pass_real_t time_factor_sq_const,
    pass_real_t rho_const,
    double ds_const,
    double L,
    pass_real_t k0l,
    pass_real_t h,
    pass_real_t k0,
    pass_real_t e1,
    pass_real_t e2,
    pass_real_t e1_s,
    pass_real_t e1_c,
    pass_real_t e2_s,
    pass_real_t e2_c,
    pass_real_t hgap,
    pass_real_t fint,
    pass_real_t fintx,
    pass_real_t s0,
    int turn,
    int slices,
    int integrator,
    int model,
    int thin
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i >= end || tag[i] <= 0)
        return;
    pass_real_t xi = x[i], pxi = px[i], yi = y[i], pyi = py[i];
    // Accumulate time-of-flight increments without the potentially large entry z.
    pass_real_t zi = 0, dpi = dp[i];
    int ti = tag[i];
    bool alive = true;
#if PASS_DIPOLE_THIN
    {
        pxi -= k0l;
        x[i] = xi;
        px[i] = pxi;
        y[i] = yi;
        py[i] = pyi;
        tag[i] = ti;
        return;
    }
#else
    pass_real_t momentum_ratio_i = (pass_real_t)1 + dpi;
    pass_real_t particle_bg = momentum_ratio_i * reference_beta_gamma;
    pass_real_t particle_beta = particle_bg / sqrt((pass_real_t)1 + particle_bg * particle_bg);
    pass_real_t beta0_over_beta = beta0 / particle_beta;
    pass_real_t rho = rho_const;
    const double ds = ds_const;
    pass_real_t time_factor = sqrt(momentum_ratio_i * momentum_ratio_i + time_factor_sq_const);
#if PASS_DIPOLE_MODEL == 1
    // RKR uses the same polar angles for both drifts around each kick.  Keep
    // these values per particle, but evaluate each distinct angle only once.
    pass_real_t rkr_sf1 = 0, rkr_cf1 = 1, rkr_sm1 = 0, rkr_cm1 = 1;
    pass_real_t rkr_sf0 = 0, rkr_cf0 = 1, rkr_sm0 = 0, rkr_cm0 = 1;
    pass_real_t rkr_shf1 = 0, rkr_shm1 = 0, rkr_shf0 = 0, rkr_shm0 = 0;
    if (fabs(h) > PASS_EPS) {
        const double rkr_base = h * ds * (pass_real_t)0.25;
        const double z1 = pass_yoshida_z1;
        const double z0 = pass_yoshida_z0;
        if (PASS_DIPOLE_INTEGRATOR == 0) {
            d_polar_trig(rkr_base * z1, rkr_sf1, rkr_cf1, rkr_shf1);
            d_polar_trig(rkr_base * (z1 + z0), rkr_sm1, rkr_cm1, rkr_shm1);
        } else {
            d_polar_trig(rkr_base * z1 * z1, rkr_sf1, rkr_cf1, rkr_shf1);
            d_polar_trig(rkr_base * z1 * (z1 + z0), rkr_sm1, rkr_cm1, rkr_shm1);
            d_polar_trig(rkr_base * z0 * z1, rkr_sf0, rkr_cf0, rkr_shf0);
            d_polar_trig(rkr_base * z0 * (z1 + z0), rkr_sm0, rkr_cm0, rkr_shm0);
        }
    }
#endif
    if (fabs(e1) > PASS_EPS)
        alive = d_yrot(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, -e1, e1_s, e1_c, beta0, time_factor, s0, turn);
    if (alive && fabs(k0) > PASS_EPS)
        alive = d_fringe(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i, fint, hgap, k0, beta0, time_factor, s0, turn);
    if (alive && fabs(e1) > PASS_EPS)
        alive = d_wedge(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, -e1, k0, e1_s, e1_c, beta0, beta0_over_beta, time_factor, s0, turn);
    for (int slice_index = 0; slice_index < slices && alive; ++slice_index) {
#if PASS_DIPOLE_MODEL == 0
#if PASS_DIPOLE_INTEGRATOR == 0
        alive = d_dkd(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, ds, h, k0, beta0, beta0_over_beta, reference_beta_gamma, s0, turn);
#else
        alive = d_dkd(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, ds * pass_yoshida_z1, h, k0, beta0, beta0_over_beta, reference_beta_gamma, s0, turn);
        if (alive)
            alive =
                d_dkd(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, ds * pass_yoshida_z0, h, k0, beta0, beta0_over_beta, reference_beta_gamma, s0, turn);
        if (alive)
            alive =
                d_dkd(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, ds * pass_yoshida_z1, h, k0, beta0, beta0_over_beta, reference_beta_gamma, s0, turn);
#endif
#else
#if PASS_DIPOLE_INTEGRATOR == 0
        // RKR outer step; internal polar drift is Yoshida-4.
        double d = ds;
        if (alive)
            alive = d_rkr_drift(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, d * (pass_real_t)0.5, h, k0, beta0, beta0_over_beta, reference_beta_gamma,
                                rho, rkr_sf1, rkr_cf1, rkr_shf1, rkr_sm1, rkr_cm1, rkr_shm1, s0, turn);
        if (alive)
            pxi -= d * k0 * h * xi;
        if (alive)
            alive = d_rkr_drift(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, d * (pass_real_t)0.5, h, k0, beta0, beta0_over_beta, reference_beta_gamma,
                                rho, rkr_sf1, rkr_cf1, rkr_shf1, rkr_sm1, rkr_cm1, rkr_shm1, s0, turn);
#else
        // RKR outer step; internal polar drift is Yoshida-4.
        double d = ds * pass_yoshida_z1;
        if (alive)
            alive = d_rkr_drift(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, d * (pass_real_t)0.5, h, k0, beta0, beta0_over_beta, reference_beta_gamma,
                                rho, rkr_sf1, rkr_cf1, rkr_shf1, rkr_sm1, rkr_cm1, rkr_shm1, s0, turn);
        if (alive)
            pxi -= d * k0 * h * xi;
        if (alive)
            alive = d_rkr_drift(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, d * (pass_real_t)0.5, h, k0, beta0, beta0_over_beta, reference_beta_gamma,
                                rho, rkr_sf1, rkr_cf1, rkr_shf1, rkr_sm1, rkr_cm1, rkr_shm1, s0, turn);

        if (alive) {
            d = ds * pass_yoshida_z0;
            alive = d_rkr_drift(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, d * (pass_real_t)0.5, h, k0, beta0, beta0_over_beta, reference_beta_gamma,
                                rho, rkr_sf0, rkr_cf0, rkr_shf0, rkr_sm0, rkr_cm0, rkr_shm0, s0, turn);
            if (alive) {
                pxi -= d * k0 * h * xi;
                alive = d_rkr_drift(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, d * (pass_real_t)0.5, h, k0, beta0, beta0_over_beta,
                                    reference_beta_gamma, rho, rkr_sf0, rkr_cf0, rkr_shf0, rkr_sm0, rkr_cm0, rkr_shm0, s0, turn);
            }
            d = ds * pass_yoshida_z1;
            if (alive)
                alive = d_rkr_drift(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, d * (pass_real_t)0.5, h, k0, beta0, beta0_over_beta,
                                    reference_beta_gamma, rho, rkr_sf1, rkr_cf1, rkr_shf1, rkr_sm1, rkr_cm1, rkr_shm1, s0, turn);
            if (alive) {
                pxi -= d * k0 * h * xi;
                alive = d_rkr_drift(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, d * (pass_real_t)0.5, h, k0, beta0, beta0_over_beta,
                                    reference_beta_gamma, rho, rkr_sf1, rkr_cf1, rkr_shf1, rkr_sm1, rkr_cm1, rkr_shm1, s0, turn);
            }
        }
#endif
#endif
    }
    if (fabs(e2) > PASS_EPS)
        alive = d_wedge(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, -e2, k0, e2_s, e2_c, beta0, beta0_over_beta, time_factor, s0, turn);
    if (alive && fabs(k0) > PASS_EPS)
        alive = d_fringe(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i, fintx, hgap, -k0, beta0, time_factor, s0, turn);
    if (alive && fabs(e2) > PASS_EPS)
        alive = d_yrot(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, -e2, e2_s, e2_c, beta0, time_factor, s0, turn);
    x[i] = xi;
    px[i] = pxi;
    y[i] = yi;
    py[i] = pyi;
    z[i] += zi;
    tag[i] = ti;
#endif
}
'''


@lru_cache(maxsize=None)
def _get_fused_kernel(dtype, model=0, integrator=0, thin=0):
    """Compile the single-launch map once per particle precision."""
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU SBend tracking requires the optional 'cuda' dependencies.") from exc
    dtype = np.dtype(dtype)
    options = (
        "--std=c++14",
        f"-DPASS_USE_FLOAT={int(dtype == np.dtype(np.float32))}",
        f"-DPASS_DIPOLE_MODEL={int(model)}",
        f"-DPASS_DIPOLE_INTEGRATOR={int(integrator)}",
        f"-DPASS_DIPOLE_THIN={int(thin)}",
    )
    if dtype == np.dtype(np.float64):
        options += ("--maxrregcount=160", )
    return cp.RawKernel(CUDA_REAL_PREAMBLE + DIPOLE_BODY, "track_sbend", options=options)


def launch_dipole(element, sim):
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU SBend tracking requires the optional 'cuda' dependencies.") from exc
    beam = sim.beams[element.beam_id]
    p = beam.particles
    real = np.float64 if element.model == "rot-kick-rot" else p.real
    threads = 256
    turn = sim.state.turn
    model = 0 if element.model == "drift-kick-drift-exact" else 1
    integrator = 0 if element.integrator == "uniform" else 1
    thin = 0 if element.is_thick else 1
    kernel = _get_fused_kernel(p.dtype.str, model, integrator, thin)
    for b in beam.bunches:
        n = b.end_idx - b.start_idx
        if n > 0:
            blocks = (n + threads - 1) // threads
            args = (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, p.lost_position, p.lost_turn, np.int32(b.start_idx), np.int32(b.end_idx), real(b.beta),
                    real(b.beta * b.gamma), real(1.0 / ((1.0 + (b.beta * b.gamma)**2) * b.beta**2)),
                    real(1.0 / element.h if abs(element.h) > const.eps else 0.0), np.float64(element.length / element.num_slice),
                    np.float64(element.length), real(element.k0l), real(element.h), real(element.k0), real(element.e1), real(element.e2),
                    real(np.sin(-element.e1)), real(np.cos(-element.e1)), real(np.sin(-element.e2)), real(np.cos(-element.e2)), real(element.hgap),
                    real(element.fint), real(element.fintx), real(element.s), np.int32(turn), np.int32(element.num_slice),
                    np.int32(0 if element.integrator == "uniform" else 1), np.int32(0 if element.model == "drift-kick-drift-exact" else 1),
                    np.int32(0 if element.is_thick else 1))
            kernel((blocks, ), (threads, ), args)
        if n > 0:
            from PASS.utils.aperture import check_aperture_gpu
            check_aperture_gpu(beam, b, element.aperture_type, element.aperture_value, element.s, turn)
        if abs(element.length) >= const.eps:
            b.t0 += element.length / (b.beta * const.c)
