import logging

import numpy as np

from PASS.commands.command import Command
from PASS.commands.element.error import FieldErrors
from PASS.utils.slicing import print_element_slicing, configure_element_slicing, run_body_slices
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.aperture import check_aperture_cpu
from PASS.commands.element.multipole import launch_multipole

logger = logging.getLogger(__name__)


@Command.register("octupole")
class Octupole(Command):
    """Track normal and skew octupoles with thin kicks or drift-kick-drift slices."""

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = kwargs["length (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]
        self.field_errors = FieldErrors(kwargs)

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
            logger.warning(
                f"Octupole {self.cmd_name} has both normal and skew components (k3l={self.k3l}, k3sl={self.k3sl}). It will act as a combined octupole."
            )

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

        configure_element_slicing(self, sim, kwargs)
        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, Length={self.length:.4f}, "
                    f"IsThick={self.is_thick}, K3L={self.k3l:.6f}, K3SL={self.k3sl:.6f}, "
                    f"NumSlice={self.num_slice:d}, Integrator={self.integrator:s}, "
                    f"ApertureType={self.aperture_type:s}, ApertureValue={self.aperture_value}")
        print_element_slicing(self)
        set_normal_logging()

    def execute_cpu(self, sim):
        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        turn = sim.state.turn

        for i, bunch in enumerate(bunches):
            self._track_octupole_cpu(beam, bunch, turn)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)
        return True

    def execute_gpu(self, sim):
        if self.field_errors.active:
            from PASS.commands.element.error import _track_field_errors_gpu
            return _track_field_errors_gpu(self, sim)
        if self._sc_nodes:
            from PASS.utils.slicing import execute_element_body_gpu
            return execute_element_body_gpu(self, sim)
        if self.is_thick:
            all_zero = (abs(self.k3l) < const.eps and abs(self.k3sl) < const.eps)
            mode = 2 if all_zero else 1
            knl = np.array([0.0, 0.0, 0.0, self.k3], dtype=np.float64)
            ksl = np.array([0.0, 0.0, 0.0, self.k3s], dtype=np.float64)
        else:
            mode = 0
            knl = np.array([0.0, 0.0, 0.0, self.k3l], dtype=np.float64)
            ksl = np.array([0.0, 0.0, 0.0, self.k3sl], dtype=np.float64)
        launch_multipole(self, sim, knl, ksl, np.array([1.0, 1.0, 0.5, 1.0 / 6.0], dtype=np.float64), mode)
        return True

    def _track_octupole_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):

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

        mask = (tag > 0).astype(np.float64)

        if not self.is_thick:
            self._octupole_kick_cpu(self.k3l, self.k3sl, x, px, y, py, tag, mask, chi)
            self.field_errors.kick_cpu(x, px, y, py, tag)
            return

        if self._sc_nodes:
            step = self._dkd_step_cpu if self.integrator == "uniform" else self._dkd_yoshida4_cpu

            def transport(ds, on_center):
                step(x, px, y, py, z, dp, tag, mask, ds, self.k3, self.k3s, chi, beta0, on_center=on_center)

            run_body_slices(self, beam, bunch, turn, transport)
            return

        if (abs(self.k3l) < const.eps and abs(self.k3sl) < const.eps) and not self.field_errors.active:
            self._drift_exact_cpu(self.length, x, px, y, py, z, dp, tag, mask, beta0)
        else:
            ds = self.length / self.num_slice
            for _ in range(self.num_slice):
                if self.integrator == "uniform":
                    self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds, self.k3, self.k3s, chi, beta0)
                elif self.integrator == "yoshida4":
                    self._dkd_yoshida4_cpu(x, px, y, py, z, dp, tag, mask, ds, self.k3, self.k3s, chi, beta0)

        newly_lost = alive_before & (tag < 0)
        if np.any(newly_lost):
            lost_position = p.lost_position[start:end]
            lost_turn = p.lost_turn[start:end]
            lost_position[newly_lost] = self.s
            lost_turn[newly_lost] = turn

    def _dkd_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask, ds, k3, k3s, chi, beta0, on_center=None):
        """Compose three drift-kick-drift steps with Yoshida coefficients."""
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, k3, k3s, chi, beta0)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z0, k3, k3s, chi, beta0, on_center=on_center)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, k3, k3s, chi, beta0)

    def _dkd_step_cpu(self, x, px, y, py, z, dp, tag, mask, ds, k3, k3s, chi, beta0, on_center=None):
        """Apply one drift-kick-drift step; Yoshida composition may use negative ds."""
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._octupole_kick_cpu(k3 * ds, k3s * ds, x, px, y, py, tag, mask, chi)
        self.field_errors.kick_cpu(x, px, y, py, tag, ds / self.length)
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

        # A particle that becomes invalid at this drift exits immediately;
        # do not transport it with the stale entry mask.
        L_mask = L * (alive & valid)

        x += L_mask * px * inv_pz
        y += L_mask * py * inv_pz
        z += L_mask * slip

    def _octupole_kick_cpu(self, k3l_eff, k3sl_eff, x, px, y, py, tag, mask, chi):
        """Apply normal and skew kicks using integrated octupole strengths."""
        if abs(k3l_eff) < const.eps and abs(k3sl_eff) < const.eps:
            return

        active = (tag > 0).astype(mask.dtype, copy=False)
        k3l_mask = k3l_eff * active

        x2 = x * x
        y2 = y * y

        re_c3 = x * (x2 - 3.0 * y2)
        im_c3 = y * (3.0 * x2 - y2)

        if abs(k3l_eff) > const.eps:
            chi_k3l = chi * k3l_mask / 6.0
            px -= chi_k3l * re_c3
            py += chi_k3l * im_c3

        if abs(k3sl_eff) > const.eps:
            k3sl_mask = k3sl_eff * active
            chi_k3sl = chi * k3sl_mask / 6.0
            px += chi_k3sl * im_c3
            py += chi_k3sl * re_c3
