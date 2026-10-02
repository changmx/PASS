import logging

import numpy as np

from PASS.commands.command import Command
from PASS.commands.element.error import AlignmentErrors, FieldErrors
from PASS.commands.element.ramping import configure_magnet_ramping, refresh_magnet_strengths, update_magnet_strengths
from PASS.commands.element.magnet_maps import _GpuBody, _apply_multipole_kick_cpu, _launch_multipole_bunch, _prepare_multipole_coefficients
from PASS.utils.aperture import check_aperture_cpu, check_aperture_gpu
from PASS.utils.slicing import print_element_slicing, configure_element_slicing, run_body_slices, transport_dkd
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const

logger = logging.getLogger(__name__)


@Command.register("multipole")
class Multipole(Command):
    """Track normal and skew multipoles using Horner evaluation.

    A zero-length element applies one integrated kick. Thick elements use
    exact drift-kick-drift slices with uniform or Yoshida integration."""

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = kwargs["length (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]
        self.field_errors = FieldErrors(kwargs)
        self.alignment_errors = AlignmentErrors(kwargs)

        configure_magnet_ramping(self, kwargs, order=None)

        if self.length < 0.0:
            raise ValueError(f"The length of Multipole {self.cmd_name} is {self.length}, which should be >= 0")
        if self.length > const.eps:
            self.is_thick = True
        else:
            self.is_thick = False

        knl_list = kwargs.get("kil", [])
        ksl_list = kwargs.get("kisl", [])

        if not isinstance(knl_list, (list, np.ndarray)):
            raise ValueError(f"KiL of {self.cmd_name} must be a list, but got {type(knl_list)}")
        if not isinstance(ksl_list, (list, np.ndarray)):
            raise ValueError(f"KiSL of {self.cmd_name} must be a list, but got {type(ksl_list)}")

        if self.ramp is not None and len(knl_list) == 0 and len(ksl_list) == 0:
            knl_list = np.zeros(self.ramp.max_order + 1)
        self.knl = np.array(knl_list, dtype=np.float64)
        self.ksl = np.array(ksl_list, dtype=np.float64)

        # Order = max(len(knl), len(ksl)) - 1; pad the shorter array with zeros
        len_n = len(self.knl)
        len_s = len(self.ksl)
        if len_n == 0 and len_s == 0 and not self.field_errors.active:
            raise ValueError(f"Multipole {self.cmd_name} has empty KiL and KiSL. At least one component is required.")

        if len_n > len_s:
            self.ksl = np.pad(self.ksl, (0, len_n - len_s), mode='constant')
        elif len_n < len_s:
            self.knl = np.pad(self.knl, (0, len_s - len_n), mode='constant')

        refresh_magnet_strengths(self)

        all_zero = np.all(self.knl == 0) and np.all(self.ksl == 0)
        if all_zero and self.ramp is None and not self.field_errors.active:
            logger.warning(f"Multipole {self.cmd_name} has zero integrated strength (all knl/ksl are zero). It will act as a pure drift.")

        self.num_slice = kwargs.get("num slices", 1)
        if self.num_slice < 1:
            logger.warning(f"The number of slices of {self.cmd_name} is {self.num_slice}, which should be >= 1. It has been changed to 1 now.")
            self.num_slice = 1

        self.integrator = kwargs.get("integrator", "adaptive")
        if self.integrator not in ["adaptive", "uniform", "yoshida4"]:
            raise ValueError(
                f"The integrator of Multipole {self.cmd_name} is {self.integrator}, which should be 'adaptive', 'uniform' or 'yoshida4'.")
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
        multipole_order = max(len(self.knl) - 1, self.ramp.max_order if self.ramp is not None else 0)
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, Length={self.length:.4f}, "
                    f"IsThick={self.is_thick}, MultipoleOrder={multipole_order:d}, "
                    f"KnL={np.array2string(self.knl, precision=6)}, "
                    f"KsL={np.array2string(self.ksl, precision=6)}, "
                    f"NumSlice={self.num_slice:d}, Integrator={self.integrator:s}, "
                    f"ApertureType={self.aperture_type:s}, ApertureValue={self.aperture_value}")
        if self.ramp is not None:
            logger.info("  Ramping: %s; entry-time sampling; effective slices=%d; files=%s", self.ramp.summary(),
                        self.slice_plan.num_slices if self.is_thick else 1, ", ".join(str(path) for path in self.ramp.sources))
        print_element_slicing(self)
        set_normal_logging()

    def execute_cpu(self, sim):
        beam = sim.beams[self.beam_id]
        turn = sim.state.turn
        masks = self.alignment_errors.enter_frame(self, beam, turn)
        try:
            for bunch in beam.bunches:
                if bunch.end_idx <= bunch.start_idx:
                    continue
                if self.ramp is not None:
                    self.update_strengths(bunch.t0)
                self._track_multipole_cpu(beam, bunch, turn)
        finally:
            self.alignment_errors.exit_frame(self, beam, turn, masks)
        for bunch in beam.bunches:
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)
        return True

    def execute_gpu(self, sim):
        beam = sim.beams[self.beam_id]
        turn = sim.state.turn
        masks = self.alignment_errors.enter_frame(self, beam, turn, gpu=True)
        try:
            for bunch in beam.bunches:
                if bunch.end_idx <= bunch.start_idx:
                    continue
                if self.ramp is not None:
                    self.update_strengths(bunch.t0)
                self._track_multipole_gpu(beam, bunch, turn)
        finally:
            self.alignment_errors.exit_frame(self, beam, turn, masks, gpu=True)
        for bunch in beam.bunches:
            check_aperture_gpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)
        return True

    def update_strengths(self, reference_time, offset=0.):
        """Update current nominal strengths from the program, without tracking."""
        update_magnet_strengths(self, reference_time, offset)

    def _refresh_strengths(self):
        normal, skew = np.asarray(self.knl, dtype=float), np.asarray(self.ksl, dtype=float)
        n = max(len(normal), len(skew), self.ramp.max_order + 1 if self.ramp is not None else 1,
                len(self.field_errors.knl) if self.field_errors.active else 1)
        self.knl = np.pad(normal, (0, n - len(normal))) if len(normal) != n else normal
        self.ksl = np.pad(skew, (0, n - len(skew))) if len(skew) != n else skew
        self.kn = self.knl / self.length if self.is_thick else np.zeros(n)
        self.ks = self.ksl / self.length if self.is_thick else np.zeros(n)
        if not hasattr(self, "inv_fact") or len(self.inv_fact) != n:
            self.inv_fact = np.ones(n)
            for index in range(1, n):
                self.inv_fact[index] = self.inv_fact[index - 1] / index

    def _track_multipole_gpu(self, beam, bunch, turn):
        kn, ks, inv_fact = _prepare_multipole_coefficients(self, self.knl, self.ksl)
        if self._sc_nodes:
            body = _GpuBody(self, beam, bunch, turn)
            launch = body.multipole_stage(kn, ks, inv_fact)
            body.run(lambda ds, on_center: transport_dkd(launch, self.integrator, ds, on_center))
        else:
            mode = 0 if not self.is_thick else 2 if not np.any(kn) and not np.any(ks) else 1
            _launch_multipole_bunch(self, beam, bunch, turn, kn, ks, inv_fact, mode)

    def _effective_strengths(self):
        """Cache nominal plus error coefficients without modifying nominal state."""
        coefficients = getattr(self, "_multipole_coefficients", None)
        if coefficients is None:
            knl, ksl = self.field_errors.combine(self.knl, self.ksl)
            coefficients = (knl, ksl, knl / self.length if self.is_thick else self.kn, ksl / self.length if self.is_thick else self.ks)
            self._multipole_coefficients = coefficients
        return coefficients

    def _track_multipole_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):

        beta0 = bunch.beta
        start = bunch.start_idx
        end = bunch.end_idx
        if end <= start:
            return

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
            knl, ksl, _, _ = self._effective_strengths()
            self._multipole_kick_cpu(knl, ksl, x, px, y, py, tag, mask, chi)
            return

        if self._sc_nodes:
            step = self._dkd_step_cpu if self.integrator == "uniform" else self._dkd_yoshida4_cpu
            _, _, kn, ks = self._effective_strengths()

            def transport(ds, on_center):
                step(x, px, y, py, z, dp, tag, mask, ds, kn, ks, chi, beta0, on_center=on_center)

            run_body_slices(self, beam, bunch, turn, transport)
            return

        knl, ksl, kn, ks = self._effective_strengths()
        all_zero = np.all(knl == 0) and np.all(ksl == 0)
        if all_zero:
            self._drift_exact_cpu(self.length, x, px, y, py, z, dp, tag, mask, beta0)
        else:
            ds = self.length / self.num_slice
            for _ in range(self.num_slice):
                if self.integrator == "uniform":
                    self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds, kn, ks, chi, beta0)
                elif self.integrator == "yoshida4":
                    self._dkd_yoshida4_cpu(x, px, y, py, z, dp, tag, mask, ds, kn, ks, chi, beta0)

        newly_lost = alive_before & (tag < 0)
        if np.any(newly_lost):
            lost_position = p.lost_position[start:end]
            lost_turn = p.lost_turn[start:end]
            lost_position[newly_lost] = self.s
            lost_turn[newly_lost] = turn

    def _dkd_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask, ds, kn, ks, chi, beta0, on_center=None):
        """Compose three drift-kick-drift steps with Yoshida coefficients."""
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, kn, ks, chi, beta0)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z0, kn, ks, chi, beta0, on_center=on_center)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, kn, ks, chi, beta0)

    def _dkd_step_cpu(self, x, px, y, py, z, dp, tag, mask, ds, kn, ks, chi, beta0, on_center=None):
        """Apply one drift-kick-drift step; Yoshida composition may use negative ds."""
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._multipole_kick_cpu(kn * ds, ks * ds, x, px, y, py, tag, mask, chi)
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
        active = alive & valid
        active = slice(None) if np.all(active) else active
        x[active] += np.float64(L) * px[active] * inv_pz[active]
        y[active] += np.float64(L) * py[active] * inv_pz[active]
        z[active] += np.float64(L) * slip[active]

    def _multipole_kick_cpu(self, knl_eff, ksl_eff, x, px, y, py, tag, mask, chi):
        """Apply the common integrated multipole polynomial."""
        _apply_multipole_kick_cpu(knl_eff, ksl_eff, self.inv_fact, x, px, y, py, tag, chi)
