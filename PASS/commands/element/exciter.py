from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const

import numpy as np
import cupy as cp
import logging

logger = logging.getLogger(__name__)

_VALID_MODES = {"single_fm", "single_fm_am", "dual_fm", "dual_fm_am"}
_VALID_DIRECTIONS = {"x", "y"}


@Command.register("exciter")
class Exciter(Command):

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id: int = beam_id
        self.s: float = kwargs["s (m)"]
        self.length: float = kwargs.get("length (m)", 0.0)
        self.cmd_type: str = self.__class__.__name__
        self.cmd_name: str = kwargs["name"]

        if self.length > 0.0:
            raise ValueError(f"The length of Exciter {self.cmd_name} is {self.length}, which should be 0.0")

        self.is_enabled: bool = kwargs["enable"]
        if not isinstance(self.is_enabled, bool):
            raise ValueError(f"is_enabled must be a boolean in {self.cmd_name}, got {type(self.is_enabled)}")

        self.mode: str = kwargs["mode"].lower()
        if self.mode not in _VALID_MODES:
            raise ValueError(f"Unknown exciter mode '{self.mode}' in {self.cmd_name}. "
                             f"Must be one of: {sorted(_VALID_MODES)}")

        self.direction: str = kwargs["direction"].lower()
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(f"Unknown direction '{self.direction}' in {self.cmd_name}. "
                             f"Must be one of: {sorted(_VALID_DIRECTIONS)}")
        self.is_x: bool = (self.direction == "x")
        self.is_y: bool = (self.direction == "y")

        self.start_turn: int = int(kwargs["start turn"])
        self.end_turn: int = int(kwargs["end turn"])

        # --- exciter hardware parameters ---
        self.voltage: float = kwargs["voltage (v)"]  # peak voltage on the plates (V)
        self.gap: float = kwargs["gap (m)"]  # spacing between plates (m)
        self.plate_length: float = kwargs["plate length (m)"]  # effective length of the plates (m)

        # --- frequency parameters (two input modes) ---
        # Mode 1 (tune): provide excite_tune + sweep_tune, cf/cfw computed at runtime
        # Mode 2 (freq): provide central_frequency + sweep_width directly
        self.excite_tune = kwargs.get("excite tune", None)
        self.sweep_tune = kwargs.get("sweep tune", None)
        self.cf = kwargs.get("central frequency (hz)", None)
        self.cfw = kwargs.get("sweep width (hz)", None)

        if self.excite_tune is not None:
            if self.sweep_tune is None:
                raise ValueError(f"excite tune is provided but sweep tune is missing in {self.cmd_name}")
            self.use_tune_mode = True
        else:
            if self.cf is None or self.cfw is None:
                raise ValueError(f"Must provide either 'excite tune'+'sweep tune' or "
                                 f"'central frequency'+'sweep width' in {self.cmd_name}")
            self.use_tune_mode = False

        self.period: float = kwargs["period (s)"]

        self.fm_dual_frequency: float = kwargs["fm dual frequency (hz)"]

        # --- AM parameters ---
        self.am_t_ext: float = kwargs["am t ext (s)"]
        self.am_r0: float = kwargs["am r0 (m)"]
        self.am_delta0: float = kwargs["am delta0"]
        self.am_k_const: float = kwargs["am k const"]

        super().__init__()

    def print(self):
        set_simple_logging()
        if self.use_tune_mode:
            freq_info = (f"excite_tune={self.excite_tune:.6f}, sweep_tune={self.sweep_tune:.6f}, "
                         f"freq_mode=tune")
        else:
            freq_info = (f"cf={self.cf:.4f}, cfw={self.cfw:.4f}, "
                         f"freq_mode=frequency")
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, "
                    f"is_enabled={self.is_enabled}, Mode={self.mode:s}, Direction={self.direction:s}, "
                    f"start_turn={self.start_turn:d}, end_turn={self.end_turn:d}, "
                    f"voltage={self.voltage:.4f}, gap={self.gap:.4f}, plate_length={self.plate_length:.4f}, "
                    f"{freq_info}, "
                    f"period={self.period:.6e}, fm_dual_frequency={self.fm_dual_frequency:.4f}, "
                    f"am_t_ext={self.am_t_ext:.6e}, am_r0={self.am_r0:.4f}, "
                    f"am_delta0={self.am_delta0:.4e}, am_k_const={self.am_k_const:.4e}")
        set_normal_logging()

    def execute_cpu(self, sim):

        if not self.is_enabled:
            return

        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        turn = sim.state.turn

        if turn < self.start_turn or turn >= self.end_turn:
            return

        effective_turn = turn - self.start_turn

        for i, bunch in enumerate(bunches):
            # normalized kick amplitude: Δpx = V·L / (d·βc·Bρ)
            kick_amplitude = (self.voltage * self.plate_length / (self.gap * bunch.beta * const.c * bunch.brho))

            v0 = bunch.beta * const.c
            t0 = bunch.t0
            frequency_0 = 1.0 / (bunch.circum / v0)

            # compute cf and cfw: tune mode or frequency mode
            if self.use_tune_mode:
                cf = self.excite_tune * frequency_0
                cfw = self.sweep_tune * frequency_0
            else:
                cf = self.cf
                cfw = self.cfw

            logger.debug(f"Exciter {self.cmd_name}: turn={turn}, effective_turn={effective_turn}, "
                         f"kick_amplitude={kick_amplitude:.6e}, exciter tune={cf/frequency_0:.6f}, sweep tune={cfw/frequency_0:.6f}, "
                         f"cf={cf:.6e}, cfw={cfw:.6e}, frequency_rev={frequency_0:.6e}")

            start = bunch.start_idx
            end = bunch.end_idx

            p = beam.particles
            z = p.z[start:end]
            px = p.px[start:end]
            py = p.py[start:end]
            tag = p.tag[start:end]

            alive = tag > 0

            # time when each particle arrives at the exciter
            time_temp = t0 - z / v0

            if self.mode == "single_fm":
                kick = self._kick_saw_fm(effective_turn, time_temp, kick_amplitude, cf, cfw)
            elif self.mode == "single_fm_am":
                kick = self._kick_saw_fm_am(effective_turn, time_temp, frequency_0, kick_amplitude, cf, cfw)
            elif self.mode == "dual_fm":
                kick = self._kick_dual_fm(effective_turn, time_temp, kick_amplitude, cf, cfw)
            elif self.mode == "dual_fm_am":
                kick = self._kick_dual_fm_am(effective_turn, time_temp, frequency_0, kick_amplitude, cf, cfw)
            else:
                kick = np.zeros(len(z), dtype=np.float64)

            if self.is_x:
                px[alive] += kick[alive]
            else:
                py[alive] += kick[alive]

    def execute_gpu(self, sim):
        pass

    # ------------------------------------------------------------------
    # AM (amplitude modulation) helpers
    # ------------------------------------------------------------------

    def _kick_am_vary(self, effective_turn, frequency_0):
        """Time-varying amplitude factor (dimensionless), based on beam diffusion / growth model.

        effective_turn: turns elapsed since excitation started (>= 0).
        """
        if effective_turn < 0:
            return 0.0
        temp_time = effective_turn / frequency_0

        exponent = np.exp(-self.am_r0**2 / self.am_delta0**2)
        delta2_t = (self.am_r0**2 * (1.0 - exponent) / (np.log(temp_time / self.am_t_ext * (1.0 - exponent) + exponent)**2 *
                                                        (self.am_t_ext * exponent + temp_time * (1.0 - exponent))))
        return np.sqrt(delta2_t / frequency_0 / self.am_k_const)

    # ------------------------------------------------------------------
    # Kick shape helpers
    # ------------------------------------------------------------------

    def _kick_saw_fm(self, effective_turn, t, amplitude, cf, cfw):
        """single_fm: sawtooth frequency modulation with constant amplitude."""
        temp = t - np.floor(t / self.period) * self.period

        theta_t = (2.0 * const.pi * cf * temp + const.pi * cfw / self.period * temp * (temp - self.period))

        kick = amplitude * np.sin(theta_t)
        return kick

    def _kick_saw_fm_am(self, effective_turn, t, frequency_0, amplitude, cf, cfw):
        """single_fm_am: sawtooth FM with varying amplitude."""
        temp = t - np.floor(t / self.period) * self.period

        theta_t = (2.0 * const.pi * cf * temp + const.pi * cfw / self.period * temp * (temp - self.period))

        am_factor = self._kick_am_vary(effective_turn, frequency_0)

        kick = amplitude * am_factor * np.sin(theta_t)
        return kick

    def _kick_dual_fm(self, effective_turn, t, amplitude, cf, cfw):
        """dual_fm: dual frequency modulation with constant amplitude."""
        temp = t - np.floor(t / self.period) * self.period
        half_period = self.period / 2.0

        mask1 = (temp >= 0) & (temp <= half_period)
        mask2 = (temp > half_period) & (temp <= self.period)

        kick = np.zeros_like(t)
        kick[mask1] = (2.0 * amplitude * np.cos(const.pi / 2.0 * cfw * temp[mask1]) *
                       np.sin(2.0 * const.pi * cf * temp[mask1] + const.pi * cfw * (self.fm_dual_frequency * temp[mask1] - 0.5) * temp[mask1]))
        kick[mask2] = (2.0 * amplitude * np.cos(const.pi / 2.0 * cfw * temp[mask2]) *
                       np.sin(2.0 * const.pi * cf * temp[mask2] + const.pi * cfw * (temp[mask2] - half_period) *
                              (self.fm_dual_frequency * temp[mask2] - 1.0)))
        return kick

    def _kick_dual_fm_am(self, effective_turn, t, frequency_0, amplitude, cf, cfw):
        """dual_fm_am: dual FM with varying amplitude."""
        temp = t - np.floor(t / self.period) * self.period
        half_period = self.period / 2.0

        mask1 = (temp >= 0) & (temp <= half_period)
        mask2 = (temp > half_period) & (temp <= self.period)

        am_factor = self._kick_am_vary(effective_turn, frequency_0)

        kick = np.zeros_like(t)
        kick[mask1] = (2.0 * amplitude * am_factor * np.cos(const.pi / 2.0 * cfw * temp[mask1]) *
                       np.sin(2.0 * const.pi * cf * temp[mask1] + const.pi * cfw * (self.fm_dual_frequency * temp[mask1] - 0.5) * temp[mask1]))
        kick[mask2] = (2.0 * amplitude * am_factor * np.cos(const.pi / 2.0 * cfw * temp[mask2]) *
                       np.sin(2.0 * const.pi * cf * temp[mask2] + const.pi * cfw * (temp[mask2] - half_period) *
                              (self.fm_dual_frequency * temp[mask2] - 1.0)))
        return kick
