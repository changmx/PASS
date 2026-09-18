import logging

import numpy as np

from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.aperture import check_aperture_cpu

logger = logging.getLogger(__name__)


@Command.register("exciter")
class Exciter(Command):

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id: int = beam_id
        self.s: float = kwargs["s (m)"]
        self.length: float = kwargs.get("length (m)", 0.0)
        self.cmd_type: str = self.__class__.__name__
        self.cmd_name: str = kwargs["name"]

        if not np.isfinite(self.length) or self.length != 0.0:
            raise ValueError(f"The length of Exciter {self.cmd_name} is {self.length}, which should be 0.0")

        self.is_enabled: bool = kwargs["enable"]
        if not isinstance(self.is_enabled, bool):
            raise ValueError(f"is_enabled must be a boolean in {self.cmd_name}, got {type(self.is_enabled)}")

        valid_modes = {"single_fm", "single_fm_am", "dual_fm", "dual_fm_am"}
        self.mode: str = kwargs["mode"].lower()
        if self.mode not in valid_modes:
            raise ValueError(f"Unknown exciter mode '{self.mode}' in {self.cmd_name}. "
                             f"Must be one of: {sorted(valid_modes)}")

        valid_directions = {"x", "y"}
        self.direction: str = kwargs["direction"].lower()
        if self.direction not in valid_directions:
            raise ValueError(f"Unknown direction '{self.direction}' in {self.cmd_name}. "
                             f"Must be one of: {sorted(valid_directions)}")
        self.is_x: bool = (self.direction == "x")
        self.is_y: bool = (self.direction == "y")

        self.start_turn: int = int(kwargs["start turn"])
        self.end_turn: int = int(kwargs["end turn"])

        self.aperture_type: str = kwargs.get("aperture type", "off").lower()
        self.aperture_value: list = kwargs.get("aperture value", [])
        if not isinstance(self.aperture_value, list):
            raise ValueError(f"Aperture value of {self.cmd_name} must be a list, but got {type(self.aperture_value)}")

        self.voltage: float = kwargs["voltage (v)"]  # signed peak interplate difference (V)
        self.gap: float = kwargs["gap (m)"]  # spacing between plates (m)
        self.plate_length: float = kwargs["plate length (m)"]  # effective length of the plates (m)
        if not all(np.isfinite(v) for v in (self.voltage, self.gap, self.plate_length)) or self.gap <= 0 or self.plate_length < 0:
            raise ValueError("Exciter requires finite voltage, positive gap and nonnegative plate length")
        self._gpu_cache = {}

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
            return False

        turn = sim.state.turn
        if turn < self.start_turn or turn >= self.end_turn:
            return False

        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        effective_turn = turn - self.start_turn

        for i, bunch in enumerate(bunches):
            self._exciter_kick_cpu(beam, bunch, effective_turn, turn)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
        return True

    def execute_gpu(self, sim):
        if not self.is_enabled:
            return False
        turn = sim.state.turn
        if turn < self.start_turn or turn >= self.end_turn:
            return False
        beam = sim.beams[self.beam_id]
        effective_turn = turn - self.start_turn
        for bunch in beam.bunches:
            kick_amplitude = self._reference_amplitude(bunch)
            v0 = bunch.beta * const.c
            frequency_0 = v0 / bunch.circum
            if self.use_tune_mode:
                cf = self.excite_tune * frequency_0
                cfw = self.sweep_tune * frequency_0
            else:
                cf = self.cf
                cfw = self.cfw
            am_factor = 1.0
            if self.mode in ("single_fm_am", "dual_fm_am"):
                am_factor = self._kick_am_vary(effective_turn, frequency_0)
            mode = {"single_fm": 0, "single_fm_am": 1, "dual_fm": 2, "dual_fm_am": 3}[self.mode]
            launch_exciter(self, sim, bunch, effective_turn, (v0, kick_amplitude, cf, cfw, self.period, self.fm_dual_frequency, am_factor, mode))
        return True

    def _reference_amplitude(self, bunch):
        """Reference impulse, subsequently multiplied by beta0*c/v_s per particle."""
        if not 0 < bunch.beta < 1 or not np.isfinite(bunch.brho) or bunch.brho <= 0:
            raise ValueError("Exciter requires a finite massive-particle reference and positive rigidity")
        value = np.sign(bunch.num_charge) * self.voltage * self.plate_length / (self.gap * bunch.beta * const.c * bunch.brho)
        if not np.isfinite(value):
            raise ValueError("Exciter integrated impulse must be finite")
        return value

    # AM (amplitude modulation) helpers

    def _kick_am_vary(self, effective_turn, frequency_0):
        """Time-varying amplitude factor (dimensionless), based on beam diffusion / growth model.

        effective_turn: turns elapsed since excitation started (>= 0).
        """
        if effective_turn < 0:
            return 0.0
        temp_time = effective_turn / frequency_0

        exponent = np.exp(-self.am_r0**2 / self.am_delta0**2)

        # Guard against log(0) or log(negative):
        #   when temp_time=0 and exponent underflows to 0, log(0) = -inf -> NaN
        log_arg = temp_time / self.am_t_ext * (1.0 - exponent) + exponent
        if log_arg <= 0.0:
            return 0.0

        delta2_t = (self.am_r0**2 * (1.0 - exponent) / (np.log(log_arg)**2 * (self.am_t_ext * exponent + temp_time * (1.0 - exponent))))
        return np.sqrt(delta2_t / frequency_0 / self.am_k_const)

    # Kick shape helpers

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

    # Exciter kick (CPU)

    def _exciter_kick_cpu(self, beam: Beam, bunch: BunchInfo, effective_turn: int, turn: int):
        """Apply the prescribed waveform at each particle arrival time."""
        kick_amplitude = self._reference_amplitude(bunch)

        v0 = bunch.beta * const.c
        frequency_0 = 1.0 / (bunch.circum / v0)

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
        dp = p.dp[start:end].astype(np.float64)
        px0, py0 = px.astype(np.float64), py.astype(np.float64)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            ps2 = (1 + dp)**2 - px0 * px0 - py0 * py0
            A2 = 1 / bunch.gamma**2 + (1 - 1 / bunch.gamma**2) * (1 + dp)**2
            valid = (dp > -1) & (ps2 > 0) & np.isfinite(ps2) & np.isfinite(A2) & np.isfinite(z)
            valid &= np.isfinite(p.x[start:end]) & np.isfinite(p.y[start:end])
            factor = np.sqrt(np.where(valid, A2, 1.) / np.where(valid, ps2, 1.))
        invalid = (tag > 0) & ~valid
        tag[invalid] = -np.abs(tag[invalid])
        p.lost_position[start:end][invalid] = self.s
        p.lost_turn[start:end][invalid] = turn
        alive = tag > 0

        # Local time coordinate: positive z arrives earlier.
        time_temp = bunch.t0 - np.where(alive, np.asarray(z, dtype=np.float64), 0.) / v0

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

        kick = kick * factor
        if self.is_x:
            px[alive] += kick[alive]
        else:
            py[alive] += kick[alive]
        with np.errstate(over='ignore', invalid='ignore'):
            ps2 = (1 + dp)**2 - px.astype(np.float64)**2 - py.astype(np.float64)**2
        invalid = alive & (~np.isfinite(ps2) | (ps2 <= 0))
        tag[invalid] = -np.abs(tag[invalid])
        p.lost_position[start:end][invalid] = self.s
        p.lost_turn[start:end][invalid] = turn


CUDA_REAL_PREAMBLE = r'''
#ifndef PASS_USE_FLOAT
#define PASS_USE_FLOAT 0
#endif
#if PASS_USE_FLOAT
using pass_real_t = float;
#else
using pass_real_t = double;
#endif
'''

EXCITER_BODY = r'''
extern "C" __global__ void track_exciter(
    pass_real_t* __restrict__ px,
    pass_real_t* __restrict__ py,
    const pass_real_t* __restrict__ z,
    const pass_real_t* __restrict__ dp,
    const pass_real_t* __restrict__ x,
    const pass_real_t* __restrict__ y,
    int* __restrict__ tag,
    float* lost_position,
    int* lost_turn,
    int start_index,
    int end_index,
    double t0,
    double inv_g2,
    double s0,
    int turn,
    double v0,
    double amplitude,
    double cf,
    double cfw,
    double period,
    double fm_dual_frequency,
    double am_factor,
    int mode,
    int direction
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index || tag[i] <= 0)
        return;
    double r = 1. + (double)dp[i], ps2 = r * r - (double)px[i] * px[i] - (double)py[i] * py[i];
    double A2 = inv_g2 + (1 - inv_g2) * r * r;
    if (!(r > 0 && ps2 > 0) || !isfinite(ps2) || !isfinite(A2) || !isfinite(z[i]) || !isfinite(x[i]) || !isfinite(y[i])) {
        tag[i] = -abs(tag[i]);
        lost_position[i] = (float)s0;
        lost_turn[i] = turn;
        return;
    }
    amplitude *= sqrt(A2 / ps2);
    const double pi = (double)3.1415926535897932384626433832795;
    double t = t0 - (double)z[i] / v0;
    double temp = t - floor(t / period) * period;
    double kick = (double)0;
    if (mode == 0 || mode == 1) {
        double theta = (double)2 * pi * cf * temp + pi * cfw / period * temp * (temp - period);
        kick = amplitude * (mode == 1 ? am_factor : (double)1) * sin(theta);
    } else {
        double half = period * (double)0.5;
        if (temp >= (double)0 && temp <= half) {
            double theta = (double)2 * pi * cf * temp + pi * cfw * (fm_dual_frequency * temp - (double)0.5) * temp;
            kick = (double)2 * amplitude * (mode == 3 ? am_factor : (double)1) * cos(pi * (double)0.5 * cfw * temp) * sin(theta);
        } else if (temp > half && temp <= period) {
            double theta = (double)2 * pi * cf * temp + pi * cfw * (temp - half) * (fm_dual_frequency * temp - (double)1);
            kick = (double)2 * amplitude * (mode == 3 ? am_factor : (double)1) * cos(pi * (double)0.5 * cfw * temp) * sin(theta);
        }
    }
    if (direction == 0)
        px[i] += kick;
    else
        py[i] += kick;
    ps2 = r * r - (double)px[i] * px[i] - (double)py[i] * py[i];
    if (!isfinite(ps2) || ps2 <= 0) {
        tag[i] = -abs(tag[i]);
        lost_position[i] = (float)s0;
        lost_turn[i] = turn;
    }
}
'''


def launch_exciter(element, sim, bunch, effective_turn, params):
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU Exciter tracking requires the optional 'cuda' dependencies.") from exc
    p = sim.beams[element.beam_id].particles
    key = (cp.cuda.runtime.getDevice(), np.dtype(p.dtype))
    if key not in element._gpu_cache:
        element._gpu_cache[key] = cp.RawKernel(
            CUDA_REAL_PREAMBLE + EXCITER_BODY,
            "track_exciter",
            options=("--std=c++14", "--fmad=false", f"-DPASS_USE_FLOAT={int(p.dtype == np.dtype(np.float32))}"),
        )
    start, end = bunch.start_idx, bunch.end_idx
    n = end - start
    if n <= 0:
        return
    real = np.float64
    threads = 256
    blocks = (n + threads - 1) // threads
    v0, amplitude, cf, cfw, period, fm_dual_frequency, am_factor, mode = params
    element._gpu_cache[key]((blocks, ), (threads, ),
                            (p.px, p.py, p.z, p.dp, p.x, p.y, p.tag, p.lost_position, p.lost_turn, np.int32(start), np.int32(end), real(
                                bunch.t0), real(1 / bunch.gamma**2), real(element.s), np.int32(sim.state.turn), real(v0), real(amplitude), real(cf),
                             real(cfw), real(period), real(fm_dual_frequency), real(am_factor), np.int32(mode), np.int32(0 if element.is_x else 1)))
    from PASS.utils.aperture import check_aperture_gpu
    check_aperture_gpu(sim.beams[element.beam_id], bunch, element.aperture_type, element.aperture_value, element.s, sim.state.turn)
