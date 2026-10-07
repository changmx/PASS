import logging

import numpy as np

from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.utils.logger import set_simple_logging, set_normal_logging
from PASS.utils.constants import const
from PASS.utils.aperture import check_aperture_cpu
from PASS.utils.program import LinearProgram

logger = logging.getLogger(__name__)


class ExciterWaveform:
    """Continuous DDS phases, integrating a sawtooth times a prescribed clock.

    The clock is piecewise linear; its product with each sweep is integrated
    analytically. Times are relative to one common trigger, in seconds.
    """

    def __init__(self, center, width, period, trigger, reference, sweep_offset=.5):
        self.center, self.width, self.period, self.trigger = center, width, period, trigger
        self.carrier = LinearProgram(center * reference.values, reference.times, origin=trigger)
        self.shifts = (sweep_offset * period, 0.)
        self.times = np.unique(np.r_[reference.times - trigger, 0.])
        self.values = reference.value(trigger, self.times)
        self.slopes = np.r_[np.diff(self.values) / np.diff(self.times), 0.]
        self.phases = np.zeros((2, len(self.times)))
        origin = int(np.searchsorted(self.times, 0.))
        for channel, shift in enumerate(self.shifts):
            for index in range(origin, len(self.times) - 1):
                increment = self._integral(self.times[index], self.times[index + 1], self.values[index], self.slopes[index], shift, np)
                self.phases[channel, index + 1] = (self.phases[channel, index] + increment) % 1.
            for index in range(origin - 1, -1, -1):
                increment = self._integral(self.times[index], self.times[index + 1], self.values[index], self.slopes[index], shift, np)
                self.phases[channel, index] = (self.phases[channel, index + 1] - increment) % 1.
        self._devices = {}

    def _moments(self, time, shift, xp, offset=0.):
        tau = xp.remainder(xp.remainder(time, self.period) + shift + offset, self.period)
        primitive = tau * (tau - self.period) / (2 * self.period)
        # Integral of the periodic primitive = -T*t/12 + periodic_part.
        periodic_part = tau * (tau * tau / (6 * self.period) - tau / 4 + self.period / 12)
        return primitive, periodic_part

    def _integral(self, start, end, frequency, slope, shift, xp, offset=0.):
        duration = (end - start) + offset
        first, first_moment = self._moments(start, shift, xp)
        last, last_moment = self._moments(end, shift, xp, offset)
        sweep = frequency * (last - first) + slope * (duration * last + self.period * duration / 12 - last_moment + first_moment)
        return self.width * sweep

    def phase_cycles(self, reference_time, offset=0., *, channel=0, xp=np):
        """Fractional cycles; keep particle offsets separate from absolute epochs."""
        if xp not in self._devices:
            self._devices[xp] = tuple(xp.asarray(v) for v in (self.times, self.values, self.slopes, self.phases))
        times, values, slopes, phases = self._devices[xp]
        elapsed = reference_time - self.trigger
        index = xp.clip(xp.searchsorted(times - elapsed, offset, side='right') - 1, 0, len(self.times) - 1)
        slope = xp.where(offset < times[0] - elapsed, 0., slopes[index])
        cycles = self.carrier.phase_cycles(reference_time, offset, xp) + phases[channel, index]
        cycles += self._integral(times[index], elapsed, values[index], slope, self.shifts[channel], xp, offset)
        return xp.remainder(cycles, 1.)

    def value(self, reference_time, offset=0., *, dual=False, xp=np):
        # A single DDS uses the unshifted sweep (channel 2 in dual mode).
        value = xp.sin(2 * np.pi * self.phase_cycles(reference_time, offset, channel=1, xp=xp))
        if dual:
            value += xp.sin(2 * np.pi * self.phase_cycles(reference_time, offset, channel=0, xp=xp))
        return xp.where(offset >= self.trigger - reference_time, value, 0.)


@Command.register("exciter")
class Exciter(Command):

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id: int = beam_id
        self.s: float = kwargs["s (m)"]
        self.length: float = 0.0
        self.cmd_type: str = self.__class__.__name__
        self.cmd_name: str = kwargs["name"]

        if kwargs.get("length (m)", 0.0) != 0.0:
            raise ValueError(f"Exciter {self.cmd_name} is a zero-length kick")

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

        if any(key in kwargs for key in ("voltage (v)", "gap (m)", "plate length (m)")):
            raise ValueError("Exciter uses 'Kick angle (rad)'; convert voltage and electrode parameters with the Exciter tool")
        self.kick_angle: float = kwargs["kick angle (rad)"]
        if not np.isfinite(self.kick_angle):
            raise ValueError("Exciter Kick angle (rad) must be finite")
        self._gpu_cache = {}

        # Tune mode follows the shared prescribed clock, not tracked energies.
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
        self.dual_sweep_offset: float = kwargs.get("dual sweep offset", .5)
        if not np.isfinite(self.period) or self.period <= 0:
            raise ValueError("Exciter period must be finite and positive")
        if not np.isfinite(self.dual_sweep_offset) or not 0 <= self.dual_sweep_offset <= 1:
            raise ValueError("Dual sweep offset must be a finite fraction in [0, 1]")
        if self.mode.startswith("dual"):
            if not np.isclose(self.dual_sweep_offset, .5, rtol=0., atol=1e-12):
                logger.warning("Exciter %s: Dual sweep offset=%g; the hardware default is half a sweep period (0.5).", self.cmd_name,
                               self.dual_sweep_offset)
            legacy_frequency = kwargs.get("fm dual frequency (hz)")
            if legacy_frequency is not None and not np.isclose(legacy_frequency * self.period, 1., rtol=1e-9, atol=0.):
                logger.warning("Exciter %s: obsolete FM dual frequency (Hz)=%g is ignored; the sweep rate is 1/Period=%g Hz.", self.cmd_name,
                               legacy_frequency, 1 / self.period)

        beam = sim.beams[beam_id]
        reference = beam.reference_program
        trigger = reference.inverse_integral(self.start_turn + self.s / beam.bunches[0].circum)
        self.am_frequency = float(reference.value(trigger))
        if self.use_tune_mode:
            center, width = self.excite_tune, self.sweep_tune
        else:
            center, width = self.cf, self.cfw
            reference = LinearProgram(1., origin=trigger)
        self.waveform = ExciterWaveform(center, width, self.period, trigger, reference, self.dual_sweep_offset)

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
                    f"kick_angle={self.kick_angle:.6e} rad, "
                    f"{freq_info}, "
                    f"period={self.period:.6e}, dual_sweep_offset={self.dual_sweep_offset:g}, "
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
        for bunch in bunches:
            self._exciter_kick_cpu(beam, bunch, turn)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
        return True

    def execute_gpu(self, sim):
        if not self.is_enabled:
            return False
        turn = sim.state.turn
        if turn < self.start_turn or turn >= self.end_turn:
            return False
        beam = sim.beams[self.beam_id]
        for bunch in beam.bunches:
            launch_exciter(self, sim, bunch)
        return True

    # AM (amplitude modulation) helpers

    def _kick_am_vary(self, elapsed, xp=np):
        """Common AM envelope at physical seconds since the DDS trigger.

        The diffusion normalization uses the prescribed startup revolution
        frequency, like the fixed Ext_Freq in the hardware waveform generator.
        """
        elapsed = xp.asarray(elapsed, dtype=xp.float64)
        active = elapsed >= 0.
        time = xp.where(active, elapsed, 0.)
        exponent = np.exp(-(self.am_r0 / self.am_delta0)**2)
        complement = -np.expm1(-(self.am_r0 / self.am_delta0)**2)
        argument = time / self.am_t_ext * complement + exponent
        valid = active & (argument > 0.)
        # Inactive samples and an underflowed initial exponential give zero.
        logarithm = xp.log(xp.where(valid, argument, 1.))
        denominator = logarithm**2 * (self.am_t_ext * exponent + time * complement)
        delta2 = self.am_r0**2 * complement / xp.where(valid, denominator, 1.)
        return xp.where(valid, xp.sqrt(delta2 / self.am_frequency / self.am_k_const), 0.)

    # Exciter kick (CPU)

    def _exciter_kick_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):
        """Apply the prescribed waveform at each particle arrival time."""
        if not 0 < bunch.beta < 1:
            raise ValueError("Exciter requires a finite reference velocity with 0 < beta < 1")
        v0 = bunch.beta * const.c
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
            valid = (dp > -1) & (ps2 > 0) & np.isfinite(ps2) & np.isfinite(z)
            valid &= np.isfinite(p.x[start:end]) & np.isfinite(p.y[start:end])
        invalid = (tag > 0) & ~valid
        tag[invalid] = -np.abs(tag[invalid])
        p.lost_position[start:end][invalid] = self.s
        p.lost_turn[start:end][invalid] = turn
        alive = tag > 0

        # Positive z arrives earlier; all bunches sample the same DDS history.
        offset = -np.where(alive, np.asarray(z, dtype=np.float64), 0.) / v0
        # Like Kicker, the nominal angle is an increment of momentum / P0.
        kick = self.kick_angle * self.waveform.value(bunch.t0, offset, dual=self.mode.startswith("dual"))
        if self.mode.endswith("_am"):
            elapsed = np.where(alive, (bunch.t0 - self.waveform.trigger) + offset, -1.)
            kick *= self._kick_am_vary(elapsed)

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
__device__ double _positive_remainder(
    double value,
    double period
) {
    double result = fmod(value, period);
    return result < 0 ? result + period : result;
}

__device__ int _find_interval(
    const double* table,
    int count,
    int stride,
    double reference,
    double offset
) {
    int start = 0, end = count;
    while (start < end) {
        int middle = (start + end) / 2;
        if (table[stride * middle] - reference <= offset)
            start = middle + 1;
        else
            end = middle;
    }
    return max(0, start - 1);
}

__device__ double _carrier_phase(
    const double* table,
    int count,
    double reference,
    double offset,
    double base,
    double frequency,
    int reference_index
) {
    if (count == 1)
        return _positive_remainder(base + frequency * offset, 1.);
    int index = _find_interval(table, count, 4, reference, offset);
    bool before = offset < table[0] - reference;
    double slope = before ? 0. : table[4 * index + 2];
    if (index == reference_index && before == (reference < table[0]))
        return _positive_remainder(base + offset * (frequency + .5 * slope * offset), 1.);
    int following = min(index + 1, count - 1);
    if (!before && fabs(reference - table[4 * following]) < fabs(reference - table[4 * index]))
        index = following;
    double dx = (reference - table[4 * index]) + offset;
    return _positive_remainder(table[4 * index + 3] + dx * (table[4 * index + 1] + .5 * slope * dx), 1.);
}

__device__ void _sweep_moments(
    double time,
    double offset,
    double shift,
    double period,
    double& primitive,
    double& moment
) {
    double tau = _positive_remainder(_positive_remainder(time, period) + shift + offset, period);
    primitive = tau * (tau - period) / (2 * period);
    moment = tau * (tau * tau / (6 * period) - tau / 4 + period / 12);
}

__device__ double _sweep_phase(
    const double* node,
    double elapsed,
    double offset,
    double slope,
    double shift,
    double period,
    double width,
    int channel
) {
    double first, first_moment, last, last_moment;
    _sweep_moments(node[0], 0., shift, period, first, first_moment);
    _sweep_moments(elapsed, offset, shift, period, last, last_moment);
    double duration = (elapsed - node[0]) + offset;
    return node[3 + channel] + width * (node[1] * (last - first) + slope * (duration * last + period * duration / 12 - last_moment + first_moment));
}

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
    const double* __restrict__ sweep_table,
    int sweep_count,
    const double* __restrict__ carrier_table,
    int carrier_count,
    double reference_time,
    double elapsed,
    double carrier_base,
    double carrier_frequency,
    int carrier_index,
    double v0,
    double period,
    double width,
    double shift,
    int dual,
    int am,
    double am_t_ext,
    double am_exponent,
    double am_complement,
    double am_scale,
    double s0,
    int turn,
    double kick_angle,
    int direction
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index || tag[i] <= 0)
        return;
    double r = 1. + (double)dp[i], ps2 = r * r - (double)px[i] * px[i] - (double)py[i] * py[i];
    if (!(r > 0 && ps2 > 0) || !isfinite(ps2) || !isfinite(z[i]) || !isfinite(x[i]) || !isfinite(y[i])) {
        tag[i] = -abs(tag[i]);
        lost_position[i] = (float)s0;
        lost_turn[i] = turn;
        return;
    }
    // Keep the local arrival offset separate from the large reference epoch.
    double offset = -(double)z[i] / v0;
    if (offset < -elapsed)
        return;
    double carrier = _carrier_phase(carrier_table, carrier_count, reference_time, offset, carrier_base, carrier_frequency, carrier_index);
    int index = _find_interval(sweep_table, sweep_count, 5, elapsed, offset);
    const double* node = sweep_table + 5 * index;
    double slope = offset < sweep_table[0] - elapsed ? 0. : node[2];
    double phase = carrier + _sweep_phase(node, elapsed, offset, slope, 0., period, width, 1);
    double waveform = sin(6.2831853071795864769 * _positive_remainder(phase, 1.));
    if (dual) {
        phase = carrier + _sweep_phase(node, elapsed, offset, slope, shift, period, width, 0);
        waveform += sin(6.2831853071795864769 * _positive_remainder(phase, 1.));
    }
    if (am) {
        double time = elapsed + offset;
        double argument = time / am_t_ext * am_complement + am_exponent;
        double logarithm = log(argument > 0 ? argument : 1.);
        double denominator = logarithm * logarithm * (am_t_ext * am_exponent + time * am_complement);
        waveform *= argument > 0 ? sqrt(am_scale / denominator) : 0.;
    }
    double kick = kick_angle * waveform;
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


def launch_exciter(element, sim, bunch):
    if not 0 < bunch.beta < 1:
        raise ValueError("Exciter requires a finite reference velocity with 0 < beta < 1")
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU Exciter tracking requires the optional 'cuda' dependencies.") from exc
    p = sim.beams[element.beam_id].particles
    key = (cp.cuda.runtime.getDevice(), np.dtype(p.dtype))
    if key not in element._gpu_cache:
        kernel = cp.RawKernel(
            CUDA_REAL_PREAMBLE + EXCITER_BODY,
            "track_exciter",
            options=("--std=c++14", "--fmad=false", f"-DPASS_USE_FLOAT={int(p.dtype == np.dtype(np.float32))}"),
        )
        waveform = element.waveform
        carrier = waveform.carrier
        sweep_table = cp.asarray(np.column_stack((waveform.times, waveform.values, waveform.slopes, waveform.phases.T)), order='C')
        carrier_table = cp.asarray(np.column_stack((carrier.times, carrier.values, carrier.slopes, carrier.phase_integrals)))
        element._gpu_cache[key] = kernel, sweep_table, carrier_table
    start, end = bunch.start_idx, bunch.end_idx
    n = end - start
    if n <= 0:
        return
    real = np.float64
    threads = 256
    blocks = (n + threads - 1) // threads
    waveform = element.waveform
    base, frequency, index = waveform.carrier.phase_anchor(bunch.t0)
    am = element.mode.endswith("_am")
    exponent = np.exp(-(element.am_r0 / element.am_delta0)**2) if am else 0.
    complement = -np.expm1(-(element.am_r0 / element.am_delta0)**2) if am else 0.
    scale = element.am_r0**2 * complement / element.am_frequency / element.am_k_const if am else 0.
    kernel, sweep_table, carrier_table = element._gpu_cache[key]
    kernel((blocks, ), (threads, ),
           (p.px, p.py, p.z, p.dp, p.x, p.y, p.tag, p.lost_position, p.lost_turn, np.int32(start), np.int32(end), sweep_table,
            np.int32(len(waveform.times)), carrier_table, np.int32(len(waveform.carrier.times)), real(bunch.t0), real(bunch.t0 - waveform.trigger),
            real(base), real(frequency), np.int32(index), real(bunch.beta * const.c), real(waveform.period), real(waveform.width),
            real(waveform.shifts[0]), np.int32(element.mode.startswith("dual")), np.int32(am), real(element.am_t_ext), real(exponent),
            real(complement), real(scale), real(element.s), np.int32(sim.state.turn), real(element.kick_angle), np.int32(0 if element.is_x else 1)))
    from PASS.utils.aperture import check_aperture_gpu
    check_aperture_gpu(sim.beams[element.beam_id], bunch, element.aperture_type, element.aperture_value, element.s, sim.state.turn)
