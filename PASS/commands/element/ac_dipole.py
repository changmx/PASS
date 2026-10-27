"""Thin AC dipole with independent physical-time kick and frequency programs."""

import logging

import numpy as np

from PASS.commands.command import Command
from PASS.para.schema.elements import ACDipoleItem
from PASS.utils.aperture import check_aperture_cpu, check_aperture_gpu
from PASS.utils.constants import const
from PASS.utils.program import LinearProgram


@Command.register('acdipole')
class ACDipole(Command):
    """Add a prescribed transverse momentum / P0 at each particle's arrival."""

    def __init__(self, beam_id, sim, **command_kwargs):
        parameters = ACDipoleItem.model_validate(command_kwargs)
        self.beam_id, self.s, self.order = beam_id, parameters.s, parameters.order
        self.cmd_name, self.cmd_type, self.length = parameters.name, 'ACDipole', 0.
        self.direction, self.is_enabled = parameters.direction, parameters.is_enabled
        self.cpu_chunk_size = parameters.cpu_chunk_size
        self.aperture_type, self.aperture_value = parameters.aperture_type.lower(), parameters.aperture_value
        self.kick = LinearProgram(parameters.kick_angle, parameters.kick_times)
        self.frequency = LinearProgram(parameters.frequency, parameters.frequency_times, origin=parameters.phase_epoch)
        # Reduce the initial phase before adding the bounded integrated cycles.
        self.phase = float(np.arctan2(np.sin(parameters.phase), np.cos(parameters.phase)))
        self.time_min = -np.inf if parameters.kick_times is None else parameters.kick_times[0]
        self.time_max = np.inf if parameters.kick_times is None else parameters.kick_times[-1]
        self._gpu_cache = {}

    def print(self):
        logging.getLogger(__name__).info('S=%g, Command=ACDipole, Name=%s, Direction=%s, Kick=%s rad, Frequency=%s Hz, Enabled=%s', self.s,
                                         self.cmd_name, self.direction, self.kick.values, self.frequency.values, self.is_enabled)

    def execute_cpu(self, sim):
        return self._execute(sim, False)

    def execute_gpu(self, sim):
        return self._execute(sim, True)

    def _execute(self, sim, gpu):
        if not self.is_enabled:
            return False
        beam, turn = sim.beams[self.beam_id], sim.state.turn
        aperture = check_aperture_gpu if gpu else check_aperture_cpu
        for bunch in beam.bunches:
            if not 0 < bunch.beta <= 1 or not np.isfinite(bunch.t0):
                raise ValueError('ACDipole requires 0 < reference beta <= 1 and a finite reference arrival time')
            if bunch.end_idx <= bunch.start_idx:
                continue
            if gpu:
                self._track_gpu(beam.particles, bunch, turn)
            else:
                self._track_cpu(beam.particles, bunch, turn)
            aperture(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
        return True

    def _track_cpu(self, p, bunch, turn):
        # Bound NumPy temporaries; None explicitly selects the whole bunch.
        chunk_size = self.cpu_chunk_size or max(1, bunch.end_idx - bunch.start_idx)
        for start in range(bunch.start_idx, bunch.end_idx, chunk_size):
            self._kick_cpu(p, bunch, turn, start, min(start + chunk_size, bunch.end_idx))

    def _kick_cpu(self, p, bunch, turn, start, end):
        indices = slice(start, end)
        px, py, z, dp, tag = (getattr(p, name)[indices] for name in ('px', 'py', 'z', 'dp', 'tag'))
        alive = tag > 0
        with np.errstate(over='ignore', invalid='ignore'):
            momentum_squared = (1 + np.asarray(dp, dtype=np.float64))**2
            ps2 = momentum_squared - np.asarray(px, dtype=np.float64)**2 - np.asarray(py, dtype=np.float64)**2
            valid = (dp > -1) & (ps2 > 0) & np.isfinite(ps2) & np.isfinite(z)
            valid &= np.isfinite(p.x[indices]) & np.isfinite(p.y[indices])
        active = alive & valid
        # Keep the small particle offset separate from the absolute clock epoch.
        offset = np.where(active, np.asarray(z, dtype=np.float64), 0.) * (-1. / (bunch.beta * const.c))
        phase = self.frequency.phase_cycles(bunch.t0, offset)
        phase *= 2 * np.pi
        phase += self.phase
        np.sin(phase, out=phase)
        amplitude = self.kick.values[0] if len(self.kick.values) == 1 else self.kick.value(bunch.t0, offset)
        phase *= amplitude
        active &= (offset >= self.time_min - bunch.t0) & (offset <= self.time_max - bunch.t0)
        # This is a normalized momentum kick, with no division by 1 + delta.
        np.add(px if self.direction == 'x' else py, phase, out=px if self.direction == 'x' else py, where=active)
        with np.errstate(over='ignore', invalid='ignore'):
            ps2 = momentum_squared - np.asarray(px, dtype=np.float64)**2 - np.asarray(py, dtype=np.float64)**2
        invalid = alive & (~valid | ~np.isfinite(ps2) | (ps2 <= 0))
        tag[invalid] = -tag[invalid]
        p.lost_position[indices][invalid] = self.s
        p.lost_turn[indices][invalid] = turn

    def _track_gpu(self, p, bunch, turn):
        import cupy as cp

        key = (cp.cuda.runtime.getDevice(), p.dtype.str)
        if key not in self._gpu_cache:
            kernel = cp.RawKernel(_CUDA_SOURCE,
                                  'track_ac_dipole',
                                  options=('--std=c++14', '--fmad=false', f'-DPASS_USE_FLOAT={int(p.dtype == np.dtype(np.float32))}'))
            kick_table = cp.asarray(np.column_stack((self.kick.times, self.kick.values, self.kick.slopes, np.zeros(len(self.kick.times)))))
            frequency_table = cp.asarray(
                np.column_stack((self.frequency.times, self.frequency.values, self.frequency.slopes, self.frequency.phase_integrals)))
            self._gpu_cache[key] = kernel, kick_table, frequency_table
        kernel, kick_table, frequency_table = self._gpu_cache[key]
        base, frequency, index = self.frequency.phase_anchor(bunch.t0)
        n = bunch.end_idx - bunch.start_idx
        kernel(((n + 255) // 256, ), (256, ),
               (p.px, p.py, p.z, p.dp, p.x, p.y, p.tag, p.lost_position, p.lost_turn, np.int32(bunch.start_idx), np.int32(bunch.end_idx), kick_table,
                np.int32(len(self.kick.times)), frequency_table, np.int32(len(self.frequency.times)), np.float64(bunch.t0),
                np.float64(-1. / (bunch.beta * const.c)), np.float64(base), np.float64(frequency), np.int32(index), np.float64(
                    self.phase), np.float64(self.time_min - bunch.t0), np.float64(self.time_max - bunch.t0), np.float64(
                        self.s), np.int32(turn), np.int32(self.direction == 'y')))


_CUDA_SOURCE = r'''
#if PASS_USE_FLOAT
using pass_real_t = float;
#else
using pass_real_t = double;
#endif

__device__ int _ac_interval(
    const double* table,
    int count,
    double reference,
    double offset
) {
    int start = 0, end = count;
    while (start < end) {
        int middle = (start + end) / 2;
        if (table[4 * middle] - reference <= offset)
            start = middle + 1;
        else
            end = middle;
    }
    return max(0, start - 1);
}

__device__ double _ac_phase(
    const double* table,
    int count,
    double reference,
    double offset,
    double base,
    double frequency,
    int reference_index
) {
    double cycles;
    if (count == 1) {
        cycles = base + frequency * offset;
    } else {
        int index = _ac_interval(table, count, reference, offset);
        bool before = offset < table[0] - reference;
        double slope = before ? 0. : table[4 * index + 2];
        if (index == reference_index && before == (reference < table[0])) {
            cycles = base + offset * (frequency + .5 * slope * offset);
        } else {
            if (!before && index + 1 < count && fabs(reference - table[4 * (index + 1)]) < fabs(reference - table[4 * index]))
                ++index;
            double dx = (reference - table[4 * index]) + offset;
            cycles = table[4 * index + 3] + dx * (table[4 * index + 1] + .5 * slope * dx);
        }
    }
    cycles = fmod(cycles, 1.);
    return cycles < 0. ? cycles + 1. : cycles;
}

extern "C" __global__ void track_ac_dipole(
    pass_real_t* __restrict__ px,
    pass_real_t* __restrict__ py,
    const pass_real_t* __restrict__ z,
    const pass_real_t* __restrict__ dp,
    const pass_real_t* __restrict__ x,
    const pass_real_t* __restrict__ y,
    int* __restrict__ tag,
    float* lost_position,
    int* lost_turn,
    int start,
    int end,
    const double* __restrict__ kick_table,
    int kick_count,
    const double* __restrict__ frequency_table,
    int frequency_count,
    double reference,
    double inverse_velocity,
    double base,
    double frequency,
    int reference_index,
    double phase,
    double time_min,
    double time_max,
    double s,
    int turn,
    int direction
) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end || tag[i] <= 0)
        return;
    double momentum_ratio = 1. + (double)dp[i];
    double ps2 = momentum_ratio * momentum_ratio - (double)px[i] * px[i] - (double)py[i] * py[i];
    bool valid = momentum_ratio > 0. && ps2 > 0. && isfinite(ps2) && isfinite(z[i]) && isfinite(x[i]) && isfinite(y[i]);
    if (valid) {
        double offset = (double)z[i] * inverse_velocity;
        if (offset >= time_min && offset <= time_max) {
            double amplitude = kick_table[1];
            if (kick_count > 1) {
                int index = _ac_interval(kick_table, kick_count, reference, offset);
                double dx = (reference - kick_table[4 * index]) + offset;
                amplitude = kick_table[4 * index + 1] + kick_table[4 * index + 2] * dx;
            }
            double cycles = _ac_phase(frequency_table, frequency_count, reference, offset, base, frequency, reference_index);
            double kick = amplitude * sin(6.2831853071795864769 * cycles + phase);
            if (direction == 0)
                px[i] += kick;
            else
                py[i] += kick;
            ps2 = momentum_ratio * momentum_ratio - (double)px[i] * px[i] - (double)py[i] * py[i];
            valid = ps2 > 0. && isfinite(ps2);
        }
    }
    if (!valid) {
        tag[i] = -tag[i];
        lost_position[i] = (float)s;
        lost_turn[i] = turn;
    }
}
'''
