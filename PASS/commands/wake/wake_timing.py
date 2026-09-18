"""Convert user-controlled z slices to physical passage times.

No particle clock state is retained. Slice intervals remain in metres until
an explicit Slicer updates them; immutable wake sources store sampled times.
"""
import numpy as np

from PASS.utils.constants import const


class WakeClock:

    def __init__(self, beam, slice_names=None):
        self.beam, self.xp = beam, beam.particles.xp
        self.slice_names = None if slice_names is None else frozenset(slice_names)

    @property
    def times(self):
        return self.sample()

    def sample(self):
        p, xp = self.beam.particles, self.xp
        times = xp.empty(len(p.z), dtype=xp.float64)
        if xp is not np:
            if self.beam.bunches:
                ranges = xp.asarray([(b.start_idx, b.end_idx) for b in self.beam.bunches], dtype=xp.int64)
                parameters = xp.asarray([(b.t0, b.beta * const.c) for b in self.beam.bunches], dtype=xp.float64)
                maximum = max(b.end_idx - b.start_idx for b in self.beam.bunches)
                self._kernel('sample_times', p.dtype)((max(1, min(128, (maximum + 255) // 256)), len(self.beam.bunches)), (256, ),
                                                      (p.z, ranges, parameters, times))
            return times
        for b in self.beam.bunches:
            bunch_slice = slice(b.start_idx, b.end_idx)
            times[bunch_slice] = b.t0 - p.z[bunch_slice].astype(xp.float64) / (b.beta * const.c)
        return times

    def slice_geometry(self, bunch, name, ids=None, alive=None):
        xp = self.xp
        slices = bunch.slice_sets[name]
        if getattr(slices, "coordinate", None) == "z_periodic":
            raise ValueError("WakeField cannot use z_periodic slices; select z_rel or arrival_phase")
        if slices.slice_table is None:
            raise ValueError('WakeField requires an explicitly generated SliceSet')
        centers = xp.asarray(slices.slice_table['z_center'], dtype=xp.float64)
        widths = xp.asarray(slices.slice_table['delta_z'], dtype=xp.float64)
        if slices.periodic:
            # Periodic bins have their own common observation window, recorded
            # by the user Slicer. They are not the local bunch reference frame.
            epoch, velocity = slices.observation_time, slices.observation_velocity
        else:
            epoch, velocity = bunch.t0, bunch.beta * const.c
        if xp is np:
            return epoch - centers / velocity, widths / velocity
        times, durations = xp.empty_like(centers), xp.empty_like(widths)
        if centers.size:
            self._kernel('slice_times',
                         np.float64)(((centers.size + 255) // 256, ), (256, ),
                                     (centers, widths, np.int64(centers.size), np.float64(epoch), np.float64(velocity), times, durations))
        return times, durations

    def _kernel(self, name, dtype):
        cache = self.__dict__.setdefault('_timing_kernels', {})
        key = (self.xp.cuda.runtime.getDevice(), name, np.dtype(dtype))
        if key not in cache:
            cache[key] = self.xp.RawKernel(_TIMING_CODE, name, options=('--std=c++17', f'-DFLOAT_PARTICLES={int(dtype==np.float32)}'))
        return cache[key]


def geometry_gpu(clock, bunch, name, ids, alive):
    return clock.slice_geometry(bunch, name, ids, alive)


_TIMING_CODE = r'''
#if FLOAT_PARTICLES
using R = float;
#else
using R = double;
#endif
extern "C" __global__ void sample_times(
    const R* z,
    const long long* ranges,
    const double* parameters,
    double* out
) {
    int b = blockIdx.y;
    long long end = ranges[2 * b + 1];
    for (long long i = ranges[2 * b] + blockIdx.x * blockDim.x + threadIdx.x; i < end; i += gridDim.x * blockDim.x)
        out[i] = parameters[2 * b] - (double)z[i] / parameters[2 * b + 1];
}
extern "C" __global__ void slice_times(
    const double* centers,
    const double* widths,
    long long n,
    double epoch,
    double velocity,
    double* times,
    double* durations
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        times[i] = epoch - centers[i] / velocity;
        durations[i] = widths[i] / velocity;
    }
}
'''


def prepare_wake_tracking(sim, sequences):
    """Prepare enabled wakes for a new run with no retained history."""
    commands = [cmd for seq in sequences for cmd in seq.cmds if cmd.cmd_type == 'WakeField' and cmd.is_enabled]
    wake_ids = {cmd.beam_id for cmd in commands}
    for cmd in commands:
        if any(s.last_turn is not None for s in cmd.group_states):
            raise ValueError('WakeField retains history from an earlier run; call reset_state() before a new run')
    for beam_id in wake_ids:
        names = {cmd.slice_set_name for cmd in commands if cmd.beam_id == beam_id}
        sim.beams[beam_id].wake_clock = WakeClock(sim.beams[beam_id], names)
    return wake_ids
