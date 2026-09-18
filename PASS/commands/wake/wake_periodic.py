"""User-triggered projection into a common physical observation window."""
import numpy as np

from PASS.utils.constants import const


def prepare_periodic_slices(p, bunch, slices, turn=None, location=None):
    xp = p.xp
    if xp is not np:
        return prepare_periodic_gpu(p, bunch, slices, turn, location)
    C = float(bunch.circum)
    if slices.explicit.z_max != 0. or not np.isclose(slices.explicit.z_min, -C, rtol=1e-13, atol=0):
        raise ValueError('Arrival-phase slices require an explicit [-C,0] interval')
    if not hasattr(slices, 'observation_time'):
        raise ValueError('Use Slicer to define a common observation event')
    bunch_slice = slice(bunch.start_idx, bunch.end_idx)
    relative = (bunch.t0 - slices.observation_time) - p.z[bunch_slice].astype(xp.float64) / (bunch.beta * const.c)
    # Use the same frequency factor and evaluation order as the CUDA path;
    # a final divide by C can move an exact full-period arrival across a seam.
    phase = -relative * (slices.observation_velocity / C)
    alive = p.tag[bunch_slice] > 0
    if not bool(xp.all(xp.isfinite(phase) | ~alive)):
        raise ValueError('Live particle arrival phase must be finite')
    history = getattr(slices, '_periodic_previous', {})
    samples = history.get(location, {})
    previous = samples.get(turn - 1) if turn is not None else None
    step = 0.
    if previous is not None and turn == previous[0] + 1 and previous[1].shape == phase.shape:
        # References and the observation epoch advance together.
        # Compare residual phase slip without folding stored coordinates.
        step = float(xp.max(xp.where(alive & previous[2], xp.abs(phase - previous[1]), 0.))) if len(phase) else 0.
    # Retain the previous-turn baseline when the user updates twice this turn.
    samples = {key: value for key, value in samples.items() if key == turn - 1}
    samples[turn] = (turn, phase.copy(), alive.copy())
    history[location] = samples
    slices._periodic_previous = history
    slices.periodic_max_step = step
    # Reduce elapsed arrival time into [0, C/v_obs). Exact integer phases
    # belong to the window start (z_phase=0), not the previous window end.
    slices._periodic_coordinate = xp.where(alive, -xp.remainder(-phase, 1.) * C, 0.).astype(p.z.dtype)


def validate_periodic_wake(beam, name, turn):
    sets = [b.slice_sets.get(name) for b in beam.bunches]
    if any(getattr(s, 'coordinate', None) == 'z_periodic' for s in sets):
        raise ValueError("WakeField cannot use z_periodic slices; select z_rel or arrival_phase")
    if not any(s is not None and s.periodic for s in sets):
        return
    if not all(s is not None and s.periodic and s.slice_table is not None for s in sets):
        raise ValueError('Periodic wake populations require common arrival-phase slices')
    windows = {(s.observation_time, s.observation_velocity) for s in sets}
    if len(windows) != 1:
        raise ValueError('Periodic slices must share one observation window')
    for s in sets:
        if getattr(s, 'periodic_max_step', 0.) > s.max_phase_slip:
            raise ValueError('Arrival slip exceeds the one-passage-per-turn wake model')


def prepare_periodic_gpu(p, bunch, slices, turn, location):
    """Fuse physical phase, validity and slip into one device pass."""
    import cupy as cp
    C = float(bunch.circum)
    if slices.explicit.z_max != 0. or not np.isclose(slices.explicit.z_min, -C, rtol=1e-13, atol=0):
        raise ValueError('Arrival-phase slices require an explicit [-C,0] interval')
    if not hasattr(slices, 'observation_time'):
        raise ValueError('Use Slicer to define a common observation event')
    n = bunch.end_idx - bunch.start_idx
    history = getattr(slices, '_periodic_previous', {})
    samples = history.get(location, {})
    previous = samples.get(turn - 1) if turn is not None else None
    use_previous = previous is not None and previous[1].shape == (n, )
    phase = cp.empty(n, dtype=cp.float64)
    alive = cp.empty(n, dtype=cp.int32)
    coordinate = cp.empty(n, dtype=p.z.dtype)
    status = cp.zeros(2, dtype=cp.float64)
    cache = slices.__dict__.setdefault('_periodic_kernels', {})
    key = (cp.cuda.runtime.getDevice(), np.dtype(p.z.dtype))
    if key not in cache:
        cache[key] = cp.RawKernel(_PERIODIC_CODE, 'arrival_phase', options=('--std=c++17', f'-DFLOAT_PARTICLES={int(p.z.dtype==np.float32)}'))
    if n:
        cache[key](
            (min(256, (n + 255) // 256), ), (256, ),
            (p.z, p.tag, np.int64(bunch.start_idx), np.int64(n), np.float64(bunch.t0 - slices.observation_time), np.float64(
                bunch.beta * const.c), np.float64(slices.observation_velocity / C), np.float64(C), previous[1] if use_previous else phase,
             previous[2] if use_previous else alive, np.int32(use_previous), phase, alive, coordinate, status))
    invalid, step = status.get()
    if invalid:
        raise ValueError('Live particle arrival phase must be finite')
    samples = {key: value for key, value in samples.items() if turn is not None and key == turn - 1}
    samples[turn] = (turn, phase, alive)
    history[location] = samples
    slices._periodic_previous = history
    slices.periodic_max_step = float(step)
    slices._periodic_coordinate = coordinate


_PERIODIC_CODE = r'''
#if FLOAT_PARTICLES
using R = float;
#else
using R = double;
#endif
extern "C" __global__ void arrival_phase(
    const R* z,
    const int* tag,
    long long start,
    long long n,
    double epoch,
    double velocity,
    double factor,
    double circumference,
    const double* old,
    const int* old_alive,
    int previous,
    double* phase,
    int* alive,
    R* coordinate,
    double* status
) {
    double maximum = 0.;
    int invalid = 0;
    for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < n; j += (long long)blockDim.x * gridDim.x) {
        int live = tag[start + j] > 0;
        double u = -(epoch - (double)z[start + j] / velocity) * factor;
        if (live && !isfinite(u))
            invalid = 1;
        if (live && previous && old_alive[j])
            maximum = fmax(maximum, fabs(u - old[j]));
        phase[j] = u;
        alive[j] = live;
        double elapsed = -u, part = elapsed - floor(elapsed);
        coordinate[j] = live ? (R)(-part * circumference) : (R)0;
    }
    if (invalid)
        atomicExch((unsigned long long*)status, __double_as_longlong(1.));
    if (maximum > 0.)
        atomicMax((unsigned long long*)(status + 1), (unsigned long long)__double_as_longlong(maximum));
}
'''
