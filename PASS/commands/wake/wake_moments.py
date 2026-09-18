"""Project current signed charge moments using a named, existing SliceSet."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from PASS.utils.constants import const
from .wake_conventions import require_cpu
from .wake_state import WakeSources, DeviceSources
from .wake_timing import WakeClock, geometry_gpu


@dataclass
class WakeProjection:
    sources: WakeSources
    witnesses: list


class WakeSourceProjector:

    def project_cpu(self, beam, slice_name, turn, components, source_shape="uniform"):
        if source_shape not in {"point", "uniform"}:
            raise ValueError("Wake source shape must be point or uniform")
        p = beam.particles
        clock = getattr(beam, "wake_clock", None)
        if clock is None:
            clock = beam.wake_clock = WakeClock(beam)
        from .wake_periodic import validate_periodic_wake
        validate_periodic_wake(beam, slice_name, turn)
        powers = {c.source_powers for c in components}
        times, widths, moments, witnesses = [], [], {key: [] for key in powers}, []
        offset = 0
        betas = []
        for bunch in beam.bunches:
            start, end = bunch.start_idx, bunch.end_idx
            if end == start:
                continue
            slices = bunch.slice_sets.get(slice_name)
            if slices is None or slices.slice_id is None:
                raise ValueError(f"WakeField requires SliceSet {slice_name!r} updated by Slicer in turn {turn}")
            ids = np.asarray(slices.slice_id)
            if ids.shape != (end - start, ) or ids.dtype.kind not in "iu":
                raise ValueError("Wake SliceSet IDs do not match the current particle range")
            alive = np.asarray(p.tag[start:end]) > 0
            count = len(slices.slice_table["z_center"])
            if np.any((ids[alive] < 0) | (ids[alive] >= count)):
                raise ValueError("A live particle has no valid wake slice; run Slicer after injection")
            if not np.any(alive):
                continue
            for component in components:
                component.model.validate_beta(bunch.beta)
                if component.velocity is not None:
                    component.velocity.validate(bunch.beta)
            t, width = clock.slice_geometry(bunch, slice_name, ids, alive)
            times.append(t)
            betas.append(np.full(count, bunch.beta))
            widths.append(width if source_shape == "uniform" else np.zeros_like(width))
            indices = np.flatnonzero(alive) + start
            local_ids = ids[alive]
            charge = bunch.ratio * bunch.num_charge * const.e
            if not np.isfinite(charge) or bunch.ratio < 0:
                raise ValueError("Wake macro charge/weight must be finite and weight nonnegative")
            for a, b in powers:
                weights = charge * np.asarray(p.x[indices], float)**a * np.asarray(p.y[indices], float)**b
                moments[(a, b)].append(np.bincount(local_ids, weights=weights, minlength=count))
            witnesses.append((bunch, indices, local_ids + offset))
            offset += count
        join = lambda values: np.concatenate(values) if values else np.empty(0, dtype=float)
        source = WakeSources(join(times), join(widths), {key: join(values) for key, values in moments.items()}, join(betas))
        return WakeProjection(source, witnesses)

    def project_gpu(self, *args, **kwargs):
        return project_gpu(*args, **kwargs)


# GPU: device source projection

_GPU_CODE = r'''
#if FLOAT_PARTICLES
using R = float;
#else
using R = double;
#endif
__device__ double monomial(
    double x,
    int n
) {
    double result = 1.;
    while (n) {
        if (n & 1)
            result *= x;
        n >>= 1;
        if (n)
            x *= x;
    }
    return result;
}
extern "C" __global__ void project_one(
    const R* x,
    const R* y,
    const int* tag,
    const int* ids,
    const double* centers,
    const double* widths,
    int float_centers,
    int float_widths,
    int start,
    int end,
    int n_slices,
    int nm,
    const int* powers,
    int use_shared,
    double q,
    double beta,
    double epoch,
    double velocity,
    int point,
    double* moments,
    double* times,
    double* duration,
    double* betas,
    int* invalid
) {
    extern __shared__ double hist[];
    if (use_shared)
        for (int j = threadIdx.x; j < nm * n_slices; j += blockDim.x)
            hist[j] = 0.;
    if (blockIdx.x == 0)
        for (int j = threadIdx.x; j < n_slices; j += blockDim.x) {
            double z = float_centers ? ((const float*)centers)[j] : centers[j];
            double dz = float_widths ? ((const float*)widths)[j] : widths[j];
            times[j] = epoch - z / velocity;
            duration[j] = point ? 0. : dz / velocity;
            betas[j] = beta;
            if (!isfinite(z) || !isfinite(dz) || dz < 0.)
                atomicExch(invalid, 3);
            if (j > 0) {
                double previous = float_centers ? ((const float*)centers)[j - 1] : centers[j - 1];
                if (!(z < previous))
                    atomicExch(invalid + 1, 1);
            }
        }
    __syncthreads();
    int has_live = 0;
    for (int i = start + blockIdx.x * blockDim.x + threadIdx.x; i < end; i += blockDim.x * gridDim.x) {
        if (tag[i] <= 0)
            continue;
        has_live = 1;
        int id = ids[i - start];
        if (id < 0 || id >= n_slices) {
            atomicExch(invalid, 1);
            continue;
        }
        for (int k = 0; k < nm; k++) {
            double v = q * monomial((double)x[i], powers[2 * k]) * monomial((double)y[i], powers[2 * k + 1]);
            if (!isfinite(v)) {
                atomicExch(invalid, 2);
                continue;
            }
            atomicAdd((use_shared ? hist : moments) + k * n_slices + id, v);
        }
    }
    if (__any_sync(0xffffffff, has_live) && ((threadIdx.x & 31) == 0))
        atomicExch(invalid + 2, 1);
    __syncthreads();
    if (use_shared)
        for (int j = threadIdx.x; j < nm * n_slices; j += blockDim.x)
            if (hist[j] != 0.)
                atomicAdd(moments + j, hist[j]);
}
extern "C" __global__ void project_batch(
    const R* x,
    const R* y,
    const int* tag,
    const int* layout,
    const unsigned long long* ptr,
    const double* parameters,
    int total,
    int nm,
    const int* powers,
    int use_shared,
    double* moments,
    double* times,
    double* widths,
    double* betas,
    int* invalid
) {
    int b = blockIdx.y, start = layout[4 * b], end = layout[4 * b + 1], n_slices = layout[4 * b + 2], offset = layout[4 * b + 3];
    const int* ids = (const int*)ptr[3 * b];
    const double* centers = (const double*)ptr[3 * b + 1];
    const double* source_widths = (const double*)ptr[3 * b + 2];
    double q = parameters[7 * b], beta = parameters[7 * b + 1], t0 = parameters[7 * b + 2], velocity = parameters[7 * b + 3];
    extern __shared__ double hist[];
    if (use_shared)
        for (int j = threadIdx.x; j < nm * n_slices; j += blockDim.x)
            hist[j] = 0.;
    if (blockIdx.x == 0)
        for (int j = threadIdx.x; j < n_slices; j += blockDim.x) {
            double z = parameters[7 * b + 4] ? ((const float*)centers)[j] : centers[j];
            double dz = parameters[7 * b + 5] ? ((const float*)source_widths)[j] : source_widths[j];
            times[offset + j] = t0 - z / velocity;
            widths[offset + j] = parameters[7 * b + 6] ? 0. : dz / velocity;
            betas[offset + j] = beta;
            if (!isfinite(z) || !isfinite(dz) || dz < 0.)
                atomicExch(invalid, 3);
            if (j > 0) {
                double previous = parameters[7 * b + 4] ? ((const float*)centers)[j - 1] : centers[j - 1];
                if (!(z < previous))
                    atomicExch(invalid + 1, 1);
            } else if (b > 0) {
                int last = layout[4 * (b - 1) + 2] - 1;
                const double* prev = (const double*)ptr[3 * (b - 1) + 1];
                double pz = parameters[7 * (b - 1) + 4] ? ((const float*)prev)[last] : prev[last];
                if (!(t0 - z / velocity > parameters[7 * (b - 1) + 2] - pz / parameters[7 * (b - 1) + 3]))
                    atomicExch(invalid + 1, 1);
            }
        }
    __syncthreads();
    int has_live = 0;
    for (int i = start + blockIdx.x * blockDim.x + threadIdx.x; i < end; i += blockDim.x * gridDim.x) {
        if (tag[i] <= 0)
            continue;
        has_live = 1;
        int id = ids[i - start];
        if (id < 0 || id >= n_slices) {
            atomicExch(invalid, 1);
            continue;
        }
        for (int k = 0; k < nm; k++) {
            double value = q * monomial((double)x[i], powers[2 * k]) * monomial((double)y[i], powers[2 * k + 1]);
            if (!isfinite(value)) {
                atomicExch(invalid, 2);
                continue;
            }
            if (use_shared)
                atomicAdd(hist + k * n_slices + id, value);
            else
                atomicAdd(moments + k * total + offset + id, value);
        }
    }
    if (__any_sync(0xffffffff, has_live) && ((threadIdx.x & 31) == 0))
        atomicExch(invalid + 2 + b, 1);
    __syncthreads();
    if (use_shared)
        for (int j = threadIdx.x; j < nm * n_slices; j += blockDim.x)
            if (hist[j] != 0.)
                atomicAdd(moments + (j / n_slices) * total + offset + j % n_slices, hist[j]);
}
'''


@lru_cache(maxsize=None)
def _get_gpu_kernels(dtype, device):
    import cupy as cp

    with cp.cuda.Device(device):
        names = ('project_batch', 'project_one')
        module = cp.RawModule(code=_GPU_CODE,
                              options=("--std=c++17", f"-DFLOAT_PARTICLES={int(np.dtype(dtype) == np.float32)}"),
                              name_expressions=names)
        return {name: module.get_function(name) for name in names}


def project_gpu(beam, slice_name, turn, components, source_shape="uniform"):
    """Project all bunches and saved slice geometries in one GPU launch."""
    import cupy as cp
    p = beam.particles
    if not isinstance(p.z, cp.ndarray):
        raise TypeError('GPU wake requires device particles')
    if source_shape not in {'point', 'uniform'}:
        raise ValueError('Wake source shape must be point or uniform')
    clock = getattr(beam, 'wake_clock', None)
    if clock is None:
        clock = beam.wake_clock = WakeClock(beam)
    from .wake_periodic import validate_periodic_wake
    validate_periodic_wake(beam, slice_name, turn)
    powers = sorted({component.source_powers for component in components})
    layout, pointers, parameters, witnesses, references = [], [], [], [], []
    total = maximum_particles = maximum_slices = 0
    checked = set()
    for bunch in beam.bunches:
        start, end = bunch.start_idx, bunch.end_idx
        if start == end:
            continue
        slice_set = bunch.slice_sets.get(slice_name)
        if slice_set is None or slice_set.slice_id is None or slice_set.slice_table is None:
            raise ValueError('WakeField requires an explicitly generated SliceSet')
        if slice_set.slice_id.shape != (end - start, ) or slice_set.slice_id.dtype.kind not in 'iu':
            raise ValueError('Wake SliceSet IDs do not match the current particle range')
        if bunch.beta not in checked:
            for component in components:
                component.model.validate_beta(bunch.beta)
                if component.velocity is not None:
                    component.velocity.validate(bunch.beta)
            checked.add(bunch.beta)
        charge = bunch.ratio * bunch.num_charge * const.e
        if not np.isfinite(charge) or bunch.ratio < 0:
            raise ValueError('Wake macro charge/weight must be finite and weight nonnegative')
        ids = cp.ascontiguousarray(slice_set.slice_id, dtype=cp.int32)
        centers = cp.ascontiguousarray(slice_set.slice_table['z_center'])
        widths = cp.ascontiguousarray(slice_set.slice_table['delta_z'])
        if centers.dtype not in (np.float32, np.float64) or widths.dtype not in (np.float32, np.float64):
            centers = centers.astype(cp.float64)
            widths = widths.astype(cp.float64)
        n_slices = len(centers)
        if widths.shape != (n_slices, ) or n_slices == 0:
            raise ValueError('Wake slice geometry must contain matching nonempty arrays')
        epoch, velocity = (slice_set.observation_time, slice_set.observation_velocity) if slice_set.periodic else (bunch.t0, bunch.beta * const.c)
        if not np.isfinite(epoch) or not np.isfinite(velocity) or velocity <= 0.:
            raise ValueError('Wake slice reference time and positive velocity must be finite')
        layout.append((start, end, n_slices, total))
        pointers.append((ids.data.ptr, centers.data.ptr, widths.data.ptr))
        parameters.append((charge, bunch.beta, epoch, velocity, centers.dtype == np.float32, widths.dtype == np.float32, source_shape == 'point'))
        witnesses.append((bunch, ids, total, n_slices))
        references.extend((ids, centers, widths))
        total += n_slices
        maximum_particles = max(maximum_particles, end - start)
        maximum_slices = max(maximum_slices, n_slices)
    moments = cp.zeros((len(powers), total), dtype=cp.float64)
    times, widths, betas = (cp.empty(total, dtype=cp.float64) for _ in range(3))
    projection = WakeProjection(DeviceSources(times, widths, {
        k: moments[i]
        for i, k in enumerate(powers)
    }, betas, source_shape == 'point'), witnesses)
    if not witnesses:
        return projection
    from .wake_models import device_arrays
    powers_gpu = device_arrays(clock, ('source_powers', tuple(powers)), (np.asarray(powers, dtype=np.int32), ))[0]
    blocks = min(128, max(1, (maximum_particles + 255) // 256))
    use_shared_memory = len(powers) * maximum_slices * 8 <= 24576
    invalid = cp.zeros(2 + len(witnesses), dtype=cp.int32)
    if len(witnesses) == 1:
        start, end, n_slices, _ = layout[0]
        charge, beta, epoch, velocity, centers_are_float32, widths_are_float32, point_source = parameters[0]
        ids, centers, saved_widths = references
        _get_gpu_kernels(p.dtype.str, cp.cuda.runtime.getDevice())['project_one'](
            (blocks, ),
            (256, ),
            (
                p.x,
                p.y,
                p.tag,
                ids,
                centers,
                saved_widths,
                np.int32(centers_are_float32),
                np.int32(widths_are_float32),
                np.int32(start),
                np.int32(end),
                np.int32(n_slices),
                np.int32(len(powers)),
                powers_gpu,
                np.int32(use_shared_memory),
                np.float64(charge),
                np.float64(beta),
                np.float64(epoch),
                np.float64(velocity),
                np.int32(point_source),
                moments,
                times,
                widths,
                betas,
                invalid,
            ),
            shared_mem=len(powers) * n_slices * 8 if use_shared_memory else 0,
        )
        error, unordered, alive = invalid.get()
        if error:
            raise ValueError('A live particle has an invalid wake slice or non-finite source coordinate/charge/geometry')
        if not alive:
            projection.sources = DeviceSources(times[:0], widths[:0], {
                k: v[:0]
                for k, v in projection.sources.moments.items()
            }, betas[:0], source_shape == 'point')
            projection.witnesses = []
        else:
            object.__setattr__(projection.sources, 'increasing', not bool(unordered))
        return projection
    # Rebuild pointer metadata from current SliceSets; injection and regrouping
    # can replace storage. References below keep all buffers alive through kick.
    cache = clock.__dict__.setdefault('_projection_metadata', {})
    layout_key = (cp.cuda.runtime.getDevice(), tuple(layout), tuple(pointers))
    if cache.get('layout_key') != layout_key:
        cache['layout_key'] = layout_key
        cache['layout'] = cp.asarray(layout, dtype=cp.int32)
        cache['pointers'] = cp.asarray(pointers, dtype=cp.uint64)
    parameter_key = (cp.cuda.runtime.getDevice(), tuple(parameters))
    if cache.get('parameter_key') != parameter_key:
        cache['parameter_key'] = parameter_key
        cache['parameters'] = cp.asarray(parameters, dtype=cp.float64)
    layout_gpu, pointers_gpu, parameters_gpu = cache['layout'], cache['pointers'], cache['parameters']
    _get_gpu_kernels(p.dtype.str, cp.cuda.runtime.getDevice())['project_batch'](
        (blocks, len(witnesses)),
        (256, ),
        (
            p.x,
            p.y,
            p.tag,
            layout_gpu,
            pointers_gpu,
            parameters_gpu,
            np.int32(total),
            np.int32(len(powers)),
            powers_gpu,
            np.int32(use_shared_memory),
            moments,
            times,
            widths,
            betas,
            invalid,
        ),
        shared_mem=len(powers) * maximum_slices * 8 if use_shared_memory else 0,
    )
    status = invalid.get()
    error, unordered = status[:2]
    if error:
        raise ValueError('A live particle has an invalid wake slice or non-finite source coordinate/charge/geometry')
    object.__setattr__(projection.sources, 'increasing', not bool(unordered))
    active = np.flatnonzero(status[2:])
    if len(active) != len(witnesses):
        indices = np.concatenate([np.arange(layout[i][3], layout[i][3] + layout[i][2], dtype=np.int64)
                                  for i in active]) if len(active) else np.empty(0, dtype=np.int64)
        projection.sources = projection.sources.ordered(cp.asarray(indices))
        compact = []
        rows = []
        offset = 0
        for i in active:
            bunch, ids, _, n_slices = witnesses[i]
            compact.append((bunch, ids, offset, n_slices))
            rows.append((bunch.start_idx, bunch.end_idx, n_slices, offset))
            offset += n_slices
        projection.witnesses = compact
        if not compact:
            return projection
        layout_gpu = cp.asarray(rows, dtype=cp.int32)
        pointers_gpu = cp.asarray([pointers[i] for i in active], dtype=cp.uint64)
    projection.device_layout = (layout_gpu, pointers_gpu, blocks, references)
    return projection
