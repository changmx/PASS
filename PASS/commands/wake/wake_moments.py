"""Project current signed charge moments using a named, existing SliceSet."""
from dataclasses import dataclass

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
        clock.sample()
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
            if slices is None or slices.valid_turn != turn or slices.slice_id is None:
                raise ValueError(f"WakeField requires SliceSet {slice_name!r} updated by Slicer in turn {turn}")
            ids = np.asarray(slices.slice_id)
            if ids.shape != (end-start,) or ids.dtype.kind not in "iu":
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
            indices = np.flatnonzero(alive)+start
            local_ids = ids[alive]
            charge = bunch.ratio*bunch.num_charge*const.e
            if not np.isfinite(charge) or bunch.ratio < 0:
                raise ValueError("Wake macro charge/weight must be finite and weight nonnegative")
            if hasattr(p, "wake_macro_charge"):
                charge = np.asarray(p.wake_macro_charge[indices])
                if not np.all(np.isfinite(charge)):
                    raise ValueError("Individual wake source charges must be finite")
            for a, b in powers:
                weights = charge*np.asarray(p.x[indices], float)**a*np.asarray(p.y[indices], float)**b
                moments[(a, b)].append(np.bincount(local_ids, weights=weights, minlength=count))
            witnesses.append((bunch, indices, local_ids+offset))
            offset += count
        join = lambda values: np.concatenate(values) if values else np.empty(0, dtype=float)
        source = WakeSources(join(times), join(widths), {key: join(values) for key, values in moments.items()}, join(betas))
        return WakeProjection(source, witnesses)

    def project_gpu(self, *args, **kwargs):
        return project_gpu(*args, **kwargs)


# ----------------------------------------------------------------------------
# GPU: device source projection
# ----------------------------------------------------------------------------


_GPU_KERNELS = {}
_GPU_CODE = r'''
#if FLOAT_PARTICLES
using R=float;
#else
using R=double;
#endif
__device__ double monomial(double x,int n){
    double result=1.;
    while(n){if(n&1)result*=x;n>>=1;if(n)x*=x;}
    return result;
}
extern "C" __global__ void project(const R* x,const R* y,const int* tag,
    const int* ids,const double* charge,double q,int individual,int start,int end,
    int ns,double* moments,int* invalid,int use_shared,int mask){
    extern __shared__ double hist[];
    if(use_shared)for(int i=threadIdx.x;i<3*ns;i+=blockDim.x)hist[i]=0.;
    __syncthreads();
    for(int i=start+blockIdx.x*blockDim.x+threadIdx.x;i<end;i+=blockDim.x*gridDim.x){
        if(tag[i]<=0)continue;
        int id=ids[i-start];
        if(id<0||id>=ns){atomicExch(invalid,1);continue;}
        double qq=individual?charge[i]:q;
        if(!isfinite(qq)||((mask&2)&&!isfinite((double)x[i]))||((mask&4)&&!isfinite((double)y[i]))){atomicExch(invalid,2);continue;}
        double* out=use_shared?hist:moments;
        if(mask&1)atomicAdd(out+id,qq);
        if(mask&2)atomicAdd(out+ns+id,qq*(double)x[i]);
        if(mask&4)atomicAdd(out+2*ns+id,qq*(double)y[i]);
    }
    __syncthreads();
    if(use_shared)for(int i=threadIdx.x;i<3*ns;i+=blockDim.x)
        if(hist[i]!=0.)atomicAdd(moments+i,hist[i]);
}
extern "C" __global__ void project_general(const R* x,const R* y,const int* tag,
    const int* ids,const double* charge,double q,int individual,int start,int end,
    int ns,double* moments,int* invalid,int use_shared,int nm,const int* powers){
    extern __shared__ double hist[];
    if(use_shared)for(int i=threadIdx.x;i<nm*ns;i+=blockDim.x)hist[i]=0.;
    __syncthreads();
    for(int i=start+blockIdx.x*blockDim.x+threadIdx.x;i<end;i+=blockDim.x*gridDim.x){
        if(tag[i]<=0)continue;
        int id=ids[i-start];
        if(id<0||id>=ns){atomicExch(invalid,1);continue;}
        double qq=individual?charge[i]:q;
        double* out=use_shared?hist:moments;
        for(int k=0;k<nm;k++){
            double value=qq*monomial((double)x[i],powers[2*k])*monomial((double)y[i],powers[2*k+1]);
            if(!isfinite(value)){atomicExch(invalid,2);continue;}
            atomicAdd(out+k*ns+id,value);
        }
    }
    __syncthreads();
    if(use_shared)for(int i=threadIdx.x;i<nm*ns;i+=blockDim.x)
        if(hist[i]!=0.)atomicAdd(moments+i,hist[i]);
}
extern "C" __global__ void project_batch(const R* x,const R* y,const int* tag,
    const double* charge,int individual,const int* layout,const unsigned long long* ptr,
    const double* parameters,int total,int nm,const int* powers,int use_shared,
    double* moments,double* times,double* widths,double* betas,int* invalid){
    int b=blockIdx.y,start=layout[4*b],end=layout[4*b+1],ns=layout[4*b+2],offset=layout[4*b+3];
    const int* ids=(const int*)ptr[3*b];
    const double* centers=(const double*)ptr[3*b+1];
    const double* source_widths=(const double*)ptr[3*b+2];
    double q=parameters[3*b],beta=parameters[3*b+1],t0=parameters[3*b+2];
    extern __shared__ double hist[];
    if(use_shared)for(int j=threadIdx.x;j<nm*ns;j+=blockDim.x)hist[j]=0.;
    if(blockIdx.x==0)for(int j=threadIdx.x;j<ns;j+=blockDim.x){
        times[offset+j]=t0+centers[j];widths[offset+j]=source_widths[j];betas[offset+j]=beta;
    }
    __syncthreads();
    for(int i=start+blockIdx.x*blockDim.x+threadIdx.x;i<end;i+=blockDim.x*gridDim.x){
        if(tag[i]<=0)continue;
        int id=ids[i-start];if(id<0||id>=ns){atomicExch(invalid,1);continue;}
        double qq=individual?charge[i]:q;
        for(int k=0;k<nm;k++){
            double value=qq*monomial((double)x[i],powers[2*k])*monomial((double)y[i],powers[2*k+1]);
            if(!isfinite(value)){atomicExch(invalid,2);continue;}
            if(use_shared)atomicAdd(hist+k*ns+id,value);
            else atomicAdd(moments+k*total+offset+id,value);
        }
    }
    __syncthreads();
    if(use_shared)for(int j=threadIdx.x;j<nm*ns;j+=blockDim.x)
        if(hist[j]!=0.)atomicAdd(moments+(j/ns)*total+offset+j%ns,hist[j]);
}
'''


def _gpu_kernels(p):
    import cupy as cp
    key = (cp.cuda.runtime.getDevice(), np.dtype(p.dtype))
    if key not in _GPU_KERNELS:
        names = ('project', 'project_general', 'project_batch')
        module = cp.RawModule(code=_GPU_CODE, options=("--std=c++17",
            f"-DFLOAT_PARTICLES={int(p.dtype == np.float32)}"), name_expressions=names)
        _GPU_KERNELS[key] = {name: module.get_function(name) for name in names}
    return _GPU_KERNELS[key]


def project_gpu(beam, slice_name, turn, components, source_shape="uniform", stationary_batch=False):
    import cupy as cp
    p = beam.particles
    if not isinstance(p.z, cp.ndarray):
        raise TypeError("GPU wake tracking requires device particle arrays")
    if source_shape not in {"uniform", "point"}:
        raise ValueError("Wake source shape must be point or uniform")
    clock = getattr(beam, "wake_clock", None)
    if clock is None:
        clock = beam.wake_clock = WakeClock(beam)
    from .wake_periodic import validate_periodic_wake
    validate_periodic_wake(beam, slice_name, turn)
    if stationary_batch and getattr(clock, "_gpu_immediate_geometry", False):
        return project_batch_gpu(clock, slice_name, turn, components, source_shape)
    if getattr(clock, "_gpu_immediate_geometry", False):
        clock.arrival._advance(allow_beta_change=False)
    else:
        clock.sample()
    powers = sorted({c.source_powers for c in components})
    standard = all(key in {(0, 0), (1, 0), (0, 1)} for key in powers)
    mask = sum({(0, 0): 1, (1, 0): 2, (0, 1): 4}[key] for key in powers) if standard else 0
    if not standard:
        cache = clock.__dict__.setdefault("_projection_powers", {})
        key = (cp.cuda.runtime.getDevice(), tuple(powers))
        if key not in cache:
            cache.clear()
            cache[key] = cp.asarray(powers, dtype=cp.int32)
        device_powers = cache[key]
    times, widths, betas, witnesses = [], [], [], []
    moments = {key: [] for key in powers}
    offset = 0
    invalid = cp.zeros(1, dtype=cp.int32)
    for b in beam.bunches:
        start, end = b.start_idx, b.end_idx
        if end == start:
            continue
        slices = b.slice_sets.get(slice_name)
        if slices is None or slices.valid_turn != turn or slices.slice_id is None:
            raise ValueError(f"WakeField requires SliceSet {slice_name!r} updated by Slicer in turn {turn}")
        ids = slices.slice_id
        if ids.shape != (end-start,) or ids.dtype.kind not in "iu":
            raise ValueError("Wake SliceSet IDs do not match the current particle range")
        immediate_count = getattr(slices, "_alive_count", None) if getattr(clock, "_gpu_immediate_geometry", False) else None
        has_alive = immediate_count > 0 if immediate_count is not None else bool(cp.any(p.tag[start:end] > 0))
        if not has_alive:
            continue
        ids = cp.ascontiguousarray(ids, dtype=cp.int32)
        ns = len(slices.slice_table["z_center"])
        for component in components:
            component.model.validate_beta(b.beta)
            if component.velocity is not None:
                component.velocity.validate(b.beta)
        charge = b.ratio*b.num_charge*const.e
        if not np.isfinite(charge) or b.ratio < 0:
            raise ValueError("Wake macro charge/weight must be finite and weight nonnegative")
        q = getattr(p, "wake_macro_charge", p.arrival_offset)
        nm = 3 if standard else len(powers)
        values = cp.zeros((nm, ns), dtype=cp.float64)
        shared = nm*ns*8 <= 24576
        extra = (np.int32(mask),) if standard else (np.int32(nm), device_powers)
        _gpu_kernels(p)["project" if standard else "project_general"]((min(256, (end-start+255)//256),), (256,),
            (p.x, p.y, p.tag, ids, q, np.float64(charge), np.int32(hasattr(p, "wake_macro_charge")),
             np.int32(start), np.int32(end), np.int32(ns), values, invalid, np.int32(shared), *extra),
            shared_mem=nm*ns*8 if shared else 0)
        t, width = geometry_gpu(clock, b, slice_name, ids, p.tag[start:end] > 0)
        times.append(t)
        widths.append(width if source_shape == "uniform" else cp.zeros_like(width))
        betas.append(cp.full(ns, b.beta, dtype=cp.float64))
        for key in powers:
            row = {(0, 0): 0, (1, 0): 1, (0, 1): 2}[key] if standard else powers.index(key)
            moments[key].append(values[row])
        witnesses.append((b, ids, offset, ns))
        offset += ns
    # One small validation transfer; no particle-sized host copies.
    if int(invalid[0]):
        raise ValueError("A live particle has an invalid wake slice or non-finite source coordinate/charge")
    join = lambda values: cp.concatenate(values) if values else cp.empty(0, dtype=cp.float64)
    grid = None
    if len(witnesses) == 1 and getattr(clock, "_gpu_immediate_geometry", False):
        step = getattr(clock, "_gpu_uniform_grids", {}).get((witnesses[0][0].bunch_id, slice_name))
        if step is not None:
            grid = (step, 0. if source_shape == "point" else step)
    source = DeviceSources(join(times), join(widths), {k: join(v) for k, v in moments.items()}, join(betas), source_shape == "point", grid)
    return WakeProjection(source, witnesses)


def project_batch_gpu(clock, slice_name, turn, components, source_shape):
    """One CUDA launch across a stationary train, including empty source slots."""
    import cupy as cp
    p = clock.beam.particles
    clock.arrival._advance(allow_beta_change=False)
    powers = sorted({c.source_powers for c in components})
    layout, pointers, parameters, witnesses, references = [], [], [], [], []
    total, maximum_particles, maximum_slices = 0, 0, 0
    checked = set()
    for b in clock.beam.bunches:
        start, end = b.start_idx, b.end_idx
        if start == end: continue
        slices = b.slice_sets.get(slice_name)
        if slices is None or slices.valid_turn != turn or slices.slice_id is None:
            raise ValueError(f"WakeField requires SliceSet {slice_name!r} updated by Slicer in turn {turn}")
        if slices.slice_id.shape != (end-start,) or slices.slice_id.dtype.kind not in "iu":
            raise ValueError("Wake SliceSet IDs do not match the current particle range")
        if b.beta not in checked:
            for c in components:
                c.model.validate_beta(b.beta)
                if c.velocity is not None: c.velocity.validate(b.beta)
            checked.add(b.beta)
        charge = b.ratio*b.num_charge*const.e
        if not np.isfinite(charge) or b.ratio < 0:
            raise ValueError("Wake macro charge/weight must be finite and weight nonnegative")
        ids = cp.ascontiguousarray(slices.slice_id, dtype=cp.int32)
        frozen = clock.frozen.get((b.bunch_id, slice_name))
        if frozen is None or frozen[0] != turn:
            raise ValueError("Stationary wake projection requires freshly frozen slice geometry")
        centers, widths = frozen[2:4]
        ns = len(centers)
        layout.append((start, end, ns, total))
        pointers.append((ids.data.ptr, centers.data.ptr, widths.data.ptr))
        parameters.append((charge, b.beta, b.t0))
        witnesses.append((b, ids, total, ns))
        references.extend((ids, centers, widths))  # Keep every raw pointer alive until kernels complete.
        total += ns
        maximum_particles, maximum_slices = max(maximum_particles, end-start), max(maximum_slices, ns)
    moment_values = cp.zeros((len(powers), total), dtype=cp.float64)
    times, widths, betas = [cp.empty(total, dtype=cp.float64) for _ in range(3)]
    projection = WakeProjection(DeviceSources(times, widths, {k:moment_values[i] for i,k in enumerate(powers)},
                                               betas, source_shape == "point"), witnesses)
    if not witnesses: return projection
    # Layout can change after injection/regrouping. It is rebuilt from current
    # metadata rather than reusing stale device pointers or bunch offsets.
    d_layout, d_ptr, d_parameters = cp.asarray(layout, dtype=cp.int32), cp.asarray(pointers, dtype=cp.uint64), cp.asarray(parameters)
    power_key = tuple(powers)
    cache = clock.__dict__.setdefault("_batch_power_arrays", {})
    if cache.get("key") != power_key:
        cache.clear()
        cache.update(key=power_key, values=cp.asarray(powers, dtype=cp.int32))
    blocks = min(128, (maximum_particles+255)//256)
    shared = len(powers)*maximum_slices*8 <= 24576
    invalid = cp.zeros(1, dtype=cp.int32)
    q = getattr(p, "wake_macro_charge", p.arrival_offset)
    _gpu_kernels(p)["project_batch"]((blocks, len(witnesses)), (256,),
        (p.x, p.y, p.tag, q, np.int32(hasattr(p, "wake_macro_charge")), d_layout, d_ptr, d_parameters,
         np.int32(total), np.int32(len(powers)), cache["values"], np.int32(shared), moment_values, times, widths, betas, invalid),
        shared_mem=len(powers)*maximum_slices*8 if shared else 0)
    if int(invalid[0]):
        raise ValueError("A live particle has an invalid wake slice or non-finite source coordinate/charge")
    if source_shape == "point": widths.fill(0)
    projection.device_layout = (d_layout, d_ptr, blocks, references)
    return projection
