"""CPU solvers with their CUDA implementations below. History is solver-independent."""
from collections import deque
from functools import lru_cache

import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from scipy.linalg import expm

from .wake_conventions import require_cpu
from .wake_models import ResonatorWakeModel
from .wake_spectrum import RationalWakeModel
from .wake_state import WakeSources


def _kernel(component, tau, widths, memory_time=None):
    model = component.model
    if memory_time is None:
        return component.scale*model.averaged(tau, widths, component.longitudinal)
    tau, widths = np.broadcast_arrays(tau, widths)
    point = widths == 0
    result = np.where(tau <= memory_time, model.evaluate(tau, component.longitudinal), 0.0)
    finite = ~point
    if np.any(finite):
        t, h = tau[finite], widths[finite]/2
        # Truncate the kernel, not merely entire historical bins.
        result[finite] = (model.primitive(np.minimum(t+h, memory_time), component.longitudinal)
                          - model.primitive(np.minimum(t-h, memory_time), component.longitudinal))/(2*h)
    return component.scale*result


class DirectSliceSolver:
    def solve_cpu(self, components, sources, targets, memory_time=None, target_betas=None):
        result = np.zeros((len(components), len(targets)), dtype=float)
        coupled = [[c.source_moment(source) for c in components] for source in sources]
        # Two-dimensional tiling bounds memory even for long history.
        for ti in range(0, len(targets), 128):
            target = targets[ti:ti+128]
            for source_index, source in enumerate(sources):
                for si in range(0, len(source.times), 1024):
                    tau = target[:, None]-source.times[None, si:si+1024]
                    width = source.widths[None, si:si+1024]
                    for ci, component in enumerate(components):
                        moment = coupled[source_index][ci][si:si+1024]
                        if np.any(moment):
                            result[ci, ti:ti+len(target)] += _kernel(component, tau, width, memory_time) @ moment
        if target_betas is not None:
            for ci, component in enumerate(components):
                result[ci] *= component.witness_factor(target_betas)
        return result

    def solve_gpu(self, components, sources, targets, memory_time=None, target_betas=None):
        import cupy as cp
        return direct_gpu(components, [DeviceSources.upload(s) for s in sources], cp.asarray(targets), memory_time,
                      None if target_betas is None else cp.asarray(target_betas))


class FFTConvolutionSolver:
    def __init__(self):
        self.resources = {}

    @staticmethod
    def compatible(sources, targets):
        if len(sources) != 1 or len(targets) < 2:
            return False
        source = sources[0]
        if not np.array_equal(source.times, targets):
            return False
        steps = np.diff(targets)
        step = (targets[-1]-targets[0])/(len(targets)-1)
        # Uniform physical times acquire a few ULPs of spacing jitter when
        # represented relative to a large revolution timestamp.
        rounding = 8*np.spacing(np.max(np.abs(targets)))
        return (step > 0 and np.all(steps > 0)
                and np.allclose(steps, step, rtol=1e-9, atol=max(abs(step)*1e-11, rounding))
                and np.allclose(source.widths, source.widths[0], rtol=1e-10, atol=abs(step)*1e-12)
                and source.widths[0] <= step*(1+1e-10)+rounding)

    def solve_cpu(self, components, sources, targets, memory_time=None, target_betas=None):
        if len(sources) == 1 and len(targets) < 2 and np.array_equal(sources[0].times, targets):
            return DirectSliceSolver().solve_cpu(components, sources, targets, memory_time, target_betas)
        if not self.compatible(sources, targets):
            raise ValueError("FFT wake requires one increasing uniform grid and common bin widths <= spacing")
        source, n = sources[0], len(targets)
        from .convolution import source_channels
        result = np.empty((len(components), n))
        step = (targets[-1]-targets[0])/(n-1)
        key = ("cpu", tuple(id(c) for c in components), n, step, source.widths[0], memory_time)
        if self.resources.get("key") != key:
            self.resources.clear()
            self.resources["key"] = key
        for causal in {c.model.causal for c in components}:
            ids = [i for i, c in enumerate(components) if c.model.causal == causal]
            selected = [components[i] for i in ids]
            channels, channel_ids = source_channels(selected)
            size = next_fast_len(2*n-1 if causal else 3*n-2)
            offset = 0 if causal else n-1
            response = self.resources.get(causal)
            if response is None:
                delays = np.arange(-offset, n, dtype=float)*step
                response = rfft(np.stack([_kernel(c, delays, source.widths[0], memory_time) for c in selected]), size, axis=1)
                self.resources[causal] = response
            transformed = rfft(np.stack([c.source_moment(source) for c in channels]), size, axis=1)
            values = irfft(transformed[channel_ids]*response, size, axis=1)[:, offset:offset+n]
            for row, ci in enumerate(ids):
                result[ci] = values[row]*(1. if target_betas is None else components[ci].witness_factor(target_betas))
        return result

    def solve_gpu(self, components, sources, targets, memory_time=None, target_betas=None):
        import cupy as cp
        if len(sources) != 1 or not bool(cp.array_equal(cp.asarray(sources[0].times), cp.asarray(targets))):
            raise ValueError("FFT wake requires a single common source/target grid")
        source = DeviceSources.upload(sources[0])
        return fft_gpu(components, source, memory_time, None if target_betas is None else cp.asarray(target_betas), resources=self.resources)


@lru_cache(maxsize=4096)
def _mode_step(omega, q, dt):
    # Dimensionless oscillator state (omega*u, u'). The third coordinate
    # integrates constant drive; scaling avoids badly conditioned SI matrices.
    matrix = np.array([[0., 1., 0.], [-1., -1/q, 1.], [0., 0., 0.]])
    step = expm(matrix*(omega*dt))
    return step[:2, :2], step[:2, 2]/omega


class RecursiveResonatorSolver:
    # Preserve the former stateless reference entry points. State evolution
    # is explicitly advance_cpu/advance_gpu, never inherited solver behavior.
    solve_cpu = DirectSliceSolver.solve_cpu
    solve_gpu = DirectSliceSolver.solve_gpu

    def advance_cpu(self, components, source, state):
        if any(not isinstance(c.model, ResonatorWakeModel) for c in components):
            raise ValueError("Recursive solver requires only resonator components")
        n = len(source.times)
        out = np.zeros((len(components), n))
        if not n:
            return out
        events = {}
        for i, (time, width) in enumerate(zip(source.times, source.widths)):
            events.setdefault(float(time), [[], [], []])[2].append(i)
            if width:
                events.setdefault(float(time-width/2), [[], [], []])[0].append((i, 1/width))
                events.setdefault(float(time+width/2), [[], [], []])[0].append((i, -1/width))
            else:
                events[float(time)][1].append(i)
        for ci, component in enumerate(components):
            model = component.model
            vector = state.mode_amplitudes.get(ci, np.zeros(2)).copy()
            last = state.last_time
            drive = 0.0
            moment = component.source_moment(source)
            output_index = 1 if component.longitudinal else 0
            amplitude = model.omega*model.r/model.q*component.scale
            for time, (edges, points, targets) in sorted(events.items()):
                if last is not None and time > last:
                    transition, forcing = _mode_step(model.omega, model.q, time-last)
                    vector = transition @ vector + forcing*drive
                if last is not None and time < last:
                    raise ValueError("Resonator events moved backwards in physical time")
                impulse = np.sum(moment[points]) if points else 0.0
                value = vector[output_index]
                if component.longitudinal:
                    value += impulse/2
                out[ci, targets] = amplitude*value*component.witness_factor(source.betas[targets])
                vector[1] += impulse
                for i, rate in edges:
                    drive += moment[i]*rate
                last = time
            state.mode_amplitudes[ci] = vector
        state.last_time = max(events)
        return out

    def advance_gpu(self, components, source, state):
        return advance_gpu(components, DeviceSources.upload(source), state, "recursive")


class RecursiveModalSolver:
    solve_cpu = DirectSliceSolver.solve_cpu
    solve_gpu = DirectSliceSolver.solve_gpu

    def advance_cpu(self, components, source, state):
        if any(not isinstance(c.model, RationalWakeModel) or not c.model.causal for c in components):
            raise ValueError("Temporal modal recursion requires causal left-half-plane rational responses")
        out = np.zeros((len(components), len(source.times)))
        if not len(source.times):
            return out
        events = {}
        for i, (time, width) in enumerate(zip(source.times, source.widths)):
            events.setdefault(float(time), [[], [], []])[2].append(i)
            if width:
                events.setdefault(float(time-width/2), [[], [], []])[0].append((i, 1/width))
                events.setdefault(float(time+width/2), [[], [], []])[0].append((i, -1/width))
            else:
                events[float(time)][1].append(i)
        ordered_events = sorted(events.items())
        for ci, component in enumerate(components):
            model = component.model
            vector = state.mode_amplitudes.get(ci, np.zeros(len(model.poles), complex)).copy()
            last, drive = state.last_time, 0.
            moment = component.source_moment(source)
            for time, (edges, points, targets) in ordered_events:
                if last is not None:
                    if time < last:
                        raise ValueError("Modal events moved backwards in physical time")
                    dt = time-last
                    vector = np.exp(model.poles*dt)*vector+np.expm1(model.poles*dt)/model.poles*drive
                impulse = np.sum(moment[points]) if points else 0.
                value = np.real(np.sum(model.residues*(vector+impulse/2)))
                out[ci, targets] = component.scale*value*component.witness_factor(source.betas[targets])
                vector += impulse
                for i, rate in edges:
                    drive += moment[i]*rate
                last = time
            state.mode_amplitudes[ci] = vector
        state.last_time = ordered_events[-1][0]
        return out

    def advance_gpu(self, components, source, state):
        return advance_gpu(components, DeviceSources.upload(source), state, "modal")


def solve_wake_cpu(components, current, state, *, turn, solver="direct", memory_turns=0, memory_time=None, fft_plan=None):
    """Return coefficients and a candidate state; caller commits after the kick."""
    if state.last_turn is not None and turn <= state.last_turn:
        raise ValueError("WakeField was executed twice or its turn counter moved backwards; reset its state before a new run")
    candidate = state.fork()
    persistent = memory_turns is None or memory_turns > 0
    if len(current.times) and persistent:
        start = float(np.min(current.times-current.widths/2))
        end = float(np.max(current.times+current.widths/2))
        if state.last_source_end is not None:
            tolerance = 32*np.spacing(max(abs(start), abs(state.last_source_end), 1e-12))
            if start < state.last_source_end-tolerance:
                raise ValueError("Wake passages overlap in physical time across tracking turns. "
                                 "This requires asynchronous passage scheduling, which PASS does not yet support; "
                                 "changing or wrapping z would give incorrect long-range kicks.")
        candidate.last_source_end = end
    if solver not in {"direct", "fft", "recursive", "modal"}:
        raise ValueError(f"Unknown explicit wake solver {solver!r}")
    recursive = solver in {"recursive", "modal"}
    if recursive:
        if memory_turns is not None or memory_time is not None:
            raise ValueError("Recursive resonator needs unlimited memory (memory_turns=None, memory_time=None)")
        algorithm = RecursiveResonatorSolver() if solver == "recursive" else RecursiveModalSolver()
        result = algorithm.advance_cpu(components, current, candidate)
    else:
        if not persistent:
            candidate.history.clear()
        else:
            while candidate.history and memory_turns is not None and candidate.history[0][0] < turn-memory_turns:
                candidate.history.popleft()
            if memory_time is not None and len(current.times):
                cutoff = float(np.min(current.times))-memory_time
                while candidate.history and np.max(candidate.history[0][1].times+candidate.history[0][1].widths/2, initial=-np.inf) < cutoff:
                    candidate.history.popleft()
        old_sources = [snapshot for _, snapshot in candidate.history]
        # Current uniform grid uses FFT even when earlier turns are irregular.
        fft_gpu = fft_plan or FFTConvolutionSolver()
        order = np.argsort(current.times, kind="stable")
        ordered = WakeSources(current.times[order], current.widths[order],
                              {key: value[order] for key, value in current.moments.items()}, current.betas[order])
        use_fft = solver == "fft"
        if use_fft:
            result = np.empty((len(components), len(order)))
            result[:, order] = fft_gpu.solve_cpu(components, [ordered], ordered.times, memory_time, ordered.betas)
            if old_sources:
                result += DirectSliceSolver().solve_cpu(components, old_sources, current.times, memory_time, current.betas)
        else:
            result = DirectSliceSolver().solve_cpu(components, [*old_sources, current], current.times, memory_time, current.betas)
        if persistent:
            candidate.history.append((turn, current))
    candidate.last_turn = turn
    return result, candidate


# ----------------------------------------------------------------------------
# GPU: explicit solvers and bounded affine prefix scans
# ----------------------------------------------------------------------------


from .wake_models import averaged_gpu, evaluate_gpu, device_arrays
from .wake_state import DeviceSources
from .wake_velocity import factor_gpu
from .wake_components import moment_gpu


def kernel_gpu(component, tau, width, point, memory_time):
    import cupy as cp
    if point:
        value = evaluate_gpu(component.model, tau, component.longitudinal)
        if memory_time is not None:
            value = cp.where(tau <= memory_time, value, 0.)
    else:
        value = averaged_gpu(component.model, tau, width, component.longitudinal, memory_time)
    return component.scale*value


def direct_gpu(components, sources, targets, memory_time=None, target_betas=None):
    import cupy as cp
    out = cp.zeros((len(components), len(targets)), dtype=cp.float64)
    for source in sources:
        coupled = [moment_gpu(c, source) for c in components]
        for ti in range(0, len(targets), 128):
            for si in range(0, len(source.times), 1024):
                tau = targets[ti:ti+128, None]-source.times[None, si:si+1024]
                for ci, component in enumerate(components):
                    out[ci, ti:ti+128] += kernel_gpu(component, tau, source.widths[None, si:si+1024],
                        source.point, memory_time) @ coupled[ci][si:si+1024]
    if target_betas is not None:
        for ci, component in enumerate(components):
            out[ci] *= factor_gpu(component, target_betas, True)
    return out


def fft_gpu(components, source, memory_time=None, target_betas=None, *, resources=None):
    import cupy as cp
    n = len(source.times)
    target_betas = source.betas if target_betas is None else target_betas
    if n < 2:
        return direct_gpu(components, [source], source.times, memory_time, target_betas)
    times = source.times
    if source.grid is None:
        step = (times[-1]-times[0])/(n-1)
        largest = cp.max(cp.abs(times))
        rounding = 8*(cp.nextafter(largest, cp.inf)-largest)
        valid = ((step > 0) & cp.all(cp.diff(times) > 0)
            & cp.all(cp.abs(cp.diff(times)-step) <= cp.maximum(abs(step)*1e-11, rounding)+1e-9*abs(step))
            & cp.all(cp.abs(source.widths-source.widths[0]) <= abs(step)*1e-12+1e-10*abs(source.widths[0]))
            & (source.widths[0] <= step*(1+1e-10)+rounding))
        if not bool(valid):
            raise ValueError("FFT wake requires one increasing uniform grid and common bin widths <= spacing")
    else:
        # A fresh equal-length Slicer with zero arrival correction supplies an
        # exact grid contract. It is invalidated by intervening transport.
        step = source.grid[0]
    from .convolution import source_channels
    result = cp.empty((len(components), n), dtype=cp.float64)
    cache = {} if resources is None else resources
    key = ("gpu", cp.cuda.runtime.getDevice(), tuple(id(c) for c in components), n, source.grid, source.point, memory_time)
    if source.grid is None or cache.get("key") != key:
        cache.clear()
        cache["key"] = key
    # Group equal FFT lengths, use batched transforms and share moment FFTs.
    for causal in {c.model.causal for c in components}:
        ids = [i for i, c in enumerate(components) if c.model.causal == causal]
        size = next_fast_len(2*n-1 if causal else 3*n-2)
        offset = 0 if causal else n-1
        selected = [components[i] for i in ids]
        channels, channel_ids = source_channels(selected)
        response = cache.get(causal)
        if response is None:
            delays = cp.arange(-offset, n, dtype=cp.float64)*step
            response = cp.fft.rfft(cp.stack([kernel_gpu(c, delays, source.widths[0], source.point, memory_time)
                                            for c in selected]), size, axis=1)
            cache[causal] = response
            cache[(causal, "channels")] = cp.asarray(channel_ids)
        moments = cp.stack([moment_gpu(c, source) for c in channels])
        spectra = cp.fft.rfft(moments, size, axis=1)
        values = cp.fft.irfft(spectra[cache[(causal, "channels")]]*response, size, axis=1)[:, offset:offset+n]
        for row, ci in enumerate(ids):
            result[ci] = values[row]*factor_gpu(components[ci], target_betas, True)
    return result


def event_arrays_gpu(source, components):
    """Combine simultaneous impulses/edges before observing half self kicks."""
    import cupy as cp
    n = len(source.times)
    local = source.times-source.times[0]
    if source.point:
        times, center = cp.unique(local, return_inverse=True)
        impulses = cp.stack([cp.bincount(center, weights=moment_gpu(c, source), minlength=len(times)) for c in components])
        drives = cp.zeros_like(impulses)
    else:
        times, indices = cp.unique(cp.concatenate((local, local-source.widths/2,
                                                   local+source.widths/2)), return_inverse=True)
        center, begin, end = indices[:n], indices[n:2*n], indices[2*n:]
        positive = source.widths > 0
        width = cp.where(positive, source.widths, 1.)
        impulses, drives = [], []
        for c in components:
            charge = moment_gpu(c, source)
            impulses.append(cp.bincount(center, weights=cp.where(positive, 0., charge), minlength=len(times)))
            rate = cp.where(positive, charge/width, 0.)
            edges = cp.bincount(begin, weights=rate, minlength=len(times))-cp.bincount(end, weights=rate, minlength=len(times))
            drives.append(cp.cumsum(edges))
        impulses, drives = cp.stack(impulses), cp.stack(drives)
    return times, center, cp.ascontiguousarray(impulses), cp.ascontiguousarray(drives)


def solve_wake_gpu(components, current, state, *, turn, solver="direct", memory_turns=0, memory_time=None, fft_plan=None):
    import cupy as cp
    if state.last_turn is not None and turn <= state.last_turn:
        raise ValueError("WakeField was executed twice or its turn counter moved backwards; reset its state before a new run")
    if solver not in {"direct", "fft", "recursive", "modal"}:
        raise ValueError(f"Unknown explicit wake solver {solver!r}")
    if not isinstance(current, DeviceSources) and len(current.times):
        # Public host-array entry: validate the contract before upload. The
        # command projection has already checked every device bunch reference.
        for c in components:
            for beta in (float(np.min(current.betas)), float(np.max(current.betas))):
                c.model.validate_beta(beta)
                if c.velocity is not None:
                    c.velocity.validate(beta)
    current = DeviceSources.upload(current)
    candidate = state.fork()
    candidate.history = deque((t, DeviceSources.upload(s)) for t, s in candidate.history)
    candidate.mode_amplitudes = {k: cp.asarray(v) for k, v in candidate.mode_amplitudes.items()}
    persistent = memory_turns is None or memory_turns > 0
    if len(current.times) and persistent:
        start, end = cp.stack((cp.min(current.times-current.widths/2), cp.max(current.times+current.widths/2))).get()
        if state.last_source_end is not None:
            tolerance = 32*np.spacing(max(abs(start), abs(state.last_source_end), 1e-12))
            if start < state.last_source_end-tolerance:
                raise ValueError("Wake passages overlap in physical time across tracking turns; asynchronous passage scheduling is required")
        candidate.last_source_end = float(end)
    if solver in {"recursive", "modal"}:
        if memory_turns is not None or memory_time is not None:
            raise ValueError("State recursion requires unlimited memory")
        result = advance_gpu(components, current, candidate, solver)
    else:
        if not persistent:
            candidate.history.clear()
        else:
            while candidate.history and memory_turns is not None and candidate.history[0][0] < turn-memory_turns:
                candidate.history.popleft()
            if memory_time is not None and len(current.times):
                cutoff = float(cp.min(current.times))-memory_time
                while candidate.history and float(cp.max(candidate.history[0][1].times+candidate.history[0][1].widths/2)) < cutoff:
                    candidate.history.popleft()
        old = [s for _, s in candidate.history]
        if solver == "fft":
            if current.grid is not None:
                result = fft_gpu(components, current, memory_time, resources=None if fft_plan is None else fft_plan.resources)
            else:
                order = cp.argsort(current.times)
                result = cp.empty((len(components), len(order)), dtype=cp.float64)
                result[:, order] = fft_gpu(components, current.ordered(order), memory_time, resources=None if fft_plan is None else fft_plan.resources)
            if old:
                result += direct_gpu(components, old, current.times, memory_time, current.betas)
        else:
            result = direct_gpu(components, [*old, current], current.times, memory_time, current.betas)
        if persistent:
            candidate.history.append((turn, current))
    candidate.last_turn = turn
    return result, candidate


_SCAN_KERNELS = {}
_SCAN_CODE = r'''
#include <cupy/complex.cuh>
using C = complex<double>;
extern "C" __global__ void rlc_scan(const double* times, const double* impulses,
    const double* drives, const double* omega, const double* q, double* state,
    double* output, int n, double last, int has_last, const int* plane) {
    const int j=threadIdx.x, m=blockIdx.x, B=128;
    __shared__ double a00[B],a01[B],a10[B],a11[B],b0[B],b1[B],base[2];
    if(j==0){base[0]=state[2*m];base[1]=state[2*m+1];} __syncthreads();
    const double w=omega[m], alpha=w/(2*q[m]);
    for(int start=0;start<n;start+=B){
        int i=start+j;
        double c=1,s=0,dt=0,fu=0,drive=0,impulse=0;
        if(i<n){
            dt=times[i]-(i>0?times[i-1]:(has_last?last:times[0]));
            dt=fmax(dt,0.);
            if(q[m]>.5){
                double d=w*sqrt((1-.5/q[m])*(1+.5/q[m])),e=exp(-alpha*dt);
                c=e*cos(d*dt);s=e*(d*dt==0?dt:sin(d*dt)/d);
            }else if(q[m]==.5){c=exp(-alpha*dt);s=c*dt;
            }else{
                double d=alpha*sqrt((1-2*q[m])*(1+2*q[m]));
                double e=exp(-w*w/(alpha+d)*dt);
                c=e*(1+exp(-2*d*dt))/2;s=-e*expm1(-2*d*dt)/(2*d);
            }
            fu=(1-c-alpha*s)/w;
            if(fmax(w,alpha)*dt<1e-4)
                fu=w*dt*dt*(.5-alpha*dt/3+(4*alpha*alpha-w*w)*dt*dt/24
                    +(4*alpha*w*w-8*alpha*alpha*alpha)*dt*dt*dt/120);
            drive=i>0?drives[m*n+i-1]:0.;impulse=impulses[m*n+i];
        }
        a00[j]=c+alpha*s;a01[j]=w*s;a10[j]=-w*s;a11[j]=c-alpha*s;
        b0[j]=fu*drive;b1[j]=s*drive+impulse;__syncthreads();
        for(int stride=1;stride<B;stride*=2){
            double u=a00[j],v=a01[j],w0=a10[j],x=a11[j],y=b0[j],z=b1[j];
            if(j>=stride){
                int k=j-stride;
                y+=u*b0[k]+v*b1[k];z+=w0*b0[k]+x*b1[k];
                double nu=u*a00[k]+v*a10[k],nv=u*a01[k]+v*a11[k];
                double nw=w0*a00[k]+x*a10[k],nx=w0*a01[k]+x*a11[k];
                u=nu;v=nv;w0=nw;x=nx;
            }
            __syncthreads();
            a00[j]=u;a01[j]=v;a10[j]=w0;a11[j]=x;b0[j]=y;b1[j]=z;
            __syncthreads();
        }
        double u=a00[j]*base[0]+a01[j]*base[1]+b0[j];
        double v=a10[j]*base[0]+a11[j]*base[1]+b1[j];
        if(i<n)output[m*n+i]=plane[m]?(v-impulse/2):u;
        __syncthreads();
        if(j==min(B,n-start)-1){base[0]=u;base[1]=v;}__syncthreads();
    }
    if(j==0){state[2*m]=base[0];state[2*m+1]=base[1];}
}

extern "C" __global__ void modal_scan(const double* times, const double* impulses,
    const double* drives, const C* poles, C* state, C* output, int n,
    double last, int has_last) {
    const int j=threadIdx.x,m=blockIdx.x,B=128;
    __shared__ C aa[B],bb[B],base;
    if(j==0)base=state[m];__syncthreads();
    const C p=poles[m];
    for(int start=0;start<n;start+=B){
        int i=start+j;C a(1,0),b(0,0);double impulse=0;
        if(i<n){
            double dt=fmax(0.,times[i]-(i>0?times[i-1]:(has_last?last:times[0])));
            C z=p*dt;a=exp(z);
            C em1(expm1(z.real())*cos(z.imag())-2*sin(z.imag()/2)*sin(z.imag()/2),
                  exp(z.real())*sin(z.imag()));
            impulse=impulses[i];
            b=em1/p*(i>0?drives[i-1]:0.)+impulse;
        }
        aa[j]=a;bb[j]=b;__syncthreads();
        for(int stride=1;stride<B;stride*=2){
            C u=aa[j],v=bb[j];
            if(j>=stride){v+=u*bb[j-stride];u*=aa[j-stride];}
            __syncthreads();aa[j]=u;bb[j]=v;__syncthreads();
        }
        C value=aa[j]*base+bb[j];
        if(i<n)output[m*n+i]=value-impulse/2;
        __syncthreads();if(j==min(B,n-start)-1)base=value;__syncthreads();
    }
    if(j==0)state[m]=base;
}
'''


def advance_gpu(components, source, state, solver):
    import cupy as cp
    if not len(source.times):
        return cp.zeros((len(components), 0), dtype=cp.float64)
    if solver == "recursive" and any(not isinstance(c.model, ResonatorWakeModel) for c in components):
        raise ValueError("Recursive solver requires only resonator components")
    if solver == "modal" and any(not isinstance(c.model, RationalWakeModel) or not c.model.causal for c in components):
        raise ValueError("Temporal modal recursion requires causal rational responses")
    times, centers, impulses, drives = event_arrays_gpu(source, components)
    origin, first, last = cp.stack((source.times[0], times[0], times[-1])).get()
    first += origin
    if state.last_time is not None and first < state.last_time-32*np.spacing(max(abs(first), abs(state.last_time), 1e-12)):
        raise ValueError("Mode events moved backwards in physical time")
    key = (cp.cuda.runtime.getDevice(), solver)
    if key not in _SCAN_KERNELS:
        _SCAN_KERNELS[key] = cp.RawKernel(_SCAN_CODE, "rlc_scan" if solver == "recursive" else "modal_scan", options=("--std=c++17",))
    n, nc = len(times), len(components)
    previous = np.float64(0. if state.last_time is None else state.last_time-origin)
    has_previous = np.int32(state.last_time is not None)
    if solver == "recursive":
        params = ([c.model.omega for c in components], [c.model.q for c in components],
                  np.array([c.longitudinal for c in components], dtype=np.int32))
        # Component tuples are stable for a command; cache on the first model,
        # including the complete tuple so distinct groups cannot alias.
        signature = tuple((c.model.omega, c.model.q, c.longitudinal) for c in components)
        omega, q, plane = device_arrays(components[0].model, ("scan", signature), params)
        vectors = cp.stack([state.mode_amplitudes.get(ci, cp.zeros(2)) for ci in range(nc)])
        output = cp.empty((nc, n), dtype=cp.float64)
        _SCAN_KERNELS[key]((nc,), (128,), (times, impulses, drives, omega, q, vectors, output,
            np.int32(n), previous, has_previous, plane))
        out = cp.empty((nc, len(source.times)), dtype=cp.float64)
        for ci, c in enumerate(components):
            out[ci] = output[ci, centers]*(c.model.omega*c.model.r/c.model.q*c.scale)*factor_gpu(c, source.betas, True)
            state.mode_amplitudes[ci] = vectors[ci].copy()
    else:
        out = cp.empty((nc, len(source.times)), dtype=cp.float64)
        for ci, c in enumerate(components):
            poles, residues = device_arrays(c.model, "scan_modes", (c.model.poles, c.model.residues))
            vector = state.mode_amplitudes.get(ci, cp.zeros(len(poles), dtype=cp.complex128)).copy()
            output = cp.empty((len(poles), n), dtype=cp.complex128)
            _SCAN_KERNELS[key]((len(poles),), (128,), (times, impulses[ci], drives[ci], poles, vector,
                output, np.int32(n), previous, has_previous))
            out[ci] = cp.real(cp.sum(residues[:, None]*output[:, centers], axis=0))*c.scale*factor_gpu(c, source.betas, True)
            state.mode_amplitudes[ci] = vector
    state.last_time = float(origin+last)
    return out
