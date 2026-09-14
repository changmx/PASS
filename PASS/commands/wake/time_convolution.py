"""Causal history on a physical-time mesh, independent of revolution periods.

Point charges use linear deposition; uniform source bins use the integral of
the same hat basis. Completed time blocks feed the existing dyadic convolution.
The last, unsealed block stays open across passages. Local exact corrections
remove the mesh error around the causal jump (including the half self wake).
Far-field time projection remains an explicit, convergent approximation.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from .convolution import (ConvolutionGrid, ConvolutionState, HistoryUpdate,
                          PartitionedConvolution, BlockPreviewTransaction, _array_state)
from .wake_state import WakeSources
from .wake_solvers import _kernel


@dataclass(frozen=True)
class TimeGrid:
    step: float
    block_size: int = 64
    origin: float | None = None

    def __post_init__(self):
        if not np.isfinite(self.step) or self.step <= 0:
            raise ValueError("Time grid step must be positive finite")
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, (int, np.integer)) or self.block_size < 2:
            raise ValueError("Time grid block size must be an integer >= 2")
        if self.origin is not None and not np.isfinite(self.origin):
            raise ValueError("Time grid origin must be finite or null")


def _subset(source, index, xp):
    if xp is np:
        return WakeSources(source.times[index], source.widths[index],
                           {k: v[index] for k, v in source.moments.items()}, source.betas[index])
    from .wake_state import DeviceSources
    return DeviceSources(source.times[index].copy(), source.widths[index].copy(),
                         {k: v[index].copy() for k, v in source.moments.items()}, source.betas[index].copy())


def _join(a, b, xp):
    if a is None or not len(a.times):
        return b
    cls = WakeSources
    if xp is not np:
        from .wake_state import DeviceSources as cls
    return cls(xp.concatenate((a.times, b.times)), xp.concatenate((a.widths, b.widths)),
               {k: xp.concatenate((a.moments[k], b.moments[k])) for k in b.moments},
               xp.concatenate((a.betas, b.betas)))


class TimeConvolutionState:
    def __init__(self, plan, start_turn=0):
        self.plan_key, self.start_turn, self.count = plan.key, int(start_turn), 0
        self.origin = plan.grid.origin
        self.last_end = None
        self.blocks = ConvolutionState(plan.blocks)
        self.open = plan.xp.zeros((len(plan.blocks.channels), 0), dtype=np.float64)
        self.recent = None

    @property
    def nbytes(self):
        recent = 0 if self.recent is None else sum(v.nbytes for v in
            (self.recent.times, self.recent.widths, self.recent.betas, *self.recent.moments.values()))
        return self.blocks.nbytes+self.open.nbytes+recent

    def state_dict(self):
        host = lambda v: v.get() if hasattr(v, "get") else np.asarray(v)
        recent = None if self.recent is None else {
            "times": host(self.recent.times).tolist(), "widths": host(self.recent.widths).tolist(),
            "betas": host(self.recent.betas).tolist(),
            "moments": [{"powers": list(k), "values": host(v).tolist()} for k, v in self.recent.moments.items()]}
        return {"kind": "physical_time", "plan_key": self.plan_key, "start_turn": self.start_turn,
                "count": self.count, "origin": self.origin, "last_end": self.last_end,
                "blocks": self.blocks.state_dict(), "open": _array_state(self.open), "recent": recent}

    @classmethod
    def restore(cls, data, plan):
        if data.get("kind") != "physical_time" or data["plan_key"] != plan.key:
            raise ValueError("Physical-time checkpoint does not match the kernel/time grid")
        for key in ("start_turn", "count"):
            if isinstance(data[key], bool) or not isinstance(data[key], int) or data[key] < 0:
                raise ValueError("Physical-time checkpoint has invalid passage counters")
        out = cls(plan, data["start_turn"])
        out.count = data["count"]
        out.origin, out.last_end = data["origin"], data["last_end"]
        for value in (out.origin, out.last_end):
            if value is not None and not np.isfinite(value):
                raise ValueError("Physical-time checkpoint has an invalid clock")
        if (out.last_end is not None and out.origin is None) or (plan.grid.origin is not None and out.origin != plan.grid.origin):
            raise ValueError("Physical-time checkpoint origin is inconsistent")
        out.blocks = ConvolutionState.restore(data["blocks"], plan.blocks)
        item = data["open"]
        shape = tuple(item["shape"])
        if len(shape) != 2 or shape[0] != len(plan.blocks.channels) or not 0 <= shape[1] <= plan.grid.block_size+2:
            raise ValueError("Physical-time checkpoint open block shape is invalid")
        value = np.asarray(item["real"], float)
        if value.size != math.prod(shape) or not np.all(np.isfinite(value)) or np.any(np.asarray(item["imag"]) != 0):
            raise ValueError("Physical-time checkpoint open block is invalid")
        out.open = plan.xp.asarray(value.reshape(shape))
        if data["recent"] is not None:
            r = data["recent"]
            out.recent = WakeSources(r["times"], r["widths"],
                {tuple(m["powers"]): m["values"] for m in r["moments"]}, r["betas"])
            if set(out.recent.moments) != {c.source_powers for c in plan.components}:
                raise ValueError("Physical-time checkpoint source moments are inconsistent")
            if plan.backend == "gpu":
                from .wake_state import DeviceSources
                out.recent = DeviceSources.upload(out.recent)
        cursor = (out.blocks.start_turn+out.blocks.count)*plan.grid.block_size
        if out.last_end is None:
            if cursor or shape[1] or out.recent is not None:
                raise ValueError("Physical-time checkpoint has buffers without a source clock")
        else:
            position = (out.last_end-out.origin)/plan.grid.step
            expected = math.floor(position)//plan.grid.block_size*plan.grid.block_size
            if out.count == 0 or cursor != expected or shape[1] != math.ceil(position)+2-cursor:
                raise ValueError("Physical-time checkpoint clock and open block are inconsistent")
            if out.recent is None or not len(out.recent.times):
                raise ValueError("Physical-time checkpoint is missing the causal correction sources")
            ends = out.recent.times+out.recent.widths/2
            tolerance = 32*abs(np.spacing(out.last_end))
            if not bool(plan.xp.all((ends <= out.last_end+tolerance) & (ends >= out.last_end-2*plan.grid.step-tolerance))):
                raise ValueError("Physical-time checkpoint recent sources are inconsistent with its clock")
        return out


class TimeHistoryUpdate:
    def __init__(self, state, operations, opened, recent, origin, end, blocks=None):
        self.state, self.count = state, state.count
        self.operations, self.opened, self.recent = operations, opened, recent
        self.origin, self.end = origin, end
        self.blocks = state.blocks if blocks is None else blocks
        self.original_blocks, self.block_count = state.blocks, state.blocks.count

    def commit(self):
        s = self.state
        if (s.count != self.count or s.blocks is not self.original_blocks
                or s.blocks.count != self.block_count):
            raise RuntimeError("Physical-time update is stale or already committed")
        s.blocks = self.blocks
        for spectrum, additions, plan in self.operations:
            HistoryUpdate(s.blocks, spectrum, additions, plan).commit()
        s.open, s.recent, s.origin, s.last_end = self.opened, self.recent, self.origin, self.end
        s.count += 1
        self.operations = ()


class TimeConvolution:
    def __init__(self, components, grid, memory_time, *, backend="cpu", method="dyadic", max_workspace_mb=1024):
        if not np.isfinite(memory_time) or memory_time <= 0:
            raise ValueError("Physical-time convolution requires finite positive Memory time")
        self.components, self.grid, self.memory_time, self.backend = tuple(components), grid, memory_time, backend
        period = grid.step*grid.block_size
        history = math.ceil(memory_time/period)+1
        self.blocks = PartitionedConvolution(components,
            ConvolutionGrid(period, 1, grid.block_size, period, grid.step), history,
            backend=backend, method=method, memory_time=memory_time, max_workspace_mb=max_workspace_mb)
        self.xp = self.blocks.xp
        self.budget = int(max_workspace_mb*1024**2)
        key = {"version": 1, "blocks": self.blocks.key, "origin": grid.origin,
               "projection": "integrated_hat_with_causal_correction"}
        self.key = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        # Samples and a trapezoidal primitive give the exact interaction of
        # the deposited hat basis without iterating over a uniform source bin.
        sample_count = math.ceil(memory_time/grid.step)+3
        state_bytes = 16*math.prod(self.blocks.frequency_shape)*(
            self.blocks.input_capacity*len(self.blocks.channels)+self.blocks.pending_capacity*len(components))
        self.estimated_bytes = self.blocks.estimated_bytes+2*state_bytes+sample_count*len(components)*16
        if self.estimated_bytes > self.budget:
            raise MemoryError("Physical-time convolution, transaction and kernel buffers exceed workspace budget")
        times = np.arange(sample_count)*grid.step
        h = np.stack([_kernel(c, times, 0., memory_time) for c in components])
        primitive = np.c_[h[:, 0]/2, h[:, 0, None]/2+np.cumsum((h[:, :-1]+h[:, 1:])/2, axis=1)]
        self.response, self.primitive = self.xp.asarray(h), self.xp.asarray(primitive)

    def _coupled(self, component, source, witness=False):
        if self.backend == "gpu":
            from .wake_velocity import factor_gpu as factor
            from .wake_components import moment_gpu as moment
            return factor(component, source.betas, True) if witness else moment(component, source)
        return component.witness_factor(source.betas) if witness else component.source_moment(source)

    def _deposit(self, source, origin, start, size):
        xp, dt = self.xp, self.grid.step
        out = xp.zeros((len(self.blocks.channels), size), dtype=np.float64)
        if not len(source.times):
            return out
        positions = (source.times-origin)/dt-start
        moments = xp.stack([self._coupled(c, source) for c in self.blocks.channels])
        if self.backend == "gpu":
            deposit_gpu(positions, source.widths/dt, moments, out)
            return out
        widths = source.widths/dt
        point = widths == 0
        left = np.floor(positions[point]).astype(np.int64)
        frac = positions[point]-left
        for ci in range(len(moments)):
            np.add.at(out[ci], left, moments[ci, point]*(1-frac))
            np.add.at(out[ci], left+1, moments[ci, point]*frac)
        for i in np.flatnonzero(~point):
            a, b = positions[i]-widths[i]/2, positions[i]+widths[i]/2
            nodes = np.arange(math.floor(a), math.ceil(b)+1)
            def hat(x):
                return np.where(x <= -1, 0., np.where(x < 0, .5*(x+1)**2,
                    np.where(x < 1, 1-.5*(1-x)**2, 1.)))
            if widths[i] < 1:
                x, half = positions[i]-nodes, widths[i]/2
                knot = np.ceil(x-half)
                alpha = .5+(knot-x)/widths[i]
                tri = lambda v: np.maximum(1-np.abs(v), 0.)
                split = .5*(tri(x-half)+tri(knot))*alpha+.5*(tri(knot)+tri(x+half))*(1-alpha)
                weights = np.where(np.abs(knot-x) < half, split, tri(x))
            else:
                weights = (hat(b-nodes)-hat(a-nodes))/widths[i]
            keep = (nodes >= 0) & (nodes < size)
            nodes, weights = nodes[keep], weights[keep]
            out[:, nodes] += moments[:, i, None]*weights
        return out

    def _sample(self, ci, x, integral=False):
        xp = self.xp
        h, p = self.response[ci], self.primitive[ci]
        k = xp.floor(x).astype(np.int64)
        f = x-k
        j = xp.clip(k, 0, len(h)-2)
        value = h[j]+f*(h[j+1]-h[j])
        if integral:
            value = p[j]+f*h[j]+.5*f*f*(h[j+1]-h[j])
            return xp.where(x < -1, 0., xp.where(x < 0, .5*h[0]*(x+1)**2,
                xp.where(x >= len(h)-1, p[-1], value)))
        return xp.where(x < -1, 0., xp.where(x < 0, h[0]*(x+1),
            xp.where(x >= len(h)-1, 0., value)))

    def _near_correction(self, source, recent, origin, frame_bytes=0):
        xp, dt = self.xp, self.grid.step
        result = xp.zeros((len(self.components), len(source.times)), dtype=np.float64)
        joined = _join(recent, source, xp)
        if not len(source.times) or not len(joined.times):
            return result
        order = xp.argsort(joined.times)
        near = _subset(joined, order, xp)
        if self.backend == "gpu":
            from .wake_models import TabulatedWakeModel
            if all(isinstance(c.model, TabulatedWakeModel) for c in self.components):
                return table_near_correction_gpu(self, source, near, origin)
        radius = float(xp.max(near.widths))/2+2*dt
        lo = xp.searchsorted(near.times, source.times-radius, side="left")
        hi = xp.searchsorted(near.times, source.times+radius, side="right")
        moments = [self._coupled(c, near) for c in self.components]
        # Bound pair workspaces even for overlapping, very wide populations.
        for begin in range(0, len(source.times), 256):
            end = min(begin+256, len(source.times))
            counts = hi[begin:end]-lo[begin:end]
            npairs = int(xp.sum(counts))
            if npairs == 0:
                continue
            if self.estimated_bytes+frame_bytes+npairs*160 > self.budget:
                raise MemoryError("Physical-time near correction exceeds workspace; reduce source overlap or slice width")
            ti = xp.repeat(xp.arange(begin, end), counts)
            starts = xp.cumsum(counts)-counts
            si = xp.repeat(lo[begin:end]-starts, counts)+xp.arange(npairs)
            tau, width = source.times[ti]-near.times[si], near.widths[si]
            keep = xp.abs(tau) <= width/2+2*dt
            ti, si, tau, width = ti[keep], si[keep], tau[keep], width[keep]
            target = (source.times[ti]-origin)/dt
            u = (near.times[si]-origin)/dt
            w = width/dt
            for ci, c in enumerate(self.components):
                if self.backend == "gpu":
                    from .wake_solvers import kernel_gpu as kernel
                    from .wake_models import TabulatedWakeModel
                    approximate = projected_pair_gpu(target, u, w, self.response[ci], self.primitive[ci])
                    exact = (exact_table_pairs_gpu(c, tau, width, self.memory_time) if isinstance(c.model, TabulatedWakeModel)
                             else kernel(c, tau, width, False, self.memory_time))
                else:
                    approximate = self._projected_pair_cpu(ci, target, u, w)
                    exact = _kernel(c, tau, width, self.memory_time)
                contribution = (exact-approximate)*moments[ci][si]
                result[ci] += xp.bincount(ti, weights=contribution, minlength=len(source.times))
        return result

    def _projected_pair_cpu(self, ci, target, position, width):
        left = np.floor(target)
        fraction = target-left
        safe = np.where(width > 0, width, 1.)
        result = 0.
        for node, weight in ((left, 1-fraction), (left+1, fraction)):
            x, half = node-position, width/2
            point = self._sample(ci, x)
            uniform = (self._sample(ci, x+half, True)-self._sample(ci, x-half, True))/safe
            # Integrate a linear interval, splitting at its only possible knot.
            # This avoids cancellation of primitives for almost pointlike bins.
            knot = np.ceil(x-half)
            alpha = .5+(knot-x)/safe
            middle = self._sample(ci, knot)
            split = .5*(self._sample(ci, x-half)+middle)*alpha+\
                    .5*(middle+self._sample(ci, x+half))*(1-alpha)
            narrow = np.where(np.abs(knot-x) < half, split, point)
            uniform = np.where(width < 1, narrow, uniform)
            result = result+weight*np.where(width > 0, uniform, point)
        return result

    def preview(self, source, state, *, turn):
        xp, dt, block = self.xp, self.grid.step, self.grid.block_size
        if isinstance(turn, bool) or not isinstance(turn, (int, np.integer)) or turn < 0:
            raise ValueError("Physical-time turn must be a nonnegative integer")
        if self.backend == "gpu":
            from .wake_state import DeviceSources
            source = DeviceSources.upload(source)
            arrays = (source.times, source.widths, source.betas, *source.moments.values())
            if any(not isinstance(v, xp.ndarray) or v.device.id != xp.cuda.runtime.getDevice() for v in arrays):
                raise ValueError("Physical-time source arrays belong to another backend/device")
            if any(v.dtype != np.float64 or not v.flags.c_contiguous for v in arrays):
                source = DeviceSources(xp.ascontiguousarray(source.times, dtype=np.float64),
                    xp.ascontiguousarray(source.widths, dtype=np.float64),
                    {k: xp.ascontiguousarray(v, dtype=np.float64) for k, v in source.moments.items()},
                    xp.ascontiguousarray(source.betas, dtype=np.float64))
        powers = {c.source_powers for c in self.components}
        if not powers.issubset(source.moments):
            raise ValueError("Physical-time sources are missing required component moments")
        if powers != set(source.moments):
            source = type(source)(source.times, source.widths,
                                  {k: source.moments[k] for k in powers}, source.betas)
        if state is None:
            state = TimeConvolutionState(self, turn)
        if state.plan_key != self.key or turn != state.start_turn+state.count:
            raise ValueError("Physical-time plan or consecutive passage counter is inconsistent")
        if not isinstance(state.open, xp.ndarray) or (self.backend == "gpu" and state.open.device.id != xp.cuda.runtime.getDevice()):
            raise ValueError("Physical-time state belongs to another backend/device; restore its checkpoint")
        if not len(source.times):
            return xp.zeros((len(self.components), 0)), TimeHistoryUpdate(state, (), state.open,
                state.recent, state.origin, state.last_end)
        if source.times.ndim != 1 or source.widths.shape != source.times.shape or source.betas.shape != source.times.shape:
            raise ValueError("Physical-time sources have incompatible shapes")
        valid = xp.all(xp.isfinite(source.times) & xp.isfinite(source.widths) & (source.widths >= 0)
                       & xp.isfinite(source.betas) & (source.betas > 0) & (source.betas <= 1))
        valid &= all(v.shape == source.times.shape for v in source.moments.values())
        for v in source.moments.values():
            valid &= xp.all(xp.isfinite(v))
        metadata = xp.stack((valid, xp.min(source.betas), xp.max(source.betas),
            xp.min(source.times-source.widths/2), xp.max(source.times+source.widths/2)))
        if self.backend == "gpu":
            metadata = metadata.get()
        valid, beta_min, beta_max, first, last = map(float, metadata)
        if not valid:
            raise ValueError("Physical-time sources must have finite times, widths, betas and moments")
        for c in self.components:
            for beta in (beta_min, beta_max):
                c.model.validate_beta(beta)
                if c.velocity is not None:
                    c.velocity.validate(beta)
        resolution = max(abs(np.spacing(first)), abs(np.spacing(last)))
        if resolution > dt*1e-3:
            raise ValueError("Physical-time clock cannot resolve the requested grid step")
        if state.last_end is not None and first < state.last_end-32*resolution:
            raise ValueError("Physical-time causal passages overlap or arrive out of order; use an event-ordered source stream")
        origin = first-2*dt if state.origin is None else state.origin
        block_state = state.blocks
        # If the complete compact-support response has expired, skip an empty
        # interval exactly. The next block starts with zero past, not a reused
        # or rescaled old spectrum. This also bounds work after long gaps.
        if first >= origin and (state.last_end is None or first > state.last_end+self.memory_time+2*dt):
            block_state = ConvolutionState(self.blocks, start_turn=math.floor((first-origin)/dt)//block)
            opened_before = xp.zeros((len(self.blocks.channels), 0), dtype=np.float64)
        else:
            opened_before = state.open
        start = (block_state.start_turn+block_state.count)*block
        min_node = math.floor((first-origin)/dt)
        stop = math.ceil((last-origin)/dt)+2
        if min_node < start:
            raise ValueError("Physical-time source precedes the open grid; check origin and arrival order")
        size = max(stop-start, opened_before.shape[1])
        nblocks = max(1, math.ceil(size/block))
        frame_bytes = nblocks*block*8*(len(self.blocks.channels)+len(self.components))
        frame_bytes += len(source.times)*8*(16+2*len(self.components)+2*len(self.blocks.channels))
        frequency_cells = math.prod(self.blocks.frequency_shape)
        frame_bytes += 16*frequency_cells*nblocks*len(self.blocks.channels)
        if self.blocks.method == "uniform":
            frame_bytes += 16*frequency_cells*nblocks*self.blocks.memory_turns*len(self.components)
        else:
            for length, count in self.blocks.levels:
                if count > 1:
                    triggers = (block_state.count+nblocks)//length-block_state.count//length
                    frame_bytes += 16*frequency_cells*triggers*(length+count-1)*len(self.components)
        if self.estimated_bytes+frame_bytes > self.budget:
            raise MemoryError("Physical-time passage span and staged history exceed workspace; increase step/block size or budget")
        density = self._deposit(source, origin, start, nblocks*block)
        density[:, :opened_before.shape[1]] += opened_before
        values = xp.empty((len(self.components), nblocks*block), dtype=np.float64)
        sealed = max(0, math.floor((last-origin)/dt)//block-block_state.start_turn-block_state.count)
        operations = []
        with BlockPreviewTransaction(block_state) as transaction:
            for bi in range(nblocks):
                row, update = self.blocks.preview_block(density[:, bi*block:(bi+1)*block][:, None, :],
                    block_state, turn=block_state.start_turn+block_state.count)
                values[:, bi*block:(bi+1)*block] = row[:, 0]
                if bi < sealed:
                    operations.append((update.spectrum, tuple(update.additions), update.plan))
                transaction.apply(update)
        position = (source.times-origin)/dt-start
        index = xp.floor(position).astype(np.int64)
        fraction = position-index
        result = values[:, index]*(1-fraction)+values[:, index+1]*fraction
        result += self._near_correction(source, state.recent, origin, frame_bytes)
        for ci, c in enumerate(self.components):
            result[ci] *= self._coupled(c, source, True)
        joined = _join(state.recent, source, xp)
        recent = _subset(joined, joined.times+joined.widths/2 >= last-2*dt, xp)
        opened = density[:, sealed*block:size].copy()
        return result, TimeHistoryUpdate(state, operations, opened, recent, origin, last, block_state)

    def step(self, source, state=None, *, turn):
        values, update = self.preview(source, state, turn=turn)
        if not bool(self.xp.all(self.xp.isfinite(values))):
            raise FloatingPointError("Physical-time convolution produced nonfinite coefficients")
        update.commit()
        return values, update.state

    @property
    def diagnostics(self):
        return {**self.blocks.diagnostics, "time_step_s": self.grid.step, "time_block_size": self.grid.block_size,
                "history_horizon_s": self.memory_time, "estimated_peak_bytes": self.estimated_bytes,
                "projection": "integrated_hat", "causal_near_correction": True}


# ----------------------------------------------------------------------------
# GPU: fused CUDA convolution kernels
# ----------------------------------------------------------------------------


_CACHE = {}
_CODE = r'''
__device__ double hat(double x) {
    if (x <= -1.) return 0.;
    if (x < 0.) return .5*(x+1.)*(x+1.);
    if (x < 1.) return 1.-.5*(1.-x)*(1.-x);
    return 1.;
}
__device__ double tri(double x) { return fmax(1.-fabs(x), 0.); }
extern "C" __global__ void deposit_time(const double* position, const double* width,
    const double* moments, double* density, long long n, long long size, int channels) {
    long long i = (long long)blockDim.x*blockIdx.x+threadIdx.x;
    if (i >= n) return;
    double u = position[i], w = width[i];
    if (w == 0.) {
        long long j = (long long)floor(u);
        double f = u-j;
        for (int c=0;c<channels;c++) {
            double q = moments[(long long)c*n+i];
            atomicAdd(density+(long long)c*size+j, q*(1.-f));
            atomicAdd(density+(long long)c*size+j+1, q*f);
        }
    } else {
        double a = u-w/2., b = u+w/2.;
        long long left=(long long)floor(a), right=(long long)ceil(b);
        for (long long j=left;j<=right;j++) {
            if (j < 0 || j >= size) continue;
            double weight=(hat(b-j)-hat(a-j))/w;
            if (w < 1.) {
                double x=u-j, half=w/2., k=ceil(x-half), alpha=.5+(k-x)/w;
                weight = fabs(k-x)<half ? .5*(tri(x-half)+tri(k))*alpha
                    +.5*(tri(k)+tri(x+half))*(1.-alpha) : tri(x);
            }
            for (int c=0;c<channels;c++)
                atomicAdd(density+(long long)c*size+j, moments[(long long)c*n+i]*weight);
        }
    }
}
__device__ double sample_time(const double* h, const double* p, long long n, double x, bool integral) {
    if (x < -1.) return 0.;
    if (x < 0.) return integral ? .5*h[0]*(x+1.)*(x+1.) : h[0]*(x+1.);
    if (x >= n-1) return integral ? p[n-1] : 0.;
    long long k=(long long)floor(x);
    double f=x-k, d=h[k+1]-h[k];
    return integral ? p[k]+f*h[k]+.5*f*f*d : h[k]+f*d;
}
__device__ double mesh_pair(double target, double u, double w,
    const double* h, const double* p, long long nh) {
    double left=floor(target), f=target-left, value=0.;
    for (int j=0;j<2;j++) {
        double x=left+j-u, v=sample_time(h,p,nh,x,false);
        if (w >= 1.) {
            v=(sample_time(h,p,nh,x+w/2.,true)-sample_time(h,p,nh,x-w/2.,true))/w;
        } else if (w > 0.) {
            double half=w/2., k=ceil(x-half);
            if (fabs(k-x)<half) {
                double alpha=.5+(k-x)/w, mid=sample_time(h,p,nh,k,false);
                v=.5*(sample_time(h,p,nh,x-half,false)+mid)*alpha
                    +.5*(mid+sample_time(h,p,nh,x+half,false))*(1.-alpha);
            }
        }
        value+=(j ? f : 1.-f)*v;
    }
    return value;
}
extern "C" __global__ void projected_pair(const double* target, const double* position,
    const double* width, const double* h, const double* p, double* result, long long n, long long nh) {
    long long i=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    if (i < n) result[i]=mesh_pair(target[i],position[i],width[i],h,p,nh);
}
__device__ double table_value(double t, const double* x, const double* y,
    const double* slope, const double* primitive, int n, bool integral) {
    if (t < 0.) return 0.;
    if (t > x[n-1]) return integral ? primitive[n-1] : 0.;
    int lo=0, hi=n-1;
    while (hi-lo>1) { int mid=(hi+lo)/2; if (x[mid]<=t) lo=mid; else hi=mid; }
    double d=t-x[lo];
    if (integral) return primitive[lo]+d*(y[lo]+.5*slope[lo]*d);
    return (y[lo]+slope[lo]*d)*(t==0. ? .5 : 1.);
}
extern "C" __global__ void exact_table_pairs(const double* tau, const double* width,
    const double* x, const double* y, const double* slope, const double* primitive,
    double* output, long long count, int n, double horizon, double scale) {
    long long i=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    if (i>=count) return;
    double t=tau[i], w=width[i], value;
    if (w>0.) {
        double a=fmin(t-w/2., horizon), b=fmin(t+w/2., horizon);
        value=(table_value(b,x,y,slope,primitive,n,true)-table_value(a,x,y,slope,primitive,n,true))/w;
    } else value=t>horizon ? 0. : table_value(t,x,y,slope,primitive,n,false);
    output[i]=scale*value;
}
extern "C" __global__ void table_near(const double* target, const double* times,
    const double* widths, const double* moments, const double* max_width,
    const double* h, const double* p, long long nh,
    const double* x, const double* y, const double* slope, const double* primitive,
    int knots, double horizon, double scale, double dt, double origin,
    long long sources, double* output) {
    // One warp owns one witness. No pair list, atomics or host pair-count
    // synchronization; broad overlapping bins also use bounded workspace.
    int lane=threadIdx.x;
    long long target_id=blockIdx.x, first=0, last=0;
    double t=target[target_id], radius=*max_width/2.+2.*dt;
    if(lane==0){
        long long lo=0,hi=sources;
        while(lo<hi){long long m=(lo+hi)/2;if(times[m]<t-radius)lo=m+1;else hi=m;}
        first=lo;lo=0;hi=sources;
        while(lo<hi){long long m=(lo+hi)/2;if(times[m]<=t+radius)lo=m+1;else hi=m;}
        last=lo;
    }
    first=__shfl_sync(0xffffffff,first,0);last=__shfl_sync(0xffffffff,last,0);
    double sum=0.;
    for(long long i=first+lane;i<last;i+=32){
        double tau=t-times[i],width=widths[i];
        if(fabs(tau)>width/2.+2.*dt)continue;
        double exact;
        if(width>0.){
            double a=fmin(tau-width/2.,horizon),b=fmin(tau+width/2.,horizon);
            exact=(table_value(b,x,y,slope,primitive,knots,true)-table_value(a,x,y,slope,primitive,knots,true))/width;
        }else exact=tau>horizon?0.:table_value(tau,x,y,slope,primitive,knots,false);
        double approximate=mesh_pair((t-origin)/dt,(times[i]-origin)/dt,width/dt,h,p,nh);
        sum+=(scale*exact-approximate)*moments[i];
    }
    for(int offset=16;offset>0;offset/=2)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if(lane==0)output[target_id]=sum;
}
'''


def deposit_gpu(position, widths, moments, density):
    import cupy as cp
    key = (cp.cuda.runtime.getDevice(), "deposit_time")
    if key not in _CACHE:
        _CACHE[key] = cp.RawKernel(_CODE, "deposit_time", options=("--std=c++17",))
    _CACHE[key](((len(position)+255)//256,), (256,),
        (position, widths, moments, density, cp.int64(len(position)), cp.int64(density.shape[1]), cp.int32(len(moments))))


def projected_pair_gpu(target, position, width, response, primitive):
    """Evaluate the mesh pair response in one launch, including tiny source bins."""
    import cupy as cp
    result = cp.empty_like(target)
    if not len(target):
        return result
    key = (cp.cuda.runtime.getDevice(), "projected_pair")
    if key not in _CACHE:
        _CACHE[key] = cp.RawKernel(_CODE, "projected_pair", options=("--std=c++17",))
    _CACHE[key](((len(target)+255)//256,), (256,),
        (target, position, width, response, primitive, result, cp.int64(len(target)), cp.int64(len(response))))
    return result


def exact_table_pairs_gpu(component, tau, width, horizon):
    """One-launch canonical causal table integration for the local correction."""
    import cupy as cp
    from .wake_models import device_arrays
    model = component.model
    arrays = device_arrays(model, "table", (model.times, model.values, model.slopes, model.integrals))
    out = cp.empty_like(tau)
    if not len(tau):
        return out
    key = (cp.cuda.runtime.getDevice(), "exact_table_pairs")
    if key not in _CACHE:
        _CACHE[key] = cp.RawKernel(_CODE, "exact_table_pairs", options=("--std=c++17",))
    _CACHE[key](((len(tau)+255)//256,), (256,),
        (tau, width, *arrays, out, cp.int64(len(tau)), cp.int32(len(model.times)),
         cp.float64(horizon), cp.float64(component.scale)))
    return out


def table_near_correction_gpu(plan, source, near, origin):
    """Fused exact-minus-mesh correction for arbitrary causal tabulated wakes."""
    import cupy as cp
    from .wake_models import device_arrays
    result = cp.empty((len(plan.components), len(source.times)), dtype=cp.float64)
    maximum = cp.max(near.widths)
    key = (cp.cuda.runtime.getDevice(), "table_near")
    if key not in _CACHE:
        _CACHE[key] = cp.RawKernel(_CODE, "table_near", options=("--std=c++17",))
    for ci, component in enumerate(plan.components):
        model = component.model
        arrays = device_arrays(model, "table", (model.times, model.values, model.slopes, model.integrals))
        moment = plan._coupled(component, near)
        _CACHE[key]((len(source.times),), (32,),
            (source.times, near.times, near.widths, moment, maximum, plan.response[ci], plan.primitive[ci],
             cp.int64(plan.response.shape[1]), *arrays, cp.int32(len(model.times)),
             cp.float64(plan.memory_time), cp.float64(component.scale), cp.float64(plan.grid.step),
             cp.float64(origin), cp.int64(len(near.times)), result[ci]))
    return result
