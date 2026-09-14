"""Stationary, gapped multibunch convolution and causal dyadic history.

The two transverse array indices here are *slot* and *time within a slot*,
not transverse particle coordinates. Their independently padded Toeplitz
embedding preserves the physical gap even when slot spacing / slice spacing
is not an integer. History partitions cover lags [L, 2L), L=1,2,4,... .
A completed L-turn input block contributes only to future turns, so no future
trajectory is required. All sample values are retained; there is no tail fit
or temporal decimation. CPU and CUDA execute the same schedule.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.fft import next_fast_len


@dataclass(frozen=True)
class ConvolutionGrid:
    period: float
    slots: int
    slices: int
    slot_spacing: float
    slice_spacing: float
    origin: float = 0.0
    width: float = 0.0
    projection: str = "exact"

    def __post_init__(self):
        for name in ("slots", "slices"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
                raise ValueError(f"Convolution grid {name} must be a positive integer")
        if not all(np.isfinite(v) and v > 0 for v in
                   (self.period, self.slot_spacing, self.slice_spacing)):
            raise ValueError("Convolution grid periods and spacings must be positive finite")
        if not np.isfinite(self.origin) or not np.isfinite(self.width) or self.width < 0:
            raise ValueError("Convolution grid origin/width must be finite; width >= 0")
        if self.width > self.slice_spacing*(1+1e-12):
            raise ValueError("Source width must not exceed slice spacing")
        span = (self.slices-1)*self.slice_spacing
        if self.slots > 1 and self.slot_spacing < span + max(self.width, self.slice_spacing)*.999999999999:
            raise ValueError("Convolution slot windows overlap")
        if (self.slots-1)*self.slot_spacing+span+self.width > self.period*(1+1e-12):
            raise ValueError("Convolution source windows exceed the turn period")
        if self.projection not in {"exact", "linear"}:
            raise ValueError("Convolution projection must be exact or linear")
        if self.projection == "linear" and self.width != 0:
            raise ValueError("Linear time projection currently requires point sources")

    def times(self, turn=0):
        return (self.origin+turn*self.period+np.arange(self.slots)[:, None]*self.slot_spacing
                +np.arange(self.slices)[None, :]*self.slice_spacing).ravel()


def source_channels(components):
    """Identical charge moments and source-speed laws share one forward FFT."""
    keys, representatives, indices = {}, [], []
    for c in components:
        law = c.velocity
        # Fixed laws and no law multiply the source by one. Witness factors
        # deliberately do not participate in this key.
        coupling = None if law is None or law.kind == "fixed" else (law.betas, law.source)
        key = (c.source_powers, coupling)
        if key not in keys:
            keys[key] = len(representatives)
            representatives.append(c)
        indices.append(keys[key])
    return tuple(representatives), np.asarray(indices, dtype=np.int32)


def _array_state(array):
    a = array.get() if hasattr(array, "get") else np.asarray(array)
    return {"shape": list(a.shape), "real": a.real.ravel().tolist(),
            "imag": a.imag.ravel().tolist()}


class ConvolutionState:
    """One location's persistent spectra; plans contain only reusable resources."""
    def __init__(self, plan, *, start_turn=0):
        self.plan_key = plan.key
        self.start_turn = int(start_turn)
        self.count = 0
        self.inputs = plan.xp.zeros((plan.input_capacity, len(plan.channels), *plan.frequency_shape), dtype=np.complex128)
        self.pending = plan.xp.zeros((plan.pending_capacity, len(plan.components), *plan.frequency_shape), dtype=np.complex128)

    @property
    def nbytes(self):
        return self.inputs.nbytes+self.pending.nbytes

    def state_dict(self):
        return {"plan_key": self.plan_key, "start_turn": self.start_turn, "count": self.count,
                "inputs": _array_state(self.inputs), "pending": _array_state(self.pending)}

    @classmethod
    def restore(cls, data, plan):
        if data["plan_key"] != plan.key:
            raise ValueError("Convolution checkpoint grid, kernel or history does not match")
        for key in ("start_turn", "count"):
            if isinstance(data[key], bool) or not isinstance(data[key], int) or data[key] < 0:
                raise ValueError("Convolution checkpoint has an invalid turn/count")
        out = cls(plan, start_turn=data["start_turn"])
        for name in ("inputs", "pending"):
            item = data[name]
            target = getattr(out, name)
            if tuple(item["shape"]) != target.shape:
                raise ValueError("Convolution checkpoint has an invalid array shape")
            value = np.asarray(item["real"])+1j*np.asarray(item["imag"])
            if value.size != target.size or not np.all(np.isfinite(value)):
                raise ValueError("Convolution checkpoint contains invalid spectra")
            target[...] = plan.xp.asarray(value.reshape(target.shape))
        out.count = data["count"]
        return out


class HistoryUpdate:
    """Staged update: no persistent ring is changed until the kick is accepted."""
    def __init__(self, state, spectrum, additions, plan=None):
        self.state, self.spectrum, self.additions = state, spectrum, additions
        self.count = state.count
        self.plan = plan

    def commit(self):
        s = self.state
        if s.count != self.count:
            raise RuntimeError("Convolution update is stale or already committed")
        n = s.count
        s.inputs[n % len(s.inputs)] = self.spectrum
        if self.plan is not None:
            schedule_uniform_gpu(self.plan, s, self.spectrum)
            s.count += 1
            self.spectrum = None
            return
        s.pending[n % len(s.pending)] = 0
        for value in self.additions:
            start = (n+1) % len(s.pending)
            first = min(len(value), len(s.pending)-start)
            s.pending[start:start+first] += value[:first]
            if first < len(value):
                s.pending[:len(value)-first] += value[first:]
        s.count += 1
        self.additions = ()
        self.spectrum = None


class BlockPreviewTransaction:
    """Sequential speculative block updates with exact, bounded rollback.

    Only overwritten ring rows are saved, once each. Use as a context manager;
    leaving it always restores the original state, including on exceptions.
    The caller retains the spectra/additions it wants to commit subsequently.
    A state cannot be used by overlapping transactions.
    """
    def __init__(self, state):
        self.state, self.count = state, state.count
        self.saved = []
        self.used = False
        self.seen = {name: np.zeros(len(getattr(state, name)), dtype=bool) for name in ("inputs", "pending")}

    def __enter__(self):
        if self.used or self.state.count != self.count:
            raise RuntimeError("Block preview transaction is stale or already used")
        if getattr(self.state, "_preview_active", False):
            raise RuntimeError("A block preview transaction is already active on this state")
        self.used = True
        self.state._preview_active = True
        return self

    def _save(self, name, start, count):
        array, seen = getattr(self.state, name), self.seen[name]
        for first, last in ((start, min(start+count, len(array))), (0, max(0, start+count-len(array)))):
            unseen = np.flatnonzero(~seen[first:last])+first
            if not len(unseen):
                continue
            boundaries = np.r_[0, np.flatnonzero(np.diff(unseen) != 1)+1, len(unseen)]
            for a, b in zip(boundaries[:-1], boundaries[1:]):
                sl = slice(int(unseen[a]), int(unseen[b-1])+1)
                self.saved.append((array, sl, array[sl].copy()))
                seen[sl] = True

    def apply(self, update):
        s = self.state
        if not getattr(s, "_preview_active", False) or update.state is not s:
            raise RuntimeError("Block update does not belong to this active transaction")
        self._save("inputs", s.count % len(s.inputs), 1)
        self._save("pending", s.count % len(s.pending), 1)
        if update.plan is not None:
            self._save("pending", 0, len(s.pending))
        for value in update.additions:
            self._save("pending", (s.count+1) % len(s.pending), len(value))
        update.commit()

    def __exit__(self, *exception):
        try:
            for array, sl, value in self.saved:
                array[sl] = value
            self.state.count = self.count
        finally:
            self.state._preview_active = False


class PartitionedConvolution:
    """Gapped spatial FFT and exact online convolution along the turn axis.

    ``method='uniform'`` uses one-turn frequency-domain history partitions;
    ``'dyadic'`` uses geometrically increasing partitions without discarding
    samples. ``memory_turns`` counts prior passages, as in the direct solver.
    Workspace limits are checked before allocating device buffers.
    """
    def __init__(self, components, grid, memory_turns, *, backend="cpu", method="dyadic",
                 memory_time=None, max_workspace_mb=1024):
        import hashlib
        from dataclasses import asdict
        import json

        if isinstance(memory_turns, bool) or not isinstance(memory_turns, int) or memory_turns < 1:
            raise ValueError("Partitioned convolution requires a finite positive Memory turns")
        if method not in {"uniform", "dyadic"}:
            raise ValueError("History partition must be uniform or dyadic")
        if backend not in {"cpu", "gpu"}:
            raise ValueError("Convolution backend must be cpu or gpu")
        if not components or any(not c.model.causal for c in components):
            raise ValueError("Online convolution requires causal response components")
        if memory_time is not None and (not np.isfinite(memory_time) or memory_time <= 0):
            raise ValueError("Memory time must be positive finite")
        self.components, self.grid = tuple(components), grid
        self.memory_turns, self.memory_time, self.method = memory_turns, memory_time, method
        self.backend = backend
        self.channels, self.channel_indices = source_channels(components)
        self.shape = (next_fast_len(2*grid.slots-1), next_fast_len(2*grid.slices-1))
        self.frequency_shape = (self.shape[0], self.shape[1]//2+1)
        self.longest = 1 << (memory_turns.bit_length()-1) if method == "dyadic" else memory_turns
        self.input_capacity = self.longest if method == "dyadic" else 1
        self.pending_capacity = 2*self.longest if method == "dyadic" else memory_turns+1
        self.levels = []
        if method == "dyadic":
            length = 1
            while length <= memory_turns:
                self.levels.append((length, min(length, memory_turns-length+1)))
                length *= 2
        else:
            self.levels = [(1, memory_turns)]
        f = math.prod(self.frequency_shape)
        nc, nu = len(components), len(self.channels)
        # Conservative peak: retained kernel spectra + rings + largest staged
        # update and FFT temporaries. Includes CPU/GPU double complex arithmetic.
        kernel_rows = sum(2*l for l, _ in self.levels) if method == "dyadic" else memory_turns
        self.estimated_bytes = int(16*f*(nc*(kernel_rows+2*self.longest+8*self.longest+1)
                                      +nu*(self.longest+4*self.longest)))
        if not np.isfinite(max_workspace_mb) or max_workspace_mb <= 0:
            raise ValueError("Max workspace must be positive finite")
        if self.estimated_bytes > max_workspace_mb*1024**2:
            raise MemoryError(f"Convolution estimated peak {self.estimated_bytes/1024**2:.1f} MiB "
                              f"exceeds {max_workspace_mb:g} MiB; reduce grid/history or raise the explicit limit")
        if backend == "gpu":
            import cupy as cp
            self.xp, self.fft = cp, cp.fft
            self.device = cp.cuda.runtime.getDevice()
            if self.estimated_bytes > cp.cuda.runtime.memGetInfo()[0]*.9:
                raise MemoryError("Convolution workspace would exhaust available GPU memory")
        else:
            import scipy.fft
            self.xp, self.fft = np, scipy.fft
            self.device = None
        self.indices = self.xp.asarray(self.channel_indices)
        self._time_plans = {}
        self._density = self.xp.zeros((nu, grid.slots, grid.slices), dtype=np.float64)
        d = np.arange(self.shape[0])
        q = np.arange(self.shape[1])
        d = np.where(d < grid.slots, d, d-self.shape[0])
        q = np.where(q < grid.slices, q, q-self.shape[1])
        self._offsets = d[:, None]*grid.slot_spacing+q[None, :]*grid.slice_spacing
        # Hash canonical sampled kernels too: model identity alone is not an
        # adequate checkpoint/cache key for user-supplied response objects.
        digest = hashlib.sha256(json.dumps({"grid": asdict(grid), "history": memory_turns,
            "memory_time": memory_time, "method": method,
            "components": [(c.plane, c.source_powers, c.test_powers, repr(c.velocity)) for c in components]},
            sort_keys=True).encode())
        self._digest = digest
        self.head = self._kernel_spectra(0, 1)[0]
        self.filters = []
        if method == "dyadic":
            for length, count in self.levels:
                kernel = self._kernel_spectra(length, count)
                # A single lag is a spectral delay line, requiring no temporal
                # FFT. This includes lag 1 and power-of-two cutoff endpoints.
                self.filters.append(kernel[0] if count == 1 else self._time_forward(kernel, 2*length))
        else:
            self.filters.append(self._kernel_spectra(1, memory_turns))
        self.key = digest.hexdigest()
        del self._digest

    def _time_plan(self, size, rows):
        """Own cuFFT handles: large levels must survive the global LRU cache.

        With two plans per level the default CuPy cache can evict long-period
        plans before their next use, causing repeated expensive CUDA planning.
        Time is contiguous for these batched transforms.
        """
        key = (size, rows)
        if key not in self._time_plans:
            from cupyx.scipy.fft import get_fft_plan
            sample = self.xp.empty((rows, *self.frequency_shape, size), dtype=np.complex128)
            self._time_plans[key] = get_fft_plan(sample, axes=(-1,), value_type="C2C")
        return self._time_plans[key]

    def _time_forward(self, data, size):
        if self.backend == "cpu":
            return self.fft.fft(data, n=size, axis=0)
        data = self.xp.ascontiguousarray(self.xp.moveaxis(data, 0, -1))
        with self._time_plan(size, data.shape[0]):
            return self.fft.fft(data, n=size, axis=-1)

    def _time_inverse(self, data, count):
        if self.backend == "cpu":
            return self.fft.ifft(data, axis=0)[:count].copy()
        with self._time_plan(data.shape[-1], data.shape[0]):
            values = self.fft.ifft(data, axis=-1)[..., :count]
        return self.xp.ascontiguousarray(self.xp.moveaxis(values, -1, 0))

    def _kernel_spectra(self, start, count):
        from .wake_solvers import _kernel
        out = self.xp.empty((count, len(self.components), *self.frequency_shape), dtype=np.complex128)
        # Initialization uses the canonical CPU evaluator, including arbitrary
        # tabulated/spectral models; no formula is reimplemented for this path.
        for j in range(count):
            tau = (start+j)*self.grid.period+self._offsets
            values = np.stack([_kernel(c, tau, self.grid.width, self.memory_time) for c in self.components])
            if not np.all(np.isfinite(values)):
                raise ValueError("Convolution kernel is not finite")
            self._digest.update(values.tobytes())
            out[j] = self.fft.rfft2(self.xp.asarray(values), axes=(-2, -1))
        return out

    def _mapping(self, source, turn):
        xp, g = self.xp, self.grid
        times = xp.asarray(source.times, dtype=np.float64)
        rel = times-(g.origin+turn*g.period)
        guard = (g.slot_spacing-(g.slices-1)*g.slice_spacing)/2
        slots = xp.floor((rel+guard)/g.slot_spacing).astype(np.int64)
        position = (rel-slots*g.slot_spacing)/g.slice_spacing
        # Absolute clocks eventually cannot resolve a fine mesh. Never silently
        # accept a tolerance comparable with an entire slice.
        resolution = abs(np.spacing(g.origin+turn*g.period))
        if resolution > g.slice_spacing*1e-3:
            raise ValueError("Absolute convolution clock cannot resolve the requested slice spacing")
        tolerance = max(64*resolution/g.slice_spacing, 2e-9)
        nearest = xp.rint(position)
        position = xp.where(xp.abs(position-nearest) <= tolerance, nearest, position)
        left = xp.floor(position).astype(np.int64)
        fraction = position-left
        valid = ((slots >= 0) & (slots < g.slots) & (left >= 0) & (left < g.slices)
                 & ((left < g.slices-1) | (fraction == 0)) & xp.isfinite(times))
        if g.projection == "exact":
            valid &= fraction == 0
        widths = xp.asarray(source.widths)
        valid &= xp.abs(widths-g.width) <= max(g.width*1e-10, g.slice_spacing*1e-12)
        if not bool(xp.all(valid)):
            raise ValueError("Source times/widths do not match the declared convolution grid; "
                             "use point-source linear projection or a compatible exact grid")
        index = slots*g.slices+left
        return index, fraction

    def _validated_state(self, state, turn):
        xp = self.xp
        if isinstance(turn, bool) or not isinstance(turn, (int, np.integer)) or turn < 0:
            raise ValueError("Convolution turn must be a nonnegative integer")
        if self.device is not None and xp.cuda.runtime.getDevice() != self.device:
            raise RuntimeError("Convolution plan belongs to a different CUDA device")
        if state is None:
            state = ConvolutionState(self, start_turn=turn)
        if state.plan_key != self.key:
            raise ValueError("Convolution state belongs to a different plan")
        if not isinstance(state.inputs, xp.ndarray) or (self.device is not None and state.inputs.device.id != self.device):
            raise ValueError("Convolution state belongs to another backend/device; restore its checkpoint on this plan")
        if turn != state.start_turn+state.count:
            raise ValueError("Convolution turns must be consecutive; submit empty sources for empty turns")
        return state

    def preview(self, source, state, *, turn):
        """Project physical sources, return witness coefficients and a staged update."""
        xp = self.xp
        state = self._validated_state(state, turn)
        if self.backend == "gpu":
            from .wake_state import DeviceSources
            source = DeviceSources.upload(source)
            index, fraction = deposit_gpu(self, source, turn)
        else:
            index, fraction = self._mapping(source, turn)
            self._density.fill(0)
            if len(source.times):
                for component in self.components:
                    component.model.validate_beta(float(np.min(source.betas)))
                    component.model.validate_beta(float(np.max(source.betas)))
            for ci, component in enumerate(self.channels):
                values = component.source_moment(source)
                if not np.all(np.isfinite(values)):
                    raise ValueError("Convolution source moments must be finite")
                flat = self._density[ci].ravel()
                if values.size == 0:
                    continue
                flat += np.bincount(index, weights=values*(1-fraction), minlength=flat.size)
                if self.grid.projection == "linear":
                    right = np.minimum(index+1, flat.size-1)
                    flat += np.bincount(right, weights=values*fraction, minlength=flat.size)
        values, update = self._convolve_block(self._density, state)
        if self.backend == "gpu":
            result = gather_gpu(self, source, values, index, fraction)
        else:
            values = values[:, :self.grid.slots, :self.grid.slices].reshape(len(self.components), -1)
            result = values[:, index]*(1-fraction)
            if self.grid.projection == "linear":
                result += values[:, xp.minimum(index+1, values.shape[1]-1)]*fraction
            for ci, component in enumerate(self.components):
                result[ci] *= component.witness_factor(source.betas)
        return result, update

    def preview_block(self, density, state, *, turn):
        """Convolve one preprojected block without changing persistent history.

        ``density`` is a finite float64 backend array with shape
        (source channels, slots, slices). Values include source velocity
        factors exactly once. The returned grid has one row per component;
        the caller applies witness factors after gathering physical times.
        The input array is not retained by the staged update.
        """
        state = self._validated_state(state, turn)
        if (not isinstance(density, self.xp.ndarray) or density.shape != self._density.shape
                or density.dtype != np.float64):
            raise ValueError("Convolution block must be a float64 backend array with the declared channel/grid shape")
        if self.device is not None and density.device.id != self.device:
            raise ValueError("Convolution block belongs to another CUDA device")
        if not bool(self.xp.all(self.xp.isfinite(density))):
            raise ValueError("Convolution block must be finite")
        values, update = self._convolve_block(density, state)
        return values[:, :self.grid.slots, :self.grid.slices].copy(), update

    def _convolve_block(self, density, state):
        """Shared spectral core; validation and coordinate mapping live at the boundary."""
        xp = self.xp
        spectrum = self.fft.rfft2(density, s=self.shape, axes=(-2, -1))
        n = state.count
        if self.backend == "gpu":
            accumulated = accumulate_gpu(self, state, spectrum)
        else:
            accumulated = self.head*spectrum[self.indices]+state.pending[n % len(state.pending)]
            if self.method == "dyadic":
                for (length, count), kernel in zip(self.levels, self.filters):
                    if count == 1 and n >= length:
                        accumulated += kernel*state.inputs[(n-length) % len(state.inputs)][self.indices]
        values = self.fft.irfft2(accumulated, s=self.shape, axes=(-2, -1))
        additions = []
        if self.method == "uniform" and self.backend == "cpu":
            # All prior lag contributions of this new source are scheduled once.
            additions.append(self.filters[0]*spectrum[self.indices][None, ...])
        elif self.method == "dyadic":
            for (length, count), kernel in zip(self.levels, self.filters):
                if count == 1 or (n+1) % length:
                    continue
                block = xp.empty((length, len(self.channels), *self.frequency_shape), dtype=np.complex128)
                if length > 1:
                    start = (n-length+1) % len(state.inputs)
                    first = min(length-1, len(state.inputs)-start)
                    block[:first] = state.inputs[start:start+first]
                    if first < length-1:
                        block[first:-1] = state.inputs[:length-1-first]
                block[-1] = spectrum
                transformed = self._time_forward(block, 2*length)
                product = (transformed[:, self.indices] if self.backend == "cpu" else transformed[self.indices])*kernel
                contribution = self._time_inverse(product, length+count-1)
                additions.append(contribution)
        return values, HistoryUpdate(state, spectrum, additions,
                                     self if self.method == "uniform" and self.backend == "gpu" else None)

    def step(self, source, state=None, *, turn):
        values, update = self.preview(source, state, turn=turn)
        if not bool(self.xp.all(self.xp.isfinite(values))):
            raise FloatingPointError("Convolution produced nonfinite coefficients")
        update.commit()
        return values, update.state

    @property
    def diagnostics(self):
        return {"partition": self.method, "source_transforms": len(self.channels),
                "components": len(self.components), "spatial_fft_shape": self.shape,
                "history_levels": len(self.levels), "estimated_peak_bytes": self.estimated_bytes}


# ----------------------------------------------------------------------------
# GPU: fused CUDA convolution kernels
# ----------------------------------------------------------------------------


_SCHEDULE = None
_ACCUMULATE = None
_GATHER = None


def accumulate_gpu(plan, state, spectrum):
    """Fuse channel selection, current response, pending field and delay lines."""
    import cupy as cp
    global _ACCUMULATE
    if _ACCUMULATE is None:
        _ACCUMULATE = cp.ElementwiseKernel(
            "raw complex128 head, raw complex128 source, raw complex128 pending, raw complex128 inputs, "
            "raw complex128 singles, raw int64 lags, raw int32 channels, int64 nf, int64 span, int64 nu, "
            "int64 count, int64 input_capacity, int64 pending_capacity, int32 nlags",
            "complex128 value", r'''
                long long src=(long long)channels[i/nf]*nf+i%nf;
                value=head[i]*source[src]+pending[(count%pending_capacity)*span+i];
                for(int j=0;j<nlags;j++)if(count>=lags[j])
                    value+=singles[(long long)j*span+i]*inputs[((count-lags[j])%input_capacity)*nu*nf+src];
            ''', "pass_wake_accumulate")
    if not hasattr(plan, "_single_filters"):
        singles = [(l, k) for (l, c), k in zip(plan.levels, plan.filters) if c == 1] if plan.method == "dyadic" else []
        plan._single_lags = cp.asarray([l for l, _ in singles], dtype=cp.int64)
        plan._single_filters = cp.stack([k for _, k in singles]) if singles else cp.empty(0, dtype=cp.complex128)
    nf = int(np.prod(plan.frequency_shape))
    result = _ACCUMULATE(plan.head, spectrum, state.pending, state.inputs, plan._single_filters,
        plan._single_lags, plan.indices, np.int64(nf), np.int64(nf*len(plan.components)),
        np.int64(len(plan.channels)), np.int64(state.count), np.int64(len(state.inputs)),
        np.int64(len(state.pending)), np.int32(len(plan._single_lags)), size=nf*len(plan.components))
    return result.reshape(len(plan.components), *plan.frequency_shape)


def gather_gpu(plan, source, values, index, fraction):
    """Gather directly from padded FFT output, fusing constant witness factors."""
    import cupy as cp
    global _GATHER
    if _GATHER is None:
        _GATHER = cp.ElementwiseKernel(
            "raw float64 field, raw int64 index, raw float64 fraction, raw float64 factors, "
            "int64 n, int64 slices, int64 padded_slices, int64 padded_span",
            "float64 value", r'''
                long long c=i/n, j=i%n, q=index[j];
                long long offset=c*padded_span+(q/slices)*padded_slices+q%slices;
                double part=fraction[j];
                value=field[offset]*(1.-part);
                if(part!=0.)value+=field[offset+1]*part;
                value*=factors[c];
            ''', "pass_wake_gather")
    if not hasattr(plan, "_witness_constants"):
        constants, varying = [], []
        for ci, c in enumerate(plan.components):
            law = c.velocity
            if law is None or law.kind == "fixed": constants.append(1.)
            elif all(v == law.witness[0] for v in law.witness): constants.append(law.witness[0])
            else:
                constants.append(1.)
                varying.append(ci)
        plan._witness_constants = cp.asarray(constants)
        plan._varying_witnesses = varying
    n = len(source.times)
    result = _GATHER(values, index, fraction, plan._witness_constants, np.int64(n),
        np.int64(plan.grid.slices), np.int64(plan.shape[1]), np.int64(np.prod(plan.shape)),
        size=n*len(plan.components)).reshape(len(plan.components), n)
    if plan._varying_witnesses:
        from .wake_velocity import factor_gpu as factor
        for ci in plan._varying_witnesses:
            result[ci] *= factor(plan.components[ci], source.betas, True)
    return result


def schedule_uniform_gpu(plan, state, spectrum):
    """Fused delayed multiply/add and consumed-slot clearing; no H-sized temporary."""
    import cupy as cp
    global _SCHEDULE
    if _SCHEDULE is None:
        _SCHEDULE = cp.ElementwiseKernel(
            "raw complex128 kernel, raw complex128 source, raw int32 channels, int64 nf, int64 span, int64 count, int64 history",
            "raw complex128 pending", r'''
                long long lag=i/span, cell=i%span;
                long long out=((count+1+lag)%(history+1))*span+cell;
                if(lag==history)pending[out]=complex<double>(0.,0.);
                else pending[out]+=kernel[i]*source[(long long)channels[cell/nf]*nf+cell%nf];
            ''', "pass_wake_schedule_uniform")
    nf = int(np.prod(plan.frequency_shape))
    span = len(plan.components)*nf
    _SCHEDULE(plan.filters[0], spectrum, plan.indices, np.int64(nf), np.int64(span), np.int64(state.count),
              np.int64(plan.memory_turns), state.pending, size=(plan.memory_turns+1)*span)

_CODE = r'''
extern "C" __global__ void scatter(const double* times,const double* widths,
    const double* betas,const double* values,int n,int channels,int slots,int slices,
    double origin,double gap,double step,double width,double tolerance,int linear,
    double beta_min,double beta_max,long long* index,double* fraction,double* density,int* invalid){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    double rel=times[i]-origin, guard=(gap-(slices-1)*step)/2.;
    double slot=floor((rel+guard)/gap),pos=(rel-slot*gap)/step;
    double near=nearbyint(pos);
    if(fabs(pos-near)<=tolerance)pos=near;
    double left=floor(pos),part=pos-left;
    if(!isfinite(times[i])||!isfinite(widths[i])||!isfinite(betas[i])||betas[i]<=0.
       ||betas[i]<beta_min||betas[i]>beta_max||slot<0||slot>=slots||left<0||left>=slices
       ||(left==slices-1&&part!=0.)||(!linear&&part!=0.)
       ||fabs(widths[i]-width)>fmax(width*1e-10,step*1e-12)){
        atomicExch(invalid,1);index[i]=0;fraction[i]=0.;return;
    }
    long long id=(long long)slot*slices+(long long)left;
    index[i]=id;fraction[i]=part;
    for(int k=0;k<channels;k++){
        double value=values[(long long)k*n+i];
        if(!isfinite(value)){atomicExch(invalid,2);continue;}
        double* out=density+(long long)k*slots*slices+id;
        atomicAdd(out,value*(1.-part));
        if(linear&&part!=0.)atomicAdd(out+1,value*part);
    }
}
'''


def deposit_gpu(plan, source, turn):
    """One scatter launch and one scalar validation transfer per passage."""
    import cupy as cp
    from .wake_components import moment_gpu as moment
    g = plan.grid
    n = len(source.times)
    arrays = [source.times, source.widths, source.betas]
    if any(a.shape != (n,) for a in arrays) or any(source.moments[c.source_powers].shape != (n,) for c in plan.channels):
        raise ValueError("Convolution source arrays must have matching one-dimensional shapes")
    times, widths, betas = [cp.ascontiguousarray(a, dtype=cp.float64) for a in arrays]
    if not hasattr(plan, "_scatter"):
        plan._scatter = cp.RawKernel(_CODE, "scatter", options=("--std=c++17",))
        plan._invalid = cp.zeros(1, dtype=cp.int32)
    index, fraction = cp.empty(n, dtype=cp.int64), cp.empty(n, dtype=cp.float64)
    plan._density.fill(0)
    if not n:
        return index, fraction
    resolution = abs(np.spacing(g.origin+turn*g.period))
    if resolution > g.slice_spacing*1e-3:
        raise ValueError("Absolute convolution clock cannot resolve the requested slice spacing")
    beta_min, beta_max = 0., 1.
    for c in plan.components:
        if getattr(c.model, "round_pipe", False):
            beta_min = max(beta_min, .99)
        law = c.velocity
        if law is not None:
            lo, hi = ((law.beta*(1-1e-12), law.beta*(1+1e-12)) if law.kind == "fixed"
                      else (law.betas[0], law.betas[-1]))
            beta_min, beta_max = max(beta_min, lo), min(beta_max, hi)
    values = cp.ascontiguousarray(cp.stack([moment(c, source) for c in plan.channels]), dtype=cp.float64)
    plan._invalid.fill(0)
    plan._scatter(((n+255)//256,), (256,), (times, widths, betas, values,
        np.int32(n), np.int32(len(plan.channels)), np.int32(g.slots), np.int32(g.slices),
        np.float64(g.origin+turn*g.period), np.float64(g.slot_spacing), np.float64(g.slice_spacing),
        np.float64(g.width), np.float64(max(64*resolution/g.slice_spacing, 2e-9)), np.int32(g.projection == "linear"),
        np.float64(beta_min), np.float64(beta_max), index, fraction, plan._density, plan._invalid))
    if int(plan._invalid[0]):
        raise ValueError("Source times/widths do not match convolution grid, or source moments/beta are invalid")
    return index, fraction
