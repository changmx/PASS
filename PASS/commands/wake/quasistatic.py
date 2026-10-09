"""Full-ring FFT history with the current revolution period frozen for all ages.

Raw emitted snapshots remain unchanged. This deliberately replaces their
physical delays and widths during evaluation; it is not a retarded-time solver.
"""
from collections import deque

import numpy as np
from scipy.fft import next_fast_len, rfft, irfft

from .convolution import source_channels
from .wake_solvers import _kernel
from .wake_state import DeviceSources, WakeSources, source_metadata_gpu


class QuasistaticFFT:
    """Current plus H previous complete passages, with one linear convolution."""

    def __init__(self, components, memory_turns, *, backend="cpu"):
        if isinstance(memory_turns, bool) or not isinstance(memory_turns, int) or memory_turns < 1:
            raise ValueError("quasistatic_fft requires positive finite Memory turns")
        if backend not in {"cpu", "gpu"}:
            raise ValueError("quasistatic_fft backend must be cpu or gpu")
        if not components or any(not component.model.causal for component in components):
            raise ValueError("quasistatic_fft requires causal wake components")
        self.components, self.memory_turns, self.backend = tuple(components), memory_turns, backend
        self.channels, self.channel_indices = source_channels(self.components)
        if backend == "gpu":
            import cupy as cp
            self.xp, self.forward, self.inverse = cp, cp.fft.rfft, cp.fft.irfft
            self.device = cp.cuda.runtime.getDevice()
        else:
            self.xp, self.forward, self.inverse, self.device = np, rfft, irfft, None
        self._geometry_cache = {}
        self._kernel_key, self._spectrum = None, None
        self._diagnostics = {}

    @property
    def diagnostics(self):
        return dict(self._diagnostics)

    def _geometry(self, source, observation_period=None):
        """Validate monotone uniform saved bins, without changing their order."""
        cached = self._geometry_cache.get(id(source))
        if cached is not None and cached[0] is source and observation_period is None:
            return cached[1]
        xp = self.xp
        n_slices = len(source.times)
        if n_slices < 2:
            raise ValueError("quasistatic_fft requires at least two full-ring slices")
        arrays = (source.times, source.widths, source.betas, *source.moments.values())
        if any(value.shape != (n_slices, ) for value in arrays):
            raise ValueError("quasistatic_fft source arrays must match the slice count")
        if not {component.source_powers for component in self.components}.issubset(source.moments):
            raise ValueError("quasistatic_fft source moments are incomplete")
        if self.backend == "gpu":
            metadata = source_metadata_gpu(self, source)
            if not metadata[0]:
                raise ValueError("quasistatic_fft requires finite times, widths, betas and moments")
            first, last, width, beta = xp.stack((source.times[0], source.times[-1], source.widths[0], source.betas[0])).get()
        else:
            if any(not np.all(np.isfinite(value)) for value in arrays):
                raise ValueError("quasistatic_fft requires finite times, widths, betas and moments")
            first, last, width, beta = source.times[0], source.times[-1], source.widths[0], source.betas[0]
        first, last, width, beta = map(float, (first, last, width, beta))
        reverse = first > last
        spacing = abs(last - first) / (n_slices - 1)
        step = spacing if observation_period is None else observation_period / n_slices
        resolution = abs(np.spacing(max(abs(first), abs(last))))
        if step <= 0 or 8 * resolution > step * 1e-3:
            raise ValueError("quasistatic_fft source clock cannot resolve its uniform grid")
        tolerance = max(8 * resolution, step * 1e-9)
        ordered_times = source.times[::-1] if reverse else source.times
        expected = min(first, last) + xp.arange(n_slices, dtype=xp.float64) * step
        point = width == 0.
        valid = xp.all(xp.abs(ordered_times - expected) <= tolerance)
        valid &= xp.all(xp.abs(source.widths - (0. if point else step)) <= step * 1e-9)
        valid &= xp.all(xp.abs(source.betas - beta) <= max(abs(beta) * 1e-12, 1e-15))
        if not 0 < beta <= 1 or not bool(valid):
            raise ValueError("quasistatic_fft requires a uniform full-ring time grid, common bin widths and one reference beta")
        geometry = (n_slices, reverse, point, beta)
        self._geometry_cache[id(source)] = (source, geometry)
        return geometry

    def _response(self, n_slices, step, width, age):
        size = (age + 1) * n_slices
        # Only the last N outputs are retained. At this length their first
        # wrapped contributor would be index 2*size-1, beyond the convolution.
        fft_size = next_fast_len(size + n_slices - 1)
        key = (n_slices, step, width, age)
        if key != self._kernel_key:
            if self.backend == "gpu":
                from .wake_models import response_gpu
                values = self.xp.empty((len(self.components), size), dtype=self.xp.float64)
                for row, component in enumerate(self.components):
                    response = response_gpu(component.model, component.longitudinal)
                    response.kernel('response_grid')(((size + 255) // 256, ), (256, ),
                                                     (response.data, np.float64(step), np.float64(width), np.int32(0), np.int32(size),
                                                      np.float64(np.inf), np.float64(component.scale), np.int32(width == 0.), values[row]))
            else:
                delays = np.arange(size) * step
                values = np.stack([_kernel(component, delays, width, None) for component in self.components])
            spectrum = self.forward(values, fft_size, axis=1)
            self._kernel_key, self._spectrum = key, spectrum
        return size, fft_size, self._spectrum

    def preview(self, source, state, *, turn, period=None, observation_period=None):
        """Return coefficients and a staged WakeState; never mutate old history.

        ``period`` is C/(beta_now*c). ``observation_period`` certifies the saved
        Slicer full-ring window, which can differ during acceleration.
        """
        if isinstance(turn, bool) or not isinstance(turn, (int, np.integer)) or turn < 0:
            raise ValueError("quasistatic_fft turn must be a nonnegative integer")
        if state.last_turn is not None and turn <= state.last_turn:
            raise ValueError("quasistatic_fft turn must advance; reset state before a new run")
        if state.mode_amplitudes or state.convolution is not None:
            raise ValueError("quasistatic_fft requires raw passage history")
        previous = -1
        for history_turn, _ in state.history:
            if (isinstance(history_turn, bool) or not isinstance(history_turn, (int, np.integer)) or history_turn <= previous
                    or state.last_turn is None or history_turn > state.last_turn):
                raise ValueError("quasistatic_fft history turns must be ordered, unique and already committed")
            previous = history_turn
        candidate = state.fork()
        candidate.history = deque((history_turn, item) for history_turn, item in state.history if history_turn >= turn - self.memory_turns)
        candidate.last_turn = int(turn)
        if self.backend == "gpu":
            if self.xp.cuda.runtime.getDevice() != self.device:
                raise RuntimeError("quasistatic_fft plan belongs to another CUDA device")
            source = DeviceSources.upload(source)
            candidate.history = deque((history_turn, DeviceSources.upload(item)) for history_turn, item in candidate.history)
        elif not isinstance(source, WakeSources):
            raise TypeError("CPU quasistatic_fft requires host WakeSources")
        if not len(source.times):
            self._geometry_cache.clear()
            self._diagnostics = {"history_horizon_turns": self.memory_turns, "quasistatic": True, "fft_size": 0}
            return self.xp.zeros((len(self.components), 0), dtype=self.xp.float64), candidate
        for name, value in (("period", period), ("observation_period", observation_period)):
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"quasistatic_fft requires an explicit positive full-ring {name}")
        n_slices, reverse, point, beta = self._geometry(source, observation_period)
        snapshots = [*candidate.history, (int(turn), source)]
        geometry = []
        for _, item in snapshots:
            entry = self._geometry(item)
            if entry[0] != n_slices or entry[2] != point:
                raise ValueError("quasistatic_fft history must retain the same bin count and source shape")
            geometry.append(entry)
        # Raw snapshot identities delimit validation-cache lifetime. Only
        # reusable resources are changed by preview, never committed state.
        self._geometry_cache = {id(item): self._geometry_cache[id(item)] for _, item in snapshots}
        step = float(period) / n_slices
        width = 0. if point else step
        for component in self.components:
            component.model.validate_beta(beta)
        source_factors = [1. if component.velocity is None else float(component.velocity.source_factor(beta)) for component in self.channels]
        witness_factors = [float(component.witness_factor(beta)) for component in self.components]
        age = int(turn - snapshots[0][0])
        size, fft_size, response = self._response(n_slices, step, width, age)
        density = self.xp.zeros((len(self.channels), size), dtype=self.xp.float64)
        for (history_turn, item), entry in zip(snapshots, geometry):
            start = (age - (turn - history_turn)) * n_slices
            for row, component in enumerate(self.channels):
                moment = item.moments[component.source_powers]
                density[row, start:start + n_slices] = moment[::-1] if entry[1] else moment
        for row, factor in enumerate(source_factors):
            density[row] *= factor
        transformed = self.forward(density, fft_size, axis=1)
        if self.backend == "gpu":
            from .wake_models import response_gpu
            from .wake_solvers import _FFT_CODE
            product = self.xp.empty_like(response)
            indices = self.xp.asarray(self.channel_indices, dtype=self.xp.int32)
            kernel = response_gpu(self.components[0].model, self.components[0].longitudinal)
            kernel.kernel('fft_product', _FFT_CODE)(((product.size + 255) // 256, ), (256, ),
                                                    (transformed, response, indices, np.int32(product.shape[1]), np.int64(product.size), product))
        else:
            product = transformed[self.channel_indices] * response
        # rfft/irfft already provide exactly one inverse normalization.
        values = self.inverse(product, fft_size, axis=1)[:, age * n_slices:age * n_slices + n_slices].copy()
        for row, factor in enumerate(witness_factors):
            values[row] *= factor
        if reverse:
            values = self.xp.ascontiguousarray(values[:, ::-1])
        candidate.history.append((int(turn), source))
        candidate.last_time = float(max(float(source.times[0]), float(source.times[-1])))
        candidate.last_source_end = candidate.last_time + (0. if point else observation_period / n_slices / 2)
        self._diagnostics = {
            "quasistatic": True,
            "history_horizon_turns": self.memory_turns,
            "current_period_s": float(period),
            "observation_period_s": float(observation_period),
            "current_beta": beta,
            "current_slice_spacing_s": step,
            "source_transforms": len(self.channels),
            "fft_size": fft_size,
            "normalization": "single_inverse"
        }
        return values, candidate
