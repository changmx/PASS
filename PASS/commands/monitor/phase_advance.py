"""Turn-by-turn fractional-tune monitor for uncoupled transverse motion."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import math
from pathlib import Path
import re

import numpy as np

from PASS import __version__
from PASS.commands.command import Command
from PASS.core.beam import Beam
from PASS.core.config import Config
from PASS.core.simulation import Simulation
from PASS.utils.constants import const
from PASS.utils.helper import get_current_time
from PASS.utils.table_io import normalize_output_format, table_path, write_table
from PASS.utils.logger import set_normal_logging, set_simple_logging

logger = logging.getLogger(__name__)


@dataclass
class _WindowSpec:
    start: int
    end: int
    written: bool = False

    @property
    def expected_intervals(self) -> int:
        return self.end - self.start - 1


@dataclass
class _WindowRuntime:
    """Per-particle accumulators for one active turn range."""

    prev_ux: object = None
    prev_vx: object = None
    prev_uy: object = None
    prev_vy: object = None
    prev_valid_x: object = None
    prev_valid_y: object = None
    phase_x_sum: object = None
    phase_y_sum: object = None
    interval_x_count: object = None
    interval_y_count: object = None


def _as_host(array):
    """Return an ndarray without importing CuPy for CPU runs."""
    return array.get() if hasattr(array, "get") else np.asarray(array)


@Command.register("phaseadvancemonitor")
class PhaseAdvanceMonitor(Command):
    """Measure per-particle fractional tunes over contiguous turn windows.

    The monitor is for uncoupled transverse optics.  At one fixed lattice
    location it transforms coordinates using fixed design Twiss/dispersion and
    accumulates the directed phase difference between consecutive turns.
    State is local to this monitor and keyed by ``abs(tag) - 1`` so normal
    ``SortBunch`` reordering cannot change particle identity.
    """

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {str(k).lower(): v for k, v in command_kwargs.items()}
        self.beam_id = int(beam_id)
        self.s = float(kwargs["s (m)"])
        self.cmd_type = self.__class__.__name__
        self.cmd_name = str(kwargs["name"])
        self.output_format = normalize_output_format(kwargs.get("output format", "hdf5-gzip1"))
        self.enable = bool(kwargs.get("enable", True))

        self.beta_x = float(kwargs["beta x (m)"])
        self.beta_y = float(kwargs["beta y (m)"])
        self.alpha_x = float(kwargs["alpha x"])
        self.alpha_y = float(kwargs["alpha y"])
        self.dx = float(kwargs.get("dx (m)", 0.0))
        self.dpx = float(kwargs.get("dpx", 0.0))
        self.x_co = float(kwargs.get("x co (m)", 0.0))
        self.px_co = float(kwargs.get("px co", 0.0))
        self.y_co = float(kwargs.get("y co (m)", 0.0))
        self.py_co = float(kwargs.get("py co", 0.0))
        self._validate_optics()

        cfg: Config = sim.cfg
        self.num_turn = int(cfg.num_turn)
        beam: Beam = sim.beams[self.beam_id]
        self._xp = beam.particles.xp
        self._particle_dtype = np.dtype(beam.particles.dtype)
        self._state_dtype = np.dtype(np.float32 if self._particle_dtype == np.dtype(np.float32) else np.float64)
        self._kernel_real = np.float32 if self._particle_dtype == np.dtype(np.float32) else np.float64
        self._capacity = int(beam.Np_total)
        requested_action = kwargs.get("min action")
        if requested_action is None:
            self.min_action = 5e-9 if beam.particles.dtype == np.dtype(np.float32) else 5e-17
        else:
            self.min_action = float(requested_action)
        if not math.isfinite(self.min_action) or self.min_action < 0.0:
            raise ValueError(f"PhaseAdvanceMonitor '{self.cmd_name}': Min action must be finite and >= 0")

        # These values are fixed for the monitor lifetime. Keeping them out of
        # the per-particle path removes repeated sqrt/division work in both
        # backends without adding a large coefficient table to the kernel.
        self._sqrt_beta_x = math.sqrt(self.beta_x)
        self._inv_sqrt_beta_x = 1.0 / self._sqrt_beta_x
        self._sqrt_beta_y = math.sqrt(self.beta_y)
        self._inv_sqrt_beta_y = 1.0 / self._sqrt_beta_y
        self._action_threshold = 2.0 * self.min_action
        self._kernel_coefficients = tuple(
            self._kernel_real(value) for value in (
                self._sqrt_beta_x,
                self._inv_sqrt_beta_x,
                self._sqrt_beta_y,
                self._inv_sqrt_beta_y,
                self.alpha_x,
                self.alpha_y,
                self.dx,
                self.dpx,
                self.x_co,
                self.px_co,
                self.y_co,
                self.py_co,
                self._action_threshold,
            ))

        self.windows = self._compile_windows(kwargs.get("turn ranges", 0))
        self._active_windows = {}
        self._phase_kernel = None
        self.output_dir = Path(cfg.output_dir_tuneSpread)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_hms = str(cfg.output_hms)
        super().__init__()

    def _validate_optics(self) -> None:
        values = (
            self.beta_x,
            self.beta_y,
            self.alpha_x,
            self.alpha_y,
            self.dx,
            self.dpx,
            self.x_co,
            self.px_co,
            self.y_co,
            self.py_co,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"PhaseAdvanceMonitor '{self.cmd_name}': optics values must be finite")
        if self.beta_x <= 0.0 or self.beta_y <= 0.0:
            raise ValueError(f"PhaseAdvanceMonitor '{self.cmd_name}': Beta x/y must be > 0")

    def _compile_windows(self, raw_windows) -> list[_WindowSpec]:
        if raw_windows == 0 or raw_windows == [] or raw_windows is None:
            return []
        if not isinstance(raw_windows, (list, tuple)):
            raise ValueError(f"PhaseAdvanceMonitor '{self.cmd_name}': Turn ranges must be 0 or a list")
        windows: list[_WindowSpec] = []
        seen: set[tuple[int, int]] = set()
        for item in raw_windows:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"PhaseAdvanceMonitor '{self.cmd_name}': each Turn ranges item must be [start, end)")
            start, end = item
            if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (start, end)):
                raise ValueError(f"PhaseAdvanceMonitor '{self.cmd_name}': Turn ranges values must be integers")
            start, end = int(start), int(end)
            original_start, original_end = start, end
            if self.num_turn <= 0 or start >= self.num_turn or end <= 0:
                logger.warning("PhaseAdvanceMonitor '%s': ignoring range [%s, %s) outside [0, %s)", self.cmd_name, start, end, self.num_turn)
                continue
            start = max(0, start)
            end = min(self.num_turn, end)
            if (start, end) != (original_start, original_end):
                logger.warning("PhaseAdvanceMonitor '%s': clipping range [%s, %s) to [%s, %s)", self.cmd_name, original_start, original_end, start,
                               end)
            if end < start:
                raise ValueError(f"PhaseAdvanceMonitor '{self.cmd_name}': range [{original_start}, {original_end}) has end before start")
            if end - start < 2:
                logger.warning("PhaseAdvanceMonitor '%s': ignoring range [%s, %s) because phase advance needs two samples", self.cmd_name, start, end)
                continue
            key = (start, end)
            if key in seen:
                logger.warning("PhaseAdvanceMonitor '%s': ignoring duplicate range [%s, %s)", self.cmd_name, start, end)
                continue
            seen.add(key)
            windows.append(_WindowSpec(start, end))
        return windows

    def _allocate_window_runtime(self) -> _WindowRuntime:
        xp = self._xp
        size = self._capacity
        state_dtype = xp.float32 if self._state_dtype == np.dtype(np.float32) else xp.float64
        real_dtype = xp.float64
        return _WindowRuntime(
            prev_ux=xp.zeros(size, dtype=state_dtype),
            prev_vx=xp.zeros(size, dtype=state_dtype),
            prev_uy=xp.zeros(size, dtype=state_dtype),
            prev_vy=xp.zeros(size, dtype=state_dtype),
            prev_valid_x=xp.zeros(size, dtype=xp.uint8),
            prev_valid_y=xp.zeros(size, dtype=xp.uint8),
            phase_x_sum=xp.zeros(size, dtype=real_dtype),
            phase_y_sum=xp.zeros(size, dtype=real_dtype),
            interval_x_count=xp.zeros(size, dtype=xp.int32),
            interval_y_count=xp.zeros(size, dtype=xp.int32),
        )

    def print(self):
        set_simple_logging()
        window_text = ", ".join(f"[{item.start},{item.end})" for item in self.windows) or "disabled"
        logger.info("S=%.4f, Command=%s, Name=%s, Enable=%s, TurnRanges=%s, MinAction=%.3e", self.s, self.cmd_type, self.cmd_name, self.enable,
                    window_text, self.min_action)
        set_normal_logging()

    def execute_cpu(self, sim: Simulation):
        return self._execute(sim, backend="cpu")

    def execute_gpu(self, sim: Simulation):
        return self._execute(sim, backend="gpu")

    def _execute(self, sim: Simulation, backend: str) -> bool:
        if not self.enable:
            return False
        turn = int(sim.state.turn)
        active = []
        for index, spec in enumerate(self.windows):
            if spec.start <= turn < spec.end and not spec.written:
                window_state = self._active_windows.get(index)
                if window_state is None:
                    window_state = self._allocate_window_runtime()
                    self._active_windows[index] = window_state
                active.append((index, spec, window_state))
        if not active:
            return False
        beam: Beam = sim.beams[self.beam_id]
        for index, spec, window_state in active:
            for bunch in beam.bunches:
                self._sample_bunch(beam.particles, bunch, window_state, backend)
            if turn == spec.end - 1:
                self._write_window(beam, window_state, spec, backend)
                spec.written = True
                del self._active_windows[index]
        return True

    def _sample_bunch(self, p, bunch, window_state: _WindowRuntime, backend: str) -> None:
        start, end = int(bunch.start_idx), int(bunch.end_idx)
        if end <= start:
            return
        if backend == "gpu":
            self._sample_bunch_gpu(p, start, end, window_state)
            return
        xp = self._xp
        tags = p.tag[start:end]
        alive = tags > 0
        raw_indices = tags[alive] - 1
        in_range = (raw_indices >= 0) & (raw_indices < self._capacity)
        if not np.any(in_range):
            return
        indices = raw_indices[in_range]

        alive_indices = np.flatnonzero(alive)[in_range]
        dp = p.dp[start:end][alive_indices]
        x = p.x[start:end][alive_indices] - self.x_co - self.dx * dp
        px = p.px[start:end][alive_indices] - self.px_co - self.dpx * dp
        y = p.y[start:end][alive_indices] - self.y_co
        py = p.py[start:end][alive_indices] - self.py_co
        ux = x * self._inv_sqrt_beta_x
        vx = self.alpha_x * ux + self._sqrt_beta_x * px
        uy = y * self._inv_sqrt_beta_y
        vy = self.alpha_y * uy + self._sqrt_beta_y * py
        valid_x = xp.isfinite(ux) & xp.isfinite(vx) & ((ux * ux + vx * vx) >= self._action_threshold)
        valid_y = xp.isfinite(uy) & xp.isfinite(vy) & ((uy * uy + vy * vy) >= self._action_threshold)

        self._accumulate_plane(indices, ux, vx, valid_x, window_state.prev_ux, window_state.prev_vx, window_state.prev_valid_x,
                               window_state.phase_x_sum, window_state.interval_x_count, xp)
        self._accumulate_plane(indices, uy, vy, valid_y, window_state.prev_uy, window_state.prev_vy, window_state.prev_valid_y,
                               window_state.phase_y_sum, window_state.interval_y_count, xp)
        window_state.prev_ux[indices] = ux
        window_state.prev_vx[indices] = vx
        window_state.prev_uy[indices] = uy
        window_state.prev_vy[indices] = vy
        window_state.prev_valid_x[indices] = valid_x
        window_state.prev_valid_y[indices] = valid_y

    @staticmethod
    def _accumulate_plane(indices, u, v, valid, prev_u, prev_v, prev_valid, phase_sum, count, xp) -> None:
        accepted = valid & (prev_valid[indices] != 0)
        accepted_indices = indices[accepted]
        previous_u = prev_u[accepted_indices]
        previous_v = prev_v[accepted_indices]
        # theta=atan2(v,u) decreases under the PASS Courant-Snyder rotation.
        # atan2(sin(theta_prev-theta_now), cos(...)) therefore gives +mu.
        delta = xp.remainder(xp.arctan2(previous_v * u[accepted] - previous_u * v[accepted], previous_u * u[accepted] + previous_v * v[accepted]),
                             2.0 * const.pi)
        phase_sum[accepted_indices] += delta
        count[accepted_indices] += 1

    def _sample_bunch_gpu(self, p, start: int, end: int, window_state: _WindowRuntime) -> None:
        kernel = self._phase_kernel
        if kernel is None:
            kernel = _get_phase_kernel(p.dtype.str)
            self._phase_kernel = kernel

        threads = 256
        blocks = (end - start + threads - 1) // threads
        kernel(
            (blocks, ),
            (threads, ),
            (
                p.x,
                p.px,
                p.y,
                p.py,
                p.dp,
                p.tag,
                np.int32(start),
                np.int32(end),
                np.int32(self._capacity),
                *self._kernel_coefficients,
                const.pi,
                window_state.prev_ux,
                window_state.prev_vx,
                window_state.prev_uy,
                window_state.prev_vy,
                window_state.prev_valid_x,
                window_state.prev_valid_y,
                window_state.phase_x_sum,
                window_state.phase_y_sum,
                window_state.interval_x_count,
                window_state.interval_y_count,
            ),
        )

    def _write_window(self, beam: Beam, window_state: _WindowRuntime, spec: _WindowSpec, backend: str) -> None:
        p = beam.particles
        host_accumulators = (
            _as_host(window_state.phase_x_sum),
            _as_host(window_state.phase_y_sum),
            _as_host(window_state.interval_x_count),
            _as_host(window_state.interval_y_count),
        )
        for bunch in beam.bunches:
            start, end = int(bunch.start_idx), int(bunch.end_idx)
            if end <= start:
                self._write_bunch(
                    beam,
                    bunch,
                    spec,
                    backend,
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float32),
                    host_accumulators,
                )
                continue
            tags = _as_host(p.tag[start:end]).astype(np.int32, copy=False)
            tag_ids = np.abs(tags) - 1
            self._write_bunch(
                beam,
                bunch,
                spec,
                backend,
                tags,
                tag_ids,
                _as_host(p.lost_turn[start:end]),
                _as_host(p.lost_position[start:end]),
                host_accumulators,
            )

    def _write_bunch(self, beam, bunch, spec, backend, tags, tag_ids, lost_turn, lost_position, host_accumulators) -> None:
        phase_x_values, phase_y_values, count_x_values, count_y_values = host_accumulators
        count_x = np.zeros(len(tags), dtype=np.int32)
        count_y = np.zeros(len(tags), dtype=np.int32)
        phase_x = np.zeros(len(tags), dtype=np.float64)
        phase_y = np.zeros(len(tags), dtype=np.float64)
        valid_ids = (tag_ids >= 0) & (tag_ids < self._capacity)
        if np.any(valid_ids):
            phase_x[valid_ids] = phase_x_values[tag_ids[valid_ids]]
            phase_y[valid_ids] = phase_y_values[tag_ids[valid_ids]]
            count_x[valid_ids] = count_x_values[tag_ids[valid_ids]]
            count_y[valid_ids] = count_y_values[tag_ids[valid_ids]]
        alive = tags > 0
        tune_x = np.full(len(tags), np.nan)
        tune_y = np.full(len(tags), np.nan)
        has_x = count_x > 0
        has_y = count_y > 0
        # A particle that has been lost during the window has no reported
        # working point, even when it accumulated intervals before loss.
        # Its row remains useful for identification and loss diagnostics.
        report_x = alive & has_x
        report_y = alive & has_y
        tune_x[report_x] = phase_x[report_x] / (2.0 * const.pi * count_x[report_x])
        tune_y[report_y] = phase_y[report_y] / (2.0 * const.pi * count_y[report_y])
        expected = spec.expected_intervals
        valid_x = alive & has_x
        valid_y = alive & has_y
        complete_x = alive & (count_x == expected)
        complete_y = alive & (count_y == expected)
        columns = {
            "tag": tags,
            "tuneXFractional": tune_x,
            "tuneYFractional": tune_y,
            "intervalCountX": count_x,
            "intervalCountY": count_y,
            "validX": valid_x,
            "validY": valid_y,
            "completeX": complete_x,
            "completeY": complete_y,
            "lostTurn": lost_turn,
            "lostPosition": lost_position,
        }
        headers = {
            "Name": "PASS Phase Advance Tune Data",
            "Command": self.cmd_type,
            "Monitor": self.cmd_name,
            "S": self.s,
            "BeamId": self.beam_id,
            "BeamName": beam.beam_name,
            "BunchId": int(bunch.bunch_id),
            "HarmonicId": int(bunch.harmonic_id),
            "HarmonicNumber": int(bunch.harmonic_number),
            "StartTurn": spec.start,
            "EndTurn": spec.end,
            "IntervalCountExpected": expected,
            "BetaX": self.beta_x,
            "BetaY": self.beta_y,
            "AlphaX": self.alpha_x,
            "AlphaY": self.alpha_y,
            "Dx": self.dx,
            "Dpx": self.dpx,
            "XCO": self.x_co,
            "PXCO": self.px_co,
            "YCO": self.y_co,
            "PYCO": self.py_co,
            "MinAction": self.min_action,
            "Backend": backend,
            "ParticlePrecision": str(beam.particles.dtype),
            "PASSVersion": __version__,
            "Time": get_current_time(),
            "Model": "uncoupled transverse optics; fractional tune only",
        }
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.cmd_name).strip("_") or "phaseadvancemonitor"
        filename = (f"{self.output_hms}_tune_beam{self.beam_id}_bunch{int(bunch.bunch_id)}"
                    f"_Np_{int(bunch.Np)}_s_{self.s:.4f}_{safe_name}"
                    f"_turn_{spec.start}_{spec.end}.tfs")
        filepath = table_path(self.output_dir / filename, self.output_format)
        write_table(filepath, columns, headers, output_format=self.output_format)
        logger.info("PhaseAdvanceMonitor '%s': saved %s", self.cmd_name, filepath)


# GPU implementation is kept at the end of the module so the monitor logic
# above remains easy to scan independently of the CUDA source.
_PHASE_KERNEL_SOURCE = r'''
#ifndef PASS_USE_FLOAT
#define PASS_USE_FLOAT 0
#endif
#if PASS_USE_FLOAT
using pass_real_t = float;
#else
using pass_real_t = double;
#endif
#ifndef PASS_STATE_FLOAT
#define PASS_STATE_FLOAT 0
#endif
#if PASS_STATE_FLOAT
using pass_state_t = float;
#else
using pass_state_t = double;
#endif
extern "C" __global__ void phase_advance(
    const pass_real_t* __restrict__ x,
    const pass_real_t* __restrict__ px,
    const pass_real_t* __restrict__ y,
    const pass_real_t* __restrict__ py,
    const pass_real_t* __restrict__ dp,
    const int* __restrict__ tag,
    int start_index,
    int end_index,
    int capacity,
    pass_real_t sqrt_beta_x,
    pass_real_t inv_sqrt_beta_x,
    pass_real_t sqrt_beta_y,
    pass_real_t inv_sqrt_beta_y,
    pass_real_t alpha_x,
    pass_real_t alpha_y,
    pass_real_t dx,
    pass_real_t dpx,
    pass_real_t x_co,
    pass_real_t px_co,
    pass_real_t y_co,
    pass_real_t py_co,
    pass_real_t action_threshold,
    double pi_value,
    pass_state_t* __restrict__ prev_ux,
    pass_state_t* __restrict__ prev_vx,
    pass_state_t* __restrict__ prev_uy,
    pass_state_t* __restrict__ prev_vy,
    unsigned char* __restrict__ prev_valid_x,
    unsigned char* __restrict__ prev_valid_y,
    double* __restrict__ phase_x_sum,
    double* __restrict__ phase_y_sum,
    int* __restrict__ interval_x_count,
    int* __restrict__ interval_y_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index || tag[i] <= 0)
        return;
    int index = abs(tag[i]) - 1;
    if (index < 0 || index >= capacity)
        return;

    pass_real_t delta = dp[i];
    pass_real_t ux = (x[i] - x_co - dx * delta) * inv_sqrt_beta_x;
    pass_real_t vx = alpha_x * ux + sqrt_beta_x * (px[i] - px_co - dpx * delta);
    pass_real_t uy = (y[i] - y_co) * inv_sqrt_beta_y;
    pass_real_t vy = alpha_y * uy + sqrt_beta_y * (py[i] - py_co);
    bool valid_x = isfinite(ux) && isfinite(vx) && ux * ux + vx * vx >= action_threshold;
    bool valid_y = isfinite(uy) && isfinite(vy) && uy * uy + vy * vy >= action_threshold;

    if (valid_x && prev_valid_x[index]) {
        double cross = (double)prev_vx[index] * (double)ux - (double)prev_ux[index] * (double)vx;
        double dot = (double)prev_ux[index] * (double)ux + (double)prev_vx[index] * (double)vx;
        double angle = atan2(cross, dot);
        if (angle < 0.0)
            angle += 2.0 * pi_value;
        phase_x_sum[index] += angle;
        interval_x_count[index] += 1;
    }
    if (valid_y && prev_valid_y[index]) {
        double cross = (double)prev_vy[index] * (double)uy - (double)prev_uy[index] * (double)vy;
        double dot = (double)prev_uy[index] * (double)uy + (double)prev_vy[index] * (double)vy;
        double angle = atan2(cross, dot);
        if (angle < 0.0)
            angle += 2.0 * pi_value;
        phase_y_sum[index] += angle;
        interval_y_count[index] += 1;
    }
    prev_ux[index] = (pass_state_t)ux;
    prev_vx[index] = (pass_state_t)vx;
    prev_uy[index] = (pass_state_t)uy;
    prev_vy[index] = (pass_state_t)vy;
    prev_valid_x[index] = valid_x;
    prev_valid_y[index] = valid_y;
}
'''


@lru_cache(maxsize=None)
def _get_phase_kernel(dtype):
    import cupy as cp

    dtype = np.dtype(dtype)
    return cp.RawKernel(
        _PHASE_KERNEL_SOURCE,
        "phase_advance",
        options=(
            "--std=c++14",
            f"-DPASS_USE_FLOAT={int(dtype == np.dtype(np.float32))}",
            f"-DPASS_STATE_FLOAT={int(dtype == np.dtype(np.float32))}",
        ),
    )
