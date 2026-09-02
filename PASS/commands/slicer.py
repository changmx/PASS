"""Longitudinal slicing command and the per-bunch :class:`SliceSet`.

``Slicer`` is a local operation: it computes IDs and statistics for each
already-established bunch range without reordering particles or changing
bunch membership.  Global regrouping belongs to ``SortBunch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import logging
import os
import numpy as np
import pandas as pd
import re
import tfs

# Compile CUDA kernels on first use in each process and retain them only in
# memory.  No Slicer-specific disk cache directory is read or written.
os.environ.setdefault("CUPY_CACHE_IN_MEMORY", "1")

from PASS import __version__
from PASS.commands.command import Command
from PASS.utils.helper import get_current_time
from PASS.utils.logger import set_simple_logging, set_normal_logging

logger = logging.getLogger(__name__)


def _normalize_model(value: str) -> str:
    """Normalize human-readable model names to stable configuration keys."""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class ExplicitRange:
    """User-defined fixed longitudinal interval in local ``z_rel``."""

    z_min: float
    z_max: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "z_min", float(self.z_min))
        object.__setattr__(self, "z_max", float(self.z_max))
        if not self.z_min < self.z_max:
            raise ValueError("Explicit slice range requires z_min < z_max")


@dataclass
class SliceSet:
    """Configuration and latest result for one named bunch-local slicer.

    ``name`` is the user-defined lookup key used by collective effects.
    ``model`` is ``equal_length`` or ``equal_particle`` (``equal_charge`` is
    accepted as a future alias). ``num_slices`` is the number of longitudinal
    bins; empty bins are allowed. ``z_range_mode`` chooses the interval:
    ``auto`` uses the actual minimum and maximum of the current live
    distribution and never excludes an observed outlier. ``explicit`` uses
    the nested :class:`ExplicitRange` object.
    ``source_command`` records the sequence entry that
    first created this set, solely for duplicate-configuration diagnostics.

    If ``number of alive particles < num_slices``, only that many slices can
    contain particles.  The remaining bins are retained with zero count and
    a warning is emitted; this is valid and does not require changing the
    configured mesh size.

    ``delta_z`` is the geometric bin width. ``real_charge`` is the number of
    real particles represented in a bin, not Coulombs. ``lind_density`` is
    the linear real-particle density ``real_charge / delta_z``. Runtime
    results are local to the owning bunch. ``slice_id`` is an integer
    array in current particle order (lost particles are -1). ``slice_table``
    contains per-slice ``z_min``, ``z_max``, ``z_center``, ``delta_z``,
    ``macro_count``, ``real_charge`` and ``lind_density``. ``real_charge``
    means equivalent real-particle count (macro count times ``bunch.ratio``),
    while physical charge requires multiplying by particle charge and e.
    ``valid_turn`` and ``valid_s`` record where the result was made;
    ``particle_order_version`` is reserved for a future order counter.
    """

    name: str
    model: str = "equal_length"
    num_slices: int = 10
    z_range_mode: str = "auto"
    explicit: ExplicitRange | None = None
    source_command: str | None = None

    slice_id: Any = field(default=None, repr=False)
    slice_table: Any = field(default=None, repr=False)
    valid_turn: int | None = field(default=None, repr=False)
    valid_s: float | None = field(default=None, repr=False)
    particle_order_version: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.model = _normalize_model(self.model)
        if self.model not in {"equal_length", "equal_particle", "equal_charge"}:
            raise ValueError(f"Unsupported slice model {self.model!r}; expected "
                             "equal_length, equal_particle, or equal_charge")
        if int(self.num_slices) < 1:
            raise ValueError("Number of slices must be >= 1")
        self.num_slices = int(self.num_slices)
        self.z_range_mode = _normalize_model(self.z_range_mode)
        if self.z_range_mode not in {"auto", "explicit"}:
            raise ValueError(f"Unsupported slice z range mode {self.z_range_mode!r}; "
                             "expected auto or explicit")
        if self.z_range_mode == "explicit":
            if self.explicit is None:
                raise ValueError("Explicit mode requires an 'explicit' range")
            if isinstance(self.explicit, dict):
                self.explicit = ExplicitRange(
                    self.explicit["z min"],
                    self.explicit["z max"],
                )
            elif not isinstance(self.explicit, ExplicitRange):
                raise TypeError("explicit must be an ExplicitRange or mapping")
        else:
            if self.explicit is not None:
                raise ValueError("Auto mode cannot include an 'explicit' block")
            self.explicit = None

    def invalidate(self) -> None:
        """Discard particle-dependent results after bunch regrouping."""
        self.slice_id = None
        self.slice_table = None
        self.valid_turn = None
        self.valid_s = None
        self.particle_order_version = None

    def configuration(self) -> tuple:
        """Return the immutable part used to detect duplicate conflicts."""
        return (
            self.model,
            self.num_slices,
            self.z_range_mode,
            self.explicit,
        )

    @classmethod
    def from_command(cls, name: str, command_data: dict, source_command: str | None = None):
        """Build the canonical nested configuration from a Slicer entry."""
        mode = _normalize_model(command_data.get("z range mode", "auto"))
        explicit = command_data.get("explicit")
        return cls(
            name=name,
            model=command_data.get("slice model", "equal_length"),
            num_slices=command_data.get("number of slices", 10),
            z_range_mode=mode,
            explicit=explicit,
            source_command=source_command,
        )


def _positive_interval(z_min: float, z_max: float, scale: float = 1.0):
    """Ensure a non-zero interval, including for a zero-length bunch."""
    if z_max > z_min:
        return float(z_min), float(z_max)
    half = max(abs(float(scale)), 1.0) * 1.0e-12
    center = 0.5 * (float(z_min) + float(z_max))
    return center - half, center + half


def _z_interval(slice_set: SliceSet, z: np.ndarray, alive: np.ndarray):
    """Resolve the current interval in bunch-relative ``z_rel``."""
    if slice_set.z_range_mode == "explicit":
        return _positive_interval(slice_set.explicit.z_min, slice_set.explicit.z_max)
    if alive.any():
        observed = z[alive]
        return _positive_interval(float(observed.min()), float(observed.max()))
    # Empty bunch: return a tiny deterministic interval.  No bunch-level
    # sigma_z fallback is used because it describes initialization, not the
    # evolved distribution.
    return -1.0e-12, 1.0e-12


def _compile_save_turns(raw_turns, num_turns: int, command_name: str) -> bytearray:
    """Compile ``Save turns`` into a fixed turn-selection table."""
    selected = bytearray(num_turns)
    for turn_range in raw_turns:
        if not isinstance(turn_range, (list, tuple)):
            raise ValueError(f"Slicer '{command_name}': each Save turns item must be "
                             "a list of one or three integers")
        if len(turn_range) == 1:
            start, end, step = turn_range[0], turn_range[0], 1
        elif len(turn_range) == 3:
            start, end, step = turn_range
        else:
            raise ValueError(f"Slicer '{command_name}': Save turns items must be "
                             "[turn] or [start, end, step]")
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (start, end, step)):
            raise ValueError(f"Slicer '{command_name}': Save turns values must be integers")
        start, end, step = int(start), int(end), int(step)
        if step <= 0:
            raise ValueError(f"Slicer '{command_name}': Save turns step must be > 0")
        if end < start:
            raise ValueError(f"Slicer '{command_name}': turn range [{start}, {end}, {step}] "
                             "has end before start")
        original_start, original_end = start, end
        if num_turns <= 0:
            logger.warning(f"Slicer '{command_name}': num_turn={num_turns}; ignoring Save turns "
                           f"range [{start}, {end}, {step}]")
            continue
        if start >= num_turns:
            logger.warning(f"Slicer '{command_name}': start turn {start} is outside "
                           f"[0, {num_turns}); ignoring range "
                           f"[{original_start}, {original_end}, {step}]")
            continue
        if end < 0:
            logger.warning(f"Slicer '{command_name}': end turn {end} is before turn 0; "
                           f"ignoring range [{original_start}, {original_end}, {step}]")
            continue

        # Keep the selected-turn phase anchored at the configured start.  For
        # example, [-1, 5, 2] selects turns 1, 3, 5 after clipping to the run.
        first_turn = start
        if first_turn < 0:
            first_turn += ((-first_turn + step - 1) // step) * step
            logger.warning(f"Slicer '{command_name}': clipping start selection {start} "
                           f"to {first_turn}")
        last_turn = min(end, num_turns - 1)
        if end >= num_turns:
            logger.warning(f"Slicer '{command_name}': clipping end turn {end} to {last_turn}")
        if first_turn > last_turn:
            logger.warning(f"Slicer '{command_name}': range [{original_start}, {original_end}, "
                           f"{step}] has no selected turn in [0, {num_turns}); ignoring it")
            continue
        count = ((last_turn - first_turn) // step) + 1
        selected[first_turn:last_turn + 1:step] = b"\x01" * count
    return selected


def _slice_one_bunch_cpu(p, bunch, slice_set: SliceSet) -> None:
    """Compute IDs and statistics for one contiguous bunch range on CPU."""
    start, end = int(bunch.start_idx), int(bunch.end_idx)
    z = np.asarray(p.z[start:end])
    n_slices = slice_set.num_slices
    alive = np.asarray(p.tag[start:end]) > 0
    n_alive = int(np.count_nonzero(alive))
    if n_alive < n_slices:
        logger.warning(f"Slicer {slice_set.name}: bunch {getattr(bunch, 'bunch_id', '?')} "
                       f"has {n_alive} alive particles but {n_slices} slices; empty slices "
                       "will be retained")
    z_min, z_max = _z_interval(slice_set, z, alive)
    outside = alive & ((z < z_min) | (z > z_max))
    if np.any(outside):
        logger.warning(f"Slicer {slice_set.name}: {int(np.count_nonzero(outside))} alive "
                       f"particles in bunch {getattr(bunch, 'bunch_id', '?')} lie outside "
                       f"[{z_min:g}, {z_max:g}] for {slice_set.z_range_mode}; assigning them "
                       "to boundary slices")
    width = (z_max - z_min) / n_slices
    local_id = np.full(z.size, -1, dtype=np.int32)
    active_z = np.clip(z[alive], z_min, z_max)
    if active_z.size:
        if slice_set.model in {"equal_particle", "equal_charge"}:
            # Rank assignment keeps populations balanced even when several
            # particles have identical z.  Sorting is temporary; no
            # ParticlePool array is reordered.
            order = np.argsort(active_z, kind="stable")
            ranks = np.empty(active_z.size, dtype=np.int64)
            ranks[order] = np.arange(active_z.size, dtype=np.int64)
            active_id = (n_slices - 1 - np.minimum((ranks * n_slices) // active_z.size, n_slices - 1)).astype(np.int32)
        else:
            active_id = np.floor((active_z - z_min) / width).astype(np.int32)
            active_id = np.clip(active_id, 0, n_slices - 1)
            active_id = n_slices - 1 - active_id
        local_id[alive] = active_id

    counts = np.bincount(local_id[local_id >= 0], minlength=n_slices).astype(np.int64)
    if active_z.size and slice_set.model in {"equal_particle", "equal_charge"}:
        quantiles = np.linspace(0.0, 1.0, n_slices + 1)
        edges = np.quantile(active_z, quantiles, method="linear")
        edges[0], edges[-1] = z_min, z_max
    else:
        edges = np.linspace(z_min, z_max, n_slices + 1, dtype=float)
    # Keep every interval's bounds valid (z_min < z_max), but expose slices
    # in descending-z order so slice 0 is the high-z end of the bunch.
    table_z_min = edges[-2::-1]
    table_z_max = edges[:0:-1]
    delta_z = table_z_max - table_z_min
    real_charge = counts.astype(float) * float(getattr(bunch, "ratio", 0.0))
    lind_density = np.divide(real_charge, delta_z, out=np.zeros_like(real_charge), where=delta_z > 0)
    slice_set.slice_id = local_id
    slice_set.slice_table = {
        "z_min": table_z_min,
        "z_max": table_z_max,
        "z_center": 0.5 * (table_z_min + table_z_max),
        "delta_z": delta_z,
        "macro_count": counts,
        "effective_num_slices": int(min(n_alive, n_slices)),
        "real_charge": real_charge,
        "lind_density": lind_density,
    }


@Command.register("Slicer")
class Slicer(Command):
    """Sequence command that updates one named :class:`SliceSet`.

    Sequence parameters:

    ``s (m)``
        Longitudinal machine position at which this command is executed.
        It is stored in ``SliceSet.valid_s`` so consumers can verify that
        their field data corresponds to the current lattice location.
    ``slice set``
        Required user-defined key, such as ``space_charge`` or
        ``beambeam_ip1``.  Only this set is updated; other named sets remain
        untouched.
    ``slice model``
        ``equal_length`` or ``equal_particle``.  This option belongs to the
        set configuration and is checked during initialization, rather than
        being reinterpreted independently at every execution.
    ``number of slices``
        Number of longitudinal bins in the selected set.
        ``z range mode``
        Selects ``auto`` or ``explicit``.
    ``explicit``
        Contains ``z min`` and ``z max`` for explicit mode.
    ``save turns``
        Optional turn selections, each ``[turn]`` or ``[start, end, step]``.
        Selected executions write a particle-to-slice TFS snapshot and a
        per-slice TFS summary.

    The last five options are registered at beam initialization and are not
    duplicated into command runtime state.  This ensures repeated commands
    referring to one key have one unambiguous configuration.
    """

    def __init__(self, beam_id: int, sim, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}
        self.beam_id = beam_id
        self.s = float(kwargs["s (m)"])
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]
        self.slice_set_name = str(kwargs["slice set"]).strip()
        if not self.slice_set_name:
            raise ValueError("Slicer requires a non-empty 'slice set' name")
        self._selected_turns = _compile_save_turns(kwargs.get("save turns", []), int(sim.cfg.num_turn), self.cmd_name)
        self.output_dir = Path(sim.cfg.output_dir_slice)
        # GPU workspaces are allocated lazily and retained for the lifetime
        # of this command.  This avoids per-turn allocations for the hot path.
        self._gpu_workspaces: dict[int, dict[str, Any]] = {}
        super().__init__()

    def execute_cpu(self, sim):
        beam = sim.beams[self.beam_id]
        turn = int(sim.state.turn)
        for bunch in beam.bunches:
            try:
                slice_set = bunch.slice_sets[self.slice_set_name]
            except KeyError as exc:
                raise KeyError(f"Bunch {bunch.bunch_id} has no SliceSet {self.slice_set_name!r}") from exc
            _slice_one_bunch_cpu(beam.particles, bunch, slice_set)
            slice_set.valid_turn = turn
            slice_set.valid_s = self.s
            if self._selected_turns[turn]:
                self._save_snapshot(sim, beam, bunch, slice_set, turn)
        return True

    def execute_gpu(self, sim):
        # CuPy JIT artifacts are kept in memory (configured at module import),
        # so this path does not depend on a writable user cache directory.
        try:
            import cupy as cp
        except (ImportError, OSError) as exc:
            raise RuntimeError("GPU Slicer requires the optional 'cuda' dependencies "
                               "(install PASS with the [cuda] extra).") from exc

        beam = sim.beams[self.beam_id]
        turn = int(sim.state.turn)
        kernels = _get_slicer_kernels(beam.particles.dtype)

        for bunch in beam.bunches:
            try:
                slice_set = bunch.slice_sets[self.slice_set_name]
            except KeyError as exc:
                raise KeyError(f"Bunch {bunch.bunch_id} has no SliceSet {self.slice_set_name!r}") from exc
            self._slice_one_bunch_gpu(beam, bunch, slice_set, kernels, cp)
            slice_set.valid_turn = turn
            slice_set.valid_s = self.s
            if self._selected_turns[turn]:
                self._save_snapshot(sim, beam, bunch, slice_set, turn)
        return True

    def _workspace(self, bunch, slice_set: SliceSet, p, cp):
        """Return a capacity-reusable workspace for one bunch/slice set."""
        n = max(1, int(bunch.end_idx) - int(bunch.start_idx))
        ns = int(slice_set.num_slices)
        ws = self._gpu_workspaces.get(int(bunch.bunch_id))
        if ws is not None and ws["capacity"] >= n and ws["num_slices"] == ns:
            return ws

        # Keep a little headroom so small bunch-size changes do not trigger a
        # new allocation.  All arrays are device-resident.
        capacity = max(n, int(ws["capacity"] * 1.5) if ws is not None else n)
        real_dtype = p.dtype
        ws = {
            "capacity": capacity,
            "num_slices": ns,
            "slice_id": cp.empty(capacity, dtype=cp.int32),
            "z_min": cp.empty(1, dtype=real_dtype),
            "z_max": cp.empty(1, dtype=real_dtype),
            "alive_count": cp.empty(1, dtype=cp.int32),
            "outside_count": cp.empty(1, dtype=cp.int32),
            "counts": cp.empty(ns, dtype=cp.int32),
            "edges": cp.empty(ns + 1, dtype=real_dtype),
            "table_z_min": cp.empty(ns, dtype=real_dtype),
            "table_z_max": cp.empty(ns, dtype=real_dtype),
            "table_z_center": cp.empty(ns, dtype=real_dtype),
            "table_delta_z": cp.empty(ns, dtype=real_dtype),
            "table_real_charge": cp.empty(ns, dtype=real_dtype),
            "table_lind_density": cp.empty(ns, dtype=real_dtype),
            "active_index": cp.empty(capacity, dtype=cp.int32),
            "sort_keys_a": cp.empty(capacity, dtype=real_dtype),
            "sort_keys_b": cp.empty(capacity, dtype=real_dtype),
            "sort_values_a": cp.empty(capacity, dtype=cp.int32),
            "sort_values_b": cp.empty(capacity, dtype=cp.int32),
            "sorter": None,
            "sort_temp": None,
        }

        # CCCL is the primary stable-sort implementation.  Keep a CuPy
        # argsort fallback for installations where cuda.compute is absent.
        try:
            from cuda.compute import SortOrder, make_radix_sort
            sorter = make_radix_sort(
                d_in_keys=ws["sort_keys_a"],
                d_out_keys=ws["sort_keys_b"],
                d_in_values=ws["sort_values_a"],
                d_out_values=ws["sort_values_b"],
                order=SortOrder.ASCENDING,
            )
            required = sorter(
                temp_storage=None,
                d_in_keys=ws["sort_keys_a"],
                d_out_keys=ws["sort_keys_b"],
                d_in_values=ws["sort_values_a"],
                d_out_values=ws["sort_values_b"],
                num_items=capacity,
            )
            ws["sorter"] = sorter
            ws["sort_temp"] = cp.empty(max(1, int(required)), dtype=cp.uint8)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            # Do not make the CPU/GPU command unavailable merely because an
            # optional CCCL build is missing.  CuPy's stable argsort remains
            # a fully GPU-resident fallback.
            ws["sorter"] = False

        self._gpu_workspaces[int(bunch.bunch_id)] = ws
        return ws

    @staticmethod
    def _launch_1d(kernel, n, args, threads=256, shared_mem=0):
        if n <= 0:
            return
        blocks = min(max(1, (int(n) + threads - 1) // threads), 4096)
        kernel((blocks, ), (threads, ), args, shared_mem=shared_mem)

    def _slice_one_bunch_gpu(self, beam, bunch, slice_set, kernels, cp):
        p = beam.particles
        start, end = int(bunch.start_idx), int(bunch.end_idx)
        n = end - start
        ws = self._workspace(bunch, slice_set, p, cp)
        ns = int(slice_set.num_slices)
        real = p.real

        if slice_set.z_range_mode == "explicit":
            z_min = real(slice_set.explicit.z_min)
            z_max = real(slice_set.explicit.z_max)
        else:
            z_min = real(np.inf)
            z_max = real(-np.inf)

        init_n = max(n, ns, 1)
        init_threads = 256
        init_blocks = min(max(1, (init_n + init_threads - 1) // init_threads), 4096)
        kernels["slicer_initialize"](
            (init_blocks, ),
            (init_threads, ),
            (np.int32(n), np.int32(ns), z_min, z_max, ws["z_min"], ws["z_max"], ws["alive_count"], ws["outside_count"], ws["slice_id"], ws["counts"]),
        )

        if slice_set.z_range_mode != "explicit":
            threads = 256
            blocks = min(max(1, (n + threads - 1) // threads), 4096)
            kernels["slicer_reduce"](
                (blocks, ),
                (threads, ),
                (p.z, p.tag, np.int32(start), np.int32(end), ws["z_min"], ws["z_max"], ws["alive_count"]),
            )
            # RawKernel scalar arguments must be host scalar values; passing
            # a 0-D CuPy array would pass its device pointer as the numeric
            # value.  This is a tiny metadata transfer and also gives us the
            # exact interval used by the subsequent assignment kernel.
            reduced_min = float(ws["z_min"].get()[0])
            reduced_max = float(ws["z_max"].get()[0])
            reduced_alive = int(ws["alive_count"].get()[0])
            if reduced_alive > 0 and np.isfinite(reduced_min) and np.isfinite(reduced_max):
                if reduced_max > reduced_min:
                    z_min, z_max = real(reduced_min), real(reduced_max)
                else:
                    # Match CPU _positive_interval for a zero-width live
                    # distribution by preserving its observed center.
                    half = max(abs(reduced_min), 1.0) * 1.0e-12
                    center = 0.5 * (reduced_min + reduced_max)
                    z_min, z_max = real(center - half), real(center + half)
            else:
                # No live finite particles: use the deterministic empty-bunch
                # interval shared with the CPU implementation.
                z_min, z_max = real(-1.0e-12), real(1.0e-12)
            if slice_set.model == "equal_length":
                # The assignment kernel counts alive particles while building
                # its histogram.  Discard the reduction count first so the
                # metadata is not accumulated twice.
                ws["alive_count"][0] = 0

        if slice_set.model == "equal_length":
            threads = 256
            blocks = min(max(1, (n + threads - 1) // threads), 4096)
            shared = ns * np.dtype(np.int32).itemsize
            # Shared histogram is fastest for normal slice counts.  For very
            # large meshes use direct global atomics to stay within limits.
            if shared <= 48 * 1024:
                kernels["slicer_assign"](
                    (blocks, ),
                    (threads, ),
                    (p.z, p.tag, np.int32(start), np.int32(end), z_min, z_max, np.int32(ns), ws["slice_id"], ws["counts"], ws["outside_count"],
                     ws["alive_count"]),
                    shared_mem=shared,
                )
            else:
                kernels["slicer_assign_global"](
                    (blocks, ),
                    (threads, ),
                    (p.z, p.tag, np.int32(start), np.int32(end), z_min, z_max, np.int32(ns), ws["slice_id"], ws["counts"], ws["outside_count"],
                     ws["alive_count"]),
                )
            self._launch_1d(
                kernels["slicer_edges_uniform"],
                ns + 1,
                (z_min, z_max, np.int32(ns), ws["edges"]),
            )
            n_alive = None
        else:
            # The active selection is stable in original particle order.  It
            # is used only as a temporary index list; ParticlePool is never
            # reordered by Slicer.
            active_index = cp.flatnonzero(p.tag[start:end] > 0)
            n_alive = int(active_index.size)
            if n_alive:
                ws["active_index"][:n_alive] = active_index
                if slice_set.z_range_mode == "auto":
                    # z_min/z_max are device scalars from the reduction.
                    pass
                else:
                    ws["z_min"][0] = z_min
                    ws["z_max"][0] = z_max
                if ws["sorter"] is not False:
                    gather_keys = ws["sort_keys_a"]
                    gather_values = ws["sort_values_a"]
                else:
                    gather_keys = ws["sort_keys_a"]
                    gather_values = ws["sort_values_a"]
                self._launch_1d(
                    kernels["slicer_gather_active"],
                    n_alive,
                    (p.z, np.int32(start), ws["active_index"], np.int32(n_alive), z_min, z_max, gather_keys, gather_values, ws["outside_count"]),
                )
                if ws["sorter"] is not False:
                    sorter = ws["sorter"]
                    sorter(
                        temp_storage=ws["sort_temp"],
                        d_in_keys=ws["sort_keys_a"],
                        d_out_keys=ws["sort_keys_b"],
                        d_in_values=ws["sort_values_a"],
                        d_out_values=ws["sort_values_b"],
                        num_items=n_alive,
                    )
                    sorted_z = ws["sort_keys_b"]
                    sorted_index = ws["sort_values_b"]
                else:
                    order = cp.argsort(ws["sort_keys_a"][:n_alive], kind="stable")
                    sorted_z = ws["sort_keys_b"][:n_alive]
                    sorted_index = ws["sort_values_b"][:n_alive]
                    sorted_z[...] = ws["sort_keys_a"][:n_alive][order]
                    sorted_index[...] = ws["sort_values_a"][:n_alive][order]
                self._launch_1d(
                    kernels["slicer_scatter_rank"],
                    n_alive,
                    (sorted_index, np.int32(n_alive), np.int32(ns), ws["slice_id"]),
                )
            else:
                # Empty bunches use the same deterministic interval as CPU.
                if slice_set.z_range_mode == "auto":
                    z_min, z_max = real(-1.0e-12), real(1.0e-12)
                # The initialization kernel already populated these scalars
                # for explicit ranges; only the empty auto case needs a
                # deterministic host-side value.
                if slice_set.z_range_mode == "auto":
                    ws["z_min"][0] = z_min
                    ws["z_max"][0] = z_max
                # Quantile edges are undefined without live particles.  Use
                # the same uniform diagnostic geometry as the CPU path so the
                # subsequent table kernel never reads uninitialized memory.
                self._launch_1d(
                    kernels["slicer_edges_uniform"],
                    ns + 1,
                    (z_min, z_max, np.int32(ns), ws["edges"]),
                )
            self._launch_1d(
                kernels["slicer_fill_counts_equal_particle"],
                ns,
                (np.int32(n_alive or 0), np.int32(ns), ws["counts"]),
            )
            if n_alive:
                # The sorter may have toggled its DoubleBuffer selector; the
                # quantile kernel needs the currently valid sorted z buffer.
                sorted_z = ws["sort_keys_b"][:n_alive]
                self._launch_1d(
                    kernels["slicer_edges_quantile"],
                    ns + 1,
                    (sorted_z, np.int32(n_alive), np.int32(ns), z_min, z_max, ws["edges"]),
                )

        # For equal_length the reduction result is needed only for diagnostics
        # and effective_num_slices.  Copying this tiny metadata is intentional;
        # particle arrays remain device-resident.
        alive_host = int(ws["alive_count"].get()[0]) if n_alive is None else n_alive
        outside_host = int(ws["outside_count"].get()[0])
        z_min_host = float(ws["z_min"].get()[0]) if slice_set.z_range_mode == "auto" else float(z_min)
        z_max_host = float(ws["z_max"].get()[0]) if slice_set.z_range_mode == "auto" else float(z_max)
        if alive_host < ns:
            logger.warning(f"Slicer {slice_set.name}: bunch {getattr(bunch, 'bunch_id', '?')} "
                           f"has {alive_host} alive particles but {ns} slices; empty slices "
                           "will be retained")
        if outside_host:
            logger.warning(f"Slicer {slice_set.name}: {outside_host} alive particles in bunch "
                           f"{getattr(bunch, 'bunch_id', '?')} lie outside "
                           f"[{z_min_host:g}, {z_max_host:g}] for {slice_set.z_range_mode}; "
                           "assigning them to boundary slices")

        self._launch_1d(
            kernels["slicer_build_table"],
            ns,
            (ws["edges"], ws["counts"], np.int32(ns), real(getattr(bunch, "ratio", 0.0)), ws["table_z_min"], ws["table_z_max"], ws["table_z_center"],
             ws["table_delta_z"], ws["table_real_charge"], ws["table_lind_density"]),
        )
        slice_set.slice_id = ws["slice_id"][:max(0, n)]
        slice_set.slice_table = {
            "z_min": ws["table_z_min"],
            "z_max": ws["table_z_max"],
            "z_center": ws["table_z_center"],
            "delta_z": ws["table_delta_z"],
            "macro_count": ws["counts"],
            "effective_num_slices": int(min(alive_host, ns)),
            "real_charge": ws["table_real_charge"],
            "lind_density": ws["table_lind_density"],
        }

    def _save_snapshot(self, sim, beam, bunch, slice_set: SliceSet, turn: int) -> None:
        """Write one same-instant particle and per-slice result snapshot."""
        start, end = int(bunch.start_idx), int(bunch.end_idx)
        particles = beam.particles
        self.output_dir.mkdir(parents=True, exist_ok=True)

        common_headers = {
            "Name": "PASS Slicer Snapshot",
            "Command": self.cmd_type,
            "CommandName": self.cmd_name,
            "SliceSet": slice_set.name,
            "SliceModel": slice_set.model,
            "ZRangeMode": slice_set.z_range_mode,
            "S": self.s,
            "Turn": turn,
            "BeamId": self.beam_id,
            "BeamName": beam.beam_name,
            "BunchId": int(bunch.bunch_id),
            "HarmonicId": int(bunch.harmonic_id),
            "NumSlices": slice_set.num_slices,
            "NumAlive": int(np.count_nonzero(_as_host(particles.tag[start:end]) > 0)),
            "ZCoordinate": "z_rel",
            "ZCenter": float(bunch.z_center),
            "PASSVersion": __version__,
            "Time": get_current_time(),
        }
        particle_df = pd.DataFrame({
            "tag": _as_host(particles.tag[start:end]),
            "z": _as_host(particles.z[start:end]),
            "slice_id": _as_host(slice_set.slice_id),
            "lost_turn": _as_host(particles.lost_turn[start:end]),
            "lost_position": _as_host(particles.lost_position[start:end]),
        })

        safe_command = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(self.cmd_name)).strip("_")
        safe_set = re.sub(r"[^A-Za-z0-9_.-]+", "_", slice_set.name).strip("_")
        safe_command = safe_command or "slicer"
        safe_set = safe_set or "slice_set"
        stem = (f"{sim.cfg.output_hms}_slice_beam{self.beam_id}_bunch{int(bunch.bunch_id)}"
                f"_{safe_command}_{safe_set}_s_{self.s:.4f}_turn_{turn}")
        tfs.write(
            str(self.output_dir / f"{stem}_particles.tfs"),
            tfs.TfsDataFrame(particle_df, headers=common_headers),
        )

        table = slice_set.slice_table
        summary_df = pd.DataFrame({
            "slice_id": np.arange(slice_set.num_slices, dtype=np.int32),
            "z_min": _as_host(table["z_min"]),
            "z_max": _as_host(table["z_max"]),
            "z_center": _as_host(table["z_center"]),
            "delta_z": _as_host(table["delta_z"]),
            "macro_count": _as_host(table["macro_count"]),
            "real_charge": _as_host(table["real_charge"]),
            "lind_density": _as_host(table["lind_density"]),
        })
        summary_df.insert(0, "effective_num_slices", table["effective_num_slices"])
        summary_df.insert(0, "num_alive", common_headers["NumAlive"])
        summary_df.insert(0, "num_slices", slice_set.num_slices)
        summary_df.insert(0, "bunch_id", int(bunch.bunch_id))
        summary_df.insert(0, "beam_id", self.beam_id)
        summary_df.insert(0, "s", self.s)
        summary_df.insert(0, "turn", turn)
        summary_headers = dict(common_headers)
        summary_headers["Name"] = "PASS Slicer Summary"
        summary_headers["DataType"] = "per_slice"
        # Keep the established CSV artifact for analysis scripts while also
        # providing the TFS representation used by newer workflows.
        summary_df.to_csv(self.output_dir / f"{stem}_summary.csv", index=False)
        tfs.write(
            str(self.output_dir / f"{stem}_summary.tfs"),
            tfs.TfsDataFrame(summary_df, headers=summary_headers),
        )
        logger.info(f"Slicer '{self.cmd_name}': saved "
                    f"{self.output_dir / f'{stem}_particles.tfs'}")

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, "
                    f"Name={self.cmd_name:s}, Slice set={self.slice_set_name:s}, "
                    f"Save turns={sum(self._selected_turns):d}")
        set_normal_logging()


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

_SLICER_CUDA_PREAMBLE = r'''
#ifndef PASS_USE_FLOAT
#define PASS_USE_FLOAT 0
#endif

#if PASS_USE_FLOAT
using pass_real_t = float;
#define PASS_FABS fabsf
#define PASS_FLOOR floorf
#define PASS_ISFINITE isfinite
#define PASS_POS_INF 3.402823466e+38F
#else
using pass_real_t = double;
#define PASS_FABS fabs
#define PASS_FLOOR floor
#define PASS_ISFINITE isfinite
#define PASS_POS_INF 1.7976931348623157e+308
#endif
'''

_SLICER_CUDA_BODY = r'''
extern "C" __device__ __forceinline__ void pass_atomic_min_real(
    pass_real_t* address, pass_real_t value)
{
    if (!PASS_ISFINITE(value)) return;
#if PASS_USE_FLOAT
    int* bits = reinterpret_cast<int*>(address);
    int old = *bits;
    while (true) {
        pass_real_t current = __int_as_float(old);
        if (current <= value) return;
        int assumed = old;
        old = atomicCAS(bits, assumed, __float_as_int(value));
        if (old == assumed) return;
    }
#else
    unsigned long long* bits = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *bits;
    while (true) {
        pass_real_t current = __longlong_as_double((long long)old);
        if (current <= value) return;
        unsigned long long assumed = old;
        old = atomicCAS(bits, assumed,
                        (unsigned long long)__double_as_longlong(value));
        if (old == assumed) return;
    }
#endif
}

extern "C" __device__ __forceinline__ void pass_atomic_max_real(
    pass_real_t* address, pass_real_t value)
{
    if (!PASS_ISFINITE(value)) return;
#if PASS_USE_FLOAT
    int* bits = reinterpret_cast<int*>(address);
    int old = *bits;
    while (true) {
        pass_real_t current = __int_as_float(old);
        if (current >= value) return;
        int assumed = old;
        old = atomicCAS(bits, assumed, __float_as_int(value));
        if (old == assumed) return;
    }
#else
    unsigned long long* bits = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *bits;
    while (true) {
        pass_real_t current = __longlong_as_double((long long)old);
        if (current >= value) return;
        unsigned long long assumed = old;
        old = atomicCAS(bits, assumed,
                        (unsigned long long)__double_as_longlong(value));
        if (old == assumed) return;
    }
#endif
}

extern "C" __global__ void slicer_reduce(
    const pass_real_t* __restrict__ z,
    const int* __restrict__ tag,
    int start,
    int end,
    pass_real_t* __restrict__ z_min,
    pass_real_t* __restrict__ z_max,
    int* __restrict__ alive_count)
{
    __shared__ pass_real_t warp_min[32];
    __shared__ pass_real_t warp_max[32];
    __shared__ int warp_count[32];

    pass_real_t local_min = (pass_real_t)PASS_POS_INF;
    pass_real_t local_max = (pass_real_t)-PASS_POS_INF;
    int local_count = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start + tid; i < end; i += stride) {
        if (tag[i] <= 0) continue;
        pass_real_t zi = z[i];
        if (!PASS_ISFINITE(zi)) continue;
        local_min = local_min < zi ? local_min : zi;
        local_max = local_max > zi ? local_max : zi;
        local_count += 1;
    }

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        pass_real_t other_min = __shfl_down_sync(0xffffffff, local_min, offset);
        pass_real_t other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        local_min = local_min < other_min ? local_min : other_min;
        local_max = local_max > other_max ? local_max : other_max;
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }
    if (lane == 0) {
        warp_min[warp] = local_min;
        warp_max[warp] = local_max;
        warp_count[warp] = local_count;
    }
    __syncthreads();
    if (warp == 0) {
        pass_real_t block_min = lane < ((blockDim.x + 31) >> 5)
            ? warp_min[lane] : (pass_real_t)PASS_POS_INF;
        pass_real_t block_max = lane < ((blockDim.x + 31) >> 5)
            ? warp_max[lane] : (pass_real_t)-PASS_POS_INF;
        int block_count = lane < ((blockDim.x + 31) >> 5)
            ? warp_count[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            pass_real_t other_min = __shfl_down_sync(0xffffffff, block_min, offset);
            pass_real_t other_max = __shfl_down_sync(0xffffffff, block_max, offset);
            block_min = block_min < other_min ? block_min : other_min;
            block_max = block_max > other_max ? block_max : other_max;
            block_count += __shfl_down_sync(0xffffffff, block_count, offset);
        }
        if (lane == 0) {
            pass_atomic_min_real(z_min, block_min);
            pass_atomic_max_real(z_max, block_max);
            atomicAdd(alive_count, block_count);
        }
    }
}

extern "C" __global__ void slicer_initialize(
    int n,
    int num_slices,
    pass_real_t z_min_value,
    pass_real_t z_max_value,
    pass_real_t* __restrict__ z_min,
    pass_real_t* __restrict__ z_max,
    int* __restrict__ alive_count,
    int* __restrict__ outside_count,
    int* __restrict__ slice_id,
    int* __restrict__ counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        z_min[0] = z_min_value;
        z_max[0] = z_max_value;
        alive_count[0] = 0;
        outside_count[0] = 0;
    }
    if (i < n) slice_id[i] = -1;
    if (i < num_slices) counts[i] = 0;
}

extern "C" __global__ void slicer_assign(
    const pass_real_t* __restrict__ z,
    const int* __restrict__ tag,
    int start,
    int end,
    pass_real_t z_min,
    pass_real_t z_max,
    int num_slices,
    int* __restrict__ slice_id,
    int* __restrict__ counts,
    int* __restrict__ outside_count,
    int* __restrict__ alive_count)
{
    extern __shared__ int local_counts[];
    for (int s = threadIdx.x; s < num_slices; s += blockDim.x)
        local_counts[s] = 0;
    __syncthreads();

    if (!(z_min < z_max)) {
        z_min = (pass_real_t)-1.0e-12;
        z_max = (pass_real_t)1.0e-12;
    }
    pass_real_t width = (z_max - z_min) / (pass_real_t)num_slices;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start + tid; i < end; i += stride) {
        int local = i - start;
        if (tag[i] <= 0) {
            slice_id[local] = -1;
            continue;
        }
        pass_real_t zi = z[i];
        bool outside = zi < z_min || zi > z_max;
        if (outside) atomicAdd(outside_count, 1);
        atomicAdd(alive_count, 1);
        pass_real_t clipped = zi < z_min ? z_min : (zi > z_max ? z_max : zi);
        int sid = (int)PASS_FLOOR((clipped - z_min) / width);
        sid = sid < 0 ? 0 : (sid >= num_slices ? num_slices - 1 : sid);
        sid = num_slices - 1 - sid;
        slice_id[local] = sid;
        atomicAdd(&local_counts[sid], 1);
    }
    __syncthreads();
    for (int s = threadIdx.x; s < num_slices; s += blockDim.x) {
        if (local_counts[s] != 0) atomicAdd(&counts[s], local_counts[s]);
    }
}

// Fallback for meshes that do not fit in per-block shared memory.
extern "C" __global__ void slicer_assign_global(
    const pass_real_t* __restrict__ z,
    const int* __restrict__ tag,
    int start,
    int end,
    pass_real_t z_min,
    pass_real_t z_max,
    int num_slices,
    int* __restrict__ slice_id,
    int* __restrict__ counts,
    int* __restrict__ outside_count,
    int* __restrict__ alive_count)
{
    if (!(z_min < z_max)) {
        z_min = (pass_real_t)-1.0e-12;
        z_max = (pass_real_t)1.0e-12;
    }
    pass_real_t width = (z_max - z_min) / (pass_real_t)num_slices;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start + tid; i < end; i += stride) {
        int local = i - start;
        if (tag[i] <= 0) {
            slice_id[local] = -1;
            continue;
        }
        pass_real_t zi = z[i];
        if (zi < z_min || zi > z_max) atomicAdd(outside_count, 1);
        atomicAdd(alive_count, 1);
        pass_real_t clipped = zi < z_min ? z_min : (zi > z_max ? z_max : zi);
        int sid = (int)PASS_FLOOR((clipped - z_min) / width);
        sid = sid < 0 ? 0 : (sid >= num_slices ? num_slices - 1 : sid);
        sid = num_slices - 1 - sid;
        slice_id[local] = sid;
        atomicAdd(&counts[sid], 1);
    }
}

extern "C" __global__ void slicer_gather_active(
    const pass_real_t* __restrict__ z,
    int start,
    const int* __restrict__ active_index,
    int n_active,
    pass_real_t z_min,
    pass_real_t z_max,
    pass_real_t* __restrict__ keys,
    int* __restrict__ values,
    int* __restrict__ outside_count)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_active) return;
    int local = active_index[j];
    pass_real_t zi = z[start + local];
    if (zi < z_min || zi > z_max) atomicAdd(outside_count, 1);
    zi = zi < z_min ? z_min : (zi > z_max ? z_max : zi);
    keys[j] = zi;
    values[j] = local;
}

extern "C" __global__ void slicer_scatter_rank(
    const int* __restrict__ sorted_index,
    int n_active,
    int num_slices,
    int* __restrict__ slice_id)
{
    int rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= n_active) return;
    int sid = num_slices - 1 - (int)(((long long)rank * (long long)num_slices) /
                    (long long)n_active);
    if (sid < 0) sid = 0;
    slice_id[sorted_index[rank]] = sid;
}

extern "C" __global__ void slicer_edges_quantile(
    const pass_real_t* __restrict__ sorted_z,
    int n_active,
    int num_slices,
    pass_real_t z_min,
    pass_real_t z_max,
    pass_real_t* __restrict__ edges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > num_slices) return;
    if (n_active <= 0) {
        edges[i] = z_min + (z_max - z_min) *
            ((pass_real_t)i / (pass_real_t)num_slices);
        return;
    }
    pass_real_t pos = ((pass_real_t)i / (pass_real_t)num_slices) *
                      (pass_real_t)(n_active - 1);
    int left = (int)PASS_FLOOR(pos);
    int right = left + 1;
    if (right >= n_active) right = n_active - 1;
    pass_real_t frac = pos - (pass_real_t)left;
    edges[i] = sorted_z[left] + frac * (sorted_z[right] - sorted_z[left]);
    if (i == 0) edges[i] = z_min;
    if (i == num_slices) edges[i] = z_max;
}

extern "C" __global__ void slicer_edges_uniform(
    pass_real_t z_min,
    pass_real_t z_max,
    int num_slices,
    pass_real_t* __restrict__ edges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > num_slices) return;
    edges[i] = z_min + (z_max - z_min) *
        ((pass_real_t)i / (pass_real_t)num_slices);
}

extern "C" __global__ void slicer_build_table(
    const pass_real_t* __restrict__ edges,
    const int* __restrict__ counts,
    int num_slices,
    pass_real_t ratio,
    pass_real_t* __restrict__ z_min,
    pass_real_t* __restrict__ z_max,
    pass_real_t* __restrict__ z_center,
    pass_real_t* __restrict__ delta_z,
    pass_real_t* __restrict__ real_charge,
    pass_real_t* __restrict__ lind_density)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_slices) return;
    int edge = num_slices - 1 - i;
    pass_real_t lo = edges[edge];
    pass_real_t hi = edges[edge + 1];
    pass_real_t dz = hi - lo;
    z_min[i] = lo;
    z_max[i] = hi;
    z_center[i] = (pass_real_t)0.5 * (lo + hi);
    delta_z[i] = dz;
    real_charge[i] = (pass_real_t)counts[i] * ratio;
    lind_density[i] = dz > (pass_real_t)0.0
        ? real_charge[i] / dz : (pass_real_t)0.0;
}

extern "C" __global__ void slicer_fill_counts_equal_particle(
    int n_active,
    int num_slices,
    int* __restrict__ counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_slices) return;
    int reverse_i = num_slices - 1 - i;
    long long hi = (((long long)(reverse_i + 1) * (long long)n_active) +
                    (long long)num_slices - 1LL) /
                   (long long)num_slices;
    long long lo = (((long long)reverse_i * (long long)n_active) +
                    (long long)num_slices - 1LL) /
                   (long long)num_slices;
    counts[i] = (int)(hi - lo);
}
'''

_SLICER_CUDA_SOURCE = _SLICER_CUDA_PREAMBLE + _SLICER_CUDA_BODY
_slicer_kernel_cache: dict[np.dtype, dict[str, Any]] = {}


def _get_slicer_kernels(dtype):
    """Compile Slicer CUDA kernels once per particle precision."""
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU Slicer requires the optional 'cuda' dependencies "
                           "(install PASS with the [cuda] extra).") from exc
    key = np.dtype(dtype)
    kernels = _slicer_kernel_cache.get(key)
    if kernels is None:
        use_float = key == np.dtype(np.float32)
        options = ("--std=c++14", f"-DPASS_USE_FLOAT={int(use_float)}")
        names = (
            "slicer_initialize",
            "slicer_reduce",
            "slicer_assign",
            "slicer_assign_global",
            "slicer_gather_active",
            "slicer_scatter_rank",
            "slicer_edges_quantile",
            "slicer_edges_uniform",
            "slicer_build_table",
            "slicer_fill_counts_equal_particle",
        )
        kernels = {name: cp.RawKernel(_SLICER_CUDA_SOURCE, name, options=options) for name in names}
        _slicer_kernel_cache[key] = kernels
    return kernels


def _is_cupy_array(value) -> bool:
    return value is not None and value.__class__.__module__.startswith("cupy")


def _as_host(value):
    return value.get() if _is_cupy_array(value) else np.asarray(value)
