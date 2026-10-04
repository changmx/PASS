from __future__ import annotations

from functools import lru_cache
import hashlib
import logging
import os
from pathlib import Path
import uuid

import numpy as np

from PASS.commands.command import Command
from PASS.core.config import Config
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.utils.logger import set_simple_logging, set_normal_logging
from PASS.utils.helper import get_current_time
from PASS.utils.particle_monitor_io import append_particle_samples, export_particle_samples_tfs, restore_particle_samples_tfs
from PASS.utils.table_io import normalize_output_format, table_path

logger = logging.getLogger(__name__)


@Command.register("particlemonitor")
class ParticleMonitor(Command):
    """Turn-by-turn particle coordinate monitor.

    Records 6D coordinates (+ tag, lost_turn, lost_position) of particles
    with ``1 <= |tag| <= max_tag`` every turn within ``[start_turn, end_turn)``
    at the monitor's s-position.

    ``Include reference`` defaults to False (11 columns). When enabled,
    three per-row reference columns are appended for physical-time and
    momentum reconstruction. Reference values are never stored in headers.

    One file stores every selected particle and recorded turn. Write interval
    (turns) bounds buffering without downsampling the trajectories. Actual
    injected coordinates are recorded independently of monitor samples. TFS
    output is converted from a private HDF5 file after recording stops.

    File naming:
        {hms}_beam{bid}_{monitor_name}_s{s:.3f}_particles.{h5|tfs}
    """

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]
        self.output_format = normalize_output_format(kwargs.get("output format", "hdf5"))
        if "output layout" in kwargs or "output_layout" in kwargs:
            raise ValueError("ParticleMonitor no longer accepts Output layout")
        self.write_interval_turns = kwargs.get("write interval (turns)", 128)
        if type(self.write_interval_turns) is not int or self.write_interval_turns < 1:
            raise ValueError("Write interval (turns) must be a positive integer")
        self.include_reference: bool = kwargs.get("include reference", False)
        self._column_names = (
            "turn",
            "x",
            "px",
            "y",
            "py",
            "z",
            "dp",
            "tag",
            "lostTurn",
            "lostPosition",
            "zCenter",
        )
        if self.include_reference:
            self._column_names += ("referenceTime", "referenceBeta", "referenceMomentum")
        self._num_columns = len(self._column_names)

        self.max_tag: int = int(kwargs.get("max tag", 0))
        if self.max_tag < 1:
            logger.warning(f"ParticleMonitor '{self.cmd_name}': max_tag={self.max_tag} < 1, "
                           f"no particles will be recorded.")

        cfg: Config = sim.cfg
        self.num_turn: int = cfg.num_turn

        # Resolve start_turn / end_turn
        self.start_turn: int = int(kwargs.get("start turn", 0))
        _end_turn_raw = int(kwargs.get("end turn", -1))
        if _end_turn_raw < 0 or _end_turn_raw > self.num_turn:
            self.end_turn: int = self.num_turn
        else:
            self.end_turn = _end_turn_raw

        if self.start_turn < 0:
            self.start_turn = 0
        if self.start_turn >= self.num_turn:
            logger.warning(f"ParticleMonitor '{self.cmd_name}': start_turn={self.start_turn} "
                           f">= num_turn={self.num_turn}, no turns will be recorded.")

        # Number of turns actually recorded
        self.num_record_turn: int = max(0, self.end_turn - self.start_turn)
        self._recorded_end = self.start_turn
        self._tables_written = False
        self._pending_samples = 0
        self._sample_count = 0
        self._file_created = False
        self._output_failed = False
        self._checkpoint_output_bytes = None
        self._checkpoint_snapshot_format = "hdf5"
        self._tfs_exported = False

        self.output_dir_particle: str = cfg.output_dir_particle
        self.output_hms: str = cfg.output_hms
        Path(self.output_dir_particle).mkdir(parents=True, exist_ok=True)
        filename = f"{self.output_hms}_beam{self.beam_id}_{self.cmd_name}_s{self.s:.3f}_particles.tfs"
        self.output_path = table_path(Path(self.output_dir_particle) / filename, self.output_format)
        self._owned_temporary_paths = set()
        self._storage_path = self._temporary_path(".h5") if self.output_format == "tfs" else self.output_path

        # Pre-allocate buffer using the same array backend as the beam
        beam: Beam = sim.beams[self.beam_id]
        xp = beam.particles.xp  # np or cp
        self._beam = beam
        self._coordinate_dtype = beam.particles.dtype
        self._buffer_turns = min(self.num_record_turn, self.write_interval_turns)
        self._sample_turns = np.zeros(max(1, self._buffer_turns), dtype=np.int64)

        if self.max_tag >= 1 and self.num_record_turn > 0:
            shape = (self._buffer_turns, self.max_tag, self._num_columns)
            self.buffer = xp.zeros(shape, dtype=xp.float64)
        else:
            # Edge case: nothing to record, use a tiny placeholder
            self.buffer = xp.zeros((1, 1, self._num_columns), dtype=xp.float64)

        self._first_index = (xp.empty(self.max_tag, dtype=xp.int32)
                             if self.max_tag >= 1 and self.num_record_turn > 0 and _is_cupy_array(beam.particles.x) else None)

        self._initial = {}
        if self.max_tag >= 1:
            self._initial = {name: np.full(self.max_tag, np.nan, dtype=self._coordinate_dtype) for name in ("x", "px", "y", "py", "z", "dp")}
            self._initial.update(particle_id=np.arange(1, self.max_tag + 1, dtype=np.int32),
                                 valid=np.zeros(self.max_tag, dtype=bool),
                                 injection_turn=np.full(self.max_tag, -1, dtype=np.int64))
            for name in ("referenceTime", "referenceBeta", "referenceMomentum"):
                self._initial[name] = np.full(self.max_tag, np.nan, dtype=np.float64)
            if not hasattr(beam, "_initial_particle_observers"):
                beam._initial_particle_observers = []
            beam._initial_particle_observers.append(self._capture_initial)

        super().__init__()

    def _capture_initial(self, destination, turn):
        """Capture actual injected coordinates before any following tracking command."""
        p = self._beam.particles
        ids = xp_get(p.tag[destination]).astype(np.int64, copy=False)
        selected = (ids >= 1) & (ids <= self.max_tag)
        destination = destination[p.xp.asarray(selected)]
        columns = ids[selected] - 1
        if not columns.size:
            return
        if np.any(self._initial["valid"][columns]):
            raise ValueError("ParticleMonitor encountered a reused injected particle identity")
        for name in ("x", "px", "y", "py", "z", "dp"):
            self._initial[name][columns] = xp_get(getattr(p, name)[destination])
        self._initial["valid"][columns] = True
        self._initial["injection_turn"][columns] = turn
        positions = xp_get(destination)
        for bunch in self._beam.bunches:
            selected_columns = columns[(positions >= bunch.start_idx) & (positions < bunch.end_idx)]
            for name, value in (("referenceTime", bunch.t0), ("referenceBeta", bunch.beta), ("referenceMomentum", bunch.p0)):
                self._initial[name][selected_columns] = value

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, "
                    f"MaxTag={self.max_tag:d}, TurnRange=[{self.start_turn:d},{self.end_turn:d}), "
                    f"NumRecordTurn={self.num_record_turn:d}, IncludeReference={self.include_reference}")
        set_normal_logging()

    def _record_one_turn(self, p, bunch, turn):
        """Fill buffer for one bunch at a given turn.

        Works for both CPU (numpy) and GPU (cupy) particle arrays.
        """
        if not self.start_turn <= turn < self.end_turn:
            return
        record_idx = self._pending_samples

        start = bunch.start_idx
        end = bunch.end_idx
        if end <= start:
            return
        tag_all = p.tag[start:end]
        is_gpu = _is_cupy_array(tag_all)
        nominal_center = bunch.harmonic_id * bunch.circum / bunch.harmonic_number

        if is_gpu:
            find_kernel, write_kernel = _get_monitor_kernels(p.dtype.str)
            threads = 256
            find_blocks = ((end - start) + threads - 1) // threads
            write_blocks = (self.max_tag + threads - 1) // threads
            reference = (bunch.t0, bunch.beta, bunch.p0) if self.include_reference else (0., 0., 0.)

            # Initialize to the exclusive end index.  A missing tag then
            # remains unwritten, matching the zero-initialized CPU buffer.
            self._first_index.fill(np.int32(end))
            find_kernel(
                (find_blocks, ),
                (threads, ),
                (p.tag, np.int32(start), np.int32(end), np.int32(self.max_tag), self._first_index),
            )
            write_kernel(
                (write_blocks, ),
                (threads, ),
                (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, p.lost_turn, p.lost_position, self._first_index, self.buffer, np.int32(end),
                 np.int32(self.max_tag), np.int32(record_idx), np.int32(self._num_columns), np.int32(turn), np.float64(nominal_center),
                 np.float64(reference[0]), np.float64(reference[1]), np.float64(reference[2])),
            )
            return

        abs_tag = np.abs(tag_all)
        indices = np.flatnonzero((abs_tag >= 1) & (abs_tag <= self.max_tag))
        columns = abs_tag[indices] - 1
        indices = indices + start
        # Particle identities are unique, including frozen loss records.
        rows = self.buffer[record_idx]
        rows[columns, 0] = turn
        for column, name in enumerate(("x", "px", "y", "py", "z", "dp", "tag", "lost_turn", "lost_position"), start=1):
            rows[columns, column] = getattr(p, name)[indices]
        rows[columns, 10] = nominal_center
        if self.include_reference:
            live = p.tag[indices] > 0
            for column, value in ((11, bunch.t0), (12, bunch.beta), (13, bunch.p0)):
                rows[columns, column] = np.where(live, value, np.nan)

    def execute_cpu(self, sim: Simulation):
        beam: Beam = sim.beams[self.beam_id]
        turn = sim.state.turn
        if self._output_failed:
            raise RuntimeError("ParticleMonitor cannot continue after a previous output failure")
        if self.max_tag < 1 or self.num_record_turn <= 0 or not self.start_turn <= turn < self.end_turn:
            return False
        if self._tables_written:
            return False
        for bunch in beam.bunches:
            self._record_one_turn(beam.particles, bunch, turn)
        self._recorded_end = turn + 1
        self._sample_count += 1
        self._sample_turns[self._pending_samples] = turn
        self._pending_samples += 1
        if self._pending_samples == self._buffer_turns or turn == self.end_turn - 1:
            self._write_samples()
        if turn == self.end_turn - 1:
            self._tables_written = True
        return True

    def execute_gpu(self, sim: Simulation):
        # Sampling dispatches to fused CUDA kernels; transfer only completed blocks.
        return self.execute_cpu(sim)

    def finalize(self, sim):
        """Flush existing samples; never sample a different lattice position."""
        if self._output_failed:
            return
        if self._pending_samples:
            self._write_samples()
        elif self._checkpoint_output_bytes is not None:
            self._restore_output_snapshot()
        if self.output_format == "tfs" and self._file_created and not self._tfs_exported:
            self._finish_tfs()
        elif self.output_format == "tfs" and self._tfs_exported:
            self._cleanup_tfs_temporary_files()
        self._tables_written = True

    def _write_samples(self):
        n_samples = self._pending_samples
        if not n_samples:
            return
        metadata = {
            "Name": "PASS Particle Monitor",
            "Time": get_current_time(),
            "Monitor": self.cmd_name,
            "S": self.s,
            "BeamId": self.beam_id,
            "MaxTag": self.max_tag,
            "IncludeReference": self.include_reference,
            "NumTurn": self._sample_count,
            "StartTurn": self.start_turn,
            "EndTurn": self._recorded_end,
            "RequestedEndTurn": self.end_turn,
            "Completed": self._sample_count == self.num_record_turn and self._recorded_end == self.end_turn,
            "CoordinateDefinition": "z=beta*c*(T-t)",
            "SamplingEvent": "monitor sequence position",
        }
        if self.include_reference:
            metadata.update(ReferenceEvent="element-exit", ReferenceAppliesTo="live particles only; NaN for loss records")
        try:
            self._restore_output_snapshot()
            values = xp_get(self.buffer[:n_samples])
            append_particle_samples(self._storage_path,
                                    values,
                                    self._column_names,
                                    self._sample_turns[:n_samples],
                                    self._initial,
                                    metadata,
                                    coordinate_dtype=self._coordinate_dtype,
                                    chunk_turns=self._buffer_turns,
                                    output_format="hdf5" if self.output_format == "tfs" else self.output_format,
                                    create=not self._file_created)
        except BaseException:
            # A write may have committed before reporting failure. Retrying can
            # duplicate samples or obscure the original error with FileExistsError.
            self._output_failed = True
            raise
        self._file_created = True
        self._pending_samples = 0
        self.buffer.fill(0)

    def state_dict(self):
        """Own both pending samples and committed output for a restart."""
        if self._output_failed:
            raise ValueError("Cannot checkpoint a ParticleMonitor after an output failure")
        output_bytes = self._checkpoint_output_bytes
        snapshot_format = self._checkpoint_snapshot_format
        if output_bytes is None and self._tfs_exported:
            output_bytes = self.output_path.read_bytes()
            snapshot_format = "tfs"
        elif output_bytes is None and self._file_created:
            output_bytes = self._storage_path.read_bytes()
            snapshot_format = "hdf5"
        return {
            "format": "PASS-particle-monitor-state-2",
            "output_format": self.output_format,
            "snapshot_format": snapshot_format,
            "buffer": np.array(xp_get(self.buffer), copy=True),
            "sample_turns": self._sample_turns.copy(),
            "initial": {
                name: value.copy()
                for name, value in self._initial.items()
            },
            "recorded_end": self._recorded_end,
            "pending_samples": self._pending_samples,
            "sample_count": self._sample_count,
            "output_bytes": output_bytes,
            "output_sha256": None if output_bytes is None else hashlib.sha256(output_bytes).hexdigest(),
        }

    def stage_checkpoint(self, data, next_turn, xp):
        """Validate detached checkpoint state without touching output files."""
        if data.get("format") != "PASS-particle-monitor-state-2" or data.get("output_format") != self.output_format:
            raise ValueError("ParticleMonitor checkpoint format does not match single-file output")
        buffer = np.asarray(data["buffer"])
        turns = np.asarray(data["sample_turns"])
        if buffer.shape != self.buffer.shape or buffer.dtype != self.buffer.dtype:
            raise ValueError("ParticleMonitor checkpoint buffer does not match")
        if turns.shape != self._sample_turns.shape or turns.dtype != self._sample_turns.dtype:
            raise ValueError("ParticleMonitor checkpoint turn buffer does not match")
        if set(data["initial"]) != set(self._initial):
            raise ValueError("ParticleMonitor checkpoint initial fields do not match")
        initial = {}
        for name, template in self._initial.items():
            value = np.asarray(data["initial"][name])
            if value.shape != template.shape or value.dtype != template.dtype:
                raise ValueError(f"ParticleMonitor checkpoint initial field {name} does not match")
            initial[name] = value.copy()
        expected_end = (max(self.start_turn, min(next_turn, self.end_turn)) if self.max_tag >= 1 and self.num_record_turn > 0 else self.start_turn)
        pending, count = data["pending_samples"], data["sample_count"]
        if type(data["recorded_end"]) is not int or data["recorded_end"] != expected_end:
            raise ValueError("ParticleMonitor checkpoint has invalid recorded turns")
        if type(count) is not int or count != expected_end - self.start_turn:
            raise ValueError("ParticleMonitor checkpoint has invalid sample count")
        if type(pending) is not int or not 0 <= pending <= min(count, self._buffer_turns):
            raise ValueError("ParticleMonitor checkpoint has invalid pending sample count")
        if not np.array_equal(turns[:pending], np.arange(expected_end - pending, expected_end)):
            raise ValueError("ParticleMonitor checkpoint pending turns are inconsistent")
        output_bytes = data["output_bytes"]
        if (output_bytes is not None) != (count > pending):
            raise ValueError("ParticleMonitor checkpoint output snapshot does not match committed samples")
        if output_bytes is not None and (not isinstance(output_bytes, bytes) or hashlib.sha256(output_bytes).hexdigest() != data["output_sha256"]):
            raise ValueError("ParticleMonitor checkpoint output snapshot is damaged")
        snapshot_format = data.get("snapshot_format")
        if snapshot_format not in {"hdf5", "tfs"} or snapshot_format == "tfs" and self.output_format != "tfs":
            raise ValueError("ParticleMonitor checkpoint snapshot encoding is inconsistent")
        return {
            "buffer": xp.asarray(buffer.copy()),
            "_sample_turns": turns.copy(),
            "_initial": initial,
            "_recorded_end": expected_end,
            "_pending_samples": pending,
            "_sample_count": count,
            "_file_created": False,
            "_tables_written": expected_end == self.end_turn,
            "_output_failed": False,
            "_checkpoint_output_bytes": output_bytes,
            "_checkpoint_snapshot_format": snapshot_format,
            "_tfs_exported": False,
            "_restoring_checkpoint": True,
        }

    def _restore_output_snapshot(self):
        """Create a fresh continuation file, preserving any previous output."""
        if not getattr(self, "_restoring_checkpoint", False):
            return
        path = self.output_path
        index = 1
        while path.exists():
            path = self.output_path.with_name(f"{self.output_path.stem}_resume{index}{self.output_path.suffix}")
            index += 1
        self.output_path = path
        self._storage_path = self._temporary_path(".h5") if self.output_format == "tfs" else self.output_path
        restored_tfs_path = None
        if self._checkpoint_output_bytes is not None:
            try:
                snapshot_path = self._temporary_path(".tfs.partial") if self._checkpoint_snapshot_format == "tfs" else self._storage_path
                with snapshot_path.open("xb") as stream:
                    stream.write(self._checkpoint_output_bytes)
                    stream.flush()
                if self._checkpoint_snapshot_format == "tfs":
                    # The current end may already include newly buffered turns.
                    # Publish bytes directly only when no samples need appending.
                    if self._recorded_end == self.end_turn and not self._pending_samples:
                        self._publish_tfs(snapshot_path)
                    else:
                        restore_particle_samples_tfs(snapshot_path,
                                                     self._storage_path,
                                                     self._column_names,
                                                     self._initial,
                                                     coordinate_dtype=self._coordinate_dtype,
                                                     chunk_turns=self._buffer_turns)
                        self._file_created = True
                    restored_tfs_path = snapshot_path
                else:
                    self._file_created = True
            except BaseException:
                if self._tfs_exported:
                    self._checkpoint_output_bytes = None
                    self._checkpoint_snapshot_format = "hdf5"
                    self._restoring_checkpoint = False
                else:
                    self._output_failed = True
                raise
            self._checkpoint_output_bytes = None
            self._checkpoint_snapshot_format = "hdf5"
        self._restoring_checkpoint = False
        if self._tfs_exported:
            self._cleanup_tfs_temporary_files()
        elif restored_tfs_path is not None:
            try:
                self._remove_temporary_file(restored_tfs_path)
            except OSError:
                logger.warning("ParticleMonitor restored its HDF5 history; temporary TFS cleanup deferred until finalization: %s", restored_tfs_path)

    def _temporary_path(self, suffix):
        """Reserve a unique name beneath this monitor's output directory."""
        path = Path(self.output_dir_particle).resolve() / f".pass_pm_{uuid.uuid4().hex}{suffix}"
        self._owned_temporary_paths.add(path)
        return path

    def _remove_temporary_file(self, path):
        """Remove only this run's explicitly owned, unchanged temporary path."""
        path = Path(path)
        if path not in self._owned_temporary_paths or path.resolve() != path or path.parent != Path(self.output_dir_particle).resolve():
            raise ValueError("Refusing to remove an unowned ParticleMonitor temporary file")
        path.unlink()
        self._owned_temporary_paths.remove(path)

    def _finish_tfs(self):
        """Publish a complete standard TFS, then remove this run's HDF5 spool."""
        staging_path = self._temporary_path(".tfs.partial")
        try:
            export_particle_samples_tfs(self._storage_path, staging_path)
            self._publish_tfs(staging_path)
        except BaseException:
            if self._tfs_exported:
                logger.exception("ParticleMonitor TFS is published at %s; calling finalize again retries temporary cleanup.", self.output_path)
            else:
                logger.exception("ParticleMonitor TFS conversion failed; HDF5 history retained at %s. Calling finalize again retries conversion.",
                                 self._storage_path)
            raise
        self._cleanup_tfs_temporary_files()

    def _publish_tfs(self, staging_path):
        """Publish without overwrite and recognize an interrupted successful link."""
        try:
            os.link(staging_path, self.output_path)
            self._tfs_exported = True
        except BaseException:
            # An exception can arrive after the OS has created the hard link.
            # File identity proves ownership without accepting an unrelated file.
            try:
                self._tfs_exported = self.output_path.samefile(staging_path)
            except OSError:
                self._tfs_exported = False
            raise

    def _cleanup_tfs_temporary_files(self):
        """Retry owned-file cleanup without affecting the published TFS."""
        failures = []
        for path in tuple(self._owned_temporary_paths):
            if not path.exists():
                self._owned_temporary_paths.discard(path)
                continue
            try:
                self._remove_temporary_file(path)
            except OSError as exc:
                failures.append(exc)
        self._file_created = self._storage_path.exists()
        if failures:
            logger.error("ParticleMonitor TFS is complete at %s; temporary cleanup failed. Calling finalize again retries cleanup.", self.output_path)
            raise failures[0]


# Backend-agnostic helpers (work for both numpy and cupy arrays)


def xp_abs(arr):
    """Absolute value that works for numpy and cupy arrays."""
    if _is_cupy_array(arr):
        import cupy as cp
        return cp.abs(arr)
    return np.abs(arr)


def xp_where(condition):
    """np.where / cp.where dispatch."""
    if _is_cupy_array(condition):
        import cupy as cp
        return cp.where(condition)
    return np.where(condition)


def xp_get(arr):
    """Copy GPU array to CPU; pass through CPU array."""
    if _is_cupy_array(arr):
        return arr.get()
    return arr


def _is_cupy_array(arr):
    """Detect a CuPy array without importing the optional CuPy package."""
    return arr.__class__.__module__.startswith("cupy")


# GPU-only implementation is kept below the monitor class so the public
# ParticleMonitor lifecycle remains easy to inspect.  The class methods resolve
# these names when called, so their placement does not affect behavior.
_MONITOR_SOURCE = r'''
#if PASS_USE_FLOAT
using pass_real_t = float;
#else
using pass_real_t = double;
#endif

extern "C" __global__ void particle_monitor_find(
    const int* tag,
    int start,
    int end,
    int max_tag,
    int* first
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i >= end)
        return;
    int value = tag[i];
    int abs_value = value < 0 ? -value : value;
    if (abs_value >= 1 && abs_value <= max_tag)
        atomicMin(&first[abs_value - 1], i);
}

extern "C" __global__ void particle_monitor_write(
    const pass_real_t* x,
    const pass_real_t* px,
    const pass_real_t* y,
    const pass_real_t* py,
    const pass_real_t* z,
    const pass_real_t* dp,
    const int* tag,
    const int* lost_turn,
    const float* lost_position,
    const int* first,
    double* out,
    int end,
    int max_tag,
    int record_idx,
    int num_columns,
    int turn,
    double z_center,
    double t0,
    double beta,
    double p0
) {
    int tag_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (tag_index >= max_tag)
        return;
    int i = first[tag_index];
    if (i < 0 || i >= end)
        return;
    size_t row = (size_t)record_idx * (size_t)max_tag + (size_t)tag_index;
    size_t base = row * (size_t)num_columns;
    out[base + 0] = (double)turn;
    out[base + 1] = (double)x[i];
    out[base + 2] = (double)px[i];
    out[base + 3] = (double)y[i];
    out[base + 4] = (double)py[i];
    out[base + 5] = (double)z[i];
    out[base + 6] = (double)dp[i];
    out[base + 7] = (double)tag[i];
    out[base + 8] = (double)lost_turn[i];
    out[base + 9] = (double)lost_position[i];
    out[base + 10] = z_center;
    if (num_columns > 11) {
        out[base + 11] = tag[i] > 0 ? t0 : nan("");
        out[base + 12] = tag[i] > 0 ? beta : nan("");
        out[base + 13] = tag[i] > 0 ? p0 : nan("");
    }
}
'''


@lru_cache(maxsize=None)
def _get_monitor_kernels(dtype):
    """Compile monitor indexing/writing kernels once per particle precision."""
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU ParticleMonitor requires the optional 'cuda' dependencies.") from exc

    dtype = np.dtype(dtype)
    return (
        cp.RawKernel(
            _MONITOR_SOURCE,
            "particle_monitor_find",
            options=("--std=c++14", f"-DPASS_USE_FLOAT={int(dtype == np.dtype(np.float32))}"),
        ),
        cp.RawKernel(
            _MONITOR_SOURCE,
            "particle_monitor_write",
            options=("--std=c++14", f"-DPASS_USE_FLOAT={int(dtype == np.dtype(np.float32))}"),
        ),
    )
