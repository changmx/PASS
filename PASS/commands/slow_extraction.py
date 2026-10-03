"""Capture slow-extraction events at a tracked plane and retire their ring tags."""

import logging
from pathlib import Path
import re
from types import MappingProxyType
from uuid import uuid4

import h5py
import numpy as np

from PASS import __version__
from PASS.commands.command import Command
from PASS.core.particle import convert_array
from PASS.para.schema.slow_extraction import SlowExtractionItem
from PASS.utils.constants import const
from PASS.utils.table_io import append_table, write_table

logger = logging.getLogger(__name__)


def _parse_settings(model, command_kwargs):
    """Accept engine lowercase aliases and Python field names consistently."""
    aliases = {}
    for name, field in model.model_fields.items():
        aliases[name.lower()] = name
        aliases[(field.alias or name).lower()] = name
    kwargs = {aliases.get(str(k).lower(), k): v for k, v in command_kwargs.items() if str(k).lower() != "name"}
    return model.model_validate(kwargs)


@Command.register("SlowExtraction")
class SlowExtraction(Command):
    """Select live particles in a single half-plane, recording their exit state.

    Coordinates are copied before tags change. Events remain in an owned host
    buffer until its row limit or finalization; they are never sampled. Disk
    errors stop tracking, and failed appends are not retried automatically.
    """

    def __init__(self, beam_id, sim, **command_kwargs):
        self.settings = _parse_settings(SlowExtractionItem, command_kwargs)
        kwargs = {str(k).lower(): v for k, v in command_kwargs.items()}
        self.beam_id = beam_id
        self.cmd_name = str(kwargs["name"])
        self.cmd_type = self.__class__.__name__
        self.s = self.settings.s
        self.output_format = self.settings.output_format
        self._cos_tilt = np.cos(self.settings.tilt)
        self._sin_tilt = np.sin(self.settings.tilt)

        # Like InjectionState, extraction state is owned by its command, not core.
        if not hasattr(sim, "_slow_extraction_sources"):
            sim._slow_extraction_sources = {}
            sim._slow_extraction_run_id = uuid4().hex
        key = (beam_id, self.cmd_name.lower())
        if key in sim._slow_extraction_sources:
            raise ValueError(f"Duplicate SlowExtraction source {self.cmd_name!r} for beam {beam_id}")
        self.run_id = sim._slow_extraction_run_id
        slug = re.sub(r"[^\w.-]+", "_", self.cmd_name).strip(".") or "extraction"
        # A source UUID also prevents sanitized command names from colliding.
        output_dir = Path(sim.cfg.output_dir_dist)
        if not getattr(sim.cfg, "flat_output", False):
            output_dir /= "slow_extraction"
        self.output_path = output_dir / f"{self.run_id}_beam{beam_id}_{slug}_{uuid4().hex[:8]}_events.h5"
        self._headers = {
            "Command": self.cmd_type,
            "Source": self.cmd_name,
            "RunId": self.run_id,
            "BeamId": beam_id,
            "S": self.s,
            "PASSVersion": __version__,
            "CoordinateDefinition": "z=beta*c*(reference_time-time); px=Px/P0; py=Py/P0; dp=(P-P0)/P0",
            "TimeUnit": "s",
            "LengthUnit": "m",
            "ReferenceMomentumUnit": "eV/c in the bunch reference convention (per nucleon for ions)",
            "RestEnergyUnit": "eV in the bunch reference convention (per nucleon for ions)",
            "Identity": "(RunId, beam_id, particle_id); particle_id=abs(tag)",
            "TagMeaning": "Extracted particles have negative ring tags; event membership distinguishes them from physical losses",
            "Side": self.settings.side,
            "Position": self.settings.position,
            "Tilt": self.settings.tilt,
            "StartTurn": self.settings.start_turn,
            "SourceStatus": "open",
            "SourceStatusDefinition": "finalized describes this source only; aborted-diagnostic rows may include incomplete retirement",
        }
        for field, header in (("end_turn", "EndTurn"), ("start_time", "StartTime"), ("end_time", "EndTime")):
            value = getattr(self.settings, field)
            if value is not None:
                self._headers[header] = value
        self._pending = []
        self._pending_rows = 0
        self._initialized = False
        self._failed = False
        self._output_failed = False
        self._finalized = False
        self.last_turn = -1
        self.batch_serial = 0
        self.reference_times = ()
        self._empty_events = self._make_empty_events(sim.beams[beam_id].particles.dtype)
        self.last_events = self._empty_events
        self.total_extracted = 0
        self.total_real = 0.0
        self._gpu_workspaces = {}
        sim._slow_extraction_sources[key] = self

    @staticmethod
    def _make_empty_events(dtype):
        columns = {
            "particle_id": np.empty(0, np.int32),
            "beam_id": np.empty(0, np.int32),
            "bunch_id": np.empty(0, np.int32),
            "turn": np.empty(0, np.int64),
            "s": np.empty(0, np.float64),
            "time": np.empty(0, np.float64),
        }
        for name in ("x", "px", "y", "py", "z", "dp"):
            columns[name] = np.empty(0, dtype)
        for name in ("reference_time", "reference_beta", "reference_momentum", "macro_weight"):
            columns[name] = np.empty(0, np.float64)
        for name in ("charge_number", "proton_number", "neutron_number"):
            columns[name] = np.empty(0, np.int32)
        columns["rest_energy"] = np.empty(0, np.float64)
        for values in columns.values():
            values.flags.writeable = False
        return MappingProxyType(columns)

    def _capture_events(self, p, bunch, indices, turn):
        n_particles = len(indices)
        if p.xp is np:
            particle_ids = convert_array(p.tag[indices], np).copy()
            coordinates = {name: convert_array(getattr(p, name)[indices], np).copy() for name in ("x", "px", "y", "py", "z", "dp")}
        else:
            workspace = self._get_gpu_workspace(p)
            if n_particles > workspace["pack_capacity"]:
                capacity = max(n_particles, 2 * workspace["pack_capacity"], 256)
                workspace["pack"] = p.xp.empty(7 * capacity, dtype=p.xp.float64)
                workspace["pack_capacity"] = capacity
            packed = workspace["pack"][:7 * n_particles].reshape(7, n_particles)
            workspace["pack_kernel"](((n_particles + 255) // 256, ), (256, ),
                                     (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, indices, np.int64(bunch.start_idx), np.int64(n_particles), packed))
            # Float64 exactly represents every int32 ID and both particle dtypes.
            # The packed slab is contiguous even when capacity exceeds this batch.
            captured = convert_array(packed, np)
            particle_ids = captured[0].astype(np.int32)
            coordinates = {name: captured[index + 1].astype(p.dtype, copy=False) for index, name in enumerate(("x", "px", "y", "py", "z", "dp"))}
        columns = {
            "particle_id": particle_ids,
            "beam_id": np.full(n_particles, self.beam_id, np.int32),
            "bunch_id": np.full(n_particles, bunch.bunch_id, np.int32),
            "turn": np.full(n_particles, turn, np.int64),
            "s": np.full(n_particles, self.s, np.float64),
            "time": np.empty(n_particles, np.float64),
        }
        columns.update(coordinates)
        columns["time"] = bunch.t0 - columns["z"].astype(np.float64) / (bunch.beta * const.c)
        for name, value in (("reference_time", bunch.t0), ("reference_beta", bunch.beta), ("reference_momentum", bunch.p0), ("macro_weight",
                                                                                                                             bunch.ratio)):
            columns[name] = np.full(n_particles, value, np.float64)
        for name, value in (("charge_number", bunch.num_charge), ("proton_number", bunch.num_proton), ("neutron_number", bunch.num_neutron)):
            columns[name] = np.full(n_particles, value, np.int32)
        columns["rest_energy"] = np.full(n_particles, bunch.m0, np.float64)
        if any(not np.all(np.isfinite(values)) for values in columns.values()):
            raise ValueError(f"SlowExtraction {self.cmd_name!r}: nonfinite extracted state")
        return columns

    def _get_gpu_workspace(self, p):
        key = (p.x.device.id, p.dtype.str)
        workspace = self._gpu_workspaces.get(key)
        if workspace is None:
            module = p.xp.RawModule(code=_SLOW_EXTRACTION_GPU_SOURCE,
                                    options=("--fmad=false", f"-DPASS_USE_FLOAT={int(p.dtype == np.dtype(np.float32))}"))
            workspace = {
                "module": module,
                "select_kernel": module.get_function("select_slow_extraction"),
                "pack_kernel": module.get_function("pack_slow_extraction"),
                "retire_kernel": module.get_function("retire_slow_extraction"),
                "mask": None,
                "mask_capacity": 0,
                "pack": None,
                "pack_capacity": 0,
            }
            self._gpu_workspaces[key] = workspace
        return workspace

    def _select_gpu(self, p, bunch):
        n_particles = bunch.end_idx - bunch.start_idx
        workspace = self._get_gpu_workspace(p)
        if n_particles > workspace["mask_capacity"]:
            capacity = max(n_particles, 2 * workspace["mask_capacity"], 256)
            workspace["mask"] = p.xp.empty(capacity, dtype=p.xp.bool_)
            workspace["mask_capacity"] = capacity
        selected = workspace["mask"][:n_particles]
        settings = self.settings
        workspace["select_kernel"](
            ((n_particles + 255) // 256, ), (256, ),
            (p.x, p.y, p.z, p.tag, np.int64(bunch.start_idx), np.int64(n_particles), np.float64(self._cos_tilt), np.float64(
                self._sin_tilt), np.float64(settings.position), np.int32(settings.side == "positive"), np.float64(
                    bunch.t0), np.float64(bunch.beta * const.c), np.int32(settings.start_time is not None), np.float64(
                        settings.start_time or 0.0), np.int32(settings.end_time is not None), np.float64(settings.end_time or 0.0), selected))
        # Stable local indices retain the CPU event order. The selected count
        # must reach the host because each invocation publishes owned events.
        return p.xp.flatnonzero(selected)

    def _retire_gpu(self, p, indices, start, turn):
        n_particles = len(indices)
        self._get_gpu_workspace(p)["retire_kernel"](
            ((n_particles + 255) // 256, ), (256, ),
            (p.tag, p.lost_turn, p.lost_position, indices, np.int64(start), np.int64(n_particles), np.int32(turn), np.float32(self.s)))

    def _execute(self, sim):
        if self._failed or self._finalized:
            raise RuntimeError("Cannot execute a failed or finalized SlowExtraction")
        turn = int(sim.state.turn)
        if turn < self.last_turn:
            raise ValueError("SlowExtraction cannot reuse event history after rewinding turns")
        if turn == self.last_turn:
            return False
        try:
            beam = sim.beams[self.beam_id]
            reference_times = tuple(float(bunch.t0) for bunch in beam.bunches)
            if beam.particles.xp is np:
                executed, events = self._capture_turn(beam, turn)
            else:
                with beam.particles.x.device:
                    executed, events = self._capture_turn(beam, turn)
        except BaseException:
            self._failed = True
            raise
        # Publish only complete captures; monitors must never consume a failed batch.
        self.last_turn = turn
        self.batch_serial += 1
        self.last_events = events
        self.reference_times = reference_times
        self.total_extracted += len(events["particle_id"])
        self.total_real += float(np.sum(events["macro_weight"]))
        return executed

    def _capture_turn(self, beam, turn):
        p = beam.particles
        settings = self.settings
        if turn < settings.start_turn or (settings.end_turn is not None and turn >= settings.end_turn):
            return False, self._empty_events

        batches = []
        selections = []
        for bunch in beam.bunches:
            start, end = bunch.start_idx, bunch.end_idx
            if start == end:
                continue
            if not np.isfinite(bunch.t0) or not np.isfinite(bunch.beta) or bunch.beta <= 0:
                raise ValueError("SlowExtraction requires finite reference time and positive reference beta")
            if p.xp is np:
                # Float64 cuts are independent of particle storage precision.
                u = p.x[start:end].astype(np.float64) * self._cos_tilt - p.y[start:end].astype(np.float64) * self._sin_tilt
                outside = u > settings.position if settings.side == "positive" else u < settings.position
                selected = (p.tag[start:end] > 0) & outside
                if settings.start_time is not None or settings.end_time is not None:
                    times = bunch.t0 - p.z[start:end].astype(np.float64) / (bunch.beta * const.c)
                    if settings.start_time is not None:
                        selected &= times >= settings.start_time
                    if settings.end_time is not None:
                        selected &= times < settings.end_time
                indices = np.flatnonzero(selected) + start
            else:
                indices = self._select_gpu(p, bunch)
            if len(indices):
                batches.append(self._capture_events(p, bunch, indices, turn))
                selections.append((indices, start))
        if not batches:
            return True, self._empty_events
        columns = batches[0] if len(batches) == 1 else {name: np.concatenate([batch[name] for batch in batches]) for name in self._empty_events}
        for values in columns.values():
            values.flags.writeable = False
        events = MappingProxyType(columns)
        self._pending.append(events)
        self._pending_rows += len(events["particle_id"])
        # Retain immutable evidence even when retirement is interrupted or ambiguous.
        # Only _execute publishes a successfully completed batch to monitors.
        if self._pending_rows >= settings.buffer_size:
            # A failed synchronous flush must not retire the current particle batch.
            self._flush_events()
        for indices, start in selections:
            if p.xp is np:
                p.lost_turn[indices] = turn
                p.lost_position[indices] = self.s
                p.tag[indices] = -p.tag[indices]
            else:
                self._retire_gpu(p, indices, start, turn)
        if p.xp is not np:
            p.xp.cuda.get_current_stream().synchronize()
        return True, events

    def execute_cpu(self, sim):
        return self._execute(sim)

    def execute_gpu(self, sim):
        return self._execute(sim)

    def _flush_events(self):
        if self._output_failed:
            raise RuntimeError("SlowExtraction output previously failed; refusing an ambiguous append retry")
        try:
            if not self._initialized:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                if self.output_path.exists():
                    raise FileExistsError(self.output_path)
            if self._pending:
                columns = {name: np.concatenate([batch[name] for batch in self._pending]) for name in self._empty_events}
                append_table(self.output_path,
                             columns,
                             self._headers,
                             chunk_rows=min(self.settings.buffer_size, 65536),
                             output_format=self.output_format)
                self._initialized = True
                self._pending.clear()
                self._pending_rows = 0
            elif not self._initialized:
                write_table(self.output_path, self._empty_events, self._headers, output_format=self.output_format)
                self._initialized = True
        except BaseException:
            self._failed = True
            self._output_failed = True
            raise

    def finalize(self, sim):
        if self._output_failed or self._finalized:
            return
        self._flush_events()
        try:
            with h5py.File(self.output_path, "r+") as stream:
                stream.attrs["SuccessfulExtracted"] = self.total_extracted
                stream.attrs["SuccessfulRealExtracted"] = self.total_real
                stream.attrs["SuccessfulBatchSerial"] = self.batch_serial
                stream.attrs["LastSuccessfulTurn"] = self.last_turn
                stream.attrs["SourceStatus"] = "aborted-diagnostic" if self._failed else "finalized"
        except BaseException:
            self._failed = True
            self._output_failed = True
            raise
        self._finalized = True

    def print(self):
        logger.info("SlowExtraction %s: S=%g m, %s side of u=%g m", self.cmd_name, self.s, self.settings.side, self.settings.position)


_SLOW_EXTRACTION_GPU_SOURCE = r'''
#ifndef PASS_USE_FLOAT
#define PASS_USE_FLOAT 0
#endif
#if PASS_USE_FLOAT
using pass_real_t = float;
#else
using pass_real_t = double;
#endif

extern "C" __global__ void select_slow_extraction(
    const pass_real_t* x,
    const pass_real_t* y,
    const pass_real_t* z,
    const int* tag,
    long long start,
    long long n_particles,
    double cos_tilt,
    double sin_tilt,
    double position,
    int positive_side,
    double reference_time,
    double beta_c,
    int has_start_time,
    double start_time,
    int has_end_time,
    double end_time,
    unsigned char* selected
) {
    long long local = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (local >= n_particles) {
        return;
    }
    long long i = start + local;
    double u = (double)x[i] * cos_tilt - (double)y[i] * sin_tilt;
    bool active = tag[i] > 0 && (positive_side ? u > position : u < position);
    if (active && (has_start_time || has_end_time)) {
        double time = reference_time - (double)z[i] / beta_c;
        active = (!has_start_time || time >= start_time) && (!has_end_time || time < end_time);
    }
    selected[local] = active;
}

extern "C" __global__ void pack_slow_extraction(
    const pass_real_t* x,
    const pass_real_t* px,
    const pass_real_t* y,
    const pass_real_t* py,
    const pass_real_t* z,
    const pass_real_t* dp,
    const int* tag,
    const long long* indices,
    long long start,
    long long n_particles,
    double* packed
) {
    long long row = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (row >= n_particles) {
        return;
    }
    long long i = start + indices[row];
    packed[row] = (double)tag[i];
    packed[n_particles + row] = (double)x[i];
    packed[2 * n_particles + row] = (double)px[i];
    packed[3 * n_particles + row] = (double)y[i];
    packed[4 * n_particles + row] = (double)py[i];
    packed[5 * n_particles + row] = (double)z[i];
    packed[6 * n_particles + row] = (double)dp[i];
}

extern "C" __global__ void retire_slow_extraction(
    int* tag,
    int* lost_turn,
    float* lost_position,
    const long long* indices,
    long long start,
    long long n_particles,
    int turn,
    float position
) {
    long long row = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (row >= n_particles) {
        return;
    }
    long long i = start + indices[row];
    lost_turn[i] = turn;
    lost_position[i] = position;
    tag[i] = -tag[i];
}
'''
