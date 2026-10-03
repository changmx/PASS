"""Capture slow-extraction events at a tracked plane and retire their ring tags."""

import logging
from pathlib import Path
import re
from types import MappingProxyType
from uuid import uuid4

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
        }
        for field, header in (("end_turn", "EndTurn"), ("start_time", "StartTime"), ("end_time", "EndTime")):
            value = getattr(self.settings, field)
            if value is not None:
                self._headers[header] = value
        self._pending = []
        self._pending_rows = 0
        self._initialized = False
        self._failed = False
        self._finalized = False
        self.last_turn = -1
        self.batch_serial = 0
        self.reference_times = ()
        self._empty_events = self._make_empty_events(sim.beams[beam_id].particles.dtype)
        self.last_events = self._empty_events
        self.total_extracted = 0
        self.total_real = 0.0
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
        columns = {
            "particle_id": convert_array(p.tag[indices], np).copy(),
            "beam_id": np.full(n_particles, self.beam_id, np.int32),
            "bunch_id": np.full(n_particles, bunch.bunch_id, np.int32),
            "turn": np.full(n_particles, turn, np.int64),
            "s": np.full(n_particles, self.s, np.float64),
            "time": np.empty(n_particles, np.float64),
        }
        for name in ("x", "px", "y", "py", "z", "dp"):
            columns[name] = convert_array(getattr(p, name)[indices], np).copy()
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

    def _execute(self, sim):
        if self._failed or self._finalized:
            raise RuntimeError("Cannot execute a failed or finalized SlowExtraction")
        turn = int(sim.state.turn)
        if turn < self.last_turn:
            raise ValueError("SlowExtraction cannot reuse event history after rewinding turns")
        if turn == self.last_turn:
            return False
        beam = sim.beams[self.beam_id]
        p = beam.particles
        self.last_turn = turn
        self.batch_serial += 1
        self.last_events = self._empty_events
        self.reference_times = tuple(float(bunch.t0) for bunch in beam.bunches)
        settings = self.settings
        if turn < settings.start_turn or (settings.end_turn is not None and turn >= settings.end_turn):
            return False

        batches = []
        selections = []
        for bunch in beam.bunches:
            start, end = bunch.start_idx, bunch.end_idx
            if start == end:
                continue
            if not np.isfinite(bunch.t0) or not np.isfinite(bunch.beta) or bunch.beta <= 0:
                raise ValueError("SlowExtraction requires finite reference time and positive reference beta")
            # Float64 geometry and time keep cuts independent of particle storage precision.
            u = p.x[start:end].astype(p.xp.float64) * self._cos_tilt - p.y[start:end].astype(p.xp.float64) * self._sin_tilt
            outside = u > settings.position if settings.side == "positive" else u < settings.position
            selected = (p.tag[start:end] > 0) & outside
            if settings.start_time is not None or settings.end_time is not None:
                times = bunch.t0 - p.z[start:end].astype(p.xp.float64) / (bunch.beta * const.c)
                if settings.start_time is not None:
                    selected &= times >= settings.start_time
                if settings.end_time is not None:
                    selected &= times < settings.end_time
            indices = p.xp.flatnonzero(selected) + start
            if len(indices):
                batches.append(self._capture_events(p, bunch, indices, turn))
                selections.append(indices)
        if not batches:
            return True
        columns = {name: np.concatenate([batch[name] for batch in batches]) for name in self._empty_events}
        for values in columns.values():
            values.flags.writeable = False
        events = MappingProxyType(columns)
        self._pending.append(events)
        self._pending_rows += len(events["particle_id"])
        if self._pending_rows >= settings.buffer_size:
            # A failed synchronous flush must not retire the current particle batch.
            self._flush_events()
        for indices in selections:
            p.lost_turn[indices] = turn
            p.lost_position[indices] = self.s
            p.tag[indices] = -p.tag[indices]
        self.last_events = events
        self.total_extracted += len(events["particle_id"])
        self.total_real += float(np.sum(events["macro_weight"]))
        return True

    def execute_cpu(self, sim):
        return self._execute(sim)

    def execute_gpu(self, sim):
        return self._execute(sim)

    def _flush_events(self):
        if self._failed:
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
            raise

    def finalize(self, sim):
        if self._failed or self._finalized:
            return
        self._flush_events()
        self._finalized = True

    def print(self):
        logger.info("SlowExtraction %s: S=%g m, %s side of u=%g m", self.cmd_name, self.s, self.settings.side, self.settings.position)
