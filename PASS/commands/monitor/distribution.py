"""Full-particle distribution snapshots."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tfs

from PASS import __version__
from PASS.commands.command import Command
from PASS.core.beam import Beam
from PASS.core.config import Config
from PASS.core.simulation import Simulation
from PASS.utils.helper import get_current_time
from PASS.utils.logger import set_normal_logging, set_simple_logging

logger = logging.getLogger(__name__)

_DATA_FIELDS = (
    "x",
    "px",
    "y",
    "py",
    "z",
    "dp",
    "tag",
    "lost_turn",
    "lost_position",
)


def is_turn_selected(turn: int, selected_turns: bytearray) -> bool:
    """Return whether *turn* is selected by a precompiled turn table."""
    return 0 <= turn < len(selected_turns) and bool(selected_turns[turn])


@Command.register("distmonitor")
class DistMonitor(Command):
    """Save born particles in every bunch at selected turns.

    ``Save turns`` is compiled once into a bytearray.  A snapshot contains
    every born particle, including lost particles, as one TFS or HDF5 file
    per bunch and turn. Reserved particles are counted in NumPending.
    """

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {str(k).lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = float(kwargs["s (m)"])
        self.cmd_type = self.__class__.__name__
        self.cmd_name = str(kwargs["name"])
        self.include_injection_metadata = kwargs.get("include injection metadata", False)
        self.output_format = kwargs.get("output format", "tfs")
        if self.output_format not in {"tfs", "hdf5"}:
            raise ValueError("DistMonitor Output format must be tfs or hdf5")

        cfg: Config = sim.cfg
        self.num_turn = int(cfg.num_turn)
        self._selected_turns = bytearray(self.num_turn)
        for turn_range in kwargs.get("save turns", []):
            if not isinstance(turn_range, (list, tuple)):
                raise ValueError(f"DistMonitor '{self.cmd_name}': each Save turns item "
                                 "must be a list of one or three integers")
            if len(turn_range) == 1:
                start, end, step = turn_range[0], turn_range[0], 1
            elif len(turn_range) == 3:
                start, end, step = turn_range
            else:
                raise ValueError(f"DistMonitor '{self.cmd_name}': Save turns items must "
                                 "be [turn] or [start, end, step]")

            if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (start, end, step)):
                raise ValueError(f"DistMonitor '{self.cmd_name}': Save turns values must "
                                 "be integers")
            start, end, step = int(start), int(end), int(step)
            if step <= 0:
                raise ValueError(f"DistMonitor '{self.cmd_name}': Save turns step must be > 0")

            original_start, original_end = start, end
            if self.num_turn <= 0:
                logger.warning(f"DistMonitor '{self.cmd_name}': num_turn={self.num_turn}; "
                               f"ignoring Save turns range [{start}, {end}, {step}]")
                continue
            if start >= self.num_turn:
                logger.warning(f"DistMonitor '{self.cmd_name}': start turn {start} is "
                               f"outside [0, {self.num_turn}); ignoring range "
                               f"[{original_start}, {original_end}, {step}]")
                continue
            if end < 0:
                logger.warning(f"DistMonitor '{self.cmd_name}': end turn {end} is before "
                               f"turn 0; ignoring range [{original_start}, {original_end}, {step}]")
                continue
            if start < 0:
                logger.warning(f"DistMonitor '{self.cmd_name}': clipping start turn {start} to 0")
                start = 0
            if end >= self.num_turn:
                logger.warning(f"DistMonitor '{self.cmd_name}': clipping end turn {end} "
                               f"to {self.num_turn - 1}")
                end = self.num_turn - 1
            if end < start:
                raise ValueError(f"DistMonitor '{self.cmd_name}': turn range [{original_start}, "
                                 f"{original_end}, {step}] has end before start")

            count = ((end - start) // step) + 1
            self._selected_turns[start:end + 1:step] = b"\x01" * count

        self.output_dir = Path(cfg.output_dir_dist)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_hms = str(cfg.output_hms)

        super().__init__()

    def print(self):
        set_simple_logging()
        selected_count = sum(self._selected_turns)
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, SelectedTurns={selected_count:d}/{self.num_turn:d}")
        set_normal_logging()

    def execute_cpu(self, sim: Simulation):
        return self._execute(sim, backend="cpu")

    def execute_gpu(self, sim: Simulation):
        return self._execute(sim, backend="gpu")

    def _execute(self, sim: Simulation, backend: str) -> bool:
        turn = int(sim.state.turn)
        if not is_turn_selected(turn, self._selected_turns):
            return False

        beam: Beam = sim.beams[self.beam_id]
        particles = beam.particles
        if backend == "gpu":
            # File I/O is host-side.  Copy only fields in the output schema.
            particles = particles.copy(np, fields=list(_DATA_FIELDS))

        for bunch in beam.bunches:
            self._save_bunch(sim, beam, bunch, particles, turn, backend)
        return True

    def _save_bunch(self, sim, beam, bunch, particles, turn: int, backend: str):
        start = int(bunch.start_idx)
        end = int(bunch.end_idx)
        df = pd.DataFrame({field: getattr(particles, field)[start:end] for field in _DATA_FIELDS})
        pending = int(np.count_nonzero(np.asarray(df["tag"]) == 0))
        df = df.loc[df["tag"] != 0].reset_index(drop=True)
        if self.include_injection_metadata:
            injection = getattr(beam, "injection_state", None)
            if injection is None:
                raise ValueError("DistMonitor injection metadata requires the beam's Injection state")
            for name, values in injection.snapshot(df["tag"].to_numpy()).items():
                df[name] = values

        tags = np.asarray(df["tag"])
        headers = {
            "Name": "PASS Distribution Data",
            "Command": self.cmd_type,
            "Monitor": self.cmd_name,
            "S": self.s,
            "BeamId": self.beam_id,
            "BeamName": beam.beam_name,
            "BunchId": int(bunch.bunch_id),
            "HarmonicId": int(bunch.harmonic_id),
            "HarmonicNumber": int(bunch.harmonic_number),
            "Turn": turn,
            "NumParticles": len(df),
            "NumAlive": int(np.count_nonzero(tags > 0)),
            "NumLost": int(np.count_nonzero(tags < 0)),
            "NumPending": pending,
            "Backend": backend,
            "ParticlePrecision": str(beam.particles.dtype),
            "PASSVersion": __version__,
            "Time": get_current_time(),
            "ZCoordinate": "z_rel",
            "ZCenter": float(bunch.harmonic_id*bunch.circum/bunch.harmonic_number),
            "ReferenceArrivalTime": float(bunch.t0),
            "ReferenceBeta": float(bunch.beta),
            "ReferenceMomentum": float(bunch.p0),
            "CoordinateDefinition": "z=beta*c*(T-t)",
            "ReferenceEvent": "element-exit",
            "ReferenceAppliesTo": "live particles only",
            "Circumference": float(bunch.circum),
        }

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.cmd_name).strip("_")
        safe_name = safe_name or "distmonitor"
        filename = (f"{self.output_hms}_dist_beam{self.beam_id}"
                    f"_bunch{int(bunch.bunch_id)}_Np_{int(bunch.Np)}"
                    f"_s_{self.s:.4f}_{safe_name}_turn_{turn}.tfs")
        filepath = self.output_dir / filename
        if self.output_format == "hdf5":
            import h5py
            filepath = filepath.with_suffix(".h5")
            with h5py.File(filepath, "w") as stream:
                for key, value in headers.items():
                    stream.attrs[key] = value
                for field in df.columns:
                    stream.create_dataset(field, data=df[field].to_numpy(), compression="gzip", compression_opts=1)
            logger.info("DistMonitor '%s': saved %s", self.cmd_name, filepath)
            return
        table = tfs.TfsDataFrame(df, headers=headers)
        tfs.write(str(filepath), table, colwidth=25, headerswidth=25)
        logger.info(f"DistMonitor '{self.cmd_name}': saved {filepath}")
