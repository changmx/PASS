from __future__ import annotations

from PASS.commands.command import Command
from PASS.core.config import Config
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.state import State
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.helper import get_current_time

import numpy as np
import pandas as pd
import logging
import tfs
from pathlib import Path
import os

logger = logging.getLogger(__name__)

# Columns stored per particle per turn
# 0:turn, 1:x, 2:px, 3:y, 4:py, 5:z, 6:dp, 7:tag, 8:lost_turn,
# 9:lost_position, 10:zCenter
_NCOLS = 11

# Column names for output TFS
_COL_NAMES = [
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
]


@Command.register("particlemonitor")
class ParticleMonitor(Command):
    """Turn-by-turn particle coordinate monitor.

    Records 6D coordinates (+ tag, lost_turn, lost_position) of particles
    with ``1 <= |tag| <= max_tag`` every turn within ``[start_turn, end_turn)``
    at the monitor's s-position.

    A pre-allocated buffer ``(max_tag, num_record_turns, NCOLS)`` is filled
    each turn.  At the end of the simulation, each particle's TBT data is
    written to a separate TFS file.

    File naming:
        {hms}_particle_beam{bid}_{monitor_name}_s_{s:.3f}_tag_{tag}.tfs
    """

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]

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

        self.output_dir_particle: str = cfg.output_dir_particle
        self.output_hms: str = cfg.output_hms
        Path(self.output_dir_particle).mkdir(parents=True, exist_ok=True)

        # Pre-allocate buffer using the same array backend as the beam
        beam: Beam = sim.beams[self.beam_id]
        xp = beam.particles.xp  # np or cp

        if self.max_tag >= 1 and self.num_record_turn > 0:
            self.buffer = xp.zeros((self.max_tag, self.num_record_turn, _NCOLS), dtype=xp.float64)
        else:
            # Edge case: nothing to record, use a tiny placeholder
            self.buffer = xp.zeros((1, 1, _NCOLS), dtype=xp.float64)

        self._first_index = (
            xp.empty(self.max_tag, dtype=xp.int32)
            if self.max_tag >= 1 and self.num_record_turn > 0
            and _is_cupy_array(beam.particles.x)
            else None
        )

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, "
                    f"MaxTag={self.max_tag:d}, TurnRange=[{self.start_turn:d},{self.end_turn:d}), "
                    f"NumRecordTurn={self.num_record_turn:d}")
        set_normal_logging()

    def _record_one_turn(self, particles, bunch, turn):
        """Fill buffer for one bunch at a given turn.

        Works for both CPU (numpy) and GPU (cupy) particle arrays.
        """
        record_idx = turn - self.start_turn
        if record_idx < 0 or record_idx >= self.num_record_turn:
            return

        start = bunch.start_idx
        end = bunch.end_idx
        tag_all = particles.tag[start:end]
        is_gpu = _is_cupy_array(tag_all)

        if is_gpu:
            find_kernel, write_kernel = _get_monitor_kernels(particles.dtype)
            threads = 256
            find_blocks = ((end - start) + threads - 1) // threads
            write_blocks = (self.max_tag + threads - 1) // threads

            # Initialize to the exclusive end index.  A missing tag then
            # remains unwritten, matching the zero-initialized CPU buffer.
            self._first_index.fill(np.int32(end))
            find_kernel(
                (find_blocks,), (threads,),
                (particles.tag, np.int32(start), np.int32(end),
                 np.int32(self.max_tag), self._first_index),
            )
            write_kernel(
                (write_blocks,), (threads,),
                (particles.x, particles.px, particles.y, particles.py,
                 particles.z, particles.dp, particles.tag,
                 particles.lost_turn, particles.lost_position,
                 self._first_index, self.buffer, np.int32(end),
                 np.int32(self.max_tag),
                 np.int32(record_idx), np.int32(self.num_record_turn),
                 np.int32(turn), np.float64(bunch.z_center)),
            )
            return

        for tag_val in range(1, self.max_tag + 1):
            # |tag| matching: find particles whose |tag| == tag_val
            # This captures both alive (tag>0) and lost (tag<0) particles
            abs_tag = xp_abs(tag_all)
            matches = xp_where(abs_tag == tag_val)[0]
            if len(matches) == 0:
                continue

            idx = start + matches[0]

            buf_row = self.buffer[tag_val - 1, record_idx]
            buf_row[0] = float(turn)
            buf_row[1] = float(particles.x[idx])
            buf_row[2] = float(particles.px[idx])
            buf_row[3] = float(particles.y[idx])
            buf_row[4] = float(particles.py[idx])
            buf_row[5] = float(particles.z[idx])
            buf_row[6] = float(particles.dp[idx])
            buf_row[7] = float(particles.tag[idx])
            buf_row[8] = float(particles.lost_turn[idx])
            buf_row[9] = float(particles.lost_position[idx])
            buf_row[10] = float(bunch.z_center)

    def execute_cpu(self, sim: Simulation):
        cfg: Config = sim.cfg
        beam: Beam = sim.beams[self.beam_id]
        state: State = sim.state
        turn = state.turn

        # Record within [start_turn, end_turn)
        did_execute = False
        if self.max_tag >= 1 and self.num_record_turn > 0:
            if self.start_turn <= turn < self.end_turn:
                for bunch in beam.bunches:
                    self._record_one_turn(beam.particles, bunch, turn)
                did_execute = True

        # Write TFS files on the last recorded turn
        if turn == self.end_turn - 1:
            self._write_tfs(sim)
            did_execute = self.max_tag >= 1 or did_execute

        return did_execute

    def execute_gpu(self, sim: Simulation):
        cfg: Config = sim.cfg
        beam: Beam = sim.beams[self.beam_id]
        state: State = sim.state
        turn = state.turn

        # Record within [start_turn, end_turn)
        # GPU: buffer stays on GPU, write directly from GPU arrays
        # No per-turn D2H copy; only one D2H copy at the end (_write_tfs)
        did_execute = False
        if self.max_tag >= 1 and self.num_record_turn > 0:
            if self.start_turn <= turn < self.end_turn:
                for bunch in beam.bunches:
                    self._record_one_turn(beam.particles, bunch, turn)
                did_execute = True

        # Write TFS files on the last recorded turn (single D2H transfer)
        if turn == self.end_turn - 1:
            self._write_tfs(sim)
            did_execute = self.max_tag >= 1 or did_execute

        return did_execute

    def _write_tfs(self, sim: Simulation):
        """Write each particle's TBT data to a separate TFS file."""
        if self.max_tag < 1:
            return

        cfg: Config = sim.cfg

        output_dir = self.output_dir_particle
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Single D2H transfer for GPU buffer
        buf_cpu = xp_get(self.buffer)

        for tag_val in range(1, self.max_tag + 1):
            tag_data = buf_cpu[tag_val - 1]  # shape (num_record_turn, NCOLS)

            # Build DataFrame
            df_data = {}
            df_data["turn"] = tag_data[:, 0].astype(np.int32)
            for i, name in enumerate(_COL_NAMES[1:], start=1):
                df_data[name] = tag_data[:, i]

            df = pd.DataFrame(df_data)

            # TFS headers
            headers = {
                "Name": "PASS Particle Monitor",
                "Time": get_current_time(),
                "Monitor": self.cmd_name,
                "S": self.s,
                "BeamId": self.beam_id,
                "Tag": tag_val,
                "NumTurn": self.num_record_turn,
                "StartTurn": self.start_turn,
                "EndTurn": self.end_turn,
            }

            table = tfs.TfsDataFrame(df, headers=headers)

            filename = (f"{self.output_hms}_beam{self.beam_id}"
                        f"_{self.cmd_name}_s{self.s:.3f}_tag{tag_val}.tfs")
            filepath = os.path.join(output_dir, filename)
            tfs.write(filepath, table)

        set_simple_logging()
        logger.info(f"ParticleMonitor '{self.cmd_name}': "
                    f"{self.max_tag} TBT files written to {output_dir}")
        set_normal_logging()


# ---------------------------------------------------------------------------
# Backend-agnostic helpers (work for both numpy and cupy arrays)
# ---------------------------------------------------------------------------


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
    const int* tag, int start, int end, int max_tag, int* first)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i >= end) return;
    int value = tag[i];
    int abs_value = value < 0 ? -value : value;
    if (abs_value >= 1 && abs_value <= max_tag)
        atomicMin(&first[abs_value - 1], i);
}

extern "C" __global__ void particle_monitor_write(
    const pass_real_t* x, const pass_real_t* px,
    const pass_real_t* y, const pass_real_t* py,
    const pass_real_t* z, const pass_real_t* dp,
    const int* tag, const int* lost_turn, const float* lost_position,
    const int* first, double* out, int end, int max_tag, int record_idx,
    int num_record_turn, int turn, double z_center)
{
    int tag_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (tag_index >= max_tag) return;
    int i = first[tag_index];
    if (i < 0 || i >= end) return;
    size_t base = ((size_t)tag_index * (size_t)num_record_turn
                   + (size_t)record_idx) * 11;
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
}
'''

_monitor_kernels = {}


def _get_monitor_kernels(dtype):
    """Compile monitor indexing/writing kernels once per particle precision."""
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "GPU ParticleMonitor requires the optional 'cuda' dependencies."
        ) from exc

    key = np.dtype(dtype)
    if key not in _monitor_kernels:
        _monitor_kernels[key] = (
            cp.RawKernel(
                _MONITOR_SOURCE,
                "particle_monitor_find",
                options=("--std=c++14", f"-DPASS_USE_FLOAT={int(key == np.dtype(np.float32))}"),
            ),
            cp.RawKernel(
                _MONITOR_SOURCE,
                "particle_monitor_write",
                options=("--std=c++14", f"-DPASS_USE_FLOAT={int(key == np.dtype(np.float32))}"),
            ),
        )
    return _monitor_kernels[key]
