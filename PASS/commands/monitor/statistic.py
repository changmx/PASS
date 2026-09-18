from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
import os
import csv

import numpy as np
import pandas as pd

from PASS.commands.command import Command
from PASS.core.config import Config
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.state import SimulationState
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.helper import get_current_time
from PASS.utils.table_io import append_table, normalize_output_format, table_path, write_table

logger = logging.getLogger(__name__)


def _fold_by_ring(z, circumference):
    """Return the ring-period representative in [-C/2, C/2)."""
    return ((z + 0.5 * circumference) % circumference) - 0.5 * circumference


def _statistics_from_centered_moments(moments, centers):
    """Recover statistics from FP64 moments near the centroid, not the origin."""
    residual = moments[[0, 12, 4, 13, 8, 10]]
    means = centers + residual
    variances = np.maximum(moments[[1, 3, 5, 7, 9, 11]] - residual**2, 0.0)
    sigma = np.sqrt(variances)
    stat = dict(zip(('x', 'px_avg', 'y', 'py_avg', 'z', 'dp'), means))
    stat['sigma'] = sigma
    stat['sig_xpx'] = moments[2] - residual[0] * residual[1]
    stat['sig_ypy'] = moments[6] - residual[2] * residual[3]
    for name, index, first, second in (('xz', 14, 0, 4), ('xy', 15, 0, 2), ('yz', 16, 2, 4)):
        stat[name] = moments[index] - residual[first] * residual[second] + means[first] * means[second]
    for name, index, third, fourth in (('x', 0, 17, 18), ('y', 2, 19, 20)):
        shift = residual[index]
        variance = variances[index]
        central_third = moments[third] - 3 * shift * variance - shift**3
        central_fourth = moments[fourth] - 4 * shift * moments[third] + 6 * shift**2 * variance + 3 * shift**4
        stat[name + '_skew'] = central_third / sigma[index]**3 if sigma[index] > 0 else 0.0
        stat[name + '_kurt'] = central_fourth / variance**2 if variance > 0 else 0.0
    return stat


def _make_stat_row(stat, reference, n_alive, injected, sigma_time):
    """Build one output row using the same scalar formulae on both backends."""
    sigma_x, sigma_px, sigma_y, sigma_py, sigma_z, sigma_dp = stat['sigma']
    sig_xpx = stat['sig_xpx']
    sig_ypy = stat['sig_ypy']
    emit_x = np.sqrt(max(sigma_x**2 * sigma_px**2 - sig_xpx**2, 0.0))
    emit_y = np.sqrt(max(sigma_y**2 * sigma_py**2 - sig_ypy**2, 0.0))

    if emit_x > 0:
        betax = sigma_x**2 / emit_x
        alphax = -sig_xpx / emit_x
        gammax = sigma_px**2 / emit_x
        invx = gammax * betax - alphax**2
    else:
        betax = alphax = gammax = invx = 0.0

    if emit_y > 0:
        betay = sigma_y**2 / emit_y
        alphay = -sig_ypy / emit_y
        gammay = sigma_py**2 / emit_y
        invy = gammay * betay - alphay**2
    else:
        betay = alphay = gammay = invy = 0.0

    xz_div = stat['xz'] / (sigma_x * sigma_z) if (sigma_x > 0 and sigma_z > 0) else 0.0

    x_skew, x_kurt = stat['x_skew'], stat['x_kurt']
    y_skew, y_kurt = stat['y_skew'], stat['y_kurt']

    beam_loss = injected - n_alive
    loss_percent = 100.0 * beam_loss / injected if injected else 0.0

    return {
        'turn': reference['turn'],
        'xAverage': stat['x'],
        'pxAverage': stat['px_avg'],
        'sigmaX': sigma_x,
        'sigmaPx': sigma_px,
        'yAverage': stat['y'],
        'pyAverage': stat['py_avg'],
        'sigmaY': sigma_y,
        'sigmaPy': sigma_py,
        'zAverage': stat['z'],
        'dpAverage': stat['dp'],
        'sigmaZ': sigma_z,
        'sigmadp': sigma_dp,
        'xEmittance': emit_x,
        'yEmittance': emit_y,
        'betax': betax,
        'betay': betay,
        'alphax': alphax,
        'alphay': alphay,
        'gammax': gammax,
        'gammay': gammay,
        'invariantx': invx,
        'invarianty': invy,
        'zCenter': reference['zCenter'],
        'referenceTime': reference['referenceTime'],
        'referenceBeta': reference['referenceBeta'],
        'referenceMomentum': reference['referenceMomentum'],
        'sigmaTime': sigma_time,
        'xzAverage': stat['xz'],
        'xyAverage': stat['xy'],
        'yzAverage': stat['yz'],
        'xzDevideSigmaxSigmaz': xz_div,
        'beamLossTotal': beam_loss,
        'numAlive': n_alive,
        'numInjected': injected,
        'numPending': reference['n_particles'] - injected,
        'lossPercent': loss_percent,
        'xSkewness': x_skew,
        'xKurtosis': x_kurt,
        'ySkewness': y_skew,
        'yKurtosis': y_kurt,
        'Ek': reference['Ek']
    }


@Command.register("statmonitor")
class StatMonitor(Command):

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]
        self.output_format = normalize_output_format(kwargs.get("output format", "hdf5-gzip1"))
        self.write_interval_turns = kwargs.get("write interval (turns)", 100)
        if (isinstance(self.write_interval_turns, bool) or not isinstance(self.write_interval_turns, (int, np.integer))
                or self.write_interval_turns < 1):
            raise ValueError("StatMonitor Write interval (turns) must be a positive integer")
        self._pending_rows = {}
        self._table_paths = {}
        self._tfs_dirty = set()
        self._output_failed = False
        self._gpu_buffer = None
        self._gpu_pending = []
        self._gpu_first_live = None
        p = sim.beams[self.beam_id].particles
        if p.xp.__name__ == "cupy":
            self._allocate_gpu_buffer(p.xp, len(sim.beams[self.beam_id].bunches))

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}")
        set_normal_logging()

    def _allocate_gpu_buffer(self, cp, n_bunches):
        # Per row: 23 centered moment sums, two counts, and seven centers.
        capacity = self.write_interval_turns * max(1, n_bunches)
        self._gpu_buffer = cp.empty((capacity, 32), dtype=cp.float64)
        self._gpu_first_live = cp.empty((), dtype=cp.int64)

    @staticmethod
    def _reference_row(bunch, turn):
        # Snapshot host values now, never use a later turn's reference at flush.
        return {
            "turn": turn,
            "zCenter": bunch.harmonic_id * bunch.circum / bunch.harmonic_number,
            "referenceTime": bunch.t0,
            "referenceBeta": bunch.beta,
            "referenceMomentum": bunch.p0,
            "Ek": bunch.Ek,
            "n_particles": bunch.Np,
        }

    def _collect_gpu_rows(self):
        if not self._gpu_pending:
            return
        # The only device-to-host transfer made by StatMonitor, once per batch.
        records = self._gpu_buffer[:len(self._gpu_pending)].get()
        for record, (csv_path, output_path, reference) in zip(records, self._gpu_pending):
            n_alive, injected = int(record[23]), int(record[24])
            moments = record[:23] / max(n_alive, 1)
            stat = _statistics_from_centered_moments(moments, record[25:31])
            sigma_time = np.sqrt(max(moments[22] - moments[21]**2, 0.0)) / (reference["referenceBeta"] * const.c)
            row = _make_stat_row(stat, reference, n_alive, injected, sigma_time)
            self._record_row(csv_path, output_path, row)
        self._gpu_pending.clear()

    def _record_row(self, output_path_csv, output_path_table, row_dict):
        self._pending_rows.setdefault(output_path_csv, []).append(row_dict)
        self._table_paths[output_path_csv] = table_path(output_path_table, self.output_format)

    def _flush_rows(self, final=False):
        if self._output_failed:
            return  # A partially failed write must not be retried as a duplicate batch.
        headers = {
            "Name": "PASS Statistic Data",
            "Monitor": self.cmd_name,
            "BeamId": self.beam_id,
            "S": self.s,
            "ZCoordinate": "z_rel_folded_by_ring",
            "ZInterval": "[-C/2,C/2)",
            "SigmaTimeCoordinate": "continuous z / (referenceBeta*c); no folding",
            "Time": get_current_time(),
        }
        try:
            self._collect_gpu_rows()
            for csv_path, rows in self._pending_rows.items():
                if not rows:
                    continue
                Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
                if self.output_format != "tfs":
                    # Columns use one batch per chunk to avoid recompressing earlier batches.
                    append_table(self._table_paths[csv_path],
                                 pd.DataFrame(rows),
                                 headers,
                                 chunk_rows=self.write_interval_turns,
                                 output_format=self.output_format)
                first_write = not Path(csv_path).is_file() or Path(csv_path).stat().st_size == 0
                with open(csv_path, "a", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    if first_write:
                        writer.writeheader()
                    writer.writerows(rows)
                rows.clear()
                if self.output_format == "tfs":
                    self._tfs_dirty.add(csv_path)
            if final:
                for csv_path in tuple(self._tfs_dirty):
                    write_table(self._table_paths[csv_path],
                                pd.read_csv(csv_path, float_precision="round_trip"),
                                headers,
                                colwidth=25,
                                headerswidth=25)
                    self._tfs_dirty.remove(csv_path)
        except BaseException:
            # A keyboard interruption during a write can also leave a partial batch.
            self._output_failed = True
            raise

    def finalize(self, sim):
        """Flush the last partial batch, including on a handled interruption."""
        self._flush_rows(final=True)

    def execute_cpu(self, sim):
        cfg: Config = sim.cfg
        beam: Beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        state: SimulationState = sim.state

        turn = state.turn
        total_turn = cfg.num_turn

        did_execute = False
        for bunch in bunches:
            bunch_id = bunch.bunch_id

            start = bunch.start_idx
            end = bunch.end_idx
            n_particles = bunch.Np
            p = beam.particles

            x = p.x[start:end]
            px = p.px[start:end]
            y = p.y[start:end]
            py = p.py[start:end]
            z = p.z[start:end]
            dp = p.dp[start:end]
            tag = p.tag[start:end]

            mask = tag > 0

            x = x[mask]
            px = px[mask]
            y = y[mask]
            py = py[mask]
            z = z[mask]
            dp = dp[mask]

            n_alive = len(x)
            if n_alive == 0:
                if n_particles == 0:
                    continue
                # Keep an explicit zero-survival row during injection/loss.
                x = px = y = py = z = dp = np.zeros(1, dtype=p.dtype)

            # z is stored bunch-relative and may remain unwrapped during
            # tracking. Statistics use one full-ring representative.
            sigma_time = float(np.std(z.astype(np.float64))) / (bunch.beta * const.c)
            z = _fold_by_ring(z.astype(np.float64), bunch.circum)

            # Remove an anchor before finding the centroid to resolve narrow beams.
            coordinates = np.array((x, px, y, py, z, dp), dtype=np.float64)
            anchors = coordinates[:, :1].copy()
            coordinates -= anchors
            offsets = coordinates.mean(axis=1)
            coordinates -= offsets[:, None]
            centers = anchors[:, 0] + offsets
            x, px, y, py, z, dp = coordinates
            moments = np.array([
                x.mean(), (x * x).mean(), (x * px).mean(), (px * px).mean(),
                y.mean(), (y * y).mean(), (y * py).mean(), (py * py).mean(),
                z.mean(), (z * z).mean(),
                dp.mean(), (dp * dp).mean(),
                px.mean(),
                py.mean(), (x * z).mean(), (x * y).mean(), (y * z).mean(), (x**3).mean(), (x**4).mean(), (y**3).mean(), (y**4).mean()
            ])
            stat = _statistics_from_centered_moments(moments, centers)
            reference = self._reference_row(bunch, turn)
            row_dict = _make_stat_row(stat, reference, n_alive, int(np.count_nonzero(tag)), sigma_time)

            output_dir = cfg.output_dir_stat
            output_filename_csv = f"{cfg.output_hms}_stat_beam{self.beam_id}_bunch{bunch_id}_Np_{bunch.Np}_s_{self.s:.4f}.csv"
            output_filename_table = f"{cfg.output_hms}_stat_beam{self.beam_id}_bunch{bunch_id}_Np_{bunch.Np}_s_{self.s:.4f}.tfs"
            output_path_csv = os.path.join(output_dir, output_filename_csv)
            output_path_table = os.path.join(output_dir, output_filename_table)

            self._record_row(output_path_csv, output_path_table, row_dict)
            did_execute = True

        if (turn + 1) % self.write_interval_turns == 0 or turn == total_turn - 1:
            self._flush_rows(final=turn == total_turn - 1)
        return did_execute

    def execute_gpu(self, sim):
        try:
            import cupy as cp
        except (ImportError, OSError) as exc:
            raise RuntimeError("GPU StatMonitor requires the optional 'cuda' dependencies "
                               "(install PASS with the [cuda] extra).") from exc
        cfg = sim.cfg
        beam = sim.beams[self.beam_id]
        p = beam.particles
        turn = sim.state.turn
        if self._gpu_buffer is None:
            self._allocate_gpu_buffer(cp, len(beam.bunches))
        initialize, calculate, center = _get_stat_kernels(p.dtype.str)
        did_execute = False
        for bunch in beam.bunches:
            start, end = bunch.start_idx, bunch.end_idx
            n_particles = end - start
            if n_particles == 0:
                continue
            index = len(self._gpu_pending)
            if index == len(self._gpu_buffer):
                # New bunches can increase capacity without an early host copy.
                larger = cp.empty((2 * len(self._gpu_buffer), 32), dtype=cp.float64)
                larger[:index] = self._gpu_buffer
                self._gpu_buffer = larger
            record = self._gpu_buffer[index]
            moments, centers = record[:25], record[25:]
            # argmax returns a device scalar; it is never converted to Python.
            cp.argmax(p.tag[start:end] > 0, out=self._gpu_first_live)
            initialize((1, ), (1, ),
                       (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, np.int32(start), self._gpu_first_live, np.float64(bunch.circum), centers))
            threads = 256
            blocks = min((n_particles + threads - 1) // threads, 512)
            arguments = (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, np.int32(start), np.int32(end), np.float64(bunch.circum), centers, moments)
            # Keep the two centered passes in FP64; counts stay on the device.
            moments.fill(0.)
            calculate((blocks, ), (threads, ), arguments)
            center((1, ), (1, ), (moments, centers))
            moments.fill(0.)
            calculate((blocks, ), (threads, ), arguments)

            filename = f"{cfg.output_hms}_stat_beam{self.beam_id}_bunch{bunch.bunch_id}_Np_{bunch.Np}_s_{self.s:.4f}"
            csv_path = os.path.join(cfg.output_dir_stat, filename + ".csv")
            output_path = os.path.join(cfg.output_dir_stat, filename + ".tfs")
            self._gpu_pending.append((csv_path, output_path, self._reference_row(bunch, turn)))
            did_execute = True

        if (turn + 1) % self.write_interval_turns == 0 or turn == cfg.num_turn - 1:
            self._flush_rows(final=turn == cfg.num_turn - 1)
        return did_execute


CUDA_REAL_PREAMBLE = r'''
#ifndef PASS_USE_FLOAT
#define PASS_USE_FLOAT 0
#endif

#if PASS_USE_FLOAT
using pass_real_t = float;
#define PASS_FLOOR floorf
#else
using pass_real_t = double;
#define PASS_FLOOR floor
#endif
'''

STAT_KERNEL_BODY = r'''
extern "C" __global__ void initialize_centers(
    const pass_real_t* x,
    const pass_real_t* px,
    const pass_real_t* y,
    const pass_real_t* py,
    const pass_real_t* z,
    const pass_real_t* dp,
    const int* tag,
    int start,
    const long long* first_live,
    double circumference,
    double* centers
) {
    int i = start + int(first_live[0]);
    if (tag[i] <= 0) {
        for (int k = 0; k < 7; ++k) {
            centers[k] = 0.0;
        }
        return;
    }
    centers[0] = double(x[i]);
    centers[1] = double(px[i]);
    centers[2] = double(y[i]);
    centers[3] = double(py[i]);
    centers[6] = double(z[i]);
    double folded_z = double(z[i]) + 0.5 * circumference;
    centers[4] = folded_z - floor(folded_z / circumference) * circumference - 0.5 * circumference;
    centers[5] = double(dp[i]);
}

extern "C" __global__ void update_centers(
    const double* moments,
    double* centers
) {
    double n_alive = moments[23];
    if (n_alive == 0.0) {
        return;
    }
    const int indices[7] = {0, 12, 4, 13, 8, 10, 21};
    for (int k = 0; k < 7; ++k) {
        centers[k] += moments[indices[k]] / n_alive;
    }
}

extern "C" __global__ void calc_all_stats(
    const pass_real_t* __restrict__ x,
    const pass_real_t* __restrict__ px,
    const pass_real_t* __restrict__ y,
    const pass_real_t* __restrict__ py,
    const pass_real_t* __restrict__ z,
    const pass_real_t* __restrict__ dp,
    const int* __restrict__ tag,
    int start,
    int end,
    double circumference,
    const double* __restrict__ centers,
    double* out // 23 moments, live count, injected count
) {
    // ===== shared memory for warp results =====
    __shared__ double warp_sum[32][25];

    // ===== register accumulation (FASTEST) =====
    double local[25] = {0.0};

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // ===== 1. GRID STRIDE LOOP =====
    for (int i = start + tid; i < end; i += stride) {

        local[24] += tag[i] != 0;
        if (tag[i] <= 0) {
            continue;
        }

        double xi = double(x[i]) - centers[0];
        double pxi = double(px[i]) - centers[1];
        double yi = double(y[i]) - centers[2];
        double pyi = double(py[i]) - centers[3];
        double zi = double(z[i]) + 0.5 * circumference;
        zi = zi - floor(zi / circumference) * circumference;
        zi = zi - 0.5 * circumference - centers[4];
        double dpi = double(dp[i]) - centers[5];

        local[0] += xi;
        local[1] += xi * xi;
        local[2] += xi * pxi;
        local[3] += pxi * pxi;
        local[4] += yi;
        local[5] += yi * yi;
        local[6] += yi * pyi;
        local[7] += pyi * pyi;
        local[8] += zi;
        local[9] += zi * zi;
        local[10] += dpi;
        local[11] += dpi * dpi;
        local[12] += pxi;
        local[13] += pyi;
        local[14] += xi * zi;
        local[15] += xi * yi;
        local[16] += yi * zi;

        double x2 = xi * xi;
        double y2 = yi * yi;

        local[17] += xi * x2;
        local[18] += x2 * x2;
        local[19] += yi * y2;
        local[20] += y2 * y2;

        double z_continuous = double(z[i]) - centers[6];
        local[21] += z_continuous;
        local[22] += z_continuous * z_continuous;
        local[23] += 1.0;
    }

    // ===== 2. WARP REDUCTION (FULL UNROLLED) =====
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int k = 0; k < 25; k++) {
            local[k] += __shfl_down_sync(0xffffffff, local[k], offset);
        }
    }

    // ===== 3. WRITE WARP RESULT =====
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < 25; k++) {
            warp_sum[warp][k] = local[k];
        }
    }

    __syncthreads();

    // ===== 4. BLOCK REDUCTION (warp0 only) =====
    if (warp == 0) {

        double sum[25] = {0.0};
        int num_warps = (blockDim.x + 31) >> 5;

        if (lane < num_warps) {
#pragma unroll
            for (int k = 0; k < 25; k++) {
                sum[k] = warp_sum[lane][k];
            }
        }

// warp reduce again
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
            for (int k = 0; k < 25; k++) {
                sum[k] += __shfl_down_sync(0xffffffff, sum[k], offset);
            }
        }

        // ===== 5. FINAL WRITE (ONE PER BLOCK) =====
        if (lane == 0) {
            for (int k = 0; k < 25; k++) {
                atomicAdd(&out[k], sum[k]);
            }
        }
    }
}
'''
STAT_SOURCE = CUDA_REAL_PREAMBLE + STAT_KERNEL_BODY


@lru_cache(maxsize=None)
def _get_stat_kernels(dtype):
    """Compile the statistics and device-side centering kernels on first use."""
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU StatMonitor requires the optional 'cuda' dependencies "
                           "(install PASS with the [cuda] extra).") from exc

    dtype = np.dtype(dtype)
    use_float = dtype == np.dtype(np.float32)
    module = cp.RawModule(code=STAT_SOURCE, options=("--std=c++14", f"-DPASS_USE_FLOAT={int(use_float)}"))
    return tuple(module.get_function(name) for name in ("initialize_centers", "calc_all_stats", "update_centers"))
