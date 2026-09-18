from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
import os
import csv

import numpy as np
import pandas as pd
import tfs

from PASS.commands.command import Command
from PASS.core.config import Config
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.state import SimulationState
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.helper import get_current_time

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


@Command.register("statmonitor")
class StatMonitor(Command):

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}")
        set_normal_logging()

    def _write_row(self, output_path_csv, output_path_tfs, row_dict, is_last_turn):

        fieldnames = list(row_dict.keys())

        if (not Path(output_path_csv).is_file()):
            with open(output_path_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row_dict)
        else:
            with open(output_path_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row_dict)

        if is_last_turn:
            df = pd.read_csv(output_path_csv)
            headers = {}
            headers["Name"] = "PASS Statistic Data"
            headers["ZCoordinate"] = "z_rel_folded_by_ring"
            headers["ZInterval"] = "[-C/2,C/2)"
            headers["SigmaTimeCoordinate"] = "continuous z / (referenceBeta*c); no folding"
            headers["Time"] = get_current_time()

            table = tfs.TfsDataFrame(df, headers=headers)
            tfs.write(output_path_tfs, table, colwidth=25, headerswidth=25)

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
            Ek = bunch.Ek

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

            injected = int(np.count_nonzero(tag))
            beam_loss = injected - n_alive
            loss_percent = 100.0 * beam_loss / injected if injected else 0.0

            row_dict = {
                'turn': turn,
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
                'zCenter': bunch.harmonic_id * bunch.circum / bunch.harmonic_number,
                'referenceTime': bunch.t0,
                'referenceBeta': bunch.beta,
                'referenceMomentum': bunch.p0,
                'sigmaTime': sigma_time,
                'xzAverage': stat['xz'],
                'xyAverage': stat['xy'],
                'yzAverage': stat['yz'],
                'xzDevideSigmaxSigmaz': xz_div,
                'beamLossTotal': beam_loss,
                'numAlive': n_alive,
                'numInjected': injected,
                'numPending': n_particles - injected,
                'lossPercent': loss_percent,
                'xSkewness': x_skew,
                'xKurtosis': x_kurt,
                'ySkewness': y_skew,
                'yKurtosis': y_kurt,
                'Ek': Ek
            }

            output_dir = cfg.output_dir_stat
            output_filename_csv = f"{cfg.output_hms}_stat_beam{self.beam_id}_bunch{bunch_id}_Np_{bunch.Np}_s_{self.s:.4f}.csv"
            output_filename_tfs = f"{cfg.output_hms}_stat_beam{self.beam_id}_bunch{bunch_id}_Np_{bunch.Np}_s_{self.s:.4f}.tfs"
            output_path_csv = os.path.join(output_dir, output_filename_csv)
            output_path_tfs = os.path.join(output_dir, output_filename_tfs)

            is_last_turn = False
            if turn == (total_turn - 1):
                is_last_turn = True
            self._write_row(output_path_csv, output_path_tfs, row_dict, is_last_turn)
            did_execute = True

        return did_execute

    def execute_gpu(self, sim):
        try:
            import cupy as cp
        except (ImportError, OSError) as exc:
            raise RuntimeError("GPU StatMonitor requires the optional 'cuda' dependencies "
                               "(install PASS with the [cuda] extra).") from exc
        cfg = sim.cfg
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
            Ek = bunch.Ek

            p = beam.particles

            x = p.x[start:end]
            px = p.px[start:end]
            y = p.y[start:end]
            py = p.py[start:end]
            z = p.z[start:end]
            dp = p.dp[start:end]
            tag = p.tag[start:end]

            # The maximum block is limited to 512, because there is atomicAdd in this kernel.
            # If the number of blocks is too large, the calculation will be slowed down due to atomicAdd
            n = end - start
            if n == 0:
                # Empty bunch: no statistics row.
                continue
            live_z = z[tag > 0].astype(cp.float64)
            sigma_time = float(cp.std(live_z)) / (bunch.beta * const.c) if live_z.size else 0.
            threads = 256
            blocks = min((n + threads - 1) // threads, 512)

            out_gpu = cp.zeros(21, dtype=cp.float64)
            count_gpu = cp.zeros(1, dtype=cp.int32)
            n_alive = int(cp.count_nonzero(tag > 0))
            centers_gpu = cp.zeros(6, dtype=cp.float64)
            if n_alive:
                first_live = int(cp.argmax(tag > 0))
                centers_gpu = cp.stack([coordinate[first_live] for coordinate in (x, px, y, py, z, dp)]).astype(cp.float64)
                centers_gpu[4] = _fold_by_ring(centers_gpu[4], bunch.circum)
                kernel = _get_stat_kernel(p.dtype.str)
                arguments = (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, np.int32(start), np.int32(end), np.float64(bunch.circum), centers_gpu, out_gpu,
                             count_gpu)
                # First find the centroid relative to a live particle, then sum centered moments.
                kernel((blocks, ), (threads, ), arguments)
                centers_gpu += out_gpu[cp.asarray([0, 12, 4, 13, 8, 10])] / n_alive
                out_gpu.fill(0.0)
                count_gpu.fill(0)
                kernel((blocks, ), (threads, ), arguments)
            moments = out_gpu.get() / max(n_alive, 1)
            stat = _statistics_from_centered_moments(moments, centers_gpu.get())
            x_avg, px_avg, y_avg, py_avg, z_avg, dp_avg = (stat[key] for key in ('x', 'px_avg', 'y', 'py_avg', 'z', 'dp'))
            xz_avg, xy_avg, yz_avg = stat['xz'], stat['xy'], stat['yz']

            injected = int((tag != 0).sum())
            beam_loss = injected - n_alive
            loss_percent = 100.0 * beam_loss / injected if injected else 0.0

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

            xz_div = xz_avg / (sigma_x * sigma_z) if (sigma_x > 0 and sigma_z > 0) else 0.0

            x_skew, x_kurt = stat['x_skew'], stat['x_kurt']
            y_skew, y_kurt = stat['y_skew'], stat['y_kurt']

            row_dict = {
                'turn': turn,
                'xAverage': x_avg,
                'pxAverage': px_avg,
                'sigmaX': sigma_x,
                'sigmaPx': sigma_px,
                'yAverage': y_avg,
                'pyAverage': py_avg,
                'sigmaY': sigma_y,
                'sigmaPy': sigma_py,
                'zAverage': z_avg,
                'dpAverage': dp_avg,
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
                'zCenter': bunch.harmonic_id * bunch.circum / bunch.harmonic_number,
                'referenceTime': bunch.t0,
                'referenceBeta': bunch.beta,
                'referenceMomentum': bunch.p0,
                'sigmaTime': sigma_time,
                'xzAverage': xz_avg,
                'xyAverage': xy_avg,
                'yzAverage': yz_avg,
                'xzDevideSigmaxSigmaz': xz_div,
                'beamLossTotal': beam_loss,
                'numAlive': n_alive,
                'numInjected': injected,
                'numPending': n_particles - injected,
                'lossPercent': loss_percent,
                'xSkewness': x_skew,
                'xKurtosis': x_kurt,
                'ySkewness': y_skew,
                'yKurtosis': y_kurt,
                'Ek': Ek
            }

            output_dir = cfg.output_dir_stat
            output_filename_csv = f"{cfg.output_hms}_stat_beam{self.beam_id}_bunch{bunch_id}_Np_{bunch.Np}_s_{self.s:.4f}.csv"
            output_filename_tfs = f"{cfg.output_hms}_stat_beam{self.beam_id}_bunch{bunch_id}_Np_{bunch.Np}_s_{self.s:.4f}.tfs"
            output_path_csv = os.path.join(output_dir, output_filename_csv)
            output_path_tfs = os.path.join(output_dir, output_filename_tfs)

            is_last_turn = False
            if turn == (total_turn - 1):
                is_last_turn = True
            self._write_row(output_path_csv, output_path_tfs, row_dict, is_last_turn)
            did_execute = True

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
    double* out, // size 21
    int* count_alive
) {
    // ===== shared memory for warp results =====
    __shared__ double warp_sum[32][21];
    __shared__ int warp_count[32];

    // ===== register accumulation (FASTEST) =====
    double local[21] = {0.0};
    int local_count = 0;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // ===== 1. GRID STRIDE LOOP =====
    for (int i = start + tid; i < end; i += stride) {

        if (tag[i] <= 0)
            continue;

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

        local_count += 1;
    }

    // ===== 2. WARP REDUCTION (FULL UNROLLED) =====
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int k = 0; k < 21; k++) {
            local[k] += __shfl_down_sync(0xffffffff, local[k], offset);
        }
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }

    // ===== 3. WRITE WARP RESULT =====
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < 21; k++) {
            warp_sum[warp][k] = local[k];
        }
        warp_count[warp] = local_count;
    }

    __syncthreads();

    // ===== 4. BLOCK REDUCTION (warp0 only) =====
    if (warp == 0) {

        double sum[21] = {0.0};
        int count_sum = 0;
        int num_warps = (blockDim.x + 31) >> 5;

        if (lane < num_warps) {
#pragma unroll
            for (int k = 0; k < 21; k++) {
                sum[k] = warp_sum[lane][k];
            }
            count_sum = warp_count[lane];
        }

// warp reduce again
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
            for (int k = 0; k < 21; k++) {
                sum[k] += __shfl_down_sync(0xffffffff, sum[k], offset);
            }
            count_sum += __shfl_down_sync(0xffffffff, count_sum, offset);
        }

        // ===== 5. FINAL WRITE (ONE PER BLOCK) =====
        if (lane == 0) {
            for (int k = 0; k < 21; k++) {
                atomicAdd(&out[k], sum[k]);
            }
            atomicAdd(count_alive, count_sum);
        }
    }
}
'''
STAT_SOURCE = CUDA_REAL_PREAMBLE + STAT_KERNEL_BODY


@lru_cache(maxsize=None)
def _get_stat_kernel(dtype):
    """Compile the CUDA statistics kernel on first GPU use."""
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU StatMonitor requires the optional 'cuda' dependencies "
                           "(install PASS with the [cuda] extra).") from exc

    dtype = np.dtype(dtype)
    use_float = dtype == np.dtype(np.float32)
    return cp.RawKernel(
        STAT_SOURCE,
        "calc_all_stats",
        options=("--std=c++14", f"-DPASS_USE_FLOAT={int(use_float)}"),
    )
