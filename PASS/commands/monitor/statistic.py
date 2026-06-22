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
import cupy as cp
import pandas as pd
import logging
import tfs
from pathlib import Path
import os
import csv

logger = logging.getLogger(__name__)


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
            headers["Time"] = get_current_time()

            table = tfs.TfsDataFrame(df, headers=headers)
            tfs.write(output_path_tfs, table)

    def execute_cpu(self, sim):
        cfg: Config = sim.cfg
        beam: Beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        state: State = sim.state

        turn = state.turn
        total_turn = cfg.num_turn

        for bunch in bunches:
            bunch_id = bunch.bunch_id

            start_idx = bunch.start_idx
            end_idx = bunch.end_idx
            Np = bunch.Np
            Ek = bunch.Ek

            p = beam.particles

            x = p.x[start_idx:end_idx]
            px = p.px[start_idx:end_idx]
            y = p.y[start_idx:end_idx]
            py = p.py[start_idx:end_idx]
            z = p.z[start_idx:end_idx]
            pz = p.pz[start_idx:end_idx]
            tag = p.tag[start_idx:end_idx]

            mask = tag > 0

            x = x[mask]
            px = px[mask]
            y = y[mask]
            py = py[mask]
            z = z[mask]
            pz = pz[mask]

            N = len(x)

            stat = {
                'x': x.mean(),
                'x2': (x**2).mean(),
                'xpx': (x * px).mean(),
                'px2': (px**2).mean(),
                'y': y.mean(),
                'y2': (y**2).mean(),
                'ypy': (y * py).mean(),
                'py2': (py**2).mean(),
                'z': z.mean(),
                'z2': (z**2).mean(),
                'pz': pz.mean(),
                'pz2': (pz**2).mean(),
                'px_avg': px.mean(),
                'py_avg': py.mean(),
                'xz': (x * z).mean(),
                'xy': (x * y).mean(),
                'yz': (y * z).mean(),
                'x3': (x**3).mean(),
                'x4': (x**4).mean(),
                'y3': (y**3).mean(),
                'y4': (y**4).mean()
            }

            sigma_x = np.sqrt(stat['x2'] - stat['x']**2)
            sigma_px = np.sqrt(stat['px2'] - stat['px_avg']**2)
            sigma_y = np.sqrt(stat['y2'] - stat['y']**2)
            sigma_py = np.sqrt(stat['py2'] - stat['py_avg']**2)
            sigma_z = np.sqrt(stat['z2'] - stat['z']**2)
            sigma_pz = np.sqrt(stat['pz2'] - stat['pz']**2)

            sig_xpx = stat['xpx'] - stat['x'] * stat['px_avg']
            sig_ypy = stat['ypy'] - stat['y'] * stat['py_avg']
            emit_x = np.sqrt(sigma_x**2 * sigma_px**2 - sig_xpx**2)
            emit_y = np.sqrt(sigma_y**2 * sigma_py**2 - sig_ypy**2)

            betax = sigma_x**2 / emit_x
            betay = sigma_y**2 / emit_y
            alphax = -sig_xpx / emit_x
            alphay = -sig_ypy / emit_y
            gammax = sigma_px**2 / emit_x
            gammay = sigma_py**2 / emit_y
            invx = gammax * betax - alphax**2
            invy = gammay * betay - alphay**2

            xz_div = stat['xz'] / (sigma_x * sigma_z) if (sigma_x > 0 and sigma_z > 0) else 0.0

            if sigma_x > 0:
                x_skew = (stat['x3'] - 3 * stat['x'] * sigma_x**2 - stat['x']**3) / sigma_x**3
                x_kurt = (stat['x4'] - 4 * stat['x'] * stat['x3'] + 2 * stat['x']**2 * stat['x2'] + 4 * stat['x']**2 * sigma_x**2 +
                          stat['x']**4) / (sigma_x**4)
            else:
                x_skew, x_kurt = 0.0, 0.0
            if sigma_y > 0:
                y_skew = (stat['y3'] - 3 * stat['y'] * sigma_y**2 - stat['y']**3) / sigma_y**3
                y_kurt = (stat['y4'] - 4 * stat['y'] * stat['y3'] + 2 * stat['y']**2 * stat['y2'] + 4 * stat['y']**2 * sigma_y**2 +
                          stat['y']**4) / (sigma_y**4)
            else:
                y_skew, y_kurt = 0.0, 0.0

            beam_loss = Np - N
            loss_percent = 100.0 * beam_loss / Np

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
                'pzAverage': stat['pz'],
                'sigmaZ': sigma_z,
                'sigmaPz': sigma_pz,
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
                'xzAverage': stat['xz'],
                'xyAverage': stat['xy'],
                'yzAverage': stat['yz'],
                'xzDevideSigmaxSigmaz': xz_div,
                'beamLossTotal': beam_loss,
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

    def execute_gpu(self, sim):
        cfg = sim.cfg
        beam: Beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        state: State = sim.state

        turn = state.turn
        total_turn = cfg.num_turn

        for bunch in bunches:
            bunch_id = bunch.bunch_id

            start_idx = bunch.start_idx
            end_idx = bunch.end_idx
            Np = bunch.Np
            Ek = bunch.Ek

            p = beam.particles

            x = p.x[start_idx:end_idx]
            px = p.px[start_idx:end_idx]
            y = p.y[start_idx:end_idx]
            py = p.py[start_idx:end_idx]
            z = p.z[start_idx:end_idx]
            pz = p.pz[start_idx:end_idx]
            tag = p.tag[start_idx:end_idx]

            # The maximum block is limited to 512, because there is atomicAdd in this kernel.
            # If the number of blocks is too large, the calculation will be slowed down due to atomicAdd
            N = end_idx - start_idx
            threads = 256
            blocks = min((N + threads - 1) // threads, 512)

            kernel = cp.RawKernel(kernel_code, "calc_all_stats")

            out_gpu = cp.zeros(22, dtype=cp.float64)

            kernel(
                (blocks, ),
                (threads, ),
                (p.x, p.px, p.y, p.py, p.z, p.pz, p.tag, np.int32(start_idx), np.int32(end_idx), out_gpu),
            )

            cp.cuda.runtime.deviceSynchronize()

            out_cpu = out_gpu.get()
            print(out_cpu)

            count_alive = int(out_cpu[21])
            inv_count = 1.0 / count_alive if count_alive > 0 else 0.0

            x_avg = out_cpu[0] * inv_count
            x2_avg = out_cpu[1] * inv_count
            xpx_avg = out_cpu[2] * inv_count
            px2_avg = out_cpu[3] * inv_count
            y_avg = out_cpu[4] * inv_count
            y2_avg = out_cpu[5] * inv_count
            ypy_avg = out_cpu[6] * inv_count
            py2_avg = out_cpu[7] * inv_count
            z_avg = out_cpu[8] * inv_count
            z2_avg = out_cpu[9] * inv_count
            pz_avg = out_cpu[10] * inv_count
            pz2_avg = out_cpu[11] * inv_count
            px_avg = out_cpu[12] * inv_count
            py_avg = out_cpu[13] * inv_count
            xz_avg = out_cpu[14] * inv_count
            xy_avg = out_cpu[15] * inv_count
            yz_avg = out_cpu[16] * inv_count
            x3_avg = out_cpu[17] * inv_count
            x4_avg = out_cpu[18] * inv_count
            y3_avg = out_cpu[19] * inv_count
            y4_avg = out_cpu[20] * inv_count

            beam_loss = Np - count_alive
            loss_percent = 100.0 * beam_loss / Np

            sigma_x = np.sqrt(x2_avg - x_avg**2)
            sigma_px = np.sqrt(px2_avg - px_avg**2)
            sigma_y = np.sqrt(y2_avg - y_avg**2)
            sigma_py = np.sqrt(py2_avg - py_avg**2)
            sigma_z = np.sqrt(z2_avg - z_avg**2)
            sigma_pz = np.sqrt(pz2_avg - pz_avg**2)

            sig_xpx = xpx_avg - x_avg * px_avg
            sig_ypy = ypy_avg - y_avg * py_avg

            emit_x = np.sqrt(sigma_x**2 * sigma_px**2 - sig_xpx**2)
            emit_y = np.sqrt(sigma_y**2 * sigma_py**2 - sig_ypy**2)

            betax = sigma_x**2 / emit_x
            betay = sigma_y**2 / emit_y
            alphax = -sig_xpx / emit_x
            alphay = -sig_ypy / emit_y
            gammax = sigma_px**2 / emit_x
            gammay = sigma_py**2 / emit_y
            invx = gammax * betax - alphax**2
            invy = gammay * betay - alphay**2

            xz_div = xz_avg / (sigma_x * sigma_z) if (sigma_x > 0 and sigma_z > 0) else 0.0

            if sigma_x > 0:
                x_skew = (x3_avg - 3 * x_avg * sigma_x**2 - x_avg**3) / sigma_x**3
                x_kurt = (x4_avg - 4 * x_avg * x3_avg + 2 * x_avg**2 * x2_avg + 4 * x_avg**2 * sigma_x**2 + x_avg**4) / (sigma_x**4)
            else:
                x_skew = x_kurt = 0.0
            if sigma_y > 0:
                y_skew = (y3_avg - 3 * y_avg * sigma_y**2 - y_avg**3) / sigma_y**3
                y_kurt = (y4_avg - 4 * y_avg * y3_avg + 2 * y_avg**2 * y2_avg + 4 * y_avg**2 * sigma_y**2 + y_avg**4) / (sigma_y**4)
            else:
                y_skew = y_kurt = 0.0

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
                'pzAverage': pz_avg,
                'sigmaZ': sigma_z,
                'sigmaPz': sigma_pz,
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
                'xzAverage': xz_avg,
                'xyAverage': xy_avg,
                'yzAverage': yz_avg,
                'xzDevideSigmaxSigmaz': xz_div,
                'beamLossTotal': beam_loss,
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


kernel_code = r'''
extern "C" __global__
void calc_all_stats(
    const double* __restrict__ x,
    const double* __restrict__ px,
    const double* __restrict__ y,
    const double* __restrict__ py,
    const double* __restrict__ z,
    const double* __restrict__ pz,
    const int* __restrict__ tag,
    int start,
    int end,
    double* out   // size 22
) {
    // ===== shared memory for warp results =====
    __shared__ double warp_sum[32][22];

    // ===== register accumulation (FASTEST) =====
    double local[22] = {0.0};

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // ===== 1. GRID STRIDE LOOP =====
    for (int i = start + tid; i < end; i += stride) {

        if (tag[i] <= 0) continue;

        double xi  = x[i];
        double pxi = px[i];
        double yi  = y[i];
        double pyi = py[i];
        double zi  = z[i];
        double pzi = pz[i];

        local[0]  += xi;
        local[1]  += xi * xi;
        local[2]  += xi * pxi;
        local[3]  += pxi * pxi;
        local[4]  += yi;
        local[5]  += yi * yi;
        local[6]  += yi * pyi;
        local[7]  += pyi * pyi;
        local[8]  += zi;
        local[9]  += zi * zi;
        local[10] += pzi;
        local[11] += pzi * pzi;
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

        local[21] += 1.0;
    }

    // ===== 2. WARP REDUCTION (FULL UNROLLED) =====
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int k = 0; k < 22; k++) {
            local[k] += __shfl_down_sync(0xffffffff, local[k], offset);
        }
    }

    // ===== 3. WRITE WARP RESULT =====
    if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < 22; k++) {
            warp_sum[warp][k] = local[k];
        }
    }

    __syncthreads();

    // ===== 4. BLOCK REDUCTION (warp0 only) =====
    if (warp == 0) {

        double sum[22] = {0.0};
        int num_warps = (blockDim.x + 31) >> 5;

        if (lane < num_warps) {
            #pragma unroll
            for (int k = 0; k < 22; k++) {
                sum[k] = warp_sum[lane][k];
            }
        }

        // warp reduce again
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int k = 0; k < 22; k++) {
                sum[k] += __shfl_down_sync(0xffffffff, sum[k], offset);
            }
        }

        // ===== 5. FINAL WRITE (ONE PER BLOCK) =====
        if (lane == 0) {
            for (int k = 0; k < 22; k++) {
                atomicAdd(&out[k], sum[k]);
            }
        }
    }
}
'''
