from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const

import numpy as np
import cupy as cp
import logging

logger = logging.getLogger(__name__)


@Command.register("drift")
class Drift(Command):

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = kwargs["length (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]

        if self.length < 0.0:
            raise ValueError(f"The length of Drift {self.cmd_name} is {self.length}, which should be >= 0")

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, Length={self.length:.4f}")
        set_normal_logging()

    def execute_cpu(self, sim):

        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches

        for i, bunch in enumerate(bunches):
            drift_exact_cpu(self.length, beam, bunch)

    def execute_gpu(self, sim):
        L = self.length
        if np.abs(L) < const.eps:
            return
        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches

        for i, bunch in enumerate(bunches):
            beta = bunch.beta
            gamma = bunch.gamma
            circum = bunch.circum
            start = bunch.start_idx
            end = bunch.end_idx

            p = beam.particles  # slicing in the kernel

            N = end - start
            threads = 256
            blocks = (N + threads - 1) // threads

            transfer_drift_kernel(
                (blocks, ),
                (threads, ),
                (p.x, p.y, p.z, p.px, p.py, p.dp, p.tag, start, end, beta, gamma, circum, L),
            )


def drift_exact_cpu(L: float, beam: Beam, bunch: BunchInfo):
    if np.abs(L) < const.eps:
        return

    beta0 = bunch.beta
    gamma0 = bunch.gamma
    circum = bunch.circum
    start = bunch.start_idx
    end = bunch.end_idx

    p = beam.particles
    x = p.x[start:end]
    px = p.px[start:end]
    y = p.y[start:end]
    py = p.py[start:end]
    z = p.z[start:end]
    dp = p.dp[start:end]
    tag = p.tag[start:end]

    beta = (1 + dp) * (gamma0 * beta0) / np.sqrt(1 + ((1 + dp) * (gamma0 * beta0))**2)
    pz_sq = (1 + dp)**2 - px**2 - py**2
    valid = (pz_sq > 0.0) & (tag > 0)
    tag[~valid] = -1
    pz_sq_safe = np.maximum(pz_sq, 0.0)
    pz = np.sqrt(pz_sq_safe)

    c_half = 0.5 * circum
    mask = (tag > 0).astype(np.float64)
    L_mask = L * mask

    x += L_mask * (px / pz)
    y += L_mask * (py / pz)
    z += L_mask * (1 - (beta0 / beta) * (1 + dp) / pz)

    over = (z > c_half).astype(np.int64)
    under = (z < -c_half).astype(np.int64)

    z += (under - over) * circum


kernel_code = r'''
extern "C" __global__
void transfer_drift(
    double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ z,
    const double* __restrict__ px,
    const double* __restrict__ py,
    const double* __restrict__ dp,
    const int* __restrict__ tag,
    int start_index,
    int end_index,
    double beta,
    double gamma,
    double circum,
    double L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;

    double r56 = L / (beta * beta * gamma * gamma);
    double c_half = 0.5 * circum;

    int mask = tag[i] > 0;

    x[i] += L * px[i] * mask;
    y[i] += L * py[i] * mask;
    z[i] += r56 * (dp[i] * beta) * beta * mask;

    int over = z[i] > c_half;
    int under = z[i] < -c_half;

    z[i] += (under - over) * circum;
}
'''
transfer_drift_kernel = cp.RawKernel(kernel_code, "transfer_drift")
