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

    def __init__(self, beam_id: int, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = kwargs["length (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, Length={self.length:.4f}")
        set_normal_logging()

    def execute_cpu(self, sim):
        pass

    def execute_gpu(self, sim):
        pass


def transfer_drift_cpu(sim: Simulation, L: float, beam_id: int):
    if np.abs(L) < const.eps:
        return
    beam = sim.beams[beam_id]
    bunches: list[BunchInfo] = beam.bunches

    for i, bunch in enumerate(bunches):
        beta = bunch.beta
        gamma = bunch.gamma
        circum = bunch.circum
        start = bunch.start
        stop = bunch.stop

        p = beam.particles[start:stop]

        r56 = L / (beta**2 * gamma**2)

        c_half = 0.5 * circum
        mask = (p.tag > 0).astype(np.float64)
        """
        tau = z/beta - ct(=0) = z/beta
        pt = DeltaE/(P0*c) = beta*DeltaP/P0

        tau0 = dev_particle.z[tid] / beta;
		pt0 = dev_particle.pz[tid] * beta;

        tau1 = tau0 + (r56 * pt0) * mask;
		dev_particle.z[tid] = tau1 * beta;
        """
        p.x += L * p.px * mask
        p.y += L * p.py * mask
        p.z += r56 * (p.pz * beta) * beta * mask

        over = (p.z > c_half).astype(np.int64)
        under = (p.z < -c_half).astype(np.int64)

        p.z += (under - over) * circum


def transfer_drift_gpu(sim: Simulation, L: float, beam_id: int):
    if np.abs(L) < const.eps:
        return
    beam = sim.beams[beam_id]
    bunches: list[BunchInfo] = beam.bunches

    for i, bunch in enumerate(bunches):
        beta = bunch.beta
        gamma = bunch.gamma
        circum = bunch.circum
        start = bunch.start
        stop = bunch.stop

        p = beam.particles  # slicing in the kernel

        N = p.x.size
        threads = 256
        blocks = (N + threads - 1) // threads

        transfer_drift_gpu(
            (blocks, ),
            (threads, ),
            (p.x, p.y, p.z, p.px, p.py, p.pz, p.tag, start, stop, beta, gamma, circum, L),
        )


kernel_code = r'''
extern "C" __global__
void transfer_drift(
    double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ z,
    const double* __restrict__ px,
    const double* __restrict__ py,
    const double* __restrict__ pz,
    const int* __restrict__ tag,
    int start_index,
    int stop_index,
    double beta,
    double gamma,
    double circum,
    double L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= stop_index) return;

    double r56 = L / (beta * beta * gamma * gamma);
    double c_half = 0.5 * circum;

    int mask = tag[i] > 0;

    x[i] += L * px[i] * mask;
    y[i] += L * py[i] * mask;
    z[i] += r56 * (pz[i] * beta) * beta * mask;

    int over = z[i] > c_half;
    int under = z[i] < -c_half;

    z[i] += (under - over) * circum;
}
'''

transfer_drift_gpu = cp.RawKernel(kernel_code, "transfer_drift")
