from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.aperture import check_aperture_cpu

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

        self.aperture_type: str = kwargs.get("aperture type", "off").lower()
        self.aperture_value: list = kwargs.get("aperture value", [])
        if not isinstance(self.aperture_value, list):
            raise ValueError(f"Aperture value of {self.cmd_name} must be a list, but got {type(self.aperture_value)}")

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, Length={self.length:.4f}, "
                    f"ApertureType={self.aperture_type:s}, ApertureValue={self.aperture_value}")
        set_normal_logging()

    def execute_cpu(self, sim):

        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        turn = sim.state.turn

        for i, bunch in enumerate(bunches):
            self._track_drift_cpu(beam, bunch, turn)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)

    def execute_gpu(self, sim):
        L = self.length
        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches

        for i, bunch in enumerate(bunches):
            beta = bunch.beta
            gamma = bunch.gamma
            start = bunch.start_idx
            end = bunch.end_idx

            p = beam.particles  # slicing in the kernel

            N = end - start
            if N > 0 and np.abs(L) >= const.eps:
                threads = 256
                blocks = (N + threads - 1) // threads
                kernel = transfer_drift_kernel_f32 if p.dtype == np.dtype(np.float32) else transfer_drift_kernel_f64
                kernel(
                    (blocks, ),
                    (threads, ),
                    (p.x, p.y, p.z, p.px, p.py, p.dp, p.tag,
                     np.int32(start), np.int32(end),
                     p.real(beta), p.real(gamma), p.real(L)),
                )
            if abs(L) >= const.eps:
                bunch.t0 += L / (bunch.beta * const.c)


    def _track_drift_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):
        if np.abs(self.length) < const.eps:
            return

        start = bunch.start_idx
        end = bunch.end_idx

        p = beam.particles
        real = p.real
        L = real(self.length)
        s_position = self.s
        beta0 = real(bunch.beta)
        gamma0 = real(bunch.gamma)
        x = p.x[start:end]
        px = p.px[start:end]
        y = p.y[start:end]
        py = p.py[start:end]
        z = p.z[start:end]
        dp = p.dp[start:end]
        tag = p.tag[start:end]
        lost_position = p.lost_position[start:end]
        lost_turn = p.lost_turn[start:end]

        one = real(1.0)
        beta = (one + dp) * (gamma0 * beta0) / np.sqrt(one + ((one + dp) * (gamma0 * beta0))**2)
        pz_sq = (one + dp)**2 - px**2 - py**2
        valid = (pz_sq > real(0.0)) & (tag > 0)
        lost_mask = ~valid
        tag[lost_mask] = -np.abs(tag[lost_mask])
        lost_position[lost_mask] = s_position
        lost_turn[lost_mask] = turn
        pz_sq_safe = np.maximum(pz_sq, real(const.eps))
        pz = np.sqrt(pz_sq_safe)

        mask = (tag > 0).astype(p.dtype, copy=False)
        L_mask = L * mask

        x += L_mask * (px / pz)
        y += L_mask * (py / pz)
        z += L_mask * (one - (beta0 / beta) * (one + dp) / pz)

CUDA_REAL_PREAMBLE = r'''
#ifndef PASS_USE_FLOAT
#define PASS_USE_FLOAT 0
#endif

#if PASS_USE_FLOAT
using pass_real_t = float;
#else
using pass_real_t = double;
#endif
'''

DRIFT_KERNEL_BODY = r'''
extern "C" __global__
void transfer_drift(
    pass_real_t* __restrict__ x,
    pass_real_t* __restrict__ y,
    pass_real_t* __restrict__ z,
    const pass_real_t* __restrict__ px,
    const pass_real_t* __restrict__ py,
    const pass_real_t* __restrict__ dp,
    const int* __restrict__ tag,
    int start_index,
    int end_index,
    pass_real_t beta,
    pass_real_t gamma,
    pass_real_t L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;

    pass_real_t r56 = L / (beta * beta * gamma * gamma);
    int mask = tag[i] > 0;

    x[i] += L * px[i] * mask;
    y[i] += L * py[i] * mask;
    z[i] += r56 * (dp[i] * beta) * beta * mask;

}
'''
DRIFT_SOURCE = CUDA_REAL_PREAMBLE + DRIFT_KERNEL_BODY
transfer_drift_kernel_f32 = cp.RawKernel(
    DRIFT_SOURCE, "transfer_drift",
    options=("--std=c++14", "-DPASS_USE_FLOAT=1"),
)
transfer_drift_kernel_f64 = cp.RawKernel(
    DRIFT_SOURCE, "transfer_drift",
    options=("--std=c++14", "-DPASS_USE_FLOAT=0"),
)
