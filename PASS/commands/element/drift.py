from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.aperture import check_aperture_cpu, check_aperture_gpu

import numpy as np
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
        turn = sim.state.turn

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
                kernel = _get_transfer_drift_kernel(p.dtype)
                kernel(
                    (blocks, ),
                    (threads, ),
                    (p.x, p.y, p.z, p.px, p.py, p.dp, p.tag,
                     p.lost_position, p.lost_turn,
                     np.int32(start), np.int32(end),
                     p.real(beta * gamma), p.real(1.0 / gamma), p.real(L),
                     p.real(self.s), np.int32(turn)),
                )
            if N > 0:
                check_aperture_gpu(
                    beam, bunch, self.aperture_type, self.aperture_value,
                    self.s, turn,
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
        # Only particles that are alive on entry can become newly lost here.
        # Preserve the first loss location/turn for particles lost earlier.
        valid = pz_sq > real(0.0)
        alive = tag > 0
        lost_mask = alive & ~valid
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
    int* __restrict__ tag,
    float* __restrict__ lost_position,
    int* __restrict__ lost_turn,
    int start_index,
    int end_index,
    pass_real_t beta_gamma,
    pass_real_t inv_gamma,
    pass_real_t L,
    pass_real_t s_position,
    int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;

    if (tag[i] <= 0) {
        return;
    }

    pass_real_t one_plus_delta = (pass_real_t)1 + dp[i];
    pass_real_t px_i = px[i];
    pass_real_t py_i = py[i];
    pass_real_t pz_sq = one_plus_delta * one_plus_delta - px_i * px_i - py_i * py_i;
    bool valid = pz_sq > (pass_real_t)0;

    if (!valid) {
        tag[i] = -abs(tag[i]);
        lost_position[i] = (float)s_position;
        lost_turn[i] = turn;
        return;
    }

    // Algebraically equivalent to the CPU beta expression, but avoids a
    // second division and keeps all particle work in registers.
    pass_real_t inv_pz = (pass_real_t)1 / sqrt(pz_sq);
    pass_real_t bg = one_plus_delta * beta_gamma;
    pass_real_t dzeta_factor = sqrt((pass_real_t)1 + bg * bg) * inv_pz * inv_gamma;

    x[i] += L * px_i * inv_pz;
    y[i] += L * py_i * inv_pz;
    z[i] += L * ((pass_real_t)1 - dzeta_factor);

}
'''
DRIFT_SOURCE = CUDA_REAL_PREAMBLE + DRIFT_KERNEL_BODY
_transfer_drift_kernels = {}


def _get_transfer_drift_kernel(dtype):
    """Compile the CUDA kernel only when the GPU backend is actually used."""
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "GPU Drift tracking requires the optional 'cuda' dependencies "
            "(install PASS with the [cuda] extra)."
        ) from exc

    key = np.dtype(dtype)
    if key not in _transfer_drift_kernels:
        use_float = key == np.dtype(np.float32)
        _transfer_drift_kernels[key] = cp.RawKernel(
            DRIFT_SOURCE,
            "transfer_drift",
            options=("--std=c++14", f"-DPASS_USE_FLOAT={int(use_float)}"),
        )
    return _transfer_drift_kernels[key]
