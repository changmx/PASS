from functools import lru_cache
import logging

import numpy as np

from PASS.commands.command import Command
from PASS.utils.slicing import print_element_slicing, configure_element_slicing, run_body_slices, transport_with_center
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.aperture import check_aperture_cpu, check_aperture_gpu

logger = logging.getLogger(__name__)


def drift_factors(px, py, dp, inv_gamma_sq):
    """Forward straight-map factors shared with finite electrostatic septa.

    Return (valid, 1/p_s, dz/ds), with safe zero-momentum substitutes for
    invalid rows.  Do not clamp positive longitudinal momentum or fold z.
    """
    real = dp.dtype.type
    with np.errstate(over="ignore", invalid="ignore"):
        transverse = px * px + py * py
        longitudinal = (real(1) + dp)**2 - transverse
    valid = (dp > -1) & (longitudinal > 0) & np.isfinite(longitudinal)
    delta = np.where(valid, dp, real(0))
    transverse = np.where(valid, transverse, real(0))
    ps = np.sqrt(np.where(valid, longitudinal, real(1)))
    inv_g2 = real(inv_gamma_sq)
    energy = np.sqrt(inv_g2 + (real(1) - inv_g2) * (real(1) + delta)**2)
    slip = (delta * (real(2) + delta) * inv_g2 - transverse) / (ps * (ps + energy))
    return valid, real(1) / ps, slip


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

        configure_element_slicing(self, sim, kwargs)
        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, Length={self.length:.4f}, "
                    f"ApertureType={self.aperture_type:s}, ApertureValue={self.aperture_value}")
        print_element_slicing(self)
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
        return True

    def _track_drift_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):
        if self._sc_nodes or self.num_slice > 1:
            offset = 0.0

            def transport(ds, on_center):
                nonlocal offset

                def advance(length):
                    nonlocal offset
                    offset += length
                    self._drift_segment_cpu(beam, bunch, turn, length, self.s - self.length + offset)

                transport_with_center(advance, ds, on_center)

            run_body_slices(self, beam, bunch, turn, transport)
        else:
            self._drift_segment_cpu(beam, bunch, turn, self.length, self.s)

    def _drift_segment_cpu(self, beam, bunch, turn, length, s_position):
        if np.abs(length) < const.eps:
            return

        start = bunch.start_idx
        end = bunch.end_idx

        p = beam.particles
        real = p.real
        L = real(length)
        inv_gamma_sq = real(1.0 / bunch.gamma**2)
        x = p.x[start:end]
        px = p.px[start:end]
        y = p.y[start:end]
        py = p.py[start:end]
        z = p.z[start:end]
        dp = p.dp[start:end]
        tag = p.tag[start:end]
        lost_position = p.lost_position[start:end]
        lost_turn = p.lost_turn[start:end]

        valid, inv_ps, slip = drift_factors(px, py, dp, inv_gamma_sq)
        # Only particles that are alive on entry can become newly lost here.
        # Preserve the first loss location/turn for particles lost earlier.
        alive = tag > 0
        newly_lost = alive & ~valid
        tag[newly_lost] = -np.abs(tag[newly_lost])
        lost_position[newly_lost] = s_position
        lost_turn[newly_lost] = turn
        active = tag > 0
        x[active] += L * px[active] * inv_ps[active]
        y[active] += L * py[active] * inv_ps[active]
        z[active] += L * slip[active]

    def execute_gpu(self, sim):
        if self._sc_nodes:
            from PASS.utils.slicing import execute_internal_sc_gpu
            return execute_internal_sc_gpu(self, sim)
        L = self.length
        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        turn = sim.state.turn

        for i, bunch in enumerate(bunches):
            gamma = bunch.gamma
            start = bunch.start_idx
            end = bunch.end_idx

            p = beam.particles  # slicing in the kernel

            n = end - start
            if n > 0 and np.abs(L) >= const.eps:
                threads = 256
                blocks = (n + threads - 1) // threads
                kernel = _get_transfer_drift_kernel(p.dtype.str)
                ds = L / self.num_slice
                for j in range(self.num_slice):
                    kernel(
                        (blocks, ),
                        (threads, ),
                        (p.x, p.y, p.z, p.px, p.py, p.dp, p.tag, p.lost_position, p.lost_turn, np.int32(start), np.int32(end), p.real(
                            1.0 / gamma**2), p.real(ds), p.real(self.s - L + (j + 1) * ds), np.int32(turn)),
                    )
            if n > 0:
                check_aperture_gpu(
                    beam,
                    bunch,
                    self.aperture_type,
                    self.aperture_value,
                    self.s,
                    turn,
                )
            if abs(L) >= const.eps:
                bunch.t0 += L / (bunch.beta * const.c)
        return True


def drift_cuda_factors():
    """Same stable formula for inclusion in fused CUDA maps (pass_real_t)."""
    return r'''
__device__ inline bool pass_drift_factors(
    pass_real_t px,
    pass_real_t py,
    pass_real_t dp,
    pass_real_t inv_g2,
    pass_real_t& inv_ps,
    pass_real_t& slip
) {
    pass_real_t transverse = px * px + py * py, ratio = (pass_real_t)1 + dp;
    pass_real_t longitudinal = ratio * ratio - transverse;
    if (!(dp > (pass_real_t)-1) || !(longitudinal > (pass_real_t)0) || !isfinite(longitudinal))
        return false;
    pass_real_t ps = sqrt(longitudinal);
    pass_real_t energy = sqrt(inv_g2 + ((pass_real_t)1 - inv_g2) * ratio * ratio);
    inv_ps = (pass_real_t)1 / ps;
    slip = (dp * ((pass_real_t)2 + dp) * inv_g2 - transverse) / (ps * (ps + energy));
    return true;
}
'''


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
extern "C" __global__ void transfer_drift(
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
    pass_real_t inv_gamma_sq,
    pass_real_t L,
    pass_real_t s_position,
    int turn
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index)
        return;

    if (tag[i] <= 0) {
        return;
    }

    pass_real_t px_i = px[i];
    pass_real_t py_i = py[i];
    pass_real_t inv_pz, slip;
    bool valid = pass_drift_factors(px_i, py_i, dp[i], inv_gamma_sq, inv_pz, slip);

    if (!valid) {
        tag[i] = -abs(tag[i]);
        lost_position[i] = (float)s_position;
        lost_turn[i] = turn;
        return;
    }

    x[i] += L * px_i * inv_pz;
    y[i] += L * py_i * inv_pz;
    z[i] += L * slip;
}
'''
DRIFT_SOURCE = CUDA_REAL_PREAMBLE + drift_cuda_factors() + DRIFT_KERNEL_BODY


@lru_cache(maxsize=None)
def _get_transfer_drift_kernel(dtype):
    """Compile the CUDA kernel only when the GPU backend is actually used."""
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU Drift tracking requires the optional 'cuda' dependencies "
                           "(install PASS with the [cuda] extra).") from exc

    dtype = np.dtype(dtype)
    use_float = dtype == np.dtype(np.float32)
    return cp.RawKernel(
        DRIFT_SOURCE,
        "transfer_drift",
        options=("--std=c++14", f"-DPASS_USE_FLOAT={int(use_float)}"),
    )
