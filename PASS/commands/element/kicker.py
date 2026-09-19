from functools import lru_cache
import logging

import numpy as np

from PASS.commands.command import Command
from PASS.commands.element.error import FieldErrors
from PASS.utils.slicing import print_element_slicing, configure_element_slicing, run_body_slices
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const, _build_yoshida_cuda_constants
from PASS.utils.aperture import check_aperture_cpu, check_aperture_gpu

logger = logging.getLogger(__name__)


@Command.register("kicker")
class Kicker(Command):
    """Apply horizontal and vertical kicks, with exact drifts for a finite length."""

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = kwargs.get("length (m)", 0.0)
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]
        self.field_errors = FieldErrors(kwargs)

        if self.length < 0.0:
            raise ValueError(f"The length of Kicker {self.cmd_name} is {self.length}, which should be >= 0")
        if self.length > const.eps:
            self.is_thick = True
        else:
            self.is_thick = False

        # Kick strengths (radians), integrated dipole strength
        self.hkick: float = kwargs.get("hkick", 0.0)
        self.vkick: float = kwargs.get("vkick", 0.0)

        # Thick lens: compute per-unit-length strength
        if self.is_thick:
            self.hk = self.hkick / self.length
            self.vk = self.vkick / self.length
        else:
            self.hk = 0.0
            self.vk = 0.0

        if abs(self.hkick) < const.eps and abs(self.vkick) < const.eps:
            logger.warning(f"Kicker {self.cmd_name} has zero kick strength "
                           f"(hkick={self.hkick}, vkick={self.vkick}). "
                           f"It will act as a {'drift' if self.is_thick else 'marker'}.")

        self.num_slice = kwargs.get("num slices", 1)
        if self.num_slice < 1:
            logger.warning(f"The number of slices of {self.cmd_name} is {self.num_slice}, which should be >= 1. It has been changed to 1 now.")
            self.num_slice = 1

        self.integrator = kwargs.get("integrator", "adaptive")
        if self.integrator not in ["adaptive", "uniform", "yoshida4"]:
            raise ValueError(f"The integrator of Kicker {self.cmd_name} is {self.integrator}, which should be 'adaptive', 'uniform' or 'yoshida4'.")
        if self.integrator == "adaptive":
            self.integrator = "uniform"

        self.aperture_type: str = kwargs.get("aperture type", "off").lower()
        self.aperture_value: list = kwargs.get("aperture value", [])
        if not isinstance(self.aperture_value, list):
            raise ValueError(f"Aperture value of {self.cmd_name} must be a list, but got {type(self.aperture_value)}")

        configure_element_slicing(self, sim, kwargs)
        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, "
                    f"Length={self.length:.4f}, IsThick={self.is_thick}, "
                    f"Hkick={self.hkick:.6e}, Vkick={self.vkick:.6e}, "
                    f"NumSlice={self.num_slice:d}, Integrator={self.integrator:s}, "
                    f"ApertureType={self.aperture_type:s}, ApertureValue={self.aperture_value}")
        print_element_slicing(self)
        set_normal_logging()

    def execute_cpu(self, sim):
        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        turn = sim.state.turn

        for i, bunch in enumerate(bunches):
            self._track_kicker_cpu(beam, bunch, turn)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)
        return True

    def execute_gpu(self, sim):
        if self.field_errors.active:
            from PASS.commands.element.error import _track_field_errors_gpu
            return _track_field_errors_gpu(self, sim)
        if self._sc_nodes:
            from PASS.utils.slicing import execute_element_body_gpu
            return execute_element_body_gpu(self, sim)
        beam = sim.beams[self.beam_id]
        turn = sim.state.turn
        p = beam.particles
        kernel = _get_kicker_kernel(p.dtype.str)

        for bunch in beam.bunches:
            start = bunch.start_idx
            end = bunch.end_idx
            n = end - start
            if n > 0:
                threads = 256
                blocks = (n + threads - 1) // threads
                kernel(
                    (blocks, ),
                    (threads, ),
                    (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, p.lost_position, p.lost_turn, np.int32(start), np.int32(end),
                     p.real(bunch.beta * bunch.gamma), p.real(1.0 / bunch.gamma), np.float64(self.length), p.real(self.hkick), p.real(self.vkick),
                     p.real(self.hk), p.real(self.vk), np.int32(self.num_slice), np.int32(1 if self.integrator == "yoshida4" else 0), p.real(
                         self.s), np.int32(turn), np.int32(1 if self.is_thick else 0)),
                )
                check_aperture_gpu(
                    beam,
                    bunch,
                    self.aperture_type,
                    self.aperture_value,
                    self.s,
                    turn,
                )
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)
        return True

    def _track_kicker_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):

        beta0 = bunch.beta
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
        lost_position = p.lost_position[start:end]
        lost_turn = p.lost_turn[start:end]
        alive_before = tag > 0

        mask = (tag > 0).astype(np.float64)

        if not self.is_thick:
            self._dipole_kick_cpu(self.hkick, self.vkick, px, py, tag, mask)
            self.field_errors.kick_cpu(x, px, y, py, tag)
            return

        if self._sc_nodes:
            step = self._dkd_step_cpu if self.integrator == "uniform" else self._dkd_yoshida4_cpu

            def transport(ds, on_center):
                step(x, px, y, py, z, dp, tag, mask, ds, self.hk, self.vk, beta0, on_center=on_center)

            run_body_slices(self, beam, bunch, turn, transport)
            return

        if (abs(self.hkick) < const.eps and abs(self.vkick) < const.eps) and not self.field_errors.active:
            self._drift_exact_cpu(self.length, x, px, y, py, z, dp, tag, mask, beta0)
        else:
            ds = self.length / self.num_slice
            for _ in range(self.num_slice):
                if self.integrator == "uniform":
                    self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds, self.hk, self.vk, beta0)
                elif self.integrator == "yoshida4":
                    self._dkd_yoshida4_cpu(x, px, y, py, z, dp, tag, mask, ds, self.hk, self.vk, beta0)

        newly_lost = alive_before & (tag <= 0)
        lost_position[newly_lost] = self.s
        lost_turn[newly_lost] = turn

    def _dkd_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask, ds, hk, vk, beta0, on_center=None):
        """Compose three drift-kick-drift steps with Yoshida coefficients."""
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, hk, vk, beta0)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z0, hk, vk, beta0, on_center=on_center)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, hk, vk, beta0)

    def _dkd_step_cpu(self, x, px, y, py, z, dp, tag, mask, ds, hk, vk, beta0, on_center=None):
        """Apply one drift-kick-drift step; Yoshida composition may use negative ds."""
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._dipole_kick_cpu(hk * ds, vk * ds, px, py, tag, mask)
        self.field_errors.kick_cpu(x, px, y, py, tag, ds / self.length)
        if on_center is not None:
            on_center()
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)

    def _drift_exact_cpu(self, L, x, px, y, py, z, dp, tag, mask, beta0):
        """Advance live particles in a straight, field-free region."""
        if abs(L) < const.eps:
            return

        momentum_ratio = 1.0 + dp
        pz_sq = momentum_ratio**2 - px**2 - py**2

        valid = pz_sq > 0.0
        alive = tag > 0
        newly_lost = alive & ~valid
        tag[newly_lost] = -np.abs(tag[newly_lost])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)
        inv_pz = 1.0 / pz

        # Rationalize 1 - beta0/beta * p/pz to retain high-energy time slip.
        inv_gamma_sq = max(0.0, 1.0 - beta0**2)
        transverse_momentum_squared = px * px + py * py
        energy_ratio = np.sqrt(inv_gamma_sq + (1.0 - inv_gamma_sq) * momentum_ratio**2)
        slip = (dp * (2.0 + dp) * inv_gamma_sq - transverse_momentum_squared) / (pz * (pz + energy_ratio))

        # Refresh the mask before applying this drift so particles that just
        # became invalid do not receive a huge displacement from pz~0.
        mask[:] = (tag > 0).astype(mask.dtype, copy=False)
        L_mask = L * mask

        x += L_mask * px * inv_pz
        y += L_mask * py * inv_pz
        z += L_mask * slip

    def _dipole_kick_cpu(self, hkick_eff, vkick_eff, px, py, tag, mask):
        """Apply the integrated horizontal and vertical dipole kicks."""
        active = mask * (tag > 0)
        px += hkick_eff * active
        py += vkick_eff * active


CUDA_REAL_PREAMBLE = _build_yoshida_cuda_constants() + f'''
#ifndef PASS_USE_FLOAT
#define PASS_USE_FLOAT 0
#endif
#if PASS_USE_FLOAT
using pass_real_t = float;
#else
using pass_real_t = double;
#endif
#define PASS_EPS ((pass_real_t){const.eps:.17g})
'''

KICKER_KERNEL_BODY = r'''
__device__ inline bool kicker_drift(
    pass_real_t& x,
    pass_real_t& y,
    pass_real_t& z,
    const pass_real_t px,
    const pass_real_t py,
    const pass_real_t dp,
    int& tag,
    float* lost_position,
    int* lost_turn,
    pass_real_t reference_beta_gamma,
    pass_real_t inv_gamma,
    double length,
    pass_real_t s_position,
    int turn
) {
    if (tag <= 0)
        return false;
    pass_real_t momentum_ratio = (pass_real_t)1 + dp;
    pass_real_t pz_sq = momentum_ratio * momentum_ratio - px * px - py * py;
    if (!(pz_sq > (pass_real_t)0)) {
        tag = -abs(tag);
        lost_position[0] = (float)s_position;
        lost_turn[0] = turn;
        return false;
    }
    pass_real_t pz = sqrt(pz_sq);
    pass_real_t inv_pz = (pass_real_t)1 / pz;
    pass_real_t inv_gamma_sq = inv_gamma * inv_gamma;
    pass_real_t transverse_momentum_squared = px * px + py * py;
    pass_real_t energy_ratio = sqrt(inv_gamma_sq + ((pass_real_t)1 - inv_gamma_sq) * momentum_ratio * momentum_ratio);
    // Stable even when the physical time slip is much smaller than float epsilon.
    pass_real_t slip = (dp * ((pass_real_t)2 + dp) * inv_gamma_sq - transverse_momentum_squared) / (pz * (pz + energy_ratio));
    x += length * px * inv_pz;
    y += length * py * inv_pz;
    z += length * slip;
    return true;
}

extern "C" __global__ void transfer_kicker(
    pass_real_t* __restrict__ x,
    pass_real_t* __restrict__ px,
    pass_real_t* __restrict__ y,
    pass_real_t* __restrict__ py,
    pass_real_t* __restrict__ z,
    const pass_real_t* __restrict__ dp,
    int* __restrict__ tag,
    float* __restrict__ lost_position,
    int* __restrict__ lost_turn,
    int start_index,
    int end_index,
    pass_real_t reference_beta_gamma,
    pass_real_t inv_gamma,
    double length,
    pass_real_t hkick,
    pass_real_t vkick,
    pass_real_t hk,
    pass_real_t vk,
    int num_slice,
    int yoshida4,
    pass_real_t s_position,
    int turn,
    int is_thick
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index || tag[i] <= 0)
        return;

    if (!is_thick) {
        px[i] += hkick;
        py[i] += vkick;
        return;
    }

    pass_real_t xi = x[i], yi = y[i], zi = z[i];
    pass_real_t pxi = px[i], pyi = py[i], dpi = dp[i];
    int ti = tag[i];
    float lp = lost_position[i];
    int lt = lost_turn[i];

    if (abs(hkick) < PASS_EPS && abs(vkick) < PASS_EPS) {
        if (kicker_drift(xi, yi, zi, pxi, pyi, dpi, ti, &lp, &lt, reference_beta_gamma, inv_gamma, length, s_position, turn)) {
            x[i] = xi;
            y[i] = yi;
            z[i] = zi;
        }
        tag[i] = ti;
        lost_position[i] = lp;
        lost_turn[i] = lt;
        return;
    }

    const double ds = length / (double)num_slice;
    for (int slice = 0; slice < num_slice && ti > 0; ++slice) {
        int nsteps = yoshida4 ? 3 : 1;
        for (int step = 0; step < nsteps && ti > 0; ++step) {
            const double factor = yoshida4 ? (step == 1 ? pass_yoshida_z0 : pass_yoshida_z1) : 1.0;
            const double eff = ds * factor;
            if (!kicker_drift(xi, yi, zi, pxi, pyi, dpi, ti, &lp, &lt, reference_beta_gamma, inv_gamma, eff * (pass_real_t)0.5, s_position, turn))
                break;
            if (ti <= 0)
                break;
            pxi += hk * eff;
            pyi += vk * eff;
            if (!kicker_drift(xi, yi, zi, pxi, pyi, dpi, ti, &lp, &lt, reference_beta_gamma, inv_gamma, eff * (pass_real_t)0.5, s_position, turn))
                break;
        }
    }
    x[i] = xi;
    px[i] = pxi;
    y[i] = yi;
    py[i] = pyi;
    z[i] = zi;
    tag[i] = ti;
    lost_position[i] = lp;
    lost_turn[i] = lt;
}
'''

KICKER_SOURCE = CUDA_REAL_PREAMBLE + KICKER_KERNEL_BODY


@lru_cache(maxsize=None)
def _get_kicker_kernel(dtype):
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU Kicker tracking requires CUDA dependencies.") from exc
    dtype = np.dtype(dtype)
    return cp.RawKernel(
        KICKER_SOURCE,
        "transfer_kicker",
        options=("--std=c++14", f"-DPASS_USE_FLOAT={int(dtype == np.dtype(np.float32))}"),
    )
