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


@Command.register("twiss")
class Twiss(Command):

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id: int = beam_id
        self.s: float = kwargs["s (m)"]
        self.cmd_type: str = self.__class__.__name__
        self.cmd_name: str = kwargs["name"]

        self.s_previous: float = kwargs["s previous (m)"]

        self.alphax: float = kwargs["alpha x"]
        self.alphay: float = kwargs["alpha y"]
        self.alphax_previous: float = kwargs["alpha x previous"]
        self.alphay_previous: float = kwargs["alpha y previous"]

        self.betax: float = kwargs["beta x (m)"]
        self.betay: float = kwargs["beta y (m)"]
        self.betax_previous: float = kwargs["beta x previous (m)"]
        self.betay_previous: float = kwargs["beta y previous (m)"]

        self.mux: float = kwargs["mu x"]
        self.muy: float = kwargs["mu y"]
        self.mux_previous: float = kwargs["mu x previous"]
        self.muy_previous: float = kwargs["mu y previous"]

        self.Dx: float = kwargs["dx (m)"]
        self.Dx_previous: float = kwargs["dx previous (m)"]

        self.Dpx: float = kwargs["dpx"]
        self.Dpx_previous: float = kwargs["dpx previous"]

        self.DQx: float = kwargs["dqx"]
        self.DQy: float = kwargs["dqy"]

        self.longitudinal_transfer: str = kwargs["longitudinal transfer"].lower()

        self.muz: float = kwargs.get("mu z", 0.0)
        self.muz_previous: float = kwargs.get("mu z previous", 0.0)

        self.phi_x: float = (self.mux - self.mux_previous) * 2.0 * const.pi
        self.phi_y: float = (self.muy - self.muy_previous) * 2.0 * const.pi
        self.phi_z: float = (self.muz - self.muz_previous) * 2.0 * const.pi

        self.aperture_type: str = kwargs.get("aperture type", "off").lower()
        self.aperture_value: list = kwargs.get("aperture value", [])
        if not isinstance(self.aperture_value, list):
            raise ValueError(f"Aperture value of {self.cmd_name} must be a list, but got {type(self.aperture_value)}")

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(
            f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, S_previous={self.s_previous:.4f}, "
            f"alphax={self.alphax:.4f}, alphay={self.alphay:.4f}, alphax_previous={self.alphax_previous:.4f}, alphay_previous={self.alphay_previous:.4f}, "
            f"betax={self.betax:.4f}, betay={self.betay:.4f}, betax_previous={self.betax_previous:.4f}, betay_previous={self.betay_previous:.4f}, "
            f"mux={self.mux:.4f}, muy={self.muy:.4f}, muz={self.muz:.4f}, "
            f"Dx={self.Dx:.4f}, Dpx={self.Dpx:.4f}, "
            f"DQx={self.DQx:.4f}, DQy={self.DQy:.4f}, "
            f"longitudinal_transfer={self.longitudinal_transfer:s}, "
            f"ApertureType={self.aperture_type:s}, ApertureValue={self.aperture_value}")
        set_normal_logging()

    def execute_cpu(self, sim):

        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        turn = sim.state.turn

        for i, bunch in enumerate(bunches):
            twiss_transfer_cpu(self, beam, bunch)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
        return True

    def execute_gpu(self, sim):
        beam = sim.beams[self.beam_id]
        turn = sim.state.turn
        length = self.s - self.s_previous
        p = beam.particles
        real = p.real
        kernel = _get_twiss_kernel(p.dtype)
        threads = 256

        dq_x = self.DQx * 2.0 * const.pi
        dq_y = self.DQy * 2.0 * const.pi
        sbx = np.sqrt(self.betax * self.betax_previous)
        bx_ratio = np.sqrt(self.betax / self.betax_previous)
        bx_prev_ratio = np.sqrt(self.betax_previous / self.betax)
        sby = np.sqrt(self.betay * self.betay_previous)
        by_ratio = np.sqrt(self.betay / self.betay_previous)
        by_prev_ratio = np.sqrt(self.betay_previous / self.betay)

        for bunch in beam.bunches:
            start = bunch.start_idx
            end = bunch.end_idx
            n = end - start
            if n > 0:
                if self.longitudinal_transfer == "drift":
                    gammat = bunch.gamma_t
                    gamma = bunch.gamma
                    m11_z = 1.0
                    m12_z = -(1.0 / gammat**2 - 1.0 / gamma**2) * length
                    m21_z = 0.0
                    m22_z = 1.0
                elif self.longitudinal_transfer == "matrix":
                    m11_z = np.cos(self.phi_z)
                    m12_z = bunch.sigma_z / bunch.dp * np.sin(self.phi_z)
                    m21_z = -bunch.dp / bunch.sigma_z * np.sin(self.phi_z)
                    m22_z = m11_z
                else:
                    m11_z = 1.0
                    m12_z = m21_z = 0.0
                    m22_z = 1.0

                blocks = (n + threads - 1) // threads
                kernel(
                    (blocks,), (threads,),
                    (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag,
                     np.int32(start), np.int32(end),
                     real(m11_z), real(m12_z), real(m21_z), real(m22_z),
                     real(self.Dx_previous), real(self.Dpx_previous),
                     real(self.Dx), real(self.Dpx),
                     real(self.phi_x), real(self.phi_y),
                     real(dq_x), real(dq_y), real(sbx), real(bx_ratio),
                     real(bx_prev_ratio), real(sby), real(by_ratio),
                     real(by_prev_ratio),
                     real(self.alphax), real(self.alphax_previous),
                     real(self.alphay), real(self.alphay_previous)),
                )

            check_aperture_gpu(
                beam, bunch, self.aperture_type, self.aperture_value,
                self.s, turn,
            )
            if abs(length) >= const.eps:
                bunch.t0 += length / (bunch.beta * const.c)
        return True


def twiss_transfer_cpu(self, beam: Beam, bunch: BunchInfo):
    """6D linear optics transfer using Twiss parameters.

    Applies the longitudinal transfer (drift/matrix/identity), removes the
    previous dispersion, rotates by the phase advance, and adds the new
    dispersion.
    """
    length = self.s - self.s_previous
    if abs(length) >= const.eps:
        bunch.t0 += length / (bunch.beta * const.c)

    if self.longitudinal_transfer == "drift":
        gammat = bunch.gamma_t
        gamma = bunch.gamma
        m11_z = 1.0
        m12_z = -1.0 * (1.0 / gammat**2 - 1.0 / gamma**2) * length
        m21_z = 0.0
        m22_z = 1.0
    elif self.longitudinal_transfer == "matrix":
        sigmaz = bunch.sigma_z
        dp_bunch = bunch.dp
        m11_z = np.cos(self.phi_z)
        m12_z = sigmaz / dp_bunch * np.sin(self.phi_z)
        m21_z = -dp_bunch / sigmaz * np.sin(self.phi_z)
        m22_z = np.cos(self.phi_z)
    else:
        m11_z = 1.0
        m12_z = 0.0
        m21_z = 0.0
        m22_z = 1.0

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

    alive = tag > 0

    z2 = z * m11_z + dp * m12_z
    dp2 = z * m21_z + dp * m22_z

    x1 = x - self.Dx_previous * dp
    px1 = px - self.Dpx_previous * dp

    y1 = y
    py1 = py

    phi_x = self.phi_x + dp * self.DQx * 2.0 * const.pi
    phi_y = self.phi_y + dp * self.DQy * 2.0 * const.pi

    cx = np.cos(phi_x)
    sx = np.sin(phi_x)
    cy = np.cos(phi_y)
    sy = np.sin(phi_y)

    sqrt_betax_betaxprev = np.sqrt(self.betax * self.betax_previous)
    sqrt_betax_de_betaxprev = np.sqrt(self.betax / self.betax_previous)
    sqrt_betaxprev_de_betax = np.sqrt(self.betax_previous / self.betax)

    sqrt_betay_betayprev = np.sqrt(self.betay * self.betay_previous)
    sqrt_betay_de_betayprev = np.sqrt(self.betay / self.betay_previous)
    sqrt_betayprev_de_betay = np.sqrt(self.betay_previous / self.betay)

    m11_x = sqrt_betax_de_betaxprev * (cx + self.alphax_previous * sx)
    m12_x = sqrt_betax_betaxprev * sx
    m21_x = -(1.0 + self.alphax * self.alphax_previous) / sqrt_betax_betaxprev * sx \
        + (self.alphax_previous - self.alphax) / sqrt_betax_betaxprev * cx
    m22_x = sqrt_betaxprev_de_betax * (cx - self.alphax * sx)

    m11_y = sqrt_betay_de_betayprev * (cy + self.alphay_previous * sy)
    m12_y = sqrt_betay_betayprev * sy
    m21_y = -(1.0 + self.alphay * self.alphay_previous) / sqrt_betay_betayprev * sy \
        + (self.alphay_previous - self.alphay) / sqrt_betay_betayprev * cy
    m22_y = sqrt_betayprev_de_betay * (cy - self.alphay * sy)

    x2 = x1 * m11_x + px1 * m12_x + self.Dx * dp2
    px2 = x1 * m21_x + px1 * m22_x + self.Dpx * dp2

    y2 = y1 * m11_y + py1 * m12_y
    py2 = y1 * m21_y + py1 * m22_y

    # --- write back (only alive particles) ---
    z[:] = np.where(alive, z2, z)
    dp[:] = np.where(alive, dp2, dp)
    x[:] = np.where(alive, x2, x)
    px[:] = np.where(alive, px2, px)
    y[:] = np.where(alive, y2, y)
    py[:] = np.where(alive, py2, py)


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

TWISS_KERNEL_BODY = r'''
extern "C" __global__
void transfer_twiss(
    pass_real_t* __restrict__ x,
    pass_real_t* __restrict__ px,
    pass_real_t* __restrict__ y,
    pass_real_t* __restrict__ py,
    pass_real_t* __restrict__ z,
    pass_real_t* __restrict__ dp,
    const int* __restrict__ tag,
    int start_index, int end_index,
    pass_real_t m11_z, pass_real_t m12_z,
    pass_real_t m21_z, pass_real_t m22_z,
    pass_real_t dx_prev, pass_real_t dpx_prev,
    pass_real_t dx, pass_real_t dpx,
    pass_real_t phi_x, pass_real_t phi_y,
    pass_real_t dqx, pass_real_t dqy,
    pass_real_t sbx, pass_real_t bx_ratio, pass_real_t bx_prev_ratio,
    pass_real_t sby, pass_real_t by_ratio, pass_real_t by_prev_ratio,
    pass_real_t alphax, pass_real_t alphax_prev,
    pass_real_t alphay, pass_real_t alphay_prev)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index || tag[i] <= 0) return;

    pass_real_t xi = x[i], pxi = px[i];
    pass_real_t yi = y[i], pyi = py[i];
    pass_real_t zi = z[i], dpi = dp[i];

    pass_real_t z2 = zi * m11_z + dpi * m12_z;
    pass_real_t dp2 = zi * m21_z + dpi * m22_z;
    pass_real_t x1 = xi - dx_prev * dpi;
    pass_real_t px1 = pxi - dpx_prev * dpi;

    pass_real_t sx, cx, sy, cy;
    sincos(phi_x + dpi * dqx, &sx, &cx);
    sincos(phi_y + dpi * dqy, &sy, &cy);

    pass_real_t m11_x = bx_ratio * (cx + alphax_prev * sx);
    pass_real_t m12_x = sbx * sx;
    pass_real_t m21_x = (-(1 + alphax * alphax_prev) / sbx * sx
                         + (alphax_prev - alphax) / sbx * cx);
    pass_real_t m22_x = bx_prev_ratio * (cx - alphax * sx);

    pass_real_t m11_y = by_ratio * (cy + alphay_prev * sy);
    pass_real_t m12_y = sby * sy;
    pass_real_t m21_y = (-(1 + alphay * alphay_prev) / sby * sy
                         + (alphay_prev - alphay) / sby * cy);
    pass_real_t m22_y = by_prev_ratio * (cy - alphay * sy);

    x[i] = x1 * m11_x + px1 * m12_x + dx * dp2;
    px[i] = x1 * m21_x + px1 * m22_x + dpx * dp2;
    y[i] = yi * m11_y + pyi * m12_y;
    py[i] = yi * m21_y + pyi * m22_y;
    z[i] = z2;
    dp[i] = dp2;
}
'''

TWISS_SOURCE = CUDA_REAL_PREAMBLE + TWISS_KERNEL_BODY
_twiss_kernels = {}


def _get_twiss_kernel(dtype):
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "GPU Twiss tracking requires the optional 'cuda' dependencies."
        ) from exc
    key = np.dtype(dtype)
    if key not in _twiss_kernels:
        _twiss_kernels[key] = cp.RawKernel(
            TWISS_SOURCE, "transfer_twiss",
            options=("--std=c++14", f"-DPASS_USE_FLOAT={int(key == np.dtype(np.float32))}"),
        )
    return _twiss_kernels[key]
