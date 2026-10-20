from functools import lru_cache
import logging
import math

import numpy as np

from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.aperture import check_aperture_cpu, check_aperture_gpu

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

    def _get_gpu_double_optics(self):
        # Scalar types matter: NumPy float32 inputs can round the original beta and DQ expressions differently.
        key = tuple(
            (type(value), float(value))
            for value in (self.alphax, self.alphax_previous, self.alphay, self.alphay_previous, self.betax, self.betax_previous, self.betay,
                          self.betay_previous, self.phi_x, self.phi_y, self.DQx, self.DQy, self.Dx_previous, self.Dpx_previous, self.Dx, self.Dpx))
        cached = getattr(self, "_gpu_double_optics_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]

        # Preserve the original sqrt rounding, then use double precision for the new host arithmetic.
        sbx = float(np.sqrt(self.betax * self.betax_previous))
        bx_ratio = float(np.sqrt(self.betax / self.betax_previous))
        bx_prev_ratio = float(np.sqrt(self.betax_previous / self.betax))
        sby = float(np.sqrt(self.betay * self.betay_previous))
        by_ratio = float(np.sqrt(self.betay / self.betay_previous))
        by_prev_ratio = float(np.sqrt(self.betay_previous / self.betay))
        alphax, alphax_prev, alphay, alphay_prev = (float(value) for value in (self.alphax, self.alphax_previous, self.alphay, self.alphay_previous))
        phi_x, phi_y = float(self.phi_x), float(self.phi_y)
        dq_x = float(self.DQx * 2.0 * const.pi)
        dq_y = float(self.DQy * 2.0 * const.pi)
        m21_sin_x = -(1.0 + alphax * alphax_prev) / sbx
        m21_cos_x = (alphax_prev - alphax) / sbx
        m21_sin_y = -(1.0 + alphay * alphay_prev) / sby
        m21_cos_y = (alphay_prev - alphay) / sby
        fixed_x = fixed_y = (0.0, 0.0, 0.0, 0.0)
        if self.DQx == 0.0:
            cx, sx = (math.cos(phi_x), math.sin(phi_x)) if math.isfinite(phi_x) else (math.nan, math.nan)
            fixed_x = (bx_ratio * (cx + alphax_prev * sx), sbx * sx, m21_sin_x * sx + m21_cos_x * cx, bx_prev_ratio * (cx - alphax * sx))
        if self.DQy == 0.0:
            cy, sy = (math.cos(phi_y), math.sin(phi_y)) if math.isfinite(phi_y) else (math.nan, math.nan)
            fixed_y = (by_ratio * (cy + alphay_prev * sy), sby * sy, m21_sin_y * sy + m21_cos_y * cy, by_prev_ratio * (cy - alphay * sy))
        double_args = (m21_sin_x, m21_cos_x, m21_sin_y, m21_cos_y, *fixed_x, *fixed_y)
        optics_args = (float(self.Dx_previous), float(self.Dpx_previous), float(self.Dx), float(self.Dpx), phi_x, phi_y, dq_x, dq_y, sbx, bx_ratio,
                       bx_prev_ratio, sby, by_ratio, by_prev_ratio, alphax, alphax_prev, alphay, alphay_prev)
        self._gpu_double_optics_cache = (key, double_args, optics_args)
        return double_args, optics_args

    def execute_gpu(self, sim):
        beam = sim.beams[self.beam_id]
        turn = sim.state.turn
        length = self.s - self.s_previous
        p = beam.particles
        real = p.real
        use_double = p.dtype == np.dtype(np.float64)
        has_in_dispersion = self.Dx_previous != 0.0 or self.Dpx_previous != 0.0
        has_out_dispersion = self.Dx != 0.0 or self.Dpx != 0.0
        has_dispersion = has_in_dispersion or has_out_dispersion
        kernel = _get_twiss_kernel(p.dtype.str, has_dispersion)
        if has_dispersion:
            dispersion_flags = (np.int32(has_in_dispersion), np.int32(has_out_dispersion), np.int32(self.longitudinal_transfer == "matrix"))
        threads = 256

        if use_double:
            double_args, optics_args = self._get_gpu_double_optics()
        else:
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
                if use_double:
                    kernel_args = (*double_args, p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, np.int32(start), np.int32(end), real(m11_z), real(m12_z),
                                   real(m21_z), real(m22_z), *optics_args)
                else:
                    kernel_args = (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, np.int32(start), np.int32(end), real(m11_z), real(m12_z), real(m21_z),
                                   real(m22_z), real(self.Dx_previous), real(self.Dpx_previous), real(self.Dx), real(self.Dpx), real(self.phi_x),
                                   real(self.phi_y), real(dq_x), real(dq_y), real(sbx), real(bx_ratio), real(bx_prev_ratio), real(sby),
                                   real(by_ratio), real(by_prev_ratio), real(self.alphax), real(self.alphax_previous), real(self.alphay),
                                   real(self.alphay_previous))
                if has_dispersion:
                    kernel_args = (real(1.0 / bunch.gamma**2), *dispersion_flags, *kernel_args)
                kernel(
                    (blocks, ),
                    (threads, ),
                    kernel_args,
                )

            check_aperture_gpu(
                beam,
                bunch,
                self.aperture_type,
                self.aperture_value,
                self.s,
                turn,
            )
            if abs(length) >= const.eps:
                bunch.t0 += length / (bunch.beta * const.c)
        return True


def twiss_transfer_cpu(self, beam: Beam, bunch: BunchInfo):
    """Optics transfer using Twiss parameters and dispersion coordinate changes.

    Removes entrance dispersion including its longitudinal companion, applies
    the prescribed longitudinal and transverse maps, and restores dispersion.
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
    has_in_dispersion = self.Dx_previous != 0.0 or self.Dpx_previous != 0.0
    has_out_dispersion = self.Dx != 0.0 or self.Dpx != 0.0
    if has_in_dispersion or has_out_dispersion:
        inv_gamma_sq = 1.0 / bunch.gamma**2
    sqrt_betax_betaxprev = np.sqrt(self.betax * self.betax_previous)
    sqrt_betax_de_betaxprev = np.sqrt(self.betax / self.betax_previous)
    sqrt_betaxprev_de_betax = np.sqrt(self.betax_previous / self.betax)

    sqrt_betay_betayprev = np.sqrt(self.betay * self.betay_previous)
    sqrt_betay_de_betayprev = np.sqrt(self.betay / self.betay_previous)
    sqrt_betayprev_de_betay = np.sqrt(self.betay_previous / self.betay)

    m21_sin_x = -(1.0 + self.alphax * self.alphax_previous) / sqrt_betax_betaxprev
    m21_cos_x = (self.alphax_previous - self.alphax) / sqrt_betax_betaxprev
    m21_sin_y = -(1.0 + self.alphay * self.alphay_previous) / sqrt_betay_betayprev
    m21_cos_y = (self.alphay_previous - self.alphay) / sqrt_betay_betayprev
    zero_dqx = self.DQx == 0.0
    zero_dqy = self.DQy == 0.0
    if zero_dqx or zero_dqy:
        # A separate zero array preserves the particle dtype without reading a possibly lost first particle.
        zero_dp = np.zeros(1, dtype=p.dtype)
    if zero_dqx:
        phi_x = self.phi_x + zero_dp * self.DQx * 2.0 * const.pi
        cx, sx = np.cos(phi_x), np.sin(phi_x)
        fixed_x = (sqrt_betax_de_betaxprev * (cx + self.alphax_previous * sx), sqrt_betax_betaxprev * sx, m21_sin_x * sx + m21_cos_x * cx,
                   sqrt_betaxprev_de_betax * (cx - self.alphax * sx))
    if zero_dqy:
        phi_y = self.phi_y + zero_dp * self.DQy * 2.0 * const.pi
        cy, sy = np.cos(phi_y), np.sin(phi_y)
        fixed_y = (sqrt_betay_de_betayprev * (cy + self.alphay_previous * sy), sqrt_betay_betayprev * sy, m21_sin_y * sy + m21_cos_y * cy,
                   sqrt_betayprev_de_betay * (cy - self.alphay * sy))

    # Keep the working arrays in cache; advance the reference clock only once above.
    for chunk_start in range(start, end, 8192):
        chunk_end = min(chunk_start + 8192, end)
        x = p.x[chunk_start:chunk_end]
        px = p.px[chunk_start:chunk_end]
        y = p.y[chunk_start:chunk_end]
        py = p.py[chunk_start:chunk_end]
        z = p.z[chunk_start:chunk_end]
        dp = p.dp[chunk_start:chunk_end]
        tag = p.tag[chunk_start:chunk_end]
        alive = tag > 0

        z1 = z
        if has_in_dispersion:
            momentum_ratio = 1.0 + dp
            beta0_over_beta = (1.0 - inv_gamma_sq) * momentum_ratio * momentum_ratio + inv_gamma_sq
            np.sqrt(beta0_over_beta, out=beta0_over_beta)
            beta0_over_beta /= momentum_ratio
            # Positive z means earlier arrival; delta's canonical derivative is beta0/beta.
            z1 = z + beta0_over_beta * (self.Dpx_previous * x - self.Dx_previous * px)

        z2 = z1 * m11_z + dp * m12_z
        dp2 = z1 * m21_z + dp * m22_z
        x1 = x - self.Dx_previous * dp
        px1 = px - self.Dpx_previous * dp

        # Retain the original NaN/Inf propagation when zero chromaticity multiplies nonfinite dp.
        finite_dp = (zero_dqx or zero_dqy) and np.isfinite(dp).all()
        if zero_dqx and finite_dp:
            m11_x, m12_x, m21_x, m22_x = fixed_x
        else:
            phi_x = self.phi_x + dp * self.DQx * 2.0 * const.pi
            cx, sx = np.cos(phi_x), np.sin(phi_x)
            m11_x = sqrt_betax_de_betaxprev * (cx + self.alphax_previous * sx)
            m12_x = sqrt_betax_betaxprev * sx
            m21_x = m21_sin_x * sx + m21_cos_x * cx
            m22_x = sqrt_betaxprev_de_betax * (cx - self.alphax * sx)
        if zero_dqy and finite_dp:
            m11_y, m12_y, m21_y, m22_y = fixed_y
        else:
            phi_y = self.phi_y + dp * self.DQy * 2.0 * const.pi
            cy, sy = np.cos(phi_y), np.sin(phi_y)
            m11_y = sqrt_betay_de_betayprev * (cy + self.alphay_previous * sy)
            m12_y = sqrt_betay_betayprev * sy
            m21_y = m21_sin_y * sy + m21_cos_y * cy
            m22_y = sqrt_betayprev_de_betay * (cy - self.alphay * sy)

        if has_out_dispersion:
            x2 = x1 * m11_x + px1 * m12_x
            px2 = x1 * m21_x + px1 * m22_x
            if not has_in_dispersion or self.longitudinal_transfer == "matrix":
                momentum_ratio = 1.0 + dp2
                beta0_over_beta = (1.0 - inv_gamma_sq) * momentum_ratio * momentum_ratio + inv_gamma_sq
                np.sqrt(beta0_over_beta, out=beta0_over_beta)
                beta0_over_beta /= momentum_ratio
            # x2 and px2 still contain only betatron motion here.
            z2 += beta0_over_beta * (self.Dx * px2 - self.Dpx * x2)
            x2 += self.Dx * dp2
            px2 += self.Dpx * dp2
        else:
            x2 = x1 * m11_x + px1 * m12_x + self.Dx * dp2
            px2 = x1 * m21_x + px1 * m22_x + self.Dpx * dp2

        y2 = y * m11_y + py * m12_y
        py2 = y * m21_y + py * m22_y

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
extern "C" __global__ void transfer_twiss(
#if PASS_HAS_DISPERSION
    pass_real_t inv_gamma_sq,
    int has_in_dispersion,
    int has_out_dispersion,
    int longitudinal_matrix,
#endif
#if !PASS_USE_FLOAT
    pass_real_t m21_sin_x,
    pass_real_t m21_cos_x,
    pass_real_t m21_sin_y,
    pass_real_t m21_cos_y,
    pass_real_t fixed_m11_x,
    pass_real_t fixed_m12_x,
    pass_real_t fixed_m21_x,
    pass_real_t fixed_m22_x,
    pass_real_t fixed_m11_y,
    pass_real_t fixed_m12_y,
    pass_real_t fixed_m21_y,
    pass_real_t fixed_m22_y,
#endif
    pass_real_t* __restrict__ x,
    pass_real_t* __restrict__ px,
    pass_real_t* __restrict__ y,
    pass_real_t* __restrict__ py,
    pass_real_t* __restrict__ z,
    pass_real_t* __restrict__ dp,
    const int* __restrict__ tag,
    int start_index,
    int end_index,
    pass_real_t m11_z,
    pass_real_t m12_z,
    pass_real_t m21_z,
    pass_real_t m22_z,
    pass_real_t dx_prev,
    pass_real_t dpx_prev,
    pass_real_t dx,
    pass_real_t dpx,
    pass_real_t phi_x,
    pass_real_t phi_y,
    pass_real_t dqx,
    pass_real_t dqy,
    pass_real_t sbx,
    pass_real_t bx_ratio,
    pass_real_t bx_prev_ratio,
    pass_real_t sby,
    pass_real_t by_ratio,
    pass_real_t by_prev_ratio,
    pass_real_t alphax,
    pass_real_t alphax_prev,
    pass_real_t alphay,
    pass_real_t alphay_prev
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index || tag[i] <= 0)
        return;

    pass_real_t xi = x[i], pxi = px[i];
    pass_real_t yi = y[i], pyi = py[i];
    pass_real_t zi = z[i], dpi = dp[i];

#if PASS_HAS_DISPERSION
    pass_real_t beta0_over_beta = 1;
    if (has_in_dispersion) {
        pass_real_t momentum_ratio = 1 + dpi;
        beta0_over_beta = sqrt(inv_gamma_sq + (1 - inv_gamma_sq) * momentum_ratio * momentum_ratio) / momentum_ratio;
        zi += beta0_over_beta * (dpx_prev * xi - dx_prev * pxi);
    }
#endif
    pass_real_t z2 = zi * m11_z + dpi * m12_z;
    pass_real_t dp2 = zi * m21_z + dpi * m22_z;
    pass_real_t x1 = xi - dx_prev * dpi;
    pass_real_t px1 = pxi - dpx_prev * dpi;

#if PASS_USE_FLOAT
    pass_real_t sx, cx, sy, cy;
    sincos(phi_x + dpi * dqx, &sx, &cx);
    sincos(phi_y + dpi * dqy, &sy, &cy);

    pass_real_t m11_x = bx_ratio * (cx + alphax_prev * sx);
    pass_real_t m12_x = sbx * sx;
    pass_real_t m21_x = (-(1 + alphax * alphax_prev) / sbx * sx + (alphax_prev - alphax) / sbx * cx);
    pass_real_t m22_x = bx_prev_ratio * (cx - alphax * sx);

    pass_real_t m11_y = by_ratio * (cy + alphay_prev * sy);
    pass_real_t m12_y = sby * sy;
    pass_real_t m21_y = (-(1 + alphay * alphay_prev) / sby * sy + (alphay_prev - alphay) / sby * cy);
    pass_real_t m22_y = by_prev_ratio * (cy - alphay * sy);
#else
    pass_real_t m11_x, m12_x, m21_x, m22_x;
    if (dqx == 0 && isfinite(dpi)) {
        m11_x = fixed_m11_x;
        m12_x = fixed_m12_x;
        m21_x = fixed_m21_x;
        m22_x = fixed_m22_x;
    } else {
        pass_real_t sx, cx;
        sincos(phi_x + dpi * dqx, &sx, &cx);
        m11_x = bx_ratio * (cx + alphax_prev * sx);
        m12_x = sbx * sx;
        m21_x = m21_sin_x * sx + m21_cos_x * cx;
        m22_x = bx_prev_ratio * (cx - alphax * sx);
    }
    pass_real_t m11_y, m12_y, m21_y, m22_y;
    if (dqy == 0 && isfinite(dpi)) {
        m11_y = fixed_m11_y;
        m12_y = fixed_m12_y;
        m21_y = fixed_m21_y;
        m22_y = fixed_m22_y;
    } else {
        pass_real_t sy, cy;
        sincos(phi_y + dpi * dqy, &sy, &cy);
        m11_y = by_ratio * (cy + alphay_prev * sy);
        m12_y = sby * sy;
        m21_y = m21_sin_y * sy + m21_cos_y * cy;
        m22_y = by_prev_ratio * (cy - alphay * sy);
    }
#endif

#if PASS_HAS_DISPERSION
    pass_real_t x2 = x1 * m11_x + px1 * m12_x;
    pass_real_t px2 = x1 * m21_x + px1 * m22_x;
    if (has_out_dispersion) {
        if (!has_in_dispersion || longitudinal_matrix) {
            pass_real_t momentum_ratio = 1 + dp2;
            beta0_over_beta = sqrt(inv_gamma_sq + (1 - inv_gamma_sq) * momentum_ratio * momentum_ratio) / momentum_ratio;
        }
        z2 += beta0_over_beta * (dx * px2 - dpx * x2);
    }
    x[i] = x2 + dx * dp2;
    px[i] = px2 + dpx * dp2;
#else
    x[i] = x1 * m11_x + px1 * m12_x + dx * dp2;
    px[i] = x1 * m21_x + px1 * m22_x + dpx * dp2;
#endif
    y[i] = yi * m11_y + pyi * m12_y;
    py[i] = yi * m21_y + pyi * m22_y;
    z[i] = z2;
    dp[i] = dp2;
}
'''

TWISS_SOURCE = CUDA_REAL_PREAMBLE + TWISS_KERNEL_BODY


@lru_cache(maxsize=None)
def _get_twiss_kernel(dtype, has_dispersion):
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU Twiss tracking requires the optional 'cuda' dependencies.") from exc
    dtype = np.dtype(dtype)
    return cp.RawKernel(
        TWISS_SOURCE,
        "transfer_twiss",
        options=("--std=c++14", f"-DPASS_USE_FLOAT={int(dtype == np.dtype(np.float32))}", f"-DPASS_HAS_DISPERSION={int(has_dispersion)}"),
    )
