from functools import lru_cache
import logging

import numpy as np

from PASS.commands.command import Command
from PASS.commands.element.error import AlignmentErrors, FieldErrors
from PASS.utils.aperture import check_aperture_cpu, check_aperture_gpu
from PASS.utils.slicing import print_element_slicing, configure_element_slicing, run_body_slices
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const, _build_yoshida_cuda_constants

logger = logging.getLogger(__name__)


@Command.register("multipole")
class Multipole(Command):
    """Track normal and skew multipoles using Horner evaluation.

    A zero-length element applies one integrated kick. Thick elements use
    exact drift-kick-drift slices with uniform or Yoshida integration."""

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = kwargs["length (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]
        self.field_errors = FieldErrors(kwargs)
        self.alignment_errors = AlignmentErrors(kwargs)

        if self.length < 0.0:
            raise ValueError(f"The length of Multipole {self.cmd_name} is {self.length}, which should be >= 0")
        if self.length > const.eps:
            self.is_thick = True
        else:
            self.is_thick = False

        knl_list = kwargs.get("kil", [])
        ksl_list = kwargs.get("kisl", [])

        if not isinstance(knl_list, (list, np.ndarray)):
            raise ValueError(f"KiL of {self.cmd_name} must be a list, but got {type(knl_list)}")
        if not isinstance(ksl_list, (list, np.ndarray)):
            raise ValueError(f"KiSL of {self.cmd_name} must be a list, but got {type(ksl_list)}")

        knl_list, ksl_list = self.field_errors.combine(knl_list, ksl_list)
        self.knl = np.array(knl_list, dtype=np.float64)
        self.ksl = np.array(ksl_list, dtype=np.float64)

        # Order = max(len(knl), len(ksl)) - 1; pad the shorter array with zeros
        len_n = len(self.knl)
        len_s = len(self.ksl)
        if len_n == 0 and len_s == 0:
            raise ValueError(f"Multipole {self.cmd_name} has empty KiL and KiSL. At least one component is required.")

        self.order = max(len_n, len_s) - 1

        if len_n > len_s:
            self.ksl = np.pad(self.ksl, (0, len_n - len_s), mode='constant')
        elif len_n < len_s:
            self.knl = np.pad(self.knl, (0, len_s - len_n), mode='constant')

        # Thick lens: compute per-unit-length strength
        if self.is_thick:
            self.kn = self.knl / self.length
            self.ks = self.ksl / self.length
        else:
            self.kn = np.zeros_like(self.knl)
            self.ks = np.zeros_like(self.ksl)

        all_zero = np.all(self.knl == 0) and np.all(self.ksl == 0)
        if all_zero:
            logger.warning(f"Multipole {self.cmd_name} has zero integrated strength (all knl/ksl are zero). It will act as a pure drift.")

        # Precompute inverse factorials: inv_fact[n] = 1/n!
        self.inv_fact = np.ones(self.order + 1)
        for n in range(1, self.order + 1):
            self.inv_fact[n] = self.inv_fact[n - 1] / n

        self.num_slice = kwargs.get("num slices", 1)
        if self.num_slice < 1:
            logger.warning(f"The number of slices of {self.cmd_name} is {self.num_slice}, which should be >= 1. It has been changed to 1 now.")
            self.num_slice = 1

        self.integrator = kwargs.get("integrator", "adaptive")
        if self.integrator not in ["adaptive", "uniform", "yoshida4"]:
            raise ValueError(
                f"The integrator of Multipole {self.cmd_name} is {self.integrator}, which should be 'adaptive', 'uniform' or 'yoshida4'.")
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
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, Length={self.length:.4f}, "
                    f"IsThick={self.is_thick}, Order={self.order:d}, "
                    f"KnL={np.array2string(self.knl, precision=6)}, "
                    f"KsL={np.array2string(self.ksl, precision=6)}, "
                    f"NumSlice={self.num_slice:d}, Integrator={self.integrator:s}, "
                    f"ApertureType={self.aperture_type:s}, ApertureValue={self.aperture_value}")
        print_element_slicing(self)
        set_normal_logging()

    def execute_cpu(self, sim):
        beam = sim.beams[self.beam_id]
        turn = sim.state.turn
        masks = self.alignment_errors.enter_frame(self, beam, turn)
        try:
            for bunch in beam.bunches:
                self._track_multipole_cpu(beam, bunch, turn)
        finally:
            self.alignment_errors.exit_frame(self, beam, turn, masks)
        for bunch in beam.bunches:
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)
        return True

    def execute_gpu(self, sim):
        beam = sim.beams[self.beam_id]
        turn = sim.state.turn
        masks = self.alignment_errors.enter_frame(self, beam, turn, gpu=True)
        try:
            if self._sc_nodes:
                from PASS.utils.slicing import execute_element_body_gpu
                execute_element_body_gpu(self, sim)
            else:
                all_zero = (np.all(self.knl == 0) and np.all(self.ksl == 0))
                mode = 0 if not self.is_thick else (2 if all_zero else 1)
                launch_multipole(self, sim, self.knl if not self.is_thick else self.kn, self.ksl if not self.is_thick else self.ks, self.inv_fact,
                                 mode)
        finally:
            self.alignment_errors.exit_frame(self, beam, turn, masks, gpu=True)
        for bunch in beam.bunches:
            check_aperture_gpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)
        return True

    def _track_multipole_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):

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

        alive_before = tag > 0

        # chi = q/q0 * m0/m  (for same-species beam, chi = 1)
        chi = 1.0

        mask = (tag > 0).astype(np.float64)

        if not self.is_thick:
            self._multipole_kick_cpu(self.knl, self.ksl, x, px, y, py, tag, mask, chi)
            return

        if self._sc_nodes:
            step = self._dkd_step_cpu if self.integrator == "uniform" else self._dkd_yoshida4_cpu

            def transport(ds, on_center):
                step(x, px, y, py, z, dp, tag, mask, ds, self.kn, self.ks, chi, beta0, on_center=on_center)

            run_body_slices(self, beam, bunch, turn, transport)
            return

        all_zero = np.all(self.knl == 0) and np.all(self.ksl == 0)
        if all_zero:
            self._drift_exact_cpu(self.length, x, px, y, py, z, dp, tag, mask, beta0)
        else:
            ds = self.length / self.num_slice
            for _ in range(self.num_slice):
                if self.integrator == "uniform":
                    self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds, self.kn, self.ks, chi, beta0)
                elif self.integrator == "yoshida4":
                    self._dkd_yoshida4_cpu(x, px, y, py, z, dp, tag, mask, ds, self.kn, self.ks, chi, beta0)

        newly_lost = alive_before & (tag < 0)
        if np.any(newly_lost):
            lost_position = p.lost_position[start:end]
            lost_turn = p.lost_turn[start:end]
            lost_position[newly_lost] = self.s
            lost_turn[newly_lost] = turn

    def _dkd_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask, ds, kn, ks, chi, beta0, on_center=None):
        """Compose three drift-kick-drift steps with Yoshida coefficients."""
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, kn, ks, chi, beta0)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z0, kn, ks, chi, beta0, on_center=on_center)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, kn, ks, chi, beta0)

    def _dkd_step_cpu(self, x, px, y, py, z, dp, tag, mask, ds, kn, ks, chi, beta0, on_center=None):
        """Apply one drift-kick-drift step; Yoshida composition may use negative ds."""
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._multipole_kick_cpu(kn * ds, ks * ds, x, px, y, py, tag, mask, chi)
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
        tag[alive & ~valid] = -np.abs(tag[alive & ~valid])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)
        inv_pz = 1.0 / pz

        # Rationalize 1 - beta0/beta * p/pz to retain high-energy time slip.
        inv_gamma_sq = max(0.0, 1.0 - beta0**2)
        transverse_momentum_squared = px * px + py * py
        energy_ratio = np.sqrt(inv_gamma_sq + (1.0 - inv_gamma_sq) * momentum_ratio**2)
        slip = (dp * (2.0 + dp) * inv_gamma_sq - transverse_momentum_squared) / (pz * (pz + energy_ratio))

        # A particle that becomes invalid at this drift exits immediately;
        # do not transport it with the stale entry mask.
        L_mask = L * (alive & valid)

        x += L_mask * px * inv_pz
        y += L_mask * py * inv_pz
        z += L_mask * slip

    def _multipole_kick_cpu(self, knl_eff, ksl_eff, x, px, y, py, tag, mask, chi):
        """Apply the common integrated multipole polynomial."""
        _apply_multipole_kick_cpu(knl_eff, ksl_eff, self.inv_fact, x, px, y, py, tag, chi)


def _apply_multipole_kick_cpu(knl, ksl, inv_fact, x, px, y, py, tag, scale=1.0):
    """Horner kick for integrated strengths; p_x=P_x/P0 and p_y=P_y/P0."""
    active = tag > 0
    # Evaluate only live particles, including when lost coordinates are nonfinite.
    xr, yi = x[active], y[active]
    re = np.full_like(xr, knl[-1] * inv_fact[-1] * scale)
    im = np.full_like(yi, ksl[-1] * inv_fact[-1] * scale)
    for i in range(len(knl) - 2, -1, -1):
        re, im = (re * xr - im * yi + knl[i] * inv_fact[i] * scale, re * yi + im * xr + ksl[i] * inv_fact[i] * scale)
    px[active] -= re
    py[active] += im


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

MULTIPOLE_KERNEL_BODY = r'''
__device__ __forceinline__ bool pass_drift(
    pass_real_t& x,
    pass_real_t& px,
    pass_real_t& y,
    pass_real_t& py,
    pass_real_t& z,
    pass_real_t dp,
    int& tag,
    float* lost_position,
    int* lost_turn,
    int index,
    double L,
    pass_real_t reference_beta_gamma,
    pass_real_t inv_gamma,
    pass_real_t s_position,
    int turn
) {
    // A tolerance matches the CPU const.eps guard and avoids exact floating-point
    // equality.  ``L`` is an element/slice length, so sub-epsilon transport is
    // intentionally treated as a no-op.
    if (fabs(L) < PASS_EPS || tag <= 0)
        return tag > 0;
    pass_real_t momentum_ratio = (pass_real_t)1 + dp;
    pass_real_t pz_sq = momentum_ratio * momentum_ratio - px * px - py * py;
    if (!(pz_sq > (pass_real_t)0)) {
        tag = -abs(tag);
        lost_position[index] = (float)s_position;
        lost_turn[index] = turn;
        return false;
    }
    pass_real_t pz = sqrt(pz_sq);
    pass_real_t inv_pz = (pass_real_t)1 / pz;
    pass_real_t inv_gamma_sq = inv_gamma * inv_gamma;
    pass_real_t transverse_momentum_squared = px * px + py * py;
    pass_real_t energy_ratio = sqrt(inv_gamma_sq + ((pass_real_t)1 - inv_gamma_sq) * momentum_ratio * momentum_ratio);
    // Stable even when the physical time slip is much smaller than float epsilon.
    pass_real_t slip = (dp * ((pass_real_t)2 + dp) * inv_gamma_sq - transverse_momentum_squared) / (pz * (pz + energy_ratio));
    x += L * px * inv_pz;
    y += L * py * inv_pz;
    z += L * slip;
    return true;
}

__device__ __forceinline__ void pass_kick(
    pass_real_t& px,
    pass_real_t& py,
    pass_real_t x,
    pass_real_t y,
    const pass_real_t* __restrict__ knl,
    const pass_real_t* __restrict__ ksl,
    const pass_real_t* __restrict__ inv_fact,
    int order,
    double scale
) {
    double dpx_mul = (double)knl[order] * inv_fact[order] * scale;
    double dpy_mul = (double)ksl[order] * inv_fact[order] * scale;
    for (int n = order; n > 0; --n) {
        double zre = dpx_mul * x - dpy_mul * y;
        double zim = dpx_mul * y + dpy_mul * x;
        dpx_mul = (double)knl[n - 1] * inv_fact[n - 1] * scale + zre;
        dpy_mul = (double)ksl[n - 1] * inv_fact[n - 1] * scale + zim;
    }
    px -= dpx_mul;
    py += dpy_mul;
}

__device__ __forceinline__ bool pass_dkd_step(
    pass_real_t& x,
    pass_real_t& px,
    pass_real_t& y,
    pass_real_t& py,
    pass_real_t& z,
    pass_real_t dp,
    int& tag,
    float* lost_position,
    int* lost_turn,
    int index,
    double ds,
    pass_real_t reference_beta_gamma,
    pass_real_t inv_gamma,
    pass_real_t s_position,
    int turn,
    const pass_real_t* __restrict__ knl,
    const pass_real_t* __restrict__ ksl,
    const pass_real_t* __restrict__ inv_fact,
    int order
) {
    if (!pass_drift(x, px, y, py, z, dp, tag, lost_position, lost_turn, index, ds * (pass_real_t)0.5, reference_beta_gamma, inv_gamma, s_position,
                    turn))
        return false;
    pass_kick(px, py, x, y, knl, ksl, inv_fact, order, ds);
    return pass_drift(x, px, y, py, z, dp, tag, lost_position, lost_turn, index, ds * (pass_real_t)0.5, reference_beta_gamma, inv_gamma, s_position,
                      turn);
}

// mode: 0 = thin integrated kick, 1 = thick sliced DKD, 2 = pure drift.
// integrator: 0 = uniform second-order DKD, 1 = Yoshida fourth-order DKD.
extern "C" __global__ void track_multipole_dkd(
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
    double L,
    pass_real_t s_position,
    int turn,
    const pass_real_t* __restrict__ knl,
    const pass_real_t* __restrict__ ksl,
    const pass_real_t* __restrict__ inv_fact,
    int order,
    int num_slice,
    int integrator,
    int mode
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (index >= end_index || tag[index] <= 0)
        return;

    pass_real_t xi = x[index], pxi = px[index];
    pass_real_t yi = y[index], pyi = py[index];
    pass_real_t zi = z[index], dpi = dp[index];
    int ti = tag[index];
    bool alive = true;

    if (mode == 0) {
        pass_kick(pxi, pyi, xi, yi, knl, ksl, inv_fact, order, (pass_real_t)1);
    } else if (mode == 2) {
        alive = pass_drift(xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn, index, L, reference_beta_gamma, inv_gamma, s_position, turn);
    } else {
        const double ds = L / (double)num_slice;
        for (int slice = 0; slice < num_slice && alive; ++slice) {
            if (integrator == 0) {
                alive = pass_dkd_step(xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn, index, ds, reference_beta_gamma, inv_gamma, s_position,
                                      turn, knl, ksl, inv_fact, order);
            } else {
                alive = pass_dkd_step(xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn, index, ds * pass_yoshida_z1, reference_beta_gamma,
                                      inv_gamma, s_position, turn, knl, ksl, inv_fact, order);
                if (alive)
                    alive = pass_dkd_step(xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn, index, ds * pass_yoshida_z0, reference_beta_gamma,
                                          inv_gamma, s_position, turn, knl, ksl, inv_fact, order);
                if (alive)
                    alive = pass_dkd_step(xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn, index, ds * pass_yoshida_z1, reference_beta_gamma,
                                          inv_gamma, s_position, turn, knl, ksl, inv_fact, order);
            }
        }
    }

    if (alive) {
        x[index] = xi;
        px[index] = pxi;
        y[index] = yi;
        py[index] = pyi;
        z[index] = zi;
    }
    tag[index] = ti;
}
'''


@lru_cache(maxsize=None)
def _get_multipole_kernel(dtype):
    """Return a dtype-specialized raw kernel, compiled lazily."""
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU multipole tracking requires the optional 'cuda' dependencies.") from exc
    dtype = np.dtype(dtype)
    return cp.RawKernel(
        CUDA_REAL_PREAMBLE + MULTIPOLE_KERNEL_BODY,
        "track_multipole_dkd",
        options=("--std=c++14", f"-DPASS_USE_FLOAT={int(dtype == np.dtype(np.float32))}"),
    )


def launch_multipole(element, sim, knl, ksl, inv_fact, mode):
    """Launch the fused map for every bunch of an element.

    ``mode`` is 0 for a thin kick, 1 for a thick sliced DKD map, and 2 for a
    zero-strength drift.  The kernel's ``integrator`` argument is 0 for the
    uniform second-order map and 1 for Yoshida-4.
    """
    import cupy as cp

    beam = sim.beams[element.beam_id]
    turn = sim.state.turn
    p = beam.particles
    real = p.real
    cache = getattr(element, "_gpu_strength_cache", None)
    if cache is None:
        cache = {}
        element._gpu_strength_cache = cache
    key = (np.dtype(p.dtype).str, cp.cuda.runtime.getDevice())
    if key not in cache:
        cache[key] = (cp.asarray(knl, dtype=p.dtype), cp.asarray(ksl, dtype=p.dtype), cp.asarray(inv_fact, dtype=p.dtype))
    knl_gpu, ksl_gpu, inv_gpu = cache[key]
    kernel = _get_multipole_kernel(p.dtype.str)
    threads = 256
    for bunch in beam.bunches:
        start, end = bunch.start_idx, bunch.end_idx
        n = end - start
        if n > 0:
            blocks = (n + threads - 1) // threads
            kernel(
                (blocks, ), (threads, ),
                (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, p.lost_position, p.lost_turn, np.int32(start), np.int32(end), real(
                    bunch.beta * bunch.gamma), real(1.0 / bunch.gamma), np.float64(element.length), real(element.s), np.int32(turn), knl_gpu, ksl_gpu,
                 inv_gpu, np.int32(len(knl) - 1), np.int32(element.num_slice), np.int32(0 if element.integrator == "uniform" else 1), np.int32(mode)))
