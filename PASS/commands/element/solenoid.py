from functools import lru_cache
import logging

import numpy as np

from PASS.commands.command import Command
from PASS.commands.element.error import AlignmentErrors, FieldErrors
from PASS.utils.aperture import check_aperture_cpu, check_aperture_gpu
from PASS.utils.slicing import print_element_slicing, configure_element_slicing, run_body_slices, transport_with_center
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const, _build_yoshida_cuda_constants

logger = logging.getLogger(__name__)


@Command.register("solenoid")
class Solenoid(Command):
    """Track a solenoid with the exact uniform-field map.

    The Larmor-frame map couples both transverse planes. Multipole errors
    use Sol-Kick-Sol slices with uniform or Yoshida integration. Without
    transverse multipoles, zero axial field reduces to a drift. At zero
    length only the integrated transverse multipole kick remains."""

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
            raise ValueError(f"The length of Solenoid {self.cmd_name} is {self.length}, which should be >= 0")
        if self.length > const.eps:
            self.is_thick = True
        else:
            self.is_thick = False

        self.ks = kwargs.get("ks", 0.0)

        # Multipole components (optional, for solenoid + multipole overlay)
        knl_list = kwargs.get("kil", [])
        ksl_list = kwargs.get("kisl", [])

        if not isinstance(knl_list, (list, np.ndarray)):
            raise ValueError(f"KiL of {self.cmd_name} must be a list, but got {type(knl_list)}")
        if not isinstance(ksl_list, (list, np.ndarray)):
            raise ValueError(f"KiSL of {self.cmd_name} must be a list, but got {type(ksl_list)}")

        knl_list, ksl_list = self.field_errors.combine(knl_list, ksl_list)
        self.knl = np.array(knl_list, dtype=np.float64)
        self.ksl = np.array(ksl_list, dtype=np.float64)

        len_n = len(self.knl)
        len_s = len(self.ksl)

        if len_n == 0 and len_s == 0:
            self.has_multipoles = False
            self.order = -1
            self.knl = np.array([0.0])
            self.ksl = np.array([0.0])
        else:
            self.order = max(len_n, len_s) - 1
            if len_n > len_s:
                self.ksl = np.pad(self.ksl, (0, len_n - len_s), mode='constant')
            elif len_n < len_s:
                self.knl = np.pad(self.knl, (0, len_s - len_n), mode='constant')
            self.has_multipoles = not (np.all(self.knl == 0) and np.all(self.ksl == 0))

        if not self.has_multipoles and abs(self.ks) < const.eps:
            logger.warning("Solenoid %s has zero axial and transverse fields; it acts as a drift or marker.", self.cmd_name)

        # Thick lens: compute per-unit-length multipole strength
        if self.is_thick and self.has_multipoles:
            self.kn = self.knl / self.length
            self.ksp = self.ksl / self.length
        else:
            self.kn = np.zeros_like(self.knl)
            self.ksp = np.zeros_like(self.ksl)

        # Precompute inverse factorials for multipole kick
        if self.has_multipoles:
            self.inv_fact = np.ones(self.order + 1)
            for n in range(1, self.order + 1):
                self.inv_fact[n] = self.inv_fact[n - 1] / n
        else:
            self.inv_fact = np.array([1.0])

        self.num_slice = kwargs.get("num slices", 1)
        if self.num_slice < 1:
            logger.warning(f"The number of slices of {self.cmd_name} is {self.num_slice}, which should be >= 1. It has been changed to 1 now.")
            self.num_slice = 1

        self.integrator = kwargs.get("integrator", "adaptive")
        if self.integrator not in ["adaptive", "uniform", "yoshida4"]:
            raise ValueError(f"The integrator of Solenoid {self.cmd_name} is {self.integrator}, which should be 'adaptive', 'uniform' or 'yoshida4'.")
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
                    f"IsThick={self.is_thick}, Ks={self.ks:.6f}, "
                    f"HasMultipoles={self.has_multipoles}, "
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
                self._track_solenoid_cpu(beam, bunch, turn)
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
            if not self.is_thick and self.has_multipoles:
                from PASS.commands.element.multipole import launch_multipole
                launch_multipole(self, sim, self.knl, self.ksl, self.inv_fact, 0)
            elif self._sc_nodes:
                from PASS.utils.slicing import execute_element_body_gpu
                execute_element_body_gpu(self, sim)
            else:
                if not self.is_thick:
                    mode = 0
                elif not self.has_multipoles:
                    mode = 2 if abs(self.ks) < const.eps else 1
                else:
                    mode = 3
                launch_solenoid(self, sim, mode)
        finally:
            self.alignment_errors.exit_frame(self, beam, turn, masks, gpu=True)
        for bunch in beam.bunches:
            check_aperture_gpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)
        return True

    def _track_solenoid_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):

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

        chi = 1.0
        mask = (tag > 0).astype(np.float64)

        if not self.is_thick:
            # The axial field has no thin kick; integrated transverse components do.
            if self.has_multipoles:
                self._multipole_kick_cpu(self.knl, self.ksl, x, px, y, py, tag, mask, chi)
            return

        if self._sc_nodes or (not self.has_multipoles and self.num_slice > 1):

            def transport(ds, on_center):
                if not self.has_multipoles:

                    def advance(length):
                        self._solenoid_exact_cpu(length, self.ks, x, px, y, py, z, dp, tag, mask, beta0)

                    transport_with_center(advance, ds, on_center)
                else:
                    step = self._sks_step_cpu if self.integrator == "uniform" else self._sks_yoshida4_cpu
                    step(x, px, y, py, z, dp, tag, mask, ds, self.ks, self.kn, self.ksp, chi, beta0, on_center=on_center)

            run_body_slices(self, beam, bunch, turn, transport)
            return

        if not self.has_multipoles:
            # Pure solenoid: one uniform-field map.
            if abs(self.ks) < const.eps:
                # ks=0: pure drift
                self._drift_exact_cpu(self.length, x, px, y, py, z, dp, tag, mask, beta0)
            else:
                self._solenoid_exact_cpu(self.length, self.ks, x, px, y, py, z, dp, tag, mask, beta0)
        else:
            # Solenoid + multipoles: Sol-Kick-Sol integrator
            ds = self.length / self.num_slice
            for _ in range(self.num_slice):
                if self.integrator == "uniform":
                    self._sks_step_cpu(x, px, y, py, z, dp, tag, mask, ds, self.ks, self.kn, self.ksp, chi, beta0)
                elif self.integrator == "yoshida4":
                    self._sks_yoshida4_cpu(x, px, y, py, z, dp, tag, mask, ds, self.ks, self.kn, self.ksp, chi, beta0)

        newly_lost = alive_before & (tag < 0)
        if np.any(newly_lost):
            lost_position = p.lost_position[start:end]
            lost_turn = p.lost_turn[start:end]
            lost_position[newly_lost] = self.s
            lost_turn[newly_lost] = turn

    def _sks_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask, ds, ks, kn, ksp, chi, beta0, on_center=None):
        """Compose three solenoid-kick-solenoid steps with Yoshida coefficients."""
        self._sks_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, ks, kn, ksp, chi, beta0)
        self._sks_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z0, ks, kn, ksp, chi, beta0, on_center=on_center)
        self._sks_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, ks, kn, ksp, chi, beta0)

    def _sks_step_cpu(self, x, px, y, py, z, dp, tag, mask, ds, ks, kn, ksp, chi, beta0, on_center=None):
        """Apply one solenoid-kick-solenoid step; ds may be negative."""
        self._solenoid_exact_cpu(ds * 0.5, ks, x, px, y, py, z, dp, tag, mask, beta0)
        self._multipole_kick_cpu(kn * ds, ksp * ds, x, px, y, py, tag, mask, chi)
        if on_center is not None:
            on_center()
        self._solenoid_exact_cpu(ds * 0.5, ks, x, px, y, py, z, dp, tag, mask, beta0)

    # Exact uniform-field solenoid map

    def _solenoid_exact_cpu(self, L, ks, x, px, y, py, z, dp, tag, mask, beta0):
        """Apply Larmor rotation and focusing in a uniform solenoid.

        The longitudinal momentum uses px + ks*y/2 and py - ks*x/2.
        z advances with the particle's own velocity."""
        if abs(L) < const.eps:
            return

        if abs(ks) < const.eps:
            self._drift_exact_cpu(L, x, px, y, py, z, dp, tag, mask, beta0)
            return

        half_strength = ks * 0.5

        momentum_ratio = 1.0 + dp

        # Momentum combinations entering the longitudinal momentum.
        larmor_px = px + half_strength * y
        larmor_py = py - half_strength * x
        transverse_momentum_squared = larmor_px * larmor_px + larmor_py * larmor_py

        pz_sq = momentum_ratio**2 - transverse_momentum_squared
        valid = pz_sq > 0.0
        alive = tag > 0
        tag[alive & ~valid] = -np.abs(tag[alive & ~valid])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)

        theta = half_strength * L / pz
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # drift_coefficient = sin(θ) / half_strength (effective drift length in Larmor frame)
        drift_coefficient = sin_theta / half_strength

        rotated_x = cos_theta * x + sin_theta * y
        rotated_px = cos_theta * px + sin_theta * py
        rotated_y = cos_theta * y - sin_theta * x
        rotated_py = cos_theta * py - sin_theta * px

        new_x = cos_theta * rotated_x + drift_coefficient * rotated_px
        new_px = cos_theta * rotated_px - half_strength * sin_theta * rotated_x
        new_y = cos_theta * rotated_y + drift_coefficient * rotated_py
        new_py = cos_theta * rotated_py - half_strength * sin_theta * rotated_y

        inv_gamma_sq = max(0.0, 1.0 - beta0**2)
        energy_ratio = np.sqrt(inv_gamma_sq + (1.0 - inv_gamma_sq) * momentum_ratio**2)
        # The mechanical transverse momentum is constant in magnitude in this map.
        add_to_z = L * (dp * (2.0 + dp) * inv_gamma_sq - transverse_momentum_squared) / (pz * (pz + energy_ratio))

        active = (alive & valid).astype(mask.dtype, copy=False)
        x[:] = new_x * active + x * (1.0 - active)
        px[:] = new_px * active + px * (1.0 - active)
        y[:] = new_y * active + y * (1.0 - active)
        py[:] = new_py * active + py * (1.0 - active)
        z += add_to_z * active

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

        L_mask = L * (alive & valid)

        x += L_mask * px * inv_pz
        y += L_mask * py * inv_pz
        z += L_mask * slip

    def _multipole_kick_cpu(self, knl_eff, ksl_eff, x, px, y, py, tag, mask, chi):
        """Apply the common integrated multipole polynomial."""
        from PASS.commands.element.multipole import _apply_multipole_kick_cpu

        _apply_multipole_kick_cpu(knl_eff, ksl_eff, self.inv_fact, x, px, y, py, tag, chi)


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

SOLENOID_BODY = r'''
__device__ __forceinline__ bool sol_drift(
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
    pass_real_t beta0,
    pass_real_t reference_beta_gamma,
    pass_real_t inv_gamma,
    pass_real_t s_position,
    int turn
) {
    if (fabs(L) < PASS_EPS || tag <= 0)
        return tag > 0;
    pass_real_t momentum_ratio = (pass_real_t)1 + dp;
    pass_real_t pz2 = momentum_ratio * momentum_ratio - px * px - py * py;
    if (!(pz2 > (pass_real_t)0)) {
        tag = -abs(tag);
        lost_position[index] = (float)s_position;
        lost_turn[index] = turn;
        return false;
    }
    pass_real_t pz = sqrt(pz2);
    pass_real_t invpz = (pass_real_t)1 / pz;
    pass_real_t inv_gamma_sq = inv_gamma * inv_gamma;
    pass_real_t transverse_momentum_squared = px * px + py * py;
    pass_real_t energy_ratio = sqrt(inv_gamma_sq + ((pass_real_t)1 - inv_gamma_sq) * momentum_ratio * momentum_ratio);
    // Stable even when the physical time slip is much smaller than float epsilon.
    pass_real_t slip = (dp * ((pass_real_t)2 + dp) * inv_gamma_sq - transverse_momentum_squared) / (pz * (pz + energy_ratio));
    x += L * px * invpz;
    y += L * py * invpz;
    z += L * slip;
    return true;
}

__device__ __forceinline__ bool sol_exact(
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
    pass_real_t ks,
    pass_real_t beta0,
    pass_real_t reference_beta_gamma,
    pass_real_t s_position,
    int turn
) {
    if (fabs(L) < PASS_EPS || tag <= 0)
        return tag > 0;
    pass_real_t half_strength = ks * (pass_real_t)0.5;
    if (fabs(half_strength) < PASS_EPS)
        return sol_drift(x, px, y, py, z, dp, tag, lost_position, lost_turn, index, L, beta0, reference_beta_gamma,
                         (pass_real_t)1 / sqrt((pass_real_t)1 + reference_beta_gamma * reference_beta_gamma), s_position, turn);

    pass_real_t momentum_ratio = (pass_real_t)1 + dp;
    pass_real_t larmor_px = px + half_strength * y;
    pass_real_t larmor_py = py - half_strength * x;
    pass_real_t pz2 = momentum_ratio * momentum_ratio - larmor_px * larmor_px - larmor_py * larmor_py;
    if (!(pz2 > (pass_real_t)0)) {
        tag = -abs(tag);
        lost_position[index] = (float)s_position;
        lost_turn[index] = turn;
        return false;
    }
    pass_real_t pz = sqrt(pz2);
    double theta = half_strength * L / pz;
    pass_real_t cos_theta = cos(theta), sin_theta = sin(theta), drift_coefficient = sin_theta / half_strength;
    pass_real_t rotated_x = cos_theta * x + sin_theta * y;
    pass_real_t rotated_px = cos_theta * px + sin_theta * py;
    pass_real_t rotated_y = cos_theta * y - sin_theta * x;
    pass_real_t rotated_py = cos_theta * py - sin_theta * px;
    pass_real_t xn = cos_theta * rotated_x + drift_coefficient * rotated_px;
    pass_real_t pxn = cos_theta * rotated_px - half_strength * sin_theta * rotated_x;
    pass_real_t yn = cos_theta * rotated_y + drift_coefficient * rotated_py;
    pass_real_t pyn = cos_theta * rotated_py - half_strength * sin_theta * rotated_y;
    pass_real_t inv_gamma_sq = (pass_real_t)1 / ((pass_real_t)1 + reference_beta_gamma * reference_beta_gamma);
    pass_real_t energy_ratio = sqrt(inv_gamma_sq + ((pass_real_t)1 - inv_gamma_sq) * momentum_ratio * momentum_ratio);
    pass_real_t transverse_momentum_squared = larmor_px * larmor_px + larmor_py * larmor_py;
    pass_real_t slip = (dp * ((pass_real_t)2 + dp) * inv_gamma_sq - transverse_momentum_squared) / (pz * (pz + energy_ratio));
    x = xn;
    px = pxn;
    y = yn;
    py = pyn;
    z += L * slip;
    return true;
}

__device__ __forceinline__ void sol_kick(
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
    double ar = (double)knl[order] * inv_fact[order] * scale;
    double ai = (double)ksl[order] * inv_fact[order] * scale;
    for (int n = order; n > 0; --n) {
        double re = ar * x - ai * y;
        double im = ar * y + ai * x;
        ar = (double)knl[n - 1] * inv_fact[n - 1] * scale + re;
        ai = (double)ksl[n - 1] * inv_fact[n - 1] * scale + im;
    }
    px -= ar;
    py += ai;
}

__device__ __forceinline__ bool sol_sks_step(
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
    pass_real_t ks,
    pass_real_t beta0,
    pass_real_t reference_beta_gamma,
    pass_real_t s_position,
    int turn,
    const pass_real_t* __restrict__ knl,
    const pass_real_t* __restrict__ ksl,
    const pass_real_t* __restrict__ inv_fact,
    int order
) {
    if (!sol_exact(x, px, y, py, z, dp, tag, lost_position, lost_turn, index, ds * (pass_real_t)0.5, ks, beta0, reference_beta_gamma, s_position,
                   turn))
        return false;
    sol_kick(px, py, x, y, knl, ksl, inv_fact, order, ds);
    return sol_exact(x, px, y, py, z, dp, tag, lost_position, lost_turn, index, ds * (pass_real_t)0.5, ks, beta0, reference_beta_gamma, s_position,
                     turn);
}

extern "C" __global__ void track_solenoid(
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
    pass_real_t beta0,
    pass_real_t reference_beta_gamma,
    double L,
    pass_real_t ks,
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
    if (mode == 0)
        return; // zero-length solenoid is a marker

    pass_real_t xi = x[index], pxi = px[index], yi = y[index], pyi = py[index];
    pass_real_t zi = z[index], dpi = dp[index];
    int ti = tag[index];
    bool alive = true;
    if (mode == 1 || mode == 2) {
        const double ds = L / (double)num_slice;
        for (int slice = 0; slice < num_slice && alive; ++slice) {
            alive = sol_exact(xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn, index, ds, mode == 1 ? ks : (pass_real_t)0, beta0,
                              reference_beta_gamma, s_position - L + (slice + 1) * ds, turn);
        }
    } else {
        const double ds = L / (double)num_slice;
        for (int slice = 0; slice < num_slice && alive; ++slice) {
            const double d = (integrator == 0) ? ds : ds * pass_yoshida_z1;
            alive = sol_sks_step(xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn, index, d, ks, beta0, reference_beta_gamma, s_position, turn,
                                 knl, ksl, inv_fact, order);
            if (integrator != 0 && alive) {
                alive = sol_sks_step(xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn, index, ds * pass_yoshida_z0, ks, beta0,
                                     reference_beta_gamma, s_position, turn, knl, ksl, inv_fact, order);
                if (alive)
                    alive = sol_sks_step(xi, pxi, yi, pyi, zi, dpi, ti, lost_position, lost_turn, index, ds * pass_yoshida_z1, ks, beta0,
                                         reference_beta_gamma, s_position, turn, knl, ksl, inv_fact, order);
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
def _get_solenoid_kernel(dtype):
    import cupy as cp

    dtype = np.dtype(dtype)
    return cp.RawKernel(
        CUDA_REAL_PREAMBLE + SOLENOID_BODY,
        "track_solenoid",
        options=("--std=c++14", f"-DPASS_USE_FLOAT={int(dtype == np.dtype(np.float32))}"),
    )


def launch_solenoid(element, sim, mode):
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU solenoid tracking requires the optional 'cuda' dependencies.") from exc
    beam = sim.beams[element.beam_id]
    p = beam.particles
    kernel = _get_solenoid_kernel(p.dtype.str)
    cache = getattr(element, "_gpu_strength_cache", None)
    if cache is None:
        cache = {}
        element._gpu_strength_cache = cache
    if np.dtype(p.dtype) not in cache:
        cache[np.dtype(p.dtype)] = (
            cp.asarray(element.kn if element.has_multipoles else [0.0], dtype=p.dtype),
            cp.asarray(element.ksp if element.has_multipoles else [0.0], dtype=p.dtype),
            cp.asarray(element.inv_fact, dtype=p.dtype),
        )
    knl, ksl, inv = cache[np.dtype(p.dtype)]
    real = p.real
    threads = 256
    turn = sim.state.turn
    for bunch in beam.bunches:
        n = bunch.end_idx - bunch.start_idx
        if n > 0:
            blocks = (n + threads - 1) // threads
            kernel((blocks, ), (threads, ),
                   (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, p.lost_position, p.lost_turn, np.int32(bunch.start_idx), np.int32(
                       bunch.end_idx), real(bunch.beta), real(bunch.beta * bunch.gamma), np.float64(element.length), real(element.ks), real(
                           element.s), np.int32(turn), knl, ksl, inv, np.int32(len(knl) - 1), np.int32(
                               element.num_slice), np.int32(0 if element.integrator == "uniform" else 1), np.int32(mode)))
