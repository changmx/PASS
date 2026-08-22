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


# ============================================================
# Yoshida 4th-order coefficients
# ============================================================
_YOSHIDA_Z1 = 1.0 / (2.0 - 2.0**(1.0/3.0))   # ≈ 1.3512071919596
_YOSHIDA_Z0 = 1.0 - 2.0 * _YOSHIDA_Z1          # ≈ -1.7024143839193


@Command.register("kicker")
class Kicker(Command):
    """
    Kicker magnet with exact drift-kick-drift tracking.

    A kicker is a pulsed dipole magnet that applies a transverse angular kick
    to the beam. Physically equivalent to an order-0 multipole (dipole).

    Tracking sequence:
      Thin lens (length=0):  single dipole kick
      Thick lens (length>0): N slices of drift-kick-drift-exact
        - uniform:   Drift(ds/2) → Kick(ds) → Drift(ds/2)  (2nd order symplectic)
        - yoshida4:  4th order Yoshida composition of DKD steps

      If hkick=0 and vkick=0 (no field), thick lens degenerates to a pure drift.

    Dipole kick (integrated strength hkick_eff = hkick * ds / L):
      Δpx = hkick_eff
      Δpy = vkick_eff

    Drift: exact drift (Table 1.1, map D), Eq. 1.86-1.88

    Coordinate convention (PASS):
      x, px, y, py, z, dp(=δ)
      px = Px/P0,  py = Py/P0,  dp = (P-P0)/P0
      z  = s - β0·c·t  (ζ coordinate)
    """

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = kwargs.get("length (m)", 0.0)
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]

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

        # --- aperture ---
        self.aperture_type: str = kwargs.get("aperture type", "off").lower()
        self.aperture_value: list = kwargs.get("aperture value", [])
        if not isinstance(self.aperture_value, list):
            raise ValueError(f"Aperture value of {self.cmd_name} must be a list, but got {type(self.aperture_value)}")

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, "
                    f"Length={self.length:.4f}, IsThick={self.is_thick}, "
                    f"Hkick={self.hkick:.6e}, Vkick={self.vkick:.6e}, "
                    f"NumSlice={self.num_slice:d}, Integrator={self.integrator:s}, "
                    f"ApertureType={self.aperture_type:s}, ApertureValue={self.aperture_value}")
        set_normal_logging()

    # ============================================================
    # Main execution
    # ============================================================

    def execute_cpu(self, sim):
        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        turn = sim.state.turn

        for i, bunch in enumerate(bunches):
            self._track_kicker_cpu(beam, bunch, turn)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)

    def execute_gpu(self, sim):
        beam = sim.beams[self.beam_id]
        turn = sim.state.turn
        p = beam.particles
        kernel = _get_kicker_kernel(p.dtype)

        for bunch in beam.bunches:
            start = bunch.start_idx
            end = bunch.end_idx
            n = end - start
            if n > 0:
                threads = 256
                blocks = (n + threads - 1) // threads
                kernel(
                    (blocks,), (threads,),
                    (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag,
                     p.lost_position, p.lost_turn,
                     np.int32(start), np.int32(end),
                     p.real(bunch.beta * bunch.gamma), p.real(1.0 / bunch.gamma),
                     p.real(self.length), p.real(self.hkick), p.real(self.vkick),
                     p.real(self.hk), p.real(self.vk), np.int32(self.num_slice),
                     np.int32(1 if self.integrator == "yoshida4" else 0),
                     p.real(_YOSHIDA_Z1), p.real(_YOSHIDA_Z0),
                     p.real(self.s), np.int32(turn),
                     np.int32(1 if self.is_thick else 0)),
                )
                check_aperture_gpu(
                    beam, bunch, self.aperture_type, self.aperture_value,
                    self.s, turn,
                )
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)

    # ============================================================
    # Full kicker tracking (CPU)
    # ============================================================

    def _track_kicker_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):
        """Track particles through the kicker: thin lens or sliced DKD-exact."""

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

        # mask for alive particles
        mask = (tag > 0).astype(np.float64)

        if not self.is_thick:
            # Thin lens: single dipole kick
            self._dipole_kick_cpu(self.hkick, self.vkick,
                                  px, py, tag, mask)
            return

        # Thick lens
        if abs(self.hkick) < const.eps and abs(self.vkick) < const.eps:
            # No field: pure drift
            self._drift_exact_cpu(self.length, x, px, y, py, z, dp, tag, mask, beta0)
        else:
            # Sliced DKD-exact
            ds = self.length / self.num_slice
            for _ in range(self.num_slice):
                if self.integrator == "uniform":
                    self._dkd_uniform_cpu(x, px, y, py, z, dp, tag, mask,
                                          ds, self.hk, self.vk, beta0)
                elif self.integrator == "yoshida4":
                    self._dkd_yoshida4_cpu(x, px, y, py, z, dp, tag, mask,
                                           ds, self.hk, self.vk, beta0)

        newly_lost = alive_before & (tag <= 0)
        lost_position[newly_lost] = self.s
        lost_turn[newly_lost] = turn

    # ============================================================
    # Body: Drift-Kick-Drift exact (uniform integrator)
    # ============================================================

    def _dkd_uniform_cpu(self, x, px, y, py, z, dp, tag, mask,
                         ds, hk, vk, beta0):
        """
        One DKD slice (uniform/leapfrog, 2nd order symplectic):

          Drift(ds/2) → Kick(ds) → Drift(ds/2)
        """
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._dipole_kick_cpu(hk * ds, vk * ds,
                              px, py, tag, mask)
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)

    # ============================================================
    # Body: Drift-Kick-Drift exact (Yoshida 4th order)
    # ============================================================

    def _dkd_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask,
                          ds, hk, vk, beta0):
        """
        One Yoshida-4 slice:

          S4(ds) = S2(z1·ds) ∘ S2(z0·ds) ∘ S2(z1·ds)

        where S2 is the standard DKD (leapfrog) step.
        """
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z1, hk, vk, beta0)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z0, hk, vk, beta0)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask,
                           ds * _YOSHIDA_Z1, hk, vk, beta0)

    def _dkd_step_cpu(self, x, px, y, py, z, dp, tag, mask,
                      ds, hk, vk, beta0):
        """Single DKD step with given effective length ds (can be negative)."""
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._dipole_kick_cpu(hk * ds, vk * ds,
                              px, py, tag, mask)
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)

    # ============================================================
    # Exact drift map (Table 1.1, map D)
    # Eq. 1.86-1.88
    # ============================================================

    def _drift_exact_cpu(self, L, x, px, y, py, z, dp, tag, mask, beta0):
        """
        Exact drift: free propagation in a straight, field-free region.

        x  += (px / pz) * L
        y  += (py / pz) * L
        z  += L * (1 - (beta0/beta) * (1+dp) / pz)

        where pz = sqrt((1+dp)² - px² - py²)
              beta = (1+dp)*beta0*gamma0 / sqrt(1 + ((1+dp)*beta0*gamma0)²)
        """
        if abs(L) < const.eps:
            return

        one_plus_delta = 1.0 + dp
        pz_sq = one_plus_delta**2 - px**2 - py**2

        valid = pz_sq > 0.0
        alive = tag > 0
        newly_lost = alive & ~valid
        tag[newly_lost] = -np.abs(tag[newly_lost])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)
        inv_pz = 1.0 / pz

        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
        bg = beta0 * gamma0
        beta = one_plus_delta_beta(one_plus_delta=one_plus_delta, bg=bg)

        # Refresh the mask before applying this drift so particles that just
        # became invalid do not receive a huge displacement from pz~0.
        mask[:] = (tag > 0).astype(mask.dtype, copy=False)
        L_mask = L * mask

        x += L_mask * px * inv_pz
        y += L_mask * py * inv_pz
        z += L_mask * (1.0 - (beta0 / beta) * one_plus_delta * inv_pz)

    # ============================================================
    # Dipole kick (thin lens)
    # ============================================================

    def _dipole_kick_cpu(self, hkick_eff, vkick_eff,
                         px, py, tag, mask):
        """
        Thin dipole kick with integrated strengths.

        Δpx = hkick_eff
        Δpy = vkick_eff

        For thin lens mode: hkick_eff = hkick, vkick_eff = vkick
        For DKD mode:       hkick_eff = hk * ds, vkick_eff = vk * ds
        """
        px += hkick_eff * mask
        py += vkick_eff * mask


# ============================================================
# Helper: compute beta from (1+delta) and beta0*gamma0
# ============================================================

def one_plus_delta_beta(one_plus_delta, bg):
    """
    Compute beta = v/c given (1+delta) and beta0*gamma0.

    From: P/P0 = 1+delta = beta*gamma / (beta0*gamma0)
    => beta*gamma = (1+delta) * beta0*gamma0
    => beta = (beta*gamma) / sqrt(1 + (beta*gamma)²)
    """
    bg_new = one_plus_delta * bg
    return bg_new / np.sqrt(1.0 + bg_new**2)


CUDA_REAL_PREAMBLE = f'''
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
    pass_real_t& x, pass_real_t& y, pass_real_t& z,
    const pass_real_t px, const pass_real_t py, const pass_real_t dp,
    int& tag, float* lost_position, int* lost_turn,
    pass_real_t beta_gamma, pass_real_t inv_gamma, pass_real_t length,
    pass_real_t s_position, int turn)
{
    if (tag <= 0) return false;
    pass_real_t one_plus_delta = (pass_real_t)1 + dp;
    pass_real_t pz_sq = one_plus_delta * one_plus_delta - px * px - py * py;
    if (!(pz_sq > (pass_real_t)0)) {
        tag = -abs(tag);
        lost_position[0] = (float)s_position;
        lost_turn[0] = turn;
        return false;
    }
    pass_real_t inv_pz = (pass_real_t)1 / sqrt(pz_sq);
    pass_real_t bg = one_plus_delta * beta_gamma;
    pass_real_t dzeta_factor = sqrt((pass_real_t)1 + bg * bg) * inv_pz * inv_gamma;
    x += length * px * inv_pz;
    y += length * py * inv_pz;
    z += length * ((pass_real_t)1 - dzeta_factor);
    return true;
}

extern "C" __global__
void transfer_kicker(
    pass_real_t* __restrict__ x, pass_real_t* __restrict__ px,
    pass_real_t* __restrict__ y, pass_real_t* __restrict__ py,
    pass_real_t* __restrict__ z, const pass_real_t* __restrict__ dp,
    int* __restrict__ tag, float* __restrict__ lost_position,
    int* __restrict__ lost_turn, int start_index, int end_index,
    pass_real_t beta_gamma, pass_real_t inv_gamma, pass_real_t length,
    pass_real_t hkick, pass_real_t vkick, pass_real_t hk, pass_real_t vk,
    int num_slice, int yoshida4, pass_real_t yoshida_z1,
    pass_real_t yoshida_z0, pass_real_t s_position, int turn, int is_thick)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index || tag[i] <= 0) return;

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
        if (kicker_drift(xi, yi, zi, pxi, pyi, dpi, ti, &lp, &lt,
                         beta_gamma, inv_gamma, length, s_position, turn)) {
            x[i] = xi; y[i] = yi; z[i] = zi;
        }
        tag[i] = ti; lost_position[i] = lp; lost_turn[i] = lt;
        return;
    }

    pass_real_t ds = length / num_slice;
    for (int slice = 0; slice < num_slice && ti > 0; ++slice) {
        int nsteps = yoshida4 ? 3 : 1;
        for (int step = 0; step < nsteps && ti > 0; ++step) {
            pass_real_t factor = yoshida4
                ? (step == 1 ? yoshida_z0 : yoshida_z1) : (pass_real_t)1;
            pass_real_t eff = ds * factor;
            if (!kicker_drift(xi, yi, zi, pxi, pyi, dpi, ti, &lp, &lt,
                              beta_gamma, inv_gamma, eff * (pass_real_t)0.5,
                              s_position, turn)) break;
            if (ti <= 0) break;
            pxi += hk * eff;
            pyi += vk * eff;
            if (!kicker_drift(xi, yi, zi, pxi, pyi, dpi, ti, &lp, &lt,
                              beta_gamma, inv_gamma, eff * (pass_real_t)0.5,
                              s_position, turn)) break;
        }
    }
    x[i] = xi; px[i] = pxi; y[i] = yi; py[i] = pyi; z[i] = zi;
    tag[i] = ti; lost_position[i] = lp; lost_turn[i] = lt;
}
'''

KICKER_SOURCE = CUDA_REAL_PREAMBLE + KICKER_KERNEL_BODY
_kicker_kernels = {}


def _get_kicker_kernel(dtype):
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU Kicker tracking requires CUDA dependencies.") from exc
    key = np.dtype(dtype)
    if key not in _kicker_kernels:
        _kicker_kernels[key] = cp.RawKernel(
            KICKER_SOURCE, "transfer_kicker",
            options=("--std=c++14", f"-DPASS_USE_FLOAT={int(key == np.dtype(np.float32))}"),
        )
    return _kicker_kernels[key]
