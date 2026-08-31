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


@Command.register("elseparator")
class ElSeparator(Command):
    """
    Electrostatic separator.

    An electrostatic separator applies a uniform transverse electric field
    to deflect charged particles. It is primarily used for beam injection
    and extraction, where a septum divides the aperture into a field-free
    region (circulating beam) and a field region (injected/extracted beam).

    The deflection kick is derived from the integrated electric field:

      Δpx = exl / (β₀·c·Bρ)
      Δpy = eyl / (β₀·c·Bρ)

    where exl = ex·L (integrated horizontal field, in Volts),
          eyl = ey·L (integrated vertical field, in Volts),
          β₀ is the reference particle velocity / c,
          Bρ is the magnetic rigidity (p₀ / q₀).

    Tracking model:
      - Thin lens (length=0): pure momentum kick (Δpx, Δpy), no drift.
      - Thick lens (length>0): Drift(L/2) → Kick → Drift(L/2) (DKD).
        For a uniform electric field, DKD is exact (the trajectory is a
        parabola and leapfrog integrates constant-acceleration motion
        without error, provided pz is approximately constant).

    Septum logic:
      Particles on the circulating-beam side of the septum do not feel
      the electric field (pure drift). Particles that cross the septum
      into the field region are deflected. Particles that hit the septum
      plate/wire (within septum_thickness) are marked as lost.

      - septum_x_position > 0: field region is x > septum_x_position + thickness
      - septum_x_position < 0: field region is x < septum_x_position - thickness
      - septum_x_position = None: all particles feel the field (no septum)

    Tilt (MAD-X convention):
      Roll angle about the longitudinal (s) axis. A positive tilt
      represents a clockwise rotation of the separator when viewed
      looking downstream along +s.

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
            raise ValueError(f"The length of ElSeparator {self.cmd_name} is {self.length}, which should be >= 0")
        if self.length > const.eps:
            self.is_thick = True
        else:
            self.is_thick = False

        # Electric field strengths (V/m)
        self.ex = kwargs.get("ex (v/m)", 0.0)
        self.ey = kwargs.get("ey (v/m)", 0.0)

        # Integrated fields (V) = E * L
        # If exl/eyl not provided, derive from ex/ey and length
        self.exl = kwargs.get("exl (v)", None)
        self.eyl = kwargs.get("eyl (v)", None)

        if self.exl is None:
            self.exl = self.ex * self.length
        if self.eyl is None:
            self.eyl = self.ey * self.length

        # Consistency: if exl given and ex not given, derive ex for thick lens
        if self.is_thick:
            if abs(self.ex) < const.eps and abs(self.exl) > const.eps:
                self.ex = self.exl / self.length
            if abs(self.ey) < const.eps and abs(self.eyl) > const.eps:
                self.ey = self.eyl / self.length

        if abs(self.exl) < const.eps and abs(self.eyl) < const.eps:
            logger.warning(f"ElSeparator {self.cmd_name} has zero integrated field "
                           f"(exl={self.exl}, eyl={self.eyl}). It will act as a drift/marker.")
        elif abs(self.exl) > const.eps and abs(self.eyl) > const.eps:
            logger.warning(f"ElSeparator {self.cmd_name} has both exl={self.exl} and eyl={self.eyl} "
                           f"non-zero. Both horizontal and vertical kicks will be applied simultaneously. "
                           f"This is unusual for a typical electrostatic separator.")

        # Tilt (rotation about s-axis)
        self.tilt = kwargs.get("tilt (rad)", 0.0)

        # Septum positions (None = no septum on that axis)
        self.septum_x_position = kwargs.get("septum x position (m)", None)
        self.septum_y_position = kwargs.get("septum y position (m)", None)
        self.septum_thickness = kwargs.get("septum thickness (m)", 0.0)

        # --- aperture ---
        self.aperture_type: str = kwargs.get("aperture type", "off").lower()
        self.aperture_value: list = kwargs.get("aperture value", [])
        if not isinstance(self.aperture_value, list):
            raise ValueError(f"Aperture value of {self.cmd_name} must be a list, but got {type(self.aperture_value)}")

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, "
                    f"Length={self.length:.4f}, Ex={self.ex:.6e}, Ey={self.ey:.6e}, "
                    f"ExL={self.exl:.6e}, EyL={self.eyl:.6e}, Tilt={self.tilt:.6f}, "
                    f"SeptumXPosition={self.septum_x_position}, SeptumYPosition={self.septum_y_position}, SeptumThickness={self.septum_thickness:.6f}, "
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
            self._track_elseparator_cpu(beam, bunch, turn)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)
        return True

    def execute_gpu(self, sim):
        beam = sim.beams[self.beam_id]
        turn = sim.state.turn
        p = beam.particles
        kernel = _get_elseparator_kernel(p.dtype)

        for bunch in beam.bunches:
            start = bunch.start_idx
            end = bunch.end_idx
            n = end - start
            if n > 0:
                threads = 256
                blocks = (n + threads - 1) // threads
                denom = bunch.beta * const.c * bunch.brho
                kick_x = self.exl / denom if abs(denom) > const.eps else 0.0
                kick_y = self.eyl / denom if abs(denom) > const.eps else 0.0
                kernel(
                    (blocks,), (threads,),
                    (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag,
                     p.lost_position, p.lost_turn,
                     np.int32(start), np.int32(end),
                     p.real(bunch.beta * bunch.gamma), p.real(1.0 / bunch.gamma),
                     p.real(self.length), p.real(kick_x), p.real(kick_y),
                     p.real(self.tilt),
                     np.int32(1 if self.septum_x_position is not None and abs(self.exl) > const.eps else 0),
                     p.real(self.septum_x_position or 0.0),
                     np.int32(1 if self.septum_y_position is not None and abs(self.eyl) > const.eps else 0),
                     p.real(self.septum_y_position or 0.0),
                     p.real(self.septum_thickness),
                     np.int32(1 if self.is_thick else 0),
                     p.real(self.s), np.int32(turn)),
                )
                check_aperture_gpu(
                    beam, bunch, self.aperture_type, self.aperture_value,
                    self.s, turn,
                )
            if abs(self.length) >= const.eps:
                bunch.t0 += self.length / (bunch.beta * const.c)
        return True

    # ============================================================
    # Full tracking (CPU)
    # ============================================================

    def _track_elseparator_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):
        """Track particles through the electrostatic separator."""

        beta0 = bunch.beta
        gamma0 = bunch.gamma
        brho = bunch.brho
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

        # Compute kick strengths from integrated field
        # Δp = exl / (β₀·c·Bρ)
        denom = beta0 * const.c * brho
        kick_x = self.exl / denom if abs(denom) > const.eps else 0.0
        kick_y = self.eyl / denom if abs(denom) > const.eps else 0.0

        mask = (tag > 0).astype(np.float64)

        # --- Tilt rotation (entry) ---
        if abs(self.tilt) > const.eps:
            self._tilt_rotate_cpu(x, px, y, py, tag, mask, self.tilt)

        # --- Classify particles: field region, field-free, septum loss ---
        field_mask = np.ones(len(x), dtype=bool)  # True = feels field
        septum_lost = np.zeros(len(x), dtype=bool)  # True = hits septum plate/wire

        if self.septum_x_position is not None and abs(self.exl) > const.eps:
            sx = self.septum_x_position
            th = self.septum_thickness
            if sx > 0:
                # Field region: x > sx + th
                # Loss zone (septum plate/wire): sx < x <= sx + th
                field_mask &= (x > sx + th)
                septum_lost |= ((x > sx) & (x <= sx + th))
            else:
                # sx < 0: field region is x < sx - th
                # Loss zone: sx - th <= x < sx
                field_mask &= (x < sx - th)
                septum_lost |= ((x < sx) & (x >= sx - th))

        if self.septum_y_position is not None and abs(self.eyl) > const.eps:
            sy = self.septum_y_position
            th = self.septum_thickness
            if sy > 0:
                field_mask &= (y > sy + th)
                septum_lost |= ((y > sy) & (y <= sy + th))
            else:
                field_mask &= (y < sy - th)
                septum_lost |= ((y < sy) & (y >= sy - th))

        # Only alive particles are classified
        alive = tag > 0
        field_mask &= alive
        septum_lost &= alive & ~field_mask

        # --- Mark particles hitting septum as lost ---
        if np.any(septum_lost):
            tag[septum_lost] = -np.abs(tag[septum_lost])
            lost_position = p.lost_position[start:end]
            lost_turn = p.lost_turn[start:end]
            lost_position[septum_lost] = self.s
            lost_turn[septum_lost] = turn

        # --- Tracking ---
        if self.is_thick:
            # Thick lens: DKD for field region, pure drift for field-free region
            # Field-free particles: full drift
            free_mask = alive & ~field_mask & (tag > 0)
            if np.any(free_mask):
                self._drift_exact_cpu(self.length, x, px, y, py, z, dp, tag,
                                      free_mask.astype(np.float64), beta0)

            # Field particles: DKD (drift L/2 → kick → drift L/2)
            if np.any(field_mask):
                fm = field_mask.astype(np.float64)
                # Check if any field particles were killed by drift (pz_sq < 0)
                self._drift_exact_cpu(self.length * 0.5, x, px, y, py, z, dp, tag, fm, beta0)
                # Update field_mask: some particles may have been killed by drift
                fm = (field_mask & (tag > 0)).astype(np.float64)
                self._kick_cpu(kick_x, kick_y, x, px, y, py, tag, fm)
                self._drift_exact_cpu(self.length * 0.5, x, px, y, py, z, dp, tag, fm, beta0)
        else:
            # Thin lens: pure kick for field particles
            if np.any(field_mask):
                fm = field_mask.astype(np.float64)
                self._kick_cpu(kick_x, kick_y, x, px, y, py, tag, fm)

        # --- Tilt rotation (exit) ---
        if abs(self.tilt) > const.eps:
            mask = (tag > 0).astype(np.float64)
            self._tilt_rotate_cpu(x, px, y, py, tag, mask, -self.tilt)

        # --- Update lost particle info ---
        newly_lost = alive_before & (tag < 0)
        if np.any(newly_lost):
            lost_position = p.lost_position[start:end]
            lost_turn = p.lost_turn[start:end]
            lost_position[newly_lost] = self.s
            lost_turn[newly_lost] = turn

    # ============================================================
    # Kick: pure momentum translation
    # ============================================================

    def _kick_cpu(self, kick_x, kick_y,
                  x, px, y, py, tag, mask):
        """Apply electrostatic kick to masked particles.

        Δpx = kick_x
        Δpy = kick_y
        """
        px += kick_x * mask
        py += kick_y * mask

    # ============================================================
    # Tilt rotation about s-axis
    # ============================================================

    def _tilt_rotate_cpu(self, x, px, y, py, tag, mask, angle):
        """Rotate (x, y, px, py) clockwise by `angle` about the s-axis.

        MAD-X convention: positive angle = clockwise rotation when viewed
        looking downstream along +s.

        x'  =  x·cos - y·sin
        y'  =  x·sin + y·cos
        px' =  px·cos - py·sin
        py' =  px·sin + py·cos
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        px_new = px * cos_a - py * sin_a
        py_new = px * sin_a + py * cos_a

        x[:] = x_new * mask + x * (1.0 - mask)
        y[:] = y_new * mask + y * (1.0 - mask)
        px[:] = px_new * mask + px * (1.0 - mask)
        py[:] = py_new * mask + py * (1.0 - mask)

    # ============================================================
    # Exact drift map (same as solenoid.py / dipole.py)
    # ============================================================

    def _drift_exact_cpu(self, L, x, px, y, py, z, dp, tag, mask, beta0):
        """Exact drift: free propagation in a straight, field-free region.

        x  += (px / pz) * L
        y  += (py / pz) * L
        z  += L * (1 - (beta0/beta) * (1+dp) / pz)
        """
        if abs(L) < const.eps:
            return

        one_plus_delta = 1.0 + dp
        pz_sq = one_plus_delta**2 - px**2 - py**2

        valid = (pz_sq > 0.0) & (tag > 0)
        tag[~valid] = -np.abs(tag[~valid])
        pz_sq_safe = np.maximum(pz_sq, const.eps)
        pz = np.sqrt(pz_sq_safe)
        inv_pz = 1.0 / pz

        gamma0 = 1.0 / np.sqrt(1.0 - beta0**2) if beta0 < 1.0 else 1e30
        bg = beta0 * gamma0
        beta = one_plus_delta_beta(one_plus_delta=one_plus_delta, bg=bg)

        # Exclude particles that became invalid in this drift before using
        # the clamped pz value; otherwise pz~0 would create a huge jump.
        mask = mask * (tag > 0)
        L_mask = L * mask

        x += L_mask * px * inv_pz
        y += L_mask * py * inv_pz
        z += L_mask * (1.0 - (beta0 / beta) * one_plus_delta * inv_pz)


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

ELSEPARATOR_KERNEL_BODY = r'''
__device__ inline bool elseparator_drift(
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
void transfer_elseparator(
    pass_real_t* __restrict__ x, pass_real_t* __restrict__ px,
    pass_real_t* __restrict__ y, pass_real_t* __restrict__ py,
    pass_real_t* __restrict__ z, const pass_real_t* __restrict__ dp,
    int* __restrict__ tag, float* __restrict__ lost_position,
    int* __restrict__ lost_turn, int start_index, int end_index,
    pass_real_t beta_gamma, pass_real_t inv_gamma, pass_real_t length,
    pass_real_t kick_x, pass_real_t kick_y, pass_real_t tilt,
    int has_x_septum, pass_real_t septum_x,
    int has_y_septum, pass_real_t septum_y, pass_real_t thickness,
    int is_thick, pass_real_t s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index || tag[i] <= 0) return;

    pass_real_t xi = x[i], pxi = px[i];
    pass_real_t yi = y[i], pyi = py[i];
    pass_real_t zi = z[i], dpi = dp[i];
    int ti = tag[i];
    float lp = lost_position[i];
    int lt = lost_turn[i];

    pass_real_t co = cos(tilt), si = sin(tilt);
    if (fabs(tilt) > PASS_EPS) {
        pass_real_t tx = xi * co - yi * si;
        pass_real_t ty = xi * si + yi * co;
        pass_real_t tpx = pxi * co - pyi * si;
        pass_real_t tpy = pxi * si + pyi * co;
        xi = tx; yi = ty; pxi = tpx; pyi = tpy;
    }

    bool field = true;
    bool septum_lost = false;
    if (has_x_septum) {
        if (septum_x > 0) {
            field = field && (xi > septum_x + thickness);
            septum_lost = septum_lost || ((xi > septum_x) && (xi <= septum_x + thickness));
        } else {
            field = field && (xi < septum_x - thickness);
            septum_lost = septum_lost || ((xi < septum_x) && (xi >= septum_x - thickness));
        }
    }
    if (has_y_septum) {
        if (septum_y > 0) {
            field = field && (yi > septum_y + thickness);
            septum_lost = septum_lost || ((yi > septum_y) && (yi <= septum_y + thickness));
        } else {
            field = field && (yi < septum_y - thickness);
            septum_lost = septum_lost || ((yi < septum_y) && (yi >= septum_y - thickness));
        }
    }
    septum_lost = septum_lost && !field;
    if (septum_lost) {
        ti = -abs(ti);
        lp = (float)s_position;
        lt = turn;
    }

    if (ti > 0) {
        if (is_thick) {
            if (field) {
                if (elseparator_drift(xi, yi, zi, pxi, pyi, dpi, ti, &lp, &lt,
                                      beta_gamma, inv_gamma, length * (pass_real_t)0.5,
                                      s_position, turn)) {
                    pxi += kick_x;
                    pyi += kick_y;
                    elseparator_drift(xi, yi, zi, pxi, pyi, dpi, ti, &lp, &lt,
                                      beta_gamma, inv_gamma, length * (pass_real_t)0.5,
                                      s_position, turn);
                }
            } else {
                elseparator_drift(xi, yi, zi, pxi, pyi, dpi, ti, &lp, &lt,
                                  beta_gamma, inv_gamma, length, s_position, turn);
            }
        } else if (field) {
            pxi += kick_x;
            pyi += kick_y;
        }
    }

    if (ti > 0 && fabs(tilt) > PASS_EPS) {
        pass_real_t tx = xi * co + yi * si;
        pass_real_t ty = -xi * si + yi * co;
        pass_real_t tpx = pxi * co + pyi * si;
        pass_real_t tpy = -pxi * si + pyi * co;
        xi = tx; yi = ty; pxi = tpx; pyi = tpy;
    }
    x[i] = xi; px[i] = pxi; y[i] = yi; py[i] = pyi; z[i] = zi;
    tag[i] = ti; lost_position[i] = lp; lost_turn[i] = lt;
}
'''

ELSEPARATOR_SOURCE = CUDA_REAL_PREAMBLE + ELSEPARATOR_KERNEL_BODY
_elseparator_kernels = {}


def _get_elseparator_kernel(dtype):
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU ElSeparator tracking requires CUDA dependencies.") from exc
    key = np.dtype(dtype)
    if key not in _elseparator_kernels:
        _elseparator_kernels[key] = cp.RawKernel(
            ELSEPARATOR_SOURCE, "transfer_elseparator",
            options=("--std=c++14", f"-DPASS_USE_FLOAT={int(key == np.dtype(np.float32))}"),
        )
    return _elseparator_kernels[key]
