from functools import lru_cache
import logging

import numpy as np

from PASS.commands.command import Command
from PASS.commands.element.error import FieldErrors
from PASS.utils.slicing import print_element_slicing, configure_element_slicing, run_body_slices, transport_with_center
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.aperture import check_aperture_cpu
from PASS.commands.element.multipole import launch_multipole

logger = logging.getLogger(__name__)


@Command.register("quadrupole")
class Quadrupole(Command):
    """Track a quadrupole with a linear matrix or exact drift-kick-drift slices."""

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = kwargs["length (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]
        self.field_errors = FieldErrors(kwargs)

        if self.length < 0.0:
            raise ValueError(f"The length of Quadrupole {self.cmd_name} is {self.length}, which should be >= 0")
        if self.length > const.eps:
            self.is_thick = True
        else:
            self.is_thick = False

        self.k1l = kwargs.get("k1l", 0.0)
        self.k1sl = kwargs.get("k1sl", 0.0)
        if self.is_thick:
            self.k1 = self.k1l / self.length
            self.k1s = self.k1sl / self.length
        else:
            self.k1 = 0.0
            self.k1s = 0.0
        if abs(self.k1l) < const.eps and abs(self.k1sl) < const.eps:
            logger.warning(f"Quadrupole {self.cmd_name} has zero integrated strength (k1l=0, k1sl=0). It will act as a pure drift.")
        if abs(self.k1l) > const.eps and abs(self.k1sl) > const.eps:
            logger.warning(
                f"Quadrupole {self.cmd_name} has both normal and skew components (k1l={self.k1l}, k1sl={self.k1sl}). It will act as a combined quadrupole."
            )

        self.model = kwargs.get("model", "adaptive")
        if self.model not in ["adaptive", "drift-kick-drift-exact", "mat-kick-mat"]:
            raise ValueError(
                f"The model of Quadrupole {self.cmd_name} is {self.model}, which should be 'adaptive', 'drift-kick-drift-exact' or 'mat-kick-mat'.")
        if self.model == "adaptive":
            self.model = "mat-kick-mat"

        self.num_slice = kwargs.get("num slices", 1)
        if self.num_slice < 1:
            logger.warning(f"The number of slices of {self.cmd_name} is {self.num_slice}, which should be >= 1. It has been changed to 1 now.")
            self.num_slice = 1

        self.integrator = kwargs.get("integrator", "adaptive")
        if self.integrator not in ["adaptive", "uniform", "yoshida4"]:
            raise ValueError(
                f"The integrator of Quadrupole {self.cmd_name} is {self.integrator}, which should be 'adaptive', 'uniform' or 'yoshida4'.")
        if self.integrator == "adaptive":
            self.integrator = "uniform"

        # MKM precomputation: rotation diagonalization
        # theta = 0.5 * arctan2(k1s, k1) is delta-independent
        # (k1 and k1s both scale by chi/(1+delta), so the ratio is unchanged)
        if abs(self.k1s) > const.eps:
            self.is_skew = True
            theta = 0.5 * np.arctan2(-self.k1s, self.k1)
            self.cos_theta = np.cos(theta)
            self.sin_theta = np.sin(theta)
            self.k_eff_base = np.sqrt(self.k1**2 + self.k1s**2)
        else:
            self.is_skew = False
            self.cos_theta = 1.0
            self.sin_theta = 0.0
            self.k_eff_base = self.k1

        self.aperture_type: str = kwargs.get("aperture type", "off").lower()
        self.aperture_value: list = kwargs.get("aperture value", [])
        if not isinstance(self.aperture_value, list):
            raise ValueError(f"Aperture value of {self.cmd_name} must be a list, but got {type(self.aperture_value)}")

        configure_element_slicing(self, sim, kwargs)
        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, Length={self.length:.4f}, "
                    f"IsThick={self.is_thick}, K1L={self.k1l:.6f}, K1SL={self.k1sl:.6f}, "
                    f"NumSlice={self.num_slice:d}, Model={self.model:s}, Integrator={self.integrator:s}, "
                    f"ApertureType={self.aperture_type:s}, ApertureValue={self.aperture_value}")
        print_element_slicing(self)
        set_normal_logging()

    def execute_cpu(self, sim):
        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        turn = sim.state.turn

        for i, bunch in enumerate(bunches):
            self._track_quadrupole_cpu(beam, bunch, turn)
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
        all_zero = (abs(self.k1l) < const.eps and abs(self.k1sl) < const.eps)
        if self.is_thick and self.model == "mat-kick-mat" and not all_zero:
            launch_quadrupole_matrix(self, sim)
            return True
        if self.is_thick:
            mode = 2 if all_zero else 1
            knl = np.array([
                0.0,
                self.k1,
            ], dtype=np.float64)
            ksl = np.array([0.0, self.k1s], dtype=np.float64)
        else:
            mode = 0
            knl = np.array([0.0, self.k1l], dtype=np.float64)
            ksl = np.array([0.0, self.k1sl], dtype=np.float64)
        launch_multipole(self, sim, knl, ksl, np.array([1.0, 1.0], dtype=np.float64), mode)
        return True

    def _track_quadrupole_cpu(self, beam: Beam, bunch: BunchInfo, turn: int):

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
            self._quadrupole_kick_cpu(self.k1l, self.k1sl, x, px, y, py, tag, mask, chi)
            self.field_errors.kick_cpu(x, px, y, py, tag)
            return

        if self._sc_nodes:

            def transport(ds, on_center):
                if self.model == "mat-kick-mat":

                    mask[:] = tag > 0
                    self._mat_kick_mat_cpu(x, px, y, py, z, dp, tag, mask, chi, beta0, ds, on_center=on_center)
                else:
                    step = self._dkd_step_cpu if self.integrator == "uniform" else self._dkd_yoshida4_cpu
                    step(x, px, y, py, z, dp, tag, mask, ds, self.k1, self.k1s, chi, beta0, on_center=on_center)

            run_body_slices(self, beam, bunch, turn, transport)
            return

        if (abs(self.k1l) < const.eps and abs(self.k1sl) < const.eps) and not self.field_errors.active:
            self._drift_exact_cpu(self.length, x, px, y, py, z, dp, tag, mask, beta0)
        else:
            if self.model == "mat-kick-mat":
                ds = self.length / self.num_slice
                for _ in range(self.num_slice):
                    self._mat_kick_mat_cpu(x, px, y, py, z, dp, tag, mask, chi, beta0, ds)
            else:  # drift-kick-drift-exact
                ds = self.length / self.num_slice
                for _ in range(self.num_slice):
                    if self.integrator == "uniform":
                        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds, self.k1, self.k1s, chi, beta0)
                    elif self.integrator == "yoshida4":
                        self._dkd_yoshida4_cpu(x, px, y, py, z, dp, tag, mask, ds, self.k1, self.k1s, chi, beta0)

        newly_lost = alive_before & (tag < 0)
        if np.any(newly_lost):
            lost_position = p.lost_position[start:end]
            lost_turn = p.lost_turn[start:end]
            lost_position[newly_lost] = self.s
            lost_turn[newly_lost] = turn

    def _mat_kick_mat_cpu(self, x, px, y, py, z, dp, tag, mask, chi, beta0, ds, on_center=None):

        def advance(length):
            mask[:] = tag > 0
            self._matrix_cpu(x, px, y, py, z, dp, tag, mask, chi, beta0, length)

        if self.field_errors.active:
            from PASS.commands.element.error import _transport_matrix_errors
            _transport_matrix_errors(advance, lambda scale: self.field_errors.kick_cpu(x, px, y, py, tag, scale), ds, self.length, self.integrator,
                                     on_center)
        else:
            transport_with_center(advance, ds, on_center)

    def _matrix_cpu(self, x, px, y, py, z, dp, tag, mask, chi, beta0, ds):
        """Apply the linear map in the quadrupole's principal axes.

        K includes the particle momentum ratio. The z update uses the path
        length of the linearized trajectory; nonlinear pz terms are omitted."""
        L = ds
        momentum_ratio = 1.0 + dp

        if self.is_skew:
            cos_theta = self.cos_theta
            sin_theta = self.sin_theta
            u = cos_theta * x + sin_theta * y
            pu = cos_theta * px + sin_theta * py
            v = -sin_theta * x + cos_theta * y
            pv = -sin_theta * px + cos_theta * py
        else:
            u = x
            pu = px
            v = y
            pv = py

        K = self.k_eff_base * chi / momentum_ratio

        K_pos_u = K > 0.0
        K_neg_u = K < 0.0
        K_zero_u = np.abs(K) < 1e-15

        sqrt_K = np.sqrt(np.abs(K))
        KL = sqrt_K * L

        Cu = np.where(K_pos_u, np.cos(KL), np.cosh(KL))
        Su = np.where(K_pos_u, np.sin(KL) / np.where(K_zero_u, 1.0, sqrt_K), np.sinh(KL) / np.where(K_zero_u, 1.0, sqrt_K))
        Su = np.where(K_zero_u, L, Su)
        Cu = np.where(K_zero_u, 1.0, Cu)

        K_v = -K
        K_pos_v = K_v > 0.0
        K_neg_v = K_v < 0.0
        K_zero_v = np.abs(K_v) < 1e-15

        sqrt_Kv = np.sqrt(np.abs(K_v))
        KLv = sqrt_Kv * L

        Cv = np.where(K_pos_v, np.cos(KLv), np.cosh(KLv))
        Sv = np.where(K_pos_v, np.sin(KLv) / np.where(K_zero_v, 1.0, sqrt_Kv), np.sinh(KLv) / np.where(K_zero_v, 1.0, sqrt_Kv))
        Sv = np.where(K_zero_v, L, Sv)
        Cv = np.where(K_zero_v, 1.0, Cv)

        slope_u = pu / momentum_ratio
        slope_v = pv / momentum_ratio

        u_new = u * Cu + slope_u * Su
        pu_new = (-K * u * Su + slope_u * Cu) * momentum_ratio

        v_new = v * Cv + slope_v * Sv
        pv_new = (-K_v * v * Sv + slope_v * Cv) * momentum_ratio

        if self.is_skew:
            x_new = cos_theta * u_new - sin_theta * v_new
            px_new = cos_theta * pu_new - sin_theta * pv_new
            y_new = sin_theta * u_new + cos_theta * v_new
            py_new = sin_theta * pu_new + cos_theta * pv_new
        else:
            x_new = u_new
            px_new = pu_new
            y_new = v_new
            py_new = pv_new

        # Integrate the squared transverse slopes along the matrix trajectory.

        A = -K * u
        B = slope_u
        C_coeff = -K_v * v
        D = slope_v

        path_length_excess = np.zeros_like(x)

        # u-plane path length correction (Kx = K)
        Kx_nonzero = ~K_zero_u
        K_safe_u = np.where(K_zero_u, 1.0, K)
        path_length_excess = np.where(
            Kx_nonzero, path_length_excess + 0.5 * (-(A**2 * Cu * Su) / (2.0 * K_safe_u) + (B**2 * Cu * Su) / 2.0 + (A**2 * L) / (2.0 * K_safe_u) +
                                                    (B**2 * L) / 2.0 - (A * B * Cu**2) / K_safe_u + (A * B) / K_safe_u), path_length_excess)
        path_length_excess = np.where(K_zero_u, path_length_excess + 0.5 * B**2 * L, path_length_excess)

        # v-plane path length correction (Ky = K_v = -K)
        Ky_nonzero = ~K_zero_v
        K_safe_v = np.where(K_zero_v, 1.0, K_v)

        path_length_excess = np.where(
            Ky_nonzero,
            path_length_excess + 0.5 * (-(C_coeff**2 * Cv * Sv) / (2.0 * K_safe_v) + (D**2 * Cv * Sv) / 2.0 + (C_coeff**2 * L) / (2.0 * K_safe_v) +
                                        (D**2 * L) / 2.0 - (C_coeff * D * Cv**2) / K_safe_v + (C_coeff * D) / K_safe_v), path_length_excess)
        path_length_excess = np.where(K_zero_v, path_length_excess + 0.5 * D**2 * L, path_length_excess)

        # Keep the small path excess and velocity correction separate from L.
        inv_gamma_sq = max(0.0, 1.0 - beta0**2)
        beta_ratio_squared_change = -inv_gamma_sq * dp * (2.0 + dp) / momentum_ratio**2
        beta0_over_beta_minus_one = beta_ratio_squared_change / (np.sqrt(1.0 + beta_ratio_squared_change) + 1.0)
        delta_z = -path_length_excess - (L + path_length_excess) * beta0_over_beta_minus_one

        active_mask = mask
        x[:] = x_new * active_mask + x * (1.0 - active_mask)
        px[:] = px_new * active_mask + px * (1.0 - active_mask)
        y[:] = y_new * active_mask + y * (1.0 - active_mask)
        py[:] = py_new * active_mask + py * (1.0 - active_mask)
        z += delta_z * active_mask

    def _dkd_yoshida4_cpu(self, x, px, y, py, z, dp, tag, mask, ds, k1, k1s, chi, beta0, on_center=None):
        """Compose three drift-kick-drift steps with Yoshida coefficients."""
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, k1, k1s, chi, beta0)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z0, k1, k1s, chi, beta0, on_center=on_center)
        self._dkd_step_cpu(x, px, y, py, z, dp, tag, mask, ds * const.yoshida_z1, k1, k1s, chi, beta0)

    def _dkd_step_cpu(self, x, px, y, py, z, dp, tag, mask, ds, k1, k1s, chi, beta0, on_center=None):
        """Apply one drift-kick-drift step; Yoshida composition may use negative ds."""
        self._drift_exact_cpu(ds * 0.5, x, px, y, py, z, dp, tag, mask, beta0)
        self._quadrupole_kick_cpu(k1 * ds, k1s * ds, x, px, y, py, tag, mask, chi)
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

    def _quadrupole_kick_cpu(self, k1l_eff, k1sl_eff, x, px, y, py, tag, mask, chi):
        """Apply normal and skew kicks using integrated quadrupole strengths."""
        if abs(k1l_eff) < const.eps and abs(k1sl_eff) < const.eps:
            return

        active = (tag > 0).astype(mask.dtype, copy=False)
        k1l_mask = k1l_eff * active

        px -= chi * k1l_mask * x
        py += chi * k1l_mask * y

        if abs(k1sl_eff) > const.eps:
            k1sl_mask = k1sl_eff * active
            px += chi * k1sl_mask * y
            py += chi * k1sl_mask * x


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

QUAD_MATRIX_BODY = r'''
extern "C" __global__ void track_quadrupole_matrix(
    pass_real_t* __restrict__ x,
    pass_real_t* __restrict__ px,
    pass_real_t* __restrict__ y,
    pass_real_t* __restrict__ py,
    pass_real_t* __restrict__ z,
    const pass_real_t* __restrict__ dp,
    const int* __restrict__ tag,
    int start_index,
    int end_index,
    pass_real_t beta0,
    pass_real_t reference_beta_gamma,
    pass_real_t inv_gamma,
    pass_real_t ds,
    pass_real_t cos_theta,
    pass_real_t sin_theta,
    pass_real_t k_eff_base,
    int num_slice
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (index >= end_index || tag[index] <= 0)
        return;

    pass_real_t xi = x[index], pxi = px[index];
    pass_real_t yi = y[index], pyi = py[index];
    pass_real_t zi = z[index], dpi = dp[index];
    pass_real_t one = (pass_real_t)1;

    for (int slice = 0; slice < num_slice; ++slice) {
        pass_real_t momentum_ratio = one + dpi;
        pass_real_t u = cos_theta * xi + sin_theta * yi;
        pass_real_t pu = cos_theta * pxi + sin_theta * pyi;
        pass_real_t v = -sin_theta * xi + cos_theta * yi;
        pass_real_t pv = -sin_theta * pxi + cos_theta * pyi;

        pass_real_t K = k_eff_base / momentum_ratio;
        pass_real_t absK = fabs(K);
        pass_real_t sqrtK = sqrt(absK);
        pass_real_t KL = sqrtK * ds;
        pass_real_t Cu, Su;
        if (absK < (pass_real_t)1e-15) {
            Cu = one;
            Su = ds;
        } else if (K > (pass_real_t)0) {
            Cu = cos(KL);
            Su = sin(KL) / sqrtK;
        } else {
            Cu = cosh(KL);
            Su = sinh(KL) / sqrtK;
        }

        pass_real_t Kv = -K;
        pass_real_t absKv = fabs(Kv);
        pass_real_t sqrtKv = sqrt(absKv);
        pass_real_t KLv = sqrtKv * ds;
        pass_real_t Cv, Sv;
        if (absKv < (pass_real_t)1e-15) {
            Cv = one;
            Sv = ds;
        } else if (Kv > (pass_real_t)0) {
            Cv = cos(KLv);
            Sv = sin(KLv) / sqrtKv;
        } else {
            Cv = cosh(KLv);
            Sv = sinh(KLv) / sqrtKv;
        }

        pass_real_t slope_u = pu / momentum_ratio;
        pass_real_t slope_v = pv / momentum_ratio;
        pass_real_t A = -K * u;
        pass_real_t B = slope_u;
        pass_real_t C = -Kv * v;
        pass_real_t D = slope_v;

        pass_real_t path_length_excess = 0;
        if (absK < (pass_real_t)1e-15) {
            path_length_excess += (pass_real_t)0.5 * B * B * ds;
        } else {
            path_length_excess +=
                (pass_real_t)0.5 * (-(A * A * Cu * Su) / ((pass_real_t)2 * K) + (B * B * Cu * Su) / (pass_real_t)2 +
                                    (A * A * ds) / ((pass_real_t)2 * K) + (B * B * ds) / (pass_real_t)2 - (A * B * Cu * Cu) / K + (A * B) / K);
        }
        if (absKv < (pass_real_t)1e-15) {
            path_length_excess += (pass_real_t)0.5 * D * D * ds;
        } else {
            path_length_excess +=
                (pass_real_t)0.5 * (-(C * C * Cv * Sv) / ((pass_real_t)2 * Kv) + (D * D * Cv * Sv) / (pass_real_t)2 +
                                    (C * C * ds) / ((pass_real_t)2 * Kv) + (D * D * ds) / (pass_real_t)2 - (C * D * Cv * Cv) / Kv + (C * D) / Kv);
        }

        pass_real_t un = u * Cu + slope_u * Su;
        pass_real_t pun = (-K * u * Su + slope_u * Cu) * momentum_ratio;
        pass_real_t vn = v * Cv + slope_v * Sv;
        pass_real_t pvn = (-Kv * v * Sv + slope_v * Cv) * momentum_ratio;

        xi = cos_theta * un - sin_theta * vn;
        pxi = cos_theta * pun - sin_theta * pvn;
        yi = sin_theta * un + cos_theta * vn;
        pyi = sin_theta * pun + cos_theta * pvn;

        pass_real_t inv_gamma_sq = inv_gamma * inv_gamma;
        pass_real_t beta_ratio_squared_change = -inv_gamma_sq * dpi * ((pass_real_t)2 + dpi) / (momentum_ratio * momentum_ratio);
        pass_real_t beta0_over_beta_minus_one = beta_ratio_squared_change / (sqrt(one + beta_ratio_squared_change) + one);
        zi += -path_length_excess - (ds + path_length_excess) * beta0_over_beta_minus_one;
    }

    x[index] = xi;
    px[index] = pxi;
    y[index] = yi;
    py[index] = pyi;
    z[index] = zi;
}
'''


@lru_cache(maxsize=None)
def _get_quadrupole_kernel(dtype):
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError("GPU quadrupole tracking requires the optional 'cuda' dependencies.") from exc

    dtype = np.dtype(dtype)
    return cp.RawKernel(
        CUDA_REAL_PREAMBLE + QUAD_MATRIX_BODY,
        "track_quadrupole_matrix",
        options=("--std=c++14", f"-DPASS_USE_FLOAT={int(dtype == np.dtype(np.float32))}"),
    )


def launch_quadrupole_matrix(element, sim):
    p = sim.beams[element.beam_id].particles
    kernel = _get_quadrupole_kernel(p.dtype.str)
    beam = sim.beams[element.beam_id]
    real = p.real
    threads = 256
    for bunch in beam.bunches:
        n = bunch.end_idx - bunch.start_idx
        if n > 0:
            blocks = (n + threads - 1) // threads
            kernel((blocks, ), (threads, ), (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, np.int32(bunch.start_idx), np.int32(
                bunch.end_idx), real(bunch.beta), real(bunch.beta * bunch.gamma), real(1.0 / bunch.gamma), real(element.length / element.num_slice),
                                             real(element.cos_theta), real(element.sin_theta), real(element.k_eff_base), np.int32(element.num_slice)))
        if n > 0:
            from PASS.utils.aperture import check_aperture_gpu
            check_aperture_gpu(beam, bunch, element.aperture_type, element.aperture_value, element.s, sim.state.turn)
        if abs(element.length) >= const.eps:
            bunch.t0 += element.length / (bunch.beta * const.c)
