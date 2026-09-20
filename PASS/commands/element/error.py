"""Element errors and their composition with nominal CPU and GPU maps."""
from functools import lru_cache

import numpy as np

from PASS.utils.constants import const


class AlignmentErrors:
    """Static magnetic DX/DY/DPSI, with apertures and SC in the design frame."""

    def __init__(self, kwargs):
        self.enabled = kwargs.get("is alignment error", False)
        self.dx = float(kwargs.get("alignment dx (m)", 0.0))
        self.dy = float(kwargs.get("alignment dy (m)", 0.0))
        self.dpsi = float(kwargs.get("alignment dpsi (rad)", 0.0))
        if not np.all(np.isfinite([self.dx, self.dy, self.dpsi])):
            raise ValueError("Alignment DX, DY and DPSI must be finite")
        for name, unit in (("ds", "m"), ("dphi", "rad"), ("dtheta", "rad")):
            for key in (f"alignment {name} ({unit})", f"alignment_{name}", name):
                if float(kwargs.get(key, 0.0)) != 0.0:
                    raise ValueError(f"Nonzero alignment {name.upper()} is not supported; use DX, DY and DPSI only")
        self.active = bool(self.enabled and (self.dx != 0 or self.dy != 0 or self.dpsi != 0))
        self.cos_psi = np.cos(self.dpsi)
        self.sin_psi = np.sin(self.dpsi)
        self._frames = {}
        self._gpu_frames = {}
        self._geometry = None

    def prepare(self, element):
        if self._geometry is None:
            self._geometry = (getattr(element, "h", 0.0), element.s - element.length, element.length)

    def enter_frame(self, element, beam, turn, *, gpu=False):
        """Enter the magnetic frame and retain the entry-live masks for exit."""
        if not self.active:
            return None
        self.prepare(element)
        p = beam.particles
        transform = self.transform_gpu if gpu else self.transform_cpu
        masks = [p.tag[bunch.start_idx:bunch.end_idx] > 0 for bunch in beam.bunches]
        for bunch, mask in zip(beam.bunches, masks):
            transform(p, bunch, mask, 0.0, turn=turn)
        return masks

    def exit_frame(self, element, beam, turn, masks, *, gpu=False):
        """Restore design coordinates, including each newly lost particle's plane."""
        if masks is None:
            return
        transform = self.transform_gpu if gpu else self.transform_cpu
        for bunch, mask in zip(beam.bunches, masks):
            transform(beam.particles, bunch, mask, element.length, inverse=True, restore_losses=True, turn=turn)

    def _frame(self, offset, inverse):
        """Rigid magnetic pose about the ideal entrance, evaluated at a plane."""
        curvature = self._geometry[0]
        angle = curvature * offset
        ca, sa = np.cos(angle), np.sin(angle)
        cp, sp = self.cos_psi, self.sin_psi
        reference_x = -2 * np.sin(angle / 2)**2 / curvature if curvature != 0 else 0.0
        qx = (cp - 1) * reference_x - cp * self.dx - sp * self.dy
        qy = -sp * reference_x + sp * self.dx - cp * self.dy
        rotation = (cp * ca**2 + sa**2, sp * ca, (1 - cp) * sa * ca, -sp * ca, cp, sp * sa, (1 - cp) * sa * ca, -sp * sa, cp * sa**2 + ca**2)
        shift = (ca * qx, qy, -sa * qx)
        if inverse:
            rotation = tuple(rotation[3 * j + i] for i in range(3) for j in range(3))
            shift = tuple(-sum(rotation[3 * i + j] * shift[j] for j in range(3)) for i in range(3))
        return (*rotation, *shift)

    def _fixed_frame(self, offset, inverse):
        key = (float(offset), inverse)
        if key not in self._frames:
            self._frames[key] = np.asarray(self._frame(offset, inverse), dtype=np.float64)
        return self._frames[key]

    def transform_cpu(self, p, bunch, mask, offset, *, inverse=False, restore_losses=False, turn=0):
        region = slice(bunch.start_idx, bunch.end_idx)
        indices = np.flatnonzero(mask) + bunch.start_idx
        if not len(indices):
            return
        if restore_losses:
            lost = p.tag[indices] <= 0
            if np.any(lost):
                # Loss positions are recorded in the design longitudinal coordinate.
                lost_mask = mask & (p.tag[region] <= 0)
                positions = np.clip(p.lost_position[indices[lost]].astype(np.float64) - self._geometry[1], 0, self._geometry[2])
                self._transform_cpu(p, bunch, indices[lost], self._frame(positions, inverse), positions, turn)
                mask = mask & ~lost_mask
                indices = np.flatnonzero(mask) + bunch.start_idx
        if len(indices):
            self._transform_cpu(p, bunch, indices, self._fixed_frame(offset, inverse), offset, turn)

    def _transform_cpu(self, p, bunch, indices, frame, offset, turn):
        x, y, px, py = (getattr(p, name)[indices].astype(np.float64) for name in ("x", "y", "px", "py"))
        rx = frame[0] * x + frame[1] * y + frame[9]
        ry = frame[3] * x + frame[4] * y + frame[10]
        if self._geometry[0] == 0 or np.all(np.asarray(offset) == 0):
            p.x[indices], p.y[indices] = rx, ry
            p.px[indices] = frame[0] * px + frame[1] * py
            p.py[indices] = frame[3] * px + frame[4] * py
            return
        momentum_ratio = 1 + p.dp[indices].astype(np.float64)
        pz_sq = momentum_ratio**2 - px**2 - py**2
        pz = np.sqrt(np.maximum(pz_sq, 0))
        ux = frame[0] * px + frame[1] * py + frame[2] * pz
        uy = frame[3] * px + frame[4] * py + frame[5] * pz
        uz = frame[6] * px + frame[7] * py + frame[8] * pz
        rz = frame[6] * x + frame[7] * y + frame[11]
        valid = (pz_sq > 0) & (uz > 0) & np.isfinite(uz) & np.isfinite(rz)
        flight = np.divide(rz, uz, out=np.zeros_like(rz), where=valid)
        p.x[indices], p.y[indices] = rx - flight * ux, ry - flight * uy
        p.px[indices], p.py[indices] = ux, uy
        energy_ratio = np.sqrt(momentum_ratio**2 + 1 / (bunch.beta * bunch.gamma)**2)
        p.z[indices] += flight * bunch.beta * energy_ratio
        newly_lost_mask = ~valid & (p.tag[indices] > 0)
        newly_lost = indices[newly_lost_mask]
        p.tag[newly_lost] = -np.abs(p.tag[newly_lost])
        p.lost_position[newly_lost] = self._geometry[1] + np.broadcast_to(offset, indices.shape)[newly_lost_mask]
        p.lost_turn[newly_lost] = turn

    def transform_gpu(self, p, bunch, mask, offset, *, inverse=False, restore_losses=False, turn=0):
        import cupy as cp

        start, end = bunch.start_idx, bunch.end_idx
        if end <= start:
            return
        key = (p.dtype.str, cp.cuda.runtime.getDevice(), float(offset), inverse)
        if key not in self._gpu_frames:
            values = np.concatenate((self._fixed_frame(offset, inverse), [self.dx, self.dy, self.cos_psi, self.sin_psi, *self._geometry]))
            self._gpu_frames[key] = cp.asarray(values, dtype=np.float64)
        _alignment_kernel(*key[:2])(((end - start + 255) // 256, ), (256, ),
                                    (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, p.lost_position, p.lost_turn, mask, np.int32(start), np.int32(end),
                                     self._gpu_frames[key], np.float64(offset), np.float64(bunch.beta), np.float64(
                                         1 / (bunch.beta * bunch.gamma)**2), np.int32(inverse), np.int32(restore_losses), np.int32(turn)))


def _apply_sc_in_design_frame(element, beam, bunch, node, command, turn, *, gpu=False):
    errors = getattr(element, "alignment_errors", None)
    apply = command.apply_bunch_gpu if gpu else command.apply_bunch_cpu
    if errors is None or not errors.active:
        return apply(element._sc_sim, beam, bunch)
    p = beam.particles
    mask = p.tag[bunch.start_idx:bunch.end_idx] > 0
    transform = errors.transform_gpu if gpu else errors.transform_cpu
    transform(p, bunch, mask, node.offset, inverse=True, turn=turn)
    try:
        return apply(element._sc_sim, beam, bunch)
    finally:
        # Include particles lost by SC: exit_frame restores their loss plane.
        transform(p, bunch, mask, node.offset, turn=turn)


@lru_cache(maxsize=None)
def _alignment_kernel(dtype, device):
    import cupy as cp

    source = r'''
#if PASS_USE_FLOAT
using alignment_real_t = float;
#else
using alignment_real_t = double;
#endif
__device__ void alignment_frame(
    const double* params,
    double offset,
    int inverse,
    double* frame
) {
    double dx = params[12], dy = params[13], cp = params[14], sp = params[15], h = params[16];
    double angle = h * offset, ca = cos(angle), sa = sin(angle);
    double sh = sin(angle / 2);
    double reference_x = h != 0 ? -2 * sh * sh / h : 0;
    double qx = (cp - 1) * reference_x - cp * dx - sp * dy;
    double qy = -sp * reference_x + sp * dx - cp * dy;
    double r[9] = {cp * ca * ca + sa * sa, sp * ca, (1 - cp) * sa * ca, -sp * ca, cp, sp * sa, (1 - cp) * sa * ca, -sp * sa, cp * sa * sa + ca * ca};
    double shift[3] = {ca * qx, qy, -sa * qx};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            frame[3 * i + j] = inverse ? r[3 * j + i] : r[3 * i + j];
        frame[9 + i] = shift[i];
        if (inverse) {
            frame[9 + i] = 0;
            for (int j = 0; j < 3; ++j)
                frame[9 + i] -= frame[3 * i + j] * shift[j];
        }
    }
}
extern "C" __global__ void transform_alignment(
    alignment_real_t* x,
    alignment_real_t* px,
    alignment_real_t* y,
    alignment_real_t* py,
    alignment_real_t* z,
    const alignment_real_t* dp,
    int* tag,
    float* lost_position,
    int* lost_turn,
    const bool* mask,
    int start,
    int end,
    const double* params,
    double offset,
    double beta0,
    double inv_beta_gamma_sq,
    int inverse,
    int restore_losses,
    int turn
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i >= end || !mask[i - start])
        return;
    double frame[12];
    for (int j = 0; j < 12; ++j)
        frame[j] = params[j];
    if (restore_losses && tag[i] <= 0) {
        offset = fmin(fmax((double)lost_position[i] - params[17], 0.), params[18]);
        alignment_frame(params, offset, inverse, frame);
    }
    double xi = x[i], yi = y[i], pxi = px[i], pyi = py[i];
    double rx = frame[0] * xi + frame[1] * yi + frame[9];
    double ry = frame[3] * xi + frame[4] * yi + frame[10];
    if (params[16] == 0 || offset == 0) {
        x[i] = rx;
        y[i] = ry;
        px[i] = frame[0] * pxi + frame[1] * pyi;
        py[i] = frame[3] * pxi + frame[4] * pyi;
        return;
    }
    double momentum_ratio = 1 + (double)dp[i];
    double pz_sq = momentum_ratio * momentum_ratio - pxi * pxi - pyi * pyi;
    double pz = sqrt(fmax(pz_sq, 0.));
    double ux = frame[0] * pxi + frame[1] * pyi + frame[2] * pz;
    double uy = frame[3] * pxi + frame[4] * pyi + frame[5] * pz;
    double uz = frame[6] * pxi + frame[7] * pyi + frame[8] * pz;
    double rz = frame[6] * xi + frame[7] * yi + frame[11];
    bool valid = pz_sq > 0 && uz > 0 && isfinite(uz) && isfinite(rz);
    double flight = valid ? rz / uz : 0;
    x[i] = rx - flight * ux;
    y[i] = ry - flight * uy;
    px[i] = ux;
    py[i] = uy;
    z[i] += flight * beta0 * sqrt(momentum_ratio * momentum_ratio + inv_beta_gamma_sq);
    if (!valid && tag[i] > 0) {
        tag[i] = -abs(tag[i]);
        lost_position[i] = params[17] + offset;
        lost_turn[i] = turn;
    }
}
'''
    with cp.cuda.Device(device):
        return cp.RawKernel(source, "transform_alignment", options=("--std=c++14", f"-DPASS_USE_FLOAT={int(np.dtype(dtype).itemsize == 4)}"))


class FieldErrors:
    """Keep errors fixed during tracking; each signed step supplies its L fraction."""

    def __init__(self, kwargs):
        self.enabled = kwargs.get("is field error", False)
        self.knl = self._coefficients(kwargs.get("field error knl", []), "Field error KNL")
        self.ksl = self._coefficients(kwargs.get("field error ksl", []), "Field error KSL")
        n = max(len(self.knl), len(self.ksl), 1)
        self.knl = np.pad(self.knl, (0, n - len(self.knl)))
        self.ksl = np.pad(self.ksl, (0, n - len(self.ksl)))
        self.active = bool(self.enabled and (np.any(self.knl != 0) or np.any(self.ksl != 0)))
        self.inv_fact = np.ones(n)
        for i in range(1, n):
            self.inv_fact[i] = self.inv_fact[i - 1] / i
        self._gpu_arrays = {}

    @staticmethod
    def _coefficients(values, name):
        result = np.asarray(values, dtype=np.float64)
        if result.ndim != 1 or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite one-dimensional coefficient array")
        return result

    def combine(self, knl, ksl):
        """Return nominal plus enabled error strengths, padding missing orders."""
        knl = self._coefficients(knl, "KiL")
        ksl = self._coefficients(ksl, "KiSL")
        if not self.active:
            return knl, ksl
        n = max(len(knl), len(ksl), len(self.knl))
        normal = np.pad(knl, (0, n - len(knl))) + np.pad(self.knl, (0, n - len(self.knl)))
        skew = np.pad(ksl, (0, n - len(ksl))) + np.pad(self.ksl, (0, n - len(self.ksl)))
        return normal, skew

    def kick_cpu(self, x, px, y, py, tag, scale=1.0):
        if not self.active:
            return
        from PASS.commands.element.multipole import _apply_multipole_kick_cpu

        _apply_multipole_kick_cpu(self.knl, self.ksl, self.inv_fact, x, px, y, py, tag, scale)

    def kick_gpu(self, p, start, end, scale=1.0):
        if not self.active or end <= start:
            return
        import cupy as cp

        key = (p.dtype.str, cp.cuda.runtime.getDevice())
        if key not in self._gpu_arrays:
            self._gpu_arrays[key] = tuple(cp.asarray(a, dtype=p.dtype) for a in (self.knl, self.ksl, self.inv_fact))
        _field_error_kernel(*key)(
            ((end - start + 255) // 256, ), (256, ),
            (p.x, p.px, p.y, p.py, p.tag, np.int32(start), np.int32(end), *self._gpu_arrays[key], np.int32(len(self.knl) - 1), np.float64(scale)))


def _transport_matrix_errors(advance, kick, ds, length, integrator, on_center=None):
    """Symmetric matrix-error composition, retaining signed Yoshida weights."""
    weights = (1.0, ) if integrator == "uniform" else (const.yoshida_z1, const.yoshida_z0, const.yoshida_z1)
    for i, weight in enumerate(weights):
        step = ds * weight
        advance(step / 2)
        kick(step / length)
        if on_center is not None and i == len(weights) // 2:
            on_center()
        advance(step / 2)


def _track_field_errors_gpu(element, sim):
    """Dispatch magnetic transport only; the caller owns aperture and clock."""
    from PASS.commands.element.multipole import launch_multipole

    kind = element.cmd_type.lower()
    if element.is_thick and (element._sc_nodes or kind == "sbend" or (kind == "quadrupole" and element.model == "mat-kick-mat")):
        from PASS.utils.slicing import execute_element_body_gpu
        return execute_element_body_gpu(element, sim)
    knl, ksl, inv_fact = _prepare_field_error_multipoles(element)
    launch_multipole(element, sim, knl, ksl, inv_fact, int(element.is_thick))
    return True


def _prepare_field_error_multipoles(element):
    """Prepare fixed multipole coefficients once for the existing DKD map."""
    coefficients = getattr(element, "_field_error_multipoles", None)
    if coefficients is not None:
        return coefficients
    kind = element.cmd_type.lower()
    if kind == "sbend":
        knl, ksl = [element.k0l], [0.0]
    elif kind == "kicker":
        knl, ksl = [-element.hkick], [element.vkick]
    else:
        order = {"quadrupole": 1, "sextupole": 2, "octupole": 3}[kind]
        knl, ksl = np.zeros(order + 1), np.zeros(order + 1)
        knl[order], ksl[order] = getattr(element, f"k{order}l"), getattr(element, f"k{order}sl")
    knl, ksl = element.field_errors.combine(knl, ksl)
    inv_fact = np.ones(len(knl))
    for i in range(1, len(knl)):
        inv_fact[i] = inv_fact[i - 1] / i
    if element.is_thick:
        knl, ksl = knl / element.length, ksl / element.length
    element._field_error_multipoles = (knl, ksl, inv_fact)
    return element._field_error_multipoles


@lru_cache(maxsize=None)
def _field_error_kernel(dtype, device):
    import cupy as cp

    from PASS.commands.element.multipole import CUDA_REAL_PREAMBLE, MULTIPOLE_KERNEL_BODY

    source = r'''
extern "C" __global__ void field_error_kick(
    const pass_real_t* x,
    pass_real_t* px,
    const pass_real_t* y,
    pass_real_t* py,
    const int* tag,
    int start,
    int end,
    const pass_real_t* knl,
    const pass_real_t* ksl,
    const pass_real_t* inv_fact,
    int order,
    double scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i >= end || tag[i] <= 0)
        return;
    pass_real_t pxi = px[i], pyi = py[i];
    pass_kick(pxi, pyi, x[i], y[i], knl, ksl, inv_fact, order, scale);
    px[i] = pxi;
    py[i] = pyi;
}
'''
    with cp.cuda.Device(device):
        return cp.RawKernel(CUDA_REAL_PREAMBLE + MULTIPOLE_KERNEL_BODY + source,
                            "field_error_kick",
                            options=("--std=c++14", f"-DPASS_USE_FLOAT={int(np.dtype(dtype).itemsize == 4)}"))
