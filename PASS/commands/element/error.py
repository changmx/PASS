"""Element errors and their composition with nominal CPU and GPU maps."""
from functools import lru_cache

import numpy as np

from PASS.utils.constants import const


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
    """Dispatch error tracking while retaining separate matrix and bend kicks."""
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
