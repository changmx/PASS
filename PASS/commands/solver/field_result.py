"""Common field results and GPU workspace/compilation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class FieldResult:
    """Potential and longitudinally integrated transverse fields.

    ``potential`` has units V m and ``integrated_ex``/``integrated_ey`` have
    units V when the supplied source density is in C/m^2.
    ``potential`` is None only when an optional field-only solve omits it.
    """

    potential: np.ndarray | None
    integrated_ex: np.ndarray
    integrated_ey: np.ndarray

    @property
    def ex(self) -> np.ndarray:
        return self.integrated_ex

    @property
    def ey(self) -> np.ndarray:
        return self.integrated_ey


class GPUFieldSolver:
    """Device/stream ownership and reusable outputs shared by GPU field solvers.

    Calls on a resource are serialized. Results own their arrays by default;
    ``copy=False`` borrows buffers until the next solve. ``prepare`` allocates
    the current slice batch, and ``close`` releases device resources.
    """

    def __init__(self, geometry, dtype):
        import cupy as cp

        self.geometry = geometry
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise TypeError("GPU field precision must be float32 or float64")
        self.scalar = self.dtype.type
        self.complex_dtype = np.complex64 if self.dtype.itemsize == 4 else np.complex128
        self.device = cp.cuda.runtime.getDevice()
        self.stream = cp.cuda.get_current_stream()
        self._work = None
        self._closed = False

    def _check_context(self):
        import cupy as cp

        if self._closed:
            raise RuntimeError("GPU field resources have been closed")
        if (
            cp.cuda.runtime.getDevice() != self.device
            or cp.cuda.get_current_stream().ptr != self.stream.ptr
        ):
            raise RuntimeError(
                "GPU resources must be used on their creation device and stream"
            )

    def _source(self, density, validate):
        import cupy as cp

        self._check_context()
        source = cp.asarray(density, dtype=self.dtype, order="C")
        squeeze = source.ndim == 2
        if squeeze:
            source = source[None]
        if (
            source.ndim != 3
            or source.shape[1:] != (self.geometry.ny, self.geometry.nx)
            or source.shape[0] < 1
        ):
            raise ValueError(
                "density must have shape (n_slice, ny, nx) matching the grid"
            )
        if validate and not bool(cp.all(cp.isfinite(source))):
            raise ValueError("density must be finite")
        self.prepare(source.shape[0])
        return source, squeeze

    def prepare(self, num_slices):
        import cupy as cp

        self._check_context()
        if (
            isinstance(num_slices, bool)
            or int(num_slices) != num_slices
            or num_slices < 1
        ):
            raise ValueError("num_slices must be a positive integer")
        num_slices = int(num_slices)
        if self._work is None or self._work["slices"] != num_slices:
            g = self.geometry
            self._work = dict(
                slices=num_slices,
                phi=cp.zeros((num_slices, g.ny, g.nx), self.dtype),
                ex=cp.empty((num_slices, g.ny, g.nx), self.dtype),
                ey=cp.empty((num_slices, g.ny, g.nx), self.dtype),
            )
            self._prepare(num_slices)

    def _result(self, potential, squeeze, copy):
        arrays = (potential, self._work["ex"], self._work["ey"])
        return FieldResult(
            *(
                None
                if a is None
                else (a[0] if squeeze else a).copy()
                if copy
                else (a[0] if squeeze else a)
                for a in arrays
            )
        )

    def _gradient(self):
        g, w = self.geometry, self._work
        _launch_gpu(
            _GRADIENT_CUDA,
            "gradient",
            w["phi"].size,
            (
                w["phi"],
                w["ex"],
                w["ey"],
                np.int32(g.nx),
                np.int32(g.ny),
                np.int64(w["phi"].size),
                self.scalar(g.dx),
                self.scalar(g.dy),
            ),
            self.dtype,
        )

    def close(self):
        import cupy as cp

        if self._closed:
            return
        with cp.cuda.Device(self.device):
            self.stream.synchronize()
            self._work = None
            for name in (
                "inv_lambda",
                "twiddles",
                "kernels",
                "coefficients",
                "indices",
                "row",
                "col",
                "values",
            ):
                if hasattr(self, name):
                    setattr(self, name, None)
            self._closed = True


@lru_cache(maxsize=None)
def _gpu_module(source, dtype, device):
    """Compile an inline source once per precision and device, on first use."""
    import cupy as cp

    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("GPU PIC supports float32 and float64")
    scalar = "float" if dtype.itemsize == 4 else "double"
    with cp.cuda.Device(device):
        return cp.RawModule(
            code=f"#define REAL {scalar}\ntypedef REAL T;\n" + source,
            options=("--std=c++17",),
        )


def _launch_gpu(source, name, size, args, dtype):
    """Launch one of a solver's cached kernels without host-array transfers."""
    import cupy as cp

    if size:
        _gpu_module(source, np.dtype(dtype).str, cp.cuda.runtime.getDevice()).get_function(name)(
            ((size + 255) // 256,), (256,), args
        )


_GRADIENT_CUDA = r"""
extern "C" __global__ void gradient(const T* phi, T* ex, T* ey,
    int nx, int ny, long long size, T dx, T dy) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    int x = i % nx, y = (i / nx) % ny;
    ex[i] = -(phi[i + (x < nx-1)] - phi[i - (x > 0)]) / (dx * ((x > 0 && x < nx-1) ? 2 : 1));
    ey[i] = -(phi[i + (y < ny-1 ? nx : 0)] - phi[i - (y > 0 ? nx : 0)]) / (dy * ((y > 0 && y < ny-1) ? 2 : 1));
}
"""
