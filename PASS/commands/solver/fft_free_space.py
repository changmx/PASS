"""Batched free-space FFT Green-function solver for transverse PIC.

This is Hockney-style zero-padded linear convolution on a rectangular grid.
It models an open transverse domain, not a conducting aperture: the potential
has an arbitrary logarithmic reference and the returned fields are the useful
physical quantities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import irfftn, next_fast_len, rfftn

from PASS.utils.constants import const
from .field_result import FieldResult, GPUFieldSolver, _launch_gpu


@dataclass
class FFTFreeSpaceSolver:
    """Cached two-dimensional free-space Green convolution kernels."""

    geometry: object
    aperture_mask: np.ndarray
    interior_mask: np.ndarray
    _shape: tuple[int, int]
    _potential_kernel_fft: np.ndarray | None
    _ex_kernel_fft: np.ndarray
    _ey_kernel_fft: np.ndarray

    @property
    def interior_indices(self) -> np.ndarray:
        return np.flatnonzero(self.interior_mask.ravel())

    def solve(self, density: np.ndarray, *, compute_potential: bool = True) -> FieldResult:
        """Convolve every density slice with free-space potential and field kernels.

        ``density`` is C/m^2.  The returned potential is V m up to an additive
        constant; ``Ex`` and ``Ey`` are integrated transverse fields in V.
        ``compute_potential=False`` skips the potential inverse transform and
        returns ``potential=None``. Default calls retain all three outputs.
        """
        source = np.asarray(density, dtype=float)
        if not np.all(np.isfinite(source)):
            raise ValueError("density must be finite")
        squeeze = source.ndim == 2
        if squeeze:
            source = source[None, ...]
        if source.ndim != 3 or source.shape[1:] != self.aperture_mask.shape:
            raise ValueError("density must have shape (n_slice, ny, nx) matching the grid")

        if compute_potential and self._potential_kernel_fft is None:
            # Tracking normally needs only Ex/Ey. Build this third cached
            # spectrum only when a caller first requests the potential.
            ny_pad, nx_pad = self._shape
            iy, ix = np.arange(ny_pad), np.arange(nx_pad)
            dy = np.where(iy < self.geometry.ny, iy, iy - ny_pad)[:, None] * self.geometry.dy
            dx = np.where(ix < self.geometry.nx, ix, ix - nx_pad)[None, :] * self.geometry.dx
            radius_squared = dx * dx + dy * dy
            with np.errstate(divide="ignore"):
                kernel = -np.log(np.sqrt(radius_squared) / np.sqrt(self.geometry.dx * self.geometry.dy)) / (2.0 * const.pi * const.epsilon0)
            kernel[0, 0] = 0.0
            self._potential_kernel_fft = rfftn(kernel)
            del dx, dy, radius_squared, kernel

        source_fft = rfftn(source, s=self._shape, axes=(-2, -1))
        # Scale once in Fourier space instead of allocating another density
        # stack. A single scratch spectrum serves all inverse transforms.
        source_fft *= self.geometry.dx * self.geometry.dy
        scratch = np.empty_like(source_fft)
        crop = (slice(None), slice(0, self.geometry.ny), slice(0, self.geometry.nx))

        def convolve(kernel):
            np.multiply(source_fft, kernel, out=scratch)
            padded = irfftn(scratch, s=self._shape, axes=(-2, -1), overwrite_x=True)
            # Own only the physical grid, releasing the padded result before
            # the next transform instead of retaining three padded bases.
            return padded[crop].copy()

        potential = convolve(self._potential_kernel_fft) if compute_potential else None
        ex = convolve(self._ex_kernel_fft)
        ey = convolve(self._ey_kernel_fft)
        if squeeze:
            return FieldResult(None if potential is None else potential[0], ex[0], ey[0])
        return FieldResult(potential, ex, ey)


def build_fft_free_space_resources(geometry) -> FFTFreeSpaceSolver:
    """Build zero-padded free-space kernels for one uniform grid geometry."""
    ny_pad = next_fast_len(2 * geometry.ny - 1)
    nx_pad = next_fast_len(2 * geometry.nx - 1)
    iy = np.arange(ny_pad)
    ix = np.arange(nx_pad)
    dy = np.where(iy < geometry.ny, iy, iy - ny_pad)[:, None] * geometry.dy
    dx = np.where(ix < geometry.nx, ix, ix - nx_pad)[None, :] * geometry.dx
    radius_squared = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        ex_kernel = dx / (2.0 * const.pi * const.epsilon0 * radius_squared)
        ey_kernel = dy / (2.0 * const.pi * const.epsilon0 * radius_squared)
    ex_kernel[0, 0] = 0.0
    ey_kernel[0, 0] = 0.0
    aperture_mask = np.ones((geometry.ny, geometry.nx), dtype=bool)
    return FFTFreeSpaceSolver(
        geometry,
        aperture_mask,
        aperture_mask.copy(),
        (ny_pad, nx_pad),
        None,
        rfftn(ex_kernel),
        rfftn(ey_kernel),
    )


def solve_poisson_fft_free_space(density: np.ndarray, geometry, *, compute_potential: bool = True):
    """One-shot wrapper around :func:`build_fft_free_space_resources`."""
    return build_fft_free_space_resources(geometry).solve(density, compute_potential=compute_potential)


class GPUFFTFreeSpaceSolver(GPUFieldSolver):
    def __init__(self, geometry, dtype="float64", *, batch_size=16):
        import cupy as cp

        super().__init__(geometry, dtype)
        if (
            isinstance(batch_size, bool)
            or int(batch_size) != batch_size
            or batch_size < 1
        ):
            raise ValueError("FFT batch_size must be a positive integer")
        self.batch_size = int(batch_size)
        g = geometry
        self.aperture_mask = np.ones((g.ny, g.nx), dtype=bool)
        self.interior_mask = self.aperture_mask.copy()
        # Small-prime padding preserves the exact zero-padded linear convolution.
        self.shape = (
            next_fast_len(2 * g.ny - 1, real=True),
            next_fast_len(2 * g.nx - 1, real=True),
        )
        py, px = self.shape
        iy, ix = np.arange(py), np.arange(px)
        y = np.where(iy < g.ny, iy, iy - py)[:, None] * g.dy
        x = np.where(ix < g.nx, ix, ix - px)[None, :] * g.dx
        r2 = x * x + y * y
        r2[0, 0] = 1
        factor = 1 / (2 * const.pi * const.epsilon0)
        kernels = [
            x / r2 * factor,
            y / r2 * factor,
            -np.log(np.sqrt(r2) / np.sqrt(g.dx * g.dy)) * factor,
        ]
        self.kernels = []
        for kernel in kernels:
            kernel[0, 0] = 0
            self.kernels.append(cp.fft.rfft2(cp.asarray(kernel, dtype=self.dtype)))

    def _prepare(self, ns):
        import cupy as cp
        from cupyx.scipy.fft import get_fft_plan

        w = self._work
        py, px = self.shape
        sizes = {min(ns, self.batch_size)}
        if ns % self.batch_size:
            sizes.add(ns % self.batch_size)
        w["fft_batches"] = {}
        for count in sizes:
            batch = {}
            batch["padded"] = cp.empty((count, py, px), self.dtype)
            batch["spectrum"] = cp.empty((count, py, px // 2 + 1), self.complex_dtype)
            batch["scratch"] = cp.empty_like(batch["spectrum"])
            batch["forward"] = get_fft_plan(
                batch["padded"], axes=(-2, -1), value_type="R2C"
            )
            batch["inverse"] = get_fft_plan(
                batch["scratch"], shape=self.shape, axes=(-2, -1), value_type="C2R"
            )
            w["fft_batches"][count] = batch

    def solve(self, density, *, compute_potential=True, validate=True, copy=True):
        import cupy as cp
        from cupy.cuda import cufft

        src, squeeze = self._source(density, validate)
        g, w = self.geometry, self._work
        py, px = self.shape
        shape_args = (np.int32(g.nx), np.int32(g.ny), np.int32(px), np.int32(py))
        for first in range(0, w["slices"], self.batch_size):
            count = min(self.batch_size, w["slices"] - first)
            batch = w["fft_batches"][count]
            padded = batch["padded"]
            _launch_gpu(
                _FFT_CUDA,
                "fft_pad",
                padded.size,
                (
                    src[first : first + count],
                    padded,
                    *shape_args,
                    np.int64(padded.size),
                ),
                self.dtype,
            )
            batch["forward"].fft(padded, batch["spectrum"], cufft.CUFFT_FORWARD)
            for kernel, target in zip(self.kernels, ("ex", "ey", "phi")):
                if target == "phi" and not compute_potential:
                    continue
                out = w[target][first : first + count]
                cp.multiply(batch["spectrum"], kernel, out=batch["scratch"])
                batch["inverse"].fft(batch["scratch"], padded, cufft.CUFFT_INVERSE)
                _launch_gpu(
                    _FFT_CUDA,
                    "fft_crop",
                    out.size,
                    (
                        padded,
                        out,
                        *shape_args,
                        np.int64(out.size),
                        self.scalar(g.dx * g.dy / (px * py)),
                    ),
                    self.dtype,
                )
        return self._result(w["phi"] if compute_potential else None, squeeze, copy)

_FFT_CUDA = r"""
extern "C" __global__ void fft_pad(const T* src, T* dst, int nx, int ny,
    int px, int py, long long size) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=size) return;
    int x=i%px,y=(i/px)%py;
    dst[i]=(x<nx && y<ny) ? src[(i/(px*py))*nx*ny+y*nx+x] : 0;
}
extern "C" __global__ void fft_crop(const T* src, T* dst, int nx, int ny,
    int px, int py, long long size, T scale) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<size) dst[i]=src[(i/(nx*ny))*px*py+((i/nx)%ny)*px+i%nx]*scale;
}
"""
