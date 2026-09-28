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
from .field_result import FieldResult, GPUFieldSolver, launch_gpu_kernel


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
    dtype: object = np.dtype("float64")
    potential_reference_length: float | None = None

    @property
    def interior_indices(self) -> np.ndarray:
        return np.flatnonzero(self.interior_mask.ravel())

    def solve(self, density: np.ndarray, *, compute_potential: bool = True, compute_fields: bool = True) -> FieldResult:
        """Convolve every density slice with free-space potential and field kernels.

        ``density`` is C/m^2.  The returned potential is V m up to an additive
        constant; ``Ex`` and ``Ey`` are integrated transverse fields in V.
        ``compute_potential=False`` skips the potential inverse transform and
        returns ``potential=None``. ``compute_fields=False`` skips both field
        transforms and returns ``integrated_ex=integrated_ey=None``. At least
        one output must be requested. Default calls retain all three outputs.
        """
        if not compute_potential and not compute_fields:
            raise ValueError("request at least one of potential or fields")
        source = np.asarray(density, dtype=self.dtype)
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
            if self.potential_reference_length is not None:
                kernel[0, 0] = (np.log(np.sqrt(self.geometry.dx * self.geometry.dy)) - _cell_mean_log_radius(self.geometry)) / (2 * const.pi *
                                                                                                                                const.epsilon0)
                kernel += np.log(self.potential_reference_length / np.sqrt(self.geometry.dx * self.geometry.dy)) / (2 * const.pi * const.epsilon0)
            self._potential_kernel_fft = rfftn(kernel.astype(self.dtype))
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
        ex = convolve(self._ex_kernel_fft) if compute_fields else None
        ey = convolve(self._ey_kernel_fft) if compute_fields else None
        if squeeze:
            return FieldResult(*(None if a is None else a[0] for a in (potential, ex, ey)))
        return FieldResult(potential, ex, ey)


def _cell_mean_log_radius(geometry):
    """Exact rectangular-cell average of log(r) for the self interaction."""
    a, b = geometry.dx / 2, geometry.dy / 2
    return np.log(np.hypot(a, b)) - 1.5 + (a / b * np.arctan(b / a) + b / a * np.arctan(a / b)) / 2


def build_fft_free_space_resources(geometry, *, dtype="float64", potential_reference_length=None) -> FFTFreeSpaceSolver:
    """Build zero-padded free-space kernels for one uniform grid geometry."""
    dtype = np.dtype(dtype)
    if dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError("FFT precision must be float32 or float64")
    if potential_reference_length is not None and (not np.isfinite(potential_reference_length) or potential_reference_length <= 0):
        raise ValueError("potential_reference_length must be positive finite")
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
        rfftn(ex_kernel.astype(dtype)),
        rfftn(ey_kernel.astype(dtype)),
        dtype,
        potential_reference_length,
    )


def solve_poisson_fft_free_space(density: np.ndarray, geometry, *, compute_potential: bool = True, compute_fields: bool = True):
    """One-shot wrapper around :func:`build_fft_free_space_resources`."""
    return build_fft_free_space_resources(geometry).solve(density, compute_potential=compute_potential, compute_fields=compute_fields)


class GPUFFTFreeSpaceSolver(GPUFieldSolver):

    def __init__(self, geometry, dtype="float64", *, batch_size=16, potential_reference_length=None):
        import cupy as cp

        super().__init__(geometry, dtype)
        self._fft_batches = {}
        if (isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1):
            raise ValueError("FFT batch_size must be a positive integer")
        self.batch_size = int(batch_size)
        if potential_reference_length is not None and (not np.isfinite(potential_reference_length) or potential_reference_length <= 0):
            raise ValueError("potential_reference_length must be positive finite")
        self.potential_reference_length = potential_reference_length
        self.aperture_mask = np.ones((geometry.ny, geometry.nx), dtype=bool)
        self.interior_mask = self.aperture_mask.copy()
        # Small-prime padding preserves the exact zero-padded linear convolution.
        self.shape = (
            next_fast_len(2 * geometry.ny - 1, real=True),
            next_fast_len(2 * geometry.nx - 1, real=True),
        )
        py, px = self.shape
        iy, ix = np.arange(py), np.arange(px)
        y = np.where(iy < geometry.ny, iy, iy - py)[:, None] * geometry.dy
        x = np.where(ix < geometry.nx, ix, ix - px)[None, :] * geometry.dx
        r2 = x * x + y * y
        r2[0, 0] = 1
        factor = 1 / (2 * const.pi * const.epsilon0)
        kernels = [
            x / r2 * factor,
            y / r2 * factor,
            -np.log(np.sqrt(r2) / np.sqrt(geometry.dx * geometry.dy)) * factor,
        ]
        self.kernels = []
        for index, kernel in enumerate(kernels):
            kernel[0, 0] = 0
            if index == 2 and potential_reference_length is not None:
                kernel[0, 0] = (np.log(np.sqrt(geometry.dx * geometry.dy)) - _cell_mean_log_radius(geometry)) * factor
                kernel += np.log(potential_reference_length / np.sqrt(geometry.dx * geometry.dy)) * factor
            self.kernels.append(cp.fft.rfft2(cp.asarray(kernel, dtype=self.dtype)))

    def _prepare(self, n_slices):
        import cupy as cp
        from cupyx.scipy.fft import get_fft_plan

        workspace = self._work
        py, px = self.shape
        sizes = [min(n_slices, self.batch_size)]
        remainder = n_slices % self.batch_size
        if remainder and remainder != sizes[0]:
            sizes.append(remainder)
        # Shape changes replace the output workspace; retain at most two FFT
        # batches so adjacent one/two-plane source solves reuse their plans.
        for count in sizes:
            if count in self._fft_batches:
                self._fft_batches[count] = self._fft_batches.pop(count)
        missing = [count for count in sizes if count not in self._fft_batches]
        if len(self._fft_batches) + len(missing) > 2:
            # Plans own temporary device storage, which must outlive queued FFTs.
            self.stream.synchronize()
        while len(self._fft_batches) + len(missing) > 2:
            del self._fft_batches[next(iter(self._fft_batches))]
        workspace["fft_batches"] = {}
        for count in sizes:
            if count not in self._fft_batches:
                batch = {}
                batch["padded"] = cp.empty((count, py, px), self.dtype)
                batch["spectrum"] = cp.empty((count, py, px // 2 + 1), self.complex_dtype)
                batch["scratch"] = cp.empty_like(batch["spectrum"])
                batch["forward"] = get_fft_plan(batch["padded"], axes=(-2, -1), value_type="R2C")
                batch["inverse"] = get_fft_plan(batch["scratch"], shape=self.shape, axes=(-2, -1), value_type="C2R")
                self._fft_batches[count] = batch
            workspace["fft_batches"][count] = self._fft_batches[count]

    def close(self):
        import cupy as cp

        if self._closed:
            return
        with cp.cuda.Device(self.device), self.stream:
            super().close()
            self._fft_batches.clear()

    def solve(self, density, *, compute_potential=True, compute_fields=True, validate=True, copy=True):
        """Convolve requested outputs; omitted potential or fields return None.

        At least one of ``compute_potential`` and ``compute_fields`` must be
        true. Results own requested arrays unless ``copy=False`` borrows them.
        """
        import cupy as cp
        from cupy.cuda import cufft

        if not compute_potential and not compute_fields:
            raise ValueError("request at least one of potential or fields")
        src, squeeze = self._source(density, validate)
        grid, workspace = self.geometry, self._work
        py, px = self.shape
        shape_args = (np.int32(grid.nx), np.int32(grid.ny), np.int32(px), np.int32(py))
        for first in range(0, workspace["slices"], self.batch_size):
            count = min(self.batch_size, workspace["slices"] - first)
            batch = workspace["fft_batches"][count]
            padded = batch["padded"]
            launch_gpu_kernel(
                _FFT_CUDA,
                "fft_pad",
                padded.size,
                (
                    src[first:first + count],
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
                if target != "phi" and not compute_fields:
                    continue
                output_view = workspace[target][first:first + count]
                cp.multiply(batch["spectrum"], kernel, out=batch["scratch"])
                batch["inverse"].fft(batch["scratch"], padded, cufft.CUFFT_INVERSE)
                launch_gpu_kernel(
                    _FFT_CUDA,
                    "fft_crop",
                    output_view.size,
                    (
                        padded,
                        output_view,
                        *shape_args,
                        np.int64(output_view.size),
                        self.scalar(grid.dx * grid.dy / (px * py)),
                    ),
                    self.dtype,
                )
        return self._result(workspace["phi"] if compute_potential else None, squeeze, copy, compute_fields=compute_fields)


_FFT_CUDA = r"""
extern "C" __global__ void fft_pad(
    const T* src,
    T* dst,
    int nx,
    int ny,
    int px,
    int py,
    long long size
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    int x = i % px, y = (i / px) % py;
    dst[i] = (x < nx && y < ny) ? src[(i / (px * py)) * nx * ny + y * nx + x] : 0;
}
extern "C" __global__ void fft_crop(
    const T* src,
    T* dst,
    int nx,
    int ny,
    int px,
    int py,
    long long size,
    T scale
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        dst[i] = src[(i / (nx * ny)) * px * py + ((i / nx) % ny) * px + i % nx] * scale;
}
"""
