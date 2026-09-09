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
from .field_result import FieldResult


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
