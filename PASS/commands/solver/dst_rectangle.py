"""Batched DST Poisson solver for a rectangular zero-Dirichlet chamber."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dstn, idstn

from PASS.utils.constants import const
from .field_result import FieldResult


@dataclass
class DSTRectangleSolver:
    """Direct sine-transform solver for ``-Laplacian(phi) = rho / epsilon0``.

    The transform acts only on the two transverse axes, retaining every leading
    slice axis.  Consequently all slice right-hand sides are solved together.
    """

    geometry: object
    aperture_mask: np.ndarray
    interior_mask: np.ndarray
    eigenvalues: np.ndarray

    @property
    def interior_indices(self) -> np.ndarray:
        return np.flatnonzero(self.interior_mask.ravel())

    def solve(self, density: np.ndarray) -> FieldResult:
        """Return potential (V m) and integrated fields (V) for all slices."""
        source = np.asarray(density, dtype=float)
        if not np.all(np.isfinite(source)):
            raise ValueError("density must be finite")
        squeeze = source.ndim == 2
        if squeeze:
            source = source[None, ...]
        if source.ndim != 3 or source.shape[1:] != self.aperture_mask.shape:
            raise ValueError("density must have shape (n_slice, ny, nx) matching the grid")

        potential = np.zeros_like(source)
        rhs = source[:, 1:-1, 1:-1] / const.epsilon0
        transformed = dstn(rhs, type=1, axes=(-2, -1), norm="ortho")
        potential[:, 1:-1, 1:-1] = idstn(
            transformed / self.eigenvalues[None, ...],
            type=1,
            axes=(-2, -1),
            norm="ortho",
        )
        dy_field, dx_field = np.gradient(
            potential,
            self.geometry.dy,
            self.geometry.dx,
            axis=(-2, -1),
            edge_order=1,
        )
        ex, ey = -dx_field, -dy_field
        if squeeze:
            return FieldResult(potential[0], ex[0], ey[0])
        return FieldResult(potential, ex, ey)


def build_dst_rectangle_resources(geometry) -> DSTRectangleSolver:
    """Precompute spectral eigenvalues for a full rectangular chamber."""
    x_mode = np.arange(1, geometry.nx - 1, dtype=float)
    y_mode = np.arange(1, geometry.ny - 1, dtype=float)
    lambda_x = 4.0 * np.sin(const.pi * x_mode / (2.0 * (geometry.nx - 1)))**2 / geometry.dx**2
    lambda_y = 4.0 * np.sin(const.pi * y_mode / (2.0 * (geometry.ny - 1)))**2 / geometry.dy**2
    aperture_mask = np.ones((geometry.ny, geometry.nx), dtype=bool)
    interior_mask = np.zeros_like(aperture_mask)
    interior_mask[1:-1, 1:-1] = True
    return DSTRectangleSolver(geometry, aperture_mask, interior_mask, lambda_y[:, None] + lambda_x[None, :])
