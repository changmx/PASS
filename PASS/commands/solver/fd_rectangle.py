"""CPU finite-difference Poisson solver for the transverse PIC pipeline.

The matrix is assembled and factorized once in :func:`build_fd_resources`.
``FDSolver.solve`` accepts a stack of slice densities and passes all right-hand
sides to the same sparse LU factorization.  This is deliberately a batched
operation: a space-charge call must not solve one linear system per slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import splu

from PASS.utils.constants import const
from .field_result import FieldResult


EPSILON_0 = const.epsilon0


@dataclass
class FDSolver:
    """Five-point Dirichlet finite-difference solver on a nodal grid."""

    geometry: object
    aperture_mask: np.ndarray
    interior_mask: np.ndarray
    matrix: csc_matrix
    _lu: object

    @property
    def interior_indices(self) -> np.ndarray:
        return np.flatnonzero(self.interior_mask.ravel())

    def solve(self, density: np.ndarray) -> FieldResult:
        """Solve all slice densities and return potential, Ex and Ey.

        ``density`` has shape ``(n_slice, ny, nx)`` (a two-dimensional array is
        accepted as a one-slice shorthand).  The source is in C/m^2, so the
        returned potential has the 2-D integrated-potential units V m and the
        returned fields have units V.  Ex/Ey use ``E = -grad(phi)``.
        """
        source = np.asarray(density, dtype=float)
        if not np.all(np.isfinite(source)):
            raise ValueError("density must be finite")
        squeeze = source.ndim == 2
        if squeeze:
            source = source[None, ...]
        if source.ndim != 3 or source.shape[1:] != self.aperture_mask.shape:
            raise ValueError(
                "density must have shape (n_slice, ny, nx) matching the grid"
            )

        n_slice, ny, nx = source.shape
        potential = np.zeros_like(source)
        interior = self.interior_indices
        if interior.size:
            rhs = (source.reshape(n_slice, ny * nx)[:, interior].T / EPSILON_0)
            # splu.solve supports a dense matrix RHS.  Keep the slice axis as
            # columns so all slices use one factorization and one solve call.
            values = self._lu.solve(rhs)
            potential.reshape(n_slice, ny * nx)[:, interior] = np.asarray(values).T

        # np.gradient returns derivatives in axis order (y, x).
        dy_field, dx_field = np.gradient(
            potential,
            self.geometry.dy,
            self.geometry.dx,
            axis=(-2, -1),
            edge_order=1,
        )
        ex = -dx_field
        ey = -dy_field
        ex *= self.aperture_mask[None, ...]
        ey *= self.aperture_mask[None, ...]
        if squeeze:
            return FieldResult(potential[0], ex[0], ey[0])
        return FieldResult(potential, ex, ey)


def _interior_mask(aperture_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(aperture_mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) < 3:
        raise ValueError("aperture_mask must be a two-dimensional grid of at least 3x3")
    interior = mask.copy()
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    interior[1:-1, 1:-1] &= (
        mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return interior


def build_fd_resources(geometry, aperture_mask=None):
    """Build reusable resources for a rectangular zero-Dirichlet domain.

    On an aligned rectangle, the Shortley-Weller distances are exactly ``dx``
    and ``dy`` and reduce to this regular five-point stencil. Arbitrary
    apertures must supply continuous geometry to
    :func:`fd_arbitrary.build_fd_arbitrary_resources`.
    """
    if aperture_mask is None:
        aperture_mask = np.ones((geometry.ny, geometry.nx), dtype=bool)
    aperture_mask = np.asarray(aperture_mask, dtype=bool)
    if aperture_mask.shape != (geometry.ny, geometry.nx):
        raise ValueError("aperture_mask shape must match geometry")
    if not np.all(aperture_mask):
        raise ValueError(
            "an arbitrary aperture needs continuous geometry; use "
            "build_fd_arbitrary_resources(..., aperture)"
        )

    interior = _interior_mask(aperture_mask)
    indices = np.flatnonzero(interior.ravel())
    index_map = np.full(interior.size, -1, dtype=np.int64)
    index_map[indices] = np.arange(indices.size)
    dx2 = 1.0 / geometry.dx**2
    dy2 = 1.0 / geometry.dy**2
    matrix = lil_matrix((indices.size, indices.size), dtype=float)
    for row, flat in enumerate(indices):
        iy, ix = divmod(int(flat), geometry.nx)
        matrix[row, row] = 2.0 * (dx2 + dy2)
        for neighbor, coefficient in (
            (flat - 1, -dx2), (flat + 1, -dx2),
            (flat - geometry.nx, -dy2), (flat + geometry.nx, -dy2),
        ):
            column = index_map[neighbor]
            if column >= 0:
                matrix[row, column] = coefficient
    sparse = csc_matrix(matrix)
    # splu does not accept a 0x0 matrix.  A None handle is equivalent to a
    # zero field for an aperture with no interior nodes.
    lu = splu(sparse) if indices.size else None
    return FDSolver(geometry, aperture_mask.copy(), interior, sparse, lu)


def build_fd_rectangle_resources(geometry):
    """Build resources for a full rectangular zero-Dirichlet domain."""
    return build_fd_resources(geometry, aperture_mask=None)


def solve_poisson_fd(
    density: np.ndarray,
    geometry=None,
    *,
    dx: float | None = None,
    dy: float | None = None,
    aperture_mask=None,
):
    """One-shot compatibility wrapper around the reusable :class:`FDSolver`.

    ``density`` may contain one or many slices.  Prefer
    :func:`build_fd_resources` plus ``FDSolver.solve`` for repeated calls so
    the sparse factorization is reused.
    """
    source = np.asarray(density)
    if geometry is None:
        if dx is None or dy is None or source.ndim not in (2, 3):
            raise TypeError("provide geometry, or dx and dy for a regular grid")
        ny, nx = source.shape[-2:]
        from .pic import GridGeometry

        geometry = GridGeometry(
            nx,
            ny,
            -float(dx) * (nx - 1) / 2,
            float(dx) * (nx - 1) / 2,
            -float(dy) * (ny - 1) / 2,
            float(dy) * (ny - 1) / 2,
        )
    return build_fd_resources(geometry, aperture_mask).solve(source)
