"""Finite-difference Poisson solvers for continuous arbitrary apertures.

The Shortley-Weller scheme replaces the regular mesh step next to a conductor
with the actual distance to the intersection of that grid line and the
aperture. Geometry-dependent masks, distances, matrix coefficients and
gradient coefficients are built once here; solving a density stack only passes
all slice right-hand sides through the cached sparse factorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import splu

from PASS.utils.constants import const
from .fd_rectangle import GPUFDSolver
from .field_result import FieldResult, _launch_gpu


from PASS.utils.aperture import (
    RectangleAperture, EllipticAperture, AllSpaceAperture, IntersectionAperture, RacetrackAperture, OctagonAperture, PolygonAperture, build_aperture
)


def build_aperture_mask(geometry, aperture: Mapping[str, Any] | None = None) -> np.ndarray:
    """Return nodal membership for the supplied continuous aperture."""
    if aperture is None:
        return np.ones((geometry.ny, geometry.nx), dtype=bool)
    xx, yy = np.meshgrid(geometry.x, geometry.y)
    return build_aperture(aperture).mask(xx, yy)


def _axis_exit_distance(
    aperture,
    x: np.ndarray,
    y: np.ndarray,
    direction_x: float,
    direction_y: float,
    step: float,
) -> np.ndarray:
    """Locate the first inside-to-outside crossing within one grid step."""
    low = np.zeros(np.broadcast(x, y).shape, dtype=float)
    high = np.full(low.shape, step, dtype=float)
    for _ in range(52):
        middle = 0.5 * (low + high)
        inside = aperture.mask(
            x + direction_x * middle,
            y + direction_y * middle,
        )
        low = np.where(inside, middle, low)
        high = np.where(inside, high, middle)
    return high


@dataclass
class ArbitraryFDSolver:
    """Cached distance-weighted Shortley-Weller solver for one aperture."""

    geometry: object
    aperture_mask: np.ndarray
    interior_mask: np.ndarray
    matrix: csc_matrix
    _lu: object
    h_left: np.ndarray
    h_right: np.ndarray
    h_bottom: np.ndarray
    h_top: np.ndarray

    @property
    def interior_indices(self) -> np.ndarray:
        return np.flatnonzero(self.interior_mask.ravel())

    def solve(self, density: np.ndarray) -> FieldResult:
        """Solve all density slices and return potential (V m) and fields (V)."""
        source = np.asarray(density, dtype=float)
        if not np.all(np.isfinite(source)):
            raise ValueError("density must be finite")
        squeeze = source.ndim == 2
        if squeeze:
            source = source[None, ...]
        if source.ndim != 3 or source.shape[1:] != self.aperture_mask.shape:
            raise ValueError("density must have shape (n_slice, ny, nx) matching the grid")

        n_slice, ny, nx = source.shape
        potential = np.zeros_like(source)
        interior = self.interior_indices
        if interior.size:
            rhs = source.reshape(n_slice, ny * nx)[:, interior].T / const.epsilon0
            values = self._lu.solve(rhs)
            potential.reshape(n_slice, ny * nx)[:, interior] = np.asarray(values).T

        ex = np.zeros_like(potential)
        ey = np.zeros_like(potential)
        if interior.size:
            iy, ix = np.divmod(interior, nx)
            left = potential[:, iy, ix - 1]
            center = potential[:, iy, ix]
            right = potential[:, iy, ix + 1]
            bottom = potential[:, iy - 1, ix]
            top = potential[:, iy + 1, ix]
            h_left = self.h_left[iy, ix][None, :]
            h_right = self.h_right[iy, ix][None, :]
            h_bottom = self.h_bottom[iy, ix][None, :]
            h_top = self.h_top[iy, ix][None, :]
            dphi_dx = (-h_right / (h_left * (h_left + h_right)) * left + (h_right - h_left) / (h_left * h_right) * center + h_left /
                       (h_right * (h_left + h_right)) * right)
            dphi_dy = (-h_top / (h_bottom * (h_bottom + h_top)) * bottom + (h_top - h_bottom) / (h_bottom * h_top) * center + h_bottom /
                       (h_top * (h_bottom + h_top)) * top)
            ex[:, iy, ix] = -dphi_dx
            ey[:, iy, ix] = -dphi_dy
        if squeeze:
            return FieldResult(potential[0], ex[0], ey[0])
        return FieldResult(potential, ex, ey)


def build_fd_arbitrary_resources(
    geometry,
    aperture: Mapping[str, Any],
    *,
    factorize=True,
):
    """Build cached FD resources for a continuous arbitrary aperture.

    The Shortley-Weller solver requires a mapping such as
    ``{'Type': 'elliptic', 'A': 0.04, 'B': 0.02}``, because node membership
    alone cannot determine a boundary intersection distance.
    """
    spec = build_aperture(aperture)
    xx, yy = np.meshgrid(geometry.x, geometry.y)
    mask = spec.mask(xx, yy)
    interior = spec.strict_mask(xx, yy)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    indices = np.flatnonzero(interior.ravel())
    index_map = np.full(interior.size, -1, dtype=np.int64)
    index_map[indices] = np.arange(indices.size)

    h_left = np.full(mask.shape, geometry.dx, dtype=float)
    h_right = np.full(mask.shape, geometry.dx, dtype=float)
    h_bottom = np.full(mask.shape, geometry.dy, dtype=float)
    h_top = np.full(mask.shape, geometry.dy, dtype=float)
    if indices.size:
        iy, ix = np.divmod(indices, geometry.nx)
        neighbor_masks = (mask[iy, ix - 1], mask[iy, ix + 1], mask[iy - 1, ix], mask[iy + 1, ix])
        directions = ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))
        for h, neighbor, step, direction in zip(
            (h_left, h_right, h_bottom, h_top),
            neighbor_masks,
            (geometry.dx, geometry.dx, geometry.dy, geometry.dy),
            directions,
        ):
            values = np.full(indices.shape, step, dtype=float)
            crossing = ~neighbor
            if np.any(crossing):
                values[crossing] = _axis_exit_distance(
                    spec,
                    xx[iy[crossing], ix[crossing]],
                    yy[iy[crossing], ix[crossing]],
                    direction[0],
                    direction[1],
                    step,
                )
            if np.any(values <= 0):
                raise ValueError("aperture boundary passes through an active FD node")
            h[iy, ix] = values

    matrix = lil_matrix((indices.size, indices.size), dtype=float)
    for row, flat in enumerate(indices):
        iy, ix = divmod(int(flat), geometry.nx)
        hl, hr = h_left[iy, ix], h_right[iy, ix]
        hb, ht = h_bottom[iy, ix], h_top[iy, ix]
        coefficients = (
            (flat - 1, -2.0 / (hl * (hl + hr))),
            (flat + 1, -2.0 / (hr * (hl + hr))),
            (flat - geometry.nx, -2.0 / (hb * (hb + ht))),
            (flat + geometry.nx, -2.0 / (ht * (hb + ht))),
        )
        matrix[row, row] = -(sum(value for _, value in coefficients))
        for neighbor, coefficient in coefficients:
            column = index_map[neighbor]
            if column >= 0:
                matrix[row, column] = coefficient
    sparse = csc_matrix(matrix)
    lu = splu(sparse) if indices.size and factorize else None
    return ArbitraryFDSolver(geometry, mask, interior, sparse, lu, h_left, h_right, h_bottom, h_top)


def solve_poisson_fd_arbitrary(density, geometry, aperture):
    """One-shot wrapper around :func:`build_fd_arbitrary_resources`."""
    return build_fd_arbitrary_resources(geometry, aperture).solve(density)


class GPUArbitraryFDSolver(GPUFDSolver):
    """cuDSS solves with continuous-aperture Shortley-Weller coefficients."""

    arbitrary = True

    def __init__(self, geometry, aperture, dtype="float64"):
        self._initialize(
            geometry, dtype, build_fd_arbitrary_resources(geometry, aperture, factorize=False)
        )

    def _prepare_geometry(self, reference):
        import cupy as cp

        indices = reference.interior_indices.astype(np.int32)
        hl, hr, hb, ht = (
            a.ravel()[indices]
            for a in (
                reference.h_left,
                reference.h_right,
                reference.h_bottom,
                reference.h_top,
            )
        )
        self.coefficients = cp.asarray(
            np.stack(
                (
                    -hr / (hl * (hl + hr)),
                    (hr - hl) / (hl * hr),
                    hl / (hr * (hl + hr)),
                    -ht / (hb * (hb + ht)),
                    (ht - hb) / (hb * ht),
                    hb / (ht * (hb + ht)),
                )
            ),
            dtype=self.dtype,
        )

    def _gradient(self):
        g, w = self.geometry, self._work
        w["ex"].fill(0)
        w["ey"].fill(0)
        _launch_gpu(
            _SW_GRADIENT_CUDA,
            "sw_gradient",
            w["slices"] * self.n,
            (
                w["phi"],
                w["ex"],
                w["ey"],
                self.indices,
                self.coefficients,
                np.int32(self.n),
                np.int32(g.nx),
                np.int32(g.nx * g.ny),
                np.int64(w["slices"] * self.n),
            ),
            self.dtype,
        )

_SW_GRADIENT_CUDA = r"""
extern "C" __global__ void sw_gradient(const T* phi, T* ex, T* ey,
    const int* indices, const T* coeff, int n, int nx, int grid, long long size) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    int k = i%n;
    long long j = (i/n)*grid + indices[k];
    ex[j] = -(coeff[k]*phi[j-1] + coeff[n+k]*phi[j] + coeff[2*n+k]*phi[j+1]);
    ey[j] = -(coeff[3*n+k]*phi[j-nx] + coeff[4*n+k]*phi[j] + coeff[5*n+k]*phi[j+nx]);
}
"""
