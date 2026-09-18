"""CPU/GPU finite-difference Poisson solver for the transverse PIC pipeline.

The matrix is assembled and factorized once in :func:`build_fd_resources`.
``FDRectangleSolver.solve`` accepts a stack of slice densities and passes all right-hand
sides to the same sparse LU factorization.  This is deliberately a batched
operation: a space-charge call must not solve one linear system per slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import splu

from PASS.utils.constants import const
from .field_result import FieldResult, GPUFieldSolver, launch_gpu_kernel


@dataclass
class FDRectangleSolver:
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
            raise ValueError("density must have shape (n_slice, ny, nx) matching the grid")

        n_slice, ny, nx = source.shape
        potential = np.zeros_like(source)
        interior = self.interior_indices
        if interior.size:
            rhs = (source.reshape(n_slice, ny * nx)[:, interior].T / const.epsilon0)
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
    interior[1:-1, 1:-1] &= (mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:])
    return interior


def build_fd_resources(geometry, aperture_mask=None, *, factorize=True):
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
        raise ValueError("an arbitrary aperture needs continuous geometry; use "
                         "build_fd_arbitrary_resources(..., aperture)")

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
            (flat - 1, -dx2),
            (flat + 1, -dx2),
            (flat - geometry.nx, -dy2),
            (flat + geometry.nx, -dy2),
        ):
            column = index_map[neighbor]
            if column >= 0:
                matrix[row, column] = coefficient
    sparse = csc_matrix(matrix)
    # splu does not accept a 0x0 matrix.  A None handle is equivalent to a
    # zero field for an aperture with no interior nodes.
    lu = splu(sparse) if indices.size and factorize else None
    return FDRectangleSolver(geometry, aperture_mask.copy(), interior, sparse, lu)


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
    """Solve once using the reusable :class:`FDRectangleSolver`.

    ``density`` may contain one or many slices.  Prefer
    :func:`build_fd_resources` plus ``FDRectangleSolver.solve`` for repeated calls so
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


class GPUFDRectangleSolver(GPUFieldSolver):
    """cuDSS factorization and batched solves for a full rectangular chamber."""

    arbitrary = False

    def __init__(self, geometry, dtype="float64"):
        self._initialize(geometry, dtype, build_fd_resources(geometry, factorize=False))

    def _initialize(self, geometry, dtype, reference):
        """Own the cuDSS lifecycle for either rectangular or Shortley-Weller FD."""
        import cupy as cp

        super().__init__(geometry, dtype)
        from nvmath.bindings import cudss

        self.api = cudss_api = cudss
        self.handle = self.config = self.data = self.matrix_handle = None
        self._dense_handles = []
        self._factored = False
        self.aperture_mask, self.interior_mask = (
            reference.aperture_mask,
            reference.interior_mask,
        )
        indices = reference.interior_indices.astype(np.int32)
        self.indices = cp.asarray(indices)
        self.n = indices.size
        self.matrix = reference.matrix.tocsr().astype(self.dtype)
        self.matrix.sort_indices()
        self._prepare_geometry(reference)
        if not self.n:
            return
        self.row = cp.asarray(self.matrix.indptr, dtype=cp.int32)
        self.col = cp.asarray(self.matrix.indices, dtype=cp.int32)
        self.values = cp.asarray(self.matrix.data)
        self.value_type = (0 if self.dtype.itemsize == 4 else 1)  # CUDA_R_32F / CUDA_R_64F
        try:
            self.handle = cudss_api.create()
            cudss_api.set_stream(self.handle, self.stream.ptr)
            self.config = cudss_api.config_create()
            self.data = cudss_api.data_create(self.handle)
            self.matrix_handle = cudss_api.matrix_create_csr(
                self.n,
                self.n,
                self.matrix.nnz,
                self.row.data.ptr,
                0,
                self.col.data.ptr,
                self.values.data.ptr,
                10,  # CUDA_R_32I row offsets (cuDSS 0.8 has separate types)
                10,  # CUDA_R_32I column indices
                self.value_type,
                cudss_api.MatrixType.GENERAL if self.arbitrary else cudss_api.MatrixType.SPD,
                cudss_api.MatrixViewType.FULL,
                cudss_api.IndexBase.ZERO,
            )
            self.prepare(1)
        except Exception:
            self.close()
            raise

    def _prepare_geometry(self, reference):
        self.coefficients = None

    def _prepare(self, n_slices):
        import cupy as cp

        if not self.n:
            return
        cudss_api, workspace = self.api, self._work
        self.stream.synchronize()  # Batch resize only; never part of a stable solve.
        for handle in self._dense_handles:
            cudss_api.matrix_destroy(handle)
        self._dense_handles = []
        workspace["rhs"] = cp.empty((n_slices, self.n), self.dtype)
        workspace["solution"] = cp.empty_like(workspace["rhs"])
        for a in (workspace["solution"], workspace["rhs"]):
            self._dense_handles.append(cudss_api.matrix_create_dn(self.n, n_slices, self.n, a.data.ptr, self.value_type, cudss_api.Layout.COL_MAJOR))
        if not self._factored:
            for phase in (cudss_api.Phase.ANALYSIS, cudss_api.Phase.FACTORIZATION):
                cudss_api.execute(
                    self.handle,
                    phase,
                    self.config,
                    self.data,
                    self.matrix_handle,
                    *self._dense_handles,
                )
            self.stream.synchronize()
            info = np.zeros(1, dtype=np.int32)
            written = np.zeros(1, dtype=np.uintp)
            cudss_api.data_get(
                self.handle,
                self.data,
                cudss_api.DataParam.INFO,
                info.ctypes.data,
                info.nbytes,
                written.ctypes.data,
            )
            if info[0]:
                raise RuntimeError(f"cuDSS factorization failed: info={info[0]}")
            self._factored = True

    def solve(self, density, *, compute_potential=True, validate=True, copy=True):
        src, squeeze = self._source(density, validate)
        grid, workspace = self.geometry, self._work
        if self.n:
            size = workspace["slices"] * self.n
            args = (np.int32(self.n), np.int32(grid.nx * grid.ny), np.int64(size))
            launch_gpu_kernel(
                _FD_CUDA,
                "fd_rhs",
                size,
                (src, workspace["rhs"], self.indices, *args, self.scalar(1 / const.epsilon0)),
                self.dtype,
            )
            cudss_api = self.api
            cudss_api.execute(
                self.handle,
                cudss_api.Phase.SOLVE,
                self.config,
                self.data,
                self.matrix_handle,
                *self._dense_handles,
            )
            launch_gpu_kernel(
                _FD_CUDA,
                "fd_scatter",
                size,
                (workspace["solution"], workspace["phi"], self.indices, *args),
                self.dtype,
            )
        self._gradient()
        return self._result(workspace["phi"], squeeze, copy)

    def close(self):
        import cupy as cp

        if self.handle is None:
            super().close()
            return
        with cp.cuda.Device(self.device):
            self.stream.synchronize()
            for handle in self._dense_handles:
                self.api.matrix_destroy(handle)
            self._dense_handles = []
            if self.matrix_handle is not None:
                self.api.matrix_destroy(self.matrix_handle)
                self.matrix_handle = None
            if self.data is not None:
                self.api.data_destroy(self.handle, self.data)
                self.data = None
            if self.config is not None:
                self.api.config_destroy(self.config)
                self.config = None
            self.api.destroy(self.handle)
            self.handle = None
        super().close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Interpreter teardown; explicit close propagates errors.


_FD_CUDA = r"""
extern "C" __global__ void fd_rhs(
    const T* rho,
    T* rhs,
    const int* indices,
    int n,
    int grid,
    long long size,
    T scale
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        rhs[i] = rho[(i / n) * grid + indices[i % n]] * scale;
}
extern "C" __global__ void fd_scatter(
    const T* values,
    T* phi,
    const int* indices,
    int n,
    int grid,
    long long size
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        phi[(i / n) * grid + indices[i % n]] = values[i];
}
"""
