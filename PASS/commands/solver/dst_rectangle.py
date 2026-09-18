"""Batched DST Poisson solver for a rectangular zero-Dirichlet chamber."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import warnings

import numpy as np
from scipy.fft import dstn, idstn

from PASS.utils.constants import const
from .field_result import FieldResult, GPUFieldSolver, _gpu_module, launch_gpu_kernel


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


_DST_CHOICES = {}


class GPUDSTRectangleSolver(GPUFieldSolver):

    def __init__(self, geometry, dtype="float64", *, implementation="auto"):
        import cupy as cp

        super().__init__(geometry, dtype)
        if implementation not in ("auto", "cufft", "fused", "cufftdx"):
            raise ValueError("DST implementation must be 'auto', 'cufft', 'fused', or 'cufftdx'")
        self.requested_implementation = implementation
        self.implementation = "cufft" if implementation == "auto" else implementation
        self.tuning = None
        lengths = (2 * (geometry.nx - 1), 2 * (geometry.ny - 1))
        self._fused_supported = all(8 <= n <= 2048 and not n & (n - 1) for n in lengths)
        if implementation not in ("auto", "cufft") and not self._fused_supported:
            raise ValueError("fused DST requires power-of-two extensions from 8 to 2048")
        self.twiddles = {}
        if implementation == "fused":
            for n in set(lengths):
                angle = -2 * np.pi * np.arange(n // 2) / n
                self.twiddles[n] = cp.asarray(np.column_stack((np.cos(angle), np.sin(angle))), dtype=self.dtype)
        self.dx_kernels = {}
        if implementation == "cufftdx":
            self.dx_kernels = {n: _cufftdx_kernel(n, self.dtype.str, self.device) for n in set(lengths)}
        reference = build_dst_rectangle_resources(geometry)
        self.aperture_mask = reference.aperture_mask
        self.interior_mask = reference.interior_mask
        # Includes epsilon0 so the division is fused into the second transform.
        self.inv_lambda = cp.asarray(1 / (const.epsilon0 * reference.eigenvalues), dtype=self.dtype)

    def prepare(self, num_slices):
        self._check_context()
        if (isinstance(num_slices, bool) or int(num_slices) != num_slices or num_slices < 1):
            raise ValueError("num_slices must be a positive integer")
        n_slices = int(num_slices)
        if (self.requested_implementation == "auto" and self._fused_supported and (self._work is None or self._work["slices"] != n_slices)):
            self._select_implementation(n_slices)
        super().prepare(n_slices)

    def _select_implementation(self, n_slices):
        # Only initialization / batch changes tune. A stable tracking call never
        # times kernels or synchronizes here. Decisions are local to this process.
        import cupy as cp

        grid = self.geometry
        key = (self.device, self.dtype.str, grid.nx, grid.ny, n_slices)
        if key not in _DST_CHOICES:
            reference = GPUDSTRectangleSolver(grid, self.dtype, implementation="cufft")
            candidate = None
            try:
                candidate = GPUDSTRectangleSolver(grid, self.dtype, implementation="cufftdx")
            except (
                    ImportError,
                    RuntimeError,
                    cp.cuda.compiler.CompileException,
            ) as exc:
                warnings.warn(f"cuFFTDx unavailable; using cuFFT DST: {exc}", RuntimeWarning)
            try:
                scores = {}
                if candidate is not None:
                    iy, ix = cp.indices((grid.ny, grid.nx), dtype=self.dtype)
                    rho = cp.broadcast_to(
                        cp.exp(-((ix / (grid.nx - 1) - 0.43)**2 + (iy / (grid.ny - 1) - 0.54)**2) * 30),
                        (n_slices, grid.ny, grid.nx),
                    ).copy()
                    expected = reference.solve(rho, validate=False)
                    actual = candidate.solve(rho, validate=False)
                    for a, b in ((actual.integrated_ex, expected.integrated_ex), (actual.integrated_ey, expected.integrated_ey)):
                        error = float(cp.linalg.norm(a - b) / cp.linalg.norm(b))
                        if not np.isfinite(error) or error > (2e-4 if self.dtype.itemsize == 4 else 5e-10):
                            raise RuntimeError(f"cuFFTDx DST validation failed: relative L2={error}")
                    for solver in (reference, candidate):
                        for _ in range(3):
                            solver.solve(rho, validate=False, copy=False)
                    self.stream.synchronize()
                    # Interleave candidates so clock/thermal changes do not
                    # systematically favor the candidate measured last.
                    samples = {"cufft": [], "cufftdx": []}
                    for index in range(6):
                        pair = ((reference, candidate) if index % 2 == 0 else (candidate, reference))
                        for solver in pair:
                            start, end = cp.cuda.Event(), cp.cuda.Event()
                            start.record()
                            solver.solve(rho, validate=False, copy=False)
                            end.record()
                            end.synchronize()
                            samples[solver.implementation].append(cp.cuda.get_elapsed_time(start, end))
                    scores = {name: float(np.median(times)) for name, times in samples.items()}
                choice = ("cufftdx" if scores and scores["cufftdx"] < 0.95 * scores["cufft"] else "cufft")
                _DST_CHOICES[key] = {"implementation": choice, "device_ms": scores}
            finally:
                reference.close()
                if candidate is not None:
                    candidate.close()
        self.tuning = _DST_CHOICES[key].copy()
        self.implementation = self.tuning["implementation"]
        if self.implementation == "cufftdx" and not self.dx_kernels:
            self.dx_kernels = {n: _cufftdx_kernel(n, self.dtype.str, self.device) for n in {2 * (grid.nx - 1), 2 * (grid.ny - 1)}}

    def _prepare(self, n_slices):
        import cupy as cp
        from cupy.cuda import cufft

        grid, workspace = self.geometry, self._work
        workspace["a"] = cp.empty((n_slices, grid.ny - 2, grid.nx - 2), self.dtype)
        workspace["b"] = cp.empty((n_slices, grid.nx - 2, grid.ny - 2), self.dtype)
        if self.implementation != "cufft":
            return
        fft_type = cufft.CUFFT_R2C if self.dtype.itemsize == 4 else cufft.CUFFT_D2Z
        workspace["transforms"] = []
        for rows, n in ((grid.ny - 2, grid.nx - 2), (grid.nx - 2, grid.ny - 2)):
            workspace["transforms"].append((
                cp.empty((n_slices * rows, 2 * (n + 1)), self.dtype),
                cp.empty((n_slices * rows, n + 2), self.complex_dtype),
                cufft.Plan1d(2 * (n + 1), fft_type, n_slices * rows),
            ))

    def solve(self, density, *, compute_potential=True, validate=True, copy=True):
        from cupy.cuda import cufft

        src, squeeze = self._source(density, validate)
        grid, workspace = self.geometry, self._work
        if self.implementation != "cufft":
            for mode, rows, n, dst in (
                (0, grid.ny - 2, grid.nx - 2, workspace["b"]),
                (1, grid.nx - 2, grid.ny - 2, workspace["a"]),
                (2, grid.ny - 2, grid.nx - 2, workspace["phi"]),
            ):
                length = 2 * (n + 1)
                if self.implementation == "cufftdx":
                    kernel, threads, shared = self.dx_kernels[length]
                    kernel(
                        (workspace["slices"] * rows, ),
                        (threads, ),
                        (src, dst, self.inv_lambda, np.int32(rows), np.int32(mode)),
                        shared_mem=shared,
                    )
                else:
                    _gpu_module(_DST_CUDA, self.dtype.str, self.device).get_function(f"dst_fused_{length}")(
                        (workspace["slices"] * rows, ),
                        (256, ),
                        (
                            src,
                            dst,
                            self.inv_lambda,
                            self.twiddles[length],
                            np.int32(rows),
                            np.int32(mode),
                        ),
                        shared_mem=2 * length * self.dtype.itemsize,
                    )
                src = dst
            self._gradient()
            return self._result(workspace["phi"], squeeze, copy)
        for step in range(4):
            axis = step % 2
            rows, n = (grid.ny - 2, grid.nx - 2) if axis == 0 else (grid.nx - 2, grid.ny - 2)
            ext, spectrum, plan = workspace["transforms"][axis]
            launch_gpu_kernel(
                _DST_CUDA,
                "dst_pack",
                ext.size,
                (
                    src,
                    ext,
                    np.int32(n),
                    np.int32(rows),
                    np.int32(step == 0),
                    np.int64(ext.size),
                ),
                self.dtype,
            )
            plan.fft(ext, spectrum, cufft.CUFFT_FORWARD)
            dst = workspace["phi"] if step == 3 else workspace["b"] if axis == 0 else workspace["a"]
            size = workspace["slices"] * rows * n
            launch_gpu_kernel(
                _DST_CUDA,
                "dst_extract",
                size,
                (
                    spectrum,
                    dst,
                    self.inv_lambda,
                    np.int32(n),
                    np.int32(rows),
                    np.int32(step == 3),
                    np.int32(step == 1),
                    np.int64(size),
                    self.scalar(1 / np.sqrt(2 * (n + 1))),
                ),
                self.dtype,
            )
            src = dst
        self._gradient()
        return self._result(workspace["phi"], squeeze, copy)


@lru_cache(maxsize=None)
def _cufftdx_kernel(length, dtype, device):
    import cupy as cp

    from importlib.metadata import distribution, PackageNotFoundError
    from pathlib import Path

    try:
        package = distribution("nvidia-mathdx")
    except PackageNotFoundError as exc:
        raise RuntimeError("cuFFTDx requires nvidia-mathdx; install the PASS [cuda] extra") from exc
    include = next(
        (Path(package.locate_file(p)).parent for p in package.files if str(p).replace("\\", "/").endswith("include/cufftdx.hpp")),
        None,
    )
    if include is None:
        raise RuntimeError("nvidia-mathdx installation does not contain cufftdx.hpp")
    properties = cp.cuda.runtime.getDeviceProperties(device)
    arch = properties["major"] * 100 + properties["minor"] * 10
    scalar = "float" if np.dtype(dtype).itemsize == 4 else "double"
    source = f"#define REAL {scalar}\n#define LENGTH {length}\n#define ARCH {arch}\n"
    source += _CUFFTDX_CUDA
    with cp.cuda.Device(device):
        mod = cp.RawModule(code=source, options=("--std=c++17", f"-I{include}"))
        info = cp.empty(2, cp.int32)
        mod.get_function("traits")((1, ), (1, ), (info, ))
        threads, shared = map(int, info.get())
        kernel = mod.get_function("dst_dx")
        if shared > 48 * 1024:
            kernel.max_dynamic_shared_size_bytes = shared
        return kernel, threads, shared


_DST_CUDA = r"""
// Every row is an odd extension of its n interior nodes, length 2(n+1).
extern "C" __global__ void dst_pack(
    const T* src,
    T* ext,
    int n,
    int rows,
    int full_grid,
    long long size
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    int length = 2 * (n + 1), k = i % length;
    long long row = i / length;
    if (k == 0 || k == n + 1) {
        ext[i] = 0;
        return;
    }
    int col = (k <= n ? k : length - k) - 1;
    long long j = full_grid ? (row / rows) * (rows + 2) * (n + 2) + (row % rows + 1) * (n + 2) + col + 1 : row * n + col;
    ext[i] = k <= n ? src[j] : -src[j];
}
// Spectrum is interleaved real/imaginary. Transpose while extracting sine modes.
extern "C" __global__ void dst_extract(
    const T* spectrum,
    T* dst,
    const T* inv_lambda,
    int n,
    int rows,
    int full_grid,
    int divide,
    long long size,
    T norm
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    int col = i % n, row = (i / n) % rows;
    long long batch = i / (n * rows), out = (batch * n + col) * rows + row;
    T value = -spectrum[2 * ((i / n) * (n + 2) + col + 1) + 1] * norm;
    if (divide)
        value *= inv_lambda[col * rows + row];
    if (full_grid)
        out = batch * (n + 2) * (rows + 2) + (col + 1) * (rows + 2) + row + 1;
    dst[out] = value;
}

template <int LENGTH>
__device__ void block_fft(
    T* re,
    T* im,
    const T* twiddle
) {
    for (int span = 2; span <= LENGTH; span *= 2) {
        for (int k = threadIdx.x; k < LENGTH / 2; k += blockDim.x) {
            int j = k % (span / 2), a = (k / (span / 2)) * span + j, b = a + span / 2;
            int w = j * (LENGTH / span);
            T wr = twiddle[2 * w], wi = twiddle[2 * w + 1];
            T br = re[b] * wr - im[b] * wi, bi = re[b] * wi + im[b] * wr, ar = re[a], ai = im[a];
            re[a] = ar + br;
            im[a] = ai + bi;
            re[b] = ar - br;
            im[b] = ai - bi;
        }
        __syncthreads();
    }
}
template <int LENGTH>
__device__ void fused_dst(
    const T* src,
    T* dst,
    const T* inv_lambda,
    const T* twiddle,
    int rows,
    int mode,
    T* memory
) {
    constexpr int n = LENGTH / 2 - 1;
    constexpr int log_length = LENGTH == 8      ? 3
                               : LENGTH == 16   ? 4
                               : LENGTH == 32   ? 5
                               : LENGTH == 64   ? 6
                               : LENGTH == 128  ? 7
                               : LENGTH == 256  ? 8
                               : LENGTH == 512  ? 9
                               : LENGTH == 1024 ? 10
                                                : 11;
    T* re = memory;
    T* im = memory + LENGTH;
    int row = blockIdx.x % rows, batch = blockIdx.x / rows;
    for (int k = threadIdx.x; k < LENGTH; k += blockDim.x) {
        T value = 0;
        if (k != 0 && k != n + 1) {
            int col = (k <= n ? k : LENGTH - k) - 1;
            long long index = mode == 0 ? (long long)batch * (rows + 2) * (n + 2) + (row + 1) * (n + 2) + col + 1 : (long long)blockIdx.x * n + col;
            value = (k <= n ? 1 : -1) * src[index];
        }
        int rev = __brev((unsigned)k) >> (32 - log_length);
        re[rev] = value;
        im[rev] = 0;
    }
    __syncthreads();
    block_fft<LENGTH>(re, im, twiddle);
    T norm = (T)(1.0 / sqrt((double)LENGTH));
    if (mode == 1) {
        // Keep the middle DST, eigenvalue scaling, and inverse DST in shared
        // memory. This removes two global intermediate stacks and launches.
        T next[(LENGTH + 255) / 256];
        for (int k = threadIdx.x; k < LENGTH; k += blockDim.x) {
            T value = 0;
            if (k != 0 && k != n + 1) {
                int col = (k <= n ? k : LENGTH - k) - 1;
                value = -im[col + 1] * norm * inv_lambda[col * rows + row] * (k <= n ? 1 : -1);
            }
            next[k / 256] = value;
        }
        __syncthreads();
        for (int k = threadIdx.x; k < LENGTH; k += blockDim.x) {
            int rev = __brev((unsigned)k) >> (32 - log_length);
            re[rev] = next[k / 256];
            im[rev] = 0;
        }
        __syncthreads();
        block_fft<LENGTH>(re, im, twiddle);
    }
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        long long index =
            mode == 2 ? (long long)batch * (rows + 2) * (n + 2) + (row + 1) * (n + 2) + col + 1 : ((long long)batch * n + col) * rows + row;
        dst[index] = -im[col + 1] * norm;
    }
}
#define FUSED_DST_WRAPPER(LENGTH) \
    extern "C" __global__ void dst_fused_##LENGTH( \
        const T* src, \
        T* dst, \
        const T* inverse, \
        const T* twiddle, \
        int rows, \
        int mode \
    ) { \
        extern __shared__ double storage[]; \
        fused_dst<LENGTH>(src, dst, inverse, twiddle, rows, mode, (T*)storage); \
    }
FUSED_DST_WRAPPER(8)
FUSED_DST_WRAPPER(16)
FUSED_DST_WRAPPER(32)
FUSED_DST_WRAPPER(64)
FUSED_DST_WRAPPER(128)
FUSED_DST_WRAPPER(256)
FUSED_DST_WRAPPER(512)
FUSED_DST_WRAPPER(1024)
FUSED_DST_WRAPPER(2048)
"""

_CUFFTDX_CUDA = r"""
#define CUFFTDX_DISABLE_CUTLASS_DEPENDENCY
#include <cufftdx.hpp>
using T = REAL;
using FFT = decltype(cufftdx::Size<LENGTH>() + cufftdx::Precision<T>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                     cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::ElementsPerThread<8>() + cufftdx::FFTsPerBlock<1>() +
                     cufftdx::SM<ARCH>() + cufftdx::Block());
static_assert(
    !FFT::requires_workspace,
    "This DST path requires a workspace-free block FFT"
);
using C = FFT::value_type;
extern "C" __global__ void traits(
    int* out
) {
    out[0] = FFT::block_dim.x;
    out[1] = FFT::shared_memory_size > sizeof(C) * LENGTH ? FFT::shared_memory_size : sizeof(C) * LENGTH;
}
extern "C" __global__ void dst_dx(
    const T* src,
    T* dst,
    const T* inv_lambda,
    int rows,
    int mode
) {
    extern __shared__ __align__(16) C buffer[];
    constexpr int n = LENGTH / 2 - 1;
    int row = blockIdx.x % rows, batch = blockIdx.x / rows;
    T norm = (T)(1.0 / sqrt((double)LENGTH));
    for (int k = threadIdx.x; k < LENGTH; k += blockDim.x) {
        T value = 0;
        if (k != 0 && k != n + 1) {
            int col = (k <= n ? k : LENGTH - k) - 1;
            long long j = mode == 0 ? (long long)batch * (rows + 2) * (n + 2) + (row + 1) * (n + 2) + col + 1 : (long long)blockIdx.x * n + col;
            value = (k <= n ? 1 : -1) * src[j];
        }
        buffer[k].x = value;
        buffer[k].y = 0;
    }
    __syncthreads();
    FFT().execute(buffer);
    __syncthreads();
    if (mode == 1) {
        T next[FFT::elements_per_thread];
        for (int k = threadIdx.x, slot = 0; k < LENGTH; k += blockDim.x, ++slot) {
            T value = 0;
            if (k != 0 && k != n + 1) {
                int col = (k <= n ? k : LENGTH - k) - 1;
                value = -buffer[col + 1].y * norm * inv_lambda[col * rows + row] * (k <= n ? 1 : -1);
            }
            next[slot] = value;
        }
        __syncthreads();
        for (int k = threadIdx.x, slot = 0; k < LENGTH; k += blockDim.x, ++slot) {
            buffer[k].x = next[slot];
            buffer[k].y = 0;
        }
        __syncthreads();
        FFT().execute(buffer);
        __syncthreads();
    }
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        long long j = mode == 2 ? (long long)batch * (rows + 2) * (n + 2) + (row + 1) * (n + 2) + col + 1 : ((long long)batch * n + col) * rows + row;
        dst[j] = -buffer[col + 1].y * norm;
    }
}
"""
