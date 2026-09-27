"""Frozen and dynamic electron-cloud electric fields in SI units, on CPU or GPU."""

from dataclasses import dataclass

import numpy as np

from PASS.commands.solver.pic import (
    build_grid_geometry,
    build_pic_resources,
    build_pic_resources_gpu,
    gather_bilinear,
    gather_fields_gpu,
    gather_quadratic,
    pic_gpu,
    solve_pic,
)
from PASS.utils.aperture import PolygonAperture, RacetrackAperture
from PASS.utils.constants import const

from .state import ElectronCloudState


class FrozenCloudFields:
    """A uniform round analytic cloud or a frozen sampled PIC distribution.

    Electron weights are positive numbers; deposition supplies their negative
    physical charge. The PIC solver returns longitudinally integrated fields,
    which are divided by the source's represented length exactly once here.
    All cloud calculations use float64, independently of beam precision.
    """

    def __init__(self, configuration, backend="cpu", dtype="float64", state=None):
        if backend not in {"cpu", "gpu"}:
            raise ValueError("electron cloud backend must be 'cpu' or 'gpu'")
        if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("electron cloud precision must be float32 or float64")
        self.configuration = configuration
        self.backend = backend
        self.dtype = np.dtype("float64")
        self.xp = np
        if backend == "gpu":
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError("no CUDA device is available")
            except (ImportError, RuntimeError) as exc:
                raise RuntimeError("GPU electron cloud requires CuPy and an available CUDA device") from exc
            self.xp = cp
        self.solver = configuration.solver
        self.electron_density = float(configuration.electron_density)
        self.radius = float(configuration.radius)
        self.center_x, self.center_y = float(configuration.center_x), float(configuration.center_y)
        if not np.all(np.isfinite([self.electron_density, self.radius, self.center_x, self.center_y
                                   ])) or self.electron_density < 0 or self.radius <= 0:
            raise ValueError("electron cloud density must be nonnegative, radius positive, and all cloud parameters finite")
        self._radius_squared = self.radius * self.radius
        if not np.isfinite(self._radius_squared) or self._radius_squared <= 0:
            raise ValueError("electron cloud radius squared must be positive and representable in float64")
        self._electrons_per_length = self.electron_density * self._radius_squared * const.pi
        self._field_at_radius = -const.e * self.electron_density / (2 * const.epsilon0) * self.radius
        self._potential_scale = -self._field_at_radius * self.radius / 2
        if not np.all(np.isfinite([self._electrons_per_length, self._field_at_radius, self._potential_scale])):
            raise ValueError("electron cloud line density and field normalization must be representable in float64")
        self.geometry = build_grid_geometry(
            nx=configuration.nx,
            ny=configuration.ny,
            grid_width_x=configuration.grid_width_x,
            grid_width_y=configuration.grid_width_y,
        )
        self.resources = None
        self.result = None
        self.state = None
        self._active = None
        if self.solver == "uniform_round_free_space":
            if state is not None:
                raise ValueError("analytic electron clouds do not have a macro-electron state")
            return
        solver_names = {"fd_dirichlet": "fd", "dst_dirichlet": "dst_rectangle", "fft_free_space": "fft_free_space"}
        if self.solver not in solver_names:
            raise ValueError(f"unsupported electron cloud solver: {self.solver!r}")
        self.deposition_method = configuration.deposition_method
        if self.deposition_method not in {"CIC", "TSC"}:
            raise ValueError("electron cloud deposition_method must be CIC or TSC")
        aperture = None if configuration.aperture_type in {"default", "off"} else {
            "Type": configuration.aperture_type,
            "Aperture Value": configuration.aperture_value,
        }
        builder = build_pic_resources if self.backend == "cpu" else build_pic_resources_gpu
        self.resources = builder(self.geometry, aperture=aperture, field_solver=solver_names[self.solver], dtype=self.dtype)
        self._active = self.xp.asarray(self.resources.field_solver.interior_mask)
        try:
            self._validate_source_disk()
            if state is None:
                n_macroparticles = configuration.n_macroparticles
                if isinstance(n_macroparticles, (bool, np.bool_)) or not isinstance(n_macroparticles, (int, np.integer)) or n_macroparticles < 1:
                    raise ValueError("electron cloud n_macroparticles must be a positive integer")
                random_seed = configuration.random_seed
                if random_seed is not None and (isinstance(random_seed, (bool, np.bool_)) or not isinstance(random_seed, (int, np.integer))):
                    raise ValueError("electron cloud random_seed must be an integer or None")
                # Encode signed integer seeds injectively into nonnegative entropy.
                entropy = None if random_seed is None else 2 * int(random_seed) if random_seed >= 0 else -2 * int(random_seed) - 1
                generator = np.random.default_rng(entropy)
                radial = self.radius * np.sqrt(generator.random(n_macroparticles))
                angle = 2 * const.pi * generator.random(n_macroparticles)
                source_length = 1.0
                n_electrons = self._electrons_per_length * source_length
                state = ElectronCloudState(
                    self.center_x + radial * np.cos(angle),
                    self.center_y + radial * np.sin(angle),
                    np.full(n_macroparticles, n_electrons / n_macroparticles),
                    source_length,
                    generator.bit_generator.state,
                )
            self.rebuild_fields(state)
        except Exception:
            self.close()
            raise

    def _validate_source_disk(self):
        """Require the entire disk's deposition stencil to fit the active mesh."""
        grid = self.geometry
        aperture = self.resources.aperture
        if isinstance(aperture, PolygonAperture):
            self._validate_disk_polygon(aperture.vertices)
        elif isinstance(aperture, RacetrackAperture) and aperture.h != aperture.b:
            # Unequal rectangle/end heights make the shoulders nonconvex.
            # Inscribed chords give sufficient containment; a valid source
            # extremely close to a curved wall can be conservatively rejected.
            n_arc_segments = 128
            angle = np.linspace(-const.pi / 2, const.pi / 2, n_arc_segments + 1)
            right_arc = np.column_stack((aperture.w + aperture.a * np.cos(angle), aperture.b * np.sin(angle)))
            right_arc[0], right_arc[-1] = (aperture.w, -aperture.b), (aperture.w, aperture.b)
            left_arc = -right_arc
            vertices = np.vstack(
                ((-aperture.w, -aperture.h), (aperture.w, -aperture.h), right_arc, (aperture.w, aperture.h), (-aperture.w, aperture.h), left_arc))
            self._validate_disk_polygon(vertices)
        support = 1.0 if self.deposition_method == "CIC" else 1.5
        # A TSC support may reach beyond the outer node; CIC has no extra
        # support once the coordinates fit inside the nodal rectangle.
        margin = 0.0 if self.deposition_method == "CIC" else 0.5
        if (self.center_x - self.radius < grid.x_min + margin * grid.dx or self.center_x + self.radius > grid.x_max - margin * grid.dx
                or self.center_y - self.radius < grid.y_min + margin * grid.dy or self.center_y + self.radius > grid.y_max - margin * grid.dy):
            raise ValueError("electron cloud source disk and deposition stencil must fit inside the PIC grid")
        x, y = np.meshgrid(grid.x, grid.y)
        distance_x = np.maximum(np.abs(x - self.center_x) - support * grid.dx, 0)
        distance_y = np.maximum(np.abs(y - self.center_y) - support * grid.dy, 0)
        potentially_used = np.hypot(distance_x, distance_y) < self.radius
        active = self.resources.field_solver.interior_mask
        active = np.asarray(active) if self.backend == "cpu" else self.xp.asnumpy(active)
        if np.any(potentially_used & ~active):
            raise ValueError(
                "electron cloud source disk needs a complete deposition stencil inside the aperture; enlarge the chamber or refine the grid")

    def _validate_disk_polygon(self, vertices):
        """Check the continuous disk, including concave features below grid size."""
        vertices = np.asarray(vertices, dtype=float)
        polygon = PolygonAperture(tuple(tuple(point) for point in vertices))
        center = np.asarray([self.center_x, self.center_y])
        if not bool(polygon.strict_mask(center[0], center[1])):
            raise ValueError("electron cloud source disk must lie strictly inside the continuous aperture")
        edges = np.roll(vertices, -1, axis=0) - vertices
        lengths_squared = np.sum(edges**2, axis=1)
        nonzero = lengths_squared > 0
        vertices, edges, lengths_squared = vertices[nonzero], edges[nonzero], lengths_squared[nonzero]
        fractions = np.clip(np.sum((center - vertices) * edges, axis=1) / lengths_squared, 0, 1)
        nearest = vertices + fractions[:, None] * edges
        distances = np.hypot(*(nearest - center).T)
        tolerance = 64 * np.finfo(float).eps * max(self.radius, float(np.max(np.abs(vertices))), float(np.max(np.abs(center))))
        if np.any(distances <= self.radius + tolerance):
            raise ValueError("electron cloud source disk intersects the continuous aperture boundary")

    def _validate_coordinates(self, x, y):
        xp = self.xp
        x, y = xp.broadcast_arrays(xp.asarray(x, dtype=self.dtype), xp.asarray(y, dtype=self.dtype))
        if not bool(xp.all(xp.isfinite(x) & xp.isfinite(y))):
            raise ValueError("electron cloud sampling coordinates must be finite")
        return x, y

    def _validate_stencil(self, x, y, label):
        """Reject uncovered coordinates instead of treating them as beam losses."""
        xp, grid = self.xp, self.geometry
        inside = ((x >= grid.x_min) & (x <= grid.x_max) & (y >= grid.y_min) & (y <= grid.y_max))
        if self.solver != "fft_free_space":
            inside &= self.resources.aperture.strict_mask(x, y)
        if not bool(xp.all(inside)):
            raise ValueError(f"electron cloud {label} coordinates lie outside the PIC grid or chamber interior")
        ux, uy = (x - grid.x_min) / grid.dx, (y - grid.y_min) / grid.dy

        def nodes(position, n_nodes):
            if self.deposition_method == "CIC":
                center = xp.minimum(xp.floor(position).astype(xp.int64), n_nodes - 2)
                return [(center, 1 - (position - center)), (center + 1, position - center)]
            center = xp.floor(position + 0.5).astype(xp.int64)
            result = []
            for offset in (-1, 0, 1):
                node = center + offset
                distance = xp.abs(position - node)
                weight = xp.where(distance < 0.5, 0.75 - distance**2, 0.5 * xp.maximum(0, 1.5 - distance)**2)
                result.append((node, weight))
            return result

        for gx, wx in nodes(ux, grid.nx):
            for gy, wy in nodes(uy, grid.ny):
                in_grid = (gx >= 0) & (gx < grid.nx) & (gy >= 0) & (gy < grid.ny)
                active = self._active[xp.clip(gy, 0, grid.ny - 1), xp.clip(gx, 0, grid.nx - 1)]
                if bool(xp.any((wx * wy > 32 * np.finfo(float).eps) & ~(in_grid & active))):
                    raise ValueError(f"electron cloud {label} coordinates require a complete interpolation stencil inside the active grid")

    def rebuild_fields(self, state=None):
        """Build new fields from a validated frozen source, without RNG draws."""
        if self.solver == "uniform_round_free_space":
            if state is not None:
                raise ValueError("analytic electron clouds do not have macro-electron state")
            return
        state = self.state if state is None else state
        if not isinstance(state, ElectronCloudState):
            raise TypeError("PIC electron cloud requires ElectronCloudState")
        if state.n_macroparticles != self.configuration.n_macroparticles:
            raise ValueError("restored electron cloud macroparticle count does not match configuration")
        radius = np.hypot(state.x - self.center_x, state.y - self.center_y)
        if np.any(radius > self.radius * (1 + 32 * np.finfo(float).eps)):
            raise ValueError("restored electron cloud coordinates lie outside the configured source disk")
        expected_count = self._electrons_per_length * state.source_length
        if not np.isfinite(expected_count) or not np.isclose(state.weight.sum(), expected_count, rtol=2e-13, atol=0):
            raise ValueError("restored electron cloud total weight does not match configured density and source length")
        xp = self.xp
        x, y = xp.asarray(state.x), xp.asarray(state.y)
        self._validate_stencil(x, y, "source")
        slice_indices = xp.zeros(state.n_macroparticles, dtype=xp.int64)
        charge = -const.e * xp.asarray(state.weight)
        if self.backend == "cpu":
            result = solve_pic({
                "x": x,
                "y": y
            },
                               slice_indices,
                               self.geometry,
                               self.resources,
                               self.deposition_method,
                               charge_per_macro=charge,
                               num_slices=1)
        else:
            result = pic_gpu(x,
                             y,
                             slice_indices,
                             charge,
                             geometry=self.geometry,
                             resources=self.resources,
                             method=self.deposition_method,
                             num_slices=1)
        if int(result.diagnostics["ignored_count"]) or int(result.diagnostics["lost_count"]) or int(result.diagnostics["boundary_count"]):
            raise ValueError("electron cloud source deposition unexpectedly dropped or clipped macroparticles")
        if not all(bool(xp.all(xp.isfinite(values))) for values in (result.density, result.potential, result.integrated_ex, result.integrated_ey)):
            raise ValueError("electron cloud field solve produced nonfinite values")
        self.state, self.result = state, result

    def sample(self, x, y):
        """Return Ex and Ey in V/m; the caller selects live beam particles."""
        x, y = self._validate_coordinates(x, y)
        xp = self.xp
        if self.solver == "uniform_round_free_space":
            normalized_x, normalized_y, radial, distance_scale = self._scaled_radius(x, y)
            radius_ratio = self.radius / distance_scale
            field_scale = self._field_at_radius * radius_ratio / xp.maximum(radius_ratio, radial)**2
            return field_scale * normalized_x, field_scale * normalized_y
        self._validate_stencil(x, y, "witness")
        p = {"x": x.ravel(), "y": y.ravel()}
        if self.backend == "cpu":
            gather = gather_bilinear if self.deposition_method == "CIC" else gather_quadratic
            ex = gather(self.result.integrated_ex, p, self.geometry, self.resources)
            ey = gather(self.result.integrated_ey, p, self.geometry, self.resources)
        else:
            ex, ey = gather_fields_gpu(self.result.integrated_ex,
                                       self.result.integrated_ey,
                                       p,
                                       self.geometry,
                                       self.resources,
                                       xp.zeros(x.size, dtype=xp.int64),
                                       method=self.deposition_method)
        return ex.reshape(x.shape) / self.state.source_length, ey.reshape(y.shape) / self.state.source_length

    def _scaled_radius(self, x, y):
        """Normalize before squaring, including finite coordinates of opposite signs."""
        xp = self.xp
        with np.errstate(over="ignore", invalid="ignore"):
            relative_x, relative_y = x - self.center_x, y - self.center_y
            overflow = ~(xp.isfinite(relative_x) & xp.isfinite(relative_y))
            distance_scale = xp.maximum(self.radius, xp.maximum(xp.abs(relative_x), xp.abs(relative_y)))
            absolute_scale = xp.maximum(max(self.radius, abs(self.center_x), abs(self.center_y)), xp.maximum(xp.abs(x), xp.abs(y)))
            distance_scale = xp.where(overflow, absolute_scale, distance_scale)
            normalized_x = xp.where(overflow, x / distance_scale - self.center_x / distance_scale, relative_x / distance_scale)
            normalized_y = xp.where(overflow, y / distance_scale - self.center_y / distance_scale, relative_y / distance_scale)
        return normalized_x, normalized_y, xp.hypot(normalized_x, normalized_y), distance_scale

    def snapshot(self):
        """Return grid axes and SI density, potential and electric-field maps."""
        xp, grid = self.xp, self.geometry
        x, y = xp.asarray(grid.x), xp.asarray(grid.y)
        if self.solver == "uniform_round_free_space":
            xx, yy = xp.meshgrid(x, y)
            _, _, radial, distance_scale = self._scaled_radius(xx, yy)
            inside = radial <= self.radius / distance_scale
            electron_density = xp.where(inside, self.electron_density, 0)
            # Phi(R)=0. Inside the disk distance_scale=R; outside, separate
            # logarithms avoid overflowing the otherwise finite ratio r/R.
            log_ratio = xp.log(xp.maximum(radial, 1)) + xp.log(distance_scale) - np.log(self.radius)
            potential = self._potential_scale * xp.where(inside, radial**2 - 1, 2 * log_ratio)
            ex, ey = self.sample(xx, yy)
            charge_density = -const.e * electron_density
        else:
            source_length = self.state.source_length
            charge_density = self.result.density[0] / source_length
            electron_density = -charge_density / const.e
            potential = self.result.potential[0] / source_length
            ex, ey = self.result.integrated_ex[0] / source_length, self.result.integrated_ey[0] / source_length
        snapshot = dict(x=x, y=y, charge_density=charge_density, electron_density=electron_density, potential=potential, ex=ex, ey=ey)
        if not all(bool(xp.all(xp.isfinite(values))) for values in snapshot.values()):
            raise ValueError("electron cloud snapshot contains nonfinite values")
        return snapshot

    def close(self):
        if self.resources is not None and hasattr(self.resources, "close"):
            self.resources.close()


@dataclass(frozen=True)
class RoundPICResult:
    """Owned SI grids: density in C/m^3, potential in V, and ex/ey in V/m."""

    density: object
    potential: object
    ex: object
    ey: object
    geometry: object


class RoundPICFields:
    """Reusable Poisson solve for a dynamic source in a grounded round chamber.

    Near a conductor, deposition and field gathering use the same normalized
    active-node CIC/TSC weights. This preserves deposited charge, but modifies
    the particle shape within one stencil of the wall. It is not a discrete
    energy-conserving scheme; wall fields require spatial convergence checks.
    The existing Shortley-Weller solver imposes zero potential at the actual
    circular boundary and supplies its boundary-aware nodal field gradients.
    """

    def __init__(self, configuration, backend="cpu", dtype="float64"):
        if backend not in {"cpu", "gpu"} or np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("RoundPICFields requires a CPU/GPU backend and float32/float64 beam precision")
        if configuration.mode != "coupled" or configuration.solver != "fd_dirichlet":
            raise ValueError("RoundPICFields requires Mode=coupled and Solver=fd_dirichlet")
        self.configuration, self.backend, self.dtype, self.xp = configuration, backend, np.dtype("float64"), np
        if backend == "gpu":
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("RoundPICFields requires an available CUDA device")
            self.xp = cp
        self.radius = float(configuration.buildup.chamber_radius)
        if self.radius <= 0 or not np.isfinite(self.radius * self.radius) or self.radius * self.radius <= 0:
            raise ValueError("round PIC chamber radius squared must be positive and representable")
        if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 5 or value % 2 == 0
                for value in (configuration.nx, configuration.ny)):
            raise ValueError("round PIC requires odd nx and ny of at least five to cover the entire circular interior")
        self.geometry = build_grid_geometry(nx=configuration.nx,
                                            ny=configuration.ny,
                                            grid_width_x=configuration.grid_width_x,
                                            grid_width_y=configuration.grid_width_y)
        grid = self.geometry
        if min(grid.x_max, grid.y_max) < self.radius:
            raise ValueError("round PIC grid widths must cover the entire circular chamber")
        if max(grid.dx, grid.dy) > self.radius / 2:
            raise ValueError("round PIC grid spacing must not exceed half the chamber radius; refine the grid")
        self._cell_area = grid.dx * grid.dy
        if not np.isfinite(self._cell_area) or self._cell_area <= 0:
            raise ValueError("round PIC cell area must be positive and representable")
        self.deposition_method = configuration.deposition_method
        if self.deposition_method not in {"CIC", "TSC"}:
            raise ValueError("round PIC deposition_method must be CIC or TSC")
        builder = build_pic_resources if backend == "cpu" else build_pic_resources_gpu
        kwargs = {"deterministic": True} if backend == "gpu" else {}
        self.resources = builder(grid, aperture={"Type": "circle", "Aperture Value": [self.radius]}, field_solver="fd", dtype=self.dtype, **kwargs)
        self._active = self.xp.asarray(self.resources.field_solver.interior_mask)
        if backend == "gpu":
            self._initialize_gpu_kernels()

    def _initialize_gpu_kernels(self):
        """Fuse shapes and validation; keep one stable particle sort per source."""
        module = self.xp.RawModule(
            code=r"""
__device__ bool round_stencil(
    double x,
    double y,
    const bool* active,
    int nx,
    int ny,
    double x_min,
    double y_min,
    double dx,
    double dy,
    double radius,
    int method,
    int* center_x,
    int* center_y,
    long long* nodes,
    double* weights,
    int* status
) {
    if (!isfinite(x) || !isfinite(y)) {
        atomicOr(status, 1);
        return false;
    }
    if (!(hypot(x, y) < radius)) {
        atomicOr(status, 2);
        return false;
    }
    double ux = (x - x_min) / dx, uy = (y - y_min) / dy;
    int width = method == 0 ? 2 : 3;
    int offset = method == 0 ? 0 : -1;
    *center_x = method == 0 ? min((int)floor(ux), nx - 2) : (int)floor(ux + 0.5);
    *center_y = method == 0 ? min((int)floor(uy), ny - 2) : (int)floor(uy + 0.5);
    double wx[3], wy[3];
    for (int i = 0; i < width; ++i) {
        if (method == 0) {
            wx[i] = i == 0 ? 1.0 - (ux - *center_x) : ux - *center_x;
            wy[i] = i == 0 ? 1.0 - (uy - *center_y) : uy - *center_y;
        } else {
            double distance_x = fabs(ux - (*center_x + i + offset));
            double distance_y = fabs(uy - (*center_y + i + offset));
            double tail_x = fmax(0.0, 1.5 - distance_x);
            double tail_y = fmax(0.0, 1.5 - distance_y);
            wx[i] = distance_x < 0.5 ? 0.75 - distance_x * distance_x : 0.5 * tail_x * tail_x;
            wy[i] = distance_y < 0.5 ? 0.75 - distance_y * distance_y : 0.5 * tail_y * tail_y;
        }
    }
    double normalizer = 0.0;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            int gx = *center_x + i + offset, gy = *center_y + j + offset;
            int stencil = i * width + j;
            bool in_grid = gx >= 0 && gx < nx && gy >= 0 && gy < ny;
            nodes[stencil] = (long long)max(0, min(gy, ny - 1)) * nx + max(0, min(gx, nx - 1));
            weights[stencil] = in_grid && active[nodes[stencil]] ? wx[i] * wy[j] : 0.0;
            normalizer += weights[stencil];
        }
    }
    if (!isfinite(normalizer) || !(normalizer > 2.2204460492503131e-16)) {
        atomicOr(status, 4);
        return false;
    }
    for (int stencil = 0; stencil < width * width; ++stencil)
        weights[stencil] /= normalizer;
    return true;
}

extern "C" __global__ void prepare_charge(
    const double* x,
    const double* y,
    const double* charge,
    const bool* active,
    long long* keys,
    double* contributions,
    int* status,
    long long n_particles,
    int nx,
    int ny,
    double x_min,
    double y_min,
    double dx,
    double dy,
    double radius,
    int method
) {
    long long particle = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= n_particles)
        return;
    keys[particle] = 9223372036854775807LL;
    if (!isfinite(charge[particle])) {
        atomicOr(status, 8);
        return;
    }
    int center_x, center_y;
    long long nodes[9];
    double weights[9];
    if (!round_stencil(x[particle], y[particle], active, nx, ny, x_min, y_min, dx, dy, radius, method, &center_x, &center_y, nodes, weights, status))
        return;
    keys[particle] = ((long long)center_y * nx + center_x) * (n_particles + 1) + particle;
    int n_stencil = method == 0 ? 4 : 9;
    double cell_area = dx * dy;
    for (int stencil = 0; stencil < n_stencil; ++stencil)
        contributions[(long long)stencil * n_particles + particle] = charge[particle] * weights[stencil] / cell_area;
}

__device__ long long lower_bound_key(
    const long long* keys,
    long long size,
    long long key
) {
    long long low = 0, high = size;
    while (low < high) {
        long long middle = low + (high - low) / 2;
        if (keys[middle] < key)
            low = middle + 1;
        else
            high = middle;
    }
    return low;
}

extern "C" __global__ void sum_charge(
    const long long* sorted_keys,
    const double* contributions,
    const bool* active,
    double* density,
    int* status,
    long long* dense_nodes,
    int dense_limit,
    long long n_particles,
    int nx,
    int ny,
    int method
) {
    long long node = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= (long long)nx * ny)
        return;
    double density_value = 0.0;
    if (active[node]) {
        int gx = node % nx, gy = node / nx;
        int width = method == 0 ? 2 : 3;
        int offset = method == 0 ? 0 : -1;
        long long stride = n_particles + 1;
        // A fixed stencil and original-particle order avoids atomics and
        // cancellation of weak remote charge in global prefix differences.
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < width; ++j) {
                int center_x = gx - i - offset, center_y = gy - j - offset;
                if (center_x < 0 || center_x >= nx || center_y < 0 || center_y >= ny)
                    continue;
                long long key = ((long long)center_y * nx + center_x) * stride;
                long long start = lower_bound_key(sorted_keys, n_particles, key);
                if (start + dense_limit < n_particles && sorted_keys[start + dense_limit] < key + stride) {
                    // Each node has one writer. Integer queue order does not
                    // affect its fixed floating-point reduction tree.
                    dense_nodes[atomicAdd(status + 1, 1)] = node;
                    return;
                }
                double total = 0.0;
                long long stencil_start = (long long)(i * width + j) * n_particles;
                for (long long index = start; index < n_particles && sorted_keys[index] < key + stride; ++index) {
                    long long particle = sorted_keys[index] - key;
                    total += contributions[stencil_start + particle];
                }
                density_value += total;
            }
        }
    }
    density[node] = density_value;
    if (!isfinite(density_value))
        atomicOr(status, 16);
    else if (density_value != 0.0)
        atomicOr(status, 128);
}

extern "C" __global__ void sum_dense_charge(
    const long long* sorted_keys,
    const double* contributions,
    const long long* dense_nodes,
    double* density,
    int* status,
    long long n_particles,
    int nx,
    int ny,
    int method
) {
    if (blockIdx.x >= status[1])
        return;
    long long node = dense_nodes[blockIdx.x];
    int gx = node % nx, gy = node / nx;
    int width = method == 0 ? 2 : 3;
    int offset = method == 0 ? 0 : -1;
    int thread = threadIdx.x;
    long long stride = n_particles + 1;
    __shared__ long long starts[9], ends[9], keys[9];
    __shared__ double warp_sums[4];
    if (thread < width * width) {
        int center_x = gx - thread / width - offset;
        int center_y = gy - thread % width - offset;
        starts[thread] = ends[thread] = keys[thread] = 0;
        if (center_x >= 0 && center_x < nx && center_y >= 0 && center_y < ny) {
            long long key = ((long long)center_y * nx + center_x) * stride;
            keys[thread] = key;
            starts[thread] = lower_bound_key(sorted_keys, n_particles, key);
            ends[thread] = lower_bound_key(sorted_keys, n_particles, key + stride);
        }
    }
    __syncthreads();
    double density_value = 0.0;
    // Exactly 128 threads, with a fixed strided partition and warp tree.
    for (int stencil = 0; stencil < width * width; ++stencil) {
        double total = 0.0;
        long long stencil_start = (long long)stencil * n_particles;
        for (long long index = starts[stencil] + thread; index < ends[stencil]; index += 128) {
            long long particle = sorted_keys[index] - keys[stencil];
            total += contributions[stencil_start + particle];
        }
        for (int distance = 16; distance > 0; distance /= 2)
            total += __shfl_down_sync(0xffffffff, total, distance);
        if (thread % 32 == 0)
            warp_sums[thread / 32] = total;
        __syncthreads();
        if (thread == 0) {
            double stencil_sum = 0.0;
            for (int warp = 0; warp < 4; ++warp)
                stencil_sum += warp_sums[warp];
            density_value += stencil_sum;
        }
        __syncthreads();
    }
    if (thread == 0) {
        density[node] = density_value;
        if (!isfinite(density_value))
            atomicOr(status, 16);
        else if (density_value != 0.0)
            atomicOr(status, 128);
    }
}

extern "C" __global__ void check_fields(
    const double* potential,
    const double* ex,
    const double* ey,
    int* status,
    long long size
) {
    long long node = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (node < size && (!isfinite(potential[node]) || !isfinite(ex[node]) || !isfinite(ey[node])))
        atomicOr(status, 32);
}

extern "C" __global__ void gather_fields(
    const double* x,
    const double* y,
    const bool* active,
    const double* ex_a,
    const double* ey_a,
    const double* ex_b,
    const double* ey_b,
    double* sampled,
    int* status,
    long long n_particles,
    int nx,
    int ny,
    double x_min,
    double y_min,
    double dx,
    double dy,
    double radius,
    int method,
    int n_fields
) {
    long long particle = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (particle < (long long)nx * ny) {
        if (!isfinite(ex_a[particle]) || !isfinite(ey_a[particle]) || (n_fields == 2 && (!isfinite(ex_b[particle]) || !isfinite(ey_b[particle]))))
            atomicOr(status, 32);
    }
    if (particle >= n_particles)
        return;
    int center_x, center_y;
    long long nodes[9];
    double weights[9];
    if (!round_stencil(x[particle], y[particle], active, nx, ny, x_min, y_min, dx, dy, radius, method, &center_x, &center_y, nodes, weights, status))
        return;
    int n_stencil = method == 0 ? 4 : 9;
    double ax = 0.0, ay = 0.0, bx = 0.0, by = 0.0;
    for (int stencil = 0; stencil < n_stencil; ++stencil) {
        long long node = nodes[stencil];
        double weight = weights[stencil];
        ax += ex_a[node] * weight;
        ay += ey_a[node] * weight;
        if (n_fields == 2) {
            bx += ex_b[node] * weight;
            by += ey_b[node] * weight;
        }
    }
    sampled[particle] = ax;
    sampled[n_particles + particle] = ay;
    if (n_fields == 2) {
        sampled[2 * n_particles + particle] = bx;
        sampled[3 * n_particles + particle] = by;
    }
    if (!isfinite(ax) || !isfinite(ay) || !isfinite(bx) || !isfinite(by))
        atomicOr(status, 64);
}
""",
            options=("--fmad=false", ),
        )
        self._prepare_charge = module.get_function("prepare_charge")
        self._sum_charge = module.get_function("sum_charge")
        self._sum_dense_charge = module.get_function("sum_dense_charge")
        self._check_fields = module.get_function("check_fields")
        self._gather_fields = module.get_function("gather_fields")
        grid = self.geometry
        self._gpu_geometry = (np.int32(grid.nx), np.int32(grid.ny), np.float64(grid.x_min), np.float64(grid.y_min), np.float64(grid.dx),
                              np.float64(grid.dy), np.float64(self.radius), np.int32(self.deposition_method == "TSC"))
        self._dense_limit = 1024

    @staticmethod
    def _check_gpu_status(status):
        value = int(status.get()[0])
        messages = (
            (1, "round PIC coordinates must be finite"),
            (2, "round PIC coordinates must lie strictly inside the circular chamber"),
            (4, "round PIC particle has no resolved active stencil; refine the grid without discarding its charge"),
            (8, "round PIC charge_per_length must be finite"),
            (16, "round PIC charge density is not representable"),
            (32, "round PIC field solve/result contains nonfinite fields; fields must be finite"),
            (64, "round PIC sampled fields are not representable"),
        )
        for flag, message in messages:
            if value & flag:
                raise ValueError(message)
        return value

    def _coordinates(self, x, y):
        if self.resources is None:
            raise RuntimeError("round PIC resources are closed")
        xp = self.xp
        x, y = xp.asarray(x), xp.asarray(y)
        if x.dtype.kind not in "fiu" or y.dtype.kind not in "fiu":
            raise ValueError("round PIC coordinates must be real numeric arrays")
        if self.backend == "gpu":
            self.resources.field_solver._check_context()
            original_x, original_y = x, y
            x, y = xp.broadcast_arrays(x.astype(xp.float64, copy=False), y.astype(xp.float64, copy=False))
            # An empty broadcast must not hide a nonfinite scalar input.
            if not x.size and (not bool(xp.all(xp.isfinite(original_x))) or not bool(xp.all(xp.isfinite(original_y)))):
                raise ValueError("round PIC coordinates must be finite")
            return x, y
        if not bool(xp.all(xp.isfinite(x))) or not bool(xp.all(xp.isfinite(y))):
            raise ValueError("round PIC coordinates must be finite")
        x, y = xp.broadcast_arrays(x.astype(xp.float64, copy=False), y.astype(xp.float64, copy=False))
        if not bool(xp.all(xp.isfinite(x) & xp.isfinite(y))):
            raise ValueError("round PIC coordinates must be finite")
        if not bool(xp.all(xp.hypot(x, y) < self.radius)):
            raise ValueError("round PIC coordinates must lie strictly inside the circular chamber")
        return x, y

    def _stencil(self, x, y):
        """Use one normalized shape for both signed-charge scatter and E gather."""
        xp, grid = self.xp, self.geometry
        ux, uy = (x.ravel() - grid.x_min) / grid.dx, (y.ravel() - grid.y_min) / grid.dy

        def nodes(position, count):
            if self.deposition_method == "CIC":
                center = xp.minimum(xp.floor(position).astype(xp.int64), count - 2)
                fraction = position - center
                return [(center, 1 - fraction), (center + 1, fraction)]
            center = xp.floor(position + .5).astype(xp.int64)
            result = []
            for offset in (-1, 0, 1):
                node = center + offset
                distance = xp.abs(position - node)
                weight = xp.where(distance < .5, .75 - distance**2, .5 * xp.maximum(0, 1.5 - distance)**2)
                result.append((node, weight))
            return result

        entries, normalizer = [], xp.zeros(x.size, dtype=xp.float64)
        for gx, wx in nodes(ux, grid.nx):
            for gy, wy in nodes(uy, grid.ny):
                in_grid = (gx >= 0) & (gx < grid.nx) & (gy >= 0) & (gy < grid.ny)
                gx_safe, gy_safe = xp.clip(gx, 0, grid.nx - 1), xp.clip(gy, 0, grid.ny - 1)
                weight = xp.where(in_grid & self._active[gy_safe, gx_safe], wx * wy, 0.)
                entries.append((gy_safe * grid.nx + gx_safe, weight))
                normalizer += weight
        if not bool(xp.all(xp.isfinite(normalizer) & (normalizer > np.finfo(float).eps))):
            raise ValueError("round PIC particle has no resolved active stencil; refine the grid without discarding its charge")
        return [(indices, weight / normalizer) for indices, weight in entries]

    def solve(self, x, y, charge_per_length):
        """Deposit signed macro line charges in C/m and return independent SI grids."""
        x, y = self._coordinates(x, y)
        xp, grid = self.xp, self.geometry
        charge = xp.asarray(charge_per_length)
        if charge.dtype.kind not in "fiu":
            raise ValueError("round PIC charge_per_length must be real numeric")
        if self.backend == "gpu":
            return self._solve_gpu(x, y, charge)
        if not bool(xp.all(xp.isfinite(charge))):
            raise ValueError("round PIC charge_per_length must be finite")
        charge = xp.broadcast_to(charge.astype(xp.float64, copy=False), x.shape).ravel()
        density = xp.zeros((grid.ny, grid.nx), dtype=xp.float64)
        for indices, weight in self._stencil(x, y):
            contribution = charge * weight / self._cell_area
            xp.add.at(density.ravel(), indices, contribution)
        if not bool(xp.all(xp.isfinite(density))):
            raise ValueError("round PIC charge density is not representable")
        if x.size == 0 or not bool(xp.any(density)):
            return RoundPICResult(density, xp.zeros_like(density), xp.zeros_like(density), xp.zeros_like(density), grid)
        solved = self.resources.field_solver.solve(density)
        if not all(bool(xp.all(xp.isfinite(values))) for values in (solved.potential, solved.integrated_ex, solved.integrated_ey)):
            raise ValueError("round PIC field solve produced nonfinite fields")
        # The solver is linear and unit-agnostic: a C/m^3 RHS directly gives
        # V and V/m, so these fields need no additional longitudinal division.
        return RoundPICResult(density, solved.potential, solved.integrated_ex, solved.integrated_ey, grid)

    def _solve_gpu(self, x, y, charge):
        xp, grid = self.xp, self.geometry
        n_particles, n_nodes = x.size, grid.nx * grid.ny
        if n_nodes > np.iinfo(np.int64).max // (n_particles + 1):
            raise ValueError("round PIC deterministic deposition keys exceed int64 capacity")
        if not n_particles:
            if not bool(xp.all(xp.isfinite(charge))):
                raise ValueError("round PIC charge_per_length must be finite")
            xp.broadcast_to(charge, x.shape)
            arrays = [xp.zeros((grid.ny, grid.nx), dtype=xp.float64) for _ in range(4)]
            return RoundPICResult(*arrays, grid)
        charge = xp.ascontiguousarray(xp.broadcast_to(charge.astype(xp.float64, copy=False), x.shape)).ravel()
        x, y = xp.ascontiguousarray(x).ravel(), xp.ascontiguousarray(y).ravel()
        keys = xp.empty(n_particles, dtype=xp.int64)
        contributions = xp.empty((4 if self.deposition_method == "CIC" else 9, n_particles), dtype=xp.float64)
        density = xp.empty((grid.ny, grid.nx), dtype=xp.float64)
        status = xp.zeros(2, dtype=xp.int32)
        # At most K nodes per overfull cell, and disjoint cells partition N.
        max_dense_nodes = min(n_nodes, contributions.shape[0] * (n_particles // (self._dense_limit + 1)))
        dense_nodes = xp.empty(max_dense_nodes, dtype=xp.int64)
        self._prepare_charge(((n_particles + 127) // 128, ), (128, ),
                             (x, y, charge, self._active, keys, contributions, status, np.int64(n_particles), *self._gpu_geometry))
        sorted_keys = xp.sort(keys)
        self._sum_charge(((n_nodes + 127) // 128, ), (128, ),
                         (sorted_keys, contributions, self._active, density, status, dense_nodes, np.int32(
                             self._dense_limit), np.int64(n_particles), np.int32(grid.nx), np.int32(grid.ny), self._gpu_geometry[-1]))
        if max_dense_nodes:
            self._sum_dense_charge((max_dense_nodes, ), (128, ),
                                   (sorted_keys, contributions, dense_nodes, density, status, np.int64(n_particles), np.int32(
                                       grid.nx), np.int32(grid.ny), self._gpu_geometry[-1]))
        # Reject an invalid source before it can enter the shared solver workspace.
        if not self._check_gpu_status(status) & 128:
            return RoundPICResult(density, xp.zeros_like(density), xp.zeros_like(density), xp.zeros_like(density), grid)
        solved = self.resources.field_solver.solve(density, validate=False)
        self._check_fields(((n_nodes + 127) // 128, ), (128, ),
                           (solved.potential, solved.integrated_ex, solved.integrated_ey, status, np.int64(n_nodes)))
        self._check_gpu_status(status)
        return RoundPICResult(density, solved.potential, solved.integrated_ex, solved.integrated_ey, grid)

    def sample(self, result, x, y):
        """Gather the boundary-aware nodal Ex/Ey in V/m at interior particles."""
        x, y = self._coordinates(x, y)
        if self.backend == "gpu":
            return self._sample_gpu((result, ), x, y)[0]
        if not isinstance(result, RoundPICResult) or result.geometry != self.geometry:
            raise ValueError("round PIC result must belong to the same grid geometry")
        xp, grid = self.xp, self.geometry
        ex, ey = xp.asarray(result.ex, dtype=xp.float64), xp.asarray(result.ey, dtype=xp.float64)
        if ex.shape != (grid.ny, grid.nx) or ey.shape != ex.shape or not bool(xp.all(xp.isfinite(ex) & xp.isfinite(ey))):
            raise ValueError("round PIC result fields must be finite and match the grid")
        sampled_x, sampled_y = xp.zeros(x.size, dtype=xp.float64), xp.zeros(x.size, dtype=xp.float64)
        for indices, weight in self._stencil(x, y):
            sampled_x += ex.ravel()[indices] * weight
            sampled_y += ey.ravel()[indices] * weight
        if not bool(xp.all(xp.isfinite(sampled_x) & xp.isfinite(sampled_y))):
            raise ValueError("round PIC sampled fields are not representable")
        return sampled_x.reshape(x.shape), sampled_y.reshape(y.shape)

    def sample_pair(self, first, second, x, y):
        """Gather two sources with one shared particle stencil and validation."""
        if self.backend == "cpu":
            return self.sample(first, x, y), self.sample(second, x, y)
        x, y = self._coordinates(x, y)
        return self._sample_gpu((first, second), x, y)

    def _sample_gpu(self, results, x, y):
        xp, grid = self.xp, self.geometry
        fields = []
        for result in results:
            if not isinstance(result, RoundPICResult) or result.geometry != grid:
                raise ValueError("round PIC result must belong to the same grid geometry")
            ex, ey = xp.asarray(result.ex, dtype=xp.float64), xp.asarray(result.ey, dtype=xp.float64)
            if ex.shape != (grid.ny, grid.nx) or ey.shape != ex.shape:
                raise ValueError("round PIC result fields must be finite and match the grid")
            fields.extend((xp.ascontiguousarray(ex), xp.ascontiguousarray(ey)))
        if len(results) == 1:
            fields.extend(fields)
        sampled = xp.empty((2 * len(results), x.size), dtype=xp.float64)
        status = xp.zeros(1, dtype=xp.int32)
        size = max(x.size, grid.nx * grid.ny)
        self._gather_fields(((size + 127) // 128, ), (128, ),
                            (xp.ascontiguousarray(x).ravel(), xp.ascontiguousarray(y).ravel(), self._active, *fields, sampled, status, np.int64(
                                x.size), *self._gpu_geometry, np.int32(len(results))))
        self._check_gpu_status(status)
        return tuple((sampled[2 * index].reshape(x.shape), sampled[2 * index + 1].reshape(y.shape)) for index in range(len(results)))

    def close(self):
        """Release the cached factorization; saved result arrays remain independent."""
        if self.resources is not None:
            if hasattr(self.resources, "close"):
                self.resources.close()
            self.resources = None
