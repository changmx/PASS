"""CPU/GPU 2.5-D particle-in-cell transverse space-charge pipeline.

The public pipeline is intentionally independent of PASS ``Simulation`` and
particle classes.  Inputs may be NumPy/CuPy arrays or an object exposing ``x``,
``y`` and optionally ``tag``.  A single call deposits every longitudinal slice
into a density stack and solves all slices as batched right-hand sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import logging

import numpy as np

from .fd_arbitrary import (
    GPUArbitraryFDSolver,
    RectangleAperture,
    build_aperture,
    build_aperture_mask as _build_continuous_aperture_mask,
    build_fd_arbitrary_resources,
)
from .dst_rectangle import GPUDSTRectangleSolver, build_dst_rectangle_resources
from .fft_free_space import GPUFFTFreeSpaceSolver, build_fft_free_space_resources
from .fd_rectangle import FDSolver, GPUFDSolver, build_fd_resources
from .field_result import _gpu_module, _launch_gpu


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridGeometry:
    """Uniform nodal grid centered on zero."""

    nx: int
    ny: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if isinstance(self.nx, bool) or isinstance(self.ny, bool) or int(self.nx) != self.nx or int(self.ny) != self.ny:
            raise TypeError("nx and ny must be integers")
        object.__setattr__(self, "nx", int(self.nx))
        object.__setattr__(self, "ny", int(self.ny))
        if self.nx < 3 or self.ny < 3:
            raise ValueError("nx and ny must be at least 3 for FD Poisson")
        if not np.all(np.isfinite((self.x_min, self.x_max, self.y_min, self.y_max))):
            raise ValueError("grid bounds must be finite")
        if not self.x_min < self.x_max or not self.y_min < self.y_max:
            raise ValueError("grid minima must be smaller than maxima")

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.nx - 1)

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / (self.ny - 1)

    @property
    def x(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.nx)

    @property
    def y(self) -> np.ndarray:
        return np.linspace(self.y_min, self.y_max, self.ny)

    def locate(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return lower node indices, fractional offsets and in-grid mask."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ux = (x - self.x_min) / self.dx
        uy = (y - self.y_min) / self.dy
        inside = (ux >= 0) & (ux <= self.nx - 1) & (uy >= 0) & (uy <= self.ny - 1)
        ix = np.floor(np.clip(ux, 0, self.nx - 1)).astype(np.int64)
        iy = np.floor(np.clip(uy, 0, self.ny - 1)).astype(np.int64)
        tx = ux - ix
        ty = uy - iy
        edge_x = ix == self.nx - 1
        edge_y = iy == self.ny - 1
        ix = np.where(edge_x, self.nx - 2, ix)
        iy = np.where(edge_y, self.ny - 2, iy)
        tx = np.where(edge_x, 1.0, tx)
        ty = np.where(edge_y, 1.0, ty)
        return ix, iy, tx, ty, inside

    def inside(self, x: Any, y: Any, margin: int = 0) -> np.ndarray:
        return ((np.asarray(x) >= self.x_min + margin * self.dx)
                & (np.asarray(x) <= self.x_max - margin * self.dx)
                & (np.asarray(y) >= self.y_min + margin * self.dy)
                & (np.asarray(y) <= self.y_max - margin * self.dy))


@dataclass
class DepositResult:
    """Deposited slice stack: density is C/m^2 and charge is C per slice."""

    density: np.ndarray
    deposited_charge: np.ndarray
    deposited_count: np.ndarray
    ignored_count: int
    boundary_count: int = 0
    min_retained_weight: float = 1.0
    lost_count: int = 0

    @property
    def total_charge(self) -> float:
        return float(np.sum(self.deposited_charge))


@dataclass
class PICResult:
    """PIC fields with potential in V m and integrated fields in V."""

    density: np.ndarray
    potential: np.ndarray | None
    integrated_ex: np.ndarray
    integrated_ey: np.ndarray
    geometry: GridGeometry
    deposited_charge: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def rho(self) -> np.ndarray:
        """Compatibility alias for the deposited charge density."""
        return self.density

    @property
    def ex(self) -> np.ndarray:
        """Compatibility alias for the integrated horizontal field."""
        return self.integrated_ex

    @property
    def ey(self) -> np.ndarray:
        """Compatibility alias for the integrated vertical field."""
        return self.integrated_ey


@dataclass
class PICResources:
    geometry: GridGeometry
    aperture: Any
    aperture_mask: np.ndarray
    field_solver: Any
    solver_name: str | None = None

    @property
    def fd_solver(self) -> Any:
        """Compatibility alias for the selected transverse field solver."""
        return self.field_solver


def build_grid_geometry(config: Mapping[str, Any] | None = None, **kwargs) -> GridGeometry:
    """Create a centered grid from a mapping or explicit keyword arguments."""
    values = dict(config or {})
    mesh = values.get("Mesh") or values.get("mesh")
    if isinstance(mesh, Mapping):
        values = {**mesh, **values}
    values.update(kwargs)

    def get(*names):
        for name in names:
            if name in values:
                return values[name]
        raise KeyError(f"missing grid parameter; expected one of {names}")

    def grid_count(name: str) -> int:
        raw = get(name, name.lower())
        if isinstance(raw, bool):
            raise TypeError(f"{name} must be an integer, not bool")
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be an integer") from exc
        try:
            is_integral = np.isfinite(raw) and value == raw
        except (TypeError, ValueError, OverflowError):
            is_integral = False
        if not is_integral:
            logger.warning("%s=%r is not an integer; truncating to %d", name, raw, value)
        return value

    nx, ny = grid_count("Nx"), grid_count("Ny")

    obsolete = {"Dx", "dx", "Dy", "dy", "X min", "x_min", "X max", "x_max",
                "Y min", "y_min", "Y max", "y_max"} & values.keys()
    if obsolete:
        raise ValueError(f"Unsupported grid extent fields: {sorted(obsolete)}; use full widths or half widths")
    def optional(*names):
        return next((values[name] for name in names if values.get(name) is not None), None)
    widths = (optional("Grid Width X (m)", "grid_width_x"),
              optional("Grid Width Y (m)", "grid_width_y"))
    halves = (optional("Grid Half Width X (m)", "grid_half_width_x"),
              optional("Grid Half Width Y (m)", "grid_half_width_y"))
    if all(value is not None for value in widths) and all(value is None for value in halves):
        hx, hy = (float(value) / 2 for value in widths)
    elif all(value is not None for value in halves) and all(value is None for value in widths):
        hx, hy = map(float, halves)
    else:
        raise ValueError("Specify either both full grid widths or both half widths")
    return GridGeometry(nx, ny, -hx, hx, -hy, hy)



def build_aperture_mask(geometry: GridGeometry, aperture: Mapping[str, Any] | None = None) -> np.ndarray:
    """Build the nodal membership mask for a continuous aperture."""
    if aperture is None:
        return np.ones((geometry.ny, geometry.nx), dtype=bool)
    return _build_continuous_aperture_mask(geometry, aperture)


def build_pic_resources(
    geometry: GridGeometry,
    aperture_mask: np.ndarray | None = None,
    *,
    aperture: Mapping[str, Any] | None = None,
    field_solver: str = "fd",
) -> PICResources:
    """Build reusable geometry, aperture, and one batched field solver.

    The density supplied to every solver is C/m^2. Potentials are V m and
    transverse fields are longitudinally integrated V.

    ``field_solver`` values are:

    - ``'fd'``: default. Regular FD for the full rectangle or Shortley-Weller
      FD for any continuous aperture geometry supported by
      :mod:`PASS.utils.aperture`.
    - ``'dst_rectangle'``: direct zero-Dirichlet DST, only for the complete
      grid-aligned rectangular chamber.
    - ``'fft_free_space'``: zero-padded Hockney free-space convolution, only for
      the complete grid rectangle and never for a conducting chamber.
    A non-rectangular boolean ``aperture_mask`` is no longer sufficient for
    finite-difference geometry. Pass the equivalent continuous ``aperture``
    mapping so wall-intersection distances can be computed.
    """
    name = str(field_solver).strip().lower().replace("-", "_")
    if name not in {"fd", "dst_rectangle", "fft_free_space"}:
        raise ValueError("field_solver must be 'fd', 'dst_rectangle', or 'fft_free_space'")
    if aperture_mask is not None:
        aperture_mask = np.asarray(aperture_mask, dtype=bool)
        if aperture_mask.shape != (geometry.ny, geometry.nx):
            raise ValueError("aperture_mask shape must match geometry")
        if not np.all(aperture_mask):
            if name in {"dst_rectangle", "fft_free_space"}:
                raise ValueError(f"{name} requires aperture_mask to be all True")
            raise ValueError(
                "boolean non-rectangular aperture masks are no longer supported; "
                "pass a continuous aperture mapping to build_pic_resources(..., aperture=...)"
            )
    if aperture is None:
        spec = RectangleAperture(geometry.x_min, geometry.x_max, geometry.y_min, geometry.y_max)
        aperture_input = None
    else:
        spec = build_aperture(aperture)
        aperture_input = aperture
    derived_mask = spec.mask(*np.meshgrid(geometry.x, geometry.y))
    if aperture_mask is not None and not np.array_equal(aperture_mask, derived_mask):
        raise ValueError("aperture and aperture_mask describe different domains")

    full_rectangle = isinstance(spec, RectangleAperture) and (
        spec.x_min == geometry.x_min and spec.x_max == geometry.x_max
        and spec.y_min == geometry.y_min and spec.y_max == geometry.y_max
    )
    if name in {"dst_rectangle", "fft_free_space"} and not full_rectangle:
        raise ValueError(f"{name} requires the full grid-aligned rectangular aperture")
    if full_rectangle:
        # Use the same boundary gradient convention for implicit/explicit walls.
        aperture_input = None
    if name == "fd":
        solver = build_fd_resources(geometry) if aperture_input is None else build_fd_arbitrary_resources(geometry, aperture_input)
    elif name == "dst_rectangle":
        solver = build_dst_rectangle_resources(geometry)
    else:  # fft_free_space
        solver = build_fft_free_space_resources(geometry)
    return PICResources(geometry, spec, solver.aperture_mask.copy(), solver, name)


def _arrays(particles, tag=None):

    def value(name, fallback=None):
        if isinstance(particles, Mapping):
            return particles.get(name, fallback)
        return getattr(particles, name, fallback)

    x, y = value("x"), value("y")
    if x is None or y is None:
        raise TypeError("particles must provide x and y arrays")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("particle x and y must be finite")
    tags = value("tag", tag)
    if tags is None:
        tags = np.ones(x.shape, dtype=bool)
    else:
        tags = np.asarray(tags)
        if tags.shape != x.shape:
            raise ValueError("tag must have the same shape as x and y")
        tags = tags > 0
    return x.ravel(), y.ravel(), tags.ravel()


def _slice_count(slice_id, num_slices=None) -> int:
    raw = np.asarray(slice_id)
    if raw.dtype.kind not in "iu":
        raise TypeError("slice_id must be an integer array")
    if num_slices is not None:
        if isinstance(num_slices, bool) or int(num_slices) != num_slices:
            raise TypeError("num_slices must be an integer")
        n = int(num_slices)
    else:
        ids = raw
        valid = ids[ids >= 0]
        n = int(valid.max()) + 1 if valid.size else 0
    if n < 1:
        raise ValueError("num_slices must be positive or slice_id must contain a valid slice")
    return n


def _require_matching_geometry(geometry: GridGeometry, owner: Any, label: str) -> None:
    """Reject a solver or resource bundle built for a different nodal grid."""
    if getattr(owner, "geometry", None) != geometry:
        raise ValueError(f"{label} geometry does not match the supplied geometry")


def _default_rectangular_resources(geometry: GridGeometry, resources: PICResources | None) -> PICResources:
    """Return explicit matching resources, or the documented full-rectangle default."""
    if resources is None:
        return build_pic_resources(geometry)
    _require_matching_geometry(geometry, resources, "PICResources")
    return resources


def _deposit_stencil(
    density: np.ndarray,
    sid: np.ndarray,
    charge: np.ndarray,
    valid: np.ndarray,
    entries: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    resources: PICResources,
) -> tuple[np.ndarray, int, float]:
    """Scatter a CIC/TSC stencil after removing conductor and wall nodes.

    The remaining weights are normalized per particle.  This is the charge
    conserving counterpart of the same normalized gather stencil below.  An
    aperture-inside particle with no active node is treated as lost and
    reported through a warning; this is a mesh/aperture resolution diagnostic.
    """
    geometry = resources.geometry
    active = resources.field_solver.interior_mask
    # Open rectangular grids have no conductor nodes to remove. Avoid the
    # repeated aperture-mask/index construction for each CIC stencil entry.
    # TSC stencils crossing the outer grid still use the normalized path below.
    if np.all(active):
        all_valid = bool(np.all(valid))
        selected = slice(None) if all_valid else np.flatnonzero(valid)
        selected_count = sid.size if all_valid else selected.size
        stencil_inside = all(np.all((gx[selected] >= 0) & (gx[selected] < geometry.nx)
                                    & (gy[selected] >= 0) & (gy[selected] < geometry.ny))
                             for gx, gy, _ in entries)
        normalizer = sum(weight[selected] for _, _, weight in entries)
        if stencil_inside and np.all(normalizer > np.finfo(float).eps):
            flat = density.ravel()
            stride = geometry.ny * geometry.nx
            scale = charge[selected] / (normalizer * geometry.dx * geometry.dy)
            for gx, gy, weight in entries:
                indices = sid[selected] * stride + gy[selected] * geometry.nx + gx[selected]
                if flat.size <= 8 * selected_count:
                    flat += np.bincount(indices, weights=scale * weight[selected], minlength=flat.size)
                else:
                    # Sparse stacks should not allocate a full-grid temporary
                    # for every stencil node just to scatter a few particles.
                    np.add.at(flat, indices, scale * weight[selected])
            return valid.copy(), 0, float(normalizer.min()) if selected_count else 1.0
    normalizer = np.zeros(sid.size, dtype=float)
    for gx, gy, weight in entries:
        in_grid = (gx >= 0) & (gx < geometry.nx) & (gy >= 0) & (gy < geometry.ny)
        local = valid & in_grid
        active_local = np.zeros_like(local)
        selected = np.flatnonzero(local)
        if selected.size:
            active_local[selected] = active[gy[selected], gx[selected]]
        normalizer[active_local] += weight[active_local]

    broken = valid & (normalizer <= np.finfo(float).eps)
    if np.any(broken):
        logger.warning(
            "%d particle(s) are inside the grid/aperture but have no active "
            "deposition node; treating them as lost",
            int(broken.sum()),
        )
    active_particles = valid & ~broken
    scale = np.zeros_like(normalizer)
    scale[active_particles] = 1.0 / normalizer[active_particles]
    flat = density.ravel()
    stride = geometry.ny * geometry.nx
    for gx, gy, weight in entries:
        in_grid = (gx >= 0) & (gx < geometry.nx) & (gy >= 0) & (gy < geometry.ny)
        local = active_particles & in_grid
        active_local = np.zeros_like(local)
        selected = np.flatnonzero(local)
        if selected.size:
            active_local[selected] = active[gy[selected], gx[selected]]
        indices = sid * stride + gy * geometry.nx + gx
        np.add.at(
            flat,
            indices[active_local],
            charge[active_local] * weight[active_local] * scale[active_local] / (geometry.dx * geometry.dy),
        )
    retained = normalizer[valid & (normalizer > np.finfo(float).eps)]
    boundary_count = int(np.count_nonzero(active_particles & ~np.isclose(normalizer, 1.0)))
    if boundary_count:
        logger.warning(
            "%d particle(s) required boundary stencil weight renormalization; "
            "the local deposited shape is modified near the aperture",
            boundary_count,
        )
    return active_particles, boundary_count, float(retained.min()) if retained.size else 1.0


def _tsc_weight(position: np.ndarray, node: np.ndarray) -> np.ndarray:
    distance = np.abs(position - node)
    outer = np.maximum(0.0, 1.5 - distance)
    return np.where(distance < 0.5, 0.75 - distance**2, 0.5 * outer**2)


def deposit_cic(particles,
                slice_id,
                geometry: GridGeometry,
                resources: PICResources | None = None,
                *,
                charge_per_macro=1.0,
                num_slices=None,
                tag=None) -> DepositResult:
    """CIC-deposit macro charges (C) into a density stack in C/m^2.

    Without ``resources`` this intentionally uses the full rectangular grid
    aperture. Supply :func:`build_pic_resources` for any physical aperture.
    """
    x, y, alive = _arrays(particles, tag)
    raw_sid = np.asarray(slice_id)
    if raw_sid.dtype.kind not in "iu":
        raise TypeError("slice_id must be an integer array")
    sid = raw_sid.astype(np.int64, copy=False).ravel()
    if sid.size != x.size:
        raise ValueError("slice_id must have one entry per particle")
    ns = _slice_count(sid, num_slices)
    try:
        charge = np.broadcast_to(np.asarray(charge_per_macro, dtype=float), x.shape)
    except ValueError as exc:
        raise ValueError("charge_per_macro must be scalar or match particle shape") from exc
    if not np.all(np.isfinite(charge)):
        raise ValueError("charge_per_macro must be finite")
    resources = _default_rectangular_resources(geometry, resources)
    ix, iy, tx, ty, in_grid = geometry.locate(x, y)
    in_aperture = resources.aperture.mask(x, y)
    valid = alive & in_grid & in_aperture & (sid >= 0) & (sid < ns)
    density = np.zeros((ns, geometry.ny, geometry.nx), dtype=float)
    entries = []
    for ox, wx in ((0, 1 - tx), (1, tx)):
        for oy, wy in ((0, 1 - ty), (1, ty)):
            entries.append((ix + ox, iy + oy, wx * wy))
    deposited, boundary_count, min_retained = _deposit_stencil(density, sid, charge, valid, entries, resources)
    deposited_charge = np.bincount(sid[deposited], weights=charge[deposited], minlength=ns)
    deposited_count = np.bincount(sid[deposited], minlength=ns).astype(np.int64)
    return DepositResult(
        density, deposited_charge, deposited_count,
        int(x.size - deposited.sum()), boundary_count, min_retained,
        int(np.count_nonzero(valid & ~deposited)),
    )


def deposit_tsc(particles,
                slice_id,
                geometry: GridGeometry,
                resources: PICResources | None = None,
                *,
                charge_per_macro=1.0,
                num_slices=None,
                tag=None) -> DepositResult:
    """TSC-deposit macro charges (C) into a density stack in C/m^2.

    Without ``resources`` this intentionally uses the full rectangular grid
    aperture. Supply :func:`build_pic_resources` for any physical aperture.
    """
    x, y, alive = _arrays(particles, tag)
    raw_sid = np.asarray(slice_id)
    if raw_sid.dtype.kind not in "iu":
        raise TypeError("slice_id must be an integer array")
    sid = raw_sid.astype(np.int64, copy=False).ravel()
    if sid.size != x.size:
        raise ValueError("slice_id must have one entry per particle")
    ns = _slice_count(sid, num_slices)
    try:
        charge = np.broadcast_to(np.asarray(charge_per_macro, dtype=float), x.shape)
    except ValueError as exc:
        raise ValueError("charge_per_macro must be scalar or match particle shape") from exc
    if not np.all(np.isfinite(charge)):
        raise ValueError("charge_per_macro must be finite")
    resources = _default_rectangular_resources(geometry, resources)
    ux = (x - geometry.x_min) / geometry.dx
    uy = (y - geometry.y_min) / geometry.dy
    valid = (alive & (ux >= 0) & (ux <= geometry.nx - 1) & (uy >= 0) & (uy <= geometry.ny - 1)
             & resources.aperture.mask(x, y) & (sid >= 0) & (sid < ns))
    density = np.zeros((ns, geometry.ny, geometry.nx), dtype=float)
    cx, cy = np.floor(ux + 0.5).astype(np.int64), np.floor(uy + 0.5).astype(np.int64)
    entries = []
    for ox in (-1, 0, 1):
        gx = cx + ox
        wx = _tsc_weight(ux, gx)
        for oy in (-1, 0, 1):
            gy = cy + oy
            entries.append((gx, gy, wx * _tsc_weight(uy, gy)))
    deposited, boundary_count, min_retained = _deposit_stencil(density, sid, charge, valid, entries, resources)
    deposited_charge = np.bincount(sid[deposited], weights=charge[deposited], minlength=ns)
    deposited_count = np.bincount(sid[deposited], minlength=ns).astype(np.int64)
    return DepositResult(
        density, deposited_charge, deposited_count,
        int(x.size - deposited.sum()), boundary_count, min_retained,
        int(np.count_nonzero(valid & ~deposited)),
    )


def deposit_particles(particles, slice_id, geometry, resources=None, method="CIC", **kwargs):
    name = str(method).strip().upper()
    if name == "CIC":
        return deposit_cic(particles, slice_id, geometry, resources, **kwargs)
    if name == "TSC":
        return deposit_tsc(particles, slice_id, geometry, resources, **kwargs)
    raise ValueError("method must be 'CIC' or 'TSC'")


def solve_pic(particles,
              slice_id,
              geometry: GridGeometry,
              resources: PICResources | None = None,
              method: str = "CIC",
              *,
              charge_per_macro=1.0,
              num_slices=None,
              tag=None,
              compute_potential: bool = True) -> PICResult:
    """Run batched transverse PIC with ``charge_per_macro`` in C.

    The deposited density is C/m^2. The returned two-dimensional potential is
    V m and ``integrated_ex``/``integrated_ey`` are V. ``delta_z`` is not an
    input here: command-level code applies it only when converting integrated
    fields to average fields before a kick.
    ``compute_potential=False`` allows FFT to omit its potential transform
    and return ``potential=None``. FD/DST still compute potential to obtain
    their fields. The default preserves the complete result for every solver.
    """
    if resources is None:
        resources = build_pic_resources(geometry)
    elif not isinstance(resources, PICResources):
        raise TypeError(
            "solve_pic requires PICResources; build them with "
            "build_pic_resources(...). Bare field solvers do not carry the "
            "continuous aperture metadata required for particle deposition."
        )
    else:
        _require_matching_geometry(geometry, resources, "PICResources")
    deposited = deposit_particles(particles, slice_id, geometry, resources, method, charge_per_macro=charge_per_macro, num_slices=num_slices, tag=tag)
    if resources.solver_name == "fft_free_space":
        solved = resources.field_solver.solve(deposited.density, compute_potential=compute_potential)
    else:
        # FD/DST need the potential to differentiate it, even for field-only
        # tracking. Their existing solve protocol remains sufficient.
        solved = resources.field_solver.solve(deposited.density)
    n_slices = deposited.density.shape[0]
    return PICResult(
        deposited.density,
        solved.potential,
        solved.integrated_ex,
        solved.integrated_ey,
        geometry,
        deposited.deposited_charge,
        {
            "n_slices": n_slices,
            "deposited_count": int(deposited.deposited_count.sum()),
            "ignored_count": deposited.ignored_count,
            "lost_count": deposited.lost_count,
            "boundary_count": deposited.boundary_count,
            "min_retained_weight": deposited.min_retained_weight,
        },
    )


def _gather(field, particles, geometry, slice_id, resources=None, quadratic=False, tag=None):
    x, y, alive = _arrays(particles, tag)
    values = np.asarray(field, dtype=float)
    if values.ndim not in (2, 3) or values.shape[-2:] != (geometry.ny, geometry.nx):
        raise ValueError("field must have shape (ny, nx) or (n_slice, ny, nx) matching the grid")
    if not np.all(np.isfinite(values)):
        raise ValueError("field must be finite")
    batched = values.ndim == 3
    if not batched:
        values = values[None, ...]
    if slice_id is None:
        sid = np.zeros(x.size, dtype=np.int64)
    else:
        raw_sid = np.asarray(slice_id)
        if raw_sid.dtype.kind not in "iu":
            raise TypeError("slice_id must be an integer array")
        sid = raw_sid.astype(np.int64, copy=False).ravel()
    if sid.size != x.size:
        raise ValueError("slice_id must have one entry per particle")
    out = np.zeros(x.size, dtype=float)
    resources = _default_rectangular_resources(geometry, resources)
    valid = (
        alive
        & resources.aperture.mask(x, y)
        & geometry.inside(x, y)
        & (sid >= 0)
        & (sid < values.shape[0])
    )
    if not quadratic and np.all(resources.field_solver.interior_mask) and np.all(valid):
        # In the usual loss-free open-grid run, Boolean/advanced indexing of
        # every coordinate and every weight only makes full-size copies.
        # Index the four field nodes directly instead, retaining normalization.
        ix, iy, tx, ty, in_grid = geometry.locate(x, y)
        if np.all(in_grid):
            w00, w10 = (1-tx)*(1-ty), tx*(1-ty)
            w01, w11 = (1-tx)*ty, tx*ty
            normalizer = w00+w10+w01+w11
            return (values[sid,iy,ix]*w00 + values[sid,iy,ix+1]*w10
                    + values[sid,iy+1,ix]*w01 + values[sid,iy+1,ix+1]*w11)/normalizer
    if quadratic:
        ux, uy = (x - geometry.x_min) / geometry.dx, (y - geometry.y_min) / geometry.dy
        cx, cy = np.floor(ux + 0.5).astype(np.int64), np.floor(uy + 0.5).astype(np.int64)
        entries = []
        for ox in (-1, 0, 1):
            gx = cx + ox
            wx = _tsc_weight(ux, gx)
            for oy in (-1, 0, 1):
                gy = cy + oy
                entries.append((gx, gy, wx * _tsc_weight(uy, gy)))
    else:
        ix, iy, tx, ty, in_grid = geometry.locate(x, y)
        valid &= in_grid
        entries = [
            (ix, iy, (1 - tx) * (1 - ty)),
            (ix + 1, iy, tx * (1 - ty)),
            (ix, iy + 1, (1 - tx) * ty),
            (ix + 1, iy + 1, tx * ty),
        ]
    active = resources.field_solver.interior_mask
    if not quadratic and np.all(active):
        selected = np.flatnonzero(valid)
        normalizer = sum(weight[selected] for _, _, weight in entries)
        accumulated = np.zeros(selected.size)
        for gx, gy, weight in entries:
            accumulated += values[sid[selected], gy[selected], gx[selected]] * weight[selected]
        out[selected] = accumulated / normalizer
        return out
    normalizer = np.zeros(x.size, dtype=float)
    for gx, gy, weight in entries:
        in_grid = (gx >= 0) & (gx < geometry.nx) & (gy >= 0) & (gy < geometry.ny)
        local = valid & in_grid
        selected = np.flatnonzero(local)
        if selected.size:
            local[selected] &= active[gy[selected], gx[selected]]
        normalizer[local] += weight[local]
    missing = valid & (normalizer <= np.finfo(float).eps)
    if np.any(missing):
        logger.warning(
            "%d particle(s) are inside the grid/aperture but have no active "
            "gather node; returning zero field",
            int(missing.sum()),
        )
        # Keep the invalid particles out of every subsequent stencil update.
        # This makes the zero-field policy explicit and prevents a future
        # change to the active-node filtering from turning this case into 0/0.
        valid &= ~missing
    for gx, gy, weight in entries:
        in_grid = (gx >= 0) & (gx < geometry.nx) & (gy >= 0) & (gy < geometry.ny)
        local = valid & in_grid
        selected = np.flatnonzero(local)
        if selected.size:
            local[selected] &= active[gy[selected], gx[selected]]
        out[local] += values[sid[local], gy[local], gx[local]] * weight[local] / normalizer[local]
    return out


def gather_bilinear(field, particles, geometry, resources=None, slice_id=None, tag=None):
    """Gather with CIC weights; no resources means a full rectangle."""
    return _gather(field, particles, geometry, slice_id, resources, False, tag)


def gather_quadratic(field, particles, geometry, resources=None, slice_id=None, tag=None):
    """Gather with TSC weights; no resources means a full rectangle."""
    return _gather(field, particles, geometry, slice_id, resources, True, tag)


def pic_cpu(
    x,
    y,
    slice_id,
    charge_per_macro,
    delta_z=None,
    geometry: GridGeometry | None = None,
    *,
    mesh: Mapping[str, Any] | None = None,
    tag=None,
    method="CIC",
    num_slices=None,
    resources=None,
    compute_potential: bool = True,
) -> PICResult:
    """Convenience array API for a batched CPU PIC solve.

    ``charge_per_macro`` is in C. ``delta_z`` is accepted at this numerical
    boundary for compatibility with the command layer; it does not alter the
    transverse density (C/m^2). When supplied, its length provides the number
    of slices, including empty ones. Returned potential is V m and fields are
    longitudinally integrated V. ``mesh`` is an alternative to constructing
    :class:`GridGeometry` directly.
    ``compute_potential=False`` allows FFT to return ``potential=None``;
    the default returns potential as well as fields.
    """
    # Preserve the compact ``pic_cpu(x, y, sid, q, geometry)`` form.
    if geometry is None and isinstance(delta_z, GridGeometry):
        geometry, delta_z = delta_z, None
    if geometry is None:
        if mesh is None:
            raise TypeError("pic_cpu requires geometry or mesh")
        geometry = build_grid_geometry(mesh)
    if num_slices is None and delta_z is not None:
        num_slices = int(np.asarray(delta_z).size)
    return solve_pic(
        {
            "x": x,
            "y": y,
            "tag": tag
        },
        slice_id,
        geometry,
        resources,
        method,
        charge_per_macro=charge_per_macro,
        num_slices=num_slices,
        compute_potential=compute_potential,
    )


@dataclass
class GPUPICResources:
    """Device-resident PIC resources with one reusable particle/grid workspace.

    Supply ``num_slices`` to avoid reading a device maximum during tracking.
    Calls must be serialized on the field solver's creation stream. Returned
    grids are owned unless ``copy=False`` explicitly borrows the workspace.
    """

    geometry: GridGeometry
    aperture: object
    field_solver: object
    solver_name: str
    active: object
    deposition_strategy: str = "atomic"
    _workspace: dict = field(default_factory=dict)

    @property
    def aperture_mask(self):
        return self.field_solver.aperture_mask

    @property
    def dtype(self):
        return self.field_solver.dtype

    @property
    def fd_solver(self):
        return self.field_solver

    def prepare(self, num_slices, num_particles=0):
        import cupy as cp

        self.field_solver.prepare(num_slices)
        w = self._workspace
        g = self.geometry
        shape = (int(num_slices), g.ny, g.nx)
        if "density" not in w or w["density"].shape != shape:
            w["density"] = cp.empty(shape, self.dtype)
        if "status" not in w or w["status"].size != num_particles:
            for name in ("status", "bins"):
                w[name] = cp.empty(num_particles, cp.int32)
            for name in ("q", "retained"):
                w[name] = cp.empty(num_particles, self.dtype)

    def close(self):
        self.field_solver.stream.synchronize()
        if hasattr(self.field_solver, "close"):
            self.field_solver.close()
        self._workspace.clear()


def build_pic_resources_gpu(
    geometry,
    aperture_mask=None,
    *,
    aperture=None,
    field_solver="fd",
    dtype="float64",
    num_slices=None,
    fft_batch_size=16,
    deposition_strategy="atomic",
    dst_implementation="auto",
):
    """Build one cached GPU solver, without constructing a CPU LU factorization."""
    import cupy as cp

    name = str(field_solver).strip().lower().replace("-", "_")
    if name not in ("fd", "dst_rectangle", "fft_free_space"):
        raise ValueError(
            "field_solver must be 'fd', 'dst_rectangle', or 'fft_free_space'"
        )
    g = geometry
    if deposition_strategy not in ("atomic", "warp", "sorted_warp"):
        raise ValueError(
            "deposition_strategy must be 'atomic', 'warp', or 'sorted_warp'"
        )
    spec = (
        RectangleAperture(g.x_min, g.x_max, g.y_min, g.y_max)
        if aperture is None
        else build_aperture(aperture)
    )
    mask = spec.mask(*np.meshgrid(g.x, g.y))
    if aperture_mask is not None:
        given = np.asarray(aperture_mask, dtype=bool)
        if (
            given.shape != mask.shape
            or not given.all()
            or not np.array_equal(given, mask)
        ):
            raise ValueError(
                "provide continuous aperture geometry instead of a non-rectangular Boolean mask"
            )
    full = isinstance(spec, RectangleAperture) and (
        spec.x_min,
        spec.x_max,
        spec.y_min,
        spec.y_max,
    ) == (g.x_min, g.x_max, g.y_min, g.y_max)
    if name != "fd" and not full:
        raise ValueError(f"{name} requires the full grid-aligned rectangular aperture")
    solver = (
        (GPUFDSolver(g, dtype) if full else GPUArbitraryFDSolver(g, aperture, dtype))
        if name == "fd"
        else GPUDSTRectangleSolver(g, dtype, implementation=dst_implementation)
        if name == "dst_rectangle"
        else GPUFFTFreeSpaceSolver(g, dtype, batch_size=fft_batch_size)
    )
    result = GPUPICResources(
        g, spec, solver, name, cp.asarray(solver.interior_mask), deposition_strategy
    )
    # Compile once during initialization, before the first particle snapshot.
    _gpu_module(_PIC_CUDA, solver.dtype.str, solver.device).get_function("deposit")
    if num_slices is not None:
        result.prepare(num_slices)
    return result


def _particles_gpu(particles, slice_id, resources, tag, validate):
    import cupy as cp

    resources.field_solver._check_context()
    get = (
        particles.get
        if isinstance(particles, Mapping)
        else lambda k, default=None: getattr(particles, k, default)
    )
    x, y = (cp.asarray(get(k), dtype=resources.dtype, order="C") for k in ("x", "y"))
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    tags = get("tag", tag)
    if tags is not None:
        tags = cp.asarray(tags)
        if tags.shape != x.shape:
            raise ValueError("tag must have the same shape as x and y")
    sid = cp.asarray(slice_id)
    if sid.dtype.kind not in "iu":
        raise TypeError("slice_id must be an integer array")
    if sid.size != x.size:
        raise ValueError("slice_id must have one entry per particle")
    sid = cp.ascontiguousarray(
        sid,
        dtype=cp.int32
        if sid.dtype.itemsize <= 4 and sid.dtype.kind == "i"
        else cp.int64,
    ).ravel()
    if validate and not bool(cp.all(cp.isfinite(x) & cp.isfinite(y))):
        raise ValueError("particle coordinates must be finite")
    x, y = x.ravel(), y.ravel()
    valid = resources.aperture.mask(
        x.astype(cp.float64, copy=False), y.astype(cp.float64, copy=False)
    )
    if tags is not None:
        valid &= tags.ravel() > 0
    return x, y, sid, valid


def _slice_count_gpu(sid, num_slices):
    if num_slices is None:
        num_slices = int(sid.max()) + 1 if sid.size else 0
    if isinstance(num_slices, bool) or int(num_slices) != num_slices or num_slices < 1:
        raise ValueError("num_slices must be a positive integer")
    return int(num_slices)


def _stencil_args_gpu(resources, n, ns, method, sid):
    method = str(method).upper()
    if method not in ("CIC", "TSC"):
        raise ValueError("method must be 'CIC' or 'TSC'")
    g = resources.geometry
    t = np.float64
    return (
        np.int64(n),
        np.int32(ns),
        np.int32(g.nx),
        np.int32(g.ny),
        t(g.x_min),
        t(g.y_min),
        t(g.dx),
        t(g.dy),
        np.int32(method == "TSC"),
        np.int32(sid.dtype.itemsize == 8),
    )


def deposit_particles_gpu(
    particles,
    slice_id,
    geometry,
    resources=None,
    method="CIC",
    *,
    charge_per_macro=1.0,
    num_slices=None,
    tag=None,
    validate=True,
    copy=True,
):
    import cupy as cp

    if resources is None:
        get = (
            particles.get
            if isinstance(particles, Mapping)
            else lambda k: getattr(particles, k)
        )
        dtype = cp.asarray(get("x")).dtype
        resources = build_pic_resources_gpu(geometry, dtype=dtype)
    if not isinstance(resources, GPUPICResources) or resources.geometry != geometry:
        raise ValueError("matching GPUPICResources are required")
    x, y, sid, valid = _particles_gpu(particles, slice_id, resources, tag, validate)
    ns = _slice_count_gpu(sid, num_slices)
    q = cp.asarray(charge_per_macro, dtype=resources.dtype, order="C").ravel()
    if q.size not in (1, x.size):
        raise ValueError("charge_per_macro must be scalar or match particle shape")
    if validate and not bool(cp.all(cp.isfinite(q))):
        raise ValueError("charge_per_macro must be finite")
    resources.prepare(ns, x.size)
    w = resources._workspace
    w["density"].fill(0)
    if resources.deposition_strategy == "sorted_warp" and x.size:
        g = geometry
        cx = cp.clip(
            cp.floor((x.astype(cp.float64) - g.x_min) / g.dx), 0, g.nx - 1
        ).astype(cp.int64)
        cy = cp.clip(
            cp.floor((y.astype(cp.float64) - g.y_min) / g.dy), 0, g.ny - 1
        ).astype(cp.int64)
        keys = cp.where(
            valid & (sid >= 0) & (sid < ns),
            (cp.clip(sid, 0, ns) * g.ny + cy) * g.nx + cx,
            ns * g.ny * g.nx,
        )
        order = cp.argsort(keys)
        x, y, sid, valid = x[order], y[order], sid[order], valid[order]
        if q.size != 1:
            q = q[order]
    args = _stencil_args_gpu(resources, x.size, ns, method, sid)
    _launch_gpu(
        _PIC_CUDA,
        "deposit",
        x.size,
        (
            x,
            y,
            sid,
            valid,
            resources.active,
            q,
            np.int32(q.size == 1),
            w["density"],
            w["status"],
            w["bins"],
            w["q"],
            w["retained"],
            *args,
            np.int32(resources.deposition_strategy != "atomic"),
        ),
        resources.dtype,
    )
    # Invalid particles occupy bin zero, keeping arbitrary invalid slice IDs
    # from increasing the histogram allocation. Counters stay on the device.
    counts = cp.zeros(ns, cp.uint64)
    charges = cp.zeros(ns, resources.dtype)
    if x.size:
        _gpu_module(_PIC_CUDA, resources.dtype.str, resources.field_solver.device).get_function(
            "deposition_totals"
        )(
            ((x.size + 255) // 256,),
            (256,),
            (w["bins"], w["q"], counts, charges, np.int64(x.size), np.int32(ns)),
            shared_mem=ns * (resources.dtype.itemsize + 4) if ns <= 1024 else 0,
        )
    boundary = cp.count_nonzero(
        (w["status"] > 0) & (~cp.isclose(w["retained"], 1.0, rtol=1e-5, atol=1e-8))
    )
    minimum = w["retained"].min() if x.size else cp.asarray(1.0, dtype=resources.dtype)
    return DepositResult(
        w["density"].copy() if copy else w["density"],
        charges,
        counts,
        x.size - counts.sum(),
        boundary,
        minimum,
        cp.count_nonzero(w["status"] < 0),
    )


def gather_fields_gpu(
    ex,
    ey,
    particles,
    geometry,
    resources,
    slice_id,
    *,
    method="CIC",
    tag=None,
    validate=True,
    out=None,
):
    """Interpolate both field components in one CUDA kernel."""
    import cupy as cp

    if resources.geometry != geometry:
        raise ValueError(
            "GPUPICResources geometry does not match the supplied geometry"
        )
    x, y, sid, valid = _particles_gpu(particles, slice_id, resources, tag, validate)
    ex, ey = (cp.asarray(a, dtype=resources.dtype, order="C") for a in (ex, ey))
    if (
        ex.shape != ey.shape
        or ex.ndim not in (2, 3)
        or ex.shape[-2:] != (geometry.ny, geometry.nx)
    ):
        raise ValueError("fields must have matching (n_slice, ny, nx) shapes")
    if validate and not bool(cp.all(cp.isfinite(ex) & cp.isfinite(ey))):
        raise ValueError("fields must be finite")
    ns = ex.shape[0] if ex.ndim == 3 else 1
    if out is None:
        out = (cp.empty_like(x), cp.empty_like(y))
    elif any(
        a.shape != x.shape or a.dtype != resources.dtype or not a.flags.c_contiguous
        for a in out
    ):
        raise ValueError(
            "gather output must be contiguous and match particle shape and precision"
        )
    _launch_gpu(
        _PIC_CUDA,
        "gather_pair",
        x.size,
        (
            x,
            y,
            sid,
            valid,
            resources.active,
            ex,
            ey,
            *out,
            *_stencil_args_gpu(resources, x.size, ns, method, sid),
        ),
        resources.dtype,
    )
    return out


def pic_gpu(
    x,
    y,
    slice_id,
    charge_per_macro,
    delta_z=None,
    geometry=None,
    *,
    mesh=None,
    tag=None,
    method="CIC",
    num_slices=None,
    resources=None,
    compute_potential=True,
    validate=True,
    copy=True,
):
    """GPU counterpart of pic_cpu: C/m^2 density, V m potential, V fields.

    ``validate=False`` skips device finite-value checks for a trusted tracking
    pipeline. Shape and resource checks still run. Diagnostics are device scalars.
    """
    import cupy as cp

    if geometry is None and isinstance(delta_z, GridGeometry):
        geometry, delta_z = delta_z, None
    if geometry is None:
        if mesh is None:
            raise TypeError("pic_gpu requires geometry or mesh")
        geometry = build_grid_geometry(mesh)
    if num_slices is None and delta_z is not None:
        num_slices = int(cp.asarray(delta_z).size)
    if resources is None:
        resources = build_pic_resources_gpu(geometry, dtype=cp.asarray(x).dtype)
    deposited = deposit_particles_gpu(
        {"x": x, "y": y, "tag": tag},
        slice_id,
        geometry,
        resources,
        method,
        charge_per_macro=charge_per_macro,
        num_slices=num_slices,
        validate=validate,
        copy=False,
    )
    solved = resources.field_solver.solve(
        deposited.density,
        compute_potential=compute_potential,
        validate=False,
        copy=copy,
    )
    return PICResult(
        deposited.density.copy() if copy else deposited.density,
        solved.potential,
        solved.integrated_ex,
        solved.integrated_ey,
        geometry,
        deposited.deposited_charge,
        dict(
            n_slices=deposited.density.shape[0],
            deposited_count=deposited.deposited_count.sum(),
            ignored_count=deposited.ignored_count,
            lost_count=deposited.lost_count,
            boundary_count=deposited.boundary_count,
            min_retained_weight=deposited.min_retained_weight,
        ),
    )

_PIC_CUDA = r"""
__device__ long long read_sid(const void* ids, long long i, int wide) {
    return wide ? ((const long long*)ids)[i] : ((const int*)ids)[i];
}
__device__ double tsc_weight(double u, int node) {
    double d=fabs(u-node), outer=fmax(0.,1.5-d);
    return d<0.5 ? 0.75-d*d : 0.5*outer*outer;
}

__device__ void deposit_add(T* address,T value,int strategy) {
    if(strategy==0) {atomicAdd(address,value);return;}
    unsigned active=__activemask();
    unsigned group=__match_any_sync(active,(unsigned long long)address);
    if(__popc(group)==1) {atomicAdd(address,value);return;}
    T sum=0;
    for(unsigned peers=group;peers;peers&=peers-1)
        sum+=__shfl_sync(group,value,__ffs(peers)-1);
    if((threadIdx.x&31)==__ffs(group)-1) atomicAdd(address,sum);
}
// Only active nodes enter either stencil; deposition and gathering use the
// same renormalization, including fractional Shortley-Weller boundaries.
__device__ int stencil(T x,T y,int nx,int ny,double xmin,double ymin,double dx,double dy,
    const bool* active,int quadratic,int* nodes,T* weights,T* norm) {
    // Geometry retains host FP64 bounds even for FP32 particles. Rounding a
    // near-wall coordinate onto the wall would otherwise erase its stencil.
    double u=((double)x-xmin)/dx,v=((double)y-ymin)/dy;
    if(!isfinite(u)||!isfinite(v)||u<0||u>nx-1||v<0||v>ny-1) return -1;
    int ix=quadratic ? (int)floor(u+(T)0.5) : min((int)floor(u),nx-2);
    int iy=quadratic ? (int)floor(v+(T)0.5) : min((int)floor(v),ny-2);
    int count=0; *norm=0;
    for(int ox=quadratic?-1:0;ox<=1;++ox) {
        int gx=ix+ox;
        double wx=quadratic?tsc_weight(u,gx):(ox?u-ix:1-(u-ix));
        for(int oy=quadratic?-1:0;oy<=1;++oy) {
            int gy=iy+oy;
            if(gx<0||gx>=nx||gy<0||gy>=ny||!active[gy*nx+gx]) continue;
            double wy=quadratic?tsc_weight(v,gy):(oy?v-iy:1-(v-iy));
            nodes[count]=gy*nx+gx; weights[count]=wx*wy; *norm+=weights[count]; ++count;
        }
    }
    return count;
}

extern "C" __global__ void deposit(const T* x,const T* y,const void* sid,
    const bool* valid,const bool* active,const T* charge,int scalar_charge,
    T* rho,int* status,int* bins,T* deposited_q,T* retained,long long size,
    int ns,int nx,int ny,double xmin,double ymin,double dx,double dy,int quadratic,int wide,int strategy) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=size) return;
    status[i]=0; bins[i]=0; deposited_q[i]=0; retained[i]=1;
    long long s=read_sid(sid,i,wide);
    if(!valid[i]||s<0||s>=ns) return;
    int nodes[9]; T weights[9],norm;
    int count=stencil(x[i],y[i],nx,ny,xmin,ymin,dx,dy,active,quadratic,nodes,weights,&norm);
    if(count<0) return;
    if(norm<=(T)2.2204460492503131e-16) { status[i]=-1; return; }
    status[i]=1; bins[i]=s+1; retained[i]=norm;
    T q=charge[scalar_charge?0:i],scale=q/(norm*dx*dy);
    deposited_q[i]=q;
    for(int k=0;k<count;++k) deposit_add(rho+s*nx*ny+nodes[k],weights[k]*scale,strategy);
}

extern "C" __global__ void gather_pair(const T* x,const T* y,const void* sid,
    const bool* valid,const bool* active,const T* ex,const T* ey,T* outx,T* outy,
    long long size,int ns,int nx,int ny,double xmin,double ymin,double dx,double dy,int quadratic,int wide) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=size) return;
    outx[i]=0;outy[i]=0;
    long long s=read_sid(sid,i,wide);
    if(!valid[i]||s<0||s>=ns) return;
    int nodes[9]; T weights[9],norm;
    int count=stencil(x[i],y[i],nx,ny,xmin,ymin,dx,dy,active,quadratic,nodes,weights,&norm);
    if(count<0||norm<=(T)2.2204460492503131e-16) return;
    T a=0,b=0;
    for(int k=0;k<count;++k) {
        long long j=s*nx*ny+nodes[k];
        a+=weights[k]*ex[j];b+=weights[k]*ey[j];
    }
    outx[i]=a/norm;outy[i]=b/norm;
}

extern "C" __global__ void deposition_totals(const int* bins,const T* q,
    unsigned long long* counts,T* charges,long long size,int ns) {
    extern __shared__ double storage[];
    T* local_q=(T*)storage;
    unsigned int* local_n=(unsigned int*)(local_q+ns);
    bool shared=ns<=1024;
    if(shared) {
        for(int j=threadIdx.x;j<ns;j+=blockDim.x) {local_q[j]=0;local_n[j]=0;}
        __syncthreads();
    }
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<size && bins[i]>0) {
        int s=bins[i]-1;
        if(shared) {atomicAdd(local_q+s,q[i]);atomicAdd(local_n+s,1u);}
        else {atomicAdd(charges+s,q[i]);atomicAdd(counts+s,1ull);}
    }
    if(shared) {
        __syncthreads();
        for(int j=threadIdx.x;j<ns;j+=blockDim.x) if(local_n[j]) {
            atomicAdd(charges+j,local_q[j]);atomicAdd(counts+j,(unsigned long long)local_n[j]);
        }
    }
}
"""
