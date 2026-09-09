"""CPU 2.5-D particle-in-cell transverse space-charge pipeline.

The public pipeline is intentionally independent of PASS ``Simulation`` and
particle classes.  Inputs may be NumPy arrays or an object exposing ``x``,
``y`` and optionally ``tag``.  A single call deposits every longitudinal slice
into a density stack and solves all slices as batched right-hand sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import logging

import numpy as np

from .fd_arbitrary import (
    RectangleAperture,
    build_aperture,
    build_aperture_mask as _build_continuous_aperture_mask,
    build_fd_arbitrary_resources,
)
from .dst_rectangle import build_dst_rectangle_resources
from .fft_free_space import build_fft_free_space_resources
from .fd_rectangle import FDSolver, build_fd_resources


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
