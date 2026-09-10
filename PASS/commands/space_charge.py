"""Transverse space-charge command using CPU PIC or analytic free-space fields.

The command consumes a previously computed bunch-local ``SliceSet``.  PIC
itself remains a pure particle-snapshot operation; this layer owns the
``delta_z`` normalization and the normalized transverse momentum kick.
"""

from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from PASS.commands.command import Command
from PASS.commands.solver.analytic import solve_analytic, sample_analytic_grid
from PASS.commands.solver.pic import (
    GridGeometry,
    build_grid_geometry,
    build_pic_resources,
    gather_bilinear,
    gather_quadratic,
    solve_pic,
)
from PASS.utils.aperture import build_aperture, aperture_bounds, RectangleAperture
from PASS.utils.constants import const
from PASS.utils.slicing import resolve_internal_sc_aperture
from PASS.utils.aperture import check_aperture_cpu
from PASS.para.schema.space_charge import (
    validate_loss_aperture, parse_element_space_charge, SLICED_ELEMENT_COMMANDS,
)
from PASS.utils.logger import set_simple_logging, set_normal_logging

logger = logging.getLogger(__name__)


def _normalise_kwargs(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key).strip().lower(): value for key, value in values.items()}


def _first(values: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in values:
            return values[name]
    return default


def _save_turn_ranges(raw: Any) -> list[tuple[int, int, int]]:
    """Normalize the shared Save turns syntax, accepting ``[0]`` shorthand."""
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("SpaceCharge 'Save turns' must be a list")
    if not raw:
        return []
    items = [raw] if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in raw) else raw
    ranges = []
    for item in items:
        if not isinstance(item, (list, tuple)):
            raise ValueError("SpaceCharge 'Save turns' items must be [turn] or [start, end, step]")
        if len(item) == 1:
            start, end, step = item[0], item[0], 1
        elif len(item) == 3:
            start, end, step = item
        else:
            raise ValueError("SpaceCharge 'Save turns' items must be [turn] or [start, end, step]")
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (start, end, step)):
            raise ValueError("SpaceCharge 'Save turns' values must be integers")
        start, end, step = int(start), int(end), int(step)
        if step <= 0 or end < start:
            raise ValueError("SpaceCharge 'Save turns' requires end >= start and step > 0")
        ranges.append((start, end, step))
    return ranges


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "space_charge"


@dataclass(frozen=True)
class SpaceChargeConfiguredResources:
    """Shared grid and a solver cache keyed by command aperture."""

    configuration_name: str
    slice_set_name: str
    configuration: Any
    geometry: GridGeometry
    pic: dict = field(default_factory=dict)


def initialize_space_charge_resources(sim) -> dict[tuple[int, str], SpaceChargeConfiguredResources]:
    """Build only referenced named resource sets and attach them to ``sim``."""
    existing = getattr(sim, "space_charge_resources", None)
    if existing is not None:
        return existing

    cfg = sim.cfg
    registry: dict[tuple[int, str], SpaceChargeConfiguredResources] = {}
    enabled_by_beam: dict[int, bool] = {}
    settings_by_beam = getattr(cfg, "space_charge", [])
    counts_by_beam = getattr(cfg, "space_charge_configuration_counts", [])

    for beam_id, data in enumerate(getattr(cfg, "input_data", [])):
        settings = settings_by_beam[beam_id] if beam_id < len(settings_by_beam) else None
        enabled = bool(getattr(settings, "enabled", False))
        enabled_by_beam[beam_id] = enabled
        sequence = data.get("sequence", {})
        commands = [(name, command) for name, command in sequence.items()
                    if isinstance(command, Mapping) and str(command.get("command", "")).strip().lower() == "spacecharge"]
        if not enabled:
            configuration_count = counts_by_beam[beam_id] if beam_id < len(counts_by_beam) else 0
            logger.info(
                "Space charge is disabled for beam %d; %d configurations and %d commands are ignored.",
                beam_id,
                configuration_count,
                len(commands),
            )
            continue

        for name, command in sequence.items():
            if not isinstance(command, Mapping) or command.get("space charge") is None:
                continue
            if str(command.get("command", "")).lower() not in SLICED_ELEMENT_COMMANDS:
                raise ValueError(f"Command {name!r} does not support internal Space charge")
            config = parse_element_space_charge(command["space charge"])
            config = resolve_internal_sc_aperture(config, command.get("aperture type", "off"),
                                                  command.get("aperture value", []), name, sim, beam_id)
            length = float(command.get("length (m)", 0.0))
            if not np.isfinite(length) or length <= const.eps:
                raise ValueError(f"Internal Space charge in {name!r} requires a thick element")
            values = _normalise_kwargs(config.model_dump(by_alias=True))
            commands.append((f"{name} (internal)", values))

        configurations = settings.configurations
        references: dict[str, list[tuple[str, Mapping]]] = {}
        for command_name, command in commands:
            configuration_name = command.get("configuration")
            if not isinstance(configuration_name, str) or not configuration_name.strip():
                raise ValueError(f"SpaceCharge command {command_name!r} requires a non-empty 'Configuration'")
            if configuration_name != configuration_name.strip():
                raise ValueError(f"SpaceCharge command {command_name!r} has surrounding whitespace in 'Configuration'")
            references.setdefault(configuration_name, []).append((command_name, command))

        missing = sorted(set(references) - set(configurations))
        if missing:
            raise KeyError(f"beam {beam_id} SpaceCharge commands reference undefined configuration(s): {missing}")
        for name in configurations:
            if name not in references:
                logger.warning(
                    "Space-charge configuration %r for beam %d is defined but not referenced; "
                    "its computational resources will not be created.",
                    name,
                    beam_id,
                )

        for name, command_names in references.items():
            configuration = configurations[name]
            try:
                geometry = build_grid_geometry(configuration.model_dump())
                configured = SpaceChargeConfiguredResources(name, configuration.slice_set, configuration, geometry)
                # Validate every referenced command, even before allocating a
                # Poisson matrix for any of its apertures.
                apertures = []
                for command_name, command in command_names:
                    values = _normalise_kwargs(command)
                    try:
                        apertures.append(_resolve_aperture(configured, values))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"command {command_name!r}: {exc}") from exc
                for aperture_type, aperture_value in apertures:
                    _pic_resources(configured, aperture_type, aperture_value)
            except Exception as exc:
                raise ValueError(f"invalid Space-charge configuration {name!r} for beam {beam_id}: {exc}") from exc
            registry[(beam_id, name)] = configured
            logger.info("Space-charge configuration %r for beam %d: %d commands, %d solver resource sets.",
                        name, beam_id, len(command_names), len(configured.pic))

    sim.space_charge_resources = registry
    sim.space_charge_enabled = enabled_by_beam
    return registry


def _resolve_aperture(configured, values):
    """Resolve the command default and check full geometric containment."""
    kind = str(values.get("aperture type", "default")).strip().lower()
    dimensions = validate_loss_aperture(kind, values.get("aperture value", []))
    grid, config = configured.geometry, configured.configuration
    if kind == "default":
        kind, dimensions = "rectangle", [(grid.x_max - grid.x_min) / 2,
                                         (grid.y_max - grid.y_min) / 2]
    spec = build_aperture({"Type": kind, "Value": dimensions})
    bounds = aperture_bounds(spec)
    if config.solver in {"fd_dirichlet", "dst_dirichlet"} and bounds is None:
        raise ValueError("Dirichlet solvers require a finite command aperture; use default or an explicit shape")
    if config.method == "pic" and bounds is not None:
        if (bounds[0] < grid.x_min or bounds[1] > grid.x_max
                or bounds[2] < grid.y_min or bounds[3] > grid.y_max):
            raise ValueError(f"command aperture bounds {bounds} exceed grid bounds "
                             f"{(grid.x_min, grid.x_max, grid.y_min, grid.y_max)}")
    if config.solver == "dst_dirichlet" and not (
        isinstance(spec, RectangleAperture)
        and bounds == (grid.x_min, grid.x_max, grid.y_min, grid.y_max)
    ):
        raise ValueError("dst_dirichlet requires the command aperture to be the full grid-aligned rectangle")
    return kind, dimensions


def _pic_resources(configured, kind, dimensions):
    if configured.configuration.method != "pic":
        return None
    solver = configured.configuration.solver
    # The free-space solver has no conducting wall and can share its kernels
    # even when command loss apertures differ.
    key = "free_space" if solver == "fft_free_space" else json.dumps([kind, dimensions], separators=(",", ":"))
    if key not in configured.pic:
        aperture = None if solver == "fft_free_space" else {"Type": kind, "Value": dimensions}
        resources = build_pic_resources(configured.geometry, aperture=aperture,
            field_solver={"fft_free_space": "fft_free_space", "fd_dirichlet": "fd",
                          "dst_dirichlet": "dst_rectangle"}[solver])
        if not np.any(resources.field_solver.interior_mask):
            raise ValueError("command aperture has no active grid nodes; increase grid resolution")
        configured.pic[key] = resources
    return configured.pic[key]


@Command.register("SpaceCharge")
class SpaceCharge(Command):
    """Apply a transverse space-charge kick at one lattice position.

    ``slice set`` names the Slicer result to consume.  ``SC length (m)`` is
    the integration length of the kick and is intentionally independent of
    the grid's longitudinal slice width.  ``delta_z`` is used only to turn
    the slice-integrated field into an average field for all methods.
    """

    def __init__(self, beam_id: int, sim, **command_kwargs):
        values = _normalise_kwargs(command_kwargs)
        self.beam_id = int(beam_id)
        self.cmd_type = self.__class__.__name__
        self.cmd_name = str(values.get("name", "space_charge"))
        self.s = float(_first(values, "s (m)", "s", default=0.0))
        if not np.isfinite(self.s):
            raise ValueError("SpaceCharge 'S (m)' must be finite")
        registry = initialize_space_charge_resources(sim)
        self.is_enabled = bool(getattr(sim, "space_charge_enabled", {}).get(self.beam_id, False))
        self.configuration_name = str(values.get("configuration", "")).strip()
        self.aperture_type = "off"
        self.aperture_value = []
        self.sc_start = None
        if not self.is_enabled:
            self.sc_length = 0.0
            self.slice_set_name = ""
            self.save_field = False
            self.save_potential = False
            self.save_density = False
            self._save_turn_ranges = []
            self.deposition_method = ""
            self.solver = ""
            self.method = ""
            self._geometry = None
            self._resources = None
            return

        allowed = {
            "name",
            "s (m)",
            "s",
            "configuration",
            "sc length (m)",
            "sc_length",
            "sc start (m)",
            "save field",
            "save potential",
            "save density",
            "save turns",
            "aperture type",
            "aperture value",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"SpaceCharge command {self.cmd_name!r} has unsupported field(s): {unknown}")
        if not self.configuration_name:
            raise ValueError(f"SpaceCharge command {self.cmd_name!r} requires a non-empty 'Configuration'")
        try:
            configured = registry[(self.beam_id, self.configuration_name)]
        except KeyError as exc:
            raise KeyError(f"SpaceCharge command {self.cmd_name!r} references undefined configuration "
                           f"{self.configuration_name!r} for beam {self.beam_id}") from exc

        self.sc_length = float(_first(values, "sc length (m)", "sc_length", default=0.0))
        if not np.isfinite(self.sc_length) or self.sc_length < 0:
            raise ValueError("SpaceCharge 'SC length (m)' must be finite and non-negative")
        raw_start = values.get("sc start (m)")
        if raw_start is not None:
            self.sc_start = float(raw_start)
            if not np.isfinite(self.sc_start):
                raise ValueError("SpaceCharge 'SC start (m)' must be finite")
        self.slice_set_name = configured.slice_set_name
        self.save_field = _first(values, "save field", default=False)
        self.save_potential = _first(values, "save potential", default=False)
        self.save_density = _first(values, "save density", default=False)
        for name in ("save_field", "save_potential", "save_density"):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"SpaceCharge '{name}' must be boolean")
            setattr(self, name, bool(value))
        self._save_turn_ranges = _save_turn_ranges(_first(values, "save turns", default=[]))
        self.configuration = configured.configuration
        self.method = self.configuration.method
        self.solver = self.configuration.solver
        self.deposition_method = (self.configuration.deposition_method or "CIC") if self.method == "pic" else "none"
        self._geometry = configured.geometry
        self.aperture_type, self.aperture_value = _resolve_aperture(configured, values)
        self._resources = _pic_resources(configured, self.aperture_type, self.aperture_value)
        if self.method != "pic":
            if self.save_potential:
                raise ValueError("analytic SpaceCharge does not support Save potential; use Save field or Save density")

    def print(self):
        set_simple_logging()
        if not self.is_enabled:
            logger.info(f"SpaceCharge {self.cmd_name}: disabled by top-level Space charge.Enabled")
            return
        logger.info(
            "S=%g m, Command=%s, Name=%s, Configuration=%s, SC length=%g m, Slice set=%s, Method=%s, Solver=%s",
            self.s, self.cmd_type, self.cmd_name, self.configuration_name, self.sc_length,
            self.slice_set_name, self.method, self.solver)
        logger.info("SC integration interval start=%s m (coverage metadata)", self.sc_start)
        if self.geometry is not None:
            grid = self.geometry
            logger.info("%s grid=%dx%d nodes, x=[%g, %g] m, y=[%g, %g] m, dx=%g m, dy=%g m",
                        "PIC" if self.method == "pic" else "Diagnostic/default-aperture",
                        grid.nx, grid.ny, grid.x_min, grid.x_max, grid.y_min, grid.y_max, grid.dx, grid.dy)
        if self.method == "pic":
            logger.info("Deposition=%s", self.deposition_method)
        elif self.method == "frozen":
            logger.info("Fixed transverse parameters: %s", {key: value for key, value in
                        self.configuration.model_dump(by_alias=True, exclude_none=True).items()
                        if key.startswith(("Center", "Sigma", "Semi-axis", "Radius", "Angle"))})
        logger.info("Aperture: Type=%s, Value=%s; %s", self.aperture_type, self.aperture_value,
                    "loss and conducting boundary" if self.solver in {"fd_dirichlet", "dst_dirichlet"}
                    else "loss only; fields are free space")
        set_normal_logging()

    @property
    def geometry(self) -> GridGeometry | None:
        return self._geometry

    def execute_cpu(self, sim):
        if not self.is_enabled or (self.sc_length == 0.0 and self.aperture_type == "off"):
            return False
        beam = sim.beams[self.beam_id]
        for bunch in beam.bunches:
            self.apply_bunch_cpu(sim, beam, bunch)
        return True

    def apply_bunch_cpu(self, sim, beam, bunch):
        """Shared entry point for explicit commands and internal element nodes.

        Consume the existing longitudinal bin membership and widths, while
        evaluating fields at current transverse coordinates. Do not advance s
        or the reference clock, rebin particles, or traverse other bunches.
        """
        if not self.is_enabled:
            return False
        turn = int(sim.state.turn)
        check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
        if self.sc_length != 0.0:
            return self._apply_bunch_cpu(beam, bunch, beam.particles, sim, turn)
        return False

    def _turn_selected(self, turn: int) -> bool:
        return any(start <= turn <= end and (turn - start) % step == 0 for start, end, step in self._save_turn_ranges)

    def _apply_bunch_cpu(self, beam, bunch, particles, sim, turn: int) -> bool:
        try:
            slice_set = bunch.slice_sets[self.slice_set_name]
        except KeyError as exc:
            raise KeyError(f"Bunch {bunch.bunch_id} has no SliceSet {self.slice_set_name!r}; "
                           "provide slice_id and slice_table before SpaceCharge") from exc
        slice_id = getattr(slice_set, "slice_id", None)
        table = getattr(slice_set, "slice_table", None)
        if slice_id is None or not isinstance(table, Mapping) or "delta_z" not in table:
            raise RuntimeError(f"SliceSet {self.slice_set_name!r} for bunch {bunch.bunch_id} "
                               "requires slice_id and slice_table.delta_z")

        start, end = int(bunch.start_idx), int(bunch.end_idx)
        n = end - start
        if n <= 0:
            return False
        local_sid = np.asarray(slice_id)
        if local_sid.ndim != 1 or local_sid.size != n:
            raise ValueError("SliceSet.slice_id length does not match the bunch particle range")
        if local_sid.dtype.kind not in "iu":
            raise ValueError("SliceSet.slice_id must contain integers")
        delta_z = np.asarray(table["delta_z"], dtype=float)
        if delta_z.ndim != 1 or delta_z.size == 0 or not np.all(np.isfinite(delta_z)) or np.any(delta_z <= 0):
            raise ValueError("SliceSet.delta_z must contain finite positive widths")
        n_slices = int(delta_z.size)
        if np.any(local_sid < -1) or np.any(local_sid >= n_slices):
            raise ValueError("SliceSet.slice_id must be -1 or a valid slice index")
        local_sid = local_sid.astype(np.int64, copy=False)

        x = np.asarray(particles.x[start:end], dtype=float)
        y = np.asarray(particles.y[start:end], dtype=float)
        tag = np.asarray(particles.tag[start:end])
        valid = (tag > 0) & (local_sid >= 0)
        if np.any(~np.isfinite(x[valid])) or np.any(~np.isfinite(y[valid])):
            raise ValueError("SpaceCharge requires finite transverse coordinates for participating particles")
        # A macro particle represents ratio real particles, each carrying the
        # signed bunch charge.  This is the physical source charge in Coulombs.
        q_macro = float(bunch.ratio) * float(bunch.num_charge) * const.e
        if not np.isfinite(q_macro):
            raise ValueError("SpaceCharge macro-particle charge must be finite")
        analytic = None
        result = None
        if self.method != "pic":
            analytic = solve_analytic(x, y, local_sid, valid, n_slices, q_macro, self.configuration)
            integrated_ex, integrated_ey = analytic.integrated_ex, analytic.integrated_ey
            if self._turn_selected(turn) and (self.save_field or self.save_density):
                result = sample_analytic_grid(analytic, self.configuration, self.geometry)
        else:
            outside = valid & (~self.geometry.inside(x, y) | ~self._resources.aperture.mask(x, y))
            if np.any(outside):
                raise ValueError(f"SpaceCharge {self.cmd_name!r}: {np.count_nonzero(outside)} participating particles "
                                 "outside the PIC grid/aperture after the local loss check; set the point's "
                                 "Aperture type/value, apply upstream losses, or enlarge the field domain")
            result, integrated_ex, integrated_ey = self._pic_fields(x, y, tag, local_sid, n_slices, q_macro, turn)

        average_ex = np.zeros(n, dtype=float)
        average_ey = np.zeros(n, dtype=float)
        average_ex[valid] = integrated_ex[valid] / delta_z[local_sid[valid]]
        average_ey[valid] = integrated_ey[valid] / delta_z[local_sid[valid]]
        # brho includes q/A; reuse exactly the same normalization for all methods.
        beta, gamma, brho = float(bunch.beta), float(bunch.gamma), float(bunch.brho)
        if beta <= 0 or gamma <= 1.0 or abs(brho) <= const.eps or not np.isfinite(beta * gamma * brho):
            raise ValueError("Bunch relativistic parameters are invalid for SpaceCharge kick")
        kick_factor = np.sign(float(bunch.num_charge)) / (beta * const.c * brho * gamma * gamma)
        if self._turn_selected(turn) and (self.save_field or self.save_potential or self.save_density):
            self._save_hdf5(sim, beam, bunch, result, delta_z, q_macro, turn, analytic)
        particles.px[start:end] += np.asarray(kick_factor * self.sc_length * average_ex, dtype=particles.px.dtype)
        particles.py[start:end] += np.asarray(kick_factor * self.sc_length * average_ey, dtype=particles.py.dtype)
        return bool(np.any(valid))

    def _pic_fields(self, x, y, tag, local_sid, n_slices, q_macro, turn):
        result = solve_pic(
            {
                "x": x,
                "y": y,
                "tag": tag
            },
            local_sid,
            self._geometry,
            self._resources,
            self.deposition_method,
            charge_per_macro=q_macro,
            num_slices=n_slices,
            compute_potential=self.save_potential and self._turn_selected(turn),
        )
        if result.diagnostics.get("lost_count", 0):
            raise ValueError(f"SpaceCharge {self.cmd_name!r}: participating particles have no active deposition "
                             "nodes inside the aperture; increase grid resolution")
        if self.deposition_method == "CIC":
            integrated_ex = gather_bilinear(result.integrated_ex, {"x": x, "y": y, "tag": tag}, self._geometry, self._resources, local_sid, tag)
            integrated_ey = gather_bilinear(result.integrated_ey, {"x": x, "y": y, "tag": tag}, self._geometry, self._resources, local_sid, tag)
        else:
            integrated_ex = gather_quadratic(result.integrated_ex, {"x": x, "y": y, "tag": tag}, self._geometry, self._resources, local_sid, tag)
            integrated_ey = gather_quadratic(result.integrated_ey, {"x": x, "y": y, "tag": tag}, self._geometry, self._resources, local_sid, tag)

        return result, integrated_ex, integrated_ey

    def _save_hdf5(self, sim, beam, bunch, result, delta_z, charge_per_macro, turn: int, analytic=None) -> None:
        """Write one self-describing HDF5 snapshot for this bunch and turn."""
        cfg = getattr(sim, "cfg", None)
        output_root = getattr(cfg, "output_dir_space_charge", None)
        if not output_root:
            output_root = str(Path(getattr(cfg, "output_dir", ".")) / "space_charge")
        flat = getattr(cfg, "flat_output", False)
        command_dir = Path(output_root) if flat else Path(output_root) / _safe_name(self.cmd_name)
        node_suffix = ""
        if hasattr(self, "internal_node_index"):
            node_suffix = f"_node{self.internal_node_index:06d}"
            if not flat:
                command_dir /= f"internal_sc/node_{self.internal_node_index:06d}"
        if not flat:
            command_dir /= f"turn_{turn:06d}"
        command_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"sc_{_safe_name(self.cmd_name)}{node_suffix}_" if flat else ""
        filename = command_dir / (prefix + f"{_safe_name(getattr(cfg, 'output_hms', 'run'))}_"
                                  f"beam{self.beam_id}_bunch{int(bunch.bunch_id)}_turn{turn:06d}.h5")
        geometry = self.geometry
        with h5py.File(filename, "w") as handle:
            if self.sc_start is not None:
                handle.attrs["sc_start"] = self.sc_start
            if hasattr(self, "internal_node_index"):
                handle.attrs["parent_element"] = self.parent_element
                handle.attrs["internal_node_index"] = self.internal_node_index
            handle.attrs.update({
                "schema_version":
                "3",
                "method": self.method,
                "grid_role": "tracking" if self.method == "pic" else "diagnostic",
                "aperture_role": "loss_and_conductor" if self.solver in {"fd_dirichlet", "dst_dirichlet"} else "loss_only",
                "backend":
                str(getattr(cfg, "backend", "cpu")),
                "solver":
                self.solver,
                "deposition_method":
                self.deposition_method,
                "nx":
                geometry.nx,
                "ny":
                geometry.ny,
                "grid_width_x":
                geometry.x_max - geometry.x_min,
                "grid_width_y":
                geometry.y_max - geometry.y_min,
                "x_min":
                geometry.x_min,
                "x_max":
                geometry.x_max,
                "y_min":
                geometry.y_min,
                "y_max":
                geometry.y_max,
                "dx":
                geometry.dx,
                "dy":
                geometry.dy,
                "turn":
                turn,
                "beam_id":
                self.beam_id,
                "beam_name":
                str(getattr(beam, "beam_name", f"beam{self.beam_id}")),
                "bunch_id":
                int(bunch.bunch_id),
                "harmonic_id":
                int(getattr(bunch, "harmonic_id", 0)),
                "harmonic_number":
                int(getattr(bunch, "harmonic_number", 1)),
                "s":
                self.s,
                "sc_length":
                self.sc_length,
                "aperture_type": self.aperture_type,
                "aperture_value": json.dumps(self.aperture_value),
                "configuration":
                self.configuration_name,
                "slice_set":
                self.slice_set_name,
                "charge_per_macro":
                charge_per_macro,
                "num_macro_particles":
                int(getattr(bunch, "Np", 0)),
                "num_real_particles":
                int(getattr(bunch, "Nrp", 0)),
                "num_alive_macro_particles":
                int(np.count_nonzero(
                    np.asarray(beam.particles.tag)[int(bunch.start_idx):int(bunch.end_idx)] > 0)) if hasattr(beam.particles, "tag") else 0,
                "particle_precision":
                str(getattr(cfg, "particle_precision", getattr(beam.particles, "dtype", "float64"))),
                "random_seed":
                self._random_seed(cfg, self.beam_id),
                "potential_gauge":
                "boundary_zero" if self.solver in {"fd_dirichlet", "dst_dirichlet"} else
                ("kernel_reference" if self.method == "pic" else "not_computed"),
                "potential_units":
                "V m",
                "integrated_field_units":
                "V",
                "density_units":
                "C/m^2",
            })
            handle.create_dataset("x", data=geometry.x)
            handle.create_dataset("y", data=geometry.y)
            handle.create_dataset("slice_id", data=np.arange(result.density.shape[0], dtype=np.int32))
            handle.create_dataset("delta_z", data=np.asarray(delta_z, dtype=np.float64))
            handle.create_dataset("slice_charge", data=np.asarray(result.deposited_charge, dtype=np.float64))
            if analytic is not None:
                handle.attrs["size_convention"] = "principal_rms" if self.solver.startswith("gaussian") else "uniform_semi_axes"
                handle.attrs["parameter_source"] = "configuration" if self.method == "frozen" else "current_slice_population_moments"
                handle.create_dataset("macro_count", data=analytic.macro_count)
                for column, name in enumerate(("center_x", "center_y", "size_x", "size_y", "angle")):
                    dataset = handle.create_dataset(name, data=analytic.parameters[:, column])
                    dataset.attrs["units"] = "rad" if name == "angle" else "m"
            if self.save_field:
                handle.create_dataset("integrated_Ex", data=result.integrated_ex, compression="gzip", compression_opts=4)
                handle.create_dataset("integrated_Ey", data=result.integrated_ey, compression="gzip", compression_opts=4)
            if self.save_density:
                handle.create_dataset("charge_density", data=result.density, compression="gzip", compression_opts=4)
            if self.save_potential:
                handle.create_dataset("potential", data=result.potential, compression="gzip", compression_opts=4)
        logger.info("SpaceCharge '%s': saved %s", self.cmd_name, filename)

    @staticmethod
    def _random_seed(cfg, beam_id: int):
        try:
            data = cfg.input_data[beam_id]
            seed = data.get("sequence", {}).get("injection", {}).get("random seed")
            return "null" if seed is None else int(seed)
        except (AttributeError, IndexError, TypeError):
            return "null"

    def execute_gpu(self, sim):
        if not self.is_enabled or (self.sc_length == 0.0 and self.aperture_type == "off"):
            return False
        raise RuntimeError("SpaceCharge GPU execution is not implemented yet; use backend='cpu' "
                           "for PIC, frozen, and quasi-frozen methods.")
