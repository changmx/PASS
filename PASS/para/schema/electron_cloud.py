"""Electron-cloud input models and named configuration resolution."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator

from PASS.para.schema.space_charge import validate_loss_aperture
from PASS.utils.constants import const


class _ElectronCloudInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, value):
        if not isinstance(value, dict):
            return value
        names = {}
        for name, field in cls.model_fields.items():
            names[name.casefold()] = name
            names[(field.alias or name).casefold()] = name
        normalized = {}
        for key, item in value.items():
            name = names.get(str(key).casefold(), key)
            if name in normalized:
                raise ValueError(f"Duplicate electron-cloud field {name!r}")
            normalized[name] = item
        return normalized


class SurfaceEmissionConfiguration(_ElectronCloudInput):
    """Phenomenological three-channel model, not a complete Furman-Pivi fit."""

    sampling: Literal["stochastic_event", "weighted_channels"] = Field(default="stochastic_event", alias="Sampling")
    true_yield_max: float = Field(default=0.0, ge=0, alias="True yield max")
    true_peak_energy_ev: float = Field(default=300.0, gt=0, alias="True peak energy (eV)")
    true_shape: float = Field(default=1.35, gt=1, alias="True shape")
    true_yield_energy_ev: list[float] = Field(default_factory=list, alias="True yield energy table (eV)")
    true_yield_table: list[float] = Field(default_factory=list, alias="True yield table")
    yield_table_outside: Literal["zero", "constant"] = Field(default="zero", alias="Yield table outside")
    yield_angle_coefficient: float = Field(default=0.0, ge=0, le=5, alias="Yield angle coefficient")
    peak_angle_coefficient: float = Field(default=0.0, ge=0, alias="Peak angle coefficient")
    true_energy_distribution: Literal["monoenergetic", "gamma"] = Field(default="gamma", alias="True energy distribution")
    true_energy_ev: float = Field(default=2.0, gt=0, alias="True mean energy (eV)")
    true_energy_shape: float = Field(default=2.0, gt=0, alias="True energy shape")
    elastic_probability: float = Field(default=0.0, ge=0, le=1, alias="Elastic probability at zero energy")
    elastic_energy_scale_ev: float = Field(default=50.0, gt=0, alias="Elastic energy scale (eV)")
    rediffused_probability: float = Field(default=0.0, ge=0, le=1, alias="Rediffused probability")
    rediffused_energy_shape: float = Field(default=1.0, gt=0, alias="Rediffused energy shape")
    max_secondaries_per_impact: StrictInt = Field(default=1000, ge=1, alias="Max secondaries per impact")

    @model_validator(mode="after")
    def _probabilities(self):
        if self.true_yield_energy_ev or self.true_yield_table:
            if len(self.true_yield_energy_ev) < 2 or len(self.true_yield_energy_ev) != len(self.true_yield_table):
                raise ValueError("True yield table requires equally sized energy/yield arrays with at least two points")
            if self.true_yield_energy_ev[0] < 0 or any(b <= a for a, b in zip(self.true_yield_energy_ev, self.true_yield_energy_ev[1:])):
                raise ValueError("True yield table energies must be nonnegative and strictly increasing")
            if any(value < 0 for value in self.true_yield_table):
                raise ValueError("True yield table must be nonnegative")
        if self.elastic_probability + self.rediffused_probability > 1:
            raise ValueError("Elastic plus rediffused probabilities must not exceed one")
        if self.rediffused_probability == 1 and (self.true_yield_max > 0 or any(self.true_yield_table)):
            raise ValueError("Nonzero true yield requires a nonzero true-emission branch probability")
        return self


class PrimarySourceConfiguration(_ElectronCloudInput):
    """Prescribed source; yields count real electrons over the station length."""

    kind: Literal["gas_ionization", "prescribed_wall", "beam_loss", "photoelectric"] = Field(alias="Kind")
    macroparticles: StrictInt = Field(default=64, ge=1, alias="Macro electrons")
    energy_ev: float = Field(default=2.0, ge=0, alias="Mean energy (eV)")
    energy_distribution: Literal["monoenergetic", "gamma"] = Field(default="monoenergetic", alias="Energy distribution")
    energy_shape: float = Field(default=2.0, gt=0, alias="Energy shape")
    electrons_per_particle_per_m: float = Field(default=0.0, ge=0, alias="Electrons per beam particle (1/m)")
    pressure_pa: float = Field(default=0.0, ge=0, alias="Gas pressure (Pa)")
    temperature_k: float = Field(default=293.15, gt=0, alias="Gas temperature (K)")
    ionization_cross_section_m2: float = Field(default=0.0, ge=0, alias="Ionization cross section (m^2)")
    loss_fraction_per_m: float = Field(default=0.0, ge=0, alias="Beam loss fraction (1/m)")
    electrons_per_lost_particle: float = Field(default=0.0, ge=0, alias="Electrons per lost particle")
    photon_rate_per_m_per_s: float = Field(default=0.0, ge=0, alias="Photon rate (1/m/s)")
    quantum_efficiency: float = Field(default=0.0, ge=0, le=1, alias="Quantum efficiency")
    wall_angle_min: float = Field(default=0.0, alias="Wall angle min (rad)")
    wall_angle_max: float = Field(default=6.283185307179586, alias="Wall angle max (rad)")

    @model_validator(mode="after")
    def _angle_range(self):
        if not 0 < self.wall_angle_max - self.wall_angle_min <= 6.283185307179586:
            raise ValueError("Wall angle interval must have width in (0, 2*pi]")
        return self


class PopulationControlConfiguration(_ElectronCloudInput):
    """Local two-representative merging with exact weight, momentum and energy."""

    target_macroparticles: StrictInt = Field(default=50000, ge=2, alias="Target macro electrons")
    spatial_cell_size: float = Field(default=0.001, gt=0, alias="Spatial cell size (m)")
    momentum_cell_size: float = Field(default=0.01, gt=0, alias="Normalized momentum cell size")
    max_weight_ratio: float = Field(default=10.0, gt=1, alias="Max weight to mean ratio")
    max_split_children: StrictInt = Field(default=64, ge=2, alias="Max split children")


class ElectronCloudBeamDriver(_ElectronCloudInput):
    """Prescribed Gaussian or tabulated relative transverse number density."""

    model: Literal["gaussian", "density_map"] = Field(default="gaussian", alias="Model")
    sigma_x: float | None = Field(default=None, gt=0, alias="Sigma X (m)")
    sigma_y: float | None = Field(default=None, gt=0, alias="Sigma Y (m)")
    center_x: float = Field(default=0., alias="Center X (m)")
    center_y: float = Field(default=0., alias="Center Y (m)")
    x: list[float] = Field(default_factory=list, alias="X (m)")
    y: list[float] = Field(default_factory=list, alias="Y (m)")
    density: list[list[float]] = Field(default_factory=list, alias="Relative density")

    @model_validator(mode="after")
    def _validate_profile(self):
        if self.model == "gaussian":
            if self.sigma_x is None or self.sigma_y is None:
                raise ValueError("Gaussian beam driver requires positive Sigma X and Sigma Y")
            if self.x or self.y or self.density:
                raise ValueError("Gaussian beam driver does not accept a density map")
        else:
            if self.sigma_x is not None or self.sigma_y is not None or self.center_x != 0 or self.center_y != 0:
                raise ValueError("density-map coordinates define its profile; Gaussian parameters are not accepted")
            if len(self.x) < 2 or len(self.y) < 2 or any(b <= a for values in (self.x, self.y) for a, b in zip(values, values[1:])):
                raise ValueError("density map axes must be strictly increasing with at least two points")
            if len(self.density) != len(self.y) or any(len(row) != len(self.x) for row in self.density):
                raise ValueError("density map shape must be (len(y), len(x))")
            if any(value < 0 for row in self.density for value in row) or not any(value > 0 for row in self.density for value in row):
                raise ValueError("density map must be nonnegative and nonzero")
        return self


class ElectronCloudBuildUpConfiguration(_ElectronCloudInput):
    """Dynamic electron initial conditions, motion and surface emission."""

    chamber_radius: float | None = Field(default=None, gt=0, alias="Chamber radius (m)")
    beam_sigma: float | None = Field(default=None, gt=0, alias="Beam sigma (m)")
    self_fields: StrictBool = Field(default=False, alias="Electron self fields")
    beam_driver: ElectronCloudBeamDriver | None = Field(default=None, alias="Beam driver")
    surface_emission: SurfaceEmissionConfiguration | None = Field(default=None, alias="Surface emission")
    primary_sources: list[PrimarySourceConfiguration] = Field(default_factory=list, alias="Primary sources")
    population_control: PopulationControlConfiguration | None = Field(default=None, alias="Population control")
    max_time_step: float = Field(gt=0, alias="Max time step (s)")
    magnetic_field: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3, alias="Magnetic field (T)")
    magnetic_gradient: float = Field(default=0.0, strict=True, alias="Magnetic gradient (T/m)")
    initial_distribution: Literal["uniform_disk", "uniform_chamber", "uniform_rectangle", "gaussian"] = Field(default="uniform_disk",
                                                                                                              alias="Initial distribution")
    initial_position_spread: list[float] | None = Field(default=None, min_length=2, max_length=2, alias="Initial position spread (m)")
    initial_line_density: float | None = Field(default=None, ge=0, alias="Initial electron line density (1/m)")
    initial_velocity_distribution: Literal["monoenergetic", "uniform", "gaussian"] = Field(default="monoenergetic",
                                                                                           alias="Initial velocity distribution")
    initial_velocity_spread: list[float] | None = Field(default=None, min_length=3, max_length=3, alias="Initial velocity spread (m/s)")
    initial_energy_ev: float = Field(default=0.0, ge=0, alias="Initial electron energy (eV)")
    primary_electrons_per_particle_per_m: float = Field(default=0.0, ge=0, alias="Primary electrons per beam particle (1/m)")
    primary_macroparticles: StrictInt = Field(default=64, ge=1, alias="Primary macro electrons")
    secondary_yield_max: float = Field(default=0.0, ge=0, alias="Secondary yield max")
    secondary_peak_energy_ev: float = Field(default=300.0, gt=0, alias="Secondary peak energy (eV)")
    secondary_shape: float = Field(default=1.35, gt=1, alias="Secondary shape")
    emission_energy_ev: float = Field(default=2.0, gt=0, alias="Emission energy (eV)")
    max_macroparticles: StrictInt = Field(default=100000, ge=1, alias="Max macro electrons")
    max_steps: StrictInt = Field(default=100000, ge=1, alias="Max steps")
    max_wall_hits_per_step: StrictInt = Field(default=32, ge=1, alias="Max wall hits per step")

    @model_validator(mode="after")
    def _validate_models(self):
        if self.initial_distribution in {"uniform_rectangle", "gaussian"}:
            if self.initial_position_spread is None or any(value <= 0 for value in self.initial_position_spread):
                raise ValueError("Initial rectangle/Gaussian distribution requires two positive Initial position spread values")
        elif self.initial_position_spread is not None:
            raise ValueError("Initial position spread is only used by rectangle/Gaussian distributions")
        if self.initial_velocity_distribution == "monoenergetic":
            if self.initial_velocity_spread is not None:
                raise ValueError("Monoenergetic initial velocities do not accept Initial velocity spread")
        else:
            if self.initial_velocity_spread is None or any(value < 0 for value in self.initial_velocity_spread):
                raise ValueError("Initial uniform/Gaussian velocities require three nonnegative Initial velocity spread values")
            if self.initial_energy_ev != 0:
                raise ValueError("Initial velocity spread cannot be combined with nonzero Initial electron energy")
            if self.initial_velocity_distribution == "uniform" and math.hypot(*self.initial_velocity_spread) >= const.c:
                raise ValueError("Uniform initial velocity box must lie strictly below the speed of light")
        if self.primary_sources and self.primary_electrons_per_particle_per_m > 0:
            raise ValueError("Primary sources cannot be combined with the legacy primary yield")
        if self.surface_emission is not None and self.secondary_yield_max > 0:
            raise ValueError("Surface emission cannot be combined with legacy Secondary yield max")
        if self.population_control is not None and self.population_control.target_macroparticles > self.max_macroparticles:
            raise ValueError("population target must not exceed Max macro electrons")
        return self


class ElectronCloudConfiguration(_ElectronCloudInput):
    """A frozen cloud or a dynamically driven electron population."""

    mode: Literal["frozen", "build_up", "coupled"] = Field(default="frozen", alias="Mode")
    solver: Literal["uniform_round_free_space", "fd_dirichlet", "dst_dirichlet", "fft_free_space",
                    "round_gaussian_beam"] = Field(default="uniform_round_free_space", alias="Solver")
    electron_density: float = Field(ge=0, alias="Electron density (1/m^3)")
    radius: float = Field(gt=0, alias="Radius (m)")
    center_x: float = Field(default=0.0, alias="Center X (m)")
    center_y: float = Field(default=0.0, alias="Center Y (m)")
    n_macroparticles: StrictInt = Field(default=10000, ge=1, alias="Number of macro electrons")
    random_seed: StrictInt | None = Field(default=None, alias="Random seed")
    nx: StrictInt = Field(default=65, ge=3, alias="Nx")
    ny: StrictInt = Field(default=65, ge=3, alias="Ny")
    grid_width_x: float = Field(default=0.1, gt=0, alias="Grid Width X (m)")
    grid_width_y: float = Field(default=0.1, gt=0, alias="Grid Width Y (m)")
    deposition_method: Literal["CIC", "TSC"] = Field(default="CIC", alias="Particle Deposition Method")
    aperture_type: str = Field(default="default", alias="Aperture type")
    aperture_value: list = Field(default_factory=list, alias="Aperture value")
    buildup: ElectronCloudBuildUpConfiguration | None = Field(default=None, alias="Build up")

    @model_validator(mode="after")
    def _validate_aperture(self):
        self.aperture_value = validate_loss_aperture(self.aperture_type, self.aperture_value)
        if self.mode in {"build_up", "coupled"}:
            required_solver = "round_gaussian_beam" if self.mode == "build_up" else "fd_dirichlet"
            if self.solver != required_solver or self.buildup is None:
                raise ValueError(f"ElectronCloud {self.mode} requires Solver={required_solver!r} and a Build up configuration")
            if self.mode == "build_up" and self.buildup.beam_driver is None and self.buildup.beam_sigma is None:
                raise ValueError("ElectronCloud build_up requires Beam sigma or Beam driver")
            if self.mode == "coupled" and self.buildup.beam_driver is not None:
                raise ValueError("ElectronCloud coupled uses actual beam slices and does not accept Beam driver")
            if self.buildup.initial_line_density is not None and self.electron_density != 0:
                raise ValueError("Initial electron line density requires Electron density=0 to avoid specifying the intensity twice")
            if self.buildup.initial_distribution == "uniform_chamber" and (self.center_x != 0 or self.center_y != 0):
                raise ValueError("Initial uniform_chamber distribution follows the chamber geometry and requires Center X=Center Y=0")
            if self.aperture_type in {"default", "off"} and self.buildup.chamber_radius is None:
                raise ValueError("ElectronCloud dynamic default/off aperture requires Chamber radius")
            if self.aperture_type not in {"default", "off", "circle", "ellipse", "rectangle", "racetrack", "polygon"}:
                raise ValueError("unsupported dynamic electron chamber geometry")
            if self.aperture_type not in {"default", "off"} and self.buildup.chamber_radius is not None:
                if self.aperture_type != "circle" or self.aperture_value != [self.buildup.chamber_radius]:
                    raise ValueError("explicit electron chamber conflicts with Chamber radius")
            if self.buildup.initial_distribution == "uniform_disk" and self.aperture_type in {
                    "default", "off", "circle"
            } and math.hypot(self.center_x, self.center_y) + self.radius >= (self.buildup.chamber_radius
                                                                             if self.aperture_type in {"default", "off"} else self.aperture_value[0]):
                raise ValueError("ElectronCloud initial disk must lie strictly inside the build-up chamber")
            if self.mode == "coupled" or self.buildup.self_fields or self.buildup.beam_driver is not None or self.aperture_type not in {
                    "default", "off", "circle"
            }:
                if self.nx < 5 or self.ny < 5 or self.nx % 2 == 0 or self.ny % 2 == 0:
                    raise ValueError("ElectronCloud coupled requires odd Nx and Ny of at least 5")
                if self.aperture_type in {"default", "off"} and min(self.grid_width_x, self.grid_width_y) < 2 * self.buildup.chamber_radius:
                    raise ValueError("ElectronCloud coupled grid widths must cover the full chamber diameter")
                if self.aperture_type in {"default", "off"} and max(self.grid_width_x / (self.nx - 1), self.grid_width_y /
                                                                    (self.ny - 1)) > self.buildup.chamber_radius / 2:
                    raise ValueError("ElectronCloud coupled grid spacing must not exceed half the chamber radius")
                if self.aperture_type not in {"default", "off"}:
                    from PASS.utils.aperture import aperture_bounds, build_aperture
                    xmin, xmax, ymin, ymax = aperture_bounds(build_aperture({"Type": self.aperture_type, "Aperture Value": self.aperture_value}))
                    if min(xmin, -xmax) < -self.grid_width_x / 2 or min(ymin, -ymax) < -self.grid_width_y / 2:
                        raise ValueError("ElectronCloud PIC grid must cover the full conducting chamber")
                    if max(self.grid_width_x / (self.nx - 1), self.grid_width_y / (self.ny - 1)) > min(xmax - xmin, ymax - ymin) / 4:
                        raise ValueError("ElectronCloud chamber is unresolved by the PIC grid spacing")
        elif self.buildup is not None or self.solver == "round_gaussian_beam":
            raise ValueError("ElectronCloud frozen mode does not accept Build up parameters or Solver='round_gaussian_beam'")
        if self.aperture_type in {"default", "off"} and self.aperture_value:
            raise ValueError("ElectronCloud default/off aperture requires an empty Aperture value")
        if self.solver in {"uniform_round_free_space", "fft_free_space"} and self.aperture_type not in {"default", "off"}:
            raise ValueError("ElectronCloud free-space solvers accept only default/off aperture; their grid is not a conducting chamber")
        if self.mode == "frozen" and self.solver in {"fd_dirichlet", "dst_dirichlet"} and self.aperture_type == "off":
            raise ValueError("ElectronCloud Dirichlet solvers require a finite conducting aperture")
        if self.solver == "dst_dirichlet" and self.aperture_type != "default":
            if self.aperture_type != "rectangle" or self.aperture_value != [self.grid_width_x / 2, self.grid_width_y / 2]:
                raise ValueError("ElectronCloud dst_dirichlet requires the full grid-aligned rectangle")
        return self


class ElectronCloudConfig(_ElectronCloudInput):
    """Per-beam switch and named cloud configurations."""

    enabled: StrictBool = Field(default=False, alias="Enabled")
    configurations: dict[str, ElectronCloudConfiguration] = Field(default_factory=dict, alias="Configurations")

    @model_validator(mode="before")
    @classmethod
    def _ignore_disabled_configurations(cls, value):
        value = cls._normalize_fields(value)
        if isinstance(value, dict) and value.get("enabled", False) is False:
            value = dict(value)
            value["configurations"] = {}
        return value

    @field_validator("configurations")
    @classmethod
    def _validate_configuration_names(cls, values):
        for name in values:
            if not name.strip() or name != name.strip():
                raise ValueError("ElectronCloud configuration names must be nonempty without surrounding whitespace")
        return values


class ElectronCloudItem(_ElectronCloudInput):
    """A cloud interaction point with an optional physical-time slice driver."""

    s: float = Field(default=0.0, alias="S (m)")
    order: StrictInt | None = Field(default=None, alias="Order")
    command: Literal["ElectronCloud"] = Field(default="ElectronCloud", alias="Command")
    configuration: str = Field(min_length=1, alias="Configuration")
    slice_set: str | None = Field(default=None, min_length=1, alias="Slice set")
    interaction_length: float = Field(ge=0, alias="Interaction length (m)")
    is_enabled: StrictBool = Field(default=True, alias="Is enabled")
    save_fields: StrictBool = Field(default=False, alias="Save fields")
    save_turns: list[list[StrictInt]] | list[StrictInt] = Field(default_factory=list, alias="Save turns")

    @field_validator("configuration")
    @classmethod
    def _validate_configuration_name(cls, value):
        if not value.strip() or value != value.strip():
            raise ValueError("ElectronCloud Configuration must be nonempty without surrounding whitespace")
        return value

    @field_validator("slice_set")
    @classmethod
    def _validate_slice_set_name(cls, value):
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("ElectronCloud Slice set must be nonempty without surrounding whitespace")
        return value

    @field_validator("save_turns")
    @classmethod
    def _validate_save_turns(cls, value):
        if not value:
            return []
        items = [value] if all(isinstance(item, int) for item in value) else value
        for item in items:
            if len(item) not in {1, 3}:
                raise ValueError("ElectronCloud Save turns entries must be [turn] or [start, end, step]")
            if item[0] < 0 or len(item) == 3 and (item[1] < item[0] or item[2] <= 0):
                raise ValueError("ElectronCloud Save turns requires 0 <= start <= end and step > 0")
        return items


def _input_value(data, name, default=None):
    matches = [value for key, value in data.items() if str(key).casefold() == name.casefold()]
    if len(matches) > 1:
        raise ValueError(f"Duplicate electron-cloud input field {name!r}")
    return matches[0] if matches else default


def load_electron_cloud(data, *, validate_sequence=True):
    """Resolve a beam's cloud block before recursive engine key folding."""
    block = ElectronCloudConfig.model_validate(_input_value(data, "Electron cloud", {}))
    if not block.enabled or not validate_sequence:
        return block
    sequence = _input_value(data, "Sequence", {})
    if not isinstance(sequence, dict):
        raise ValueError("ElectronCloud configuration resolution requires a Sequence object")
    for name, values in sequence.items():
        if not isinstance(values, dict) or str(_input_value(values, "Command", "")).casefold() != "electroncloud":
            continue
        normalized = dict(values)
        command_key = next(key for key in normalized if str(key).casefold() == "command")
        normalized[command_key] = "ElectronCloud"
        item = ElectronCloudItem.model_validate(normalized)
        if item.is_enabled and item.configuration not in block.configurations:
            raise ValueError(f"ElectronCloud command {name!r} references undefined configuration {item.configuration!r}")
        if item.is_enabled and block.configurations[item.configuration].mode in {"build_up", "coupled"} and item.slice_set is None:
            raise ValueError(f"ElectronCloud {block.configurations[item.configuration].mode} command {name!r} requires Slice set")
    return block


class ElectronCloudResponseItem(_ElectronCloudInput):
    """A response station; charge, reference factors and length are in the file."""

    s: float = Field(default=0.0, alias="S (m)")
    order: StrictInt | None = Field(default=None, alias="Order")
    command: Literal["ElectronCloudResponse"] = Field(default="ElectronCloudResponse", alias="Command")
    response_file: str = Field(min_length=1, alias="Response file")
    slice_set: str = Field(min_length=1, alias="Slice set")
    timeline_mode: Literal["finite", "periodic"] = Field(default="finite", alias="Timeline mode")
    maximum_centroid_displacement: float | None = Field(default=None, gt=0, alias="Maximum centroid displacement (m)")
    profile_tolerance: float | None = Field(
        default=None,
        ge=0,
        alias="Profile tolerance",
        description="Relative Frobenius tolerance of the centered transverse covariance; higher moments are not checked")
    is_enabled: StrictBool = Field(default=True, alias="Is enabled")

    @field_validator("response_file", "slice_set")
    @classmethod
    def _validate_name(cls, value):
        if value != value.strip() or not value.strip():
            raise ValueError("Response file and Slice set must be nonempty without surrounding whitespace")
        return value
