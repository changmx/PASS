"""Electron-cloud input models and named configuration resolution."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator

from PASS.para.schema.space_charge import validate_loss_aperture


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


class ElectronCloudBuildUpConfiguration(_ElectronCloudInput):
    """Dynamic electron motion, circular wall, and weighted secondary emission."""

    chamber_radius: float = Field(gt=0, alias="Chamber radius (m)")
    beam_sigma: float = Field(gt=0, alias="Beam sigma (m)")
    max_time_step: float = Field(gt=0, alias="Max time step (s)")
    magnetic_field: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3, alias="Magnetic field (T)")
    magnetic_gradient: float = Field(default=0.0, strict=True, alias="Magnetic gradient (T/m)")
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
        if self.mode in {"build_up", "coupled"}:
            required_solver = "round_gaussian_beam" if self.mode == "build_up" else "fd_dirichlet"
            if self.solver != required_solver or self.buildup is None:
                raise ValueError(f"ElectronCloud {self.mode} requires Solver={required_solver!r} and a Build up configuration")
            if self.aperture_type not in {"default", "off"}:
                raise ValueError(f"ElectronCloud {self.mode} uses Build up.Chamber radius (m); Aperture type must be default/off")
            if math.hypot(self.center_x, self.center_y) + self.radius >= self.buildup.chamber_radius:
                raise ValueError("ElectronCloud initial disk must lie strictly inside the build-up chamber")
            if self.mode == "coupled":
                if self.nx < 5 or self.ny < 5 or self.nx % 2 == 0 or self.ny % 2 == 0:
                    raise ValueError("ElectronCloud coupled requires odd Nx and Ny of at least 5")
                if min(self.grid_width_x, self.grid_width_y) < 2 * self.buildup.chamber_radius:
                    raise ValueError("ElectronCloud coupled grid widths must cover the full chamber diameter")
                if max(self.grid_width_x / (self.nx - 1), self.grid_width_y / (self.ny - 1)) > self.buildup.chamber_radius / 2:
                    raise ValueError("ElectronCloud coupled grid spacing must not exceed half the chamber radius")
        elif self.buildup is not None or self.solver == "round_gaussian_beam":
            raise ValueError("ElectronCloud frozen mode does not accept Build up parameters or Solver='round_gaussian_beam'")
        self.aperture_value = validate_loss_aperture(self.aperture_type, self.aperture_value)
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
