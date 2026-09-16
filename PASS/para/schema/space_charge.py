"""Schemas for named transverse space-charge resources and commands."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator


class SpaceChargeResourceConfig(BaseModel):
    """One named PIC or free-space analytic transverse field configuration."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)

    slice_set: str = Field(
        default="space_charge",
        min_length=1,
        alias="Slice set",
        description="Named Slicer result consumed by commands using this configuration",
    )
    nx: int = Field(default=128, ge=3, alias="Nx")
    ny: int = Field(default=128, ge=3, alias="Ny")
    grid_width_x: float | None = Field(default=None, gt=0.0, alias="Grid Width X (m)")
    grid_width_y: float | None = Field(default=None, gt=0.0, alias="Grid Width Y (m)")
    grid_half_width_x: float | None = Field(default=None, gt=0.0, alias="Grid Half Width X (m)")
    grid_half_width_y: float | None = Field(default=None, gt=0.0, alias="Grid Half Width Y (m)")
    method: Literal["pic", "frozen", "quasi-frozen"] = Field(default="pic", alias="Method")
    solver: Literal[
        "fft_free_space", "fd_dirichlet", "dst_dirichlet",
        "gaussian_round_free_space", "gaussian_ellipse_free_space",
        "uniform_round_free_space", "uniform_ellipse_free_space",
        "parabolic_round_free_space", "parabolic_ellipse_free_space",
    ] = Field(default="fd_dirichlet", alias="Solver")
    deposition_method: Literal["CIC", "TSC"] | None = Field(
        default=None,
        alias="Particle Deposition Method",
    )
    center_x: float | None = Field(default=None, alias="Center X (m)")
    center_y: float | None = Field(default=None, alias="Center Y (m)")
    angle: float | None = Field(default=None, alias="Angle (rad)")
    sigma: float | None = Field(default=None, gt=0, alias="Sigma (m)")
    sigma_x: float | None = Field(default=None, gt=0, alias="Sigma X (m)")
    sigma_y: float | None = Field(default=None, gt=0, alias="Sigma Y (m)")
    radius: float | None = Field(default=None, gt=0, alias="Radius (m)")
    a: float | None = Field(default=None, gt=0, alias="Semi-axis A (m)")
    b: float | None = Field(default=None, gt=0, alias="Semi-axis B (m)")

    @model_validator(mode="after")
    def validate_method_parameters(self):
        pic = self.solver in {"fft_free_space", "fd_dirichlet", "dst_dirichlet"}
        if (self.method == "pic") != pic:
            raise ValueError(f"Method {self.method!r} does not support Solver {self.solver!r}")
        parameters = {"center_x", "center_y", "angle", "sigma", "sigma_x", "sigma_y", "radius", "a", "b"}
        supplied = {name for name in parameters if getattr(self, name) is not None}
        if self.method != "frozen":
            if supplied:
                raise ValueError(f"fixed transverse parameters require Method='frozen': {sorted(supplied)}")
        else:
            sizes = {
                "gaussian_round_free_space": {"sigma"},
                "gaussian_ellipse_free_space": {"sigma_x", "sigma_y"},
                "uniform_round_free_space": {"radius"},
                "uniform_ellipse_free_space": {"a", "b"},
                "parabolic_round_free_space": {"radius"},
                "parabolic_ellipse_free_space": {"a", "b"},
            }[self.solver]
            missing = sizes - supplied
            if missing:
                raise ValueError(f"frozen {self.solver} requires {sorted(missing)}")
            extra = supplied - sizes - {"center_x", "center_y", "angle"}
            if extra:
                raise ValueError(f"parameters do not apply to {self.solver}: {sorted(extra)}")
            if "round" in self.solver and self.angle not in {None, 0.0}:
                raise ValueError("round profiles do not have an orientation angle")
        if not pic and self.deposition_method is not None:
            raise ValueError("Particle Deposition Method is only valid for Method='pic'")
        widths = (self.grid_width_x, self.grid_width_y)
        halves = (self.grid_half_width_x, self.grid_half_width_y)
        if all(value is None for value in widths + halves):
            self.grid_width_x = self.grid_width_y = 0.02
        elif all(value is not None for value in widths) and all(value is None for value in halves):
            pass
        elif all(value is not None for value in halves) and all(value is None for value in widths):
            pass
        else:
            raise ValueError("Specify either both Grid Width X/Y (m) or both Grid Half Width X/Y (m), not a mixture")
        return self


class SpaceChargeConfig(BaseModel):
    """Top-level ``Space charge`` block with named resource configurations."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    enabled: StrictBool = Field(default=False, alias="Enabled")
    coverage_check: Literal["warn", "error", "off"] = Field(default="warn", alias="Coverage check")
    coverage_mode: Literal["full-ring", "partial"] = Field(default="full-ring", alias="Coverage mode")
    expected_sc_length: float | None = Field(default=None, ge=0, allow_inf_nan=False,
                                             alias="Expected SC length (m)")
    configurations: dict[str, SpaceChargeResourceConfig] = Field(
        default_factory=dict,
        alias="Configurations",
    )

    @model_validator(mode="before")
    @classmethod
    def ignore_configurations_when_disabled(cls, value):
        if not isinstance(value, dict):
            return value
        enabled = value.get("Enabled", value.get("enabled", False))
        if enabled is not False:
            return value
        value = dict(value)
        if "Configurations" in value:
            value["Configurations"] = {}
        if "configurations" in value:
            value["configurations"] = {}
        return value

    @field_validator("configurations")
    @classmethod
    def validate_configuration_names(
        cls, configurations: dict[str, SpaceChargeResourceConfig]
    ) -> dict[str, SpaceChargeResourceConfig]:
        for name in configurations:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("space-charge configuration names must be non-empty strings")
            if name != name.strip():
                raise ValueError(
                    f"space-charge configuration name {name!r} must not have surrounding whitespace"
                )
        return configurations

    @model_validator(mode="after")
    def validate_coverage_target(self):
        if self.coverage_mode == "full-ring" and self.expected_sc_length is not None:
            raise ValueError("Expected SC length (m) applies only to Coverage mode='partial'; full-ring uses circumference")
        return self


def validate_loss_aperture(aperture_type: str, value: list) -> list:
    """Validate point-local loss geometry without building field resources."""
    sizes = {"circle": 1, "rectangle": 2, "ellipse": 2, "rectcircle": 3,
             "rectellipse": 4, "racetrack": 4, "octagon": 3}
    if aperture_type not in {"off", "default", "polygon", *sizes}:
        raise ValueError(f"Unsupported particle loss aperture type: {aperture_type!r}")
    if not isinstance(value, list):
        raise ValueError("Aperture value must be a list")
    if aperture_type in {"off", "default"}:
        return value
    try:
        if aperture_type == "polygon":
            if len(value) < 3 or any(not isinstance(vertex, list) or len(vertex) != 2 for vertex in value):
                raise ValueError("polygon aperture requires at least three [x, y] vertices")
            result = [[float(x), float(y)] for x, y in value]
            numbers = [number for vertex in result for number in vertex]
        else:
            if len(value) != sizes[aperture_type]:
                raise ValueError(f"{aperture_type} aperture requires {sizes[aperture_type]} values")
            result = [float(number) for number in value]
            numbers = result
        if not all(math.isfinite(number) for number in numbers):
            raise ValueError("Aperture value must contain finite numbers")
        from PASS.utils.aperture import build_aperture

        build_aperture({"Type": aperture_type, "Value": result})
    except (TypeError, IndexError, OverflowError) as exc:
        raise ValueError(f"Invalid {aperture_type} particle loss aperture: {value}") from exc
    return result


class SpaceCharge(BaseModel):
    """A position-local command referencing one named resource configuration."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    s: float = Field(default=0.0, alias="S (m)")
    command: Literal["SpaceCharge"] = Field(default="SpaceCharge", alias="Command")
    configuration: str = Field(
        min_length=1,
        alias="Configuration",
        description="Name in the top-level Space charge.Configurations mapping",
    )
    sc_length: float = Field(default=0.0, ge=0.0, alias="SC length (m)")
    sc_start: float | None = Field(default=None, allow_inf_nan=False, alias="SC start (m)",
                                  description="Optional start of the represented integration interval; does not transport particles")
    aperture_type: str = Field(
        default="default", alias="Aperture type",
        description="Particle aperture and Dirichlet conducting boundary; default is the configuration grid rectangle",
    )
    aperture_value: list = Field(
        default_factory=list, alias="Aperture value",
        description="Aperture dimensions in meters, using the standard element aperture syntax",
    )
    save_field: bool = Field(default=False, alias="Save field")
    save_potential: bool = Field(default=False, alias="Save potential")
    save_density: bool = Field(default=False, alias="Save density")
    save_turns: list[list[int]] | list[int] = Field(default_factory=list, alias="Save turns")

    @model_validator(mode="after")
    def validate_aperture(self):
        self.aperture_value = validate_loss_aperture(self.aperture_type, self.aperture_value)
        return self

    @field_validator("configuration")
    @classmethod
    def validate_configuration_name(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("Configuration must not have surrounding whitespace")
        return value


class ElementSpaceCharge(BaseModel):
    """Internal SC placement; body length and aperture are owned by the element."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    configuration: str = Field(min_length=1, alias="Configuration")
    num_kicks: StrictInt = Field(default=1, ge=1, alias="Num kicks")
    aperture_type: str = Field(default="default", alias="Aperture type",
        description="default inherits the parent element; conflicting explicit values warn and are overridden")
    aperture_value: list = Field(default_factory=list, alias="Aperture value")
    save_field: StrictBool = Field(default=False, alias="Save field")
    save_potential: StrictBool = Field(default=False, alias="Save potential")
    save_density: StrictBool = Field(default=False, alias="Save density")
    save_turns: list[list[int]] | list[int] = Field(default_factory=list, alias="Save turns")

    @field_validator("configuration")
    @classmethod
    def validate_configuration_name(cls, value):
        if value != value.strip():
            raise ValueError("Configuration must not have surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_aperture(self):
        self.aperture_value = validate_loss_aperture(self.aperture_type, self.aperture_value)
        return self


SLICED_ELEMENT_COMMANDS = frozenset({
    "drift", "sbend", "quadrupole", "sextupole", "octupole",
    "multipole", "kicker", "bump", "solenoid", "elseparator",
})


def parse_element_space_charge(raw):
    """Accept exported aliases or case-normalized runtime JSON, strictly."""
    if isinstance(raw, ElementSpaceCharge):
        return raw
    if not isinstance(raw, dict):
        raise ValueError("Element 'Space charge' must be an object or null")
    fields = {field.alias.lower(): name for name, field in ElementSpaceCharge.model_fields.items()}
    values = {}
    for key, value in raw.items():
        name = fields.get(str(key).lower(), str(key).lower())
        if name in values:
            raise ValueError(f"Duplicate element Space charge field: {key}")
        values[name] = value
    return ElementSpaceCharge.model_validate(values)
