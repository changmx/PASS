"""Schemas for named transverse space-charge resources and commands."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator


class SpaceChargeResourceConfig(BaseModel):
    """One named, independently allocated transverse PIC resource set."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    slice_set: str = Field(
        default="space_charge",
        min_length=1,
        alias="Slice set",
        description="Named Slicer result consumed by commands using this configuration",
    )
    nx: int = Field(default=128, ge=3, alias="Nx")
    ny: int = Field(default=128, ge=3, alias="Ny")
    grid_width_x: float = Field(default=0.02, gt=0.0, alias="Grid Width X (m)")
    grid_width_y: float = Field(default=0.02, gt=0.0, alias="Grid Width Y (m)")
    field_solver: Literal["fd", "dst_rectangle", "fft_green"] = Field(
        default="fd",
        alias="Field solver",
    )
    deposition_method: Literal["CIC", "TSC"] = Field(
        default="CIC",
        alias="Particle Deposition Method",
    )
    aperture: dict | None = Field(default=None, alias="Aperture")


class SpaceChargeConfig(BaseModel):
    """Top-level ``Space charge`` block with named resource configurations."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    enabled: StrictBool = Field(default=False, alias="Enabled")
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
    save_field: bool = Field(default=False, alias="Save field")
    save_potential: bool = Field(default=False, alias="Save potential")
    save_density: bool = Field(default=False, alias="Save density")
    save_turns: list[list[int]] | list[int] = Field(default_factory=list, alias="Save turns")

    @field_validator("configuration")
    @classmethod
    def validate_configuration_name(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("Configuration must not have surrounding whitespace")
        return value
