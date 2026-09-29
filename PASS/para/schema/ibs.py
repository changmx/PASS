"""Intrabeam-scattering input models and named configuration resolution."""

from collections.abc import Mapping
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator

from PASS.para.schema.twiss import TwissItem


class _IBSInput(BaseModel):
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
                raise ValueError(f"Duplicate IBS field {name!r}")
            normalized[name] = item
        return normalized


class IBSOpticsConfig(_IBSInput):
    """Uncoupled local optics, including horizontal and vertical dispersion."""

    beta_x: float = Field(gt=0, alias="Beta x (m)")
    beta_y: float = Field(gt=0, alias="Beta y (m)")
    alpha_x: float = Field(default=0.0, alias="Alpha x")
    alpha_y: float = Field(default=0.0, alias="Alpha y")
    dx: float = Field(default=0.0, alias="Dx (m)")
    dpx: float = Field(default=0.0, alias="Dpx")
    dy: float = Field(default=0.0, alias="Dy (m)")
    dpy: float = Field(default=0.0, alias="Dpy")

    @classmethod
    def from_twiss(cls, twiss: TwissItem | Mapping, *, endpoint: Literal["entrance", "exit"] = "exit", dy: float = 0.0, dpy: float = 0.0) -> Self:
        """Copy one Twiss map endpoint; vertical dispersion is explicit input.

        ``entrance`` selects previous optics and ``exit`` selects current
        optics. This constructs input values only; it does not inspect the
        runtime lattice or modify transport. A mapping must contain a complete
        TwissItem input, not an arbitrary external optics-table row.
        """
        if endpoint not in {"entrance", "exit"}:
            raise ValueError("IBS optics endpoint must be 'entrance' or 'exit'")
        if isinstance(twiss, TwissItem):
            twiss = TwissItem.model_validate(twiss.model_dump())
        elif isinstance(twiss, Mapping):
            twiss = TwissItem.model_validate(dict(twiss))
        else:
            raise TypeError("IBS optics construction requires a TwissItem or its complete input mapping")
        if twiss.command.casefold() != "twiss":
            raise ValueError("IBS optics source must have Command='Twiss'")
        suffix = "_previous" if endpoint == "entrance" else ""
        values = {name: getattr(twiss, name + suffix) for name in ("beta_x", "beta_y", "alpha_x", "alpha_y", "dx", "dpx")}
        return cls(**values, dy=dy, dpy=dpy)


class IBSConfiguration(_IBSInput):
    """Physical model and numerical controls shared by named IBS points."""

    method: Literal["bjorken_mtingwa", "kinetic", "binary"] = Field(default="bjorken_mtingwa", alias="Method")
    coulomb_log: float = Field(gt=0, alias="Coulomb log")
    coulomb_log_note: str | None = Field(default=None, alias="Coulomb log note")
    random_seed: StrictInt | None = Field(default=None, ge=0, alias="Random Seed")
    bunched: StrictBool = Field(default=True, alias="Bunched")
    slice_set: str | None = Field(default=None, min_length=1, alias="Slice set")
    grid_shape: tuple[StrictInt, StrictInt, StrictInt] = Field(default=(8, 8, 8), alias="Grid shape")
    grid_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = Field(default=None, alias="Grid bounds (m)")
    collision_steps: StrictInt = Field(default=1, ge=1, alias="Collision steps")
    matching_tolerance: float = Field(default=0.1, gt=0, le=0.25, alias="Matching tolerance")
    max_scattering: float = Field(default=0.05, gt=0, le=0.05, alias="Max scattering")
    max_substeps: StrictInt = Field(default=1000, ge=1, alias="Max substeps")

    @field_validator("coulomb_log_note")
    @classmethod
    def _validate_coulomb_log_note(cls, value):
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("IBS Coulomb log note must be nonempty without surrounding whitespace")
        return value

    @field_validator("grid_shape", mode="before")
    @classmethod
    def _accept_json_grid_shape(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @field_validator("grid_shape")
    @classmethod
    def _validate_grid_shape(cls, value):
        if min(value) < 1:
            raise ValueError("IBS Grid shape requires three positive integers")
        return value

    @field_validator("grid_bounds", mode="before")
    @classmethod
    def _accept_json_grid_bounds(cls, value):
        if isinstance(value, (tuple, list)):
            return tuple(tuple(row) if isinstance(row, list) else row for row in value)
        return value

    @field_validator("grid_bounds")
    @classmethod
    def _validate_grid_bounds(cls, value):
        if value is not None and any(upper <= lower for lower, upper in zip(*value)):
            raise ValueError("IBS Grid bounds (m) requires upper bounds greater than lower bounds along every axis")
        return value

    @field_validator("slice_set")
    @classmethod
    def _validate_slice_name(cls, value):
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("IBS Slice set must be nonempty without surrounding whitespace")
        return value

    @model_validator(mode="after")
    def _validate_method_options(self):
        if self.slice_set is not None and self.method != "kinetic":
            raise ValueError("IBS Slice set is supported only by Method='kinetic'")
        if self.method != "binary" and self.grid_shape != (8, 8, 8):
            raise ValueError("IBS Grid shape is supported only by Method='binary'")
        if self.method != "binary" and self.grid_bounds is not None:
            raise ValueError("IBS Grid bounds (m) is supported only by Method='binary'")
        if self.method != "binary" and self.collision_steps != 1:
            raise ValueError("IBS Collision steps is supported only by Method='binary'")
        if self.method == "binary" and self.matching_tolerance != 0.1:
            raise ValueError("IBS Matching tolerance is supported only by Gaussian methods")
        return self


class IBSConfig(_IBSInput):
    """Top-level ``Intrabeam scattering`` block."""

    enabled: StrictBool = Field(default=False, alias="Enabled")
    configurations: dict[str, IBSConfiguration] = Field(default_factory=dict, alias="Configurations")

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
                raise ValueError("IBS configuration names must be nonempty without surrounding whitespace")
        return values


class IBSItem(_IBSInput):
    """Local IBS diagnostic or kick with a positive physical exposure length."""

    s: float = Field(default=0.0, alias="S (m)")
    order: StrictInt | None = Field(default=None, alias="Order")
    command: Literal["IBS"] = Field(default="IBS", alias="Command")
    configuration: str = Field(min_length=1, alias="Configuration")
    interaction_length: float = Field(ge=0, alias="Interaction length (m)")
    optics: IBSOpticsConfig | None = Field(default=None, alias="Optics")
    is_enabled: StrictBool = Field(default=True, alias="Is enabled")
    save_diagnostics: StrictBool = Field(default=False, alias="Save diagnostics")
    save_turns: list[list[StrictInt]] | list[StrictInt] = Field(default_factory=list, alias="Save turns")

    @field_validator("configuration")
    @classmethod
    def _validate_configuration_name(cls, value):
        if not value.strip() or value != value.strip():
            raise ValueError("IBS Configuration must be nonempty without surrounding whitespace")
        return value

    @field_validator("save_turns")
    @classmethod
    def _validate_save_turns(cls, value):
        if not value:
            return []
        items = [value] if all(isinstance(item, int) for item in value) else value
        for item in items:
            if len(item) not in {1, 3}:
                raise ValueError("IBS Save turns entries must be [turn] or [start, end, step]")
            if item[0] < 0 or len(item) == 3 and (item[1] < item[0] or item[2] <= 0):
                raise ValueError("IBS Save turns requires 0 <= start <= end and step > 0")
        return items


def _input_value(data, name, default=None):
    matches = [value for key, value in data.items() if str(key).casefold() == name.casefold()]
    if len(matches) > 1:
        raise ValueError(f"Duplicate IBS input field {name!r}")
    return matches[0] if matches else default


def load_intrabeam_scattering(data, *, validate_sequence=True):
    """Resolve IBS before recursive engine key folding changes named keys."""
    block = IBSConfig.model_validate(_input_value(data, "Intrabeam scattering", {}))
    if not block.enabled or not validate_sequence:
        return block
    sequence = _input_value(data, "Sequence", {})
    if not isinstance(sequence, dict):
        raise ValueError("IBS configuration resolution requires a Sequence object")
    for name, values in sequence.items():
        if not isinstance(values, dict) or str(_input_value(values, "Command", "")).casefold() != "ibs":
            continue
        normalized = IBSItem._normalize_fields(values)
        normalized["command"] = "IBS"
        item = IBSItem.model_validate(normalized)
        if not item.is_enabled:
            continue
        if item.configuration not in block.configurations:
            raise ValueError(f"IBS command {name!r} references undefined configuration {item.configuration!r}")
        model = block.configurations[item.configuration]
        if not model.bunched:
            for injection in sequence.values():
                if isinstance(injection, dict) and str(_input_value(injection, "Command", "")).casefold() == "injection":
                    if _input_value(injection, "Harmonic Number", 1) != 1:
                        raise ValueError("Coasting IBS requires Harmonic Number=1 and one bunch group representing the complete ring")
        if model.method in {"bjorken_mtingwa", "kinetic"} and item.optics is None:
            raise ValueError(f"IBS {model.method} command {name!r} requires Optics")
        if model.method == "binary" and item.optics is not None:
            raise ValueError(f"IBS binary command {name!r} does not use Optics; omit it")
    return block
