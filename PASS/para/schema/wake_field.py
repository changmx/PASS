"""WakeField sequence schema. Kernel units follow the chosen component order."""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator, field_validator


class WakeParameters(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def engine_keys(cls, data):
        if not isinstance(data, dict):
            return data
        aliases = {str(f.alias or name).lower(): f.alias or name for name, f in cls.model_fields.items()}
        names = {name.lower(): f.alias or name for name, f in cls.model_fields.items()}
        converted = {}
        for key, value in data.items():
            canonical = aliases.get(str(key).lower(), names.get(str(key).lower(), key))
            if canonical in converted:
                raise ValueError(f"Duplicate wake parameter {key!r}")
            converted[canonical] = value
        return converted


class ConstantWake(WakeParameters):
    kind: Literal["constant"] = Field(default="constant", alias="Kind")
    amplitude: float = Field(alias="Amplitude")
    duration: float = Field(gt=0, alias="Duration (s)")


class ResonatorWake(WakeParameters):
    kind: Literal["resonator"] = Field(default="resonator", alias="Kind")
    r: float = Field(gt=0, alias="R")
    q: float = Field(gt=0, alias="Q")
    frequency: float = Field(gt=0, alias="Frequency (Hz)")


class UltrarelativisticWallWake(WakeParameters):
    kind: Literal["ultrarelativistic_wall"] = Field(default="ultrarelativistic_wall", alias="Kind")
    radius: float = Field(gt=0, alias="Radius (m)")
    conductivity: float = Field(gt=0, alias="Conductivity (S/m)")
    length: float = Field(gt=0, alias="Length (m)")


class TabulatedWake(WakeParameters):
    kind: Literal["tabulated"] = Field(default="tabulated", alias="Kind")
    times: list[float] = Field(min_length=2, alias="Times (s)")
    values: list[float] = Field(min_length=2, alias="Values")
    causal: StrictBool = Field(default=True, alias="Causal")

    @model_validator(mode="after")
    def table(self):
        if (len(self.times) != len(self.values) or (self.causal and self.times[0] != 0)
                or any(b <= a for a, b in zip(self.times, self.times[1:]))):
            raise ValueError("Tabulated wake needs matching increasing times; causal tables start at zero")
        return self


class ImpedanceWake(WakeParameters):
    kind: Literal["impedance"] = Field(default="impedance", alias="Kind")
    frequencies: list[float] = Field(min_length=2, alias="Frequencies (Hz)")
    real: list[float] = Field(min_length=2, alias="Real")
    imag: list[float] = Field(min_length=2, alias="Imag")
    reconstruction: Literal["two_sided", "causal_projection"] = Field(alias="Reconstruction")

    @model_validator(mode="after")
    def samples(self):
        if (len(self.frequencies) != len(self.real) or len(self.real) != len(self.imag)
                or self.frequencies[0] < 0 or any(b <= a for a, b in zip(self.frequencies, self.frequencies[1:]))):
            raise ValueError("Impedance samples must match increasing nonnegative frequencies")
        return self


class FittedImpedanceWake(ImpedanceWake):
    kind: Literal["fitted_impedance"] = Field(default="fitted_impedance", alias="Kind")
    reconstruction: Literal["two_sided", "causal_projection"] = Field(default="causal_projection", alias="Reconstruction")
    initial_poles: list[tuple[float, float]] = Field(min_length=1, alias="Initial poles (1/s)")
    optimize_poles: StrictBool = Field(default=True, alias="Optimize poles")
    max_evaluations: Annotated[StrictInt, Field(ge=1)] = Field(default=500, alias="Max evaluations")
    relative_floor: float = Field(default=1e-3, gt=0, le=1, alias="Relative floor")
    tolerance: float = Field(default=.02, gt=0, alias="Fit tolerance")


class ModalWake(WakeParameters):
    kind: Literal["modes"] = Field(default="modes", alias="Kind")
    poles: list[tuple[float, float]] = Field(min_length=1, alias="Poles (1/s)")
    residues: list[tuple[float, float]] = Field(min_length=1, alias="Residues")


class ResistiveWallWake(WakeParameters):
    kind: Literal["resistive_wall"] = Field(default="resistive_wall", alias="Kind")
    radius: float = Field(gt=0, alias="Radius (m)")
    conductivity: float = Field(gt=0, alias="Conductivity (S/m)")
    length: float = Field(gt=0, alias="Length (m)")
    beta: float = Field(gt=0, le=1, alias="Beta")
    frequencies: list[float] = Field(min_length=2, alias="Frequencies (Hz)")
    wall_thickness: float | None = Field(default=None, gt=0, alias="Wall thickness (m)")
    skin_depth_ratio_max: float = Field(default=.1, gt=0, le=.1, alias="Max skin depth ratio")
    surface_impedance_ratio_max: float = Field(default=.1, gt=0, le=.1, alias="Max surface impedance ratio")


class WakeVelocity(WakeParameters):
    kind: Literal["fixed", "factorized", "ideal"] = Field(alias="Kind")
    beta: float | None = Field(default=None, gt=0, le=1, alias="Beta")
    betas: list[float] = Field(default_factory=list, alias="Betas")
    source: list[float] = Field(default_factory=list, alias="Source")
    witness: list[float] = Field(default_factory=list, alias="Witness")

    @model_validator(mode="after")
    def law(self):
        from PASS.commands.wake.wake_velocity import VelocityLaw
        if self.kind == "ideal":
            if self.beta is not None or self.betas or self.source or self.witness:
                raise ValueError("Ideal velocity-independent point coupling takes no beta table")
        else:
            VelocityLaw(**self.model_dump())
            if self.kind == "fixed" and (self.betas or self.source or self.witness):
                raise ValueError("Fixed velocity does not take a coupling table")
            if self.kind == "factorized" and self.beta is not None:
                raise ValueError("Factorized velocity takes beta knots, not a reference beta")
        return self


class WakeFileConvention(WakeParameters):
    data_kind: Literal["wake_function", "impedance"] = Field(alias="Data kind")
    axis: Literal["time", "distance", "frequency"] = Field(alias="Axis")
    axis_unit: str = Field(alias="Axis unit")
    value_unit: str = Field(alias="Value unit")
    positive_trailing: StrictBool = Field(alias="Positive trailing")
    longitudinal_positive_loss: StrictBool = Field(alias="Longitudinal positive loss")
    fourier_exponent: Literal[-1, 1] = Field(default=-1, alias="Fourier exponent")
    transverse_impedance_factor: Literal["i", "-i", "1"] = Field(default="i", alias="Transverse impedance factor")
    shunt_impedance_convention: str = Field(default="not_applicable", alias="Shunt impedance convention")
    integrated: StrictBool = Field(alias="Integrated")
    reference_beta: float = Field(gt=0, le=1, alias="Reference beta")


class FileWake(WakeParameters):
    kind: Literal["file"] = Field(default="file", alias="Kind")
    file_path: str = Field(min_length=1, alias="File path")
    format: Literal["table", "headtail"] = Field(default="table", alias="Format")
    convention: WakeFileConvention = Field(alias="Convention")
    axis_column: Annotated[StrictInt, Field(ge=0)] = Field(default=0, alias="Axis column")
    value_column: Annotated[StrictInt, Field(ge=0)] = Field(alias="Value column")
    imag_column: Annotated[StrictInt, Field(ge=0)] | None = Field(default=None, alias="Imag column")
    delimiter: str | None = Field(default=None, alias="Delimiter")
    skiprows: Annotated[StrictInt, Field(ge=0)] = Field(default=0, alias="Skip rows")
    causal: StrictBool = Field(default=True, alias="Causal")
    reconstruction: Literal["two_sided", "causal_projection"] = Field(default="two_sided", alias="Reconstruction")
    length: float | None = Field(default=None, gt=0, alias="Length (m)")


class WakeSpatialTerm(WakeParameters):
    plane: Literal["x", "y", "z"] = Field(alias="Plane")
    source_powers: tuple[Annotated[StrictInt, Field(ge=0)], Annotated[StrictInt, Field(ge=0)]] = Field(default=(0, 0), alias="Source powers")
    test_powers: tuple[Annotated[StrictInt, Field(ge=0)], Annotated[StrictInt, Field(ge=0)]] = Field(default=(0, 0), alias="Test powers")

    @model_validator(mode="after")
    def powers(self):
        from PASS.commands.wake.wake_components import SpatialTerm
        SpatialTerm(**self.model_dump())
        return self


WakeModelConfig = Annotated[Union[ConstantWake, ResonatorWake, ResistiveWallWake,
    UltrarelativisticWallWake, TabulatedWake, ImpedanceWake, FittedImpedanceWake,
    ModalWake, FileWake], Field(discriminator="kind")]
WakeComponentKind = Literal["longitudinal", "constant_x", "constant_y", "dipolar_x", "dipolar_y",
                            "dipolar_xy", "dipolar_yx", "quadrupolar_x", "quadrupolar_y",
                            "quadrupolar_xy", "quadrupolar_yx", "custom"]


class WakeComponentConfig(WakeParameters):
    component: WakeComponentKind = Field(alias="Component")
    model: WakeModelConfig = Field(alias="Model")
    spatial: WakeSpatialTerm | None = Field(default=None, alias="Spatial")
    scale: float = Field(default=1.0, alias="Scale")
    velocity: WakeVelocity | None = Field(default=None, alias="Velocity")
    field_content: Literal["wake", "finite_conductivity_correction", "pec_image", "direct_space_charge", "total"] = Field(default="wake", alias="Field content")

    @field_validator("model", mode="before")
    @classmethod
    def model_discriminator(cls, value):
        if isinstance(value, dict):
            # The engine recursively lowercases JSON keys before construction.
            value = dict(value)
            if "kind" in value and "Kind" not in value:
                value["Kind"] = value.pop("kind")
        return value

    @model_validator(mode="after")
    def wall_components(self):
        if (self.component == "custom") != (self.spatial is not None):
            raise ValueError("Custom components require Spatial; named components already define it")
        if self.model.kind in {"resistive_wall", "ultrarelativistic_wall"} and self.component not in {"longitudinal", "dipolar_x", "dipolar_y"}:
            raise ValueError("Round-wall model supports longitudinal, dipolar_x and dipolar_y only")
        if self.model.kind == "resistive_wall":
            if self.velocity is not None and (self.velocity.kind != "fixed" or self.velocity.beta != self.model.beta):
                raise ValueError("Finite-beta wall requires the same fixed beta as its spectrum")
            self.velocity = WakeVelocity(kind="fixed", beta=self.model.beta)
            if self.field_content not in {"wake", "finite_conductivity_correction"}:
                raise ValueError("Wall model contains only the finite-conductivity correction")
            self.field_content = "finite_conductivity_correction"
        elif self.velocity is None:
            raise ValueError("Specify a fixed, factorized, or explicitly ideal velocity law")
        return self


class WakeConvolutionGrid(WakeParameters):
    period: float = Field(gt=0, alias="Period (s)")
    slots: Annotated[StrictInt, Field(ge=1)] = Field(alias="Slots")
    slices: Annotated[StrictInt, Field(ge=1)] = Field(alias="Slices")
    slot_spacing: float = Field(gt=0, alias="Slot spacing (s)")
    slice_spacing: float = Field(gt=0, alias="Slice spacing (s)")
    origin: float = Field(default=0., alias="Origin (s)")
    width: float = Field(default=0., ge=0, alias="Width (s)")
    projection: Literal["exact", "linear"] = Field(default="exact", alias="Projection")

    @model_validator(mode="after")
    def geometry(self):
        from PASS.commands.wake.convolution import ConvolutionGrid
        ConvolutionGrid(**self.model_dump())
        return self


class WakeTimeGrid(WakeParameters):
    step: float = Field(gt=0, alias="Step (s)")
    block_size: Annotated[StrictInt, Field(ge=2)] = Field(default=64, alias="Block size")
    origin: float | None = Field(default=None, alias="Origin (s)")

    @model_validator(mode="after")
    def geometry(self):
        from PASS.commands.wake.time_convolution import TimeGrid
        TimeGrid(**self.model_dump())
        return self


class WakeSolverGroup(WakeParameters):
    name: str = Field(min_length=1, alias="Name")
    components: list[WakeComponentConfig] = Field(min_length=1, alias="Components")
    solver: Literal["direct", "fft", "recursive", "modal", "partitioned_fft", "time_fft"] = Field(alias="Solver")
    history: Literal["none", "direct", "state", "partitioned"] = Field(alias="History")
    convolution_grid: WakeConvolutionGrid | None = Field(default=None, alias="Convolution grid")
    time_grid: WakeTimeGrid | None = Field(default=None, alias="Time grid")
    partition: Literal["uniform", "dyadic"] | None = Field(default=None, alias="Partition")
    max_workspace_mb: float | None = Field(default=None, gt=0, alias="Max workspace (MiB)")
    source_shape: Literal["point", "uniform"] = Field(default="uniform", alias="Source shape")
    memory_turns: Annotated[StrictInt, Field(ge=1)] | None = Field(default=None, alias="Memory turns")
    memory_time: float | None = Field(default=None, gt=0, alias="Memory time (s)")
    boundary: Literal["causal_passages", "isolated", "periodic"] = Field(default="causal_passages", alias="Boundary")
    periodic_images: Annotated[StrictInt, Field(ge=1)] | None = Field(default=None, alias="Periodic images")
    period: float | None = Field(default=None, gt=0, alias="Period (s)")

    @model_validator(mode="after")
    def combinations(self):
        if self.history not in {"direct", "partitioned"} and (self.memory_turns is not None or self.memory_time is not None):
            raise ValueError("Memory cutoffs are only used with direct or partitioned history")
        if (self.solver in {"partitioned_fft", "time_fft"}) != (self.history == "partitioned"):
            raise ValueError("Partitioned FFT requires partitioned history")
        if self.solver == "partitioned_fft":
            if self.convolution_grid is None or self.memory_turns is None:
                raise ValueError("Partitioned FFT requires Convolution grid and finite Memory turns")
            if self.source_shape == "point" and self.convolution_grid.width != 0:
                raise ValueError("Point sources require zero convolution grid width")
            if self.source_shape == "uniform" and self.convolution_grid.width == 0:
                raise ValueError("Uniform sources require positive convolution grid width")
        elif self.convolution_grid is not None:
            raise ValueError("Convolution grid requires partitioned_fft")
        if self.solver == "time_fft":
            if self.time_grid is None or self.memory_time is None or self.memory_turns is not None:
                raise ValueError("time_fft requires Time grid and finite Memory time, without Memory turns")
        elif self.time_grid is not None:
            raise ValueError("Time grid requires time_fft")
        if self.solver not in {"partitioned_fft", "time_fft"} and (self.partition is not None or self.max_workspace_mb is not None):
            raise ValueError("Partition/workspace options require a partitioned history solver")
        if (self.solver in {"recursive", "modal"}) != (self.history == "state"):
            raise ValueError("Recursive/modal solvers require state history; direct/FFT use none or direct history")
        if self.solver == "recursive" and any(c.model.kind != "resonator" for c in self.components):
            raise ValueError("Recursive solver requires resonators")
        if self.solver == "modal" and any(c.model.kind not in {"modes", "fitted_impedance"} for c in self.components):
            raise ValueError("Modal solver requires explicit or fitted poles/residues")
        if self.boundary != "causal_passages" and self.history != "none":
            raise ValueError("Spatial isolated/periodic boundaries do not admit transient passage history")
        if self.boundary == "periodic":
            if self.periodic_images is None or self.period is None or self.solver != "direct":
                raise ValueError("Periodic response requires explicit period, image count and direct solver")
        elif self.periodic_images is not None or self.period is not None:
            raise ValueError("Periodic parameters require periodic boundary")
        for c in self.components:
            m = c.model
            two_sided = (m.kind == "resistive_wall" or m.kind == "tabulated" and not m.causal
                         or m.kind == "impedance" and m.reconstruction == "two_sided"
                         or m.kind == "file" and (m.convention.data_kind == "wake_function" and not m.causal
                             or m.convention.data_kind == "impedance" and m.reconstruction == "two_sided"))
            if two_sided and self.boundary == "causal_passages":
                raise ValueError("Two-sided responses require explicit isolated or periodic spatial boundary")
            if self.boundary != "causal_passages" and c.velocity.kind != "fixed":
                raise ValueError("Steady spatial response requires a fixed common beta")
        return self


class WakeResourceConfig(WakeParameters):
    """Reusable model and solver settings, never shared runtime history."""
    groups: list[WakeSolverGroup] = Field(min_length=1, alias="Groups")

    @model_validator(mode="after")
    def unique_groups(self):
        if len({g.name for g in self.groups}) != len(self.groups):
            raise ValueError("Wake solver group names must be unique at a physical location")
        return self


class WakeFieldConfig(WakeParameters):
    """Optional top-level 'Wake field' block."""
    enabled: StrictBool = Field(default=True, alias="Enabled")
    configurations: dict[str, WakeResourceConfig] = Field(default_factory=dict, alias="Configurations")

    @field_validator("configurations")
    @classmethod
    def configuration_names(cls, value):
        if any(not name.strip() or name != name.strip() for name in value):
            raise ValueError("Wake configuration names must be nonempty without surrounding whitespace")
        return value


class WakeField(WakeParameters):
    s: float = Field(ge=0, alias="S (m)")
    command: Literal["WakeField"] = Field(default="WakeField", alias="Command")
    slice_set: str = Field(min_length=1, alias="Slice set")
    groups: list[WakeSolverGroup] | None = Field(default=None, min_length=1, alias="Groups")
    configuration: str | None = Field(default=None, min_length=1, alias="Configuration")
    is_enabled: StrictBool = Field(default=True, alias="Is enabled")

    @field_validator("slice_set")
    @classmethod
    def slice_name(cls, value):
        if not value.strip() or value != value.strip():
            raise ValueError("Slice set must be nonempty without leading/trailing whitespace")
        return value

    @model_validator(mode="after")
    def combinations(self):
        if (self.groups is None) == (self.configuration is None):
            raise ValueError("Specify either inline Groups or a named Configuration, exclusively")
        if self.configuration is not None and self.configuration != self.configuration.strip():
            raise ValueError("Configuration must not have surrounding whitespace")
        if self.groups is not None:
            WakeResourceConfig(groups=self.groups)
        return self


def resolve_wake_point(point: dict, block: WakeFieldConfig | None = None) -> dict:
    """Return an independent inline command, keeping the input unmodified."""
    config = WakeField.model_validate(point)
    if config.configuration is not None:
        resources = {} if block is None else block.configurations
        if config.configuration not in resources:
            raise ValueError(f"Unknown Wake field Configuration: {config.configuration!r}")
        groups = resources[config.configuration].groups
    else:
        groups = config.groups
    result = config.model_dump(by_alias=True, mode="json", exclude={"configuration"})
    result["Groups"] = [g.model_dump(by_alias=True, mode="json") for g in groups]
    result["Is enabled"] = config.is_enabled and (block is None or block.enabled)
    return result


def expand_wake_configurations(data: dict) -> None:
    """Resolve input references before tracking; each command builds its own state.

    Work on a loader-owned input object. Existing inline-only input remains valid.
    Validate all candidates before changing the sequence.
    """
    block = None
    keys = {str(key).casefold(): key for key in data}
    if "wake field" in keys:
        block = WakeFieldConfig.model_validate(data[keys["wake field"]])
    sequence = data.get(keys.get("sequence"), {})
    replacements = {}
    for name, point in sequence.items():
        if not isinstance(point, dict):
            continue
        command_keys = [key for key in point if str(key).casefold() == "command"]
        if command_keys and str(point[command_keys[0]]).casefold() == "wakefield":
            values = dict(point)
            # Like the engine registry, accept the command type in any case.
            values[command_keys[0]] = "WakeField"
            replacements[name] = resolve_wake_point(values, block)
    sequence.update(replacements)
