"""BeamBeam input models and joint validation, without allocating particles."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_serializer, model_validator


class _CollisionInput(BaseModel):
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
                raise ValueError(f"Duplicate field {name!r}")
            normalized[name] = item
        return normalized


class FrozenParameters(_CollisionInput):
    """An uncoupled prescribed transverse profile; no arbitrary covariance."""

    center_x: float = Field(default=0.0, alias="Center X (m)")
    center_y: float = Field(default=0.0, alias="Center Y (m)")
    angle: float = Field(default=0.0, alias="Angle (rad)")
    sigma: float | None = Field(default=None, gt=0, alias="Sigma (m)")
    sigma_x: float | None = Field(default=None, gt=0, alias="Sigma X (m)")
    sigma_y: float | None = Field(default=None, gt=0, alias="Sigma Y (m)")
    radius: float | None = Field(default=None, gt=0, alias="Radius (m)")
    a: float | None = Field(default=None, gt=0, alias="Semi-axis A (m)")
    b: float | None = Field(default=None, gt=0, alias="Semi-axis B (m)")
    sigma_delta: float = Field(default=0.0, ge=0, alias="Sigma delta")

    def validate_profile(self, solver):
        sizes = {"sigma", "sigma_x", "sigma_y", "radius", "a", "b"}
        if solver.startswith("gaussian_"):
            required = {"sigma"} if "round" in solver else {"sigma_x", "sigma_y"}
        else:
            required = {"radius"} if "round" in solver else {"a", "b"}
        supplied = {name for name in sizes if getattr(self, name) is not None}
        if supplied != required:
            raise ValueError(f"Frozen {solver} requires exactly these sizes: {sorted(required)}")
        if "round" in solver and self.angle != 0.0:
            raise ValueError("A round frozen profile has no orientation angle")


class BeamBeamSourceConfig(_CollisionInput):
    """A source description, or only a slice reference for a weak target."""

    slice_set: str = Field(min_length=1, alias="Slice set")
    method: Literal["pic", "frozen", "quasi-frozen"] | None = Field(default=None, alias="Method")
    solver: Literal["fft_free_space", "gaussian_round_free_space", "gaussian_ellipse_free_space", "uniform_round_free_space",
                    "uniform_ellipse_free_space", "parabolic_round_free_space", "parabolic_ellipse_free_space"] | None = Field(default=None,
                                                                                                                               alias="Solver")
    nx: StrictInt = Field(default=128, ge=3, alias="Nx")
    ny: StrictInt = Field(default=128, ge=3, alias="Ny")
    grid_half_width_x: float | None = Field(default=None, gt=0, alias="Grid Half Width X (m)")
    grid_half_width_y: float | None = Field(default=None, gt=0, alias="Grid Half Width Y (m)")
    deposition_method: Literal["CIC", "TSC"] = Field(default="TSC", alias="Particle Deposition Method")
    propagation_step: float | None = Field(default=None, gt=0, alias="Propagation step (m)")
    frozen_parameters: FrozenParameters | None = Field(default=None, alias="Frozen parameters")
    slice_parameters: dict[int | str, FrozenParameters] = Field(default_factory=dict, alias="Slice parameters")
    frozen_optics_reference: str | None = Field(default=None, min_length=1, alias="Frozen optics reference")
    source_center_slopes: list[float] = Field(default_factory=lambda: [0.0, 0.0], min_length=2, max_length=2, alias="Source center slopes")
    statistics_precision: Literal["float32", "float64"] | None = Field(default=None, alias="Statistics precision")

    @model_validator(mode="after")
    def _validate_source(self):
        if self.slice_set != self.slice_set.strip():
            raise ValueError("Slice set must not have surrounding whitespace")
        pic_fields = {"nx", "ny", "grid_half_width_x", "grid_half_width_y", "deposition_method", "propagation_step"}
        frozen_fields = {"frozen_parameters", "slice_parameters", "frozen_optics_reference", "source_center_slopes"}
        if self.method is None:
            if self.model_fields_set - {"slice_set", "method", "solver"} or self.solver is not None:
                raise ValueError("A target-only entry contains only Slice set")
            return self
        if self.solver is None or (self.method == "pic") != (self.solver == "fft_free_space"):
            raise ValueError("PIC requires fft_free_space; frozen/quasi-frozen require a free-space analytic solver")
        if self.method != "pic" and self.model_fields_set & pic_fields:
            raise ValueError("Grid and deposition parameters apply only to PIC")
        if self.method == "pic" and (self.grid_half_width_x is None or self.grid_half_width_y is None):
            raise ValueError("PIC requires both Grid Half Width X/Y (m)")
        if self.method == "pic" and self.propagation_step is None:
            raise ValueError("PIC requires an explicit positive Propagation step (m); automatic step selection is not supported")
        if self.method != "frozen" and self.model_fields_set & frozen_fields:
            raise ValueError("Prescribed profile/optics parameters apply only to frozen")
        if self.method == "frozen":
            if self.frozen_parameters is None:
                raise ValueError("Frozen requires Frozen parameters")
            self.frozen_parameters.validate_profile(self.solver)
            normalized = {}
            for index, parameters in self.slice_parameters.items():
                if type(index) is not int and not (isinstance(index, str) and index.isdecimal()):
                    raise ValueError("Slice parameters keys must be nonnegative integer indices")
                index = int(index)
                if index in normalized:
                    raise ValueError("Duplicate Slice parameters index")
                normalized[index] = parameters
            self.slice_parameters = normalized
            for index, parameters in self.slice_parameters.items():
                if index < 0:
                    raise ValueError("Slice parameters indices must be nonnegative")
                parameters.validate_profile(self.solver)
            if self.frozen_optics_reference is None and any(parameters.sigma_delta > 0
                                                            for parameters in [self.frozen_parameters, *self.slice_parameters.values()]):
                raise ValueError("Sigma delta requires Frozen optics reference to define its dispersion")
        return self

    @model_serializer(mode="wrap")
    def _serialize_source(self, handler, info):
        data = handler(self)
        names = {"slice_set"}
        if self.method is not None:
            names.update({"method", "solver", "statistics_precision"})
        if self.method == "pic":
            names.update({"nx", "ny", "grid_half_width_x", "grid_half_width_y", "deposition_method", "propagation_step"})
        if self.method == "frozen":
            names.update({"frozen_parameters", "slice_parameters", "frozen_optics_reference", "source_center_slopes"})
        keys = {(type(self).model_fields[name].alias if info.by_alias else name) for name in names}
        return {k: v for k, v in data.items() if k in keys and v is not None}


class BeamBeamLuminosityConfig(_CollisionInput):
    """Sample complete encounters into a small TFS or HDF5 table."""

    enabled: StrictBool = Field(default=False, alias="Enabled")
    sample_interval_turns: StrictInt = Field(default=100, ge=1, alias="Sample interval (turns)")
    output_format: Literal["tfs", "hdf5", "hdf5-gzip1"] = Field(default="tfs", alias="Output format")
    reference_luminosity: float | None = Field(default=None, gt=0, alias="Reference luminosity (cm^-2 s^-1)")
    collision_frequency: float | None = Field(default=None, gt=0, alias="Collision frequency (Hz)")

    @model_serializer(mode="wrap")
    def _serialize_output_format(self, handler, info):
        data = handler(self)
        # Preserve the configuration digest of existing TFS checkpoints.
        if self.output_format == "tfs":
            data.pop("Output format" if info.by_alias else "output_format", None)
        return data


class BeamBeamConfiguration(_CollisionInput):
    beams: list[StrictInt] = Field(default_factory=lambda: [0, 1], min_length=2, max_length=2, alias="Beams")
    mode: Literal["strong-strong", "weak-strong", "weak-weak"] = Field(default="strong-strong", alias="Mode")
    weak_beam: StrictInt | None = Field(default=None, alias="Weak beam")
    bunch_pairs: list[list[StrictInt]] | None = Field(default=None, min_length=1, alias="Bunch pairs")
    full_crossing_angle: float = Field(default=0.0, gt=-math.pi, lt=math.pi, alias="Full crossing angle (rad)")
    crossing_plane: float = Field(default=0.0, alias="Crossing plane (rad)")
    interaction_map: Literal["synchro_beam_6d"] = Field(default="synchro_beam_6d", alias="Interaction map")
    potential_reference_length: float = Field(default=1.0, gt=0, alias="Potential reference length (m)")
    sources: dict[str, BeamBeamSourceConfig] = Field(alias="Sources")
    luminosity: BeamBeamLuminosityConfig | None = Field(default=None, alias="Luminosity")

    @model_serializer(mode="wrap")
    def _serialize_luminosity(self, handler, info):
        data = handler(self)
        if self.luminosity is None:
            data.pop("Luminosity" if info.by_alias else "luminosity", None)
        return data

    @model_validator(mode="after")
    def _validate_pair(self):
        if self.beams != [0, 1]:
            raise ValueError("Beams must be [0, 1] in this two-beam implementation")
        if self.mode == "weak-strong":
            if self.weak_beam not in self.beams:
                raise ValueError("weak-strong requires Weak beam=0 or 1")
        elif self.weak_beam is not None:
            raise ValueError("Weak beam applies only to weak-strong")
        if set(self.sources) != {"0", "1"}:
            raise ValueError("Sources must contain exactly beam keys '0' and '1'")
        for pair in self.bunch_pairs or []:
            if len(pair) != 2 or any(index < 0 for index in pair):
                raise ValueError("Bunch pairs must contain nonnegative [bunch0, bunch1] indices")
        for side in (0, 1):
            if self.bunch_pairs is not None and len({pair[side] for pair in self.bunch_pairs}) != len(self.bunch_pairs):
                raise ValueError("Bunch pairs must be one-to-one in each beam")
            emits = self.mode == "strong-strong" or self.mode == "weak-strong" and side != self.weak_beam
            if emits and self.sources[str(side)].method is None:
                raise ValueError(f"Beam {side} emits a field and requires Method/Solver")
        if self.luminosity is not None and self.luminosity.enabled and all(source.method is None for source in self.sources.values()):
            raise ValueError("Luminosity requires at least one explicit source Method/Solver, including weak-weak reference runs")
        return self


class BeamBeamConfig(_CollisionInput):
    """One input's block. Absent Enabled has no vote in the shared switch."""

    enabled: StrictBool | None = Field(default=None, alias="Enabled")
    configurations: dict[str, BeamBeamConfiguration] = Field(default_factory=dict, alias="Configurations")

    @model_serializer(mode="wrap")
    def _serialize_switch(self, handler, info):
        data = handler(self)
        if self.enabled is None:
            data.pop("Enabled" if info.by_alias else "enabled", None)
        return data

    @model_validator(mode="after")
    def _validate_names(self):
        if "enabled" in self.model_fields_set and self.enabled is None:
            raise ValueError("Explicit Enabled must be true or false, not null")
        for name in self.configurations:
            if not name.strip() or name != name.strip():
                raise ValueError("BeamBeam configuration names must be nonempty without surrounding whitespace")
        return self


class BeamBeamItem(_CollisionInput):
    s: float = Field(alias="S (m)")
    order: StrictInt | None = Field(default=None, alias="Order")
    command: Literal["BeamBeam"] = Field(default="BeamBeam", alias="Command")
    configuration: str = Field(min_length=1, alias="Configuration")


def _value(data, name, default=None):
    if not isinstance(data, dict):
        raise ValueError(f"The object containing {name!r} must be a mapping")
    return next((value for key, value in data.items() if str(key).casefold() == name.casefold()), default)


def load_beam_beam(inputs, *, validate_sequences=True):
    """Merge the shared switch and named configurations before key folding."""
    enabled_values, configurations = [], {}
    for data in inputs:
        if _value(data, "Is beam-beam", False):
            raise ValueError("Is beam-beam is obsolete; use Beam beam.Enabled and BeamBeam commands")
        block = BeamBeamConfig.model_validate(_value(data, "Beam beam", {}))
        if block.enabled is not None:
            enabled_values.append(block.enabled)
        for name, config in block.configurations.items():
            if name in configurations:
                raise ValueError(f"BeamBeam configuration {name!r} is declared more than once")
            configurations[name] = config
    if enabled_values and any(value != enabled_values[0] for value in enabled_values):
        raise ValueError("Both inputs must agree on explicitly supplied Beam beam.Enabled")
    enabled = bool(enabled_values and enabled_values[0])
    if enabled:
        if len(inputs) != 2:
            raise ValueError("Enabled BeamBeam requires two beam input files")
        if not configurations:
            raise ValueError("Enabled BeamBeam requires at least one configuration")
        counts = [int(_value(_value(_value(data, "Sequence", {}), "injection", {}), "Harmonic Number", 1)) for data in inputs]
        if counts[0] != counts[1]:
            raise ValueError("Fixed BeamBeam pairing requires equal bunch counts in both beams")
        pairings = set()
        for config in configurations.values():
            pairs = config.bunch_pairs if config.bunch_pairs is not None else [[index, index] for index in range(counts[0])]
            if any({pair[side] for pair in pairs} != set(range(counts[side])) for side in (0, 1)):
                raise ValueError("Bunch pairs must be a complete bijection of the declared bunches")
            pairings.add(tuple(sorted(tuple(pair) for pair in pairs)))
        if len(pairings) != 1:
            raise ValueError("All IPs must use the same fixed bunch pairing")
        for field, default in (("Backend (GPU/CPU)", "cpu"), ("Particle Precision", "float64"), ("Number of turns", 0)):
            values = [_value(data, field, default) for data in inputs]
            if values[0] != values[1]:
                raise ValueError(f"Both BeamBeam inputs must agree on {field}")
        if validate_sequences:
            _validate_sequences(inputs, configurations)
    return enabled, configurations


def _validate_sequences(inputs, configurations):
    from PASS.utils.command_order import sort_commands
    from PASS.utils.constants import const

    plans = []
    for beam_id, data in enumerate(inputs):
        sequence = _value(data, "Sequence", {})
        entries = sort_commands(list(sequence.items()), key=lambda row: row[1])
        plan, frame, frame_s, slices = [], None, None, {}
        for name, raw in entries:
            kind = str(_value(raw, "Command", "")).casefold()
            ref = _value(raw, "Configuration")
            if kind == "injection":
                n_bunches = int(_value(raw, "Harmonic Number", 1))
                for config in configurations.values():
                    if any(pair[beam_id] >= n_bunches for pair in config.bunch_pairs or []):
                        raise ValueError("BeamBeam bunch index exceeds the Injection bunch count")
            if kind in {"beambeam", "crossingangle"}:
                if ref not in configurations:
                    raise ValueError(f"{name}: undefined BeamBeam Configuration {ref!r}")
            if kind == "crossingangle":
                direction = str(_value(raw, "Direction", "forward")).casefold()
                if direction == "forward":
                    if frame is not None:
                        raise ValueError(f"{name}: nested CrossingAngle frames are not allowed")
                    frame = ref
                    frame_s = float(_value(raw, "S (m)", 0))
                elif direction == "inverse":
                    if frame != ref:
                        raise ValueError(f"{name}: inverse CrossingAngle does not match the open frame")
                    if abs(float(_value(raw, "S (m)", 0)) - frame_s) > const.eps:
                        raise ValueError(f"{name}: CrossingAngle forward/inverse must be at the same IP position")
                    frame = None
                    slices.clear()
                else:
                    raise ValueError(f"{name}: Direction must be forward or inverse")
            elif frame is not None and kind not in {"slicer", "beambeam"}:
                raise ValueError(f"{name}: only collision Slicer and BeamBeam are allowed inside a CrossingAngle frame")
            if kind in {"sortbunch", "reorganizebunch"}:
                slices.clear()
                if kind == "reorganizebunch":
                    raise ValueError("ReorganizeBunch is incompatible with fixed BeamBeam bunch pairing")
            if kind == "slicer":
                if frame is not None and (_value(raw, "Purpose", "general") != "beam_beam" or ref != frame):
                    raise ValueError(f"{name}: the collision-frame Slicer must have matching Purpose/Configuration")
                slices[_value(raw, "Slice set")] = (raw, frame)
            if kind != "beambeam":
                continue
            config = configurations[ref]
            if config.full_crossing_angle != 0 and frame != ref:
                raise ValueError(f"{name}: nonzero crossing angle requires a matching CrossingAngle frame")
            if frame is not None and frame != ref:
                raise ValueError(f"{name}: BeamBeam and collision frame configurations differ")
            if frame is not None and abs(float(_value(raw, "S (m)", 0)) - frame_s) > const.eps:
                raise ValueError(f"{name}: CrossingAngle and BeamBeam must be at the same IP position")
            source = config.sources[str(beam_id)]
            if source.slice_set not in slices:
                raise ValueError(f"{name}: run Slicer for {source.slice_set!r} before BeamBeam")
            sliced, slice_frame = slices[source.slice_set]
            coordinate = _value(sliced, "Coordinate", "z_rel")
            if (_value(sliced, "Purpose", "general") != "beam_beam" or _value(sliced, "Configuration") != ref or slice_frame != frame
                    or coordinate != ("collision_z" if frame else "z_rel")):
                raise ValueError(f"{name}: Slicer Purpose, Configuration and Coordinate must match this collision")
            if abs(float(_value(sliced, "S (m)", 0)) - float(_value(raw, "S (m)", 0))) > const.eps:
                raise ValueError(f"{name}: Slicer and BeamBeam must be at the same lattice position")
            if source.method == "frozen" and source.frozen_optics_reference is not None:
                optics = next((item for key, item in sequence.items() if key.casefold() == source.frozen_optics_reference.casefold()), None)
                if optics is None or str(_value(optics, "Command", "")).casefold() != "twiss":
                    raise ValueError(f"{name}: Frozen optics reference must identify an existing Twiss entry")
                if abs(float(_value(optics, "S (m)", 0)) - float(_value(raw, "S (m)", 0))) > const.eps:
                    raise ValueError(f"{name}: Frozen optics reference must be at the collision IP")
            if source.method == "frozen":
                if any(index >= int(_value(sliced, "Number of slices", 10)) for index in source.slice_parameters):
                    raise ValueError(f"{name}: Slice parameters index exceeds Number of slices")
            plan.append(ref)
            slices.pop(source.slice_set)
        if frame is not None:
            raise ValueError(f"Beam {beam_id} leaves a CrossingAngle frame open")
        if set(plan) != set(configurations):
            raise ValueError(f"Beam {beam_id} must contain BeamBeam commands for all configurations")
        plans.append(plan)
    if plans[0] != plans[1]:
        raise ValueError("Both beams must encounter IP configurations in the same order")
