"""Schema for the longitudinal :class:`PASS.commands.slicer.Slicer` command."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from PASS.utils.coordinates import resolve_slice_coordinate


class SlicerItem(BaseModel):
    """Configure a named bunch-local slice set and optional snapshots."""

    model_config = ConfigDict(populate_by_name=True)

    s: float = Field(alias="S (m)")
    order: StrictInt | None = Field(default=None, alias="Order")
    command: str = Field(default="Slicer", alias="Command")
    output_format: Literal["tfs", "hdf5", "hdf5-gzip1"] = Field(default="hdf5-gzip1",
                                                                alias="Output format",
                                                                description="Particle details only; slice summaries remain TFS/CSV")
    slice_set: str = Field(alias="Slice set")
    purpose: Literal["general", "beam_beam"] = Field(default="general", alias="Purpose")
    configuration: str | None = Field(default=None, min_length=1, alias="Configuration")
    slice_model: str = Field(default="equal_length", alias="Slice model")
    num_slices: int = Field(default=10, ge=1, alias="Number of slices")
    z_range_mode: str = Field(default="auto", alias="Z range mode")
    explicit: dict | None = Field(default=None, alias="Explicit")
    periodic: StrictBool = Field(default=False, alias="Periodic")
    coordinate: str | None = Field(
        default=None,
        alias="Coordinate",
        description="z_rel (continuous), z_periodic (folded by circumference, required by SC), or arrival_phase (wake clock)")
    max_phase_slip: float = Field(default=0.05, gt=0, le=0.1, allow_inf_nan=False, alias="Max phase slip")
    save_turns: list[list[int]] = Field(default_factory=list, alias="Save turns")

    @model_validator(mode="after")
    def resolve_coordinate(self):
        self.coordinate = resolve_slice_coordinate(self.coordinate, self.periodic)
        self.periodic = self.coordinate == "arrival_phase"
        if self.purpose == "beam_beam":
            if self.configuration is None or self.coordinate not in {"z_rel", "collision_z"}:
                raise ValueError("Beam-beam Slicer requires Configuration and Coordinate=z_rel or collision_z")
        elif self.configuration is not None or self.coordinate == "collision_z":
            raise ValueError("Collision Configuration/Coordinate require Purpose=beam_beam")
        return self
