"""Schema for the longitudinal :class:`PASS.commands.slicer.Slicer` command."""

from pydantic import BaseModel, ConfigDict, Field


class Slicer(BaseModel):
    """Configure a named bunch-local slice set and optional snapshots."""

    model_config = ConfigDict(populate_by_name=True)

    s: float = Field(alias="S (m)")
    command: str = Field(default="Slicer", alias="Command")
    slice_set: str = Field(alias="Slice set")
    slice_model: str = Field(default="equal_length", alias="Slice model")
    num_slices: int = Field(default=10, ge=1, alias="Number of slices")
    z_range_mode: str = Field(default="auto", alias="Z range mode")
    explicit: dict | None = Field(default=None, alias="Explicit")
    save_turns: list[list[int]] = Field(default_factory=list, alias="Save turns")
