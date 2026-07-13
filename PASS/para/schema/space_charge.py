"""Space charge and other simulation module schemas."""

from pydantic import BaseModel, Field, ConfigDict


class SpaceChargeConfig(BaseModel):
    """Space-charge simulation parameters.

    Appears as a top-level key in the input JSON, not inside Sequence.
    """

    model_config = ConfigDict(populate_by_name=True)

    is_enabled: bool = Field(default=True, alias="Is enable space charge")
    num_slices: int = Field(default=100, ge=1, alias="Number of slices")
    slice_model: str = Field(default="Equal particle", alias="Slice model")
    field_solver: str = Field(default="PIC_FD_CUDSS", alias="Field solver")
