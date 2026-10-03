"""Slow-extraction actions and their event-based spill monitors."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator


class _SlowExtractionWindow(BaseModel):
    """Intersect half-open turn and physical-time windows when both are set."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    order: StrictInt | None = Field(default=None, alias="Order")
    s: float = Field(ge=0, allow_inf_nan=False, alias="S (m)")
    start_turn: StrictInt = Field(default=0, ge=0, alias="Start turn")
    end_turn: StrictInt | None = Field(default=None, ge=0, alias="End turn")
    start_time: float | None = Field(default=None, allow_inf_nan=False, alias="Start time (s)")
    end_time: float | None = Field(default=None, allow_inf_nan=False, alias="End time (s)")
    output_format: Literal["hdf5", "hdf5-gzip1"] = Field(default="hdf5-gzip1", alias="Output format")

    @model_validator(mode="after")
    def _validate_window(self):
        if self.end_turn is not None and self.end_turn <= self.start_turn:
            raise ValueError("End turn must be greater than Start turn; the end is exclusive")
        if self.start_time is not None and self.end_time is not None and self.end_time <= self.start_time:
            raise ValueError("End time (s) must be greater than Start time (s); the end is exclusive")
        return self


class SlowExtractionItem(_SlowExtractionWindow):
    """Collect and retire live particles on one side of a transverse plane."""

    command: Literal["SlowExtraction"] = Field(default="SlowExtraction", alias="Command")
    position: float = Field(allow_inf_nan=False, alias="Position (m)")
    side: Literal["positive", "negative"] = Field(default="positive", alias="Side")
    tilt: float = Field(default=0.0, allow_inf_nan=False, alias="Tilt (rad)")
    buffer_size: StrictInt = Field(default=65536, ge=1, alias="Buffer size (particles)")


class SlowExtractionMonitorItem(_SlowExtractionWindow):
    """Bin one source's extraction events by turn, physical time, or both."""

    command: Literal["SlowExtractionMonitor"] = Field(default="SlowExtractionMonitor", alias="Command")
    source: str = Field(min_length=1, alias="Source")
    bin_by: Literal["turn", "time", "both"] = Field(default="both", alias="Bin by")
    turn_bin_width: StrictInt = Field(default=1, ge=1, alias="Turn bin width")
    time_bin_width: float = Field(default=1.e-3, gt=0, allow_inf_nan=False, alias="Time bin width (s)")
    time_origin: float = Field(default=0.0, allow_inf_nan=False, alias="Time origin (s)")
    write_interval_turns: StrictInt = Field(default=100, ge=1, alias="Write interval (turns)")

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value):
        if not value.strip():
            raise ValueError("Source must be a nonempty sequence name")
        return value
