"""Monitor schemas.

Consumed by PASS.commands.monitor.* via Command.create(**kwargs).
"""

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class StatMonitorItem(BaseModel):
    """Statistic monitor: records bunch statistics at a position."""

    model_config = ConfigDict(populate_by_name=True)

    s: float = Field(alias="S (m)")
    command: str = Field(default="StatMonitor", alias="Command")


class DistMonitorItem(BaseModel):
    """Distribution monitor: saves full particle distribution at specified turns."""
    include_injection_metadata: bool = Field(default=False, alias="Include injection metadata")
    output_format: Literal["tfs", "hdf5"] = Field(default="tfs", alias="Output format")

    model_config = ConfigDict(populate_by_name=True)

    s: float = Field(alias="S (m)")
    command: str = Field(default="DistMonitor", alias="Command")
    save_turns: list[list[int]] = Field(
        default_factory=list,
        alias="Save turns",
        description=("Each item is [turn] or [start, end, step], with inclusive "
                     "endpoints; an empty list disables saving"),
    )


class PhaseAdvanceMonitorItem(BaseModel):
    """Measure uncoupled fractional tunes from consecutive-turn coordinates."""

    model_config = ConfigDict(populate_by_name=True)

    s: float = Field(alias="S (m)")
    command: str = Field(default="PhaseAdvanceMonitor", alias="Command")
    enable: bool = Field(default=True, alias="Enable")
    beta_x: float = Field(alias="Beta x (m)")
    beta_y: float = Field(alias="Beta y (m)")
    alpha_x: float = Field(alias="Alpha x")
    alpha_y: float = Field(alias="Alpha y")
    dx: float = Field(default=0.0, alias="Dx (m)")
    dpx: float = Field(default=0.0, alias="Dpx")
    x_co: float = Field(default=0.0, alias="X CO (m)")
    px_co: float = Field(default=0.0, alias="PX CO")
    y_co: float = Field(default=0.0, alias="Y CO (m)")
    py_co: float = Field(default=0.0, alias="PY CO")
    turn_ranges: list[list[int]] | int = Field(
        default=0,
        alias="Turn ranges",
        description=("Each item is [start, end) with zero-based turns; 0 disables analysis"),
    )
    min_action: float | None = Field(
        default=None,
        alias="Min action",
        description="Minimum normalized Courant-Snyder action; defaults by particle precision",
    )


class ParticleMonitorItem(BaseModel):
    """Particle monitor: records turn-by-turn coordinates of specified particles.

    Records particles with 1 <= |tag| <= max_tag every turn (within
    [start_turn, end_turn)) at the monitor's s-position.
    Each particle's TBT data is saved to a separate TFS file at the
    end of the simulation.
    """

    model_config = ConfigDict(populate_by_name=True)

    s: float = Field(alias="S (m)")
    command: str = Field(default="ParticleMonitor", alias="Command")
    include_reference: bool = Field(
        default=False,
        alias="Include reference",
        description="Append per-row reference time, beta and momentum; omit their storage by default",
    )
    max_tag: int = Field(
        alias="Max tag",
        description="Record particles with 1 <= |tag| <= max_tag",
    )
    start_turn: int = Field(
        default=0,
        alias="Start turn",
        description="Turn from which recording starts (inclusive, 0-based)",
    )
    end_turn: int = Field(
        default=-1,
        alias="End turn",
        description="Turn at which recording stops (exclusive, -1 means last turn inclusive)",
    )
