"""Monitor schemas.

Consumed by PASS.commands.monitor.* via Command.create(**kwargs).
"""

from pydantic import BaseModel, Field, ConfigDict


class StatMonitor(BaseModel):
    """Statistic monitor: records bunch statistics at a position."""

    model_config = ConfigDict(populate_by_name=True)

    s: float = Field(alias="S (m)")
    command: str = Field(default="StatMonitor", alias="Command")


class DistMonitor(BaseModel):
    """Distribution monitor: saves full particle distribution at specified turns."""

    model_config = ConfigDict(populate_by_name=True)

    s: float = Field(alias="S (m)")
    command: str = Field(default="DistMonitor", alias="Command")
    save_turns: list[list[int]] = Field(
        default_factory=list,
        alias="Save turns",
        description=(
            "Each item is [turn] or [start, end, step], with inclusive "
            "endpoints; an empty list disables saving"
        ),
    )


class PhaseMonitor(BaseModel):
    """Phase monitor: records phase advance."""

    model_config = ConfigDict(populate_by_name=True)

    s: float = Field(alias="S (m)")
    command: str = Field(default="PhaseMonitor", alias="Command")
    is_enabled: bool = Field(default=True, alias="Is enable phase monitor")
    beta_x: float = Field(alias="Beta x (m)")
    beta_y: float = Field(alias="Beta y (m)")
    alpha_x: float = Field(alias="Alpha x")
    alpha_y: float = Field(alias="Alpha y")
    save_turns: list[list[int]] = Field(
        default_factory=list,
        alias="Save turns",
    )


class ParticleMonitor(BaseModel):
    """Particle monitor: records turn-by-turn coordinates of specified particles.

    Records particles with 1 <= |tag| <= max_tag every turn (within
    [start_turn, end_turn)) at the monitor's s-position.
    Each particle's TBT data is saved to a separate TFS file at the
    end of the simulation.
    """

    model_config = ConfigDict(populate_by_name=True)

    s: float = Field(alias="S (m)")
    command: str = Field(default="ParticleMonitor", alias="Command")
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
