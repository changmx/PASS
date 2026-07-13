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
        description="Each sub-list is [start, stop, step] or [turn1, turn2, ...]",
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
    """Particle monitor: saves specified particles at an observer position."""

    model_config = ConfigDict(populate_by_name=True)

    s: float = Field(alias="S (m)")
    command: str = Field(default="ParticleMonitor", alias="Command")
    observer_id: int = Field(alias="Observer Id")
