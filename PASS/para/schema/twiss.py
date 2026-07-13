"""Twiss transport point (TwissPoint).

Consumed by:
    - PASS.commands.twiss.Twiss.__init__

Each twiss point stores current + previous optical functions.
The engine computes the transfer matrix from (previous → current).
"""

from pydantic import BaseModel, Field, ConfigDict


class TwissPoint(BaseModel):
    """A single twiss transport point in the sequence."""

    model_config = ConfigDict(populate_by_name=True)

    # --- position ---
    s: float = Field(alias="S (m)")
    command: str = Field(default="Twiss", alias="Command")
    s_previous: float = Field(alias="S previous (m)")

    # --- current twiss ---
    alpha_x: float = Field(alias="Alpha x")
    alpha_y: float = Field(alias="Alpha y")
    beta_x: float = Field(alias="Beta x (m)")
    beta_y: float = Field(alias="Beta y (m)")
    mu_x: float = Field(alias="Mu x")
    mu_y: float = Field(alias="Mu y")
    mu_z: float = Field(default=0.0, alias="Mu z")
    dx: float = Field(alias="Dx (m)")
    dpx: float = Field(alias="Dpx")

    # --- previous twiss ---
    alpha_x_previous: float = Field(alias="Alpha x previous")
    alpha_y_previous: float = Field(alias="Alpha y previous")
    beta_x_previous: float = Field(alias="Beta x previous (m)")
    beta_y_previous: float = Field(alias="Beta y previous (m)")
    mu_x_previous: float = Field(alias="Mu x previous")
    mu_y_previous: float = Field(alias="Mu y previous")
    mu_z_previous: float = Field(default=0.0, alias="Mu z previous")
    dx_previous: float = Field(default=0.0, alias="Dx previous (m)")
    dpx_previous: float = Field(default=0.0, alias="Dpx previous")

    # --- chromaticity ---
    dqx: float = Field(default=0.0, alias="DQx")
    dqy: float = Field(default=0.0, alias="DQy")

    # --- longitudinal ---
    longitudinal_transfer: str = Field(
        default="off",
        alias="Longitudinal transfer",
        description="off / drift / matrix",
    )
