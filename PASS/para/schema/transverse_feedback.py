"""Bunch-by-bunch transverse pickups, delayed FIR feedback, and filter design."""

import math
from numbers import Integral
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator


class _TransverseFeedbackInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)

    order: StrictInt | None = Field(default=None, alias="Order")
    s: float = Field(ge=0, alias="S (m)")

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
                raise ValueError(f"Duplicate transverse-feedback field {name!r}")
            normalized[name] = item
        return normalized


class TransversePickupItem(_TransverseFeedbackInput):
    """Measure live-particle centroids at one physical tracking boundary."""

    command: Literal["TransversePickup"] = Field(default="TransversePickup", alias="Command")
    plane: Literal["x", "y", "xy"] = Field(default="x", alias="Plane", description="Measured transverse planes; xy measures both independently.")
    reference_x: float = Field(default=0.0,
                               alias="Reference x (m)",
                               description="Fixed horizontal pickup reference subtracted from the centroid, in m.")
    reference_y: float = Field(default=0.0,
                               alias="Reference y (m)",
                               description="Fixed vertical pickup reference subtracted from the centroid, in m.")


class TransverseFeedbackItem(_TransverseFeedbackInput):
    """Apply a delayed, flat bunch kick from one named transverse pickup."""

    command: Literal["TransverseFeedback"] = Field(default="TransverseFeedback", alias="Command")
    pickup: str = Field(min_length=1, alias="Pickup", description="Exact TransversePickup command name in this beam's Sequence.")
    enabled: StrictBool = Field(default=True, alias="Enable")
    start_turn: StrictInt = Field(default=0, ge=0, alias="Start turn")
    end_turn: StrictInt | None = Field(default=None, ge=0, alias="End turn", description="Exclusive last turn; null continues to the run end.")
    delay_turns: StrictInt = Field(default=1, ge=1, alias="Delay turns", description="Sample n-d-k is used at turn n; d is at least one turn.")
    coefficients_x: list[float] | None = Field(default=None,
                                               min_length=1,
                                               alias="FIR coefficients x",
                                               description="Horizontal FIR coefficients a[k], newest delayed sample first.")
    coefficients_y: list[float] | None = Field(default=None,
                                               min_length=1,
                                               alias="FIR coefficients y",
                                               description="Vertical FIR coefficients a[k], newest delayed sample first.")
    gain_x: float = Field(default=0.0,
                          alias="Gain x (1/m)",
                          description="Horizontal gain G; the normalized momentum kick is -G times the filtered position.")
    gain_y: float = Field(default=0.0,
                          alias="Gain y (1/m)",
                          description="Vertical gain G; the normalized momentum kick is -G times the filtered position.")
    max_kick_x: float | None = Field(default=None,
                                     ge=0,
                                     alias="Max kick x",
                                     description="Limit on abs(delta px), dimensionless; null means unlimited.")
    max_kick_y: float | None = Field(default=None,
                                     ge=0,
                                     alias="Max kick y",
                                     description="Limit on abs(delta py), dimensionless; null means unlimited.")
    bunch_ids: list[Annotated[StrictInt, Field(ge=0)]] | None = Field(default=None,
                                                                      alias="Harmonic IDs",
                                                                      description="Unique grouping-slot IDs to kick; null selects every slot.")
    diagnostics_interval: StrictInt = Field(default=0,
                                            ge=0,
                                            alias="Diagnostics interval (turns)",
                                            description="Feedback diagnostic cadence; zero disables output.")

    @field_validator("pickup")
    @classmethod
    def _validate_pickup(cls, value):
        if not value.strip() or value != value.strip():
            raise ValueError("Pickup must be a nonempty sequence name without surrounding whitespace")
        return value

    @field_validator("bunch_ids")
    @classmethod
    def _validate_bunch_ids(cls, value):
        if value is not None and len(value) != len(set(value)):
            raise ValueError("Harmonic IDs must be unique")
        return value

    @model_validator(mode="after")
    def _validate_window(self):
        if self.end_turn is not None and self.end_turn <= self.start_turn:
            raise ValueError("End turn must be greater than Start turn; the end is exclusive")
        return self


def design_feedback_fir(tune, phase_advance, delay_turns=1, tap_count=5):
    """Return minimum-norm real taps with zero DC and unit target-tune response.

    ``phase_advance`` is the pickup-to-feedback phase in radians within the
    chosen turn numbering. For x[n]=cos(2*pi*Q*n+phi), the delayed response is
    exp(i*(phase_advance+pi/2)); a positive gain in delta_px=-G*filtered_x
    therefore opposes the coherent momentum at the feedback location.
    This narrow-band design does not establish closed-loop stability.
    """
    if not isinstance(delay_turns, Integral) or isinstance(delay_turns, (bool, np.bool_)) or delay_turns < 1:
        raise ValueError("delay_turns must be an integer of at least one")
    if not isinstance(tap_count, Integral) or isinstance(tap_count, (bool, np.bool_)) or tap_count < 3:
        raise ValueError("tap_count must be an integer of at least three")
    if isinstance(tune, (bool, np.bool_)) or isinstance(phase_advance, (bool, np.bool_)):
        raise ValueError("tune and phase_advance must be finite real numbers")
    try:
        tune, phase_advance = float(tune), float(phase_advance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tune and phase_advance must be finite real numbers") from exc
    if not math.isfinite(tune) or not math.isfinite(phase_advance):
        raise ValueError("tune and phase_advance must be finite real numbers")
    omega = 2.0 * np.pi * math.remainder(tune, 1.0)
    phase = np.arange(tap_count, dtype=np.float64) * omega
    matrix = np.vstack((np.ones(tap_count), np.cos(phase), np.sin(phase)))
    target_phase = math.remainder(phase_advance + np.pi / 2.0 + omega * delay_turns, 2.0 * np.pi)
    target = np.array([0.0, np.cos(target_phase), -np.sin(target_phase)])
    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(matrix, target, rcond=1.e-12)
    # Integer and half-integer tunes cannot provide three independent constraints.
    if rank != 3 or singular_values[-1] <= singular_values[0] / 1.e8:
        raise ValueError("FIR design is ill-conditioned near an integer or half-integer tune; choose another tune or tap count")
    error = np.max(np.abs(matrix @ coefficients - target))
    if not np.all(np.isfinite(coefficients)) or error > 5.e-10:
        raise ValueError(f"FIR design constraints could not be resolved accurately; maximum residual={error:.3e}")
    return coefficients.tolist()
