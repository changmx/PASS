"""Element schemas for the PASS sequence.

Each element type has its own pydantic model with aliases matching
the JSON keys consumed by the corresponding engine Command class.

All elements share common fields (s, length, aperture) defined in ElementBase.
Specific element types add their own physical parameters.

Consumed by PASS.commands.element.* via Command.create(**kwargs).
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import ClassVar


class ElementBase(BaseModel):
    """Base model for all physical elements in the sequence.

    Subclasses set ``command`` to their registered Command name.
    """

    model_config = ConfigDict(populate_by_name=True)

    s: float = Field(alias="S (m)")
    command: str = Field(alias="Command")
    length: float = Field(default=0.0, ge=0, alias="Length (m)")

    # aperture (shared by all elements)
    aperture_type: str = Field(default="off", alias="Aperture type")
    aperture_value: list = Field(default_factory=list, alias="Aperture value")


# ============================================================
# Drift
# ============================================================

class DriftElement(ElementBase):
    command: str = Field(default="Drift", alias="Command")


# ============================================================
# Marker
# ============================================================

class MarkerElement(ElementBase):
    """Marker has no length or physical effect, only records position."""
    command: str = Field(default="Marker", alias="Command")
    length: float = Field(default=0.0, alias="Length (m)")


# ============================================================
# SBend (dipole)
# ============================================================

class SBendElement(ElementBase):
    command: str = Field(default="SBend", alias="Command")
    k0l: float = Field(default=0.0, alias="K0L")
    e1: float = Field(default=0.0, alias="E1 (rad)")
    e2: float = Field(default=0.0, alias="E2 (rad)")
    hgap: float = Field(default=0.0, alias="Hgap (m)")
    fint: float = Field(default=0.0, alias="Fint")
    fintx: float = Field(default=0.0, alias="Fintx")

    # field error
    is_field_error: bool = Field(default=False, alias="Is field error")
    field_error_knl: list[float] = Field(default_factory=list, alias="Field error KNL")
    field_error_ksl: list[float] = Field(default_factory=list, alias="Field error KSL")

    # ramping
    is_ramping: bool = Field(default=False, alias="Is ramping")
    k0l_ramping_file: str = Field(default="", alias="K0L ramping file")

    # slicing
    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


# ============================================================
# Quadrupole
# ============================================================

class QuadrupoleElement(ElementBase):
    command: str = Field(default="Quadrupole", alias="Command")
    k1l: float = Field(default=0.0, alias="K1L")
    k1sl: float = Field(default=0.0, alias="K1SL")

    # field error
    is_field_error: bool = Field(default=False, alias="Is field error")
    field_error_knl: list[float] = Field(default_factory=list, alias="Field error KNL")
    field_error_ksl: list[float] = Field(default_factory=list, alias="Field error KSL")

    # ramping
    is_ramping: bool = Field(default=False, alias="Is ramping")
    k1l_ramping_file: str = Field(default="", alias="K1L ramping file")
    k1sl_ramping_file: str = Field(default="", alias="K1SL ramping file")

    # slicing
    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


# ============================================================
# Sextupole
# ============================================================

class SextupoleElement(ElementBase):
    command: str = Field(default="Sextupole", alias="Command")
    k2l: float = Field(default=0.0, alias="K2L")
    k2sl: float = Field(default=0.0, alias="K2SL")

    is_field_error: bool = Field(default=False, alias="Is field error")
    field_error_knl: list[float] = Field(default_factory=list, alias="Field error KNL")
    field_error_ksl: list[float] = Field(default_factory=list, alias="Field error KSL")

    is_ramping: bool = Field(default=False, alias="Is ramping")
    k2l_ramping_file: str = Field(default="", alias="K2L ramping file")
    k2sl_ramping_file: str = Field(default="", alias="K2SL ramping file")

    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


# ============================================================
# Octupole
# ============================================================

class OctupoleElement(ElementBase):
    command: str = Field(default="Octupole", alias="Command")
    k3l: float = Field(default=0.0, alias="K3L")
    k3sl: float = Field(default=0.0, alias="K3SL")

    is_field_error: bool = Field(default=False, alias="Is field error")
    field_error_knl: list[float] = Field(default_factory=list, alias="Field error KNL")
    field_error_ksl: list[float] = Field(default_factory=list, alias="Field error KSL")

    is_ramping: bool = Field(default=False, alias="Is ramping")
    k3l_ramping_file: str = Field(default="", alias="K3L ramping file")
    k3sl_ramping_file: str = Field(default="", alias="K3SL ramping file")

    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


# ============================================================
# Multipole
# ============================================================

class MultipoleElement(ElementBase):
    command: str = Field(default="Multipole", alias="Command")
    knl: list[float] = Field(default_factory=list, alias="KiL")
    ksl: list[float] = Field(default_factory=list, alias="KiSL")

    is_ramping: bool = Field(default=False, alias="Is ramping")
    kl_ramping_file: str = Field(default="", alias="KL ramping file")

    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


# ============================================================
# Solenoid
# ============================================================

class SolenoidElement(ElementBase):
    command: str = Field(default="Solenoid", alias="Command")
    ks: float = Field(default=0.0, alias="KS")

    is_field_error: bool = Field(default=False, alias="Is field error")
    field_error_knl: list[float] = Field(default_factory=list, alias="Field error KNL")
    field_error_ksl: list[float] = Field(default_factory=list, alias="Field error KSL")

    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


# ============================================================
# Kicker
# ============================================================

class KickerElement(ElementBase):
    command: str = Field(default="Kicker", alias="Command")
    hkick: float = Field(default=0.0, alias="HKICK")
    vkick: float = Field(default=0.0, alias="VKICK")

    is_field_error: bool = Field(default=False, alias="Is field error")
    field_error_knl: list[float] = Field(default_factory=list, alias="Field error KNL")
    field_error_ksl: list[float] = Field(default_factory=list, alias="Field error KSL")

    is_ramping: bool = Field(default=False, alias="Is ramping")
    kick_ramping_file: str = Field(default="", alias="Kick ramping file")

    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


# ============================================================
# ElSeparator (electrostatic separator)
# ============================================================

class ElSeparatorElement(ElementBase):
    command: str = Field(default="ElSeparator", alias="Command")
    ex: float = Field(default=0.0, alias="EX (V/m)")
    ey: float = Field(default=0.0, alias="EY (V/m)")
    exl: float = Field(default=None, alias="EXL (V)")
    eyl: float = Field(default=None, alias="EYL (V)")
    tilt: float = Field(default=0.0, alias="Tilt (rad)")
    septum_x_position: float = Field(default=None, alias="Septum x position (m)")
    septum_y_position: float = Field(default=None, alias="Septum y position (m)")
    septum_thickness: float = Field(default=0.0, alias="Septum thickness (m)")


# ============================================================
# Exciter (tune exciter)
# ============================================================

class ExciterElement(ElementBase):
    command: str = Field(default="Exciter", alias="Command")
    is_enabled: bool = Field(default=True, alias="Enable")
    mode: str = Field(alias="Mode")
    direction: str = Field(alias="Direction")
    start_turn: int = Field(alias="Start turn")
    end_turn: int = Field(alias="End turn")
    voltage: float = Field(alias="Voltage (V)")
    gap: float = Field(alias="Gap (m)")
    plate_length: float = Field(alias="Plate length (m)")

    # frequency (two modes)
    excite_tune: float | None = Field(default=None, alias="Excite tune")
    sweep_tune: float | None = Field(default=None, alias="Sweep tune")
    central_frequency: float | None = Field(default=None, alias="Central frequency (Hz)")
    sweep_width: float | None = Field(default=None, alias="Sweep width (Hz)")

    period: float = Field(alias="Period (s)")
    fm_dual_frequency: float = Field(alias="FM dual frequency (Hz)")

    # AM parameters
    am_t_ext: float = Field(alias="AM t ext (s)")
    am_r0: float = Field(alias="AM r0 (m)")
    am_delta0: float = Field(alias="AM delta0")
    am_k_const: float = Field(alias="AM k const")


# ============================================================
# RFCavity
# ============================================================

class RFCavityElement(ElementBase):
    command: str = Field(default="RFCavity", alias="Command")
    voltage: float = Field(default=0.0, alias="Voltage (V)")
    harmonic: int = Field(default=1, alias="Harmonic")
    phase: float = Field(default=0.0, alias="Phase (rad)")
    phi_offset: float = Field(default=0.0, alias="Phi offset (rad)")
    rf_data_file: str | None = Field(default=None, alias="RF data file")
    is_enabled: bool = Field(default=True, alias="Is enabled")
    dp_aperture: list[float] | None = Field(default=None, alias="Dp aperture")


# ============================================================
# ReorganizeBunch (bunch index redistribution, no physical tracking)
# ============================================================

class ReorganizeBunchElement(ElementBase):
    """Reorganize bunch command: redistribute particle indices among bunches.

    Only modifies ``start_idx`` / ``end_idx`` of each bunch;
    does not touch any particle coordinates.
    """
    command: str = Field(default="ReorganizeBunch", alias="Command")
    mode: str = Field(default="merge", alias="Mode")
    start_turn: int = Field(default=0, alias="Start turn")
    end_turn: int = Field(default=-1, alias="End turn")
    new_num_bunch: int = Field(default=1, ge=1, alias="New num bunch")


# ============================================================
# Convenience registry
# ============================================================

ELEMENT_REGISTRY: dict[str, type[ElementBase]] = {
    "drift": DriftElement,
    "marker": MarkerElement,
    "sbend": SBendElement,
    "quadrupole": QuadrupoleElement,
    "sextupole": SextupoleElement,
    "octupole": OctupoleElement,
    "multipole": MultipoleElement,
    "solenoid": SolenoidElement,
    "kicker": KickerElement,
    "elseparator": ElSeparatorElement,
    "exciter": ExciterElement,
    "rfcavity": RFCavityElement,
    "reorganizebunch": ReorganizeBunchElement,
}
