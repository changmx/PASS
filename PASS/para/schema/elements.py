"""Element schemas for the PASS sequence.

Each element type has its own pydantic model with aliases matching
the JSON keys consumed by the corresponding engine Command class.

All elements share common fields (s, length, aperture) defined in ElementBase.
Specific element types add their own physical parameters.

Consumed by PASS.commands.element.* via Command.create(**kwargs).
"""

from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import ClassVar, Literal
from pydantic import StrictInt
from PASS.para.schema.space_charge import ElementSpaceCharge


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

    @model_validator(mode="before")
    @classmethod
    def reject_unsupported_internal_sc(cls, value):
        if isinstance(value, dict) and "space_charge" not in cls.model_fields:
            if value.get("Space charge", value.get("space_charge")) is not None:
                raise ValueError(f"{cls.__name__} does not support internal Space charge")
        return value


class SlicedElementBase(ElementBase):
    """Body transport with optional internally scheduled space charge."""

    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    space_charge: ElementSpaceCharge | None = Field(default=None, alias="Space charge")

    @model_validator(mode="after")
    def validate_internal_sc_length(self):
        if self.space_charge is not None and self.length <= 0:
            raise ValueError("Internal Space charge requires a positive element length")
        return self


# ============================================================
# Drift
# ============================================================

class DriftElement(SlicedElementBase):
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

class SBendElement(SlicedElementBase):
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
    model: str = Field(default="adaptive", alias="Model")
    integrator: str = Field(default="adaptive", alias="Integrator")


# ============================================================
# Quadrupole
# ============================================================

class QuadrupoleElement(SlicedElementBase):
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

    # model
    model: str = Field(default="adaptive", alias="Model")

    # slicing
    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


# ============================================================
# Sextupole
# ============================================================

class SextupoleElement(SlicedElementBase):
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

class OctupoleElement(SlicedElementBase):
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

class MultipoleElement(SlicedElementBase):
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

class SolenoidElement(SlicedElementBase):
    command: str = Field(default="Solenoid", alias="Command")
    ks: float = Field(default=0.0, alias="KS")
    knl: list[float] = Field(default_factory=list, alias="KiL")
    ksl: list[float] = Field(default_factory=list, alias="KiSL")

    is_field_error: bool = Field(default=False, alias="Is field error")
    field_error_knl: list[float] = Field(default_factory=list, alias="Field error KNL")
    field_error_ksl: list[float] = Field(default_factory=list, alias="Field error KSL")

    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


# ============================================================
# Kicker
# ============================================================

class KickerElement(SlicedElementBase):
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


class BumpElement(SlicedElementBase):
    model_config = ConfigDict(populate_by_name=True, allow_inf_nan=False)
    command: str = Field(default="Bump", alias="Command")
    waveform_file: str = Field(alias="Waveform file", min_length=1)
    time_mode: Literal["reference", "particle"] = Field(default="particle", alias="Time mode")
    time_offset: float = Field(default=0.0, alias="Time offset (s)")
    enabled: bool = Field(default=True, alias="Enable")
    num_slices: StrictInt = Field(default=1, ge=1, alias="Num slices")


# ============================================================
# ElSeparator (electrostatic separator)
# ============================================================

class ElSeparatorElement(SlicedElementBase):
    """Infinite-height parallel electrodes, with a thick or thin electric kick."""
    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)
    command: str = Field(default="ElSeparator", alias="Command")
    voltage: float | None = Field(default=None, alias="V (V)",
        description="Signed septum-minus-high-voltage-electrode potential difference; specify exactly one of V and VL")
    voltage_length: float | None = Field(default=None, alias="VL (V m)",
        description="Signed longitudinal integral of the interplate voltage difference, in V m; integrated electric field is VL / Gap")
    gap: float = Field(gt=0, alias="Gap (m)")
    tilt: float = Field(default=0.0, alias="Tilt (rad)")
    septum_position: float = Field(alias="Septum position (m)")
    septum_thickness: float = Field(default=0.0, ge=0, alias="Septum thickness (m)")
    num_slices: StrictInt = Field(default=1, ge=1, alias="Num slices")

    @model_validator(mode="after")
    def validate_geometry(self):
        from PASS.para.schema.space_charge import validate_loss_aperture
        import math
        if not math.isfinite(self.s-self.length):
            raise ValueError("ElSeparator entrance S - Length must be finite")
        outer = self.septum_position + self.septum_thickness
        counter = outer + self.gap
        if not math.isfinite(counter) or counter <= outer:
            raise ValueError("Gap must resolve distinct finite electrode surfaces")
        if self.septum_thickness > 0 and outer <= self.septum_position:
            raise ValueError("Positive septum thickness must resolve distinct surfaces")
        if (self.voltage is None) == (self.voltage_length is None):
            raise ValueError("Specify exactly one of V (V) and VL (V m); zero is allowed")
        if self.voltage is not None:
            if self.length == 0 and self.voltage != 0:
                raise ValueError("Nonzero V requires positive Length; use VL for a zero-length kick")
            field = self.voltage / self.gap
            integrated_field = field * self.length
        else:
            integrated_field = self.voltage_length / self.gap
            field = integrated_field / self.length if self.length > 0 else 0.
        if not math.isfinite(field) or not math.isfinite(integrated_field):
            raise ValueError("Electric field and integrated electric field must be finite")
        kind = self.aperture_type.lower()
        self.aperture_type = {"circular": "circle", "elliptic": "ellipse", "rectangular": "rectangle"}.get(kind, kind)
        self.aperture_value = validate_loss_aperture(self.aperture_type, self.aperture_value)
        if self.aperture_type == "polygon":
            from PASS.validation.geometry import validate_polygon
            validate_polygon(self.aperture_value)
        return self


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

from PASS.para.schema.rf import RFComponent


class RFCavityElement(ElementBase):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)
    command: str = Field(default="RFCavity", alias="Command")
    components: list[RFComponent] = Field(min_length=1, alias="Components")
    is_enabled: bool = Field(default=True, alias="Is enabled")
    dp_aperture: list[float] | None = Field(default=None, alias="Dp aperture")

    @model_validator(mode="after")
    def validate_thin(self):
        if self.length != 0:
            raise ValueError("RFCavity is a zero-length effective-voltage kick")
        if self.dp_aperture is not None:
            if len(self.dp_aperture) != 2 or not self.dp_aperture[0] < self.dp_aperture[1]:
                raise ValueError("Dp aperture requires two ordered finite bounds")
        return self


# ============================================================
# ReorganizeBunch (bunch index redistribution, no physical tracking)
# ============================================================

class ReorganizeBunchElement(ElementBase):
    """Reorganize bunch command: switch to a new harmonic (bucket grid).

    All particles are sorted by longitudinal position, reassigned to the
    nearest new bucket center, and the beam harmonic number / bunch structure
    are updated to the new harmonic (one bunch per bucket).
    """
    command: str = Field(default="ReorganizeBunch", alias="Command")
    start_turn: int = Field(default=0, alias="Start turn")
    new_harmonic: int | None = Field(
        default=None, ge=1, alias="New harmonic number",
        description="Harmonic number after reorganization (bucket count). "
                    "Particles are re-sorted and assigned to the nearest new "
                    "bucket center; beam harmonic and bunch count are "
                    "updated accordingly.",
    )


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
    "bump": BumpElement,
    "elseparator": ElSeparatorElement,
    "exciter": ExciterElement,
    "rfcavity": RFCavityElement,
    "reorganizebunch": ReorganizeBunchElement,
}
