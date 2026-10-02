"""Element schemas for the PASS sequence.

Each element type has its own pydantic model with aliases matching
the JSON keys consumed by the corresponding engine Command class.

All elements share common fields (s, length, aperture) defined in ElementBase.
Specific element types add their own physical parameters.

Consumed by PASS.commands.element.* via Command.create(**kwargs).
"""

import math
from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator, StrictBool, StrictInt

from PASS.para.schema.space_charge import ElementSpaceCharge
from PASS.utils.constants import const


class ElementBase(BaseModel):
    """Base model for all physical elements in the sequence.

    Subclasses set ``command`` to their registered Command name.
    """

    model_config = ConfigDict(populate_by_name=True)

    s: float = Field(alias="S (m)")
    command: str = Field(alias="Command")
    length: float = Field(default=0.0, ge=0, alias="Length (m)")
    order: StrictInt | None = Field(default=None, alias="Order")

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

    @model_validator(mode="before")
    @classmethod
    def reject_alignment_on_unsupported_elements(cls, value):
        if isinstance(value, dict) and "is_alignment_error" not in cls.model_fields:
            keys = {
                "is alignment error", "is_alignment_error", "alignment dx (m)", "alignment_dx", "alignment dy (m)", "alignment_dy",
                "alignment dpsi (rad)", "alignment_dpsi", "alignment ds (m)", "alignment_ds", "alignment dphi (rad)", "alignment_dphi",
                "alignment dtheta (rad)", "alignment_dtheta"
            }
            if any(v for k, v in value.items() if str(k).lower() in keys):
                raise ValueError(f"{cls.__name__} does not support alignment errors")
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


class MagneticElementBase(SlicedElementBase):
    """Static magnetic alignment and absolute integrated multipole errors."""
    is_field_error: bool = Field(default=False, alias="Is field error")
    field_error_knl: list[float] = Field(default_factory=list, alias="Field error KNL")
    field_error_ksl: list[float] = Field(default_factory=list, alias="Field error KSL")
    is_alignment_error: bool = Field(default=False, alias="Is alignment error")
    alignment_dx: float = Field(default=0.0, alias="Alignment DX (m)", allow_inf_nan=False)
    alignment_dy: float = Field(default=0.0, alias="Alignment DY (m)", allow_inf_nan=False)
    alignment_dpsi: float = Field(default=0.0, alias="Alignment DPSI (rad)", allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def reject_unsupported_alignment(cls, value):
        if isinstance(value, dict):
            values = {str(k).lower(): v for k, v in value.items()}
            for name, unit in (("ds", "m"), ("dphi", "rad"), ("dtheta", "rad")):
                for key in (f"alignment {name} ({unit})", f"alignment_{name}", name):
                    if float(values.get(key, 0.0)) != 0.0:
                        raise ValueError(f"Nonzero alignment {name.upper()} is not supported; use DX, DY and DPSI only")
        return value

    @field_validator("field_error_knl", "field_error_ksl")
    @classmethod
    def validate_field_error_coefficients(cls, values):
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Field-error coefficients must be finite")
        return values


class RampingMagneticElementBase(MagneticElementBase):
    """Time programs of normalized strengths for straight multipole magnets."""

    is_ramping: bool = Field(default=False, alias="Is ramping")
    ramping_file: str = Field(default="", alias="Ramping file")

    @model_validator(mode="after")
    def validate_ramping_sources(self):
        if self.is_ramping:
            from PASS.utils.magnet_program import _ramping_sources
            order = {"QuadrupoleItem": 1, "SextupoleItem": 2, "OctupoleItem": 3}.get(type(self).__name__)
            _ramping_sources(self.model_dump(by_alias=True), order=order)
        return self


# Drift


class DriftItem(SlicedElementBase):
    command: str = Field(default="Drift", alias="Command")


# Marker


class MarkerItem(ElementBase):
    """Marker has no length or physical effect, only records position."""
    command: str = Field(default="Marker", alias="Command")
    length: float = Field(default=0.0, alias="Length (m)")


# SBend (dipole)


class SBendItem(MagneticElementBase):
    command: str = Field(default="SBend", alias="Command")
    k0l: float = Field(default=0.0, alias="K0L")
    e1: float = Field(default=0.0, alias="E1 (rad)")
    e2: float = Field(default=0.0, alias="E2 (rad)")
    hgap: float = Field(default=0.0, alias="Hgap (m)")
    fint: float = Field(default=0.0, alias="Fint")
    fintx: float = Field(default=0.0, alias="Fintx")

    # ramping
    is_ramping: bool = Field(default=False, alias="Is ramping")
    k0l_ramping_file: str = Field(default="", alias="K0L ramping file")

    # slicing
    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    model: str = Field(default="adaptive", alias="Model")
    integrator: str = Field(default="adaptive", alias="Integrator")


# Quadrupole


class QuadrupoleItem(RampingMagneticElementBase):
    command: str = Field(default="Quadrupole", alias="Command")
    k1l: float = Field(default=0.0, alias="K1L")
    k1sl: float = Field(default=0.0, alias="K1SL")

    # ramping
    is_ramping: bool = Field(default=False, alias="Is ramping")
    k1l_ramping_file: str = Field(default="", alias="K1L ramping file")
    k1sl_ramping_file: str = Field(default="", alias="K1SL ramping file")

    # model
    model: str = Field(default="adaptive", alias="Model")

    # slicing
    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


# Sextupole


class SextupoleItem(RampingMagneticElementBase):
    command: str = Field(default="Sextupole", alias="Command")
    k2l: float = Field(default=0.0, alias="K2L")
    k2sl: float = Field(default=0.0, alias="K2SL")

    is_ramping: bool = Field(default=False, alias="Is ramping")
    k2l_ramping_file: str = Field(default="", alias="K2L ramping file")
    k2sl_ramping_file: str = Field(default="", alias="K2SL ramping file")

    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


# Octupole


class OctupoleItem(RampingMagneticElementBase):
    command: str = Field(default="Octupole", alias="Command")
    k3l: float = Field(default=0.0, alias="K3L")
    k3sl: float = Field(default=0.0, alias="K3SL")

    is_ramping: bool = Field(default=False, alias="Is ramping")
    k3l_ramping_file: str = Field(default="", alias="K3L ramping file")
    k3sl_ramping_file: str = Field(default="", alias="K3SL ramping file")

    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


# Multipole


class MultipoleItem(RampingMagneticElementBase):
    command: str = Field(default="Multipole", alias="Command")
    knl: list[float] = Field(default_factory=list, alias="KiL")
    ksl: list[float] = Field(default_factory=list, alias="KiSL")

    is_ramping: bool = Field(default=False, alias="Is ramping")
    kl_ramping_file: str = Field(default="", alias="KL ramping file")

    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


# Solenoid


class SolenoidItem(MagneticElementBase):
    command: str = Field(default="Solenoid", alias="Command")
    ks: float = Field(default=0.0, alias="KS")
    knl: list[float] = Field(default_factory=list, alias="KiL")
    ksl: list[float] = Field(default_factory=list, alias="KiSL")

    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


def _normalize_cooler_fields(model, value):
    """Accept generated aliases and the engine's case-normalized dictionaries."""
    if not isinstance(value, dict):
        return value
    names = {}
    for name, field in model.model_fields.items():
        names[name.casefold()] = name
        names[(field.alias or name).casefold()] = name
    normalized = {}
    for key, item in value.items():
        name = names.get(str(key).casefold(), key)
        if name in normalized:
            raise ValueError(f"Duplicate electron-cooler field {name!r}")
        if name == "command" and isinstance(item, str) and item.casefold() == "electroncooler":
            item = "ElectronCooler"
        normalized[name] = item
    return normalized


class ElectronBeamConfig(BaseModel):
    """Prescribed electron reservoir; temperatures and covariance are in its rest frame."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)

    kinetic_energy: float = Field(gt=0, alias="Kinetic energy (eV)")
    profile: Literal["uniform_round", "gaussian"] = Field(default="uniform_round", alias="Profile")
    radius: float | None = Field(default=None, gt=0, alias="Radius (m)")
    radius_exit: float | None = Field(default=None, gt=0, alias="Exit radius (m)")
    sigma_x: float | None = Field(default=None, gt=0, alias="Sigma x (m)")
    sigma_y: float | None = Field(default=None, gt=0, alias="Sigma y (m)")
    sigma_x_exit: float | None = Field(default=None, gt=0, alias="Exit sigma x (m)")
    sigma_y_exit: float | None = Field(default=None, gt=0, alias="Exit sigma y (m)")
    mode: Literal["dc", "gaussian_bunch"] = Field(default="dc", alias="Mode")
    current: float | None = Field(default=None, ge=0, alias="Current (A)")
    bunch_charge: float | None = Field(default=None, ge=0, alias="Bunch charge (C)")
    sigma_time: float | None = Field(default=None, gt=0, alias="Sigma time (s)")
    repetition_frequency: float | None = Field(default=None, gt=0, alias="Repetition frequency (Hz)")
    bunch_center_time: float = Field(default=0.0, alias="Bunch center time (s)")
    center_x: float = Field(default=0.0, alias="Center x (m)")
    center_y: float = Field(default=0.0, alias="Center y (m)")
    angle_x: float = Field(default=0.0, gt=-math.pi / 2, lt=math.pi / 2, alias="Angle x (rad)")
    angle_y: float = Field(default=0.0, gt=-math.pi / 2, lt=math.pi / 2, alias="Angle y (rad)")
    temperature_transverse: float | None = Field(default=None, gt=0, alias="Transverse temperature (eV)")
    temperature_longitudinal: float | None = Field(default=None, gt=0, alias="Longitudinal temperature (eV)")
    velocity_covariance: list[list[float]] | None = Field(default=None, alias="Velocity covariance (m2/s2)")
    velocity_gradient: list[list[float]] | None = Field(default=None, alias="Velocity gradient (1/s)")

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, value):
        return _normalize_cooler_fields(cls, value)

    @field_validator("velocity_covariance")
    @classmethod
    def _validate_velocity_covariance(cls, value):
        if value is None:
            return value
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("Electron velocity covariance requires a finite 3 by 3 matrix")
        scale = float(np.max(np.abs(matrix)))
        if scale == 0 or not np.allclose(matrix / scale, matrix.T / scale, rtol=0, atol=1.e-12):
            raise ValueError("Electron velocity covariance must be symmetric positive definite")
        if np.min(np.linalg.eigvalsh(matrix / scale)) <= 0:
            raise ValueError("Electron velocity covariance must be positive definite")
        return value

    @field_validator("velocity_gradient")
    @classmethod
    def _validate_velocity_gradient(cls, value):
        if value is not None:
            matrix = np.asarray(value, dtype=float)
            if matrix.shape != (3, 2) or not np.all(np.isfinite(matrix)):
                raise ValueError("Electron velocity gradient requires a finite 3 by 2 matrix")
        return value

    @model_validator(mode="after")
    def _validate_electron_beam(self):
        if self.profile == "uniform_round":
            if self.radius is None:
                raise ValueError("uniform_round electron beams require Radius (m)")
            if any(value is not None for value in (self.sigma_x, self.sigma_y, self.sigma_x_exit, self.sigma_y_exit)):
                raise ValueError("uniform_round electron beams use Radius (m), not Gaussian widths")
        else:
            if self.sigma_x is None or self.sigma_y is None:
                raise ValueError("gaussian electron beams require Sigma x (m) and Sigma y (m)")
            if self.radius is not None or self.radius_exit is not None:
                raise ValueError("gaussian electron beams use Gaussian widths, not Radius (m)")
        if self.mode == "dc":
            if self.current is None:
                raise ValueError("DC electron beams require Current (A)")
            if any(value is not None for value in (self.bunch_charge, self.sigma_time, self.repetition_frequency)) or self.bunch_center_time != 0:
                raise ValueError("DC electron beams do not use Gaussian-bunch charge or timing fields")
        else:
            if self.bunch_charge is None or self.sigma_time is None:
                raise ValueError("Gaussian electron bunches require Bunch charge (C) and Sigma time (s)")
            if self.current is not None:
                raise ValueError("Gaussian electron bunches use Bunch charge (C), not Current (A)")
        temperatures = (self.temperature_transverse, self.temperature_longitudinal)
        if self.velocity_covariance is None:
            if any(value is None for value in temperatures):
                raise ValueError("Supply both electron temperatures or a full Velocity covariance (m2/s2)")
        elif any(value is not None for value in temperatures):
            raise ValueError("Electron temperatures and Velocity covariance (m2/s2) are alternative inputs")
        if self.velocity_covariance is None:
            largest_variance = max(temperatures) * const.e / const.m_e_kg
        else:
            largest_variance = float(np.max(np.linalg.eigvalsh(np.asarray(self.velocity_covariance))))
        if largest_variance > (0.05 * const.c)**2:
            raise ValueError("Electron cooling requires nonrelativistic electron thermal speeds (rms <= 0.05 c)")
        return self


class ElectronCoolerItem(ElementBase):
    """A physical cooling section with positive forward transport substeps."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)

    command: Literal["ElectronCooler"] = Field(default="ElectronCooler", alias="Command")
    electron_beam: ElectronBeamConfig = Field(alias="Electron beam")
    model: Literal["gaussian", "parkhomchuk", "magnetized_collision"] = Field(default="gaussian", alias="Model")
    collisions: StrictBool = Field(default=True, alias="Collisions")
    diffusion: StrictBool = Field(default=True, alias="Diffusion")
    effective_velocity_spread: float = Field(default=0.0, ge=0, alias="Effective velocity spread (m/s)")
    magnetic_field: float = Field(default=0.0, alias="Magnetic field (T)")
    mean_space_charge: StrictBool = Field(default=False, alias="Mean space charge")
    coulomb_log: float | None = Field(default=None, gt=0, alias="Coulomb log")
    min_impact_parameter: float | None = Field(default=None, gt=0, alias="Min impact parameter (m)")
    max_impact_parameter: float | None = Field(default=None, gt=0, alias="Max impact parameter (m)")
    quadrature_order: StrictInt = Field(default=64, ge=8, alias="Quadrature order")
    radial_order: StrictInt = Field(default=16, ge=4, alias="Radial order")
    polar_order: StrictInt = Field(default=12, ge=4, alias="Polar order")
    azimuthal_order: StrictInt = Field(default=16, ge=4, alias="Azimuthal order")
    time_order: StrictInt = Field(default=64, ge=8, alias="Time order")
    quadrature_rtol: float = Field(default=0.02, ge=1.e-6, le=0.1, alias="Quadrature relative tolerance")
    max_refinements: StrictInt = Field(default=2, ge=1, alias="Max refinements")
    num_slices: StrictInt = Field(default=1, ge=1, alias="Num slices")
    max_substeps: StrictInt = Field(default=1000, ge=1, alias="Max substeps")
    max_fractional_step: float = Field(default=0.05, gt=0, le=0.1, alias="Max fractional step")
    random_seed: StrictInt | None = Field(default=None, ge=0, alias="Random Seed")
    save_diagnostics: StrictBool = Field(default=False, alias="Save diagnostics")
    save_turns: list[list[StrictInt]] | list[StrictInt] = Field(default_factory=list, alias="Save turns")

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, value):
        return _normalize_cooler_fields(cls, value)

    @field_validator("save_turns")
    @classmethod
    def _validate_save_turns(cls, value):
        if not value:
            return []
        items = [value] if all(isinstance(item, int) for item in value) else value
        for item in items:
            if len(item) not in {1, 3}:
                raise ValueError("ElectronCooler Save turns entries must be [turn] or [start, end, step]")
            if item[0] < 0 or len(item) == 3 and (item[1] < item[0] or item[2] <= 0):
                raise ValueError("ElectronCooler Save turns requires 0 <= start <= end and step > 0")
        return items

    @model_validator(mode="after")
    def _validate_cooling_model(self):
        if self.model in {"parkhomchuk", "magnetized_collision"} and (self.electron_beam.angle_x != 0 or self.electron_beam.angle_y != 0):
            raise ValueError("Magnetized cooling requires the electron mean direction parallel to the axial solenoid")
        if self.model == "parkhomchuk":
            if self.diffusion:
                raise ValueError("Parkhomchuk provides a friction force only; set Diffusion=False")
            if self.magnetic_field == 0:
                raise ValueError("Parkhomchuk requires a nonzero Magnetic field (T)")
        elif self.effective_velocity_spread != 0:
            raise ValueError("Effective velocity spread (m/s) is supported only by parkhomchuk")
        if self.model == "magnetized_collision":
            if self.magnetic_field <= 0:
                raise ValueError("magnetized_collision requires a positive Magnetic field (T)")
            if self.min_impact_parameter is None or self.max_impact_parameter is None or self.max_impact_parameter <= self.min_impact_parameter:
                raise ValueError("magnetized_collision requires explicit 0 < Min impact parameter (m) < Max impact parameter (m)")
            if self.coulomb_log is not None:
                raise ValueError("magnetized_collision integrates explicit impact-parameter cutoffs; omit Coulomb log")
            covariance = self.electron_beam.velocity_covariance
            if covariance is not None:
                matrix = np.asarray(covariance)
                scale = float(np.max(np.abs(matrix)))
                if (not np.allclose(matrix - np.diag(np.diag(matrix)), 0, rtol=0, atol=1.e-12 * scale)
                        or not np.isclose(matrix[0, 0], matrix[1, 1], rtol=1.e-10, atol=0)):
                    raise ValueError("magnetized_collision requires a diagonal gyrotropic electron velocity covariance with Cxx=Cyy")
            gradient = self.electron_beam.velocity_gradient
            if gradient is not None and np.any(np.asarray(gradient)[:2] != 0):
                raise ValueError("magnetized_collision supports longitudinal electron velocity gradients only")
        elif self.min_impact_parameter is not None:
            raise ValueError("Min impact parameter (m) is supported only by magnetized_collision")
        if self.model != "gaussian" and self.quadrature_order != 64:
            raise ValueError("Quadrature order is supported only by gaussian")
        if self.model != "magnetized_collision":
            if (self.radial_order, self.polar_order, self.azimuthal_order, self.time_order, self.quadrature_rtol,
                    self.max_refinements) != (16, 12, 16, 64, 0.02, 2):
                raise ValueError("Magnetized quadrature controls are supported only by magnetized_collision")
        if self.coulomb_log is not None and self.max_impact_parameter is not None:
            raise ValueError("A fixed Coulomb log does not use Max impact parameter (m)")
        electron = self.electron_beam
        if self.mean_space_charge and electron.mode == "gaussian_bunch":
            maximum_size = max(value for value in (electron.radius, electron.radius_exit, electron.sigma_x, electron.sigma_x_exit, electron.sigma_y,
                                                   electron.sigma_y_exit) if value is not None)
            electron_gamma = 1 + electron.kinetic_energy / const.m_e_eV
            rest_length = np.sqrt((electron_gamma - 1) * (electron_gamma + 1)) * const.c * electron.sigma_time
            if rest_length < 10 * maximum_size:
                raise ValueError("The long-beam mean field requires the electron rest-frame RMS bunch length >= 10 transverse beam sizes")
        return self


# Kicker


class KickerItem(MagneticElementBase):
    command: str = Field(default="Kicker", alias="Command")
    hkick: float = Field(default=0.0, alias="HKICK")
    vkick: float = Field(default=0.0, alias="VKICK")

    is_ramping: bool = Field(default=False, alias="Is ramping")
    kick_ramping_file: str = Field(default="", alias="Kick ramping file")

    num_slices: int = Field(default=1, ge=1, alias="Num slices")
    integrator: str = Field(default="adaptive", alias="Integrator")


class BumpItem(SlicedElementBase):
    model_config = ConfigDict(populate_by_name=True, allow_inf_nan=False)
    command: str = Field(default="Bump", alias="Command")
    waveform_file: str = Field(alias="Waveform file", min_length=1)
    time_mode: Literal["reference", "particle"] = Field(default="particle", alias="Time mode")
    time_offset: float = Field(default=0.0, alias="Time offset (s)")
    enabled: bool = Field(default=True, alias="Enable")
    num_slices: StrictInt = Field(default=1, ge=1, alias="Num slices")


# ElSeparator (electrostatic separator)


class ElSeparatorItem(SlicedElementBase):
    """Infinite-height parallel electrodes, with a thick or thin electric kick."""
    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)
    command: str = Field(default="ElSeparator", alias="Command")
    voltage: float | None = Field(default=None,
                                  alias="V (V)",
                                  description="Signed septum-minus-high-voltage-electrode potential difference; specify exactly one of V and VL")
    voltage_length: float | None = Field(
        default=None,
        alias="VL (V m)",
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
        if not math.isfinite(self.s - self.length):
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


# Exciter (tune exciter)


class ExciterItem(ElementBase):
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


# RFCavity

from PASS.para.schema.rf import RFComponent


class RFCavityItem(ElementBase):
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


# ReorganizeBunch (bunch index redistribution, no physical tracking)


class ReorganizeBunchItem(ElementBase):
    """Reorganize bunch command: switch to a new harmonic (bucket grid).

    All particles are sorted by longitudinal position, reassigned to the
    nearest new bucket center, and the beam harmonic number / bunch structure
    are updated to the new harmonic (one bunch per bucket).
    """
    command: str = Field(default="ReorganizeBunch", alias="Command")
    start_turn: int = Field(default=0, alias="Start turn")
    new_harmonic: int | None = Field(
        default=None,
        ge=1,
        alias="New harmonic number",
        description="Harmonic number after reorganization (bucket count). "
        "Particles are re-sorted and assigned to the nearest new "
        "bucket center; beam harmonic and bunch count are "
        "updated accordingly.",
    )


class CollisionElementBase(ElementBase):
    """Strict zero-length collision-adjacent elements; aliases are case insensitive."""

    model_config = ConfigDict(populate_by_name=True, extra='forbid', allow_inf_nan=False)
    length: float = Field(default=0., ge=0, le=0, alias='Length (m)')
    name: str | None = Field(default=None, exclude=True)

    @model_validator(mode='before')
    @classmethod
    def normalize_aliases(cls, value):
        if not isinstance(value, dict):
            return value
        aliases = {str(field.alias or name).lower(): name for name, field in cls.model_fields.items()}
        return {aliases.get(str(key).lower(), str(key).lower()): item for key, item in value.items()}

    @model_validator(mode='after')
    def validate_collision_aperture(self):
        if self.aperture_type != 'off' or self.aperture_value:
            raise ValueError('Collision-adjacent ideal elements do not impose a physical aperture')
        return self


class CrossingAngleItem(CollisionElementBase):
    command: str = Field(default='CrossingAngle', alias='Command')
    configuration: str = Field(min_length=1, alias='Configuration')
    direction: Literal['forward', 'inverse'] = Field(alias='Direction')


class EquivalentDispersion(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='forbid', allow_inf_nan=False)
    dx: float = Field(default=0., alias='Dx (m)')
    dpx: float = Field(default=0., alias='Dpx')
    dy: float = Field(default=0., alias='Dy (m)')
    dpy: float = Field(default=0., alias='Dpy')

    @model_validator(mode='before')
    @classmethod
    def normalize_aliases(cls, value):
        if not isinstance(value, dict):
            return value
        aliases = {str(field.alias or name).lower(): name for name, field in cls.model_fields.items()}
        return {aliases.get(str(key).lower(), str(key).lower()): item for key, item in value.items()}


class IPEquivalentItem(CollisionElementBase):
    optics_reference: str = Field(min_length=1, alias='Optics reference')
    side: Literal['before', 'after'] = Field(alias='Side')
    equivalent_dispersion: EquivalentDispersion = Field(default_factory=EquivalentDispersion, alias='Equivalent dispersion')
    longitudinal_shear: float = Field(default=0., alias='Longitudinal shear (m)')


class CrabCavityItem(IPEquivalentItem):
    command: str = Field(default='CrabCavity', alias='Command')
    plane: Literal['x', 'y'] = Field(alias='Plane')
    phase_advance: float = Field(alias='Phase advance (rad)')
    equivalent_kick: float = Field(alias='Equivalent kick')
    frequency: float = Field(gt=0., alias='Frequency (Hz)')
    phase: float = Field(default=0., alias='Phase (rad)')
    phase_epoch: float | None = Field(default=None, alias='Phase epoch (s)')


class FloatWaisterItem(IPEquivalentItem):
    command: str = Field(default='FloatWaister', alias='Command')
    mode: Literal['rfq', 'theory'] = Field(alias='Mode')
    phase_advance_x: float = Field(alias='Phase advance x (rad)')
    phase_advance_y: float = Field(alias='Phase advance y (rad)')
    strength_x: float | None = Field(default=None, alias='Strength x')
    strength_y: float | None = Field(default=None, alias='Strength y')
    gx: float | None = Field(default=None, alias='Equivalent gx (1/m)')
    gy: float | None = Field(default=None, alias='Equivalent gy (1/m)')
    frequency: float | None = Field(default=None, gt=0., alias='Frequency (Hz)')
    phase: float | None = Field(default=None, alias='Phase (rad)')
    phase_epoch: float | None = Field(default=None, alias='Phase epoch (s)')

    @model_validator(mode='after')
    def validate_model_parameters(self):
        if self.mode == 'theory':
            if self.strength_x is None or self.strength_y is None:
                raise ValueError('Theory FloatWaister requires Strength x and Strength y')
            if any(value is not None for value in (self.gx, self.gy, self.frequency, self.phase, self.phase_epoch)):
                raise ValueError('Theory FloatWaister does not use RF gradients, frequency or phase')
        else:
            if self.gx is None or self.gy is None or self.frequency is None:
                raise ValueError('RFQ FloatWaister requires Equivalent gx/gy and Frequency')
            if self.strength_x is not None or self.strength_y is not None:
                raise ValueError('RFQ FloatWaister does not use theory strengths')
        return self


# Convenience registry

ELEMENT_REGISTRY: dict[str, type[ElementBase]] = {
    "drift": DriftItem,
    "marker": MarkerItem,
    "sbend": SBendItem,
    "quadrupole": QuadrupoleItem,
    "sextupole": SextupoleItem,
    "octupole": OctupoleItem,
    "multipole": MultipoleItem,
    "solenoid": SolenoidItem,
    "electroncooler": ElectronCoolerItem,
    "kicker": KickerItem,
    "bump": BumpItem,
    "elseparator": ElSeparatorItem,
    "exciter": ExciterItem,
    "rfcavity": RFCavityItem,
    "crossingangle": CrossingAngleItem,
    "crabcavity": CrabCavityItem,
    "floatwaister": FloatWaisterItem,
    "reorganizebunch": ReorganizeBunchItem,
}
