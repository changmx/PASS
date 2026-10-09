"""Injection parameters and explicit Cartesian scan coordinates.

Consumed by:
    - PASS.commands.injection.InjectionBunchInfo  (bunch0/bunch1/...)
    - PASS.commands.injection._read_offset_fromfile (offset file columns)

The injection JSON node is nested inside Sequence as:
    "Injection": {"S (m)": 0.0, "Command": "Injection", "bunch0": {...}}
"""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class OffsetConfig(BaseModel):
    """Injection offset configuration (x or y direction)."""

    model_config = ConfigDict(populate_by_name=True)

    is_offset: bool = Field(
        default=False,
        alias="Is Offset",
    )
    is_load_from_file: bool = Field(
        default=False,
        alias="Is Load From File",
    )
    file_path: str = Field(
        default="",
        alias="File Path",
    )
    file_time_kind: str = Field(
        default="turn",
        alias="File Time Kind",
    )
    offset_position: float = Field(
        default=0.0,
        alias="Offset Position (m)",
    )
    offset_momentum: float = Field(
        default=0.0,
        alias="Offset Momentum (rad)",
    )


class ScanGridConfig(BaseModel):
    """A Cartesian x/y grid repeated for every initial momentum deviation."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)

    x_range: list[float] = Field(alias="X range (m)", min_length=2, max_length=2)
    y_range: list[float] = Field(alias="Y range (m)", min_length=2, max_length=2)
    num_x: StrictInt = Field(alias="Number of x points", ge=1)
    num_y: StrictInt = Field(alias="Number of y points", ge=1)
    dp_values: list[float] = Field(alias="dp values", min_length=1)
    px: float = Field(default=0.0, alias="px")
    py: float = Field(default=0.0, alias="py")
    z: float = Field(default=0.0, alias="z (m)")

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, values):
        if not isinstance(values, dict):
            return values
        aliases = {(field.alias or name).casefold(): name for name, field in cls.model_fields.items()}
        return {aliases.get(str(key).casefold(), key): value for key, value in values.items()}

    @model_validator(mode="after")
    def _validate_grid(self):
        for name, bounds, count in (("x", self.x_range, self.num_x), ("y", self.y_range, self.num_y)):
            if count == 1 and bounds[0] != bounds[1]:
                raise ValueError(f"A one-point {name} range must have identical endpoints")
            if count > 1 and bounds[0] >= bounds[1]:
                raise ValueError(f"A multi-point {name} range must have increasing endpoints")
        if len(set(self.dp_values)) != len(self.dp_values):
            raise ValueError("dp values must not contain duplicates")
        if any(1.0 + dp <= math.hypot(self.px, self.py) for dp in self.dp_values):
            raise ValueError("Scan momenta require dp > -1 and positive real longitudinal momentum")
        if self.num_particles > 2147483647:
            raise ValueError("Scan grid exceeds the int32 particle identity capacity")
        return self

    @property
    def num_particles(self):
        return self.num_x * self.num_y * len(self.dp_values)

    def generate(self):
        import numpy as np
        from PASS.utils.scan_grid import generate_scan_grid

        return generate_scan_grid(np.linspace(*self.x_range, self.num_x),
                                  np.linspace(*self.y_range, self.num_y),
                                  self.dp_values,
                                  px=self.px,
                                  py=self.py,
                                  z=self.z)


class BunchConfig(BaseModel):
    """Per-bunch injection parameters.

    One BunchConfig per bunch in the beam.
    """

    model_config = ConfigDict(populate_by_name=True)

    # --- energy & intensity ---
    kinetic_energy: float = Field(
        ...,
        alias="Kinetic Energy per Nucleon (eV/u)",
        description="Kinetic energy per nucleon in eV/u",
    )
    num_real_particles: int = Field(
        ...,
        alias="Number of Real Particles",
        description="Number of real particles per bunch",
    )
    num_macro_particles: int = Field(
        ...,
        alias="Number of Macro Particles",
        description="Number of macro particles per bunch",
    )

    # --- distribution loading ---
    is_load_from_file: bool = Field(
        default=False,
        alias="Is Load Distribution from File",
    )
    file_path: str = Field(
        default="",
        alias="Distribution File Path",
    )
    file_mode: Literal["sequential", "repeat"] = Field(
        default="sequential",
        alias="Distribution File Mode",
        description="Read successive bunch-local rows, or repeat the first batch each event.",
    )

    reference_arrival_time: float | None = Field(default=None, alias="Reference arrival time (s)")

    # --- injection timing ---
    injection_turns: int = Field(
        default=1,
        ge=1,
        alias="Total Injection Turns",
    )
    injection_interval: int = Field(
        default=1,
        ge=1,
        alias="Injection Interval",
    )

    # --- transverse twiss ---
    alpha_x: float = Field(default=0.0, alias="Alpha x")
    alpha_y: float = Field(default=0.0, alias="Alpha y")
    beta_x: float = Field(default=1.0, gt=0, alias="Beta x (m)")
    beta_y: float = Field(default=1.0, gt=0, alias="Beta y (m)")

    # --- emittance ---
    emit_x: float = Field(default=0.0,
                          ge=0,
                          alias="RMS geometric emittance x (m'rad)",
                          description="Horizontal RMS geometric emittance before dispersion and centroid offsets.")
    emit_y: float = Field(default=0.0,
                          ge=0,
                          alias="RMS geometric emittance y (m'rad)",
                          description="Vertical RMS geometric emittance before dispersion and centroid offsets.")

    # --- dispersion ---
    dx: float = Field(default=0.0, alias="Dx (m)")
    dpx: float = Field(default=0.0, alias="Dpx")

    # --- longitudinal ---
    sigma_z: float = Field(default=0.1, gt=0, alias="Sigma z (m)")
    dp: float = Field(default=0.001, gt=0, alias="Sigma dp/p")

    # --- distribution type ---
    dist_trans: Literal["kv", "gaussian", "uniform-real", "uniform-phase", "waterbag", "parabolic"] = Field(
        default="gaussian",
        alias="Transverse dist",
        description="kv / gaussian / uniform-real / uniform-phase / waterbag / parabolic",
    )
    dist_longi: str = Field(
        default="gaussian",
        alias="Longitudinal dist",
        description="gaussian / coasting / matchz / matchdp",
    )

    # --- RF (for matchz/matchdp) ---
    rf_voltage: float = Field(default=0.0, alias="RF Voltage (V)")
    rf_phase: float = Field(default=0.0, alias="RF Phase (rad)")
    harmonic_id: int = Field(default=0, ge=0, alias="Harmonic ID of this bunch")
    rf_s_position: float = Field(
        default=0.0,
        alias="RF S Position Refer to Inj. Point (m)",
    )

    # --- momentum offset (ddp / dde, mutually exclusive) ---
    momentum_offset_dp: float = Field(
        default=0.0,
        alias="Momentum Offset dp",
        description="Bunch-level average momentum deviation (dp/p). "
        "Mutually exclusive with kinetic energy offset.",
    )
    kinetic_energy_offset: float = Field(
        default=0.0,
        alias="Kinetic Energy Offset (eV)",
        description="Bunch-level kinetic energy offset in eV. "
        "Converted to dp internally. "
        "Mutually exclusive with momentum offset dp.",
    )

    # --- offsets ---
    offset_x: OffsetConfig = Field(
        default_factory=OffsetConfig,
        alias="Offset x",
    )
    offset_y: OffsetConfig = Field(
        default_factory=OffsetConfig,
        alias="Offset y",
    )

    # --- misc ---
    save_init_dist: bool = Field(
        default=False,
        alias="Is Save Initial Distribution",
    )
    output_format: Literal["tfs", "hdf5", "hdf5-gzip1"] = Field(default="hdf5",
                                                                alias="Output format",
                                                                description="Initial distribution output format")
    insert_particle: list[list[float]] = Field(
        default_factory=list,
        alias="Insert Particle Coordinate",
        description="Manual particle coordinates [[x,px,y,py,z,dp], ...]",
    )
    insert_particle_file: str | None = Field(default=None,
                                             alias="Insert Particle File",
                                             description="Six-column TFS/HDF5 coordinates replacing first-batch rows after injection offsets")
    scan_grid: ScanGridConfig | None = Field(default=None,
                                             alias="Scan Grid",
                                             description="Cartesian coordinates replacing first-batch rows after injection offsets")

    @model_validator(mode="before")
    @classmethod
    def _validate_emittance_keys(cls, values):
        if isinstance(values, dict):
            for key in values:
                if str(key).casefold() in {"emittance x (m'rad)", "emittance y (m'rad)"}:
                    raise ValueError("Use RMS geometric emittance x/y (m'rad) for injection emittances")
        return values

    @model_validator(mode="after")
    def _validate_insert_source(self):
        if self.insert_particle_file is not None and not self.insert_particle_file.strip():
            self.insert_particle_file = None
        selected = bool(self.insert_particle) + bool(self.insert_particle_file) + (self.scan_grid is not None)
        if selected > 1:
            raise ValueError("Insert Particle Coordinate, Insert Particle File and Scan Grid are mutually exclusive")
        if self.scan_grid is not None:
            events = (self.injection_turns + self.injection_interval - 1) // self.injection_interval
            first_count = self.num_macro_particles // events + self.num_macro_particles % events
            if self.scan_grid.num_particles > first_count:
                raise ValueError(f"Scan Grid needs {self.scan_grid.num_particles} particles but the first injection batch contains {first_count}")
        return self


class InjectionItem(BaseModel):
    """The Injection sequence node.

    Consumed by PASS.commands.injection.Injection.__init__.
    ``harmonic_number`` is declared ONCE at the injection level; it defines
    how many longitudinal bunch groups are created.  Every bunch dict must
    carry its ``Harmonic ID of this bunch`` (group slot in
    [0, harmonic_number)).
    """

    model_config = ConfigDict(populate_by_name=True)

    order: StrictInt | None = Field(default=None, alias="Order")
    s: float = Field(
        default=0.0,
        alias="S (m)",
        description="Injection position (must be 0)",
    )
    command: str = Field(
        default="Injection",
        alias="Command",
    )
    harmonic_number: int = Field(
        default=1,
        ge=1,
        alias="Harmonic Number",
        description="Beam bunch grouping count. Determines the number of "
        "longitudinal groups (C/h spacing) and the number of "
        "bunch dictionaries created at injection. It does not "
        "restrict RF cavity harmonics.",
    )
    random_seed: StrictInt | None = Field(
        default=None,
        alias="Random Seed",
        description="Optional seed for Injection particle-distribution "
        "generation. Omit it for a non-deterministic seed.",
    )
    bunches: list[BunchConfig] = Field(
        default_factory=lambda: [BunchConfig(
            kinetic_energy=33.2e6,
            num_real_particles=int(1e11),
            num_macro_particles=int(1e5),
        )],
        description="List of bunch configurations (bunch0, bunch1, ...)",
    )

    def to_sequence_dict(self) -> dict:
        """Convert to engine-compatible dict with bunch0/bunch1/... keys."""
        if len(self.bunches) != self.harmonic_number:
            raise ValueError("InjectionItem requires exactly one BunchConfig per "
                             f"harmonic group: harmonic_number={self.harmonic_number}, "
                             f"bunches={len(self.bunches)}")
        harmonic_ids = [bunch.harmonic_id for bunch in self.bunches]
        if set(harmonic_ids) != set(range(self.harmonic_number)):
            raise ValueError("InjectionItem bunch harmonic ids must be a permutation of "
                             f"[0, {self.harmonic_number}); got {harmonic_ids}")

        populated = [b for b in self.bunches if b.num_macro_particles > 0]
        if populated:
            source = populated[0]
            if any(b.num_real_particles * source.num_macro_particles != source.num_real_particles * b.num_macro_particles for b in populated):
                raise ValueError("All populated bunches in a beam must have the same fixed macro-particle weight")

        result = {
            "S (m)": self.s,
            "Command": self.command,
            "Harmonic Number": self.harmonic_number,
            "Random Seed": self.random_seed,
        }
        if self.order is not None:
            result["Order"] = self.order
        for i, bunch in enumerate(self.bunches):
            result[f"bunch{i}"] = bunch.model_dump(by_alias=True)
        return result
