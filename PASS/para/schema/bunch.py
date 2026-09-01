"""Injection parameters (BunchConfig, OffsetConfig, InjectionItem).

Consumed by:
    - PASS.commands.injection.InjectionBunchInfo  (bunch0/bunch1/...)
    - PASS.commands.injection._read_offset_fromfile (offset file columns)

The injection JSON node is nested inside Sequence as:
    "Injection": {"S (m)": 0.0, "Command": "Injection", "bunch0": {...}}
"""

from pydantic import BaseModel, ConfigDict, Field, StrictInt


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

    # --- injection timing ---
    injection_turns: int = Field(
        default=1, ge=1,
        alias="Total Injection Turns",
    )
    injection_interval: int = Field(
        default=1, ge=1,
        alias="Injection Interval",
    )

    # --- transverse twiss ---
    alpha_x: float = Field(default=0.0, alias="Alpha x")
    alpha_y: float = Field(default=0.0, alias="Alpha y")
    beta_x: float = Field(default=1.0, gt=0, alias="Beta x (m)")
    beta_y: float = Field(default=1.0, gt=0, alias="Beta y (m)")

    # --- emittance ---
    emit_x: float = Field(default=0.0, ge=0, alias="Emittance x (m'rad)")
    emit_y: float = Field(default=0.0, ge=0, alias="Emittance y (m'rad)")

    # --- dispersion ---
    dx: float = Field(default=0.0, alias="Dx (m)")
    dpx: float = Field(default=0.0, alias="Dpx")

    # --- longitudinal ---
    sigma_z: float = Field(default=0.1, gt=0, alias="Sigma z (m)")
    dp: float = Field(default=0.001, gt=0, alias="Sigma dp/p")

    # --- distribution type ---
    dist_trans: str = Field(
        default="gaussian",
        alias="Transverse dist",
        description="kv / gaussian / uniform / waterbag / parabolic",
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
    insert_particle: list[list[float]] = Field(
        default_factory=list,
        alias="Insert Particle Coordinate",
        description="Manual particle coordinates [[x,px,y,py,z,dp], ...]",
    )


class InjectionItem(BaseModel):
    """The Injection sequence node.

    Consumed by PASS.commands.injection.Injection.__init__.
    ``harmonic_number`` is declared ONCE at the injection level; it defines
    how many longitudinal bunch groups are created.  Every bunch dict must
    carry its ``Harmonic ID of this bunch`` (group slot in
    [0, harmonic_number)).
    """

    model_config = ConfigDict(populate_by_name=True)

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
            raise ValueError(
                "InjectionItem requires exactly one BunchConfig per "
                f"harmonic group: harmonic_number={self.harmonic_number}, "
                f"bunches={len(self.bunches)}"
            )
        harmonic_ids = [bunch.harmonic_id for bunch in self.bunches]
        if set(harmonic_ids) != set(range(self.harmonic_number)):
            raise ValueError(
                "InjectionItem bunch harmonic ids must be a permutation of "
                f"[0, {self.harmonic_number}); got {harmonic_ids}"
            )

        result = {
            "S (m)": self.s,
            "Command": self.command,
            "Harmonic Number": self.harmonic_number,
            "Random Seed": self.random_seed,
        }
        for i, bunch in enumerate(self.bunches):
            result[f"bunch{i}"] = bunch.model_dump(by_alias=True)
        return result
