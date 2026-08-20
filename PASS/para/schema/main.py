"""Global simulation parameters (MainConfig).

Consumed by:
    - PASS.core.config.Config.load_input  (root-level keys)
    - PASS.core.beam.Beam._load_input     (is_space_charge, is_beambeam)
    - PASS.core.bunch.BunchInfo._load_input  (gamma_t, protons, neutrons, charges, circumference)

All aliases must match the JSON keys the engine expects (case-insensitive).
"""

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class MainConfig(BaseModel):
    """Global parameters for a PASS simulation.

    These appear at the root level of the input JSON file.
    """

    model_config = ConfigDict(populate_by_name=True)

    # --- particle identity ---
    beam_name: str = Field(
        default="proton",
        alias="Beam Name",
        description="Arbitrary label for the beam species",
    )
    num_proton: int = Field(
        default=1, ge=0,
        alias="Number of Protons",
        description="Proton count per particle (0 for electron/positron)",
    )
    num_neutron: int = Field(
        default=0, ge=0,
        alias="Number of Neutrons",
        description="Neutron count per particle (>0 for ion)",
    )
    num_electron: int = Field(
        default=1,
        alias="Number of Charges",
        description="Charge count per particle (can be negative, not zero)",
    )

    # --- ring ---
    gamma_t: float = Field(
        default=7.635,
        alias="Transition Gamma",
        description="Transition gamma of the lattice",
    )
    circumference: float = Field(
        default=569.1, gt=0,
        alias="Circumference (m)",
        description="Ring circumference in meters",
    )

    # --- simulation control ---
    num_turns: int = Field(
        default=100, ge=1,
        alias="Number of turns",
        description="Total number of turns to simulate",
    )
    backend: str = Field(
        default="cpu",
        alias="Backend (gpu/cpu)",
        description="Compute backend: 'cpu' or 'gpu'",
    )
    particle_precision: Literal["float32", "float64"] = Field(
        default="float64",
        alias="Particle Precision",
        description="Storage precision for the six particle coordinates",
    )
    num_gpu: int = Field(
        default=1, ge=1,
        alias="Number of GPU devices",
        description="Number of GPU devices to use",
    )
    gpu_id: list[int] = Field(
        default_factory=lambda: [0],
        alias="Device Id",
        description="List of GPU device IDs",
    )
    output_dir: str = Field(
        default="./output",
        alias="Output directory",
        description="Root output directory (absolute or relative to input file)",
    )
    is_plot: bool = Field(
        default=False,
        alias="Is plot figure",
        description="Whether to generate plots after simulation",
    )

    # --- optional module flags (read by Beam._load_input) ---
    is_space_charge: bool = Field(
        default=False,
        alias="Is space charge",
        description="Whether space-charge module is active",
    )
    is_beambeam: bool = Field(
        default=False,
        alias="Is beam-beam",
        description="Whether beam-beam module is active",
    )
