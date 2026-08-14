"""Generate distribution-only PASS inputs for Example 01.

Cases:
    transverse       Five transverse distributions with longitudinal Gaussian
    longi-gaussian   One Gaussian/Gaussian bunch
    longi-matchz     One Gaussian/MatchZ bunch with h=1
    longi-matchdp    One Gaussian/MatchDp bunch with h=1
    coasting         One Gaussian/coasting bunch with h=1

Usage:
    python make_input.py
    python make_input.py --case transverse
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PASS.para.api import build_sequence, generate_input
from PASS.para.schema.bunch import BunchConfig
from PASS.para.schema.main import MainConfig


SCRIPT_DIR = Path(__file__).resolve().parent

CIRCUM = 251.327
GAMMA_T = 4.8
NUM_TURNS = 1
NUM_MACRO_PARTICLES = 100000
NUM_REAL_PARTICLES = int(1e11)

KINETIC_ENERGY = 45e6
ALPHA_X = -2.614303952
ALPHA_Y = 1.57442348
BETA_X = 0.5
BETA_Y = 0.5
EMIT_X = 200e-6
EMIT_Y = 100e-6

GAUSSIAN_SIGMA_Z = 5.0
GAUSSIAN_SIGMA_DP = 1e-3

# Keep the longitudinal matching setup from the former example beam0.json.
MATCH_SIGMA_Z = 30.0
MATCH_SIGMA_DP = 5e-3
MATCH_RF_VOLTAGE = 100e3
MATCH_RF_PHASE = 0.5235987755982988

CASES = {
    "transverse": {
        "input_name": "beam0_transverse.json",
        "description": "Five transverse distributions with longitudinal Gaussian bunches.",
        "bunches": [
            {"transverse": "gaussian", "longitudinal": "gaussian"},
            {"transverse": "kv", "longitudinal": "gaussian"},
            {"transverse": "waterbag", "longitudinal": "gaussian"},
            {"transverse": "parabolic", "longitudinal": "gaussian"},
            {"transverse": "uniform", "longitudinal": "gaussian"},
        ],
    },
    "longi-gaussian": {
        "input_name": "beam0_longi_gaussian.json",
        "description": "Gaussian transverse and Gaussian longitudinal distribution.",
        "bunches": [{"transverse": "gaussian", "longitudinal": "gaussian"}],
    },
    "longi-matchz": {
        "input_name": "beam0_longi_matchz.json",
        "description": "Gaussian transverse and MatchZ longitudinal distribution with h=1.",
        "bunches": [{"transverse": "gaussian", "longitudinal": "matchz"}],
    },
    "longi-matchdp": {
        "input_name": "beam0_longi_matchdp.json",
        "description": "Gaussian transverse and MatchDp longitudinal distribution with h=1.",
        "bunches": [{"transverse": "gaussian", "longitudinal": "matchdp"}],
    },
    "coasting": {
        "input_name": "beam0_coasting.json",
        "description": "Gaussian transverse and coasting longitudinal distribution with h=1.",
        "bunches": [{"transverse": "gaussian", "longitudinal": "coasting"}],
    },
}


def input_path(case_name: str) -> Path:
    """Return the generated JSON path for a named case."""
    return SCRIPT_DIR / CASES[case_name]["input_name"]


def make_main(case_name: str) -> MainConfig:
    """Create a one-turn, CPU-only configuration for initial generation."""
    return MainConfig(
        beam_name="proton",
        num_proton=1,
        num_neutron=0,
        num_electron=1,
        gamma_t=GAMMA_T,
        circumference=CIRCUM,
        num_turns=NUM_TURNS,
        backend="cpu",
        num_gpu=1,
        gpu_id=[0],
        output_dir=str(SCRIPT_DIR / "output" / case_name),
        is_plot=True,
        is_space_charge=False,
        is_beambeam=False,
    )


def make_bunch(spec: dict) -> BunchConfig:
    """Build one bunch from a compact transverse/longitudinal case spec."""
    longitudinal = spec["longitudinal"]
    is_rf_matched = longitudinal in {"matchz", "matchdp"}
    sigma_z = CIRCUM if longitudinal == "coasting" else GAUSSIAN_SIGMA_Z
    sigma_dp = GAUSSIAN_SIGMA_DP
    if is_rf_matched:
        sigma_z = MATCH_SIGMA_Z
        sigma_dp = MATCH_SIGMA_DP

    return BunchConfig(
        kinetic_energy=KINETIC_ENERGY,
        num_real_particles=NUM_REAL_PARTICLES,
        num_macro_particles=NUM_MACRO_PARTICLES,
        injection_turns=1,
        injection_interval=1,
        alpha_x=ALPHA_X,
        alpha_y=ALPHA_Y,
        beta_x=BETA_X,
        beta_y=BETA_Y,
        emit_x=EMIT_X,
        emit_y=EMIT_Y,
        dx=0.0,
        dpx=0.0,
        sigma_z=sigma_z,
        dp=sigma_dp,
        dist_trans=spec["transverse"],
        dist_longi=longitudinal,
        rf_voltage=MATCH_RF_VOLTAGE if is_rf_matched else 0.0,
        rf_phase=MATCH_RF_PHASE if is_rf_matched else 0.0,
        rf_s_position=0.0,
        save_init_dist=True,
    )


def make_case(case_name: str) -> Path:
    """Generate the named PASS input JSON."""
    case = CASES[case_name]
    bunches = [make_bunch(spec) for spec in case["bunches"]]
    sequence = build_sequence(items=[], names=[], bunches=bunches)
    path = input_path(case_name)

    generate_input(make_main(case_name), sequence, str(path))

    print(f"[Done] {case_name}: {path.name}")
    print(f"  {case['description']}")
    print(f"  {len(bunches)} bunch(es), injection harmonic number = {len(bunches)}")
    for bunch_id, spec in enumerate(case["bunches"]):
        print(
            f"  bunch{bunch_id}: transverse={spec['transverse']}, "
            f"longitudinal={spec['longitudinal']}"
        )
    return path


def selected_cases(case_name: str) -> list[str]:
    """Expand the all shortcut and validate a user-provided case name."""
    if case_name == "all":
        return list(CASES)
    return [case_name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Example 01 inputs.")
    parser.add_argument(
        "--case",
        choices=["all", *CASES],
        default="all",
        help="Input case to generate (default: all).",
    )
    args = parser.parse_args()

    for case_name in selected_cases(args.case):
        make_case(case_name)


if __name__ == "__main__":
    main()
