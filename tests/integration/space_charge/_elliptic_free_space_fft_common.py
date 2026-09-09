"""Shared implementation for elliptic transverse-field FFT validations.

Run from the repository root, for example::

    python -m tests.integration.space_charge._elliptic_free_space_fft_common gaussian simana
    python -m tests.integration.space_charge._elliptic_free_space_fft_common kv simana
    python -m tests.integration.space_charge._elliptic_free_space_fft_common gaussian ana

Each simulation stores one SpaceCharge HDF5 grid.  Analysis compares that grid
with the corresponding analytic elliptic field without rerunning tracking.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from PASS.commands.solver.formula_gaussian_ellipse import gaussian_elliptic_field
from PASS.commands.solver.formula_uniform_ellipse import uniform_elliptic_field
from PASS.main import main as pass_main
from PASS.para.api import generate_input
from PASS.para.schema.bunch import BunchConfig, InjectionItem
from PASS.para.schema.main import MainConfig
from PASS.para.schema.sequence import Sequence
from PASS.para.schema.elements import MarkerElement
from PASS.para.schema.slicer import Slicer
from PASS.para.schema.space_charge import (
    SpaceCharge,
    SpaceChargeConfig,
    SpaceChargeResourceConfig,
)
from tests.integration.space_charge.test_round_gaussian_free_space_fft import (
    _grid_consistency_checks,
    _plot_results,
    _scan_points,
    _validate_generated_input,
    SCAN_RADIAL_STEP_M,
)


ROOT = Path(__file__).resolve().parent
NUM_MACRO_PARTICLES = 1_000_000
NUM_REAL_PARTICLES = 100_000_000_000
GRID_HALF_WIDTH_M = 0.100
GRID_POINTS = 512
DELTA_Z_M = 1.0
SC_LENGTH_M = 0.1
KINETIC_ENERGY_EV_U = 33.2e6
ELEMENTARY_CHARGE_C = 1.602176634e-19
RELATIVE_ERROR_MINIMUM_THEORY_PEAK_FRACTION = 0.2


@dataclass(frozen=True)
class EllipticCase:
    distribution: str
    title: str
    random_seed: int
    scale_x_m: float
    scale_y_m: float
    scale_names: tuple[str, str]
    theory_factory: Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]]

    @property
    def case_name(self) -> str:
        return f"elliptic_{self.distribution}_free_space_fft"

    @property
    def configuration_name(self) -> str:
        return f"elliptic_{self.distribution}_free_space_fft"

    @property
    def default_run_dir(self) -> Path:
        return ROOT / "output" / self.case_name / "run_001"

    @property
    def emittance_x_m_rad(self) -> float:
        rms_size = 0.5 * self.scale_x_m if self.distribution == "kv" else self.scale_x_m
        return rms_size**2

    @property
    def emittance_y_m_rad(self) -> float:
        rms_size = 0.5 * self.scale_y_m if self.distribution == "kv" else self.scale_y_m
        return rms_size**2


SIGMA_X_M = 0.025
SIGMA_Y_M = 0.015
SEMI_AXIS_X_M = 0.060
SEMI_AXIS_Y_M = 0.035


CASES = {
    "gaussian": EllipticCase(
        distribution="gaussian",
        title="Elliptic Gaussian FFT",
        random_seed=20260909,
        scale_x_m=SIGMA_X_M,
        scale_y_m=SIGMA_Y_M,
        scale_names=("sigma_x_m", "sigma_y_m"),
        theory_factory=lambda x, y, charge: gaussian_elliptic_field(
            x, y, charge, SIGMA_X_M, SIGMA_Y_M
        ),
    ),
    "kv": EllipticCase(
        distribution="kv",
        title="Elliptic KV (uniform spatial projection) FFT",
        random_seed=20260910,
        scale_x_m=SEMI_AXIS_X_M,
        scale_y_m=SEMI_AXIS_Y_M,
        scale_names=("semi_axis_x_m", "semi_axis_y_m"),
        theory_factory=lambda x, y, charge: uniform_elliptic_field(
            x, y, charge, SEMI_AXIS_X_M, SEMI_AXIS_Y_M
        ),
    ),
}


def _parameters(case: EllipticCase) -> dict:
    return {
        "case": case.case_name,
        "distribution": case.distribution,
        "random_seed": case.random_seed,
        "num_macro_particles": NUM_MACRO_PARTICLES,
        "num_real_particles": NUM_REAL_PARTICLES,
        case.scale_names[0]: case.scale_x_m,
        case.scale_names[1]: case.scale_y_m,
        "emittance_x_m_rad": case.emittance_x_m_rad,
        "emittance_y_m_rad": case.emittance_y_m_rad,
        "scan_radial_step_m": SCAN_RADIAL_STEP_M,
        "grid_half_width_m": GRID_HALF_WIDTH_M,
        "grid_points": GRID_POINTS,
        "delta_z_m": DELTA_Z_M,
        "sc_length_m": SC_LENGTH_M,
        "solver": "fft_free_space",
        "deposition_method": "CIC",
        "space_charge_configuration": case.configuration_name,
    }


def _write_input(case: EllipticCase, run_dir: Path) -> Path:
    input_dir = run_dir / "input"
    simulation_dir = run_dir / "simulation"
    input_dir.mkdir(parents=True, exist_ok=True)
    bunch = BunchConfig(
        kinetic_energy=KINETIC_ENERGY_EV_U,
        num_real_particles=NUM_REAL_PARTICLES,
        num_macro_particles=NUM_MACRO_PARTICLES,
        is_load_from_file=False,
        beta_x=1.0,
        beta_y=1.0,
        emit_x=case.emittance_x_m_rad,
        emit_y=case.emittance_y_m_rad,
        sigma_z=0.1,
        dp=1.0e-6,
        dist_trans=case.distribution,
        dist_longi="coasting",
        save_init_dist=False,
    )
    main = MainConfig(
        beam_name=case.case_name,
        num_turns=1,
        backend="cpu",
        particle_precision="float64",
        circumference=100.0,
        output_dir=str(simulation_dir.resolve()),
        is_plot=False,
    )
    sequence = Sequence()
    # Physical acceptance is independent of the field solver. Same-s Marker
    # runs before SpaceCharge; SpaceCharge filters the current live tags.
    sequence.add("transverse_acceptance", MarkerElement(
        s=0.0, aperture_type="rectangle",
        aperture_value=[GRID_HALF_WIDTH_M, GRID_HALF_WIDTH_M]))
    sequence.add(
        "injection",
        InjectionItem(harmonic_number=1, random_seed=case.random_seed, bunches=[bunch]),
    )
    sequence.add(
        "slicer",
        Slicer(
            s=0.0,
            slice_set="space_charge",
            slice_model="equal_length",
            num_slices=1,
            z_range_mode="explicit",
            explicit={"z min": -0.5 * DELTA_Z_M, "z max": 0.5 * DELTA_Z_M},
            # The field HDF5 is sufficient for analysis; do not duplicate the
            # million particles in a Slicer diagnostic file.
            save_turns=[],
        ),
    )
    space_charge_command = SpaceCharge(
        s=0.0,
        configuration=case.configuration_name,
        sc_length=SC_LENGTH_M,
        save_field=True,
        save_potential=True,
        save_density=True,
        save_turns=[[0]],
    )
    sequence.add("space_charge", space_charge_command)
    input_path = input_dir / "beam0.json"
    space_charge = SpaceChargeConfig(
        enabled=True,
        configurations={
            case.configuration_name: SpaceChargeResourceConfig(
                slice_set="space_charge",
                nx=GRID_POINTS,
                ny=GRID_POINTS,
                grid_width_x=2.0 * GRID_HALF_WIDTH_M,
                grid_width_y=2.0 * GRID_HALF_WIDTH_M,
                solver='fft_free_space',
                deposition_method="CIC",
            )
        },
    )
    generate_input(main, sequence, str(input_path), space_charge=space_charge)
    _validate_generated_input(input_path, space_charge, space_charge_command, bunch)
    (run_dir / "parameters.json").write_text(
        json.dumps(_parameters(case), indent=2), encoding="utf-8"
    )
    return input_path


def simulate(case: EllipticCase, run_dir: Path) -> Path:
    input_path = _write_input(case, run_dir)
    pass_main(str(input_path))
    files = sorted((run_dir / "simulation").rglob("*.h5"), key=lambda item: item.stat().st_mtime)
    if not files:
        raise AssertionError(f"PASS produced no SpaceCharge HDF5 snapshot under {run_dir / 'simulation'}")
    field_file = files[-1]
    manifest = {
        "field_file": str(field_file.relative_to(run_dir)),
        "input_file": str(input_path.relative_to(run_dir)),
    }
    (run_dir / "simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[sim] saved {field_file}")
    return field_file


def _field_file(run_dir: Path) -> Path:
    manifest_path = run_dir / "simulation_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        field_file = run_dir / manifest["field_file"]
        if field_file.is_file():
            return field_file
        raise FileNotFoundError(f"manifest points to missing field file: {field_file}")
    files = sorted((run_dir / "simulation").rglob("*.h5"), key=lambda item: item.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no HDF5 field snapshot under {run_dir / 'simulation'}; run with 'sim' first")
    return files[-1]


def _relative_evaluation_mask(
    case: EllipticCase,
    x: np.ndarray,
    y: np.ndarray,
    theory_magnitude: np.ndarray,
    theory_peak: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    # The vector field vanishes at the origin, where macro-particle shot noise
    # makes a pointwise relative ratio ill-conditioned.  The independent
    # peak-normalized check below still covers every probe, including the core.
    mask = theory_magnitude >= RELATIVE_ERROR_MINIMUM_THEORY_PEAK_FRACTION * theory_peak
    if case.distribution != "kv":
        return mask, None
    normalized_radius = np.sqrt((x / case.scale_x_m) ** 2 + (y / case.scale_y_m) ** 2)
    normalized_edge_half_width = 2.0 * max(
        float(x_grid[1] - x_grid[0]) / case.scale_x_m,
        float(y_grid[1] - y_grid[0]) / case.scale_y_m,
    )
    mask &= np.abs(normalized_radius - 1.0) > normalized_edge_half_width
    return mask, normalized_edge_half_width


def analyse(case: EllipticCase, run_dir: Path) -> dict:
    field_file = _field_file(run_dir)
    analysis_dir = run_dir / "analysis"
    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(field_file, "r") as handle:
        x_grid = np.asarray(handle["x"], dtype=float)
        y_grid = np.asarray(handle["y"], dtype=float)
        delta_z = float(np.asarray(handle["delta_z"])[0])
        charge = float(np.asarray(handle["slice_charge"])[0])
        ex_grid = np.asarray(handle["integrated_Ex"][0], dtype=float)
        ey_grid = np.asarray(handle["integrated_Ey"][0], dtype=float)
        density = np.asarray(handle["charge_density"][0], dtype=float)
        potential = np.asarray(handle["potential"][0], dtype=float)
        attrs = {key: handle.attrs[key] for key in handle.attrs}

    x_probe, y_probe, angle_deg = _scan_points(max_radius_m=0.095)
    grid_metrics, gauss_frame, residual_fields = _grid_consistency_checks(
        x_grid, y_grid, density, potential, ex_grid, ey_grid, charge
    )
    gauss_frame.to_csv(analysis_dir / "gauss_law.csv", index=False)
    points = (y_probe, x_probe)
    sample_ex = RegularGridInterpolator((y_grid, x_grid), ex_grid, bounds_error=True)(points)
    sample_ey = RegularGridInterpolator((y_grid, x_grid), ey_grid, bounds_error=True)(points)
    theory_ex, theory_ey = case.theory_factory(x_probe, y_probe, charge)
    angle_rad = np.deg2rad(angle_deg)
    sample_er = sample_ex * np.cos(angle_rad) + sample_ey * np.sin(angle_rad)
    sample_et = -sample_ex * np.sin(angle_rad) + sample_ey * np.cos(angle_rad)
    theory_er = theory_ex * np.cos(angle_rad) + theory_ey * np.sin(angle_rad)
    theory_et = -theory_ex * np.sin(angle_rad) + theory_ey * np.cos(angle_rad)
    error = np.hypot(sample_ex - theory_ex, sample_ey - theory_ey)
    theory_magnitude = np.hypot(theory_ex, theory_ey)
    theory_peak = float(theory_magnitude.max())
    relative_error = np.divide(
        error,
        theory_magnitude,
        out=np.zeros_like(error),
        where=theory_magnitude > 0.0,
    )
    relative_evaluation, normalized_edge_half_width = _relative_evaluation_mask(
        case, x_probe, y_probe, theory_magnitude, theory_peak, x_grid, y_grid
    )
    exclusion_reason = np.where(
        relative_evaluation,
        "included",
        "field_below_relative_error_threshold",
    ).astype(object)
    exclusion_reason[theory_magnitude == 0.0] = "zero_theory_at_origin"
    if case.distribution == "kv":
        normalized_radius = np.sqrt(
            (x_probe / case.scale_x_m) ** 2 + (y_probe / case.scale_y_m) ** 2
        )
        near_edge = (
            np.abs(normalized_radius - 1.0) <= normalized_edge_half_width
        )
        exclusion_reason[near_edge] = "near_kv_density_edge"
    peak_normalized_error = error / theory_peak
    frame = pd.DataFrame(
        {
            "angle_deg": angle_deg,
            "x_m": x_probe,
            "y_m": y_probe,
            "r_m": np.hypot(x_probe, y_probe),
            "pic_Ex_v": sample_ex,
            "pic_Ey_v": sample_ey,
            "pic_Er_v": sample_er,
            "pic_Etheta_v": sample_et,
            "theory_Ex_v": theory_ex,
            "theory_Ey_v": theory_ey,
            "theory_Er_v": theory_er,
            "theory_Etheta_v": theory_et,
            "theory_field_magnitude_v": theory_magnitude,
            "absolute_field_error_v": error,
            "relative_field_error": relative_error,
            "included_in_relative_evaluation": relative_evaluation,
            "relative_error_exclusion_reason": exclusion_reason,
        }
    )
    frame.to_csv(analysis_dir / "field_scan.csv", index=False)
    np.savez(
        analysis_dir / "field_data.npz",
        x_grid=x_grid,
        y_grid=y_grid,
        charge_density=density,
        potential=potential,
        integrated_Ex=ex_grid,
        integrated_Ey=ey_grid,
        **residual_fields,
        **{name: frame[name].to_numpy() for name in frame.columns},
    )

    nominal_charge = NUM_REAL_PARTICLES * ELEMENTARY_CHARGE_C
    summary = {
        **_parameters(case),
        "field_file": str(field_file),
        "solver": str(attrs.get("solver", "fft_free_space")),
        "deposition_method": str(attrs.get("deposition_method", "CIC")),
        "slice_charge_c": charge,
        "nominal_charge_c": nominal_charge,
        "deposited_charge_fraction": charge / nominal_charge,
        "delta_z_m": delta_z,
        "field_units": "slice-integrated transverse field (V)",
        "potential_units": str(attrs.get("potential_units", "V m")),
        "relative_error_minimum_theory_peak_fraction": RELATIVE_ERROR_MINIMUM_THEORY_PEAK_FRACTION,
        "kv_normalized_edge_exclusion_half_width": normalized_edge_half_width,
        "max_relative_field_error_evaluation_region": float(
            relative_error[relative_evaluation].max()
        ),
        "rms_relative_field_error_evaluation_region": float(
            np.sqrt(np.mean(relative_error[relative_evaluation] ** 2))
        ),
        "max_peak_normalized_field_error": float(peak_normalized_error.max()),
        "rms_peak_normalized_field_error": float(np.sqrt(np.mean(peak_normalized_error**2))),
        "theory_peak_field_v": theory_peak,
        "relative_evaluation_probe_count": int(relative_evaluation.sum()),
        "relative_error_tolerance": 0.03,
        "peak_normalized_error_tolerance": 0.01,
        **grid_metrics,
        "charge_conservation_tolerance": 1.0e-9,
        "potential_field_relative_l2_tolerance": 1.0e-3,
        "poisson_relative_l2_tolerance": 0.15,
        "gauss_law_relative_error_tolerance": 0.03,
        "field_inversion_rms_over_peak_tolerance": 0.005,
        "potential_inversion_rms_over_range_tolerance": 0.001,
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_results(
        frame,
        x_grid,
        y_grid,
        density,
        potential,
        ex_grid,
        ey_grid,
        figures_dir,
        summary["potential_units"],
        gauss_frame,
        residual_fields,
        case_title=case.title,
        tangential_is_residual=False,
    )
    checks = (
        ("relative field error", summary["max_relative_field_error_evaluation_region"], summary["relative_error_tolerance"]),
        ("peak-normalized field error", summary["max_peak_normalized_field_error"], summary["peak_normalized_error_tolerance"]),
        ("charge-conservation relative error", grid_metrics["charge_conservation_relative_error"], summary["charge_conservation_tolerance"]),
        ("potential-field relative L2 residual", grid_metrics["potential_field_relative_l2_residual"], summary["potential_field_relative_l2_tolerance"]),
        ("Poisson relative L2 residual", grid_metrics["poisson_relative_l2_residual"], summary["poisson_relative_l2_tolerance"]),
        ("Gauss-law relative error", grid_metrics["max_gauss_law_relative_error"], summary["gauss_law_relative_error_tolerance"]),
        ("field inversion-symmetry residual", grid_metrics["field_inversion_rms_over_peak"], summary["field_inversion_rms_over_peak_tolerance"]),
        ("potential inversion-symmetry residual", grid_metrics["potential_inversion_rms_over_range"], summary["potential_inversion_rms_over_range_tolerance"]),
    )
    failures = [
        f"{name}: measured={measured:.8g}, tolerance={tolerance:.8g}"
        for name, measured, tolerance in checks
        if measured > tolerance
    ]
    if failures:
        raise AssertionError(
            f"{case.title} validation failed:\n- "
            + "\n- ".join(failures)
            + f"\nsee {analysis_dir / 'summary.json'}"
        )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distribution", choices=tuple(CASES))
    parser.add_argument("mode", choices=("sim", "ana", "simana"))
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    case = CASES[args.distribution]
    run_dir = (args.run_dir or case.default_run_dir).resolve()
    if args.mode in {"sim", "simana"}:
        simulate(case, run_dir)
    if args.mode in {"ana", "simana"}:
        analyse(case, run_dir)


if __name__ == "__main__":
    main()
