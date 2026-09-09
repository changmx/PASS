"""Round KV transverse field test with separated simulation/analysis modes.

Run from the repository root, for example::

    python -m tests.integration.space_charge.test_round_kv_free_space_fft simana
    python -m tests.integration.space_charge.test_round_kv_free_space_fft ana

The 4-D KV source projects to a uniform area density inside a circle.  The simulation stores the
SpaceCharge grid snapshot once; analysis samples that snapshot without
rerunning the million-macro-particle simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from PASS.main import main as pass_main
from PASS.para.api import generate_input
from PASS.para.schema.bunch import BunchConfig, InjectionItem
from PASS.para.schema.main import MainConfig
from PASS.para.schema.sequence import Sequence
from PASS.para.schema.slicer import Slicer
from PASS.para.schema.space_charge import (
    SpaceCharge,
    SpaceChargeConfig,
    SpaceChargeResourceConfig,
)
from tests.integration.space_charge.test_round_gaussian_free_space_fft import (
    SCAN_RADIAL_STEP_M,
    _grid_consistency_checks,
    _plot_results,
    _scan_points,
    _validate_generated_input,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = ROOT / "output" / "round_kv_free_space_fft" / "run_001"
RANDOM_SEED = 20260908
NUM_MACRO_PARTICLES = 1_000_000
NUM_REAL_PARTICLES = 100_000_000_000
RADIUS_M = 0.060
RMS_SIZE_M = 0.5 * RADIUS_M
EMITTANCE_M_RAD = RMS_SIZE_M**2
GRID_HALF_WIDTH_M = 0.100
GRID_POINTS = 512
DELTA_Z_M = 1.0
SC_LENGTH_M = 0.1
KINETIC_ENERGY_EV_U = 33.2e6
ELEMENTARY_CHARGE_C = 1.602176634e-19
EPSILON_0 = 8.8541878128e-12
SPACE_CHARGE_CONFIGURATION = "round_kv_free_space_fft"


def _parameters() -> dict:
    return {
        "random_seed": RANDOM_SEED,
        "num_macro_particles": NUM_MACRO_PARTICLES,
        "num_real_particles": NUM_REAL_PARTICLES,
        "radius_m": RADIUS_M,
        "rms_size_m": RMS_SIZE_M,
        "emittance_m_rad": EMITTANCE_M_RAD,
        "transverse_distribution": "kv",
        "grid_half_width_m": GRID_HALF_WIDTH_M,
        "grid_points": GRID_POINTS,
        "delta_z_m": DELTA_Z_M,
        "sc_length_m": SC_LENGTH_M,
        "solver": "fft_free_space",
        "deposition_method": "CIC",
        "space_charge_configuration": SPACE_CHARGE_CONFIGURATION,
    }


def _write_input(run_dir: Path) -> Path:
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
        emit_x=EMITTANCE_M_RAD,
        emit_y=EMITTANCE_M_RAD,
        sigma_z=0.1,
        dp=1.0e-6,
        dist_trans="kv",
        dist_longi="coasting",
        save_init_dist=False,
    )
    main = MainConfig(
        beam_name="round-kv-field-scan",
        num_turns=1,
        backend="cpu",
        particle_precision="float64",
        circumference=100.0,
        output_dir=str(simulation_dir.resolve()),
        is_plot=False,
    )
    sequence = Sequence()
    sequence.add("injection", InjectionItem(harmonic_number=1, random_seed=RANDOM_SEED, bunches=[bunch]))
    sequence.add(
        "slicer",
        Slicer(
            s=0.0,
            slice_set="space_charge",
            slice_model="equal_length",
            num_slices=1,
            z_range_mode="explicit",
            explicit={"z min": -0.5 * DELTA_Z_M, "z max": 0.5 * DELTA_Z_M},
            # Only the SpaceCharge HDF5 snapshot is used by this validation.
            # Avoid writing a second million-particle Slicer snapshot.
            save_turns=[],
        ),
    )
    space_charge_command = SpaceCharge(
        s=0.0,
        configuration=SPACE_CHARGE_CONFIGURATION,
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
            SPACE_CHARGE_CONFIGURATION: SpaceChargeResourceConfig(
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
    (run_dir / "parameters.json").write_text(json.dumps(_parameters(), indent=2), encoding="utf-8")
    return input_path


def simulate(run_dir: Path) -> Path:
    """Run PASS once and record the exact resulting HDF5 snapshot."""
    input_path = _write_input(run_dir)
    pass_main(str(input_path))
    files = sorted((run_dir / "simulation").rglob("*.h5"), key=lambda item: item.stat().st_mtime)
    if not files:
        raise AssertionError(f"PASS produced no SpaceCharge HDF5 snapshot under {run_dir / 'simulation'}")
    field_file = files[-1]
    manifest = {
        "field_file": str(field_file.relative_to(run_dir)),
        "input_file": str(input_path.relative_to(run_dir)),
    }
    (run_dir / "simulation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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


def _kv_round_projected_field(
    x: np.ndarray,
    y: np.ndarray,
    charge: float,
    radius_m: float = RADIUS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the analytic field of the KV beam's uniform x-y projection."""
    r = np.hypot(x, y)
    inside = r <= radius_m
    er = np.empty_like(r)
    er[inside] = charge * r[inside] / (2.0 * np.pi * EPSILON_0 * radius_m**2)
    er[~inside] = charge / (2.0 * np.pi * EPSILON_0 * r[~inside])
    ex = np.divide(er * x, r, out=np.zeros_like(r), where=r > 0.0)
    ey = np.divide(er * y, r, out=np.zeros_like(r), where=r > 0.0)
    return ex, ey, er


def analyse(run_dir: Path) -> dict:
    """Compare one saved FFT grid against the uniform-disk analytic field."""
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

    x_probe, y_probe, angle_deg = _scan_points()
    grid_metrics, gauss_frame, residual_fields = _grid_consistency_checks(
        x_grid, y_grid, density, potential, ex_grid, ey_grid, charge
    )
    gauss_frame.to_csv(analysis_dir / "gauss_law.csv", index=False)
    sample_ex = RegularGridInterpolator((y_grid, x_grid), ex_grid, bounds_error=True)((y_probe, x_probe))
    sample_ey = RegularGridInterpolator((y_grid, x_grid), ey_grid, bounds_error=True)((y_probe, x_probe))
    theory_ex, theory_ey, theory_er = _kv_round_projected_field(x_probe, y_probe, charge)
    angle_rad = np.deg2rad(angle_deg)
    sample_er = sample_ex * np.cos(angle_rad) + sample_ey * np.sin(angle_rad)
    sample_et = -sample_ex * np.sin(angle_rad) + sample_ey * np.cos(angle_rad)
    error = np.hypot(sample_ex - theory_ex, sample_ey - theory_ey)
    relative_error = np.divide(error, theory_er, out=np.zeros_like(error), where=theory_er > 0.0)
    radii = np.hypot(x_probe, y_probe)
    nonzero = theory_er > 0.0
    # Very near the origin the analytic field tends to zero, so finite-particle
    # shot noise dominates a relative-error ratio.  Retain the peak-normalized
    # absolute-error check there and start the relative check at 10 mm.
    relative_error_min_radius_m = 0.010
    edge_exclusion_half_width_m = 2.0 * max(
        float(x_grid[1] - x_grid[0]), float(y_grid[1] - y_grid[0])
    )
    away_from_edge = np.abs(radii - RADIUS_M) > edge_exclusion_half_width_m
    relative_evaluation = nonzero & (radii >= relative_error_min_radius_m - 1.0e-12) & away_from_edge
    exclusion_reason = np.full(radii.shape, "included", dtype=object)
    exclusion_reason[~away_from_edge] = "near_kv_density_edge"
    exclusion_reason[radii < relative_error_min_radius_m - 1.0e-12] = (
        "core_relative_error_ill_conditioned"
    )
    exclusion_reason[~nonzero] = "zero_theory_at_origin"
    theory_peak = float(theory_er.max())
    peak_normalized_error = error / theory_peak
    frame = pd.DataFrame(
        {
            "angle_deg": angle_deg,
            "x_m": x_probe,
            "y_m": y_probe,
            "r_m": radii,
            "pic_Ex_v": sample_ex,
            "pic_Ey_v": sample_ey,
            "pic_Er_v": sample_er,
            "pic_Etheta_v": sample_et,
            "theory_Ex_v": theory_ex,
            "theory_Ey_v": theory_ey,
            "theory_Er_v": theory_er,
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

    max_relative = float(relative_error[nonzero].max())
    max_relative_evaluation = float(relative_error[relative_evaluation].max())
    rms_relative = float(np.sqrt(np.mean(relative_error[nonzero] ** 2)))
    max_peak_normalized = float(peak_normalized_error.max())
    azimuthal_rms = float(np.sqrt(np.mean(sample_et[nonzero] ** 2)))
    azimuthal_max = float(np.max(np.abs(sample_et[nonzero])))
    nominal_charge = NUM_REAL_PARTICLES * ELEMENTARY_CHARGE_C
    summary = {
        "case": "round_kv_free_space_fft",
        "transverse_distribution": "kv",
        "field_file": str(field_file),
        "solver": str(attrs.get("solver", "fft_free_space")),
        "deposition_method": str(attrs.get("deposition_method", "CIC")),
        "num_macro_particles": NUM_MACRO_PARTICLES,
        "num_real_particles": NUM_REAL_PARTICLES,
        "random_seed": RANDOM_SEED,
        "slice_charge_c": charge,
        "nominal_charge_c": nominal_charge,
        "deposited_charge_fraction": charge / nominal_charge,
        "theory_charge_c": charge,
        "radius_m": RADIUS_M,
        "rms_size_m": RMS_SIZE_M,
        "emittance_m_rad": EMITTANCE_M_RAD,
        "scan_radial_step_m": SCAN_RADIAL_STEP_M,
        "grid_points": GRID_POINTS,
        "grid_half_width_m": GRID_HALF_WIDTH_M,
        "delta_z_m": delta_z,
        "field_units": "slice-integrated transverse field (V)",
        "potential_units": str(attrs.get("potential_units", "V m")),
        "max_relative_field_error": max_relative,
        "relative_error_min_radius_m": relative_error_min_radius_m,
        "edge_exclusion_half_width_m": edge_exclusion_half_width_m,
        "max_relative_field_error_evaluation_region": max_relative_evaluation,
        "rms_relative_field_error": rms_relative,
        "max_peak_normalized_field_error": max_peak_normalized,
        "theory_peak_radial_field_v": theory_peak,
        "azimuthal_field_rms_v": azimuthal_rms,
        "azimuthal_field_max_abs_v": azimuthal_max,
        "azimuthal_field_rms_over_theory_peak": azimuthal_rms / theory_peak,
        "azimuthal_field_max_abs_over_theory_peak": azimuthal_max / theory_peak,
        "nonzero_probe_count": int(nonzero.sum()),
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
        "azimuthal_field_rms_over_theory_peak_tolerance": 0.005,
        "azimuthal_field_max_abs_over_theory_peak_tolerance": 0.01,
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
        case_title="Round KV (uniform spatial projection) FFT",
    )
    checks = (
        (
            "relative field error away from the origin and disk edge",
            max_relative_evaluation,
            summary["relative_error_tolerance"],
        ),
        ("peak-normalized absolute field error", max_peak_normalized, summary["peak_normalized_error_tolerance"]),
        ("charge-conservation relative error", grid_metrics["charge_conservation_relative_error"], summary["charge_conservation_tolerance"]),
        ("potential-field relative L2 residual", grid_metrics["potential_field_relative_l2_residual"], summary["potential_field_relative_l2_tolerance"]),
        ("Poisson relative L2 residual", grid_metrics["poisson_relative_l2_residual"], summary["poisson_relative_l2_tolerance"]),
        ("Gauss-law relative error", grid_metrics["max_gauss_law_relative_error"], summary["gauss_law_relative_error_tolerance"]),
        ("field inversion-symmetry residual", grid_metrics["field_inversion_rms_over_peak"], summary["field_inversion_rms_over_peak_tolerance"]),
        ("potential inversion-symmetry residual", grid_metrics["potential_inversion_rms_over_range"], summary["potential_inversion_rms_over_range_tolerance"]),
        ("azimuthal-field RMS residual", azimuthal_rms / theory_peak, summary["azimuthal_field_rms_over_theory_peak_tolerance"]),
        ("azimuthal-field maximum residual", azimuthal_max / theory_peak, summary["azimuthal_field_max_abs_over_theory_peak_tolerance"]),
    )
    failures = [
        f"{name}: measured={measured:.8g}, tolerance={tolerance:.8g}"
        for name, measured, tolerance in checks
        if measured > tolerance
    ]
    if failures:
        raise AssertionError(
            "round KV FFT validation failed:\n- "
            + "\n- ".join(failures)
            + f"\nsee {analysis_dir / 'summary.json'}"
        )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sim", "ana", "simana"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.mode in {"sim", "simana"}:
        simulate(run_dir)
    if args.mode in {"ana", "simana"}:
        analyse(run_dir)


def test_full_workflow(sc_workflow):
    """Generate the full source, run PASS, and enforce the existing analysis checks."""
    sc_workflow("round_kv_free_space_fft")


if __name__ == "__main__":
    main()
