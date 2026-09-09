"""Round Gaussian transverse field test with separated simulation/analysis modes.

Run from the repository root, for example::

    python -m tests.integration.space_charge.test_round_gaussian_free_space_fft simana
    python -m tests.integration.space_charge.test_round_gaussian_free_space_fft ana

The simulation stores the SpaceCharge grid snapshot once.  Analysis samples that
snapshot at the requested rays, so plotting changes do not rerun the million-
macro-particle simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

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


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = ROOT / "output" / "round_gaussian_free_space_fft" / "run_001"
RANDOM_SEED = 20260904
NUM_MACRO_PARTICLES = 1_000_000
NUM_REAL_PARTICLES = 100_000_000_000
SIGMA_M = 0.030
EMITTANCE_M_RAD = SIGMA_M**2
GRID_HALF_WIDTH_M = 0.100
GRID_POINTS = 512
SCAN_RADIAL_STEP_M = 0.00025
SCAN_MARKER_SIZE_PT = 2.0
DELTA_Z_M = 1.0
SC_LENGTH_M = 0.1
KINETIC_ENERGY_EV_U = 33.2e6
ELEMENTARY_CHARGE_C = 1.602176634e-19
EPSILON_0 = 8.8541878128e-12
SPACE_CHARGE_CONFIGURATION = "round_gaussian_free_space_fft"


def _parameters() -> dict:
    return {
        "random_seed": RANDOM_SEED,
        "num_macro_particles": NUM_MACRO_PARTICLES,
        "num_real_particles": NUM_REAL_PARTICLES,
        "sigma_m": SIGMA_M,
        "emittance_m_rad": EMITTANCE_M_RAD,
        "transverse_distribution": "gaussian",
        "scan_radial_step_m": SCAN_RADIAL_STEP_M,
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
        dist_trans="gaussian",
        dist_longi="coasting",
        save_init_dist=False,
    )
    main = MainConfig(
        beam_name="round-gaussian-field-scan",
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


def _validate_generated_input(
    input_path: Path,
    space_charge: SpaceChargeConfig,
    command: SpaceCharge,
    bunch: BunchConfig,
    command_name: str = "space_charge",
) -> None:
    """Require generated phase space and the current space-charge input layout."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    obsolete = {"Is space charge", "Space-charge simulation parameters"}.intersection(payload)
    if obsolete:
        raise AssertionError(f"generated input contains obsolete space-charge keys: {sorted(obsolete)}")
    expected_global = space_charge.model_dump(by_alias=True)
    measured_global = payload.get("Space charge")
    if measured_global != expected_global:
        raise AssertionError(
            "generated top-level Space charge block differs from SpaceChargeConfig: "
            f"measured={measured_global!r}, expected={expected_global!r}"
        )
    expected_command = command.model_dump(by_alias=True)
    measured_command = payload.get("Sequence", {}).get(command_name)
    if measured_command != expected_command:
        raise AssertionError(
            "generated SpaceCharge command differs from the command schema: "
            f"measured={measured_command!r}, expected={expected_command!r}"
        )
    measured_bunch = payload.get("Sequence", {}).get("injection", {}).get("bunch0")
    expected_bunch = bunch.model_dump(by_alias=True)
    if measured_bunch != expected_bunch:
        raise AssertionError(
            "generated Injection bunch differs from BunchConfig: "
            f"measured={measured_bunch!r}, expected={expected_bunch!r}"
        )
    if measured_bunch["Is Load Distribution from File"]:
        raise AssertionError("field validation must generate phase space, not load a distribution")
    if measured_bunch["Distribution File Path"]:
        raise AssertionError("generated phase-space validation must not set a distribution file")
    for key in ("Emittance x (m'rad)", "Emittance y (m'rad)"):
        if measured_bunch[key] <= 0.0:
            raise AssertionError(f"{key} must be positive so x/px and y/py are generated")


def simulate(run_dir: Path) -> Path:
    """Run PASS once and record the exact resulting HDF5 snapshot."""
    input_path = _write_input(run_dir)
    pass_main(str(input_path))
    files = sorted((run_dir / "simulation").rglob("*.h5"), key=lambda item: item.stat().st_mtime)
    if not files:
        raise AssertionError(f"PASS produced no SpaceCharge HDF5 snapshot under {run_dir / 'simulation'}")
    field_file = files[-1]
    manifest = {"field_file": str(field_file.relative_to(run_dir)), "input_file": str(input_path.relative_to(run_dir))}
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


def _scan_points(
    center_x_m: float = 0.0,
    center_y_m: float = 0.0,
    max_radius_m: float = 0.095,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.deg2rad(np.arange(0.0, 360.0, 22.5))
    # 0.25 mm is ten times denser than the original 2.5 mm scan and is also
    # smaller than the 512-point field-grid spacing (~0.391 mm).
    radii = np.arange(0.0, max_radius_m + 0.5 * SCAN_RADIAL_STEP_M, SCAN_RADIAL_STEP_M)
    radius, angle = np.meshgrid(radii, angles, indexing="ij")
    return (
        (center_x_m + radius * np.cos(angle)).ravel(),
        (center_y_m + radius * np.sin(angle)).ravel(),
        np.rad2deg(angle).ravel(),
    )


def _gaussian_field(
    x: np.ndarray,
    y: np.ndarray,
    charge: float,
    center_x_m: float = 0.0,
    center_y_m: float = 0.0,
    sigma_m: float = SIGMA_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    relative_x = x - center_x_m
    relative_y = y - center_y_m
    r = np.hypot(relative_x, relative_y)
    enclosed = -np.expm1(-(r / sigma_m) ** 2 / 2.0)
    er = np.divide(charge * enclosed, 2.0 * np.pi * EPSILON_0 * r, out=np.zeros_like(r), where=r > 0.0)
    ex = np.divide(er * relative_x, r, out=np.zeros_like(r), where=r > 0.0)
    ey = np.divide(er * relative_y, r, out=np.zeros_like(r), where=r > 0.0)
    return ex, ey, er, np.zeros_like(er)


def _grid_consistency_checks(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    density: np.ndarray,
    potential: np.ndarray,
    ex_grid: np.ndarray,
    ey_grid: np.ndarray,
    charge: float,
    center_x_m: float = 0.0,
    center_y_m: float = 0.0,
) -> tuple[dict, pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate charge, field-potential, Poisson, Gauss, and symmetry residuals."""
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    density_charge = float(np.sum(density) * dx * dy)
    charge_error = abs(density_charge - charge) / abs(charge)

    dphi_dy, dphi_dx = np.gradient(potential, y_grid, x_grid, edge_order=2)
    potential_field_residual = np.hypot(ex_grid + dphi_dx, ey_grid + dphi_dy)
    field_magnitude = np.hypot(ex_grid, ey_grid)
    field_peak = float(np.max(field_magnitude))

    divergence = np.gradient(ex_grid, x_grid, axis=1, edge_order=2) + np.gradient(
        ey_grid, y_grid, axis=0, edge_order=2
    )
    poisson_source = density / EPSILON_0
    poisson_residual = divergence - poisson_source

    margin = 2
    interior = np.s_[margin:-margin, margin:-margin]
    potential_field_relative_l2 = float(
        np.linalg.norm(potential_field_residual[interior]) / np.linalg.norm(field_magnitude[interior])
    )
    poisson_relative_l2 = float(
        np.linalg.norm(poisson_residual[interior]) / np.linalg.norm(poisson_source[interior])
    )

    xx, yy = np.meshgrid(x_grid, y_grid)
    mirrored_points = np.column_stack(
        ((2.0 * center_y_m - yy).ravel(), (2.0 * center_x_m - xx).ravel())
    )
    mirrored_ex = RegularGridInterpolator(
        (y_grid, x_grid), ex_grid, bounds_error=False, fill_value=np.nan
    )(mirrored_points).reshape(ex_grid.shape)
    mirrored_ey = RegularGridInterpolator(
        (y_grid, x_grid), ey_grid, bounds_error=False, fill_value=np.nan
    )(mirrored_points).reshape(ey_grid.shape)
    mirrored_potential = RegularGridInterpolator(
        (y_grid, x_grid), potential, bounds_error=False, fill_value=np.nan
    )(mirrored_points).reshape(potential.shape)
    symmetry_valid = np.isfinite(mirrored_ex) & np.isfinite(mirrored_ey) & np.isfinite(mirrored_potential)
    field_inversion_residual = np.full_like(field_magnitude, np.nan)
    field_inversion_residual[symmetry_valid] = np.hypot(
        ex_grid[symmetry_valid] + mirrored_ex[symmetry_valid],
        ey_grid[symmetry_valid] + mirrored_ey[symmetry_valid],
    )
    field_inversion_rms_over_peak = float(
        np.sqrt(np.mean(field_inversion_residual[symmetry_valid] ** 2)) / field_peak
    )
    potential_range = float(np.ptp(potential))
    potential_inversion_rms_over_range = float(
        np.sqrt(np.mean((potential[symmetry_valid] - mirrored_potential[symmetry_valid]) ** 2)) / potential_range
    )

    gauss_rows = []
    for half_width_m in (0.020, 0.040, 0.060, 0.080):
        ix_min = int(np.argmin(np.abs(x_grid - (center_x_m - half_width_m))))
        ix_max = int(np.argmin(np.abs(x_grid - (center_x_m + half_width_m))))
        iy_min = int(np.argmin(np.abs(y_grid - (center_y_m - half_width_m))))
        iy_max = int(np.argmin(np.abs(y_grid - (center_y_m + half_width_m))))
        field_flux = (
            np.trapezoid(ex_grid[iy_min : iy_max + 1, ix_max], y_grid[iy_min : iy_max + 1])
            - np.trapezoid(ex_grid[iy_min : iy_max + 1, ix_min], y_grid[iy_min : iy_max + 1])
            + np.trapezoid(ey_grid[iy_max, ix_min : ix_max + 1], x_grid[ix_min : ix_max + 1])
            - np.trapezoid(ey_grid[iy_min, ix_min : ix_max + 1], x_grid[ix_min : ix_max + 1])
        )
        enclosed_charge = float(np.sum(density[iy_min : iy_max + 1, ix_min : ix_max + 1]) * dx * dy)
        charge_flux = enclosed_charge / EPSILON_0
        relative_error = float((field_flux - charge_flux) / charge_flux)
        gauss_rows.append(
            {
                "half_width_m": half_width_m,
                "field_flux_v_m": field_flux,
                "enclosed_charge_c": enclosed_charge,
                "charge_over_epsilon0_v_m": charge_flux,
                "relative_error": relative_error,
            }
        )
    gauss_frame = pd.DataFrame(gauss_rows)
    metrics = {
        "density_integral_charge_c": density_charge,
        "charge_conservation_relative_error": charge_error,
        "potential_field_relative_l2_residual": potential_field_relative_l2,
        "poisson_relative_l2_residual": poisson_relative_l2,
        "max_gauss_law_relative_error": float(np.max(np.abs(gauss_frame["relative_error"]))),
        "field_inversion_rms_over_peak": field_inversion_rms_over_peak,
        "potential_inversion_rms_over_range": potential_inversion_rms_over_range,
    }
    residual_fields = {
        "potential_field_residual_over_peak": potential_field_residual / field_peak,
        "poisson_residual_over_source_peak": poisson_residual / float(np.max(np.abs(poisson_source))),
        "field_inversion_residual_over_peak": field_inversion_residual / field_peak,
    }
    return metrics, gauss_frame, residual_fields


def analyse(run_dir: Path) -> dict:
    """Sample one saved grid field and write all analysis artifacts."""
    field_file = _field_file(run_dir)
    analysis_dir = run_dir / "analysis"
    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(field_file, "r") as handle:
        x_grid = np.asarray(handle["x"], dtype=float)
        y_grid = np.asarray(handle["y"], dtype=float)
        delta_z = float(np.asarray(handle["delta_z"])[0])
        charge = float(np.asarray(handle["slice_charge"])[0])
        # The 2.5D solver returns the transverse field integrated through the
        # slice (V). The analytic sheet-charge formula uses the same convention;
        # SpaceCharge divides by delta_z only when applying the particle kick.
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
    # HDF5 stores arrays as [y, x], matching the solver's (ny, nx) convention.
    sample_ex = RegularGridInterpolator((y_grid, x_grid), ex_grid, bounds_error=True)((y_probe, x_probe))
    sample_ey = RegularGridInterpolator((y_grid, x_grid), ey_grid, bounds_error=True)((y_probe, x_probe))
    nominal_charge = NUM_REAL_PARTICLES * ELEMENTARY_CHARGE_C
    # The finite mesh clips the unbounded Gaussian tail.  Use the charge that
    # was actually deposited for the field comparison and report the retained
    # fraction separately, so the solver error is not mixed with truncation.
    theory_ex, theory_ey, theory_er, _ = _gaussian_field(x_probe, y_probe, charge)
    sample_er = sample_ex * np.cos(np.deg2rad(angle_deg)) + sample_ey * np.sin(np.deg2rad(angle_deg))
    sample_et = -sample_ex * np.sin(np.deg2rad(angle_deg)) + sample_ey * np.cos(np.deg2rad(angle_deg))
    error = np.hypot(sample_ex - theory_ex, sample_ey - theory_ey)
    relative_error = np.divide(error, theory_er, out=np.zeros_like(error), where=theory_er > 0.0)
    nonzero = theory_er > 0.0
    # With Injection-generated macro particles, the first few millimetres are
    # dominated by the finite-sample field-noise floor while the analytic
    # field tends to zero.  Retain the all-probe peak-normalized check there.
    relative_error_min_radius_m = 0.0075
    radii = np.hypot(x_probe, y_probe)
    relative_evaluation = nonzero & (radii >= relative_error_min_radius_m - 1.0e-12)
    exclusion_reason = np.where(
        ~nonzero,
        "zero_theory_at_origin",
        np.where(relative_evaluation, "included", "core_relative_error_ill_conditioned"),
    )
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
    rms_peak_normalized = float(np.sqrt(np.mean(peak_normalized_error**2)))
    azimuthal_rms = float(np.sqrt(np.mean(sample_et[nonzero] ** 2)))
    azimuthal_max = float(np.max(np.abs(sample_et[nonzero])))
    summary = {
        "case": "round_gaussian_free_space_fft",
        "field_file": str(field_file),
        "solver": str(attrs.get("solver", "fft_free_space")),
        "deposition_method": str(attrs.get("deposition_method", "CIC")),
        "num_macro_particles": NUM_MACRO_PARTICLES,
        "num_real_particles": NUM_REAL_PARTICLES,
        "random_seed": RANDOM_SEED,
        "transverse_distribution": "gaussian",
        "slice_charge_c": charge,
        "nominal_charge_c": nominal_charge,
        "deposited_charge_fraction": charge / nominal_charge,
        "theory_charge_c": charge,
        "sigma_m": SIGMA_M,
        "emittance_m_rad": EMITTANCE_M_RAD,
        "scan_radial_step_m": SCAN_RADIAL_STEP_M,
        "grid_points": GRID_POINTS,
        "grid_half_width_m": GRID_HALF_WIDTH_M,
        "delta_z_m": delta_z,
        "field_units": "slice-integrated transverse field (V)",
        "potential_units": str(attrs.get("potential_units", "V m")),
        "max_relative_field_error": max_relative,
        "relative_error_min_radius_m": relative_error_min_radius_m,
        "max_relative_field_error_evaluation_region": max_relative_evaluation,
        "rms_relative_field_error": rms_relative,
        "max_peak_normalized_field_error": max_peak_normalized,
        "rms_peak_normalized_field_error": rms_peak_normalized,
        "theory_peak_radial_field_v": theory_peak,
        "azimuthal_field_rms_v": azimuthal_rms,
        "azimuthal_field_max_abs_v": azimuthal_max,
        "azimuthal_field_rms_over_theory_peak": azimuthal_rms / theory_peak,
        "azimuthal_field_max_abs_over_theory_peak": azimuthal_max / theory_peak,
        "nonzero_probe_count": int(nonzero.sum()),
        "relative_error_tolerance": 0.03,
        "peak_normalized_error_tolerance": 0.012,
        "rms_peak_normalized_error_tolerance": 0.004,
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
    )
    checks = (
        ("relative field error for r >= 7.5 mm", max_relative_evaluation, summary["relative_error_tolerance"]),
        ("peak-normalized absolute field error", max_peak_normalized, summary["peak_normalized_error_tolerance"]),
        ("RMS peak-normalized field error", rms_peak_normalized, summary["rms_peak_normalized_error_tolerance"]),
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
            "round Gaussian FFT validation failed:\n- "
            + "\n- ".join(failures)
            + f"\nsee {analysis_dir / 'summary.json'}"
        )
    print(json.dumps(summary, indent=2))
    return summary


def _plot_results(
    frame: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    density: np.ndarray,
    potential: np.ndarray,
    ex_grid: np.ndarray,
    ey_grid: np.ndarray,
    figures_dir: Path,
    potential_units: str,
    gauss_frame: pd.DataFrame,
    residual_fields: dict[str, np.ndarray],
    case_title: str = "Round Gaussian FFT",
    tangential_is_residual: bool = True,
) -> None:
    for component in ("Ex", "Ey", "Er"):
        fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
        for angle, group in frame.groupby("angle_deg"):
            ax.plot(group["r_m"] * 1e3, group[f"pic_{component}_v"], ".-", markersize=SCAN_MARKER_SIZE_PT, label=f"PIC {angle:g} deg", alpha=0.65)
            ax.plot(group["r_m"] * 1e3, group[f"theory_{component}_v"], "k--", linewidth=1.0, alpha=0.35)
        ax.set(xlabel="radius (mm)", ylabel=f"{component} (V)", title=f"{case_title}: {component}")
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=7)
        fig.savefig(figures_dir / f"{component.lower()}_scan.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for angle, group in frame.groupby("angle_deg"):
        ax.plot(group["r_m"] * 1e3, group["relative_field_error"] * 100.0, ".-", markersize=SCAN_MARKER_SIZE_PT, label=f"{angle:g} deg")
    ax.set(
        xlabel="radius (mm)",
        ylabel="relative electric-field error (%)",
        title=f"{case_title}: relative electric-field error",
    )
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=7)
    fig.savefig(figures_dir / "relative_error.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for angle, group in frame.groupby("angle_deg"):
        ax.plot(group["r_m"] * 1e3, group["pic_Etheta_v"], ".-", markersize=SCAN_MARKER_SIZE_PT, label=f"{angle:g} deg")
        if not tangential_is_residual:
            ax.plot(
                group["r_m"] * 1e3,
                group["theory_Etheta_v"],
                "k--",
                linewidth=1.0,
                alpha=0.35,
            )
    ax.axhline(0.0, color="black", linewidth=1.0)
    tangential_title = (
        case_title + r": azimuthal residual (theory: $E_\theta=0$)"
        if tangential_is_residual
        else case_title + r": azimuthal field component"
    )
    ax.set(
        xlabel="radius (mm)",
        ylabel=r"azimuthal field $E_\theta$ (V)",
        title=tangential_title,
    )
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=7)
    fig.savefig(figures_dir / "tangential_residual.png", dpi=150)
    plt.close(fig)

    fig, (ax_flux, ax_error) = plt.subplots(2, 1, figsize=(6.8, 7.0), sharex=True, constrained_layout=True)
    half_width_mm = gauss_frame["half_width_m"] * 1e3
    ax_flux.plot(half_width_mm, gauss_frame["charge_over_epsilon0_v_m"], "o-", markersize=3.0, label=r"$Q_{enc}/\epsilon_0$")
    ax_flux.plot(
        half_width_mm,
        gauss_frame["field_flux_v_m"],
        "s--",
        markersize=3.0,
        label=r"$\oint_C \mathbf{E}\cdot\mathbf{n}\,dl$",
    )
    ax_flux.set(ylabel="flux (V m)", title="Gauss-law check on centered square contours")
    ax_flux.grid(alpha=0.3)
    ax_flux.legend()
    ax_error.plot(half_width_mm, gauss_frame["relative_error"] * 100.0, "o-", markersize=3.0)
    ax_error.axhline(0.0, color="black", linewidth=1.0)
    ax_error.set(xlabel="square half-width (mm)", ylabel="relative error (%)")
    ax_error.grid(alpha=0.3)
    fig.savefig(figures_dir / "gauss_law.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    extent = [x_grid[0] * 1e3, x_grid[-1] * 1e3, y_grid[0] * 1e3, y_grid[-1] * 1e3]
    image = ax.imshow(density, origin="lower", extent=extent, aspect="equal")
    step = max(len(x_grid) // 24, 1)
    xx, yy = np.meshgrid(x_grid[::step], y_grid[::step])
    # The field grid is already integrated over the slice width in the solver.
    qx = ex_grid[::step, ::step]
    qy = ey_grid[::step, ::step]
    ax.quiver(xx * 1e3, yy * 1e3, qx, qy, color="white", alpha=0.75)
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="Source density and PIC field")
    fig.colorbar(image, ax=ax, label="charge density (C/m2)")
    fig.savefig(figures_dir / "density_field.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
    consistency_panels = (
        ("potential_field_residual_over_peak", r"$|\mathbf{E}+\nabla\phi|/|E|_{max}$ (%)", "magma", 0.0, None),
        ("poisson_residual_over_source_peak", r"$(\nabla\cdot\mathbf{E}-\rho/\epsilon_0)/(\rho/\epsilon_0)_{max}$ (%)", "RdBu_r", None, None),
        ("field_inversion_residual_over_peak", r"$|\mathbf{E}(\mathbf{r})+\mathbf{E}(-\mathbf{r})|/|E|_{max}$ (%)", "magma", 0.0, None),
    )
    for ax, (name, title, cmap, vmin, vmax) in zip(axes, consistency_panels):
        values = residual_fields[name] * 100.0
        if name == "poisson_residual_over_source_peak":
            limit = float(np.percentile(np.abs(values), 99.0))
            vmin, vmax = -limit, limit
        image = ax.imshow(values, origin="lower", extent=extent, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=title)
        fig.colorbar(image, ax=ax, shrink=0.82)
    fig.savefig(figures_dir / "grid_consistency.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    image = ax.imshow(potential, origin="lower", extent=extent, aspect="equal", cmap="viridis")
    contours = ax.contour(x_grid * 1e3, y_grid * 1e3, potential, levels=16, colors="white", linewidths=0.7)
    ax.clabel(contours, inline=True, fontsize=7, colors="white")
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="PIC potential (arbitrary additive gauge)", aspect="equal")
    ax.grid(alpha=0.2)
    fig.colorbar(image, ax=ax, label=f"potential ({potential_units})")
    fig.savefig(figures_dir / "potential.png", dpi=150)
    plt.close(fig)

    surface_step = max(len(x_grid) // 128, 1)
    surface_x = x_grid[::surface_step] * 1e3
    surface_y = y_grid[::surface_step] * 1e3
    surface_xx, surface_yy = np.meshgrid(surface_x, surface_y)

    fig = plt.figure(figsize=(7.4, 5.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        surface_xx,
        surface_yy,
        potential[::surface_step, ::surface_step],
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
    )
    ax.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        zlabel=f"potential ({potential_units})",
        title="PIC potential surface",
    )
    fig.colorbar(surface, ax=ax, shrink=0.68, pad=0.10, label=f"potential ({potential_units})")
    fig.savefig(figures_dir / "potential_3d.png", dpi=150)
    plt.close(fig)

    field_magnitude = np.hypot(ex_grid, ey_grid)
    field_surface = field_magnitude[::surface_step, ::surface_step]
    field_max = float(np.max(field_surface))
    fig = plt.figure(figsize=(7.4, 5.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        surface_xx,
        surface_yy,
        field_surface,
        cmap="viridis",
        vmin=0.0,
        vmax=field_max,
        linewidth=0.15,
        edgecolor=(0.1, 0.1, 0.1, 0.18),
        alpha=0.78,
        antialiased=True,
    )
    ax.contour(
        surface_xx,
        surface_yy,
        field_surface,
        zdir="z",
        offset=0.0,
        levels=np.linspace(0.0, field_max, 12),
        cmap="viridis",
        linewidths=0.8,
    )
    ax.set_zlim(0.0, 1.05 * field_max)
    ax.view_init(elev=34.0, azim=-58.0)
    ax.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        zlabel="|E| (V)",
        title="PIC integrated transverse-field magnitude",
    )
    fig.colorbar(surface, ax=ax, shrink=0.68, pad=0.10, label="|E| (V), linear color scale")
    fig.savefig(figures_dir / "field_magnitude_3d.png", dpi=150)
    plt.close(fig)


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
    sc_workflow("round_gaussian_free_space_fft")


if __name__ == "__main__":
    main()
