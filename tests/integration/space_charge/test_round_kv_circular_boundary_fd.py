"""Full FD test of a round KV beam inside a larger circular conductor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PASS.main import main as pass_main
from PASS.para.api import generate_input
from PASS.para.schema.bunch import BunchConfig, InjectionItem
from PASS.para.schema.main import MainConfig
from PASS.para.schema.sequence import Sequence
from PASS.para.schema.slicer import Slicer
from PASS.para.schema.space_charge import SpaceCharge, SpaceChargeConfig, SpaceChargeResourceConfig
from PASS.utils.constants import const
from tests.integration.space_charge.test_round_gaussian_free_space_fft import (
    SCAN_MARKER_SIZE_PT,
    SCAN_RADIAL_STEP_M,
    _field_file,
    _scan_points,
    _validate_generated_input,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = ROOT / "output" / "round_kv_circular_boundary_fd" / "run_001"
RANDOM_SEED = 20260916
NUM_MACRO_PARTICLES = 1_000_000
NUM_REAL_PARTICLES = 100_000_000_000
BEAM_RADIUS_M = 0.035
APERTURE_RADIUS_M = 0.060
EMITTANCE_M_RAD = (0.5 * BEAM_RADIUS_M) ** 2
GRID_HALF_WIDTH_M = 0.065
GRID_POINTS = 257
DELTA_Z_M = 1.0
SC_LENGTH_M = 0.1
KINETIC_ENERGY_EV_U = 33.2e6
CONFIGURATION = "round_kv_circular_boundary_fd"


def _parameters() -> dict:
    return {
        "case": "round_kv_circular_boundary_fd",
        "transverse_distribution": "kv",
        "random_seed": RANDOM_SEED,
        "num_macro_particles": NUM_MACRO_PARTICLES,
        "num_real_particles": NUM_REAL_PARTICLES,
        "beam_radius_m": BEAM_RADIUS_M,
        "aperture_radius_m": APERTURE_RADIUS_M,
        "beam_to_aperture_radius_ratio": BEAM_RADIUS_M / APERTURE_RADIUS_M,
        "emittance_m_rad": EMITTANCE_M_RAD,
        "grid_points": GRID_POINTS,
        "grid_half_width_m": GRID_HALF_WIDTH_M,
        "scan_radial_step_m": SCAN_RADIAL_STEP_M,
        "solver": "fd_dirichlet",
        "deposition_method": "CIC",
    }


def _write_input(run_dir: Path) -> Path:
    input_dir = run_dir / "input"
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
        beam_name="round-kv-circular-fd",
        num_turns=1,
        backend="cpu",
        particle_precision="float64",
        circumference=100.0,
        output_dir=str((run_dir / "simulation").resolve()),
        is_plot=False,
    )
    sequence = Sequence()
    sequence.add(
        "injection",
        InjectionItem(harmonic_number=1, random_seed=RANDOM_SEED, bunches=[bunch]),
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
            save_turns=[],
        ),
    )
    command = SpaceCharge(
        s=0.0,
        configuration=CONFIGURATION,
        sc_length=SC_LENGTH_M,
        aperture_type="circle", aperture_value=[APERTURE_RADIUS_M],
        save_field=True,
        save_potential=True,
        save_density=True,
        save_turns=[[0]],
    )
    sequence.add("space_charge", command)
    space_charge = SpaceChargeConfig(
        enabled=True,
        configurations={
            CONFIGURATION: SpaceChargeResourceConfig(
                slice_set="space_charge",
                nx=GRID_POINTS,
                ny=GRID_POINTS,
                grid_width_x=2.0 * GRID_HALF_WIDTH_M,
                grid_width_y=2.0 * GRID_HALF_WIDTH_M,
                solver='fd_dirichlet',
                deposition_method="CIC",
            )
        },
    )
    input_path = input_dir / "beam0.json"
    generate_input(main, sequence, str(input_path), space_charge=space_charge)
    _validate_generated_input(input_path, space_charge, command, bunch)
    (run_dir / "parameters.json").write_text(json.dumps(_parameters(), indent=2), encoding="utf-8")
    return input_path


def simulate(run_dir: Path = DEFAULT_RUN_DIR) -> Path:
    input_path = _write_input(run_dir)
    pass_main(str(input_path))
    files = list((run_dir / "simulation").rglob("*.h5"))
    if not files:
        raise AssertionError("PASS produced no circular FD field snapshot")
    field_file = max(files, key=lambda path: path.stat().st_mtime)
    (run_dir / "simulation_manifest.json").write_text(
        json.dumps({"field_file": str(field_file.relative_to(run_dir))}, indent=2),
        encoding="utf-8",
    )
    print(f"[sim] saved {field_file}")
    return field_file


def _analytic_solution(
    x: np.ndarray, y: np.ndarray, charge: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.hypot(x, y)
    coefficient = charge / (2.0 * const.pi * const.epsilon0)
    inside_beam = radius <= BEAM_RADIUS_M
    inside_aperture = radius <= APERTURE_RADIUS_M
    er = np.zeros_like(radius)
    er[inside_beam] = coefficient * radius[inside_beam] / BEAM_RADIUS_M**2
    between = (~inside_beam) & inside_aperture
    er[between] = coefficient / radius[between]
    potential = np.zeros_like(radius)
    potential[between] = coefficient * np.log(APERTURE_RADIUS_M / radius[between])
    potential[inside_beam] = coefficient * (
        np.log(APERTURE_RADIUS_M / BEAM_RADIUS_M)
        + 0.5 * (1.0 - (radius[inside_beam] / BEAM_RADIUS_M) ** 2)
    )
    ex = np.divide(er * x, radius, out=np.zeros_like(er), where=radius > 0.0)
    ey = np.divide(er * y, radius, out=np.zeros_like(er), where=radius > 0.0)
    return potential, ex, ey


def _plot(
    frame: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    potential: np.ndarray,
    ex_grid: np.ndarray,
    ey_grid: np.ndarray,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for angle, group in frame.groupby("angle_deg"):
        ax.plot(group["r_m"] * 1.0e3, group["fd_Er_v"], ".-", markersize=SCAN_MARKER_SIZE_PT, alpha=0.65, label=f"FD {angle:g} deg")
        ax.plot(group["r_m"] * 1.0e3, group["theory_Er_v"], "k--", linewidth=0.8, alpha=0.25)
    ax.axvline(BEAM_RADIUS_M * 1.0e3, color="tab:blue", linestyle=":", label="KV edge")
    ax.axvline(APERTURE_RADIUS_M * 1.0e3, color="tab:red", linestyle=":", label="conductor")
    ax.set(xlabel="radius (mm)", ylabel="Er (V)", title="Round KV beam in circular FD aperture")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=7)
    fig.savefig(figures_dir / "radial_field.png", dpi=150)
    plt.close(fig)

    extent = [x_grid[0] * 1.0e3, x_grid[-1] * 1.0e3, y_grid[0] * 1.0e3, y_grid[-1] * 1.0e3]
    for name, values, units in (
        ("potential", potential, "V m"),
        ("field_magnitude", np.hypot(ex_grid, ey_grid), "V"),
    ):
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        image = ax.imshow(values, origin="lower", extent=extent, aspect="equal", cmap="viridis")
        levels = np.linspace(0.1 * float(values.max()), 0.9 * float(values.max()), 9)
        contours = ax.contour(x_grid * 1.0e3, y_grid * 1.0e3, values, levels=levels, colors="white", linewidths=0.55)
        ax.clabel(contours, fontsize=6, fmt="%.0f")
        theta = np.linspace(0.0, 2.0 * np.pi, 721)
        ax.plot(APERTURE_RADIUS_M * np.cos(theta) * 1.0e3, APERTURE_RADIUS_M * np.sin(theta) * 1.0e3, "w--", linewidth=0.9)
        fig.colorbar(image, ax=ax, label=units)
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"Round KV circular FD: {name.replace('_', ' ')}")
        fig.savefig(figures_dir / f"{name}.png", dpi=150)
        plt.close(fig)


def analyse(run_dir: Path = DEFAULT_RUN_DIR) -> dict:
    field_file = _field_file(run_dir)
    with h5py.File(field_file, "r") as handle:
        x_grid = np.asarray(handle["x"], dtype=float)
        y_grid = np.asarray(handle["y"], dtype=float)
        charge = float(np.asarray(handle["slice_charge"])[0])
        density = np.asarray(handle["charge_density"][0], dtype=float)
        potential = np.asarray(handle["potential"][0], dtype=float)
        ex_grid = np.asarray(handle["integrated_Ex"][0], dtype=float)
        ey_grid = np.asarray(handle["integrated_Ey"][0], dtype=float)

    x_probe, y_probe, angle_deg = _scan_points(max_radius_m=0.064)
    points = (y_probe, x_probe)
    fd_ex = RegularGridInterpolator((y_grid, x_grid), ex_grid)(points)
    fd_ey = RegularGridInterpolator((y_grid, x_grid), ey_grid)(points)
    theory_potential, theory_ex, theory_ey = _analytic_solution(x_probe, y_probe, charge)
    angle_rad = np.deg2rad(angle_deg)
    fd_er = fd_ex * np.cos(angle_rad) + fd_ey * np.sin(angle_rad)
    theory_er = theory_ex * np.cos(angle_rad) + theory_ey * np.sin(angle_rad)
    field_error = np.hypot(fd_ex - theory_ex, fd_ey - theory_ey)
    theory_magnitude = np.hypot(theory_ex, theory_ey)
    theory_peak = float(theory_magnitude.max())
    radius = np.hypot(x_probe, y_probe)
    grid_step = max(float(x_grid[1] - x_grid[0]), float(y_grid[1] - y_grid[0]))
    away_from_edges = (
        (np.abs(radius - BEAM_RADIUS_M) > 2.0 * grid_step)
        & (np.abs(radius - APERTURE_RADIUS_M) > 2.0 * grid_step)
    )
    included = away_from_edges & (theory_magnitude >= 0.2 * theory_peak)
    relative_error = np.divide(field_error, theory_magnitude, out=np.zeros_like(field_error), where=theory_magnitude > 0.0)
    reason = np.where(included, "included", "field_below_threshold_or_near_edge").astype(object)
    reason[radius > APERTURE_RADIUS_M] = "outside_grounded_conductor"
    reason[radius == 0.0] = "zero_theory_at_origin"
    frame = pd.DataFrame(
        {
            "angle_deg": angle_deg,
            "x_m": x_probe,
            "y_m": y_probe,
            "r_m": radius,
            "fd_Ex_v": fd_ex,
            "fd_Ey_v": fd_ey,
            "fd_Er_v": fd_er,
            "theory_Ex_v": theory_ex,
            "theory_Ey_v": theory_ey,
            "theory_Er_v": theory_er,
            "theory_potential_v_m": theory_potential,
            "relative_field_error": relative_error,
            "included_in_relative_evaluation": included,
            "relative_error_exclusion_reason": reason,
        }
    )
    analysis_dir = run_dir / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(analysis_dir / "field_scan.csv", index=False)
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    density_charge = float(density.sum() * dx * dy)
    summary = {
        **_parameters(),
        "field_file": str(field_file),
        "slice_charge_c": charge,
        "charge_conservation_relative_error": abs(density_charge - charge) / charge,
        "max_relative_field_error_evaluation_region": float(relative_error[included].max()),
        "rms_relative_field_error_evaluation_region": float(np.sqrt(np.mean(relative_error[included] ** 2))),
        "max_peak_normalized_field_error_away_from_edges": float((field_error[away_from_edges] / theory_peak).max()),
        "relative_error_tolerance": 0.03,
        "peak_normalized_error_tolerance": 0.01,
        "charge_conservation_tolerance": 1.0e-9,
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(analysis_dir / "field_data.npz", x_grid=x_grid, y_grid=y_grid, charge_density=density, potential=potential, integrated_Ex=ex_grid, integrated_Ey=ey_grid)
    _plot(frame, x_grid, y_grid, potential, ex_grid, ey_grid, figures_dir)
    checks = (
        (summary["max_relative_field_error_evaluation_region"], summary["relative_error_tolerance"], "relative field error"),
        (summary["max_peak_normalized_field_error_away_from_edges"], summary["peak_normalized_error_tolerance"], "peak-normalized field error"),
        (summary["charge_conservation_relative_error"], summary["charge_conservation_tolerance"], "charge conservation"),
    )
    failures = [f"{name}: measured={value:.8g}, tolerance={limit:.8g}" for value, limit, name in checks if value > limit]
    if failures:
        raise AssertionError("round KV circular FD validation failed:\n" + "\n".join(failures))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sim", "ana", "simana"), nargs="?", default="simana")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.mode in ("sim", "simana"):
        simulate(run_dir)
    if args.mode in ("ana", "simana"):
        analyse(run_dir)


def test_full_workflow(sc_workflow):
    """Generate the full source, run PASS, and enforce the existing analysis checks."""
    sc_workflow("round_kv_circular_boundary_fd")


if __name__ == "__main__":
    main()
