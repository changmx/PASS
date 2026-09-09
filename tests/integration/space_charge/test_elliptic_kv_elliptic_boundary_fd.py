"""Full PASS elliptic-KV validation of the finite-difference field solver.

The 4-D KV distribution has a uniform x-y projection.  This is an intentional
special analytic case in which the KV edge coincides with the grounded
elliptic conductor; Poisson's equation then has an exact quadratic potential
and linear electric field.  It is complemented by the separated beam/aperture
case in ``test_round_kv_circular_boundary_fd.py``.  This workflow runs the real
Injection -> Slicer -> SpaceCharge command chain and writes HDF5, CSV, NPZ,
JSON, and PNG results.
"""

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
from PASS.para.schema.space_charge import (
    SpaceCharge,
    SpaceChargeConfig,
    SpaceChargeResourceConfig,
)
from PASS.utils.constants import const
from tests.integration.space_charge.test_round_gaussian_free_space_fft import (
    SCAN_MARKER_SIZE_PT,
    SCAN_RADIAL_STEP_M,
    _field_file,
    _scan_points,
    _validate_generated_input,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = ROOT / "output" / "elliptic_kv_elliptic_boundary_fd" / "run_001"
RANDOM_SEED = 20260914
NUM_MACRO_PARTICLES = 1_000_000
NUM_REAL_PARTICLES = 100_000_000_000
SEMI_AXIS_X_M = 0.060
SEMI_AXIS_Y_M = 0.035
EMITTANCE_X_M_RAD = (0.5 * SEMI_AXIS_X_M) ** 2
EMITTANCE_Y_M_RAD = (0.5 * SEMI_AXIS_Y_M) ** 2
GRID_HALF_WIDTH_M = 0.065
GRID_POINTS = 257
DELTA_Z_M = 1.0
SC_LENGTH_M = 0.1
KINETIC_ENERGY_EV_U = 33.2e6
ELEMENTARY_CHARGE_C = 1.602176634e-19
SPACE_CHARGE_CONFIGURATION = "elliptic_kv_elliptic_boundary_fd"


def _parameters() -> dict:
    return {
        "case": "elliptic_kv_elliptic_boundary_fd",
        "transverse_distribution": "kv",
        "random_seed": RANDOM_SEED,
        "num_macro_particles": NUM_MACRO_PARTICLES,
        "num_real_particles": NUM_REAL_PARTICLES,
        "semi_axis_x_m": SEMI_AXIS_X_M,
        "semi_axis_y_m": SEMI_AXIS_Y_M,
        "emittance_x_m_rad": EMITTANCE_X_M_RAD,
        "emittance_y_m_rad": EMITTANCE_Y_M_RAD,
        "scan_radial_step_m": SCAN_RADIAL_STEP_M,
        "grid_half_width_m": GRID_HALF_WIDTH_M,
        "grid_points": GRID_POINTS,
        "solver": "fd_dirichlet",
        "deposition_method": "CIC",
        "aperture": {
            "type": "grounded elliptic conductor",
            "semi_axis_x_m": SEMI_AXIS_X_M,
            "semi_axis_y_m": SEMI_AXIS_Y_M,
        },
        "beam_fills_aperture": True,
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
        emit_x=EMITTANCE_X_M_RAD,
        emit_y=EMITTANCE_Y_M_RAD,
        sigma_z=0.1,
        dp=1.0e-6,
        dist_trans="kv",
        dist_longi="coasting",
        save_init_dist=False,
    )
    main = MainConfig(
        beam_name="elliptic-kv-fd-field-scan",
        num_turns=1,
        backend="cpu",
        particle_precision="float64",
        circumference=100.0,
        output_dir=str(simulation_dir.resolve()),
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
        configuration=SPACE_CHARGE_CONFIGURATION,
        sc_length=SC_LENGTH_M,
        aperture_type="ellipse", aperture_value=[SEMI_AXIS_X_M, SEMI_AXIS_Y_M],
        save_field=True,
        save_potential=True,
        save_density=True,
        save_turns=[[0]],
    )
    sequence.add("space_charge", command)
    space_charge = SpaceChargeConfig(
        enabled=True,
        configurations={
            SPACE_CHARGE_CONFIGURATION: SpaceChargeResourceConfig(
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
    (run_dir / "parameters.json").write_text(
        json.dumps(_parameters(), indent=2), encoding="utf-8"
    )
    return input_path


def simulate(run_dir: Path = DEFAULT_RUN_DIR) -> Path:
    input_path = _write_input(run_dir)
    pass_main(str(input_path))
    files = sorted((run_dir / "simulation").rglob("*.h5"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise AssertionError("PASS produced no FD SpaceCharge HDF5 snapshot")
    field_file = files[-1]
    (run_dir / "simulation_manifest.json").write_text(
        json.dumps(
            {
                "field_file": str(field_file.relative_to(run_dir)),
                "input_file": str(input_path.relative_to(run_dir)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[sim] saved {field_file}")
    return field_file


def _grounded_ellipse_solution(
    x: np.ndarray,
    y: np.ndarray,
    charge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized_radius_squared = (x / SEMI_AXIS_X_M) ** 2 + (y / SEMI_AXIS_Y_M) ** 2
    inside = normalized_radius_squared <= 1.0
    density = charge / (const.pi * SEMI_AXIS_X_M * SEMI_AXIS_Y_M)
    coefficient = density / (
        2.0
        * const.epsilon0
        * (1.0 / SEMI_AXIS_X_M**2 + 1.0 / SEMI_AXIS_Y_M**2)
    )
    potential = np.where(inside, coefficient * (1.0 - normalized_radius_squared), 0.0)
    ex = np.where(inside, 2.0 * coefficient * x / SEMI_AXIS_X_M**2, 0.0)
    ey = np.where(inside, 2.0 * coefficient * y / SEMI_AXIS_Y_M**2, 0.0)
    return potential, ex, ey


def _plot_results(
    frame: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    density: np.ndarray,
    potential: np.ndarray,
    ex_grid: np.ndarray,
    ey_grid: np.ndarray,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for component in ("Ex", "Ey", "Er"):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        for angle, group in frame.groupby("angle_deg"):
            ax.plot(
                group["r_m"] * 1.0e3,
                group[f"fd_{component}_v"],
                ".-",
                markersize=SCAN_MARKER_SIZE_PT,
                alpha=0.65,
                label=f"FD {angle:g} deg",
            )
            ax.plot(
                group["r_m"] * 1.0e3,
                group[f"theory_{component}_v"],
                "k--",
                linewidth=0.9,
                alpha=0.25,
            )
        ax.set(
            xlabel="radius (mm)",
            ylabel=f"{component} (V)",
            title=f"Elliptic KV FD: {component} scan",
        )
        ax.grid(alpha=0.3)
        ax.legend(ncol=3, fontsize=7)
        fig.savefig(figures_dir / f"{component.lower()}_scan.png", dpi=150)
        plt.close(fig)

    included = frame["included_in_relative_evaluation"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for angle, group in frame[included].groupby("angle_deg"):
        ax.plot(
            group["r_m"] * 1.0e3,
            group["relative_field_error"] * 100.0,
            ".-",
            markersize=SCAN_MARKER_SIZE_PT,
            label=f"{angle:g} deg",
        )
    ax.set(xlabel="radius (mm)", ylabel="relative field error (%)", title="Elliptic KV FD error")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=7)
    fig.savefig(figures_dir / "relative_error.png", dpi=150)
    plt.close(fig)

    extent = [x_grid[0] * 1.0e3, x_grid[-1] * 1.0e3, y_grid[0] * 1.0e3, y_grid[-1] * 1.0e3]
    boundary_angle = np.linspace(0.0, 2.0 * np.pi, 721)
    boundary_x_mm = SEMI_AXIS_X_M * np.cos(boundary_angle) * 1.0e3
    boundary_y_mm = SEMI_AXIS_Y_M * np.sin(boundary_angle) * 1.0e3
    for name, values, units in (
        ("charge_density", density, "C/m²"),
        ("potential", potential, "V m"),
        ("field_magnitude", np.hypot(ex_grid, ey_grid), "V"),
    ):
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        image = ax.imshow(values, origin="lower", extent=extent, aspect="equal", cmap="viridis")
        fig.colorbar(image, ax=ax, label=units)
        if name in ("potential", "field_magnitude"):
            positive_max = float(np.max(values))
            levels = np.linspace(0.1 * positive_max, 0.9 * positive_max, 9)
            contours = ax.contour(
                x_grid * 1.0e3,
                y_grid * 1.0e3,
                values,
                levels=levels,
                colors="white",
                linewidths=0.55,
                alpha=0.8,
            )
            ax.clabel(contours, inline=True, fontsize=6, fmt="%.0f")
        ax.plot(boundary_x_mm, boundary_y_mm, "w--", linewidth=0.9, alpha=0.9)
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"Elliptic KV FD: {name.replace('_', ' ')}")
        fig.savefig(figures_dir / f"{name}.png", dpi=150)
        plt.close(fig)


def analyse(run_dir: Path = DEFAULT_RUN_DIR) -> dict:
    field_file = _field_file(run_dir)
    analysis_dir = run_dir / "analysis"
    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(field_file, "r") as handle:
        x_grid = np.asarray(handle["x"], dtype=float)
        y_grid = np.asarray(handle["y"], dtype=float)
        charge = float(np.asarray(handle["slice_charge"])[0])
        density = np.asarray(handle["charge_density"][0], dtype=float)
        potential = np.asarray(handle["potential"][0], dtype=float)
        ex_grid = np.asarray(handle["integrated_Ex"][0], dtype=float)
        ey_grid = np.asarray(handle["integrated_Ey"][0], dtype=float)
        attrs = {key: handle.attrs[key] for key in handle.attrs}

    x_probe, y_probe, angle_deg = _scan_points(max_radius_m=0.064)
    points = (y_probe, x_probe)
    fd_potential = RegularGridInterpolator((y_grid, x_grid), potential)(points)
    fd_ex = RegularGridInterpolator((y_grid, x_grid), ex_grid)(points)
    fd_ey = RegularGridInterpolator((y_grid, x_grid), ey_grid)(points)
    theory_potential, theory_ex, theory_ey = _grounded_ellipse_solution(x_probe, y_probe, charge)
    angle_rad = np.deg2rad(angle_deg)
    fd_er = fd_ex * np.cos(angle_rad) + fd_ey * np.sin(angle_rad)
    theory_er = theory_ex * np.cos(angle_rad) + theory_ey * np.sin(angle_rad)
    field_error = np.hypot(fd_ex - theory_ex, fd_ey - theory_ey)
    theory_magnitude = np.hypot(theory_ex, theory_ey)
    theory_peak = float(theory_magnitude.max())
    relative_error = np.divide(
        field_error,
        theory_magnitude,
        out=np.zeros_like(field_error),
        where=theory_magnitude > 0.0,
    )
    normalized_radius = np.sqrt(
        (x_probe / SEMI_AXIS_X_M) ** 2 + (y_probe / SEMI_AXIS_Y_M) ** 2
    )
    theory_boundary_radius = 1.0 / np.sqrt(
        (np.cos(angle_rad) / SEMI_AXIS_X_M) ** 2
        + (np.sin(angle_rad) / SEMI_AXIS_Y_M) ** 2
    )
    edge_half_width = 2.0 * max(
        float(x_grid[1] - x_grid[0]) / SEMI_AXIS_X_M,
        float(y_grid[1] - y_grid[0]) / SEMI_AXIS_Y_M,
    )
    inside = normalized_radius < 1.0
    away_from_edge = np.abs(normalized_radius - 1.0) > edge_half_width
    included = inside & away_from_edge & (theory_magnitude >= 0.2 * theory_peak)
    reason = np.full(x_probe.shape, "included", dtype=object)
    reason[~inside] = "outside_grounded_conductor"
    reason[~away_from_edge] = "near_kv_and_conductor_edge"
    reason[inside & (theory_magnitude < 0.2 * theory_peak)] = (
        "field_below_relative_error_threshold"
    )
    reason[inside & (theory_magnitude == 0.0)] = "zero_theory_at_origin"

    frame = pd.DataFrame(
        {
            "angle_deg": angle_deg,
            "x_m": x_probe,
            "y_m": y_probe,
            "r_m": np.hypot(x_probe, y_probe),
            "normalized_elliptic_radius": normalized_radius,
            "theory_boundary_radius_m": theory_boundary_radius,
            "distance_from_theory_boundary_m": np.hypot(x_probe, y_probe)
            - theory_boundary_radius,
            "fd_potential_v_m": fd_potential,
            "fd_Ex_v": fd_ex,
            "fd_Ey_v": fd_ey,
            "fd_Er_v": fd_er,
            "theory_potential_v_m": theory_potential,
            "theory_Ex_v": theory_ex,
            "theory_Ey_v": theory_ey,
            "theory_Er_v": theory_er,
            "absolute_field_error_v": field_error,
            "relative_field_error": relative_error,
            "included_in_relative_evaluation": included,
            "relative_error_exclusion_reason": reason,
        }
    )
    frame.to_csv(analysis_dir / "field_scan.csv", index=False)

    boundary_rows = []
    for angle, group in frame.groupby("angle_deg"):
        boundary_radius = float(group["theory_boundary_radius_m"].iloc[0])
        field_magnitude = np.hypot(group["fd_Ex_v"], group["fd_Ey_v"])
        outside_zero = group[
            (group["r_m"] >= boundary_radius - 1.0e-12)
            & (field_magnitude < 1.0e-12)
        ]
        first_zero_radius = float(outside_zero["r_m"].iloc[0])
        boundary_rows.append(
            {
                "angle_deg": float(angle),
                "theory_boundary_radius_m": boundary_radius,
                "fd_first_exact_zero_radius_m": first_zero_radius,
                "zero_offset_m": first_zero_radius - boundary_radius,
            }
        )
    boundary_frame = pd.DataFrame(boundary_rows)
    boundary_frame.to_csv(analysis_dir / "boundary_zero_offsets.csv", index=False)

    xx, yy = np.meshgrid(x_grid, y_grid)
    theory_potential_grid, theory_ex_grid, theory_ey_grid = _grounded_ellipse_solution(
        xx, yy, charge
    )
    normalized_grid_radius = np.sqrt(
        (xx / SEMI_AXIS_X_M) ** 2 + (yy / SEMI_AXIS_Y_M) ** 2
    )
    grid_interior = normalized_grid_radius <= 1.0 - edge_half_width
    potential_relative_l2 = float(
        np.linalg.norm((potential - theory_potential_grid)[grid_interior])
        / np.linalg.norm(theory_potential_grid[grid_interior])
    )
    measured_field_grid = np.stack((ex_grid, ey_grid))
    theory_field_grid = np.stack((theory_ex_grid, theory_ey_grid))
    field_grid_mask = np.broadcast_to(grid_interior, measured_field_grid.shape)
    field_relative_l2 = float(
        np.linalg.norm((measured_field_grid - theory_field_grid)[field_grid_mask])
        / np.linalg.norm(theory_field_grid[field_grid_mask])
    )
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    density_charge = float(density.sum() * dx * dy)
    charge_error = abs(density_charge - charge) / charge
    summary = {
        **_parameters(),
        "field_file": str(field_file),
        "solver": str(attrs.get("solver", "fd_dirichlet")),
        "slice_charge_c": charge,
        "density_integral_charge_c": density_charge,
        "charge_conservation_relative_error": charge_error,
        "edge_exclusion_normalized_half_width": edge_half_width,
        "max_boundary_zero_offset_m": float(boundary_frame["zero_offset_m"].abs().max()),
        "max_relative_field_error_evaluation_region": float(relative_error[included].max()),
        "rms_relative_field_error_evaluation_region": float(
            np.sqrt(np.mean(relative_error[included] ** 2))
        ),
        "max_peak_normalized_field_error_away_from_edge": float(
            (field_error[away_from_edge] / theory_peak).max()
        ),
        "potential_relative_l2_interior": potential_relative_l2,
        "field_relative_l2_interior": field_relative_l2,
        "relative_error_tolerance": 0.03,
        "peak_normalized_error_tolerance": 0.01,
        "potential_relative_l2_tolerance": 0.005,
        "field_relative_l2_tolerance": 0.01,
        "charge_conservation_tolerance": 1.0e-9,
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez(
        analysis_dir / "field_data.npz",
        x_grid=x_grid,
        y_grid=y_grid,
        charge_density=density,
        potential=potential,
        integrated_Ex=ex_grid,
        integrated_Ey=ey_grid,
        theory_potential=theory_potential_grid,
        theory_Ex=theory_ex_grid,
        theory_Ey=theory_ey_grid,
        **{name: frame[name].to_numpy() for name in frame.columns},
    )
    _plot_results(frame, x_grid, y_grid, density, potential, ex_grid, ey_grid, figures_dir)

    checks = (
        ("relative field error", summary["max_relative_field_error_evaluation_region"], summary["relative_error_tolerance"]),
        ("peak-normalized field error", summary["max_peak_normalized_field_error_away_from_edge"], summary["peak_normalized_error_tolerance"]),
        ("potential relative L2", potential_relative_l2, summary["potential_relative_l2_tolerance"]),
        ("field relative L2", field_relative_l2, summary["field_relative_l2_tolerance"]),
        ("charge conservation", charge_error, summary["charge_conservation_tolerance"]),
    )
    failures = [f"{name}: measured={value:.8g}, tolerance={limit:.8g}" for name, value, limit in checks if value > limit]
    if failures:
        raise AssertionError("elliptic KV FD validation failed:\n" + "\n".join(failures))
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
    sc_workflow("elliptic_kv_elliptic_boundary_fd")


if __name__ == "__main__":
    main()
