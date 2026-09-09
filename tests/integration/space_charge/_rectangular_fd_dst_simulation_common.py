"""Full PASS FD/DST comparisons for fixed-seed bunches in a rectangle.

Both solvers consume the same deposited charge on the same 257x257 grounded
rectangular grid.  The first SpaceCharge command changes px/py only, so the
second command sees identical x/y coordinates and slice membership.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from PASS.para.schema.elements import MarkerElement
from PASS.para.schema.slicer import Slicer
from PASS.para.schema.space_charge import (
    SpaceCharge,
    SpaceChargeConfig,
    SpaceChargeResourceConfig,
)
from tests.integration.space_charge.test_round_gaussian_free_space_fft import (
    SCAN_MARKER_SIZE_PT,
    SCAN_RADIAL_STEP_M,
    _scan_points,
    _validate_generated_input,
)


ROOT = Path(__file__).resolve().parent
NUM_MACRO_PARTICLES = 1_000_000
NUM_REAL_PARTICLES = 100_000_000_000
GRID_HALF_WIDTH_M = 0.100
GRID_POINTS = 257
DELTA_Z_M = 1.0
SC_LENGTH_M = 0.1
KINETIC_ENERGY_EV_U = 33.2e6


@dataclass(frozen=True)
class RectangularCase:
    name: str
    title: str
    distribution: str
    random_seed: int
    emit_x_m_rad: float
    emit_y_m_rad: float
    scale_x_m: float
    scale_y_m: float
    scale_names: tuple[str, str]

    @property
    def default_run_dir(self) -> Path:
        return ROOT / "output" / self.name / "run_001"

    @property
    def fd_configuration(self) -> str:
        return f"{self.name}_fd"

    @property
    def dst_configuration(self) -> str:
        return f"{self.name}_dst"

    @property
    def free_space_fft_run_dir(self) -> Path:
        prefix = self.name.removesuffix("_rectangular_fd_dst")
        return ROOT / "output" / f"{prefix}_free_space_fft" / "run_001"


CASES = {
    "round_gaussian": RectangularCase(
        "round_gaussian_rectangular_fd_dst", "Round Gaussian in a grounded rectangle",
        "gaussian", 20260904, 0.030**2, 0.030**2, 0.030, 0.030,
        ("sigma_x_m", "sigma_y_m"),
    ),
    "round_kv": RectangularCase(
        "round_kv_rectangular_fd_dst", "Round KV in a grounded rectangle",
        "kv", 20260908, (0.060 / 2.0) ** 2, (0.060 / 2.0) ** 2, 0.060, 0.060,
        ("semi_axis_x_m", "semi_axis_y_m"),
    ),
    "elliptic_gaussian": RectangularCase(
        "elliptic_gaussian_rectangular_fd_dst", "Elliptic Gaussian in a grounded rectangle",
        "gaussian", 20260909, 0.025**2, 0.015**2, 0.025, 0.015,
        ("sigma_x_m", "sigma_y_m"),
    ),
    "elliptic_kv": RectangularCase(
        "elliptic_kv_rectangular_fd_dst", "Elliptic KV in a grounded rectangle",
        "kv", 20260910, (0.060 / 2.0) ** 2, (0.035 / 2.0) ** 2, 0.060, 0.035,
        ("semi_axis_x_m", "semi_axis_y_m"),
    ),
}


def _parameters(case: RectangularCase) -> dict:
    return {
        "case": case.name,
        "transverse_distribution": case.distribution,
        "random_seed": case.random_seed,
        "num_macro_particles": NUM_MACRO_PARTICLES,
        "num_real_particles": NUM_REAL_PARTICLES,
        case.scale_names[0]: case.scale_x_m,
        case.scale_names[1]: case.scale_y_m,
        "emittance_x_m_rad": case.emit_x_m_rad,
        "emittance_y_m_rad": case.emit_y_m_rad,
        "grid_points": GRID_POINTS,
        "grid_half_width_m": GRID_HALF_WIDTH_M,
        "scan_radial_step_m": SCAN_RADIAL_STEP_M,
        "field_solvers": ["fd_dirichlet", "dst_dirichlet"],
        "deposition_method": "CIC",
        "boundary": "grounded rectangle (Dirichlet phi=0)",
    }


def _write_input(case: RectangularCase, run_dir: Path) -> Path:
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
        emit_x=case.emit_x_m_rad,
        emit_y=case.emit_y_m_rad,
        sigma_z=0.1,
        dp=1.0e-6,
        dist_trans=case.distribution,
        dist_longi="coasting",
        save_init_dist=False,
    )
    main = MainConfig(
        beam_name=case.name.replace("_", "-"),
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
            save_turns=[],
        ),
    )
    fd_command = SpaceCharge(
        s=0.0,
        configuration=case.fd_configuration,
        sc_length=SC_LENGTH_M,
        save_field=True,
        save_potential=True,
        save_density=True,
        save_turns=[[0]],
    )
    dst_command = SpaceCharge(
        s=0.0,
        configuration=case.dst_configuration,
        sc_length=SC_LENGTH_M,
        save_field=True,
        save_potential=True,
        save_density=True,
        save_turns=[[0]],
    )
    sequence.add("space_charge_fd", fd_command)
    sequence.add("space_charge_dst", dst_command)
    common_resource = {
        "slice_set": "space_charge",
        "nx": GRID_POINTS,
        "ny": GRID_POINTS,
        "grid_width_x": 2.0 * GRID_HALF_WIDTH_M,
        "grid_width_y": 2.0 * GRID_HALF_WIDTH_M,
        "deposition_method": "CIC",
    }
    space_charge = SpaceChargeConfig(
        enabled=True,
        configurations={
            case.fd_configuration: SpaceChargeResourceConfig(
                **common_resource, solver='fd_dirichlet'
            ),
            case.dst_configuration: SpaceChargeResourceConfig(
                **common_resource, solver='dst_dirichlet'
            ),
        },
    )
    input_path = input_dir / "beam0.json"
    generate_input(main, sequence, str(input_path), space_charge=space_charge)
    _validate_generated_input(
        input_path, space_charge, fd_command, bunch, command_name="space_charge_fd"
    )
    _validate_generated_input(
        input_path, space_charge, dst_command, bunch, command_name="space_charge_dst"
    )
    (run_dir / "parameters.json").write_text(
        json.dumps(_parameters(case), indent=2), encoding="utf-8"
    )
    return input_path


def _latest_command_file(run_dir: Path, command_name: str) -> Path:
    files = [
        path
        for path in (run_dir / "simulation").rglob("*.h5")
        if command_name in path.parts
    ]
    if not files:
        raise FileNotFoundError(f"no HDF5 snapshot found for {command_name}")
    return max(files, key=lambda path: path.stat().st_mtime)


def simulate(case: RectangularCase, run_dir: Path) -> tuple[Path, Path]:
    input_path = _write_input(case, run_dir)
    pass_main(str(input_path))
    fd_file = _latest_command_file(run_dir, "space_charge_fd")
    dst_file = _latest_command_file(run_dir, "space_charge_dst")
    (run_dir / "simulation_manifest.json").write_text(
        json.dumps(
            {
                "input_file": str(input_path.relative_to(run_dir)),
                "fd_field_file": str(fd_file.relative_to(run_dir)),
                "dst_field_file": str(dst_file.relative_to(run_dir)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[sim] FD saved {fd_file}")
    print(f"[sim] DST saved {dst_file}")
    return fd_file, dst_file


def _load_field(path: Path) -> dict[str, np.ndarray | float | str]:
    with h5py.File(path, "r") as handle:
        return {
            "x": np.asarray(handle["x"], dtype=float),
            "y": np.asarray(handle["y"], dtype=float),
            "density": np.asarray(handle["charge_density"][0], dtype=float),
            "potential": np.asarray(handle["potential"][0], dtype=float),
            "ex": np.asarray(handle["integrated_Ex"][0], dtype=float),
            "ey": np.asarray(handle["integrated_Ey"][0], dtype=float),
            "charge": float(np.asarray(handle["slice_charge"])[0]),
            "solver": str(handle.attrs.get("solver", "unknown")),
        }


def _relative_l2(measured: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(measured - reference) / np.linalg.norm(reference))


def _plot_comparison(
    case: RectangularCase,
    frame: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    fd: dict,
    dst: dict,
    fft: dict | None,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for angle, group in frame.groupby("angle_deg"):
        ax.plot(
            group["r_m"] * 1.0e3,
            group["fd_Er_v"],
            ".-",
            markersize=SCAN_MARKER_SIZE_PT,
            alpha=0.65,
            label=f"FD {angle:g} deg",
        )
        ax.plot(
            group["r_m"] * 1.0e3,
            group["dst_Er_v"],
            "k--",
            linewidth=0.8,
            alpha=0.25,
        )
    ax.set(xlabel="radius (mm)", ylabel="Er (V)", title=f"{case.title}: FD versus DST")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=7)
    fig.savefig(figures_dir / "radial_field_comparison.png", dpi=150)
    plt.close(fig)

    if fft is not None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        # selected_angles = (0.0, 45.0, 90.0, 135.0)
        selected_angles = (0.0, 45.0)
        for angle, group in frame.groupby("angle_deg"):
            if not any(np.isclose(angle, selected) for selected in selected_angles):
                continue
            (line,) = ax.plot(
                group["r_m"] * 1.0e3,
                group["fft_free_space_Er_v"],
                ":",
                linewidth=1.0,
                label=f"FFT free space, {angle:g} deg",
            )
            color = line.get_color()
            ax.plot(group["r_m"] * 1.0e3, group["fd_Er_v"], "-", color=color,
                    linewidth=0.8, label=f"FD rectangle, {angle:g} deg")
            ax.plot(group["r_m"] * 1.0e3, group["dst_Er_v"], "--", color=color,
                    linewidth=0.8, label=f"DST rectangle, {angle:g} deg")
        ax.set(
            xlabel="radius (mm)",
            ylabel="Er (V)",
            title=f"{case.title}: free-space FFT vs rectangular FD/DST",
        )
        ax.grid(alpha=0.3)
        ax.legend(ncol=3, fontsize=6)
        fig.savefig(
            figures_dir / "free_space_fft_vs_rectangular_fd_dst_field.png", dpi=150
        )
        plt.close(fig)

    extent = [x_grid[0] * 1.0e3, x_grid[-1] * 1.0e3, y_grid[0] * 1.0e3, y_grid[-1] * 1.0e3]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, label, data in (
        (axes[0], "FD", np.hypot(fd["ex"], fd["ey"])),
        (axes[1], "DST", np.hypot(dst["ex"], dst["ey"])),
    ):
        image = ax.imshow(data, origin="lower", extent=extent, aspect="equal", cmap="viridis")
        levels = np.linspace(0.1 * float(data.max()), 0.9 * float(data.max()), 9)
        contours = ax.contour(
            x_grid * 1.0e3,
            y_grid * 1.0e3,
            data,
            levels=levels,
            colors="white",
            linewidths=0.5,
        )
        ax.clabel(contours, fontsize=6, fmt="%.0f")
        fig.colorbar(image, ax=ax, label="V")
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"{label} field magnitude")
    fig.savefig(figures_dir / "field_magnitude_comparison.png", dpi=150)
    plt.close(fig)

    field_difference = np.hypot(fd["ex"] - dst["ex"], fd["ey"] - dst["ey"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    image = ax.imshow(
        np.log10(np.maximum(field_difference, np.finfo(float).tiny)),
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="magma",
    )
    fig.colorbar(image, ax=ax, label="log10 |E_FD - E_DST| (V)")
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="FD-DST field difference")
    fig.savefig(figures_dir / "field_difference.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for angle, group in frame.groupby("angle_deg"):
        if not any(np.isclose(angle, selected) for selected in (0.0, 45.0, 90.0, 135.0)):
            continue
        ax.semilogy(
            group["r_m"] * 1.0e3,
            np.maximum(group["fd_relative_error_vs_dst"], np.finfo(float).tiny),
            "o-",
            markersize=SCAN_MARKER_SIZE_PT,
            linewidth=0.7,
            label=f"FD relative to DST, {angle:g} deg",
        )
        ax.semilogy(
            group["r_m"] * 1.0e3,
            np.maximum(group["dst_relative_error_vs_fd"], np.finfo(float).tiny),
            "x--",
            markersize=SCAN_MARKER_SIZE_PT,
            linewidth=0.7,
            label=f"DST relative to FD, {angle:g} deg",
        )
    ax.set(
        xlabel="radius (mm)",
        ylabel="pointwise relative field difference",
        title=f"{case.title}: two-way FD/DST relative error",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=6)
    fig.savefig(figures_dir / "fd_dst_relative_error_same_plot.png", dpi=150)
    plt.close(fig)


def analyse(case: RectangularCase, run_dir: Path, *, fft_run_dir: Path | None = None) -> dict:
    fd_file = _latest_command_file(run_dir, "space_charge_fd")
    dst_file = _latest_command_file(run_dir, "space_charge_dst")
    fd = _load_field(fd_file)
    dst = _load_field(dst_file)
    try:
        fft_file = _latest_command_file(
            case.free_space_fft_run_dir if fft_run_dir is None else fft_run_dir,
            "space_charge",
        )
        fft = _load_field(fft_file)
    except FileNotFoundError:
        fft_file = None
        fft = None
    np.testing.assert_array_equal(fd["x"], dst["x"])
    np.testing.assert_array_equal(fd["y"], dst["y"])
    x_grid = fd["x"]
    y_grid = fd["y"]

    x_probe, y_probe, angle_deg = _scan_points(max_radius_m=0.095)
    points = (y_probe, x_probe)
    angle_rad = np.deg2rad(angle_deg)
    sampled = {}
    for solver_name, data in (("fd", fd), ("dst", dst)):
        sampled_ex = RegularGridInterpolator((y_grid, x_grid), data["ex"])(points)
        sampled_ey = RegularGridInterpolator((y_grid, x_grid), data["ey"])(points)
        sampled[f"{solver_name}_Ex_v"] = sampled_ex
        sampled[f"{solver_name}_Ey_v"] = sampled_ey
        sampled[f"{solver_name}_Er_v"] = (
            sampled_ex * np.cos(angle_rad) + sampled_ey * np.sin(angle_rad)
        )
    if fft is not None:
        fft_points = (y_probe, x_probe)
        fft_ex = RegularGridInterpolator((fft["y"], fft["x"]), fft["ex"])(fft_points)
        fft_ey = RegularGridInterpolator((fft["y"], fft["x"]), fft["ey"])(fft_points)
        sampled["fft_free_space_Ex_v"] = fft_ex
        sampled["fft_free_space_Ey_v"] = fft_ey
        sampled["fft_free_space_Er_v"] = (
            fft_ex * np.cos(angle_rad) + fft_ey * np.sin(angle_rad)
        )
    scan_error = np.hypot(
        sampled["fd_Ex_v"] - sampled["dst_Ex_v"],
        sampled["fd_Ey_v"] - sampled["dst_Ey_v"],
    )
    dst_peak = float(np.hypot(sampled["dst_Ex_v"], sampled["dst_Ey_v"]).max())
    fd_magnitude = np.hypot(sampled["fd_Ex_v"], sampled["fd_Ey_v"])
    dst_magnitude = np.hypot(sampled["dst_Ex_v"], sampled["dst_Ey_v"])
    relative_floor = max(fd_magnitude.max(), dst_magnitude.max()) * 1.0e-14
    frame = pd.DataFrame(
        {
            "angle_deg": angle_deg,
            "x_m": x_probe,
            "y_m": y_probe,
            "r_m": np.hypot(x_probe, y_probe),
            **sampled,
            "absolute_field_difference_v": scan_error,
            "peak_normalized_field_difference": scan_error / dst_peak,
            "fd_relative_error_vs_dst": scan_error / np.maximum(dst_magnitude, relative_floor),
            "dst_relative_error_vs_fd": scan_error / np.maximum(fd_magnitude, relative_floor),
        }
    )

    boundary_model_metrics = {}
    if fft is not None:
        fft_magnitude = np.hypot(
            sampled["fft_free_space_Ex_v"], sampled["fft_free_space_Ey_v"]
        )
        fft_peak = float(fft_magnitude.max())
        for solver_name in ("fd", "dst"):
            difference = np.hypot(
                sampled[f"{solver_name}_Ex_v"] - sampled["fft_free_space_Ex_v"],
                sampled[f"{solver_name}_Ey_v"] - sampled["fft_free_space_Ey_v"],
            )
            frame[f"{solver_name}_rectangle_vs_fft_free_space_peak_normalized_difference"] = (
                difference / fft_peak
            )
            boundary_model_metrics[
                f"max_{solver_name}_rectangle_vs_fft_free_space_peak_normalized_difference"
            ] = float((difference / fft_peak).max())

    component_errors = {
        "fd_potential_relative_l2_vs_dst": _relative_l2(fd["potential"], dst["potential"]),
        "fd_Ex_relative_l2_vs_dst": _relative_l2(fd["ex"], dst["ex"]),
        "fd_Ey_relative_l2_vs_dst": _relative_l2(fd["ey"], dst["ey"]),
        "dst_potential_relative_l2_vs_fd": _relative_l2(dst["potential"], fd["potential"]),
        "dst_Ex_relative_l2_vs_fd": _relative_l2(dst["ex"], fd["ex"]),
        "dst_Ey_relative_l2_vs_fd": _relative_l2(dst["ey"], fd["ey"]),
    }
    density_max_abs_difference = float(np.max(np.abs(fd["density"] - dst["density"])))
    summary = {
        **_parameters(case),
        "fd_field_file": str(fd_file),
        "dst_field_file": str(dst_file),
        "free_space_fft_field_file": None if fft_file is None else str(fft_file),
        "fd_solver_attribute": fd["solver"],
        "dst_solver_attribute": dst["solver"],
        "density_max_abs_difference": density_max_abs_difference,
        "slice_charge_difference_c": float(fd["charge"] - dst["charge"]),
        **component_errors,
        **boundary_model_metrics,
        "max_scan_peak_normalized_field_difference": float((scan_error / dst_peak).max()),
        "relative_error_definition": (
            "FD-vs-DST and DST-vs-FD are the same absolute vector difference divided "
            "by the named reference magnitude, with a 1e-14 peak floor near field zeros"
        ),
        "relative_l2_tolerance": 1.0e-11,
        "density_absolute_tolerance": 0.0,
    }
    analysis_dir = run_dir / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(analysis_dir / "field_scan.csv", index=False)
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez(
        analysis_dir / "field_data.npz",
        x_grid=x_grid,
        y_grid=y_grid,
        fd_density=fd["density"],
        dst_density=dst["density"],
        fd_potential=fd["potential"],
        dst_potential=dst["potential"],
        fd_Ex=fd["ex"],
        dst_Ex=dst["ex"],
        fd_Ey=fd["ey"],
        dst_Ey=dst["ey"],
    )
    _plot_comparison(case, frame, x_grid, y_grid, fd, dst, fft, figures_dir)

    assert density_max_abs_difference == 0.0, (
        "FD/DST density mismatch: "
        f"measured_max_abs={density_max_abs_difference:.8g}, theory=0, tolerance=0"
    )
    for name, error in component_errors.items():
        assert error < summary["relative_l2_tolerance"], (
            f"full PASS FD/DST {name} mismatch: measured={error:.8g}, "
            f"theory=0, tolerance={summary['relative_l2_tolerance']:.8g}"
        )
    print(json.dumps(summary, indent=2))
    return summary
