"""Shared calculations for distribution-specific FD space-charge tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PASS.commands.solver.pic import (
    GridGeometry,
    build_pic_resources,
    gather_bilinear,
    gather_quadratic,
    solve_pic,
)
from PASS.commands.space_charge import SpaceCharge as SpaceChargeCommand
from PASS.para.schema.space_charge import SpaceChargeConfig, SpaceChargeResourceConfig
from PASS.utils.constants import const


RECTANGLE_HALF_WIDTH_X_M = 0.080
RECTANGLE_HALF_WIDTH_Y_M = 0.050
ELLIPSE_A_M = 0.060
ELLIPSE_B_M = 0.035
POTENTIAL_SCALE_V_M = 1200.0
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


def write_summary(case_name: str, summary: dict) -> Path:
    analysis_dir = OUTPUT_ROOT / case_name / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    path = analysis_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def write_convergence_outputs(
    case_name: str,
    grid_sizes: tuple[int, ...],
    errors: np.ndarray,
    potential_orders: np.ndarray,
    field_orders: np.ndarray,
    summary: dict,
) -> Path:
    case_dir = OUTPUT_ROOT / case_name
    analysis_dir = case_dir / "analysis"
    figures_dir = case_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = analysis_dir / "convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["grid_points", "potential_relative_l2", "field_relative_l2", "potential_order", "field_order"]
        )
        for index, grid_points in enumerate(grid_sizes):
            writer.writerow(
                [
                    grid_points,
                    float(errors[index, 0]),
                    float(errors[index, 1]),
                    "" if index == 0 else float(potential_orders[index - 1]),
                    "" if index == 0 else float(field_orders[index - 1]),
                ]
            )

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.loglog(grid_sizes, errors[:, 0], "o-", markersize=3.0, label="Potential")
    ax.loglog(grid_sizes, errors[:, 1], "s-", markersize=3.0, label="Electric field")
    reference = errors[0, 0] * (grid_sizes[0] / np.asarray(grid_sizes, dtype=float)) ** 2
    ax.loglog(grid_sizes, reference, "--", label="Second-order reference")
    ax.set_xlabel("Grid points per axis")
    ax.set_ylabel("Relative L2 error")
    ax.set_title(summary["title"])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "convergence.png", dpi=150)
    plt.close(fig)

    summary = dict(summary)
    summary["grid_sizes"] = list(grid_sizes)
    summary["potential_relative_l2"] = errors[:, 0].tolist()
    summary["field_relative_l2"] = errors[:, 1].tolist()
    summary["potential_observed_orders"] = potential_orders.tolist()
    summary["field_observed_orders"] = field_orders.tolist()
    summary["convergence_csv"] = str(csv_path)
    summary["convergence_figure"] = str(figures_dir / "convergence.png")
    return write_summary(case_name, summary)


def _relative_l2(measured: np.ndarray, theory: np.ndarray, mask: np.ndarray) -> float:
    return float(np.linalg.norm((measured - theory)[mask]) / np.linalg.norm(theory[mask]))


def _observed_orders(errors: np.ndarray) -> np.ndarray:
    return np.log2(errors[:-1] / errors[1:])


def _rectangle_manufactured_error(
    grid_points: int, field_solver: str = "fd"
) -> tuple[float, float]:
    a = RECTANGLE_HALF_WIDTH_X_M
    b = RECTANGLE_HALF_WIDTH_Y_M
    geometry = GridGeometry(grid_points, grid_points, -a, a, -b, b)
    xx, yy = np.meshgrid(geometry.x, geometry.y)
    kx = np.pi / (2.0 * a)
    ky = np.pi / (2.0 * b)
    potential = POTENTIAL_SCALE_V_M * np.cos(kx * xx) * np.cos(ky * yy)
    density = const.epsilon0 * (kx * kx + ky * ky) * potential
    theory_ex = POTENTIAL_SCALE_V_M * kx * np.sin(kx * xx) * np.cos(ky * yy)
    theory_ey = POTENTIAL_SCALE_V_M * ky * np.cos(kx * xx) * np.sin(ky * yy)

    result = build_pic_resources(
        geometry, field_solver=field_solver
    ).field_solver.solve(density)
    interior = np.ones_like(potential, dtype=bool)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    potential_error = _relative_l2(result.potential, potential, interior)
    measured_field = np.stack((result.integrated_ex, result.integrated_ey))
    theory_field = np.stack((theory_ex, theory_ey))
    field_mask = np.broadcast_to(interior, measured_field.shape)
    field_error = _relative_l2(measured_field, theory_field, field_mask)
    return potential_error, field_error


def _ellipse_geometry(grid_points: int) -> GridGeometry:
    return GridGeometry(
        grid_points,
        grid_points,
        -ELLIPSE_A_M,
        ELLIPSE_A_M,
        -ELLIPSE_B_M,
        ELLIPSE_B_M,
    )


def _ellipse_resources(geometry: GridGeometry):
    return build_pic_resources(
        geometry,
        aperture={"Type": "ellipse", "A": ELLIPSE_A_M, "B": ELLIPSE_B_M},
        field_solver="fd",
    )


def _elliptic_uniform_charge_metrics() -> dict[str, float]:
    geometry = _ellipse_geometry(129)
    xx, yy = np.meshgrid(geometry.x, geometry.y)
    normalized_radius_squared = (xx / ELLIPSE_A_M) ** 2 + (yy / ELLIPSE_B_M) ** 2
    potential = POTENTIAL_SCALE_V_M * (1.0 - normalized_radius_squared)
    density_value = 2.0 * const.epsilon0 * POTENTIAL_SCALE_V_M * (
        1.0 / ELLIPSE_A_M**2 + 1.0 / ELLIPSE_B_M**2
    )
    density = np.where(normalized_radius_squared <= 1.0, density_value, 0.0)
    theory_ex = 2.0 * POTENTIAL_SCALE_V_M * xx / ELLIPSE_A_M**2
    theory_ey = 2.0 * POTENTIAL_SCALE_V_M * yy / ELLIPSE_B_M**2

    resources = _ellipse_resources(geometry)
    result = resources.field_solver.solve(density)
    interior = resources.field_solver.interior_mask
    potential_error = _relative_l2(result.potential, potential, interior)
    measured_field = np.stack((result.integrated_ex, result.integrated_ey))
    theory_field = np.stack((theory_ex, theory_ey))
    field_mask = np.broadcast_to(interior, measured_field.shape)
    field_error = _relative_l2(measured_field, theory_field, field_mask)

    outside = ~resources.aperture_mask
    outside_max = float(
        max(
            np.max(np.abs(result.potential[outside])),
            np.max(np.abs(result.integrated_ex[outside])),
            np.max(np.abs(result.integrated_ey[outside])),
        )
    )
    return {
        "potential_relative_l2": potential_error,
        "field_relative_l2": field_error,
        "outside_max_abs": outside_max,
    }


def _elliptic_quartic_manufactured_error(grid_points: int) -> tuple[float, float]:
    geometry = _ellipse_geometry(grid_points)
    xx, yy = np.meshgrid(geometry.x, geometry.y)
    u = (xx / ELLIPSE_A_M) ** 2 + (yy / ELLIPSE_B_M) ** 2
    inside = u <= 1.0
    potential = np.where(inside, POTENTIAL_SCALE_V_M * (1.0 - u) ** 2, 0.0)
    inverse_square_sum = 1.0 / ELLIPSE_A_M**2 + 1.0 / ELLIPSE_B_M**2
    weighted_radius = xx**2 / ELLIPSE_A_M**4 + yy**2 / ELLIPSE_B_M**4
    density = np.where(
        inside,
        const.epsilon0
        * POTENTIAL_SCALE_V_M
        * (4.0 * (1.0 - u) * inverse_square_sum - 8.0 * weighted_radius),
        0.0,
    )
    theory_ex = 4.0 * POTENTIAL_SCALE_V_M * xx * (1.0 - u) / ELLIPSE_A_M**2
    theory_ey = 4.0 * POTENTIAL_SCALE_V_M * yy * (1.0 - u) / ELLIPSE_B_M**2

    resources = _ellipse_resources(geometry)
    result = resources.field_solver.solve(density)
    interior = resources.field_solver.interior_mask
    potential_error = _relative_l2(result.potential, potential, interior)
    measured_field = np.stack((result.integrated_ex, result.integrated_ey))
    theory_field = np.stack((theory_ex, theory_ey))
    field_mask = np.broadcast_to(interior, measured_field.shape)
    field_error = _relative_l2(measured_field, theory_field, field_mask)
    return potential_error, field_error


def _command_simulation(field_solver: str, deposition_method: str):
    x = np.array([-6.0e-3, -2.0e-3, 0.0, 2.0e-3, 6.0e-3])
    y = np.array([0.0, 3.0e-3, 0.0, -3.0e-3, 0.0])
    particle_count = x.size
    particles = SimpleNamespace(
        x=x,
        y=y,
        px=np.zeros(particle_count),
        py=np.zeros(particle_count),
        z=np.linspace(-0.2, 0.2, particle_count),
        dp=np.linspace(-1.0e-3, 1.0e-3, particle_count),
        tag=np.ones(particle_count, dtype=np.int32),
    )
    slice_set = SimpleNamespace(
        slice_id=np.zeros(particle_count, dtype=np.int32),
        slice_table={"delta_z": np.array([0.01])},
    )
    bunch = SimpleNamespace(
        bunch_id=0,
        harmonic_id=0,
        harmonic_number=1,
        start_idx=0,
        end_idx=particle_count,
        slice_sets={"space_charge": slice_set},
        ratio=1.0e9,
        num_charge=1,
        beta=0.5,
        gamma=1.1547005383792517,
        brho=3.0,
        Np=particle_count,
        Nrp=int(particle_count * 1.0e9),
    )
    beam = SimpleNamespace(particles=particles, bunches=[bunch], beam_name="fd-dst-command")
    settings = SpaceChargeConfig(
        enabled=True,
        configurations={
            "test": SpaceChargeResourceConfig(
                slice_set="space_charge",
                nx=33,
                ny=33,
                grid_width_x=0.04,
                grid_width_y=0.04,
                solver={'fd': 'fd_dirichlet', 'dst_rectangle': 'dst_dirichlet', 'fft_free_space': 'fft_free_space'}[field_solver],
                deposition_method=deposition_method,
            )
        },
    )
    cfg = SimpleNamespace(
        output_dir=".",
        output_dir_space_charge=".",
        output_hms="test",
        backend="cpu",
        particle_precision="float64",
        input_data=[
            {
                "sequence": {
                    "slicer": {"command": "slicer", "slice set": "space_charge"},
                    "space_charge": {"command": "spacecharge", "configuration": "test"},
                }
            }
        ],
        space_charge=[settings],
        space_charge_configuration_counts=[1],
    )
    return SimpleNamespace(beams=[beam], state=SimpleNamespace(turn=0), cfg=cfg)


def _fd_dst_pic_metrics() -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(20260912)
    particle_count = 8192
    particles = {
        "x": rng.normal(0.0, 0.006, particle_count),
        "y": rng.normal(0.0, 0.005, particle_count),
        # The PIC solver uses x/y only, but keep a complete generated
        # transverse phase space so this remains representative beam data.
        "px": rng.normal(0.0, 0.006, particle_count),
        "py": rng.normal(0.0, 0.005, particle_count),
        "tag": np.ones(particle_count, dtype=np.int8),
    }
    slice_id = np.arange(particle_count, dtype=np.int32) % 2
    geometry = GridGeometry(65, 65, -0.030, 0.030, -0.025, 0.025)
    charge_per_macro = 2.5e-15

    metrics: dict[str, dict[str, float]] = {}
    for method, gather in (("CIC", gather_bilinear), ("TSC", gather_quadratic)):
        fd_resources = build_pic_resources(geometry, field_solver="fd")
        dst_resources = build_pic_resources(geometry, field_solver="dst_rectangle")
        fd = solve_pic(
            particles,
            slice_id,
            geometry,
            fd_resources,
            method,
            charge_per_macro=charge_per_macro,
            num_slices=2,
        )
        dst = solve_pic(
            particles,
            slice_id,
            geometry,
            dst_resources,
            method,
            charge_per_macro=charge_per_macro,
            num_slices=2,
        )

        method_metrics = {
            "density_max_abs_difference": float(np.max(np.abs(fd.density - dst.density))),
            "deposited_charge_max_abs_difference": float(
                np.max(np.abs(fd.deposited_charge - dst.deposited_charge))
            ),
        }
        for name in ("potential", "integrated_ex", "integrated_ey"):
            measured = getattr(fd, name)
            theory = getattr(dst, name)
            relative_l2 = float(np.linalg.norm(measured - theory) / np.linalg.norm(theory))
            method_metrics[f"{name}_relative_l2"] = relative_l2

        for component in ("integrated_ex", "integrated_ey"):
            fd_gathered = gather(
                getattr(fd, component),
                particles,
                geometry,
                fd_resources,
                slice_id,
            )
            dst_gathered = gather(
                getattr(dst, component),
                particles,
                geometry,
                dst_resources,
                slice_id,
            )
            relative_l2 = float(
                np.linalg.norm(fd_gathered - dst_gathered) / np.linalg.norm(dst_gathered)
            )
            method_metrics[f"gathered_{component}_relative_l2"] = relative_l2
        metrics[method] = method_metrics
    return metrics


def _execute_command(field_solver: str, deposition_method: str):
    simulation = _command_simulation(field_solver, deposition_method)
    particles = simulation.beams[0].particles
    unchanged = {
        name: getattr(particles, name).copy() for name in ("x", "y", "z", "dp", "tag")
    }
    command = SpaceChargeCommand(
        0,
        simulation,
        **{"S (m)": 0.0, "Configuration": "test", "SC length (m)": 0.1},
    )
    assert command.execute_cpu(simulation) is True
    for name, expected in unchanged.items():
        np.testing.assert_array_equal(getattr(particles, name), expected)
    return particles.px.copy(), particles.py.copy()


def _fd_dst_command_kick_metrics() -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for method in ("CIC", "TSC"):
        fd_px, fd_py = _execute_command("fd", method)
        dst_px, dst_py = _execute_command("dst_rectangle", method)
        measured = np.concatenate((fd_px, fd_py))
        theory = np.concatenate((dst_px, dst_py))
        relative_l2 = float(np.linalg.norm(measured - theory) / np.linalg.norm(theory))
        metrics[method] = {
            "reference_kick_l2_norm": float(np.linalg.norm(theory)),
            "kick_relative_l2": relative_l2,
        }
    return metrics
