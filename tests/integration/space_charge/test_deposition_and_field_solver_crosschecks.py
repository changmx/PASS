"""Cross-check PIC deposition, field solvers, and particle gathering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm, qmc

from PASS.commands.solver.formula_gaussian_round import gaussian_round_field
from PASS.commands.solver.pic import GridGeometry, build_pic_resources, solve_pic


SLICE_CHARGE_C = 1.602176634e-8
SIGMA_M = 0.015
GRID_HALF_WIDTH_M = 0.100
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "output"
    / "deposition_and_field_solver_crosschecks"
    / "analysis"
)


def _write_result(name: str, payload: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _round_gaussian_density(geometry: GridGeometry) -> np.ndarray:
    xx, yy = np.meshgrid(geometry.x, geometry.y)
    density = np.exp(-(xx * xx + yy * yy) / (2.0 * SIGMA_M**2))
    # Normalize the represented, finite-grid source exactly.  At +/-6.7 sigma
    # the distinction from an unbounded Gaussian is below this test's error.
    return density * SLICE_CHARGE_C / (density.sum() * geometry.dx * geometry.dy)


def _probe_coordinates() -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    radii = np.linspace(0.010, 0.075, 27)
    radius, angle = np.meshgrid(radii, angles, indexing="ij")
    return (radius * np.cos(angle)).ravel(), (radius * np.sin(angle)).ravel()


def _sample_field(
    geometry: GridGeometry,
    ex_grid: np.ndarray,
    ey_grid: np.ndarray,
    x_probe: np.ndarray,
    y_probe: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = (y_probe, x_probe)
    ex = RegularGridInterpolator((geometry.y, geometry.x), ex_grid, bounds_error=True)(points)
    ey = RegularGridInterpolator((geometry.y, geometry.x), ey_grid, bounds_error=True)(points)
    return ex, ey


def _fft_error(grid_points: int) -> tuple[float, float]:
    geometry = GridGeometry(
        grid_points,
        grid_points,
        -GRID_HALF_WIDTH_M,
        GRID_HALF_WIDTH_M,
        -GRID_HALF_WIDTH_M,
        GRID_HALF_WIDTH_M,
    )
    density = _round_gaussian_density(geometry)
    solver = build_pic_resources(geometry, field_solver="fft_free_space").field_solver
    result = solver.solve(density)
    x_probe, y_probe = _probe_coordinates()
    measured_ex, measured_ey = _sample_field(
        geometry, result.integrated_ex, result.integrated_ey, x_probe, y_probe
    )
    theory_ex, theory_ey = gaussian_round_field(
        x_probe, y_probe, SLICE_CHARGE_C, SIGMA_M
    )
    theory_peak = np.hypot(theory_ex, theory_ey).max()
    error = np.hypot(measured_ex - theory_ex, measured_ey - theory_ey) / theory_peak
    return float(error.max()), float(np.sqrt(np.mean(error**2)))


def test_fft_free_space_round_gaussian_converges_at_second_order():
    grid_sizes = (65, 129, 257)
    errors = np.asarray([_fft_error(points) for points in grid_sizes])
    max_errors = errors[:, 0]
    rms_errors = errors[:, 1]
    orders = np.log2(rms_errors[:-1] / rms_errors[1:])
    _write_result(
        "fft_free_space_round_gaussian_convergence",
        {
            "distribution": "analytic round Gaussian density",
            "grid_sizes": list(grid_sizes),
            "max_peak_normalized_errors": max_errors.tolist(),
            "rms_peak_normalized_errors": rms_errors.tolist(),
            "observed_orders": orders.tolist(),
        },
    )

    assert np.all(np.diff(max_errors) < 0.0), (
        "FFT maximum error did not decrease monotonically: "
        f"grid_sizes={grid_sizes}, max_errors={max_errors.tolist()}"
    )
    assert np.all(orders > 1.8), (
        "FFT field did not approach second-order convergence: "
        f"grid_sizes={grid_sizes}, rms_errors={rms_errors.tolist()}, orders={orders.tolist()}"
    )
    assert rms_errors[-1] < 3.5e-4, (
        "257x257 FFT RMS peak-normalized field error exceeded tolerance: "
        f"measured={rms_errors[-1]:.8g}, theory=0, tolerance=3.5e-4"
    )


def _sobol_gaussian_particles() -> dict[str, np.ndarray]:
    exponent = 18
    unit_samples = qmc.Sobol(2, scramble=True, seed=20260911).random_base2(exponent)
    coordinates = norm.ppf(np.clip(unit_samples, 1.0e-12, 1.0 - 1.0e-12)) * 0.020
    momentum_unit_samples = qmc.Sobol(2, scramble=True, seed=20260913).random_base2(exponent)
    momenta = norm.ppf(
        np.clip(momentum_unit_samples, 1.0e-12, 1.0 - 1.0e-12)
    ) * 0.020
    return {
        "x": coordinates[:, 0],
        "y": coordinates[:, 1],
        "px": momenta[:, 0],
        "py": momenta[:, 1],
        "tag": np.ones(2**exponent, dtype=np.int8),
    }


def test_fft_cic_and_tsc_deposition_conserve_charge_and_match_theory():
    geometry = GridGeometry(129, 129, -0.100, 0.100, -0.100, 0.100)
    resources = build_pic_resources(geometry, field_solver="fft_free_space")
    particles = _sobol_gaussian_particles()
    particle_count = particles["x"].size
    slice_id = np.zeros(particle_count, dtype=np.int32)
    x_probe, y_probe = _probe_coordinates()
    # This source intentionally has a different sigma from the convergence
    # test, exercising the deposition path rather than an analytic node array.
    particle_sigma_m = 0.020
    theory_ex, theory_ey = gaussian_round_field(
        x_probe, y_probe, SLICE_CHARGE_C, particle_sigma_m
    )
    theory_peak = np.hypot(theory_ex, theory_ey).max()

    method_results = {}
    for method in ("CIC", "TSC"):
        result = solve_pic(
            particles,
            slice_id,
            geometry,
            resources,
            method,
            charge_per_macro=SLICE_CHARGE_C / particle_count,
            num_slices=1,
        )
        measured_charge = float(result.deposited_charge[0])
        charge_error = abs(measured_charge - SLICE_CHARGE_C) / SLICE_CHARGE_C
        measured_ex, measured_ey = _sample_field(
            geometry,
            result.integrated_ex[0],
            result.integrated_ey[0],
            x_probe,
            y_probe,
        )
        field_error = np.hypot(measured_ex - theory_ex, measured_ey - theory_ey) / theory_peak
        max_error = float(field_error.max())
        rms_error = float(np.sqrt(np.mean(field_error**2)))
        method_results[method] = {
            "deposited_charge_c": measured_charge,
            "charge_relative_error": charge_error,
            "max_peak_normalized_field_error": max_error,
            "rms_peak_normalized_field_error": rms_error,
        }

        assert charge_error < 1.0e-10, (
            f"{method} charge conservation failed: measured={measured_charge:.12g} C, "
            f"theory={SLICE_CHARGE_C:.12g} C, relative_error={charge_error:.8g}, "
            "tolerance=1e-10"
        )
        assert max_error < 3.0e-3, (
            f"{method} maximum peak-normalized field error failed: "
            f"measured={max_error:.8g}, theory=0, tolerance=0.003"
        )
        assert rms_error < 1.6e-3, (
            f"{method} RMS peak-normalized field error failed: "
            f"measured={rms_error:.8g}, theory=0, tolerance=0.0016"
        )
    _write_result(
        "round_gaussian_cic_tsc",
        {
            "distribution": "fixed-seed Sobol Gaussian x/px/y/py phase space",
            "coordinate_random_seed": 20260911,
            "momentum_random_seed": 20260913,
            "field_solver": "fft_free_space",
            "results": method_results,
        },
    )


def test_fd_and_dst_rectangle_solve_the_same_discrete_poisson_problem():
    geometry = GridGeometry(65, 65, -0.080, 0.080, -0.060, 0.060)
    density = _round_gaussian_density(geometry)
    # Dirichlet solvers ignore source values on the conducting wall.
    density[[0, -1], :] = 0.0
    density[:, [0, -1]] = 0.0
    fd = build_pic_resources(geometry, field_solver="fd").field_solver.solve(density)
    dst = build_pic_resources(geometry, field_solver="dst_rectangle").field_solver.solve(density)

    component_errors = {}
    for name in ("potential", "integrated_ex", "integrated_ey"):
        measured = getattr(fd, name)
        theoretical_reference = getattr(dst, name)
        relative_l2 = float(
            np.linalg.norm(measured - theoretical_reference)
            / np.linalg.norm(theoretical_reference)
        )
        component_errors[name] = relative_l2
        assert relative_l2 < 1.0e-12, (
            f"FD and DST {name} disagree for the same discrete equation: "
            f"measured_relative_l2={relative_l2:.8g}, theory=0, tolerance=1e-12"
        )

    boundary = np.concatenate(
        (fd.potential[0, :], fd.potential[-1, :], fd.potential[:, 0], fd.potential[:, -1])
    )
    boundary_max = float(np.max(np.abs(boundary)))
    _write_result(
        "round_gaussian_rectangular_fd_dst_discrete_equation",
        {
            "distribution": "analytic round Gaussian density",
            "field_solvers": ["fd", "dst_rectangle"],
            "component_relative_l2_errors": component_errors,
            "fd_boundary_max_abs": boundary_max,
        },
    )
    assert boundary_max == 0.0, (
        "FD zero-Dirichlet boundary failed: "
        f"measured_max_abs={boundary_max:.8g} V m, theory=0 V m, tolerance=0"
    )
