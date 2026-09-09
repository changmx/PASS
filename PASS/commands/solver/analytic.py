"""Particle-local free-space fields with fixed or per-slice transverse moments.

All fields are longitudinally integrated (V); charges are signed Coulombs.
The caller supplies slice membership. No slicing history or particle z is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .formula_gaussian_round import gaussian_round_field
from .formula_gaussian_ellipse import gaussian_elliptic_field
from .formula_uniform_round import uniform_round_field
from .formula_uniform_ellipse import uniform_elliptic_field
from .pic import PICResult


@dataclass
class AnalyticResult:
    integrated_ex: np.ndarray
    integrated_ey: np.ndarray
    slice_charge: np.ndarray
    macro_count: np.ndarray
    # Columns: center_x, center_y, size_x, size_y, angle; sizes are Gaussian
    # principal RMS widths or uniform semi-axes. Empty slices use NaN parameters.
    parameters: np.ndarray


def evaluate_profile(x, y, charge, parameters, solver):
    """Return lab-axis fields and local coordinates for one source profile."""
    cx, cy, sx, sy, angle = parameters
    c, s = np.cos(angle), np.sin(angle)
    dx, dy = x - cx, y - cy
    u, v = c * dx + s * dy, -s * dx + c * dy
    if solver == "gaussian_round_free_space":
        eu, ev = gaussian_round_field(u, v, charge, sx)
    elif solver == "gaussian_ellipse_free_space":
        eu, ev = gaussian_elliptic_field(u, v, charge, sx, sy)
    elif solver == "uniform_round_free_space":
        eu, ev = uniform_round_field(u, v, charge, sx)
    elif solver == "uniform_ellipse_free_space":
        eu, ev = uniform_elliptic_field(u, v, charge, sx, sy)
    else:
        raise ValueError(f"unsupported analytic solver {solver!r}")
    return c * eu - s * ev, s * eu + c * ev, u, v


def solve_analytic(x, y, slice_id, valid, num_slices, charge_per_macro, configuration):
    """Evaluate each nonempty slice; population moments use denominator N."""
    ex, ey = np.zeros_like(x, dtype=float), np.zeros_like(y, dtype=float)
    counts = np.bincount(slice_id[valid], minlength=num_slices)
    charges = counts * charge_per_macro
    parameters = np.full((num_slices, 5), np.nan)
    solver = configuration.solver
    round_profile = "_round_" in solver
    gaussian = solver.startswith("gaussian_")
    # Group once, rather than scanning all particles for every slice.
    active = np.flatnonzero(valid)
    order = active[np.argsort(slice_id[active], kind="stable")]
    offsets = np.r_[0, np.cumsum(counts)]
    for sid in np.flatnonzero(counts):
        indices = order[offsets[sid]:offsets[sid + 1]]
        if configuration.method == "frozen":
            cx, cy = configuration.center_x or 0.0, configuration.center_y or 0.0
            angle = configuration.angle or 0.0
            if round_profile:
                sx = sy = configuration.sigma if gaussian else configuration.radius
            elif gaussian:
                sx, sy = configuration.sigma_x, configuration.sigma_y
            else:
                sx, sy = configuration.a, configuration.b
        else:
            minimum = 2 if round_profile else 3
            if indices.size < minimum:
                raise ValueError(f"quasi-frozen slice {sid}: macro_count={indices.size}, requires at least {minimum}")
            cx, cy = float(np.mean(x[indices])), float(np.mean(y[indices]))
            dx, dy = x[indices] - cx, y[indices] - cy
            covariance = np.array([[np.mean(dx * dx), np.mean(dx * dy)],
                                   [np.mean(dx * dy), np.mean(dy * dy)]])
            if not np.all(np.isfinite(covariance)):
                raise ValueError(f"quasi-frozen slice {sid}: non-finite covariance")
            if round_profile:
                variance = np.trace(covariance) / 2.0
                if variance <= 0:
                    raise ValueError(f"quasi-frozen slice {sid}: zero transverse size")
                sx = sy = np.sqrt(variance) * (1.0 if gaussian else 2.0)
                angle = 0.0
            else:
                eigenvalues, axes = np.linalg.eigh(covariance)
                if eigenvalues[0] <= 64 * np.finfo(float).eps * eigenvalues[1]:
                    raise ValueError(f"quasi-frozen slice {sid}: degenerate covariance, eigenvalues={eigenvalues}")
                sy, sx = np.sqrt(eigenvalues) * (1.0 if gaussian else 2.0)
                angle = (np.arctan2(axes[1, 1], axes[0, 1]) + np.pi / 2) % np.pi - np.pi / 2
        parameters[sid] = cx, cy, sx, sy, angle
        ex[indices], ey[indices], _, _ = evaluate_profile(
            x[indices], y[indices], charges[sid], parameters[sid], solver)
    return AnalyticResult(ex, ey, charges, counts, parameters)


def sample_analytic_grid(result, configuration, geometry):
    """Diagnostic sampling only; no grid interpolation enters tracking."""
    x, y = np.meshgrid(geometry.x, geometry.y)
    shape = (result.slice_charge.size, geometry.ny, geometry.nx)
    density, ex, ey = (np.zeros(shape) for _ in range(3))
    gaussian = configuration.solver.startswith("gaussian_")
    for sid in np.flatnonzero(result.macro_count):
        charge = result.slice_charge[sid]
        ex[sid], ey[sid], u, v = evaluate_profile(
            x, y, charge, result.parameters[sid], configuration.solver)
        sx, sy = result.parameters[sid, 2:4]
        radius2 = (u / sx)**2 + (v / sy)**2
        if gaussian:
            density[sid] = charge / (2 * np.pi * sx * sy) * np.exp(-0.5 * radius2)
        else:
            density[sid] = charge / (np.pi * sx * sy) * (radius2 <= 1)
    return PICResult(density, None, ex, ey, geometry, result.slice_charge)
