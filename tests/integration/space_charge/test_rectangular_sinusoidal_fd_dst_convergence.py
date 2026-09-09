"""FD and DST convergence against a rectangular sinusoidal exact solution."""

import csv
import json

import matplotlib

import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tests.integration.space_charge._fd_validation_common import (
    OUTPUT_ROOT,
    _observed_orders,
    _rectangle_manufactured_error,
)


def test_rectangular_sinusoidal_fd_and_dst_converge_at_second_order():
    grid_sizes = (33, 65, 129)
    errors = {
        solver: np.asarray(
            [_rectangle_manufactured_error(points, solver) for points in grid_sizes]
        )
        for solver in ("fd", "dst_rectangle")
    }
    orders = {
        solver: {
            "potential": _observed_orders(values[:, 0]),
            "field": _observed_orders(values[:, 1]),
        }
        for solver, values in errors.items()
    }
    case_dir = OUTPUT_ROOT / "rectangular_sinusoidal_fd_dst_convergence"
    analysis_dir = case_dir / "analysis"
    figures_dir = case_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / "fd_dst_relative_error.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "grid_points",
                "fd_potential_relative_l2",
                "dst_potential_relative_l2",
                "fd_field_relative_l2",
                "dst_field_relative_l2",
            ]
        )
        for index, points in enumerate(grid_sizes):
            writer.writerow(
                [points, errors["fd"][index, 0], errors["dst_rectangle"][index, 0],
                 errors["fd"][index, 1], errors["dst_rectangle"][index, 1]]
            )
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    styles = {
        ("fd", 0): "o-", ("dst_rectangle", 0): "s--",
        ("fd", 1): "^-", ("dst_rectangle", 1): "v--",
    }
    for solver, label in (("fd", "FD"), ("dst_rectangle", "DST")):
        ax.loglog(grid_sizes, errors[solver][:, 0], styles[(solver, 0)], markersize=3,
                  label=f"{label} potential vs exact")
        ax.loglog(grid_sizes, errors[solver][:, 1], styles[(solver, 1)], markersize=3,
                  label=f"{label} field vs exact")
    ax.set(
        xlabel="Grid points per axis",
        ylabel="Relative L2 error",
        title="Rectangular sinusoidal solution: FD and DST relative errors",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(figures_dir / "fd_dst_relative_error_same_plot.png", dpi=150)
    plt.close(fig)
    summary = {
        "case": "rectangular_sinusoidal_fd_dst_convergence",
        "boundary": "grounded rectangle",
        "reference": "analytic sinusoidal potential and electric field",
        "grid_sizes": list(grid_sizes),
        "errors": {name: value.tolist() for name, value in errors.items()},
        "orders": {
            name: {component: value.tolist() for component, value in solver_orders.items()}
            for name, solver_orders in orders.items()
        },
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    for solver, values in errors.items():
        assert np.all(orders[solver]["potential"] > 1.95), (
            f"{solver} potential convergence mismatch: measured_errors={values[:, 0].tolist()}, "
            f"orders={orders[solver]['potential'].tolist()}, theory_order=2, tolerance_min=1.95"
        )
        assert np.all(orders[solver]["field"] > 1.95), (
            f"{solver} field convergence mismatch: measured_errors={values[:, 1].tolist()}, "
            f"orders={orders[solver]['field'].tolist()}, theory_order=2, tolerance_min=1.95"
        )
        assert values[-1, 0] < 6.0e-5 and values[-1, 1] < 6.0e-5, (
            f"{solver} 129x129 error exceeded tolerance: potential={values[-1, 0]:.8g}, "
            f"field={values[-1, 1]:.8g}, theory=0, tolerance=6e-5"
        )
