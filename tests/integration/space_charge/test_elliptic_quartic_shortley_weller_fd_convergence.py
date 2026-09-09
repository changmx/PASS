"""Shortley-Weller FD convergence on an ellipse with a quartic exact solution."""

import numpy as np

from tests.integration.space_charge._fd_validation_common import (
    _elliptic_quartic_manufactured_error,
    _observed_orders,
    write_convergence_outputs,
)


def test_elliptic_quartic_shortley_weller_fd_converges_at_second_order():
    grid_sizes = (33, 65, 129)
    errors = np.asarray(
        [_elliptic_quartic_manufactured_error(points) for points in grid_sizes]
    )
    potential_orders = _observed_orders(errors[:, 0])
    field_orders = _observed_orders(errors[:, 1])
    write_convergence_outputs(
        "elliptic_quartic_shortley_weller_fd_convergence",
        grid_sizes,
        errors,
        potential_orders,
        field_orders,
        {
            "title": "Elliptic quartic manufactured solution: Shortley-Weller FD convergence",
            "source": "analytic quartic charge density",
            "field_solver": "fd",
            "boundary_discretization": "Shortley-Weller",
            "expected_order": 2.0,
        },
    )

    assert np.all(potential_orders > 1.85), (
        "elliptic Shortley-Weller potential convergence is too slow: "
        f"grid_sizes={grid_sizes}, measured_errors={errors[:, 0].tolist()}, "
        f"measured_orders={potential_orders.tolist()}, expected_near=2, tolerance_min=1.85"
    )
    assert np.all(field_orders > 1.90), (
        "elliptic Shortley-Weller field convergence is too slow: "
        f"grid_sizes={grid_sizes}, measured_errors={errors[:, 1].tolist()}, "
        f"measured_orders={field_orders.tolist()}, expected_near=2, tolerance_min=1.90"
    )
    assert errors[-1, 0] < 3.5e-4 and errors[-1, 1] < 3.5e-4, (
        "129x129 elliptic Shortley-Weller error exceeded tolerance: "
        f"potential_measured={errors[-1, 0]:.8g}, field_measured={errors[-1, 1]:.8g}, "
        "theory=0, tolerance=3.5e-4"
    )
