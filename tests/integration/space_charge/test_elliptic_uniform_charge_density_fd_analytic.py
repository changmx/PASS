"""Analytic FD validation for uniform charge density in an elliptic conductor."""

from tests.integration.space_charge._fd_validation_common import (
    ELLIPSE_A_M,
    ELLIPSE_B_M,
    _elliptic_uniform_charge_metrics,
    write_summary,
)


def test_elliptic_uniform_charge_density_fd_matches_exact_field():
    metrics = _elliptic_uniform_charge_metrics()
    write_summary(
        "elliptic_uniform_charge_density_fd_analytic",
        {
            "title": "Elliptic uniform charge density: FD analytic validation",
            "source": "analytic uniform spatial charge density (not PASS uniform distribution)",
            "field_solver": "fd",
            "ellipse_semi_axis_x_m": ELLIPSE_A_M,
            "ellipse_semi_axis_y_m": ELLIPSE_B_M,
            **metrics,
        },
    )

    assert metrics["potential_relative_l2"] < 5.0e-12, (
        "elliptic FD uniform-charge potential failed: "
        f"measured_relative_l2={metrics['potential_relative_l2']:.8g}, theory=0, tolerance=5e-12"
    )
    assert metrics["field_relative_l2"] < 5.0e-12, (
        "elliptic FD uniform-charge field failed: "
        f"measured_relative_l2={metrics['field_relative_l2']:.8g}, theory=0, tolerance=5e-12"
    )
    assert metrics["outside_max_abs"] == 0.0, (
        "elliptic FD returned a nonzero solution outside the conductor: "
        f"measured_max_abs={metrics['outside_max_abs']:.8g}, theory=0, tolerance=0"
    )
