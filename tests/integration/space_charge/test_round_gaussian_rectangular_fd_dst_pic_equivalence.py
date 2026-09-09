"""FD/DST PIC equivalence for a fixed-seed Gaussian source in a rectangle."""

from tests.integration.space_charge._fd_validation_common import _fd_dst_pic_metrics, write_summary


def test_gaussian_fd_and_dst_match_for_pic_deposit_solve_and_gather():
    metrics = _fd_dst_pic_metrics()
    write_summary(
        "round_gaussian_rectangular_fd_dst_pic_equivalence",
        {
            "title": "Round Gaussian rectangular-boundary FD/DST PIC equivalence",
            "source": "complete Gaussian x/px/y/py phase space generated with numpy seed 20260912",
            "field_solvers": ["fd", "dst_rectangle"],
            "deposition_methods": ["CIC", "TSC"],
            "metrics": metrics,
        },
    )

    for method, method_metrics in metrics.items():
        assert method_metrics["density_max_abs_difference"] == 0.0
        assert method_metrics["deposited_charge_max_abs_difference"] == 0.0
        for name, relative_l2 in method_metrics.items():
            if name.endswith("_relative_l2"):
                assert relative_l2 < 1.0e-11, (
                    f"FD and DST two-slice {method} {name} disagree: "
                    f"measured_relative_l2={relative_l2:.8g}, theory=0, tolerance=1e-11"
                )
