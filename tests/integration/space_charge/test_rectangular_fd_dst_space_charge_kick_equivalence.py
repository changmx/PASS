"""SpaceCharge kick equivalence between FD and DST on a rectangular boundary."""

from tests.integration.space_charge._fd_validation_common import (
    _fd_dst_command_kick_metrics,
    write_summary,
)


def test_fd_and_dst_commands_produce_the_same_cic_and_tsc_kicks():
    metrics = _fd_dst_command_kick_metrics()
    write_summary(
        "rectangular_fd_dst_space_charge_kick_equivalence",
        {
            "title": "Rectangular-boundary FD/DST SpaceCharge kick equivalence",
            "source": "deterministic symmetric five-particle command probe",
            "field_solvers": ["fd", "dst_rectangle"],
            "deposition_methods": ["CIC", "TSC"],
            "metrics": metrics,
        },
    )

    for method, method_metrics in metrics.items():
        assert method_metrics["reference_kick_l2_norm"] > 0.0, (
            f"{method} reference kick unexpectedly vanished: measured_norm=0, theory_norm>0"
        )
        assert method_metrics["kick_relative_l2"] < 1.0e-12, (
            f"FD and DST command-level {method} kicks disagree: "
            f"measured_relative_l2={method_metrics['kick_relative_l2']:.8g}, "
            "theory=0, tolerance=1e-12"
        )
