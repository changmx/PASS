"""Full PASS FD/DST comparison for a round KV bunch in a rectangle."""

import argparse
from pathlib import Path

from tests.integration.space_charge._rectangular_fd_dst_simulation_common import (
    CASES,
    analyse as _analyse,
    simulate as _simulate,
)

CASE = CASES["round_kv"]
DEFAULT_RUN_DIR = CASE.default_run_dir


def simulate(run_dir: Path = DEFAULT_RUN_DIR):
    return _simulate(CASE, run_dir.resolve())


def analyse(run_dir: Path = DEFAULT_RUN_DIR) -> dict:
    return _analyse(CASE, run_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sim", "ana", "simana"), nargs="?", default="simana")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    if args.mode in {"sim", "simana"}:
        simulate(args.run_dir)
    if args.mode in {"ana", "simana"}:
        analyse(args.run_dir)


def test_full_workflow(sc_workflow):
    """Generate the full source, run PASS, and enforce the existing analysis checks."""
    sc_workflow("round_kv_rectangular_fd_dst")


if __name__ == "__main__":
    main()
