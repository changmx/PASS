"""Elliptic KV transverse-field validation for the FFT Green solver.

Run from the repository root, for example::

    python -m tests.integration.space_charge.test_elliptic_kv_free_space_fft simana
    python -m tests.integration.space_charge.test_elliptic_kv_free_space_fft ana
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.integration.space_charge._elliptic_free_space_fft_common import (
    CASES,
    analyse as _analyse,
    simulate as _simulate,
)


CASE = CASES["kv"]
DEFAULT_RUN_DIR = CASE.default_run_dir


def simulate(run_dir: Path = DEFAULT_RUN_DIR) -> Path:
    """Generate and track a 4-D KV source with a uniform elliptic x-y projection."""
    return _simulate(CASE, run_dir.resolve())


def analyse(run_dir: Path = DEFAULT_RUN_DIR) -> dict:
    """Compare the saved field with the KV projected uniform-ellipse field."""
    return _analyse(CASE, run_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sim", "ana", "simana"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    if args.mode in {"sim", "simana"}:
        simulate(args.run_dir)
    if args.mode in {"ana", "simana"}:
        analyse(args.run_dir)


def test_full_workflow(sc_workflow):
    """Generate the full source, run PASS, and enforce the existing analysis checks."""
    sc_workflow("elliptic_kv_free_space_fft")


if __name__ == "__main__":
    main()
