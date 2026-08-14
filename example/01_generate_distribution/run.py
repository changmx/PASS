"""Run an Example 01 distribution-generation case.

Usage:
    python run.py
    python run.py --case longi-matchz
    python run.py --case all
    python run.py --beam0 path/to/input.json
"""

import argparse
from pathlib import Path

from PASS.main import main as pass_main

from make_input import CASES, input_path, selected_cases


def run(beam0_path: str, is_cal_phase: bool = True):
    """Run one generated input file."""
    pass_main(beam0_path, None, is_cal_phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PASS simulation")
    parser.add_argument(
        "--case",
        choices=["all", *CASES],
        default="transverse",
        help="Generated case to run (default: transverse).",
    )
    parser.add_argument(
        "--beam0",
        default=None,
        help="Explicit input path. Cannot be combined with --case all.",
    )
    parser.add_argument("--no-cal-phase", action="store_true",
                        help="Disable phase calculation")
    args = parser.parse_args()

    if args.beam0:
        if args.case == "all":
            parser.error("--beam0 cannot be combined with --case all")
        paths = [Path(args.beam0)]
    else:
        paths = [input_path(case_name) for case_name in selected_cases(args.case)]

    for path in paths:
        if not path.exists():
            parser.error(f"Input file does not exist: {path}. Run make_input.py first.")
        print(f"[Run] {path.name}")
        run(str(path), is_cal_phase=not args.no_cal_phase)
