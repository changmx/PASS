"""Run PASS simulation for one Example-05 case.

Usage:
    python run.py
    python run.py --case twiss_h1_fixed
    python run.py --case all
    python run.py --beam0 path/to/input.json

Each case reads beam0_<name>.json and writes output/<name>/YYYY_MMDD/HHMM_SS/.
"""

import argparse
from pathlib import Path

from PASS.main import main as pass_main

from make_input import CASES, input_path, selected_cases


def run_case(name: str) -> None:
    beam0 = input_path(name)
    if not beam0.exists():
        raise FileNotFoundError(f"Missing input file: {beam0} (run make_input.py first)")
    print(f"[run] {beam0}")
    pass_main(str(beam0))


def run(beam0_path: str, is_cal_phase: bool = True) -> None:
    pass_main(beam0_path, None, is_cal_phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Example 05 simulation")
    parser.add_argument(
        "--case",
        choices=["all", *CASES],
        default="all",
        help="Generated case to run (default: all).",
    )
    parser.add_argument(
        "--beam0",
        default=None,
        help="Explicit input path. Overrides --case.",
    )
    parser.add_argument("--no-cal-phase", action="store_true",
                        help="Disable phase calculation")
    args = parser.parse_args()

    if args.beam0:
        paths = [Path(args.beam0)]
    else:
        paths = [input_path(case_name) for case_name in selected_cases(args.case)]

    for path in paths:
        if not path.exists():
            parser.error(f"Input file does not exist: {path}. Run make_input.py first.")
        print(f"[Run] {path.name}")
        run(str(path), is_cal_phase=not args.no_cal_phase)
