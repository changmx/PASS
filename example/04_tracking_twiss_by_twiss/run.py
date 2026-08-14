"""Run PASS simulation.

Usage:
    python run.py
    python run.py --beam0 path/to/beam0.json --no-cal-phase
"""

import argparse

from PASS.main import main as pass_main

from make_input import input_path


def run(beam0_path: str, beam1_path: str | None = None, is_cal_phase: bool = True):
    pass_main(beam0_path, beam1_path, is_cal_phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PASS simulation")
    parser.add_argument("--beam0", default=str(input_path()),
                        help="Path to beam0.json")
    parser.add_argument("--beam1", default=None, help="Path to beam1.json (optional)")
    parser.add_argument("--no-cal-phase", action="store_true",
                        help="Disable phase calculation")
    args = parser.parse_args()

    if not input_path().exists() and args.beam0 == str(input_path()):
        parser.error(f"Input file does not exist: {input_path()}. Run make_input.py first.")

    run(args.beam0, args.beam1, is_cal_phase=not args.no_cal_phase)
