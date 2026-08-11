"""Run PASS simulation.

Usage:
    python run.py
    python run.py --beam0 path/to/beam0.json --no-cal-phase
"""

import argparse
from pathlib import Path

from PASS.main import main as pass_main


def run(beam0_path: str, beam1_path: str | None = None, is_cal_phase: bool = True):
    pass_main(beam0_path, beam1_path, is_cal_phase)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Run PASS simulation")
    parser.add_argument("--beam0", default=str(script_dir / "beam0.json"),
                        help="Path to beam0.json")
    parser.add_argument("--beam1", default=None, help="Path to beam1.json (optional)")
    parser.add_argument("--no-cal-phase", action="store_true",
                        help="Disable phase calculation")
    args = parser.parse_args()

    run(args.beam0, args.beam1, is_cal_phase=not args.no_cal_phase)
