from PASS.main import main

import argparse
from pathlib import Path


def run(beam0_path: str, beam1_path: str | None = None, is_cal_phase: bool = True):
    abs_beam0 = str(Path(beam0_path).resolve())
    abs_beam1 = str(Path(beam1_path).resolve()) if beam1_path else None
    main(abs_beam0, abs_beam1, is_cal_phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PASS simulation")
    parser.add_argument("--beam0", required=True, help="Path to beam0.json (required)")
    parser.add_argument("--beam1", help="Path to beam1.json (optional)")
    parser.add_argument("--no-cal-phase", action="store_true", help="Disable phase calculation (default: calculate phase)")

    args = parser.parse_args()

    is_cal_phase = not args.no_cal_phase
    run(args.beam0, args.beam1, is_cal_phase)
