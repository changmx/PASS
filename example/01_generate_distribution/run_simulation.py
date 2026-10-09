"""Run Example 01 cases through the pass-run command-line entry point.

Usage:
    python run_simulation.py
    python run_simulation.py --case longi-matchz
    python run_simulation.py --case all
    python run_simulation.py --beam0 path/to/input.json
"""

import argparse
from pathlib import Path

from PASS.main import cli_main
from generate_input import CASES, SCRIPT_DIR, input_path, selected_cases


def run(beam0_path: str, *, output: str | None = None, stop_file: str | None = None) -> int:
    """Run one generated input file and return the pass-run exit code."""
    argv = ["--beam0", str(beam0_path)]
    if output is not None:
        argv.extend(["--output", str(output)])
    if stop_file is not None:
        argv.extend(["--stop-file", str(stop_file)])
    return cli_main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run generated Example 01 cases. For a single input, use pass-run --beam0 FILE.")
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
    parser.add_argument("--output", help="Override the output root for every selected case")
    parser.add_argument("--work-dir",
                        type=Path,
                        default=SCRIPT_DIR,
                        help="Directory containing generated case inputs (default: this example directory).")
    parser.add_argument("--stop-file", help="Stop when this file exists (forwarded to pass-run)")
    args = parser.parse_args(argv)

    if args.beam0:
        if args.case == "all":
            parser.error("--beam0 cannot be combined with --case all")
        paths = [Path(args.beam0)]
    else:
        paths = [input_path(case_name, args.work_dir) for case_name in selected_cases(args.case)]

    for path in paths:
        if not path.exists():
            parser.error(f"Input file does not exist: {path}. Run generate_input.py first.")
        print(f"[Run] {path.name}")
        exit_code = run(str(path), output=args.output, stop_file=args.stop_file)
        if exit_code != 0:
            return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
