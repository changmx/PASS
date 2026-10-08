"""Run Example 05 cases through the pass-run command-line entry point.

Usage:
    python run_simulation.py
    python run_simulation.py --case twiss_h1_fixed
    python run_simulation.py --case all
    python run_simulation.py --beam0 path/to/input.json

Each case reads beam0_<name>.json and writes output/<name>/YYYY_MMDD/HHMM_SS/.
"""

import argparse
from pathlib import Path

from PASS.main import cli_main
from generate_input import CASES, input_path, selected_cases


def run_case(name: str, *, output: str | None = None, stop_file: str | None = None) -> int:
    beam0 = input_path(name)
    if not beam0.exists():
        raise FileNotFoundError(f"Missing input file: {beam0} (run generate_input.py first)")
    print(f"[run] {beam0}")
    return run(str(beam0), output=output, stop_file=stop_file)


def run(beam0_path: str, *, output: str | None = None, stop_file: str | None = None) -> int:
    argv = ["--beam0", str(beam0_path)]
    if output is not None:
        argv.extend(["--output", str(output)])
    if stop_file is not None:
        argv.extend(["--stop-file", str(stop_file)])
    return cli_main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run generated Example 05 cases. For a single input, use pass-run --beam0 FILE.")
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
    parser.add_argument("--output", help="Override the output root for every selected case")
    parser.add_argument("--stop-file", help="Stop when this file exists (forwarded to pass-run)")
    args = parser.parse_args(argv)

    if args.beam0:
        paths = [Path(args.beam0)]
    else:
        paths = [input_path(case_name) for case_name in selected_cases(args.case)]

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
