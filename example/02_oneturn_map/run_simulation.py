"""Run Example 02 through the pass-run command-line entry point.

Usage:
    python run_simulation.py
    pass-run --beam0 beam0.json
    python run_simulation.py --beam0 path/to/beam0.json
"""

import argparse
from pathlib import Path

from PASS.main import cli_main


def run(beam0_path: str, beam1_path: str | None = None, *, output: str | None = None, stop_file: str | None = None) -> int:
    argv = ["--beam0", str(beam0_path)]
    if beam1_path is not None:
        argv.extend(["--beam1", str(beam1_path)])
    if output is not None:
        argv.extend(["--output", str(output)])
    if stop_file is not None:
        argv.extend(["--stop-file", str(stop_file)])
    return cli_main(argv)


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Run Example 02. You can also use pass-run --beam0 FILE.")
    parser.add_argument("--beam0", default=str(script_dir / "beam0.json"), help="Path to beam0.json")
    parser.add_argument("--beam1", default=None, help="Path to beam1.json (optional)")
    parser.add_argument("--output", help="Override the output root")
    parser.add_argument("--stop-file", help="Stop when this file exists (forwarded to pass-run)")
    args = parser.parse_args(argv)

    return run(args.beam0, args.beam1, output=args.output, stop_file=args.stop_file)


if __name__ == "__main__":
    raise SystemExit(main())
