"""Run Example 07 through the pass-run command-line entry point."""

import argparse
from pathlib import Path

from PASS.main import cli_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Example 07. You can also use pass-run --beam0 FILE.")
    parser.add_argument("--beam0", "--input", type=Path, default=Path(__file__).with_name("beam0.json"), help="Input JSON (--input is also accepted)")
    parser.add_argument("--output", help="Override the output root")
    parser.add_argument("--stop-file", help="Stop when this file exists (forwarded to pass-run)")
    args = parser.parse_args(argv)
    cli_argv = ["--beam0", str(args.beam0)]
    if args.output is not None:
        cli_argv.extend(["--output", args.output])
    if args.stop_file is not None:
        cli_argv.extend(["--stop-file", args.stop_file])
    return cli_main(cli_argv)


if __name__ == "__main__":
    raise SystemExit(main())
