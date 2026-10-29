"""Replot saved results: python -m PASS.plot RUN_DIRECTORY."""

import argparse
import logging

from PASS.plot.plot_main import plot_run


def main():
    parser = argparse.ArgumentParser(description="Plot saved PASS monitor data without running tracking.")
    parser.add_argument("run_directory")
    parser.add_argument("--output", help="Image directory (default: RUN_DIRECTORY/plot)")
    parser.add_argument("--format", nargs="+", choices=("png", "pdf"), default=["png"], dest="formats")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--bins", type=int, default=80)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = plot_run(args.run_directory, save_dir=args.output, formats=args.formats, dpi=args.dpi, bins=args.bins)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return int(result["summary"]["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
