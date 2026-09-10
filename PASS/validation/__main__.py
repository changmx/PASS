"""Usage: python -m PASS.validation beam0.json [beam1.json] --report report.json."""
import argparse
import json
from pathlib import Path

from . import validate_files


def main():
    parser = argparse.ArgumentParser(description="PASS comprehensive JSON preflight (no tracking)")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--report", type=Path, help="Write a machine-readable JSON report")
    args = parser.parse_args()
    report = validate_files(args.inputs)
    print(report.text())
    if args.report:
        args.report.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
