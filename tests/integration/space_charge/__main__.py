"""Batch space-charge validation: python -m tests.integration.space_charge --help."""
import argparse
from pathlib import Path
import sys

from ._suite import GROUPS, WORKFLOWS, WorkflowRunner, prepare_output_dir, pytest_targets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", nargs="?", choices=GROUPS, default="all")
    parser.add_argument("--case", action="append", choices=WORKFLOWS,
                        help="Select a workflow by name; repeat to select several (overrides group).")
    parser.add_argument("--mode", choices=("test", "sim", "ana", "simana"), default="test",
                        help="test runs pytest; other modes run only the selected full workflows.")
    parser.add_argument("--output-dir", type=Path, help="New batch directory; ana requires an existing batch.")
    parser.add_argument("--list", action="store_true", help="List selected pytest targets without running them.")
    arguments = list(sys.argv[1:] if argv is None else argv)
    forwarded = []
    if "--" in arguments:
        split = arguments.index("--")
        forwarded, arguments = arguments[split + 1:], arguments[:split]
    args, pytest_args = parser.parse_known_args(arguments)
    pytest_args += forwarded
    targets = pytest_targets(args.group, args.case)
    if args.list:
        print("\n".join(targets))
        return 0
    if args.mode == "test":
        import pytest
        options = ["-q", *targets, *pytest_args]
        if args.output_dir:
            options += ["--sc-output-dir", str(args.output_dir.resolve())]
        if args.group == "regression" and not args.case and args.output_dir:
            parser.error("--output-dir applies to integration cases; regression uses pytest tmp_path.")
        return int(pytest.main(options))
    if pytest_args:
        parser.error("Additional pytest arguments are only supported with --mode test.")
    selected = args.case or [case for case in GROUPS[args.group] if case in WORKFLOWS]
    if not selected:
        parser.error("This group has no full workflows; use --mode test.")
    if args.mode == "ana" and args.output_dir is None:
        parser.error("--mode ana requires --output-dir pointing to an existing batch.")
    try:
        output_dir = prepare_output_dir(args.output_dir, args.mode)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Output directory: {output_dir}")
    runner = WorkflowRunner(output_dir, args.mode)
    failures = []
    for case in dict.fromkeys(selected):
        try:
            runner.run(case)
        except RuntimeError as exc:
            failures.append(case)
            print(exc)
    print(f"Completed {len(runner.completed)} workflows; failed selections: {len(failures)}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
