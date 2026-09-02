"""Run Slicer integration cases through the PASS executor."""

from __future__ import annotations

from pathlib import Path

from PASS.main import main as pass_main

from .make_input import create_input


TEST_DIR = Path(__file__).resolve().parent


def run_case(case: str, backend: str, work_dir: str | Path | None = None) -> Path:
    """Run one case and retain its files below the test package directory."""
    work_dir = TEST_DIR if work_dir is None else Path(work_dir)
    output_root = work_dir / "output" / f"{case}_{backend}"
    input_path = create_input(
        output_root / f"beam0_{case}_{backend}.json",
        case,
        backend,
        output_root,
    )
    # Use the same public entry point as a normal PASS simulation.  The main
    # function owns logging, device selection, Beam construction, sequence
    # sorting, and Executor invocation.
    pass_main(str(input_path))

    # ``PASS.main.main`` currently catches runtime exceptions and returns
    # None, so locate the output it created and require its completion marker.
    candidates = [
        path
        for path in output_root.glob("*/*")
        if path.is_dir() and (path / "distribution").is_dir() and (path / "slice").is_dir()
    ]
    if not candidates:
        raise AssertionError(f"PASS.main produced no completed output under {output_root}")
    result_dir = max(candidates, key=lambda path: path.stat().st_mtime)
    logs = sorted(result_dir.glob("*.log"))
    if not logs:
        raise AssertionError(f"PASS.main produced no log under {result_dir}")
    log_text = logs[-1].read_text(encoding="utf-8")
    if "Simulation Completed" not in log_text:
        raise AssertionError(f"PASS.main did not complete; inspect {logs[-1]}")
    return result_dir
