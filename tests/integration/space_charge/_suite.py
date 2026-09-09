"""Case selection and serial execution shared by pytest and the batch CLI."""
from __future__ import annotations

from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = DIRECTORY.parents[2]
PACKAGE = "tests.integration.space_charge"
PREFIXES = ("round_gaussian", "round_kv", "elliptic_gaussian", "elliptic_kv")
FFT = tuple(f"{name}_free_space_fft" for name in PREFIXES)
RECTANGLE = tuple(f"{name}_rectangular_fd_dst" for name in PREFIXES)
APERTURE = ("round_kv_circular_boundary_fd", "elliptic_kv_elliptic_boundary_fd")
WORKFLOWS = FFT + RECTANGLE + APERTURE
CROSSCHECK = "deposition_and_field_solver_crosschecks"
CHECKS = (
    CROSSCHECK, "rectangular_sinusoidal_fd_dst_convergence",
    "round_gaussian_rectangular_fd_dst_pic_equivalence",
    "rectangular_fd_dst_space_charge_kick_equivalence",
    "elliptic_uniform_charge_density_fd_analytic",
    "elliptic_quartic_shortley_weller_fd_convergence",
    "all_supported_aperture_geometries_fd",
)
GROUPS = {
    "all": WORKFLOWS + CHECKS + ("analytic_free_space_tracking",),
    "analytic": ("analytic_free_space_tracking",),
    "workflows": WORKFLOWS,
    "checks": CHECKS,
    "fft": FFT + (
        CROSSCHECK + "::test_fft_free_space_round_gaussian_converges_at_second_order",
        CROSSCHECK + "::test_fft_cic_and_tsc_deposition_conserve_charge_and_match_theory",
    ),
    "rectangle": RECTANGLE + CHECKS[1:4] + (
        CROSSCHECK + "::test_fd_and_dst_rectangle_solve_the_same_discrete_poisson_problem",
    ),
    "aperture": APERTURE + CHECKS[4:],
    # These are existing local exploratory tests; keep their explicit discovery.
    "regression": (),
}
REGRESSION_PATHS = (
    "tests/unit/test_space_charge.py",
    "tests/codex/test_space_charge_analytic.py",
    "tests/codex/test_space_charge_command_geometry.py",
    "tests/codex/test_space_charge_local_aperture.py",
    "tests/codex/test_space_charge_stage1.py",
    "tests/codex/test_space_charge_configuration.py",
    "tests/codex/test_pic.py",
    "tests/codex/test_pic_cpu_field_validation.py",
    "tests/codex/test_space_charge_requested_fixes.py",
)


def pytest_targets(group: str, cases: list[str] | None = None) -> list[str]:
    if cases:
        selected = tuple(dict.fromkeys(cases))
    elif group == "regression":
        return [str(REPOSITORY / path) for path in REGRESSION_PATHS]
    else:
        selected = GROUPS[group]
    return [str(DIRECTORY / f"test_{name.partition('::')[0]}.py")
            + ("::" + name.partition("::")[2] if "::" in name else "")
            for name in selected]


def new_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPOSITORY / "tests" / "codex" / "space_charge_runs" / f"{stamp}_{uuid4().hex[:8]}"


def prepare_output_dir(path: Path | None, mode: str) -> Path:
    root = (path or new_output_dir()).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"Output path is not a directory: {root}")
    if mode == "ana":
        if not root.is_dir():
            raise ValueError(f"Analysis requires an existing output directory: {root}")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"Output directory is not empty: {root}. Choose a new directory; no files were deleted.")
        root.mkdir(parents=True, exist_ok=True)
    return root


class WorkflowRunner:
    """Run each requested workflow and its FFT reference once per batch."""

    def __init__(self, output_dir: Path, mode: str = "simana"):
        self.output_dir = output_dir
        self.mode = mode
        self.completed: set[str] = set()
        self.results: list[dict] = []

    def run(self, case: str) -> Path:
        if case not in WORKFLOWS:
            raise ValueError(f"Unknown workflow: {case}")
        run_dir = self.output_dir / case / "run_001"
        if case in self.completed:
            return run_dir
        if case in RECTANGLE:
            # Never silently compare a new FD/DST result with an unrelated old FFT.
            reference = case.removesuffix("_rectangular_fd_dst") + "_free_space_fft"
            self.run(reference)
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{case}_{self.mode}.log"
        started = time.perf_counter()
        print(f"[{self.mode}] {case}", flush=True)
        with log_file.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                [sys.executable, "-m", f"{PACKAGE}._suite", case, self.mode, str(self.output_dir)],
                cwd=REPOSITORY, stdout=stream, stderr=subprocess.STDOUT,
                encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        self.results.append({
            "case": case, "mode": self.mode, "exit_code": process.returncode,
            "seconds": time.perf_counter() - started,
            "log": str(log_file), "run_dir": str(run_dir),
        })
        (self.output_dir / f"workflow_results_{self.mode}.json").write_text(
            json.dumps(self.results, indent=2), encoding="utf-8",
        )
        if process.returncode:
            tail = "\n".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-35:])
            raise RuntimeError(f"{case} failed (exit {process.returncode}); see {log_file}\n{tail}")
        self.completed.add(case)
        return run_dir


def _worker(case: str, mode: str, output_dir: Path) -> None:
    module = importlib.import_module(f"{PACKAGE}.test_{case}")
    run_dir = output_dir / case / "run_001"
    if mode in {"sim", "simana"}:
        module.simulate(run_dir)
    if mode in {"ana", "simana"}:
        if case in RECTANGLE:
            common = importlib.import_module(f"{PACKAGE}._rectangular_fd_dst_simulation_common")
            reference = case.removesuffix("_rectangular_fd_dst") + "_free_space_fft"
            common.analyse(module.CASE, run_dir, fft_run_dir=output_dir / reference / "run_001")
        else:
            module.analyse(run_dir)


if __name__ == "__main__":
    _worker(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
