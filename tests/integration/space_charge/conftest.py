"""Run all integration cases automatically, with isolated serial output."""
import importlib
from pathlib import Path

import pytest

from ._suite import PACKAGE, WorkflowRunner, prepare_output_dir


def pytest_addoption(parser):
    parser.addoption("--sc-output-dir", type=Path, default=None,
                     help="Empty output directory for this space-charge integration run.")


def pytest_configure(config):
    if config.getoption("numprocesses", default=0):
        raise pytest.UsageError("Space-charge generated-input workflows must run serially; omit -n.")


@pytest.fixture(scope="session")
def sc_output_dir(request):
    try:
        root = prepare_output_dir(request.config.getoption("--sc-output-dir"), "test")
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    request.config._space_charge_output_dir = root
    return root


@pytest.fixture(scope="session")
def sc_workflow(sc_output_dir):
    return WorkflowRunner(sc_output_dir).run


@pytest.fixture(autouse=True)
def sc_output_paths(monkeypatch, sc_output_dir):
    """Keep existing numerical checks and assertions; redirect only artifacts."""
    common = importlib.import_module(f"{PACKAGE}._fd_validation_common")
    monkeypatch.setattr(common, "OUTPUT_ROOT", sc_output_dir)
    sinusoidal = importlib.import_module(f"{PACKAGE}.test_rectangular_sinusoidal_fd_dst_convergence")
    monkeypatch.setattr(sinusoidal, "OUTPUT_ROOT", sc_output_dir)
    crosschecks = importlib.import_module(f"{PACKAGE}.test_deposition_and_field_solver_crosschecks")
    monkeypatch.setattr(crosschecks, "OUTPUT_DIR", sc_output_dir / "deposition_and_field_solver_crosschecks" / "analysis")
    apertures = importlib.import_module(f"{PACKAGE}.test_all_supported_aperture_geometries_fd")
    monkeypatch.setattr(apertures, "OUTPUT_DIR", sc_output_dir / "gaussian_and_kv_uniform_all_aperture_geometries_fd")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    root = getattr(config, "_space_charge_output_dir", None)
    if root is not None:
        terminalreporter.write_sep("-", f"Space-charge artifacts: {root}")
