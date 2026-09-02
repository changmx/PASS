from __future__ import annotations

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("slicer integration")
    group.addoption(
        "--backend",
        choices=("cpu", "gpu"),
        default="cpu",
        help="backend used by the Slicer integration cases (default: cpu)",
    )


@pytest.fixture
def backend(request) -> str:
    return request.config.getoption("--backend")
