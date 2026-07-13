"""Ramping file converters for specific element types.

Each function wraps data_converter.convert_external_to_tfs with
the correct data column names for that element type.
"""

from PASS.para.tools.data_converter import convert_external_to_tfs
from typing import Callable


def convert_k0l_ramping(
    input_path: str,
    output_path: str,
    revolution_freq: float | Callable | None = None,
    num_turns: int | None = None,
    method: str = "linear",
) -> str:
    """Dipole K0L ramping → TFS."""
    return convert_external_to_tfs(
        input_path, output_path,
        data_cols=["k0l"],
        revolution_freq=revolution_freq, num_turns=num_turns,
        method=method,
        title="Dipole K0L Ramping", data_type="RAMPING",
    )


def convert_k1l_ramping(
    input_path: str,
    output_path: str,
    revolution_freq: float | Callable | None = None,
    num_turns: int | None = None,
    method: str = "linear",
) -> str:
    """Quadrupole K1L/K1SL ramping → TFS."""
    return convert_external_to_tfs(
        input_path, output_path,
        data_cols=["k1l", "k1sl"],
        revolution_freq=revolution_freq, num_turns=num_turns,
        method=method,
        title="Quadrupole K1L Ramping", data_type="RAMPING",
    )


def convert_k2l_ramping(
    input_path: str,
    output_path: str,
    revolution_freq: float | Callable | None = None,
    num_turns: int | None = None,
    method: str = "linear",
) -> str:
    """Sextupole K2L/K2SL ramping → TFS."""
    return convert_external_to_tfs(
        input_path, output_path,
        data_cols=["k2l", "k2sl"],
        revolution_freq=revolution_freq, num_turns=num_turns,
        method=method,
        title="Sextupole K2L Ramping", data_type="RAMPING",
    )


def convert_k3l_ramping(
    input_path: str,
    output_path: str,
    revolution_freq: float | Callable | None = None,
    num_turns: int | None = None,
    method: str = "linear",
) -> str:
    """Octupole K3L/K3SL ramping → TFS."""
    return convert_external_to_tfs(
        input_path, output_path,
        data_cols=["k3l", "k3sl"],
        revolution_freq=revolution_freq, num_turns=num_turns,
        method=method,
        title="Octupole K3L Ramping", data_type="RAMPING",
    )


def convert_kick_ramping(
    input_path: str,
    output_path: str,
    revolution_freq: float | Callable | None = None,
    num_turns: int | None = None,
    method: str = "linear",
) -> str:
    """Kicker hkick/vkick ramping → TFS."""
    return convert_external_to_tfs(
        input_path, output_path,
        data_cols=["hkick", "vkick"],
        revolution_freq=revolution_freq, num_turns=num_turns,
        method=method,
        title="Kicker Ramping", data_type="RAMPING",
    )
