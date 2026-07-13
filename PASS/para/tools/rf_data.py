"""RF data converter: external RF data → PASS TFS format.

Engine consumption (rfcavity.py:_load_rf_table):
    tfs.read(filepath)
    Required columns: HARMONIC, VOLTAGE, PHASE, PHI_OFFSET
    One row per turn (row 0 = turn 1).
"""

from PASS.para.tools.data_converter import convert_external_to_tfs
from typing import Callable


def convert_rf_data(
    input_path: str,
    output_path: str,
    revolution_freq: float | Callable | None = None,
    num_turns: int | None = None,
    method: str = "linear",
) -> str:
    """Convert external RF data to PASS TFS format.

    The external file must contain columns: voltage, harmonic, phase, phi_offset
    (case-insensitive, any order).

    Args:
        input_path: external RF data file (CSV/TXT/TFS).
        output_path: output .tfs file path.
        revolution_freq: revolution frequency in Hz (for time→turn conversion).
        num_turns: target number of turns.
        method: interpolation method.

    Returns:
        Output file path.
    """
    return convert_external_to_tfs(
        input_path, output_path,
        data_cols=["voltage", "harmonic", "phase", "phi_offset"],
        revolution_freq=revolution_freq, num_turns=num_turns,
        method=method,
        title="RF Ramping Data", data_type="RF_RAMPING",
    )
