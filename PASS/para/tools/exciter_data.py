"""Exciter data converter: external exciter frequency/voltage data → PASS TFS.

The PASS exciter currently uses analytical formulas (FM/AM modes) rather
than reading external data files at runtime. This converter is provided
for future use when exciter data needs to be pre-processed from external
measurements (e.g. LLRF frequency logs, BD42TUNE voltage traces).
"""

from PASS.para.tools.data_converter import convert_external_to_tfs
from typing import Callable


def convert_exciter_data(
    input_path: str,
    output_path: str,
    revolution_freq: float | Callable | None = None,
    num_turns: int | None = None,
    method: str = "linear",
) -> str:
    """Convert external exciter data to PASS TFS format.

    Expected columns: frequency, voltage
    (case-insensitive, any order).

    Args:
        input_path: external exciter data file.
        output_path: output .tfs file path.
        revolution_freq: revolution frequency in Hz.
        num_turns: target number of turns.
        method: interpolation method.

    Returns:
        Output file path.
    """
    return convert_external_to_tfs(
        input_path, output_path,
        data_cols=["frequency", "voltage"],
        revolution_freq=revolution_freq, num_turns=num_turns,
        method=method,
        title="Exciter Data", data_type="EXCITER",
    )
