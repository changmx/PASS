"""Physical-time magnet program I/O and legacy element-specific converters.

The legacy element-specific functions retain their turn-resampling interfaces.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tfs
from tfs.reader import _read_metadata

from PASS.para.tools.data_converter import convert_external_to_tfs
from PASS.utils.magnet_program import _strength_component, _strength_unit, _validate_samples


def validate_magnet_ramping(times, columns, headers=None):
    """Return owned TIME and canonical strength arrays after format validation.

    Kn and KnS are per-length coefficients; KnL and KnSL are integrated.
    All values are normalized to the current bunch reference momentum.
    Validation does not read or write a file and does not require a magnet length.
    """
    return _validate_samples(times, columns, headers)


def read_magnet_ramping_source(input_path, *, delimiter=None, header=0, skiprows=0):
    """Read source columns with round-trip precision, retaining TFS metadata.

    ``header`` and ``skiprows`` apply to CSV/TXT sources only. TFS declares its
    own column names, types and metadata rows. No interpolation is performed.
    """
    path = Path(input_path)
    headers = {}
    if path.suffix.lower() == ".tfs":
        try:
            metadata = _read_metadata(path)
        except UnboundLocalError as exc:
            raise ValueError("Source TFS is empty") from exc
        if metadata.column_names is None or metadata.column_types is None:
            raise ValueError("Source TFS requires column names and types")
        names = [str(name).casefold() for name in metadata.column_names]
        if len(set(names)) != len(names):
            raise ValueError("Source column names must be unique, ignoring case")
        dtypes = {
            name: dtype if np.dtype(dtype).kind in "iuf" else str
            for name, dtype in zip(metadata.column_names, metadata.column_types, strict=True)
        }
        frame = pd.read_csv(path,
                            sep=r"\s+",
                            names=metadata.column_names,
                            dtype=dtypes,
                            skiprows=metadata.non_data_lines,
                            float_precision="round_trip",
                            na_values=["nil"])
        headers = metadata.headers
    else:
        separator = delimiter if delimiter is not None else "," if path.suffix.lower() == ".csv" else r"\s+"
        frame = pd.read_csv(path, sep=separator, header=header, skiprows=skiprows, comment="#", float_precision="round_trip")
    return tfs.TfsDataFrame(frame, headers=headers)


def write_magnet_ramping(output_path, times, columns):
    """Write absolute normalized strengths against physical seconds as TFS."""
    times, columns = validate_magnet_ramping(times, columns)
    path = Path(output_path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        raise ValueError("Magnet ramping output must use TFS")
    headers = {"TIME_UNIT": "s", "STRENGTH_CONVENTION": "normalized"}
    for name in columns:
        order, _, integrated = _strength_component(name)
        headers[f"{name}_UNIT"] = _strength_unit(order, integrated)
    frame = tfs.TfsDataFrame({"TIME": times, **columns}, headers=headers)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 18 significant digits retain distinct adjacent float64 time knots.
    tfs.write(path, frame, colwidth=25, headerswidth=25)
    return str(path)


def convert_magnet_ramping(input_path, output_path, *, time_column="TIME", column_mapping=None, time_scale=1.0, delimiter=None):
    """Convert named CSV/TXT/TFS columns without resampling or changing the clock.

    ``column_mapping`` maps source names to Kn/KnS/KnL/KnSL output names.
    When omitted, source strength names must already follow this convention.
    ``time_scale`` explicitly converts the source time unit to seconds.
    """
    if isinstance(time_scale, (bool, np.bool_)) or not np.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("Magnet ramping time_scale must be finite and positive")
    path = Path(input_path)
    output = Path(output_path)
    if path.resolve() == output.resolve() or (output.exists() and path.samefile(output)):
        raise ValueError("Choose a distinct output path to preserve the source magnet table")
    frame = read_magnet_ramping_source(path, delimiter=delimiter)
    headers = {str(key).upper(): value for key, value in frame.headers.items()}
    if "TIME_UNIT" in headers:
        expected_scale = {"s": 1., "ms": 1e-3, "us": 1e-6, "ns": 1e-9}.get(headers["TIME_UNIT"])
        if expected_scale is None or time_scale != expected_scale:
            raise ValueError("Source TIME_UNIT and time_scale must agree on conversion to seconds")
    names = {str(name).casefold(): name for name in frame.columns}
    if len(names) != len(frame.columns):
        raise ValueError("Source column names must be unique, ignoring case")

    def find_column(name):
        found = names.get(str(name).casefold())
        if found is None:
            raise ValueError(f"Source column {name!r} is missing; available columns: {list(frame.columns)}")
        return found

    selected_time = find_column(time_column)
    if column_mapping is None:
        column_mapping = {name: name for name in frame if name != selected_time and str(name).upper() != "TURN"}
    output = {}
    output_headers = {key: value for key, value in headers.items() if key == "STRENGTH_CONVENTION"}
    for source, target in column_mapping.items():
        target = str(target).upper()
        if target in output:
            raise ValueError(f"Duplicate target strength column {target}")
        output[target] = frame[find_column(source)].to_numpy()
        source_unit = f"{str(source).upper()}_UNIT"
        if source_unit in headers:
            output_headers[f"{target}_UNIT"] = headers[source_unit]
    times = frame[selected_time].to_numpy()
    # Validate real types before scaling, so strings and complex data cannot coerce.
    times, output = _validate_samples(times, output, output_headers)
    return write_magnet_ramping(output_path, times * time_scale, output)


def convert_k0l_ramping(
    input_path: str,
    output_path: str,
    revolution_freq: float | Callable | None = None,
    num_turns: int | None = None,
    method: str = "linear",
) -> str:
    """Dipole K0L ramping → TFS."""
    return convert_external_to_tfs(
        input_path,
        output_path,
        data_cols=["k0l"],
        revolution_freq=revolution_freq,
        num_turns=num_turns,
        method=method,
        title="Dipole K0L Ramping",
        data_type="RAMPING",
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
        input_path,
        output_path,
        data_cols=["k1l", "k1sl"],
        revolution_freq=revolution_freq,
        num_turns=num_turns,
        method=method,
        title="Quadrupole K1L Ramping",
        data_type="RAMPING",
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
        input_path,
        output_path,
        data_cols=["k2l", "k2sl"],
        revolution_freq=revolution_freq,
        num_turns=num_turns,
        method=method,
        title="Sextupole K2L Ramping",
        data_type="RAMPING",
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
        input_path,
        output_path,
        data_cols=["k3l", "k3sl"],
        revolution_freq=revolution_freq,
        num_turns=num_turns,
        method=method,
        title="Octupole K3L Ramping",
        data_type="RAMPING",
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
        input_path,
        output_path,
        data_cols=["hkick", "vkick"],
        revolution_freq=revolution_freq,
        num_turns=num_turns,
        method=method,
        title="Kicker Ramping",
        data_type="RAMPING",
    )
