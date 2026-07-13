"""Generic data converter: external files → PASS TFS format.

Four-step pipeline, each step is an independent function:

    1. load_raw_data      — read external file (CSV/TXT/TFS/...)
    2. time_to_turn       — convert time axis → turn axis (if needed)
    3. interpolate_to_continuous_turns — fill gaps / ensure consecutive turns
    4. write_tfs_ramping  — write PASS unified TFS file

Users can call convert_external_to_tfs() for the full pipeline,
or call individual steps for custom workflows.

Output TFS format:
    @ TITLE     "..."
    @ TYPE      "..."
    @ NUM_TURNS N
    *
    TURN  TIME (S)  <data_col1>  <data_col2>  ...
    1     1.23e-6   ...
    ...

The engine reads ramping TFS files via tfs.read() and indexes by row
(row 0 = turn 1). TURN column is for human readability; TIME (S) is optional.
"""

import numpy as np
import tfs
from pathlib import Path
from typing import Callable
from scipy.interpolate import interp1d


# ============================================================
# Step 1: Read external data
# ============================================================

def load_raw_data(
    file_path: str,
    turn_col: str = "turn",
    time_col: str = "time (s)",
    data_cols: list[str] | None = None,
    delimiter: str = ",",
    skiprows: int = 0,
) -> dict:
    """Read raw data from an external file.

    Auto-detects format by extension:
        .tfs  → tfs.read()
        .csv  → np.genfromtxt (delimiter=',')
        .txt  → np.genfromtxt (delimiter=whitespace or given)

    Returns:
        {"turn": np.ndarray | None,
         "time": np.ndarray | None,
         "data": {col_name: np.ndarray}}

    Args:
        file_path: path to the external file.
        turn_col: expected column name for turn (case-insensitive).
        time_col: expected column name for time (case-insensitive).
        data_cols: explicit list of data column names to extract.
                   If None, all columns except turn/time are treated as data.
        delimiter: delimiter for CSV/TXT files (ignored for .tfs).
        skiprows: number of header rows to skip for CSV/TXT (ignored for .tfs).
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".tfs":
        df = tfs.read(str(path))
        df.columns = df.columns.str.lower().str.strip()
    else:
        # Generic CSV/TXT reader
        raw = np.genfromtxt(
            str(path),
            delimiter=delimiter if delimiter != "whitespace" else None,
            skip_header=skiprows,
            names=True,   # read header row as field names
            dtype=None,
            encoding="utf-8",
            deletechars="",
        )
        # Convert structured array to dict-like
        col_names = [n.strip().lower() for n in raw.dtype.names]
        data_dict = {}
        for orig_name, clean_name in zip(raw.dtype.names, col_names):
            data_dict[clean_name] = np.asarray(raw[orig_name], dtype=float)

        # Build a simple namespace that mimics df.columns / df[col]
        class _DF:
            columns = col_names

            def __getitem__(self, key):
                return data_dict[key.lower().strip()]

        df = _DF()

    result = {"turn": None, "time": None, "data": {}}

    # Detect turn column (flexible: "turn", "Turn", "TURN", "turn (n)", etc.)
    for col in df.columns:
        if col.startswith("turn"):
            result["turn"] = np.asarray(df[col], dtype=float)
            break

    # Detect time column (flexible: "time", "Time", "time (s)", etc.)
    if result["turn"] is None:
        for col in df.columns:
            if col.startswith("time"):
                result["time"] = np.asarray(df[col], dtype=float)
                break

    # Read data columns
    if data_cols is None:
        exclude = {turn_col.lower(), time_col.lower()}
        data_cols = [c for c in df.columns if c not in exclude]

    for col in data_cols:
        target = col.lower().strip()
        for df_col in df.columns:
            if df_col == target:
                result["data"][col] = np.asarray(df[df_col], dtype=float)
                break

    return result


# ============================================================
# Step 2: Time ↔ Turn conversion
# ============================================================

def time_to_turn(
    time_arr: np.ndarray,
    revolution_freq: float | Callable,
    num_turns: int | None = None,
) -> np.ndarray:
    """Convert time axis → turn axis.

    turn = time * f_rev  (rounded to nearest integer)

    Args:
        time_arr: array of time values in seconds.
        revolution_freq: revolution frequency in Hz. Can be a constant
                         or a callable(turn) -> Hz for variable frequency.
        num_turns: if given, clip turn values to [1, num_turns].

    Returns:
        Integer array of turn numbers.
    """
    time_arr = np.asarray(time_arr, dtype=float)

    if callable(revolution_freq):
        # Variable frequency: integrate dt = 1/f_rev(turn) cumulatively
        turn_arr = np.zeros(len(time_arr), dtype=float)
        t_accum = 0.0
        turn_accum = 0.0
        turn_arr[0] = 0.0
        for i in range(1, len(time_arr)):
            dt = time_arr[i] - time_arr[i - 1]
            f_avg = revolution_freq(turn_accum)
            turn_accum += dt * f_avg
            turn_arr[i] = turn_accum
        turn_arr = np.round(turn_arr).astype(int)
    else:
        turn_arr = np.round(time_arr * revolution_freq).astype(int)

    if num_turns is not None:
        turn_arr = np.clip(turn_arr, 1, num_turns)

    return turn_arr


def turn_to_time(
    turn_arr: np.ndarray,
    revolution_freq: float | Callable,
) -> np.ndarray:
    """Convert turn axis → time axis.

    time = turn / f_rev

    Args:
        turn_arr: array of turn numbers.
        revolution_freq: Hz (constant or callable(turn) -> Hz).

    Returns:
        Float array of time values in seconds.
    """
    turn_arr = np.asarray(turn_arr, dtype=float)

    if callable(revolution_freq):
        time_arr = np.zeros_like(turn_arr)
        t_accum = 0.0
        time_arr[0] = 0.0
        for i in range(1, len(turn_arr)):
            f = revolution_freq(turn_arr[i])
            t_accum += 1.0 / f
            time_arr[i] = t_accum
        return time_arr
    else:
        return turn_arr / revolution_freq


# ============================================================
# Step 3: Interpolate to continuous turns
# ============================================================

def interpolate_to_continuous_turns(
    turn_arr: np.ndarray,
    data_dict: dict[str, np.ndarray],
    start_turn: int = 1,
    end_turn: int | None = None,
    method: str = "linear",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Interpolate sparse/non-consecutive turn data to consecutive turns.

    Args:
        turn_arr: original turn numbers (may have gaps).
        data_dict: {col_name: data_array} aligned with turn_arr.
        start_turn: first turn (default 1).
        end_turn: last turn (default = max(turn_arr)).
        method: interpolation method: "linear", "cubic", "nearest".

    Returns:
        (continuous_turns, interpolated_data_dict)
    """
    turn_arr = np.asarray(turn_arr, dtype=float)

    if end_turn is None:
        end_turn = int(np.max(turn_arr))

    continuous_turns = np.arange(start_turn, end_turn + 1, dtype=int)

    # Remove duplicate s-points (same turn appearing multiple times)
    unique_mask = np.ones(len(turn_arr), dtype=bool)
    for i in range(1, len(turn_arr)):
        if abs(turn_arr[i] - turn_arr[i - 1]) < 1e-10:
            unique_mask[i] = False
    turn_arr = turn_arr[unique_mask]

    result = {}
    for col, values in data_dict.items():
        values = np.asarray(values, dtype=float)[unique_mask]
        if len(turn_arr) == 1:
            # Single data point: constant value
            result[col] = np.full(len(continuous_turns), values[0])
        else:
            interp = interp1d(
                turn_arr, values,
                kind=method,
                bounds_error=False,
                fill_value=(values[0], values[-1]),
            )
            result[col] = interp(continuous_turns)

    return continuous_turns, result


# ============================================================
# Step 4: Write TFS file
# ============================================================

def write_tfs_ramping(
    output_path: str,
    turn: np.ndarray,
    time: np.ndarray | None,
    data: dict[str, np.ndarray],
    title: str = "PASS Ramping Data",
    data_type: str = "RAMPING",
    headers: dict | None = None,
) -> str:
    """Write a PASS unified TFS ramping file.

    Args:
        output_path: output file path (.tfs).
        turn: integer array of turn numbers (1-based, consecutive).
        time: optional float array of time values in seconds.
        data: {col_name: float_array} of data columns.
        title: TFS header Title.
        data_type: TFS header Type.
        headers: additional TFS headers.

    Returns:
        The output path.
    """
    turn = np.asarray(turn, dtype=np.int64)
    columns = {"TURN": turn}

    if time is not None:
        columns["TIME_S"] = np.asarray(time, dtype=np.float64)

    for col, arr in data.items():
        columns[col.upper()] = np.asarray(arr, dtype=np.float64)

    df = tfs.TfsDataFrame(columns)
    df.headers["TITLE"] = title
    df.headers["TYPE"] = data_type
    df.headers["NUM_TURNS"] = len(turn)

    if headers:
        df.headers.update(headers)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tfs.write(output_path, df)
    return output_path


# ============================================================
# High-level API: full pipeline
# ============================================================

def convert_external_to_tfs(
    input_path: str,
    output_path: str,
    data_cols: list[str] | None = None,
    revolution_freq: float | Callable | None = None,
    num_turns: int | None = None,
    method: str = "linear",
    turn_col: str = "turn",
    time_col: str = "time (s)",
    delimiter: str = ",",
    skiprows: int = 0,
    title: str = "PASS Ramping Data",
    data_type: str = "RAMPING",
) -> str:
    """Convert an external data file to PASS unified TFS format.

    Pipeline:
        1. load_raw_data: read external file
        2. If only time column exists: time_to_turn conversion
        3. interpolate_to_continuous_turns: ensure consecutive turns
        4. turn_to_time: generate time column (if freq available)
        5. write_tfs_ramping: write output

    Args:
        input_path: path to the external data file.
        output_path: path for the output .tfs file.
        data_cols: explicit data column names. If None, auto-detect.
        revolution_freq: Hz (for time↔turn conversion). Constant or callable.
        num_turns: target number of turns. If None, use max(turn).
        method: interpolation method (linear/cubic/nearest).
        turn_col: expected turn column name.
        time_col: expected time column name.
        delimiter: for CSV/TXT input.
        skiprows: header rows to skip for CSV/TXT input.
        title: TFS Title header.
        data_type: TFS Type header.

    Returns:
        The output path.
    """
    # Step 1
    raw = load_raw_data(input_path, turn_col, time_col, data_cols, delimiter, skiprows)

    # Step 2: determine turn axis
    if raw["turn"] is not None:
        turn_arr = raw["turn"]
    elif raw["time"] is not None and revolution_freq is not None:
        turn_arr = time_to_turn(raw["time"], revolution_freq, num_turns)
    else:
        raise ValueError(
            "Cannot determine turn axis: need either a 'turn' column "
            "or a 'time' column + revolution_freq"
        )

    # Step 3: interpolate to continuous turns
    end_turn = num_turns if num_turns else int(np.max(turn_arr))
    turn_cont, data_cont = interpolate_to_continuous_turns(
        turn_arr, raw["data"],
        start_turn=1, end_turn=end_turn, method=method,
    )

    # Step 4: generate time column
    time_cont = None
    if revolution_freq is not None:
        time_cont = turn_to_time(turn_cont, revolution_freq)

    # Step 5: write TFS
    return write_tfs_ramping(
        output_path, turn_cont, time_cont, data_cont,
        title=title, data_type=data_type,
    )
