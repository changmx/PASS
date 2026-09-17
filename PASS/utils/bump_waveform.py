"""Bump tables use seconds and integrated Delta(Px,Py)/P0."""
from pathlib import Path
import numpy as np
import pandas as pd
import tfs
from tfs.errors import AbsentColumnNameError, AbsentColumnTypeError
from tfs.reader import _read_metadata


def read_bump_waveform(path):
    """Return the shared interpolation table and each plane's supplied range.

    Converted tables hold each plane constant outside its original range.
    The range headers preserve that boundary for runtime warnings even when
    the other plane supplies earlier or later time nodes.
    """
    path = Path(path)
    # tfs.read does not expose pandas' float_precision option. Reuse its TFS
    # metadata parser (including compressed input), but parse the real columns
    # with round-trip precision so adjacent double time nodes stay distinct.
    try:
        metadata = _read_metadata(path)
    except UnboundLocalError as exc:
        # The TFS metadata parser leaves its line counter unset on an empty
        # stream, including an empty compressed file. Expose an input error.
        raise ValueError("Bump waveform file is empty") from exc
    if metadata.column_names is None:
        raise AbsentColumnNameError(path)
    if metadata.column_types is None:
        raise AbsentColumnTypeError(path)
    columns = ["TIME", "HKICK", "VKICK"]
    types = dict(zip(metadata.column_names, metadata.column_types, strict=True))
    for column in columns:
        if column not in types:
            raise KeyError(column)
        if (not np.issubdtype(types[column], np.number)
                or np.issubdtype(types[column], np.complexfloating)):
            raise ValueError(f"Bump {column} must contain real numbers")
    frame = pd.read_csv(path, sep=r"\s+", names=metadata.column_names,
                        usecols=columns, dtype=np.float64, engine="c",
                        skiprows=metadata.non_data_lines, float_precision="round_trip",
                        na_values=["nil"])
    values = frame[columns].to_numpy(dtype=np.float64)
    headers = metadata.headers
    if len(values) < 2 or not np.all(np.isfinite(values)) or np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("Bump requires at least two finite rows with strictly increasing TIME")
    for key, expected in (("TIME_UNIT", "s"), ("KICK_CONVENTION", "delta_p_over_p0")):
        if key in headers and headers[key] != expected:
            raise ValueError(f"Bump {key} must be {expected}")
    bounds = []
    for column, plane in ((1, "HKICK"), (2, "VKICK")):
        keys = (f"{plane}_START", f"{plane}_END")
        if (keys[0] in headers) != (keys[1] in headers):
            raise ValueError(f"Bump requires both {keys[0]} and {keys[1]}")
        endpoints = []
        for key, fallback in zip(keys, (values[0, 0], values[-1, 0])):
            limit = float(headers.get(key, fallback))
            if not np.isfinite(limit):
                raise ValueError(f"Bump {key} must be finite")
            index = int(np.searchsorted(values[:, 0], limit))
            if index == len(values) or values[index, 0] != limit:
                raise ValueError(f"Bump {key} must identify a TIME node")
            endpoints.append(limit)
        start, end = endpoints
        if start >= end:
            raise ValueError(f"Bump {plane} range must be strictly increasing")
        # Range metadata must describe endpoint padding, not hide extra data.
        padded = np.interp(np.clip(values[:, 0], start, end), values[:, 0], values[:, column])
        if not np.array_equal(values[:, column], padded):
            raise ValueError(f"Bump {plane} must hold its endpoint values outside its supplied range")
        bounds.append((start, end))
    return values, np.asarray(bounds, dtype=np.float64)


def convert_cisp_bump(horizontal, vertical, output):
    """Merge independent CSV time grids, holding each plane's endpoints.

    Row counts and time spans may differ or be disjoint. No samples are
    cropped and neither laboratory clock is shifted.
    """
    arrays = [np.loadtxt(path, delimiter=",", ndmin=2) for path in (horizontal, vertical)]
    for values in arrays:
        if (values.shape[1] != 2 or len(values) < 2 or not np.all(np.isfinite(values))
                or np.any(np.diff(values[:, 0]) <= 0)):
            raise ValueError("CISP kick CSV requires two finite columns and increasing time")
    times = np.unique(np.concatenate([a[:, 0] for a in arrays]))
    headers = {"TIME_UNIT": "s", "KICK_CONVENTION": "delta_p_over_p0",
               "H_SOURCE": str(Path(horizontal).resolve()), "V_SOURCE": str(Path(vertical).resolve()),
               "HKICK_START": float(arrays[0][0, 0]), "HKICK_END": float(arrays[0][-1, 0]),
               "VKICK_START": float(arrays[1][0, 0]), "VKICK_END": float(arrays[1][-1, 0])}
    frame = tfs.TfsDataFrame({"TIME": times, "HKICK": np.interp(times, *arrays[0].T),
                             "VKICK": np.interp(times, *arrays[1].T)}, headers=headers)
    tfs.write(output, frame, colwidth=25, headerswidth=25)
    return frame
