"""Bump tables use seconds and integrated Delta(Px,Py)/P0."""
from pathlib import Path
import numpy as np
import tfs


def read_bump_waveform(path):
    frame = tfs.read(path)
    values = frame[["TIME", "HKICK", "VKICK"]].to_numpy(dtype=np.float64)
    if len(values) < 2 or not np.all(np.isfinite(values)) or np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("Bump requires at least two finite rows with strictly increasing TIME")
    for key, expected in (("TIME_UNIT", "s"), ("KICK_CONVENTION", "delta_p_over_p0")):
        if key in frame.headers and frame.headers[key] != expected:
            raise ValueError(f"Bump {key} must be {expected}")
    return values


def convert_cisp_bump(horizontal, vertical, output):
    """Combine two CSV functions without shifting their laboratory clocks."""
    arrays = [np.loadtxt(path, delimiter=",", ndmin=2) for path in (horizontal, vertical)]
    for values in arrays:
        if (values.shape[1] != 2 or len(values) < 2 or not np.all(np.isfinite(values))
                or np.any(np.diff(values[:, 0]) <= 0)):
            raise ValueError("CISP kick CSV requires two finite columns and increasing time")
    lo, hi = max(a[0, 0] for a in arrays), min(a[-1, 0] for a in arrays)
    if hi <= lo:
        raise ValueError("Horizontal and vertical waveforms have no common interval")
    times = np.unique(np.concatenate([a[:, 0] for a in arrays] + [[lo, hi]]))
    times = times[(times >= lo) & (times <= hi)]
    headers = {"TIME_UNIT": "s", "KICK_CONVENTION": "delta_p_over_p0",
               "H_SOURCE": str(Path(horizontal).resolve()), "V_SOURCE": str(Path(vertical).resolve()),
               "COMMON_START": float(lo), "COMMON_END": float(hi)}
    frame = tfs.TfsDataFrame({"TIME": times, "HKICK": np.interp(times, *arrays[0].T),
                             "VKICK": np.interp(times, *arrays[1].T)}, headers=headers)
    tfs.write(output, frame)
    return frame
