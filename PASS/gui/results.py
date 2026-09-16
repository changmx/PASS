"""Aligned numeric result columns and reference metadata for the plot page."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PASS.utils.constants import const


@dataclass
class ResultTable:
    columns: dict[str, list[float]]
    metadata: dict


def read_result(path):
    path = Path(path)
    metadata = {}
    if path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py
        with h5py.File(path, "r") as stream:
            if any(isinstance(value, h5py.Dataset) and value.ndim > 1 for value in stream.values()):
                raise ValueError("该 HDF5 包含多维场数组；请使用专门的切片场分析，不能将网格轴当作粒子列配对。")
            metadata = {key: value.decode() if isinstance(value, bytes) else value.item() if isinstance(value, np.generic) else value
                        for key, value in stream.attrs.items()}
            columns = {key: np.asarray(value).tolist() for key, value in stream.items()
                       if isinstance(value, h5py.Dataset) and value.ndim == 1 and np.issubdtype(value.dtype, np.number)}
        if not columns:
            raise ValueError("该 HDF5 没有一维粒子数据列；切片场等多维数组需要专门的场图。")
    else:
        import pandas as pd
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            import tfs
            frame = tfs.read(path)
            metadata = dict(frame.headers)
        columns = {}
        for key in frame:
            numeric = pd.to_numeric(frame[key], errors="coerce")
            if numeric.notna().any():
                # Preserve row alignment even when one cell is non-numeric.
                columns[str(key)] = numeric.to_numpy().tolist()
    if len({len(v) for v in columns.values()}) > 1:
        raise ValueError("数据列长度不一致，不能配对绘图。")
    if not columns:
        raise ValueError("文件没有可绘制的数值列。")
    names = {key.casefold(): key for key in columns}
    refs = {str(key).casefold(): value for key, value in metadata.items()}
    if "z" in names and "tag" in names:
        count = len(columns[names["z"]])
        time = columns.get(names.get("referencetime"), refs.get("referencearrivaltime"))
        beta = columns.get(names.get("referencebeta"), refs.get("referencebeta"))
        if time is not None and beta is not None:
            t = np.broadcast_to(np.asarray(time, dtype=float), (count,))
            b = np.broadcast_to(np.asarray(beta, dtype=float), (count,))
            z = np.asarray(columns[names["z"]], dtype=float)
            tag = np.asarray(columns[names["tag"]])
            valid = (tag > 0) & (b > 0) & (b < 1) & np.isfinite(t) & np.isfinite(b) & np.isfinite(z)
            arrived = np.full(count, np.nan)
            arrived[valid] = t[valid] - z[valid] / (b[valid] * const.c)
            columns["arrival_time_s"] = arrived.tolist()
    return ResultTable(columns, metadata)


def select_rows(columns, status="all", batch=None):
    if not columns:
        return {}
    names = {key.casefold(): key for key in columns}
    mask = np.ones(len(next(iter(columns.values()))), dtype=bool)
    if "tag" in names:
        tags = np.asarray(columns[names["tag"]])
        if status == "alive":
            mask &= tags > 0
        elif status == "lost":
            mask &= tags < 0
    if batch is not None and "injection_batch" in names:
        mask &= np.asarray(columns[names["injection_batch"]]) == batch
    return {key: np.asarray(value)[mask].tolist() for key, value in columns.items()}
