"""Aligned numeric result columns and reference metadata for the plot page."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from PASS.utils.constants import const
from PASS.utils.table_io import read_table


@dataclass
class ResultTable:
    columns: dict[str, np.ndarray | list[float]]
    metadata: dict
    _filter_key: tuple | None = field(default=None, init=False, repr=False)
    _row_mask: np.ndarray | slice = field(default_factory=lambda: slice(None), init=False, repr=False)

    def select_columns(self, keys, status="all", batch=None):
        """Filter only displayed columns and reuse the current particle mask."""
        key = (status, batch)
        if key != self._filter_key:
            self._row_mask = row_mask(self.columns, status, batch)
            self._filter_key = key
        return {name: np.asarray(self.columns[name])[self._row_mask] for name in keys if name in self.columns}


def read_result(path, *, arrays=False):
    path = Path(path)
    metadata = {}
    if path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py
        with h5py.File(path, "r") as stream:
            if any(isinstance(value, h5py.Dataset) and value.ndim > 1 for value in stream.values()):
                raise ValueError("该 HDF5 包含多维场数组；请使用专门的切片场分析，不能将网格轴当作粒子列配对。")
            definitions = {name: dict(value.attrs) for name, value in stream.items() if isinstance(value, h5py.Dataset) and value.attrs}
        frame = read_table(path)
        metadata = dict(frame.headers)
        if definitions:
            metadata["column_definitions"] = definitions
        columns = {key: frame[key].to_numpy() for key in frame if frame[key].dtype.kind in "biuf"}
        if not columns:
            raise ValueError("该 HDF5 没有一维粒子数据列；切片场等多维数组需要专门的场图。")
    else:
        import pandas as pd
        if path.suffix.lower() == ".csv":
            if Path(str(path) + ".metadata.json").exists():
                from PASS.tool.data_conversion import read_tables

                table = next(read_tables(path))
                frame = pd.DataFrame(table.columns)
                metadata = _conversion_metadata(table.metadata)
            else:
                frame = pd.read_csv(path, float_precision="round_trip")
        else:
            import tfs
            frame = tfs.read(path)
            metadata = dict(frame.headers)
            if "PASS_CONVERSION_METADATA" in metadata:
                from PASS.tool.data_conversion import read_tables

                metadata = _conversion_metadata(next(read_tables(path)).metadata)
        columns = {}
        for key in frame:
            numeric = pd.to_numeric(frame[key], errors="coerce")
            if frame[key].dtype.kind in "biuf" or numeric.notna().any():
                # Preserve row alignment even when one cell is non-numeric.
                columns[str(key)] = numeric.to_numpy()
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
            t = np.broadcast_to(np.asarray(time, dtype=float), (count, ))
            b = np.broadcast_to(np.asarray(beta, dtype=float), (count, ))
            z = np.asarray(columns[names["z"]], dtype=float)
            tag = np.asarray(columns[names["tag"]])
            valid = (tag > 0) & (b > 0) & (b < 1) & np.isfinite(t) & np.isfinite(b) & np.isfinite(z)
            arrived = np.full(count, np.nan)
            arrived[valid] = t[valid] - z[valid] / (b[valid] * const.c)
            columns["arrival_time_s"] = arrived
    if not arrays:
        columns = {key: value.tolist() for key, value in columns.items()}
    return ResultTable(columns, metadata)


def _conversion_metadata(metadata):
    result = dict(metadata.get("parameters", {}))
    for key in ("column_definitions", "plot_export"):
        if key in metadata:
            result[key] = metadata[key]
    return result


def row_mask(columns, status="all", batch=None):
    """Return a cheap full-column view when no particle filter is active."""
    if not columns:
        return slice(None)
    names = {key.casefold(): key for key in columns}
    has_status = "tag" in names and status in {"alive", "lost"}
    has_batch = batch is not None and "injection_batch" in names
    if not has_status and not has_batch:
        return slice(None)
    mask = np.ones(len(next(iter(columns.values()))), dtype=bool)
    if has_status:
        tags = np.asarray(columns[names["tag"]])
        if status == "alive":
            mask &= tags > 0
        elif status == "lost":
            mask &= tags < 0
    if has_batch:
        mask &= np.asarray(columns[names["injection_batch"]]) == batch
    return mask


def select_rows(columns, status="all", batch=None):
    """Compatibility adapter for callers expecting lists for every column."""
    mask = row_mask(columns, status, batch)
    return {key: np.asarray(value)[mask].tolist() for key, value in columns.items()}
