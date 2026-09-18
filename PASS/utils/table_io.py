"""Typed diagnostic tables shared by TFS and HDF5 producers and readers."""

from collections.abc import Mapping
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tfs


def normalize_output_format(value="hdf5-gzip1") -> str:
    """Validate the public table format without silently falling back."""
    if value not in ("tfs", "hdf5", "hdf5-gzip1"):
        raise ValueError("Output format must be 'tfs', 'hdf5' or 'hdf5-gzip1'")
    return value


def _hdf5_filters(output_format):
    output_format = normalize_output_format(output_format)
    if output_format == "tfs":
        raise ValueError("HDF5 tables require 'hdf5' or 'hdf5-gzip1'")
    if output_format == "hdf5":
        return {}
    return {"compression": "gzip", "compression_opts": 1, "shuffle": True}


def table_path(path, output_format="hdf5-gzip1") -> Path:
    return Path(path).with_suffix(".tfs" if normalize_output_format(output_format) == "tfs" else ".h5")


def find_table_files(directory, pattern="*", *, recursive=False) -> list[Path]:
    """Find table stems, preferring HDF5 when a legacy TFS copy also exists."""
    directory = Path(directory)
    found = {}
    for suffix in (".tfs", ".hdf5", ".h5"):
        paths = directory.rglob(pattern + suffix) if recursive else directory.glob(pattern + suffix)
        for path in paths:
            if path.is_file():
                found[path.with_suffix("")] = path
    return sorted(found.values())


def _table_columns(frame):
    if not isinstance(frame, Mapping) and not frame.columns.is_unique:
        raise ValueError("Table column names must be unique")
    columns = {}
    for name, column in frame.items():
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(f"Invalid HDF5 table column name: {name!r}")
        values = np.asarray(column)
        if values.ndim != 1:
            raise ValueError(f"HDF5 table column {name!r} must be one-dimensional")
        if values.dtype.kind not in "biufc":
            raise ValueError(f"HDF5 table column {name!r} must contain numeric or boolean values")
        columns[name] = values
    if not columns:
        raise ValueError("A diagnostic table must define at least one column")
    if len({len(values) for values in columns.values()}) != 1:
        raise ValueError("HDF5 table columns must have equal lengths")
    return columns


def _write_metadata(stream, headers, columns):
    for key, value in headers.items():
        if str(key).startswith("_pass_table_"):
            raise ValueError("Header names beginning with _pass_table_ are reserved")
        stream.attrs[key] = value
    stream.attrs["_pass_table_version"] = 1
    stream.attrs["_pass_table_columns"] = json.dumps(list(columns))


def write_table(path, frame, headers=None, *, colwidth=20, headerswidth=20, output_format=None) -> Path:
    """Write column arrays; infer TFS or default compressed HDF5 from the suffix when omitted."""
    path = Path(path)
    is_hdf5 = path.suffix.lower() in (".h5", ".hdf5")
    if output_format is None:
        output_format = "hdf5-gzip1" if is_hdf5 else "tfs"
    output_format = normalize_output_format(output_format)
    if is_hdf5 != (output_format != "tfs"):
        raise ValueError("Output format does not match the table file extension")
    headers = dict(getattr(frame, "headers", {}) if headers is None else headers)
    if output_format == "tfs":
        if isinstance(frame, Mapping):
            frame = pd.DataFrame(frame, copy=False)
        tfs.write(path, frame, headers_dict=headers, colwidth=colwidth, headerswidth=headerswidth)
        return path
    columns = _table_columns(frame)
    filters = _hdf5_filters(output_format)
    with h5py.File(path, "w") as stream:
        _write_metadata(stream, headers, columns)
        for name, values in columns.items():
            # About 64 KiB per column chunk, with valid chunks for empty tables.
            chunk_rows = max(1, min(len(values), 65536 // values.dtype.itemsize))
            stream.create_dataset(name, data=values, maxshape=(None, ), chunks=(chunk_rows, ), **filters)
    return path


def append_table(path, frame, headers, *, chunk_rows, output_format="hdf5-gzip1"):
    """Append one batch to a column-oriented HDF5 table and close the file."""
    columns = _table_columns(frame)
    n_rows = len(next(iter(columns.values())))
    if n_rows == 0:
        return
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, (int, np.integer)) or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    filters = _hdf5_filters(output_format)
    with h5py.File(path, "a") as stream:
        if not list(stream.keys()):
            _write_metadata(stream, headers, columns)
            for name, values in columns.items():
                stream.create_dataset(name, shape=(0, ), maxshape=(None, ), dtype=values.dtype, chunks=(int(chunk_rows), ), **filters)
        expected = json.loads(stream.attrs["_pass_table_columns"])
        if list(columns) != expected:
            raise ValueError("HDF5 append columns do not match the existing table")
        lengths = {len(stream[name]) for name in expected}
        if len(lengths) != 1:
            raise ValueError("HDF5 table columns have inconsistent lengths")
        for name, values in columns.items():
            if stream[name].dtype != values.dtype:
                raise ValueError(f"HDF5 append dtype changed for {name!r}")
        start = lengths.pop()
        end = start + n_rows
        for name, values in columns.items():
            stream[name].resize((end, ))
            stream[name][start:end] = values


def read_table(path):
    """Read a numeric table as a TfsDataFrame, including legacy HDF5 snapshots."""
    path = Path(path)
    if path.suffix.lower() not in (".h5", ".hdf5"):
        return tfs.read(path)
    with h5py.File(path, "r") as stream:
        names = json.loads(stream.attrs["_pass_table_columns"]) if "_pass_table_columns" in stream.attrs else list(stream.keys())
        columns = {}
        for name in names:
            dataset = stream[name]
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1:
                raise ValueError("HDF5 table columns must be one-dimensional; use a field reader for multidimensional SpaceCharge data")
            columns[name] = dataset[:]
        if not columns or len({len(values) for values in columns.values()}) != 1:
            raise ValueError("HDF5 file must contain nonempty column definitions of equal length")
        headers = {}
        for key, value in stream.attrs.items():
            if not key.startswith("_pass_table_"):
                headers[key] = value.decode("utf-8") if isinstance(value, bytes) else value.item() if isinstance(value, np.generic) else value
    return tfs.TfsDataFrame(pd.DataFrame(columns), headers=headers)
