"""Read-only selection and explicit table conversion, independent of Qt.

Python indices (turns, rows and axes) are zero-based; stops are exclusive.
No coordinate, unit, interpolation or reference-particle transformation occurs.
"""
from __future__ import annotations

import base64
import csv
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import shlex
import tempfile
from typing import Callable

import numpy as np


@dataclass
class DataSelection:
    """Explicit source selection, also serializable with ``dataclasses.asdict``.

    ``datasets`` are absolute HDF5 paths. ``indices`` fixes axes by zero-based
    integer, leaving None axes free. ``axes`` maps an axis number to its 1-D
    coordinate dataset. ``mode`` is columns, matrix or long.
    ``bpms``, ``bunch_ids`` and ``turns`` select OMC3 SDDS samples.
    ``tbt_columns`` maps BPM/BUNCH/TURN/X/Y to CSV/TFS source columns.
    ``filters`` contains (column, comparison, value) triples combined with AND.
    None selects all available columns/parameters; an empty list selects none.
    """
    datasets: list[str] = field(default_factory=list)
    columns: list[str] | None = None
    parameters: list[str] | None = None
    bpms: list[str] | None = None
    bunch_ids: list[int] | None = None
    turns: tuple[int, int | None] = (0, None)
    tbt_columns: dict[str, str] = field(default_factory=dict)
    rows: tuple[int, int | None, int] = (0, None, 1)
    mode: str = "columns"
    indices: list[int | None] = field(default_factory=list)
    axes: dict[str, str] = field(default_factory=dict)
    filters: list[tuple[str, str, object]] = field(default_factory=list)
    column_types: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in {"columns", "matrix", "long"}:
            raise ValueError("Unknown selection mode")
        start, stop, step = self.rows
        if any(type(v) is not int for v in (start, step)) or start < 0 or step < 1:
            raise ValueError("Rows require nonnegative start and positive integer step")
        if stop is not None and (type(stop) is not int or stop < start):
            raise ValueError("Row stop must be an integer >= start")
        start, stop = self.turns
        if type(start) is not int or start < 0 or (stop is not None and (type(stop) is not int or stop <= start)):
            raise ValueError("Turns require 0 <= start < stop, with an exclusive stop")
        if self.bunch_ids is not None and any(type(value) is not int or value < 0 for value in self.bunch_ids):
            raise ValueError("Bunch identifiers must be nonnegative integers")


@dataclass
class DataTable:
    columns: dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)
    label: str = "table"
    notices: list[str] = field(default_factory=list)

    @property
    def row_count(self):
        return len(next(iter(self.columns.values()))) if self.columns else 0


@dataclass
class ConversionReport:
    outputs: list[str]
    row_counts: list[int]
    notices: list[str]


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    result = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if path.suffix.lower() == ".csv":
        metadata = Path(str(path) + ".metadata.json")
        stat = metadata.stat() if metadata.exists() else None
        result["csv_metadata"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} if stat else None
    return result


def _digest_file(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_format(path):
    """Detect container signatures before considering table suffixes."""
    path = Path(path)
    with path.open("rb") as stream:
        head = stream.read(512)
    if head.startswith(b"SDDS"):
        return "sdds"
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return "hdf5"
    # HDF5 permits a user block before the superblock.
    if path.suffix.lower() in {".h5", ".hdf5", ".hdf"}:
        import h5py
        if h5py.is_hdf5(path):
            return "hdf5"
        raise ValueError("Not a valid HDF5 file")
    if path.suffix.lower() == ".csv":
        return "csv"
    if path.suffix.lower() == ".tfs":
        return "tfs"
    raise ValueError("Supported inputs: SDDS, HDF5, TFS, CSV")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        native = value.item()
        if isinstance(native, np.generic):
            raise ValueError(f"Unsupported extended scalar type: {value.dtype}")
        return _json_value(native)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported metadata type: {type(value).__name__}")


def _restore_json(value):
    if isinstance(value, dict):
        if set(value) == {"nonfinite_float"}:
            return float(value["nonfinite_float"])
        return {k: _restore_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_json(v) for v in value]
    return value


def _read_flat(path, column_types=None):
    import pandas as pd
    metadata = {}
    notices = []
    if detect_format(path) == "tfs":
        import tfs
        frame = tfs.read(path)
        # Reparse tokens with Python/NumPy conversion: pandas' default float
        # parser can move a correctly written 17-digit value by one ULP.
        tokens, names, types = [], None, None
        with Path(path).open(encoding="utf-8-sig") as stream:
            for line in stream:
                text = line.strip()
                if not text or text.startswith(("@", "#", "!")):
                    continue
                if text.startswith("*"):
                    names = shlex.split(text[1:])
                elif text.startswith("$"):
                    types = shlex.split(text[1:])
                elif names is not None and types is not None:
                    values = shlex.split(text)
                    if len(values) != len(names):
                        raise ValueError("TFS row length does not match its column definitions")
                    tokens.append(values)
        if names is None or types is None or len(names) != len(types):
            raise ValueError("Missing TFS column/type definitions")
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("TFS requires unique nonempty column names")
        exact = {}
        for index, (name, dtype) in enumerate(zip(names, types)):
            values = [row[index] for row in tokens]
            if re.fullmatch(r"%\d*s", dtype):
                exact[name] = np.array(values, dtype=object)
            elif dtype in {"%b"}:
                if any(v.lower() not in {"true", "false", "0", "1"} for v in values):
                    raise ValueError(f"Invalid TFS boolean column: {name}")
                exact[name] = np.array([v.lower() in {"true", "1"} for v in values], dtype=bool)
            elif dtype in {"%d", "%hd", "%ld"}:
                exact[name] = np.array(values, dtype=np.int64)
            elif dtype in {"%le", "%f", "%e", "%lf"}:
                exact[name] = np.array(values, dtype=np.float64)
            else:
                raise ValueError(f"Unsupported TFS type {dtype}: {name}")
        headers = frame.headers
        frame = tfs.TfsDataFrame(exact, headers=headers)
        metadata = {"parameters": dict(frame.headers)}
        packed = metadata["parameters"].pop("PASS_CONVERSION_METADATA", None)
        if packed:
            metadata.update(_restore_json(json.loads(base64.b64decode(packed).decode("utf-8"))))
    else:
        sidecar = Path(str(path) + ".metadata.json")
        dtypes = {}
        if sidecar.exists():
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            if saved.get("format") != "pass-table-metadata-v1":
                raise ValueError("Unsupported CSV metadata sidecar")
            if saved.get("sha256") != _digest_file(path):
                raise ValueError("CSV changed since its metadata was written; use a matching sidecar or explicitly rename the stale sidecar")
            metadata = _restore_json(saved["metadata"])
            dtypes = {k: (str if v in {"string", "object"} or v.startswith(("<U", "|S")) else v) for k, v in saved["dtypes"].items()}
        else:
            notices.append(
                "CSV types are inferred; units, acquisition information and descriptions require metadata. Override column_types if necessary.")
        # Parse explicit mappings as text first, so pandas cannot narrow or round
        # integers before the checked conversion in _select_table.
        dtypes.update({k: str for k in (column_types or {})})
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if not header or len(set(header)) != len(header) or any(not n for n in header):
            raise ValueError("CSV requires nonempty unique column names")
        if "BPM" in header:
            dtypes.setdefault("BPM", str)
        missing = {key: ["nan", "NaN"] for key, dtype in dtypes.items() if dtype is not str and np.dtype(dtype).kind == "f"}
        frame = pd.read_csv(path, dtype=dtypes or None, keep_default_na=False, na_values=missing, float_precision="round_trip")
    return DataTable({str(k): frame[k].to_numpy() for k in frame.columns}, metadata, "table", notices)


def inspect_file(path):
    """Describe data without constructing a GUI or altering the source."""
    kind = detect_format(path)
    if kind == "hdf5":
        from PASS.tool.hdf5_io import inspect_hdf5
        result = inspect_hdf5(path)
    elif kind == "sdds":
        from PASS.tool.sdds_io import inspect_sdds
        result = inspect_sdds(path)
    else:
        table = _read_flat(path)
        result = {
            "format": kind,
            "fields": [{
                "name": k,
                "dtype": str(v.dtype),
                "shape": list(v.shape),
                "kind": "column"
            } for k, v in table.columns.items()],
            "metadata": _json_value(table.metadata),
            "notices": table.notices
        }
    result["signature"] = file_signature(path)
    return result


def _select_table(table, selection, limit=None):
    columns = table.columns
    if len({len(v) for v in columns.values()}) > 1:
        raise ValueError("Selected columns have different lengths; export separate tables")
    start, stop, step = selection.rows
    columns = {k: np.asarray(v)[start:stop:step] for k, v in columns.items()}
    for name, dtype in selection.column_types.items():
        if name not in columns:
            raise ValueError(f"Unknown column: {name}")
        columns[name] = _cast_column(columns[name], dtype)
    for name, operation, value in selection.filters:
        if name not in columns:
            raise ValueError(f"Unknown filter column: {name}")
        array = columns[name]
        if array.dtype.kind in "iuf":
            value = int(value) if array.dtype.kind in "iu" else float(value)
        comparisons = {"==": np.equal, "!=": np.not_equal, "<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}
        if operation not in comparisons:
            raise ValueError(f"Unsupported comparison: {operation}")
        mask = comparisons[operation](array, value)
        columns = {k: v[mask] for k, v in columns.items()}
    names = list(columns) if selection.columns is None else selection.columns
    if not names or len(set(names)) != len(names):
        raise ValueError("Select at least one unique output column")
    if any(k not in columns for k in names):
        raise ValueError("Selected columns are missing from the source")
    selected = {k: columns[k][:limit] for k in names}
    metadata = dict(table.metadata)
    params = metadata.get("parameters", {})
    if selection.parameters is not None:
        if any(k not in params for k in selection.parameters):
            raise ValueError("Selected parameters are missing from the source")
        metadata["parameters"] = {k: params[k] for k in selection.parameters}
    return DataTable(selected, metadata, table.label, table.notices)


def _cast_column(values, dtype):
    if dtype == "string":
        return values.astype(str)
    target = np.dtype(dtype)
    if target.kind not in "iufb":
        raise ValueError("Column types must be string, integer, float or bool")
    if target.kind in "iu":
        converted = []
        bounds = np.iinfo(target)
        for value in values:
            if isinstance(value, (float, np.floating)) and (not np.isfinite(value) or value != np.trunc(value)):
                raise ValueError("Integer conversion would discard a fractional or nonfinite value")
            integer = int(value)
            if not bounds.min <= integer <= bounds.max:
                raise ValueError("Integer conversion would overflow the requested type")
            converted.append(integer)
        return np.asarray(converted, dtype=target)
    if target.kind == "b":
        if not all(str(v).lower() in {"true", "false", "0", "1"} for v in values):
            raise ValueError("Boolean conversion accepts only true/false/0/1")
        return np.array([str(v).lower() in {"true", "1"} for v in values], dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        result = values.astype(target)
    if values.dtype.kind in "iuf" and np.any(np.isfinite(values) & ~np.isfinite(result)):
        raise ValueError("Floating conversion would overflow the requested type")
    if values.dtype.kind in "iu" and any(int(a) != int(b) for a, b in zip(values, result)):
        raise ValueError("Floating conversion would lose integer precision")
    if values.dtype.kind in "OUS":
        for original, converted in zip(values, result):
            if re.fullmatch(r"[+-]?\d+", str(original).strip()) and (not np.isfinite(converted) or int(original) != int(converted)):
                raise ValueError("Floating conversion would lose integer precision")
    return result


def read_tables(path, selection=None, *, preview_limit=None):
    """Yield selected tables. HDF5 preview uses bounded dataset reads."""
    selection = selection or DataSelection()
    kind = detect_format(path)
    if kind == "hdf5":
        from PASS.tool.hdf5_io import read_hdf5_tables
        yield from read_hdf5_tables(path, selection, preview_limit=preview_limit)
    elif kind == "sdds":
        from PASS.tool.sdds_io import read_sdds_tables
        yield from read_sdds_tables(path, selection, preview_limit=preview_limit)
    else:
        if selection.mode != "columns":
            raise ValueError("CSV/TFS inputs support columns mode")
        read_types = dict(selection.column_types)
        mapped_types = {}
        if selection.tbt_columns:
            for name, dtype in (("BPM", "string"), ("BUNCH", "int64"), ("TURN", "int64")):
                column = selection.tbt_columns.get(name, name)
                if column:
                    read_types[column] = dtype
                    mapped_types[column] = dtype
        table = _read_flat(path, read_types)
        for column, dtype in mapped_types.items():
            if column in table.columns and column not in selection.column_types:
                try:
                    table.columns[column] = _cast_column(table.columns[column], dtype)
                except (ValueError, TypeError, OverflowError):
                    # Invalid identifiers remain visible for the preflight report.
                    # A numeric filter must never silently become a text comparison.
                    if any(name == column for name, _, _ in selection.filters):
                        raise ValueError(f"Cannot apply an OMC3 identifier filter before conversion to {dtype}: {column}") from None
        yield _select_table(table, selection, preview_limit)


def preview_file(path, selection=None, *, limit=500, check_sdds=False):
    """Preview selected rows, optionally checking the full OMC3 return path.

    ``check_sdds`` scans the entire SDDS selection in bounded blocks, or checks
    the CSV/TFS table already loaded in memory. The display remains limited.
    """
    if not 1 <= limit <= 5000:
        raise ValueError("Preview limit must be between 1 and 5000")
    before = file_signature(path)
    selection = selection or DataSelection()
    kind = detect_format(path)
    validation = None
    if check_sdds and kind == "sdds":
        from PASS.tool.sdds_io import preview_sdds_selection
        table, validation = preview_sdds_selection(path, selection, limit)
        source_tables = [table]
    else:
        source_tables = read_tables(path, selection, preview_limit=None if check_sdds and kind in {"csv", "tfs"} else limit)
    tables = []
    for table in source_tables:
        if check_sdds and kind in {"csv", "tfs"}:
            from PASS.tool.sdds_io import check_sdds_table
            validation = check_sdds_table(table, selection.tbt_columns)
        # Strings keep wide integers and nonfinite numbers intact across JSON/UI.
        rows = [[str(v.item() if isinstance(v, np.generic) else v) for v in row]
                for row in zip(*(values[:limit] for values in table.columns.values()))]
        tables.append({
            "label": table.label,
            "columns": list(table.columns),
            "rows": rows,
            "dtypes": {
                k: str(v.dtype)
                for k, v in table.columns.items()
            },
            "metadata": _json_value(table.metadata),
            "notices": table.notices
        })
        break
    if before != file_signature(path):
        raise ValueError("Source changed during preview; reopen it")
    return {"tables": tables, "limit": limit, "signature": before, "sdds_check": validation}


def _write_table(table, destination, kind, *, csv_metadata=True, sdds_mode="binary", sdds_columns=None):
    import pandas as pd
    columns = table.columns
    frame = pd.DataFrame(columns)
    extra = []
    if kind == "csv":
        frame.to_csv(destination, index=False, float_format="%.17g", na_rep="nan")
        if csv_metadata:
            extra_path = Path(str(destination) + ".metadata.json")
            extra_path.write_text(json.dumps(
                {
                    "format": "pass-table-metadata-v1",
                    "sha256": _digest_file(destination),
                    "metadata": _json_value(table.metadata),
                    "dtypes": {
                        k: str(v.dtype)
                        for k, v in columns.items()
                    }
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False),
                                  encoding="utf-8")
            extra.append(extra_path)
    elif kind == "tfs":
        import tfs
        for name, values in columns.items():
            if any(c.isspace() for c in name) or any(c in name for c in '\\"'):
                raise ValueError(f"TFS column name contains unsupported whitespace/quotes: {name!r}")
            if values.dtype.kind == "u" and values.size and values.max() > np.iinfo(np.int64).max:
                raise ValueError(f"TFS integer column exceeds int64: {name}; use CSV/HDF5")
            if values.dtype.kind in "OUS" and any(any(c in str(v) for c in '\\"\r\n') for v in values):
                raise ValueError(f"TFS text column {name} contains quotes, backslashes or newlines; use CSV/HDF5")
        headers = {k: v for k, v in table.metadata.get("parameters", {}).items() if isinstance(v, (str, int, float, np.number, bool))}
        headers = {
            k: v
            for k, v in headers.items()
            if not any(c.isspace() for c in k) and not any(c in k for c in '\\"') and not (isinstance(v, str) and any(c in v for c in '\\"\r\n'))
        }
        # TFS quoted strings cannot portably carry JSON's nested double quotes.
        headers["PASS_CONVERSION_METADATA"] = base64.b64encode(
            json.dumps(_json_value(table.metadata), ensure_ascii=False, allow_nan=False).encode("utf-8")).decode("ascii")
        tfs.write(destination, tfs.TfsDataFrame(frame, headers=headers), colwidth=25, headerswidth=25)
    elif kind == "hdf5":
        import h5py
        with h5py.File(destination, "w") as stream:
            group = stream.create_group("table")
            group.attrs["PASS_CONVERSION_METADATA"] = json.dumps(_json_value(table.metadata), ensure_ascii=False, allow_nan=False)
            for name, values in columns.items():
                if not name or "/" in name or name in {".", ".."}:
                    raise ValueError(f"Column name cannot be an HDF5 child name: {name!r}")
                if values.dtype.kind in "OUS":
                    if not all(isinstance(v, (str, bytes)) for v in values):
                        raise ValueError(f"Column {name}: mixed/non-string objects cannot be saved as strings")
                    values = np.array([v.decode("utf-8") if isinstance(v, bytes) else v for v in values], dtype=object)
                    group.create_dataset(name, data=values, dtype=h5py.string_dtype("utf-8"))
                else:
                    group.create_dataset(name, data=values, compression="gzip", compression_opts=1)
                definition = table.metadata.get("column_definitions", {}).get(name, {})
                for key in ("units", "description", "symbol"):
                    if definition.get(key):
                        group[name].attrs[key] = definition[key]
    else:
        from PASS.tool.sdds_io import write_sdds_table
        write_sdds_table(table, destination, mode=sdds_mode, columns_map=sdds_columns)
    return extra


def _write_container_chunks(source, destination, selection, kind, csv_metadata, progress):
    """Stream selected HDF5 or OMC3 SDDS samples to one flat table."""
    import pandas as pd
    if detect_format(source) == "hdf5":
        from PASS.tool.hdf5_io import iter_hdf5_chunks as iter_chunks
    else:
        from PASS.tool.sdds_io import iter_sdds_chunks as iter_chunks
    first = None
    total = 0
    notices = []
    for table in iter_chunks(source, selection):
        if first is None:
            first = table
            _write_table(table, destination, kind, csv_metadata=False)
        elif kind == "csv":
            pd.DataFrame(table.columns).to_csv(destination, mode="a", header=False, index=False, float_format="%.17g", na_rep="nan")
        else:
            chunk_path = destination.with_name(destination.name + ".chunk")
            _write_table(table, chunk_path, kind)
            with chunk_path.open(encoding="utf-8") as input_stream, destination.open("a", encoding="utf-8", newline="") as output_stream:
                for line in input_stream:
                    if not line.lstrip().startswith(("@", "*", "$")):
                        output_stream.write(line)
            chunk_path.unlink()
        total += table.row_count
        notices.extend(table.notices)
        if progress:
            progress(1, "table", total)
    if first is None:
        raise ValueError("No data selected")
    extra = []
    if kind == "csv" and csv_metadata:
        sidecar = Path(str(destination) + ".metadata.json")
        sidecar.write_text(json.dumps(
            {
                "format": "pass-table-metadata-v1",
                "sha256": _digest_file(destination),
                "metadata": _json_value(first.metadata),
                "dtypes": {
                    k: str(v.dtype)
                    for k, v in first.columns.items()
                }
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False),
                           encoding="utf-8")
        extra.append(sidecar)
    return total, extra, notices


def convert_file(source,
                 destination,
                 selection=None,
                 *,
                 output_format=None,
                 overwrite=False,
                 csv_metadata=True,
                 sdds_mode="binary",
                 expected_signature=None,
                 progress: Callable | None = None,
                 staging_directory=None):
    """Convert selected tables, staging complete files before publishing them.

    OMC3 SDDS arrays become a BPM/BUNCH/TURN/X/Y table.
    ``overwrite=False`` is the script default. Sources can never be overwritten.
    A CSV sidecar is optional; disabling it explicitly discards its metadata.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    kind = (output_format or destination.suffix.lstrip(".")).lower()
    kind = "hdf5" if kind in {"h5", "hdf"} else kind
    source_kind = detect_format(source)
    if kind not in {"csv", "tfs", "hdf5", "sdds"}:
        raise ValueError("Choose CSV, TFS, HDF5 or SDDS output")
    if source_kind in {"sdds", "hdf5"} and kind in {"sdds", "hdf5"}:
        raise ValueError("SDDS/HDF5 inputs export to CSV or TFS; direct container conversion is not supported")
    if source == destination:
        raise ValueError("The source must not be overwritten")
    before = file_signature(source)
    if expected_signature is not None and before != expected_signature:
        raise ValueError("Source changed since preview; reopen and preview it again")
    selection = selection or DataSelection()
    destination.parent.mkdir(parents=True, exist_ok=True)
    notices, counts, staged = [], [], []
    with tempfile.TemporaryDirectory(prefix="pass-conversion-", dir=staging_directory or destination.parent) as temporary:
        if source_kind in {"hdf5", "sdds"}:
            target = destination if destination.suffix else destination.with_suffix("." + kind)
            path = Path(temporary) / target.name
            count, extra, messages = _write_container_chunks(source, path, selection, kind, csv_metadata, progress)
            staged.append((path, target))
            staged.extend((p, target.with_name(p.name)) for p in extra)
            counts.append(count)
            notices.extend(messages)
        for index, table in enumerate(read_tables(source, selection) if source_kind not in {"hdf5", "sdds"} else []):
            suffix = destination.suffix or (".h5" if kind == "hdf5" else "." + kind)
            target = destination.with_name(destination.stem + suffix)
            if target == source:
                raise ValueError("An output would overwrite the source")
            path = Path(temporary) / target.name
            extra = _write_table(table, path, kind, csv_metadata=csv_metadata, sdds_mode=sdds_mode, sdds_columns=selection.tbt_columns)
            staged.append((path, target))
            staged.extend((p, target.with_name(p.name)) for p in extra)
            counts.append(table.row_count)
            notices.extend(table.notices)
            if progress:
                progress(index + 1, table.label, table.row_count)
        if not staged:
            raise ValueError("No tables selected")
        targets = [target for _, target in staged]
        if len(set(targets)) != len(targets):
            raise ValueError("Output filenames collide")
        if any(target == source or (target.exists() and os.path.samefile(source, target)) for target in targets):
            raise ValueError("An output aliases the source file")
        existing = [str(p) for p in targets if p.exists()]
        if existing and not overwrite:
            raise FileExistsError("Output already exists: " + ", ".join(existing))
        if source_kind == "csv" and Path(str(source) + ".metadata.json") in targets:
            raise ValueError("An output would overwrite source metadata")
        if before != file_signature(source):
            raise ValueError("Source changed during conversion; output was not published")
        if kind == "csv" and not csv_metadata:
            notices.append("CSV metadata sidecar disabled: units, types and descriptions are not preserved.")
            for _, target in staged:
                if Path(str(target) + ".metadata.json").exists():
                    raise FileExistsError("A previous metadata sidecar exists; choose a new output name or keep metadata enabled")
        for path, target in staged:
            if overwrite:
                os.replace(path, target)
            else:
                # Exclusive publication avoids overwriting a file created after the check.
                os.link(path, target)
    return ConversionReport([str(p) for p in targets], counts, list(dict.fromkeys(notices)))


def convert_hdf5(source, destination, selection=None, **kwargs):
    """Convert HDF5 to CSV/TFS, or CSV/TFS to HDF5."""
    if detect_format(source) not in {"hdf5", "csv", "tfs"}:
        raise ValueError("Expected HDF5, CSV or TFS input")
    if detect_format(source) in {"csv", "tfs"}:
        kwargs["output_format"] = "hdf5"
    return convert_file(source, destination, selection, **kwargs)


def convert_sdds(source, destination, selection=None, **kwargs):
    """Convert OMC3 LHC/TbT SDDS to CSV/TFS, or a complete BPM table to SDDS."""
    if detect_format(source) not in {"sdds", "csv", "tfs"}:
        raise ValueError("Expected SDDS, CSV or TFS input")
    if detect_format(source) in {"csv", "tfs"}:
        kwargs["output_format"] = "sdds"
    return convert_file(source, destination, selection, **kwargs)
