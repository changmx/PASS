"""Explicit numeric signal selection for offline spectral analysis.

Algorithms consume arrays; this module handles file layout and sampling checks.
No particles are regrouped, missing samples filled, or lost tails interpolated.
"""

from contextlib import contextmanager
import csv
import json
from pathlib import Path
import shlex
import zipfile

import numpy as np


def _format(path):
    suffix = Path(path).suffix.lower()
    formats = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".txt": "text",
        ".dat": "text",
        ".tfs": "tfs",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".hdf": "hdf5",
        ".npy": "npy",
        ".npz": "npz"
    }
    if suffix not in formats:
        raise ValueError("Supported signal files: CSV, TSV, TXT, DAT, TFS, HDF5, NPY and NPZ")
    return formats[suffix]


def _native(value):
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()] if value.ndim else _native(value.item())
    if isinstance(value, np.generic):
        return _native(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def _numeric(values):
    """Parse an entire column without dropping any row on conversion failure."""
    stripped = [value.strip() for value in values]
    if stripped and all(value.lower() in {"true", "false"} for value in stripped):
        return np.asarray([value.lower() == "true" for value in stripped], dtype=bool)
    converted = ["nan" if not value else value for value in stripped]
    for dtype in (np.int64, np.float64, np.complex128):
        try:
            return np.asarray(converted, dtype=dtype)
        except (ValueError, OverflowError):
            continue
    return np.asarray(values, dtype=str)


def _read_text(path, kind, delimiter, skip_rows, header):
    if isinstance(skip_rows, bool) or not isinstance(skip_rows, (int, np.integer)) or skip_rows < 0:
        raise ValueError("skip_rows must be a nonnegative integer")
    if isinstance(header, bool):
        header = "present" if header else "absent"
    if header not in {"auto", "present", "absent"}:
        raise ValueError("header must be 'auto', 'present' or 'absent'")
    if delimiter in (None, ""):
        delimiter = {"csv": ",", "tsv": "\t"}.get(kind)
    if delimiter in ("whitespace", r"\s+"):
        delimiter = None
    if delimiter == r"\t":
        delimiter = "\t"
    if delimiter is not None and (not isinstance(delimiter, str) or len(delimiter) != 1):
        raise ValueError("delimiter must be one character or 'whitespace'")
    comments = []
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for line_index, line in enumerate(stream):
            if line_index < skip_rows or not line.strip():
                continue
            if line.lstrip().startswith(("#", "!")):
                comments.append(line.rstrip())
                continue
            values = shlex.split(line, comments=False) if delimiter is None else next(csv.reader([line], delimiter=delimiter, strict=True))
            rows.append(values)
    if not rows:
        raise ValueError("The text file has no data rows after skipped lines and comments")
    has_header = header == "present" or (header == "auto" and any(_numeric([value]).dtype.kind not in "biufc" for value in rows[0]))
    names = [name.strip() for name in rows.pop(0)] if has_header else [f"column_{index}" for index in range(len(rows[0]))]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Column names must be nonempty and unique")
    if any(len(row) != len(names) for row in rows):
        raise ValueError("Text rows have different numbers of columns; no rows were discarded")
    metadata = {"header": names if has_header else None, "delimiter": delimiter, "skip_rows": int(skip_rows), "comments": comments}
    return {name: _numeric([row[index] for row in rows]) for index, name in enumerate(names)}, metadata


def _read_table(path, kind, delimiter, skip_rows, header):
    if kind == "tfs" or (kind == "csv" and Path(str(path) + ".metadata.json").exists()):
        if delimiter not in (None, "", ",") or skip_rows or header not in ("auto", "present", True):
            raise ValueError("TFS and CSV metadata sidecars require their original header and delimiter")
        if kind == "tfs":
            from PASS.utils.particle_monitor_read import particle_monitor_tfs_metadata, read_particle_monitor_tfs

            if particle_monitor_tfs_metadata(path) is not None:
                records, parameters = read_particle_monitor_tfs(path)
                selected = records["record"] == 1
                shape = (parameters["ValidSamples"], parameters["MaxTag"])
                columns = {name: values[selected].reshape(shape) for name, values in records.items() if name not in {"record", "turn", "particle_id"}}
                columns["turn"] = records["turn"][selected].reshape(shape)[:, 0]
                columns["particle_id"] = np.arange(1, shape[1] + 1, dtype=np.int64)
                return columns, {"parameters": parameters, "notices": ["Initial records excluded; samples grouped by particle_id"]}
        from PASS.tool.data_conversion import read_tables

        table = next(read_tables(path))
        metadata = dict(table.metadata)
        metadata["notices"] = list(table.notices)
        return table.columns, metadata
    return _read_text(path, kind, delimiter, skip_rows, header)


def _array_description(name, shape, dtype, attributes=None):
    return {"name": name, "shape": None if shape is None else list(shape), "dtype": str(dtype), "attributes": _native(attributes or {})}


def _npy_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version in ((2, 0), (3, 0)):
        # Numeric, unstructured headers are ASCII in both versions.
        return np.lib.format.read_array_header_2_0(stream)
    raise ValueError(f"Unsupported NPY header version {version}; save a plain numeric NumPy array")


def inspect_data(path, *, delimiter=None, skip_rows=0, header="auto"):
    """List numeric arrays/columns, shapes and metadata without loading HDF5 data.

    NPY/NPZ inspection reads array headers only. Text/TFS inspection reads the
    table. Nonnumeric and scalar entries appear in ``excluded`` rather than
    selectable ``arrays``. HDF5 dataset names are absolute paths. A standalone
    NPY array is named ``data``. Headerless text columns are ``column_0``, etc.
    """
    path = Path(path)
    kind = _format(path)
    arrays = []
    excluded = []
    metadata = {}

    def add(name, shape, dtype, attributes=None):
        description = _array_description(name, shape, dtype, attributes)
        if dtype.kind in "biufc" and shape is not None and len(shape) >= 1:
            arrays.append(description)
        else:
            description["reason"] = "A signal requires a numeric nonscalar array"
            excluded.append(description)

    if kind == "hdf5":
        import h5py

        with h5py.File(path, "r") as stream:
            metadata = {"parameters": _native(dict(stream.attrs)), "groups": {}}

            def visit(name, item):
                if isinstance(item, h5py.Dataset):
                    add("/" + name, item.shape, item.dtype, dict(item.attrs))
                elif item.attrs:
                    metadata["groups"]["/" + name] = _native(dict(item.attrs))

            stream.visititems(visit)
    elif kind == "npy":
        with path.open("rb") as stream:
            shape, _, dtype = _npy_header(stream)
        add("data", shape, dtype)
    elif kind == "npz":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith(".npy"):
                    continue
                with archive.open(name) as stream:
                    shape, _, dtype = _npy_header(stream)
                add(name[:-4], shape, dtype)
    else:
        columns, metadata = _read_table(path, kind, delimiter, skip_rows, header)
        for name, array in columns.items():
            add(name, array.shape, array.dtype)
    metadata.update({"source": str(path.resolve()), "format": kind})
    return {"arrays": arrays, "excluded": excluded, "metadata": _native(metadata)}


@contextmanager
def _open_data(path, kind, delimiter, skip_rows, header):
    if kind == "hdf5":
        import h5py

        with h5py.File(path, "r") as stream:
            yield stream, {"parameters": _native(dict(stream.attrs))}
    elif kind == "npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            yield {"data": array}, {}
        finally:
            if isinstance(array, np.memmap):
                array._mmap.close()
    elif kind == "npz":
        with np.load(path, allow_pickle=False) as archive:
            yield archive, {}
    else:
        yield _read_table(path, kind, delimiter, skip_rows, header)


def _get_array(source, name):
    if not isinstance(name, str) or not name:
        raise ValueError("Select a named numeric column or dataset")
    if name not in source:
        raise ValueError(f"Unknown signal column/dataset: {name}")
    try:
        array = source[name]
    except ValueError as error:
        raise ValueError(f"Cannot read {name!r} as a numeric array without pickle: {error}") from error
    if not hasattr(array, "dtype") or array.dtype.kind not in "biufc" or array.shape is None or len(array.shape) < 1:
        raise ValueError(f"{name!r} must contain a numeric nonscalar array; object/string/structured arrays are unsupported")
    return array


def _hdf5_selection_metadata(array, metadata):
    """Restore PASS table provenance only along the selected dataset's path."""
    metadata = dict(metadata)
    groups = []
    parent = array.parent
    while True:
        groups.append(parent)
        if parent.name == "/":
            break
        parent = parent.parent
    metadata["group_attributes"] = {}
    for group in reversed(groups):
        attributes = _native(dict(group.attrs))
        metadata["group_attributes"][group.name] = attributes
        packed = attributes.get("PASS_CONVERSION_METADATA")
        if packed:
            from PASS.tool.data_conversion import _restore_json

            restored = _restore_json(json.loads(packed))
            if not isinstance(restored, dict):
                raise ValueError("PASS_CONVERSION_METADATA must contain a metadata object")
            metadata.update(restored)
    return metadata


def _bounds(value, length, label):
    if value is None:
        return 0, length
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{label} must be [start, end), with two indices")
    start, end = value
    start = 0 if start is None else start
    end = length if end is None else end
    if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)) for item in (start, end)):
        raise ValueError(f"{label} indices must be integers")
    if not 0 <= start < end <= length:
        raise ValueError(f"{label} must satisfy 0 <= start < end <= {length}")
    return int(start), int(end)


def _select_array(array, sample_axis, sample_range, object_range):
    if isinstance(sample_axis, (bool, np.bool_)) or not isinstance(sample_axis, (int, np.integer)):
        raise ValueError("sample_axis must be an integer")
    if not -array.ndim <= sample_axis < array.ndim:
        raise ValueError(f"sample_axis is outside the {array.ndim} array dimensions")
    sample_axis %= array.ndim
    start, end = _bounds(sample_range, array.shape[sample_axis], "sample_range")
    object_shape = tuple(length for axis, length in enumerate(array.shape) if axis != sample_axis)
    n_objects = int(np.prod(object_shape)) if object_shape else 1
    if n_objects == 0 or end - start < 2:
        raise ValueError("Select at least one signal with at least two samples")
    indices = [slice(None)] * array.ndim
    indices[sample_axis] = slice(start, end)
    if object_range is None:
        signal = np.moveaxis(np.asarray(array[tuple(indices)]), sample_axis, -1).copy()
        object_ids = np.arange(n_objects).reshape(object_shape)
    else:
        first, last = _bounds(object_range, n_objects, "object_range")
        if not object_shape:
            signal = np.asarray(array[tuple(indices)]).copy().reshape(1, end - start)
        elif len(object_shape) == 1:
            indices[1 - sample_axis] = slice(first, last)
            signal = np.moveaxis(np.asarray(array[tuple(indices)]), sample_axis, -1).copy()
        else:
            # Read only requested objects; HDF5 cannot pair fancy indices on all axes.
            rows = []
            for index in range(first, last):
                coordinates = iter(np.unravel_index(index, object_shape))
                indices = [slice(start, end) if axis == sample_axis else next(coordinates) for axis in range(array.ndim)]
                rows.append(np.asarray(array[tuple(indices)]))
            signal = np.stack(rows)
        object_ids = np.arange(first, last)
    return signal, object_ids, sample_axis, (start, end)


def _sampling_coordinates(source, coordinate, n_original, bounds, sample_spacing, *, allow_tail=False):
    start, end = bounds
    if coordinate is None:
        if isinstance(sample_spacing, (bool, np.bool_)) or not np.isscalar(sample_spacing):
            raise ValueError("sample_spacing must be a finite positive number")
        try:
            spacing = float(sample_spacing)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("sample_spacing must be a finite positive number") from error
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("sample_spacing must be a finite positive number")
        return np.arange(start, end, dtype=float) * spacing, spacing
    values = _get_array(source, coordinate)
    shape_matches = values.ndim == 1 and (values.shape[0] >= n_original if allow_tail else values.shape[0] == n_original)
    if not shape_matches or values.dtype.kind == "c":
        raise ValueError("The sample coordinate must be a real 1-D array matching the original sampling axis")
    coordinates = np.asarray(values[start:end], dtype=np.float64)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Sample coordinates contain nonfinite values")
    differences = np.diff(coordinates)
    if np.any(differences <= 0):
        raise ValueError(
            "Sample coordinates must increase strictly; repeated turns may indicate multiple particles. Select/group one trajectory first")
    spacing = float(np.median(differences))
    # Large absolute epochs must not make a missing sample look like roundoff.
    tolerance = min(max(spacing * 1e-9, np.max(np.abs(coordinates)) * np.finfo(float).eps * 8), spacing * 1e-6)
    if not np.all(np.abs(differences - spacing) <= tolerance):
        raise ValueError("Sample coordinates are not uniformly spaced or their numeric precision is insufficient; select a continuous interval")
    return coordinates.copy(), spacing


def load_signal(path,
                selection,
                *,
                sample_axis=-1,
                sample_range=None,
                object_range=None,
                coordinate=None,
                sample_spacing=1.0,
                delimiter=None,
                skip_rows=0,
                header="auto",
                alive_selection=None):
    """Read a numeric signal and return samples on its last axis.

    ``selection`` and ``coordinate`` name explicit columns or datasets. The
    sampling coordinate, when supplied, determines the returned spacing and
    overrides ``sample_spacing``. Otherwise spacing is in caller-chosen units.
    All ranges are half-open Python indices, not turn/time coordinate values.
    ``object_range`` selects flattened nonsampling dimensions in C order and
    produces ``(objects, samples)``; omitting it preserves leading dimensions.

    Text delimiter defaults are comma for CSV, tab for TSV and whitespace for
    TXT/DAT. ``header`` is auto/present/absent (True/False also work), and
    ``skip_rows`` counts physical initial lines. TFS and CSV metadata sidecars
    reuse PASS's typed readers and preserve their metadata.

    ``alive_selection`` explicitly names a real mask or signed tag (>0 live),
    with the signal's original shape or one value per sample. PASS Particle
    Monitor tables automatically check their tag column. Lost/unavailable
    samples cause an error; select a complete live interval to analyze them.
    NumPy object arrays are never unpickled. HDF5 and NPY selections are sliced
    before copying; compressed NPZ and text tables require loading the member.
    Single-file PASS ParticleMonitor data always use axis 0 for samples;
    their particle IDs replace generic object indices in the result.
    """
    path = Path(path)
    kind = _format(path)
    with _open_data(path, kind, delimiter, skip_rows, header) as (source, metadata):
        array = _get_array(source, selection)
        if kind == "hdf5":
            metadata = _hdf5_selection_metadata(array, metadata)
        parameters = metadata.get("parameters", {})
        monitor = parameters.get("Name") == "PASS Particle Monitor"
        single_file = monitor and parameters.get("Layout") == "single_file"
        if monitor and parameters.get("Layout") not in {"single_file", "table"}:
            raise ValueError("ParticleMonitor requires the current single-file format")
        if single_file:
            if array.ndim != 2 or (kind == "hdf5" and array.parent.name != "/"):
                raise ValueError("Select a root turn-by-turn dataset from single-file ParticleMonitor output")
            committed = parameters.get("ValidSamples", -1)
            if (parameters.get("FormatVersion") != 2 or isinstance(committed, (bool, np.bool_)) or not isinstance(committed, (int, np.integer))
                    or not 0 <= committed <= array.shape[0]):
                raise ValueError("Unsupported or incomplete single-file ParticleMonitor output")
            sample_range = _bounds(sample_range, committed, "sample_range")
            sample_axis = 0
        signal, object_ids, axis, bounds = _select_array(array, sample_axis, sample_range, object_range)
        if single_file:
            object_ids = np.asarray(source["particle_id"][int(object_ids[0]):int(object_ids[-1]) + 1])
        prefix = array.parent.name.rstrip("/") + "/" if kind == "hdf5" and array.parent.name != "/" else ""
        turn_selection = prefix + "turn"
        tag_selection = prefix + "tag"
        if monitor:
            if coordinate is None and turn_selection in source:
                coordinate = turn_selection
            if alive_selection is None and tag_selection in source:
                alive_selection = tag_selection
        coordinates, spacing = _sampling_coordinates(source,
                                                     coordinate,
                                                     committed if single_file else array.shape[axis],
                                                     bounds,
                                                     sample_spacing,
                                                     allow_tail=single_file)
        if alive_selection is not None:
            alive = _get_array(source, alive_selection)
            if alive.dtype.kind == "c":
                raise ValueError("Alive masks must be real, with positive values marking live samples")
            if alive.shape == array.shape or (single_file and alive.ndim == 2 and alive.shape[0] >= committed and alive.shape[1] == array.shape[1]):
                live_values = _select_array(alive, sample_axis, sample_range, object_range)[0]
            elif alive.shape == (array.shape[axis], ):
                live_values = np.asarray(alive[bounds[0]:bounds[1]])
            else:
                raise ValueError("Alive mask shape must match the signal or its sample coordinate")
            if not np.all(np.isfinite(live_values) & (live_values > 0)):
                raise ValueError("Selection contains lost/unavailable samples; choose a continuous live interval before loss")
        if not np.all(np.isfinite(signal)):
            raise ValueError("Selection contains nonfinite signal samples; choose a complete continuous interval (rows are never dropped)")
        if monitor:
            tags = (live_values if alive_selection == tag_selection else _select_array(source[tag_selection], sample_axis, sample_range,
                                                                                       object_range)[0]) if tag_selection in source else None
            if tags is not None and np.any(tags != tags[..., :1]):
                raise ValueError("A Particle Monitor signal must follow one particle tag; group trajectories explicitly")
        metadata.update({
            "source": str(path.resolve()),
            "format": kind,
            "selection": selection,
            "original_shape": list(array.shape),
            "sample_axis": int(axis),
            "sample_range": list(bounds),
            "object_range": object_range,
            "coordinate": coordinate,
            "spacing_source": "coordinate" if coordinate is not None else "sample_spacing",
            "alive_selection": alive_selection
        })
        if hasattr(array, "attrs"):
            metadata["dataset_attributes"] = _native(dict(array.attrs))
        if coordinate is not None and hasattr(source[coordinate], "attrs"):
            metadata["coordinate_attributes"] = _native(dict(source[coordinate].attrs))
    return {"signal": signal, "sample_spacing": spacing, "sample_coordinates": coordinates, "object_ids": object_ids, "metadata": _native(metadata)}
