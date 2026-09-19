"""HDF5 inspection and explicit table/slice selection; no implicit axis pairing."""
from dataclasses import replace
import json

import h5py
import numpy as np

from PASS.tool.data_conversion import DataTable, _json_value, _restore_json, _select_table


def _attributes(obj):
    result = {}
    for key, value in obj.attrs.items():
        try:
            result[key] = _json_value(value)
        except (ValueError, UnicodeError):
            result[key] = {"unsupported_attribute_type": str(type(value).__name__)}
    return result


def inspect_hdf5(path):
    """List groups/datasets and metadata without reading array payloads."""
    fields, notices = [], []
    with h5py.File(path, "r") as stream:
        metadata = {"parameters": _restore_json(_attributes(stream))}

        def visit(name, obj):
            item = {"name": "/" + name, "attributes": _attributes(obj)}
            if isinstance(obj, h5py.Group):
                item.update(kind="group", shape=[], dtype="")
            else:
                item.update(kind="dataset",
                            shape=list(obj.shape) if obj.shape is not None else None,
                            dtype=str(obj.dtype),
                            fields=list(obj.dtype.names or []))
                if obj.shape == () and (obj.dtype.kind in "biufSU" or h5py.check_string_dtype(obj.dtype) is not None):
                    value = obj.asstr()[()] if h5py.check_string_dtype(obj.dtype) is not None else obj[()]
                    item["value"] = _json_value(value)
                    metadata["parameters"][item["name"]] = value
            fields.append(item)
            for key, value in item["attributes"].items():
                if key != "PASS_CONVERSION_METADATA":
                    metadata["parameters"][item["name"] + "@" + key] = _restore_json(value)
                else:
                    metadata.update(_restore_json(json.loads(value)))

        stream.visititems(visit)
        if "PASS_CONVERSION_METADATA" in stream.attrs:
            metadata.update(_restore_json(json.loads(stream.attrs["PASS_CONVERSION_METADATA"])))
    return {"format": "hdf5", "fields": fields, "metadata": _json_value(metadata), "notices": notices}


def _read(dataset, key):
    dtype = dataset.dtype
    if dtype.names:
        raise ValueError("Select scalar fields of a record dataset")
    if h5py.check_string_dtype(dtype) is not None:
        return np.asarray(dataset.asstr()[key])
    if dtype.kind not in "biuf":
        raise ValueError(f"Unsupported HDF5 datatype {dtype}: {dataset.name}")
    if dtype.kind == "f" and dtype.itemsize > 8:
        raise ValueError(f"HDF5 long-double conversion is unsupported: {dataset.name}")
    return np.asarray(dataset[key])


def read_hdf5_tables(path, selection, *, preview_limit=None):
    """Read explicitly selected columns, matrix or long-form array coordinates.

    HDF5 preview limits apply to source reads when there is no row filter.
    Filtered previews scan selected data to preserve row-selection semantics.
    """
    if not selection.datasets:
        raise ValueError("Select HDF5 datasets explicitly; equal lengths alone do not imply shared rows")
    if len(set(selection.datasets)) != len(selection.datasets):
        raise ValueError("Select each HDF5 dataset only once")
    if preview_limit is not None and selection.filters:
        tables = []
        count = 0
        empty = None
        for table in iter_hdf5_chunks(path, selection):
            if table.row_count:
                tables.append(table)
            elif empty is None:
                empty = table
            count += table.row_count
            if count >= preview_limit:
                break
        if tables:
            first = tables[0]
            yield DataTable({name: np.concatenate([table.columns[name] for table in tables])[:preview_limit]
                             for name in first.columns}, first.metadata, first.label, first.notices)
        elif empty is not None:
            yield empty
        return
    info = inspect_hdf5(path)
    metadata = _restore_json(info["metadata"])
    metadata["source_datasets"] = selection.datasets
    metadata["selection"] = {"mode": selection.mode, "indices": selection.indices, "axes": selection.axes}
    notices = [
        "Unselected datasets, links and array dimensions are not exported. HDF5 hierarchy is recorded as metadata, not reconstructed in tables."
    ]
    with h5py.File(path, "r") as stream:
        datasets = []
        for name in selection.datasets:
            if name not in stream or not isinstance(stream[name], h5py.Dataset):
                raise ValueError(f"Not an HDF5 dataset: {name}")
            dataset = stream[name]
            if dataset.shape is None:
                raise ValueError(f"Empty HDF5 dataspace is not a table: {name}")
            datasets.append(dataset)
        columns = {}
        start, stop, step = selection.rows
        effective_stop = stop
        if preview_limit is not None and not selection.filters:
            effective_stop = min(stop, start + preview_limit * step) if stop is not None else start + preview_limit * step
        row_slice = slice(start, effective_stop, step)
        metadata["column_definitions"] = dict(metadata.get("column_definitions", {}))
        if selection.mode == "columns":
            if any(d.ndim != 1 for d in datasets):
                raise ValueError("Columns mode requires 1-D datasets; choose matrix or long mode for arrays")
            if len({len(d) for d in datasets}) != 1:
                raise ValueError("Datasets have different lengths; select a common row axis or export separately")
            short_names = [d.name.rsplit("/", 1)[-1] for d in datasets]
            for dataset, short_name in zip(datasets, short_names):
                name = short_name if short_names.count(short_name) == 1 else dataset.name
                if dataset.dtype.names:
                    for field in dataset.dtype.names:
                        field_dtype = dataset.dtype.fields[field][0]
                        if field_dtype.subdtype or field_dtype.names or field_dtype.kind not in "biufSUO":
                            raise ValueError(f"Nested/non-scalar record field is unsupported: {field}")
                        if field_dtype.kind == "O" and h5py.check_string_dtype(field_dtype) is None:
                            raise ValueError(f"Non-string variable-length record field is unsupported: {field}")
                        if field_dtype.kind == "f" and field_dtype.itemsize > 8:
                            raise ValueError(f"Long-double record field is unsupported: {field}")
                        key = field if len(datasets) == 1 else name + "." + field
                        if key in columns:
                            raise ValueError(f"Output column name collision: {key}")
                        values = np.asarray(dataset.fields(field)[row_slice])
                        if values.dtype.kind in "SO":
                            values = np.array([v.decode("utf-8") if isinstance(v, bytes) else v for v in values], dtype=object)
                        columns[key] = values
                else:
                    columns[name] = _read(dataset, row_slice)
                    metadata["column_definitions"][name] = _attributes(dataset)
        elif selection.mode in {"matrix", "long"}:
            shapes = {d.shape for d in datasets}
            if len(shapes) != 1:
                raise ValueError("Selected arrays must have identical shapes and explicit shared axes")
            shape = datasets[0].shape
            if not shape:
                raise ValueError("Scalar datasets are available as parameters; select an array for matrix/long export")
            indices = selection.indices or [None] * len(shape)
            if any(axis not in {str(i) for i in range(len(shape))} for axis in selection.axes):
                raise ValueError("Coordinate mapping names an axis outside the array")
            if len(indices) != len(shape):
                raise ValueError("Supply one fixed index or None for each array axis")
            for size, index in zip(shape, indices):
                if index is not None and (type(index) is not int or not 0 <= index < size):
                    raise ValueError("Fixed array index is outside its axis")
            free_axes = [axis for axis, index in enumerate(indices) if index is None]
            if selection.mode == "matrix":
                if len(datasets) != 1 or len(free_axes) != 2:
                    raise ValueError("Matrix export requires one dataset and exactly two unfixed axes")
                key = [slice(None) if index is None else index for index in indices]
                key[free_axes[0]] = row_slice
                values = _read(datasets[0], tuple(key))
                names = datasets[0].attrs.get("columns", [])
                names = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in names]
                if len(names) != values.shape[1]:
                    names = [f"column_{i}" for i in range(values.shape[1])]
                if len(set(names)) != len(names):
                    raise ValueError("Matrix column labels are not unique")
                columns = {name: values[:, i] for i, name in enumerate(names)}
            else:
                selected_shape = tuple(shape[axis] for axis in free_axes)
                total = int(np.prod(selected_shape, dtype=object)) if selected_shape else 1
                row_indices = np.arange(*row_slice.indices(total), dtype=np.int64)
                unravelled = np.unravel_index(row_indices, selected_shape) if selected_shape else []
                coordinates = {}
                for axis, fixed in enumerate(indices):
                    values = np.full(len(row_indices), fixed, dtype=np.int64) if fixed is not None else unravelled[free_axes.index(axis)]
                    coordinates[axis] = values
                    coordinate_path = selection.axes.get(str(axis))
                    name = f"axis_{axis}"
                    if coordinate_path:
                        coordinate = stream[coordinate_path]
                        if not isinstance(coordinate, h5py.Dataset) or coordinate.shape != (shape[axis], ):
                            raise ValueError(f"Coordinate dataset must match axis {axis}: {coordinate_path}")
                        unique, inverse = np.unique(values, return_inverse=True)
                        mapped = _read(coordinate, unique)[inverse]
                        name = coordinate_path.rsplit("/", 1)[-1]
                        metadata["column_definitions"][name] = _attributes(coordinate)
                    else:
                        mapped = values
                    if name in columns:
                        raise ValueError("Axis names collide; choose distinct coordinate datasets")
                    columns[name] = mapped
                for dataset in datasets:
                    name = dataset.name.rsplit("/", 1)[-1]
                    if name in columns:
                        raise ValueError(f"Output column names collide: {name}")
                    # h5py does not support paired fancy indices on several axes.
                    # Read contiguous runs on the last axis to bound memory/I/O.
                    values = np.empty(len(row_indices), dtype=dataset.dtype)
                    pos = 0
                    while pos < len(row_indices):
                        end = pos + 1
                        while end < len(row_indices) and all(coordinates[a][end] == coordinates[a][pos] for a in range(len(shape) - 1)):
                            end += 1
                        prefix = tuple(int(coordinates[a][pos]) for a in range(len(shape) - 1))
                        last = coordinates[len(shape) - 1][pos:end]
                        values[pos:end] = _read(dataset, prefix + (last, ))
                        pos = end
                    columns[name] = values
                    metadata["column_definitions"][name] = _attributes(dataset)
                metadata["array_shape"] = list(shape)
        else:
            raise ValueError("HDF5 supports columns, matrix and long modes")
        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate output columns")
        table = DataTable(columns, metadata, "table", notices)
        # Source row slicing already occurred above; filters still precede preview truncation.
        yield _select_table(table, replace(selection, rows=(0, None, 1)), preview_limit)


def iter_hdf5_chunks(path, selection, *, chunk_size=65536):
    """Bound the number of selected source rows held by the export pipeline."""
    if not selection.datasets:
        raise ValueError("Select HDF5 datasets explicitly")
    with h5py.File(path, "r") as stream:
        dataset = stream[selection.datasets[0]]
        shape = dataset.shape
        if shape is None:
            raise ValueError("Empty HDF5 dataspace is not a table")
        indices = selection.indices or [None] * len(shape)
        if selection.mode == "columns":
            total = shape[0] if shape else 0
        elif selection.mode == "matrix":
            free = [i for i, fixed in enumerate(indices) if fixed is None]
            total = shape[free[0]] if free else 0
        else:
            total = int(np.prod([n for n, fixed in zip(shape, indices) if fixed is None], dtype=object))
    rows = range(*slice(*selection.rows).indices(total))
    if not rows:
        yield from read_hdf5_tables(path, replace(selection, rows=(0, 0, 1)))
        return
    for offset in range(0, len(rows), chunk_size):
        block = rows[offset:offset + chunk_size]
        yield from read_hdf5_tables(path, replace(selection, rows=(block.start, block.stop, block.step)))


def preview_hdf5(path, selection=None, *, limit=500):
    from PASS.tool.data_conversion import preview_file
    return preview_file(path, selection, limit=limit)


def convert_hdf5(source, destination, selection=None, **kwargs):
    from PASS.tool.data_conversion import convert_hdf5 as convert
    return convert(source, destination, selection, **kwargs)
