"""OMC3 LHC/TbT SDDS1 arrays, using the PyLHC SDDS implementation."""
from dataclasses import asdict, replace
from pathlib import Path
import re

import numpy as np

from PASS.tool.data_conversion import DataTable, _cast_column, _json_value, _select_table


def _backend():
    try:
        import sdds
    except ImportError as exc:
        raise ImportError('OMC3 SDDS conversion requires: python -m pip install --editable ".[conversion]"') from exc
    if not hasattr(sdds, "SddsFile") or not hasattr(sdds, "classes"):
        raise ImportError("OMC3 conversion requires the PyLHC sdds package, not another package named sdds")
    return sdds


def _positive_count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"OMC3 {name} must be a positive integer")
    return int(value)


def _load_omc3(path):
    backend = _backend()
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"SDDS1":
            raise ValueError("Only OMC3 LHC/TbT SDDS1 files are supported")
    try:
        source = backend.read(path)
    except (AssertionError, KeyError, NotImplementedError) as exc:
        raise ValueError("Expected an OMC3 LHC/TbT SDDS1 file containing BPM position arrays") from exc
    values = source.values
    required = ("nbOfCapBunches", "nbOfCapTurns", "acqStamp", "bpmNames", "horPositionsConcentratedAndSorted", "verPositionsConcentratedAndSorted")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError("Not an OMC3 LHC/TbT file; missing: " + ", ".join(missing))
    if any(isinstance(item, backend.classes.Column) for item in source.definitions.values()):
        raise ValueError("General SDDS column files are outside the OMC3 conversion scope")
    for name in required:
        expected_type = backend.classes.Parameter if name in {"nbOfCapBunches", "nbOfCapTurns", "acqStamp"} else backend.classes.Array
        if not isinstance(source.definitions[name], expected_type):
            raise ValueError(f"Invalid OMC3 SDDS field definition: {name}")
    n_bunches = _positive_count(values["nbOfCapBunches"], "nbOfCapBunches")
    n_turns = _positive_count(values["nbOfCapTurns"], "nbOfCapTurns")
    names = np.asarray(values["bpmNames"], dtype=object)
    if names.ndim != 1 or not len(names) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("bpmNames must be a nonempty string array")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate BPM names cannot be mapped to a unique table")
    bunch_key = "BunchId" if "BunchId" in values else "horBunchId"
    if bunch_key not in values:
        raise ValueError("Missing OMC3 BunchId / horBunchId array")
    if not isinstance(source.definitions[bunch_key], backend.classes.Array):
        raise ValueError("OMC3 bunch identifiers must be an SDDS array")
    raw_ids = np.asarray(values[bunch_key]).ravel()
    if raw_ids.dtype.kind not in "iu" or len(raw_ids) < n_bunches:
        raise ValueError("OMC3 bunch identifiers must be integers and match nbOfCapBunches")
    bunch_ids = raw_ids[:n_bunches].astype(np.int64)
    if len(set(bunch_ids)) != n_bunches:
        raise ValueError("Duplicate OMC3 bunch identifiers")
    if np.any(raw_ids[:n_bunches] < 0) or np.any(raw_ids[:n_bunches] > np.iinfo(np.int32).max):
        raise ValueError("OMC3 bunch identifiers must fit nonnegative int32")
    shape = (len(names), n_bunches, n_turns)
    positions, definitions = {}, {}
    for plane, name in (("X", "horPositionsConcentratedAndSorted"), ("Y", "verPositionsConcentratedAndSorted")):
        array = np.asarray(values[name])
        if array.dtype.kind != "f" or array.dtype.itemsize not in (4, 8) or array.size != int(np.prod(shape)):
            raise ValueError(f"{name} must contain float32/float64 data with shape BPM x bunch x turn = {shape}")
        positions[plane] = array.reshape(shape)
        definitions[plane] = {key: value for key, value in asdict(source.definitions[name]).items() if value is not None}
    parameters = {name: values[name] for name, item in source.definitions.items() if isinstance(item, backend.classes.Parameter)}
    stamp = parameters["acqStamp"]
    if isinstance(stamp, (bool, np.bool_)) or not isinstance(stamp, (int, np.integer)) or not -(2**63) <= int(stamp) < 2**63:
        raise ValueError("acqStamp must be an int64 nanosecond timestamp")
    known_arrays = {"bpmNames", bunch_key, "horPositionsConcentratedAndSorted", "verPositionsConcentratedAndSorted"}
    excluded = [name for name, item in source.definitions.items() if isinstance(item, backend.classes.Array) and name not in known_arrays]
    notices = [
        "OMC3 SDDS is read as BPM/bunch/turn arrays. Position values and units are unchanged; no units are inferred.",
        "SDDS arrays are loaded in memory; table preview and export are processed in bounded blocks."
    ]
    if excluded:
        notices.append("Additional arrays are not exported: " + ", ".join(excluded))
    metadata = {
        "parameters": parameters,
        "column_definitions": definitions,
        "omc3_tbt": {
            "layout": "lhc",
            "bpm_names": names.tolist(),
            "bunch_ids": bunch_ids.tolist(),
            "n_turns": n_turns,
            "axis_order": ["BPM", "BUNCH", "TURN"]
        }
    }
    return names, bunch_ids, positions, metadata, notices


def inspect_sdds(path):
    """Describe OMC3 BPMs, bunches, turns and normalized table columns."""
    names, bunch_ids, positions, metadata, notices = _load_omc3(path)
    shape = positions["X"].shape
    fields = [{
        "name": name,
        "kind": "column",
        "dtype": dtype,
        "shape": [int(np.prod(shape))],
        "required": name in {"BPM", "BUNCH", "TURN"},
        "attributes": _json_value(metadata["column_definitions"].get(name, {}))
    } for name, dtype in (("BPM", "string"), ("BUNCH", "int64"), ("TURN", "int64"), ("X", str(positions["X"].dtype)), ("Y",
                                                                                                                       str(positions["Y"].dtype)))]
    fields.append({"name": "BPMs", "kind": "group", "shape": [], "dtype": ""})
    fields.extend({"name": "BPMs/" + name, "bpm_name": name, "kind": "bpm", "shape": list(shape[1:]), "dtype": "X / Y"} for name in names)
    return {
        "format": "sdds",
        "profile": "omc3-lhc",
        "fields": fields,
        "metadata": _json_value(metadata),
        "bunch_ids": bunch_ids.tolist(),
        "n_turns": shape[2],
        "notices": notices
    }


def _select_indices(available, requested, label):
    names = list(available)
    selected = names if requested is None else requested
    if not selected or len(set(selected)) != len(selected) or any(name not in names for name in selected):
        raise ValueError(f"Select existing, distinct {label}")
    return np.array([names.index(name) for name in selected], dtype=np.int64)


def iter_sdds_chunks(path, selection, *, chunk_size=65536):
    """Flatten only explicitly selected BPM/bunch/turn samples, in bounded blocks."""
    if selection.mode != "columns":
        raise ValueError("OMC3 SDDS uses the BPM/BUNCH/TURN/X/Y table layout")
    names, bunch_ids, positions, metadata, notices = _load_omc3(path)
    bpm_indices = _select_indices(names, selection.bpms, "BPM names")
    bunch_indices = _select_indices(bunch_ids, selection.bunch_ids, "bunch identifiers")
    n_turns = positions["X"].shape[2]
    start, stop = selection.turns
    stop = n_turns if stop is None else stop
    if not 0 <= start < stop <= n_turns:
        raise ValueError(f"Turn selection must satisfy 0 <= start < stop <= {n_turns}")
    selected_shape = (len(bpm_indices), len(bunch_indices), stop - start)
    metadata["omc3_tbt"]["selected_bpms"] = names[bpm_indices].tolist()
    metadata["omc3_tbt"]["selected_bunch_ids"] = bunch_ids[bunch_indices].tolist()
    metadata["omc3_tbt"]["selected_turn_range"] = [start, stop]
    columns = ["BPM", "BUNCH", "TURN", "X", "Y"] if selection.columns is None else selection.columns
    if not any(name in columns for name in ("X", "Y")) or any(name not in {"BPM", "BUNCH", "TURN", "X", "Y"} for name in columns):
        raise ValueError("Select X and/or Y; BPM, BUNCH and TURN identifiers are always retained")
    if any(name in selection.column_types for name in ("BPM", "BUNCH", "TURN")):
        raise ValueError("OMC3 identifier types cannot be changed")
    columns = ["BPM", "BUNCH", "TURN"] + [name for name in ("X", "Y") if name in columns]
    rows = range(*slice(*selection.rows).indices(int(np.prod(selected_shape))))
    offsets = range(0, len(rows), chunk_size) if rows else [0]
    for offset in offsets:
        block = rows[offset:offset + chunk_size]
        indices = np.arange(block.start, block.stop, block.step, dtype=np.int64)
        bpm, bunch, turn = np.unravel_index(indices, selected_shape)
        original_bpm, original_bunch, original_turn = bpm_indices[bpm], bunch_indices[bunch], turn + start
        data = {
            "BPM": names[original_bpm],
            "BUNCH": bunch_ids[original_bunch],
            "TURN": original_turn,
            "X": positions["X"][original_bpm, original_bunch, original_turn],
            "Y": positions["Y"][original_bpm, original_bunch, original_turn]
        }
        table = DataTable(data, metadata, "omc3_tbt", list(notices))
        yield _select_table(table, replace(selection, rows=(0, None, 1), columns=columns))


def read_sdds_tables(path, selection, *, preview_limit=None):
    tables, count, first = [], 0, None
    chunk_size = 65536 if selection.filters or preview_limit is None else preview_limit
    for table in iter_sdds_chunks(path, selection, chunk_size=chunk_size):
        if first is None:
            first = table
        if table.row_count:
            remaining = None if preview_limit is None else preview_limit - count
            tables.append({name: values[:remaining] for name, values in table.columns.items()})
            count += min(table.row_count, remaining) if remaining is not None else table.row_count
        if preview_limit is not None and count >= preview_limit:
            break
    if tables:
        yield DataTable({name: np.concatenate([table[name] for table in tables])
                         for name in first.columns}, first.metadata, first.label, first.notices)
    elif first is not None:
        yield first


def preview_sdds_selection(path, selection, limit):
    """Check all selected samples while retaining only bounded table blocks."""
    first, preview, preview_count = None, [], 0
    names, bunches = set(), set()
    total, turn_mask = 0, None
    for table in iter_sdds_chunks(path, selection):
        if first is None:
            first = table
            turn_mask = np.zeros(table.metadata["omc3_tbt"]["n_turns"], dtype=bool)
        if not table.row_count:
            continue
        total += table.row_count
        names.update(table.columns["BPM"])
        bunches.update(table.columns["BUNCH"])
        turn_mask[table.columns["TURN"]] = True
        remaining = limit - preview_count
        if remaining > 0:
            preview.append({name: values[:remaining] for name, values in table.columns.items()})
            preview_count += min(remaining, table.row_count)
    columns = {name: np.concatenate([block[name] for block in preview]) if preview else values[:0] for name, values in first.columns.items()}
    table = replace(first, columns=columns)
    # Source selectors preserve unique BPM/bunch/turn tuples. Cardinalities are
    # therefore sufficient to check the full Cartesian grid without storing it.
    sample = replace(table, columns={name: values[:1] for name, values in columns.items()})
    result = check_sdds_table(sample)
    turn_count = int(np.count_nonzero(turn_mask))
    start = int(np.argmax(turn_mask)) if turn_count else 0
    stop = len(turn_mask) - int(np.argmax(turn_mask[::-1])) if turn_count else 0
    result.update(row_count=total, bpm_count=len(names), bunch_count=len(bunches), turn_count=turn_count, turn_start=start, turn_stop=stop)
    if result["compatible"]:
        if any(not name.isascii() for name in names):
            result.update(compatible=False, code="invalid_data", reason="BPM names must be ASCII strings for PyLHC SDDS output")
        elif turn_count != stop - start:
            result.update(compatible=False, code="turn_gaps", reason="OMC3 requires consecutive turn samples")
        elif total != len(names) * len(bunches) * turn_count:
            result.update(compatible=False, code="incomplete_grid", reason="Incomplete BPM/bunch/turn grid")
        elif total > np.iinfo(np.int32).max:
            result.update(compatible=False, code="invalid_data", reason="Selected OMC3 array exceeds the SDDS1 int32 dimension range")
    return table, result


class SddsTableError(ValueError):
    """Machine-readable reason shared by preview and export validation."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _prepare_sdds_file(table, columns_map=None):
    """Validate the complete table before preview approval or file writing."""
    import pandas as pd
    backend = _backend()
    mapping = {name: name for name in ("BPM", "BUNCH", "TURN", "X", "Y")}
    mapping.update(columns_map or {})
    if set(mapping) != {"BPM", "BUNCH", "TURN", "X", "Y"} or len(set(mapping.values())) != 5:
        raise SddsTableError("mapping", "Map BPM, BUNCH, TURN, X and Y to five distinct source columns")
    if any(name not in table.columns for name in mapping.values()):
        raise SddsTableError("missing_columns", "OMC3 output requires BPM, BUNCH, TURN, X and Y; choose the corresponding source columns")
    data = {name: np.asarray(table.columns[column]) for name, column in mapping.items()}
    if not len(data["BPM"]):
        raise SddsTableError("empty", "No samples selected for OMC3 output")
    if any(not isinstance(name, str) or not name or not name.isascii() for name in data["BPM"]):
        raise ValueError("BPM names must be nonempty ASCII strings; retain string types when reading CSV")
    data["BUNCH"] = _cast_column(data["BUNCH"], "int32")
    data["TURN"] = _cast_column(data["TURN"], "int64")
    if np.any(data["BUNCH"] < 0) or np.any(data["TURN"] < 0):
        raise ValueError("Bunch identifiers and turn indices must be nonnegative")
    for plane in ("X", "Y"):
        if data[plane].dtype.kind != "f" or data[plane].dtype.itemsize not in (4, 8):
            data[plane] = _cast_column(data[plane], "float64")
    frame = pd.DataFrame(data)
    if frame.duplicated(["BPM", "BUNCH", "TURN"]).any():
        raise SddsTableError("duplicate", "Duplicate BPM/BUNCH/TURN samples cannot be converted to OMC3")
    names, bunch_ids = list(pd.unique(frame["BPM"])), list(pd.unique(frame["BUNCH"]))
    turns = np.unique(data["TURN"])
    if np.any(np.diff(turns) != 1):
        raise SddsTableError("turn_gaps", "OMC3 requires consecutive turn samples; missing or downsampled turns cannot be inferred")
    if len(frame) != len(names) * len(bunch_ids) * len(turns):
        raise SddsTableError("incomplete_grid", "Incomplete BPM/bunch/turn grid: every selected BPM and bunch needs both planes for every turn")
    index = pd.MultiIndex.from_product([names, bunch_ids, turns], names=["BPM", "BUNCH", "TURN"])
    ordered = frame.set_index(["BPM", "BUNCH", "TURN"]).reindex(index)
    parameters = table.metadata.get("parameters", {})
    stamp = int(_cast_column(np.asarray([parameters.get("acqStamp", 0)], dtype=object), "int64")[0])
    definitions = [
        backend.classes.Parameter("acqStamp", "llong"),
        backend.classes.Parameter("nbOfCapBunches", "long"),
        backend.classes.Parameter("nbOfCapTurns", "long")
    ]
    values = [stamp, len(bunch_ids), len(turns)]
    if max(len(names), len(bunch_ids), len(turns), len(frame)) > np.iinfo(np.int32).max:
        raise ValueError("Selected OMC3 array exceeds the SDDS1 int32 dimension range")
    for name, value in parameters.items():
        if name in {"acqStamp", "nbOfCapBunches", "nbOfCapTurns"}:
            continue
        if name in {"BunchId", "bpmNames", "horPositionsConcentratedAndSorted", "verPositionsConcentratedAndSorted"}:
            raise ValueError(f"Parameter name conflicts with an OMC3 array: {name}")
        if isinstance(value, np.generic):
            value = value.item()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]*", name) or type(value) not in {int, float, str}:
            table.notices.append(f"Parameter not representable in OMC3 SDDS: {name}")
            continue
        if isinstance(value, str) and not value.isascii():
            raise ValueError(f"PyLHC SDDS output requires ASCII string parameters: {name}")
        dtype = "string" if isinstance(value, str) else "double" if isinstance(value, float) else "llong"
        if dtype == "llong" and not -(2**63) <= value < 2**63:
            raise ValueError(f"SDDS parameter exceeds int64: {name}")
        definitions.append(backend.classes.Parameter(name, dtype))
        values.append(value)
    definitions.extend([backend.classes.Array("BunchId", "long"), backend.classes.Array("bpmNames", "string")])
    values.extend([bunch_ids, names])
    expected = {}
    for plane, name in (("X", "horPositionsConcentratedAndSorted"), ("Y", "verPositionsConcentratedAndSorted")):
        array = ordered[plane].to_numpy()
        dtype = "float" if array.dtype.itemsize == 4 else "double"
        units = table.metadata.get("column_definitions", {}).get(mapping[plane], {}).get("units")
        if units is not None and (not isinstance(units, str) or any(char in units for char in ',="\r\n')):
            raise ValueError(f"Unsupported SDDS units attribute for {plane}")
        definitions.append(backend.classes.Array(name, dtype, units=units))
        values.append(array)
        expected[plane] = array.reshape(len(names), len(bunch_ids), len(turns))
    details = {
        "compatible": True,
        "reason": "",
        "code": "",
        "row_count": len(frame),
        "bpm_count": len(names),
        "bunch_count": len(bunch_ids),
        "turn_count": len(turns),
        "turn_start": int(turns[0]),
        "turn_stop": int(turns[-1]) + 1,
        "timestamp": str(stamp) if "acqStamp" in parameters else None,
        "units": {
            plane: table.metadata.get("column_definitions", {}).get(mapping[plane], {}).get("units")
            for plane in ("X", "Y")
        }
    }
    if "acqStamp" not in parameters:
        table.notices.append("No acquisition timestamp supplied; acqStamp=0 denotes an unspecified time.")
    if turns[0] != 0:
        table.notices.append(f"OMC3 stores turns starting at 0; original selected turn {turns[0]} becomes turn 0.")
    return backend.SddsFile("SDDS1", None, definitions, values), expected, details


def check_sdds_table(table, columns_map=None):
    """Check the entire table, including samples beyond the display preview."""
    candidate = replace(table, notices=list(table.notices))
    try:
        _, _, result = _prepare_sdds_file(candidate, columns_map)
        result["notices"] = candidate.notices
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        return {"compatible": False, "code": getattr(exc, "code", "invalid_data"), "reason": str(exc), "row_count": table.row_count}


def write_sdds_table(table, destination, *, mode="binary", columns_map=None):
    """Write binary SDDS1 without narrowing positions, then verify OMC3 reading."""
    import turn_by_turn as tbt
    if mode != "binary":
        raise ValueError("OMC3 output uses binary SDDS1; general SDDS ASCII export is not supported")
    source, expected, details = _prepare_sdds_file(table, columns_map)
    _backend().write(source, destination)
    restored = tbt.read_tbt(destination, datatype="lhc")
    if list(restored.bunch_ids) != list(source.values["BunchId"]) or restored.nturns != details["turn_count"]:
        raise ValueError("OMC3 read-back changed bunch identifiers or turn count")
    for bunch, matrix in enumerate(restored.matrices):
        for plane in ("X", "Y"):
            frame = getattr(matrix, plane)
            if list(frame.index) != list(
                    source.values["bpmNames"]) or not np.array_equal(frame.to_numpy(), expected[plane][:, bunch, :], equal_nan=True):
                raise ValueError(f"OMC3 read-back changed {plane} data")


def preview_sdds(path, selection=None, *, limit=500):
    from PASS.tool.data_conversion import preview_file
    return preview_file(path, selection, limit=limit)


def convert_sdds(source, destination, selection=None, **kwargs):
    from PASS.tool.data_conversion import convert_sdds as convert
    return convert(source, destination, selection, **kwargs)
