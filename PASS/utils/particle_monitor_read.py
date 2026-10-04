"""Read current single-file ParticleMonitor histories."""

import io
from pathlib import Path
import shlex

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _read_header(stream):
    attributes, names, types = {}, None, None
    while line := stream.readline():
        text = line.decode("utf-8-sig").strip()
        if not text or text.startswith(("#", "!")):
            continue
        fields = shlex.split(text)
        if fields[0] == "@" and len(fields) == 4:
            kind, value = fields[2:]
            attributes[fields[1]] = value if kind.endswith("s") else int(value) if kind.endswith("d") else float(value)
        elif fields[0] == "*":
            names = fields[1:]
        elif fields[0] == "$":
            types = fields[1:]
            break
        else:
            raise ValueError("Invalid ParticleMonitor TFS header")
    if not names or types is None or len(names) != len(types) or len(set(names)) != len(names):
        raise ValueError("ParticleMonitor TFS requires unique typed columns")
    return attributes, names, types, stream.tell()


def _inspect(stream):
    # Leave other TFS dialects to their existing reader.
    identifiers = {}
    while line := stream.readline():
        if line.lstrip().startswith(b"$"):
            break
        if line.lstrip().startswith(b"@"):
            fields = shlex.split(line.decode("utf-8-sig"))
            if len(fields) == 4 and fields[1] in {"Name", "Layout"}:
                identifiers[fields[1]] = fields[3]
    if identifiers.get("Name") != "PASS Particle Monitor" or identifiers.get("Layout") == "table":
        return None
    if identifiers.get("Layout") != "single_file":
        raise ValueError("ParticleMonitor requires the current single-file format")
    stream.seek(0)
    attributes, names, types, start = _read_header(stream)
    if attributes.get("FormatVersion") != 2:
        raise ValueError("Unsupported ParticleMonitor format; current format version is 2")
    required = {"record", "turn", "particle_id", "x", "px", "y", "py", "z", "dp", "tag", "lostTurn", "lostPosition"}
    if not required.issubset(names) or any(kind not in {"%d", "%le"} for kind in types):
        raise ValueError("ParticleMonitor TFS requires numeric trajectory and identity columns")
    for name in ("record", "turn", "particle_id", "tag", "lostTurn"):
        if types[names.index(name)] != "%d":
            raise ValueError(f"ParticleMonitor {name} must be an integer column")
    _integer(attributes.get("MaxTag"), "MaxTag", 1)
    for name in ("ValidSamples", "ValidRows", "NumTurn", "StartTurn", "EndTurn", "RequestedEndTurn"):
        _integer(attributes.get(name), name)
    if attributes["NumTurn"] != attributes["ValidSamples"]:
        raise ValueError("ParticleMonitor sample counts disagree")
    if attributes.get("Completed") not in (0, 1):
        raise ValueError("ParticleMonitor Completed must be 0 or 1")
    attributes["Completed"] = bool(attributes["Completed"])
    return attributes, names, types, start


def particle_monitor_tfs_metadata(path):
    """Return current PM metadata, or None for an ordinary TFS table."""
    with Path(path).open("rb") as stream:
        description = _inspect(stream)
    return description[0] if description is not None else None


def iter_particle_monitor_tfs(path, *, block_rows=65536, cancel=None):
    """Yield bounded numeric row blocks and validate the complete TFS table.

    Initial rows precede the turn-major sample rows. The iterator must be
    exhausted to verify the final row/sample counts.
    """
    block_rows = _integer(block_rows, "block_rows", 1)
    with Path(path).open("rb") as stream:
        description = _inspect(stream)
        if description is None:
            raise ValueError("Not a current ParticleMonitor TFS")
        attributes, names, types, start = description
        n_particles = attributes["MaxTag"]
        dtype = [(name, np.int64 if kind == "%d" else np.float64) for name, kind in zip(names, types)]
        n_rows, n_samples, last_initial_id, previous_turn = 0, 0, 0, -1
        stream.seek(start)
        while True:
            rows = []
            while len(rows) < block_rows and (line := stream.readline()):
                text = line.strip()
                if text.startswith(b"# PASS_PM_COMMIT"):
                    raise ValueError("ParticleMonitor TFS must be a finalized standard table")
                if text and not text.startswith((b"#", b"!")):
                    rows.append(line)
            if not rows:
                break
            if cancel is not None and cancel():
                raise InterruptedError("ParticleMonitor reading cancelled")
            array = np.loadtxt(io.BytesIO(b"".join(rows)), dtype=dtype, ndmin=1)
            values = {name: array[name] for name in names}
            record, turn, particle_id = (values[name] for name in ("record", "turn", "particle_id"))
            if np.any((record != 0) & (record != 1)) or np.any(turn < 0) or np.any((particle_id < 1) | (particle_id > n_particles)):
                raise ValueError("Invalid ParticleMonitor record, turn or particle identity")
            initial = record == 0
            initial_ids = particle_id[initial]
            if len(initial_ids):
                if n_samples or initial_ids[0] <= last_initial_id or np.any(np.diff(initial_ids) <= 0) or np.any(np.diff(record) < 0):
                    raise ValueError("Initial ParticleMonitor rows must precede samples in particle-ID order")
                if np.any(values["tag"][initial] != initial_ids):
                    raise ValueError("Initial ParticleMonitor rows require positive particle tags")
                last_initial_id = int(initial_ids[-1])
            selected = ~initial
            sample_ids, sample_turns = particle_id[selected], turn[selected]
            if len(sample_ids):
                positions = n_samples + np.arange(len(sample_ids), dtype=np.int64)
                if np.any(sample_ids != positions % n_particles + 1):
                    raise ValueError("ParticleMonitor samples must contain every particle ID in order")
                differences = np.diff(np.r_[previous_turn, sample_turns])
                starts = positions % n_particles == 0
                if np.any(differences[starts] <= 0) or np.any(differences[~starts] != 0):
                    raise ValueError("ParticleMonitor samples have inconsistent or non-increasing turns")
                tags = values["tag"][selected]
                if np.any((tags != 0) & (np.abs(tags) != sample_ids)):
                    raise ValueError("ParticleMonitor sample tags disagree with particle IDs")
                previous_turn = int(sample_turns[-1])
                n_samples += len(sample_ids)
            n_rows += len(array)
            yield values
        if n_samples != attributes["ValidSamples"] * n_particles or n_rows != attributes["ValidRows"]:
            raise ValueError("ParticleMonitor data do not cover the declared row and sample counts")
        if n_samples and attributes["EndTurn"] != previous_turn + 1:
            raise ValueError("ParticleMonitor EndTurn disagrees with its samples")
        if cancel is not None and cancel():
            raise InterruptedError("ParticleMonitor reading cancelled")


def read_particle_monitor_tfs(path, *, requested_turn=None, last_sample_only=False, cancel=None):
    """Read all rows, or initial rows plus the latest requested sample."""
    if requested_turn is not None:
        requested_turn = _integer(requested_turn, "requested_turn")
    attributes = particle_monitor_tfs_metadata(path)
    if attributes is None:
        raise ValueError("Not a current ParticleMonitor TFS")
    initial_parts, sample_parts, selected_turn = [], [], -1
    for block in iter_particle_monitor_tfs(path, cancel=cancel):
        initial = block["record"] == 0
        if np.any(initial):
            initial_parts.append({name: values[initial] for name, values in block.items()})
        selected = block["record"] == 1
        if last_sample_only:
            if requested_turn is not None:
                selected &= block["turn"] <= requested_turn
            if not np.any(selected):
                continue
            turn = int(block["turn"][selected][-1])
            if turn > selected_turn:
                sample_parts = []
                selected_turn = turn
            selected &= block["turn"] == selected_turn
        if np.any(selected):
            sample_parts.append({name: values[selected] for name, values in block.items()})
    parts = initial_parts + sample_parts
    if parts:
        columns = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    else:
        with Path(path).open("rb") as stream:
            _, names, types, _ = _read_header(stream)
        columns = {name: np.empty(0, dtype=np.int64 if kind == "%d" else np.float64) for name, kind in zip(names, types)}
    return columns, attributes


def read_particle_trajectories(path, *, max_tag=None):
    """Return particle-ID keyed trajectories from one current PM file."""
    if max_tag is not None:
        max_tag = _integer(max_tag, "max_tag", 1)
    if isinstance(path, (list, tuple)):
        if len(path) != 1:
            raise ValueError("Select exactly one ParticleMonitor file")
        path = path[0]
    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as stream:
            if (stream.attrs.get("Name") != "PASS Particle Monitor" or stream.attrs.get("Layout") != "single_file"
                    or stream.attrs.get("FormatVersion") != 2):
                raise ValueError("ParticleMonitor requires the current single-file format version 2")
            count = _integer(stream.attrs.get("ValidSamples"), "ValidSamples")
            ids = stream["particle_id"][:]
            if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids <= 0) or np.any(ids > np.iinfo(np.int64).max)
                    or len(np.unique(ids)) != len(ids)):
                raise ValueError("ParticleMonitor requires unique positive integer particle IDs")
            if stream["turn"].ndim != 1 or len(stream["turn"]) < count:
                raise ValueError("ParticleMonitor turn data do not cover ValidSamples")
            turns = stream["turn"][:count]
            if turns.dtype.kind not in "iu" or np.any(turns < 0) or np.any(turns > np.iinfo(np.int64).max) or np.any(turns[1:] <= turns[:-1]):
                raise ValueError("ParticleMonitor sample turns must be strictly increasing nonnegative int64 integers")
            fields = {name: item for name, item in stream.items() if name not in {"particle_id", "turn"} and isinstance(item, h5py.Dataset)}
            if any(item.ndim != 2 or item.shape[0] < count or item.shape[1] != len(ids) for item in fields.values()):
                raise ValueError("ParticleMonitor datasets do not cover the sample/particle shape")
            selected = np.flatnonzero(ids <= max_tag) if max_tag is not None else np.arange(len(ids))
            trajectories = {int(ids[index]): {"turn": turns.copy()} for index in selected}
            if not len(selected):
                return trajectories
            contiguous = selected[-1] - selected[0] + 1 == len(selected)
            columns = slice(int(selected[0]), int(selected[-1]) + 1) if contiguous else selected
            for name, item in fields.items():
                # Read each selected hyperslab once instead of decoding the same
                # HDF5 chunks separately for every particle's column.
                values = item[:count, columns]
                if name == "tag":
                    selected_ids = ids[selected].astype(np.int64, copy=False)[None, :]
                    if values.dtype.kind not in "iu" or np.any((values != 0) & (values != selected_ids) & (values != -selected_ids)):
                        raise ValueError("ParticleMonitor sample tags disagree with particle IDs")
                for column, index in enumerate(selected):
                    trajectories[int(ids[index])][name] = values[:, column]
            return trajectories
    records, attributes = read_particle_monitor_tfs(path)
    selected = records["record"] == 1
    shape = (attributes["ValidSamples"], attributes["MaxTag"])
    columns = {name: values[selected].reshape(shape) for name, values in records.items() if name not in {"record", "particle_id"}}
    return {
        index + 1: {
            name: values[:, index]
            for name, values in columns.items()
        }
        for index in range(shape[1] if max_tag is None else min(shape[1], max_tag))
    }
