"""Dynamic-aperture diagnostics from initial coordinates and observed particle states."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np


def _integer_values(value, name, *, minimum=None):
    """Accept exact numeric integers without truncating them."""
    values = np.asarray(value)
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain finite integers")
    if values.dtype.kind == "f":
        if (not np.all(np.isfinite(values)) or np.any(values != np.floor(values)) or np.any(values < -(2**63)) or np.any(values >= 2**63)):
            raise ValueError(f"{name} must contain finite int64 integers")
    elif values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds int64 capacity")
    values = values.astype(np.int64, copy=False)
    if minimum is not None and np.any(values < minimum):
        raise ValueError(f"{name} must be at least {minimum}")
    return values


def _integer_scalar(value, name, *, minimum=0):
    values = _integer_values(value, name, minimum=minimum)
    if values.ndim != 0:
        raise ValueError(f"{name} must be one integer")
    return int(values)


def compute_dynamic_aperture(initial_coordinates,
                             *,
                             particle_id=None,
                             tag,
                             coordinates=None,
                             sample_turn,
                             lost_turn=None,
                             lost_position=None,
                             injection_turn=None,
                             initial_valid=None,
                             requested_turn=None,
                             metadata=None):
    """Classify an x-y scan at an explicitly observed monitor event.

    Initial coordinates are ``(particles, 6)`` in x, px, y, py, z, dp order.
    Sample arrays are ``(samples, particles)``; coordinates append a six-axis.
    Positive/negative/zero tags denote live/lost/unavailable particles. IDs must
    already be aligned to the initial rows; signed tags are checked against them.
    ``requested_turn`` is an inclusive simulation turn at this monitor, not a
    number of completed revolutions. If omitted, the last supplied turn is used.

    Only the last sample at or before that turn is classified. A live particle
    without a sample on the requested turn is incomplete. Numerical validity is
    checked in that sample when coordinates are supplied; historical excursions
    are not inferred. Missing initial coordinates are unavailable. Loss times
    are absolute simulation turns, not particle ages. This does not add physics,
    track particles, fit an outer hull, or assume symmetry or a simply connected
    stable region. Initial dp groups remain unchanged by subsequent RF kicks.
    """
    if np.iscomplexobj(initial_coordinates):
        raise ValueError("initial_coordinates must contain real coordinates")
    initial = np.asarray(initial_coordinates, dtype=float)
    if initial.ndim != 2 or initial.shape[1] != 6:
        raise ValueError("initial_coordinates must have shape (particles, 6)")
    n_particles = len(initial)
    ids = np.arange(1, n_particles + 1, dtype=np.int64) if particle_id is None else np.asarray(particle_id)
    if ids.shape != (n_particles, ) or ids.dtype.kind not in "iu" or np.any(ids <= 0) or len(np.unique(ids)) != n_particles:
        raise ValueError("particle_id must contain one unique positive integer per initial row")
    ids = _integer_values(ids, "particle_id", minimum=1)
    turns = np.asarray(sample_turn)
    if turns.ndim != 1 or turns.dtype.kind not in "iu":
        raise ValueError("sample_turn must be strictly increasing nonnegative integers")
    turns = _integer_values(turns, "sample_turn", minimum=0)
    if np.any(np.diff(turns) <= 0):
        raise ValueError("sample_turn must be strictly increasing nonnegative integers")
    tags = _integer_values(tag, "tag")
    if tags.shape != (len(turns), n_particles):
        raise ValueError("tag must have shape (samples, particles)")
    if not np.all(np.isfinite(tags)) or not np.all((tags == 0) | (np.abs(tags) == ids[None, :])):
        raise ValueError("nonzero signed tags must match the aligned particle_id")
    if requested_turn is None:
        requested_turn = int(turns[-1]) if len(turns) else 0
    if isinstance(requested_turn, (bool, np.bool_)) or not isinstance(requested_turn, (int, np.integer)) or requested_turn < 0:
        raise ValueError("requested_turn must be a nonnegative integer")
    requested_turn = int(requested_turn)

    def particle_array(value, default, dtype, name):
        array = np.full(n_particles, default, dtype=dtype) if value is None else np.asarray(value, dtype=dtype)
        if array.shape != (n_particles, ):
            raise ValueError(f"{name} must have shape (particles,)")
        return array

    if initial_valid is not None:
        initial_valid = np.asarray(initial_valid)
        if initial_valid.dtype.kind not in "biuf" or np.any((initial_valid != 0) & (initial_valid != 1)):
            raise ValueError("initial_valid must contain booleans or numeric 0/1 values")
    valid = particle_array(initial_valid, True, bool, "initial_valid") & np.all(np.isfinite(initial), axis=1)
    born = particle_array(None if injection_turn is None else _integer_values(injection_turn, "injection_turn", minimum=-1), -1, np.int64,
                          "injection_turn")
    status = np.full(n_particles, "unavailable", dtype="<U12")
    observed = np.full(n_particles, -1, dtype=np.int64)
    loss = np.full(n_particles, -1, dtype=np.int64)
    position = np.full(n_particles, np.nan)
    last = np.full((n_particles, 6), np.nan)
    selected = np.searchsorted(turns, requested_turn, side="right") - 1

    def sample_array(value, name):
        if value is None:
            return None
        array = np.asarray(value)
        if array.shape != tags.shape:
            raise ValueError(f"{name} must have shape (samples, particles)")
        return array

    loss_samples = sample_array(lost_turn, "lost_turn")
    position_samples = sample_array(lost_position, "lost_position")
    if coordinates is not None:
        coordinates = np.asarray(coordinates)
        if np.iscomplexobj(coordinates):
            raise ValueError("coordinates must contain real coordinates")
        if coordinates.shape != tags.shape + (6, ):
            raise ValueError("coordinates must have shape (samples, particles, 6)")
    if selected >= 0:
        current = tags[selected]
        if np.any((current != 0) & (born > turns[selected])):
            raise ValueError("a recorded particle state cannot precede its injection turn")
        available = valid & (current != 0) & ((born < 0) | (born <= requested_turn))
        observed[available] = int(turns[selected])
        status[available] = "incomplete"
        live = available & (current > 0)
        if int(turns[selected]) == requested_turn:
            status[live] = "survived"
        dead = available & (current < 0)
        if loss_samples is not None:
            values = _integer_values(loss_samples[selected], "lost_turn", minimum=-1)
            known = dead & (values >= 0)
            loss[known] = values[known]
            if np.any(known & (loss > turns[selected])):
                raise ValueError("a recorded loss turn cannot be later than its sample")
            if np.any(known & (born >= 0) & (loss < born)):
                raise ValueError("a recorded loss turn cannot precede its injection turn")
        status[dead] = "lost"
        if position_samples is not None:
            position[dead] = position_samples[selected, dead]
        if coordinates is not None:
            last[:] = coordinates[selected]
            finite = np.all(np.isfinite(last), axis=1)
            # px and py are mechanical momenta normalized to reference momentum.
            with np.errstate(over="ignore", invalid="ignore"):
                momentum_ratio = 1.0 + last[:, 5]
                longitudinal_squared = momentum_ratio**2 - last[:, 1]**2 - last[:, 3]**2
            physical = finite & (momentum_ratio > 0) & np.isfinite(longitudinal_squared) & (longitudinal_squared > 0)
            status[available & ~physical] = "invalid"
    result_metadata = dict(metadata or {})
    result_metadata.update({
        "requested_turn": requested_turn,
        "sample_event": "at the selected monitor; turn labels do not establish completed revolutions",
        "coordinate_order": ["x", "px", "y", "py", "z", "dp"],
        "dp_grouping": "initial_coordinates[:, 5]",
        "validity_scope": "selected sample only" if coordinates is not None else "coordinates not checked",
    })
    return {
        "particle_id": ids,
        "initial_coordinates": initial.copy(),
        "initial_valid": valid,
        "injection_turn": born,
        "status": status,
        "observed_turn": observed,
        "lost_turn": loss,
        "lost_position": position,
        "final_coordinates": last,
        "dp_values": np.unique(initial[valid, 5]),
        "metadata": result_metadata,
    }


def _attributes(stream):
    return {
        key: value.decode("utf-8") if isinstance(value, bytes) else value.item() if isinstance(value, np.generic) else value
        for key, value in stream.attrs.items()
    }


def _check_cancel(cancel):
    if cancel is not None and cancel():
        raise InterruptedError("Dynamic-aperture reading cancelled")


def _read_single_file_tfs(path, requested_turn, cancel, clamp_to_available):
    from PASS.utils.particle_monitor_read import read_particle_monitor_tfs

    columns, attrs = read_particle_monitor_tfs(path, requested_turn=requested_turn, last_sample_only=True, cancel=cancel)
    ids = np.arange(1, attrs["MaxTag"] + 1, dtype=np.int64)
    initial = np.full((len(ids), 6), np.nan)
    born = np.full(len(ids), -1, dtype=np.int64)
    valid = np.zeros(len(ids), dtype=bool)
    initial_rows = columns["record"] == 0
    initial_ids = columns["particle_id"][initial_rows]
    if np.any(columns["tag"][initial_rows] != initial_ids):
        raise ValueError("Initial ParticleMonitor records require their positive particle tag")
    coordinate_names = ("x", "px", "y", "py", "z", "dp")
    initial[initial_ids - 1] = np.column_stack([columns[name][initial_rows] for name in coordinate_names])
    born[initial_ids - 1] = columns["turn"][initial_rows]
    valid[initial_ids - 1] = True
    sample_rows = columns["record"] == 1
    turns = np.unique(columns["turn"][sample_rows])
    # The TFS reader validates every row even when it returns only an earlier sample.
    last_sample_turn = int(attrs["EndTurn"]) - 1 if attrs["ValidSamples"] > 0 else None
    target = _resolve_turn(requested_turn, attrs, turns, clamp_to_available=clamp_to_available, last_sample_turn=last_sample_turn)
    coordinates = np.column_stack([columns[name][sample_rows] for name in coordinate_names]).reshape(len(turns), len(ids), 6)
    return compute_dynamic_aperture(initial,
                                    particle_id=ids,
                                    tag=columns["tag"][sample_rows].reshape(len(turns), len(ids)),
                                    coordinates=coordinates,
                                    sample_turn=turns,
                                    lost_turn=columns["lostTurn"][sample_rows].reshape(len(turns), len(ids)),
                                    lost_position=columns["lostPosition"][sample_rows].reshape(len(turns), len(ids)),
                                    injection_turn=born,
                                    initial_valid=valid,
                                    requested_turn=target,
                                    metadata={
                                        **attrs, "sources": [str(path)],
                                        "initial_source": "injection capture",
                                        "last_sample_turn": last_sample_turn,
                                        "requested_turn_input": requested_turn
                                    })


def read_dynamic_aperture(path, *, requested_turn=None, cancel=None, clamp_to_available=False):
    """Read one current ParticleMonitor file without loading its full history.

    Injection coordinates are taken from the monitor's initial records. The
    default horizon is RequestedEndTurn-1; requested_turn is an inclusive
    simulation turn at the monitor location. With ``clamp_to_available=True``,
    an explicit turn beyond the last committed sample is reduced to that sample's
    turn. Automatic horizons and unsampled gaps are never clamped; they retain
    incomplete-state detection. Empty histories remain unavailable. Metadata
    reports ``requested_turn_input``, the effective ``requested_turn``, and
    ``last_sample_turn`` (None for an empty history).
    """
    import h5py

    path = Path(path)
    if not path.is_file():
        raise ValueError("Select one ParticleMonitor file")
    _check_cancel(cancel)
    if not isinstance(clamp_to_available, (bool, np.bool_)):
        raise ValueError("clamp_to_available must be a boolean")
    if requested_turn is not None:
        requested_turn = _resolve_turn(requested_turn, {}, np.empty(0, dtype=np.int64))
    if path.suffix.lower() == ".tfs":
        return _read_single_file_tfs(path, requested_turn, cancel, clamp_to_available)
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("ParticleMonitor analysis supports HDF5 or TFS")
    with h5py.File(path, "r") as stream:
        attrs = _attributes(stream)
        if attrs.get("Name") != "PASS Particle Monitor" or attrs.get("Layout") != "single_file" or attrs.get("FormatVersion") != 2:
            raise ValueError("ParticleMonitor requires the current single-file format version 2")
        if "initial" not in stream:
            raise ValueError("ParticleMonitor has no injection initial coordinates")
        ids = stream["particle_id"][:]
        group = stream["initial"]
        if not np.array_equal(ids, group["particle_id"][:]):
            raise ValueError("ParticleMonitor initial IDs do not match its trajectory columns")
        initial = np.column_stack([group[name][:] for name in ("x", "px", "y", "py", "z", "dp")])
        born = group["injection_turn"][:]
        valid = group["valid"][:]
        count = _integer_scalar(attrs["ValidSamples"], "ValidSamples")
        if stream["turn"].ndim != 1 or count > len(stream["turn"]):
            raise ValueError("ParticleMonitor turn data must be one-dimensional and cover ValidSamples")
        for name in ("x", "px", "y", "py", "z", "dp", "tag", "lostTurn", "lostPosition"):
            dataset = stream[name]
            if dataset.ndim != 2 or dataset.shape[0] < count or dataset.shape[1] != len(ids):
                raise ValueError(f"ParticleMonitor {name} does not cover ValidSamples and particle IDs")
        turns = _integer_values(stream["turn"][:count], "sample turn", minimum=0)
        if np.any(np.diff(turns) <= 0):
            raise ValueError("Recorded sample turns must be strictly increasing")
        last_sample_turn = int(turns[-1]) if len(turns) else None
        target = _resolve_turn(requested_turn, attrs, turns, clamp_to_available=clamp_to_available, last_sample_turn=last_sample_turn)
        index = int(np.searchsorted(turns, target, side="right") - 1)
        rows = slice(index, index + 1) if index >= 0 else slice(0, 0)
        coordinates = np.stack([stream[name][rows] for name in ("x", "px", "y", "py", "z", "dp")], axis=-1)
        result = compute_dynamic_aperture(initial,
                                          particle_id=ids,
                                          tag=stream["tag"][rows],
                                          coordinates=coordinates,
                                          sample_turn=turns[rows],
                                          lost_turn=stream["lostTurn"][rows],
                                          lost_position=stream["lostPosition"][rows],
                                          injection_turn=born,
                                          initial_valid=valid,
                                          requested_turn=target,
                                          metadata={
                                              **attrs, "sources": [str(path)],
                                              "initial_source": "injection capture",
                                              "last_sample_turn": last_sample_turn,
                                              "requested_turn_input": requested_turn
                                          })
    _check_cancel(cancel)
    return result


def _resolve_turn(requested_turn, attrs, turns, *, clamp_to_available=False, last_sample_turn=None):
    if requested_turn is not None:
        if isinstance(requested_turn, bool) or not isinstance(requested_turn, (int, np.integer)) or requested_turn < 0:
            raise ValueError("requested_turn must be a nonnegative integer")
        return min(int(requested_turn), last_sample_turn) if clamp_to_available and last_sample_turn is not None else int(requested_turn)
    end = attrs.get("RequestedEndTurn", attrs.get("EndTurn"))
    if end is not None:
        end = _integer_scalar(end, "monitor EndTurn")
        if end > 0:
            return end - 1
    return _integer_scalar(turns[-1], "sample turn") if len(turns) else 0


def export_dynamic_aperture(result, path):
    """Stage numeric exports before publication; restore prior files on failure."""
    path = Path(path)
    if path.suffix.lower() not in {".npz", ".csv"}:
        raise ValueError("DA exports require .npz or .csv")
    metadata = json.dumps(result["metadata"],
                          ensure_ascii=False,
                          indent=2,
                          default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value))
    targets = [path] + ([path.with_suffix(".json")] if path.suffix.lower() == ".csv" else [])
    for target in targets:
        if target.exists() and not target.is_file():
            raise ValueError(f"Export destination is not a file: {target}")
    directory = Path(tempfile.mkdtemp(prefix=".pass-da-", dir=path.parent))
    staged = directory / path.name
    keep_recovery = False
    try:
        if path.suffix.lower() == ".npz":
            # A file handle avoids NumPy appending .npz to an uppercase .NPZ path.
            with staged.open("wb") as stream:
                np.savez(stream, **{key: value for key, value in result.items() if key != "metadata"}, metadata_json=np.array(metadata))
        else:
            names = ("particle_id", "x0", "px0", "y0", "py0", "z0", "dp0", "initial_valid", "injection_turn", "status", "observed_turn", "lost_turn",
                     "lost_position")
            with staged.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(names)
                for index, particle_id in enumerate(result["particle_id"]):
                    writer.writerow([
                        particle_id, *result["initial_coordinates"][index], result["initial_valid"][index], result["injection_turn"][index],
                        result["status"][index], result["observed_turn"][index], result["lost_turn"][index], result["lost_position"][index]
                    ])
            staged.with_suffix(".json").write_text(metadata, encoding="utf-8")
        backups, published = {}, []
        keep_recovery = True
        try:
            for index, target in enumerate(targets):
                if target.exists() or target.is_symlink():
                    backup = directory / f"previous-{index}"
                    # Register intent first: rename can succeed before an
                    # interruption is delivered to the caller.
                    backups[target] = backup
                    target.replace(backup)
                published.append(target)
                (directory / target.name).replace(target)
        except BaseException:
            # Keep old data recoverable even if restoration itself is blocked.
            recovery_failed = False
            for target in reversed(targets):
                try:
                    if target in backups:
                        if backups[target].exists() or backups[target].is_symlink():
                            backups[target].replace(target)
                    elif target in published:
                        target.unlink(missing_ok=True)
                except OSError:
                    recovery_failed = True
            if recovery_failed:
                raise OSError(f"Export publication and recovery failed; recovery files retained in {directory}")
            keep_recovery = False
            raise
        keep_recovery = False
    finally:
        if not keep_recovery:
            shutil.rmtree(directory)
    return path
