"""Single-file particle histories written from bounded monitor buffers."""

import json
from pathlib import Path

import h5py
import numpy as np

from PASS.utils.table_io import _hdf5_filters


def _update_sample_metadata(stream, count):
    """Derive summary attributes from a complete sample prefix."""
    start_turn = int(stream.attrs["StartTurn"])
    requested_end = int(stream.attrs["RequestedEndTurn"])
    end_turn = int(stream["turn"][count - 1]) + 1 if count else start_turn
    stream.attrs.update(NumTurn=count, EndTurn=end_turn, Completed=count == requested_end - start_turn and end_turn == requested_end)


def append_particle_samples(path, values, column_names, turns, initial, metadata, *, coordinate_dtype, chunk_turns, output_format, create=False):
    """Append one complete sample block; publish ValidSamples only after writing.

    ``values`` has axes (sample, particle, field), excluding no fields from the
    monitor buffer. Integer identity/loss fields are converted before storage.
    Initial coordinates come from Injection callbacks, never a monitor sample.
    """
    path = Path(path)
    n_samples, n_particles, _ = values.shape
    metadata = {name: value for name, value in metadata.items() if name not in {"Layout", "FormatVersion", "ValidSamples"}}
    filters = _hdf5_filters(output_format)
    with h5py.File(path, "x" if create else "a") as stream:
        if create:
            stream.attrs.update(metadata)
            stream.attrs.update(Layout="single_file", FormatVersion=2, ValidSamples=0)
            stream.create_dataset("particle_id", data=np.arange(1, n_particles + 1, dtype=np.int32))
            stream.create_dataset("turn", shape=(0, ), maxshape=(None, ), dtype=np.int64, chunks=(chunk_turns, ), **filters)
            _update_sample_metadata(stream, 0)
            for name in column_names[1:]:
                dtype = (np.int32 if name == "tag" else np.int64 if name == "lostTurn" else np.float32
                         if name == "lostPosition" else np.float64 if name == "zCenter" or name.startswith("reference") else coordinate_dtype)
                # Keep chunks near 256 KiB without spanning unnecessary particles.
                particle_chunk = max(1, min(n_particles, 262144 // (chunk_turns * np.dtype(dtype).itemsize)))
                dataset = stream.create_dataset(name,
                                                shape=(0, n_particles),
                                                maxshape=(None, n_particles),
                                                dtype=dtype,
                                                chunks=(chunk_turns, particle_chunk),
                                                **filters)
                dataset.attrs["Axes"] = "sample,particle"
            group = stream.create_group("initial")
            group.attrs["Event"] = "injection-after-reference-conversion"
            group.attrs["CoordinateDefinition"] = "z=beta*c*(T-t)"
            for name, data in initial.items():
                group.create_dataset(name, data=np.zeros_like(data) if name == "valid" else data)
        elif stream.attrs.get("Layout") != "single_file" or stream.attrs.get("FormatVersion") != 2:
            raise ValueError("ParticleMonitor append requires the current single-file format version 2")
        start = int(stream.attrs["ValidSamples"])
        end = start + n_samples
        try:
            if n_samples:
                stream["turn"].resize((end, ))
                stream["turn"][start:end] = turns
                for index, name in enumerate(column_names[1:], start=1):
                    dataset = stream[name]
                    dataset.resize((end, n_particles))
                    dataset[start:end] = values[:, :, index]
            for name, data in initial.items():
                if name != "valid":
                    stream["initial"][name][:] = data
            # Publish new injection records only after all their fields exist.
            stream["initial/valid"][:] = initial["valid"]
            stream.attrs.update(metadata)
            _update_sample_metadata(stream, end)
            stream.flush()
            stream.attrs["ValidSamples"] = end
            stream.flush()
        except BaseException as error:
            try:
                _update_sample_metadata(stream, int(stream.attrs["ValidSamples"]))
                stream.flush()
            except BaseException as repair_error:
                error.add_note(f"ParticleMonitor metadata repair also failed: {repair_error}; ValidSamples remains the commit marker")
            raise


def export_particle_samples_tfs(source, destination, *, chunk_rows=65536):
    """Convert committed HDF5 history into standard TFS using bounded rows.

    The caller publishes the completed destination and owns temporary cleanup.
    No headers or comments occur after the TFS column definition.
    """
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("TFS conversion chunk_rows must be a positive integer")
    field_names = ("x", "px", "y", "py", "z", "dp", "tag", "lostTurn", "lostPosition", "zCenter", "referenceTime", "referenceBeta",
                   "referenceMomentum")
    names = ("record", "turn", "particle_id") + field_names
    integer_names = {"record", "turn", "particle_id", "tag", "lostTurn"}
    formats = ["%.0f" if name in integer_names else "%.17g" for name in names]
    with h5py.File(source, "r") as history:
        if history.attrs.get("Layout") != "single_file" or history.attrs.get("FormatVersion") != 2:
            raise ValueError("TFS export requires ParticleMonitor format version 2")
        n_samples = int(history.attrs["ValidSamples"])
        n_particles = len(history["particle_id"])
        initial = history["initial"]
        n_initial = 0
        for start in range(0, n_particles, chunk_rows):
            n_initial += int(np.count_nonzero(initial["valid"][start:start + chunk_rows]))
        headers = dict(history.attrs)
        end_turn = int(history["turn"][n_samples - 1]) + 1 if n_samples else int(headers["StartTurn"])
        headers.update(NumTurn=n_samples,
                       EndTurn=end_turn,
                       ValidRows=n_initial + n_samples * n_particles,
                       Completed=n_samples == headers["RequestedEndTurn"] - headers["StartTurn"] and end_turn == headers["RequestedEndTurn"])
        with Path(destination).open("x", encoding="utf-8", newline="\n") as stream:
            for name, value in headers.items():
                if isinstance(value, (bool, int, np.bool_, np.integer)):
                    stream.write(f"@ {name} %d {int(value)}\n")
                elif isinstance(value, (float, np.floating)):
                    stream.write(f"@ {name} %le {float(value):.17g}\n")
                else:
                    stream.write(f"@ {name} %s {json.dumps(str(value), ensure_ascii=False)}\n")
            stream.write("* " + " ".join(names) + "\n")
            stream.write("$ " + " ".join("%d" if name in integer_names else "%le" for name in names) + "\n")
            for start in range(0, n_particles, chunk_rows):
                end = min(start + chunk_rows, n_particles)
                valid = initial["valid"][start:end]
                rows = np.full((int(np.count_nonzero(valid)), len(names)), np.nan, dtype=np.float64)
                rows[:, 0] = 0
                rows[:, 1] = initial["injection_turn"][start:end][valid]
                rows[:, 2] = initial["particle_id"][start:end][valid]
                for index, name in enumerate(field_names, start=3):
                    if name in initial:
                        rows[:, index] = initial[name][start:end][valid]
                rows[:, names.index("tag")] = rows[:, 2]
                rows[:, names.index("lostTurn")] = -1
                np.savetxt(stream, rows, fmt=formats)
            for sample in range(n_samples):
                turn = int(history["turn"][sample])
                for start in range(0, n_particles, chunk_rows):
                    end = min(start + chunk_rows, n_particles)
                    rows = np.full((end - start, len(names)), np.nan, dtype=np.float64)
                    rows[:, 0] = 1
                    rows[:, 1] = turn
                    rows[:, 2] = history["particle_id"][start:end]
                    for index, name in enumerate(field_names, start=3):
                        if name in history:
                            rows[:, index] = history[name][sample, start:end]
                    np.savetxt(stream, rows, fmt=formats)
            stream.flush()


def restore_particle_samples_tfs(source, destination, column_names, initial, *, coordinate_dtype, chunk_turns):
    """Rebuild a private HDF5 history when resuming a finalized TFS checkpoint."""
    from PASS.utils.particle_monitor_read import iter_particle_monitor_tfs, particle_monitor_tfs_metadata

    metadata = particle_monitor_tfs_metadata(source)
    n_samples, n_particles = metadata["ValidSamples"], metadata["MaxTag"]
    empty = np.empty((0, n_particles, len(column_names)), dtype=np.float64)
    append_particle_samples(destination,
                            empty,
                            column_names,
                            np.empty(0, dtype=np.int64),
                            initial,
                            metadata,
                            coordinate_dtype=coordinate_dtype,
                            chunk_turns=chunk_turns,
                            output_format="hdf5",
                            create=True)
    with h5py.File(destination, "a") as history:
        history["turn"].resize((n_samples, ))
        for name in column_names[1:]:
            history[name].resize((n_samples, n_particles))
        written_rows = 0
        for block in iter_particle_monitor_tfs(source):
            selected = block["record"] == 1
            values = {name: value[selected] for name, value in block.items()}
            start = 0
            while start < len(values["turn"]):
                sample, particle = divmod(written_rows, n_particles)
                count = min(n_particles - particle, len(values["turn"]) - start)
                history["turn"][sample] = values["turn"][start]
                for name in column_names[1:]:
                    history[name][sample, particle:particle + count] = values[name][start:start + count]
                start += count
                written_rows += count
        if written_rows != n_samples * n_particles:
            raise ValueError("TFS checkpoint sample count does not match its history")
        history.attrs.update(metadata)
        history.flush()
