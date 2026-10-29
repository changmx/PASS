"""Plot every committed live particle sample in a twelve-panel monitor figure."""

from pathlib import Path

import h5py
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy as np

from PASS.plot.style import coordinate_spec, create_figure, plotting_style, save_figure, style_axes
from PASS.utils.particle_monitor_read import read_particle_trajectories


def _iter_particle_blocks(path):
    """Yield bounded HDF5 reads, or validated TFS particle histories."""
    names = ("x", "px", "y", "py", "z", "dp")
    path = Path(path)
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        trajectories = read_particle_trajectories(path)
        for particle_id, history in trajectories.items():
            values = {name: np.asarray(history[name])[:, None] for name in names}
            yield np.array([particle_id]), np.asarray(history["turn"]), values, np.asarray(history["tag"])[:, None]
        return

    with h5py.File(path, "r") as stream:
        if (stream.attrs.get("Name") != "PASS Particle Monitor" or stream.attrs.get("Layout") != "single_file"
                or stream.attrs.get("FormatVersion") != 2):
            raise ValueError("ParticleMonitor plotting requires single-file format version 2")
        n_samples = stream.attrs.get("ValidSamples")
        if isinstance(n_samples, (bool, np.bool_)) or not isinstance(n_samples, (int, np.integer)) or n_samples < 0:
            raise ValueError("ParticleMonitor ValidSamples must be a nonnegative integer")
        ids = stream["particle_id"][:]
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids <= 0) or np.any(ids > np.iinfo(np.int64).max)
                or len(np.unique(ids)) != len(ids)):
            raise ValueError("ParticleMonitor requires unique positive integer particle IDs")
        if stream["turn"].ndim != 1 or len(stream["turn"]) < n_samples:
            raise ValueError("ParticleMonitor turn data do not cover ValidSamples")
        turns = stream["turn"][:n_samples]
        if (turns.dtype.kind not in "iu" or np.any(turns < 0) or np.any(turns > np.iinfo(np.int64).max) or np.any(turns[1:] <= turns[:-1])):
            raise ValueError("ParticleMonitor sample turns must be strictly increasing nonnegative integers")
        for name in (*names, "tag"):
            dataset = stream[name]
            if dataset.ndim != 2 or dataset.shape[0] < n_samples or dataset.shape[1] != len(ids):
                raise ValueError(f"ParticleMonitor {name} does not cover the sample/particle shape")
        # Limit coordinate reads to about 8 MiB per field where possible. A
        # single full trajectory is the smallest block to preserve its gaps.
        block_particles = max(1, min(64, 1048576 // max(1, int(n_samples))))
        for start in range(0, len(ids), block_particles):
            end = min(start + block_particles, len(ids))
            selected_ids = ids[start:end].astype(np.int64, copy=False)
            tags = stream["tag"][:n_samples, start:end]
            if tags.dtype.kind not in "iu" or np.any((tags != 0) & (tags != selected_ids[None, :]) & (tags != -selected_ids[None, :])):
                raise ValueError("ParticleMonitor sample tags disagree with particle IDs")
            values = {name: stream[name][:n_samples, start:end] for name in names}
            yield selected_ids, turns, values, tags


def plot_particle_monitor(path, output_stem, *, formats=("png", ), dpi=300):
    """Save six time traces and six projections using all saved particle IDs.

    Only committed monitor samples are plotted. Injection records are separate
    events and are not inserted into trajectories at the monitor position.
    Nonpositive tags and samples with nonfinite coordinates are excluded;
    missing samples break time traces instead of joining their neighbours.
    """
    spec = coordinate_spec()
    fields = ("x", "y", "z", "px", "py", "dp")
    pairs = (("x", "px"), ("y", "py"), ("z", "dp"), ("x", "y"), ("z", "x"), ("z", "y"))
    particle_count, sample_count, valid_sample_count, plotted_particle_count = 0, 0, 0, 0
    first_turn, last_turn = None, None
    with mpl.rc_context(plotting_style()):
        figure, axes = create_figure(4, 3, (15, 15))
        axes = np.asarray(axes).reshape(4, 3)
        try:
            for ids, turns, values, tags in _iter_particle_blocks(path):
                particle_count += len(ids)
                sample_count = len(turns)
                if not sample_count:
                    continue
                first_turn, last_turn = int(turns[0]), int(turns[-1])
                valid = tags > 0
                for name in fields:
                    valid &= np.isfinite(values[name])
                valid_sample_count += int(np.count_nonzero(valid))
                plotted_particle_count += int(np.count_nonzero(np.any(valid, axis=0)))
                if not np.any(valid):
                    continue
                scaled = {name: np.where(valid, values[name] * spec[name][0], np.nan) for name in fields}
                isolated = valid.copy()
                isolated[1:] &= ~valid[:-1]
                isolated[:-1] &= ~valid[1:]
                for index, name in enumerate(fields):
                    axis = axes.flat[index]
                    # Each column is an independent particle. NaNs preserve
                    # pending/lost gaps; rasterization keeps PDF sizes bounded.
                    lines = axis.plot(turns, scaled[name], color="#0072B2", linewidth=0.35, alpha=0.18, rasterized=True)
                    for column, line in enumerate(lines):
                        if np.any(isolated[:, column]):
                            line.set_marker(".")
                            line.set_markersize(3)
                            line.set_markevery(np.flatnonzero(isolated[:, column]).tolist())
                for index, (horizontal, vertical) in enumerate(pairs, start=6):
                    axes.flat[index].scatter(scaled[horizontal][valid],
                                             scaled[vertical][valid],
                                             color="#0072B2",
                                             marker=".",
                                             s=0.35,
                                             alpha=0.15,
                                             linewidths=0,
                                             rasterized=True)

            metadata = {
                "particle_count": particle_count,
                "plotted_particle_count": plotted_particle_count,
                "sample_count": sample_count,
                "valid_sample_count": valid_sample_count,
                "turn_range": [first_turn, last_turn],
                "selection": "All saved IDs and committed monitor samples; tag > 0 and all six coordinates finite",
                "downsampled": False,
            }
            if not valid_sample_count:
                return {"status": "skipped", "reason": "No finite live particle samples", **metadata}
            for index, name in enumerate(fields):
                axis = axes.flat[index]
                axis.set_xlabel("Turn")
                axis.set_ylabel(spec[name][1])
                if first_turn != last_turn:
                    axis.set_xlim(first_turn, last_turn)
                style_axes(axis)
                axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            for index, (horizontal, vertical) in enumerate(pairs, start=6):
                axis = axes.flat[index]
                axis.set_xlabel(spec[horizontal][1])
                axis.set_ylabel(spec[vertical][1])
                style_axes(axis)
            return {"status": "completed", "files": save_figure(figure, output_stem, formats=formats, dpi=dpi), **metadata}
        finally:
            figure.clear()
