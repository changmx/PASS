"""Automatic and offline plots from finalized monitor files."""

from collections import defaultdict
import hashlib
import json
import logging
from pathlib import Path
import re
from urllib.parse import quote

import h5py
import numpy as np
import pandas as pd
import tfs
from tfs.reader import read_headers

from PASS.plot.plot_distribution import expand_limits, plot_distribution, snapshot_coordinates
from PASS.plot.style import available_path, coordinate_spec
from PASS.utils.table_io import read_table

logger = logging.getLogger(__name__)


def _find_sources(directory):
    found = {}
    for suffix in (".csv", ".tfs", ".hdf5", ".h5"):
        for path in sorted(directory.glob("*" + suffix)):
            if path.is_file():
                found[path.stem] = path
    return list(found.values())


def _read_headers(path):
    if path.suffix in (".h5", ".hdf5"):
        with h5py.File(path, "r") as stream:
            return {key: value for key, value in stream.attrs.items() if not key.startswith("_pass_")}
    if path.suffix == ".tfs":
        return read_headers(path)
    return {}


def _read_frame(path):
    if path.suffix == ".csv":
        return tfs.TfsDataFrame(pd.read_csv(path), headers={})
    return read_table(path)


def _identity(path, headers):
    identity = {}
    for field, pattern in (("BeamId", r"_beam(\d+)"), ("BunchId", r"_bunch(\d+)")):
        match = re.search(pattern, path.stem)
        identity[field] = headers.get(field, int(match.group(1)) if match else "unknown")
    match = re.search(r"_cmd_(.+)_s_([^_]+)$", path.stem)
    identity["Monitor"] = headers.get("Monitor", match.group(1) if match else path.stem)
    identity["S"] = headers.get("S", match.group(2) if match else "unknown")
    identity["Turn"] = headers.get("Turn", "unknown")
    return identity


def _output_stem(directory, kind, path, headers):
    identity = _identity(path, headers)
    name = "beam" + quote(str(identity["BeamId"]), safe="._-")
    if kind != "particle":
        name += "_bunch" + quote(str(identity["BunchId"]), safe="._-")
    monitor = "injection" if identity["Turn"] == "Injection" else str(identity["Monitor"])
    # Bound the escaped component too: Unicode names expand to percent-encoded bytes.
    encoded_monitor = quote(monitor, safe="._-")
    if len(encoded_monitor) > 70:
        prefix = re.sub(r"%[0-9A-F]?$", "", encoded_monitor[:55])
        encoded_monitor = prefix + "_" + hashlib.sha256(monitor.encode()).hexdigest()[:10]
    name += "_" + encoded_monitor
    try:
        name += f"_s{float(identity['S']):.4f}"
    except (ValueError, TypeError):
        pass
    if kind == "distribution":
        name += "_turn_" + quote(str(identity["Turn"]), safe="._-")
    return directory / kind / name


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def plot_run(output_directory, *, save_dir=None, formats=("png", ), dpi=300, bins=80):
    """Render saved monitors and save a per-file manifest, preserving prior exports.

    Tracking state and Qt are not required. A bad monitor is recorded as failed
    without suppressing other plots. Distribution coordinate limits are shared
    by monitor; each six-panel figure determines its own count colour scale.
    """
    from PASS.plot.plot_particle_monitor import plot_particle_monitor
    from PASS.plot.plot_statistics import plot_statistics

    source = Path(output_directory).resolve()
    if not source.is_dir():
        raise ValueError(f"Run directory does not exist: {source}")
    if isinstance(formats, str):
        formats = (formats, )
    formats = tuple(dict.fromkeys(formats))
    if not formats or any(value not in ("png", "pdf") for value in formats):
        raise ValueError("Plot formats must be png and/or pdf")
    for name, value in (("dpi", dpi), ("bins", bins)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    destination = Path(save_dir).resolve() if save_dir is not None else source / "plot"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "source_directory": str(source),
        "output_directory": str(destination),
        "formats": list(formats),
        "dpi": int(dpi),
        "bins": int(bins),
        "entries": []
    }
    groups = defaultdict(list)

    def failed(entry, error):
        entry.update(status="failed", error=f"{type(error).__name__}: {error}")
        logger.warning("Plot failed for %s: %s", entry["source"], error)

    for folder, expected in ((source, "statistics"), (source / "distribution", "distribution"), (source / "particle", "particle")):
        for path in _find_sources(folder):
            entry = {"source": str(path), "type": expected}
            try:
                headers = _read_headers(path)
                names = {"statistics": "PASS Statistic Data", "distribution": "PASS Distribution Data", "particle": "PASS Particle Monitor"}
                if headers.get("Name") != names[expected]:
                    if not (expected == "statistics" and path.suffix == ".csv" and "_stat_" in path.stem):
                        continue
                manifest["entries"].append(entry)
                entry["metadata"] = headers
                stem = _output_stem(destination, expected, path, headers)
                if expected == "distribution":
                    identity = _identity(path, headers)
                    key = tuple(str(identity[field]) for field in ("BeamId", "BunchId", "Monitor", "S"))
                    groups[key].append((path, stem, entry))
                    continue
                logger.info("Plotting %s: %s", expected, path.name)
                if expected == "statistics":
                    result = plot_statistics(_read_frame(path), stem, formats=formats, dpi=dpi)
                else:
                    result = plot_particle_monitor(path, stem, formats=formats, dpi=dpi)
                entry["status"] = "completed"
                entry.update(result)
            except Exception as error:
                if not any(item is entry for item in manifest["entries"]):
                    manifest["entries"].append(entry)
                failed(entry, error)

    for members in groups.values():
        # Only one snapshot is resident at a time; all snapshots define shared limits.
        extrema = {name: [np.inf, -np.inf] for name in coordinate_spec()}
        valid_members = []
        for path, stem, entry in members:
            try:
                values, invalid_count = snapshot_coordinates(_read_frame(path))
                if not len(values["x"]):
                    entry.update(status="skipped", reason="No finite live particles", invalid_live_count=invalid_count)
                    continue
                for name, array in values.items():
                    extrema[name][0] = min(extrema[name][0], float(array.min()))
                    extrema[name][1] = max(extrema[name][1], float(array.max()))
                valid_members.append((path, stem, entry))
            except Exception as error:
                failed(entry, error)
        if not valid_members:
            continue
        limits = {name: expand_limits(*bounds) for name, bounds in extrema.items()}
        for path, stem, entry in valid_members:
            try:
                logger.info("Plotting distribution: %s", path.name)
                result = plot_distribution(_read_frame(path), stem, limits=limits, bins=bins, formats=formats, dpi=dpi)
                entry["status"] = "completed"
                entry.update(result)
            except Exception as error:
                failed(entry, error)

    manifest["summary"] = {status: sum(entry["status"] == status for entry in manifest["entries"]) for status in ("completed", "skipped", "failed")}
    path = available_path(destination / "manifest.json")
    manifest["manifest_path"] = str(path)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    logger.info("Plotting results: %s; manifest: %s", manifest["summary"], path)
    return manifest


def plot_main(sim):
    """Keep the existing executor hook; plot failures do not invalidate tracking."""
    try:
        return plot_run(sim.cfg.output_dir, save_dir=sim.cfg.output_dir_plot)
    except Exception:
        logger.exception("Post-processing failed; saved simulation data remain available")
        return {"summary": {"completed": 0, "skipped": 0, "failed": 1}}


def list_files(target_dir: str, recursive: bool = False) -> list[str]:
    """Retain the existing file-list helper for external callers."""
    root = Path(target_dir).resolve()
    if not root.is_dir():
        return []
    paths = root.rglob("*") if recursive else root.iterdir()
    return [str(path.resolve()) for path in paths if path.is_file()]
