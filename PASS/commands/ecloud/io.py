"""SI field diagnostics and portable frozen/dynamic electron-cloud files."""

import json
from pathlib import Path

import h5py
import numpy as np

from PASS.utils.constants import const


def _host(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


def _source_array_names(format_name):
    if format_name == "PASS-frozen-electrons-1":
        return ("x", "y", "weight")
    if format_name == "PASS-dynamic-electrons-1":
        return ("x", "y", "ux", "uy", "uz", "weight")
    raise ValueError("Unsupported ElectronCloud source state format")


def _write_source(group, state):
    metadata = dict(state)
    for name in _source_array_names(state.get("format")):
        values = np.asarray(metadata.pop(name), dtype=np.float64)
        data = group.create_dataset(name, data=values, compression="gzip", compression_opts=1)
        data.attrs["unit"] = "m" if name in {"x", "y"} else "1"
    group.attrs["metadata_json"] = json.dumps(metadata, allow_nan=False, sort_keys=True)


def _read_source(group):
    state = json.loads(group.attrs["metadata_json"])
    if not isinstance(state, dict):
        raise ValueError("ElectronCloud source metadata must be an object")
    array_names = _source_array_names(state.get("format"))
    if set(group.keys()) != set(array_names) or any(name in state for name in array_names):
        raise ValueError("ElectronCloud source array datasets do not match its state format")
    for name in array_names:
        values = np.asarray(group[name], dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"ElectronCloud source {name} must be one-dimensional")
        state[name] = values.tolist()
    return state


def write_field_snapshot(path, fields, metadata):
    """Create a new field file; a snapshot never replaces an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = fields.snapshot()
    with h5py.File(path, "x") as output:
        output.attrs["format"] = "PASS-electron-cloud-fields-1"
        output.attrs["metadata_json"] = json.dumps(metadata, allow_nan=False, sort_keys=True)
        grid = output.create_group("grid")
        for name in ("x", "y"):
            grid.create_dataset(name, data=_host(snapshot[name])).attrs["unit"] = "m"
        group = output.create_group("fields")
        for name, unit in (("charge_density", "C/m^3"), ("electron_density", "1/m^3"), ("potential", "V"), ("ex", "V/m"), ("ey", "V/m")):
            group.create_dataset(name, data=_host(snapshot[name]), compression="gzip", compression_opts=1).attrs["unit"] = unit
        if fields.state is not None:
            _write_source(output.create_group("source"), fields.state.state_dict())
    return path


def write_cloud_state(path, state):
    """Save a cloud source, independently of complete beam/turn checkpointing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as output:
        output.attrs["format"] = state["format"]
        output.attrs["configuration_sha256"] = state["configuration_sha256"]
        output.attrs["has_cloud"] = state["cloud"] is not None
        if state["cloud"] is not None:
            _write_source(output.create_group("source"), state["cloud"])
    return path


def read_cloud_state(path):
    with h5py.File(path, "r") as source:
        if source.attrs.get("format") != "PASS-electron-cloud-1":
            raise ValueError("Unsupported ElectronCloud source file format")
        if "has_cloud" not in source.attrs or "configuration_sha256" not in source.attrs:
            raise ValueError("ElectronCloud source file is missing its configuration identity")
        has_cloud = source.attrs["has_cloud"]
        if not isinstance(has_cloud, (bool, np.bool_)) or bool(has_cloud) != ("source" in source):
            raise ValueError("ElectronCloud source file has inconsistent source presence")
        return {
            "format": source.attrs["format"],
            "configuration_sha256": source.attrs["configuration_sha256"],
            "cloud": _read_source(source["source"]) if has_cloud else None,
        }


def write_buildup_snapshot(path, cloud, metadata, history):
    """Save dynamic sources/history and, for coupled mode, the final cloud field."""
    state = getattr(cloud, "state", cloud).state_dict()
    if state.get("format") != "PASS-dynamic-electrons-1":
        raise ValueError("ElectronCloud build-up snapshots require a dynamic electron state")
    if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
        raise ValueError("ElectronCloud build-up history must be a list of diagnostic objects")
    metadata_json = json.dumps(metadata, allow_nan=False, sort_keys=True)
    history_json = json.dumps(history, allow_nan=False, sort_keys=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x", libver="latest", track_order=True) as output:
        output.attrs["format"] = "PASS-electron-cloud-buildup-1"
        output.attrs["metadata_json"] = metadata_json
        output.attrs["history_json"] = history_json
        _write_source(output.create_group("source"), state)
        pic = getattr(cloud, "pic", None)
        if pic is not None:
            result = pic.solve(cloud.state.x, cloud.state.y, -const.e * cloud.state.weight / cloud.state.source_length)
            grid = output.create_group("grid")
            for name in ("x", "y"):
                grid.create_dataset(name, data=_host(getattr(pic.geometry, name))).attrs["unit"] = "m"
            group = output.create_group("fields")
            group.attrs["source"] = "electrons at the final observation time"
            for name, values, unit in (("charge_density", result.density, "C/m^3"), ("electron_density", -result.density / const.e, "1/m^3"),
                                       ("potential", result.potential, "V"), ("ex", result.ex, "V/m"), ("ey", result.ey, "V/m")):
                group.create_dataset(name, data=_host(values), compression="gzip", compression_opts=1).attrs["unit"] = unit
    return path
