"""Grouped electron-cloud diagnostics with physical SI fields and number weights."""

from copy import deepcopy
from dataclasses import dataclass
import json

import numpy as np


@dataclass
class ElectronCloudResult:
    grid: dict
    fields: dict
    source: dict
    history: dict
    metadata: dict

    @property
    def columns(self):
        return {}

    @property
    def nbytes(self):
        return sum(value.nbytes for group in (self.grid, self.fields, self.source, self.history) for value in group.values())

    @property
    def views(self):
        return [name for name, values in (("fields", self.fields), ("source", self.source), ("history", self.history)) if values]

    def unit(self, group, name):
        return self.metadata.get("dataset_definitions", {}).get(group, {}).get(name, {}).get("units", "")

    def freeze(self):
        groups = []
        for group in (self.grid, self.fields, self.source, self.history):
            arrays = {}
            for name, value in group.items():
                arrays[name] = np.asarray(value).view()
                arrays[name].flags.writeable = False
            groups.append(arrays)
        return ElectronCloudResult(*groups, deepcopy(self.metadata))


def _json_attribute(group, name, default):
    try:
        return json.loads(group.attrs[name]) if name in group.attrs else default
    except (ValueError, TypeError) as exc:
        raise ValueError(f"电子云 {name} 不是有效的 JSON。") from exc


def _history_unit(name):
    if name in {"time_start", "time_end"}:
        return "s"
    if name == "line_charge":
        return "C/m"
    if name == "max_electric_field":
        return "V/m"
    if name in {
            "max_kick", "n_alive", "bunch_id", "slice_index", "steps", "n_real_particles", "primary_electrons", "n_electrons_before",
            "n_electrons_after", "incident_electrons", "emitted_electrons"
    }:
        return "1"
    return ""


def read_electron_cloud_result(stream, *, cancelled=lambda: False, progress=lambda message: None):
    """Read only diagnostic snapshot formats, preserving per-dataset units."""
    metadata = _json_attribute(stream, "metadata_json", {})
    if not isinstance(metadata, dict):
        raise ValueError("电子云 metadata_json 必须为对象。")
    metadata["format"] = str(stream.attrs["format"])
    definitions = metadata["dataset_definitions"] = {}
    groups = {}
    for group_name in ("grid", "fields", "source"):
        arrays, definitions[group_name] = {}, {}
        if group_name in stream:
            group = stream[group_name]
            for name, dataset in group.items():
                if cancelled():
                    raise InterruptedError("读取已取消")
                progress(f"读取电子云 {group_name}/{name}")
                if not hasattr(dataset, "dtype") or dataset.dtype.kind not in "biuf":
                    raise ValueError(f"电子云 {group_name}/{name} 必须是数值数据集。")
                arrays[name] = np.asarray(dataset)
                unit = dataset.attrs.get("unit", "")
                definitions[group_name][name] = {"units": unit.decode("utf-8") if isinstance(unit, bytes) else str(unit)}
            if group_name == "source":
                source_metadata = _json_attribute(group, "metadata_json", {})
                if not isinstance(source_metadata, dict):
                    raise ValueError("电子云源 metadata_json 必须为对象。")
                metadata["source_metadata"] = source_metadata
            if group_name == "fields":
                metadata["field_metadata"] = dict(group.attrs)
        groups[group_name] = arrays
    grid, fields, source = (groups[name] for name in ("grid", "fields", "source"))
    if fields:
        for name in ("x", "y"):
            axis = grid.get(name, np.empty(0))
            if axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
                raise ValueError(f"电子云 grid/{name} 必须为有限且严格递增的一维网格。")
        for name, values in fields.items():
            if values.shape != (len(grid["y"]), len(grid["x"])):
                raise ValueError(f"电子云 fields/{name} 必须按二维 (y, x) 保存。")
    if source:
        if not {"x", "y", "weight"} <= set(source):
            raise ValueError("电子云源缺少 x、y 或 weight。")
        if any(values.ndim != 1 for values in source.values()) or len({len(values) for values in source.values()}) != 1:
            raise ValueError("电子云源数组必须一维且等长。")
        if any(not np.isfinite(values).all() for values in source.values()) or np.any(source["weight"] < 0):
            raise ValueError("电子云源坐标、动量和权重必须有限，电子数权重不得为负。")
    rows = _json_attribute(stream, "history_json", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("电子云 history_json 必须为对象列表。")
    history, definitions["history"] = {}, {}
    names = dict.fromkeys(name for row in rows for name in row)
    for name in names:
        if cancelled():
            raise InterruptedError("读取已取消")
        progress(f"读取电子云历史 {name}")
        values = [row.get(name) for row in rows]
        if all(value is None or isinstance(value, (int, float)) for value in values):
            history[name] = np.asarray([np.nan if value is None else value for value in values], dtype=float)
        elif all(value is None or isinstance(value, str) for value in values):
            history[name] = np.asarray(["" if value is None else value for value in values], dtype=str)
        else:
            raise ValueError(f"电子云历史 {name} 必须为数值或文本标量列。")
        definitions["history"][name] = {"units": _history_unit(name)}
    if history and ("time_end" not in history or history["time_end"].dtype.kind not in "biuf" or not np.isfinite(history["time_end"]).all()):
        raise ValueError("电子云历史必须包含有限的物理 time_end。")
    if not any((fields, source, history)):
        raise ValueError("电子云诊断没有可显示的场、源或历史数据。")
    if cancelled():
        raise InterruptedError("读取已取消")
    return ElectronCloudResult(grid, fields, source, history, metadata)


def electron_cloud_export_columns(result, view):
    """Export complete selected groups; source weights and event strings survive."""
    definitions = deepcopy(result.metadata["dataset_definitions"].get(view, {}))
    if view == "fields":
        x, y = np.meshgrid(result.grid["x"], result.grid["y"])
        columns = {"x": x.ravel(), "y": y.ravel(), **{name: value.ravel() for name, value in result.fields.items()}}
        definitions.update(deepcopy(result.metadata["dataset_definitions"]["grid"]))
    elif view in {"source", "history"}:
        columns = getattr(result, view)
    else:
        raise ValueError("请选择电子云场、源或历史视图。")
    return columns, definitions


def electron_cloud_figure_spec(result, view, quantity="", x_name="x", y_name="y", bins=80):
    if view == "fields":
        # Export workers transfer the chosen map without unrelated particle arrays.
        selected = ElectronCloudResult(result.grid, {quantity: result.fields[quantity]}, {}, {}, result.metadata)
        return "electron_cloud_field", {"result": selected, "field": quantity}
    if view == "source":

        def label(name):
            return f"{name} [{result.unit('source', name)}]" if result.unit("source", name) else name

        return "density", {
            "x": result.source[x_name],
            "y": result.source[y_name],
            "weights": result.source["weight"],
            "bins": bins,
            "x_label": label(x_name),
            "y_label": label(y_name),
            "weight_label": "Electrons / bin (saved source length)",
            "title": f"Source length = {result.metadata.get('source_metadata', {}).get('source_length', '?')} m"
        }
    if view == "history":
        unit = result.unit("history", quantity)
        return "series", {
            "series": [(quantity, result.history["time_end"], result.history[quantity], "line")],
            "x_label": "Physical interval-end time [s]",
            "y_label": f"{quantity} [{unit}]" if unit else quantity,
            "title": f"Source length = {result.metadata.get('source_metadata', {}).get('source_length', '?')} m"
        }
    raise ValueError("请选择电子云场、源或历史视图。")
