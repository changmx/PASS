"""Full-resolution result analysis and explicit SpaceCharge field semantics."""

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np


@dataclass
class FieldResult:
    x: np.ndarray
    y: np.ndarray
    slice_ids: np.ndarray
    delta_z: np.ndarray
    fields: dict
    metadata: dict

    @property
    def columns(self):
        return {}

    @property
    def nbytes(self):
        return sum(value.nbytes for value in (self.x, self.y, self.slice_ids, self.delta_z, *self.fields.values()))

    def plane(self, name, index, average=False):
        units = {
            "charge_density": ("density_units", "C/m^2", "C/m^3"),
            "potential": ("potential_units", "V m", "V"),
            "integrated_Ex": ("integrated_field_units", "V", "V/m"),
            "integrated_Ey": ("integrated_field_units", "V", "V/m")
        }
        key, default, averaged = units[name]
        values = self.fields[name][index]
        unit = str(self.metadata.get(key, default))
        if average:
            width = float(self.delta_z[index])
            if not np.isfinite(width) or width <= 0:
                raise ValueError("Δz 必须为正有限值，才能计算切片平均场。")
            if unit != default:
                raise ValueError(f"无法自动转换未知单位 {unit!r}；请使用原始积分量。")
            return values / width, averaged
        return values, unit


@dataclass(frozen=True)
class ExportSelection:
    """Widget-free snapshot; arrays are borrowed read-only until the task ends."""
    source: str
    result: object
    x_name: str = ""
    y_name: str = ""
    status: str = "all"
    batch: int | None = None
    mode: str = "auto"
    comparison: str = "single"
    baseline_path: str | None = None
    baseline: object = None
    bins: int = 80
    field: str = ""
    index: int | None = None
    average: bool = False
    aperture: bool = True


def freeze_result(result):
    """Copy metadata and array views, never a table-sized buffer on the GUI thread."""
    from PASS.gui.results import ResultTable

    if result is None:
        return None

    def readonly(values):
        view = np.asarray(values).view()
        view.flags.writeable = False
        return view

    metadata = deepcopy(result.metadata)
    if isinstance(result, FieldResult):
        return FieldResult(*(readonly(value) for value in (result.x, result.y, result.slice_ids, result.delta_z)), {
            name: readonly(value)
            for name, value in result.fields.items()
        }, metadata)
    # The new table owns its filter cache; the worker never mutates the GUI's cache.
    return ResultTable({name: readonly(value) for name, value in result.columns.items()}, metadata)


def read_plot_result(path, *, cancelled=lambda: False, progress=lambda message: None):
    """Route multidimensional SC data separately from aligned particle tables."""
    from PASS.gui.results import read_result

    path = Path(path)
    progress(f"读取 {path.name}")
    if cancelled():
        raise InterruptedError("读取已取消")
    if path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as stream:
            field_names = [name for name in ("charge_density", "potential", "integrated_Ex", "integrated_Ey") if name in stream]
            if field_names:
                required = ("x", "y", "slice_id", "delta_z")
                if any(name not in stream for name in required):
                    raise ValueError("SC 场文件缺少 x、y、slice_id 或 delta_z。")
                x, y, slices, widths = (np.asarray(stream[name]) for name in required)
                for name, axis in (("x", x), ("y", y)):
                    if axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
                        raise ValueError(f"SC {name} 网格必须是一维、有限且严格递增，至少包含两个节点。")
                if slices.ndim != 1 or widths.shape != slices.shape or not slices.size:
                    raise ValueError("SC 切片编号和 Δz 必须是一维且长度相同。")
                fields = {}
                for name in field_names:
                    progress(f"读取 {path.name} · {name}")
                    if cancelled():
                        raise InterruptedError("读取已取消")
                    if stream[name].shape != (len(slices), len(y), len(x)):
                        raise ValueError(f"SC {name} 必须按 (slice, y, x) 保存，当前形状为 {stream[name].shape}。")
                    fields[name] = np.asarray(stream[name])
                return FieldResult(x, y, slices, widths, fields, dict(stream.attrs))
    result = read_result(path, arrays=True)
    if cancelled():
        raise InterruptedError("读取已取消")
    progress(f"完成 {path.name}")
    return result


def compare_columns(current, baseline, x_name, y_name, mode, status="all", batch=None):
    """Compare exact matching X rows; never sort, interpolate, or divide by zero."""
    names = {key.casefold(): key for key in baseline.columns}
    bx, by = names.get(x_name.casefold()), names.get(y_name.casefold())
    if bx is None or by is None:
        raise ValueError("基线缺少所选 X / Y 列。")
    for key_a, key_b in ((x_name, bx), (y_name, by)):
        if column_unit(current, key_a) != column_unit(baseline, key_b):
            raise ValueError(f"两份数据的 {key_a} 单位不一致或一份缺少单位，不能直接比较。")
    a = current.select_columns((x_name, y_name), status, batch)
    b = baseline.select_columns((bx, by), status, batch)
    x, y, reference = a[x_name], a[y_name], b[by]
    if x.shape != b[bx].shape or not np.isfinite(x).all() or not np.array_equal(x, b[bx]):
        raise ValueError("比较需要 X 值和行顺序严格一致且有限；不会自动排序或插值。")
    for key in ("ZCoordinate", "CoordinateDefinition"):
        if key in current.metadata or key in baseline.metadata:
            if current.metadata.get(key) != baseline.metadata.get(key):
                raise ValueError(f"两份数据的 {key} 不一致，不能直接比较。")
    current_names = {key.casefold(): key for key in current.columns}
    if "particle_id" in current_names or "particle_id" in names:
        if "particle_id" not in current_names or "particle_id" not in names:
            raise ValueError("粒子逐行比较需要两份数据均提供 particle_id。")
        key_a, key_b = current_names["particle_id"], names["particle_id"]
        if not np.array_equal(current.select_columns((key_a, ), status, batch)[key_a], baseline.select_columns((key_b, ), status, batch)[key_b]):
            raise ValueError("两份数据的 particle_id 顺序不一致，不能逐行比较。")
    if {x_name.casefold(), y_name.casefold()} & {"px", "py", "dp"}:
        for key in ("ReferenceMomentum", "ReferenceBeta"):
            if current.metadata.get(key) != baseline.metadata.get(key):
                raise ValueError(f"归一化动量比较需要相同的 {key}。")
        for key in ("referencemomentum", "referencebeta"):
            if key in current_names or key in names:
                if key not in current_names or key not in names:
                    raise ValueError("归一化动量比较缺少配套逐行参考量。")
                key_a, key_b = current_names[key], names[key]
                if not np.array_equal(
                        current.select_columns((key_a, ), status, batch)[key_a],
                        baseline.select_columns((key_b, ), status, batch)[key_b]):
                    raise ValueError("归一化动量比较的逐行参考量不一致。")
    if mode == "overlay":
        return x, y, reference
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        difference = np.asarray(y, dtype=float) - np.asarray(reference, dtype=float)
        if mode == "relative":
            valid = np.isfinite(difference) & np.isfinite(reference) & (reference != 0)
            relative = np.full(difference.shape, np.nan)
            np.divide(difference, reference, out=relative, where=valid)
            relative[~np.isfinite(relative)] = np.nan
            return x, relative, reference
    difference[~np.isfinite(difference)] = np.nan
    return x, difference, reference


def density_histogram(x, y, bins=80):
    """Count macro-particle rows, without charge weighting or interpolation."""
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("密度图的 X / Y 必须一维且等长。")
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        raise ValueError("当前筛选没有有限的 X / Y 配对。")
    histogram, x_edges, y_edges = np.histogram2d(x[valid], y[valid], bins=int(bins))
    return histogram.T, x_edges, y_edges


def column_unit(result, name):
    definitions = result.metadata.get("column_definitions", {})
    definition = next((value for key, value in definitions.items() if key.casefold() == name.casefold()), {})
    explicit = definition.get("units")
    if isinstance(explicit, bytes):
        explicit = explicit.decode("utf-8")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    if not any(key in result.metadata for key in ("PASSVersion", "ZCoordinate", "ReferenceMomentum")):
        return None
    units = {
        "x": "m",
        "y": "m",
        "z": "m",
        "px": "Px/P0",
        "py": "Py/P0",
        "dp": "ΔP/P0",
        "arrival_time_s": "s",
        "referencetime": "s",
        "referencebeta": "β",
        "referencemomentum": "eV/c; ions per nucleon",
        "lost_position": "m"
    }
    return units.get(name.casefold())


def column_label(result, name):
    unit = column_unit(result, name)
    return f"{name} [{unit}]" if unit else name


def _task_context(context):
    from PASS.gui.jobs import TaskContext

    return context if context is not None else TaskContext()


def _selected_columns(result, names, selection, context):
    columns = {}
    for index, name in enumerate(names):
        context.report(f"筛选完整数据 · {index + 1}/{len(names)} 列")
        columns.update(result.select_columns((name, ), selection.status, selection.batch))
    context.check()
    return columns


def export_selection_data(selection, path, *, context=None):
    """Filter a frozen selection and export every row, including invalid values."""
    context = _task_context(context)
    context.report("准备完整筛选数据…")
    result = selection.result
    metadata = {"source": selection.source, "status": selection.status, "batch": selection.batch, "display_reduction": False}
    definitions = dict(result.metadata.get("column_definitions", {}))
    if isinstance(result, FieldResult):
        index = selection.index
        x, y = np.meshgrid(result.x, result.y)
        columns = {"x_m": x.ravel(), "y_m": y.ravel()}
        definitions.update({"x_m": {"units": "m"}, "y_m": {"units": "m"}})
        for name in result.fields:
            context.check()
            columns[name] = result.fields[name][index].ravel()
            definitions[name] = {"units": result.plane(name, index, False)[1]}
        metadata.update(slice_id=result.slice_ids[index], delta_z_m=result.delta_z[index], axis_order="y,x")
        if selection.average:
            values, unit = result.plane(selection.field, index, True)
            columns[selection.field + "_slice_average"] = values.ravel()
            definitions[selection.field + "_slice_average"] = {"units": unit}
            metadata["average_units"] = unit
    else:
        columns = _selected_columns(result, tuple(result.columns), selection, context)
        if selection.comparison != "single":
            context.report("检查基线对齐并计算完整比较数据…")
            _, values, reference = compare_columns(result, selection.baseline, selection.x_name, selection.y_name, selection.comparison,
                                                   selection.status, selection.batch)
            columns["comparison_baseline"] = reference
            unit = column_unit(result, selection.y_name)
            definitions["comparison_baseline"] = {"units": unit} if unit else {}
            if selection.comparison != "overlay":
                name = "comparison_" + selection.comparison
                columns[name] = values
                definitions[name] = {"units": "1"} if selection.comparison == "relative" else dict(definitions["comparison_baseline"])
            metadata.update(baseline=selection.baseline_path, comparison=selection.comparison, alignment="exact X and row order; no interpolation")
    parameters = {key: value for key, value in result.metadata.items() if key not in {"column_definitions", "plot_export"}}
    export_columns(path, columns, {"parameters": parameters, "column_definitions": definitions, "plot_export": metadata}, context=context)


def selection_figure_spec(selection, *, context=None):
    """Build a full-resolution specification without consulting any Qt object."""
    context = _task_context(context)
    context.report("准备完整图像数据…")
    result = selection.result
    if isinstance(result, FieldResult):
        # Transfer only the selected field plane, retaining every grid node.
        index = selection.index
        result = FieldResult(result.x, result.y, result.slice_ids[index:index + 1], result.delta_z[index:index + 1],
                             {selection.field: result.fields[selection.field][index:index + 1]}, result.metadata)
        return "field", {"result": result, "field": selection.field, "index": 0, "average": selection.average, "aperture": selection.aperture}
    columns = _selected_columns(result, (selection.x_name, selection.y_name), selection, context)
    x, y = columns[selection.x_name], columns[selection.y_name]
    mode = selection.mode
    if mode == "auto":
        mode = "scatter" if {name.casefold() for name in result.columns} & {"tag", "particle_id"} else "line"
    x_label, y_label = column_label(result, selection.x_name), column_label(result, selection.y_name)
    if selection.comparison != "single":
        if mode == "density":
            raise ValueError("文件比较请先选择折线或散点；密度与投影用于当前文件。")
        x, y, reference = compare_columns(result, selection.baseline, selection.x_name, selection.y_name, selection.comparison, selection.status,
                                          selection.batch)
        context.check()
        if selection.comparison == "overlay":
            series = [(Path(selection.source).name + " (A)", x, y, mode), (Path(selection.baseline_path).name + " (B)", x, reference, mode)]
        else:
            y_label = f"({selection.y_name} A−B)/B [1]" if selection.comparison == "relative" else f"Δ {y_label} (A−B)"
            series = [(selection.comparison, x, y, "line")]
    elif mode == "density":
        return "density", {"x": x, "y": y, "bins": selection.bins, "x_label": x_label, "y_label": y_label}
    else:
        series = [(Path(selection.source).name, x, y, mode)]
    return "series", {"series": series, "x_label": x_label, "y_label": y_label}


class ExportRecoveryError(OSError):
    """A storage failure also blocked rollback; preserve the backup directory."""


@contextmanager
def _export_directory(parent):
    directory = Path(tempfile.mkdtemp(prefix=".pass-export-", dir=parent))
    preserve = False
    try:
        yield directory
    except ExportRecoveryError:
        preserve = True
        raise
    finally:
        if not preserve:
            shutil.rmtree(directory)


def _publish_pair(staged, destinations, context):
    """Rollback ordinary failures; two paths cannot be crash-atomic as a pair."""
    backup_directory = staged[0].parent / "backups"
    backup_directory.mkdir()
    backups = []
    for index, target in enumerate(destinations):
        if target.exists():
            backup = backup_directory / f"previous-{index}"
            context.report("准备原文件备份…")
            context.copy_file(target, backup)
            backups.append(backup)
        else:
            backups.append(None)
    context.report("发布数据和元数据…")
    context.check()
    # No cancellation inside this short boundary: either publish both or restore.
    committed = []
    try:
        for source, target in zip(staged, destinations):
            os.replace(source, target)
            committed.append(target)
    except OSError as publish_error:
        restore_errors = []
        for index, target in reversed(list(enumerate(destinations))):
            if target not in committed:
                continue
            try:
                if backups[index] is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backups[index], target)
            except OSError as exc:
                restore_errors.append(str(exc))
        if restore_errors:
            raise ExportRecoveryError(f"发布失败且存储系统阻止恢复：{publish_error}。原文件备份保留在 {staged[0].parent}；恢复错误：{'；'.join(restore_errors)}") from publish_error
        raise


def export_columns(path, columns, metadata=None, *, context=None):
    """Chunked CSV and converter-compatible typed/hash-checked sidecar publication."""
    import pandas as pd
    from PASS.tool.data_conversion import _json_value

    context = _task_context(context)
    path = Path(path)
    columns = {name: np.asarray(values) for name, values in columns.items()}
    if any(value.ndim != 1 for value in columns.values()) or len({len(value) for value in columns.values()}) > 1:
        raise ValueError("导出列必须一维且等长。")
    count = len(next(iter(columns.values()), []))
    with _export_directory(path.parent) as directory:
        # Internal names must never collide with an arbitrary destination basename.
        temporary = Path(directory) / "table.csv"
        sidecar = Path(str(temporary) + ".metadata.json")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            for start in range(0, max(1, count), 32768):
                context.report(f"写入 CSV · {min(start, count):,}/{count:,} 行")
                pd.DataFrame({
                    name: values[start:start + 32768]
                    for name, values in columns.items()
                }).to_csv(stream, index=False, header=start == 0, float_format="%.17g", na_rep="nan")
            stream.flush()
            os.fsync(stream.fileno())
        context.report("校验 CSV 并写入元数据…")
        digest = hashlib.sha256()
        with temporary.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                context.check()
                digest.update(block)
        with sidecar.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "format": "pass-table-metadata-v1",
                    "sha256": digest.hexdigest(),
                    "metadata": _json_value(metadata or {}),
                    "dtypes": {
                        name: str(value.dtype)
                        for name, value in columns.items()
                    }
                },
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_pair((temporary, sidecar), (path, Path(str(path) + ".metadata.json")), context)


def _render_figure(spec, path):
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    with rc_context({"path.simplify": False}):
        figure = create_figure(*spec, dpi=160)
        FigureCanvasAgg(figure)
        try:
            figure.savefig(path, dpi=160)
        finally:
            figure.clear()


def _render_isolated(spec, path, directory, context):
    request = Path(directory) / "figure.pickle"
    context.report("传递完整图像数据…")
    with request.open("wb") as stream:
        pickle.dump(spec, stream, protocol=pickle.HIGHEST_PROTOCOL)
    context.check()
    # A separate interpreter isolates Matplotlib rcParams/font caches from GUI paints.
    environment = dict(os.environ, MPLBACKEND="Agg")
    package_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
    context.report("正在独立进程中生成完整图像…")
    with (Path(directory) / "renderer.log").open("w+b") as log:
        process = subprocess.Popen([sys.executable, "-m", "PASS.gui.plot_analysis", "--render-export",
                                    str(request), str(path)],
                                   stdout=log,
                                   stderr=log,
                                   env=environment,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while process.poll() is None:
                context.cancelled.wait(0.05)
                context.check()
            if process.returncode:
                log.seek(0)
                details = log.read(16384).decode("utf-8", errors="replace")
                raise ValueError("图像生成失败：" + details)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            process.wait()


def export_selection_image(selection, path, *, context=None, isolated=False):
    """Render all selected rows, then replace the image at one publication point."""
    context = _task_context(context)
    spec = selection_figure_spec(selection, context=context)
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix=".pass-image-", dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        context.check()
        if isolated:
            _render_isolated(spec, temporary, directory, context)
        else:
            context.report("正在生成完整图像…")
            _render_figure(spec, temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        context.report("发布图像…")
        context.check()
        os.replace(temporary, path)


def create_figure(kind, payload, *, size=(10, 6), dpi=100):
    """Build standalone scientific figures lazily; data always has full resolution."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=size, dpi=dpi, layout="constrained")
    if kind == "density":
        values, x_edges, y_edges = density_histogram(payload["x"], payload["y"], payload.get("bins", 80))
        grid = figure.add_gridspec(4, 4)
        axis = figure.add_subplot(grid[1:, :3])
        top = figure.add_subplot(grid[0, :3], sharex=axis)
        right = figure.add_subplot(grid[1:, 3], sharey=axis)
        image = axis.pcolormesh(x_edges, y_edges, values, shading="flat", cmap="viridis")
        top.stairs(values.sum(axis=0), x_edges)
        right.stairs(values.sum(axis=1), y_edges, orientation="horizontal")
        top.set_ylabel("Count")
        right.set_xlabel("Count")
        top.tick_params(labelbottom=False)
        right.tick_params(labelleft=False)
        figure.colorbar(image, ax=[axis, top, right], label="Macro-particle rows / bin", shrink=0.7)
        axis.set_xlabel(payload["x_label"])
        axis.set_ylabel(payload["y_label"])
    elif kind == "field":
        figure.set_layout_engine("compressed")
        result, name, index, average = payload["result"], payload["field"], payload["index"], payload["average"]
        values, unit = result.plane(name, index, average)
        axis = figure.add_subplot(111)
        image = axis.pcolormesh(result.x, result.y, np.ma.masked_invalid(values), shading="nearest", cmap="RdBu_r" if "E" in name else "viridis")
        figure.colorbar(image, ax=axis, label=f"{name} [{unit}]")
        axis.set(xlabel="x [m]", ylabel="y [m]", title=f"slice_id={result.slice_ids[index]}, Δz={result.delta_z[index]:.6g} m")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(result.x[0], result.x[-1])
        axis.set_ylim(result.y[0], result.y[-1])
        if payload.get("aperture", True) and result.metadata.get("aperture_type") not in (None, "off", "default"):
            from PASS.utils.aperture import build_aperture

            values = result.metadata.get("aperture_value", "[]")
            parameters = json.loads(values) if isinstance(values, (str, bytes)) else values
            aperture = build_aperture({"Type": result.metadata["aperture_type"], "Value": parameters})
            x = np.linspace(result.x[0], result.x[-1], max(256, len(result.x)))
            y = np.linspace(result.y[0], result.y[-1], max(256, len(result.y)))
            mask = aperture.mask(x[None, :], y[:, None])
            if mask.any() and not mask.all():
                axis.contour(x, y, mask.astype(float), levels=[0.5], colors="black", linewidths=1)
    else:
        axis = figure.add_subplot(111)
        for label, x, y, mode in payload["series"]:
            if mode == "scatter":
                axis.scatter(x, y, s=3, label=label, rasterized=True)
            else:
                axis.plot(x, y, label=label, marker="o" if np.count_nonzero(np.isfinite(x) & np.isfinite(y)) == 1 else None, markersize=3)
        axis.set(xlabel=payload["x_label"], ylabel=payload["y_label"])
        axis.grid(alpha=0.25)
        if len(payload["series"]) > 1:
            axis.legend()
    return figure


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--render-export":
        raise SystemExit("Internal GUI image renderer; use the result export controls.")
    # The request is created privately by this process's parent, never user input.
    with Path(sys.argv[2]).open("rb") as request_stream:
        _render_figure(pickle.load(request_stream), Path(sys.argv[3]))
