"""Validate input tables using the same parser and columns as tracking."""
from pathlib import Path
import json
import re

import numpy as np


def _inspect_distribution(file, chunk_rows=65536):
    """Cache only row counts and diagnostics, never a full coordinate table."""
    from PASS.utils.injection_io import DistributionBatchReader

    fields = ("x", "px", "y", "py", "z", "dp")
    issues = []
    nonfinite = {name: [0, None] for name in fields}
    momentum_count, momentum_first = 0, None
    n_rows = 0

    def _check_columns(names):
        if len({str(name).lower() for name in names}) != len(names):
            issues.append(("file.columns", "输入表格列名重复（包括大小写冲突）"))
            return False
        missing = [name for name in fields if name not in names]
        if missing:
            issues.append(("file.columns", f"缺少必需列：{missing}；现有列：{names}"))
            return False
        return True

    def _check_values(values):
        nonlocal n_rows, momentum_count, momentum_first
        for column, name in enumerate(fields):
            bad = np.flatnonzero(~np.isfinite(values[:, column]))
            if len(bad):
                nonfinite[name][0] += len(bad)
                if nonfinite[name][1] is None:
                    nonfinite[name][1] = n_rows + int(bad[0]) + 1
        dp, px, py = values[:, 5], values[:, 1], values[:, 3]
        with np.errstate(over="ignore", invalid="ignore"):
            bad = np.flatnonzero(np.isfinite(dp) & np.isfinite(px) & np.isfinite(py) & (1 + dp <= np.hypot(px, py)))
        if len(bad):
            momentum_count += len(bad)
            if momentum_first is None:
                momentum_first = n_rows + int(bad[0]) + 1
        n_rows += len(values)

    reader = DistributionBatchReader(file)
    if file.suffix.lower() in (".h5", ".hdf5"):
        import h5py

        with h5py.File(file, "r") as stream:
            names = json.loads(stream.attrs["_pass_table_columns"]) if "_pass_table_columns" in stream.attrs else list(stream.keys())
            if not _check_columns(names):
                return None, issues
            lengths = set()
            for name in names:
                dataset = stream[name]
                if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1:
                    raise ValueError("HDF5 table columns must be one-dimensional")
                lengths.add(len(dataset))
            if len(lengths) != 1:
                raise ValueError("HDF5 file must contain column definitions of equal length")
            for name in fields:
                if stream[name].dtype.kind not in "iuf":
                    issues.append(("file.numeric", f"列 {name!r} 必须使用数值类型"))
            if issues:
                return None, issues
            total_rows = lengths.pop()
            for start in range(0, total_rows, chunk_rows):
                end = min(start + chunk_rows, total_rows)
                values = np.empty((end - start, len(fields)), dtype=np.float64)
                for column, name in enumerate(fields):
                    values[:, column] = stream[name][start:end]
                _check_values(values)
    else:
        with file.open("rb") as stream:
            reader._read_tfs_header(stream, require_coordinates=False)
            if not _check_columns(reader._column_names):
                return None, issues
            for name, kind in zip(reader._column_names, reader._column_types):
                if name in fields and kind not in {"%le", "%f", "%hd", "%d"}:
                    issues.append(("file.numeric", f"列 {name!r} 必须使用数值类型"))
            if issues:
                return None, issues
            while True:
                lines = []
                for _ in range(chunk_rows):
                    line = reader._read_data_line(stream)
                    if line is None:
                        break
                    lines.append(line)
                if not lines:
                    break
                _check_values(reader._parse_tfs_rows(lines, n_rows))
    if not n_rows:
        issues.append(("file.empty", "输入表格没有数据行"))
    for name, (count, first) in nonfinite.items():
        if count:
            issues.append(("file.nonfinite", f"列 {name!r} 有 {count} 个非有限数值，第一个位于数据行 {first}"))
    if momentum_count:
        issues.append(("distribution.momentum", f"{momentum_count} 行无法得到正的实数纵向动量；首个数据行 {momentum_first}"))
    return n_rows, issues


INPUT_FILE_FIELDS = frozenset({
    "waveform file",
    "distribution file path",
    "insert particle file",
    "file path",
    "file_path",
    "program file",
    "ramping file",
    "k0l ramping file",
    "k1l ramping file",
    "k1sl ramping file",
    "k2l ramping file",
    "k2sl ramping file",
    "k3l ramping file",
    "k3sl ramping file",
    "kl ramping file",
    "kick ramping file",
})


def resolve_input_paths(data, base):
    """Use the input JSON directory for dependencies, as the validator does."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key.casefold() in INPUT_FILE_FIELDS and isinstance(value, str) and value.strip():
                path = Path(value).expanduser()
                data[key] = str((path if path.is_absolute() else Path(base) / path).resolve())
            elif isinstance(value, (dict, list)):
                resolve_input_paths(value, base)
    elif isinstance(data, list):
        for value in data:
            resolve_input_paths(value, base)
    return data


def check_wake_files(check, values, path):
    """Validate canonical wake metadata and physics with the tracking reader."""
    if not check.check_files:
        return
    from copy import deepcopy
    from PASS.para.schema.wake_field import WakeComponentConfig
    from PASS.commands.wake_field import _build_component
    groups = values.get("Groups", [])
    if not isinstance(groups, list):
        return
    for gi, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        entries = group.get("Components", [])
        if not isinstance(entries, list):
            continue
        for ci, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            try:
                config = WakeComponentConfig.model_validate(entry)
            except ValueError:
                continue  # The schema validator already reports this entry.
            if config.model.kind != "file":
                continue
            location = (*path, "Groups", gi, "Components", ci, "Model", "File path")
            prepared = deepcopy(entry)
            resolve_input_paths(prepared, check.base)
            try:
                loaded = _build_component(WakeComponentConfig.model_validate(prepared))
                if group.get("Boundary", "causal_passages") == "causal_passages" and not loaded.model.causal:
                    raise ValueError("A two-sided response cannot use causal passage scheduling; select isolated or periodic Boundary")
                actual = loaded.model.input_metadata["path"]
                if actual not in check.report.checked_files:
                    check.report.checked_files.append(actual)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                check.add(location, "wake.file", f"Wake file validation failed: {exc}", not values.get("Is enabled", True))


def check_table(check, value, path, kind, active, minimum_rows, maximum_rows=None):
    if value is None or value == "":
        if active:
            check.add(path, "file.required", "此功能已启用，必须选择输入文件")
        return
    if not isinstance(value, str) or not value.strip():
        check.add(path, "file.path", "文件路径必须为非空字符串")
        return
    if not active:
        check.add(path, "file.unused", "此文件对应的功能未启用，运行时不会读取", True)
    if not check.check_files:
        return
    try:
        file = Path(value).expanduser()
        if not file.is_absolute():
            file = check.base / file
        file = file.resolve()
        if not file.is_file():
            check.add(path, "file.missing", f"输入文件不存在或不是普通文件：{file}", not active)
            return
        if kind != "distribution" and file.suffix.lower() in (".h5", ".hdf5"):
            check.add(path, "file.format", "此输入仍使用 TFS；HDF5 输入仅支持粒子分布", not active)
            return
        if kind == "distribution":
            cache_key = (file, "distribution")
            if cache_key not in check.file_cache:
                try:
                    check.file_cache[cache_key] = _inspect_distribution(file)
                except Exception as exc:
                    check.file_cache[cache_key] = (None, [("file.format", f"无法读取输入表格：{str(exc) or type(exc).__name__}")])
                if str(file) not in check.report.checked_files:
                    check.report.checked_files.append(str(file))
            n_rows, issues = check.file_cache[cache_key]
            for code, message in issues:
                check.add(path, code, message, not active)
            if n_rows is None or not n_rows:
                return
            if n_rows < minimum_rows:
                check.add(path, "distribution.rows", f"该 bunch 需要文件至少 {minimum_rows} 行，实际 {n_rows} 行；不足部分不会正确初始化", not active)
            if maximum_rows is not None and n_rows > maximum_rows:
                check.add(path, "injection.insert_count", f"显式坐标文件含 {n_rows} 行，不能超过首次注入的宏粒子数 {maximum_rows}", not active)
            return n_rows
        # Check every declared table, including inactive resources, but inactive
        # failures are warnings since they cannot affect the selected execution.
        cached = check.file_cache.get(file)
        if cached is None:
            from PASS.utils.table_io import read_table
            try:
                cached = read_table(file)
            except Exception as exc:
                cached = str(exc) or type(exc).__name__
            check.file_cache[file] = cached
            check.report.checked_files.append(str(file))
        if isinstance(cached, str):
            check.add(path, "file.format", f"无法读取输入表格：{cached}", not active)
            return
        frame = cached
        if frame.empty:
            check.add(path, "file.empty", "输入表格没有数据行", not active)
            return
        names = list(frame.columns)
        lower = [str(name).lower() for name in names]
        if len(set(lower)) != len(lower):
            check.add(path, "file.columns", "输入表格列名重复（包括大小写冲突）", not active)
            return
        required = []
        if kind.startswith("offset_"):
            axis = kind[-1]
            patterns = [r"(?:time|turn)\s*(?:\(\s*s\s*\))?", rf"{axis}\s*(?:\(\s*m\s*\))?", rf"p{axis}\s*(?:\(\s*rad\s*\))?"]
            for pattern in patterns:
                matched = [n for n in names if re.fullmatch(pattern, str(n), re.I)]
                if len(matched) != 1:
                    check.add(path, "file.columns", f"偏移文件需要唯一的时间/圈数列及 {axis}, p{axis} 列；匹配 {pattern!r} 得到 {matched}", not active)
                    return
                required.append(matched[0])
        else:
            return  # Specialized program readers validate their own formats.
        missing = [n for n in required if n not in names]
        if missing:
            check.add(path, "file.columns", f"缺少必需列：{missing}；现有列：{names}", not active)
            return
        arrays = {}
        for column in required:
            series = frame[column]
            if not np.issubdtype(series.dtype, np.number) or np.issubdtype(series.dtype, np.bool_):
                check.add(path, "file.numeric", f"列 {column!r} 必须使用数值类型", not active)
                continue
            array = series.to_numpy()
            bad = np.flatnonzero(~np.isfinite(array))
            if len(bad):
                check.add(path, "file.nonfinite", f"列 {column!r} 有 {len(bad)} 个非有限数值，第一个位于数据行 {bad[0] + 1}", not active)
                continue
            arrays[column] = array
        if len(arrays) != len(required):
            return
        if kind.startswith("offset_"):
            time = arrays[required[0]]
            if np.any(time < 0) or np.any(np.diff(time) <= 0):
                check.add(path, "offset.time", "偏移时间/圈数必须非负且严格递增，不能重复", not active)
            if str(required[0]).lower().startswith("turn") and np.any(time != np.floor(time)):
                check.add(path, "offset.turn", "偏移文件 turn 列必须为整数", not active)
            if time[0] > 0:
                check.add(path, "offset.coverage", "偏移表从 0 之后开始；最初阶段将使用第一行", True)
        return len(frame)
    except (OSError, ValueError, OverflowError) as exc:
        check.add(path, "file.read", str(exc), not active)


def check_rf_files(check, values, path):
    if not check.check_files:
        return
    from copy import deepcopy
    from PASS.commands.element.rfcavity import RFWaveform
    from PASS.utils.program import LinearProgram
    entries = values.get('Components', [])
    if not isinstance(entries, list):
        return  # The schema diagnostic owns malformed component containers.
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            continue
        if not item.get('Program file'):
            continue
        prepared = deepcopy(item)
        resolve_input_paths(prepared, check.base)
        try:
            RFWaveform(prepared, LinearProgram(1.))
            check.report.checked_files.append(prepared['Program file'])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            check.add((*path, 'Components', index, 'Program file'), 'rf.program', str(exc))


def check_magnet_ramping(check, values, path, *, order=None):
    """Use the runtime reader for enabled and inactive magnet program files."""
    from copy import deepcopy
    from tfs.errors import TfsFormatError
    from PASS.utils.magnet_program import _ramping_sources, load_magnet_ramp

    active = bool(values.get("Is ramping", False))
    prepared = deepcopy(values)
    resolve_input_paths(prepared, check.base)
    declared = any(value for key, value in prepared.items() if key.lower() == "ramping file" or key.lower().endswith(" ramping file"))
    if not active and not declared:
        return
    if not active:
        check.add((*path, "Ramping file"), "file.unused", "磁铁 ramping 未启用，所选文件不会参与跟踪", True)
    read_errors = (OSError, ValueError, KeyError, TypeError, OverflowError, TfsFormatError)
    try:
        sources = _ramping_sources(prepared, order=order)
        if not check.check_files:
            return
        prepared["Is ramping"] = True
        length = prepared.get("Length (m)", 0.)
        cache_key = (sources, length, order)
        # Cache only this validation pass, including length/order-specific errors.
        # Store diagnostics, not large programs; every reference keeps its location.
        if cache_key not in check.magnet_ramp_cache:
            try:
                load_magnet_ramp(prepared, length=length, order=order)
                check.magnet_ramp_cache[cache_key] = None
            except read_errors as exc:
                check.magnet_ramp_cache[cache_key] = str(exc)
        error = check.magnet_ramp_cache[cache_key]
        if error is not None:
            check.add((*path, "Ramping file"), "magnet.ramping", error, not active)
            return
        for source in sources:
            if str(source) not in check.report.checked_files:
                check.report.checked_files.append(str(source))
    except read_errors as exc:
        check.add((*path, "Ramping file"), "magnet.ramping", str(exc), not active)
