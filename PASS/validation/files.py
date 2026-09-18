"""Validate input TFS contents using the same parser and columns as tracking."""
from pathlib import Path
import re

import numpy as np

INPUT_FILE_FIELDS = frozenset({
    "waveform file",
    "distribution file path",
    "file path",
    "file_path",
    "program file",
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
    """Use the actual wake reader for file validation, never the TFS parser."""
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
                actual = loaded.model.input_metadata["path"]
                if actual not in check.report.checked_files:
                    check.report.checked_files.append(actual)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                check.add(location, "wake.file", f"Wake file validation failed: {exc}", not values.get("Is enabled", True))


def check_table(check, value, path, kind, active, minimum_rows):
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
        # Check every declared table, including inactive resources, but inactive
        # failures are warnings since they cannot affect the selected execution.
        cached = check.file_cache.get(file)
        if cached is None:
            import tfs
            try:
                cached = tfs.read(file)
            except Exception as exc:
                cached = str(exc) or type(exc).__name__
            check.file_cache[file] = cached
            check.report.checked_files.append(str(file))
        if isinstance(cached, str):
            check.add(path, "file.format", f"无法读取 TFS：{cached}", not active)
            return
        frame = cached
        if frame.empty:
            check.add(path, "file.empty", "TFS 文件没有数据行", not active)
            return
        names = list(frame.columns)
        lower = [str(name).lower() for name in names]
        if len(set(lower)) != len(lower):
            check.add(path, "file.columns", "TFS 列名重复（包括大小写冲突）", not active)
            return
        required = []
        if kind == "distribution":
            # Injection._load_dist indexes literal lowercase columns.
            required = ["x", "px", "y", "py", "z", "dp"]
        elif kind.startswith("offset_"):
            axis = kind[-1]
            patterns = [r"(?:time|turn)\s*(?:\(\s*s\s*\))?", rf"{axis}\s*(?:\(\s*m\s*\))?", rf"p{axis}\s*(?:\(\s*rad\s*\))?"]
            for pattern in patterns:
                matched = [n for n in names if re.fullmatch(pattern, str(n), re.I)]
                if len(matched) != 1:
                    check.add(path, "file.columns", f"偏移文件需要唯一的时间/圈数列及 {axis}, p{axis} 列；匹配 {pattern!r} 得到 {matched}", not active)
                    return
                required.append(matched[0])
        else:
            return  # No engine-defined magnetic ramping format exists yet.
        missing = [n for n in required if n not in names]
        if missing:
            check.add(path, "file.columns", f"缺少必需列：{missing}；现有列：{names}", not active)
            return
        arrays = {}
        for column in required:
            series = frame[column]
            if not np.issubdtype(series.dtype, np.number) or np.issubdtype(series.dtype, np.bool_):
                check.add(path, "file.numeric", f"列 {column!r} 必须使用 TFS 数值类型", not active)
                continue
            array = series.to_numpy()
            bad = np.flatnonzero(~np.isfinite(array))
            if len(bad):
                check.add(path, "file.nonfinite", f"列 {column!r} 有 {len(bad)} 个非有限数值，第一个位于数据行 {bad[0] + 1}", not active)
                continue
            arrays[column] = array
        if len(arrays) != len(required):
            return
        if kind == "distribution":
            if len(frame) < minimum_rows:
                check.add(path, "distribution.rows", f"该 bunch 在粒子池中的索引要求文件至少 {minimum_rows} 行，实际 {len(frame)} 行；不足部分不会正确初始化", not active)
            dp, px, py = arrays["dp"], arrays["px"], arrays["py"]
            with np.errstate(over="ignore", invalid="ignore"):
                bad = np.flatnonzero((dp <= -1) | ((1 + dp)**2 <= px**2 + py**2))
            if len(bad):
                check.add(path, "distribution.momentum", f"{len(bad)} 行无法得到正的实数纵向动量；首个数据行 {bad[0] + 1}", not active)
        if kind.startswith("offset_"):
            time = arrays[required[0]]
            if np.any(time < 0) or np.any(np.diff(time) <= 0):
                check.add(path, "offset.time", "偏移时间/圈数必须非负且严格递增，不能重复", not active)
            if str(required[0]).lower().startswith("turn") and np.any(time != np.floor(time)):
                check.add(path, "offset.turn", "偏移文件 turn 列必须为整数", not active)
            if time[0] > 0:
                check.add(path, "offset.coverage", "偏移表从 0 之后开始；最初阶段将使用第一行", True)
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
