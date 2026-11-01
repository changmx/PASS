"""Qt-free input snapshots shared by command-line and GUI runs."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Iterator


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def unique_filename(name: str, existing) -> str:
    """Keep readable names portable, numbering collisions before the extension."""
    name = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", name).rstrip(". ") or "file"
    if re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", name):
        name = "_" + name
    names = {str(value).casefold() for value in existing}
    path = PurePosixPath(name)
    candidate, suffix = name, 1
    while candidate.casefold() in names:
        candidate = f"{path.stem}{suffix}{path.suffix}"
        suffix += 1
    return candidate


def atomic_write(path: Path, content: bytes) -> None:
    """Replace only after a complete, flushed write in the same directory."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def resolved_file(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def resolve_output_base(value, input_path) -> Path:
    """Resolve output paths against the original beam-0 JSON, before relocation."""
    if value is None or (isinstance(value, str) and value.lower() == "default"):
        value = "output"
    path = Path(value)
    return (path if path.is_absolute() else Path(input_path).resolve().parent / path).resolve()


def read_task_name(data, *, settings=False, default=None):
    """Read the canonical or legacy folder field without coercing its value."""
    aliases = {"task_name", "output_subdirectory", "output_name"} if settings else {"task name", "output subdirectory", "output name"}
    keys = [key for key in data if isinstance(key, str) and key.casefold() in aliases]
    if len(keys) > 1:
        raise ValueError("Specify Task name only once; do not combine it with legacy output-name fields")
    return data[keys[0]] if keys else default


def canonicalize_task_name(data, *, settings=False, python_names=False):
    """Copy aliases, placing the task name directly after the output directory."""
    aliases = {"task_name", "output_subdirectory", "output_name"} if settings else {"task name", "output subdirectory", "output name"}
    if python_names:
        aliases |= {"task_name", "output_subdirectory", "output_name"}
    keys = [key for key in data if isinstance(key, str) and key.casefold() in aliases]
    if len(keys) > 1:
        raise ValueError("Specify Task name only once; do not combine it with legacy output-name fields")
    if not keys:
        return dict(data)
    task_key = "task_name" if settings else "Task name"
    output_keys = {"output_directory"} if settings else {"output directory"}
    if python_names and not settings:
        output_keys.add("output_dir")
    has_output = any(isinstance(key, str) and key.casefold() in output_keys for key in data)
    result = {}
    for key, value in data.items():
        if key == keys[0]:
            if not has_output:
                result[task_key] = value
            continue
        result[key] = value
        if isinstance(key, str) and key.casefold() in output_keys:
            result[task_key] = data[keys[0]]
    return result


def normalize_task_name(value) -> str:
    """Validate a task name; empty/default values request an automatic timestamp."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Task name must be a string or null")
    name = value.strip()
    if not name or name.casefold() == "default":
        return ""
    if name in {".", ".."} or name.endswith(".") or re.search(r'[\x00-\x1f<>:"/\\|?*]', name):
        raise ValueError("Task name must be a single folder name without path separators or invalid filename characters")
    if re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\..*)?", name):
        raise ValueError("Task name must not be a reserved device name")
    return name


def normalize_output_subdirectory(value) -> str:
    """Compatibility alias for the former output-subdirectory validator."""
    return normalize_task_name(value)


def normalize_output_name(value) -> str:
    """Compatibility alias for the former output-name validator."""
    return normalize_task_name(value)


def _resolve_task_name_override(task_name, output_subdirectory, output_name):
    values = [value for value in (task_name, output_subdirectory, output_name) if value is not None]
    if len(values) > 1:
        raise ValueError("Use task_name only once; do not combine it with legacy output-name keywords")
    return values[0] if values else None


def create_run_directory(output_root, task_name=None, *, output_subdirectory=None, output_name=None) -> Path:
    """Reserve one run folder, numbering collisions without reusing results."""
    task_name = normalize_task_name(_resolve_task_name_override(task_name, output_subdirectory, output_name))
    base_name = task_name or datetime.now().strftime("%Y_%m_%d_%H_%M")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        name = base_name if attempt == 0 else f"{base_name}_{attempt}"
        directory = root / name
        try:
            directory.mkdir()
        except FileExistsError:
            attempt += 1
            continue
        return directory


def file_references(value: object, pointer: str = "") -> Iterator[tuple[dict, str, str]]:
    """Yield declared input paths using the same catalog as preflight validation."""
    from PASS.validation.files import INPUT_FILE_FIELDS

    if isinstance(value, dict):
        for key, item in value.items():
            address = pointer + "/" + str(key).replace("~", "~0").replace("/", "~1")
            if str(key).casefold() in INPUT_FILE_FIELDS and isinstance(item, str) and item.strip():
                yield value, key, address
            elif isinstance(item, (dict, list)):
                yield from file_references(item, address)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from file_references(item, f"{pointer}/{index}")


def copy_input_file(source, destination, check=None):
    """Hash the exact bytes copied; never replace an existing dependency file."""
    source, destination = Path(source), Path(destination)
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_stream, destination.open("xb") as target_stream:
        for block in iter(lambda: source_stream.read(1024 * 1024), b""):
            if check is not None:
                check()
            target_stream.write(block)
            digest.update(block)
        target_stream.flush()
        os.fsync(target_stream.fileno())
    return digest.hexdigest()


def archive_input_documents(documents,
                            snapshot,
                            output,
                            *,
                            task_name=None,
                            output_subdirectory=None,
                            output_name=None,
                            comparison_documents=None,
                            check=None,
                            copy_file=None,
                            record=None):
    """Archive raw configurations and resolved execution inputs in one fresh directory.

    Callers validate before and after archiving. Missing inactive inputs remain
    explicit in the record; missing active inputs fail the caller's validation.
    Execution JSON paths are relative to the snapshot and never point at copied
    dependencies' original locations. Dependencies are direct files in assets/;
    filename collisions receive numeric suffixes. Original dictionaries are not mutated.
    An explicit task_name overrides execution inputs only, including an empty
    string to request an automatic timestamp; None preserves their stored names.
    """
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    task_name = _resolve_task_name_override(task_name, output_subdirectory, output_name)
    if task_name is not None:
        task_name = normalize_task_name(task_name)
    if record is None:
        record = {}
    for name in ("inputs", "dependencies", "random_seeds", "unavailable_dependencies"):
        record.setdefault(name, [])
    assets = snapshot / "assets"
    asset_names = {path.name for path in assets.iterdir()} if assets.exists() else set()
    copied, paths = {}, []
    for index, (name, original, base) in enumerate(documents):
        if check is not None:
            check()
        data = canonicalize_task_name(deepcopy(original))
        configuration = snapshot / f"configuration{index}.json"
        path = snapshot / f"beam{index}.json"
        # Callers normally allocate a fresh directory; also reject accidental reuse.
        if configuration.exists() or path.exists():
            raise FileExistsError(f"Input snapshot already exists: {path}")
        original_content = json_bytes(comparison_documents[index] if comparison_documents is not None else original)
        atomic_write(configuration, original_content)
        for mapping, key, pointer in file_references(data):
            if check is not None:
                check()
            source = resolved_file(mapping[key], Path(base))
            if not source.is_file():
                # Freeze absence too: a source restored later must not bypass the archive.
                unavailable = snapshot / "unavailable" / str(len(record["unavailable_dependencies"])) / (source.name or "input")
                if os.path.lexists(unavailable):
                    raise FileExistsError(f"Unavailable input placeholder already exists: {unavailable}")
                relative = unavailable.relative_to(snapshot).as_posix()
                record["unavailable_dependencies"].append({"input": name, "pointer": pointer, "source": str(source), "file": relative})
                mapping[key] = relative
                continue
            if source not in copied:
                target = assets / unique_filename(source.name, asset_names)
                digest = copy_file(source, target) if copy_file is not None else copy_input_file(source, target, check)
                asset_names.add(target.name)
                copied[source] = target
                record["dependencies"].append({
                    "file": target.relative_to(snapshot).as_posix(),
                    "sha256": digest,
                    "source": str(source),
                    "size_bytes": target.stat().st_size,
                })
            mapping[key] = copied[source].relative_to(snapshot).as_posix()
        # Keep exactly one key even for inputs using accepted case variations.
        output_key = next((key for key in data if key.casefold() == "output directory"), "Output directory")
        data[output_key] = str(output)
        if task_name is not None:
            name_key = next((key for key in data if key.casefold() == "task name"), "Task name")
            data[name_key] = task_name
        content = json_bytes(canonicalize_task_name(data))
        atomic_write(path, content)
        values = {str(key).casefold(): value for key, value in data.items()}
        record["inputs"].append({
            "name": name,
            "file": path.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "configuration": configuration.name,
            "configuration_sha256": hashlib.sha256(original_content).hexdigest(),
            "backend": values.get("backend (gpu/cpu)", "cpu"),
            "turns": values.get("number of turns"),
        })
        for command_name, command in values.get("sequence", {}).items():
            if isinstance(command, dict):
                kwargs = {str(key).casefold(): value for key, value in command.items()}
                if str(kwargs.get("command", "")).casefold() == "injection":
                    record["random_seeds"].append({"input": name, "command": command_name, "seed": kwargs.get("random seed")})
        paths.append(path)
    return paths
