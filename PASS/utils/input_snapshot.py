"""Qt-free input snapshots shared by command-line and GUI runs."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


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
        return Path(__file__).resolve().parents[2] / "output"
    path = Path(value)
    return (path if path.is_absolute() else Path(input_path).resolve().parent / path).resolve()


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


def archive_input_documents(documents, snapshot, output, *, comparison_documents=None, check=None, copy_file=None, record=None):
    """Archive raw configurations and resolved execution inputs in one fresh directory.

    Callers validate before and after archiving. Missing inactive inputs remain
    explicit in the record; missing active inputs fail the caller's validation.
    Execution JSON paths are relative to the snapshot and never point at copied
    dependencies' original locations. Original dictionaries are not mutated.
    """
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    if record is None:
        record = {}
    for name in ("inputs", "dependencies", "random_seeds", "unavailable_dependencies"):
        record.setdefault(name, [])
    copied, paths = {}, []
    for index, (name, original, base) in enumerate(documents):
        if check is not None:
            check()
        data = deepcopy(original)
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
                target = snapshot / "assets" / str(len(copied)) / source.name
                digest = copy_file(source, target) if copy_file is not None else copy_input_file(source, target, check)
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
        content = json_bytes(data)
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
