"""Portable, inspectable PASS projects: ZIP containers with ordinary input files.

This module has no Qt dependency. The tracking engine still consumes JSON files;
projects materialize immutable, fully resolved inputs before launching it.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Iterator
from uuid import uuid4
import zipfile

from PASS import __version__

FORMAT_VERSION = 1
FILE_FIELDS = frozenset({
    "distribution file path", "file path", "program file",
    "k0l ramping file", "k1l ramping file", "k1sl ramping file",
    "k2l ramping file", "k2sl ramping file", "k3l ramping file",
    "k3sl ramping file", "kl ramping file", "kick ramping file",
})
JSON_LIMIT = 64 * 1024 * 1024


class ProjectError(ValueError):
    """An incomplete, unsupported or malformed project."""


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_json(content: bytes) -> dict:
    from PASS.validation.report import parse_json
    value, report = parse_json(content)
    if not report.ok:
        if any(issue.code == "json.syntax" for issue in report.errors):
            # Preserve JSONDecodeError's line/column for the source editor.
            json.loads(content.decode("utf-8-sig"))
        raise ProjectError(report.text())
    return value


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


def file_references(value: object, pointer: str = "") -> Iterator[tuple[dict, str, str]]:
    """Yield only schema-owned input paths, never output paths or arbitrary text."""
    if isinstance(value, dict):
        for key, item in value.items():
            address = pointer + "/" + str(key).replace("~", "~0").replace("/", "~1")
            if str(key).casefold() in FILE_FIELDS and isinstance(item, str) and item.strip():
                yield value, key, address
            elif isinstance(item, (dict, list)):
                yield from file_references(item, address)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from file_references(item, f"{pointer}/{index}")


def resolved_file(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def missing_files(data: dict, base: Path) -> list[str]:
    return [f"{pointer}: input file not found: {mapping[key]}"
            for mapping, key, pointer in file_references(data)
            if not resolved_file(mapping[key], base).is_file()]


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_name(name: str, existing) -> str:
    names = {str(n).casefold() for n in existing}
    candidate, suffix = name, 2
    while candidate.casefold() in names:
        candidate = f"{name}_{suffix}"
        suffix += 1
    return candidate


def safe_member(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or "\x00" in name or path.is_absolute()
            or any(part in (".", "..", "") or part.endswith((".", " "))
                   or re.search(r'[<>"|?*]', part)
                   or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part)
                   for part in name.split("/"))):
        raise ProjectError(f"Invalid archive path: {name!r}")
    return name


@dataclass
class InputConfig:
    id: str
    name: str
    data: dict


@dataclass
class Asset:
    id: str
    path: str
    original_name: str
    sha256: str
    size_bytes: int
    kind: str = "input"


class Project:
    """One project and its private editing cache; no persistent sidecar files."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="pass-project-")
        self.root = Path(self._temporary.name)
        (self.root / "configs").mkdir()
        self.id = uuid4().hex
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.configs: dict[str, InputConfig] = {}
        self.assets: dict[str, Asset] = {}
        self.recipes: list[dict] = []
        self.active_config_id = ""
        self.run_settings: dict = {}
        self.path: Path | None = None
        self.dirty = False

    def close(self) -> None:
        self._temporary.cleanup()

    @property
    def config_base(self) -> Path:
        return self.root / "configs"

    def add_asset(self, source: Path, kind: str = "input") -> Asset:
        source = Path(source).resolve()
        if not source.is_file():
            raise ProjectError(f"Input file not found: {source}")
        # Copy first, then hash the snapshot rather than a changing external file.
        asset_id = uuid4().hex
        name = source.name
        relative = safe_member(f"assets/{asset_id}/{name}")
        target = self.root / relative
        target.parent.mkdir(parents=True)
        shutil.copyfile(source, target)
        digest = digest_file(target)
        for asset in self.assets.values():
            if asset.sha256 == digest and asset.original_name == name and asset.kind == kind:
                target.unlink()
                target.parent.rmdir()
                return asset
        asset = Asset(asset_id, relative, name, digest, target.stat().st_size, kind)
        self.assets[asset_id] = asset
        self.dirty = True
        return asset

    def _capture(self, data: dict, base: Path) -> dict:
        value = deepcopy(data)
        problems = missing_files(value, base)
        if problems:
            raise ProjectError("\n".join(problems))
        by_path = {(self.root / asset.path).resolve(): asset for asset in self.assets.values()}
        for mapping, key, _ in file_references(value):
            source = resolved_file(mapping[key], base)
            asset = by_path.get(source) or self.add_asset(source)
            mapping[key] = "../" + asset.path
        # Validate JSON before publishing a change to the project.
        json_bytes(value)
        return value

    def add_config(self, name: str, data: dict, base: Path) -> str:
        value = self._capture(data, base)
        config_id = uuid4().hex
        name = unique_name(Path(name).stem or "beam", [c.name for c in self.configs.values()])
        self.configs[config_id] = InputConfig(config_id, name, value)
        if not self.active_config_id:
            self.active_config_id = config_id
        self.dirty = True
        return config_id

    def update_config(self, config_id: str, data: dict, base: Path | None = None) -> None:
        value = self._capture(data, base or self.config_base)
        config = self.configs[config_id]
        if config.data != value:
            config.data = value
            self.dirty = True

    def add_recipe(self, recipe: dict, config_id: str) -> None:
        value = deepcopy(recipe)
        value["config_id"] = config_id
        sources = value.pop("source_files", {})
        value["source_assets"] = {key: self.add_asset(Path(path), "source").id
                                  for key, path in sources.items() if path}
        value.setdefault("id", uuid4().hex)
        value.setdefault("pass_version", __version__)
        self.recipes.append(value)
        self.dirty = True

    def _dependency_records(self, config: InputConfig) -> list[dict]:
        assets = {(self.root / a.path).resolve(): a.id for a in self.assets.values()}
        dependencies = []
        for mapping, key, pointer in file_references(config.data):
            value = mapping[key]
            if Path(value).is_absolute() or "\\" in value or ":" in value:
                raise ProjectError(f"{config.name}{pointer}: project input must reference an embedded asset")
            asset_id = assets.get(resolved_file(value, self.config_base))
            if not asset_id:
                raise ProjectError(f"{config.name}{pointer}: missing embedded asset {value}")
            dependencies.append({"pointer": pointer, "asset_id": asset_id})
        return dependencies

    def save(self, destination: Path) -> None:
        if not self.configs or self.active_config_id not in self.configs:
            raise ProjectError("Project needs an active input JSON")
        destination = Path(destination).resolve()
        entries: dict[str, bytes | Path] = {}
        configs = []
        for config in self.configs.values():
            path = f"configs/{config.id}.json"
            content = json_bytes(config.data)
            entries[path] = content
            configs.append({"id": config.id, "name": config.name, "path": path,
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "dependencies": self._dependency_records(config)})
        assets = []
        for asset in self.assets.values():
            source = self.root / asset.path
            if not source.is_file() or digest_file(source) != asset.sha256:
                raise ProjectError(f"Asset changed or missing: {asset.original_name}")
            entries[asset.path] = source
            assets.append(vars(asset).copy())
        recipes = []
        for index, recipe in enumerate(self.recipes):
            path = f"recipes/{index:04}.json"
            content = json_bytes(recipe)
            entries[path] = content
            recipes.append({"path": path, "sha256": hashlib.sha256(content).hexdigest()})
        manifest = {
            "format": "pass-project", "format_version": FORMAT_VERSION,
            "project_id": self.id, "created_at": self.created_at,
            "saved_with": {"pass_version": __version__},
            "active_config_id": self.active_config_id, "configs": configs,
            "assets": assets, "recipes": recipes, "run_settings": self.run_settings,
        }
        self._write_archive(destination, entries, manifest)
        self.path = destination
        self.dirty = False

    @staticmethod
    def _write_archive(destination: Path, entries: dict, manifest: dict | None = None) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                if manifest is not None:
                    archive.writestr("manifest.json", json_bytes(manifest))
                for name, value in entries.items():
                    safe_member(name)
                    if isinstance(value, Path):
                        archive.write(value, name)
                    else:
                        archive.writestr(name, value)
            with zipfile.ZipFile(temporary) as archive:
                bad = archive.testzip()
                if bad:
                    raise ProjectError(f"Archive verification failed: {bad}")
            with open(temporary, "r+b") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def open(cls, path: Path) -> Project:
        project = cls()
        try:
            project._load(Path(path))
            return project
        except Exception as exc:
            project.close()
            raise ProjectError(f"Cannot open project: {exc}") from exc

    def _load(self, path: Path) -> None:
        with zipfile.ZipFile(path) as archive:
            names = set()
            total = 0
            for info in archive.infolist():
                safe_member(info.filename)
                folded = info.filename.casefold()
                if folded in names or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
                    raise ProjectError(f"Duplicate, linked or encrypted member: {info.filename}")
                names.add(folded)
                total += info.file_size
            if len(names) > 100000 or total > 256 * 1024**3:
                raise ProjectError("Archive exceeds project size limits")

            def read_entry(name: str, digest: str | None = None) -> dict:
                safe_member(name)
                if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                    raise ProjectError(f"Invalid checksum: {name}")
                if archive.getinfo(name).file_size > JSON_LIMIT:
                    raise ProjectError(f"JSON member too large: {name}")
                content = archive.read(name)
                if digest and hashlib.sha256(content).hexdigest() != digest:
                    raise ProjectError(f"Checksum mismatch: {name}")
                return read_json(content)

            manifest = read_entry("manifest.json")
            if manifest.get("format") != "pass-project" or type(manifest.get("format_version")) is not int or manifest["format_version"] != FORMAT_VERSION:
                raise ProjectError("Unsupported PASS project format/version")
            self.id = str(manifest["project_id"])
            self.created_at = manifest["created_at"]
            registered = {"manifest.json"}
            def register(name):
                safe_member(name)
                if name in registered:
                    raise ProjectError(f"Member used more than once: {name}")
                registered.add(name)
            for entry in manifest["assets"]:
                asset = Asset(**entry)
                if asset.id in self.assets or not re.fullmatch(r"[a-zA-Z0-9_-]+", asset.id):
                    raise ProjectError("Invalid or duplicate asset ID")
                if not re.fullmatch(r"[0-9a-f]{64}", asset.sha256):
                    raise ProjectError("Invalid asset checksum")
                register(asset.path)
                if not asset.path.startswith(f"assets/{asset.id}/") or archive.getinfo(asset.path).file_size != asset.size_bytes:
                    raise ProjectError(f"Asset metadata mismatch: {asset.path}")
                target = self.root / asset.path
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(asset.path) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                if digest_file(target) != asset.sha256:
                    raise ProjectError(f"Checksum mismatch: {asset.path}")
                self.assets[asset.id] = asset
            for entry in manifest["configs"]:
                identifier = entry["id"]
                if identifier in self.configs or not re.fullmatch(r"[a-zA-Z0-9_-]+", identifier):
                    raise ProjectError("Invalid or duplicate config ID")
                if entry["path"] != f"configs/{identifier}.json":
                    raise ProjectError("Invalid config path")
                register(entry["path"])
                data = read_entry(entry["path"], entry["sha256"])
                config = InputConfig(identifier, str(entry["name"]), data)
                self.configs[identifier] = config
                if self._dependency_records(config) != entry["dependencies"]:
                    raise ProjectError(f"Dependency index mismatch: {config.name}")
            for entry in manifest["recipes"]:
                register(entry["path"])
                if not entry["path"].startswith("recipes/"):
                    raise ProjectError("Invalid recipe path")
                recipe = read_entry(entry["path"], entry["sha256"])
                if recipe["config_id"] not in self.configs or any(i not in self.assets for i in recipe.get("source_assets", {}).values()):
                    raise ProjectError("Recipe references unknown config or source")
                self.recipes.append(recipe)
            if {name.casefold() for name in registered} != names:
                raise ProjectError("Archive contains unregistered members")
            self.active_config_id = manifest["active_config_id"]
            if self.active_config_id not in self.configs:
                raise ProjectError("Active input not found")
            self.run_settings = manifest.get("run_settings", {})
            if not isinstance(self.run_settings, dict):
                raise ProjectError("Run settings must be an object")
        self.path = path.resolve()
        self.dirty = False

    def materialize(self, config_ids: list[str], destination: Path, output: Path) -> list[Path]:
        """Write a run snapshot with absolute input paths, without mutating configs."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, identifier in enumerate(config_ids):
            config = self.configs[identifier]
            self._dependency_records(config)
            data = deepcopy(config.data)
            for mapping, key, _ in file_references(data):
                source = resolved_file(mapping[key], self.config_base)
                relative = source.relative_to(self.root)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source, target)
                mapping[key] = str(target.resolve())
            data["Output directory"] = str(Path(output).resolve())
            target = destination / f"beam{index}.json"
            atomic_write(target, json_bytes(data))
            paths.append(target)
        return paths

    def export_bundle(self, config_ids: list[str], destination: Path) -> None:
        """Export portable JSON + assets with a launcher resolving paths at run time."""
        entries: dict[str, bytes | Path] = {}
        for index, identifier in enumerate(config_ids):
            config = self.configs[identifier]
            dependencies = self._dependency_records(config)
            data = deepcopy(config.data)
            for mapping, key, _ in file_references(data):
                mapping[key] = mapping[key].removeprefix("../")
            data["Output directory"] = "output"
            entries[f"beam{index}.json"] = json_bytes(data)
            for record in dependencies:
                asset = self.assets[record["asset_id"]]
                entries[asset.path] = self.root / asset.path
        # Engine readers historically use process-relative paths. A controlled
        # cwd makes this extracted package directly runnable with existing PASS.
        entries["run.py"] = (
            '"""Run this exported PASS input bundle: python run.py."""\n'
            'import os\nfrom pathlib import Path\nfrom PASS.gui.runner import run_inputs\n'
            'root = Path(__file__).resolve().parent\nos.chdir(root)\n'
            + ('raise SystemExit(run_inputs(str(root / "beam0.json"), str(root / "beam1.json")))\n' if len(config_ids) == 2
               else 'raise SystemExit(run_inputs(str(root / "beam0.json")))\n')
        ).encode("utf-8")
        entries["README.txt"] = b"Extract all files together. Install PASS, then run: python run.py\nInputs use paths relative to this folder. Outputs are written under output/.\n"
        self._write_archive(Path(destination), entries)

    def copy_command(self, source_id: str, name: str, target: Project, target_id: str) -> str:
        """Copy a command and its named SC/Slicer/file dependencies without overwrite."""
        source = self.configs[source_id].data
        result = deepcopy(target.configs[target_id].data)
        sequence = result.setdefault("Sequence", {})
        command = deepcopy(source["Sequence"][name])
        config_names = {}
        slice_names = {}

        def copy_slice(slice_name):
            if slice_name in slice_names:
                return slice_names[slice_name]
            slicers = [(n, item) for n, item in source.get("Sequence", {}).items()
                       if isinstance(item, dict) and item.get("Command") == "Slicer" and item.get("Slice set") == slice_name]
            if not slicers:
                raise ProjectError(f"Missing Slicer dependency: {slice_name}")
            used = [item.get("Slice set") for item in sequence.values() if isinstance(item, dict) and item.get("Command") == "Slicer"]
            replacement = unique_name(slice_name, used)
            slice_names[slice_name] = replacement
            for slicer_name, item in slicers:
                item = target._capture(item, self.config_base)
                item["Slice set"] = replacement
                sequence[unique_name(slicer_name, sequence)] = item
            return replacement

        def visit(value):
            if isinstance(value, dict):
                if "Configuration" in value:
                    old = value["Configuration"]
                    if old not in config_names:
                        resource = source.get("Space charge", {}).get("Configurations", {}).get(old)
                        if not isinstance(resource, dict):
                            raise ProjectError(f"Missing Space charge configuration: {old}")
                        block = result.setdefault("Space charge", {})
                        resources = block.setdefault("Configurations", {})
                        new = unique_name(old, resources)
                        copied = target._capture(resource, self.config_base)
                        copied["Slice set"] = copy_slice(copied["Slice set"])
                        resources[new] = copied
                        # Preserve target physics switches; create a usable block
                        # only when the target has no module settings yet.
                        block.setdefault("Enabled", source.get("Space charge", {}).get("Enabled", True))
                        config_names[old] = new
                    value["Configuration"] = config_names[old]
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(command)
        command = target._capture(command, self.config_base)
        new_name = unique_name(name, sequence)
        sequence[new_name] = command
        target.update_config(target_id, result)
        return new_name
