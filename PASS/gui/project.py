"""Editable directory projects, with read-only compatibility for ZIP projects.

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
from uuid import uuid4
import zipfile

from PASS import __version__
from PASS.utils.input_snapshot import atomic_write, file_references, json_bytes, resolved_file, unique_filename

FORMAT_VERSION = 3
JSON_LIMIT = 64 * 1024 * 1024


def input_file_fields():
    """Resolve the shared field catalog only after GUI schema preparation."""
    from PASS.validation.files import INPUT_FILE_FIELDS
    return INPUT_FILE_FIELDS


def __getattr__(name):
    if name == "FILE_FIELDS":
        return input_file_fields()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ProjectError(ValueError):
    """An incomplete, unsupported or malformed project."""


def read_json(content: bytes) -> dict:
    from PASS.validation.report import parse_json
    value, report = parse_json(content)
    if not report.ok:
        if any(issue.code == "json.syntax" for issue in report.errors):
            # Preserve JSONDecodeError's line/column for the source editor.
            json.loads(content.decode("utf-8-sig"))
        raise ProjectError(report.text())
    return value


def normalize_command(data: dict) -> dict:
    """Copy one command using canonical schema keys without filling defaults."""
    from PASS.validation.report import ValidationReport
    from PASS.validation.rules import MODELS, _normalize_fields

    kind = next((value for key, value in data.items() if isinstance(key, str) and key.casefold() == "command"), None)
    canonical = next((name for name in MODELS if isinstance(kind, str) and name.casefold() == kind.casefold()), None)
    report = ValidationReport(full=False)
    annotation = MODELS[canonical] if canonical is not None else {"Command": str}
    normalized = _normalize_fields(data, annotation, {}, report, python_names=True)
    if not report.ok:
        raise ProjectError(report.text())
    if canonical is not None:
        normalized["Command"] = canonical
    return normalized


def normalize_input(data: dict) -> dict:
    """Canonicalize input aliases while preserving names, unknown fields and values."""
    from PASS.para.schema.main import MainConfig
    from PASS.validation.report import ValidationReport
    from PASS.validation.rules import _normalize_fields

    if not isinstance(data, dict):
        raise ProjectError("JSON 根节点必须是对象")
    report = ValidationReport(full=False)
    normalized = _normalize_fields(data, MainConfig, {}, report, python_names=True)
    if not report.ok:
        raise ProjectError(report.text())
    sequence = normalized.get("Sequence")
    if isinstance(sequence, dict):
        normalized["Sequence"] = {name: normalize_command(command) if isinstance(command, dict) else command for name, command in sequence.items()}
    return normalized


def missing_files(data: dict, base: Path) -> list[str]:
    return [
        f"{pointer}: input file not found: {mapping[key]}" for mapping, key, pointer in file_references(data)
        if not resolved_file(mapping[key], base).is_file()
    ]


def digest_file(path: Path, *, context=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if context is not None:
                context.check()
            digest.update(block)
    return digest.hexdigest()


def unique_name(name: str, existing) -> str:
    names = {str(n).casefold() for n in existing}
    candidate, suffix = name, 2
    while candidate.casefold() in names:
        candidate = f"{name}_{suffix}"
        suffix += 1
    return candidate


def _flat_member(name: str, directory: str) -> str:
    """Require a direct file within the format's named directory."""
    safe_member(name)
    path = PurePosixPath(name)
    if len(path.parts) != 2 or path.parts[0] != directory:
        raise ProjectError(f"Expected a file directly in {directory}/: {name}")
    return name


def _electron_cloud_block(data):
    """Read cloud aliases without validating away disabled configuration drafts."""
    from PASS.para.schema.electron_cloud import ElectronCloudConfig
    keys = [key for key in data if str(key).casefold() == "electron cloud"]
    if len(keys) > 1:
        raise ProjectError("Duplicate Electron cloud input blocks")
    key = keys[0] if keys else "Electron cloud"
    raw = data.get(key, {})
    if not isinstance(raw, dict):
        raise ProjectError("Electron cloud must be an object")
    normalized = ElectronCloudConfig._normalize_fields(deepcopy(raw))
    block = {
        (ElectronCloudConfig.model_fields[name].alias if name in ElectronCloudConfig.model_fields else name): value
        for name, value in normalized.items()
    }
    return key, block


def _electron_cloud_mode(resource):
    from PASS.para.schema.electron_cloud import ElectronCloudConfiguration
    return ElectronCloudConfiguration._normalize_fields(deepcopy(resource)).get("mode", "frozen") if isinstance(resource, dict) else "frozen"


def safe_member(name: str) -> str:
    if not isinstance(name, str):
        raise ProjectError("Project file paths must be strings")
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or "\x00" in name or path.is_absolute() or any(part in (".", "..", "") or part.endswith(
        (".", " ")) or re.search(r'[<>"|?*]', part) or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part)
                                                                                               for part in name.split("/"))):
        raise ProjectError(f"Invalid archive path: {name!r}")
    return name


def _check_project_links(root: Path, target: Path, name: str) -> None:
    for current in (target, *target.parents):
        if current == root:
            break
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ProjectError(f"Linked project paths are not supported: {name}")


def _directory_member(root: Path, name: str) -> Path:
    """Resolve a portable member without following links out of the project."""
    safe_member(name)
    root = root.resolve()
    target = root / name
    if not target.resolve().is_relative_to(root):
        raise ProjectError(f"Project path escapes its directory: {name}")
    _check_project_links(root, target, name)
    return target


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
    """One project and an immutable private cache for GUI edits and run snapshots."""

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
        self.legacy = False
        self.name = ""
        self._config_paths: dict[str, str] = {}
        self._asset_paths: dict[str, str] = {}
        self._recipe_paths: dict[str, str] = {}
        self._disk_files: dict[str, str] = {}
        self._manifest_extra: dict = {}
        self.dirty = False
        self._leases = 0
        self._close_requested = False
        self._cache_closed = False

    def close(self) -> None:
        self._close_requested = True
        if not self._leases and not self._cache_closed:
            self._temporary.cleanup()
            self._cache_closed = True

    def retain(self):
        """Keep immutable input files alive while a background task reads them."""
        if self._cache_closed or self._close_requested:
            raise ProjectError("Project cache is closing")
        self._leases += 1
        return self

    def release(self):
        if self._leases <= 0:
            raise ProjectError("Unbalanced project cache release")
        self._leases -= 1
        if self._close_requested:
            self.close()

    def clone(self, *, context=None):
        """Copy into a private staging cache; callers publish only on success."""
        candidate = Project()
        try:
            for source in self.root.rglob("*"):
                if source.is_file():
                    target = candidate.root / source.relative_to(self.root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if context is not None:
                        context.report(f"准备项目文件：{source.name}")
                    # Imported assets are immutable. A staging cache can share
                    # their file storage; new imports always use unused paths.
                    if source.relative_to(self.root).parts[0] == "assets":
                        try:
                            os.link(source, target)
                            continue
                        except OSError:
                            pass
                    if context is not None:
                        context.copy_file(source, target)
                    else:
                        shutil.copyfile(source, target)
            for key in ("id", "created_at", "configs", "assets", "recipes", "active_config_id", "run_settings", "path", "dirty", "legacy", "name",
                        "_config_paths", "_asset_paths", "_recipe_paths", "_disk_files", "_manifest_extra"):
                setattr(candidate, key, deepcopy(getattr(self, key)))
        except Exception:
            candidate.close()
            raise
        return candidate

    @property
    def config_base(self) -> Path:
        return self.root / "configs"

    def add_asset(self, source: Path, kind: str = "input", *, context=None) -> Asset:
        source = Path(source).resolve()
        if not source.is_file():
            raise ProjectError(f"Input file not found: {source}")
        for asset in self.assets.values():
            known_paths = [(self.root / asset.path).resolve()]
            if self.path is not None and not self.legacy and asset.id in self._asset_paths:
                known_paths.append((self.path.parent / self._asset_paths[asset.id]).resolve())
            if source in known_paths:
                if digest_file(source, context=context) != asset.sha256:
                    raise ProjectError(f"Project file changed while importing: {source.name}. Reload the project.")
                # A dependency can also be the source of a generator recipe.
                if asset.kind == "input" and kind != "input":
                    asset.kind = kind
                    self.dirty = True
                return asset
        # Copy first, then hash the snapshot rather than a changing external file.
        asset_id = uuid4().hex
        name = source.name
        directory = self.root / "assets"
        existing = [PurePosixPath(asset.path).name for asset in self.assets.values()]
        if directory.exists():
            existing.extend(path.name for path in directory.iterdir())
        relative = f"assets/{unique_filename(name, existing)}"
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if context is not None:
            context.report(f"复制依赖：{name}")
            context.copy_file(source, target)
        else:
            shutil.copyfile(source, target)
        digest = digest_file(target, context=context)
        for asset in self.assets.values():
            if asset.sha256 == digest and asset.original_name == name and asset.kind == kind:
                target.unlink()
                self._remember_asset_path(asset, source)
                return asset
        asset = Asset(asset_id, relative, name, digest, target.stat().st_size, kind)
        self.assets[asset_id] = asset
        self._remember_asset_path(asset, source)
        self.dirty = True
        return asset

    def _remember_asset_path(self, asset: Asset, source: Path) -> None:
        """Keep files explicitly added inside an existing project at their paths."""
        if self.path is None or self.legacy or asset.id in self._asset_paths or not source.is_relative_to(self.path.parent):
            return
        name = source.relative_to(self.path.parent).as_posix()
        reserved = {
            self.path.name.casefold(), *(path.casefold() for path in self._config_paths.values()),
            *(path.casefold() for path in self._recipe_paths.values())
        }
        if name.casefold() in reserved:
            return
        _directory_member(self.path.parent, name)
        if digest_file(source) != asset.sha256:
            raise ProjectError(f"Project file changed while importing: {name}")
        self._asset_paths[asset.id] = name
        self._disk_files[str(source)] = asset.sha256

    def _capture(self, data: dict, base: Path, *, context=None) -> dict:
        # Commands and resource fragments also use this path; retain their schema context.
        value = deepcopy(data)
        problems = missing_files(value, base)
        if problems:
            raise ProjectError("\n".join(problems))
        by_path = {(self.root / asset.path).resolve(): asset for asset in self.assets.values()}
        if self.path is not None and not self.legacy:
            by_path.update({
                (self.path.parent / name).resolve(): self.assets[identifier]
                for identifier, name in self._asset_paths.items() if identifier in self.assets
            })
        for mapping, key, _ in file_references(value):
            source = resolved_file(mapping[key], base)
            asset = by_path.get(source) or self.add_asset(source, context=context)
            mapping[key] = "../" + asset.path
        # Validate JSON before publishing a change to the project.
        json_bytes(value)
        return value

    def editable_input(self, config_id: str, *, data: dict | None = None) -> tuple[dict, Path]:
        """Expose real, editable paths while retaining immutable cached inputs."""
        data = deepcopy(self.configs[config_id].data if data is None else data)
        if self.path is None or self.legacy or config_id not in self._config_paths:
            return data, self.config_base
        base = (self.path.parent / self._config_paths[config_id]).parent
        assets = {(self.root / asset.path).resolve(): identifier for identifier, asset in self.assets.items()}
        for mapping, key, _ in file_references(data):
            identifier = assets.get(resolved_file(mapping[key], self.config_base))
            if identifier in self._asset_paths:
                mapping[key] = Path(os.path.relpath(self.path.parent / self._asset_paths[identifier], base)).as_posix()
            else:
                # New unsaved imports still live in the private staging cache.
                mapping[key] = str(resolved_file(mapping[key], self.config_base))
        return data, base

    def add_config(self, name: str, data: dict, base: Path, *, context=None) -> str:
        value = self._capture(normalize_input(data), base, context=context)
        config_id = uuid4().hex
        filename = unique_filename(f"{Path(name).stem or 'beam'}.json", [f"{c.name}.json" for c in self.configs.values()])
        name = PurePosixPath(filename).stem
        self.configs[config_id] = InputConfig(config_id, name, value)
        if not self.active_config_id:
            self.active_config_id = config_id
        self.dirty = True
        return config_id

    def update_config(self, config_id: str, data: dict, base: Path | None = None, *, context=None) -> None:
        value = self._capture(normalize_input(data), base or self.config_base, context=context)
        config = self.configs[config_id]
        if config.data != value:
            config.data = value
            self.dirty = True

    def add_recipe(self, recipe: dict, config_id: str, *, context=None) -> None:
        value = deepcopy(recipe)
        identifier = value.setdefault("id", uuid4().hex)
        if not isinstance(identifier, str) or not identifier or any(existing.get("id") == identifier for existing in self.recipes):
            raise ProjectError("Invalid or duplicate recipe ID")
        value["config_id"] = config_id
        sources = value.pop("source_files", {})
        value["source_assets"] = {key: self.add_asset(Path(path), "source", context=context).id for key, path in sources.items() if path}
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

    def save(self, destination: Path, *, context=None) -> None:
        """Save ordinary files and publish the v3 manifest after successful writes."""
        if not self.configs or self.active_config_id not in self.configs:
            raise ProjectError("Project needs an active input JSON")
        self.check_external_changes(context=context)
        destination = Path(destination).expanduser()
        if destination.is_dir() or not destination.suffix:
            destination /= "project.passproj"
        if destination.suffix.lower() != ".passproj":
            raise ProjectError("A project entry must use the .passproj extension")
        if destination.is_symlink():
            raise ProjectError("Cannot save through a linked project entry")
        destination = destination.resolve()
        if self.legacy and self.path is not None and destination.parent == self.path.parent:
            raise ProjectError("Save the legacy ZIP project into a new directory; the original archive is preserved")
        if destination.exists() and (destination != self.path or self.legacy):
            raise ProjectError(f"Refusing to overwrite another project: {destination}")
        root = destination.parent
        used = {destination.name.casefold()}
        directories = set()
        source_directory = None
        config_paths = {}
        asset_paths = {}
        recipe_paths = {}
        recipe_ids = set()
        for recipe in self.recipes:
            identifier = recipe.setdefault("id", uuid4().hex)
            if not isinstance(identifier, str) or not identifier or identifier in recipe_ids:
                raise ProjectError("Invalid or duplicate recipe ID")
            recipe_ids.add(identifier)

        def allocate(preferred, fallback, directory=""):
            if preferred:
                name = safe_member(preferred)
                if name.casefold() in used:
                    raise ProjectError(f"Duplicate project file path: {name}")
            else:
                parent = PurePosixPath(directory)
                existing = [
                    PurePosixPath(path).name for path in used | directories if PurePosixPath(path).parent == PurePosixPath(directory.casefold())
                ]
                if (root / directory).is_dir():
                    existing.extend(path.name for path in (root / directory).iterdir())
                name = (parent / unique_filename(fallback, existing)).as_posix()
            parents = [path.as_posix().casefold() for path in PurePosixPath(name).parents if path != PurePosixPath(".")]
            if name.casefold() in directories or any(parent in used for parent in parents):
                raise ProjectError(f"Project file/directory path conflict: {name}")
            target = _directory_member(root, name)
            if target.exists() and not target.is_file():
                raise ProjectError(f"Project destination is not a file: {target}")
            for parent in target.parents:
                if parent == root:
                    break
                if parent.exists() and not parent.is_dir():
                    raise ProjectError(f"Project parent path is not a directory: {parent}")
            used.add(name.casefold())
            directories.update(parents)
            return name

        def allocate_asset(name):
            nonlocal source_directory
            name = unique_filename(name, ())
            if PurePosixPath(name).suffix.lower() != ".passproj":
                return allocate(None, name)
            if source_directory is None:
                existing = list(used)
                if root.is_dir():
                    existing.extend(path.name for path in root.iterdir()
                                    if not path.is_dir() or path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()))
                source_directory = unique_filename("sources", existing)
                _directory_member(root, source_directory)
            # Auxiliary project descriptions must not participate in root discovery.
            return allocate(None, name, source_directory)

        def retain_path(name):
            member = PurePosixPath(name)
            if len(member.parts) == 1 and member.suffix.lower() == ".passproj":
                if self.path is None or root != self.path.parent:
                    # Save As can relocate the copy without removing the original.
                    return False
                if destination != root / "project.passproj":
                    raise ProjectError(f"Auxiliary project file {name} conflicts with directory project selection. "
                                       "Move it into a subdirectory, update its project references and reload, "
                                       "or Save As into a new directory.")
            return True

        # Reserve retained paths first so newly imported files cannot take them.
        for identifier in self.configs:
            if identifier in self._config_paths:
                config_paths[identifier] = allocate(self._config_paths[identifier], "")
        for identifier in self.assets:
            if identifier in self._asset_paths:
                name = self._asset_paths[identifier]
                if retain_path(name):
                    asset_paths[identifier] = allocate(name, "")
        for recipe in self.recipes:
            identifier = recipe.get("id")
            if identifier in self._recipe_paths and retain_path(self._recipe_paths[identifier]):
                recipe_paths[identifier] = allocate(self._recipe_paths[identifier], "")
        for identifier, config in self.configs.items():
            if identifier not in config_paths:
                config_paths[identifier] = allocate(None, f"{config.name}.json")
        for identifier, asset in self.assets.items():
            if identifier not in asset_paths:
                asset_paths[identifier] = allocate_asset(asset.original_name)
        for index, recipe in enumerate(self.recipes):
            identifier = recipe.setdefault("id", uuid4().hex)
            if identifier not in recipe_paths:
                recipe_paths[identifier] = allocate(None, f"recipe{index + 1}.json")

        entries: dict[str, bytes | Path] = {}
        for config in self.configs.values():
            if context is not None:
                context.report(f"整理输入：{config.name}")
            self._dependency_records(config)
            data = deepcopy(config.data)
            by_path = {(self.root / asset.path).resolve(): identifier for identifier, asset in self.assets.items()}
            base = (root / config_paths[config.id]).parent
            for mapping, key, _ in file_references(data):
                identifier = by_path[resolved_file(mapping[key], self.config_base)]
                mapping[key] = Path(os.path.relpath(root / asset_paths[identifier], base)).as_posix()
            entries[config_paths[config.id]] = json_bytes(data)
        for asset in self.assets.values():
            if context is not None:
                context.report(f"验证依赖：{asset.original_name}")
            source = self.root / asset.path
            if not source.is_file() or digest_file(source, context=context) != asset.sha256:
                raise ProjectError(f"Asset changed or missing: {asset.original_name}")
            entries[asset_paths[asset.id]] = source
        for recipe in self.recipes:
            value = deepcopy(recipe)
            try:
                value["config"] = config_paths[value.pop("config_id")]
                value["source_files"] = {key: asset_paths[identifier] for key, identifier in value.pop("source_assets", {}).items()}
            except KeyError as exc:
                raise ProjectError("Recipe references an unknown input or source") from exc
            entries[recipe_paths[recipe["id"]]] = json_bytes(value)
        settings = deepcopy(self.run_settings)
        for key in ("beam0", "beam1"):
            identifier = settings.get(key)
            if identifier not in (None, ""):
                if identifier not in config_paths:
                    raise ProjectError(f"Unknown project run input: {key}")
                settings[key] = config_paths[identifier]
        manifest = deepcopy(self._manifest_extra)
        manifest.update({
            "format": "pass-project",
            "format_version": FORMAT_VERSION,
            "name": self.name or destination.parent.name,
            "configs": list(config_paths.values()),
            "active_config": config_paths[self.active_config_id],
            "run": settings,
        })
        # Auxiliary originals and recipes remain editable, without a hash registry.
        sources = [{"path": asset_paths[asset.id], "kind": asset.kind} for asset in self.assets.values() if asset.kind != "input"]
        referenced = {record["asset_id"] for config in self.configs.values() for record in self._dependency_records(config)}
        sources.extend({
            "path": asset_paths[asset.id],
            "kind": asset.kind
        } for asset in self.assets.values() if asset.kind == "input" and asset.id not in referenced)
        if sources:
            manifest["sources"] = sources
        if self.recipes:
            manifest["recipes"] = [recipe_paths[recipe["id"]] for recipe in self.recipes]
        entries[destination.name] = json_bytes(manifest)
        expected_files = self._write_directory(destination, entries, context=context)
        self.path = destination
        self.legacy = False
        self.name = manifest["name"]
        self._config_paths = config_paths
        self._asset_paths = asset_paths
        self._recipe_paths = recipe_paths
        self._disk_files = expected_files
        # Publication is complete; a late cancellation must not report it undone.
        self.check_external_changes()
        self.dirty = False

    @staticmethod
    def resolve_path(path: Path) -> Path:
        """Prefer the standard entry, otherwise require one unambiguous project."""
        path = Path(path).expanduser()
        if path.is_dir():
            default = path / "project.passproj"
            if default.exists():
                if not default.is_file():
                    raise ProjectError(f"Project entry is not a file: {default}")
                return default.resolve()
            candidates = sorted((entry for entry in path.iterdir() if entry.is_file() and entry.suffix.lower() == ".passproj"),
                                key=lambda entry: entry.name.casefold())
            if not candidates:
                raise ProjectError(f"No .passproj project file found in directory: {path}. Create project.passproj or select a project file.")
            if len(candidates) != 1:
                raise ProjectError(f"Multiple .passproj project files found in {path}; specify the project file explicitly")
            path = candidates[0]
        return path.resolve()

    def check_external_changes(self, *, context=None, expected_files: dict[str, str] | None = None) -> None:
        """Reject stale GUI caches rather than overwrite or run unseen disk edits."""
        files = tuple((self._disk_files if expected_files is None else expected_files).items())
        for name, expected in files:
            if context is not None:
                context.report(f"检查外部修改：{Path(name).name}")
            try:
                changed = digest_file(Path(name), context=context) != expected
            except OSError:
                changed = True
            if changed:
                raise ProjectError(f"Project file changed or missing on disk: {name}. Reload the project before saving or running. "
                                   "工程文件已在外部修改或移除，请重新加载后再保存或运行。")

    def _write_directory(self, destination: Path, entries: dict, *, context=None) -> dict[str, str]:
        root = destination.parent
        same_directory = self.path is not None and not self.legacy and root == self.path.parent
        for name, value in entries.items():
            target = _directory_member(root, name)
            # Save As must not overwrite source members when the directories overlap.
            if target.exists() and (not same_directory or str(target) not in self._disk_files):
                raise ProjectError(f"Refusing to overwrite an unrelated file: {target}")
            if target.exists() and not target.is_file():
                raise ProjectError(f"Project destination is not a file: {target}")
        root.mkdir(parents=True, exist_ok=True)
        # Stage all data and backups before changing any existing project member.
        with tempfile.TemporaryDirectory(prefix=".pass-save-", dir=root) as temporary:
            staging = Path(temporary)
            backups = {}
            written = []
            expected_files = {}
            for index, (name, value) in enumerate(entries.items()):
                if context is not None:
                    context.report(f"准备保存：{name}")
                target = _directory_member(root, name)
                staged = staging / str(index)
                if isinstance(value, Path):
                    if context is None:
                        shutil.copyfile(value, staged)
                    else:
                        context.copy_file(value, staged)
                else:
                    staged.write_bytes(value)
                expected_files[str(target)] = digest_file(staged, context=context)
                if target.exists():
                    backup = staging / f"backup-{index}"
                    shutil.copyfile(target, backup)
                    backups[name] = backup
            self.check_external_changes(context=context)
            try:
                for index, name in enumerate(entries):
                    if context is not None:
                        context.report(f"保存项目：{name}")
                    target = _directory_member(root, name)
                    if target.exists() and name not in backups:
                        raise ProjectError(f"Project destination appeared while saving: {target}")
                    if name in backups:
                        if not target.is_file() or digest_file(target) != self._disk_files[str(target)]:
                            raise ProjectError(f"Project file changed while saving: {target}. Reload the project.")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if name in backups:
                        os.replace(staging / str(index), target)
                    else:
                        # Exclusive creation also protects files appearing after the check.
                        with (staging / str(index)).open("rb") as source, target.open("xb") as output:
                            shutil.copyfileobj(source, output, 1024 * 1024)
                            output.flush()
                            os.fsync(output.fileno())
                    written.append(name)
            except Exception:
                # Restore previous members; new files are kept for inspection.
                for name in reversed(written):
                    target = root / name
                    if name in backups and target.is_file() and digest_file(target) == expected_files[str(target)]:
                        os.replace(backups[name], root / name)
                raise
        return expected_files

    @staticmethod
    def _write_archive(destination: Path, entries: dict, manifest: dict | None = None, *, context=None) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                if manifest is not None:
                    archive.writestr("manifest.json", json_bytes(manifest))
                for name, value in entries.items():
                    if context is not None:
                        context.report(f"写入项目：{name}")
                    safe_member(name)
                    if isinstance(value, Path) and context is not None:
                        with value.open("rb") as reader, archive.open(name, "w", force_zip64=True) as writer:
                            while block := reader.read(1024 * 1024):
                                context.check()
                                writer.write(block)
                    elif isinstance(value, Path):
                        archive.write(value, name)
                    else:
                        archive.writestr(name, value)
            with zipfile.ZipFile(temporary) as archive:
                if context is None:
                    bad = archive.testzip()
                    if bad:
                        raise ProjectError(f"Archive verification failed: {bad}")
                else:
                    for name in archive.namelist():
                        context.report(f"校验项目：{name}")
                        with archive.open(name) as stream:
                            while stream.read(1024 * 1024):
                                context.check()
            with open(temporary, "r+b") as stream:
                os.fsync(stream.fileno())
            if context is not None:
                context.check()
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def open(cls, path: Path) -> Project:
        project = cls()
        try:
            project._load(cls.resolve_path(path))
            return project
        except Exception as exc:
            project.close()
            raise ProjectError(f"Cannot open project: {exc}") from exc

    def _load(self, path: Path) -> None:
        if zipfile.is_zipfile(path):
            self._load_legacy(path)
        else:
            self._load_directory(path)

    def _load_directory(self, path: Path) -> None:
        root = path.parent
        records = {}
        total = 0
        registered = {path.name.casefold(): "manifest"}
        disk_assets = {}

        def read_member(name):
            nonlocal total
            target = _directory_member(root, name)
            if not target.is_file():
                raise ProjectError(f"Project file not found: {name}")
            if target.stat().st_size > JSON_LIMIT:
                raise ProjectError(f"JSON member too large: {name}")
            content = target.read_bytes()
            if len(content) > JSON_LIMIT:
                raise ProjectError(f"JSON member too large: {name}")
            total += len(content)
            if total > 256 * 1024**3 or len(records) >= 100000:
                raise ProjectError("Project exceeds file count or size limits")
            records[str(target)] = hashlib.sha256(content).hexdigest()
            return read_json(content)

        manifest = read_member(path.name)
        if (manifest.get("format") != "pass-project" or type(manifest.get("format_version")) is not int
                or manifest["format_version"] != FORMAT_VERSION):
            raise ProjectError("Unsupported PASS project format/version")
        self.name = manifest.get("name", root.name)
        if not isinstance(self.name, str):
            raise ProjectError("Project name must be a string")
        self._manifest_extra = {
            key: deepcopy(value)
            for key, value in manifest.items()
            if key not in {"format", "format_version", "name", "configs", "active_config", "run", "sources", "recipes"}
        }
        settings = manifest.get("run", {})
        if not isinstance(settings, dict):
            raise ProjectError("Project run settings must be an object")
        names = manifest.get("configs")
        if names is None:
            names = []
            for name in (manifest.get("active_config"), settings.get("beam0"), settings.get("beam1")):
                if name not in (None, "") and name not in names:
                    names.append(name)
        if not isinstance(names, list) or not names or len(names) > 100000:
            raise ProjectError("Project configs must be a nonempty list of JSON paths")
        identifiers = {}
        for name in names:
            safe_member(name)
            if PurePosixPath(name).suffix.lower() != ".json":
                raise ProjectError(f"Project input must be a JSON file: {name}")
            if name.casefold() in registered:
                raise ProjectError(f"Duplicate project member: {name}")
            registered[name.casefold()] = "config"
            identifier = uuid4().hex
            identifiers[name] = identifier
            self._config_paths[identifier] = name

        def load_asset(name, kind="input"):
            nonlocal total
            target = _directory_member(root, name)
            folded = name.casefold()
            if folded in disk_assets:
                asset = self.assets[disk_assets[folded]]
                original = _directory_member(root, self._asset_paths[asset.id])
                if not target.is_file():
                    raise ProjectError(f"Project input file not found: {name}")
                if not target.samefile(original):
                    raise ProjectError(f"Project input paths differ only by case: {self._asset_paths[asset.id]} and {name}")
                if kind != "input":
                    asset.kind = kind
                return asset
            if folded in registered:
                raise ProjectError(f"Project member has conflicting uses: {name}")
            if not target.is_file():
                raise ProjectError(f"Project input file not found: {name}")
            total += target.stat().st_size
            if total > 256 * 1024**3 or len(registered) >= 100000:
                raise ProjectError("Project exceeds file count or size limits")
            registered[folded] = "asset"
            identifier = uuid4().hex
            internal = "assets/" + unique_filename(target.name, [PurePosixPath(asset.path).name for asset in self.assets.values()])
            cached = self.root / internal
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, cached)
            digest = digest_file(cached)
            if digest_file(target) != digest:
                raise ProjectError(f"Project file changed while loading: {name}")
            asset = Asset(identifier, internal, target.name, digest, cached.stat().st_size, kind)
            self.assets[identifier] = asset
            self._asset_paths[identifier] = name
            records[str(target)] = digest
            disk_assets[folded] = identifier
            return asset

        sources = manifest.get("sources", [])
        if not isinstance(sources, list):
            raise ProjectError("Project sources must be a list")
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("kind", "source"), str):
                raise ProjectError("Project sources need a path and a string kind")
            load_asset(source.get("path"), source.get("kind", "source"))
        for name, identifier in identifiers.items():
            data = normalize_input(read_member(name))
            base = (root / name).parent
            for mapping, key, pointer in file_references(data):
                value = mapping[key]
                if Path(value).is_absolute():
                    raise ProjectError(f"{name}{pointer}: project dependencies must use relative paths")
                raw_source = base / value
                source = Path(os.path.normpath(raw_source))
                if not source.is_relative_to(root):
                    raise ProjectError(f"{name}{pointer}: dependency is outside the project directory")
                # Parent traversal must not hide a link in the original spelling.
                _check_project_links(root, raw_source, value)
                member = source.relative_to(root).as_posix()
                asset = load_asset(member)
                mapping[key] = "../" + asset.path
            display = unique_name(PurePosixPath(name).stem, [config.name for config in self.configs.values()])
            self.configs[identifier] = InputConfig(identifier, display, data)
        recipe_names = manifest.get("recipes", [])
        if not isinstance(recipe_names, list):
            raise ProjectError("Project recipes must be a list of JSON paths")
        for name in recipe_names:
            safe_member(name)
            if name.casefold() in registered:
                raise ProjectError(f"Duplicate project member: {name}")
            registered[name.casefold()] = "recipe"
            recipe = read_member(name)
            config = recipe.pop("config", None)
            if not isinstance(config, str) or config not in identifiers:
                raise ProjectError("Recipe references an unknown config")
            recipe["config_id"] = identifiers[config]
            source_files = recipe.pop("source_files", {})
            if not isinstance(source_files, dict):
                raise ProjectError("Recipe source_files must be an object")
            recipe["source_assets"] = {key: load_asset(value, "source").id for key, value in source_files.items()}
            identifier = recipe.setdefault("id", uuid4().hex)
            if not isinstance(identifier, str) or not identifier or identifier in self._recipe_paths:
                raise ProjectError("Invalid or duplicate recipe ID")
            self._recipe_paths[identifier] = name
            self.recipes.append(recipe)
        active = manifest.get("active_config", names[0])
        if not isinstance(active, str) or active not in identifiers:
            raise ProjectError("Project active_config must name an input in configs")
        self.active_config_id = identifiers[active]
        self.run_settings = deepcopy(settings)
        for key in ("beam0", "beam1"):
            name = self.run_settings.get(key)
            if name not in (None, ""):
                if not isinstance(name, str) or name not in identifiers:
                    raise ProjectError(f"Project {key} must name an input in configs")
                self.run_settings[key] = identifiers[name]
        if "output_directory" in settings and settings["output_directory"] is not None and not isinstance(settings["output_directory"], str):
            raise ProjectError("Project output_directory must be a path string")
        if total > 256 * 1024**3 or len(registered) > 100000:
            raise ProjectError("Project exceeds file count or size limits")
        self.path = path
        self._disk_files = records
        self.check_external_changes()
        self.dirty = False

    def _load_legacy(self, path: Path) -> None:
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
            if manifest.get("format") != "pass-project" or type(manifest.get("format_version")) is not int or manifest["format_version"] != 2:
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
                _flat_member(asset.path, "assets")
                if archive.getinfo(asset.path).file_size != asset.size_bytes:
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
                _flat_member(entry["path"], "configs")
                if PurePosixPath(entry["path"]).suffix.lower() != ".json":
                    raise ProjectError("Invalid config path")
                register(entry["path"])
                data = read_entry(entry["path"], entry["sha256"])
                config = InputConfig(identifier, str(entry["name"]), data)
                self.configs[identifier] = config
                if self._dependency_records(config) != entry["dependencies"]:
                    raise ProjectError(f"Dependency index mismatch: {config.name}")
                # Archive pointers and checksums refer to the submitted key spelling.
                config.data = normalize_input(data)
                # Python field names may reveal file references absent from the raw index.
                self._dependency_records(config)
            recipe_ids = set()
            for entry in manifest["recipes"]:
                register(entry["path"])
                if not entry["path"].startswith("recipes/"):
                    raise ProjectError("Invalid recipe path")
                recipe = read_entry(entry["path"], entry["sha256"])
                if recipe["config_id"] not in self.configs or any(i not in self.assets for i in recipe.get("source_assets", {}).values()):
                    raise ProjectError("Recipe references unknown config or source")
                # Older archives identified files by index and could repeat IDs.
                identifier = recipe.get("id")
                if not isinstance(identifier, str) or not identifier or identifier in recipe_ids:
                    recipe["id"] = uuid4().hex
                recipe_ids.add(recipe["id"])
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
        self.legacy = True
        self.name = path.stem
        self._disk_files = {str(self.path): digest_file(self.path)}
        self.dirty = False

    def materialize(self, config_ids: list[str], destination: Path, output: Path) -> list[Path]:
        """Write a run snapshot with absolute input paths, without mutating configs."""
        self.check_external_changes()
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

    def export_bundle(self, config_ids: list[str], destination: Path, *, context=None) -> None:
        """Export portable JSON + assets with a launcher resolving paths at run time."""
        self.check_external_changes(context=context)
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
        # Inputs resolve assets beside their JSON; leave cwd intact for CLI overrides.
        entries["run.py"] = ('"""Run this exported PASS input bundle: python run.py."""\n'
                             'from pathlib import Path\nimport sys\nfrom PASS.main import cli_main\n'
                             'root = Path(__file__).resolve().parent\n' +
                             ('raise SystemExit(cli_main([str(root / "beam0.json"), str(root / "beam1.json"), *sys.argv[1:]]))\n'
                              if len(config_ids) == 2 else 'raise SystemExit(cli_main([str(root / "beam0.json"), *sys.argv[1:]]))\n')).encode("utf-8")
        entries["README.txt"] = (b"Extract all files together. Install PASS, then run: python run.py\n"
                                 b"Use python run.py --help for CLI options; no GUI installation is required.\n"
                                 b"Inputs use paths relative to this folder. Outputs are written under output/.\n"
                                 b"Override output with --output DIR; a relative DIR uses your terminal's working directory.\n")
        self._write_archive(Path(destination), entries, context=context)

    def copy_command(self, source_id: str, name: str, target: Project, target_id: str, *, context=None) -> str:
        """Copy dependencies without overwrite; the target derives its own design clock."""
        from PASS.para.schema.main import MainConfig
        source = self.configs[source_id].data
        try:
            MainConfig.reject_removed_reference_clock(source)
            MainConfig.reject_removed_reference_clock(target.configs[target_id].data)
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc
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
                item = target._capture(item, self.config_base, context=context)
                item["Slice set"] = replacement
                sequence[unique_name(slicer_name, sequence)] = item
            return replacement

        def visit(value):
            if isinstance(value, dict):
                if value.get("Configuration") is not None:
                    old = value["Configuration"]
                    collision = value.get("Command") in {"BeamBeam", "CrossingAngle"} or (value.get("Command") == "Slicer"
                                                                                          and value.get("Purpose") == "beam_beam")
                    if collision:
                        source_resources = [config.data.get("Beam beam", {}).get("Configurations", {}).get(old) for config in self.configs.values()]
                        source_resources = [resource for resource in source_resources if isinstance(resource, dict)]
                        target_resources = [(result if identifier == target_id else config.data).get("Beam beam", {}).get("Configurations",
                                                                                                                          {}).get(old)
                                            for identifier, config in target.configs.items()]
                        target_resources = [resource for resource in target_resources if isinstance(resource, dict)]
                        if len(source_resources) != 1 or len(target_resources) != 1 or source_resources[0] != target_resources[0]:
                            raise ProjectError("束束命令依赖双束共享 IP 配置。请先在目标项目的一个输入中声明同名、相同的 Beam beam 配置；"
                                               "复制单个命令不会复制或改写另一束的配置。")
                        # The shared definition already exists exactly once. The
                        # user places paired commands and slicing in each input.
                        return
                    module = {
                        "WakeField": "Wake field",
                        "ElectronCloud": "Electron cloud",
                        "IBS": "Intrabeam scattering"
                    }.get(value.get("Command"), "Space charge")
                    identity = (module, old)
                    if identity not in config_names:
                        source_block = source.get(module, {})
                        if module == "Electron cloud":
                            _, source_block = _electron_cloud_block(source)
                        resource = source_block.get("Configurations", {}).get(old)
                        if not isinstance(resource, dict):
                            raise ProjectError(f"Missing {module} configuration: {old}")
                        if module == "Electron cloud":
                            block_key, block = _electron_cloud_block(result)
                            result[block_key] = block
                        else:
                            block = result.setdefault(module, {})
                        resources = block.setdefault("Configurations", {})
                        new = unique_name(old, resources)
                        copied = target._capture(resource, self.config_base, context=context)
                        if module == "Space charge":
                            copied["Slice set"] = copy_slice(copied["Slice set"])
                        elif module == "Intrabeam scattering":
                            from PASS.para.schema.ibs import IBSConfiguration
                            ibs = IBSConfiguration.model_validate(copied)
                            if ibs.slice_set is not None:
                                copied = ibs.model_dump(by_alias=True)
                                copied["Slice set"] = copy_slice(ibs.slice_set)
                        resources[new] = copied
                        # Preserve target physics switches; create a usable block
                        # only when the target has no module settings yet.
                        block.setdefault("Enabled", source_block.get("Enabled", module != "Electron cloud"))
                        config_names[identity] = new
                    value["Configuration"] = config_names[identity]
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(command)
        if command.get("Command") == "WakeField":
            command["Slice set"] = copy_slice(command["Slice set"])
        elif command.get("Command") == "ElectronCloud":
            _, block = _electron_cloud_block(result)
            resource = block["Configurations"][command["Configuration"]]
            if _electron_cloud_mode(resource) in {"build_up", "coupled"}:
                if not command.get("Slice set"):
                    raise ProjectError("Dynamic ElectronCloud requires a Slice set dependency")
                command["Slice set"] = copy_slice(command["Slice set"])
        command = target._capture(command, self.config_base, context=context)
        new_name = unique_name(name, sequence)
        sequence[new_name] = command
        target.update_config(target_id, result, context=context)
        if context is not None:
            context.check()
        return new_name
