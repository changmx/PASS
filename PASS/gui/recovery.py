"""Existing recovery drafts, including temporary input dependencies."""

from copy import deepcopy
import json
from pathlib import Path

from PySide6.QtCore import QLockFile, QObject, QSettings, QStandardPaths
from PySide6.QtWidgets import QInputDialog, QMessageBox

from PASS.gui.jobs import TaskCancelled
from PASS.gui.project import Asset, InputConfig, Project, atomic_write, digest_file, json_bytes, safe_member


def _replace_paths(value, replacements):
    if isinstance(value, dict):
        return {key: _replace_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_paths(item, replacements) for item in value]
    if isinstance(value, str):
        for source, destination in replacements:
            for old, new in ((str(source), str(destination)), (source.as_posix(), destination.as_posix())):
                value = value.replace(old.replace("\\", "\\\\"), new.replace("\\", "\\\\"))
                value = value.replace(old, new)
    return value


class RecoveryStore:

    def __init__(self, directory):
        self.directory = Path(directory)

    @staticmethod
    def read(path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (not isinstance(payload, dict) or payload.get("format") != "pass-draft-1" or not isinstance(payload.get("state"), dict)
                or not isinstance(payload.get("saved_at"), str)):
            raise ValueError("恢复副本格式无效")
        if not isinstance(payload["state"].get("path", ""), (str, type(None))):
            raise ValueError("恢复副本路径无效")
        return payload

    @staticmethod
    def resolve(path):
        path = Path(path)
        if path.exists():
            payload = RecoveryStore.read(path)
            payload["status"] = "resolved"
            atomic_write(path, json_bytes(payload))

    def pending(self, context=None):
        entries = []
        for path in self.directory.glob("*.json"):
            if context is not None:
                context.report(f"读取已有恢复副本：{path.name}")
            lock = QLockFile(str(path) + ".lock")
            lock.setStaleLockTime(0)
            if not lock.tryLock(0):
                continue
            try:
                payload = self.read(path)
                if payload.get("status") == "pending":
                    entries.append((path, payload))
            except (OSError, ValueError, TypeError):
                continue
            finally:
                lock.unlock()
        if context is not None:
            context.check()
        return sorted(entries, key=lambda item: item[1]["saved_at"], reverse=True)


class RecoveryManager(QObject):

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        if owner.settings.format() == QSettings.IniFormat:
            directory = Path(owner.settings.fileName()).parent / "recovery"
        else:
            directory = Path(QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)) / "recovery"
        self.store = RecoveryStore(directory)
        self._restored_from = None

    def invalidate(self):
        if self._restored_from is not None:
            try:
                self.store.resolve(self._restored_from)
                self._restored_from = None
            except (OSError, ValueError) as exc:
                self.owner.statusBar().showMessage(f"恢复副本状态未更新，原副本保留：{exc}", 10000)

    def restore(self):
        try:
            entries = self.owner._run_document_task("读取已有恢复副本", self.store.pending)
        except TaskCancelled:
            return
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self.owner, "恢复失败", str(exc))
            return
        if not entries:
            QMessageBox.information(self.owner, "恢复草稿", "没有其他已退出窗口留下的未保存草稿。")
            return
        titles = [f"{payload['saved_at']} · {Path(payload['state'].get('path') or '未命名输入').name} · {path.stem[:8]}" for path, payload in entries]
        selected, accepted = QInputDialog.getItem(self.owner, "恢复未保存草稿", "选择草稿（恢复为未保存文档，不覆盖原文件）：", titles, 0, False)
        if not accepted or not self.owner._confirm_replace():
            return
        path, payload = entries[titles.index(selected)]
        state = deepcopy(payload["state"])
        state["path"] = ""
        preview = None
        candidate = None
        try:
            if state.get("project"):
                candidate, state = self.owner._run_document_task("恢复项目依赖", lambda context: self._prepare_project(state, context))
            # Validate restoration on a detached editor before replacing live data.
            preview = type(self.owner.config)()
            preview.restore_draft_state(state)
            self.owner.config.restore_draft_state(state)
            # Releasing a cache must not resolve the copy currently being restored.
            previous_source, self._restored_from = self._restored_from, None
            try:
                self.owner._release_project()
            finally:
                self._restored_from = previous_source
            if candidate is not None:
                self.owner.project = candidate
                self.owner._active_input_id = candidate.active_config_id
                self.owner._changing_input = True
                try:
                    selector = self.owner.config.input_selector
                    selector.clear()
                    for config in candidate.configs.values():
                        selector.addItem(config.name + ".json", config.id)
                    selector.setCurrentIndex(selector.findData(candidate.active_config_id))
                finally:
                    self.owner._changing_input = False
            self.owner.config._data_dirty = True
            self.owner._update_document_ui()
            self.owner._show_page(0)
            self.owner.run.refresh_inputs()
            if self._restored_from != path:
                self.invalidate()
            self._restored_from = path
        except TaskCancelled:
            return
        except (OSError, ValueError, TypeError, KeyError) as exc:
            if candidate is not None and candidate is not self.owner.project:
                candidate.close()
            QMessageBox.warning(self.owner, "恢复失败", str(exc))
        finally:
            if preview is not None:
                preview.deleteLater()

    def _prepare_project(self, state, context):
        saved = state["project"]
        root = Path(saved["root"]).resolve()
        if not root.is_relative_to((self.store.directory / "assets").resolve()):
            raise ValueError("恢复项目的依赖目录不在恢复副本内")
        project = Project()
        try:
            project.id = saved["id"]
            project.created_at = saved["created_at"]
            project.configs = {config["id"]: InputConfig(**config) for config in saved["configs"]}
            project.assets = {asset["id"]: Asset(**asset) for asset in saved["assets"]}
            project.active_config_id = saved["active_input_id"]
            if project.active_config_id not in project.configs:
                raise ValueError("恢复项目缺少活动输入")
            for asset in project.assets.values():
                member = safe_member(asset.path)
                source, target = root / member, project.root / member
                context.report(f"恢复依赖：{asset.original_name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                context.copy_file(source, target)
                if target.stat().st_size != asset.size_bytes or digest_file(target, context=context) != asset.sha256:
                    raise ValueError(f"恢复依赖的大小或校验值不符：{asset.original_name}")
            state = _replace_paths(state, [(root, project.root)])
            project.configs = {config["id"]: InputConfig(**config) for config in state["project"]["configs"]}
            project.recipes = deepcopy(state["project"]["recipes"])
            project.run_settings = deepcopy(state["project"]["run_settings"])
            project.dirty = True
            context.check()
            return project, state
        except Exception:
            project.close()
            raise

    def shutdown(self):
        self.invalidate()
        return True
