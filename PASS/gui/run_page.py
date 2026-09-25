"""Background run preparation, process control, logs, and reproducible run records."""
from __future__ import annotations

import codecs
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import difflib
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from uuid import uuid4

from PySide6.QtCore import QProcess, QSettings, QThread, QTimer, Qt, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from PASS import __version__
from PASS.gui.appearance import code_font
from PASS.gui.project import atomic_write, file_references, json_bytes, read_json, resolved_file
from PASS.gui.runner import RunExitCode
from PASS.gui.widgets import BusyProgressBar, PropertyComboBox, button


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


class PreparationCancelled(Exception):
    """A run was cancelled before any tracking process started."""


class RunPreparation(QThread):
    """Copy inputs and dependencies into a private, validated run snapshot."""
    progress = Signal(str)
    prepared = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, documents, output_root, parent=None, *, previous_record=None):
        super().__init__(parent)
        self.documents = documents
        self.output_root = Path(output_root).resolve()
        self.previous_record = previous_record
        self.record_path = None
        self.record = None
        self._comparison_documents = None

    def _check(self):
        if self.isInterruptionRequested():
            raise PreparationCancelled()

    def _copy_file(self, source, destination):
        digest = hashlib.sha256()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as source_stream, destination.open("xb") as target_stream:
            for block in iter(lambda: source_stream.read(1024 * 1024), b""):
                self._check()
                target_stream.write(block)
                digest.update(block)
            target_stream.flush()
            os.fsync(target_stream.fileno())
        return digest.hexdigest()

    def _verify_record(self):
        path = Path(self.previous_record)
        record = read_json(path.read_bytes())
        self.progress.emit("核对原运行快照和依赖校验和…")
        configurations = [{
            "file": item["configuration"],
            "sha256": item["configuration_sha256"]
        } for item in record.get("inputs", []) if "configuration_sha256" in item]
        for entry in [*record.get("inputs", []), *record.get("dependencies", []), *configurations]:
            self._check()
            source = resolved_file(entry["file"], path.parent)
            if not source.is_relative_to(path.parent.resolve()):
                raise ValueError("运行快照引用了快照目录以外的文件。")
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    self._check()
                    digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"运行快照已被修改，无法按原快照重跑：{source.name}")
        inputs = record.get("inputs", [])
        if not inputs or len(inputs) > 2:
            raise ValueError("运行记录必须包含一份或两份输入快照。")
        self.documents = [(entry.get("name", entry["file"]), read_json(resolved_file(entry["file"], path.parent).read_bytes()), path.parent)
                          for entry in inputs]
        self._comparison_documents = [read_json(resolved_file(entry["configuration"], path.parent).read_bytes()) for entry in inputs]
        return record

    def _prepare(self):
        from PASS.validation.rules import validate_documents
        previous = self._verify_record() if self.previous_record else None
        self._check()
        self.progress.emit("全面校验输入和文件依赖…")
        report = validate_documents(self.documents)
        self._check()
        if not report.ok:
            raise ValueError(report.text())
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:10]
        snapshot = self.output_root / "input_snapshots" / run_id
        output = self.output_root / "runs" / run_id
        snapshot.mkdir(parents=True, exist_ok=False)
        self.record_path = snapshot / "run.json"
        self.record = {
            "format_version": 1,
            "id": run_id,
            "status": "preparing",
            "created_at": _utc_now(),
            "started_at": None,
            "ended_at": None,
            "exit_code": None,
            "pass_version": __version__,
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "source": self._source_identity(),
            "package_versions": {},
            "backend": self.documents[0][1].get("Backend (gpu/cpu)", "cpu"),
            "particle_precision": self.documents[0][1].get("Particle Precision", "float64"),
            "configured_device_ids": deepcopy(self.documents[0][1].get("Device Id", [])),
            "observed_gpu": None,
            "output_root": str(self.output_root),
            "output_directory": str(output),
            "results_directory": str(output),
            "inputs": [],
            "dependencies": [],
            "random_seeds": [],
            "source_run": previous.get("id") if previous else None,
            "warnings": [str(issue) for issue in report.warnings],
        }
        atomic_write(self.record_path, json_bytes(self.record))
        for package in ("numpy", "scipy", "pydantic", "PySide6", "cupy-cuda13x"):
            try:
                self.record["package_versions"][package] = version(package)
            except PackageNotFoundError:
                pass
        copied = {}
        paths = []
        self.progress.emit("复制依赖并计算快照校验和…")
        for index, (name, original, base) in enumerate(self.documents):
            self._check()
            data = deepcopy(original)
            original_path = snapshot / f"configuration{index}.json"
            original_content = json_bytes(self._comparison_documents[index] if self._comparison_documents is not None else original)
            atomic_write(original_path, original_content)
            for mapping, key, _pointer in file_references(data):
                self._check()
                source = resolved_file(mapping[key], Path(base))
                if source not in copied:
                    target = snapshot / "assets" / str(len(copied)) / source.name
                    digest = self._copy_file(source, target)
                    copied[source] = target
                    self.record["dependencies"].append({
                        "file": target.relative_to(snapshot).as_posix(),
                        "sha256": digest,
                        "source": str(source),
                        "size_bytes": target.stat().st_size
                    })
                mapping[key] = str(copied[source])
            data["Output directory"] = str(output)
            path = snapshot / f"beam{index}.json"
            content = json_bytes(data)
            atomic_write(path, content)
            self.record["inputs"].append({
                "name": name,
                "file": path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "configuration": original_path.name,
                "configuration_sha256": hashlib.sha256(original_content).hexdigest(),
                "backend": data.get("Backend (gpu/cpu)", "cpu"),
                "turns": data.get("Number of turns")
            })
            for command_name, command in data.get("Sequence", {}).items():
                if isinstance(command, dict) and command.get("Command") == "Injection":
                    self.record["random_seeds"].append({"input": name, "command": command_name, "seed": command.get("Random Seed")})
            paths.append(path)
        self._check()
        self.progress.emit("校验复制后的固定快照…")
        checked = validate_documents([(path.name, read_json(path.read_bytes()), path.parent) for path in paths])
        self._check()
        if not checked.ok:
            raise ValueError(checked.text())
        self.record["status"] = "ready"
        atomic_write(self.record_path, json_bytes(self.record))
        return {"paths": paths, "snapshot": snapshot, "output": output, "record_path": self.record_path, "record": self.record}

    def _source_identity(self):
        root = Path(__file__).resolve().parents[2]
        digest = hashlib.sha256()
        paths = sorted((root / "PASS").rglob("*.py"))
        for path in paths:
            self._check()
            digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        source = {
            "root": str(root),
            "python_source_sha256": digest.hexdigest(),
            "python_source_files": len(paths),
            "git_head": None,
            "git_dirty": None
        }
        try:
            head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True)
            dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal", "--", "PASS"],
                                   capture_output=True,
                                   text=True,
                                   timeout=5,
                                   check=True)
            source.update(git_head=head.stdout.strip(), git_dirty=bool(dirty.stdout.strip()))
        except (OSError, subprocess.SubprocessError):
            pass
        return source

    def run(self):
        try:
            result = self._prepare()
            self._check()
        except PreparationCancelled:
            self._record_failure("cancelled")
            self.cancelled.emit()
        except Exception as exc:
            self._record_failure("preparation_failed", str(exc))
            self.failed.emit(str(exc))
        else:
            self.prepared.emit(result)

    def _record_failure(self, status, error=None):
        if self.record is None:
            return
        self.record.update(status=status, ended_at=_utc_now())
        if error:
            self.record["error"] = error
        try:
            atomic_write(self.record_path, json_bytes(self.record))
        except OSError:
            pass


class RunHistoryDialog(QDialog):
    """Inspect immutable input snapshots and their independent result directories."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.records = []
        self.setWindowTitle("运行历史")
        self.resize(1020, 620)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["开始 / 创建时间", "输入", "状态", "退出码", "运行 ID"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._show_record)
        layout.addWidget(self.table, 1)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMaximumHeight(210)
        layout.addWidget(self.detail)
        actions = QHBoxLayout()
        for title, callback in (("打开结果目录", self._open_output), ("打开快照目录", self._open_snapshot), ("按原快照重跑", self._rerun), ("比较所选两次配置", self._compare)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            actions.addWidget(button)
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        actions.addWidget(close)
        layout.addLayout(actions)
        owner.run_finished.connect(self.reload)
        self.reload()

    def reload(self, *_args):
        self.records = []
        for path in reversed(self.owner.history_paths):
            try:
                record = read_json(Path(path).read_bytes())
            except (OSError, ValueError):
                continue
            self.records.append((Path(path), record))
        self.table.setRowCount(len(self.records))
        for row, (_path, record) in enumerate(self.records):
            values = (record.get("started_at") or record.get("created_at", ""), " + ".join(i["name"] for i in record.get("inputs", [])),
                      record.get("status", ""), record.get("exit_code"), record.get("id", ""))
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem("" if value is None else str(value)))
        self._show_record()

    def _selected(self):
        return [self.records[index.row()] for index in self.table.selectionModel().selectedRows()]

    def _show_record(self):
        selected = self._selected()
        self.detail.setPlainText(json.dumps(selected[0][1], indent=2, ensure_ascii=False) if selected else "")

    def _open_output(self):
        selected = self._selected()
        if selected:
            self.owner._open_directory(selected[0][1]["results_directory"])

    def _open_snapshot(self):
        selected = self._selected()
        if selected:
            self.owner._open_directory(selected[0][0].parent)

    def _rerun(self):
        selected = self._selected()
        if len(selected) == 1 and self.owner.rerun_record(selected[0][0]):
            self.accept()

    def _compare(self):
        selected = self._selected()
        if len(selected) != 2:
            QMessageBox.information(self, "选择两次运行", "使用 Ctrl 选择两条运行记录后比较配置。")
            return
        try:
            texts = []
            for path, record in selected:
                configs = [read_json(resolved_file(item["configuration"], path.parent).read_bytes()) for item in record["inputs"]]
                texts.append(json.dumps(configs, indent=2, ensure_ascii=False, sort_keys=True).splitlines())
            difference = "\n".join(difflib.unified_diff(*texts, fromfile=selected[0][1]["id"], tofile=selected[1][1]["id"], lineterm=""))
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "比较失败", str(exc))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("运行配置差异")
        dialog.resize(980, 640)
        layout = QVBoxLayout(dialog)
        view = QPlainTextEdit(difference or "两次运行的配置相同。")
        view.setReadOnly(True)
        view.setFont(code_font())
        layout.addWidget(view)
        dialog.exec()
        dialog.deleteLater()


class RunPage(QWidget):
    shutdown_finished = Signal()
    run_finished = Signal(object)

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.controller = None
        self.process: QProcess | None = None
        self.started_at = 0.0
        self._stopped = False
        self._input_project = None
        self._refreshing = False
        self._log_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._progress_tail = ""
        self.preparation = None
        self._leased_projects = []
        self._preparation_result = None
        self._preparation_error = None
        self._cancel_preparation_requested = False
        self._closing = False
        self._forced = False
        self._finish_recorded = False
        self._record_path = None
        self._record = None
        self._output_path = None
        self._stop_path = None
        self._log_path = None
        self._log_lines = deque(maxlen=20000)
        self._log_partial = ""
        self._displayed_log = ""
        self._last_log_level = 20
        self._updating_log = False
        self._history_settings = QSettings("PASS", "Editor")
        self.history_paths = self._history_settings.value("run/history", [], type=list)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        header = QHBoxLayout()
        header.addWidget(QLabel("运行"))
        self.run_path = QLabel("使用当前输入")
        self.run_path.setObjectName("muted")
        header.addWidget(self.run_path, 1)
        self.start_button = button("开始运行", "primary")
        self.start_button.clicked.connect(self.start_run)
        self.stop_button = button("正常停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_run)
        header.addWidget(self.start_button)
        header.addWidget(self.stop_button)
        self.force_button = button("强制结束")
        self.force_button.setToolTip("立即结束进程，尚未写入的输出可能丢失。正常停止会等待当前圈结束并保存缓冲输出。")
        self.force_button.setEnabled(False)
        self.force_button.clicked.connect(self.force_stop)
        header.addWidget(self.force_button)
        self.history_button = button("运行历史…")
        self.history_button.clicked.connect(self.show_history)
        header.addWidget(self.history_button)
        root.addLayout(header)
        inputs = QHBoxLayout()
        inputs.addWidget(QLabel("Beam 0"))
        self.beam0 = PropertyComboBox()
        self.beam1 = PropertyComboBox()
        inputs.addWidget(self.beam0, 1)
        inputs.addWidget(QLabel("Beam 1（双束运行，可选）"))
        inputs.addWidget(self.beam1, 1)
        self.beam0.currentIndexChanged.connect(self._settings_changed)
        self.beam1.currentIndexChanged.connect(self._settings_changed)
        root.addLayout(inputs)
        output = QHBoxLayout()
        output.addWidget(QLabel("输出目录"))
        self.output_directory = QLineEdit()
        self.output_directory.setPlaceholderText("output（相对于 JSON / 项目所在目录）")
        self.output_directory.textChanged.connect(self._settings_changed)
        output.addWidget(self.output_directory, 1)
        browse = button("选择目录…")
        browse.clicked.connect(self._choose_output)
        output.addWidget(browse)
        root.addLayout(output)
        hint = QLabel("运行使用当前编辑内容的固定快照；继续编辑不会改变已经启动的任务。")
        hint.setObjectName("muted")
        root.addWidget(hint)
        self.progress = BusyProgressBar()
        self.progress.hide()
        root.addWidget(self.progress)
        stats = QHBoxLayout()
        self.elapsed = QLabel("耗时：--")
        self.eta = QLabel("预计剩余：等待日志")
        self.state = QLabel("空闲")
        for widget in (self.state, self.elapsed, self.eta):
            widget.setObjectName("runStat")
            stats.addWidget(widget)
        stats.addStretch()
        root.addLayout(stats)
        log_controls = QHBoxLayout()
        self.follow_log = QCheckBox("跟随日志")
        self.follow_log.setChecked(True)
        self.follow_log.toggled.connect(lambda checked: self._scroll_log_end() if checked else None)
        log_controls.addWidget(self.follow_log)
        self.log_level = QComboBox()
        self.log_level.addItem("全部级别", 0)
        self.log_level.addItem("警告与错误", 30)
        self.log_level.addItem("仅错误", 40)
        self.log_level.currentIndexChanged.connect(self._refresh_log)
        log_controls.addWidget(self.log_level)
        self.log_search = QLineEdit()
        self.log_search.setPlaceholderText("搜索日志；Enter 查找下一处")
        self.log_search.returnPressed.connect(self._find_log)
        log_controls.addWidget(self.log_search, 1)
        find = button("查找")
        find.clicked.connect(self._find_log)
        log_controls.addWidget(find)
        export = button("导出完整日志…")
        export.clicked.connect(self.export_log)
        log_controls.addWidget(export)
        self.open_output_button = button("打开结果目录")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self.open_output)
        log_controls.addWidget(self.open_output_button)
        root.addLayout(log_controls)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.log.setObjectName("codeEditor")
        self.log.setFont(code_font())
        self.log.verticalScrollBar().valueChanged.connect(self._log_scrolled)
        root.addWidget(self.log, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_elapsed)
        self.log_timer = QTimer(self)
        self.log_timer.setSingleShot(True)
        self.log_timer.setInterval(100)
        self.log_timer.timeout.connect(self._refresh_log)
        self.refresh_inputs()

    @property
    def busy(self):
        return bool(self.preparation is not None or self.process and self.process.state() != QProcess.NotRunning)

    def show_history(self):
        dialog = RunHistoryDialog(self)
        dialog.exec()
        dialog.deleteLater()

    def refresh_inputs(self) -> None:
        owner = self.controller
        project = owner.project if owner and hasattr(owner, "project") else None
        changed = (project.id if project else None) != self._input_project
        previous = self.input_settings()
        settings = project.run_settings if project and changed else previous
        self._refreshing = True
        self.beam0.clear()
        self.beam1.clear()
        self.beam1.addItem("不使用第二束", "")
        if project:
            for entry in project.configs.values():
                self.beam0.addItem(entry.name + ".json", entry.id)
                self.beam1.addItem(entry.name + ".json", entry.id)
            first = settings.get("beam0", project.active_config_id)
            self.beam0.setCurrentIndex(max(0, self.beam0.findData(first)))
            self.beam1.setCurrentIndex(max(0, self.beam1.findData(settings.get("beam1", ""))))
        else:
            self.beam0.addItem("当前 JSON", "")
        self.beam0.setEnabled(project is not None)
        self.beam1.setEnabled(project is not None)
        if changed or not self.output_directory.text():
            self.output_directory.setText(settings.get("output_directory", "output"))
        self._input_project = project.id if project else None
        self._refreshing = False

    def _settings_changed(self, *_args):
        if self._refreshing:
            return
        owner = self.controller
        if owner and getattr(owner, "project", None):
            owner.project.run_settings = self.input_settings()
            owner.project.dirty = True
            owner._update_document_ui()

    def input_settings(self) -> dict:
        return {
            "beam0": self.beam0.currentData() or "",
            "beam1": self.beam1.currentData() or "",
            "output_directory": self.output_directory.text() or "output"
        }

    def selected_input_ids(self) -> list[str]:
        values = [self.beam0.currentData()]
        if self.beam1.currentData():
            values.append(self.beam1.currentData())
        if not values[0] or len(set(values)) != len(values):
            raise ValueError("请选择有效输入；双束运行使用两份独立 JSON。")
        return values

    def _choose_output(self):
        directory = QFileDialog.getExistingDirectory(self, "选择运行输出目录", self.output_directory.text())
        if directory:
            self.output_directory.setText(directory)

    def start_run(self) -> None:
        if self.busy or self._closing or not self.config.commit_pending():
            return
        owner = self.controller
        project = getattr(owner, "project", None)
        try:
            if project:
                ids = self.selected_input_ids()
                documents = []
                for cid in ids:
                    active = cid == owner._active_input_id
                    entry = project.configs[cid]
                    documents.append((entry.name, deepcopy(self.config.data if active else entry.data),
                                      Path(self.config.base_dir if active else project.config_base)))
                base = project.path.parent if project.path else Path.cwd()
            else:
                documents = [(Path(self.config.path).name if self.config.path else "beam0", deepcopy(self.config.data), Path(self.config.base_dir))]
                base = self.config.base_dir
            output = Path(self.output_directory.text() or "output").expanduser()
            if not output.is_absolute():
                output = base / output
            sources = ([project] if project else []) + list(getattr(owner, "_standalone_sources", []))
            self._start_preparation(documents, output, sources)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法运行", str(exc))

    def rerun_record(self, record_path):
        if self.busy or self._closing:
            QMessageBox.information(self, "运行尚未结束", "请等待当前准备或运行结束后重跑。")
            return False
        try:
            record = read_json(Path(record_path).read_bytes())
            self._start_preparation([], Path(record["output_root"]), [], previous_record=Path(record_path))
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "无法重跑", str(exc))
            return False
        return True

    def _start_preparation(self, documents, output, sources, *, previous_record=None):
        self._previous_state = self.state.text()
        self._preparation_result = self._preparation_error = None
        self._cancel_preparation_requested = False
        self._leased_projects = []
        try:
            for project in dict.fromkeys(sources):
                project.retain()
                self._leased_projects.append(project)
        except Exception:
            for project in self._leased_projects:
                project.release()
            self._leased_projects = []
            raise
        worker = RunPreparation(documents, output, self, previous_record=previous_record)
        self.preparation = worker
        worker.progress.connect(self.state.setText)
        worker.prepared.connect(self._prepared)
        worker.failed.connect(self._preparation_failed)
        worker.cancelled.connect(self._preparation_cancelled)
        worker.finished.connect(self._preparation_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setText("取消准备")
        self.stop_button.setEnabled(True)
        self.force_button.setEnabled(False)
        self.state.setText("正在准备运行…")
        self.progress.show()
        worker.start()

    def _prepared(self, result):
        self._preparation_result = result

    def _preparation_failed(self, message):
        self._preparation_error = message

    def _preparation_cancelled(self):
        self._preparation_error = ""

    def _preparation_finished(self):
        worker = self.preparation
        result = self._preparation_result
        self.preparation = None
        for project in self._leased_projects:
            project.release()
        self._leased_projects = []
        if result is not None and not self._closing and not self._cancel_preparation_requested:
            self._launch(result)
        else:
            if result is not None:
                record = result["record"]
                record.update(status="cancelled", ended_at=_utc_now())
                try:
                    atomic_write(result["record_path"], json_bytes(record))
                except OSError:
                    pass
            self.progress.hide()
            self.start_button.setEnabled(not self._closing)
            self.stop_button.setEnabled(False)
            self.stop_button.setText("正常停止")
            self.state.setText(self._previous_state)
            if self._preparation_error and not self._closing:
                QMessageBox.warning(self, "无法运行", self._preparation_error)
        worker.deleteLater()
        if self._closing and not self.busy:
            self.shutdown_finished.emit()

    def _launch(self, prepared):
        if self.process is not None:
            self.process.deleteLater()
        self.process = QProcess(self)
        self._record_path = prepared["record_path"]
        self._record = prepared["record"]
        self._record.update(status="starting", started_at=_utc_now())
        self._output_path = prepared["output"]
        self._stop_path = prepared["snapshot"] / "stop.requested"
        self._log_path = prepared["snapshot"] / "gui.log"
        self._save_record()
        self._remember_record()
        self.process.setProgram(sys.executable)
        self.process.setArguments([
            "-u", "-X", "utf8", "-m", "PASS.gui.runner", *map(str, prepared["paths"]), "--stop-file",
            str(self._stop_path), "--record",
            str(self._record_path)
        ])
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[2]))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self.process.started.connect(self._process_started)
        self._log_lines.clear()
        self._log_partial = ""
        self._displayed_log = ""
        self._last_log_level = 20
        self._updating_log = True
        self.log.clear()
        self._updating_log = False
        self._log_decoder.reset()
        self._progress_tail = ""
        self._append_output(f"输入快照：{prepared['snapshot']}\n输出目录：{prepared['output']}\n\n")
        self.run_path.setText(" + ".join(entry["name"] for entry in self._record["inputs"]))
        self.started_at = time.monotonic()
        self._stopped = self._forced = self._finish_recorded = False
        self.start_button.setEnabled(False)
        self.stop_button.setText("正常停止")
        self.stop_button.setEnabled(True)
        self.force_button.setEnabled(True)
        self.open_output_button.setEnabled(True)
        self.progress.show()
        self.state.setText("正在启动…")
        self.timer.start(1000)
        self.process.start()

    def _remember_record(self):
        path = str(self._record_path)
        self.history_paths = [item for item in self.history_paths if item != path] + [path]
        self.history_paths = self.history_paths[-100:]
        self._history_settings.setValue("run/history", self.history_paths)

    def _save_record(self):
        try:
            atomic_write(self._record_path, json_bytes(self._record))
        except (OSError, ValueError) as exc:
            self._append_output(f"WARNING 运行记录写入失败：{exc}\n")

    def _process_started(self):
        self.state.setText("运行中")
        self._record["status"] = "running"
        self._save_record()

    def _process_error(self, error):
        if self.process:
            self._append_output("ERROR " + self.process.errorString() + "\n")
        if error == QProcess.FailedToStart:
            self._finished(-1, QProcess.CrashExit)

    def _update_elapsed(self):
        if self.started_at:
            self.elapsed.setText(f"耗时：{time.monotonic() - self.started_at:.1f} s")

    def _read_output(self) -> None:
        if self.process:
            self._append_output(self._log_decoder.decode(bytes(self.process.readAllStandardOutput())))

    def _append_output(self, output: str) -> None:
        if not output:
            return
        if self._log_path is not None:
            try:
                with self._log_path.open("a", encoding="utf-8") as stream:
                    stream.write(output)
            except OSError:
                pass
        lines = (self._log_partial + output).split("\n")
        self._log_partial = lines.pop()
        for line in lines:
            match = re.search(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b", line)
            if match:
                self._last_log_level = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}[match.group(1)]
            self._log_lines.append((self._last_log_level, line))
        if not self.log_timer.isActive():
            self.log_timer.start()
        self._progress_tail = (self._progress_tail + output)[-4096:]
        turns = re.findall(r"\b[Tt]urn[:\s]+(\d+)(?:/(\d+))?", self._progress_tail)
        if turns and not self._stopped:
            current, total = turns[-1]
            self.state.setText(f"运行中 · turn {current}" + (f"/{total}" if total else ""))
        estimates = re.findall(r"\bETA:\s*([^|\r\n]+)", self._progress_tail)
        if estimates:
            self.eta.setText("预计剩余：" + estimates[-1].strip())

    def _refresh_log(self, *_args):
        if not hasattr(self, "log"):
            return
        self.log_timer.stop()
        minimum = self.log_level.currentData() or 0
        lines = [line for level, line in self._log_lines if level >= minimum]
        if self._log_partial and self._last_log_level >= minimum:
            lines.append(self._log_partial)
        scrollbar = self.log.verticalScrollBar()
        old_value = scrollbar.value()
        cursor = self.log.textCursor()
        position, anchor = cursor.position(), cursor.anchor()
        self._updating_log = True
        text = "\n".join(lines)
        if text.startswith(self._displayed_log):
            insertion = QTextCursor(self.log.document())
            insertion.movePosition(QTextCursor.End)
            insertion.insertText(text[len(self._displayed_log):])
        else:
            self.log.setPlainText(text)
        self._displayed_log = text
        if self.follow_log.isChecked():
            scrollbar.setValue(scrollbar.maximum())
        else:
            cursor = self.log.textCursor()
            maximum = self.log.document().characterCount() - 1
            cursor.setPosition(min(anchor, maximum))
            cursor.setPosition(min(position, maximum), QTextCursor.KeepAnchor)
            self.log.setTextCursor(cursor)
            scrollbar.setValue(old_value)
        self._updating_log = False

    def _log_scrolled(self, value):
        if not self._updating_log and value < self.log.verticalScrollBar().maximum():
            self.follow_log.setChecked(False)

    def _scroll_log_end(self):
        self._updating_log = True
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        self._updating_log = False

    def _find_log(self):
        text = self.log_search.text()
        if not text:
            return
        self.follow_log.setChecked(False)
        if not self.log.find(text):
            cursor = self.log.textCursor()
            cursor.movePosition(QTextCursor.Start)
            self.log.setTextCursor(cursor)
            self.log.find(text)

    def export_log(self):
        path, _filter = QFileDialog.getSaveFileName(self, "导出完整运行日志", "pass-run.log", "Log (*.log);;Text (*.txt)")
        if path:
            try:
                if self._log_path is not None and self._log_path.is_file():
                    if Path(path).resolve() != self._log_path.resolve():
                        shutil.copyfile(self._log_path, path)
                else:
                    Path(path).write_text("".join(line + "\n" for _level, line in self._log_lines) + self._log_partial, encoding="utf-8")
            except OSError as exc:
                QMessageBox.warning(self, "日志导出失败", str(exc))

    def open_output(self):
        if self._output_path:
            self._open_directory(self._output_path)

    def _open_directory(self, path):
        path = Path(path)
        if path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
        else:
            QMessageBox.information(self, "目录尚未生成", "运行尚未生成该结果目录。")

    def _finished(self, code: int, status: QProcess.ExitStatus) -> None:
        if self._finish_recorded:
            return
        self._finish_recorded = True
        self._read_output()
        self._append_output(self._log_decoder.decode(b"", final=True))
        self._refresh_log()
        self.timer.stop()
        self.progress.hide()
        self.start_button.setEnabled(not self._closing)
        self.stop_button.setEnabled(False)
        self.force_button.setEnabled(False)
        result = "failed"
        if self._forced:
            result = "force_stopped"
        elif status == QProcess.NormalExit:
            result = {
                RunExitCode.COMPLETED: "completed",
                RunExitCode.STOPPED: "stopped",
                RunExitCode.INTERRUPTED: "interrupted",
            }.get(code, "failed")
        self.state.setText({
            "force_stopped": "已强制结束",
            "stopped": "已在圈边界停止",
            "interrupted": "运行已中断（当前圈可能未完成）",
            "completed": "完成",
            "failed": f"失败（退出码 {code}）",
        }[result])
        self._update_elapsed()
        self.eta.setText("预计剩余：--")
        if self._record_path:
            try:
                self._record = read_json(self._record_path.read_bytes())
            except (OSError, ValueError):
                pass
            self._record.update(status=result,
                                exit_code=code,
                                ended_at=_utc_now(),
                                duration_seconds=time.monotonic() - self.started_at,
                                stop_requested=self._stopped)
            self._output_path = Path(self._record["results_directory"])
            self._save_record()
            self.run_finished.emit(deepcopy(self._record))
        if self._closing and not self.busy:
            self.shutdown_finished.emit()

    def stop_run(self) -> None:
        if self.preparation is not None:
            self._cancel_preparation_requested = True
            self.preparation.requestInterruption()
            self.stop_button.setEnabled(False)
            self.state.setText("正在取消准备…")
        elif self.process and self.process.state() != QProcess.NotRunning:
            try:
                self._stop_path.write_text("Stop at the next complete turn boundary.\n", encoding="utf-8")
            except OSError as exc:
                QMessageBox.warning(self, "正常停止请求失败", str(exc))
                return
            self._stopped = True
            self.stop_button.setEnabled(False)
            self.state.setText("等待当前圈完成并保存输出…")

    def force_stop(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self._forced = self._stopped = True
            self.process.kill()

    def shutdown(self):
        """Request cooperative shutdown; emit shutdown_finished when safe to close."""
        self._closing = True
        if self.busy:
            self.stop_run()
            return False
        self.log_timer.stop()
        return True
