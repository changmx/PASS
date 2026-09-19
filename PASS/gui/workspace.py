"""Document actions and the project resource browser, separate from physics forms."""
from __future__ import annotations

from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import shutil
import shlex
from uuid import uuid4

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PASS.gui.project import Project, ProjectError, atomic_write, file_references, json_bytes, read_json, resolved_file


def display_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


class ResourceDialog(QDialog):
    """Read-only source browsing with explicit export and copy-into-document actions."""

    def __init__(self, owner, project: Project, external: bool = False) -> None:
        super().__init__(owner)
        self.owner = owner
        self.project = project
        self.external = external
        self.selection = None
        self.value = None
        self.parameter_values = []
        self.setWindowTitle("项目内容 · " + (project.path.name if project.path else "未保存项目"))
        self.resize(940, 600)
        self.setMinimumSize(700, 420)
        root = QVBoxLayout(self)
        tools = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索文件、命令或参数")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        tools.addWidget(self.search, 1)
        if not external:
            add = QPushButton("添加源文件…")
            add.clicked.connect(self._add_source)
            tools.addWidget(add)
        export = QPushButton("导出此文件…")
        export.clicked.connect(self.export_file)
        tools.addWidget(export)
        root.addLayout(tools)
        splitter = QSplitter(Qt.Horizontal)
        self.files = QTreeWidget()
        self.files.setHeaderHidden(True)
        self.files.setMinimumWidth(180)
        self.files.currentItemChanged.connect(self._select_file)
        splitter.addWidget(self.files)
        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(8, 0, 0, 0)
        self.location = QComboBox()
        self.location.currentIndexChanged.connect(self._select_location)
        layout.addWidget(self.location)
        self.tabs = QTabWidget()
        self.parameters = QTableWidget(0, 2)
        self.parameters.setHorizontalHeaderLabels(["参数", "值"])
        self.parameters.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.parameters.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.parameters.setColumnWidth(0, 270)
        self.parameters.verticalHeader().hide()
        self.parameters.verticalHeader().setDefaultSectionSize(25)
        self.parameters.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.parameters.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.parameters, "参数")
        self.raw = QPlainTextEdit()
        self.raw.setReadOnly(True)
        self.raw.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.tabs.addTab(self.raw, "原始内容")
        self.data_table = QTableWidget()
        self.data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tabs.addTab(self.data_table, "数据预览")
        self.tabs.setTabVisible(2, False)
        layout.addWidget(self.tabs, 1)
        row = QHBoxLayout()
        self.copy_value_button = QPushButton("复制值")
        self.copy_value_button.clicked.connect(self.copy_value)
        row.addWidget(self.copy_value_button)
        copy_json = QPushButton("复制当前 JSON")
        copy_json.clicked.connect(self.copy_json)
        row.addWidget(copy_json)
        self.copy_command_button = QPushButton("复制命令到当前输入")
        self.copy_command_button.clicked.connect(self.copy_command)
        row.addWidget(self.copy_command_button)
        self.restore_button = QPushButton("载入生成设置")
        self.restore_button.clicked.connect(self.restore_recipe)
        row.addWidget(self.restore_button)
        layout.addLayout(row)
        splitter.addWidget(right)
        splitter.setSizes([250, 660])
        root.addWidget(splitter, 1)
        self.info = QLabel("只读预览。复制完整命令会带入引用的计算配置、切片器和文件依赖。")
        self.info.setWordWrap(True)
        root.addWidget(self.info)
        self.refresh()

    def refresh(self):
        self.files.clear()
        configs = QTreeWidgetItem(["输入 JSON"])
        self.files.addTopLevelItem(configs)
        for entry in self.project.configs.values():
            item = QTreeWidgetItem([entry.name + ".json"])
            item.setData(0, Qt.UserRole, ("config", entry.id))
            configs.addChild(item)
        assets = QTreeWidgetItem(["输入与源文件"])
        self.files.addTopLevelItem(assets)
        for entry in self.project.assets.values():
            item = QTreeWidgetItem([entry.original_name])
            item.setToolTip(0, f"{entry.path}\n{entry.size_bytes:,} bytes\nSHA-256: {entry.sha256}")
            item.setData(0, Qt.UserRole, ("asset", entry.id))
            assets.addChild(item)
        recipes = QTreeWidgetItem(["生成参数"])
        self.files.addTopLevelItem(recipes)
        for index, recipe in enumerate(self.project.recipes):
            item = QTreeWidgetItem([f"{recipe.get('kind', 'generator')} · {index + 1}"])
            item.setData(0, Qt.UserRole, ("recipe", index))
            recipes.addChild(item)
        self.files.expandAll()
        if configs.childCount():
            self.files.setCurrentItem(configs.child(0))

    def _select_file(self, item, _previous=None):
        self.selection = item.data(0, Qt.UserRole) if item else None
        self.location.blockSignals(True)
        self.location.clear()
        self.value = None
        if self.selection:
            kind, identifier = self.selection
            if kind == "config":
                data = self.project.configs[identifier].data
                self.location.addItem("完整输入 JSON", None)
                self.location.addItem("全局配置", "__global__")
                for name, command in data.get("Sequence", {}).items():
                    if isinstance(command, dict):
                        self.location.addItem(f"{name} / {command.get('Command', '')}", name)
            else:
                self.location.addItem("生成参数" if kind == "recipe" else "原始文件", None)
        self.location.blockSignals(False)
        self._select_location()

    def _select_location(self, _index=0):
        self.tabs.setTabVisible(2, False)
        self.copy_command_button.setEnabled(False)
        self.restore_button.setVisible(bool(self.selection and self.selection[0] == "recipe"))
        if not self.selection:
            self.raw.clear()
            self.parameters.setRowCount(0)
            return
        kind, identifier = self.selection
        if kind == "config":
            data = self.project.configs[identifier].data
            name = self.location.currentData()
            self.value = ({
                k: v
                for k, v in data.items() if k != "Sequence"
            } if name == "__global__" else data.get("Sequence", {}).get(name) if name else data)
            self.copy_command_button.setEnabled(bool(name and name != "__global__"))
        elif kind == "recipe":
            self.value = self.project.recipes[identifier]
        else:
            asset = self.project.assets[identifier]
            self.value = vars(asset)
            path = self.project.root / asset.path
            with path.open("rb") as stream:
                content = stream.read(2 * 1024 * 1024)
            if b"\0" in content[:8192]:
                text = "二进制文件，可使用“导出此文件”获取原始内容。\n\n" + display_json(self.value)
            else:
                text = content.decode("utf-8-sig", errors="replace")
                if asset.size_bytes > len(content):
                    text += "\n\n[预览截断；导出可取得完整文件]"
            self.raw.setPlainText(text)
            self.tabs.setCurrentIndex(1)
            if path.suffix.casefold() in (".tfs", ".csv") and b"\0" not in content[:8192]:
                self._preview_table(path.suffix.casefold(), text)
        if kind != "asset":
            text = display_json(self.value)
            self.raw.setPlainText(text[:2 * 1024 * 1024] + ("\n[预览截断]" if len(text) > 2 * 1024 * 1024 else ""))
            self.tabs.setCurrentIndex(0)
        self.parameter_values = []

        def flatten(value, prefix=""):
            if len(self.parameter_values) >= 5000:
                return
            if isinstance(value, dict) and value:
                for key, item in value.items():
                    flatten(item, f"{prefix} / {key}" if prefix else str(key))
            else:
                self.parameter_values.append((prefix, value))

        flatten(self.value)
        self.parameters.setRowCount(len(self.parameter_values))
        for row, (name, value) in enumerate(self.parameter_values):
            self.parameters.setItem(row, 0, QTableWidgetItem(name))
            cell = QTableWidgetItem(json.dumps(value, ensure_ascii=False)[:200])
            cell.setToolTip(json.dumps(value, ensure_ascii=False)[:1000])
            self.parameters.setItem(row, 1, cell)
        self.info.setText("只读预览 · 参数表最多显示 5,000 项；完整内容可导出。" if len(self.parameter_values) >= 5000 else "只读预览 · 复制完整命令会带入引用的计算配置、切片器和文件依赖。")
        self._filter(self.search.text())

    def _preview_table(self, suffix, text):
        try:
            if suffix == ".csv":
                rows = list(csv.reader(io.StringIO(text)))[:501]
                header, rows = rows[0], rows[1:]
            else:
                lines = text.splitlines()
                start = next(i for i, line in enumerate(lines) if line.lstrip().startswith("*"))
                header = shlex.split(lines[start].lstrip()[1:])
                rows = [shlex.split(line) for line in lines[start + 1:] if line.strip() and not line.lstrip().startswith(("@", "$", "#"))][:500]
            if not header:
                return
            rows = [row for row in rows if len(row) == len(header)]
            self.data_table.setColumnCount(len(header))
            self.data_table.setHorizontalHeaderLabels(header)
            self.data_table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, value in enumerate(row):
                    self.data_table.setItem(r, c, QTableWidgetItem(value))
            self.tabs.setTabVisible(2, True)
            self.tabs.setCurrentIndex(2)
        except (ValueError, StopIteration, IndexError, csv.Error):
            pass  # Raw preview and exact export remain available for partial files.

    def _filter(self, text):
        needle = text.casefold().strip()
        for root_index in range(self.files.topLevelItemCount()):
            root = self.files.topLevelItem(root_index)
            for index in range(root.childCount()):
                child = root.child(index)
                child.setHidden(bool(needle and needle not in child.text(0).casefold()))
        for row, (name, value) in enumerate(self.parameter_values):
            self.parameters.setRowHidden(row, bool(needle and needle not in name.casefold() and needle not in str(value).casefold()))

    def copy_value(self):
        if self.tabs.currentWidget() is self.data_table:
            item = self.data_table.currentItem()
            if item:
                QApplication.clipboard().setText(item.text())
            return
        row = self.parameters.currentRow()
        if 0 <= row < len(self.parameter_values):
            value = self.parameter_values[row][1]
            QApplication.clipboard().setText(value if isinstance(value, str) else display_json(value))
            self.info.setText("已复制选中的参数值。")

    def copy_json(self):
        if self.value is not None:
            QApplication.clipboard().setText(display_json(self.value))
            self.info.setText("已复制当前 JSON。")

    def copy_command(self):
        if not self.selection or self.selection[0] != "config":
            return
        name = self.location.currentData()
        if not name or name == "__global__":
            return
        copied = self.owner.copy_project_command(self.project, self.selection[1], name)
        if copied:
            self.info.setText(f"已复制为 {copied}；引用的配置和切片集发生重名时自动使用新名称。")

    def export_file(self):
        if not self.selection:
            return
        kind, identifier = self.selection
        if kind == "asset":
            name = self.project.assets[identifier].original_name
        elif kind == "config":
            name = self.project.configs[identifier].name + ".json"
        else:
            name = f"recipe-{identifier + 1}.json"
        path, _ = QFileDialog.getSaveFileName(self, "导出项目内的原始文件", name, "All files (*)")
        if not path:
            return
        try:
            if kind == "asset":
                shutil.copyfile(self.project.root / self.project.assets[identifier].path, path)
            else:
                value = self.project.configs[identifier].data if kind == "config" else self.project.recipes[identifier]
                atomic_write(Path(path), json_bytes(value))
            self.info.setText("已导出文件。JSON 中引用的资产需一同导出才能运行。" if kind == "config" else "已导出文件。")
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def _add_source(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "添加源文件到项目", "", "All files (*)")
        try:
            for path in paths:
                self.project.add_asset(Path(path), "source")
            if paths:
                self.owner._update_document_ui()
                self.refresh()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "添加失败", str(exc))

    def restore_recipe(self):
        if self.selection and self.selection[0] == "recipe":
            if self.owner.restore_project_recipe(self.project, self.project.recipes[self.selection[1]]):
                self.accept()


class DocumentWindowMixin:
    """File ownership is explicit: one standalone JSON or one multi-input project."""

    def _init_documents(self):
        self.project: Project | None = None
        self._active_input_id = ""
        self._changing_input = False
        self._standalone_sources: list[Project] = []
        self.actions = {}
        menu = self.file_menu

        def action(key, text, handler, shortcut=None):
            item = QAction(text, self)
            if shortcut:
                item.setShortcut(shortcut)
            item.triggered.connect(handler)
            menu.addAction(item)
            self.actions[key] = item
            return item

        menu.addSection("JSON 输入")
        action("new_json", "新建 JSON", self.new_json, QKeySequence.New)
        action("open_json", "打开 JSON…", self.open_json, QKeySequence.Open)
        action("save_json", "保存 JSON", self.save_document)
        action("save_json_as", "JSON 另存为…", lambda: self.save_document(True))
        menu.addSeparator()
        menu.addSection("项目")
        action("new_project", "从当前输入创建项目…", self.new_project)
        action("open_project", "打开项目…", self.open_project)
        action("save_project", "保存项目", self.save_document)
        action("save_project_as", "项目另存为…", lambda: self.save_document(True))
        action("import_json", "导入 JSON 到项目…", self.import_json)
        action("contents", "项目内容…", self.show_resources)
        action("browse_project", "从其他项目查看 / 复制参数…", self.browse_project)
        menu.addSeparator()
        menu.addSection("导出副本")
        action("export_json", "导出当前 JSON…", self.export_current_json)
        action("export_bundle", "导出可运行输入包…", self.export_bundle)
        menu.addSeparator()
        action("undo", "撤销参数修改", lambda: self._undo(False), QKeySequence.Undo)
        action("redo", "重做参数修改", lambda: self._undo(True), QKeySequence.Redo)
        self.save_action = QAction(self)
        self.save_action.setShortcut(QKeySequence.Save)
        self.save_action.triggered.connect(self.save_document)
        self.addAction(self.save_action)
        self.config.input_selector.currentIndexChanged.connect(self._switch_input)
        self.config.contents_button.clicked.connect(self.show_resources)
        self.config.import_input_button.clicked.connect(self.import_json)
        self.config.changed.connect(self._update_document_ui)
        self.config.file_changed.connect(self._update_document_ui)
        self._update_document_ui()

    def _undo(self, redo):
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QPlainTextEdit)) and not focused.isReadOnly():
            focused.redo() if redo else focused.undo()
        else:
            self.config.undo_data(redo)

    def _commit_current(self) -> bool:
        if not self.config.commit_pending():
            return False
        if self.project:
            try:
                self.project.update_config(self._active_input_id, self.config.data, self.config.base_dir)
                existing = {r["id"] for r in self.project.recipes}
                for recipe in self.config.recipes:
                    if recipe["id"] not in existing:
                        self.project.add_recipe(recipe, self._active_input_id)
                        existing.add(recipe["id"])
                # Keep the active form bound to the document object. Save and
                # input switching replace/rebuild that object deliberately.
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "项目输入不完整", str(exc))
                return False
        return True

    def _dirty(self):
        return self.config.has_unsaved_changes() or bool(self.project and self.project.dirty)

    def _confirm_replace(self, *, force=False, target="当前输入") -> bool:
        if not self._dirty():
            if force:
                return QMessageBox.question(self, "确认替换", f"将替换{target}，是否继续？", QMessageBox.Yes | QMessageBox.Cancel,
                                            QMessageBox.Cancel) == QMessageBox.Yes
            return True
        result = QMessageBox.question(self, "未保存修改", f"即将替换{target}。是否先保存当前修改？", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                      QMessageBox.Save)
        return self.save_document() if result == QMessageBox.Save else result == QMessageBox.Discard

    def _set_input(self, data, path="", recipes=None, base=None):
        self.config.data = deepcopy(data)
        self.config.path = str(path)
        self.config.base_dir = Path(base) if base is not None else Path(path).resolve().parent if path else Path.cwd()
        self.config.recipes = deepcopy(recipes or [])
        self.config._data_dirty = self.config._json_dirty = self.config._form_dirty = False
        self.config.reset_history()
        self.config._sync_editor()
        self.config._clear_form()
        self.config._refresh_tree()
        self.config._set_sync_status("项目输入" if self.project else "JSON 输入")
        self._update_document_ui()

    def _release_project(self):
        if self.project:
            self.project.close()
            self.project = None
        for project in self._standalone_sources:
            project.close()
        self._standalone_sources.clear()

    def new_json(self):
        if not self._confirm_replace():
            return
        from PASS.para.schema.main import MainConfig
        self._release_project()
        data = MainConfig().model_dump(by_alias=True)
        data["Sequence"] = {}
        self._set_input(data)

    def open_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开独立 JSON", "", "JSON (*.json)")
        if not path:
            return
        self.open_json_path(path)

    def open_json_path(self, path):
        from PASS.gui.file_drop import identify_file
        try:
            if identify_file(path) != "json":
                raise ValueError("请选择 PASS 输入 JSON。")
            data = read_json(Path(path).read_bytes())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "打开失败", str(exc))
            return
        if self.project:
            dialog = QMessageBox(self)
            dialog.setWindowTitle("加载输入配置")
            name = self.project.configs[self._active_input_id].name
            dialog.setText(f"{Path(path).name}\n当前项目输入：{name}\n选择加载方式：")
            add = dialog.addButton("作为新配置加入项目", QMessageBox.AcceptRole)
            replace_input = dialog.addButton("替换当前配置", QMessageBox.DestructiveRole)
            standalone = dialog.addButton("打开为独立 JSON", QMessageBox.ActionRole)
            dialog.addButton(QMessageBox.Cancel)
            dialog.exec()
            chosen = dialog.clickedButton()
            if chosen == add:
                self.import_json_paths([path])
                return
            if chosen == replace_input:
                if not self._confirm_replace(force=True, target=f"项目中的 {name} 配置为 {Path(path).name}"):
                    return
                candidate = None
                try:
                    candidate = self._copy_project()
                    candidate.update_config(self._active_input_id, data, Path(path).resolve().parent)
                    candidate.add_asset(Path(path), "source-json")
                    candidate.recipes = [r for r in candidate.recipes if r["config_id"] != self._active_input_id]
                except (OSError, ValueError) as exc:
                    if candidate:
                        candidate.close()
                    QMessageBox.warning(self, "替换失败", str(exc))
                    return
                old = self.project
                self.project = candidate
                self._activate_project_input(self._active_input_id)
                old.close()
                return
            if chosen != standalone:
                return
        if self._confirm_replace(force=True, target=f"当前文档为 {Path(path).name}"):
            self._release_project()
            self._set_input(data, path)

    def _copy_project(self):
        """Stage edits in a private cache without changing the live document."""
        candidate = Project()
        try:
            shutil.copytree(self.project.root, candidate.root, dirs_exist_ok=True)
            for key in ("id", "created_at", "configs", "assets", "recipes", "active_config_id", "run_settings", "path", "dirty"):
                setattr(candidate, key, deepcopy(getattr(self.project, key)))
        except Exception:
            candidate.close()
            raise
        return candidate

    def new_project(self):
        if not self._commit_current():
            return
        candidate = Project()
        try:
            cid = candidate.add_config(Path(self.config.path).name if self.config.path else "beam0", self.config.data, self.config.base_dir)
            if self.config.path and Path(self.config.path).is_file():
                candidate.add_asset(Path(self.config.path), "source-json")
            for recipe in self.config.recipes:
                # Re-home source assets when starting from another project.
                value = self._recipe_with_sources(self.project, recipe) if self.project else recipe
                candidate.add_recipe(value, cid)
        except (OSError, ValueError) as exc:
            candidate.close()
            QMessageBox.warning(self, "无法创建完整项目", str(exc))
            return
        if self.project and not self._confirm_replace():
            candidate.close()
            return
        self._release_project()
        self.project = candidate
        self._activate_project_input(cid)
        self.save_document(True)

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开 PASS 项目", "", "PASS project (*.passproj)")
        if not path:
            return
        self.open_project_path(path)

    def open_project_path(self, path):
        try:
            candidate = Project.open(Path(path))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "打开失败", str(exc))
            return
        if not self._confirm_replace(force=True, target=f"当前文档为项目 {Path(path).name}"):
            candidate.close()
            return
        self._release_project()
        self.project = candidate
        self._activate_project_input(candidate.active_config_id)
        self.run.refresh_inputs()

    def _activate_project_input(self, cid):
        self._active_input_id = cid
        self.project.active_config_id = cid
        config = self.project.configs[cid]
        recipes = [r for r in self.project.recipes if r["config_id"] == cid]
        self._changing_input = True
        self.config.input_selector.clear()
        for entry in self.project.configs.values():
            self.config.input_selector.addItem(entry.name + ".json", entry.id)
        self.config.input_selector.setCurrentIndex(self.config.input_selector.findData(cid))
        self._changing_input = False
        self._set_input(config.data, recipes=recipes, base=self.project.config_base)
        self.run.refresh_inputs()

    def _switch_input(self, index):
        if self._changing_input or not self.project:
            return
        cid = self.config.input_selector.itemData(index)
        if not cid or cid == self._active_input_id:
            return
        if not self._commit_current():
            self._changing_input = True
            self.config.input_selector.setCurrentIndex(self.config.input_selector.findData(self._active_input_id))
            self._changing_input = False
            return
        self.project.dirty = True
        self._activate_project_input(cid)

    def import_json(self):
        if not self.project or not self._commit_current():
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "导入 JSON 及其全部输入依赖", "", "JSON (*.json)")
        self.import_json_paths(paths)

    def import_json_paths(self, paths):
        if not paths or not self.project or not self._commit_current():
            return
        candidate = None
        try:
            candidate = self._copy_project()
            last = None
            for name in paths:
                from PASS.gui.file_drop import identify_file
                if identify_file(name) != "json":
                    raise ValueError("请选择 PASS 输入 JSON。")
                path = Path(name)
                last = candidate.add_config(path.name, read_json(path.read_bytes()), path.resolve().parent)
                candidate.add_asset(path, "source-json")
            if last:
                old = self.project
                self.project = candidate
                self._activate_project_input(last)
                old.close()
        except (OSError, ValueError) as exc:
            if candidate and candidate is not self.project:
                candidate.close()
            QMessageBox.warning(self, "导入失败", str(exc))
            self._update_document_ui()

    def save_document(self, save_as: bool = False) -> bool:
        if not self._commit_current():
            return False
        if self.project:
            path = self.project.path
            if not path or save_as:
                value, _ = QFileDialog.getSaveFileName(self, "保存完整 PASS 项目", str(path or "beam.passproj"), "PASS project (*.passproj)")
                if not value:
                    return False
                path = Path(value)
                if path.suffix.lower() != ".passproj":
                    path = path.with_suffix(".passproj")
            try:
                self.project.run_settings = self.run.input_settings()
                self.project.save(path)
                self.config.data = deepcopy(self.project.configs[self._active_input_id].data)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "保存项目失败", str(exc))
                return False
        else:
            path = Path(self.config.path) if self.config.path else None
            if not path or save_as:
                value, _ = QFileDialog.getSaveFileName(self, "保存独立 JSON", str(path or "beam.json"), "JSON (*.json)")
                if not value:
                    return False
                path = Path(value)
            data = deepcopy(self.config.data)
            for mapping, key, _ in file_references(data):
                source = resolved_file(mapping[key], self.config.base_dir)
                for cached in self._standalone_sources:
                    asset = next((a for a in cached.assets.values() if (cached.root / a.path).resolve() == source), None)
                    if asset:
                        durable = path.resolve().parent / (path.stem + "_files") / asset.id / asset.original_name
                        if durable.resolve() != source:
                            try:
                                durable.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copyfile(source, durable)
                            except OSError as exc:
                                QMessageBox.warning(self, "保存输入文件失败", str(exc))
                                return False
                        source = durable.resolve()
                        break
                try:
                    mapping[key] = Path(os.path.relpath(source, path.resolve().parent)).as_posix()
                except ValueError:  # Different Windows volumes.
                    mapping[key] = str(source)
            try:
                atomic_write(path, json_bytes(data))
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "保存 JSON 失败", str(exc))
                return False
            self.config.path = str(path.resolve())
            self.config.base_dir = path.resolve().parent
            self.config.data = data
        self.config._form_dirty = self.config._json_dirty = self.config._data_dirty = False
        self.config._sync_editor()
        self.config.cancel_form()
        self.config._set_sync_status("已保存")
        self._update_document_ui()
        return True

    def export_current_json(self):
        if not self._commit_current():
            return
        title = "导出 JSON 副本（不含引用的输入文件）"
        path, _ = QFileDialog.getSaveFileName(self, title, "beam.json", "JSON (*.json)")
        if path:
            try:
                atomic_write(Path(path), json_bytes(self.config.data))
                self.statusBar().showMessage("已导出 JSON 副本；需要携带依赖时请选择“导出可运行输入包”。", 10000)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "导出失败", str(exc))

    def export_bundle(self):
        if not self._commit_current():
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出运行页所选输入及依赖", "pass-inputs.zip", "ZIP (*.zip)")
        if not path:
            return
        temporary = None
        try:
            project = self.project
            if project:
                ids = self.run.selected_input_ids()
            else:
                temporary = project = Project()
                ids = [project.add_config("beam0", self.config.data, self.config.base_dir)]
            project.export_bundle(ids, Path(path))
            self.statusBar().showMessage("已导出输入包；解压后执行 python run.py。", 10000)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
        finally:
            if temporary:
                temporary.close()

    def show_resources(self):
        if self.project and self._commit_current():
            ResourceDialog(self, self.project).exec()

    def browse_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "只读查看其他项目", "", "PASS project (*.passproj)")
        if not path:
            return
        try:
            source = Project.open(Path(path))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "打开失败", str(exc))
            return
        try:
            ResourceDialog(self, source, True).exec()
        finally:
            source.close()

    def copy_project_command(self, source, source_id, name):
        if not self._commit_current():
            return None
        temporary = None
        try:
            if self.project:
                target, target_id = self.project, self._active_input_id
            else:
                # Retain a private asset cache until this standalone document is
                # saved/closed. Save-as then writes durable adjacent input files.
                temporary = target = Project()
                target_id = target.add_config("beam", self.config.data, self.config.base_dir)
            policy = "check"
            difference = source.command_clock_difference(source_id, name, target, target_id)
            if difference:
                choice = QMessageBox(self)
                choice.setWindowTitle("命令的规定时钟不同")
                choice.setText("复制的命令依赖规定时钟。请选择它在目标项目中使用的时钟。")
                choice.setInformativeText("源：" + display_json(difference["source"]) + "\n目标：" + display_json(difference["target"]) +
                                          ("\n" + difference["detail"] if difference.get("detail") else "") + "\n复制源时钟也会影响目标中已有的谐波 RF、Bump 和到达相位切片。")
                keep = choice.addButton("使用目标时钟", QMessageBox.AcceptRole)
                copy_clock = choice.addButton("复制源时钟", QMessageBox.ActionRole)
                choice.addButton(QMessageBox.Cancel)
                choice.exec()
                if choice.clickedButton() not in (keep, copy_clock):
                    return None
                policy = "target" if choice.clickedButton() == keep else "source"
            copied = source.copy_command(source_id, name, target, target_id, clock_policy=policy)
            if self.project:
                self._activate_project_input(target_id)
            else:
                data = deepcopy(target.configs[target_id].data)
                for mapping, key, _ in file_references(data):
                    mapping[key] = str(resolved_file(mapping[key], target.config_base))
                self.config.data = data
                self._standalone_sources.append(target)
                temporary = None
                self.config._sync_editor()
                self.config._refresh_tree()
            self.config._data_dirty = True
            self.config._select_sequence_item(copied)
            self._update_document_ui()
            return copied
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "复制失败", str(exc))
            return None
        finally:
            if temporary:
                temporary.close()

    @staticmethod
    def _recipe_with_sources(project, recipe):
        value = deepcopy(recipe)
        if project:
            value = deepcopy(next((r for r in project.recipes if r.get("id") == recipe.get("id")), recipe))
            value["source_files"] = {key: str(project.root / project.assets[aid].path) for key, aid in value.get("source_assets", {}).items()}
        return value

    def restore_project_recipe(self, project, recipe):
        if not self.config._confirm_form_navigation():
            return False
        value = self._recipe_with_sources(project, recipe)
        if value.get("kind") == "optics":
            parameters = value["parameters"]
            self.config.configure_optics_generator(parameters["mode"])
            fields = self.config._optics_fields
        elif value.get("kind") == "madx":
            parameters = value["parameters"]
            self.config.configure_madx_import(parameters.get("source_kind", "twiss"))
            fields = self.config._madx_fields
            # Copy sources so closing a read-only source project cannot invalidate
            # a restored generator form before its first preview.
            cache = Project()
            for key, path in value.get("source_files", {}).items():
                asset = cache.add_asset(Path(path), "source")
                parameters[key] = str(cache.root / asset.path)
            self._standalone_sources.append(cache)
        else:
            return False
        from PySide6.QtWidgets import QCheckBox
        for key, value in parameters.items():
            field = fields.get(key)
            if isinstance(field, QLineEdit):
                field.setText(str(value))
            elif isinstance(field, QComboBox):
                field.setCurrentText(str(value))
            elif isinstance(field, QCheckBox):
                field.setChecked(bool(value))
        self._show_page(0)
        return True

    def _update_document_ui(self, *_args):
        if not hasattr(self, "project"):
            return
        project = self.project
        dirty = self._dirty()
        label = (project.path.name if project.path else "未保存项目.passproj") if project else (
            Path(self.config.path).name if self.config.path else "beam.json")
        self.config.file_label.setText(label + (" ●" if dirty else ""))
        self.config.file_label.setToolTip(str(project.path or "") if project else self.config.path)
        self.config.input_selector.setVisible(project is not None)
        self.config.input_count_label.setVisible(project is not None)
        self.config.input_count_label.setText(f"当前输入配置（共 {len(project.configs)} 份）" if project else "当前输入配置")
        self.config.input_selector.setToolTip(f"当前输入配置（共 {len(project.configs)} 份）" if project else "当前输入配置")
        self.config.import_input_button.setVisible(project is not None)
        self.config.contents_button.setVisible(project is not None)
        self.setWindowTitle(f"PASS · {label}" + (" *" if dirty else ""))
        for key in ("save_json", "save_json_as"):
            self.actions[key].setEnabled(project is None)
        for key in ("save_project", "save_project_as", "contents", "import_json"):
            self.actions[key].setEnabled(project is not None)
        self.actions["save_json"].setText("保存 JSON" + ("\tCtrl+S" if project is None else ""))
        self.actions["save_project"].setText("保存项目" + ("\tCtrl+S" if project else ""))
