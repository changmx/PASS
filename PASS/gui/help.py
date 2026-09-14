"""Help menu, documentation progress and project information dialogs."""
from __future__ import annotations

from email.utils import getaddresses
from html import escape
from importlib.metadata import distribution, PackageNotFoundError
import platform

from PySide6 import __version__ as pyside_version
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel, QMenu, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout,
)

from PASS import __version__
from PASS.gui.appearance import code_font
from PASS.gui.documentation import DocumentationBuilder, local_document, source_checkout


class HelpMenu(QMenu):
    def __init__(self, owner) -> None:
        super().__init__(owner)
        self.owner = owner
        self.root = source_checkout()
        self.urls = {
            "Documentation": "https://changmx.github.io/PASS/",
            "Repository": "https://github.com/changmx/PASS",
            "Issues": "https://github.com/changmx/PASS/issues",
        }
        self.title = "Particle Accelerator Simulation Studio"
        self.authors = "Mingxuan Chang · Jie Liu · Lei Wang"
        self.package = None
        try:
            self.package = distribution("pass-sim")
            metadata = self.package.metadata
            self.title = metadata.get("Summary") or self.title
            names = [name for name, _ in getaddresses(metadata.get_all("Author-email") or []) if name]
            self.authors = " · ".join(names) or metadata.get("Author") or self.authors
            for item in metadata.get_all("Project-URL") or []:
                key, separator, value = item.partition(",")
                if separator and key.strip() in self.urls:
                    self.urls[key.strip()] = value.strip()
        except PackageNotFoundError:
            pass
        self.builder = DocumentationBuilder(self.root, self)
        self.build_dialog = None
        self.about_dialog = None
        self.license_dialog = None
        online = self.addMenu("在线说明文档")
        for language, label in (("zh", "中文"), ("en", "English")):
            online.addAction(label, lambda checked=False, lang=language: self.open_online(lang))
        self.addAction("源代码", lambda: self.open_url(self.urls["Repository"]))
        self.addSeparator()
        local = self.addMenu("阅读本地文档")
        for language, label in (("zh", "中文"), ("en", "English")):
            local.addAction(label, lambda checked=False, lang=language: self.read_local(lang))
        self.rebuild_action = self.addAction("重新编译本地文档", self.rebuild)
        self.log_action = self.addAction("查看编译日志…", self.show_build_dialog)
        self.log_action.setVisible(False)
        self.addSeparator()
        self.addAction("关于 PASS", self.show_about)
        self.builder.running_changed.connect(self._running_changed)
        self.builder.finished.connect(lambda ok, message: owner.statusBar().showMessage(message, 12000))

    def open_url(self, url: str | QUrl) -> None:
        target = url if isinstance(url, QUrl) else QUrl(url)
        if not QDesktopServices.openUrl(target):
            QMessageBox.warning(self.owner, "无法打开链接", f"请使用浏览器打开：\n{target.toString()}")

    def open_online(self, language: str) -> None:
        self.open_url(self.urls["Documentation"].rstrip("/") + f"/{language}/")

    def read_local(self, language: str) -> None:
        page = local_document(self.root, language)
        if page is not None:
            self.open_url(QUrl.fromLocalFile(str(page.resolve())))
            return
        if self.builder.running:
            self.show_build_dialog()
            return
        box = QMessageBox(self.owner)
        box.setWindowTitle("尚无本地文档")
        box.setText("尚未生成本地文档。" if self.root else "当前安装不包含文档源码。请获取完整的 PASS 源码并以可编辑模式安装。")
        build = box.addButton("立即编译", QMessageBox.AcceptRole) if self.root else None
        online = box.addButton("在线文档", QMessageBox.ActionRole)
        box.addButton("关闭", QMessageBox.RejectRole)
        box.exec()
        if build is not None and box.clickedButton() == build:
            self.rebuild()
        elif box.clickedButton() == online:
            self.open_online(language)

    def show_build_dialog(self) -> None:
        if self.build_dialog is None:
            self.build_dialog = DocumentationDialog(self, self.owner)
        self.build_dialog.show()
        self.build_dialog.raise_()
        self.build_dialog.activateWindow()

    def rebuild(self) -> None:
        self.show_build_dialog()
        if self.builder.running:
            return
        self.build_dialog.log.clear()
        try:
            self.builder.start()
        except (OSError, ValueError) as exc:
            self.build_dialog.completed(False, str(exc))
            self.build_dialog.log.setPlainText(str(exc))
        self.log_action.setVisible(True)

    def _running_changed(self, running: bool) -> None:
        self.rebuild_action.setEnabled(not running)
        self.rebuild_action.setText("文档正在编译…" if running else "重新编译本地文档")
        if running:
            self.owner.statusBar().showMessage("正在重新编译本地文档…")

    def confirm_close(self) -> bool:
        if not self.builder.running:
            return True
        return QMessageBox.question(
            self.owner, "文档仍在编译", "停止文档编译并关闭 PASS？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) == QMessageBox.Yes

    def show_about(self) -> None:
        if self.about_dialog is None:
            dialog = self.about_dialog = QDialog(self.owner)
            dialog.setWindowTitle("关于 PASS")
            dialog.setMinimumWidth(520)
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(24, 20, 24, 20)
            layout.setSpacing(14)
            brand = QLabel("PASS")
            brand.setObjectName("brand")
            layout.addWidget(brand)
            info = QLabel(
                f"<b>{escape(self.title)}</b><p>用于粒子加速器束流动力学建模、粒子追踪与分析。</p>"
                f"<p>版本：{escape(__version__)}</p><p>作者：{escape(self.authors)}</p>"
                "<p>中国科学院近代物理研究所<br>"
                "Institute of Modern Physics, Chinese Academy of Sciences</p>"
                '<p>许可证：<a href="license">Apache License 2.0</a></p>'
                "<p>© 2025–2026 Institute of Modern Physics,<br>Chinese Academy of Sciences</p>"
            )
            info.setWordWrap(True)
            info.setTextInteractionFlags(Qt.TextBrowserInteraction)
            info.linkActivated.connect(self.show_license)
            layout.addWidget(info)
            links = QHBoxLayout()
            for label, url in (("在线文档", self.urls["Documentation"]),
                               ("源代码", self.urls["Repository"]), ("问题反馈", self.urls["Issues"])):
                link = QLabel(f'<a href="{escape(url, quote=True)}">{label}</a>')
                link.linkActivated.connect(self.open_url)
                links.addWidget(link)
            links.addStretch()
            layout.addLayout(links)
            row = QHBoxLayout()
            copy = QPushButton("复制版本与环境信息")
            copy.clicked.connect(self.copy_environment)
            row.addWidget(copy)
            row.addStretch()
            close = QPushButton("关闭")
            close.clicked.connect(dialog.close)
            row.addWidget(close)
            layout.addLayout(row)
        self.about_dialog.show()
        self.about_dialog.raise_()
        self.about_dialog.activateWindow()

    def copy_environment(self) -> None:
        QApplication.clipboard().setText(
            f"PASS: {__version__}\nPython: {platform.python_version()}\n"
            f"PySide6: {pyside_version}\nOS: {platform.platform()}\nArchitecture: {platform.machine()}"
        )
        self.owner.statusBar().showMessage("已复制版本与环境信息。", 5000)

    def show_license(self, _link=None) -> None:
        if self.license_dialog is None:
            candidates = [self.root / "LICENSE"] if self.root else []
            if self.package is not None:
                candidates.extend(self.package.locate_file(path) for path in self.package.files or []
                                  if path.name == "LICENSE")
            content = None
            for path in candidates:
                try:
                    content = path.read_text(encoding="utf-8")
                    break
                except (OSError, UnicodeError):
                    continue
            if content is None:
                self.open_url(self.urls["Repository"].rstrip("/") + "/blob/HEAD/LICENSE")
                return
            dialog = self.license_dialog = QDialog(self.about_dialog or self.owner)
            dialog.setWindowTitle("PASS · Apache License 2.0")
            dialog.resize(760, 560)
            layout = QVBoxLayout(dialog)
            text = QPlainTextEdit(content)
            text.setReadOnly(True)
            text.setFont(code_font())
            layout.addWidget(text)
            close = QPushButton("关闭")
            close.clicked.connect(dialog.close)
            layout.addWidget(close, alignment=Qt.AlignRight)
        self.license_dialog.show()
        self.license_dialog.raise_()


class DocumentationDialog(QDialog):
    def __init__(self, menu: HelpMenu, parent) -> None:
        super().__init__(parent)
        self.menu = menu
        self.setWindowTitle("重新编译本地文档")
        self.resize(820, 520)
        layout = QVBoxLayout(self)
        self.state = QLabel("准备编译中文和英文文档。")
        self.state.setWordWrap(True)
        self.state.setTextFormat(Qt.PlainText)
        layout.addWidget(self.state)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(code_font())
        layout.addWidget(self.log, 1)
        row = QHBoxLayout()
        self.open_buttons = {}
        for language, label in (("zh", "打开中文文档"), ("en", "打开英文文档")):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, lang=language: menu.read_local(lang))
            button.setEnabled(local_document(menu.root, language) is not None)
            self.open_buttons[language] = button
            row.addWidget(button)
        row.addStretch()
        self.stop = QPushButton("停止编译")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(menu.builder.stop)
        row.addWidget(self.stop)
        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        row.addWidget(close)
        layout.addLayout(row)
        note = QLabel("关闭此窗口后编译仍在后台继续；可从帮助菜单查看编译日志。")
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        menu.builder.output.connect(self.append_output)
        menu.builder.running_changed.connect(self.running_changed)
        menu.builder.finished.connect(self.completed)

    def append_output(self, text: str) -> None:
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)

    def running_changed(self, running: bool) -> None:
        self.progress.setVisible(running)
        self.stop.setEnabled(running)
        if running:
            self.state.setText("正在重新编译中文和英文文档…")

    def completed(self, success: bool, message: str) -> None:
        self.state.setText(message)
        for language, button in self.open_buttons.items():
            button.setEnabled(local_document(self.menu.root, language) is not None)
