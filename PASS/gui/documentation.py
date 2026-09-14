"""Local documentation discovery and asynchronous, full Sphinx rebuilds."""
from __future__ import annotations

import codecs
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
from uuid import uuid4

from PySide6.QtCore import QObject, QProcess, Signal

from PASS.gui.project import atomic_write


def source_checkout() -> Path | None:
    """Find the checkout containing this installed module, never the user's cwd."""
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "PASS" / "__init__.py").is_file():
        return root
    return None


def local_document(root: Path | None, language: str) -> Path | None:
    """Prefer the last successful GUI build; accept the conventional CLI output."""
    if root is None or language not in ("zh", "en"):
        return None
    build = root / "docs" / "build"
    try:
        name = json.loads((build / "gui" / "latest.json").read_text(encoding="utf-8"))["directory"]
        # The manifest identifies a single child directory, never an arbitrary path.
        if isinstance(name, str) and name not in ("", ".", "..") and Path(name).name == name:
            page = build / "gui" / name / language / "index.html"
            if page.is_file():
                return page
    except (OSError, ValueError, KeyError, TypeError):
        pass
    page = build / "html" / language / "index.html"
    return page if page.is_file() else None


class DocumentationBuilder(QObject):
    """Publish a new reading location only after both language indexes exist."""

    output = Signal(str)
    running_changed = Signal(bool)
    finished = Signal(bool, str)

    def __init__(self, root: Path | None, parent=None) -> None:
        super().__init__(parent)
        self.root = root
        self.process: QProcess | None = None
        self.output_dir: Path | None = None
        self.running = False
        self._stopped = False
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def start(self) -> None:
        if self.running:
            return
        if self.root is None:
            raise ValueError("当前安装中没有 PASS 源码目录。请从源代码链接获取项目并以可编辑模式安装。")
        source = self.root / "docs" / "source"
        for relative in ("conf.py", "index.rst", "zh/index.rst", "en/index.rst"):
            if not (source / relative).is_file():
                raise ValueError(f"缺少文档源码：{source / relative}\n请获取完整的 PASS 源码。")
        missing = [name for name in ("sphinx", "sphinx_rtd_theme")
                   if importlib.util.find_spec(name) is None]
        if missing:
            raise ValueError(
                "缺少文档编译依赖：" + ", ".join(missing)
                + f'\n请使用以下 Python 环境，在 PASS 源码目录安装依赖：\n{sys.executable}'
                + '\npython -m pip install --editable ".[gui,docs]"'
            )
        name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        self.output_dir = self.root / "docs" / "build" / "gui" / name
        self.output_dir.mkdir(parents=True, exist_ok=False)
        if self.process is not None:
            self.process.deleteLater()
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments([
            "-u", "-X", "utf8", "-m", "sphinx", "-E", "-a", "-N", "-W",
            "-b", "html", str(source), str(self.output_dir),
        ])
        self.process.setWorkingDirectory(str(self.root))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self._decoder.reset()
        self._stopped = False
        self.running = True
        self.running_changed.emit(True)
        self.output.emit(f"Python：{sys.executable}\n源码：{source}\n输出：{self.output_dir}\n\n")
        self.process.start()

    def _read_output(self) -> None:
        text = self._decoder.decode(bytes(self.process.readAllStandardOutput()))
        if text:
            self.output.emit(text)

    def _error(self, error) -> None:
        self.output.emit(self.process.errorString() + "\n")
        if error == QProcess.FailedToStart:
            self._finished(-1, QProcess.CrashExit)

    def _finished(self, code: int, status) -> None:
        if not self.running:
            return
        self._read_output()
        tail = self._decoder.decode(b"", final=True)
        if tail:
            self.output.emit(tail)
        success = not self._stopped and code == 0 and status == QProcess.NormalExit
        message = "编译成功，可以阅读中文和英文文档。"
        if success:
            try:
                for relative in ("index.html", "zh/index.html", "en/index.html"):
                    if not (self.output_dir / relative).is_file():
                        raise OSError(f"缺少编译结果：{relative}")
                atomic_write(self.output_dir.parent / "latest.json",
                             json.dumps({"directory": self.output_dir.name}).encode("utf-8"))
            except OSError as exc:
                success = False
                message = f"无法更新本地文档入口：{exc}"
        else:
            message = "编译已停止。" if self._stopped else f"编译失败（退出码 {code}），请查看日志。"
        if not success:
            message += " 若已有成功编译的文档，仍可继续阅读。"
        self.running = False
        self.running_changed.emit(False)
        self.output.emit("\n" + message + "\n")
        self.finished.emit(success, message)

    def stop(self) -> None:
        if self.running:
            self._stopped = True
            self.process.kill()

    def shutdown(self) -> None:
        """Reap the child before its owning window is destroyed."""
        self.stop()
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self.process.waitForFinished(3000)
