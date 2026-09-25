"""Cooperative disk tasks with an event-responsive, ownership-safe progress dialog."""

from pathlib import Path
import os
import tempfile
import threading

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout


class TaskCancelled(Exception):
    """Cancellation observed before the operation's publication boundary."""


class TaskContext:

    def __init__(self, cancelled=None, progress=None):
        self.cancelled = cancelled if cancelled is not None else threading.Event()
        self.progress = progress or (lambda _message: None)

    def check(self):
        if self.cancelled.is_set():
            raise TaskCancelled("操作已取消")

    def report(self, message):
        self.check()
        self.progress(message)

    def copy_file(self, source, destination):
        self.check()
        with Path(source).open("rb") as reader, Path(destination).open("wb") as writer:
            while block := reader.read(1024 * 1024):
                self.check()
                writer.write(block)

    def copy_atomic(self, source, destination):
        destination = Path(destination)
        fd, name = tempfile.mkstemp(prefix=".pass-copy-", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            self.copy_file(source, temporary)
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            self.check()
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


class TaskWorker(QThread):
    progress = Signal(str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.context = TaskContext(progress=self.progress.emit)
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.operation(self.context)
        except Exception as exc:
            self.error = exc


class TaskDialog(QDialog):

    def __init__(self, parent, title, operation):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.worker = TaskWorker(operation, self)
        layout = QVBoxLayout(self)
        self.message = QLabel("正在准备…")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        progress = QProgressBar()
        progress.setRange(0, 0)
        layout.addWidget(progress)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.cancel)
        layout.addWidget(self.cancel_button)
        self.worker.progress.connect(self.message.setText)
        self.worker.finished.connect(self.accept)

    def cancel(self):
        self.worker.context.cancelled.set()
        self.cancel_button.setEnabled(False)
        self.message.setText("正在取消，等待当前文件操作结束…")

    def reject(self):
        if self.worker.isRunning():
            self.cancel()
        else:
            super().reject()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.cancel()
            event.ignore()
        else:
            super().closeEvent(event)


def run_task_dialog(parent, title, operation):
    """Return after completion while Qt continues to paint and dispatch events.

    The modal boundary prevents document replacement while a task borrows its
    files. Operations must not access widgets or publish into the live document.
    After their atomic commit they return success even if cancellation races it.
    """
    dialog = TaskDialog(parent, title, operation)
    dialog.worker.start()
    dialog.exec()
    dialog.worker.wait()
    error, result = dialog.worker.error, dialog.worker.result
    dialog.deleteLater()
    if error is not None:
        raise error
    return result
