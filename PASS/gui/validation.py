"""Responsive, filterable validation report with exact JSON-path navigation."""
from copy import deepcopy
import json
from pathlib import Path

from PySide6.QtCore import Qt, QEvent, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from PASS.validation import validate_input, ValidationReport
from PASS.gui.appearance import THEMES


class ValidationWorker(QThread):
    completed = Signal(object)

    def __init__(self, data, base, parent):
        super().__init__(parent)
        self.data, self.base = deepcopy(data), base

    def run(self):
        try:
            report = validate_input(self.data, self.base)
        except Exception as exc:
            report = ValidationReport()
            report.add((), "validation.internal", f"检测未完成：{type(exc).__name__}: {exc}")
        self.completed.emit(report)


class ValidationDialog(QDialog):
    navigate = Signal(object)

    def __init__(self, parent=None, report=None):
        super().__init__(parent)
        self.report, self.worker = report, None
        self.setWindowTitle("JSON 全面检测")
        self.resize(940, 540)
        self.setMinimumSize(640, 320)
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.summary = QLabel()
        top.addWidget(self.summary, 1)
        self.filter = QComboBox()
        self.filter.addItems(["全部问题", "仅错误", "仅警告"])
        self.filter.currentIndexChanged.connect(self.refresh)
        top.addWidget(self.filter)
        root.addLayout(top)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["级别", "位置 / JSON 路径", "检测结果", "规则"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setColumnWidth(0, 58)
        self.table.setColumnWidth(1, 240)
        self.table.setColumnWidth(2, 400)
        self.table.setColumnWidth(3, 160)
        self.table.cellDoubleClicked.connect(self.locate)
        self.table.currentCellChanged.connect(self.show_detail)
        root.addWidget(self.table, 1)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText("选择一项，查看完整位置和检测原因。")
        self.detail.setMaximumHeight(78)
        root.addWidget(self.detail)
        root.addWidget(QLabel("双击定位配置。错误会阻止运行；警告保留运行选择。"))
        actions = QHBoxLayout()
        self.copy_button = QPushButton("复制报告")
        self.copy_button.clicked.connect(lambda: QApplication.clipboard().setText(self.report.text()) if self.report else None)
        self.export_button = QPushButton("导出报告…")
        self.export_button.clicked.connect(self.export_report)
        actions.addWidget(self.copy_button)
        actions.addWidget(self.export_button)
        actions.addStretch()
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.accept)
        actions.addWidget(self.close_button)
        root.addLayout(actions)
        self.refresh()

    def start(self, data, base):
        self.summary.setText("正在检查全部参数、执行依赖和输入文件内容…")
        self.worker = ValidationWorker(data, base, self)
        self.worker.completed.connect(self.set_report)
        self.worker.finished.connect(lambda: self.close_button.setEnabled(True))
        self.close_button.setEnabled(False)
        self.worker.start()

    def set_report(self, report):
        self.report = report
        self.refresh()

    def refresh(self, *_args):
        report = self.report
        self.copy_button.setEnabled(report is not None)
        self.export_button.setEnabled(report is not None)
        if report is None:
            return
        prefix = "全面检测" if report.full else "参数预检（尚未检查文件内容）"
        self.summary.setText(f"{prefix}：{len(report.errors)} 错误 · {len(report.warnings)} 警告 · {report.command_count} 个命令")
        severity = (None, "error", "warning")[self.filter.currentIndex()]
        issues = [d for d in report.diagnostics if severity is None or d.severity == severity]
        self.table.setRowCount(len(issues))
        for row, diagnostic in enumerate(issues):
            for col, value in enumerate(("错误" if diagnostic.severity == "error" else "警告", diagnostic.pointer
                                         or "/", diagnostic.message, diagnostic.code)):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                item.setData(Qt.UserRole, diagnostic)
                if col == 0:
                    dark = self.palette().window().color().lightness() < 128
                    color = THEMES["dark" if dark else "light"][diagnostic.severity]
                    item.setForeground(QColor(color))
                self.table.setItem(row, col, item)
            self.table.setRowHeight(row, 28)
        self.detail.clear()
        if issues:
            self.table.setCurrentCell(0, 0)
            self.show_detail(0)

    def show_detail(self, row, *_args):
        item = self.table.item(row, 0)
        if item and hasattr(self, "detail"):
            issue = item.data(Qt.UserRole)
            self.detail.setPlainText(f"{issue.pointer or '/'}\n{issue.message}\n{issue.code}")

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange) and hasattr(self, "close_button"):
            self.refresh()

    def locate(self, row, _column=0):
        if self.worker and self.worker.isRunning():
            return
        item = self.table.item(row, 0)
        if item:
            self.navigate.emit(item.data(Qt.UserRole))

    def export_report(self):
        if self.report is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出检测报告", "validation-report.json", "JSON (*.json)")
        if path:
            try:
                Path(path).write_text(json.dumps(self.report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                QMessageBox.warning(self, "无法导出", str(exc))

    def done(self, result):
        # Keep the worker alive until its result is delivered. Never terminate
        # a thread while the TFS parser owns a file handle.
        if self.worker and self.worker.isRunning():
            return
        super().done(result)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            event.ignore()
        else:
            super().closeEvent(event)
