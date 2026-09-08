"""PASS GUI application.

The GUI is intentionally an orchestration layer. PASS schemas and the tracking
engine remain the source of truth; this module handles project files, process
control, and presentation.
"""

from __future__ import annotations

import csv
from collections import Counter
from copy import deepcopy
import ctypes
import json
import re
import sys
import time
from pathlib import Path

from PySide6.QtCore import QProcess, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QFont, QIntValidator, QPainter, QPalette, QPen, QTextFormat
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QCheckBox,
    QDialog,
    QInputDialog,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QHeaderView,
    QToolButton,
)

from PASS import __version__


BLUE = "#61afef"
PANEL = "#21252b"
BASE = "#282c34"
MUTED = "#8b93a1"


# These values are constrained by PASS command implementations.  Unknown
# string fields deliberately remain text fields so newer schemas stay usable.
ENUM_OPTIONS = {
    "Backend (gpu/cpu)": ("cpu", "gpu"),
    "Aperture type": (
        "off", "default", "circle", "rectangle", "ellipse", "rectcircle",
        "rectellipse", "racetrack", "octagon", "polygon",
    ),
    "Integrator": ("adaptive", "uniform", "yoshida4"),
    "Model": ("adaptive", "drift-kick-drift-exact", "mat-kick-mat"),
    "Slice model": ("equal_length", "equal_particle", "equal_charge", "Equal particle"),
    "Z range mode": ("auto", "explicit"),
    "Direction": ("x", "y"),
    "Mode": ("single_fm", "single_fm_am", "dual_fm", "dual_fm_am"),
    "Transverse dist": ("kv", "gaussian", "uniform", "waterbag", "parabolic"),
    "Longitudinal dist": ("gaussian", "coasting", "matchz", "matchdp"),
    "Particle Precision": ("float32", "float64"),
    "Longitudinal transfer": ("off", "drift", "matrix"),
    "Field solver": ("fd", "dst_rectangle", "fft_green"),
    "Method": ("fd", "dst_rectangle", "fft_green"),
    "Particle Deposition Method": ("CIC", "TSC"),
    "File Time Kind": ("turn", "second"),
}

TIMING_MODE_OPTIONS = ("off", "turn", "command", "synchronized-command")

# Descriptions are copied from the public schema lazily by ``_schema_help``.
# Keeping this lookup in the GUI means new schema fields automatically get the
# same hover help without duplicating every description here.
_SCHEMA_HELP: dict[str, str] | None = None

FIELD_HELP = {
    "S (m)": "Command 在环中的纵向位置，单位为 m。",
    "S previous (m)": "Twiss 传输矩阵的上一光学点位置，单位为 m。",
    "Harmonic Number": "束团分组数；添加或删除 bunch 时由界面自动保持一致。",
    "Harmonic ID of this bunch": "该 bunch 的零起始分组编号，由界面按顺序维护。",
    "Random Seed": "分布生成随机种子。留空（null）时每次运行使用非确定性随机数。",
    "Timing": "运行进度和 ETA 的输出方式。",
    "Device Id": "GPU 后端使用的设备编号列表。",
    "Insert Particle Coordinate": "手动输入粒子坐标数组 [[x, px, y, py, z, dp], ...]。",
}


def button(text: str, object_name: str = "") -> QPushButton:
    item = QPushButton(text)
    if object_name:
        item.setObjectName(object_name)
    item.setCursor(Qt.PointingHandCursor)
    return item


class PropertyComboBox(QComboBox):
    """A property selector changed only through its drop-down list."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


class CollapsibleSection(QWidget):
    """A small independently toggled section for the vertical component library."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.header = QToolButton()
        self.header.setObjectName("librarySectionHeader")
        self.header.setText(title)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setLayoutDirection(Qt.LeftToRight)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.setStyleSheet("text-align: left;")
        self.header.setArrowType(Qt.RightArrow)
        self.header.setCheckable(True)
        self.header.setChecked(False)
        self.header.toggled.connect(self._set_expanded)

        self.body = QWidget()
        self.body.setObjectName("librarySectionBody")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(10, 3, 4, 8)
        self.body_layout.setSpacing(4)
        self.body.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header_row = QHBoxLayout()
        self.header_row.setContentsMargins(0, 0, 0, 0)
        self.header_row.setSpacing(6)
        self.header_row.addWidget(self.header, 1)
        layout.addLayout(self.header_row)
        layout.addWidget(self.body)

    def _set_expanded(self, expanded: bool) -> None:
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.body.setVisible(expanded)


class BusyProgressBar(QWidget):
    """Small indeterminate progress bar rendered without native Qt styling."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("busyProgress")
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self._offset = -0.25
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self.setFixedHeight(4)
        self.setMinimumWidth(80)

    def _advance(self) -> None:
        self._offset += 0.035
        if self._offset > 1.0:
            self._offset = -0.25
        self.update()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._timer.start(30)

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#11151a"))
        painter.drawRect(self.rect())
        width = max(48, int(self.width() * 0.22))
        x = int((self.width() + width) * self._offset - width)
        painter.fillRect(x, 0, width, self.height(), QColor("#8b93a1"))


class LineNumberEditor(QPlainTextEdit):
    """Plain-text editor with a compact, non-editable line-number gutter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.line_number_area = QWidget(self)
        self.line_number_area.setObjectName("lineNumberArea")
        self.line_number_area.paintEvent = self._paint_line_numbers
        self.blockCountChanged.connect(self._update_line_number_width)
        self.updateRequest.connect(self._update_line_number_area)
        self._update_line_number_width(0)

    def _line_number_width(self) -> int:
        digits = len(str(max(1, self.blockCount())))
        return 10 + self.fontMetrics().horizontalAdvance("9") * digits + 10

    def _update_line_number_width(self, _count: int) -> None:
        self.setViewportMargins(self._line_number_width(), 0, 0, 0)

    def _update_line_number_area(self, rect, dy: int) -> None:
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), self.line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_number_width(0)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        contents = self.contentsRect()
        self.line_number_area.setGeometry(contents.left(), contents.top(), self._line_number_width(), contents.height())

    def _paint_line_numbers(self, event) -> None:
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor("#161a20"))
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(QColor("#6b7788"))
                painter.drawText(0, top, self.line_number_area.width() - 6, self.fontMetrics().height(), Qt.AlignRight, str(block_number + 1))
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1


class PlotCanvas(QWidget):
    """Small dependency-free line plot for CSV/TFS numeric columns."""

    def __init__(self) -> None:
        super().__init__()
        self.x_values: list[float] = []
        self.values: list[float] = []
        self._x_limits: tuple[float, float] | None = None
        self._y_limits: tuple[float, float] | None = None
        self.setMinimumHeight(300)

    def set_series(self, x_values: list[float], values: list[float]) -> None:
        length = min(len(x_values), len(values))
        self.x_values = x_values[:length]
        self.values = values[:length]
        self.fit_view()

    def fit_view(self) -> None:
        if self.x_values and self.values:
            self._x_limits = (min(self.x_values), max(self.x_values))
            self._y_limits = (min(self.values), max(self.values))
        else:
            self._x_limits = self._y_limits = None
        self.update()

    def set_values(self, values: list[float]) -> None:
        """Compatibility helper that uses the row index as the X coordinate."""
        self.set_series(list(range(len(values))), values)

    def _plot_area(self):
        return self.rect().adjusted(52, 24, -24, -42)

    @staticmethod
    def _expanded_range(low: float, high: float) -> tuple[float, float]:
        if low == high:
            padding = abs(low) * 0.05 or 1.0
            return low - padding, high + padding
        return low, high

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        if len(self.values) < 2 or self._x_limits is None or self._y_limits is None:
            event.ignore()
            return
        area = self._plot_area()
        if not area.contains(event.position().toPoint()):
            event.ignore()
            return
        factor = 0.8 if event.angleDelta().y() > 0 else 1.25
        x_ratio = (event.position().x() - area.left()) / max(area.width(), 1)
        y_ratio = 1.0 - (event.position().y() - area.top()) / max(area.height(), 1)
        x_low, x_high = self._x_limits
        y_low, y_high = self._y_limits
        x_anchor = x_low + (x_high - x_low) * x_ratio
        y_anchor = y_low + (y_high - y_low) * y_ratio
        self._x_limits = (x_anchor - (x_anchor - x_low) * factor, x_anchor + (x_high - x_anchor) * factor)
        self._y_limits = (y_anchor - (y_anchor - y_low) * factor, y_anchor + (y_high - y_anchor) * factor)
        self.update()
        event.accept()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1b1f24"))
        painter.setRenderHint(QPainter.Antialiasing)
        area = self._plot_area()
        painter.setPen(QPen(QColor("#3b4350"), 1))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = area.bottom() - fraction * area.height()
            painter.drawLine(area.left(), int(y), area.right(), int(y))
        if len(self.values) < 2 or self._x_limits is None or self._y_limits is None:
            painter.setPen(QColor(MUTED))
            painter.drawText(area, Qt.AlignCenter, "选择 X / Y 数值列以绘图")
            return
        x_low, x_high = self._expanded_range(*self._x_limits)
        low, high = self._expanded_range(*self._y_limits)
        x_span = x_high - x_low
        span = high - low
        points = []
        for x_value, value in zip(self.x_values, self.values):
            x = area.left() + (x_value - x_low) / x_span * area.width()
            y = area.bottom() - (value - low) / span * area.height()
            points.append((int(x), int(y)))
        painter.setPen(QPen(QColor(BLUE), 2))
        for first, second in zip(points, points[1:]):
            painter.drawLine(*first, *second)
        painter.setPen(QColor(MUTED))
        painter.drawText(8, area.top() + 5, f"max {high:.5g}")
        painter.drawText(8, area.bottom(), f"min {low:.5g}")
        painter.drawText(area.left(), self.height() - 12, f"{x_low:.5g}")
        painter.drawText(area.right() - 65, self.height() - 12, f"{x_high:.5g}")


class ConfigPage(QWidget):
    file_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        # Start from the public root schema so a new project is already a
        # complete PASS input skeleton rather than a Sequence-only fragment.
        from PASS.para.schema.main import MainConfig

        self.data: dict = MainConfig().model_dump(by_alias=True)
        self.data["Sequence"] = {}
        self.path: str = ""
        self._selected_mapping: dict | None = None
        self._selected_path: tuple[str, str | None] | None = None
        self._pending_command: str | None = None
        self._form_fields: dict[str, QWidget] = {}
        self._bunch_fields: dict[tuple[str, str | None], QWidget] = {}
        self._name_field: QLineEdit | None = None
        self._bunch_selector: QComboBox | None = None
        self._active_bunch_key: str | None = None
        self._injection_pending = False
        self._madx_fields: dict[str, QWidget] = {}
        self._madx_preview: tuple[list, list[str], float] | None = None
        self._madx_preview_signature: tuple | None = None
        self._timing_fields: dict[str, QWidget] = {}
        self._space_charge_fields: dict[str, QWidget] = {}
        self._space_charge_selector: PropertyComboBox | None = None
        self._space_charge_name_field: QLineEdit | None = None
        self._space_charge_enabled_field: QCheckBox | None = None
        self._active_space_charge_configuration: str | None = None
        self._editor_syncing = False
        self._json_dirty = False
        self._form_dirty = False
        self._data_dirty = False
        self._validation_issues: list[str] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("配置"))
        self.file_label = QLabel("尚未加载文件")
        self.file_label.setObjectName("muted")
        toolbar.addWidget(self.file_label, 1)
        self.sync_status = QLabel("配置已同步")
        self.sync_status.setObjectName("syncStatus")
        toolbar.addWidget(self.sync_status)
        self.validation_label = QPushButton()
        self.validation_label.setObjectName("validationStatus")
        self.validation_label.setToolTip("配置校验状态")
        self.validation_label.setFlat(True)
        self.validation_label.setCursor(Qt.PointingHandCursor)
        self.validation_label.clicked.connect(self._show_validation_issues)
        toolbar.addWidget(self.validation_label)
        load = button("打开 JSON")
        load.setToolTip("打开一个已有的 PASS JSON 配置")
        load.clicked.connect(self.load_json)
        save = button("保存 JSON", "primary")
        save.setToolTip("将当前配置保存到已打开的 JSON 文件")
        save.clicked.connect(self.save_json)
        save_as = button("另存为")
        save_as.clicked.connect(self.export_json)
        toolbar.addWidget(load)
        toolbar.addWidget(save)
        toolbar.addWidget(save_as)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("配置与组件库"))
        library_scroll = QScrollArea()
        library_scroll.setObjectName("libraryScroll")
        library_scroll.setWidgetResizable(True)
        library_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        library_body = QWidget()
        library_layout = QVBoxLayout(library_body)
        library_layout.setContentsMargins(0, 4, 4, 4)
        library_layout.setSpacing(2)
        self.library_sections: dict[str, CollapsibleSection] = {}

        def add_section(title: str, entries: tuple[tuple[str, str, object], ...], expanded: bool = False) -> None:
            section = CollapsibleSection(title)
            self.library_sections[title] = section
            for text, tip, handler in entries:
                item = button(text)
                item.setToolTip(tip)
                item.clicked.connect(handler)
                section.body_layout.addWidget(item)
            section.body_layout.addStretch()
            section.header.setChecked(expanded)
            library_layout.addWidget(section)

        add_section(
            "必需项",
            (
                ("全局配置  · 必需", "编辑 PASS 输入 JSON 根对象中的全部全局配置。", self.configure_global),
                ("Injection  · 必需", "必需项：Sequence 中需要一个 Injection command。", lambda: self.select_command("Injection")),
            ),
            expanded=True,
        )
        add_section(
            "Twiss 与光学",
            (
                ("从 MAD-X 文件导入 Twiss 点", "读取 MAD-X 导出的 Twiss/TFS 文件，转换为 Twiss command 并追加到 Sequence。", self.configure_madx_twiss),
                ("Twiss", "浏览 Twiss 光学传输参数；确认后才插入 Sequence。", lambda checked=False: self.select_command("Twiss")),
            ),
        )
        add_section(
            "序列工具",
            tuple((command, "浏览默认参数；确认后才插入 Sequence。", lambda checked=False, cmd=command: self.select_command(cmd))
                  for command in ("SortBunch", "ReorganizeBunch", "Slicer")),
        )
        add_section(
            "元件",
            (
                ("从 MAD-X 文件导入元件", "读取 MAD-X 导出的 Twiss/TFS 表，转换为 PASS 元件并追加到 Sequence。", self.configure_madx_elements),
            ) + tuple((command, "浏览默认参数；确认后才插入 Sequence。", lambda checked=False, cmd=command: self.select_command(cmd))
                  for command in ("Marker", "Drift", "SBend", "Quadrupole", "Sextupole", "Octupole", "Multipole", "Solenoid", "Kicker", "ElSeparator", "RFCavity", "Exciter")),
        )
        add_section(
            "监测与诊断",
            tuple((command, "浏览默认参数；确认后才插入 Sequence。", lambda checked=False, cmd=command: self.select_command(cmd))
                  for command in ("StatMonitor", "ParticleMonitor", "DistMonitor", "PhaseAdvanceMonitor")),
        )
        add_section(
            "物理模块",
            (
                ("空间电荷全局配置", "编辑顶层 Space charge 及其命名资源配置。", self.configure_space_charge),
                ("空间电荷计算点", "浏览并插入引用全局配置的 SpaceCharge command。", lambda: self.select_command("SpaceCharge")),
                ("束束效应（待实现）", "束束效应模块尚未接入。", lambda checked=False: None),
            ),
        )
        self.library_sections["物理模块"].body_layout.itemAt(2).widget().setEnabled(False)
        library_layout.addStretch()
        library_scroll.setWidget(library_body)
        left_layout.addWidget(library_scroll, 1)
        splitter.addWidget(left_frame)
        editor_frame = QFrame()
        editor_layout = QVBoxLayout(editor_frame)
        self.editor_tabs = QTabWidget()
        overview_panel = QWidget()
        overview_layout = QVBoxLayout(overview_panel)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        self.tree_section = CollapsibleSection("全局配置与 Sequence")
        self.tree_section.setObjectName("treeSection")
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._tree_clicked)
        self.tree_section.body_layout.addWidget(self.tree)
        self.tree_section.header.setChecked(False)
        overview_layout.addWidget(self.tree_section)
        self.sequence_section = CollapsibleSection("执行序列详情（Sequence）")
        self.sequence_section.setObjectName("sequenceSection")
        sequence_toolbar = QHBoxLayout()
        sequence_toolbar.addWidget(QLabel("过滤"))
        self.sequence_filter = QLineEdit()
        self.sequence_filter.setPlaceholderText("名称或 Command")
        self.sequence_filter.setClearButtonEnabled(True)
        self.sequence_filter.textChanged.connect(self._filter_sequence_table)
        sequence_toolbar.addWidget(self.sequence_filter, 1)
        sequence_toolbar.addWidget(QLabel("Command"))
        self.sequence_command_filter = QComboBox()
        self.sequence_command_filter.setObjectName("sequenceCommandFilter")
        self.sequence_command_filter.addItem("全部 Command")
        self.sequence_command_filter.currentTextChanged.connect(self._filter_sequence_table)
        sequence_toolbar.addWidget(self.sequence_command_filter)
        refresh_sequence = button("刷新")
        refresh_sequence.setToolTip("根据当前 JSON 重新生成 Sequence 表格")
        refresh_sequence.clicked.connect(self._refresh_sequence_table)
        sequence_toolbar.addWidget(refresh_sequence)
        delete_sequence = button("删除选中")
        delete_sequence.setToolTip("删除执行序列中当前选中的一个或多个 command")
        delete_sequence.clicked.connect(self.delete_selected_sequence_rows)
        sequence_toolbar.addWidget(delete_sequence)
        self.sequence_table = QTableWidget(0, 4)
        self.sequence_table.setHorizontalHeaderLabels(["名称", "Command", "S (m)", "状态"])
        self.sequence_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sequence_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.sequence_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sequence_table.setAlternatingRowColors(True)
        self.sequence_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.sequence_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.sequence_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.sequence_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.sequence_table.cellClicked.connect(self._sequence_row_clicked)
        self.sequence_section.body_layout.addLayout(sequence_toolbar)
        self.sequence_section.body_layout.addWidget(self.sequence_table, 1)
        self.sequence_section.header.setChecked(True)
        overview_layout.addWidget(self.sequence_section, 1)
        self.editor_tabs.addTab(overview_panel, "配置概览")
        self.editor = LineNumberEditor()
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor.setFont(QFont("Cascadia Code", 10))
        self.editor.setPlainText(json.dumps(self.data, indent=4, ensure_ascii=False))
        self.editor.textChanged.connect(self._mark_json_dirty)
        self.editor_tabs.addTab(self.editor, "JSON 源码")
        editor_layout.addWidget(self.editor_tabs)
        splitter.addWidget(editor_frame)
        self.form_title = QLabel("参数配置")
        self.form_title.setObjectName("formTitle")
        form_frame = QFrame()
        form_layout = QVBoxLayout(form_frame)
        form_layout.setContentsMargins(8, 0, 0, 0)
        form_layout.addWidget(self.form_title)
        self.form_hint = QLabel("在配置概览中选择配置组或 Sequence command。")
        self.form_hint.setObjectName("muted")
        self.form_hint.setWordWrap(True)
        form_layout.addWidget(self.form_hint)
        legend = QHBoxLayout()
        legend.setSpacing(6)
        for text, object_name in (("只读", "legendReadonly"), ("选项", "legendChoice"), ("输入", "legendInput"), ("复合", "legendJson")):
            marker = QLabel(text)
            marker.setObjectName(object_name)
            marker.setAlignment(Qt.AlignCenter)
            legend.addWidget(marker)
        legend.addStretch()
        form_layout.addLayout(legend)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.form_body = QWidget()
        self.form_layout = QFormLayout(self.form_body)
        self.form_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
        self.form_layout.setFormAlignment(Qt.AlignTop)
        self.form_layout.setHorizontalSpacing(12)
        self.form_layout.setVerticalSpacing(8)
        scroll.setWidget(self.form_body)
        form_layout.addWidget(scroll, 1)
        actions = QHBoxLayout()
        self.form_apply = button("确认修改", "primary")
        self.form_apply.setEnabled(False)
        self.form_apply.clicked.connect(self.apply_form)
        actions.addWidget(self.form_apply)
        self.madx_import_button = button("导入到 Sequence", "primary")
        self.madx_import_button.setToolTip("读取所选 TFS，并把项目追加到当前 Sequence。已有项目不会被覆盖。")
        self.madx_import_button.setVisible(False)
        self.madx_import_button.clicked.connect(self.import_madx_twiss)
        actions.addWidget(self.madx_import_button)
        self.madx_preview_button = button("预览导入")
        self.madx_preview_button.setToolTip("只读取并预览 MAD-X 文件，不修改当前 Sequence")
        self.madx_preview_button.setVisible(False)
        self.madx_preview_button.clicked.connect(self.preview_madx_import)
        actions.addWidget(self.madx_preview_button)
        self.insert_button = button("插入到 Sequence", "primary")
        self.insert_button.setVisible(False)
        self.insert_button.clicked.connect(self.insert_pending_command)
        actions.addWidget(self.insert_button)
        self.duplicate_button = button("复制")
        self.duplicate_button.setVisible(False)
        self.duplicate_button.clicked.connect(self.duplicate_selected)
        actions.addWidget(self.duplicate_button)
        self.delete_button = button("删除")
        self.delete_button.setVisible(False)
        self.delete_button.clicked.connect(self.delete_selected)
        actions.addWidget(self.delete_button)
        form_layout.addLayout(actions)
        # Keep labels and editors readable while allowing the center overview
        # to consume the remaining space on both desktop and small screens.
        form_frame.setFixedWidth(500)
        splitter.addWidget(form_frame)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 680, 500])
        root.addWidget(splitter, 1)
        self._refresh_tree()

    def load_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "加载 PASS JSON", "", "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as stream:
                data = json.load(stream)
            if not isinstance(data, dict):
                raise ValueError("JSON 根对象必须是 object")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return
        self.data, self.path = data, path
        self._sync_editor()
        self._data_dirty = self._json_dirty = self._form_dirty = False
        self.file_label.setText(path)
        self.file_changed.emit(path)
        self._refresh_tree()
        self._clear_form()

    def has_unsaved_changes(self) -> bool:
        """Return whether the current project has changes not written to disk."""
        return self._data_dirty or self._json_dirty or self._form_dirty

    def save_json(self) -> bool:
        """Validate the editor and save to the currently opened file."""
        if not self.path:
            return self.export_json()
        if self._json_dirty and not self.apply_json():
            return False
        if self._form_dirty:
            self._set_sync_status("存在未确认的表单修改", "warning")
            return False
        try:
            with open(self.path, "w", encoding="utf-8") as stream:
                json.dump(self.data, stream, indent=4, ensure_ascii=False)
        except OSError as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return False
        self.file_changed.emit(self.path)
        self._data_dirty = self._json_dirty = self._form_dirty = False
        self._set_sync_status("已保存", "ok")
        return True

    def apply_json(self) -> bool:
        try:
            data = json.loads(self.editor.toPlainText())
            if not isinstance(data, dict):
                raise ValueError("JSON 根对象必须是 object")
        except json.JSONDecodeError as exc:
            message = f"JSON 无效（第 {exc.lineno} 行，第 {exc.colno} 列）：{exc.msg}"
            self._set_validation_status("JSON 无效", "error", message)
            cursor = self.editor.textCursor()
            cursor.setPosition(exc.pos)
            self.editor.setTextCursor(cursor)
            self.editor.ensureCursorVisible()
            QMessageBox.warning(self, "JSON 无效", message)
            return False
        except ValueError as exc:
            QMessageBox.warning(self, "JSON 无效", str(exc))
            return False
        self.data = data
        self._json_dirty = False
        self._form_dirty = False
        self._data_dirty = True
        self._refresh_tree()
        self._clear_form()
        self.file_changed.emit(self.path)
        self._set_sync_status("JSON 修改已确认", "ok")
        return True

    def export_json(self) -> bool:
        if self._json_dirty and not self.apply_json():
            return False
        if self._form_dirty:
            self._set_sync_status("存在未确认的表单修改", "warning")
            return False
        path, _ = QFileDialog.getSaveFileName(self, "导出 PASS JSON", self.path or "beam0.json", "JSON files (*.json)")
        if not path:
            return False
        try:
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(self.data, stream, indent=4, ensure_ascii=False)
        except OSError as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return False
        self.path = path
        self.file_label.setText(path)
        self.file_changed.emit(path)
        self._data_dirty = self._json_dirty = self._form_dirty = False
        self._set_sync_status("已导出", "ok")
        return True

    def save_pending_changes(self) -> bool:
        """Apply pending editors and persist the project, if possible."""
        if self._json_dirty and not self.apply_json():
            return False
        if self._form_dirty:
            self.apply_form()
            if self._form_dirty:
                return False
        if not self._data_dirty:
            return True
        return self.save_json()

    def _mark_json_dirty(self) -> None:
        if self._editor_syncing:
            return
        self._json_dirty = True
        self._set_sync_status("JSON 有未确认修改", "warning")

    def _set_sync_status(self, text: str, state: str = "") -> None:
        self.sync_status.setText(text)
        self.sync_status.setProperty("state", state)
        self.sync_status.style().unpolish(self.sync_status)
        self.sync_status.style().polish(self.sync_status)

    def _command_template(self, command: str) -> dict:
        """Return a safe preview template without changing project data."""
        sequence = self.data.get("Sequence", {})
        if not isinstance(sequence, dict):
            return {}
        position = max(
            (item.get("S (m)", 0.0) for item in sequence.values() if isinstance(item, dict)),
            default=0.0,
        )
        if command == "Injection":
            # Reuse the public schema so an inserted beam source is complete.
            from PASS.para.schema.bunch import InjectionItem

            template = InjectionItem().to_sequence_dict()
            template["S (m)"] = position
            return template
        if command == "Twiss":
            from PASS.para.schema.twiss import TwissPoint

            return TwissPoint(
                s=position, s_previous=position,
                alpha_x=0.0, alpha_y=0.0, beta_x=1.0, beta_y=1.0,
                mu_x=0.0, mu_y=0.0, dx=0.0, dpx=0.0,
                alpha_x_previous=0.0, alpha_y_previous=0.0,
                beta_x_previous=1.0, beta_y_previous=1.0,
                mu_x_previous=0.0, mu_y_previous=0.0,
            ).model_dump(by_alias=True)
        from PASS.para.schema.elements import ELEMENT_REGISTRY
        from PASS.para.schema.monitors import DistMonitor, ParticleMonitor, PhaseAdvanceMonitor, StatMonitor
        from PASS.para.schema.slicer import Slicer

        element = ELEMENT_REGISTRY.get(command.casefold())
        if element is not None:
            required = {"S (m)": position}
            if command == "Exciter":
                required.update({
                    "Mode": "single_fm", "Direction": "x", "Start turn": 0, "End turn": -1,
                    "Voltage (V)": 0.0, "Gap (m)": 0.0, "Plate length (m)": 0.0,
                    "Period (s)": 1.0, "FM dual frequency (Hz)": 0.0, "AM t ext (s)": 0.0,
                    "AM r0 (m)": 0.0, "AM delta0": 0.0, "AM k const": 0.0,
                })
            return element(**required).model_dump(by_alias=True)
        monitor_models = {
            "StatMonitor": (StatMonitor, {}),
            "DistMonitor": (DistMonitor, {}),
            "ParticleMonitor": (ParticleMonitor, {"Max tag": 1}),
            "PhaseAdvanceMonitor": (
                PhaseAdvanceMonitor,
                {"Beta x (m)": 1.0, "Beta y (m)": 1.0, "Alpha x": 0.0, "Alpha y": 0.0},
            ),
            "Slicer": (Slicer, {"Slice set": "space_charge"}),
        }
        if command == "SpaceCharge":
            from PASS.para.schema.space_charge import SpaceCharge

            block = self.data.get("Space charge", {})
            configurations = block.get("Configurations", {}) if isinstance(block, dict) else {}
            configuration = next(iter(configurations), "default") if isinstance(configurations, dict) else "default"
            return SpaceCharge(s=position, configuration=configuration).model_dump(by_alias=True)
        if command in monitor_models:
            model, required = monitor_models[command]
            return model(**({"S (m)": position} | required)).model_dump(by_alias=True)
        if command == "SortBunch":
            return {"S (m)": position, "Command": "SortBunch"}
        raise KeyError(f"Unknown GUI command: {command}")

    def select_command(self, command: str) -> None:
        """Show a command preview; insertion requires explicit confirmation."""
        if not self._confirm_form_navigation():
            return
        template = self._command_template(command)
        if not template:
            QMessageBox.warning(self, "Sequence 无效", "请先修正 JSON 中的 Sequence 对象。")
            return
        self._populate_form(f"预览 · {command}", template, pending=True)
        self._pending_command = command
        self.form_hint.setText("这是默认参数预览。修改参数后点击“插入到 Sequence”才会写入项目。")

    def add_command(self, command: str) -> None:
        """Compatibility alias for browsing a command template."""
        self.select_command(command)

    def insert_pending_command(self) -> None:
        if not self._pending_command:
            return
        command = self._pending_command
        template = dict(self._selected_mapping or self._command_template(command))
        sequence = self.data.setdefault("Sequence", {})
        if not isinstance(sequence, dict):
            QMessageBox.warning(self, "Sequence 无效", "请先修正 JSON 中的 Sequence 对象。")
            return
        proposed = self._name_field.text().strip() if self._name_field is not None else ""
        if not proposed:
            QMessageBox.warning(self, "名称无效", "名称不能为空。")
            return
        name = proposed
        if name in sequence:
            QMessageBox.warning(self, "名称重复", f"Sequence 中已存在 {name}。")
            return
        sequence[name] = template
        try:
            self._write_form_values(sequence[name])
        except ValueError as exc:
            sequence.pop(name, None)
            QMessageBox.warning(self, "字段无效", str(exc))
            return
        self._sync_editor()
        self._form_dirty = False
        self._data_dirty = True
        self._refresh_tree()
        self._pending_command = None
        self._select_sequence_item(name)

    def configure_madx_elements(self) -> None:
        """Open the MAD-X element importer with element mode selected."""
        self.configure_madx_import(source_kind="elements")

    def configure_madx_twiss(self) -> None:
        """Open the MAD-X Twiss-point importer with file selection."""
        self.configure_madx_import(source_kind="twiss")

    def configure_madx_import(self, source_kind: str = "file") -> None:
        """Show import settings for the MAD-X Twiss/TFS readers in PASS."""
        if not self._confirm_form_navigation():
            return
        self._clear_form()
        if source_kind == "elements":
            self.form_title.setText("从 MAD-X 文件导入元件")
            self.form_hint.setText("读取 MAD-X 导出的 Twiss/TFS 表，转换为 PASS 元件；可选择合并连续 Drift。")
        elif source_kind == "twiss":
            self.form_title.setText("从 MAD-X 文件导入 Twiss 点")
            self.form_hint.setText("选择 MAD-X 导出的 Twiss/TFS 文件，转换为 Twiss 传输 command。")
        else:
            self.form_title.setText("导入 MAD-X Twiss (TFS)")
            self.form_hint.setText("PASS 读取 MAD-X 导出的 Twiss/TFS 表，不直接解析 .madx 源脚本。")
        self._madx_fields = {}
        self._madx_preview = None
        self._madx_preview_signature = None
        self._madx_fields["source_kind"] = source_kind
        self._add_madx_path_field(
            "Twiss TFS 文件",
            "选择 Twiss/TFS 文件",
            required=True,
            directory=False,
        )
        self._add_madx_path_field("误差 TFS 文件", "可选的 MAD-X error TFS 文件")

        merge = PropertyComboBox()
        merge.setObjectName("choiceField")
        merge.addItems(("是", "否"))
        merge.setCurrentText("是")
        merge.setToolTip("是否将导入结果中的连续 Drift 合并为一个项目。")
        self._madx_fields["merge_drift"] = merge
        self.form_layout.addRow(self._field_label("是否合并连续 Drift", ""), merge)
        errors = QCheckBox("附加场误差")
        errors.setObjectName("booleanField")
        errors.setToolTip("根据误差 TFS 为匹配元件附加场误差。")
        self._madx_fields["field_errors"] = errors
        self.form_layout.addRow(self._field_label("误差选项", ""), errors)

        transfer = PropertyComboBox()
        transfer.setObjectName("choiceField")
        transfer.addItems(ENUM_OPTIONS["Longitudinal transfer"])
        transfer.setToolTip("仅 Twiss 传输模式：纵向传输模型。")
        self._madx_fields["longitudinal_transfer"] = transfer
        self.form_layout.addRow(self._field_label("纵向传输", ""), transfer)
        for key, text, tooltip in (
            ("muz", "Mu z", "仅 Twiss 传输模式：纵向 tune。"),
            ("dqx", "DQx", "仅 Twiss 传输模式：留为 from_file 时使用 TFS 头信息。"),
            ("dqy", "DQy", "仅 Twiss 传输模式：留为 from_file 时使用 TFS 头信息。"),
            ("patterns", "薄元件正则", "仅 Twiss 传输模式：用逗号分隔要保留为薄元件的正则表达式。"),
        ):
            field = QLineEdit("0.0" if key == "muz" else "from_file" if key in ("dqx", "dqy") else "")
            field.setObjectName("valueField")
            field.setToolTip(tooltip)
            self._madx_fields[key] = field
            self.form_layout.addRow(self._field_label(text, ""), field)
        update_circumference = QCheckBox("用 TFS 环周长更新全局配置")
        update_circumference.setObjectName("booleanField")
        update_circumference.setChecked(True)
        update_circumference.setToolTip("导入后将 TFS 的 LENGTH 写入 Circumference (m)。")
        self._madx_fields["update_circumference"] = update_circumference
        self.form_layout.addRow(self._field_label("全局配置", ""), update_circumference)
        self.madx_import_button.setVisible(True)
        self.madx_preview_button.setVisible(True)
        self.form_apply.setEnabled(False)

    def _add_madx_path_field(self, key: str, title: str, required: bool = False, directory: bool = False) -> None:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        path = QLineEdit()
        path.setObjectName("valueField")
        path.setPlaceholderText("必填" if required else "可选")
        path.setToolTip("请选择 MAD-X 导出的 Twiss/TFS 表。" if not directory else "请选择包含 MAD-X Twiss/TFS 输出的文件夹。")
        browse = button("浏览")
        browse.setToolTip("选择文件")
        if directory:
            browse.clicked.connect(lambda: self._browse_madx_directory(path, title))
        else:
            browse.clicked.connect(lambda: self._browse_madx_file(path, title))
        layout.addWidget(path, 1)
        layout.addWidget(browse)
        self._madx_fields[key] = path
        self.form_layout.addRow(self._field_label(key, ""), row)

    @staticmethod
    def _browse_madx_file(field: QLineEdit, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            field.window(), title, "", "MAD-X Twiss/TFS (*.tfs *.TFS *.dat *.DAT *.madx *.MADX);;All files (*)",
        )
        if path:
            field.setText(path)

    @staticmethod
    def _browse_madx_directory(field: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(field.window(), title)
        if path:
            field.setText(path)

    @staticmethod
    def _optional_float(field: QLineEdit, key: str) -> float | str:
        text = field.text().strip()
        if text.casefold() == "from_file":
            return "from_file"
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{key} 必须是数字或 from_file。") from exc

    def _madx_import_signature(self) -> tuple:
        fields = self._madx_fields
        def value(key: str, default: object = "") -> object:
            field = fields.get(key)
            if isinstance(field, QLineEdit):
                return field.text().strip()
            if isinstance(field, QComboBox):
                return field.currentText()
            if isinstance(field, QCheckBox):
                return field.isChecked()
            return default

        return (
            value("source_kind"), value("Twiss TFS 文件"), value("误差 TFS 文件"),
            value("merge_drift"), value("field_errors"), value("longitudinal_transfer"),
            value("muz"), value("dqx"), value("dqy"), value("patterns"),
        )

    def _read_madx_import(self) -> tuple[list, list[str], float] | None:
        source_field = self._madx_fields.get("Twiss TFS 文件")
        source = source_field.text().strip() if isinstance(source_field, QLineEdit) else ""
        if "Twiss TFS 文件" not in self._madx_fields:
            QMessageBox.warning(self, "缺少文件", "请选择 MAD-X 导出的 Twiss/TFS 文件。")
            return None
        if not source:
            QMessageBox.warning(self, "缺少文件", "请选择 MAD-X 导出的 Twiss/TFS 文件。")
            return None
        if not Path(source).is_file():
            QMessageBox.warning(self, "文件不存在", "所选文件不存在；.madx 源脚本需要先生成 Twiss/TFS 表。")
            return None
        if Path(source).suffix.casefold() == ".madx":
            QMessageBox.warning(self, "需要 Twiss/TFS", "PASS 当前读取的是 MAD-X 导出的 Twiss/TFS 表，请先由该 .madx 脚本生成表文件。")
            return None
        error_file = self._madx_fields["误差 TFS 文件"].text().strip()
        try:
            from PASS.para.madx import read_madx_elements, read_madx_twiss

            element_mode = self._madx_fields.get("source_kind") != "twiss"
            merge_drift = self._madx_fields["merge_drift"].currentText() == "是"
            if element_mode:
                items, names, circumference = read_madx_elements(
                    source,
                    error_file=error_file,
                    is_merge_drift=merge_drift,
                    is_field_error=self._madx_fields["field_errors"].isChecked(),
                )
            else:
                patterns = [part.strip() for part in self._madx_fields["patterns"].text().split(",") if part.strip()]
                items, names, circumference = read_madx_twiss(
                    source,
                    error_file=error_file,
                    muz=float(self._madx_fields["muz"].text().strip()),
                    dqx=self._optional_float(self._madx_fields["dqx"], "DQx"),
                    dqy=self._optional_float(self._madx_fields["dqy"], "DQy"),
                    is_field_error=self._madx_fields["field_errors"].isChecked(),
                    insert_patterns=patterns or None,
                    longitudinal_transfer=self._madx_fields["longitudinal_transfer"].currentText(),
                    is_merge_drift=merge_drift,
                )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            QMessageBox.critical(self, "导入失败", str(exc))
            return None
        return items, names, circumference

    @staticmethod
    def _madx_keyword_counts(source: str) -> Counter[str]:
        """Return TFS KEYWORD counts without making import dependent on them."""
        try:
            import tfs

            table = tfs.read(source)
            if "KEYWORD" not in table.columns:
                return Counter()
            return Counter(str(keyword).casefold() for keyword in table["KEYWORD"])
        except Exception:
            # The PASS reader is the authority for whether a file is importable.
            # This supplemental preview detail should never block an import.
            return Counter()

    @staticmethod
    def _format_madx_type_counts(counts: Counter[str], *, twiss_points: bool = False) -> str:
        """Format MAD-X KEYWORD statistics in a stable, operator-friendly order."""
        counts = counts.copy()
        labels = {
            "drift": "Drift",
            "quadrupole": "四极铁",
            "sextupole": "六极铁",
            "octupole": "八极铁",
            "multipole": "多极铁",
            "sbend": "扇形弯铁",
            "rbend": "矩形弯铁",
            "hkicker": "水平校正铁",
            "vkicker": "垂直校正铁",
            "kicker": "Kicker",
            "tkicker": "时间 Kicker",
            "monitor": "监测器",
            "marker": "Marker",
            "solenoid": "螺线管",
            "rfcavity": "RF 腔",
            "elseparator": "静电分离器",
        }
        order = tuple(labels)
        entries = []
        for keyword in order:
            count = counts.pop(keyword, 0)
            if count:
                suffix = "处 Twiss 点" if twiss_points else ""
                entries.append(f"{labels[keyword]}{suffix}：{count}")
        for keyword, count in sorted(counts.items()):
            suffix = "处 Twiss 点" if twiss_points else ""
            entries.append(f"其他（{keyword.upper()}）{suffix}：{count}")
        return "\n".join(entries) or "（无法从 TFS 读取 KEYWORD 列）"

    @staticmethod
    def _command_counts(items: list) -> Counter[str]:
        counts: Counter[str] = Counter()
        for item in items:
            command = getattr(item, "command", None)
            if command:
                counts[str(command)] += 1
            else:
                counts[type(item).__name__] += 1
        return counts

    @staticmethod
    def _format_command_counts(counts: Counter[str]) -> str:
        return "\n".join(
            f"{command}：{count}" for command, count in sorted(counts.items())
        ) or "（无项目）"

    def preview_madx_import(self) -> None:
        """Read MAD-X output and show a non-mutating import summary."""
        signature = self._madx_import_signature()
        if self._madx_preview is None or self._madx_preview_signature != signature:
            preview = self._read_madx_import()
            if preview is None:
                return
            self._madx_preview = preview
            self._madx_preview_signature = signature
        items, names, circumference = self._madx_preview
        source = str(self._madx_fields["Twiss TFS 文件"].text().strip())
        source_kind = "Twiss 光学点" if self._madx_fields.get("source_kind") == "twiss" else "元件"
        merge = self._madx_fields["merge_drift"].currentText() == "是"
        preview_names = ", ".join(str(name) for name in names[:8])
        if len(names) > 8:
            preview_names += f" 等 {len(names)} 项"
        keyword_counts = self._madx_keyword_counts(source)
        if source_kind == "Twiss 光学点":
            source_summary = self._format_madx_type_counts(keyword_counts, twiss_points=True)
            source_heading = "Twiss 点按来源元件（TFS 合并前）："
        else:
            source_summary = self._format_madx_type_counts(keyword_counts)
            source_heading = "来源元件类型（TFS 合并前）："
        command_summary = self._format_command_counts(self._command_counts(items))
        text = (
            f"文件：{Path(source).name}\n"
            f"类型：MAD-X {source_kind}\n"
            f"最终导入项目：{len(items)}\n"
            f"环周长：{circumference:.6g} m\n"
            f"合并连续 Drift：{'是' if merge else '否'}\n"
            f"\n{source_heading}\n{source_summary}\n"
            f"\n最终 PASS command：\n{command_summary}\n"
            f"\n项目示例：{preview_names or '（无项目）'}\n\n"
            "预览不会修改当前配置；点击“导入到 Sequence”后才会写入。"
        )
        QMessageBox.information(self, "MAD-X 导入预览", text)

    def import_madx_twiss(self) -> None:
        signature = self._madx_import_signature()
        if self._madx_preview is not None and self._madx_preview_signature == signature:
            items, names, circumference = self._madx_preview
        else:
            preview = self._read_madx_import()
            if preview is None:
                return
            items, names, circumference = preview
            self._madx_preview = preview
            self._madx_preview_signature = signature
        sequence = self.data.setdefault("Sequence", {})
        if not isinstance(sequence, dict):
            QMessageBox.warning(self, "Sequence 无效", "请先修正 JSON 中的 Sequence 对象。")
            return
        first_name = ""
        for name, item in zip(names, items):
            unique_name = self._unique_sequence_name(str(name), sequence)
            sequence[unique_name] = item.model_dump(by_alias=True)
            first_name = first_name or unique_name
        if self._madx_fields["update_circumference"].isChecked():
            self.data["Circumference (m)"] = circumference
        self._sync_editor()
        self._data_dirty = True
        self._refresh_tree()
        if first_name:
            self._select_sequence_item(first_name)
        QMessageBox.information(self, "导入完成", f"已导入 {len(items)} 个项目，环周长 {circumference:.6g} m。")

    @staticmethod
    def _unique_sequence_name(name: str, sequence: dict) -> str:
        base = name or "imported"
        candidate, suffix = base, 2
        while candidate in sequence:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def configure_space_charge(self) -> None:
        """Create or edit the top-level named space-charge configurations."""
        if not self._confirm_form_navigation():
            return
        if not isinstance(self.data.get("Space charge"), dict):
            from PASS.para.schema.space_charge import SpaceChargeConfig, SpaceChargeResourceConfig

            self.data["Space charge"] = SpaceChargeConfig(
                enabled=True,
                configurations={"default": SpaceChargeResourceConfig()},
            ).model_dump(by_alias=True)
        self._sync_editor()
        self._data_dirty = True
        self._refresh_tree()
        self._select_top_level_item("__root__", "Space charge")

    @staticmethod
    def _default_space_charge_resource() -> dict:
        from PASS.para.schema.space_charge import SpaceChargeResourceConfig

        return SpaceChargeResourceConfig().model_dump(by_alias=True)

    def _populate_space_charge_configuration(self, active_name: str | None = None) -> None:
        """Render the top-level block and one named PIC resource as typed fields."""
        block = self.data.get("Space charge")
        if not isinstance(block, dict):
            return
        self._clear_form()
        self._selected_mapping = block
        self._selected_path = ("__root__", "Space charge")
        self.form_title.setText("空间电荷全局配置")
        self.form_hint.setText(
            "顶层配置负责共享的切片集、PIC 网格和求解器；Sequence 中的计算点按名称引用这里的配置。"
        )

        enabled = QCheckBox("启用空间电荷模块")
        enabled.setObjectName("booleanField")
        enabled.setChecked(block.get("Enabled", False) is True)
        enabled.setToolTip("关闭时所有 SpaceCharge 配置和计算点均被忽略。")
        self._track_field(enabled)
        self._space_charge_enabled_field = enabled
        self.form_layout.addRow(self._field_label("Enabled", False), enabled)

        configurations = block.get("Configurations")
        if not isinstance(configurations, dict):
            configurations = block["Configurations"] = {}
        names = list(configurations)
        if active_name not in configurations:
            active_name = names[0] if names else None
        self._active_space_charge_configuration = active_name

        selector_box = QGroupBox("命名资源配置")
        selector_box.setObjectName("configGroup")
        selector_layout = QVBoxLayout(selector_box)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("当前配置"))
        self._space_charge_selector = PropertyComboBox()
        self._space_charge_selector.setObjectName("choiceField")
        self._space_charge_selector.addItems(names)
        if active_name:
            self._space_charge_selector.setCurrentText(active_name)
        self._space_charge_selector.currentTextChanged.connect(self._select_space_charge_configuration)
        selector_row.addWidget(self._space_charge_selector, 1)
        add = button("添加")
        add.clicked.connect(self.add_space_charge_configuration)
        selector_row.addWidget(add)
        copy_button = button("复制")
        copy_button.clicked.connect(self.copy_space_charge_configuration)
        copy_button.setEnabled(active_name is not None)
        selector_row.addWidget(copy_button)
        delete = button("删除")
        delete.clicked.connect(self.delete_space_charge_configuration)
        delete.setEnabled(active_name is not None)
        selector_row.addWidget(delete)
        selector_layout.addLayout(selector_row)

        if active_name is not None and isinstance(configurations.get(active_name), dict):
            name_row = QHBoxLayout()
            name_row.addWidget(QLabel("配置名称"))
            self._space_charge_name_field = QLineEdit(active_name)
            self._space_charge_name_field.setObjectName("valueField")
            self._space_charge_name_field.setToolTip("SpaceCharge command 的 Configuration 字段引用此名称。")
            self._track_field(self._space_charge_name_field)
            self._space_charge_name_field.textChanged.connect(self._preview_space_charge_configuration_name)
            name_row.addWidget(self._space_charge_name_field, 1)
            selector_layout.addLayout(name_row)

            resource = configurations[active_name]
            resource_form = QFormLayout()
            resource_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            resource_form.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
            resource_form.setHorizontalSpacing(12)
            resource_form.setVerticalSpacing(7)
            for key, default_value in self._default_space_charge_resource().items():
                value = resource.get(key, default_value)
                field = self._make_field(key, value)
                self._space_charge_fields[key] = field
                resource_form.addRow(self._field_label(key, value), field)
            selector_layout.addLayout(resource_form)
        else:
            empty = QLabel("尚无资源配置。点击“添加”创建一个配置后，SpaceCharge 计算点才能引用它。")
            empty.setObjectName("muted")
            empty.setWordWrap(True)
            selector_layout.addWidget(empty)
        self.form_layout.addRow(selector_box)
        self.form_apply.setEnabled(True)

    def _preview_space_charge_configuration_name(self, name: str) -> None:
        """Keep the current-configuration selector in sync while its name is edited."""
        selector = self._space_charge_selector
        if selector is None or selector.currentIndex() < 0:
            return
        selector.blockSignals(True)
        selector.setItemText(selector.currentIndex(), name)
        selector.blockSignals(False)

    def _refresh_after_space_charge_change(self, active_name: str | None) -> None:
        """Synchronize every configuration view after a structural change."""
        self._form_dirty = False
        self._data_dirty = True
        self._sync_editor()
        self._refresh_tree()
        self._populate_space_charge_configuration(active_name)
        self._set_sync_status("空间电荷配置已更新，尚未保存", "warning")

    def _write_space_charge_configuration(self) -> str | None:
        """Validate and write the currently displayed top-level resource."""
        from PASS.para.schema.space_charge import SpaceChargeConfig, SpaceChargeResourceConfig

        block = self.data.get("Space charge")
        if not isinstance(block, dict) or self._space_charge_enabled_field is None:
            raise ValueError("Space charge 顶层配置无效。")
        configurations = block.get("Configurations")
        if not isinstance(configurations, dict):
            raise ValueError("Space charge.Configurations 必须是对象。")
        old_name = self._active_space_charge_configuration
        new_name = old_name
        candidate_configurations = deepcopy(configurations)
        if old_name is not None:
            if self._space_charge_name_field is None:
                raise ValueError("空间电荷配置名称控件缺失。")
            new_name = self._space_charge_name_field.text().strip()
            if not new_name:
                raise ValueError("空间电荷配置名称不能为空。")
            if new_name != old_name and new_name in candidate_configurations:
                raise ValueError(f"空间电荷配置名称 {new_name!r} 已存在。")
            original = configurations.get(old_name)
            if not isinstance(original, dict):
                raise ValueError(f"Space charge.Configurations.{old_name} 必须是对象。")
            resource = {
                key: self._read_field_value(key, field, original.get(key))
                for key, field in self._space_charge_fields.items()
            }
            canonical_resource = SpaceChargeResourceConfig.model_validate(resource).model_dump(by_alias=True)
            candidate_configurations.pop(old_name, None)
            candidate_configurations[new_name] = canonical_resource
        enabled = self._space_charge_enabled_field.isChecked()
        candidate = {"Enabled": enabled, "Configurations": candidate_configurations}
        if enabled:
            SpaceChargeConfig.model_validate(candidate)
        block.clear()
        block.update(candidate)
        if old_name is not None and new_name != old_name:
            sequence = self.data.get("Sequence")
            if isinstance(sequence, dict):
                for item in sequence.values():
                    if (
                        isinstance(item, dict)
                        and item.get("Command") == "SpaceCharge"
                        and item.get("Configuration") == old_name
                    ):
                        item["Configuration"] = new_name
        self._active_space_charge_configuration = new_name
        return new_name

    def _select_space_charge_configuration(self, name: str) -> None:
        if not name or name == self._active_space_charge_configuration:
            return
        try:
            self._write_space_charge_configuration()
        except ValueError as exc:
            QMessageBox.warning(self, "空间电荷配置无效", str(exc))
            if self._space_charge_selector:
                self._space_charge_selector.blockSignals(True)
                self._space_charge_selector.setCurrentText(self._active_space_charge_configuration or "")
                self._space_charge_selector.blockSignals(False)
            return
        self._refresh_after_space_charge_change(name)

    def add_space_charge_configuration(self) -> None:
        block = self.data.get("Space charge")
        configurations = block.get("Configurations") if isinstance(block, dict) else None
        if not isinstance(configurations, dict):
            return
        if self._active_space_charge_configuration is not None:
            try:
                self._write_space_charge_configuration()
            except ValueError as exc:
                QMessageBox.warning(self, "空间电荷配置无效", str(exc))
                return
            block = self.data.get("Space charge")
            configurations = block.get("Configurations") if isinstance(block, dict) else None
            if not isinstance(configurations, dict):
                return
        index = 1
        name = "configuration_1"
        while name in configurations:
            index += 1
            name = f"configuration_{index}"
        configurations[name] = self._default_space_charge_resource()
        self._refresh_after_space_charge_change(name)

    def copy_space_charge_configuration(self) -> None:
        block = self.data.get("Space charge")
        configurations = block.get("Configurations") if isinstance(block, dict) else None
        source_name = self._active_space_charge_configuration
        if not isinstance(configurations, dict) or source_name not in configurations:
            return
        try:
            source_name = self._write_space_charge_configuration()
        except ValueError as exc:
            QMessageBox.warning(self, "空间电荷配置无效", str(exc))
            return
        block = self.data.get("Space charge")
        configurations = block.get("Configurations") if isinstance(block, dict) else None
        if not isinstance(configurations, dict) or source_name not in configurations:
            return
        base = f"{source_name}_copy"
        name, suffix = base, 2
        while name in configurations:
            name = f"{base}{suffix}"
            suffix += 1
        configurations[name] = deepcopy(configurations[source_name])
        self._refresh_after_space_charge_change(name)

    def delete_space_charge_configuration(self) -> None:
        block = self.data.get("Space charge")
        configurations = block.get("Configurations") if isinstance(block, dict) else None
        name = self._active_space_charge_configuration
        if not isinstance(configurations, dict) or name not in configurations:
            return
        sequence = self.data.get("Sequence")
        references = [
            str(command_name)
            for command_name, item in sequence.items()
            if (
                isinstance(sequence, dict)
                and isinstance(item, dict)
                and item.get("Command") == "SpaceCharge"
                and item.get("Configuration") == name
            )
        ] if isinstance(sequence, dict) else []
        replacement = None
        alternatives = [configuration_name for configuration_name in configurations if configuration_name != name]
        if references:
            reference_text = "、".join(references[:8])
            if len(references) > 8:
                reference_text += f" 等 {len(references)} 项"
            if not alternatives:
                QMessageBox.warning(
                    self,
                    "无法删除空间电荷配置",
                    f"配置 {name} 正被计算点 {reference_text} 引用，且没有其他配置可供替换。\n"
                    "请先新增一个全局配置，或删除这些空间电荷计算点。",
                )
                return
            replacement, accepted = QInputDialog.getItem(
                self,
                "重新分配空间电荷计算点",
                f"配置 {name} 正被以下计算点引用：\n{reference_text}\n\n删除前请选择替代配置：",
                alternatives,
                0,
                False,
            )
            if not accepted:
                return
            if QMessageBox.question(
                self,
                "确认替换并删除",
                f"将 {len(references)} 个计算点改为引用 {replacement}，然后删除 {name}。是否继续？",
            ) != QMessageBox.Yes:
                return
        elif QMessageBox.question(self, "删除空间电荷配置", f"确定删除配置 {name}？") != QMessageBox.Yes:
            return
        if references and isinstance(sequence, dict):
            for command_name in references:
                sequence[command_name]["Configuration"] = replacement
        configurations.pop(name)
        next_name = replacement if replacement is not None else alternatives[0] if alternatives else None
        self._refresh_after_space_charge_change(next_name)

    def configure_global(self) -> None:
        """Open the complete root input schema from the left-side library."""
        if not self._confirm_form_navigation():
            return
        self.editor_tabs.setCurrentIndex(0)
        self._populate_root_configuration()

    def configure_timing(self) -> None:
        """Open the structured Timing editor from the project navigation."""
        if not self._confirm_form_navigation():
            return
        self.editor_tabs.setCurrentIndex(0)
        self._populate_timing_configuration()

    def _refresh_tree(self) -> None:
        self.tree.clear()
        root_values = [(key, value) for key, value in self.data.items() if key != "Sequence"]
        if root_values:
            root = QTreeWidgetItem(["全局配置"])
            root.setData(0, Qt.UserRole, ("__root__", None))
            root.setToolTip(0, "PASS 输入 JSON 的根级参数，包括束流身份、环参数、Timing 和物理开关。")
            self.tree.addTopLevelItem(root)
            for key, value in root_values:
                child = QTreeWidgetItem([str(key)])
                child.setData(0, Qt.UserRole, ("__root__", key))
                child.setToolTip(0, json.dumps(value, ensure_ascii=False)[:500])
                root.addChild(child)
        sequence = self.data.get("Sequence")
        sequence_count = len(sequence) if isinstance(sequence, dict) else 0
        sequence_node = QTreeWidgetItem([f"Sequence · 执行序列（{sequence_count} 项）"])
        sequence_node.setData(0, Qt.UserRole, ("Sequence", None))
        sequence_node.setToolTip(0, "按 S (m) 排序执行的 command 集合；浏览不会自动插入。")
        self.tree.addTopLevelItem(sequence_node)
        if isinstance(sequence, dict):
            for name, value in sequence.items():
                if not isinstance(value, dict):
                    continue
                command = str(value.get("Command", ""))
                child = QTreeWidgetItem([f"{name}  ·  {command}"])
                child.setData(0, Qt.UserRole, ("Sequence", str(name)))
                child.setToolTip(0, json.dumps(value, ensure_ascii=False)[:500])
                sequence_node.addChild(child)
        self.tree.expandAll()
        self._refresh_sequence_table()
        self._validate_configuration()

    def _validate_configuration(self) -> list[str]:
        issues: list[str] = []
        sequence = self.data.get("Sequence")
        injection = sequence.get("injection") if isinstance(sequence, dict) else None
        if not isinstance(sequence, dict):
            issues.append("Sequence 必须是对象")
        if not isinstance(injection, dict) or injection.get("Command") != "Injection":
            issues.append("未找到 Injection command，请在 Sequence 中添加 Injection")
        try:
            from PASS.para.schema.main import MainConfig

            MainConfig.model_validate(
                {key: value for key, value in self.data.items() if key not in {"Sequence", "Space charge"}}
            )
        except Exception as exc:
            issues.append(f"全局配置: {self._validation_detail(exc)}")
        space_charge = None
        try:
            from PASS.core.config import Config

            space_charge, _ = Config._load_space_charge(self.data)
        except Exception as exc:
            issues.append(f"Space charge: {self._validation_detail(exc)}")
        if isinstance(sequence, dict):
            for name, item in sequence.items():
                issue = self._validate_sequence_item(str(name), item)
                if issue:
                    issues.append(issue)
            if space_charge is not None and space_charge.enabled:
                configured = set(space_charge.configurations)
                for name, item in sequence.items():
                    if not isinstance(item, dict) or item.get("Command") != "SpaceCharge":
                        continue
                    reference = item.get("Configuration")
                    if reference not in configured:
                        issues.append(
                            f"Sequence.{name}: 未定义 Space charge configuration {reference!r}"
                        )
        if issues:
            self._validation_issues = issues
            detail = "配置检查未通过：\n" + "\n".join(f"• {issue}" for issue in issues)
            self._set_validation_status(f"配置检查：{len(issues)} 项问题", "error", detail)
        else:
            self._validation_issues = []
            self._set_validation_status("配置有效", "ok", "配置检查通过：全局配置和 Sequence 均有效。")

        return issues

    def _set_validation_status(self, text: str, state: str, detail: str) -> None:
        """Show a compact validation result without consuming overview space."""
        self.validation_label.setText(text)
        self.validation_label.setToolTip(detail)
        self.validation_label.setProperty("state", state)
        self.validation_label.style().unpolish(self.validation_label)
        self.validation_label.style().polish(self.validation_label)

    def _show_validation_issues(self) -> None:
        """Present validation findings without consuming permanent layout space."""
        if not self._validation_issues:
            QMessageBox.information(self, "配置检查", "配置检查通过。")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("配置检查")
        dialog.setMinimumSize(520, 280)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("双击问题可定位到相关配置。"))
        issues = QListWidget()
        issues.addItems(self._validation_issues)
        issues.itemDoubleClicked.connect(lambda item: self._navigate_to_validation_issue(item.text(), dialog))
        layout.addWidget(issues, 1)
        close = button("关闭")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close, alignment=Qt.AlignRight)
        dialog.exec()

    def _navigate_to_validation_issue(self, issue: str, dialog: QDialog) -> None:
        """Open the input area most likely to resolve a validation issue."""
        if not self._confirm_form_navigation():
            return
        if issue.startswith("未找到 Injection command"):
            self.select_command("Injection")
        elif issue.startswith("Sequence."):
            name = issue.removeprefix("Sequence.").split(":", 1)[0]
            self._select_sequence_item(name)
        else:
            self.configure_global()
        dialog.accept()

    @staticmethod
    def _validation_detail(exc: Exception) -> str:
        """Reduce a Pydantic error to the most useful first field message."""
        errors = getattr(exc, "errors", None)
        if callable(errors):
            details = errors()
            if details:
                detail = details[0]
                location = ".".join(str(part) for part in detail.get("loc", ()))
                message = str(detail.get("msg", "校验失败"))
                return f"{location}: {message}" if location else message
        return str(exc).splitlines()[0] if str(exc) else "校验失败"

    @classmethod
    def _validate_sequence_item(cls, name: str, item: object) -> str | None:
        if not isinstance(item, dict):
            return f"Sequence.{name}: command 必须是对象"
        command = item.get("Command")
        if not isinstance(command, str) or not command:
            return f"Sequence.{name}: 缺少 Command"
        try:
            from PASS.para.schema.bunch import BunchConfig, InjectionItem
            from PASS.para.schema.elements import ELEMENT_REGISTRY
            from PASS.para.schema.monitors import DistMonitor, ParticleMonitor, PhaseAdvanceMonitor, StatMonitor
            from PASS.para.schema.slicer import Slicer
            from PASS.para.schema.space_charge import SpaceCharge
            from PASS.para.schema.twiss import TwissPoint

            if command == "Injection":
                bunches = []
                for key in sorted(
                    (str(key) for key, value in item.items()
                     if re.fullmatch(r"bunch\d+", str(key)) and isinstance(value, dict)),
                    key=lambda key: int(key[5:]),
                ):
                    bunches.append(BunchConfig.model_validate(item[key]))
                injection_data = {key: value for key, value in item.items() if not re.fullmatch(r"bunch\d+", str(key))}
                injection_data["bunches"] = bunches
                InjectionItem.model_validate(injection_data).to_sequence_dict()
                return None
            if command == "Twiss":
                TwissPoint.model_validate(item)
                return None
            if command == "SortBunch":
                float(item["S (m)"])
                return None
            models = {
                "StatMonitor": StatMonitor,
                "DistMonitor": DistMonitor,
                "ParticleMonitor": ParticleMonitor,
                "PhaseAdvanceMonitor": PhaseAdvanceMonitor,
                "Slicer": Slicer,
                "SpaceCharge": SpaceCharge,
            }
            model = models.get(command) or ELEMENT_REGISTRY.get(command.casefold())
            if model is None:
                return f"Sequence.{name}: 未知 Command {command}"
            model.model_validate(item)
        except (KeyError, TypeError, ValueError) as exc:
            return f"Sequence.{name}: {cls._validation_detail(exc)}"
        return None

    def _refresh_sequence_table(self) -> None:
        sequence = self.data.get("Sequence", {})
        rows = []
        if isinstance(sequence, dict):
            for name, value in sequence.items():
                if not isinstance(value, dict):
                    continue
                command = str(value.get("Command", ""))
                try:
                    position = float(value.get("S (m)", 0.0))
                except (TypeError, ValueError):
                    position = 0.0
                rows.append((position, str(name), command, value))
        rows.sort(key=lambda row: (row[0], row[1]))
        selected_command = self.sequence_command_filter.currentText()
        commands = sorted({command for _, _, command, _ in rows if command})
        self.sequence_command_filter.blockSignals(True)
        self.sequence_command_filter.clear()
        self.sequence_command_filter.addItem("全部 Command")
        self.sequence_command_filter.addItems(commands)
        self.sequence_command_filter.setCurrentText(
            selected_command if selected_command in commands else "全部 Command"
        )
        self.sequence_command_filter.blockSignals(False)
        self.sequence_table.setRowCount(len(rows))
        for row_index, (position, name, command, value) in enumerate(rows):
            self.sequence_table.setItem(row_index, 0, QTableWidgetItem(name))
            self.sequence_table.setItem(row_index, 1, QTableWidgetItem(command))
            self.sequence_table.setItem(row_index, 2, QTableWidgetItem(f"{position:.6g}"))
            status = "禁用" if value.get("Is enable", True) is False else "启用"
            self.sequence_table.setItem(row_index, 3, QTableWidgetItem(status))
            self.sequence_table.item(row_index, 0).setData(Qt.UserRole, name)
        self._filter_sequence_table()

    def _filter_sequence_table(self, _value: str = "") -> None:
        """Hide rows that do not match the command name or type."""
        needle = self.sequence_filter.text().strip().casefold()
        selected_command = self.sequence_command_filter.currentText()
        for row in range(self.sequence_table.rowCount()):
            name_item = self.sequence_table.item(row, 0)
            command_item = self.sequence_table.item(row, 1)
            name = name_item.text() if name_item else ""
            command = command_item.text() if command_item else ""
            text_matches = not needle or needle in name.casefold() or needle in command.casefold()
            type_matches = selected_command == "全部 Command" or command == selected_command
            visible = text_matches and type_matches
            self.sequence_table.setRowHidden(row, not visible)

    def _sequence_row_clicked(self, row: int, column: int) -> None:
        item = self.sequence_table.item(row, 0)
        if item is not None:
            self.editor_tabs.setCurrentIndex(0)
            self._select_sequence_item(str(item.data(Qt.UserRole)))

    def _tree_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        if not self._confirm_form_navigation():
            return
        root_key, child_key = item.data(0, Qt.UserRole)
        if root_key == "__root__":
            if child_key is None:
                self._populate_root_configuration()
            else:
                self._populate_root_field(str(child_key))
            return
        if root_key == "Sequence" and child_key is None:
            self._clear_form("Sequence 包含所有按位置执行的 command；请在下方表格或树中选择具体项目。")
            self.form_title.setText("Sequence")
            self.editor_tabs.setCurrentIndex(0)
            return
        target = self.data.get(root_key)
        title = str(root_key)
        if child_key is not None and isinstance(target, dict):
            target = target.get(child_key)
            title = f"{root_key} · {child_key}"
        if isinstance(target, dict):
            self._populate_form(title, target, name_value=child_key if root_key == "Sequence" else None)
            self._selected_path = (root_key, child_key)
            self._pending_command = None
            self._update_action_visibility()
        else:
            self._clear_form("所选条目不是可编辑的对象。")

    def _populate_root_field(self, key: str) -> None:
        if key not in self.data:
            return
        if key == "Space charge" and isinstance(self.data[key], dict):
            self._populate_space_charge_configuration()
            return
        if key == "Timing" and isinstance(self.data[key], dict):
            self._populate_timing_configuration()
            return
        if isinstance(self.data[key], dict):
            self._populate_form(f"全局配置 · {key}", self.data[key])
            self._selected_path = ("__root__", key)
            self._update_action_visibility()
            return
        self._clear_form()
        self._selected_mapping = self.data
        self._selected_path = ("__root__", key)
        self.form_title.setText(f"全局配置 · {key}")
        self.form_hint.setText("该值位于 JSON 根对象。")
        field = self._make_field(key, self.data[key])
        self.form_layout.addRow(self._field_label(key, self.data[key]), field)
        self._form_fields[key] = field
        self.form_apply.setEnabled(True)

    def _populate_root_configuration(self) -> None:
        """Present the complete root schema in one form for new projects."""
        self._clear_form()
        self._selected_mapping = self.data
        self._selected_path = ("__root__", None)
        self.form_title.setText("全局配置")
        self.form_hint.setText("这些字段会写入输入 JSON 的根对象；Sequence 在下方单独管理。")
        sections = (
            ("束流信息", ("Beam Name", "Number of Protons", "Number of Neutrons", "Number of Charges")),
            ("环参数", ("Transition Gamma", "Circumference (m)")),
            ("模拟控制", ("Number of turns", "Particle Precision")),
            ("计算后端", ("Backend (gpu/cpu)", "Number of GPU devices", "Device Id")),
            ("输出与文件", ("Output directory", "Is plot figure")),
            ("物理模型开关", ("Is beam-beam",)),
        )
        shown = set()
        for section_name, keys in sections:
            box = QGroupBox(section_name)
            box.setObjectName("configGroup")
            section_layout = QFormLayout(box)
            section_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            section_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
            section_layout.setHorizontalSpacing(12)
            section_layout.setVerticalSpacing(7)
            has_fields = False
            for key in keys:
                if key not in self.data:
                    continue
                value = self.data[key]
                field = self._make_field(key, value)
                section_layout.addRow(self._field_label(key, value), field)
                self._form_fields[key] = field
                shown.add(key)
                has_fields = True
            if has_fields:
                self.form_layout.addRow(box)
        for key, value in self.data.items():
            if key in {"Sequence", "Timing", "Space charge"} or key in shown:
                continue
            field = self._make_field(key, value)
            self.form_layout.addRow(self._field_label(key, value), field)
            self._form_fields[key] = field
        timing = self.data.get("Timing")
        if isinstance(timing, dict):
            self._add_timing_section(timing)
        self.form_apply.setEnabled(bool(self._form_fields))

    def _populate_timing_configuration(self) -> None:
        """Edit Timing as typed settings instead of a JSON text blob."""
        timing = self.data.get("Timing")
        if not isinstance(timing, dict):
            return
        self._clear_form()
        self._selected_mapping = self.data
        self._selected_path = ("__root__", "Timing")
        self.form_title.setText("全局配置 · Timing")
        self.form_hint.setText("控制进度日志和 ETA 的输出节奏。")
        self._add_timing_section(timing)
        self.form_apply.setEnabled(bool(self._timing_fields))

    def _add_timing_section(self, timing: dict) -> None:
        timing_box = QGroupBox("Timing")
        timing_box.setObjectName("timingBox")
        layout = QFormLayout(timing_box)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(7)
        for key, value in timing.items():
            field = self._make_timing_field(str(key), value)
            self._timing_fields[str(key)] = field
            layout.addRow(self._field_label(str(key), value), field)
        self.form_layout.addRow(self._field_label("Timing", timing), timing_box)

    def _make_timing_field(self, key: str, value: object) -> QWidget:
        if key == "Mode":
            field = PropertyComboBox()
            field.setObjectName("choiceField")
            field.addItems(TIMING_MODE_OPTIONS)
            field.setCurrentText(str(value))
            field.setToolTip("进度记录模式：关闭、每 turn、每 command 或同步 command。")
            return field
        return self._make_field(key, value)

    def _make_space_charge_reference_field(self, value: str) -> PropertyComboBox:
        """Create a command reference selector from top-level configuration names."""
        field = PropertyComboBox()
        field.setObjectName("choiceField")
        block = self.data.get("Space charge")
        configurations = block.get("Configurations") if isinstance(block, dict) else None
        names = list(configurations) if isinstance(configurations, dict) else []
        if value and value not in names:
            names.insert(0, value)
        field.addItems(names)
        if value:
            field.setCurrentText(value)
        field.setMinimumWidth(220)
        field.setToolTip("选择顶层 Space charge.Configurations 中的命名配置。")
        self._track_field(field)
        return field

    def _populate_form(self, title: str, target: dict, pending: bool = False, name_value: str | None = None) -> None:
        if target.get("Command") == "Injection":
            self._populate_injection_form(title, target, pending, name_value)
            return
        self._clear_form()
        self._selected_mapping = target
        self.form_title.setText(title)
        self.form_hint.setText("字段会写回当前配置。列表和嵌套对象使用 JSON 编辑器；Command 为只读。")
        if pending or name_value is not None:
            default_name = "injection" if target.get("Command") == "Injection" else f"{str(target.get('Command', 'command')).lower()}_1"
            self._name_field = QLineEdit(name_value or default_name)
            self._name_field.setToolTip("Sequence 中的唯一名称")
            self.form_layout.addRow(QLabel("名称"), self._name_field)
        for key, value in target.items():
            if target.get("Command") == "SpaceCharge" and key == "Configuration":
                field = self._make_space_charge_reference_field(str(value))
            else:
                field = self._make_field(str(key), value)
            self.form_layout.addRow(self._field_label(str(key), value), field)
            self._form_fields[key] = field
        self.form_apply.setEnabled(bool(self._form_fields) or self._name_field is not None)
        self.insert_button.setVisible(pending)
        self._update_action_visibility(pending=pending)

    def _populate_injection_form(
        self, title: str, target: dict, pending: bool = False, name_value: str | None = None,
    ) -> None:
        """Render Injection and its bunches as structured controls, not raw JSON."""
        requested_bunch = self._active_bunch_key
        self._clear_form()
        self._selected_mapping = target
        self._injection_pending = pending
        self.form_title.setText(title)
        self.form_hint.setText("bunch 使用结构化字段编辑；添加、复制或删除时分组编号会自动更新。")
        if pending or name_value is not None:
            self._name_field = QLineEdit(name_value or "injection")
            self._name_field.setObjectName("valueField")
            self._name_field.setToolTip("Sequence 中的唯一名称")
            self.form_layout.addRow(self._field_label("名称", ""), self._name_field)
        for key, value in target.items():
            if re.fullmatch(r"bunch\d+", str(key)):
                continue
            if key == "Harmonic Number":
                field = QLineEdit(str(len(self._injection_keys(target))))
                field.setObjectName("readonlyField")
                field.setReadOnly(True)
                field.setToolTip(FIELD_HELP[key])
            else:
                field = self._make_field(str(key), value)
                self._form_fields[key] = field
            self.form_layout.addRow(self._field_label(str(key), value), field)

        bunch_box = QGroupBox("Bunch 配置")
        bunch_box.setObjectName("bunchBox")
        bunch_layout = QVBoxLayout(bunch_box)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("当前 bunch"))
        self._bunch_selector = PropertyComboBox()
        self._bunch_selector.setObjectName("choiceField")
        keys = self._injection_keys(target)
        self._bunch_selector.addItems(keys)
        if requested_bunch not in keys:
            requested_bunch = keys[0] if keys else None
        self._active_bunch_key = requested_bunch
        if self._active_bunch_key:
            self._bunch_selector.setCurrentText(self._active_bunch_key)
        self._bunch_selector.currentTextChanged.connect(self._select_bunch)
        selector_row.addWidget(self._bunch_selector, 1)
        add = button("添加")
        add.setToolTip("添加一个默认 bunch，并使 Harmonic Number 与 bunch 数保持一致。")
        add.clicked.connect(self.add_bunch)
        selector_row.addWidget(add)
        copy_button = button("复制")
        copy_button.setToolTip("复制当前 bunch 的参数。")
        copy_button.clicked.connect(self.copy_bunch)
        selector_row.addWidget(copy_button)
        delete = button("删除")
        delete.setToolTip("删除当前 bunch；至少保留一个 bunch。")
        delete.clicked.connect(self.delete_bunch)
        delete.setEnabled(len(keys) > 1)
        selector_row.addWidget(delete)
        bunch_layout.addLayout(selector_row)
        if self._active_bunch_key and isinstance(target.get(self._active_bunch_key), dict):
            fields = QFormLayout()
            fields.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            fields.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
            fields.setHorizontalSpacing(12)
            fields.setVerticalSpacing(7)
            bunch = target[self._active_bunch_key]
            for key, value in bunch.items():
                if key in ("Offset x", "Offset y") and isinstance(value, dict):
                    offset_box = QGroupBox(str(key))
                    offset_layout = QFormLayout(offset_box)
                    offset_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
                    offset_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
                    for child_key, child_value in value.items():
                        field = self._make_field(str(child_key), child_value)
                        self._bunch_fields[(str(key), str(child_key))] = field
                        offset_layout.addRow(self._field_label(str(child_key), child_value), field)
                    fields.addRow(offset_box)
                    continue
                field = self._make_field(str(key), value)
                self._bunch_fields[(str(key), None)] = field
                fields.addRow(self._field_label(str(key), value), field)
            bunch_layout.addLayout(fields)
        self.form_layout.addRow(bunch_box)
        self.form_apply.setEnabled(True)
        self.insert_button.setVisible(pending)
        self._update_action_visibility(pending=pending)

    @staticmethod
    def _injection_keys(target: dict) -> list[str]:
        return sorted(
            (str(key) for key, value in target.items() if re.fullmatch(r"bunch\d+", str(key)) and isinstance(value, dict)),
            key=lambda key: int(key[5:]),
        )

    def _normalize_injection(self, target: dict) -> None:
        keys = self._injection_keys(target)
        if not keys:
            self._add_default_bunch(target)
            keys = self._injection_keys(target)
        bunches = [target[key] for key in keys]
        for key in keys:
            target.pop(key, None)
        for index, bunch in enumerate(bunches):
            bunch["Harmonic ID of this bunch"] = index
            target[f"bunch{index}"] = bunch
        target["Harmonic Number"] = len(bunches)

    @staticmethod
    def _add_default_bunch(target: dict) -> None:
        from PASS.para.schema.bunch import BunchConfig

        index = len(ConfigPage._injection_keys(target))
        target[f"bunch{index}"] = BunchConfig(
            kinetic_energy=33.2e6,
            num_real_particles=int(1e11),
            num_macro_particles=int(1e5),
            harmonic_id=index,
        ).model_dump(by_alias=True)

    def _rebuild_injection_form(self, active_key: str | None = None) -> None:
        target = self._selected_mapping
        if not isinstance(target, dict):
            return
        title = self.form_title.text()
        name_value = self._name_field.text() if self._name_field is not None else None
        selected_path = self._selected_path
        pending_command = self._pending_command
        self._active_bunch_key = active_key
        self._populate_injection_form(title, target, self._injection_pending, name_value)
        self._selected_path = selected_path
        self._pending_command = pending_command
        self._update_action_visibility(pending=self._injection_pending)

    def _select_bunch(self, key: str) -> None:
        if not key or key == self._active_bunch_key:
            return
        try:
            self._write_bunch_values(self._selected_mapping)
        except ValueError as exc:
            QMessageBox.warning(self, "字段无效", str(exc))
            if self._bunch_selector:
                self._bunch_selector.setCurrentText(self._active_bunch_key or "")
            return
        self._rebuild_injection_form(key)

    def add_bunch(self) -> None:
        target = self._selected_mapping
        if not isinstance(target, dict):
            return
        self._write_bunch_values(target)
        self._normalize_injection(target)
        self._add_default_bunch(target)
        self._normalize_injection(target)
        self._data_dirty = True
        self._rebuild_injection_form(f"bunch{len(self._injection_keys(target)) - 1}")

    def copy_bunch(self) -> None:
        target = self._selected_mapping
        key = self._active_bunch_key
        if not isinstance(target, dict) or not key or not isinstance(target.get(key), dict):
            return
        self._write_bunch_values(target)
        self._normalize_injection(target)
        new_key = f"bunch{len(self._injection_keys(target))}"
        target[new_key] = deepcopy(target[key])
        self._normalize_injection(target)
        self._data_dirty = True
        self._rebuild_injection_form(new_key)

    def delete_bunch(self) -> None:
        target = self._selected_mapping
        key = self._active_bunch_key
        if not isinstance(target, dict) or not key or len(self._injection_keys(target)) <= 1:
            return
        self._write_bunch_values(target)
        target.pop(key, None)
        self._normalize_injection(target)
        self._data_dirty = True
        self._rebuild_injection_form("bunch0")

    def _field_label(self, key: str, value: object) -> QLabel:
        label = QLabel(key)
        label.setMinimumWidth(168)
        label.setMaximumWidth(210)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        help_text = self._field_help(key, value)
        label.setToolTip(help_text)
        return label

    @classmethod
    def _field_help(cls, key: str, value: object) -> str:
        if key in FIELD_HELP:
            return FIELD_HELP[key]
        schema_help = cls._schema_help().get(key, "")
        if schema_help:
            return schema_help
        if isinstance(value, bool):
            return "布尔开关：勾选为启用，取消勾选为关闭。"
        if isinstance(value, (list, dict)):
            return "复合参数：以 JSON 数组或对象编辑，应用时检查格式。"
        if key in ENUM_OPTIONS:
            return "受限选项：请从下拉列表中选择。"
        return f"输入参数，当前值类型为 {type(value).__name__}。"

    @staticmethod
    def _schema_help() -> dict[str, str]:
        """Collect field descriptions from PASS schemas for hover help."""
        global _SCHEMA_HELP
        if _SCHEMA_HELP is not None:
            return _SCHEMA_HELP
        result: dict[str, str] = {}
        try:
            from PASS.para.schema.bunch import BunchConfig, InjectionItem, OffsetConfig
            from PASS.para.schema.elements import ELEMENT_REGISTRY
            from PASS.para.schema.main import MainConfig, TimingConfig
            from PASS.para.schema.monitors import DistMonitor, ParticleMonitor, PhaseAdvanceMonitor, StatMonitor
            from PASS.para.schema.space_charge import (
                SpaceCharge,
                SpaceChargeConfig,
                SpaceChargeResourceConfig,
            )
            from PASS.para.schema.slicer import Slicer
            from PASS.para.schema.twiss import TwissPoint

            models = [
                MainConfig, TimingConfig, BunchConfig, InjectionItem, OffsetConfig,
                SpaceChargeConfig, SpaceChargeResourceConfig, SpaceCharge, Slicer, TwissPoint,
                StatMonitor, DistMonitor, ParticleMonitor, PhaseAdvanceMonitor,
                *ELEMENT_REGISTRY.values(),
            ]
            for model in models:
                for field in model.model_fields.values():
                    alias = field.alias or field.validation_alias
                    description = field.description
                    if alias and description:
                        result[str(alias)] = description
        except (ImportError, AttributeError):
            # The form remains usable with the local fallback help map when a
            # lightweight installation omits an optional schema module.
            pass
        _SCHEMA_HELP = result
        return result

    def _make_field(self, key: str, value: object) -> QWidget:
        if key == "Command":
            field = QLineEdit(str(value))
            field.setObjectName("readonlyField")
            field.setReadOnly(True)
            field.setMinimumWidth(220)
            field.setToolTip("Command 类型由组件库确定，不能在此修改。")
            self._track_field(field)
            return field
        if isinstance(value, bool):
            field = QCheckBox()
            field.setObjectName("booleanField")
            field.setChecked(value)
            field.setToolTip(self._field_help(key, value))
            self._track_field(field)
            return field
        if key in ENUM_OPTIONS:
            field = PropertyComboBox()
            field.setObjectName("choiceField")
            options = list(ENUM_OPTIONS[key])
            value_text = "" if value is None else str(value)
            if value_text not in options:
                options.insert(0, value_text)
            field.addItems(options)
            field.setCurrentText(value_text)
            field.setMinimumWidth(220)
            field.setToolTip(self._field_help(key, value))
            self._track_field(field)
            return field
        if isinstance(value, (list, dict)) or key == "Aperture":
            field = QPlainTextEdit(json.dumps(value, indent=2, ensure_ascii=False))
            field.setObjectName("jsonField")
            field.setFont(QFont("Cascadia Code", 10))
            field.setMinimumHeight(82)
            field.setMinimumWidth(220)
            field.setToolTip(self._field_help(key, value))
            self._track_field(field)
            return field
        field = QLineEdit("" if value is None else str(value))
        field.setObjectName("valueField")
        field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        field.setMinimumWidth(220)
        if isinstance(value, int) and not isinstance(value, bool):
            field.setValidator(QIntValidator(field))
        elif isinstance(value, float):
            field.setValidator(QDoubleValidator(field))
        if value is None:
            field.setPlaceholderText("null")
        if "file" in key.casefold() or "path" in key.casefold():
            field.setToolTip(FIELD_HELP.get(key, "文件路径，可直接输入绝对路径或相对输入 JSON 的路径。"))
        else:
            field.setToolTip(self._field_help(key, value))
        self._track_field(field)
        return field

    def _track_field(self, field: QWidget) -> None:
        """Mark form edits without changing data until the user applies them."""
        if isinstance(field, QLineEdit):
            field.textChanged.connect(lambda: self._mark_form_dirty())
        elif isinstance(field, QComboBox):
            field.currentTextChanged.connect(lambda: self._mark_form_dirty())
        elif isinstance(field, QCheckBox):
            field.stateChanged.connect(lambda: self._mark_form_dirty())
        elif isinstance(field, QPlainTextEdit):
            field.textChanged.connect(self._mark_form_dirty)

    def _mark_form_dirty(self) -> None:
        if self._selected_mapping is None:
            return
        self._form_dirty = True
        self._set_sync_status("表单有未确认修改", "warning")

    @staticmethod
    def _read_field_value(key: str, field: QWidget, old_value: object) -> object:
        if isinstance(field, QCheckBox):
            return field.isChecked()
        if isinstance(field, QComboBox):
            return field.currentText()
        if isinstance(field, QPlainTextEdit):
            try:
                value = json.loads(field.toPlainText())
            except json.JSONDecodeError as exc:
                raise ValueError(f"{key} 的 JSON 无效：{exc.msg}") from exc
            if isinstance(old_value, list) and not isinstance(value, list):
                raise ValueError(f"{key} 必须是 JSON 数组。")
            if isinstance(old_value, dict) and not isinstance(value, dict):
                raise ValueError(f"{key} 必须是 JSON 对象。")
            return value
        if not isinstance(field, QLineEdit):
            raise ValueError(f"{key} 使用了未知的编辑控件。")
        text = field.text().strip()
        if old_value is None:
            return None if not text or text.casefold() == "null" else text
        if isinstance(old_value, int) and not isinstance(old_value, bool):
            return int(text)
        if isinstance(old_value, float):
            return float(text)
        return text

    def _write_form_values(self, target: dict) -> None:
        for key, field in self._form_fields.items():
            target[key] = self._read_field_value(str(key), field, target.get(key))
        if self._timing_fields:
            timing = target.setdefault("Timing", {})
            if not isinstance(timing, dict):
                timing = target["Timing"] = {}
            for key, field in self._timing_fields.items():
                timing[key] = self._read_field_value(key, field, timing.get(key))
        if target.get("Command") == "Injection":
            self._write_bunch_values(target)
            self._normalize_injection(target)

    def _write_bunch_values(self, target: dict | None) -> None:
        if not isinstance(target, dict) or not self._active_bunch_key:
            return
        bunch = target.get(self._active_bunch_key)
        if not isinstance(bunch, dict):
            return
        for (key, child_key), field in self._bunch_fields.items():
            if child_key is None:
                bunch[key] = self._read_field_value(key, field, bunch.get(key))
                continue
            nested = bunch.setdefault(key, {})
            if not isinstance(nested, dict):
                nested = bunch[key] = {}
            nested[child_key] = self._read_field_value(child_key, field, nested.get(child_key))

    def _update_action_visibility(self, pending: bool | None = None) -> None:
        if pending is None:
            pending = bool(self._pending_command)
        is_sequence_item = bool(self._selected_path and self._selected_path[0] == "Sequence" and self._selected_path[1])
        self.duplicate_button.setVisible(is_sequence_item and not pending)
        self.delete_button.setVisible(is_sequence_item and not pending)

    def apply_form(self) -> None:
        if self._selected_mapping is None:
            return
        if self._name_field is not None and self._selected_path and self._selected_path[0] == "Sequence":
            new_name = self._name_field.text().strip()
            old_name = self._selected_path[1]
            sequence = self.data.get("Sequence")
            if not new_name:
                QMessageBox.warning(self, "名称无效", "名称不能为空。")
                return
            if isinstance(sequence, dict) and new_name != old_name and new_name in sequence:
                QMessageBox.warning(self, "名称重复", f"Sequence 中已存在 {new_name}。")
                return
            if isinstance(sequence, dict) and old_name in sequence and new_name != old_name:
                sequence[new_name] = sequence.pop(old_name)
                self._selected_path = ("Sequence", new_name)
        try:
            if self._selected_path == ("__root__", "Space charge"):
                active_space_charge_name = self._write_space_charge_configuration()
            else:
                active_space_charge_name = None
                self._write_form_values(self._selected_mapping)
        except ValueError as exc:
            QMessageBox.warning(self, "字段无效", str(exc))
            return
        self._form_dirty = False
        self._sync_editor()
        self._refresh_tree()
        if self._selected_path and self._selected_path[0] == "Sequence":
            self._select_sequence_item(self._selected_path[1])
        elif self._selected_path == ("__root__", "Space charge"):
            self._populate_space_charge_configuration(active_space_charge_name)
        self.file_changed.emit(self.path)
        self._data_dirty = True
        self._set_sync_status("表单修改已确认，尚未保存", "warning")

    def duplicate_selected(self) -> None:
        if not self._selected_path or self._selected_path[0] != "Sequence" or not self._selected_path[1]:
            return
        sequence = self.data.get("Sequence")
        if not isinstance(sequence, dict):
            return
        source_name = self._selected_path[1]
        source = sequence.get(source_name)
        if not isinstance(source, dict):
            return
        base = f"{source_name}_copy"
        name, suffix = base, 2
        while name in sequence:
            name = f"{base}{suffix}"
            suffix += 1
        sequence[name] = deepcopy(source)
        self._sync_editor()
        self._data_dirty = True
        self._refresh_tree()
        self._select_sequence_item(name)

    def delete_selected(self) -> None:
        if not self._selected_path or self._selected_path[0] != "Sequence" or not self._selected_path[1]:
            return
        sequence = self.data.get("Sequence")
        if not isinstance(sequence, dict):
            return
        name = self._selected_path[1]
        answer = QMessageBox.question(self, "删除 command", f"确定删除 {name}？")
        if answer != QMessageBox.Yes:
            return
        sequence.pop(name, None)
        self._sync_editor()
        self._data_dirty = True
        self._refresh_tree()
        self._clear_form("已删除。")
        self.file_changed.emit(self.path)

    def delete_selected_sequence_rows(self) -> None:
        """Delete all currently selected visible sequence-table rows."""
        sequence = self.data.get("Sequence")
        if not isinstance(sequence, dict):
            return
        names = []
        for row in sorted({index.row() for index in self.sequence_table.selectionModel().selectedRows()}):
            item = self.sequence_table.item(row, 0)
            if item is not None:
                name = item.data(Qt.UserRole)
                if isinstance(name, str) and name in sequence:
                    names.append(name)
        if not names:
            return
        answer = QMessageBox.question(self, "删除 command", f"确定删除选中的 {len(names)} 个 command？")
        if answer != QMessageBox.Yes:
            return
        for name in names:
            sequence.pop(name, None)
        self._sync_editor()
        self._data_dirty = True
        self._refresh_tree()
        self._clear_form(f"已删除 {len(names)} 个 command。")
        self.file_changed.emit(self.path)

    def _clear_form(self, hint: str = "在配置概览中选择配置组或 Sequence command。") -> None:
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)
        self._selected_mapping = None
        self._selected_path = None
        self._form_fields = {}
        self._bunch_fields = {}
        self._name_field = None
        self._bunch_selector = None
        self._active_bunch_key = None
        self._injection_pending = False
        self._madx_fields = {}
        self._madx_preview = None
        self._madx_preview_signature = None
        self._timing_fields = {}
        self._space_charge_fields = {}
        self._space_charge_selector = None
        self._space_charge_name_field = None
        self._space_charge_enabled_field = None
        self._active_space_charge_configuration = None
        self._pending_command = None
        self.form_title.setText("参数配置")
        self.form_hint.setText(hint)
        self.form_apply.setEnabled(False)
        self.madx_import_button.setVisible(False)
        self.madx_preview_button.setVisible(False)
        self.insert_button.setVisible(False)
        self.duplicate_button.setVisible(False)
        self.delete_button.setVisible(False)

    def _confirm_form_navigation(self) -> bool:
        """Ask before replacing an unconfirmed property form with another one."""
        if not self._form_dirty:
            return True
        answer = QMessageBox.question(
            self,
            "未确认表单修改",
            "当前属性修改尚未确认。切换会放弃这些修改，是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return False
        self._form_dirty = False
        self._set_sync_status("已放弃未确认的表单修改", "warning")
        return True

    def _sync_editor(self) -> None:
        self._editor_syncing = True
        self.editor.setPlainText(json.dumps(self.data, indent=4, ensure_ascii=False))
        self._editor_syncing = False
        self._json_dirty = False

    def _select_sequence_item(self, name: str) -> None:
        if not self._confirm_form_navigation():
            return
        sequence = self.data.get("Sequence")
        target = sequence.get(name) if isinstance(sequence, dict) else None
        if not isinstance(target, dict):
            return
        self._populate_form(f"Sequence · {name}", target, name_value=name)
        self._selected_path = ("Sequence", name)
        self._pending_command = None
        self._update_action_visibility()
        for row in range(self.sequence_table.rowCount()):
            item = self.sequence_table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == name:
                self.sequence_table.selectRow(row)
                break

    def _select_top_level_item(self, key: str, child_key: str | None = None) -> None:
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            if item.data(0, Qt.UserRole) == (key, child_key):
                self.tree.setCurrentItem(item)
                self._tree_clicked(item, 0)
                return
            iterator += 1


class RunPage(QWidget):
    def __init__(self, config: ConfigPage) -> None:
        super().__init__()
        self.config = config
        self.process: QProcess | None = None
        self.started_at = 0.0
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        header = QHBoxLayout()
        header.addWidget(QLabel("运行"))
        header.addStretch()
        self.run_path = QLabel("请先在配置页加载 JSON")
        self.run_path.setObjectName("muted")
        header.addWidget(self.run_path)
        self.start_button = button("开始运行", "primary")
        self.start_button.clicked.connect(self.start_run)
        self.stop_button = button("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_run)
        header.addWidget(self.start_button)
        header.addWidget(self.stop_button)
        root.addLayout(header)
        self.progress = BusyProgressBar()
        self.progress.setVisible(False)
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
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Cascadia Code", 10))
        root.addWidget(self.log, 1)
        config.file_changed.connect(self._path_changed)

    def _path_changed(self, path: str) -> None:
        self.run_path.setText(path or "请先在配置页加载 JSON")

    def start_run(self) -> None:
        path = self.config.path
        if not path or not Path(path).exists():
            QMessageBox.information(self, "需要输入文件", "请先在配置页加载并导出一个 JSON 文件。")
            return
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        code = "from PASS.main import main; main(sys.argv[1])"
        self.process.setArguments(["-u", "-c", "import sys; " + code, path])
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(lambda _: self.log.appendPlainText("[PASS] 进程启动失败"))
        self.log.clear()
        self.started_at = time.monotonic()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setVisible(True)
        self.state.setText("运行中")
        self.process.start()

    def _read_output(self) -> None:
        if self.process:
            output = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
            self.log.appendPlainText(output.rstrip())
            turns = re.findall(r"(?:turn|Turn)\s+(\d+)", output)
            if turns:
                self.state.setText(f"运行中 · turn {turns[-1]}")

    def _finished(self, code: int, status: QProcess.ExitStatus) -> None:
        self._read_output()
        self.progress.setVisible(False)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.state.setText("完成" if code == 0 else f"失败（退出码 {code}）")
        self.elapsed.setText(f"耗时：{time.monotonic() - self.started_at:.1f} s")
        self.eta.setText("预计剩余：--")

    def stop_run(self) -> None:
        if self.process and self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self.state.setText("已停止")


class PlotPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.columns: dict[str, list[float]] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        header = QHBoxLayout()
        header.addWidget(QLabel("绘图"))
        header.addStretch()
        load = button("加载 CSV / TFS")
        load.clicked.connect(self.load_data)
        header.addWidget(load)
        root.addLayout(header)
        self.canvas = PlotCanvas()
        controls = QHBoxLayout()
        controls.addWidget(QLabel("X 列"))
        self.x_column_box = QComboBox()
        self.x_column_box.currentTextChanged.connect(self._select_series)
        controls.addWidget(self.x_column_box)
        controls.addWidget(QLabel("Y 列"))
        self.column_box = QComboBox()
        self.column_box.currentTextChanged.connect(self._select_series)
        controls.addWidget(self.column_box)
        fit = button("适配视图")
        fit.setToolTip("恢复到当前数据的完整范围")
        fit.clicked.connect(self.canvas.fit_view)
        controls.addWidget(fit)
        controls.addWidget(QLabel("滚轮缩放；StatMonitor、ParticleMonitor 和 tune 插件可继续接入"))
        controls.addStretch()
        root.addLayout(controls)
        root.addWidget(self.canvas, 1)
        self.info = QLabel("未加载数据")
        self.info.setObjectName("muted")
        root.addWidget(self.info)

    def load_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "加载结果文件", "", "Data files (*.csv *.tfs);;All files (*)")
        if not path:
            return
        try:
            columns = self._read_table(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "读取失败", str(exc))
            return
        self.columns = columns
        self.x_column_box.blockSignals(True)
        self.column_box.clear()
        self.x_column_box.clear()
        self.x_column_box.addItems(list(columns))
        self.column_box.addItems(list(columns))
        self.x_column_box.setCurrentIndex(0)
        if self.column_box.count() > 1:
            self.column_box.setCurrentIndex(1)
        self.x_column_box.blockSignals(False)
        self._select_series()
        self.info.setText(f"{path} · {len(next(iter(columns.values()), []))} rows · {len(columns)} numeric columns")

    @staticmethod
    def _read_table(path: str) -> dict[str, list[float]]:
        raw = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        lines = [line.strip() for line in raw if line.strip()]
        if not lines:
            raise ValueError("文件为空")
        if Path(path).suffix.lower() == ".csv":
            rows = list(csv.reader(lines))
            header = [cell.strip().strip('"') for cell in rows[0]]
            data_rows = rows[1:]
        else:
            header_index = next((i for i, line in enumerate(lines) if line.startswith("*")), None)
            if header_index is None:
                header = lines[0].split()
                data_rows = [line.split() for line in lines[1:]]
            else:
                header = lines[header_index][1:].split()
                data_rows = [
                    line.split()
                    for line in lines[header_index + 1:]
                    if not line.startswith(("@", "$", "*", "#"))
                ]
        result = {name: [] for name in header}
        for row in data_rows:
            if len(row) != len(header):
                continue
            for name, value in zip(header, row):
                try:
                    result[name].append(float(value.strip().strip('"')))
                except ValueError:
                    pass
        return {key: values for key, values in result.items() if values}

    def _select_series(self, _column: str = "") -> None:
        self.canvas.set_series(
            self.columns.get(self.x_column_box.currentText(), []),
            self.columns.get(self.column_box.currentText(), []),
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"PASS v{__version__}")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(
            """
            QWidget { background: #282c34; color: #d7dae0; font-size: 14px; }
            QMainWindow { background: #282c34; }
            QLabel { padding: 2px; }
            QLabel#brand { color: #61afef; font-size: 22px; font-weight: 700; }
            QLabel#muted, QLabel#runStat { color: #8b93a1; }
            QPushButton { background: #2c313a; border: 1px solid #4b5564; border-radius: 4px; padding: 8px 14px; }
            QPushButton:hover { border-color: #61afef; background: #263b50; }
            QPushButton#primary { background: #2d5f91; border-color: #61afef; }
            QPushButton#required { border-color: #e5c07b; color: #e5c07b; }
            QPushButton#required:hover { border-color: #e5c07b; background: #3c3628; }
            QScrollArea#libraryScroll { background: transparent; border: 0; }
            QWidget#librarySectionBody { background: #20252d; border-left: 1px solid #3b4350; border-right: 1px solid #3b4350; border-bottom: 1px solid #3b4350; }
            QToolButton#librarySectionHeader { background: #21252b; color: #d7dae0; border: 1px solid #3b4350; border-radius: 3px; padding: 8px 10px; text-align: left; font-weight: 600; }
            QToolButton#librarySectionHeader:hover { background: #263b50; border-color: #61afef; }
            QToolButton#librarySectionHeader:checked { color: #61afef; border-color: #61afef; border-bottom-left-radius: 0; border-bottom-right-radius: 0; }
            QLabel#syncStatus { color: #98c379; padding: 4px 8px; }
            QLabel#syncStatus[state="warning"] { color: #e5c07b; }
            QPushButton#validationStatus { background: #20252d; border: 1px solid #3b4350; padding: 7px 10px; border-radius: 4px; color: #98c379; }
            QPushButton#validationStatus:hover { background: #263b50; border-color: #61afef; }
            QPushButton#validationStatus[state="error"] { color: #e06c75; border-color: #7b3842; }
            QComboBox#sequenceCommandFilter { min-width: 150px; }
            QLabel#legendReadonly, QLabel#legendChoice, QLabel#legendInput, QLabel#legendJson { border: 1px solid #3b4350; border-radius: 3px; padding: 2px 7px; font-size: 11px; }
            QLabel#legendReadonly { color: #a9b3c1; border-left: 3px solid #6b7788; }
            QLabel#legendChoice { color: #c678dd; border-left: 3px solid #c678dd; }
            QLabel#legendInput { color: #61afef; border-left: 3px solid #61afef; }
            QLabel#legendJson { color: #e5c07b; border-left: 3px solid #e5c07b; }
            QTreeWidget, QPlainTextEdit, QListWidget, QComboBox, QTableWidget { background: #1b1f24; border: 1px solid #3b4350; border-radius: 4px; }
            QLineEdit { background: #1b1f24; border: 1px solid #3b4350; border-radius: 4px; padding: 6px; }
            QLineEdit#valueField { border-left: 3px solid #61afef; }
            QLineEdit#readonlyField { background: #20252d; color: #a9b3c1; border-left: 3px solid #6b7788; }
            QComboBox#choiceField { border-left: 3px solid #c678dd; padding: 4px 6px; }
            QCheckBox#booleanField { color: #98c379; padding: 5px 2px; }
            QPlainTextEdit#jsonField { border-left: 3px solid #e5c07b; background: #1b1f24; }
            QGroupBox#bunchBox { border: 1px solid #3b4350; border-radius: 4px; margin-top: 10px; padding: 8px; }
            QGroupBox#bunchBox::title { color: #61afef; subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QGroupBox { border: 1px solid #3b4350; border-radius: 4px; margin-top: 8px; padding: 6px; }
            QGroupBox::title { color: #8b93a1; subcontrol-origin: margin; left: 8px; padding: 0 3px; }
            QGroupBox#configGroup { border: 1px solid #3b4350; border-radius: 4px; margin-top: 10px; padding: 8px; }
            QGroupBox#configGroup::title { color: #61afef; subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QTreeWidget::item:selected, QListWidget::item:selected { background: #263b50; color: #d7dae0; }
            QTableWidget::item:selected { background: #263b50; color: #d7dae0; }
            QScrollBar:vertical { background: #11151a; width: 12px; margin: 0; border: 1px solid #3b4350; }
            QScrollBar::handle:vertical { background: #6b7788; min-height: 24px; border-radius: 4px; }
            QScrollBar::handle:vertical:hover { background: #8b93a1; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { background: #11151a; height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: #11151a; }
            QScrollBar:horizontal { background: #11151a; height: 12px; margin: 0; border: 1px solid #3b4350; }
            QScrollBar::handle:horizontal { background: #6b7788; min-width: 24px; border-radius: 4px; }
            QScrollBar::handle:horizontal:hover { background: #8b93a1; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { background: #11151a; width: 0; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: #11151a; }
            QHeaderView::section { background: #21252b; color: #8b93a1; border: 0; border-bottom: 1px solid #3b4350; padding: 6px; }
            QTabWidget::pane { border: 1px solid #3b4350; border-radius: 4px; }
            QTabBar::tab { background: #21252b; color: #8b93a1; padding: 7px 12px; border: 1px solid #3b4350; border-bottom: 0; }
            QTabBar::tab:selected { background: #263b50; color: #d7dae0; border-color: #61afef; }
            QSplitter::handle { background: #3b4350; }
            QStatusBar { background: #21252b; color: #8b93a1; }
            QWidget#lineNumberArea { background: #161a20; border-right: 1px solid #3b4350; }
            """
        )
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.setContentsMargins(20, 14, 20, 14)
        brand = QLabel(f"PASS v{__version__}")
        brand.setObjectName("brand")
        header.addWidget(brand)
        header.addSpacing(24)
        self.project_label = QLabel("未加载项目")
        self.project_label.setObjectName("muted")
        header.addWidget(self.project_label)
        header.addStretch()
        outer.addLayout(header)
        nav = QHBoxLayout()
        nav.setContentsMargins(20, 0, 20, 12)
        self.nav = []
        for index, label in enumerate(("配置", "运行", "绘图")):
            item = button(label)
            item.setMinimumWidth(92)
            item.clicked.connect(lambda checked=False, i=index: self._show_page(i))
            nav.addWidget(item)
            self.nav.append(item)
        nav.addStretch()
        outer.addLayout(nav)
        self.stack = QStackedWidget()
        self.config = ConfigPage()
        self.run = RunPage(self.config)
        self.plot = PlotPage()
        for page in (self.config, self.run, self.plot):
            self.stack.addWidget(page)
        outer.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.config.file_changed.connect(self._project_changed)
        self._show_page(0)

    def _show_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for i, item in enumerate(self.nav):
            item.setObjectName("primary" if i == index else "")
            item.style().unpolish(item)
            item.style().polish(item)

    def _project_changed(self, path: str) -> None:
        self.project_label.setText(Path(path).name if path else "未加载项目")
        self.statusBar().showMessage(f"已加载：{path}" if path else "配置已更新")

    def closeEvent(self, event) -> None:
        """Prevent an accidental close from discarding a configuration edit."""
        if not self.config.has_unsaved_changes():
            event.accept()
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("未保存修改")
        if self.config._form_dirty:
            dialog.setText("当前配置包含未确认的属性表单修改。保存并退出会先确认当前表单。")
        else:
            dialog.setText("当前配置有未保存修改。")
        save = dialog.addButton("保存并退出", QMessageBox.AcceptRole)
        discard = dialog.addButton("放弃修改", QMessageBox.DestructiveRole)
        dialog.addButton("取消", QMessageBox.RejectRole)
        dialog.exec()
        if dialog.clickedButton() is save:
            if self.config.save_pending_changes():
                event.accept()
            else:
                event.ignore()
        elif dialog.clickedButton() is discard:
            event.accept()
        else:
            event.ignore()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("PASS")
    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(BASE))
    palette.setColor(QPalette.WindowText, QColor("#d7dae0"))
    palette.setColor(QPalette.Base, QColor("#1b1f24"))
    palette.setColor(QPalette.Text, QColor("#d7dae0"))
    palette.setColor(QPalette.Button, QColor(PANEL))
    palette.setColor(QPalette.ButtonText, QColor("#d7dae0"))
    app.setPalette(palette)
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, lambda: _enable_dark_title_bar(window))
    sys.exit(app.exec())


def _enable_dark_title_bar(window: QMainWindow) -> None:
    """Use the native dark title bar where Windows supports it; no-op on Linux."""
    if sys.platform != "win32":
        return
    try:
        value = ctypes.c_int(1)
        hwnd = ctypes.c_void_p(int(window.winId()))
        for attribute in (20, 19):  # Windows 11, then older Windows 10 builds.
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value),
            ) == 0:
                break
    except (AttributeError, OSError):
        pass


__all__ = ["main", "MainWindow"]
