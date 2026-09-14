"""PASS GUI application.

The GUI is intentionally an orchestration layer. PASS schemas and the tracking
engine remain the source of truth; this module handles project files, process
control, and presentation.
"""

from __future__ import annotations

import csv
import codecs
from collections import Counter
from copy import deepcopy
import json
import math
import re
import sys
import time
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QEvent, QProcess, QTimer, Qt, Signal, QSettings, QSize
from PySide6.QtGui import QAction, QDoubleValidator, QIntValidator, QPainter, QPalette, QPen, QTextCursor, QTextFormat
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
    QLayout,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QMenu,
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
from PASS.gui.appearance import THEMES, JsonHighlighter, apply_application_theme, code_font, icon
from PASS.gui.help import HelpMenu
from PASS.gui.project import FILE_FIELDS, missing_files, read_json
from PASS.gui.tools import ToolsPage
from PASS.gui.structured import (
    ApertureEditor, CoefficientsEditor, DevicesEditor, InternalSpaceChargeEditor,
    ListEditor, NumericTable, ObjectEditor, ObjectListEditor, ParticleEditor, RangeEditor,
    StructuredField, TurnsEditor, Column,
)


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
    "Method": ("pic", "frozen", "quasi-frozen"),
    "Solver": ("fd_dirichlet", "dst_dirichlet", "fft_free_space",
               "gaussian_round_free_space", "gaussian_ellipse_free_space",
               "uniform_round_free_space", "uniform_ellipse_free_space"),
    "Particle Deposition Method": ("CIC", "TSC"),
    "Coverage check": ("warn", "error", "off"),
    "Coverage mode": ("full-ring", "partial"),
    "File Time Kind": ("turn", "second"),
}

TIMING_MODE_OPTIONS = ("off", "turn", "command", "synchronized-command")

# Descriptions are copied from the public schema lazily by ``_schema_help``.
# Keeping this lookup in the GUI means new schema fields automatically get the
# same hover help without duplicating every description here.
_SCHEMA_HELP: dict[str, str] | None = None

FIELD_HELP = {
    "Coverage check": "跟踪前检查 SC 权重和区间：warn 警告，error 停止，off 不检查。",
    "Coverage mode": "full-ring 要求覆盖全环；partial 允许仅覆盖部分区段。",
    "Expected SC length (m)": "partial 模式可选的每圈 SC 总作用长度；full-ring 模式留空并使用环长。",
    "SC start (m)": "该显式 SC command 代表的积分区间起点，仅用于覆盖检查；不改变踢的位置或粒子传输。",
    "S (m)": "Command 在环中的纵向位置，单位为 m。",
    "S previous (m)": "Twiss 传输矩阵的上一光学点位置，单位为 m。",
    "Grid Width X (m)": "水平网格全宽，单位 m；与水平半宽二选一。",
    "Grid Width Y (m)": "垂直网格全宽，单位 m；与垂直半宽二选一。",
    "Grid Half Width X (m)": "水平网格半宽，范围为 [-半宽, +半宽]，单位 m。",
    "Grid Half Width Y (m)": "垂直网格半宽，范围为 [-半宽, +半宽]，单位 m。",
    "Center X (m)": "frozen 解析源分布的水平中心，单位 m；留空取 0。不是 PIC 网格中心。",
    "Center Y (m)": "frozen 解析源分布的垂直中心，单位 m；留空取 0。不是 PIC 网格中心。",
    "Angle (rad)": "frozen 椭圆分布局部 x 主轴相对实验室 x 轴的逆时针角度，单位 rad；留空取 0。",
    "Sigma (m)": "圆形高斯分布的单轴 RMS 尺寸 σ，单位 m；必须大于 0。",
    "Sigma X (m)": "椭圆高斯分布局部 x 主轴的 RMS 尺寸，单位 m；必须大于 0。",
    "Sigma Y (m)": "椭圆高斯分布局部 y 主轴的 RMS 尺寸，单位 m；必须大于 0。",
    "Radius (m)": "均匀圆盘源分布的外半径，单位 m；不是管壁半径，必须大于 0。",
    "Semi-axis A (m)": "均匀椭圆源分布局部 x 方向的半轴，单位 m；必须大于 0。",
    "Semi-axis B (m)": "均匀椭圆源分布局部 y 方向的半轴，单位 m；必须大于 0。",
    "Harmonic Number": "束团分组数；添加或删除 bunch 时由界面自动保持一致。",
    "Harmonic ID of this bunch": "该 bunch 的零起始分组编号，由界面按顺序维护。",
    "Random Seed": "分布生成随机种子。留空（null）时每次运行使用非确定性随机数。",
    "Timing": "运行进度和 ETA 的输出方式。",
    "Device Id": "GPU 后端使用的设备编号列表。",
    "Insert Particle Coordinate": "每行一个粒子：x、px、y、py、z_rel、dp/p。行数就是手动插入粒子数，包含在宏粒子总数内。",
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


class CompactFormBody(QWidget):
    """Use the form's natural height instead of distributing spare scroll space."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Let Qt recompute the size as sections are inserted or shown. Caching
        # maximumHeight during LayoutRequest can freeze a new form at its empty
        # height before nested sections have been polished.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)


class CollapsibleSection(QWidget):
    """A small independently toggled section for the vertical component library."""

    def __init__(self, title: str, parent: QWidget | None = None, *, depth: int = 0) -> None:
        super().__init__(parent)
        self.title = title
        self.header = QPushButton()
        self.header.setObjectName("librarySectionHeader")
        self.header.setProperty("depth", depth)
        self.header.setText(title)
        self.header.setLayoutDirection(Qt.LeftToRight)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.setCheckable(True)
        self.header.setChecked(False)
        self.header.toggled.connect(self._set_expanded)

        self.body = QWidget()
        self.body.setObjectName("librarySectionBody")
        self.body.setAttribute(Qt.WA_StyledBackground, True)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(14, 3, 0, 5)
        self.body_layout.setSpacing(2)
        self.body.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header_row = QHBoxLayout()
        self.header_row.setContentsMargins(0, 0, 0, 0)
        self.header_row.setSpacing(0)
        self.header_row.addWidget(self.header, 1)
        self.toggle_button = QToolButton()
        self.toggle_button.setObjectName("librarySectionToggle")
        self.toggle_button.setProperty("depth", depth)
        self.toggle_button.setFixedSize(22, 28)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.setAccessibleName(f"展开或收起{title}")
        self.toggle_button.clicked.connect(self.header.toggle)
        self.header_row.addWidget(self.toggle_button)
        layout.addLayout(self.header_row)
        layout.addWidget(self.body)

    def _set_expanded(self, expanded: bool) -> None:
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.body.setVisible(expanded and self.body_layout.count() > 0)


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
        painter.setBrush(self.palette().base())
        painter.drawRect(self.rect())
        width = max(48, int(self.width() * 0.22))
        x = int((self.width() + width) * self._offset - width)
        painter.fillRect(x, 0, width, self.height(), self.palette().link())


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
        painter.fillRect(event.rect(), self.palette().color(QPalette.Base))
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(self.palette().color(QPalette.PlaceholderText))
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
        painter.fillRect(self.rect(), self.palette().color(QPalette.Base))
        painter.setRenderHint(QPainter.Antialiasing)
        area = self._plot_area()
        painter.setPen(QPen(self.palette().color(QPalette.Mid), 1))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = area.bottom() - fraction * area.height()
            painter.drawLine(area.left(), int(y), area.right(), int(y))
        if len(self.values) < 2 or self._x_limits is None or self._y_limits is None:
            painter.setPen(self.palette().color(QPalette.PlaceholderText))
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
        painter.setPen(QPen(self.palette().color(QPalette.Link), 2))
        for first, second in zip(points, points[1:]):
            painter.drawLine(*first, *second)
        painter.setPen(self.palette().color(QPalette.PlaceholderText))
        painter.drawText(8, area.top() + 5, f"max {high:.5g}")
        painter.drawText(8, area.bottom(), f"min {low:.5g}")
        painter.drawText(area.left(), self.height() - 12, f"{x_low:.5g}")
        painter.drawText(area.right() - 65, self.height() - 12, f"{x_high:.5g}")


class ConfigPage(QWidget):
    file_changed = Signal(str)
    changed = Signal()

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
        self._optics_mode: str | None = None
        self._optics_fields: dict[str, QWidget] = {}
        self._optics_preview: dict | None = None
        self._timing_fields: dict[str, QWidget] = {}
        self._space_charge_fields: dict[str, QWidget] = {}
        self._space_charge_selector: PropertyComboBox | None = None
        self._space_charge_name_field: QLineEdit | None = None
        self._space_charge_enabled_field: QCheckBox | None = None
        self._space_charge_coverage_fields: dict[str, QWidget] = {}
        self._active_space_charge_configuration: str | None = None
        self._editor_syncing = False
        self._json_dirty = False
        self._form_dirty = False
        self._data_dirty = False
        self._validation_issues: list[str] = []
        self.recipes: list[dict] = []
        self.base_dir = Path.cwd()
        self._history: list[dict] = [deepcopy(self.data)]
        self._history_index = 0
        self._history_restoring = False
        self._field_defaults: dict = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)
        toolbar = QHBoxLayout()
        self.file_label = QLabel("beam.json")
        toolbar.addWidget(self.file_label)
        self.input_selector = PropertyComboBox()
        self.input_selector.setMinimumWidth(180)
        self.input_selector.hide()
        toolbar.addWidget(self.input_selector)
        toolbar.addStretch()
        self.sync_status = QLabel("JSON 输入")
        self.sync_status.setObjectName("syncStatus")
        toolbar.addWidget(self.sync_status)
        self.contents_button = button("项目内容")
        self.contents_button.hide()
        toolbar.addWidget(self.contents_button)
        root.addLayout(toolbar)
        self.splitter = splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        left_frame = QFrame()
        left_frame.setObjectName("libraryPanel")
        left_frame.setMinimumWidth(164)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(8, 8, 6, 8)
        library_scroll = QScrollArea()
        library_scroll.setObjectName("libraryScroll")
        library_scroll.setWidgetResizable(True)
        library_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        library_body = QWidget()
        library_layout = QVBoxLayout(library_body)
        library_layout.setContentsMargins(0, 2, 0, 2)
        library_layout.setSpacing(2)
        self.library_sections: dict[str, CollapsibleSection] = {}

        def add_section(title: str, entries: tuple[tuple[str, str, object], ...], expanded: bool = False) -> None:
            section = CollapsibleSection(title)
            if title == "输入配置":
                section.header.setText("输入配置（必需）")
            section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.library_sections[title] = section
            for text, tip, handler in entries:
                item = button(text)
                item.setToolTip(tip)
                item.setMinimumWidth(0)
                item.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
                item.clicked.connect(handler)
                section.body_layout.addWidget(item)
            section.header.setChecked(expanded)
            library_layout.addWidget(section)

        add_section(
            "输入配置",
            (
                ("全局配置", "编辑 PASS 输入 JSON 根对象中的全部全局配置。", self.configure_global),
                ("束流 / Injection", "编辑束流注入参数。", self.configure_injection),
                ("执行序列", "查看执行序列。", lambda: self.editor_tabs.setCurrentIndex(0)),
            ),
            expanded=True,
        )
        add_section(
            "Twiss 与光学",
            (
                ("导入 MAD-X Twiss…", "读取 MAD-X 导出的 Twiss/TFS 文件，转换为 Twiss command 并追加到 Sequence。", self.configure_madx_twiss),
                ("插入单圈传输矩阵", "输入一组周期光学参数和单圈 tune，生成一圈 Twiss 传输。", lambda: self.configure_optics_generator("one_turn")),
                ("平滑近似 Twiss 序列", "按一圈分段数生成等间距 Twiss 传输点。", lambda: self.configure_optics_generator("smooth")),
                ("插入 Twiss 传输点", "填写起点与终点光学参数，手动插入一段 Twiss 传输。", lambda: self.select_command("Twiss")),
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
                ("导入 MAD-X 元件…", "读取 MAD-X 导出的 Twiss/TFS 表，转换为 PASS 元件并追加到 Sequence。", self.configure_madx_elements),
            ) + tuple((command, "浏览默认参数；确认后才插入 Sequence。", lambda checked=False, cmd=command: self.select_command(cmd))
                  for command in ("Marker", "Drift", "SBend", "Quadrupole", "Sextupole", "Octupole", "Multipole", "Solenoid", "Kicker", "ElSeparator", "RFCavity", "Exciter")),
        )
        add_section(
            "监测与诊断",
            tuple((command, "浏览默认参数；确认后才插入 Sequence。", lambda checked=False, cmd=command: self.select_command(cmd))
                  for command in ("StatMonitor", "ParticleMonitor", "DistMonitor", "PhaseAdvanceMonitor")),
        )
        add_section("物理效应", ())
        physics_layout = self.library_sections["物理效应"].body_layout
        self.space_charge_menu = CollapsibleSection("空间电荷", depth=1)
        self.space_charge_menu.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for text, tip, handler in (
            ("计算配置", "管理模块开关、切片集、网格和求解器。", self.configure_space_charge),
            ("插入计算点", "手动插入引用命名计算配置的 SpaceCharge command。", lambda: self.select_command("SpaceCharge")),
        ):
            item = button(text)
            item.setToolTip(tip)
            item.clicked.connect(handler)
            self.space_charge_menu.body_layout.addWidget(item)
        physics_layout.addWidget(self.space_charge_menu)
        for title in ("尾场", "束束效应", "电子云"):
            section = CollapsibleSection(title, depth=1)
            section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            section.header.setEnabled(False)
            section.toggle_button.setVisible(False)
            section.header.setToolTip("尚未提供配置界面。")
            physics_layout.addWidget(section)
        library_layout.addStretch()
        library_scroll.setWidget(library_body)
        left_layout.addWidget(library_scroll, 1)
        splitter.addWidget(left_frame)
        editor_frame = QFrame()
        editor_frame.setObjectName("editorPanel")
        editor_frame.setMinimumWidth(350)
        editor_layout = QVBoxLayout(editor_frame)
        editor_layout.setContentsMargins(10, 8, 10, 8)
        title = QHBoxLayout()
        sequence_title = QLabel("执行序列")
        sequence_title.setObjectName("formTitle")
        title.addWidget(sequence_title)
        self.sequence_count = QLabel()
        self.sequence_count.setObjectName("muted")
        title.addWidget(self.sequence_count)
        title.addStretch()
        self.validation_label = button("未校验", "validationStatus")
        self.validation_label.clicked.connect(self._show_validation_issues)
        title.addWidget(self.validation_label)
        self.validate_button = button("校验")
        self.validate_button.clicked.connect(self.validate_input)
        title.addWidget(self.validate_button)
        editor_layout.addLayout(title)
        self.editor_tabs = QTabWidget()
        overview_panel = QWidget()
        overview_layout = QVBoxLayout(overview_panel)
        overview_layout.setContentsMargins(0, 8, 0, 0)
        # This internal tree remains an index for schema navigation. The sequence
        # table is the single visible overview, avoiding duplicate command lists.
        self.tree = QTreeWidget(self)
        self.tree.hide()
        self.tree.itemClicked.connect(self._tree_clicked)
        sequence_toolbar = QHBoxLayout()
        self.sequence_filter = QLineEdit()
        self.sequence_filter.setPlaceholderText("搜索名称或 Command")
        self.sequence_filter.setClearButtonEnabled(True)
        self.sequence_filter.textChanged.connect(self._filter_sequence_table)
        sequence_toolbar.addWidget(self.sequence_filter, 1)
        self.sequence_command_filter = PropertyComboBox()
        self.sequence_command_filter.setMaximumWidth(160)
        self.sequence_command_filter.addItem("全部 Command")
        self.sequence_command_filter.currentTextChanged.connect(self._filter_sequence_table)
        sequence_toolbar.addWidget(self.sequence_command_filter)
        self.delete_sequence_button = button("删除")
        self.delete_sequence_button.clicked.connect(self.delete_selected_sequence_rows)
        sequence_toolbar.addWidget(self.delete_sequence_button)
        overview_layout.addLayout(sequence_toolbar)
        self.sequence_table = QTableWidget(0, 8)
        self.sequence_table.setHorizontalHeaderLabels(["名称", "Command", "s / m", "状态", "计算配置", "作用长度 / m", "孔径类型", "切片集"])
        self.sequence_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sequence_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.sequence_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sequence_table.setAlternatingRowColors(False)
        self.sequence_table.verticalHeader().hide()
        self.sequence_table.verticalHeader().setDefaultSectionSize(26)
        header = self.sequence_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(50)
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self._column_menu)
        header.setToolTip("拖动分隔线调整列宽；右键选择可选列。")
        self._restore_columns()
        header.sectionResized.connect(self._save_columns)
        self.sequence_table.cellClicked.connect(self._sequence_row_clicked)
        overview_layout.addWidget(self.sequence_table, 1)
        self.editor_tabs.addTab(overview_panel, "执行序列")
        self.editor = LineNumberEditor()
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor.setObjectName("codeEditor")
        self.editor.setFont(code_font())
        self.json_highlighter = JsonHighlighter(self.editor.document())
        self.editor.setPlainText(json.dumps(self.data, indent=4, ensure_ascii=False))
        self.editor.document().contentsChange.connect(self._mark_json_dirty)
        self.editor_tabs.addTab(self.editor, "JSON 源码")
        editor_layout.addWidget(self.editor_tabs)
        splitter.addWidget(editor_frame)
        self.form_title = QLabel("属性")
        self.form_title.setObjectName("formTitle")
        self.form_frame = form_frame = QFrame()
        form_frame.setObjectName("propertyPanel")
        form_layout = QVBoxLayout(form_frame)
        form_layout.setContentsMargins(10, 10, 8, 8)
        form_layout.setSpacing(6)
        form_header = QHBoxLayout()
        self.form_title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        form_header.addWidget(self.form_title, 1)
        self.form_command = QLineEdit()
        self.form_command.setObjectName("commandBadge")
        self.form_command.setReadOnly(True)
        self.form_command.setFocusPolicy(Qt.NoFocus)
        self.form_command.setToolTip("Command 类型（只读）")
        self.form_command.hide()
        form_header.addWidget(self.form_command)
        form_layout.addLayout(form_header)
        self.form_hint = QLabel("选择执行序列中的一项，或从组件库插入。")
        self.form_hint.setObjectName("muted")
        self.form_hint.setWordWrap(True)
        self.form_hint.setMaximumHeight(44)
        form_layout.addWidget(self.form_hint)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.form_body = CompactFormBody()
        self.form_layout = QFormLayout(self.form_body)
        self.form_layout.setContentsMargins(0, 4, 2, 4)
        self.form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.form_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.form_layout.setFormAlignment(Qt.AlignTop)
        self.form_layout.setHorizontalSpacing(8)
        self.form_layout.setVerticalSpacing(6)
        scroll.setWidget(self.form_body)
        form_layout.addWidget(scroll, 1)
        actions = QGridLayout()
        actions.setSpacing(5)
        self.form_apply = button("应用修改", "primary")
        self.form_apply.setEnabled(False)
        self.form_apply.clicked.connect(self.apply_form)
        self.cancel_form_button = button("取消修改")
        self.cancel_form_button.clicked.connect(self.cancel_form)
        actions.addWidget(self.form_apply, 0, 0)
        actions.addWidget(self.cancel_form_button, 0, 1)
        self.madx_import_button = button("导入序列", "primary")
        self.madx_import_button.clicked.connect(self.import_madx_twiss)
        self.madx_preview_button = button("预览导入")
        self.madx_preview_button.clicked.connect(self.preview_madx_import)
        self.optics_preview_button = button("预览生成")
        self.optics_preview_button.clicked.connect(self.preview_optics)
        self.optics_insert_button = button("插入序列", "primary")
        self.optics_insert_button.clicked.connect(self.insert_optics)
        self.insert_button = button("插入序列", "primary")
        self.insert_button.clicked.connect(self.insert_pending_command)
        self.duplicate_button = button("复制此项")
        self.duplicate_button.clicked.connect(self.duplicate_selected)
        self.delete_button = button("删除此项")
        self.delete_button.clicked.connect(self.delete_selected)
        for item, row, column in [(self.madx_preview_button, 1, 0), (self.madx_import_button, 1, 1),
                                  (self.optics_preview_button, 2, 0), (self.optics_insert_button, 2, 1),
                                  (self.insert_button, 3, 0), (self.duplicate_button, 4, 0), (self.delete_button, 4, 1)]:
            item.hide()
            actions.addWidget(item, row, column)
        form_layout.addLayout(actions)
        form_frame.setMinimumWidth(282)
        splitter.addWidget(form_frame)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([190, 658, 312])
        root.addWidget(splitter, 1)
        self._refresh_tree()

    def configure_injection(self) -> None:
        sequence = self.data.get("Sequence", {})
        for name, item in sequence.items():
            if isinstance(item, dict) and item.get("Command") == "Injection":
                self._select_sequence_item(name)
                return
        self.select_command("Injection")

    def _restore_columns(self, reset: bool = False) -> None:
        settings = QSettings("PASS", "Editor")
        self._restoring_columns = True
        try:
            for index, width in enumerate((280, 230, 105, 80, 120, 110, 110, 110)):
                if not reset:
                    width = settings.value(f"sequence/width/{index}", width, type=int)
                self.sequence_table.setColumnWidth(index, max(50, min(2000, width)))
                visible = index < 3 or (not reset and settings.value(f"sequence/visible/{index}", False, type=bool))
                self.sequence_table.setColumnHidden(index, not visible)
        finally:
            self._restoring_columns = False
        if reset:
            self._save_columns()

    def _save_columns(self, *_args) -> None:
        if getattr(self, "_restoring_columns", False):
            return
        settings = QSettings("PASS", "Editor")
        for index in range(self.sequence_table.columnCount()):
            if not self.sequence_table.isColumnHidden(index):
                settings.setValue(f"sequence/width/{index}", self.sequence_table.columnWidth(index))
            settings.setValue(f"sequence/visible/{index}", not self.sequence_table.isColumnHidden(index))

    def _column_menu(self, position) -> None:
        menu = QMenu(self)
        menu.addSection("显示的列")
        for index in range(self.sequence_table.columnCount()):
            label = self.sequence_table.horizontalHeaderItem(index).text()
            action = menu.addAction(label + ("（必备）" if index < 3 else ""))
            action.setCheckable(True)
            action.setChecked(not self.sequence_table.isColumnHidden(index))
            action.setEnabled(index >= 3)
            action.toggled.connect(lambda checked, i=index: self._toggle_column(i, checked))
        menu.addSeparator()
        menu.addAction("恢复默认列与宽度", lambda: self._restore_columns(True))
        menu.exec(self.sequence_table.horizontalHeader().mapToGlobal(position))

    def _toggle_column(self, index: int, visible: bool) -> None:
        if index < 3:
            return
        self.sequence_table.setColumnHidden(index, not visible)
        self._save_columns()

    def cancel_form(self) -> None:
        selected = self._selected_path
        self._form_dirty = False
        if selected and selected[0] == "Sequence" and selected[1]:
            self._select_sequence_item(selected[1])
        elif selected == ("__root__", "Space charge"):
            self._populate_space_charge_configuration(self._active_space_charge_configuration)
        elif selected:
            self.configure_global()
        else:
            self._clear_form()
        self.changed.emit()

    def commit_pending(self) -> bool:
        if self._form_dirty and self._json_dirty:
            QMessageBox.warning(self, "两处均有修改", "属性与 JSON 源码均有未应用修改。请先取消其中一处的修改，再应用另一处。")
            return False
        if self._json_dirty and not self.apply_json():
            return False
        if self._form_dirty:
            self.apply_form()
            if self._form_dirty:
                return False
        return True

    def validate_input(self) -> None:
        if self.commit_pending():
            self._show_validation_issues(full=True)

    def reset_history(self) -> None:
        self._history = [deepcopy(self.data)]
        self._history_index = 0

    def _record_history(self) -> None:
        if self._history_restoring or self._history[self._history_index] == self.data:
            return
        self._history = self._history[:self._history_index + 1]
        self._history.append(deepcopy(self.data))
        if len(self._history) > 40:
            self._history.pop(0)
        self._history_index = len(self._history) - 1

    def undo_data(self, redo: bool = False) -> None:
        index = self._history_index + (1 if redo else -1)
        if not 0 <= index < len(self._history) or not self._confirm_form_navigation():
            return
        self._history_index = index
        self.data = deepcopy(self._history[index])
        self._history_restoring = True
        self._sync_editor()
        self._history_restoring = False
        self._form_dirty = False
        self._data_dirty = True
        self._clear_form()
        self._refresh_tree()
        self.changed.emit()


    def has_unsaved_changes(self) -> bool:
        """Return whether the current project has changes not written to disk."""
        return self._data_dirty or self._json_dirty or self._form_dirty


    def apply_json(self) -> bool:
        try:
            data = read_json(self.editor.toPlainText().encode("utf-8"))
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
        self._record_history()
        self._json_dirty = False
        self._form_dirty = False
        self._data_dirty = True
        self._refresh_tree()
        self._clear_form()
        self.file_changed.emit(self.path)
        self._set_sync_status("JSON 修改已确认", "ok")
        return True



    def _mark_json_dirty(self, _position: int, removed: int, added: int) -> None:
        # Syntax highlighting changes formats and emits textChanged even when
        # the source is untouched. Only actual character edits dirty the JSON.
        if self._editor_syncing or not (removed or added):
            return
        self._json_dirty = True
        self._set_sync_status("JSON 有未确认修改", "warning")

    def _set_sync_status(self, text: str, state: str = "") -> None:
        self.sync_status.setText(text)
        self.sync_status.setProperty("state", state)
        self.sync_status.style().unpolish(self.sync_status)
        self.sync_status.style().polish(self.sync_status)
        self.changed.emit()

    def _command_template(self, command: str) -> dict:
        """Return a safe preview template without changing project data."""
        sequence = self.data.get("Sequence", {})
        if not isinstance(sequence, dict):
            return {}
        position = max(
            (item["S (m)"] for item in sequence.values() if isinstance(item, dict)
             and type(item.get("S (m)")) in (int, float) and abs(item["S (m)"]) < 1e290),
            default=0.0,
        )
        if command == "Injection":
            # Reuse the public schema so an inserted beam source is complete.
            from PASS.para.schema.bunch import InjectionItem

            template = InjectionItem().to_sequence_dict()
            template["S (m)"] = 0.0
            # The Gaussian rejection sampler requires a non-degenerate ellipse.
            # These are GUI starting values; the public physics schema is unchanged.
            template["bunch0"]["Emittance x (m'rad)"] = 1e-6
            template["bunch0"]["Emittance y (m'rad)"] = 1e-6
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
            if command == "RFCavity":
                required["Components"] = [{"Voltage (V)": 0.0, "Harmonic": 1, "Phase (rad)": 0.0}]
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
        if command == "Twiss":
            self.form_title.setText("插入 Twiss 传输点")
            self.form_hint.setText("填写起点与终点的光学参数，插入从起点到终点的一段 Twiss 传输。Mu 的单位为周（2π）。")
        elif command == "SpaceCharge":
            self.form_title.setText("空间电荷 · 插入计算点")
            self.form_hint.setText("引用共享计算配置；孔径默认与网格同尺寸，壁上和壁外粒子在求场前损失。Aperture type/value 在 FD/DST 中同时定义导体边界；DST 必须使用完整网格矩形，自由空间方法仅用孔径判断损失。")

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

    def configure_optics_generator(self, mode: str) -> None:
        """Open a non-mutating generator with explicit preview and insertion."""
        if not self._confirm_form_navigation():
            return
        if self._json_dirty and not self.apply_json():
            return
        self._clear_form()
        self._optics_mode = mode
        smooth = mode == "smooth"
        self.form_title.setText("生成平滑近似 Twiss 序列" if smooth else "插入单圈传输矩阵")
        self.form_hint.setText(
            "N 段传输生成 N＋1 个 Twiss 点，包含 0 和 C；仅生成光学传输，不插入空间电荷计算点。"
            if smooth else
            "只需一组周期光学参数。自动设置起点 S=0、终点 S=C，以及一圈的相位推进。"
        )
        self.form_apply.hide()
        self.optics_preview_button.show()
        self.optics_insert_button.show()
        self.optics_insert_button.setEnabled(False)

        def add_field(layout: QFormLayout, key: str, label: str, default, tip: str = "") -> QWidget:
            field = self._make_field("Longitudinal transfer" if key == "longitudinal_transfer" else key, default)
            field.setAccessibleName(label)
            if tip:
                field.setToolTip(tip)
            self._optics_fields[key] = field
            layout.addRow(label, field)
            signal = field.currentTextChanged if isinstance(field, QComboBox) else field.textChanged
            signal.connect(self._optics_changed)
            return field

        add_field(self.form_layout, "name", "名称前缀" if smooth else "名称", "smooth" if smooth else "one_turn")
        circumference = self.data.get("Circumference (m)", 1.0)
        add_field(self.form_layout, "circumference", "周长 C (m)", str(circumference), "默认读取全局周长；插入时将全局周长同步为此值。")
        tune_tip = "平滑近似应填写包含整数部分的完整工作点，例如 9.47，而不是 0.47。"
        add_field(self.form_layout, "qx", "Qx（完整工作点）" if smooth else "Qx", 1.0, "单圈水平 tune，单位为周（2π）；" + tune_tip)
        add_field(self.form_layout, "qy", "Qy（完整工作点）" if smooth else "Qy", 1.0, "单圈垂直 tune，单位为周（2π）；" + tune_tip)
        if smooth:
            add_field(self.form_layout, "num_segments", "一圈分段数 N", 100, "正整数；生成 N 段传输、N＋1 个 Twiss 点。")
        else:
            for key, label, default in (
                ("alpha_x", "αx", 0.0), ("alpha_y", "αy", 0.0),
                ("beta_x", "βx (m)", 1.0), ("beta_y", "βy (m)", 1.0),
            ):
                add_field(self.form_layout, key, label, default)
        add_field(self.form_layout, "longitudinal_transfer", "纵向传输", "off")
        add_field(self.form_layout, "muz", "Qs", 0.0, "matrix 模式的单圈纵向 tune，单位为周（2π）。")
        if smooth:
            add_field(self.form_layout, "alpha_x", "αx", 0.0, "标准平滑近似取 0。")
            add_field(self.form_layout, "alpha_y", "αy", 0.0, "标准平滑近似取 0。")
        for key, label in (("dx", "Dx (m)"), ("dpx", "Dpx"), ("dqx", "单圈色品 DQx"), ("dqy", "单圈色品 DQy")):
            add_field(self.form_layout, key, label, 0.0)
        self._optics_summary = QLabel()
        self._optics_summary.setWordWrap(True)
        self.form_layout.addRow(self._optics_summary)
        self._optics_details = QPlainTextEdit()
        self._optics_details.setReadOnly(True)
        self._optics_details.setMaximumHeight(150)
        self._optics_details.hide()
        self.form_layout.addRow(self._optics_details)
        self._optics_table = QTableWidget(0, 4)
        self._optics_table.setHorizontalHeaderLabels(["名称", "起点 S (m)", "终点 S (m)", "ΔMu x / y"])
        self._optics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._optics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._optics_table.setMaximumHeight(220)
        self._optics_table.hide()
        self.form_layout.addRow(self._optics_table)
        self._optics_changed()

    def _read_optics_parameters(self):
        from pydantic import ValidationError
        from PASS.gui.optics import OpticsParameters

        values = {"mode": self._optics_mode}
        for key, field in self._optics_fields.items():
            text = field.currentText() if isinstance(field, QComboBox) else field.text().strip()
            try:
                if key in ("name", "longitudinal_transfer"):
                    values[key] = text
                elif key == "num_segments":
                    values[key] = int(text)
                elif key == "muz" and self._optics_fields["longitudinal_transfer"].currentText() != "matrix":
                    values[key] = 0.0
                else:
                    values[key] = float(text)
            except ValueError as exc:
                expected = "正整数" if key == "num_segments" else "数值"
                raise ValueError(f"{field.accessibleName()}：请输入{expected}。") from exc
        try:
            return OpticsParameters.model_validate(values)
        except ValidationError as exc:
            messages = []
            for error in exc.errors(include_url=False):
                key = error["loc"][0] if error["loc"] else None
                field = self._optics_fields.get(key)
                label = field.accessibleName() if field is not None else "生成参数"
                explanation = {
                    "finite_number": "请输入有限数值。",
                    "greater_than": "必须大于 0。",
                    "greater_than_equal": "必须至少为 1。",
                    "string_too_short": "不能为空。",
                }.get(error["type"], str(error.get("ctx", {}).get("error", "请检查输入参数。")))
                messages.append(f"{label}：{explanation}")
            raise ValueError("\n".join(messages)) from exc

    def _optics_changed(self) -> None:
        """Invalidate previews immediately; derive beta and step without allocating points."""
        self._optics_preview = None
        self.optics_insert_button.setEnabled(False)
        self._optics_table.hide()
        self._optics_details.hide()
        self._optics_fields["muz"].setEnabled(
            self._optics_fields["longitudinal_transfer"].currentText() == "matrix")
        try:
            parameters = self._read_optics_parameters()
        except (ValueError, OverflowError) as exc:
            self._optics_summary.setText(f"请完善有效参数：{exc}")
            return
        bx, by = parameters.betas
        segments = parameters.num_segments if parameters.mode == "smooth" else 1
        points = segments + 1 if parameters.mode == "smooth" else 1
        self._optics_summary.setText(
            f"βx = {bx:.8g} m，βy = {by:.8g} m\n"
            f"{segments} 段传输，{points} 个 Twiss 点；Δs = {parameters.circumference / segments:.8g} m\n"
            f"总相位推进：Qx = {parameters.qx:.8g}，Qy = {parameters.qy:.8g}，Qs = {parameters.muz:.8g}\n"
            "填写完成后点击“预览生成”。"
        )

    def preview_optics(self) -> None:
        if not self._optics_mode:
            return
        if self._json_dirty:
            QMessageBox.warning(self, "JSON 有未确认修改", "请先确认 JSON 修改，再重新打开生成器并预览。")
            return
        try:
            parameters = self._read_optics_parameters()
            sequence = self.data.get("Sequence")
            if not isinstance(sequence, dict):
                raise ValueError("Sequence 必须是对象。")
            items, proposed_names = parameters.generate()
        except (ValueError, OverflowError) as exc:
            self._optics_preview = None
            self.optics_insert_button.setEnabled(False)
            QMessageBox.warning(self, "生成参数无效", str(exc))
            return
        reserved = dict(sequence)
        names = []
        for name in proposed_names:
            unique = self._unique_sequence_name(name, reserved)
            reserved[unique] = None
            names.append(unique)
        renamed = sum(a != b for a, b in zip(proposed_names, names))
        overlaps = []
        transport_commands = {"twiss", "drift", "sbend", "quadrupole", "sextupole", "octupole", "multipole", "solenoid"}
        for name, item in sequence.items():
            if not isinstance(item, dict) or str(item.get("Command", "")).lower() not in transport_commands:
                continue
            try:
                end = float(item.get("S (m)", 0))
                start = float(item.get("S previous (m)", end))
            except (ValueError, TypeError):
                continue
            if min(start, end) <= parameters.circumference and max(start, end) >= 0:
                overlaps.append(name)
        lines = [
            f"位置范围：0 ～ {parameters.circumference:.8g} m；插入时同步全局周长。",
            f"将插入 {len(items)} 个 Twiss 点；表格显示前 {min(100, len(items))} 项。",
            f"名称冲突自动添加后缀：{renamed} 项；最终名称见表格。",
        ]
        if overlaps:
            lines.append(f"注意：此范围已有 {len(overlaps)} 个光学传输命令（{', '.join(overlaps[:5])}）。新增传输会叠加，请核对。")
        lines.append("预览未修改项目。确认后点击“插入到 Sequence”。")
        self._optics_details.setPlainText("\n".join(lines))
        self._optics_details.show()
        self._optics_table.setRowCount(min(100, len(items)))
        for row, (name, item) in enumerate(zip(names[:100], items[:100])):
            values = [name, f"{item.s_previous:.8g}", f"{item.s:.8g}",
                      f"{item.mu_x - item.mu_x_previous:.8g} / {item.mu_y - item.mu_y_previous:.8g}"]
            for column, value in enumerate(values):
                self._optics_table.setItem(row, column, QTableWidgetItem(value))
        self._optics_table.show()
        self._optics_preview = {
            "parameters": parameters, "items": items, "names": names,
            "sequence": deepcopy(sequence), "circumference": self.data.get("Circumference (m)"),
        }
        self._optics_summary.setText(self._optics_summary.text().replace(
            "填写完成后点击“预览生成”。", "预览已就绪，可核对下方结果并插入。"
        ))
        self.optics_insert_button.setEnabled(True)

    def insert_optics(self) -> None:
        preview = self._optics_preview
        if preview is None:
            return
        if (self._json_dirty or self.data.get("Sequence") != preview["sequence"]
                or self.data.get("Circumference (m)") != preview["circumference"]):
            self.optics_insert_button.setEnabled(False)
            self._optics_preview = None
            QMessageBox.warning(self, "预览已失效", "项目已发生变化，请重新预览后插入。")
            return
        sequence = self.data["Sequence"]
        sequence.update({name: item.model_dump(by_alias=True)
                         for name, item in zip(preview["names"], preview["items"])})
        self.data["Circumference (m)"] = preview["parameters"].circumference
        self.recipes.append({"id": uuid4().hex, "kind": "optics", "pass_version": __version__,
                             "parameters": preview["parameters"].model_dump(), "generated_names": preview["names"]})
        count = len(preview["items"])
        self._form_dirty = False
        self._data_dirty = True
        self._sync_editor()
        self._refresh_tree()
        self._select_sequence_item(preview["names"][0])
        self.file_changed.emit(self.path)
        self._set_sync_status(f"已插入 {count} 个 Twiss 点，尚未保存", "warning")

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
        if source_kind == "twiss":
            sampling = PropertyComboBox()
            sampling.setObjectName("choiceField")
            sampling.addItems(("保留原始位置", "等间距插值"))
            self._madx_fields["sampling"] = sampling
            self.form_layout.addRow("采样方式", sampling)
            segments = QLineEdit("100")
            segments.setObjectName("valueField")
            segments.setToolTip("正整数 N：一圈 N 段，包含 0 和 C 的 N＋1 个基础点；薄元件、场误差和光学跳变处另行拆分。")
            self._madx_fields["num_segments"] = segments
            self.form_layout.addRow("一圈分段数 N", segments)
            grid_summary = QLabel("五次 Hermite（相位约束）；Δs=C/N。预览后显示实际间距与点数。")
            grid_summary.setWordWrap(True)
            self._madx_fields["grid_summary"] = grid_summary
            self.form_layout.addRow(grid_summary)
            sampling.currentTextChanged.connect(self._madx_sampling_changed)
            segments.textChanged.connect(self._madx_sampling_changed)
            self._madx_sampling_changed()
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
        for field in self._madx_fields.values():
            if isinstance(field, QWidget):
                self._track_field(field)

    def _madx_sampling_changed(self) -> None:
        interpolation = self._madx_fields["sampling"].currentText() == "等间距插值"
        self._madx_fields["merge_drift"].setEnabled(not interpolation)
        self._madx_fields["num_segments"].setEnabled(interpolation)
        self._madx_fields["grid_summary"].setText(
            "五次 Hermite（相位约束）；Δs=C/N。预览后显示实际间距与点数。")
        self._madx_preview = self._madx_preview_signature = None

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
            if isinstance(field, str):
                return field
            return default

        def file_signature(key: str) -> tuple:
            path = Path(str(value(key)))
            try:
                stat = path.stat()
                return str(path), stat.st_size, stat.st_mtime_ns
            except OSError:
                return (str(path),)

        return (
            value("source_kind"), file_signature("Twiss TFS 文件"), file_signature("误差 TFS 文件"),
            value("merge_drift"), value("field_errors"), value("longitudinal_transfer"),
            value("muz"), value("dqx"), value("dqy"), value("patterns"),
            value("sampling"), value("num_segments"),
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
            from PASS.para.madx import read_madx_elements, read_madx_twiss, read_madx_twiss_interpolated

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
                interpolation = self._madx_fields["sampling"].currentText() == "等间距插值"
                options = {"is_merge_drift": merge_drift}
                reader = read_madx_twiss
                if interpolation:
                    try:
                        segments = int(self._madx_fields["num_segments"].text().strip())
                    except ValueError as exc:
                        raise ValueError("一圈分段数 N 必须是正整数。") from exc
                    if segments < 1:
                        raise ValueError("一圈分段数 N 必须是正整数。")
                    options = {"num_interp_slice": segments+1}
                    reader = read_madx_twiss_interpolated
                items, names, circumference = reader(
                    source,
                    error_file=error_file,
                    muz=float(self._madx_fields["muz"].text().strip()),
                    dqx=self._optional_float(self._madx_fields["dqx"], "DQx"),
                    dqy=self._optional_float(self._madx_fields["dqy"], "DQy"),
                    is_field_error=self._madx_fields["field_errors"].isChecked(),
                    insert_patterns=patterns or None,
                    longitudinal_transfer=self._madx_fields["longitudinal_transfer"].currentText(),
                    **options,
                )
                if interpolation:
                    twiss = [item for item in items if item.command == "Twiss"]
                    positions = len({item.s for item in twiss})
                    self._madx_fields["grid_summary"].setText(
                        f"五次 Hermite（相位约束）\nΔs = {circumference/segments:.10g} m；"
                        f"基础点 {segments+1}，附加位置 {positions-segments-1}，Twiss command {len(twiss)}\n"
                        f"累计相位差：Qx = {twiss[-1].mu_x-twiss[0].mu_x_previous:.12g}，"
                        f"Qy = {twiss[-1].mu_y-twiss[0].mu_y_previous:.12g}")
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
        sampling = self._madx_fields.get("sampling")
        interpolation = isinstance(sampling, QComboBox) and sampling.currentText() == "等间距插值"
        sampling_summary = (
            "采样方式：等间距插值（替换原始 Twiss 点）\n"
            + self._madx_fields["grid_summary"].text() + "\n"
            if interpolation else f"合并连续 Drift：{'是' if merge else '否'}\n")
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
            f"{sampling_summary}"
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
        generated_names = []
        for name, item in zip(names, items):
            unique_name = self._unique_sequence_name(str(name), sequence)
            sequence[unique_name] = item.model_dump(by_alias=True)
            first_name = first_name or unique_name
            generated_names.append(unique_name)
        if self._madx_fields["update_circumference"].isChecked():
            self.data["Circumference (m)"] = circumference
        parameters, sources = {}, {}
        for key, field in self._madx_fields.items():
            if isinstance(field, QLineEdit):
                value = field.text()
                if key in ("Twiss TFS 文件", "误差 TFS 文件"):
                    if value:
                        sources[key] = str(Path(value).resolve())
                    continue
                parameters[key] = value
            elif isinstance(field, QComboBox):
                parameters[key] = field.currentText()
            elif isinstance(field, QCheckBox):
                parameters[key] = field.isChecked()
            elif isinstance(field, str):
                parameters[key] = field
        self.recipes.append({"id": uuid4().hex, "kind": "madx", "pass_version": __version__,
                             "parameters": parameters, "source_files": sources, "generated_names": generated_names})
        self._form_dirty = False
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
                configurations={"default": SpaceChargeResourceConfig(deposition_method="CIC")},
            ).model_dump(by_alias=True)
        self._sync_editor()
        self._data_dirty = True
        self._refresh_tree()
        self._select_top_level_item("__root__", "Space charge")

    @staticmethod
    def _default_space_charge_resource() -> dict:
        from PASS.para.schema.space_charge import SpaceChargeResourceConfig

        return SpaceChargeResourceConfig(deposition_method="CIC").model_dump(by_alias=True)

    def _populate_space_charge_configuration(self, active_name: str | None = None) -> None:
        """Render the top-level block and one named PIC resource as typed fields."""
        block = self.data.get("Space charge")
        if not isinstance(block, dict):
            return
        self._clear_form()
        self._selected_mapping = block
        self._selected_path = ("__root__", "Space charge")
        self.form_title.setText("空间电荷 · 计算配置")
        self.form_hint.setText(
            "共享切片集、网格和场模型，并设置覆盖检查；显式计算点独立设置孔径，元件内部 SC 使用元件孔径。"
        )

        enabled = QCheckBox("启用空间电荷模块")
        enabled.setObjectName("booleanField")
        enabled.setChecked(block.get("Enabled", False) is True)
        enabled.setToolTip("关闭时所有 SpaceCharge 配置和计算点均被忽略。")
        self._track_field(enabled)
        self._space_charge_enabled_field = enabled
        self.form_layout.addRow(self._field_label("Enabled", False), enabled)
        for key, default in (("Coverage check", "warn"), ("Coverage mode", "full-ring"),
                             ("Expected SC length (m)", None)):
            value = block.get(key, default)
            field = self._make_field(key, value)
            self._space_charge_coverage_fields[key] = field
            self.form_layout.addRow(self._field_label(key, value), field)

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
            self._space_charge_resource_form = resource_form
            resource_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            resource_form.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
            resource_form.setHorizontalSpacing(8)
            resource_form.setVerticalSpacing(6)
            self._space_charge_extent_mode = PropertyComboBox()
            self._space_charge_extent_mode.addItems(["全宽", "半宽"])
            self._space_charge_extent_mode.setCurrentText(
                "半宽" if resource.get("Grid Half Width X (m)") is not None else "全宽")
            self._track_field(self._space_charge_extent_mode)
            resource_form.addRow("网格范围输入", self._space_charge_extent_mode)
            for key, default_value in self._default_space_charge_resource().items():
                value = resource.get(key, default_value)
                if key == "Particle Deposition Method":
                    value = value or "CIC"
                field = self._make_field(key, value)
                self._space_charge_fields[key] = field
                resource_form.addRow(self._field_label(key, value), field)
            selector_layout.addLayout(resource_form)
            self._space_charge_method_hint = QLabel()
            self._space_charge_method_hint.setWordWrap(True)
            selector_layout.addWidget(self._space_charge_method_hint)
            self._space_charge_fields["Method"].currentTextChanged.connect(self._update_space_charge_fields)
            self._space_charge_fields["Solver"].currentTextChanged.connect(self._update_space_charge_fields)
            self._space_charge_extent_mode.currentTextChanged.connect(self._change_space_charge_extent_mode)
            self._update_space_charge_fields()
        else:
            empty = QLabel("尚无资源配置。点击“添加”创建一个配置后，SpaceCharge 计算点才能引用它。")
            empty.setObjectName("muted")
            empty.setWordWrap(True)
            selector_layout.addWidget(empty)
        self.form_layout.addRow(selector_box)
        self.form_apply.setEnabled(True)

    def _change_space_charge_extent_mode(self) -> None:
        """Convert the current extent when switching its input convention."""
        half = self._space_charge_extent_mode.currentText() == "半宽"
        target_prefix = "Grid Half Width" if half else "Grid Width"
        source_prefix = "Grid Width" if half else "Grid Half Width"
        for axis in ("X", "Y"):
            target = self._space_charge_fields[f"{target_prefix} {axis} (m)"]
            source = self._space_charge_fields[f"{source_prefix} {axis} (m)"]
            try:
                value = float(source.text())
                target.setText(str(value / 2 if half else value * 2))
            except ValueError:
                target.clear()
        self._update_space_charge_fields()

    def _update_space_charge_fields(self) -> None:
        """Limit solver choices and expose only inputs used by the chosen method."""
        method = self._space_charge_fields["Method"].currentText()
        solver_field = self._space_charge_fields["Solver"]
        choices = (
            ("fft_free_space", "fd_dirichlet", "dst_dirichlet") if method == "pic" else
            ("gaussian_round_free_space", "gaussian_ellipse_free_space",
             "uniform_round_free_space", "uniform_ellipse_free_space")
        )
        solver = solver_field.currentText()
        if solver not in choices:
            solver = "fd_dirichlet" if method == "pic" else choices[0]
        solver_field.blockSignals(True)
        solver_field.clear()
        solver_field.addItems(choices)
        solver_field.setCurrentText(solver)
        solver_field.blockSignals(False)
        prefix = "Grid Half Width" if self._space_charge_extent_mode.currentText() == "半宽" else "Grid Width"
        other = "Grid Width" if prefix == "Grid Half Width" else "Grid Half Width"
        for axis in ("X", "Y"):
            target = self._space_charge_fields[f"{prefix} {axis} (m)"]
            if not target.text().strip():
                try:
                    value = float(self._space_charge_fields[f"{other} {axis} (m)"].text())
                    target.setText(str(value / 2 if prefix == "Grid Half Width" else value * 2))
                except ValueError:
                    pass
        active = {"Slice set", "Nx", "Ny", f"{prefix} X (m)", f"{prefix} Y (m)", "Method", "Solver"}
        if method == "pic":
            active.add("Particle Deposition Method")
            hint = "PIC 从粒子沉积电荷并求场，沉积方式默认 CIC。导体边界由各 SpaceCharge command 的 Aperture type/value 定义；DST 要求与网格重合的矩形。网格仅接受全宽或半宽，步长由节点数计算。"
        elif method == "frozen":
            active.update({"Center X (m)", "Center Y (m)"})
            active.update({
                "gaussian_round_free_space": {"Sigma (m)"},
                "gaussian_ellipse_free_space": {"Sigma X (m)", "Sigma Y (m)", "Angle (rad)"},
                "uniform_round_free_space": {"Radius (m)"},
                "uniform_ellipse_free_space": {"Semi-axis A (m)", "Semi-axis B (m)", "Angle (rad)"},
            }[solver])
            hint = "frozen 使用固定的源分布中心和尺寸；对应尺寸必须填写，中心和角度留空取零。网格用于诊断采样及 command 缺省矩形孔径，公式场为自由空间。"
        else:
            hint = "quasi-frozen 每次按切片粒子重新计算中心、尺寸和方向，无需输入固定分布参数。网格用于诊断采样及 command 缺省矩形孔径，公式场为自由空间。"
        self._space_charge_active_fields = active
        for key, field in self._space_charge_fields.items():
            self._space_charge_resource_form.setRowVisible(field, True)
            field.setEnabled(key in active)
            field.setToolTip(self._field_help(key, None) + ("" if key in active else "\n当前方法、求解器或网格输入方式不使用此项。"))
        self._space_charge_method_hint.setText(hint)

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
                if key in self._space_charge_active_fields else None
                for key, field in self._space_charge_fields.items()
            }
            canonical_resource = SpaceChargeResourceConfig.model_validate(resource).model_dump(by_alias=True)
            candidate_configurations.pop(old_name, None)
            candidate_configurations[new_name] = canonical_resource
        enabled = self._space_charge_enabled_field.isChecked()
        coverage = {key: self._read_field_value(key, widget, block.get(key))
                    for key, widget in self._space_charge_coverage_fields.items()}
        canonical_coverage = SpaceChargeConfig.model_validate({"Enabled": enabled, **coverage}).model_dump(by_alias=True)
        candidate = {"Enabled": enabled, "Configurations": candidate_configurations,
                     **{key: canonical_coverage[key] for key in coverage}}
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
                    internal = item.get("Space charge") if isinstance(item, dict) else None
                    if isinstance(internal, dict) and internal.get("Configuration") == old_name:
                        internal["Configuration"] = new_name
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
                    "请先新增一个计算配置，或删除这些空间电荷计算点。",
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

    def collect_validation_issues(self, data: dict, base_dir: Path) -> list[str]:
        from PASS.validation import validate_input
        return [str(issue) for issue in validate_input(data, base_dir).errors]

    def _apply_validation_report(self, report) -> list[str]:
        self._validation_report = report
        self._validation_issues = [str(issue) for issue in report.errors]
        label = "全面检测" if report.full else "参数预检"
        if report.errors or report.warnings:
            state = "error" if report.errors else "warning"
            self._set_validation_status(
                f"{label}：{len(report.errors)} 错误 · {len(report.warnings)} 警告", state, report.text())
        else:
            self._set_validation_status(
                "全面检测通过" if report.full else "参数预检通过", "ok",
                report.text() + ("" if report.full else "\n点击校验可进一步检查全部输入文件内容。"))
        return self._validation_issues

    def _validate_configuration(self, *, full=False) -> list[str]:
        from PASS.validation import validate_input
        return self._apply_validation_report(validate_input(self.data, self.base_dir, check_files=full))

    def _set_validation_status(self, text: str, state: str, detail: str) -> None:
        """Show a compact validation result without consuming overview space."""
        self.validation_label.setText(text)
        self.validation_label.setToolTip(detail)
        self.validation_label.setProperty("state", state)
        self.validation_label.style().unpolish(self.validation_label)
        self.validation_label.style().polish(self.validation_label)

    def _show_validation_issues(self, *, full=False) -> None:
        from PASS.gui.validation import ValidationDialog
        dialog = ValidationDialog(self, None if full else getattr(self, "_validation_report", None))
        dialog.navigate.connect(lambda issue: self._navigate_to_validation_issue(issue, dialog))
        if full:
            dialog.start(self.data, self.base_dir)
        dialog.exec()
        if dialog.report is not None:
            self._apply_validation_report(dialog.report)

    def _navigate_to_validation_issue(self, issue, dialog: QDialog) -> None:
        if not self._confirm_form_navigation():
            return
        path = issue.path
        if len(path) >= 2 and path[0] == "Sequence":
            if path[1] in self.data.get("Sequence", {}):
                if len(path) >= 3 and re.fullmatch(r"bunch\d+", str(path[2])):
                    self._active_bunch_key = path[2]
                self._select_sequence_item(path[1])
                if len(path) >= 3:
                    if len(path) >= 4 and re.fullmatch(r"bunch\d+", str(path[2])):
                        child = path[4] if len(path) >= 5 and isinstance(path[4], str) else None
                        field = self._bunch_fields.get((path[3], child))
                    else:
                        field = self._form_fields.get(path[2])
                    if field is not None:
                        field.setFocus(Qt.OtherFocusReason)
                        indices = [part for part in path[3:] if isinstance(part, int)]
                        table = field.findChild(QTableWidget)
                        if table is not None and indices and indices[0] < table.rowCount():
                            column = min(indices[1] if len(indices) > 1 else 0, table.columnCount() - 1)
                            table.setCurrentCell(indices[0], column)
                            table.scrollTo(table.model().index(indices[0], column))
            elif path[1] == "injection":
                self.select_command("Injection")
        elif path and path[0] == "Space charge":
            name = path[2] if len(path) >= 3 and path[1] == "Configurations" else None
            self._populate_space_charge_configuration(name)
        else:
            self.configure_global()
            if path and path[0] in self._form_fields:
                self._form_fields[path[0]].setFocus(Qt.OtherFocusReason)
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
        from PASS.validation.rules import Validator
        from PASS.validation import ValidationReport
        from PASS.para.schema.main import MainConfig
        data = MainConfig().model_dump(by_alias=True)
        validator = Validator(data, Path.cwd(), ValidationReport(full=False), False)
        validator.globals()
        validator.command(name, item)
        return str(validator.report.errors[0]) if validator.report.errors else None

    def _refresh_sequence_table(self) -> None:
        from PASS.commands import command_priority
        from PASS.utils.constants import const

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
                if not math.isfinite(position):
                    position = 0.0
                rows.append((position, str(name), command, value))
        # Match the engine's stable ordering, including positions in the same
        # tolerance bin; names must not change the displayed execution order.
        rows.sort(key=lambda row: (
            round(row[0] / const.eps) if const.eps > 0 and abs(row[0]) < 1e290 else row[0],
            command_priority(row[2]),
        ))
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
        self.sequence_count.setText(f"{len(rows):,} 项")
        for row_index, (position, name, command, value) in enumerate(rows):
            self.sequence_table.setItem(row_index, 0, QTableWidgetItem(name))
            self.sequence_table.setItem(row_index, 1, QTableWidgetItem(command))
            self.sequence_table.setItem(row_index, 2, QTableWidgetItem(f"{position:.6g}"))
            enabled = True
            if command == "RFCavity":
                enabled = value.get("Is enabled", True)
            elif command == "Exciter":
                enabled = value.get("Enable", True)
            elif command == "SpaceCharge":
                block = self.data.get("Space charge", {})
                enabled = block.get("Enabled", False) if isinstance(block, dict) else False
            status = "启用" if enabled else "禁用"
            self.sequence_table.setItem(row_index, 3, QTableWidgetItem(status))
            for column, field in [(4, "Configuration"), (5, "SC length (m)"), (6, "Aperture type"), (7, "Slice set")]:
                cell = value.get(field, value.get("Length (m)", "") if column == 5 else "")
                self.sequence_table.setItem(row_index, column, QTableWidgetItem(str(cell)))
            self.sequence_table.item(row_index, 2).setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
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
            section_layout.setHorizontalSpacing(8)
            section_layout.setVerticalSpacing(6)
            has_fields = False
            for key in keys:
                if key not in self.data:
                    continue
                value = self.data[key]
                field = self._make_field(key, value)
                self._add_property_row(section_layout, self._field_label(key, value), field)
                self._form_fields[key] = field
                shown.add(key)
                has_fields = True
            if has_fields:
                self.form_layout.addRow(box)
        for key, value in self.data.items():
            if key in {"Sequence", "Timing", "Space charge"} or key in shown:
                continue
            field = self._make_field(key, value)
            self._add_property_row(self.form_layout, self._field_label(key, value), field)
            self._form_fields[key] = field
        self._connect_structured_fields()
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
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
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
        field.setMinimumWidth(100)
        field.setToolTip("选择顶层 Space charge.Configurations 中的命名配置。")
        self._track_field(field)
        return field

    @staticmethod
    def _property_sections(values: dict) -> list[tuple[str | None, list[str]]]:
        """Present command parameters by purpose, independently of schema inheritance."""
        command = values.get("Command")
        special = {
            "Exciter": [
                ("激励频率", ("Excite tune", "Sweep tune", "Central frequency (Hz)", "Sweep width (Hz)", "Period (s)", "FM dual frequency (Hz)")),
                ("幅度调制", ("AM t ext (s)", "AM r0 (m)", "AM delta0", "AM k const")),
            ],
            "RFCavity": [("射频波形", ("Components",))],
            "Slicer": [("切片范围", ("Z range mode", "Explicit"))],
            "PhaseAdvanceMonitor": [
                ("参考光学", ("Alpha x", "Alpha y", "Beta x (m)", "Beta y (m)", "Dx (m)", "Dpx", "X CO (m)", "PX CO", "Y CO (m)", "PY CO")),
                ("分析范围", ("Turn ranges", "Min action")),
            ],
        }
        definitions = [
            *special.get(command, []),
            ("场误差", ("Is field error", "Field error KNL", "Field error KSL")),
            ("孔径", ("Aperture type", "Aperture value", "Dp aperture")),
            ("Ramping", tuple(key for key in values if "ramping" in key.casefold())),
            ("诊断输出", ("Save field", "Save potential", "Save density", "Save turns")),
            ("内部空间电荷", ("Space charge",)),
        ]
        sections = [(title, [key for key in keys if key in values]) for title, keys in definitions]
        grouped = {key for _, keys in sections for key in keys}
        leading = ("S (m)", "Length (m)", "Enable", "Is enabled")
        tracking = ("Model", "Integrator", "Num slices")
        basic = [key for key in leading if key in values]
        basic += [key for key in values if key not in grouped and key not in (*leading, *tracking, "Command")]
        basic += [key for key in tracking if key in values]
        return [(None, basic), *((title, keys) for title, keys in sections if keys)]

    def _show_command_badge(self, command: str) -> None:
        self.form_command.setText(command)
        self.form_command.setFixedWidth(max(60, self.form_command.fontMetrics().horizontalAdvance(command) + 16))
        self.form_command.show()
        self._form_fields["Command"] = self.form_command

    def _property_section(self, title: str) -> QFormLayout:
        box = QGroupBox(title)
        box.setObjectName("propertySection")
        layout = QFormLayout(box)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        self.form_layout.addRow(box)
        return layout

    def _populate_form(self, title: str, target: dict, pending: bool = False, name_value: str | None = None) -> None:
        if target.get("Command") == "Injection":
            self._populate_injection_form(title, target, pending, name_value)
            return
        self._clear_form()
        self._selected_mapping = target
        self.form_title.setText(title)
        self.form_hint.setText("全部字段展开。数组按用途填写数值、范围或表格，无需输入括号和逗号。")
        if target.get("Command") == "SpaceCharge":
            self.form_hint.setText("default 孔径使用配置网格矩形；修改后点击应用。")
        if pending or name_value is not None:
            default_name = "injection" if target.get("Command") == "Injection" else f"{str(target.get('Command', 'command')).lower()}_1"
            self._name_field = QLineEdit(name_value or default_name)
            self._name_field.setToolTip("Sequence 中的唯一名称")
            self.form_layout.addRow(QLabel("名称"), self._name_field)
            self._track_field(self._name_field)
        twiss_layouts = {}
        if target.get("Command") == "Twiss":
            self.form_hint.setText("使用起点与终点光学参数构造一段传输；Mu 为累计相位，单位为周（2π）。")
            for group in ("起点光学参数", "终点光学参数", "传输设置"):
                box = QGroupBox(group)
                layout = QFormLayout(box)
                layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
                layout.setSizeConstraint(QLayout.SetMinimumSize)
                twiss_layouts[group] = layout
                self.form_layout.addRow(box)
        try:
            complete = self._command_template(str(target.get("Command", "")))
        except (KeyError, ValueError, TypeError):
            complete = {}
        complete.update(target)
        self._field_defaults = complete
        self._field_context = complete
        if "Command" in complete:
            self._show_command_badge(str(complete["Command"]))
        sections = [(None, [key for key in complete if key != "Command"])] if twiss_layouts else self._property_sections(complete)
        for section, keys in sections:
            section_layout = self._property_section(section) if section else self.form_layout
            for key in keys:
                value = complete[key]
                if target.get("Command") == "SpaceCharge" and key == "Configuration":
                    field = self._make_space_charge_reference_field(str(value))
                else:
                    field = self._make_field(str(key), value)
                label = self._field_label(str(key), value)
                layout = section_layout
                if twiss_layouts:
                    if "previous" in key.casefold():
                        layout = twiss_layouts["起点光学参数"]
                        label.setText(re.sub(r" previous", "", str(key), flags=re.IGNORECASE))
                    elif key in {"S (m)", "Alpha x", "Alpha y", "Beta x (m)", "Beta y (m)", "Mu x", "Mu y", "Mu z", "Dx (m)", "Dpx"}:
                        layout = twiss_layouts["终点光学参数"]
                    else:
                        layout = twiss_layouts["传输设置"]
                if key == "Space charge" and section == "内部空间电荷":
                    layout.addRow(field)
                else:
                    self._add_property_row(layout, label, field)
                self._form_fields[key] = field
        self._connect_structured_fields()
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
            self._track_field(self._name_field)
        defaults = self._command_template("Injection")
        complete = {key: value for key, value in defaults.items() if not re.fullmatch(r"bunch\d+", str(key))}
        complete.update(target)
        self._field_defaults = complete
        for key, value in complete.items():
            if re.fullmatch(r"bunch\d+", str(key)):
                continue
            if key == "Command":
                self._show_command_badge(str(value))
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
            fields.setHorizontalSpacing(8)
            fields.setVerticalSpacing(6)
            bunch = deepcopy(defaults["bunch0"])
            bunch.update(target[self._active_bunch_key])
            for offset in ("Offset x", "Offset y"):
                if isinstance(bunch.get(offset), dict):
                    bunch[offset] = defaults["bunch0"][offset] | bunch[offset]
            for key, value in bunch.items():
                if key in ("Offset x", "Offset y") and isinstance(value, dict):
                    offset_box = QGroupBox(str(key))
                    offset_layout = QFormLayout(offset_box)
                    offset_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
                    offset_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
                    for child_key, child_value in value.items():
                        field = self._make_field(str(child_key), child_value)
                        self._bunch_fields[(str(key), str(child_key))] = field
                        self._add_property_row(offset_layout, self._field_label(str(child_key), child_value), field)
                    fields.addRow(offset_box)
                    continue
                field = self._make_field(str(key), value)
                self._bunch_fields[(str(key), None)] = field
                self._add_property_row(fields, self._field_label(str(key), value), field)
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
            emit_x=1e-6,
            emit_y=1e-6,
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
        if not self._commit_bunch_fields(target):
            return
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
        if not self._commit_bunch_fields(target):
            return
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
        if not self._commit_bunch_fields(target):
            return
        target.pop(key, None)
        self._normalize_injection(target)
        self._data_dirty = True
        self._rebuild_injection_form("bunch0")

    @staticmethod
    def _add_property_row(layout, label, field):
        if isinstance(field, StructuredField):
            # Tables use the full property-pane width, with a permanent label.
            label.setMaximumWidth(16777215)
            layout.addRow(label)
            layout.addRow(field)
        else:
            layout.addRow(label, field)

    def _connect_structured_fields(self):
        fields = self._form_fields
        aperture = fields.get("Aperture value")
        kind = fields.get("Aperture type")
        if isinstance(aperture, ApertureEditor) and isinstance(kind, QComboBox):
            kind.currentTextChanged.connect(aperture.set_kind)
        explicit = fields.get("Explicit")
        mode = fields.get("Z range mode")
        if isinstance(explicit, RangeEditor) and isinstance(mode, QComboBox):
            explicit.blockSignals(True)
            explicit.set_active(mode.currentText() == "explicit")
            explicit.blockSignals(False)
            mode.currentTextChanged.connect(lambda text: explicit.set_active(text == "explicit"))
        devices = fields.get("Device Id")
        count = fields.get("Number of GPU devices")
        backend = fields.get("Backend (gpu/cpu)")
        if isinstance(devices, DevicesEditor) and isinstance(count, QLineEdit):
            count.setReadOnly(True)
            count.setToolTip("由 GPU 设备列表自动计算。")
            def sync_devices():
                count.setText(str(devices.table.rowCount()))
            devices.changed.connect(sync_devices)
            if isinstance(backend, QComboBox):
                devices.setEnabled(backend.currentText() == "gpu")
                backend.currentTextChanged.connect(lambda text: devices.setEnabled(text == "gpu"))

    def _commit_bunch_fields(self, target):
        try:
            self._write_bunch_values(target)
            return True
        except ValueError as exc:
            QMessageBox.warning(self, "粒子参数无效", str(exc))
            return False

    def _field_label(self, key: str, value: object) -> QLabel:
        short = {"S (m)": "位置 s / m", "SC length (m)": "作用长度 / m", "SC start (m)": "区间起点 / m",
                 "Configuration": "计算配置", "Aperture type": "孔径类型", "Aperture value": "孔径参数 / m",
                 "Save field": "保存电场", "Save potential": "保存电势", "Save density": "保存电荷密度", "Save turns": "保存圈数",
                 "Turn ranges": "分析圈数范围", "Insert Particle Coordinate": "手动插入粒子", "Dp aperture": "动量接受范围",
                 "Device Id": "GPU 设备列表", "Explicit": "显式切片范围", "Space charge": "内部空间电荷"}
        label = QLabel(short.get(key, key))
        label.setMinimumWidth(96)
        label.setMaximumWidth(126)
        label.setWordWrap(True)
        label.setAttribute(Qt.WA_LayoutUsesWidgetRect)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        help_text = self._field_help(key, value)
        label.setToolTip(key + "\n" + help_text)
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
            return "按标签填写数值或表格；应用时检查范围和参数关系。"
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
        structured = None
        total_turns = int(self.data.get("Number of turns", 100))
        if key == "Save turns":
            structured = TurnsEditor(value, total_turns)
        elif key == "Turn ranges":
            structured = TurnsEditor(value, total_turns, analysis=True)
        elif key == "Insert Particle Coordinate":
            structured = ParticleEditor(value)
        elif key in {"KiL", "KiSL", "Field error KNL", "Field error KSL"}:
            structured = CoefficientsEditor(value, key)
        elif key == "Device Id":
            structured = DevicesEditor(value)
        elif key == "Aperture value":
            context = getattr(self, "_field_context", {})
            structured = ApertureEditor(context.get("Aperture type", "off"), value)
        elif key in {"Explicit", "Dp aperture"}:
            structured = RangeEditor(value, explicit=key == "Explicit")
        elif key == "Space charge":
            block = self.data.get("Space charge", {})
            names = list(block.get("Configurations", {})) if isinstance(block, dict) else []
            structured = InternalSpaceChargeEditor(value, names, total_turns)
        elif (key == "Components" and isinstance(value, list)
              and getattr(self, "_field_context", {}).get("Command") == "RFCavity"):
            from PASS.para.schema.rf import RFComponent
            default = RFComponent(harmonic=1).model_dump(by_alias=True)
            structured = ObjectListEditor(value, self._make_field, self._read_field_value, default)
        elif isinstance(value, list):
            structured = (NumericTable([Column(f"第 {i + 1} 列") for i in range(len(value[0]))], value)
                          if value and isinstance(value[0], list) else ListEditor(value))
        elif isinstance(value, dict):
            structured = ObjectEditor(value, self._make_field, self._read_field_value)
        if structured is not None:
            self._track_field(structured)
            return structured
        if key == "Command":
            field = QLineEdit(str(value))
            field.setObjectName("readonlyField")
            field.setReadOnly(True)
            field.setMinimumWidth(100)
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
            field.setMinimumWidth(100)
            field.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            field.setMinimumContentsLength(8)
            field.setToolTip(self._field_help(key, value))
            self._track_field(field)
            return field
        field = QLineEdit("" if value is None else str(value))
        field.setObjectName("valueField")
        field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        field.setMinimumWidth(100)
        if key.casefold() in FILE_FIELDS:
            choose = field.addAction(icon("folder", self.palette().color(QPalette.Text).name()), QLineEdit.TrailingPosition)
            choose.setProperty("themeIcon", "folder")
            choose.setToolTip("选择输入文件")
            def browse():
                path, _ = QFileDialog.getOpenFileName(self, key, str(self.base_dir), "All files (*)")
                if path:
                    field.setText(path)
            choose.triggered.connect(browse)
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
        field.setAttribute(Qt.WA_LayoutUsesWidgetRect)
        if isinstance(field, StructuredField):
            field.changed.connect(self._mark_form_dirty)
        elif isinstance(field, QLineEdit):
            field.textChanged.connect(lambda: self._mark_form_dirty())
        elif isinstance(field, QComboBox):
            field.currentTextChanged.connect(lambda: self._mark_form_dirty())
        elif isinstance(field, QCheckBox):
            field.stateChanged.connect(lambda: self._mark_form_dirty())
        elif isinstance(field, QPlainTextEdit):
            field.textChanged.connect(self._mark_form_dirty)

    def _mark_form_dirty(self) -> None:
        if self._selected_mapping is None and self._optics_mode is None and not self._madx_fields:
            return
        self._form_dirty = True
        self._set_sync_status("表单有未确认修改", "warning")

    @staticmethod
    def _read_field_value(key: str, field: QWidget, old_value: object) -> object:
        if isinstance(field, StructuredField):
            try:
                return field.get_value()
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{key}：{exc}") from exc
        if isinstance(field, QCheckBox):
            return field.isChecked()
        if isinstance(field, QComboBox):
            return None if old_value is None and not field.currentText() else field.currentText()
        if not isinstance(field, QLineEdit):
            raise ValueError(f"{key} 使用了未知的编辑控件。")
        text = field.text().strip()
        if key in {"Harmonic", "Frequency (Hz)", "Time (s)"} and (not text or text.casefold() == "null"):
            return None
        if old_value is None:
            if not text or text.casefold() == "null":
                return None
            if key.casefold() not in FILE_FIELDS:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass
            return text
        if isinstance(old_value, int) and not isinstance(old_value, bool):
            return int(text)
        if isinstance(old_value, float):
            return float(text)
        return text

    def _write_form_values(self, target: dict) -> None:
        for key, field in self._form_fields.items():
            target[key] = self._read_field_value(str(key), field, target.get(key, self._field_defaults.get(key)))
        if self._timing_fields:
            timing = target.setdefault("Timing", {})
            if not isinstance(timing, dict):
                timing = target["Timing"] = {}
            for key, field in self._timing_fields.items():
                timing[key] = self._read_field_value(key, field, timing.get(key))
        if target.get("Command") == "Injection":
            self._write_bunch_values(target)
            self._normalize_injection(target)
        if "Device Id" in self._form_fields:
            devices = target["Device Id"]
            if target.get("Backend (gpu/cpu)") == "gpu" and not devices:
                raise ValueError("GPU 后端至少需要一个设备编号")
            target["Number of GPU devices"] = max(1, len(devices))

    def _write_bunch_values(self, target: dict | None) -> None:
        if not isinstance(target, dict) or not self._active_bunch_key:
            return
        original = target.get(self._active_bunch_key)
        if not isinstance(original, dict):
            return
        bunch = deepcopy(original)
        for (key, child_key), field in self._bunch_fields.items():
            if child_key is None:
                bunch[key] = self._read_field_value(key, field, bunch.get(key))
                continue
            nested = bunch.setdefault(key, {})
            if not isinstance(nested, dict):
                nested = bunch[key] = {}
            nested[child_key] = self._read_field_value(child_key, field, nested.get(child_key))
        count = len(bunch.get("Insert Particle Coordinate", []))
        # Manual coordinates replace particles in the first injection block.
        turns = bunch.get("Total Injection Turns", 1)
        maximum = bunch.get("Number of Macro Particles", 0)
        interval = max(1, int(bunch.get("Injection Interval", 1)))
        events = max(1, (turns + interval - 1) // interval)
        first_injection = maximum // events + maximum % events
        if count > first_injection:
            raise ValueError(f"手动粒子为 {count} 个，不能超过首次注入的宏粒子数 {first_injection}")
        original.clear()
        original.update(bunch)

    def _update_action_visibility(self, pending: bool | None = None) -> None:
        if pending is None:
            pending = bool(self._pending_command)
        is_sequence_item = bool(self._selected_path and self._selected_path[0] == "Sequence" and self._selected_path[1])
        self.duplicate_button.setVisible(is_sequence_item and not pending)
        self.delete_button.setVisible(is_sequence_item and not pending)

    def apply_form(self) -> None:
        if self._json_dirty:
            QMessageBox.warning(self, "JSON 源码有修改", "请先应用 JSON 修改，或撤销 JSON 编辑，再应用属性。")
            return
        if self._selected_mapping is None:
            return
        previous_data = deepcopy(self.data)
        previous_path = self._selected_path
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
            # A bad later field must not leave earlier fields or a rename applied.
            self.data.clear()
            self.data.update(previous_data)
            self._selected_path = previous_path
            if previous_path and previous_path[0] == "Sequence":
                self._selected_mapping = self.data["Sequence"][previous_path[1]]
            elif previous_path and previous_path[1]:
                self._selected_mapping = self.data.get(previous_path[1])
            else:
                self._selected_mapping = self.data
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
        self.form_command.hide()
        self.form_command.clear()
        self._bunch_fields = {}
        self._name_field = None
        self._bunch_selector = None
        self._active_bunch_key = None
        self._injection_pending = False
        self._madx_fields = {}
        self._madx_preview = None
        self._madx_preview_signature = None
        self._optics_mode = None
        self._optics_fields = {}
        self._optics_preview = None
        self._timing_fields = {}
        self._space_charge_fields = {}
        self._space_charge_selector = None
        self._space_charge_name_field = None
        self._space_charge_enabled_field = None
        self._space_charge_coverage_fields = {}
        self._active_space_charge_configuration = None
        self._pending_command = None
        self.form_title.setText("参数配置")
        self.form_hint.setText(hint)
        self.form_apply.setEnabled(False)
        self.form_apply.show()
        self.optics_preview_button.hide()
        self.optics_insert_button.hide()
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
        self._record_history()
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
        self.controller = None
        self.process: QProcess | None = None
        self.started_at = 0.0
        self._stopped = False
        self._input_project = None
        self._refreshing = False
        self._log_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._progress_tail = ""
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        header = QHBoxLayout()
        header.addWidget(QLabel("运行"))
        self.run_path = QLabel("使用当前输入")
        self.run_path.setObjectName("muted")
        header.addWidget(self.run_path, 1)
        self.start_button = button("开始运行", "primary")
        self.start_button.clicked.connect(self.start_run)
        self.stop_button = button("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_run)
        header.addWidget(self.start_button)
        header.addWidget(self.stop_button)
        root.addLayout(header)
        inputs = QHBoxLayout()
        inputs.addWidget(QLabel("Beam 0"))
        self.beam0 = PropertyComboBox()
        self.beam1 = PropertyComboBox()
        inputs.addWidget(self.beam0, 1)
        inputs.addWidget(QLabel("Beam 1（可选）"))
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
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.log.setObjectName("codeEditor")
        self.log.setFont(code_font())
        root.addWidget(self.log, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_elapsed)
        self.refresh_inputs()

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
        return {"beam0": self.beam0.currentData() or "", "beam1": self.beam1.currentData() or "",
                "output_directory": self.output_directory.text() or "output"}

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
        from PASS.gui.project import Project, atomic_write, json_bytes
        from uuid import uuid4
        owner = self.controller
        if self.process and self.process.state() != QProcess.NotRunning:
            return
        if owner and not owner._commit_current():
            return
        if not owner and not self.config.commit_pending():
            return
        temporary = None
        try:
            project = owner.project if owner else None
            if project:
                ids = self.selected_input_ids()
                base = project.path.parent if project.path else Path.cwd()
                from PASS.validation.rules import validate_documents
                report = validate_documents([(project.configs[cid].name, project.configs[cid].data,
                                               project.config_base) for cid in ids])
                if not report.ok:
                    raise ValueError(report.text())
            else:
                issues = self.config._validate_configuration(full=True)
                if issues:
                    raise ValueError("\n".join(issues))
                temporary = project = Project()
                ids = [project.add_config("beam0", self.config.data, self.config.base_dir)]
                base = self.config.base_dir
            output = Path(self.output_directory.text() or "output").expanduser()
            if not output.is_absolute():
                output = base / output
            output = output.resolve()
            snapshot = output / "input_snapshots" / uuid4().hex
            paths = project.materialize(ids, snapshot, output)
            atomic_write(snapshot / "run.json", json_bytes({"pass_version": __version__, "input_ids": ids,
                         "inputs": [p.name for p in paths], "output_directory": str(output)}))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法运行", str(exc))
            return
        finally:
            if temporary:
                temporary.close()
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(["-u", "-X", "utf8", "-m", "PASS.gui.runner", *map(str, paths)])
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[2]))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self.log.clear()
        self._log_decoder.reset()
        self._progress_tail = ""
        self.log.appendPlainText(f"输入快照：{snapshot}\n输出目录：{output}\n\n")
        self.run_path.setText(" + ".join(project.configs[cid].name + ".json" for cid in ids))
        self.started_at = time.monotonic()
        self._stopped = False
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.show()
        self.state.setText("运行中")
        self.timer.start(1000)
        self.process.start()

    def _process_error(self, error):
        if self.process:
            self.log.appendPlainText(self.process.errorString())
        if error == QProcess.FailedToStart:
            self._finished(-1, QProcess.CrashExit)

    def _update_elapsed(self):
        self.elapsed.setText(f"耗时：{time.monotonic() - self.started_at:.1f} s")

    def _read_output(self) -> None:
        if self.process:
            self._append_output(self._log_decoder.decode(bytes(self.process.readAllStandardOutput())))

    def _append_output(self, output: str) -> None:
        if not output:
            return
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(output)
        self._progress_tail = (self._progress_tail + output)[-4096:]
        turns = re.findall(r"\b[Tt]urn[:\s]+(\d+)(?:/(\d+))?", self._progress_tail)
        if turns:
            current, total = turns[-1]
            self.state.setText(f"运行中 · turn {current}" + (f"/{total}" if total else ""))
        estimates = re.findall(r"\bETA:\s*([^|\r\n]+)", self._progress_tail)
        if estimates:
            self.eta.setText("预计剩余：" + estimates[-1].strip())

    def _finished(self, code: int, status: QProcess.ExitStatus) -> None:
        self._read_output()
        self._append_output(self._log_decoder.decode(b"", final=True))
        self.timer.stop()
        self.progress.hide()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.state.setText("已停止" if self._stopped else "完成" if code == 0 and status == QProcess.NormalExit else f"失败（退出码 {code}）")
        self._update_elapsed()
        self.eta.setText("预计剩余：--")

    def stop_run(self) -> None:
        if self.process and self.process.state() != QProcess.NotRunning:
            self._stopped = True
            self.process.kill()


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
        controls.addWidget(QLabel("滚轮缩放，适配视图恢复完整范围"))
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


from PASS.gui.workspace import DocumentWindowMixin


class MainWindow(DocumentWindowMixin, QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings("PASS", "Editor")
        self.resize(1200, 760)
        self.setMinimumSize(1000, 650)
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QHBoxLayout()
        header.setContentsMargins(20, 8, 18, 8)
        header.setSpacing(12)
        brand = QLabel("PASS")
        brand.setObjectName("brand")
        header.addWidget(brand)
        version = QLabel(f"v{__version__}")
        version.setObjectName("muted")
        header.addWidget(version)
        header.addSpacing(12)
        self.file_button = QToolButton()
        self.file_button.setText("文件")
        self.file_button.setPopupMode(QToolButton.InstantPopup)
        self.file_menu = QMenu(self.file_button)
        self.file_button.setMenu(self.file_menu)
        header.addWidget(self.file_button)
        header.addSpacing(14)
        self.nav = []
        for index, label in enumerate(("配置", "运行", "绘图", "工具")):
            item = button(label, "nav")
            item.setCheckable(True)
            item.clicked.connect(lambda checked=False, i=index: self._show_page(i))
            header.addWidget(item)
            self.nav.append(item)
        header.addStretch()
        self.theme_button = QToolButton()
        self.theme_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.theme_button.setPopupMode(QToolButton.InstantPopup)
        theme_menu = QMenu(self.theme_button)
        self.theme_actions = {}
        for mode, label in (("dark", "深色 · One Dark Pro"), ("light", "浅色"), ("system", "跟随系统")):
            action = theme_menu.addAction(label)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, m=mode: self.apply_theme(m))
            self.theme_actions[mode] = action
        self.theme_button.setMenu(theme_menu)
        header.addWidget(self.theme_button)
        self.help_button = QToolButton()
        self.help_button.setText("帮助")
        self.help_button.setPopupMode(QToolButton.InstantPopup)
        self.help_menu = HelpMenu(self)
        self.help_button.setMenu(self.help_menu)
        header.addWidget(self.help_button)
        outer.addLayout(header)
        self.stack = QStackedWidget()
        self.config = ConfigPage()
        self.run = RunPage(self.config)
        self.run.controller = self
        self.plot = PlotPage()
        self.tools = ToolsPage()
        for page in (self.config, self.run, self.plot, self.tools):
            self.stack.addWidget(page)
        outer.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self._init_documents()
        self._show_page(0)
        geometry = self.settings.value("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)
        split = self.settings.value("window/splitter")
        if split:
            self.config.splitter.restoreState(split)
        self.theme_preference = self.settings.value("theme", "dark", type=str)
        self.apply_theme(self.theme_preference)
        QApplication.instance().styleHints().colorSchemeChanged.connect(self._system_theme_changed)

    def apply_theme(self, preference: str) -> None:
        if preference not in ("dark", "light", "system"):
            preference = "dark"
        self.theme_preference = preference
        theme = preference
        if preference == "system":
            theme = "light" if QApplication.instance().styleHints().colorScheme() == Qt.ColorScheme.Light else "dark"
        self.current_theme = theme
        self.settings.setValue("theme", preference)
        apply_application_theme(theme)
        self.tools.set_theme(theme)
        self.config.json_highlighter.set_theme(theme)
        self.theme_button.setText({"dark": "深色", "light": "浅色", "system": "跟随系统"}[preference])
        self.theme_button.setIcon(icon("moon" if theme == "dark" else "sun", THEMES[theme]["muted"]))
        for mode, action in self.theme_actions.items():
            action.setChecked(mode == preference)
        for name, glyph in [("输入配置", "settings"), ("Twiss 与光学", "optics"), ("元件", "box"),
                            ("序列工具", "tools"), ("监测与诊断", "chart"), ("物理效应", "layers")]:
            self.config.library_sections[name].header.setIcon(icon(glyph, THEMES[theme]["muted"]))
        for widget, glyph in [(self.config.validate_button, "check"), (self.config.delete_sequence_button, "trash"),
                              (self.config.contents_button, "folder"), (self.config.form_apply, "check")]:
            widget.setIcon(icon(glyph, THEMES[theme]["muted"]))
        for action in self.config.findChildren(QAction):
            if glyph := action.property("themeIcon"):
                action.setIcon(icon(glyph, THEMES[theme]["text"]))

    def _system_theme_changed(self, _scheme):
        if self.theme_preference == "system":
            self.apply_theme("system")

    def _show_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for i, item in enumerate(self.nav):
            item.setChecked(i == index)
        if index == 1:
            self.run.refresh_inputs()

    def closeEvent(self, event) -> None:
        if not self._confirm_replace():
            event.ignore()
            return
        if not self.help_menu.confirm_close():
            event.ignore()
            return
        if self.run.process and self.run.process.state() != QProcess.NotRunning:
            answer = QMessageBox.question(self, "任务仍在运行", "停止当前运行并关闭窗口？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.run.stop_run()
            self.run.process.waitForFinished(2000)
        self.help_menu.builder.shutdown()
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/splitter", self.config.splitter.saveState())
        self._release_project()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("PASS")
    app.setOrganizationName("PASS")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


__all__ = ["main", "MainWindow"]
