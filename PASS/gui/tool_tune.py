"""Embedded resonance diagram with editable working points and line controls."""
import csv
import io
import math
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtCore import QSignalBlocker, QSize, QTimer, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QAbstractSpinBox, QApplication, QCheckBox, QColorDialog,
    QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QPushButton, QScrollArea, QSplitter, QTableWidget, QTabWidget,
    QTableWidgetItem, QVBoxLayout, QWidget)

from PASS.gui.appearance import THEMES
from PASS.gui.structured import IntegerSpinBox
from PASS.gui.tool_beam import hint, number
from PASS.gui.tool_formulas import FormulaDialog, TUNE_FORMULAS
from PASS.tool.tune_diagram import ResonanceLine, resonance_lines


POINT_COLORS = ("#3479dc", "#d78532", "#9767c3", "#3b987a", "#cf647d", "#629baa")
MARKERS = (("圆点", "o"), ("方块", "s"), ("三角", "^"), ("菱形", "D"), ("加号", "+"))


def parse_working_points(text):
    """Atomic CSV/TSV import: name,Qx,Qy[,color,marker], optional header.

    Names are optional. Two-column Qx,Qy data have blank names.
    """
    rows = list(csv.reader(io.StringIO(text.lstrip("\ufeff")), delimiter="\t" if "\t" in text.splitlines()[0] else ",")) if text.strip() else []
    points = []
    for line_no, row in enumerate(rows, 1):
        if not row or all(not s.strip() for s in row):
            continue
        row = [s.strip() for s in row]
        if line_no == 1 and [s.lower() for s in row[:3]] in (["name", "qx", "qy"], ["名称", "qx", "qy"]):
            continue
        if line_no == 1 and [s.lower() for s in row] == ["qx", "qy"]:
            continue
        try:
            if len(row) == 2:
                name, x, y = "", float(row[0]), float(row[1])
                color, marker = POINT_COLORS[len(points) % len(POINT_COLORS)], "o"
            elif 3 <= len(row) <= 5:
                name, x, y = row[0], float(row[1]), float(row[2])
                color = row[3] if len(row) >= 4 else POINT_COLORS[len(points) % len(POINT_COLORS)]
                marker = row[4] if len(row) == 5 else "o"
            else:
                raise ValueError("列数应为 2、3、4 或 5")
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("坐标须为有限数")
            if not QColor(color).isValid() or marker not in dict(MARKERS).values():
                raise ValueError("无效颜色或符号（o、s、^、D、+）")
        except ValueError as exc:
            raise ValueError(f"工作点数据第 {line_no} 行：{exc}") from exc
        points.append((name, x, y, QColor(color).name(), marker))
    if not points:
        raise ValueError("没有可导入的工作点。")
    return points


class TuneDiagramPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme = "dark"
        self.formula_dialog = None
        self._building = True
        self._valid_plot = False
        self._next_color = 0
        self._artists = {}
        self.point_legend = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self.redraw)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        header = QHBoxLayout()
        title = QLabel("共振线图")
        title.setObjectName("formTitle")
        header.addWidget(title)
        header.addStretch()
        formula = QPushButton("详细公式…")
        formula.clicked.connect(self.show_formulas)
        header.addWidget(formula)
        self.export_button = QPushButton("导出图形…")
        self.export_button.clicked.connect(self.export_plot)
        header.addWidget(self.export_button)
        layout.addLayout(header)
        self.splitter = split = QSplitter()
        split.setChildrenCollapsible(False)
        layout.addWidget(split, 1)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(6, 6), dpi=100)
        self.figure.set_layout_engine("constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(360, 360)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setIconSize(QSize(18, 18))
        ll.addWidget(self.toolbar)
        ll.addWidget(self.canvas, 1)
        self.details = hint("选择图中工作点查看坐标。")
        ll.addWidget(self.details)
        points_page = QWidget()
        pgl = QVBoxLayout(points_page)
        pgl.setContentsMargins(8, 8, 8, 8)
        point_actions = QGridLayout()
        for index, (text, callback) in enumerate((("添加", self.add_point), ("删除选中", self.remove_points), ("粘贴", self.paste_points),
                                                 ("导入 CSV…", self.import_points), ("导出 CSV…", self.export_points))):
            button = QPushButton(text)
            button.clicked.connect(callback)
            point_actions.addWidget(button, index // 3, index % 3)
        pgl.addLayout(point_actions)
        self.points = QTableWidget(0, 4)
        self.points.setHorizontalHeaderLabels(["显示", "名称", "Qx", "Qy"])
        self.points.horizontalHeaderItem(1).setToolTip("名称可留空；非空名称显示在绘图区右上角图例。")
        self.points.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.points.verticalHeader().hide()
        self.points.setMinimumHeight(150)
        self.points.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for column, width in ((0, 36), (2, 68), (3, 68)):
            self.points.setColumnWidth(column, width)
        self.points.itemChanged.connect(self.schedule)
        self.points.currentCellChanged.connect(self._refresh_point_style)
        pgl.addWidget(self.points, 1)
        point_style = QHBoxLayout()
        point_style.addWidget(QLabel("颜色"))
        self.point_color = QPushButton()
        self.point_color.setFixedWidth(42)
        self.point_color.setAccessibleName("选中工作点的颜色")
        self.point_color.clicked.connect(self._choose_point_color)
        point_style.addWidget(self.point_color)
        point_style.addWidget(QLabel("符号"))
        self.point_marker = QComboBox()
        for label, symbol in MARKERS:
            self.point_marker.addItem(label, symbol)
        self.point_marker.currentIndexChanged.connect(self._change_point_marker)
        point_style.addWidget(self.point_marker, 1)
        pgl.addLayout(point_style)
        pgl.addWidget(hint("名称可留空；非空名称显示在右上角图例。双击编辑；选中行可调整颜色和符号。"))
        split.addWidget(left)
        controls_panel = QWidget()
        controls_panel.setMinimumWidth(310)
        controls_panel.setMaximumWidth(460)
        panel_layout = QVBoxLayout(controls_panel)
        panel_layout.setContentsMargins(4, 0, 0, 0)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls = QWidget()
        controls_scroll.setWidget(controls)
        cl = QVBoxLayout(controls)
        cl.setContentsMargins(8, 8, 8, 8)
        ranges = QGroupBox("坐标范围（完整 tune）")
        rg = QGridLayout(ranges)
        self.ranges = {}
        for row, axis in enumerate(("Qx", "Qy")):
            rg.addWidget(QLabel(axis), row, 0)
            for col, edge in enumerate(("min", "max"), 1):
                spin = number(9 if edge == "min" else 10, -1e6)
                spin.setMaximum(1e6)
                spin.setToolTip(f"{axis} {'下限' if edge == 'min' else '上限'}")
                self.ranges[f"{axis}_{edge}"] = spin
                spin.valueChanged.connect(self.schedule)
                rg.addWidget(spin, row, col)
        panel_layout.addWidget(ranges)
        self.control_tabs = QTabWidget()
        self.control_tabs.addTab(points_page, "工作点")
        self.control_tabs.addTab(controls_scroll, "共振线")
        panel_layout.addWidget(self.control_tabs, 1)
        order_group = QGroupBox("共振阶数")
        og = QGridLayout(order_group)
        self.orders = {}
        for order in range(1, 13):
            check = QCheckBox(f"{order} 阶")
            check.setChecked(order <= 4)
            check.toggled.connect(self.schedule)
            self.orders[order] = check
            og.addWidget(check, (order - 1) // 3, (order - 1) % 3)
        self.kinds = {}
        for row, (key, label) in enumerate((("single", "单平面共振"), ("sum", "和共振（实线）"), ("diff", "差共振（虚线）")), 4):
            check = QCheckBox(label)
            check.setChecked(True)
            check.toggled.connect(self.schedule)
            self.kinds[key] = check
            og.addWidget(check, row, 0, 1, 3)
        cl.addWidget(order_group)
        custom_group = QGroupBox("自定义：m Qx + n Qy = l")
        cgl = QVBoxLayout(custom_group)
        inputs = QHBoxLayout()
        self.coefficients = []
        for label, default in (("m", 3), ("n", 0), ("l", 28)):
            column = QVBoxLayout()
            column.addWidget(QLabel(label))
            spin = IntegerSpinBox(default, -1000000, 1000000)
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spin.setMinimumWidth(45)
            self.coefficients.append(spin)
            column.addWidget(spin)
            inputs.addLayout(column)
        cgl.addLayout(inputs)
        add_line = QPushButton("添加共振线")
        add_line.clicked.connect(self.add_line)
        cgl.addWidget(add_line)
        self.lines = QTableWidget(0, 2)
        self.lines.setHorizontalHeaderLabels(["显示", "共振方程"])
        self.lines.setColumnWidth(0, 42)
        self.lines.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.lines.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.lines.setMinimumHeight(95)
        self.lines.setMaximumHeight(110)
        self.lines.itemChanged.connect(self.schedule)
        cgl.addWidget(self.lines)
        remove_line = QPushButton("删除选中共振线")
        remove_line.clicked.connect(self.remove_lines)
        cgl.addWidget(remove_line)
        cl.addWidget(custom_group)
        cl.addWidget(hint("自定义线使用强调色；相同几何直线只绘制一次。"))
        cl.addStretch()
        split.addWidget(controls_panel)
        split.setStretchFactor(0, 1)
        split.setSizes([700, 330])
        self.error = hint()
        layout.addWidget(self.error)
        self.canvas.mpl_connect("pick_event", self._pick)
        self.canvas.mpl_connect("motion_notify_event", self._hover)
        self._append_points([("", 9.47, 9.43, POINT_COLORS[0], "o")])
        self._building = False
        self.redraw()

    def schedule(self, *_):
        if not self._building:
            self._timer.start()

    def _checked_item(self):
        item = QTableWidgetItem()
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        return item

    def _append_points(self, points):
        with QSignalBlocker(self.points):
            for name, x, y, color, marker in points:
                row = self.points.rowCount()
                self.points.insertRow(row)
                self.points.setItem(row, 0, self._checked_item())
                for column, value in ((1, name), (2, repr(x)), (3, repr(y))):
                    self.points.setItem(row, column, QTableWidgetItem(value))
                self.points.item(row, 1).setData(Qt.UserRole, color)
                self.points.item(row, 1).setData(Qt.UserRole + 1, marker)
                self._next_color += 1
        self.points.setCurrentCell(self.points.rowCount() - 1, 1)
        self._refresh_point_style()
        self.schedule()

    def add_point(self):
        x = (self.ranges["Qx_min"].value() + self.ranges["Qx_max"].value()) / 2
        y = (self.ranges["Qy_min"].value() + self.ranges["Qy_max"].value()) / 2
        self._append_points([("", x, y, POINT_COLORS[self._next_color % len(POINT_COLORS)], "o")])

    def _refresh_point_style(self, *_):
        item = self.points.item(self.points.currentRow(), 1)
        self.point_color.setEnabled(item is not None)
        self.point_marker.setEnabled(item is not None)
        color = item.data(Qt.UserRole) if item else None
        self.point_color.setStyleSheet(f"background-color: {color}; border: 1px solid #888;" if color else "")
        self.point_color.setToolTip(color or "先选择一个工作点")
        with QSignalBlocker(self.point_marker):
            self.point_marker.setCurrentIndex(self.point_marker.findData(item.data(Qt.UserRole + 1)) if item else -1)

    def _choose_point_color(self):
        item = self.points.item(self.points.currentRow(), 1)
        if item is None:
            return
        color = QColorDialog.getColor(QColor(item.data(Qt.UserRole)), self, "工作点颜色")
        if color.isValid():
            item.setData(Qt.UserRole, color.name())
            self._refresh_point_style()

    def _change_point_marker(self):
        item = self.points.item(self.points.currentRow(), 1)
        if item is not None:
            item.setData(Qt.UserRole + 1, self.point_marker.currentData())

    def remove_points(self):
        for row in sorted({i.row() for i in self.points.selectedIndexes()}, reverse=True):
            self.points.removeRow(row)
        self._refresh_point_style()
        self.schedule()

    def remove_lines(self):
        for row in sorted({i.row() for i in self.lines.selectedIndexes()}, reverse=True):
            self.lines.removeRow(row)
        self.schedule()

    def add_line(self):
        try:
            line = ResonanceLine(*(s.value() for s in self.coefficients))
            if any(self.lines.item(row, 1).data(Qt.UserRole).key == line.key for row in range(self.lines.rowCount())):
                raise ValueError("自定义列表中已有同一条几何直线。")
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        with QSignalBlocker(self.lines):
            row = self.lines.rowCount()
            self.lines.insertRow(row)
            self.lines.setItem(row, 0, self._checked_item())
            item = QTableWidgetItem(line.label)
            item.setData(Qt.UserRole, line)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.lines.setItem(row, 1, item)
        self.schedule()

    def point_data(self, visible_only=False):
        data = []
        for row in range(self.points.rowCount()):
            if visible_only and self.points.item(row, 0).checkState() != Qt.Checked:
                continue
            try:
                name = self.points.item(row, 1).text().strip()
                x, y = (float(self.points.item(row, column).text()) for column in (2, 3))
                if not all(math.isfinite(v) for v in (x, y)):
                    raise ValueError()
            except (ValueError, AttributeError):
                raise ValueError(f"工作点第 {row + 1} 行：Qx/Qy 必须是有限数值。") from None
            item = self.points.item(row, 1)
            data.append((name, x, y, item.data(Qt.UserRole), item.data(Qt.UserRole + 1)))
        return data

    def redraw(self):
        self._timer.stop()
        self._valid_plot = False
        self.export_button.setEnabled(False)
        colors = THEMES[self.theme]
        self.ax.clear()
        self._artists.clear()
        self.point_legend = None
        self.figure.set_facecolor(colors["bg"])
        self.ax.set_facecolor(colors["input"])
        self.ax.tick_params(colors=colors["text"], labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color(colors["line"])
        self.ax.set_xlabel("$Q_x$", color=colors["text"])
        self.ax.set_ylabel("$Q_y$", color=colors["text"])
        try:
            xr = (self.ranges["Qx_min"].value(), self.ranges["Qx_max"].value())
            yr = (self.ranges["Qy_min"].value(), self.ranges["Qy_max"].value())
            orders = [n for n, w in self.orders.items() if w.isChecked()]
            lines = resonance_lines(orders, xr, yr, [k for k, w in self.kinds.items() if w.isChecked()])
            points = self.point_data(visible_only=True)
            custom = [self.lines.item(row, 1).data(Qt.UserRole) for row in range(self.lines.rowCount()) if self.lines.item(row, 0).checkState() == Qt.Checked]
            custom_keys = {line.key for line in custom}
            palette = (colors["muted"], "#c97859", "#559975", "#a07fc2", "#519aa8", "#bd9a48")
            legends = []
            for order in orders:
                color = palette[(order - 1) % len(palette)]
                for kind in ("single", "sum", "diff"):
                    segments = [line.segment(xr, yr) for line in lines if line.order == order and line.kind == kind and line.key not in custom_keys]
                    if segments:
                        self.ax.add_collection(LineCollection(segments, colors=color, linewidths=max(.5, 1.3 - .07*order), linestyles="dashed" if kind == "diff" else "solid", alpha=.72))
                legends.append(Line2D([], [], color=color, linewidth=1.2, label=f"{order}"))
            for line in custom:
                if segment := line.segment(xr, yr):
                    self.ax.add_collection(LineCollection([segment], colors=colors["accent"], linewidths=2, linestyles="dashed" if line.kind == "diff" else "solid"))
            outside = 0
            point_handles, point_names = [], []
            for name, x, y, color, marker in points:
                if not (xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1]):
                    outside += 1
                    continue
                artist, = self.ax.plot(x, y, marker=marker, color=color, markersize=6, linestyle="none", picker=7, zorder=5)
                self._artists[artist] = (name, x, y)
                if name:
                    point_handles.append(artist)
                    point_names.append(name)
            self.ax.set(xlim=xr, ylim=yr, aspect="equal")
            if point_handles:
                self.point_legend = self.ax.legend(
                    handles=point_handles, labels=[" "] * len(point_names), loc="upper right",
                    fontsize=8, numpoints=1, markerscale=.9, borderpad=.3, labelspacing=.25,
                    handlelength=.9, handletextpad=.4, borderaxespad=.5, framealpha=.9,
                    facecolor=colors["panel"], edgecolor=colors["line"], labelcolor=colors["text"])
                # Literal names, including leading underscores, work across Matplotlib versions.
                for text, name in zip(self.point_legend.get_texts(), point_names):
                    text.set_text(name)
                    text.set_parse_math(False)
                    text.set_fontfamily(["Microsoft YaHei", "DejaVu Sans"])
                if legends:
                    self.ax.add_artist(self.point_legend)
                    self.point_legend.set_clip_on(False)
            if legends:
                # Keep the outside legend as Axes.legend_ so constrained layout reserves its space.
                legend = self.ax.legend(handles=legends, title="Order", loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=min(6, len(legends)), fontsize=8,
                                        title_fontsize=8, facecolor=colors["panel"], edgecolor=colors["line"], labelcolor=colors["text"])
                legend.get_title().set_color(colors["text"])
            self.error.setText(f"{len(lines)} 条自动线 · {len(custom)} 条自定义线 · {len(points)-outside} 个可见工作点" + (f" · {outside} 个工作点在范围外" if outside else ""))
            self.details.setText("选择图中工作点查看坐标；缩放和平移使用图上方工具栏。")
            self._valid_plot = True
            self.export_button.setEnabled(True)
        except (ValueError, OverflowError) as exc:
            self.error.setText(str(exc))
            self.details.setText("请修正输入后重新绘图。")
        self.toolbar.update()
        self.canvas.draw_idle()

    def set_theme(self, theme):
        self.theme = theme
        # Matplotlib creates toolbar icons once; recolor them after Qt's palette changes.
        for _, _, image, callback in self.toolbar.toolitems:
            if callback and callback in self.toolbar._actions:
                self.toolbar._actions[callback].setIcon(self.toolbar._icon(image + ".png"))
        self.redraw()
        if self.formula_dialog:
            self.formula_dialog.update()

    def _pick(self, event):
        if data := self._artists.get(event.artist):
            self._show_point_details(data)

    def _hover(self, event):
        if event.inaxes is self.ax:
            for artist, data in self._artists.items():
                if artist.contains(event)[0]:
                    self._show_point_details(data)
                    return

    def _show_point_details(self, data):
        prefix = f"{data[0]}   " if data[0] else ""
        self.details.setText(f"{prefix}Qx = {data[1]:.9g}   Qy = {data[2]:.9g}")

    def paste_points(self):
        self.load_points_text(QApplication.clipboard().text())

    def load_points_text(self, text):
        try:
            points = parse_working_points(text)
        except ValueError as exc:
            self.error.setText(str(exc))
            return False
        self._append_points(points)
        return True

    def import_points(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入工作点（追加）", "", "CSV / TSV (*.csv *.tsv);;All files (*)")
        if path:
            try:
                self.load_points_text(Path(path).read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError) as exc:
                self.error.setText(str(exc))

    def export_points(self):
        try:
            points = self.point_data()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出全部工作点", "tune-working-points.csv", "CSV (*.csv)")
        if path:
            try:
                with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("name", "Qx", "Qy", "color", "marker"))
                    writer.writerows(points)
            except OSError as exc:
                self.error.setText(str(exc))

    def export_plot(self):
        if self._timer.isActive():
            self.redraw()
        if not self._valid_plot:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出共振线图", "tune-diagram.svg", "SVG (*.svg);;PNG (*.png);;PDF (*.pdf)")
        if path:
            try:
                self.figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=self.figure.get_facecolor())
            except (OSError, ValueError) as exc:
                self.error.setText(str(exc))

    def show_formulas(self):
        if self.formula_dialog is None:
            self.formula_dialog = FormulaDialog("共振线图 · 详细公式", TUNE_FORMULAS, self)
        self.formula_dialog.show()
        self.formula_dialog.raise_()
