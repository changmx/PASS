"""Shared independent reference inputs, result fields and plot export controls."""
import csv
import json
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QSignalBlocker, QSize, QTimer, Signal, Qt
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSplitter,
    QTabWidget, QVBoxLayout, QWidget)

from PASS.gui.appearance import THEMES
from PASS.gui.structured import IntegerSpinBox
from PASS.gui.tool_beam import format_number, hint, number, output, ParticleEditor
from PASS.gui.tool_formulas import FormulaDialog
from PASS.gui.beam_calculator import solve_kinematics
from PASS.tool.particles import ParticleSpec


def integer(value, minimum=0, maximum=10000):
    widget = IntegerSpinBox(value, minimum, maximum)
    widget.setButtonSymbols(QAbstractSpinBox.NoButtons)
    return widget


class ReferenceBeamBox(QWidget):
    changed = Signal()

    def __init__(self, source=None):
        super().__init__()
        self.source = source
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.editor = ParticleEditor(searchable=False, title="参考束流")
        outer.addWidget(self.editor)
        for name in ("mass_number", "charge_state", "atomic_number", "species", "identity"):
            setattr(self, name, getattr(self.editor, name))
        self.ek = number(100)
        self.ek_label = QLabel("Ek (AMeV)")
        self.import_button = QPushButton("读取束流计算器")
        self.import_button.setEnabled(source is not None)
        row = QHBoxLayout()
        row.addWidget(self.ek_label)
        row.addWidget(self.ek, 1)
        row.addWidget(self.import_button)
        self.editor.layout().addLayout(row, 3, 0, 1, 4)
        self.mass_results = {}
        self.mass_labels = {}
        for column, (key, label) in enumerate((("rest_mass", "静止质量 m₀/A (MeV/c²)"),
                                               ("mass_ratio", "质量比 μ = m₀/u"))):
            self.mass_results[key] = output()
            self.mass_labels[key] = QLabel(label)
            self.editor.layout().addWidget(self.mass_labels[key], 4, 2 * column)
            self.editor.layout().addWidget(self.mass_results[key], 4, 2 * column + 1)
        self.import_button.clicked.connect(self.read_source)
        self.editor.changed.connect(self.refresh)
        self.ek.valueChanged.connect(self.refresh)
        self.refresh()

    def current(self):
        return solve_kinematics(self.editor.current(), "kinetic_energy", self.ek.value() * 1e6)

    def refresh(self):
        try:
            particle = self.editor.current()
            self.ek_label.setText(f"Ek ({particle.kinetic_energy_unit})")
            suffix = "/A" if particle.uses_nucleon_units else ""
            self.mass_labels["rest_mass"].setText(f"静止质量 m₀{suffix} (MeV/c²)")
            mass_ratio = particle.mass_in_u
            self.mass_results["rest_mass"].setText(format_number(particle.normalized_rest_mass_ev_c2 / 1e6))
            self.mass_results["mass_ratio"].setText(format_number(mass_ratio))
        except (ValueError, OverflowError):
            for field in self.mass_results.values():
                field.setText("—")
        self.changed.emit()

    def read_source(self):
        k = self.source() if self.source else None
        if k is None:
            self.identity.setText("束流计算器当前参数无效，请先修正后再读取。")
            return
        with QSignalBlocker(self.editor):
            self.editor.set_particle(k.particle)
        with QSignalBlocker(self.ek):
            self.ek.setValue(k.ek_ev / 1e6)
        self.refresh()

    def snapshot(self):
        particle = self.editor.current()
        return {"A": self.mass_number.value(), "q": self.charge_state.value(), "Z": self.atomic_number.value(),
                "mu_mass_ratio": self.mass_results["mass_ratio"].text(),
                "energy_normalization": "per_nucleon" if particle.uses_nucleon_units else "per_particle",
                "Ek_AMeV" if particle.uses_nucleon_units else "Ek_MeV": self.ek.value(),
                "species": self.species.currentData()}


class ResultFields(QGroupBox):
    def __init__(self, fields, title="计算结果", columns=1):
        super().__init__(title)
        self.fields = fields  # key, displayed label, multiplier from SI/eV
        self.outputs = {}
        self.labels = {}
        layout = QGridLayout(self)
        for index, (key, label, _) in enumerate(fields):
            row, col = index // columns, 2 * (index % columns)
            self.labels[key] = QLabel(label)
            layout.addWidget(self.labels[key], row, col)
            widget = output()
            self.outputs[key] = widget
            layout.addWidget(widget, row, col+1)
            layout.setColumnStretch(col+1, 1)

    def clear(self):
        for widget in self.outputs.values():
            widget.setText("—")

    def set_values(self, values):
        for key, _, scale in self.fields:
            value = values[key]
            self.outputs[key].setText(format_number(None if value is None else value * scale))

    def snapshot(self):
        return {self.labels[key].text(): self.outputs[key].text() for key, _, _ in self.fields}


class PhysicsToolPage(QWidget):
    """Base for lazy tool pages. Subclasses supply recalculate and CSV rows."""
    def __init__(self, title, formulas, source=None, *, plot=False, stacked=False):
        super().__init__()
        self.title, self.formulas = title, formulas
        self.theme = "dark"
        self.formula_dialog = None
        self.copy_payload = None
        self.plot_enabled = plot
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self.recalculate)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 12, 16, 12)
        header = QHBoxLayout()
        heading = QLabel(title)
        heading.setObjectName("formTitle")
        header.addWidget(heading)
        header.addStretch()
        self.copy_button = QPushButton("复制结果")
        self.copy_button.clicked.connect(self.copy_results)
        header.addWidget(self.copy_button)
        if plot:
            self.data_button = QPushButton("导出数据…")
            self.data_button.clicked.connect(self.export_data)
            header.addWidget(self.data_button)
            self.export_button = QPushButton("导出图形…")
            self.export_button.clicked.connect(self.export_plot)
            header.addWidget(self.export_button)
        formula = QPushButton("详细公式…")
        formula.clicked.connect(self.show_formulas)
        header.addWidget(formula)
        self.layout.addLayout(header)
        self.reference = ReferenceBeamBox(source)
        self.reference.changed.connect(self.schedule)
        self.layout.addWidget(self.reference)
        if plot:
            self.splitter = QSplitter()
            self.splitter.setChildrenCollapsible(False)
            self.layout.addWidget(self.splitter, 1)
            plot_panel = QWidget()
            pl = QVBoxLayout(plot_panel)
            pl.setContentsMargins(0, 0, 4, 0)
            self.figure = Figure(figsize=(6, 5), dpi=100, layout="constrained")
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setMinimumSize(350, 260)
            self.ax = self.figure.add_subplot(111)
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setIconSize(QSize(18, 18))
            pl.addWidget(self.toolbar)
            pl.addWidget(self.canvas, 1)
            self.summary = hint()
            pl.addWidget(self.summary)
            self.splitter.addWidget(plot_panel)
            self.tabs = QTabWidget()
            self.tabs.setMinimumWidth(340)
            self.tabs.setMaximumWidth(400)
            self.parameters = QWidget()
            self.controls = QVBoxLayout(self.parameters)
            self.controls.setContentsMargins(8, 8, 8, 8)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(self.parameters)
            self.tabs.addTab(scroll, "参数")
            self.result_page = QWidget()
            self.result_layout = QVBoxLayout(self.result_page)
            result_scroll = QScrollArea()
            result_scroll.setWidgetResizable(True)
            result_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            result_scroll.setWidget(self.result_page)
            if stacked:
                result_scroll.takeWidget()
                self.result_page.setParent(self.parameters)
                self.tabs.tabBar().hide()
                result_scroll.deleteLater()
            else:
                self.tabs.addTab(result_scroll, "计算结果")
            self.splitter.addWidget(self.tabs)
            self.splitter.setSizes([650, 350])
            self.splitter.setStretchFactor(0, 1)
        self.error = hint()
        self.layout.addWidget(self.error)

    def bind(self, widget):
        signal = widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.valueChanged
        signal.connect(self.schedule)
        return widget

    def schedule(self, *_):
        self.timer.start()

    def prepare_plot(self):
        colors = THEMES[self.theme]
        self.ax.clear()
        self.figure.set_facecolor(colors["bg"])
        self.ax.set_facecolor(colors["input"])
        self.ax.tick_params(colors=colors["text"], labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color(colors["line"])
        self.ax.grid(alpha=.2, color=colors["muted"])
        return colors

    def finish_plot(self):
        self.ax.legend(loc="upper right", fontsize=8, facecolor=THEMES[self.theme]["panel"],
                                edgecolor=THEMES[self.theme]["line"], labelcolor=THEMES[self.theme]["text"])
        self.toolbar.update()
        self.canvas.draw_idle()

    def set_valid(self, payload):
        self.error.clear()
        self.copy_payload = {"tool": self.title, "reference": self.reference.snapshot(), **payload}
        self.copy_button.setEnabled(True)
        if self.plot_enabled:
            self.data_button.setEnabled(True)
            self.export_button.setEnabled(True)

    def set_error(self, message):
        self.copy_payload = None
        self.copy_button.setEnabled(False)
        self.error.setText(str(message))
        if self.plot_enabled:
            self.data_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.results.clear()
            self.summary.clear()
            self.prepare_plot()
            self.canvas.draw_idle()

    def set_theme(self, theme):
        self.theme = theme
        if self.plot_enabled:
            for _, _, image, callback in self.toolbar.toolitems:
                if callback and callback in self.toolbar._actions:
                    self.toolbar._actions[callback].setIcon(self.toolbar._icon(image + ".png"))
        self.recalculate()

    def show_formulas(self):
        if self.formula_dialog is None:
            self.formula_dialog = FormulaDialog(self.title + " · 详细公式", self.formulas, self)
        self.formula_dialog.show()
        self.formula_dialog.raise_()

    def copy_results(self):
        if self.timer.isActive():
            self.recalculate()
        if self.copy_payload is not None:
            QApplication.clipboard().setText(json.dumps(self.copy_payload, ensure_ascii=False, indent=2))

    def export_plot(self):
        if self.timer.isActive():
            self.recalculate()
        if self.copy_payload is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出图形", "physics-tool.svg", "SVG (*.svg);;PNG (*.png);;PDF (*.pdf)")
        if path:
            try:
                self.figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=self.figure.get_facecolor())
            except (OSError, ValueError) as exc:
                self.error.setText(str(exc))

    def export_data(self):
        if self.timer.isActive():
            self.recalculate()
        if self.copy_payload is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出曲线数据", "physics-tool.csv", "CSV (*.csv)")
        if path:
            try:
                columns, rows = self.export_rows()
                with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(columns)
                    writer.writerows(rows)
            except (OSError, ValueError) as exc:
                self.error.setText(str(exc))
