"""Focused tests for GUI data adapters that do not need a visible window."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from PASS.gui.app import PlotCanvas, PlotPage


def test_plot_reader_extracts_numeric_csv_columns(tmp_path):
    source = tmp_path / "stats.csv"
    source.write_text("turn,x_rms,label\n0,1.0,first\n1,2.5,second\n", encoding="utf-8")

    result = PlotPage._read_table(str(source))

    assert result == {"turn": [0.0, 1.0], "x_rms": [1.0, 2.5]}


def test_plot_reader_extracts_tfs_columns(tmp_path):
    source = tmp_path / "particle.tfs"
    source.write_text(
        "@ TITLE %s \"PASS\"\n* TURN X PX\n$ %d %le %le\n0 0.001 0.0\n1 0.002 0.1\n",
        encoding="utf-8",
    )

    result = PlotPage._read_table(str(source))

    assert result == {"TURN": [0.0, 1.0], "X": [0.001, 0.002], "PX": [0.0, 0.1]}


def test_plot_canvas_tracks_selected_x_and_y_series():
    app = QApplication.instance() or QApplication([])
    canvas = PlotCanvas()

    canvas.set_series([10.0, 20.0, 30.0], [2.0, 4.0, 3.0])

    assert canvas.x_values == [10.0, 20.0, 30.0]
    assert canvas.values == [2.0, 4.0, 3.0]
    assert canvas._x_limits == (10.0, 30.0)
    assert canvas._y_limits == (2.0, 4.0)
