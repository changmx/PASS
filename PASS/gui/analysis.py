"""Independent spectrum and frequency-map workspace with asynchronous numerical work."""

import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSplitter, QStackedWidget,
                               QVBoxLayout, QWidget)

from PASS.gui.widgets import file_dialog_directory


class AnalysisWorker(QThread):
    """Own each operation until completion; cancellation never destroys a running thread."""

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.result = None
        self.error = None

    def check(self):
        if self.isInterruptionRequested():
            raise InterruptedError("分析操作已取消。")

    def run(self):
        try:
            self.check()
            self.result = self.operation(self.check)
        except Exception as exc:
            self.error = exc


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _compute(snapshot, check):
    from PASS.analysis.data_io import load_signal
    from PASS.gui.analysis_fma import compute_frequency_map
    from PASS.gui.analysis_spectrum import compute_spectrum

    check()
    loaded_x = load_signal(snapshot["path"], snapshot["selection_x"], **snapshot["load_options"])
    check()
    loaded_y = None
    if snapshot["task"] == "fma":
        loaded_y = load_signal(snapshot["path"], snapshot["selection_y"], **snapshot["load_options"])
        check()
        result = compute_frequency_map(loaded_x, loaded_y, snapshot["kwargs"])
    else:
        result = compute_spectrum(loaded_x, snapshot["method"], snapshot["kwargs"])
    check()
    return dict(snapshot, loaded_x=loaded_x, loaded_y=loaded_y, result=result)


def _python_invocation(snapshot):
    options = ", ".join(f"{key}={value!r}" for key, value in snapshot["load_options"].items())
    source = f"{snapshot['path']!r}, {snapshot['selection_x']!r}, {options}"
    function = "compute_fma" if snapshot["task"] == "fma" else ("compute_refined_fft" if snapshot["method"] == "refined_fft" else "compute_fft")
    lines = [f"from PASS.analysis import {function}", "from PASS.analysis.data_io import load_signal", "", f"data_x = load_signal({source})"]
    kwargs = ", ".join(f"{key}={value!r}" for key, value in snapshot["kwargs"].items())
    if snapshot["task"] == "fma":
        lines.append(f"data_y = load_signal({snapshot['path']!r}, {snapshot['selection_y']!r}, {options})")
        lines.append(f"result = {function}(data_x['signal'], data_y['signal'], sample_spacing=data_x['sample_spacing'], {kwargs})")
    else:
        lines.append(f"result = {function}(data_x['signal'], sample_spacing=data_x['sample_spacing'], {kwargs})")
    return "\n".join(lines) + "\n"


def _export_numeric(payload, path, check):
    """Publish a complete result and calculation metadata in one atomic file."""
    path = Path(path)
    snapshot = {key: value for key, value in payload.items() if key not in ("loaded_x", "loaded_y", "result")}
    metadata = dict(configuration=snapshot,
                    result=payload["result"].get("metadata", {}),
                    source_x=payload["loaded_x"].get("metadata", {}),
                    source_y=None if payload["loaded_y"] is None else payload["loaded_y"].get("metadata", {}))
    fd, name = tempfile.mkstemp(prefix=".pass-analysis-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        check()
        if path.suffix.lower() == ".npz":
            arrays = {key: np.asarray(value) for key, value in payload["result"].items() if key != "metadata"}
            for side in ("x", "y"):
                loaded = payload[f"loaded_{side}"]
                if loaded is None:
                    continue
                for key in ("signal", "sample_coordinates", "object_ids"):
                    if loaded.get(key) is not None:
                        arrays[f"input_{side}_{key}"] = np.asarray(loaded[key])
            arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, default=_json_value))
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
        else:
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                stream.write("# " + json.dumps(metadata, ensure_ascii=False, default=_json_value) + "\n")
                writer = csv.writer(stream)
                if payload["task"] == "fma":
                    from PASS.gui.analysis_fma import fma_rows

                    columns = fma_rows(payload)
                    writer.writerow(columns)
                    for i, row in enumerate(zip(*columns.values())):
                        if i % 4096 == 0:
                            check()
                        writer.writerow(row)
                else:
                    from PASS.gui.analysis_data import object_labels

                    result = payload["result"]
                    frequencies = np.asarray(result["frequency"])
                    amplitude = np.asarray(result["amplitude"]).reshape(-1, frequencies.size)
                    phase = np.asarray(result["phase"]).reshape(amplitude.shape)
                    coefficients = np.asarray(result["coefficients"]).reshape(amplitude.shape)
                    identities = object_labels(payload["loaded_x"], amplitude.shape[0])
                    writer.writerow(
                        ("object", "record_type", "frequency", "amplitude", "phase_rad", "coefficient_real", "coefficient_imag", "valid", "quality"))
                    for i in range(amplitude.shape[0]):
                        for start in range(0, frequencies.size, 4096):
                            check()
                            end = min(start + 4096, frequencies.size)
                            writer.writerows(
                                zip(np.full(end - start, identities[i]), ["spectrum"] * (end - start), frequencies[start:end],
                                    amplitude[i, start:end], phase[i, start:end], coefficients[i, start:end].real, coefficients[i, start:end].imag,
                                    [""] * (end - start), [""] * (end - start)))
                    if "peak_frequency" in result:
                        peaks = np.asarray(result["peak_frequency"]).reshape(amplitude.shape[0], -1)
                        peak_amplitude = np.asarray(result["peak_amplitude"]).reshape(peaks.shape)
                        peak_phase = np.asarray(result["peak_phase"]).reshape(peaks.shape)
                        valid = np.asarray(result["valid"]).reshape(peaks.shape)
                        quality = np.asarray(result["quality"]).reshape(peaks.shape)
                        for i in range(peaks.shape[0]):
                            check()
                            for j in range(peaks.shape[1]):
                                writer.writerow(
                                    (identities[i], "peak", peaks[i, j], peak_amplitude[i, j], peak_phase[i, j], "", "", valid[i, j], quality[i, j]))
        check()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


class AnalysisPage(QWidget):
    shutdown_finished = Signal()
    analysis_finished = Signal()
    analysis_failed = Signal(str)
    load_finished = Signal()

    def __init__(self):
        super().__init__()
        self.theme = "dark"
        self._activated = False
        self._worker = None
        self._operation = None
        self._closing = False
        self._cancelled = False
        self._inventory_key = None
        self.result = None
        self.dynamic_aperture = None
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(12, 12, 12, 12)

    def activate(self):
        if self._activated or self._closing:
            return
        from PASS.gui.analysis_data import AnalysisResultView, SourceControls
        from PASS.gui.analysis_fma import FmaControls
        from PASS.gui.analysis_spectrum import SpectrumControls
        from PASS.gui.dynamic_aperture import DynamicAperturePage
        from PASS.gui.tools import ToolNavigation

        self._activated = True
        self.navigation = ToolNavigation(items=(("频谱分析", "spectrum"), ("频率图分析", "resonance"), ("动力学孔径", "dynamic_aperture")))
        self._layout.addWidget(self.navigation)
        self.stack = QStackedWidget()
        signal_page = QWidget()
        signal_layout = QVBoxLayout(signal_page)
        signal_layout.setContentsMargins(16, 12, 16, 12)
        self.stack.addWidget(signal_page)
        self.dynamic_aperture = DynamicAperturePage()
        self.dynamic_aperture.shutdown_finished.connect(self._child_shutdown_finished)
        self.stack.addWidget(self.dynamic_aperture)
        self._layout.addWidget(self.stack, 1)
        header = QHBoxLayout()
        self.analysis_title = QLabel("频谱分析")
        self.analysis_title.setObjectName("formTitle")
        header.addWidget(self.analysis_title)
        header.addStretch()
        self.copy_button = QPushButton("复制本次 Python 调用")
        self.export_button = QPushButton("导出数值…")
        self.image_button = QPushButton("导出图形…")
        for button in (self.copy_button, self.export_button, self.image_button):
            header.addWidget(button)
            button.setEnabled(False)
        self.copy_button.clicked.connect(self.copy_python)
        self.export_button.clicked.connect(self.export_data)
        self.image_button.clicked.connect(self.export_image)
        signal_layout.addLayout(header)
        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        controls = QWidget()
        self.controls_layout = QVBoxLayout(controls)
        self.controls_layout.setContentsMargins(0, 0, 8, 0)
        # Retain programmatic mode selection; the sidebar is the only visible selector.
        self.task = QComboBox(self)
        self.task.addItem("频谱分析", "spectrum")
        self.task.addItem("频率图分析 FMA", "fma")
        self.task.hide()
        self.source = SourceControls()
        self.controls_layout.addWidget(self.source)
        self.settings = QStackedWidget()
        self.spectrum, self.fma = SpectrumControls(), FmaControls()
        self.settings.addWidget(self.spectrum)
        self.settings.addWidget(self.fma)
        self.controls_layout.addWidget(self.settings)
        self.controls_layout.addStretch()
        self.control_scroll = QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setWidget(controls)
        self.control_scroll.setMinimumWidth(330)
        splitter.addWidget(self.control_scroll)
        self.view = AnalysisResultView()
        splitter.addWidget(self.view)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([370, 700])
        signal_layout.addWidget(splitter, 1)
        footer = QHBoxLayout()
        self.run_button = QPushButton("运行分析")
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.status = QLabel("选择文件并读取可用列 / 数组。")
        self.status.setWordWrap(True)
        footer.addWidget(self.run_button)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.status, 1)
        signal_layout.addLayout(footer)
        self.source.browse.clicked.connect(self.choose_file)
        self.source.inspect_requested.connect(self.inspect_source)
        self.source.changed.connect(self._settings_changed)
        self.source.path.returnPressed.connect(self.inspect_source)
        self.spectrum.changed.connect(self._settings_changed)
        self.fma.changed.connect(self._settings_changed)
        self.task.currentIndexChanged.connect(self._task_changed)
        self.navigation.currentRowChanged.connect(self._select_analysis)
        self.run_button.clicked.connect(self.run_analysis)
        self.cancel_button.clicked.connect(self.cancel)
        self.navigation.setCurrentRow(0)
        self.set_theme(self.theme)

    def _select_analysis(self, index):
        if index == 2:
            self.stack.setCurrentWidget(self.dynamic_aperture)
            self.dynamic_aperture.activate()
        else:
            self.stack.setCurrentIndex(0)
            self.analysis_title.setText(self.navigation.buttons[index].text())
            self.task.setCurrentIndex(index)

    def _child_shutdown_finished(self):
        if self._closing and not self.busy:
            self.shutdown_finished.emit()

    def _task_changed(self):
        self.settings.setCurrentIndex(self.task.currentIndex())
        self.source.set_fma(self.task.currentData() == "fma")
        self.navigation.setCurrentRow(self.task.currentIndex())
        self._settings_changed()

    def _settings_changed(self):
        if self.result is not None and not self.busy:
            self.status.setText("设置已更改；当前图形、导出及 Python 调用仍对应上次计算。请重新运行以更新结果。")

    def _source_key(self):
        return (str(Path(self.source.path.text()).expanduser().resolve()), repr(self.source.file_options()))

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择分析数据",
                                              self.source.path.text() or file_dialog_directory(self),
                                              "信号数据 (*.csv *.tsv *.txt *.dat *.tfs *.h5 *.hdf5 *.npy *.npz);;所有文件 (*)")
        if path:
            self.open_path(path)

    def open_path(self, path):
        self.activate()
        if self.busy or self._closing:
            if self._activated:
                self.status.setText("请等待当前操作结束后再打开文件。")
            return False
        self.navigation.setCurrentRow(self.task.currentIndex())
        self.source.path.setText(str(Path(path).resolve()))
        self.inspect_source()
        return True

    def inspect_source(self):
        if self.busy or self._closing:
            return
        try:
            path = str(Path(self.source.path.text()).expanduser().resolve())
            if not Path(path).is_file():
                raise ValueError("请选择存在的数据文件。")
            options = self.source.file_options()
            key = self._source_key()

            def operation(check):
                from PASS.analysis.data_io import inspect_data

                check()
                value = inspect_data(path, **options)
                check()
                return key, value

            self._start("inspect", operation, "正在读取列 / 数组信息…")
        except Exception as exc:
            self._failed(exc)

    def _snapshot(self):
        if self._inventory_key != self._source_key():
            raise ValueError("文件或文本读取设置已更改；请先重新读取列 / 数组列表。")
        selection_x = self.source.selection_x.currentData()
        selection_y = self.source.selection_y.currentData()
        if selection_x is None or (self.task.currentData() == "fma" and selection_y is None):
            raise ValueError("请选择有效信号列 / 数组。")
        panel = self.fma if self.task.currentData() == "fma" else self.spectrum
        method, kwargs = panel.parameters()
        return dict(path=str(Path(self.source.path.text()).expanduser().resolve()),
                    task=self.task.currentData(),
                    method=method,
                    kwargs=kwargs,
                    selection_x=selection_x,
                    selection_y=selection_y,
                    load_options=self.source.load_options(),
                    sampling_unit=self.source.unit.currentData())

    def run_analysis(self):
        if self.busy or self._closing:
            return
        try:
            snapshot = self._snapshot()
            self._start("compute", lambda check: _compute(snapshot, check), "正在读取信号并计算；界面可继续响应…")
        except Exception as exc:
            self._failed(exc)

    def _start(self, kind, operation, message):
        self._operation = kind
        self._cancelled = False
        self._worker = AnalysisWorker(operation, self)
        self._worker.finished.connect(self._finished)
        self._set_busy(True)
        self.status.setText(message)
        self._worker.start()

    def _set_busy(self, enabled):
        self.control_scroll.setEnabled(not enabled)
        for button in self.navigation.buttons[:2]:
            button.setEnabled(not enabled)
        self.run_button.setEnabled(not enabled)
        self.cancel_button.setEnabled(enabled)
        for button in (self.copy_button, self.export_button, self.image_button):
            button.setEnabled(not enabled and self.result is not None)

    def _finished(self):
        worker, self._worker = self._worker, None
        kind, self._operation = self._operation, None
        if worker is None:
            return
        value, error = worker.result, worker.error
        worker.deleteLater()
        if self._closing:
            self._child_shutdown_finished()
            return
        self._set_busy(False)
        if (self._cancelled and not (kind == "export" and value is not None)) or isinstance(error, InterruptedError):
            self.status.setText("操作已取消；已完成的上次结果仍可使用。")
            return
        if error is not None:
            self._failed(error)
            return
        try:
            if kind == "inspect":
                self._inventory_key, inventory = value
                self.source.set_inventory(inventory)
                self.status.setText("列 / 数组列表已读取；选择信号、采样坐标和分析方法。")
                self.load_finished.emit()
            elif kind == "compute":
                self.result = value
                self._display_result()
                self._set_busy(False)
                self.status.setText("分析完成；导出及 Python 调用保留本次数据选择与参数。")
                self.analysis_finished.emit()
            else:
                self.status.setText("已导出：" + str(value))
        except Exception as exc:
            self._failed(exc)

    def _failed(self, error):
        message = str(error)
        self.status.setText("分析失败：" + message)
        self.analysis_failed.emit(message)

    def _display_result(self):
        if self.result["task"] == "fma":
            from PASS.gui.analysis_fma import display_frequency_map

            display_frequency_map(self.view, self.result, self.theme)
        else:
            from PASS.gui.analysis_spectrum import display_spectrum

            display_spectrum(self.view, self.result, self.theme)

    def copy_python(self):
        if self.result is not None:
            QApplication.clipboard().setText(_python_invocation(self.result))
            self.status.setText("已复制本次计算的 Python 调用。")

    def export_data(self, path=None):
        if self.busy or self.result is None or self._closing:
            return
        if not isinstance(path, (str, Path)):
            path, selected = QFileDialog.getSaveFileName(self, "导出完整分析结果", "analysis_result.npz", "NumPy 结果与元数据 (*.npz);;CSV 数值表 (*.csv)")
            if path and not Path(path).suffix:
                path += ".csv" if selected.startswith("CSV") else ".npz"
        if path:
            if Path(path).suffix.lower() not in (".npz", ".csv"):
                self._failed(ValueError("数值导出支持 .npz 或 .csv。"))
                return
            payload = self.result
            self._start("export", lambda check: _export_numeric(payload, path, check), "正在导出完整数值结果…")

    def export_image(self, path=None):
        if self.busy or self.result is None or self._closing:
            return
        if not isinstance(path, (str, Path)):
            path, _ = QFileDialog.getSaveFileName(self, "导出当前分析图形", "analysis_result.png", "PNG 图形 (*.png);;SVG 图形 (*.svg)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".png"
        if Path(path).suffix.lower() not in (".png", ".svg"):
            self._failed(ValueError("图形导出支持 .png 或 .svg。"))
            return
        # Display work is bounded; saving this owned Figure cannot race a worker.
        temporary = None
        try:
            destination = Path(path)
            fd, name = tempfile.mkstemp(prefix=".pass-analysis-", suffix=destination.suffix, dir=destination.parent)
            os.close(fd)
            temporary = Path(name)
            self.view.figure.savefig(temporary, dpi=160, facecolor=self.view.figure.get_facecolor())
            temporary.replace(destination)
            self.status.setText("已导出图形：" + str(path))
        except Exception as exc:
            self._failed(exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def cancel(self):
        if self._worker is not None:
            self._cancelled = True
            self._worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.status.setText("正在取消；当前读取或数值运算结束后释放资源…")

    @property
    def busy(self):
        return self._worker is not None or (self.dynamic_aperture is not None and self.dynamic_aperture.busy)

    def shutdown(self):
        self._closing = True
        self.cancel()
        if self.dynamic_aperture is not None:
            self.dynamic_aperture.shutdown()
        return not self.busy

    def set_theme(self, theme):
        self.theme = theme
        if self.dynamic_aperture is not None:
            self.dynamic_aperture.set_theme(theme)
        if self._activated:
            self.navigation.set_theme(theme)
            if self.result is not None:
                self._display_result()
            else:
                from PASS.gui.analysis_data import style_figure

                style_figure(self.view.figure, theme)
                self.view.canvas.draw_idle()
