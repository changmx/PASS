"""Spectrum task settings and bounded result rendering."""

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from PASS.gui.analysis_data import MethodControls, bounded_line_indices, object_labels, style_figure
from PASS.gui.appearance import THEMES


class SpectrumControls(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.method = MethodControls()
        self.method.changed.connect(self.changed)
        layout.addWidget(self.method)
        note = QLabel("基础 FFT 显示离散频谱；精细 FFT 使用窗、补零和峰值插值。补零细化频率网格，不增加观测时长或真实频率分辨率。")
        note.setWordWrap(True)
        layout.addWidget(note)

    def parameters(self):
        return self.method.parameters()


def compute_spectrum(loaded, method, kwargs):
    from PASS.analysis import compute_fft, compute_refined_fft

    operation = compute_refined_fft if method == "refined_fft" else compute_fft
    return operation(loaded["signal"], sample_spacing=loaded["sample_spacing"], **kwargs)


def spectrum_rows(payload):
    result = payload["result"]
    if "peak_frequency" in result:
        peaks = np.asarray(result["peak_frequency"])
        n_peaks = peaks.shape[-1] if peaks.ndim else 1
        rows = dict(object=np.repeat(object_labels(payload["loaded_x"], max(1, peaks.size // n_peaks)), n_peaks),
                    peak=np.tile(np.arange(n_peaks), max(1, peaks.size // n_peaks)),
                    frequency=peaks.reshape(-1))
        for key, label in (("peak_amplitude", "amplitude"), ("peak_phase", "phase_rad"), ("valid", "valid"), ("quality", "quality")):
            if key in result:
                values = np.asarray(result[key])
                if values.size == peaks.size:
                    rows[label] = values.reshape(-1)
        return rows
    frequency = np.asarray(result["frequency"])
    amplitude = np.asarray(result["amplitude"]).reshape(-1, frequency.size)
    phase = np.asarray(result["phase"]).reshape(-1, frequency.size)
    indices = np.arange(min(200, amplitude.size))
    rows, bins = np.divmod(indices, frequency.size)
    identities = object_labels(payload["loaded_x"], amplitude.shape[0])
    return dict(object=identities[rows], frequency=frequency[bins], amplitude=amplitude[rows, bins], phase_rad=phase[rows, bins])


def draw_spectrum(figure, payload, theme):
    figure.clear()
    signal_ax, spectrum_ax = figure.subplots(2, 1)
    loaded, result = payload["loaded_x"], payload["result"]
    signal = np.asarray(loaded["signal"])
    rows = signal.reshape(-1, signal.shape[-1])
    coordinates = loaded.get("sample_coordinates")
    if coordinates is None:
        coordinates = np.arange(rows.shape[-1]) * loaded["sample_spacing"]
    first = rows[0]
    indices = bounded_line_indices(first.real, 5000 if np.iscomplexobj(first) else 10000)
    if np.iscomplexobj(first):
        indices = np.unique(np.concatenate((indices, bounded_line_indices(first.imag, 5000))))
    color = THEMES[theme]["accent"]
    signal_ax.plot(np.asarray(coordinates)[indices], first.real[indices], color=color, lw=.8, label="实部")
    if np.iscomplexobj(first):
        signal_ax.plot(np.asarray(coordinates)[indices], first.imag[indices], color=THEMES[theme]["warning"], lw=.8, label="虚部")
        signal_ax.legend()
    unit = payload["sampling_unit"]
    signal_ax.set_xlabel(f"采样坐标 ({unit})")
    signal_ax.set_ylabel("信号")
    signal_ax.set_title("原始信号 · 首个选定对象")
    frequency = np.asarray(result["frequency"])
    amplitude = np.asarray(result["amplitude"]).reshape(-1, frequency.size)[0]
    indices = bounded_line_indices(amplitude)
    spectrum_ax.plot(frequency[indices], amplitude[indices], color=color, lw=.8)
    spectrum_ax.set_xlabel("频率 (Hz)" if unit == "s" else f"频率 (cycles/{unit})")
    spectrum_ax.set_ylabel("幅值")
    spectrum_ax.set_title("频谱 · 首个选定对象")
    if "peak_frequency" in result:
        peaks = np.asarray(result["peak_frequency"])
        amplitudes = np.asarray(result["peak_amplitude"])
        count = peaks.shape[-1] if peaks.ndim else 1
        spectrum_ax.scatter(peaks.reshape(-1, count)[0], amplitudes.reshape(-1, count)[0], color=THEMES[theme]["warning"], s=25)
    style_figure(figure, theme)


def display_spectrum(view, payload, theme):
    draw_spectrum(view.figure, payload, theme)
    view.set_rows(spectrum_rows(payload))
    signal = np.asarray(payload["loaded_x"]["signal"])
    n_objects = int(np.prod(signal.shape[:-1])) if signal.ndim > 1 else 1
    method = "精细 FFT" if payload["method"] == "refined_fft" else "基础 FFT"
    view.summary.setText(f"{method} · {n_objects} 个对象 × {signal.shape[-1]} 个样本；图示首个对象，最多 10,000 点；表格最多 200 行。")
    view.details.setText("频率、相位和质量标记随完整数值结果导出。导出及 Python 代码使用本次计算的数据选择和参数快照。")
    view.canvas.draw_idle()
