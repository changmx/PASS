"""Spectrum task settings and bounded result rendering."""

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from PASS.gui.analysis_data import MethodControls, bounded_line_indices, object_labels, style_figure
from PASS.gui.appearance import THEMES
from PASS.gui.data_cursor import CursorSeries, set_cursor_sources


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
        if payload.get("task") == "spectrum":
            rows.pop("object")
        return rows
    frequency = np.asarray(result["frequency"])
    amplitude = np.asarray(result["amplitude"]).reshape(-1, frequency.size)
    phase = np.asarray(result["phase"]).reshape(-1, frequency.size)
    indices = np.arange(min(200, amplitude.size))
    rows, bins = np.divmod(indices, frequency.size)
    identities = object_labels(payload["loaded_x"], amplitude.shape[0])
    values = dict(object=identities[rows], frequency=frequency[bins], amplitude=amplitude[rows, bins], phase_rad=phase[rows, bins])
    if payload.get("task") == "spectrum":
        values.pop("object")
    return values


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
    object_id = object_labels(loaded, len(rows))[0]
    trace_label = (f"统计量 {payload.get('selection_x', '')}"
                   if payload.get("task") == "spectrum" else f"粒子 {object_id}" if payload.get("task") == "particle_spectrum" else f"对象 {object_id}")
    coordinates = np.asarray(coordinates)

    def signal_tip(index):
        value = first[index]
        text = f"数据点 · {trace_label}\n采样坐标 = {coordinates[index]:.10g} {payload['sampling_unit']}\n实部 = {value.real:.10g}"
        if np.iscomplexobj(first):
            text += f"\n虚部 = {value.imag:.10g}"
        return text

    sources = [CursorSeries(coordinates, first.real, signal_tip, lookup_mode="x")]
    if np.iscomplexobj(first):
        sources.append(CursorSeries(coordinates, first.imag, signal_tip, lookup_mode="x"))
    set_cursor_sources(signal_ax, sources)
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
    signal_ax.set_title("原始信号 · " + trace_label)
    frequency = np.asarray(result["frequency"])
    amplitude = np.asarray(result["amplitude"]).reshape(-1, frequency.size)[0]
    phase = np.asarray(result["phase"]).reshape(-1, frequency.size)[0]
    frequency_unit = "Hz" if unit == "s" else f"cycles/{unit}"

    def spectrum_tip(index):
        return (f"频谱数据点 · {trace_label}\n频率 = {frequency[index]:.10g} {frequency_unit}\n幅值 = {amplitude[index]:.10g}"
                f"\n相位 = {phase[index]:.10g} rad")

    spectrum_sources = [CursorSeries(frequency, amplitude, spectrum_tip, lookup_mode="x")]
    indices = bounded_line_indices(amplitude)
    spectrum_ax.plot(frequency[indices], amplitude[indices], color=color, lw=.8)
    spectrum_ax.set_xlabel("频率 (Hz)" if unit == "s" else f"频率 (cycles/{unit})")
    spectrum_ax.set_ylabel("幅值")
    spectrum_ax.set_title("频谱 · " + trace_label)
    if "peak_frequency" in result:
        peaks = np.asarray(result["peak_frequency"])
        amplitudes = np.asarray(result["peak_amplitude"])
        count = peaks.shape[-1] if peaks.ndim else 1
        spectrum_ax.scatter(peaks.reshape(-1, count)[0], amplitudes.reshape(-1, count)[0], color=THEMES[theme]["warning"], s=25)
        peak_frequency = peaks.reshape(-1, count)[0]
        peak_amplitude = amplitudes.reshape(-1, count)[0]
        peak_phase = np.asarray(result["peak_phase"]).reshape(-1, count)[0]

        def peak_tip(index):
            return (f"精细谱峰 {index + 1} · {trace_label}\n频率 = {peak_frequency[index]:.10g} {frequency_unit}\n幅值 = {peak_amplitude[index]:.10g}"
                    f"\n相位 = {peak_phase[index]:.10g} rad")

        spectrum_sources.append(CursorSeries(peak_frequency, peak_amplitude, peak_tip, priority=1, hit_radius=10.))
    set_cursor_sources(spectrum_ax, spectrum_sources)
    style_figure(figure, theme)


def display_spectrum(view, payload, theme):
    draw_spectrum(view.figure, payload, theme)
    view.set_rows(spectrum_rows(payload))
    if payload.get("task") == "particle_spectrum" and view.table.columnCount():
        view.table.horizontalHeaderItem(0).setText("粒子 ID")
    signal = np.asarray(payload["loaded_x"]["signal"])
    n_objects = int(np.prod(signal.shape[:-1])) if signal.ndim > 1 else 1
    method = "精细 FFT" if payload["method"] == "refined_fft" else "基础 FFT"
    if payload.get("task") == "spectrum":
        view.summary.setText(f"统计数据频谱 · {payload.get('selection_x', '')} · {method} · {signal.shape[-1]} 个样本；图形最多 10,000 点，表格最多 200 行。")
    else:
        view.summary.setText(f"PM 粒子频谱 · {method} · {n_objects} 个粒子 × {signal.shape[-1]} 个样本；图示首个选定粒子，最多 10,000 点；表格最多 200 行。")
    view.details.setText("频率、相位和质量标记随完整数值结果导出。导出及 Python 代码使用本次计算的数据选择和参数快照。")
    view.canvas.draw_idle()
