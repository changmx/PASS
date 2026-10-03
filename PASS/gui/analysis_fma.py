"""Frequency-map task controls, paired input checks and result display."""

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

from PASS.gui.analysis_data import MethodControls, _integer, _pair, _responsive_form, bounded_indices, object_labels, style_figure


class FmaControls(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.method = MethodControls(fma=True)
        self.method.changed.connect(self.changed)
        layout.addWidget(self.method)
        box = QGroupBox("两个分析窗口")
        form = QFormLayout(box)
        self.automatic = QCheckBox("自动使用两个等长窗口")
        self.automatic.setToolTip("使用信号前后两个等长且不重叠的窗口；奇数长度信号舍去末尾一个样本。")
        self.automatic.setChecked(True)
        form.addRow(self.automatic)
        self.first_start, self.first_end = _integer(0, 0), _integer(128, 1)
        self.second_start, self.second_end = _integer(128, 0), _integer(256, 1)
        form.addRow("窗口一 [起, 止)", _pair(self.first_start, self.first_end))
        form.addRow("窗口二 [起, 止)", _pair(self.second_start, self.second_end))
        _responsive_form(form)
        self.automatic.toggled.connect(self._update_windows)
        self.automatic.toggled.connect(self.changed)
        for widget in (self.first_start, self.first_end, self.second_start, self.second_end):
            widget.valueChanged.connect(self.changed)
        self._update_windows()
        layout.addWidget(box)
        note = QLabel("窗口索引相对于数据选择后的信号；X/Y 必须按同一对象及采样坐标对齐。频率漂移是有限观测窗的稳定性指标，不能单独证明混沌。")
        note.setWordWrap(True)
        layout.addWidget(note)

    def _update_windows(self):
        for widget in (self.first_start, self.first_end, self.second_start, self.second_end):
            widget.setEnabled(not self.automatic.isChecked())

    def parameters(self):
        method, kwargs = self.method.parameters()
        kwargs["windows"] = (None if self.automatic.isChecked() else
                             ((self.first_start.value(), self.first_end.value()), (self.second_start.value(), self.second_end.value())))
        return method, kwargs


def compute_frequency_map(loaded_x, loaded_y, kwargs):
    from PASS.analysis import compute_fma

    if loaded_x["signal"].shape != loaded_y["signal"].shape:
        raise ValueError("FMA 的 X/Y 数据形状必须相同；请显式选择对应对象和采样轴。")
    if loaded_x["sample_spacing"] != loaded_y["sample_spacing"]:
        raise ValueError("FMA 的 X/Y 采样间隔不一致。")
    for key, label in (("sample_coordinates", "采样坐标"), ("object_ids", "对象标识")):
        first, second = loaded_x.get(key), loaded_y.get(key)
        if (first is None) != (second is None) or (first is not None and not np.array_equal(first, second)):
            raise ValueError(f"FMA 的 X/Y {label}不一致。")
    return compute_fma(loaded_x["signal"], loaded_y["signal"], sample_spacing=loaded_x["sample_spacing"], **kwargs)


def fma_rows(payload):
    result = payload["result"]
    count = np.asarray(result["qx_first"]).size
    rows = {"object": object_labels(payload["loaded_x"], count)}
    for key in ("qx_first", "qy_first", "qx_second", "qy_second", "delta_qx", "delta_qy", "drift", "diffusion_log10", "valid", "quality"):
        if key in result and np.asarray(result[key]).size == count:
            rows[key] = np.asarray(result[key]).reshape(-1)
    return rows


def draw_frequency_map(figure, payload, theme):
    figure.clear()
    ax = figure.add_subplot(111)
    result = payload["result"]
    qx, qy = np.asarray(result["qx_first"]).reshape(-1), np.asarray(result["qy_first"]).reshape(-1)
    drift = np.asarray(result["drift"]).reshape(-1)
    valid = np.asarray(result["valid"]).reshape(-1) & np.isfinite(qx) & np.isfinite(qy) & np.isfinite(drift)
    indices = np.flatnonzero(valid)
    indices = indices[bounded_indices(indices.size)]
    if indices.size:
        scatter = ax.scatter(qx[indices], qy[indices], c=drift[indices], cmap="viridis", s=20)
        colorbar = figure.colorbar(scatter, ax=ax)
        colorbar.set_label("频率漂移 |ΔQ|")
    unit = payload["sampling_unit"]
    label = "Hz" if unit == "s" else f"cycles/{unit}"
    ax.set_xlabel(f"窗口一 Qx ({label})")
    ax.set_ylabel(f"窗口一 Qy ({label})")
    ax.set_title("频率图 · 颜色表示两窗口频率漂移")
    style_figure(figure, theme)


def display_frequency_map(view, payload, theme):
    draw_frequency_map(view.figure, payload, theme)
    rows = fma_rows(payload)
    view.set_rows(rows)
    valid = np.asarray(payload["result"]["valid"])
    view.summary.setText(f"FMA · {int(valid.sum())}/{valid.size} 个有效对象；频率图最多 10,000 个对象，表格最多 200 行。")
    view.details.setText("漂移 = sqrt(ΔQx² + ΔQy²)；零漂移的 log10 为 -∞。数值结果保留有效标记和质量标记。")
    view.canvas.draw_idle()
