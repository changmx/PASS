"""Design explicit feedback coefficients without changing the input document."""

import cmath
import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout

from PASS.gui.parameters import IntegerValidator
from PASS.para.schema.transverse_feedback import design_feedback_fir


class FeedbackFIRDialog(QDialog):
    """Preview all measured planes together before filling an editable draft."""

    def __init__(self, parent=None, *, pickup_name, plane, delay_turns=1, tap_count=5):
        super().__init__(parent)
        if plane not in ("x", "y", "xy"):
            raise ValueError("拾取器 Plane 必须为 x、y 或 xy")
        self.result_values = None
        self._preview_values = None
        self.setWindowTitle("生成 FIR 系数")
        self.resize(700, 690)
        root = QVBoxLayout(self)
        note = QLabel(f"拾取器：{pickup_name} · 平面：{plane}\n"
                      "分别填写各平面的 tune 与同一圈编号内拾取器→踢点的有符号相移（rad）。相移不能仅由 S 推断。")
        note.setTextFormat(Qt.PlainText)
        note.setWordWrap(True)
        root.addWidget(note)
        settings = QFormLayout()
        self.method = QComboBox()
        self.method.addItem("minimum_norm · 最小系数范数", "minimum_norm")
        self.method.addItem("flat · 目标 tune 处一阶平坦", "flat")
        self.method.setToolTip("minimum_norm 至少 3 个系数；flat 至少 5 个。两者均抑制直流，并在目标 tune 处具有单位响应。")
        settings.addRow("设计方法", self.method)
        self.taps = QSpinBox()
        self.taps.setRange(3, 256)
        self.taps.setValue(tap_count)
        self.taps.setToolTip("交互生成器支持 3–256 个系数；手动输入及 Python 设计函数不受此上限限制。")
        settings.addRow("系数数量", self.taps)
        self.delay = QLineEdit(str(delay_turns))
        self.delay.setValidator(IntegerValidator(self.delay))
        settings.addRow("延迟（圈，至少 1）", self.delay)
        root.addLayout(settings)
        planes = QHBoxLayout()
        self.tunes, self.phases = {}, {}
        for axis in plane:
            group = QGroupBox(f"{axis} 平面")
            layout = QFormLayout(group)
            tune, phase = QLineEdit(), QLineEdit()
            tune.setPlaceholderText("目标 betatron tune")
            phase.setPlaceholderText("有符号相移，rad")
            layout.addRow("Tune", tune)
            layout.addRow("相移（rad）", phase)
            self.tunes[axis], self.phases[axis] = tune, phase
            planes.addWidget(group)
            tune.textChanged.connect(self._invalidate_preview)
            phase.textChanged.connect(self._invalidate_preview)
        root.addLayout(planes)
        self.generate_button = QPushButton("计算并预览")
        self.generate_button.clicked.connect(self.generate)
        root.addWidget(self.generate_button)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("系数按最新延迟采样在前排列：a[0] 对应 n−d，a[k] 对应 n−d−k。")
        root.addWidget(self.preview, 1)
        self.status = QLabel("先计算，再填入草稿。生成系数不会自动设置增益；零增益仍不产生反馈踢。")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.status)
        hint = QLabel("填入时更新两个平面的系数字段（未测量平面设为 null）及延迟，保留增益和其他参数。\n"
                      "可继续手动修改系数，再点击应用／插入。局部响应设计不保证闭环稳定，flat 也不保证更好的抑制效果。")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        root.addWidget(hint)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.apply_button = QPushButton("填入草稿")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        actions.addWidget(self.apply_button)
        actions.addWidget(cancel)
        root.addLayout(actions)
        self.method.currentIndexChanged.connect(self._invalidate_preview)
        self.taps.valueChanged.connect(self._invalidate_preview)
        self.delay.textChanged.connect(self._invalidate_preview)
        # Enter must not silently accept a stale preview after editing a value.
        self.generate_button.setDefault(True)
        self.apply_button.setAutoDefault(False)
        cancel.setAutoDefault(False)

    def _invalidate_preview(self, *_args):
        self._preview_values = None
        self.apply_button.setEnabled(False)
        self.preview.clear()
        self.status.setText("参数已更改，请重新计算并预览。")

    def generate(self):
        self._invalidate_preview()
        values = {"FIR coefficients x": None, "FIR coefficients y": None}
        lines = []
        try:
            text = self.delay.text().strip()
            if self.delay.validator().validate(text, 0)[0] != self.delay.validator().State.Acceptable:
                raise ValueError("延迟必须为至少 1 圈的整数")
            delay = int(text)
            if delay < 1:
                raise ValueError("延迟必须为至少 1 圈的整数")
            values["Delay turns"] = delay
            for axis in self.tunes:
                try:
                    tune = float(self.tunes[axis].text())
                    phase = float(self.phases[axis].text())
                    coefficients = design_feedback_fir(tune, phase, delay, self.taps.value(), method=self.method.currentData())
                except (ValueError, OverflowError) as exc:
                    raise ValueError(f"{axis} 平面：请检查 tune、相移和系数数量。{exc}") from exc
                values[f"FIR coefficients {axis}"] = coefficients
                omega = 2.0 * math.pi * math.remainder(tune, 1.0)
                response = sum(value * cmath.exp(-1j * omega * (delay + k)) for k, value in enumerate(coefficients))
                target = cmath.exp(1j * math.remainder(phase + math.pi / 2.0, 2.0 * math.pi))
                lines.extend((f"{axis} · a[0] → a[{len(coefficients) - 1}]", ", ".join(repr(value) for value in coefficients),
                              f"Σa = {math.fsum(coefficients):.3e}   |H| = {abs(response):.9g}   |H − 目标| = {abs(response - target):.3e}", ""))
        except (ValueError, OverflowError) as exc:
            self.status.setText(f"无法生成：{exc}")
            return False
        self._preview_values = values
        self.preview.setPlainText("\n".join(lines))
        self.status.setText("计算完成。填入草稿后仍可编辑；增益保持原值。")
        self.apply_button.setEnabled(True)
        return True

    def accept(self):
        if self._preview_values is not None:
            self.result_values = self._preview_values
            super().accept()
