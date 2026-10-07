"""Exciter voltage/kick conversion and FM/AM signal preview."""
from dataclasses import asdict

import numpy as np
from matplotlib.ticker import FuncFormatter
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QFormLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from PASS.gui.structured import format_scientific
from PASS.gui.tool_beam import hint, create_number_input
from PASS.gui.tool_optics_formulas import EXCITER_FORMULAS
from PASS.gui.tool_physics_common import PhysicsToolPage, ResultFields
from PASS.gui.exciter_calculator import ExciterSettings, calculate_exciter, kick_angle_to_voltage, voltage_to_kick_angle


class ExciterPage(PhysicsToolPage):

    def __init__(self, source=None):
        super().__init__("激励计算与绘制", EXCITER_FORMULAS, source, plot=True, stacked=True)
        self.result = None
        self._offset_warning_active = False
        self._offset_warning = QMessageBox(QMessageBox.Warning, "DDS 扫频错开比例", "当前两路 DDS 未错开半个扫频周期；将按设定比例计算。", QMessageBox.Ok, self)
        self._offset_warning.setModal(False)
        self.inputs = {}
        self.scales = {}
        box, form = self._create_section("激励设置")
        self.mode = QComboBox()
        for label, key in (("单频 FM", "single_fm"), ("单频 FM + AM", "single_fm_am"), ("双频 FM", "dual_fm"), ("双频 FM + AM", "dual_fm_am")):
            self.mode.addItem(label, key)
        form.addRow("激励模式", self.mode)
        self.add_input(form, "circumference", "环周长 C (m)", 100)
        self.add_input(form, "kick_angle", "带符号基准踢角 A0 (rad)", 1e-4, minimum=-1e100)
        self.inputs["kick_angle"].setToolTip("以 rad 输入，例如 1e-6；双频每路 DDS 使用同一幅度后相加，正负号由输入角度决定。")
        self.open_conversion = QPushButton("电压换算…")
        self.open_conversion.clicked.connect(self.show_conversion)
        form.addRow(self.open_conversion)
        self.controls.addWidget(box)
        self.conversion_dialog = QDialog(self)
        self.conversion_dialog.setWindowTitle("电压 ↔ 踢角")
        self.conversion_dialog.setMinimumWidth(440)
        conversion_form = QFormLayout(self.conversion_dialog)
        self.conversion_reference = hint()
        conversion_form.addRow(self.conversion_reference)
        self.conversion_inputs = {}
        self.conversion_scales = {"voltage": 1, "kick_angle": 1, "gap": .001, "plate_length": 1}
        self.converted_kick = None
        self.converted_voltage = None
        for key, label, value in (("voltage", "带符号峰值电压差 V (V)", 1000), ("kick_angle", "带符号踢角 (rad)", 1e-4), ("gap", "极板间距 d (mm)", 50),
                                  ("plate_length", "极板有效长 L (m)", .5)):
            widget = create_number_input(value, -1e100 if key in {"voltage", "kick_angle"} else 0, scientific=key == "kick_angle")
            self.conversion_inputs[key] = widget
            conversion_form.addRow(label, widget)
            if key in {"voltage", "kick_angle"}:
                widget.valueChanged.connect(lambda _value, key=key: self.calculate_conversion(key))
                widget.lineEdit().returnPressed.connect(lambda key=key: self.calculate_conversion(key))
            else:
                widget.valueChanged.connect(lambda _value: self.calculate_conversion())
        self.conversion_results = self._create_results([("field", "电场 V/d (V/m)", 1), ("transit", "过板时间 (ns)", 1e9)])
        conversion_form.addRow(self.conversion_results)
        self.apply_conversion = QPushButton("填入基准踢角")
        self.apply_conversion.setAutoDefault(False)
        self.apply_conversion.setDefault(False)
        self.apply_conversion.clicked.connect(self.apply_converted_kick)
        conversion_form.addRow(self.apply_conversion)
        self.conversion_error = hint()
        conversion_form.addRow(self.conversion_error)
        conversion_form.addRow(hint("电压或踢角输入后按回车互算；间距、长度改变时由电压计算踢角。\n点击按钮才填入主页面踢角。"))
        self.reference.changed.connect(self.calculate_conversion)
        frequency_box, self.frequency_form = self._create_section("扫频参数")
        ff = self.frequency_form
        self.frequency_mode = QComboBox()
        self.frequency_mode.addItem("Tune → 频率（默认）", "tune")
        self.frequency_mode.addItem("直接输入频率", "frequency")
        ff.addRow("输入方式", self.frequency_mode)
        for key, label, value, scale in (("excite_tune", "中心激励 tune", .47, 1), ("sweep_tune", "扫频 tune 全宽", .02, 1),
                                         ("central_frequency", "中心频率 fc (kHz)", 600, 1000), ("sweep_width", "扫频全宽 Δf (kHz)", 20, 1000),
                                         ("period", "扫频周期 T (ms)", 1, .001), ("dual_sweep_offset", "DDS 扫频错开比例 δ/T", .5, 1)):
            self.add_input(ff, key, label, value, scale)
        self.inputs["dual_sweep_offset"].setMaximum(1)
        self.inputs["dual_sweep_offset"].setToolTip("DDS 1 的扫频进度领先 DDS 2 的比例；默认错开半周期。两路同时启动，初始正弦相位均为 0。")
        self.controls.addWidget(frequency_box)
        self.am_box, am = self._create_section("AM 参数")
        for key, label, value, scale in (("am_t_ext", "t_ext (ms)", 100, .001), ("am_r0", "r0 (mm)", 10, .001), ("am_delta0", "δ0 (mm)", 5, .001),
                                         ("am_k_const", "模型系数 k", 1, 1)):
            self.add_input(am, key, label, value, scale)
        self.controls.addWidget(self.am_box)
        plotting, pf = self._create_section("绘图与到达时间")
        self.plot_kind = QComboBox()
        for label, key in (("踢角与包络", "wave"), ("FM 频率", "frequency"), ("采样频谱", "spectrum"), ("AM 因子", "am")):
            self.plot_kind.addItem(label, key)
        self.plot_kind.setCurrentIndex(self.plot_kind.findData("frequency"))
        pf.addRow("显示", self.bind(self.plot_kind))
        self.show_turns = QCheckBox("叠加逐圈采样点")
        self.show_turns.toggled.connect(self.schedule)
        pf.addRow(self.show_turns)
        for key, label, value, scale in (("start_time", "起始时间 (ms)", 0, .001), ("duration", "时间跨度 (ms)", 2, .001),
                                         ("reference_clock_start", "启动时参考钟 t0 (ms)", 0, .001), ("z_rel", "z_rel (m)", 0, 1)):
            self.add_input(pf, key, label, value, scale, -1e100 if key in ("reference_clock_start", "z_rel") else 0)
        self.controls.addWidget(plotting)
        self.results = self._create_results([("f0", "回旋频率 f0 (kHz)", .001), ("cf", "中心频率 fc (kHz)", .001), ("width", "扫频全宽 Δf (kHz)", .001),
                                             ("tune", "中心激励 tune", 1), ("sweep_tune", "扫频 tune 全宽", 1), ("amplitude", "基准踢角 A0 (rad)", 1),
                                             ("peak", "采样最大 |踢角| (rad)", 1), ("rms", "采样 RMS 踢角 (rad)", 1), ("rate", "采样率 (MHz)", 1e-6),
                                             ("resolution", "频谱间隔 (Hz)", 1)])
        self.result_layout.setContentsMargins(0, 0, 0, 0)
        result_section, result_form = self._create_section("计算结果")
        result_form.addRow(self.results)
        self.result_layout.addWidget(result_section)
        self.result_layout.addWidget(hint("预览按所设粒子到达时间计算，不包含束流响应。"))
        self.controls.addWidget(self.result_page)
        self.controls.addStretch()
        self.mode.currentIndexChanged.connect(self.change_mode)
        self.frequency_mode.currentIndexChanged.connect(self.change_frequency_mode)
        self.calculate_conversion()
        self.change_mode()

    def add_input(self, form, key, label, value, scale=1, minimum=0):
        widget = self.bind(create_number_input(value, minimum, scientific=key == "kick_angle"))
        self.inputs[key], self.scales[key] = widget, scale
        form.addRow(label, widget)

    def _create_section(self, title):
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 8, 0, 4)
        label = QLabel(title)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        layout.addWidget(label)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        layout.addLayout(form)
        return section, form

    def _create_results(self, fields):
        results = ResultFields(fields, title="")
        results.setFlat(True)
        results.setStyleSheet("QGroupBox { border: none; margin: 0; padding: 0; }")
        results.layout().setContentsMargins(0, 0, 0, 0)
        return results

    def show_conversion(self):
        self.calculate_conversion()
        self.conversion_dialog.ensurePolished()
        self.conversion_dialog.layout().activate()
        self.conversion_dialog.adjustSize()
        self.conversion_dialog.show()
        self.conversion_dialog.raise_()
        self.conversion_dialog.activateWindow()

    def calculate_conversion(self, known="voltage"):
        self.converted_kick = None
        self.converted_voltage = None
        try:
            kinematics = self.reference.current()
            self.conversion_reference.setText(f"使用主页面参考束流：q={kinematics.particle.charge_state:g}，"
                                              f"Ek={self.reference.ek.value():g} {kinematics.particle.kinetic_energy_unit}。")
            values = {key: widget.value() * self.conversion_scales[key] for key, widget in self.conversion_inputs.items()}
            if known == "voltage":
                self.converted_voltage = values["voltage"]
                self.converted_kick = voltage_to_kick_angle(kinematics, self.converted_voltage, values["plate_length"], values["gap"])
                with QSignalBlocker(self.conversion_inputs["kick_angle"]):
                    self.conversion_inputs["kick_angle"].setValue(self.converted_kick)
            else:
                self.converted_kick = values["kick_angle"]
                self.converted_voltage = kick_angle_to_voltage(kinematics, self.converted_kick, values["plate_length"], values["gap"])
                with QSignalBlocker(self.conversion_inputs["voltage"]):
                    self.conversion_inputs["voltage"].setValue(self.converted_voltage)
            self.conversion_results.set_values(
                dict(field=self.converted_voltage / values["gap"], transit=values["plate_length"] / kinematics.velocity))
            self.conversion_error.clear()
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            self.converted_kick = None
            self.converted_voltage = None
            self.conversion_results.clear()
            self.conversion_error.setText(str(exc))
        self.apply_conversion.setEnabled(self.converted_kick is not None)

    def apply_converted_kick(self):
        self.calculate_conversion()
        if self.converted_kick is not None:
            self.inputs["kick_angle"].setValue(self.converted_kick)
            self.recalculate()
            self.conversion_dialog.hide()

    def change_frequency_mode(self):
        if self.result is not None:
            values = {
                "excite_tune": self.result.central_frequency / self.result.frequency_0,
                "sweep_tune": self.result.sweep_width / self.result.frequency_0,
                "central_frequency": self.result.central_frequency,
                "sweep_width": self.result.sweep_width
            }
            for key, value in values.items():
                with QSignalBlocker(self.inputs[key]):
                    self.inputs[key].setValue(value / self.scales[key])
        self.change_mode()

    def change_mode(self):
        tune = self.frequency_mode.currentData() == "tune"
        for key in ("excite_tune", "sweep_tune", "central_frequency", "sweep_width"):
            self.frequency_form.setRowVisible(self.inputs[key], tune == (key in ("excite_tune", "sweep_tune")))
        self.frequency_form.setRowVisible(self.inputs["dual_sweep_offset"], self.mode.currentData().startswith("dual"))
        self.am_box.setVisible(self.mode.currentData().endswith("_am"))
        self.recalculate()

    def recalculate(self):
        self.timer.stop()
        self.result = None
        try:
            settings = ExciterSettings(mode=self.mode.currentData(),
                                       frequency_mode=self.frequency_mode.currentData(),
                                       **{
                                           key: widget.value() * self.scales[key]
                                           for key, widget in self.inputs.items()
                                       })
            r = calculate_exciter(self.reference.current(), settings)
            offset_warning = settings.mode.startswith("dual") and not np.isclose(settings.dual_sweep_offset, .5, rtol=0, atol=1e-12)
            if offset_warning and not self._offset_warning_active:
                self._offset_warning.show()
            elif not offset_warning:
                self._offset_warning.close()
            self._offset_warning_active = offset_warning
            colors = self.prepare_plot()
            kind = self.plot_kind.currentData()
            time = r.times * 1000
            if kind == "wave":
                self.ax.plot(time, r.kick, lw=.6, color=colors["accent"], label="Kick")
                self.ax.plot(time, r.envelope, lw=1, color="#c98a42", label="Envelope")
                self.ax.plot(time, -r.envelope, lw=1, color="#c98a42")
                if self.show_turns.isChecked():
                    self.ax.plot(r.turn_times * 1000, r.turn_kick, ".", ms=2, color=colors["text"], label="Turn samples")
                ylabel = "Kick (rad)"
            elif kind == "frequency":
                self.ax.plot(time,
                             r.dds1_frequency / 1000,
                             color=colors["accent"],
                             label="FM frequency" if settings.mode.startswith("single") else "DDS 1")
                if settings.mode.startswith("dual"):
                    self.ax.plot(time, r.dds2_frequency / 1000, color="#c98a42", label="DDS 2")
                ylabel = "Frequency (kHz)"
            elif kind == "am":
                self.ax.plot(time, r.am_factor, color=colors["accent"], label="AM factor")
                ylabel = "AM factor"
            else:
                frequencies, amplitude = r.spectrum()
                self.ax.plot(frequencies / 1000, amplitude, color=colors["accent"], lw=.8, label="Hann amplitude spectrum")
                limit = max(abs(r.dds1_frequency).max(), abs(r.dds2_frequency).max(), 1 / settings.duration)
                self.ax.set_xlim(0, min(r.sample_rate / 2, 2 * limit))
                ylabel = "Kick amplitude (rad)"
            if kind in {"wave", "spectrum"}:
                self.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: format_scientific(float(f"{value:.8g}"))))
            self.ax.set_xlabel("Frequency (kHz)" if kind == "spectrum" else "Elapsed time (ms)", color=colors["text"])
            self.ax.set_ylabel(ylabel, color=colors["text"])
            self.finish_plot()
            values = dict(f0=r.frequency_0,
                          cf=r.central_frequency,
                          width=r.sweep_width,
                          tune=r.central_frequency / r.frequency_0,
                          sweep_tune=r.sweep_width / r.frequency_0,
                          amplitude=r.kick_amplitude,
                          peak=float(abs(r.kick).max()),
                          rms=float(np.sqrt(np.mean(r.kick**2))),
                          rate=r.sample_rate,
                          resolution=1 / settings.duration)
            self.results.set_values(values)
            for key in ("amplitude", "peak", "rms"):
                self.results.outputs[key].setText(format_scientific(values[key]))
            self.summary.setText(f"A0={format_scientific(r.kick_amplitude)} rad · fc={r.central_frequency/1000:.9g} kHz · {len(r.times)} 个采样点")
            self.result = r
            self.set_valid({"settings_SI": asdict(settings), "results": self.results.snapshot()})
        except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError) as exc:
            self.set_error(exc)

    def export_rows(self):
        r = self.result
        if self.plot_kind.currentData() == "spectrum":
            return ("frequency_Hz", "kick_amplitude_rad"), zip(*r.spectrum())
        return ("elapsed_time_s", "arrival_time_s", "kick_rad", "envelope_rad", "am_factor", "dds1_frequency_Hz",
                "dds2_frequency_Hz"), zip(r.times, r.arrival_times, r.kick, r.envelope, r.am_factor, r.dds1_frequency, r.dds2_frequency)
