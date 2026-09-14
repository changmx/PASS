"""Exciter voltage/kick conversion and FM/AM signal preview."""
from dataclasses import asdict
import numpy as np
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox

from PASS.gui.tool_beam import hint, number
from PASS.gui.tool_optics_formulas import EXCITER_FORMULAS
from PASS.gui.tool_physics_common import PhysicsToolPage, ResultFields
from PASS.gui.exciter_calculator import ExciterSettings, calculate_exciter


class ExciterPage(PhysicsToolPage):
    def __init__(self, source=None):
        super().__init__("激励计算与绘制", EXCITER_FORMULAS, source, plot=True, stacked=True)
        self.result = None
        self.inputs = {}
        self.scales = {}
        box = QGroupBox("激励器与环")
        form = QFormLayout(box)
        self.mode = QComboBox()
        for label, key in (("单频 FM", "single_fm"), ("单频 FM + AM", "single_fm_am"),
                           ("双频 FM", "dual_fm"), ("双频 FM + AM", "dual_fm_am")):
            self.mode.addItem(label, key)
        form.addRow("激励模式", self.mode)
        for key, label, value, scale in (("circumference", "环周长 C (m)", 100, 1),
                ("voltage", "电压幅值 V (V)", 1000, 1), ("gap", "极板间距 d (mm)", 50, .001),
                ("plate_length", "极板有效长 L (m)", .5, 1)):
            self.add_input(form, key, label, value, scale)
        self.controls.addWidget(box)
        frequency_box = QGroupBox("扫频参数")
        self.frequency_form = ff = QFormLayout(frequency_box)
        self.frequency_mode = QComboBox()
        self.frequency_mode.addItem("Tune → 频率", "tune")
        self.frequency_mode.addItem("直接输入频率", "frequency")
        ff.addRow("输入方式", self.frequency_mode)
        for key, label, value, scale in (("excite_tune", "中心激励 tune", .47, 1),
                ("sweep_tune", "扫频 tune 全宽", .02, 1),
                ("central_frequency", "中心频率 fc (kHz)", 600, 1000),
                ("sweep_width", "扫频全宽 Δf (kHz)", 20, 1000),
                ("period", "扫频周期 T (ms)", 1, .001),
                ("dual_frequency", "双频参数 fd (Hz)", 1000, 1)):
            self.add_input(ff, key, label, value, scale)
        self.controls.addWidget(frequency_box)
        self.am_box = QGroupBox("AM 扩散模型")
        am = QFormLayout(self.am_box)
        for key, label, value, scale in (("am_t_ext", "t_ext (ms)", 100, .001),
                ("am_r0", "r0 (mm)", 10, .001), ("am_delta0", "δ0 (mm)", 5, .001),
                ("am_k_const", "模型系数 k", 1, 1)):
            self.add_input(am, key, label, value, scale)
        self.controls.addWidget(self.am_box)
        plotting = QGroupBox("绘图与到达时间")
        pf = QFormLayout(plotting)
        self.plot_kind = QComboBox()
        for label, key in (("踢角与包络", "wave"), ("FM 频率", "frequency"), ("采样频谱", "spectrum"), ("AM 因子", "am")):
            self.plot_kind.addItem(label, key)
        self.plot_kind.setCurrentIndex(self.plot_kind.findData("frequency"))
        pf.addRow("显示", self.bind(self.plot_kind))
        self.show_turns = QCheckBox("叠加逐圈采样点")
        self.show_turns.toggled.connect(self.schedule)
        pf.addRow(self.show_turns)
        for key, label, value, scale in (("start_time", "起始时间 (ms)", 0, .001),
                ("duration", "时间跨度 (ms)", 2, .001), ("reference_clock_start", "启动时参考钟 t0 (ms)", 0, .001),
                ("z_rel", "z_rel (m)", 0, 1)):
            self.add_input(pf, key, label, value, scale, -1e100 if key in ("reference_clock_start", "z_rel") else 0)
        self.controls.addWidget(plotting)
        self.results = ResultFields([
            ("f0", "回旋频率 f0 (kHz)", .001), ("cf", "中心频率 fc (kHz)", .001),
            ("width", "扫频全宽 Δf (kHz)", .001), ("tune", "中心激励 tune", 1),
            ("sweep_tune", "扫频 tune 全宽", 1), ("amplitude", "基准踢角 A0 (mrad)", 1000),
            ("peak", "采样最大 |踢角| (mrad)", 1000), ("rms", "采样 RMS 踢角 (mrad)", 1000),
            ("field", "基准电场 V/d (V/m)", 1), ("transit", "过板时间 (ns)", 1e9),
            ("rate", "采样率 (MHz)", 1e-6), ("resolution", "频谱间隔 (Hz)", 1)])
        self.result_layout.setContentsMargins(0, 0, 0, 0)
        self.result_layout.addWidget(self.results)
        self.result_layout.addWidget(hint("按 PASS Exciter 的公式绘制外加信号；逐圈点是粒子实际采样值。此处不计算束流响应或发射度增长。"))
        self.controls.addWidget(self.result_page)
        self.controls.addStretch()
        self.mode.currentIndexChanged.connect(self.change_mode)
        self.frequency_mode.currentIndexChanged.connect(self.change_frequency_mode)
        self.change_mode()

    def add_input(self, form, key, label, value, scale=1, minimum=0):
        widget = self.bind(number(value, minimum))
        self.inputs[key], self.scales[key] = widget, scale
        form.addRow(label, widget)

    def change_frequency_mode(self):
        if self.result is not None:
            values = {"excite_tune": self.result.central_frequency/self.result.frequency_0,
                "sweep_tune": self.result.sweep_width/self.result.frequency_0,
                "central_frequency": self.result.central_frequency, "sweep_width": self.result.sweep_width}
            for key, value in values.items():
                with QSignalBlocker(self.inputs[key]):
                    self.inputs[key].setValue(value/self.scales[key])
        self.change_mode()

    def change_mode(self):
        tune = self.frequency_mode.currentData() == "tune"
        for key in ("excite_tune", "sweep_tune", "central_frequency", "sweep_width"):
            self.frequency_form.setRowVisible(self.inputs[key], tune == (key in ("excite_tune", "sweep_tune")))
        self.frequency_form.setRowVisible(self.inputs["dual_frequency"], self.mode.currentData().startswith("dual"))
        self.am_box.setVisible(self.mode.currentData().endswith("_am"))
        self.recalculate()

    def recalculate(self):
        self.timer.stop()
        self.result = None
        try:
            settings = ExciterSettings(mode=self.mode.currentData(), frequency_mode=self.frequency_mode.currentData(),
                **{key: widget.value()*self.scales[key] for key, widget in self.inputs.items()})
            r = calculate_exciter(self.reference.current(), settings)
            colors = self.prepare_plot()
            kind = self.plot_kind.currentData()
            time = r.times*1000
            if kind == "wave":
                self.ax.plot(time, r.kick*1000, lw=.6, color=colors["accent"], label="Kick")
                self.ax.plot(time, r.envelope*1000, lw=1, color="#c98a42", label="Envelope")
                self.ax.plot(time, -r.envelope*1000, lw=1, color="#c98a42")
                if self.show_turns.isChecked():
                    self.ax.plot(r.turn_times*1000, r.turn_kick*1000, ".", ms=2, color=colors["text"], label="Turn samples")
                ylabel = "Kick (mrad)"
            elif kind == "frequency":
                self.ax.plot(time, r.lower_frequency/1000, color=colors["accent"], label="FM frequency" if settings.mode.startswith("single") else "Lower branch")
                if settings.mode.startswith("dual"):
                    self.ax.plot(time, r.upper_frequency/1000, color="#c98a42", label="Upper branch")
                ylabel = "Frequency (kHz)"
            elif kind == "am":
                self.ax.plot(time, r.am_factor, color=colors["accent"], label="AM factor (per turn)")
                ylabel = "AM factor"
            else:
                frequencies, amplitude = r.spectrum()
                self.ax.plot(frequencies/1000, amplitude*1000, color=colors["accent"], lw=.8, label="Hann amplitude spectrum")
                limit = max(abs(r.lower_frequency).max(), abs(r.upper_frequency).max(), 1/settings.duration)
                self.ax.set_xlim(0, min(r.sample_rate/2, 2*limit))
                ylabel = "Kick amplitude (mrad)"
            self.ax.set_xlabel("Frequency (kHz)" if kind == "spectrum" else "Elapsed time (ms)", color=colors["text"])
            self.ax.set_ylabel(ylabel, color=colors["text"])
            self.finish_plot()
            values = dict(f0=r.frequency_0, cf=r.central_frequency, width=r.sweep_width,
                tune=r.central_frequency/r.frequency_0, sweep_tune=r.sweep_width/r.frequency_0,
                amplitude=r.kick_amplitude, peak=float(abs(r.kick).max()), rms=float(np.sqrt(np.mean(r.kick**2))),
                field=settings.voltage/settings.gap, transit=settings.plate_length/r.velocity,
                rate=r.sample_rate, resolution=1/settings.duration)
            self.results.set_values(values)
            self.summary.setText(f"A0={r.kick_amplitude*1000:.9g} mrad · fc={r.central_frequency/1000:.9g} kHz · {len(r.times)} 个采样点")
            self.result = r
            self.set_valid({"settings_SI": asdict(settings), "results": self.results.snapshot()})
        except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError) as exc:
            self.set_error(exc)

    def export_rows(self):
        r = self.result
        if self.plot_kind.currentData() == "spectrum":
            return ("frequency_Hz", "kick_amplitude_rad"), zip(*r.spectrum())
        return ("elapsed_time_s", "arrival_time_s", "kick_rad", "envelope_rad", "am_factor", "fm_lower_Hz", "fm_upper_Hz"), zip(
            r.times, r.arrival_times, r.kick, r.envelope, r.am_factor, r.lower_frequency, r.upper_frequency)
