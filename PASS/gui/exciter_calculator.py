"""GUI Exciter calculation backend using the CPU command's exact FM/AM formulas.

FM and AM use the same particle time from the common trigger:
u=t_elapsed-z_rel/(beta*c), without folding z.
The signed kick angle directly sets each DDS amplitude; envelopes are magnitudes.
Voltage conversion is a separate calculation at the selected reference state.
It does not track beam response.
"""
from dataclasses import dataclass
import math
import warnings

import numpy as np

from PASS.gui.beam_calculator import Kinematics
from PASS.gui.optics_calculator import finite_number
from PASS.utils.program import LinearProgram


@dataclass(frozen=True)
class ExciterSettings:
    mode: str = "single_fm"
    circumference: float = 100.
    kick_angle: float = 1e-4
    frequency_mode: str = "tune"
    excite_tune: float = .47
    sweep_tune: float = .02
    central_frequency: float = 600000.
    sweep_width: float = 20000.
    period: float = .001
    dual_frequency: float | None = None
    am_t_ext: float = .1
    am_r0: float = .01
    am_delta0: float = .005
    am_k_const: float = 1.
    start_time: float = 0.
    duration: float = .002
    reference_clock_start: float = 0.
    z_rel: float = 0.
    dual_sweep_offset: float = .5


@dataclass(frozen=True)
class ExciterResult:
    settings: ExciterSettings
    frequency_0: float
    central_frequency: float
    sweep_width: float
    kick_amplitude: float
    velocity: float
    times: np.ndarray
    arrival_times: np.ndarray
    kick: np.ndarray
    envelope: np.ndarray
    am_factor: np.ndarray
    dds1_frequency: np.ndarray
    dds2_frequency: np.ndarray
    turn_times: np.ndarray
    turn_kick: np.ndarray
    sample_rate: float

    def spectrum(self):
        """One-sided Hann-window amplitude spectrum; includes DC, no detrend."""
        window = np.hanning(len(self.times))
        magnitude = 2 * np.abs(np.fft.rfft(self.kick * window)) / window.sum()
        magnitude[0] /= 2
        if len(self.times) % 2 == 0:
            magnitude[-1] /= 2
        return np.fft.rfftfreq(len(self.times), 1 / self.sample_rate), magnitude


def _am_factor(settings, elapsed, frequency_0):
    if not settings.mode.endswith("_am"):
        return np.ones_like(elapsed)
    if np.any(elapsed >= settings.am_t_ext):
        raise ValueError("AM 模型在 t_ext 发散；绘图区间必须止于 t_ext 之前。")
    exponent = math.exp(-(settings.am_r0 / settings.am_delta0)**2)
    complement = -math.expm1(-(settings.am_r0 / settings.am_delta0)**2)
    argument = elapsed / settings.am_t_ext * complement + exponent
    result = np.zeros_like(elapsed)
    valid = (elapsed >= 0) & (argument > 0)
    denominator = np.log(argument[valid])**2 * (settings.am_t_ext * exponent + elapsed[valid] * complement)
    if np.any(denominator <= 0):
        raise ValueError("AM 参数退化，无法计算有限振幅。")
    result[valid] = np.sqrt(settings.am_r0**2 * complement / denominator / frequency_0 / settings.am_k_const)
    return result


def exciter_waveform(settings, elapsed, frequency_0, velocity, amplitude, cf, width):
    """Return physical arrival, kick, envelope, AM factor and fixed DDS frequencies."""
    elapsed = np.asarray(elapsed, dtype=float)
    reference = float(elapsed.flat[0]) if elapsed.size else 0.
    offset = (elapsed - reference) - settings.z_rel / velocity
    time = reference + offset
    arrival = settings.reference_clock_start + time
    sweep_reference = np.remainder(reference, settings.period)
    tau = np.remainder(sweep_reference + offset, settings.period)
    am = _am_factor(settings, time, frequency_0)
    carrier = LinearProgram(cf).phase_cycles(reference, offset)
    sweep_cycles = width / (2 * settings.period) * tau * (tau - settings.period)
    cycles = carrier + sweep_cycles
    frequency = cf + width * (tau / settings.period - .5)
    if settings.mode.startswith("single"):
        signal = np.sin(2 * np.pi * np.remainder(cycles, 1))
        modulation = np.ones_like(time)
        dds1 = dds2 = frequency
    else:
        shift = np.remainder(settings.dual_sweep_offset, 1) * settings.period
        shifted_tau = np.remainder(sweep_reference + shift + offset, settings.period)
        shifted_sweep_cycles = width / (2 * settings.period) * (shifted_tau * (shifted_tau - settings.period) - shift * (shift - settings.period))
        shifted_cycles = carrier + shifted_sweep_cycles
        signal = np.sin(2 * np.pi * np.remainder(shifted_cycles, 1)) + np.sin(2 * np.pi * np.remainder(cycles, 1))
        modulation = 2 * np.cos(np.pi * (shifted_sweep_cycles - sweep_cycles))
        dds1 = cf + width * (shifted_tau / settings.period - .5)
        dds2 = frequency
    active = offset >= -reference
    kick = np.where(active, amplitude * am * signal, 0.)
    envelope = np.where(active, abs(amplitude * am * modulation), 0.)
    return arrival, kick, envelope, am, dds1, dds2


def voltage_to_kick_angle(kinematics: Kinematics, voltage: float, plate_length: float, gap: float) -> float:
    """Convert signed interplate voltage to a reference kick in rad (small-angle limit)."""
    if not kinematics.particle.charge_state or not kinematics.brho or not kinematics.velocity:
        raise ValueError("电压换算需要带电粒子且 Ek > 0。")
    voltage = finite_number(voltage, "带符号极板间峰值电压差")
    plate_length = finite_number(plate_length, "极板有效长", minimum=0)
    gap = finite_number(gap, "极板间距", positive=True)
    charge_sign = 1 if kinematics.particle.charge_state > 0 else -1
    return finite_number(charge_sign * voltage * plate_length / (gap * kinematics.velocity * kinematics.brho), "换算踢角")


def kick_angle_to_voltage(kinematics: Kinematics, kick_angle: float, plate_length: float, gap: float) -> float:
    """Convert a signed reference kick in rad to interplate voltage (small-angle limit)."""
    if not kinematics.particle.charge_state or not kinematics.brho or not kinematics.velocity:
        raise ValueError("电压换算需要带电粒子且 Ek > 0。")
    kick_angle = finite_number(kick_angle, "带符号基准踢角")
    plate_length = finite_number(plate_length, "极板有效长", positive=True)
    gap = finite_number(gap, "极板间距", positive=True)
    charge_sign = 1 if kinematics.particle.charge_state > 0 else -1
    return finite_number(charge_sign * kick_angle * gap * kinematics.velocity * kinematics.brho / plate_length, "换算电压")


def calculate_exciter(kinematics: Kinematics, settings: ExciterSettings, *, max_samples=200000):
    if settings.mode not in ("single_fm", "single_fm_am", "dual_fm", "dual_fm_am"):
        raise ValueError("请选择 PASS 支持的 FM/AM 激励模式。")
    if settings.frequency_mode not in ("tune", "frequency"):
        raise ValueError("频率输入应为 tune 或 frequency。")
    if not kinematics.velocity:
        raise ValueError("激励预览需要 Ek > 0。")
    for key in ("circumference", "period", "duration"):
        finite_number(getattr(settings, key), key, positive=True)
    amplitude = finite_number(settings.kick_angle, "带符号基准踢角")
    finite_number(settings.start_time, "绘图起始时间", minimum=0)
    for key in ("reference_clock_start", "z_rel"):
        finite_number(getattr(settings, key), key)
    f0 = kinematics.velocity / settings.circumference
    if settings.frequency_mode == "tune":
        cf = finite_number(settings.excite_tune, "激励 tune", minimum=0) * f0
        width = finite_number(settings.sweep_tune, "扫频 tune 全宽", minimum=0) * f0
    else:
        cf = finite_number(settings.central_frequency, "中心频率", minimum=0)
        width = finite_number(settings.sweep_width, "扫频全宽", minimum=0)
    if settings.mode.startswith("dual"):
        offset = finite_number(settings.dual_sweep_offset, "DDS 扫频错开比例", minimum=0)
        if offset > 1:
            raise ValueError("DDS 扫频错开比例必须在 0 到 1 之间。")
        if settings.dual_frequency is not None:
            frequency = finite_number(settings.dual_frequency, "旧双频参数", minimum=0)
            if not math.isclose(frequency * settings.period, 1, rel_tol=1e-12, abs_tol=1e-12):
                warnings.warn("旧双频参数 fd 与 1/T 不一致；该参数已停用，按扫频周期和 DDS 错开比例计算。", UserWarning, stacklevel=2)
    if settings.mode.endswith("_am"):
        for key in ("am_t_ext", "am_r0", "am_delta0", "am_k_const"):
            finite_number(getattr(settings, key), key, positive=True)
        if settings.start_time + settings.duration - settings.z_rel / kinematics.velocity >= settings.am_t_ext:
            raise ValueError("AM 模型在 t_ext 发散；请缩短绘图时间或增大 t_ext。")
    # The FFT is a finite sampled signal, not an analytic bandwidth.
    bound = cf + width / 2
    needed = max(2048, math.ceil(settings.duration * bound * 24))
    if needed > max_samples:
        raise ValueError("当前频率下绘图窗口过长；请缩短时间跨度（最多 200000 个采样点）。")
    times = settings.start_time + np.arange(needed) * (settings.duration / needed)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        arrays = exciter_waveform(settings, times, f0, kinematics.velocity, amplitude, cf, width)
        start_turn = math.ceil(settings.start_time * f0)
        stop_turn = math.ceil((settings.start_time + settings.duration) * f0)
        if stop_turn - start_turn > max_samples:
            raise ValueError("绘图窗口包含过多圈数，请缩短时间跨度。")
        turn_times = np.arange(start_turn, stop_turn) / f0
        turn_kick = exciter_waveform(settings, turn_times, f0, kinematics.velocity, amplitude, cf, width)[1]
        if not all(np.all(np.isfinite(a)) for a in (*arrays, turn_kick)):
            raise ValueError("激励参数超出有限数值范围。")
    return ExciterResult(settings, f0, cf, width, amplitude, kinematics.velocity, times, *arrays[:4], arrays[4], arrays[5], turn_times, turn_kick,
                         needed / settings.duration)
