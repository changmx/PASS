"""GUI Exciter calculation backend using the CPU command's exact FM/AM formulas.

Elapsed time controls the effective turn for AM. Absolute arrival time controls
FM: t_arrive=T_start+t_elapsed-z_rel/(beta*c), without folding z.
The preview uses fixed reference kinematics; it does not track beam response.
"""
from dataclasses import dataclass
import math
import numpy as np

from PASS.gui.beam_calculator import Kinematics
from PASS.gui.optics_calculator import finite_number


@dataclass(frozen=True)
class ExciterSettings:
    mode: str = "single_fm"
    circumference: float = 100.
    voltage: float = 1000.
    gap: float = .05
    plate_length: float = .5
    frequency_mode: str = "tune"
    excite_tune: float = .47
    sweep_tune: float = .02
    central_frequency: float = 600000.
    sweep_width: float = 20000.
    period: float = .001
    dual_frequency: float = 1000.
    am_t_ext: float = .1
    am_r0: float = .01
    am_delta0: float = .005
    am_k_const: float = 1.
    start_time: float = 0.
    duration: float = .002
    reference_clock_start: float = 0.
    z_rel: float = 0.


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
    lower_frequency: np.ndarray
    upper_frequency: np.ndarray
    turn_times: np.ndarray
    turn_kick: np.ndarray
    sample_rate: float

    def spectrum(self):
        """One-sided Hann-window amplitude spectrum; includes DC, no detrend."""
        window = np.hanning(len(self.times))
        magnitude = 2*np.abs(np.fft.rfft(self.kick*window))/window.sum()
        magnitude[0] /= 2
        if len(self.times) % 2 == 0:
            magnitude[-1] /= 2
        return np.fft.rfftfreq(len(self.times), 1/self.sample_rate), magnitude


def _am_factor(settings, elapsed, frequency_0):
    if not settings.mode.endswith("_am"):
        return np.ones_like(elapsed)
    # AM is constant during each turn, exactly as Exciter._kick_am_vary.
    model_time = np.floor(elapsed*frequency_0 + 1e-10)/frequency_0
    if np.any(model_time >= settings.am_t_ext):
        raise ValueError("AM 模型在 t_ext 发散；绘图区间必须止于 t_ext 之前。")
    exponent = math.exp(-(settings.am_r0/settings.am_delta0)**2)
    complement = -math.expm1(-(settings.am_r0/settings.am_delta0)**2)
    argument = model_time/settings.am_t_ext*complement + exponent
    result = np.zeros_like(elapsed)
    valid = argument > 0
    denominator = np.log(argument[valid])**2 * (settings.am_t_ext*exponent + model_time[valid]*complement)
    if np.any(denominator <= 0):
        raise ValueError("AM 参数退化，无法计算有限振幅。")
    result[valid] = np.sqrt(settings.am_r0**2 * complement/denominator/frequency_0/settings.am_k_const)
    return result


def exciter_waveform(settings, elapsed, frequency_0, velocity, amplitude, cf, width):
    """Return arrival time, kick, magnitude envelope, AM factor, two FM branches."""
    elapsed = np.asarray(elapsed, dtype=float)
    arrival = settings.reference_clock_start + elapsed - settings.z_rel/velocity
    tau = arrival - np.floor(arrival/settings.period)*settings.period
    am = _am_factor(settings, elapsed, frequency_0)
    if settings.mode.startswith("single"):
        phase = 2*np.pi*cf*tau + np.pi*width/settings.period*tau*(tau-settings.period)
        modulation = np.ones_like(tau)
        carrier = cf + width*(tau/settings.period-.5)
        low = high = carrier
    else:
        half = settings.period/2
        first = tau <= half
        phase = np.where(first,
            2*np.pi*cf*tau + np.pi*width*(settings.dual_frequency*tau-.5)*tau,
            2*np.pi*cf*tau + np.pi*width*(tau-half)*(settings.dual_frequency*tau-1))
        modulation = 2*np.cos(np.pi*.5*width*tau)
        carrier = cf + width*settings.dual_frequency*tau - np.where(first, width/4,
                                                                 width/2+width*settings.dual_frequency*settings.period/4)
        low, high = carrier-width/4, carrier+width/4
    kick = amplitude*am*modulation*np.sin(phase)
    return arrival, kick, abs(amplitude*am*modulation), am, low, high


def calculate_exciter(kinematics: Kinematics, settings: ExciterSettings, *, max_samples=200000):
    if settings.mode not in ("single_fm", "single_fm_am", "dual_fm", "dual_fm_am"):
        raise ValueError("请选择 PASS 支持的 FM/AM 激励模式。")
    if settings.frequency_mode not in ("tune", "frequency"):
        raise ValueError("频率输入应为 tune 或 frequency。")
    if not kinematics.particle.charge_state or not kinematics.brho or not kinematics.velocity:
        raise ValueError("激励计算需要带电粒子且 Ek > 0。")
    for key in ("circumference", "gap", "plate_length", "period", "duration"):
        finite_number(getattr(settings, key), key, positive=True)
    finite_number(settings.voltage, "电压", minimum=0)
    finite_number(settings.start_time, "绘图起始时间", minimum=0)
    for key in ("reference_clock_start", "z_rel"):
        finite_number(getattr(settings, key), key)
    f0 = kinematics.velocity/settings.circumference
    if settings.frequency_mode == "tune":
        cf = finite_number(settings.excite_tune, "激励 tune", minimum=0)*f0
        width = finite_number(settings.sweep_tune, "扫频 tune 全宽", minimum=0)*f0
    else:
        cf = finite_number(settings.central_frequency, "中心频率", minimum=0)
        width = finite_number(settings.sweep_width, "扫频全宽", minimum=0)
    if settings.mode.startswith("dual"):
        finite_number(settings.dual_frequency, "双频参数", minimum=0)
    if settings.mode.endswith("_am"):
        for key in ("am_t_ext", "am_r0", "am_delta0", "am_k_const"):
            finite_number(getattr(settings, key), key, positive=True)
        if settings.start_time+settings.duration >= settings.am_t_ext:
            raise ValueError("AM 模型在 t_ext 发散；请缩短绘图时间或增大 t_ext。")
    # Bound smooth branch frequencies over one sweep; resets add broadband
    # content. The FFT is a finite sampled signal, not an analytic bandwidth.
    bound = cf+width/2 if settings.mode.startswith("single") else cf+width*(settings.dual_frequency*settings.period+1)
    needed = max(2048, math.ceil(settings.duration*bound*24))
    if needed > max_samples:
        raise ValueError("当前频率下绘图窗口过长；请缩短时间跨度（最多 200000 个采样点）。")
    times = settings.start_time + np.arange(needed)*(settings.duration/needed)
    amplitude = settings.voltage*settings.plate_length/(settings.gap*kinematics.velocity*kinematics.brho)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        arrays = exciter_waveform(settings, times, f0, kinematics.velocity, amplitude, cf, width)
        start_turn = math.ceil(settings.start_time*f0)
        stop_turn = math.ceil((settings.start_time+settings.duration)*f0)
        if stop_turn-start_turn > max_samples:
            raise ValueError("绘图窗口包含过多圈数，请缩短时间跨度。")
        turn_times = np.arange(start_turn, stop_turn)/f0
        turn_kick = exciter_waveform(settings, turn_times, f0, kinematics.velocity, amplitude, cf, width)[1]
        if not all(np.all(np.isfinite(a)) for a in (*arrays, turn_kick)):
            raise ValueError("激励参数超出有限数值范围。")
    return ExciterResult(settings, f0, cf, width, amplitude, kinematics.velocity, times,
                         *arrays[:4], arrays[4], arrays[5], turn_times, turn_kick, needed/settings.duration)
