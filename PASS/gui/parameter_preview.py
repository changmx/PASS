"""Read-only physical waveform and electrode previews for configuration forms."""
from copy import deepcopy
from pathlib import Path
import math

import numpy as np
from PySide6.QtWidgets import QDialog, QFormLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PASS.gui.project import resolved_file


def bump_preview_error(error):
    """Present form errors in Chinese without exposing Pydantic diagnostics."""
    from pydantic import ValidationError
    labels = {"Waveform file": "波形文件", "Time mode": "时间模式", "Time offset (s)": "时间偏移（秒）",
              "S (m)": "位置（米）", "Length (m)": "元件长度（米）", "Num slices": "本体切片数",
              "Enable": "启用开关", "Space charge": "内部空间电荷"}
    if isinstance(error, ValidationError):
        lines = []
        for issue in error.errors():
            path = issue["loc"]
            field = " / ".join(labels.get(str(part), str(part)) for part in path) or "Bump 参数"
            kind, context = issue["type"], issue.get("ctx", {})
            if path == ("Waveform file",):
                message = "请先选择波形文件（TFS 格式）"
            elif kind == "literal_error" and path == ("Time mode",):
                message = "请选择规定参考时钟（reference）或实际粒子到达时间（particle）"
            elif kind in {"greater_than", "greater_than_equal", "less_than", "less_than_equal"}:
                relation, bound = {"greater_than": ("大于", "gt"), "greater_than_equal": ("大于或等于", "ge"),
                                   "less_than": ("小于", "lt"), "less_than_equal": ("小于或等于", "le")}[kind]
                message = f"必须{relation} {context[bound]}"
            elif kind in {"int_type", "int_parsing"}:
                message = "请填写整数"
            elif kind in {"finite_number", "float_type", "float_parsing"}:
                message = "请填写有限数值"
            elif kind == "missing":
                message = "请填写此必填参数"
            elif "Internal Space charge requires a positive element length" in issue["msg"]:
                message = "启用内部空间电荷时，元件长度必须大于零"
            else:
                message = "参数无效，请检查数值范围及选项之间的关系"
            lines.append(f"{field}：{message}。")
        return "\n".join(lines)
    message = str(error)
    for name, label in labels.items():
        message = message.replace(name, label)
    if any("\u4e00" <= char <= "\u9fff" for char in message):
        return message
    return "无法预览 Bump 波形，请检查波形文件及参数；文件须包含 TIME、HKICK、VKICK 三列有限数值。"


def read_bump_preview(command, base_dir):
    """Use the tracking reader, translating its file errors at the GUI boundary."""
    from PASS.utils.bump_waveform import read_bump_waveform
    from tfs.errors import AbsentColumnNameError, AbsentColumnTypeError, TfsFormatError
    value = command.get("Waveform file")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("请先选择 Bump 波形文件（TFS 格式）。")
    path = resolved_file(value, Path(base_dir))
    if path.is_dir():
        raise ValueError(f"所选路径是文件夹，请选择 TFS 波形文件：\n{path}")
    try:
        return read_bump_waveform(path)
    except FileNotFoundError as exc:
        raise ValueError(f"找不到 Bump 波形文件，请检查路径：\n{path}") from exc
    except PermissionError as exc:
        raise ValueError(f"无法读取 Bump 波形文件，请检查读取权限或文件占用情况：\n{path}") from exc
    except OSError as exc:
        raise ValueError(f"读取 Bump 波形文件失败，请检查文件路径和磁盘状态：\n{path}") from exc
    except AbsentColumnNameError as exc:
        raise ValueError("波形文件缺少 TFS 列名行，请添加：\n* TIME HKICK VKICK") from exc
    except AbsentColumnTypeError as exc:
        raise ValueError("波形文件缺少 TFS 列类型行，请在列名行后添加：\n$ %le %le %le") from exc
    except TfsFormatError as exc:
        raise ValueError("波形文件的 TFS 格式无效，请检查表头、列名及类型声明。") from exc
    except KeyError as exc:
        raise ValueError("波形文件缺少必需列：TIME（时间）、HKICK（水平踢量）、VKICK（垂直踢量）。请检查列名及大小写。") from exc
    except UnicodeError as exc:
        raise ValueError("波形文件的文本编码无法读取，请将文件保存为 UTF-8 编码。") from exc
    except (ValueError, TypeError) as exc:
        message = str(exc)
        if message == "Bump requires at least two finite rows with strictly increasing TIME":
            translated = "波形文件至少需要两行有限数值；TIME 必须严格递增，不能重复或倒序。"
        elif message == "Bump TIME_UNIT must be s":
            translated = "波形文件的时间单位必须为秒，请将 TIME_UNIT 设为 s，并按秒换算时间数据。"
        elif message == "Bump KICK_CONVENTION must be delta_p_over_p0":
            translated = "波形文件的踢量约定必须为积分动量变化 ΔP/P0；KICK_CONVENTION 应设为 delta_p_over_p0。"
        else:
            translated = "波形表格无法解析，请检查列数及数据类型；TIME、HKICK、VKICK 必须填写数值。"
        raise ValueError(translated) from exc


def rf_waveforms(command, data, base_dir):
    from PASS.commands.element.rfcavity import Waveform
    from PASS.gui.clock import reference_clock_snapshot
    from PASS.utils.program import LinearProgram
    clock = reference_clock_snapshot(data)
    reference = LinearProgram(clock["Revolution frequency (Hz)"], clock["Time (s)"], origin=clock["Time origin (s)"])
    components = deepcopy(command["Components"])
    for item in components:
        if item.get("Program file"):
            item["Program file"] = str(resolved_file(item["Program file"], Path(base_dir)))
    return [Waveform(item, reference) for item in components]


def sample_rf(waveforms, start, end):
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("时间窗口需要有限的开始、结束值且结束 > 开始")
    rate = max(float(np.max(w.frequency.values)) * w.harmonic
               + float(np.max(np.abs(w.phase.slopes))) / (2 * np.pi) for w in waveforms)
    count = max(1000, math.ceil((end-start) * rate * 32) + 1)
    if count > 200000:
        raise ValueError("窗口内频率过高；请缩短时间窗口（每周期至少 32 点，上限 200000 点）")
    time = np.linspace(start, end, count)
    values = [np.broadcast_to(w.value(start, time-start), time.shape) for w in waveforms]
    return time, values


class ParameterPreview(QDialog):
    def __init__(self, command, data, base_dir, parent=None):
        # Reject bad files before allocating a dialog/canvas, so failed previews
        # do not leave partially constructed widgets attached to the form.
        samples = read_bump_preview(command, base_dir) if command.get("Command") == "Bump" else None
        super().__init__(parent)
        self.command = command
        self.setWindowTitle("预览ES" if command["Command"] == "ElSeparator" else command["Command"] + " 参数预览")
        self.resize(940, 650)
        root = QVBoxLayout(self)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.status = QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        kind = command["Command"]
        if kind == "RFCavity":
            self.waveforms = rf_waveforms(command, data, base_dir)
            start = self.waveforms[0].frequency.origin
            rate = min(float(w.frequency.value(start)) * w.harmonic for w in self.waveforms)
            self.start, self.end = QLineEdit(str(start)), QLineEdit(str(start + 2/rate))
            form = QFormLayout()
            form.addRow("物理时间开始 / s", self.start)
            form.addRow("物理时间结束 / s", self.end)
            root.addLayout(form)
            refresh = QPushButton("更新波形")
            refresh.clicked.connect(self.draw_rf)
            root.addWidget(refresh)
            self.draw_rf()
        elif kind == "Bump":
            times = samples[:, 0] - command.get("Time offset (s)", 0.)
            pad = .05*(times[-1]-times[0])
            ax = self.figure.subplots()
            for i, label in enumerate(("HKICK", "VKICK"), 1):
                ax.plot(times, samples[:, i], label=label)
                ax.plot([times[0]-pad, times[0]], [0, 0], color="gray", linestyle="--")
                ax.plot([times[-1], times[-1]+pad], [0, 0], color="gray", linestyle="--")
            ax.set(xlabel="Sampling time before offset (s)", ylabel="Integrated kick: delta P / P0")
            ax.grid(alpha=.25)
            ax.legend()
            self.status.setText("TFS 线性插值，端点包含在内、表外为零。横轴已减 Time offset；reference 使用规定时钟的圈时刻，particle 使用实际局部粒子到达时间。此图是输入波形，不是跟踪轨迹。")
        else:
            self.draw_separator()
        root.addWidget(NavigationToolbar2QT(self.canvas, self))
        root.addWidget(self.canvas, 1)

    def draw_rf(self):
        try:
            time, values = sample_rf(self.waveforms, float(self.start.text()), float(self.end.text()))
            self.figure.clear()
            ax = self.figure.subplots()
            for i, value in enumerate(values):
                ax.plot(time, value, label=f"Component {i+1}", alpha=.7)
            ax.plot(time, np.sum(values, axis=0), label="Sum", color="black", linewidth=1.6)
            ax.set(xlabel="Physical time (s)", ylabel="Physical voltage (V)")
            ax.grid(alpha=.25)
            ax.legend()
            self.status.setText("按规定频率的积分、相位及线性插值计算各分量与总电压；显示有符号物理电压，尚未乘粒子电荷。不推算同步桶、质心或束流响应。")
        except (ValueError, TypeError, OverflowError) as exc:
            self.figure.clear()
            self.status.setText(str(exc))
        self.canvas.draw_idle()

    def draw_separator(self):
        from matplotlib.patches import Polygon
        from PASS.para.schema.elements import ElSeparatorElement
        keys = ("Gap (m)", "Electrode height (m)", "Septum position (m)",
                "Septum thickness (m)", "Electrode center (m)", "Tilt (rad)")
        # Reuse the physical geometry checks, independently of tracking settings.
        # An unspecified voltage permits geometry only; never report it as zero.
        voltage = self.command.get("Voltage (V)")
        p = ElSeparatorElement.model_validate({"S (m)": 0., "Voltage (V)": 0. if voltage is None else voltage,
            **{k: self.command[k] for k in keys if k in self.command}}).model_dump(by_alias=True)
        d, thickness, gap = (p[k] for k in ("Septum position (m)", "Septum thickness (m)", "Gap (m)"))
        center, height, angle = (p[k] for k in ("Electrode center (m)", "Electrode height (m)", "Tilt (rad)"))
        ax = self.figure.subplots()
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for left, right, color, label in (
            (d, d+thickness, "#bb5555", "Septum"),
            (d+thickness, d+thickness+gap, "#abd9ec", "Open field gap"),
            (d+thickness+gap, d+thickness+gap+.3*gap, "#777777", "Counter electrode (continues)"),
        ):
            vertices = np.array([[left, center-height/2], [right, center-height/2],
                                 [right, center+height/2], [left, center+height/2]]) @ rotation.T
            ax.add_patch(Polygon(vertices, closed=True, facecolor=color, edgecolor=color, label=label))
        ax.plot(0, 0, "+", color="black", label="Beam origin")
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(xlabel="Beam x (m)", ylabel="Beam y (m)")
        ax.grid(alpha=.25)
        ax.legend()
        field = ("电压未填写，仅预览几何，不计算电场" if voltage is None else
                 f"开口场隙内 Eu=V/g={p['Voltage (V)']/gap:.6g} V/m")
        self.status.setText(f"电极截面（真空孔径未绘制）。{field}；对电极向外无限延伸，仅截取一段显示。电极边界吸收粒子，零电压仍有材料损失。")
