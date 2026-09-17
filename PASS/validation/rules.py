"""Static checks for every currently registered PASS command.

Never construct a Simulation, particle pool, Poisson matrix, or output folder.
Schema defaults are used for analysis only; missing engine-required fields are
reported rather than silently repairing the input being inspected.
"""
from functools import lru_cache
import math
from pathlib import Path
import re
from typing import Annotated, get_args, get_origin

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from PASS.para.schema.bunch import BunchConfig, InjectionItem, OffsetConfig
from PASS.para.schema.elements import ELEMENT_REGISTRY
from PASS.para.schema.main import MainConfig
from PASS.para.schema.monitors import DistMonitor, ParticleMonitor, PhaseAdvanceMonitor, StatMonitor
from PASS.para.schema.slicer import Slicer
from PASS.para.schema.wake_field import WakeField, WakeFieldConfig, resolve_wake_point
from PASS.para.schema.space_charge import SpaceCharge, SpaceChargeConfig, SpaceChargeResourceConfig, validate_loss_aperture
from PASS.para.schema.twiss import TwissPoint
from .report import ValidationReport, parse_json


class SortBunchModel(BaseModel):
    s: float = Field(alias="S (m)")
    command: str = Field(default="SortBunch", alias="Command")


class TwissModel(TwissPoint):
    # The engine supports these fields in addition to the optics schema.
    aperture_type: str = Field(default="off", alias="Aperture type")
    aperture_value: list = Field(default_factory=list, alias="Aperture value")


MODELS = {model.model_fields["command"].default: model for model in ELEMENT_REGISTRY.values()}
MODELS.update(Injection=InjectionItem, Twiss=TwissModel, SortBunch=SortBunchModel,
              StatMonitor=StatMonitor, DistMonitor=DistMonitor, ParticleMonitor=ParticleMonitor,
              PhaseAdvanceMonitor=PhaseAdvanceMonitor, Slicer=Slicer, SpaceCharge=SpaceCharge,
              WakeField=WakeField)


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def integer(value):
    return type(value) is int


@lru_cache(maxsize=None)
def field_adapter(model, name):
    field = model.model_fields[name]
    annotation = Annotated[field.annotation, *field.metadata] if field.metadata else field.annotation
    return TypeAdapter(annotation)


def _nested_models(annotation):
    if get_origin(annotation) is dict:
        return None  # A named-resource mapping is not one child model.
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for argument in get_args(annotation):
        if isinstance(argument, type) and issubclass(argument, BaseModel):
            return argument
    return None


def json_location(value, location):
    """Remove Pydantic union branch labels that are not JSON path segments."""
    result = []
    for part in location:
        if isinstance(value, list) and isinstance(part, int):
            result.append(part)
            value = value[part] if 0 <= part < len(value) else None
        elif isinstance(value, dict) and part in value:
            result.append(part)
            value = value[part]
        elif isinstance(value, dict):
            if part == value.get("Kind", value.get("kind")):
                continue  # Discriminated-union branch label, not a JSON key.
            # A missing required nested key still has a meaningful JSON path.
            result.append(part)
            value = None
    return tuple(result)


class Validator:
    def __init__(self, data, base, report, check_files):
        self.data, self.base, self.report = data, Path(base), report
        self.check_files = check_files
        self.commands = {}
        self.resources = {}
        self.slice_sets = {}
        self.total_particles = 0
        self.last_injection = 0
        self.bunch_models = []
        self.file_cache = {}

    def add(self, path, code, message, warning=False):
        self.report.add(path, code, message, "warning" if warning else "error")

    def require(self, raw, fields, path):
        for key in fields:
            if key not in raw:
                self.add((*path, key), "field.required", "缺少引擎必需的字段")

    def model(self, model, raw, path, excluded=()):
        """Validate fields independently, so one failure never masks siblings."""
        if not isinstance(raw, dict):
            self.add(path, "field.object", "必须是对象")
            return {}
        known = {f.alias or name: (name, f) for name, f in model.model_fields.items() if name not in excluded}
        for key in raw.keys() - known.keys():
            self.add((*path, key), "field.unknown", "未知字段；请使用当前 JSON schema 中的名称")
        values, failed = {}, False
        for alias, (name, f) in known.items():
            if alias not in raw:
                if f.is_required():
                    self.add((*path, alias), "field.required", "缺少必需参数")
                    failed = True
                else:
                    default = f.get_default(call_default_factory=True)
                    values[alias] = default.model_dump(by_alias=True) if isinstance(default, BaseModel) else default
                continue
            value = raw[alias]
            child = _nested_models(f.annotation)
            if child and isinstance(value, dict):
                self.model(child, value, (*path, alias))
            try:
                field_adapter(model, name).validate_python(value, strict=True)
                values[alias] = value
            except ValidationError as exc:
                failed = True
                for issue in exc.errors():
                    self.add((*path, alias, *json_location(value, issue["loc"])), "field.type", issue["msg"])
        if not failed and not excluded:
            try:
                checked = model.model_validate(raw, strict=True)
                return checked.model_dump(by_alias=True)
            except ValidationError as exc:
                for issue in exc.errors():
                    self.add((*path, *json_location(raw, issue["loc"])), "field.constraint", issue["msg"])
        return values

    def numeric(self, data, key, path, *, minimum=None, positive=False):
        value = data.get(key)
        if value is None:
            return
        if not number(value):
            self.add((*path, key), "number.finite", "必须是有限数值")
        elif positive and value <= 0 or minimum is not None and value < minimum:
            self.add((*path, key), "number.range", "必须大于 0" if positive else f"必须 ≥ {minimum}")

    def choice(self, data, key, choices, path):
        if key in data and data[key] not in choices:
            self.add((*path, key), "field.choice", "可选值：" + ", ".join(sorted(choices)))

    def scan(self, value, path=()):
        if isinstance(value, dict):
            seen = set()
            for key, child in value.items():
                if not isinstance(key, str):
                    self.add(path, "json.key", "JSON 键必须是字符串")
                    continue
                if key.casefold() in seen:
                    self.add((*path, key), "json.duplicate", "键名存在大小写冲突；引擎会覆盖其中一个值")
                seen.add(key.casefold())
                self.scan(child, (*path, key))
        elif isinstance(value, list):
            for i, child in enumerate(value):
                self.scan(child, (*path, i))
        elif isinstance(value, float) and not math.isfinite(value):
            self.add(path, "json.nonfinite", "必须是有限数值；不允许 NaN、Infinity 或数值溢出")
        elif value is not None and type(value) not in (str, bool, int, float):
            self.add(path, "json.type", "不支持的 JSON 数据类型")

    def aperture(self, values, path):
        if "Aperture type" not in values:
            return
        kind, dims = values["Aperture type"], values.get("Aperture value", [])
        try:
            for row in dims:
                for v in (row if isinstance(row, list) else [row]):
                    if not number(v):
                        raise ValueError("孔径尺寸必须是有限数值，不接受字符串或布尔值")
            validate_loss_aperture(kind, dims)
            if kind in {"off", "default"} and dims:
                self.add((*path, "Aperture value"), "aperture.unused", "当前孔径类型不使用这些尺寸", True)
            if kind == "polygon":
                from .geometry import validate_polygon
                validate_polygon(dims)
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            self.add((*path, "Aperture value"), "aperture.geometry", str(exc))

    def turns(self, raw, path, *, analysis=False, flat=False):
        if analysis and type(raw) is int and raw == 0:
            return
        if not isinstance(raw, list):
            self.add(path, "turns.shape", "应为圈数范围列表" + ("，或 0（关闭分析）" if analysis else ""))
            return
        if flat and raw and all(integer(v) for v in raw):
            raw = [[v] for v in raw]
        seen = set()
        for i, row in enumerate(raw):
            p = (*path, i)
            if not isinstance(row, list) or len(row) not in ({2} if analysis else {1, 3}) or not all(integer(v) for v in row):
                self.add(p, "turns.shape", "需要两个整数 [开始圈, 结束圈)" if analysis else "需要 [单圈] 或 [开始圈, 结束圈, 步长]，且每项为整数")
                continue
            start, end = row[:2] if len(row) > 1 else (row[0], row[0])
            step = row[2] if len(row) == 3 else 1
            if start < 0 or end < start or step <= 0 or analysis and end - start < 2:
                self.add(p, "turns.range", "圈数必须非负、结束不早于开始、步长 > 0；相位分析至少连续两圈")
                continue
            if start >= self.turn_count or end > self.turn_count - (0 if analysis else 1):
                self.add(p, "turns.clipped", f"范围超出本次运行 [0, {self.turn_count})，引擎会裁剪或忽略", True)
            if analysis and min(end, self.turn_count) - start < 2:
                self.add(p, "turns.empty_analysis", "本次运行内不足连续两圈，无法进行相位分析", True)
            if tuple(row) in seen:
                self.add(p, "turns.duplicate", "重复的圈数范围", True)
            seen.add(tuple(row))

    def window(self, v, p, allow_minus_one=False):
        start, end = v.get("Start turn", 0), v.get("End turn", self.turn_count)
        if not integer(start) or not integer(end):
            return
        if allow_minus_one and end == -1:
            end = self.turn_count
        if start < 0 or end <= start:
            self.add(p, "turns.window", "需要 0 ≤ 开始圈 < 结束圈（结束圈不包含在内）")
        elif start >= self.turn_count or end > self.turn_count:
            self.add(p, "turns.clipped", f"执行窗口超出本次运行 [0, {self.turn_count})", True)

    def globals(self):
        raw = {k: v for k, v in self.data.items() if k not in {"Sequence", "Space charge", "Wake field"}}
        self.wake_config = None
        if "Wake field" in self.data:
            self.model(WakeFieldConfig, self.data["Wake field"], ("Wake field",))
            try:
                self.wake_config = WakeFieldConfig.model_validate(self.data["Wake field"])
            except (ValueError, TypeError):
                pass
        self.global_values = self.model(MainConfig, raw, ())
        g = self.global_values
        self.require(raw, ["Number of turns", "Number of Protons", "Number of Neutrons", "Number of Charges",
                           "Transition Gamma", "Circumference (m)", "Beam Name"], ())
        self.turn_count = g.get("Number of turns", 0)
        self.circumference = g.get("Circumference (m)", 0)
        if not integer(self.turn_count) or self.turn_count < 1:
            self.turn_count = 0
        if not number(self.circumference) or self.circumference <= 0:
            self.circumference = 0
        clock = g.get("Reference clock")
        if isinstance(clock, dict) and self.circumference:
            frequencies = clock.get("Revolution frequency (Hz)")
            frequencies = frequencies if isinstance(frequencies, list) else [frequencies]
            from PASS.utils.constants import const
            if frequencies and all(number(f) for f in frequencies) and max(frequencies) * self.circumference >= const.c:
                self.add(("Reference clock", "Revolution frequency (Hz)"), "clock.speed",
                         "规定回旋频率 × 周长必须小于光速")
        self.backend = g.get("Backend (gpu/cpu)", "cpu")
        self.choice(g, "Backend (gpu/cpu)", {"cpu", "gpu"}, ())
        self.numeric(g, "Transition Gamma", (), positive=True)
        charge = g.get("Number of Charges")
        if charge == 0:
            self.add(("Number of Charges",), "beam.charge", "电荷数不能为 0")
        if g.get("Number of Protons") == 0 and g.get("Number of Neutrons") == 0 and charge not in {-1, 1}:
            self.add(("Number of Charges",), "beam.species", "电子或正电子的电荷数必须为 -1 或 +1")
        if g.get("Number of Protons") == 1 and g.get("Number of Neutrons") == 0 and charge not in {-1, 1}:
            self.add(("Number of Charges",), "beam.species", "当前单质子质量模型只支持单位电荷数")
        if g.get("Is beam-beam"):
            self.add(("Is beam-beam",), "feature.unsupported", "当前引擎没有注册 BeamBeam 命令，开启此开关不会产生束束作用")
        ids = g.get("Device Id", [])
        if isinstance(ids, list) and all(integer(i) for i in ids):
            if any(i < 0 for i in ids) or len(set(ids)) != len(ids) or self.backend == "gpu" and not ids:
                self.add(("Device Id",), "gpu.ids", "GPU ID 必须非负且不重复；GPU 模式至少配置一个 ID")
            if self.backend == "gpu" and (len(ids) != g.get("Number of GPU devices") or len(ids) > 1):
                self.add(("Device Id",), "gpu.single_device", "当前进程只使用第一个 GPU；数量应与列表一致", True)
        output = g.get("Output directory")
        if isinstance(output, str) and output != "default":
            if not output.strip():
                self.add(("Output directory",), "output.empty", "输出目录不能留空")
            elif self.check_files:
                try:
                    p = Path(output).expanduser()
                    if not p.is_absolute():
                        p = self.base / p
                    if p.exists() and not p.is_dir():
                        self.add(("Output directory",), "output.directory", "输出路径是文件，必须选择目录")
                    elif any(parent.exists() and not parent.is_dir() for parent in p.parents):
                        self.add(("Output directory",), "output.parent", "输出路径的父级是文件")
                except (OSError, ValueError) as exc:
                    self.add(("Output directory",), "output.path", str(exc))

    def injection(self, raw, path):
        bunch_keys = sorted([k for k in raw if isinstance(k, str) and re.fullmatch(r"bunch\d+", k)], key=lambda k: int(k[5:]))
        root = {k: v for k, v in raw.items() if k not in bunch_keys}
        v = self.model(InjectionItem, root, path, excluded={"bunches"})
        self.require(root, ["S (m)", "Harmonic Number"], path)
        if v.get("S (m)") != 0:
            self.add((*path, "S (m)"), "injection.position", "Injection 必须位于 S = 0")
        harmonic = v.get("Harmonic Number")
        if integer(harmonic) and harmonic > 0 and (len(bunch_keys) != harmonic or any(key != f"bunch{i}" for i, key in enumerate(bunch_keys))):
            self.add(path, "injection.bunches", f"Harmonic Number={harmonic}，需要连续的 bunch0 至 bunch{harmonic - 1}；空桶也要声明")
        ids, start_index = [], 0
        weight_source = None
        for key in bunch_keys:
            p = (*path, key)
            b = self.model(BunchConfig, raw[key], p)
            if not isinstance(raw[key], dict):
                continue
            self.require(raw[key], ["Total Injection Turns", "Injection Interval", "Alpha x", "Alpha y", "Beta x (m)", "Beta y (m)",
                                   "Emittance x (m'rad)", "Emittance y (m'rad)", "Dx (m)", "Dpx", "Sigma z (m)", "Sigma dp/p",
                                   "Transverse dist", "Longitudinal dist", "Offset x", "Offset y", "Insert Particle Coordinate"], p)
            ids.append(b.get("Harmonic ID of this bunch", 0))
            self.numeric(b, "Kinetic Energy per Nucleon (eV/u)", p, positive=True)
            for field in ("Number of Real Particles", "Number of Macro Particles"):
                self.numeric(b, field, p, minimum=0)
            n, real = b.get("Number of Macro Particles", 0), b.get("Number of Real Particles", 0)
            n = max(n, 0) if integer(n) else 0
            if n > 0 and integer(real) and real >= 0:
                if weight_source is None:
                    weight_source = (real, n)
                elif real * weight_source[1] != weight_source[0] * n:
                    self.add(p, "injection.weight", "同一束流的所有非空束团必须具有相同且固定的真实粒子数/宏粒子数")
            self.total_particles += n
            if n == 0 and number(real) and real > 0:
                self.add(p, "injection.empty", "存在真实粒子却没有宏粒子，束流强度将无法表示")
            elif n > 0 and real == 0:
                self.add(p, "injection.test_particles", "宏粒子的权重为 0，可用于测试粒子，但不会产生空间电荷", True)
            stop, interval = b.get("Total Injection Turns", 1), b.get("Injection Interval", 1)
            events = (stop + interval - 1) // interval if integer(stop) and integer(interval) and min(stop, interval) > 0 else 1
            last = (events - 1) * interval if integer(interval) else 0
            self.last_injection = max(self.last_injection, last if n else 0)
            first = n // events + n % events
            if n and last >= self.turn_count:
                self.add(p, "injection.incomplete", f"最后一次注入在第 {last} 圈，本次运行结束前无法完成", True)
            rows = b.get("Insert Particle Coordinate", [])
            if len(rows) > first:
                self.add((*p, "Insert Particle Coordinate"), "injection.manual_count", f"手动粒子不能超过首次注入的宏粒子数 {first}（按注入间隔计算）")
            for i, row in enumerate(rows):
                if not isinstance(row, list) or len(row) != 6 or not all(number(x) for x in row):
                    self.add((*p, "Insert Particle Coordinate", i), "injection.coordinates", "手动粒子每行必须有 6 个有限数值：x, px, y, py, z_rel, dp")
                else:
                    if row[5] <= -1 or 1 + row[5] <= math.hypot(row[1], row[3]):
                        self.add((*p, "Insert Particle Coordinate", i), "injection.momentum", "要求 dp > -1 且 px² + py² < (1+dp)²，以保证纵向动量为实数")
            self.choice(b, "Transverse dist", {"gaussian", "kv", "uniform", "waterbag", "parabolic"}, p)
            self.choice(b, "Longitudinal dist", {"gaussian", "coasting", "matchz", "matchdp"}, p)
            if b.get("Transverse dist") == "gaussian" and not b.get("Is Load Distribution from File") and n:
                if any(b.get(f"Emittance {axis} (m'rad)", 0) <= 0 for axis in "xy"):
                    self.add(p, "injection.gaussian", "Gaussian generation requires positive Emittance x/y (m'rad); Gaussian 发射度必须大于 0")
            ddp, dde = b.get("Momentum Offset dp", 0), b.get("Kinetic Energy Offset (eV)", 0)
            if number(ddp) and number(dde):
                if ddp and dde:
                    self.add(p, "injection.offset_conflict", "Momentum Offset dp 与 Kinetic Energy Offset (eV) 只能有一项非零")
                if ddp <= -1:
                    self.add((*p, "Momentum Offset dp"), "injection.momentum", "动量偏移必须 > -1")
                energy = b.get("Kinetic Energy per Nucleon (eV/u)")
                if number(energy) and energy + dde <= 0:
                    self.add((*p, "Kinetic Energy Offset (eV)"), "injection.energy", "偏移后的动能必须大于 0")
            self.bunch_models.append((p, b))
            self.file(b, "Distribution File Path", p, "distribution", active=b.get("Is Load Distribution from File", False), minimum_rows=first if b.get("Distribution File Mode") == "repeat" else n)
            start_index += n
            for axis in "xy":
                offset = b.get(f"Offset {axis}")
                op = (*p, f"Offset {axis}")
                if isinstance(offset, dict):
                    self.choice(offset, "File Time Kind", {"turn", "second"}, op)
                    self.file(offset, "File Path", op, f"offset_{axis}", active=offset.get("Is Offset", False) and offset.get("Is Load From File", False))
                    if offset.get("Is Offset") and not offset.get("Is Load From File"):
                        self.require(raw[key].get(f"Offset {axis}", {}), ["Offset Position (m)", "Offset Momentum (rad)"], op)
        if integer(harmonic) and harmonic > 0 and all(integer(i) for i in ids):
            if len(set(ids)) != len(ids) or any(i < 0 or i >= harmonic for i in ids):
                self.add(path, "injection.harmonic_ids", f"Harmonic ID 必须不重复并覆盖 [0, {harmonic})")
        return v

    def file(self, values, key, path, kind, *, active=True, minimum_rows=0):
        from .files import check_table
        return check_table(self, values.get(key), (*path, key), kind, active, minimum_rows)

    def command(self, name, raw):
        p = ("Sequence", name)
        if not isinstance(raw, dict):
            self.add(p, "command.object", "command 必须是对象")
            return
        kind = raw.get("Command")
        if not isinstance(kind, str) or kind not in MODELS:
            self.add((*p, "Command"), "command.unknown", f"未知或缺少 Command：{kind!r}；可选值：{', '.join(MODELS)}")
            return
        if kind == "Injection":
            v = self.injection(raw, p)
        else:
            v = self.model(MODELS[kind], raw, p)
        if kind == "WakeField":
            try:
                v = resolve_wake_point(raw, self.wake_config)
            except (ValueError, TypeError) as exc:
                if raw.get("Configuration") is not None:
                    self.add((*p, "Configuration"), "wake.configuration", str(exc))
            # The file checker expects a list even for an invalid draft.
            if not isinstance(v.get("Groups"), list):
                v["Groups"] = []
        self.require(raw, ["S (m)"], p)
        required = {"Drift": ["Length (m)"], "SBend": ["Length (m)", "K0L"],
                    "Quadrupole": ["Length (m)"], "Sextupole": ["Length (m)"],
                    "Octupole": ["Length (m)"], "Multipole": ["Length (m)"],
                    "Solenoid": ["Length (m)"], "Exciter": ["Enable"]}
        self.require(raw, required.get(kind, []), p)
        self.commands[name] = (kind, v)
        s = v.get("S (m)")
        if number(s) and self.circumference and not 0 <= s <= self.circumference:
            self.add((*p, "S (m)"), "sequence.position", f"命令位置应在 [0, {self.circumference:g}] m 内")
        length = v.get("Length (m)", 0)
        if number(length) and number(s) and length > s + 1e-10:
            self.add((*p, "Length (m)"), "sequence.body_start", "S 是元件出口位置；该元件入口 S-Length 小于 0", True)
        if kind in {"Marker", "RFCavity", "Exciter", "ReorganizeBunch"} and length != 0:
            self.add((*p, "Length (m)"), "element.thin", "该命令是点操作，Length 必须为 0")
        self.aperture(v, p)
        if "Integrator" in v:
            self.choice(v, "Integrator", {"adaptive", "uniform", "yoshida4"}, p)
        if kind in {"SBend", "Quadrupole"}:
            self.choice(v, "Model", {"adaptive", "drift-kick-drift-exact", "rot-kick-rot" if kind == "SBend" else "mat-kick-mat"}, p)
        if kind == "SBend":
            self.numeric(v, "Hgap (m)", p, minimum=0)
            for edge in ("E1 (rad)", "E2 (rad)"):
                if number(v.get(edge)) and abs(math.cos(v[edge])) < 1e-12:
                    self.add((*p, edge), "sbend.edge", "端面角的 cos 接近 0，边缘聚焦公式奇异")
        if kind == "Multipole" and not v.get("KiL") and not v.get("KiSL"):
            self.add(p, "multipole.empty", "KiL 与 KiSL 至少有一项包含分量")
        if v.get("Is ramping"):
            self.add((*p, "Is ramping"), "feature.unsupported", "当前磁铁跟踪未实现 ramping；此开关和 ramping 文件不会更新磁场，请关闭")
        for key in v:
            if key.endswith(" ramping file"):
                self.file(v, key, p, "ramping", active=False)
        if "Is field error" in v and not v["Is field error"] and (v.get("Field error KNL") or v.get("Field error KSL")):
            self.add((*p, "Is field error"), "field_error.disabled", "场误差系数已填写，但场误差开关关闭", True)
        if kind == "Bump":
            from .files import resolve_input_paths
            from PASS.utils.bump_waveform import read_bump_waveform
            from tfs.errors import TfsFormatError
            values = {"Waveform file": v.get("Waveform file", "")}
            resolve_input_paths(values, self.base)
            if self.check_files:
                try:
                    read_bump_waveform(values["Waveform file"])
                except (ValueError, OSError, KeyError, TypeError, TfsFormatError) as exc:
                    self.add((*p, "Waveform file"), "bump.waveform", str(exc))
        if kind in {"Twiss", "PhaseAdvanceMonitor"}:
            for field in v:
                if field.startswith("Beta "):
                    self.numeric(v, field, p, positive=True)
        if kind == "Twiss":
            self.require(raw, ["Dx previous (m)", "Dpx previous", "DQx", "DQy", "Longitudinal transfer"], p)
            self.choice(v, "Longitudinal transfer", {"off", "drift", "matrix"}, p)
            previous = v.get("S previous (m)")
            if number(s) and number(previous) and (previous < 0 or previous > s):
                self.add((*p, "S previous (m)"), "twiss.position", "要求 0 ≤ S previous ≤ S；反向间隔会产生反向漂移")
        if "Save turns" in v:
            self.turns(v["Save turns"], (*p, "Save turns"), flat=kind == "SpaceCharge")
        if kind == "PhaseAdvanceMonitor":
            self.turns(v.get("Turn ranges", 0), (*p, "Turn ranges"), analysis=True)
            self.numeric(v, "Min action", p, minimum=0)
        if kind == "ParticleMonitor":
            self.numeric(v, "Max tag", p, minimum=0)
            self.window(v, p, allow_minus_one=True)
        if kind == "ReorganizeBunch":
            if v.get("New harmonic number") is None:
                self.add((*p, "New harmonic number"), "field.required", "必须指定重组后的谐波数")
            self.numeric(v, "Start turn", p, minimum=0)
        if kind == "RFCavity":
            pair = v.get("Dp aperture")
            if pair is not None and (len(pair) != 2 or not all(number(x) for x in pair) or pair[0] >= pair[1] or pair[0] < -1):
                self.add((*p, "Dp aperture"), "rf.dp_aperture", "需要 [-1 ≤ 下限 < 上限] 的两个有限数值")
            from .files import check_rf_files
            check_rf_files(self, v, p)
        if kind == "Exciter":
            self.exciter(v, p)
        if kind == "Slicer":
            self.slicer(v, p)
        if kind == "WakeField":
            from .files import check_wake_files
            location = ("Wake field", "Configurations", raw["Configuration"]) if raw.get("Configuration") is not None else p
            check_wake_files(self, v, location)
        internal = v.get("Space charge")
        if isinstance(internal, dict):
            self.aperture(internal, (*p, "Space charge"))
            self.turns(internal.get("Save turns", []), (*p, "Space charge", "Save turns"), flat=True)
            if number(length) and length <= 1e-10:
                self.add((*p, "Space charge"), "sc.thick", "元件内空间电荷要求 Length > 1e-10 m（引擎厚元件阈值）")

    def exciter(self, v, p):
        self.choice(v, "Mode", {"single_fm", "single_fm_am", "dual_fm", "dual_fm_am"}, p)
        self.choice(v, "Direction", {"x", "y"}, p)
        self.window(v, p)
        for key in ("Gap (m)", "Period (s)"):
            self.numeric(v, key, p, positive=True)
        self.numeric(v, "Plate length (m)", p, minimum=0)
        tune = [v.get(k) is not None for k in ("Excite tune", "Sweep tune")]
        freq = [v.get(k) is not None for k in ("Central frequency (Hz)", "Sweep width (Hz)")]
        if not (all(tune) and not any(freq) or all(freq) and not any(tune)):
            self.add(p, "exciter.frequency", "必须且只能填写一组完整的激励/扫频 tune，或中心频率/扫频宽度")
        for key in ("Excite tune", "Sweep tune", "Central frequency (Hz)", "Sweep width (Hz)", "FM dual frequency (Hz)"):
            self.numeric(v, key, p, minimum=0)
        if str(v.get("Mode", "")).endswith("_am"):
            for key in ("AM t ext (s)", "AM r0 (m)", "AM delta0", "AM k const"):
                self.numeric(v, key, p, positive=True)

    def slicer(self, v, p):
        self.choice(v, "Slice model", {"equal_length", "equal_particle", "equal_charge"}, p)
        self.choice(v, "Z range mode", {"auto", "explicit"}, p)
        self.choice(v, "Coordinate", {"z_rel", "z_periodic", "arrival_phase"}, p)
        name = v.get("Slice set")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            self.add((*p, "Slice set"), "slicer.name", "Slice set 名称不能为空或包含首尾空白")
            return
        explicit = v.get("Explicit")
        if isinstance(explicit, dict):
            for key in explicit.keys() - {"z min", "z max"}:
                self.add((*p, "Explicit", key), "field.unknown", "Explicit 只包含 z min 与 z max")
            if not all(number(explicit.get(k)) for k in ("z min", "z max")):
                self.add((*p, "Explicit"), "slicer.range", "z min 与 z max 必须为有限数值")
                return
        try:
            from PASS.commands.slicer import SliceSet
            candidate = SliceSet.from_command(name, {k.lower(): value for k, value in v.items()})
            if candidate.coordinate == "arrival_phase" and self.circumference:
                from PASS.utils.coordinates import ring_interval
                ring_interval(candidate.explicit.z_min, candidate.explicit.z_max, self.circumference)
                if candidate.coordinate == "arrival_phase" and candidate.explicit.z_max != 0.:
                    raise ValueError("arrival_phase requires Explicit [-circumference, 0]")
            if candidate.coordinate == "z_periodic" and self.circumference and candidate.explicit is not None:
                if candidate.explicit.z_min < -self.circumference / 2 or candidate.explicit.z_max > self.circumference / 2:
                    raise ValueError("z_periodic Explicit range must lie within [-C/2, C/2]")
            if name in self.slice_sets and self.slice_sets[name] != candidate.configuration():
                self.add(p, "slicer.conflict", f"多个 Slicer 对 Slice set {name!r} 的配置不一致")
            self.slice_sets[name] = candidate.configuration()
        except (ValueError, TypeError, KeyError) as exc:
            self.add(p, "slicer.config", str(exc))

    def run(self):
        self.scan(self.data)
        self.globals()
        sequence = self.data.get("Sequence")
        if not isinstance(sequence, dict):
            self.add(("Sequence",), "sequence.object", "Sequence 必须是对象")
            sequence = {}
        injection = sequence.get("injection")
        if not isinstance(injection, dict) or injection.get("Command") != "Injection":
            self.add(("Sequence", "injection"), "injection.missing", "未找到 Injection command，请在 Sequence.injection 中声明")
        for name, raw in sequence.items():
            if not isinstance(name, str):
                continue
            if not name.strip() or name != name.strip():
                self.add(("Sequence", name), "command.name", "命令名不能为空或包含首尾空白")
            if isinstance(raw, dict) and raw.get("Command") == "Injection" and name != "injection":
                self.add(("Sequence", name), "injection.unique", "只能有一个 Injection，且键名必须为 injection")
            self.command(name, raw)
        self.report.command_count = len(sequence)
        from .relations import check_relations
        check_relations(self)
        return self.report


def validate_input(data, base_dir=".", *, check_files=True):
    report = ValidationReport(full=check_files)
    if not isinstance(data, dict):
        report.add((), "json.root", "JSON 根节点必须是对象")
        return report
    return Validator(data, base_dir, report, check_files).run()


def validate_file(path, *, check_files=True):
    path = Path(path)
    report = ValidationReport(full=check_files)
    try:
        data, report = parse_json(path.read_bytes(), report)
    except (OSError, ValueError) as exc:
        report.add((), "file.read", str(exc))
        return report
    if data is not None:
        Validator(data, path.resolve().parent, report, check_files).run()
    return report


def validate_files(paths, *, check_files=True):
    """Aggregate every selected beam and verify shared execution settings."""
    from dataclasses import replace
    result = ValidationReport(full=check_files)
    inputs = []
    for path in paths:
        report = validate_file(path, check_files=check_files)
        result.diagnostics.extend(replace(d, source=str(path)) for d in report.diagnostics)
        result.command_count += report.command_count
        result.checked_files.extend(p for p in report.checked_files if p not in result.checked_files)
        try:
            data, parsed = parse_json(Path(path).read_bytes())
            if data is not None and parsed.ok:
                inputs.append((path, data))
        except OSError:
            pass
    if len(paths) not in (1, 2):
        result.add((), "run.inputs", "一次运行需要一个或两个束流输入")
    check_shared_inputs(inputs, result)
    return result


def check_shared_inputs(inputs, result):
    if len(inputs) == 2:
        try:
            first, second = [MainConfig.model_validate(data, strict=True).model_dump(by_alias=True) for _, data in inputs]
        except ValidationError:
            return  # Each input's field errors are already present in the report.
        for field in ("Number of turns", "Backend (gpu/cpu)", "Particle Precision", "Device Id", "Number of GPU devices", "Timing", "Is plot figure"):
            if first[field] != second[field]:
                result.add((field,), "run.mismatch", f"两个输入的 {field} 必须一致；引擎共享同一运行设置", source=str(inputs[1][0]))


def validate_documents(documents):
    """Validate unsaved project inputs before creating any run snapshot.

    Each document is (display_name, data, dependency_base_directory).
    """
    from dataclasses import replace
    report = ValidationReport()
    for name, data, base in documents:
        checked = validate_input(data, base)
        report.diagnostics.extend(replace(d, source=name) for d in checked.diagnostics)
        report.checked_files.extend(f for f in checked.checked_files if f not in report.checked_files)
        report.command_count += checked.command_count
    check_shared_inputs([(name, data) for name, data, _base in documents], report)
    return report
