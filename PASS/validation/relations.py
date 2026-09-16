"""Cross-command execution, space-charge geometry, and resource checks."""
import math
from types import SimpleNamespace

from PASS.para.schema.space_charge import SpaceChargeConfig, SpaceChargeResourceConfig
from .rules import number, integer


def check_relations(check):
    from PASS.commands import command_priority
    from PASS.utils.constants import const
    raw = check.data.get("Space charge", {})
    root = ("Space charge",)
    sc = check.model(SpaceChargeConfig, {k: v for k, v in raw.items() if k != "Configurations"}, root) if isinstance(raw, dict) else check.model(SpaceChargeConfig, raw, root)
    resources = raw.get("Configurations", {}) if isinstance(raw, dict) else {}
    enabled = sc.get("Enabled", False) is True
    if not isinstance(resources, dict):
        check.add((*root, "Configurations"), "field.object", "Configurations 必须是命名配置的对象")
        resources = {}
    for name, resource in resources.items():
        p = (*root, "Configurations", name)
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            check.add(p, "sc.name", "配置名称不能为空或包含首尾空白")
        values = check.model(SpaceChargeResourceConfig, resource, p)
        check_resource_combinations(check, values, p)
        try:
            check.resources[name] = SpaceChargeResourceConfig.model_validate(values, strict=True)
        except (ValueError, TypeError):
            pass
    ordered = [(name, kind, v) for name, (kind, v) in check.commands.items() if number(v.get("S (m)"))]
    ordered.sort(key=lambda row: (round(row[2]["S (m)"] / const.eps) if abs(row[2]["S (m)"]) < 1e290 else row[2]["S (m)"], command_priority(row[1])))
    valid_slices, used, contributions = set(), set(), []
    for name, kind, v in ordered:
        p = ("Sequence", name)
        if kind == "Slicer" and v.get("Slice set") in check.slice_sets:
            valid_slices.add(v["Slice set"])
        if kind in {"SortBunch", "ReorganizeBunch"}:
            start = v.get("Start turn", 0)
            if kind == "SortBunch" or integer(start) and 0 <= start < check.turn_count:
                valid_slices.clear()
            if kind == "ReorganizeBunch" and integer(start):
                if start >= check.turn_count:
                    check.add((*p, "Start turn"), "reorganize.inactive", "本次运行不会到达重组圈数", True)
                elif start < check.last_injection:
                    check.add(p, "reorganize.injection", "重组发生在注入结束之前，会改变后续注入所依赖的 bunch 数量或粒子索引")
        if kind == "WakeField" and v.get("Is enabled", True):
            slice_name = v.get("Slice set")
            if slice_name in check.slice_sets and check.slice_sets[slice_name][4] == "z_periodic":
                check.add(p, "wake.slice_coordinate", "WakeField 不能使用 z_periodic；请选择 z_rel 或 arrival_phase 切片")
            if slice_name not in check.slice_sets:
                check.add((*p, "Slice set"), "wake.slicer_missing", f"未定义 Slice set {slice_name!r}")
            elif slice_name not in valid_slices:
                check.add(p, "wake.slicer_order", "WakeField 之前必须运行对应的 Slicer")
        if kind == "ParticleMonitor":
            maximum = v.get("Max tag", 0)
            if integer(maximum):
                if maximum > check.total_particles:
                    check.add((*p, "Max tag"), "monitor.tags", f"Max tag 超过宏粒子总数 {check.total_particles}，会分配多余缓冲区", True)
                start, end = v.get("Start turn", 0), v.get("End turn", -1)
                if integer(start) and integer(end):
                    turns = max(0, min(check.turn_count, check.turn_count if end == -1 else end) - start)
                    columns = 14 if v.get("Include reference", False) else 11
                    size = max(0, maximum) * turns * columns * 8
                    if size > 512 * 1024**2:
                        check.add(p, "monitor.memory", f"仅坐标缓冲区预计占用 {size / 1024**3:.2f} GiB，请确认内存容量", True)
        internal = v.get("Space charge")
        if kind != "SpaceCharge" and not isinstance(internal, dict):
            continue
        cp = (*p, "Space charge") if isinstance(internal, dict) else p
        values = dict(internal) if isinstance(internal, dict) else v
        ref = values.get("Configuration")
        if ref not in resources:
            check.add((*cp, "Configuration"), "sc.reference", f"未定义 Space charge configuration {ref!r}")
        if not enabled:
            check.add(cp, "sc.disabled", "全局 Space charge.Enabled 已关闭，此配置不会产生空间电荷作用", True)
        else:
            used.add(ref)
        config = check.resources.get(ref)
        if config is None:
            continue
        slice_name = config.slice_set
        length = v.get("Length (m)", 0) if isinstance(internal, dict) else v.get("SC length (m)", 0)
        if slice_name not in check.slice_sets:
            check.add((*root, "Configurations", ref, "Slice set"), "sc.slicer_missing", f"未定义 Slice set {slice_name!r}；请添加对应 Slicer")
        elif enabled and number(length) and length > 0 and slice_name not in valid_slices:
            check.add(cp, "sc.slicer_order", f"执行到此命令前 Slice set {slice_name!r} 尚未计算，或已被 SortBunch/ReorganizeBunch 失效；请在其后、SC 之前放置 Slicer")
        if slice_name in check.slice_sets and check.slice_sets[slice_name][4] != "z_periodic":
            check.add(cp, "sc.slice_coordinate", "SpaceCharge 只接受 Coordinate=z_periodic 切片；请显式设置对应 Slicer")
        if config.method != "pic" and values.get("Save potential"):
            check.add((*cp, "Save potential"), "sc.analytic_potential", "解析空间电荷不支持保存电势，可保存场或密度")
        if isinstance(internal, dict):
            parent_kind, parent_dims = v.get("Aperture type", "off"), v.get("Aperture value", [])
            if parent_kind == "default":
                parent_kind, parent_dims = "rectangle", [1.0, 1.0]
            if values.get("Aperture type", "default") != "default" and (values.get("Aperture type"), values.get("Aperture value")) != (parent_kind, parent_dims):
                check.add((*cp, "Aperture type"), "sc.aperture_override", "元件内 SC 将使用父元件的孔径，覆盖这里的不同设置", True)
            values.update({"Aperture type": parent_kind, "Aperture value": parent_dims})
        try:
            from PASS.commands.solver.pic import build_grid_geometry
            from PASS.commands.space_charge import _resolve_aperture
            from .geometry import has_interior_node
            grid = build_grid_geometry(config.model_dump(by_alias=True))
            owner = SimpleNamespace(geometry=grid, configuration=config)
            kind_aper, dims = _resolve_aperture(owner, {k.lower(): value for k, value in values.items()})
            if enabled and check.check_files and config.solver in {"fd_dirichlet", "dst_dirichlet"} and not has_interior_node(grid, kind_aper, dims):
                check.add((*cp, "Aperture value"), "sc.grid_empty", "孔径内没有有效网格节点，请提高网格分辨率或扩大孔径")
            if grid.nx * grid.ny > 4_000_000:
                check.add((*root, "Configurations", ref), "sc.grid_memory", f"横向网格有 {grid.nx * grid.ny:,} 个节点；各 slice 的场数组及求解器可能占用大量内存", True)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            check.add(cp, "sc.geometry", str(exc))
        if enabled and number(length) and length >= 0:
            if isinstance(internal, dict):
                # Coverage only needs total body weight, not N kick objects.
                contributions.append(SimpleNamespace(cmd_name=name, length=length, s=v["S (m)"],
                    _sc_nodes={0: (None, SimpleNamespace(sc_length=length))}))
            else:
                start = v.get("SC start (m)")
                if start is None or number(start):
                    contributions.append(SimpleNamespace(cmd_name=name, cmd_type="SpaceCharge", is_enabled=True,
                        sc_length=length, sc_start=start))
    for ref in resources.keys() - used:
        check.add((*root, "Configurations", ref), "sc.unused", "此空间电荷资源未被启用的命令引用", True)
    if enabled and check.circumference > 0:
        policy = sc.get("Coverage check", "warn")
        if policy == "off":
            check.add(root, "sc.coverage_disabled", "空间电荷覆盖检查已关闭", True)
        else:
            from PASS.utils.sc_coverage import analyse_sc_coverage
            try:
                coverage = analyse_sc_coverage(contributions, check.circumference,
                    mode=sc.get("Coverage mode", "full-ring"), expected_length=sc.get("Expected SC length (m)"))
                for issue in coverage["issues"]:
                    check.add(root, "sc.coverage", issue, policy != "error")
            except (ValueError, OverflowError) as exc:
                check.add(root, "sc.coverage_numeric", f"覆盖参数导致数值溢出或无效区间：{exc}")
    check_longitudinal(check)


def check_resource_combinations(check, values, path):
    """Collect independent combinations even if a model validator fails early."""
    widths = [values.get(k) is not None for k in ("Grid Width X (m)", "Grid Width Y (m)")]
    halves = [values.get(k) is not None for k in ("Grid Half Width X (m)", "Grid Half Width Y (m)")]
    if any(widths + halves) and not (all(widths) and not any(halves) or all(halves) and not any(widths)):
        check.add(path, "sc.grid_pair", "必须完整填写一组 X/Y 全宽或 X/Y 半宽，不能缺项或混用")
    method, solver = values.get("Method"), values.get("Solver")
    pic_solvers = {"fft_free_space", "fd_dirichlet", "dst_dirichlet"}
    analytic_sizes = {
        "gaussian_round_free_space": {"Sigma (m)"},
        "gaussian_ellipse_free_space": {"Sigma X (m)", "Sigma Y (m)"},
        "uniform_round_free_space": {"Radius (m)"},
        "uniform_ellipse_free_space": {"Semi-axis A (m)", "Semi-axis B (m)"},
        "parabolic_round_free_space": {"Radius (m)"},
        "parabolic_ellipse_free_space": {"Semi-axis A (m)", "Semi-axis B (m)"},
    }
    if method and solver and (method == "pic") != (solver in pic_solvers):
        check.add((*path, "Solver"), "sc.method_solver", "PIC 方法使用 PIC 求解器；frozen/quasi-frozen 使用解析求解器")
    if method != "pic" and values.get("Particle Deposition Method") is not None:
        check.add((*path, "Particle Deposition Method"), "sc.deposition", "粒子沉积方式只适用于 PIC")
    sizes = set().union(*analytic_sizes.values())
    parameters = sizes | {"Center X (m)", "Center Y (m)", "Angle (rad)"}
    supplied = {key for key in parameters if values.get(key) is not None}
    if method != "frozen":
        for key in sorted(supplied):
            check.add((*path, key), "sc.fixed_profile", "固定中心、方向和尺寸参数仅适用于 frozen 方法")
    elif solver in analytic_sizes:
        for key in sorted(analytic_sizes[solver] - supplied):
            check.add((*path, key), "sc.profile_required", f"{solver} 必须指定此尺寸")
        for key in sorted((supplied & sizes) - analytic_sizes[solver]):
            check.add((*path, key), "sc.profile_unused", f"此尺寸不适用于 {solver}")
        if "round" in solver and values.get("Angle (rad)") not in (None, 0.0):
            check.add((*path, "Angle (rad)"), "sc.round_angle", "圆对称分布没有方向角，应为 0 或 null")


def check_longitudinal(check):
    from PASS.utils.constants import const
    g = check.global_values
    gt = g.get("Transition Gamma")
    if not number(gt) or gt <= 0:
        return
    proton, neutron = g.get("Number of Protons"), g.get("Number of Neutrons")
    mass = const.m_e_eV if proton == neutron == 0 else const.m_p_eV if proton == 1 and neutron == 0 else const.m_u_eV
    for path, bunch in check.bunch_models:
        energy = bunch.get("Kinetic Energy per Nucleon (eV/u)")
        if not number(energy) or energy <= 0:
            continue
        gamma = 1 + energy / mass
        eta = 1 / gt / gt - 1 / gamma / gamma
        if not math.isfinite(eta):
            check.add(("Transition Gamma",), "beam.eta", "Transition Gamma 导致滑移因子数值溢出")
            return
        if bunch.get("Longitudinal dist") in {"matchz", "matchdp"} and not bunch.get("Is Load Distribution from File") and bunch.get("Number of Macro Particles", 0) > 0:
            voltage, phase = bunch.get("RF Voltage (V)"), bunch.get("RF Phase (rad)")
            if number(voltage) and number(phase) and -voltage * eta * math.cos(phase) <= 0:
                check.add(path, "injection.rf_stability", "匹配分布要求 -V·eta·cos(phi_s) > 0；当前参数没有稳定 RF 桶或位于过渡能量")
        beta_squared = 1 - 1 / gamma / gamma
        if beta_squared <= 0:
            check.add((*path, "Kinetic Energy per Nucleon (eV/u)"), "beam.beta", "动能太小，浮点精度下 beta 为 0，无法跟踪")
            continue
        frequency = math.sqrt(beta_squared) * const.c / check.circumference if check.circumference else 0
        for name, (kind, values) in check.commands.items():
            if kind != "Exciter" or not str(values.get("Mode", "")).endswith("_am") or not values.get("Enable", True):
                continue
            ext = values.get("AM t ext (s)")
            if not number(ext) or ext <= 0 or not frequency:
                continue
            start, end = values.get("Start turn"), values.get("End turn")
            if not integer(start) or not integer(end):
                continue
            if min(end, check.turn_count) - start - 1 >= ext * frequency:
                check.add(("Sequence", name, "AM t ext (s)"), "exciter.am_singularity", "激励窗口达到 AM t ext，AM 公式在该时刻奇异；请缩短窗口或增大 AM t ext", True)
