# Example 05 — RF Cavity 纵向动力学测试

## 概述

本 example 系统测试 PASS 高频腔元件 `RFCavity`（`PASS/commands/element/rfcavity.py`）的物理正确性与功能特性，束流为**低能重离子 ²³⁸U³⁵⁺ @ 17 MeV/u**，环光学采用 example 03/04 的 FODO（`fodo.tfs` 头部参数：**C = 234.4 m，γ_t = 3.3746**）。

测试覆盖 4 个 case（由 `make_input.py` 中单一 `CASES` 配置源驱动，`analyse.py` 反向 import 保证参数与理论期望值不重复定义）：

| case | lattice | RF 模式 | 谐波 | 专项验证 |
|------|---------|---------|------|---------|
| `twiss_h1_fixed` | 单 Twiss 点整环 map（`longitudinal_transfer="drift"`） | 固定标量 | h=1 | 能量增益 / Qs / bucket / 阻尼 / 丢失 |
| `twiss_h2_fixed` | 同上 | 固定标量 | h=2 | RF 周期 $C/h$ 的相位对称性 |
| `twiss_h1_ramping` | 同上 | **TFS 波形文件**（每圈一行） | h=1 | 逐圈参数读取 + 越界钳制 |
| `element_h1_fixed` | 真实 FODO 环（fodo.tfs 元件） | 固定标量 | h=1 | 精确逐元素纵向 / 动量压缩涌现验证 |

**关于光学与 transition**：γ_t = 3.3746 的 FODO 上，17 MeV/u 重离子 γ = 1.01825 远低于 γ_t，η = 1/γ_t² − 1/γ² = **−0.8767**（transition 之下，稳定加速相位 0<φ_s<π/2）。加速 2048 圈 γ 仅增至 ~1.019，距 γ_t 差约 3600 倍，**不会穿越 transition**（穿越需能量 ~2.2 GeV/u，η 变号后 φ_s 需移至 (π/2, π)，属另一测试场景）。

**关于 K 值归一化**：`fodo.tfs` 的 K1L/K0L 等是**归一化强度**（除以参考磁刚度 Bρ），在 PASS 与 MADX 中均作为与粒子能量无关的光学参数直接使用（PASS `quadrupole.py` 中 `Δp_x = −K1L·x`，chi=1 无额外缩放）。这对应"磁铁梯度随粒子磁刚度缩放"的缩场同步加速模式（K 恒定是同步加速器的标准运行方式），因此 2 TeV 质子标定的 K 值可直接用于 17 MeV/u 重离子，横向光学（tune、β 函数）保持不变。element case 使用 fodo.tfs 真实元件，其纵向动量压缩（γ_t=3.3746）应从 dipole 的几何路径差中**自然涌现**——这是本 example 对 PASS dipole 纵向映射的专项验证（见 §6）。

## 物理模型

### RF kick（薄透镜，PASS 约定）

每圈在 s=0 处施加纵向 kick：

$$dE = (q/A)\,V\,\sin\left(\phi_s + \phi_{\text{off}} - \frac{2\pi h}{C}z_{\text{lab}}\right),\qquad z_{\text{lab}}=z_{\text{rel}}+z_{\text{center}}$$

- PASS 存储的粒子坐标是相对本束团中心的 $z_{\text{rel}}$；$z_{\text{center}}=h_{\text{id}}C/h_{\text{group}}$ 是束团在整环上的固定标签。因此 RF 相位总是使用实验室纵向坐标 $z_{\text{lab}}$，不使用偶数谐波专用补偿。
- 本 example 只有一个束团，$z_{\text{center}}=0$，所以同步粒子为 $z_{\text{rel}}=0$，与 RF 谐波奇偶无关。相差一个 RF 周期 $C/h$ 的粒子获得相同 kick；h=2 时 $z=\pm C/2$ 与 $z=0$ 同相。
- 当腔谐波与束团分组数相同或为其整数倍时，所有束团中心获得相同参考增益。若二者不是整数倍，程序仍按每束团自己的 $z_{\text{center}}$ 计算相位和参考能量，因此不同束团可获得不同增益。
- 能量→动量→δ 转换为**精确相对论**（无线性化）；每个束团的参考系随自身中心粒子的 $dE_{\text{ref}}$ 更新。横向动量重缩放 $p_{x,y} \leftarrow p_{x,y}\cdot(p_0^{\text{old}}/p_0^{\text{new}})$（绝热阻尼）。

### 一圈 map 与同步振荡

Twiss case：`Injection → RFCavity(s=0) → monitors(s=0) → Twiss(s=C, s_prev=0)`，Twiss 点提供整环传输，纵向为一阶漂移 $z \leftarrow z - \eta\,C\,\delta p$（`twiss.py` 的 `"drift"` 模式，用真实 γ_t、γ）。element case：fodo.tfs 真实元件逐元素传输，纵向由 drift/dipole 的精确公式 $z \leftarrow z + L\big(1 - \beta_0(1+\delta p)/(\beta\,p_z)\big)$ 及 dipole 几何路径差提供——动量压缩（γ_t=3.3746）从 dipole 映射中自然涌现，与 twiss 的一阶 η 应一致（§6 验证）。

小振幅线性化一圈 map 给出同步振荡频率与 bucket 参数（一阶理论，与 PASS `injection.py` 公式同源）：

$$Q_s = \sqrt{\frac{-(q/A)\,h\,V\,\eta\,\cos\phi_s}{2\pi\beta^2 E}},\qquad
\Delta p_{\max} = \sqrt{\frac{-(q/A)\,V\big[2\cos\phi_s - (\pi-2\phi_s)\sin\phi_s\big]}{\pi\beta^2 E\, h\, \eta}},\qquad
z_{\max} = \frac{R(\pi-2\phi_s)}{h}$$

分离线由纵向 Hamiltonian 等值线数值给出（`analyse.bucket_separatrix()`）。

## 束流与参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 离子 | ²³⁸U³⁵⁺（92p, 146n, q=35） | q/A = 0.14706 |
| 动能 | 17 MeV/u | γ=1.01825, β=0.18847 |
| 周长 C | 234.4 m（fodo.tfs） | R=37.31 m |
| γ_t | 3.3746（fodo.tfs） | η = −0.87667 |
| 腔电压 V | 20 kV | |
| 同步相位 φ_s | 0.1 rad | η<0 → 稳定加速相位 0<φ_s<π/2 |
| 圈数 | 2048（ramping: 200） | |

**圈数选择（2048 而非 1024）**：Qs ≈ 3.5e-3 → 同步周期 ~287 圈。1024 圈仅 3.6 个周期，FFT 分辨率 1/1024 = 9.8e-4（仅比 Qs 小 3.6 倍），谱峰解析差；2048 圈 = 7.1 个周期，分辨率 4.9e-4，配合零填充/抛物线插值达到 ~1e-5 精度。此外 bucket 边界粒子（tag 10）滑相至丢失需 ~500 圈，1024 圈余量不足。

**理论期望值**（`make_input.calc_theory()` 自动计算）：

| 量 | h=1 | h=2 |
|----|-----|-----|
| dE_syn | 293.628 eV/u/圈 | 293.628 eV/u/圈 |
| Q_s | 3.4811e-3（周期 287 圈） | 4.9230e-3（周期 203 圈） |
| Δp_max | 7.332e-3 | 5.185e-3 |
| z_max | 109.74 m（< C/2=117.2 m） | 54.87 m |

**分布粒子**（5000 个，KV 横向 / 高斯纵向）：σ_z = 5 m，σ_dp = 1e-3。约束：z 方向裕度 22×（vs z_max），dp 方向裕度 7.3×（vs Δp_max）——**dp 是约束方**。束团匹配长度（σ_z = σ_dp·|η|C/2πQ_s）约 9.4 m，本测试取"欠匹配"（dp 主导），不影响稳定性与统计。

## 测试粒子（tag 分组）

基准同步位置为 $z_{\text{rel}}=0$（本 example 的 $z_{\text{center}}=0$）。13+2 个 tag 粒子：

| tag | 坐标（相对 z_sync） | 用途 |
|-----|---------------------|------|
| 1 | z=0, dp=0 | 同步粒子 → 能量增益 / 参考系跟踪 |
| 2–3 | z=±3 m | Qs（z 振荡，δ_amp≈3.2e-4 线性区） |
| 4–5 | dp=±1e-3 | Qs（dp 振荡，z_amp≈9.4 m） |
| 6–7 | dp=±0.5·Δp_max | bucket 边界扫描 |
| 8–9 | dp=±0.8·Δp_max | 同上 |
| 10–11 | dp=±1.0·Δp_max | 边界（理论临界） |
| 12 | dp=+1.2·Δp_max | 桶外 → dp aperture 丢失路径 |
| 13 | x=3 mm, px=1e-4 | 绝热阻尼（束团级验证） |
| 14–15 | z=±C/2（仅 h=2） | 相差一个 RF 周期的同相性 |

dp 孔径 `±1.08·Δp_max`（自动随 case 计算）：tag 12 第一圈即被截断，tag 10/11 边界粒子可观测滑相/环绕。

## 验证结果

### 1. 能量增益（4 个 case 全部）

| 量 | 实测 | 理论 | 误差 |
|----|------|------|------|
| dE/dn（Ek 斜率） | 293.627696 eV/u/圈 | 293.627696 eV/u/圈 | **0.0000 %** |

### 2. 同步振荡 Qs（FFT，Hann 窗 + 零填充 65536 + 抛物线插值）

| case | 实测 Qs | 理论 Qs(γ₀) | 相对 | 理论 ⟨Qs(γ)⟩ | 相对 |
|------|---------|-------------|------|--------------|------|
| twiss_h1 | 3.4422e-3 | 3.4811e-3 | 1.12 % | 3.4499e-3 | **0.22 %** |
| twiss_h2 | 4.8553e-3 | 4.9230e-3 | 1.38 % | 4.8789e-3 | **0.48 %** |
| element_h1 | 3.4418e-3 | 3.4811e-3 | 1.13 % | 3.4499e-3 | **0.24 %** |

**绝热漂移**：2048 圈参考能量 +0.60 MeV/u（相对总能量 0.063%），但低 β 时 Qs 分母 β²E = (γ−1/γ)m₀ 对 γ 高度敏感（放大因子 ~1/β²），Qs 全程下移 ~1.7%。FFT 测到全程平均，与绝热平均理论 ⟨Qs(γ)⟩ 吻合（0.2–0.5%）。

**element case 的动量压缩验证**：真实 FODO 逐元素模拟（精确漂移 + dipole 几何路径差）得到的 Qs 与 twiss 一阶模拟**一致到 0.012%**（见 §6）——证明 PASS dipole 的纵向映射（含动量压缩 γ_t=3.3746）是正确的，动量压缩从 dipole 映射中自然涌现。

### 3. Bucket 边界扫描（dp 孔径 ±1.08·Δp_max）

| dp₀/Δp_max | twiss_h1 | twiss_h2 | element_h1 |
|------------|----------|----------|------------|
| ±0.5 | 稳定 ✓ | 稳定 ✓ | 稳定 ✓ |
| ±0.8 | 稳定 ✓ | 稳定 ✓ | 稳定 ✓ |
| +1.0 | 滑相丢失 | 滑相丢失 | **稳定 ✓** |
| −1.0 | 稳定 ✓ | 稳定 ✓ | **稳定 ✓** |
| +1.2 | turn 0 丢失 ✓ | turn 0 丢失 ✓ | turn 0 丢失 ✓ |

tag 12 第一圈被 dp 孔径截断（lost_turn=0），丢失路径正确。**±1.0Δp_max 边界粒子在 element（精确漂移）下稳定**，而在 twiss（一阶漂移）下 +1.0 滑出丢失——一阶近似的桶边界判据偏移是近似误差而非物理（理论 Δp_max 为一阶值，精确桶略大）。±方向不对称来自 φ_s≠0 的 bucket 上下不对称。

### 4. h=2 的 RF 周期对称性

| tag | 初始 z | max\|dp\| | max\|z−z₀\| |
|-----|--------|-----------|-------------|
| 14 | +117.2 m（=+C/2） | 6.2e-15 | 0.0 m |
| 15 | −117.2 m（=−C/2） | 6.2e-15 | 0.0 m |

两个粒子与 $z=0$ 相差一个 h=2 的 RF 周期 $C/h=C/2$，因此增益与参考系一致（δ 恒 0）。该结果不依赖坐标折叠或偶数谐波专用补偿。

### 5. 能量斜坡（rf data file，V(n)=20 kV·(1+0.02n)，50 行）

| 验证 | 结果 |
|------|------|
| E(n) vs Σ V(k)sinφ_s 逐圈模型 | max 相对偏差 **1.5e-11** |
| 50 行后钳制斜率 vs 期望冻结值 | **0.000 %** |

### 6. Twiss vs Element：一阶漂移近似误差

两个 case 使用同一 FODO（γ_t=3.3746）：twiss 用一阶纵向漂移 $z \leftarrow z-\eta C\delta p$，element 用精确逐元素映射（动量压缩从 dipole 几何中涌现）。

**(a) Qs 一致性**（一阶 vs 精确的实现自洽性）

| 量 | 值 |
|----|-----|
| Qs(twiss) 实测 | 3.4484e-3 |
| Qs(element) 实测 | 3.4480e-3 |
| 比值 | **0.999879**（理论 1.0，相同 γ_t） |

一阶漂移与精确逐元素传输的 Qs 一致到 0.012%，同时证明 PASS dipole 的纵向映射（动量压缩）正确。

**(b) 轨迹差异 vs dp**（一阶近似误差）

| dp₀ | max\|Δz\| |
|-----|----------|
| ±1e-3 | 0.05 m |
| ±3.7e-3 | 0.10–0.36 m |
| ±5.9e-3 | 2.4–2.8 m |
| ±7.3e-3 | 172–218 m（边界分叉：twiss 一阶下 tag10 滑出丢失，element 精确下稳定） |

一阶轨迹误差随 \|δp\| 增长；正常束流（δp≲1e-3）误差 ~5 cm（占 z 振荡振幅 ~0.5%），接近 bucket 边界时一阶近似失效（见 §3）。

### 7. 绝热阻尼（束团级）

| 量 | 相对变化 |
|----|----------|
| σ_py·p₀（守恒量） | 7.2e-3（≈ 5000 粒子统计噪声水平） |
| tag 13 Jx·p₀²（单粒子） | ~5e-3（Jx 的 x²/β 项不被 kick 缩放，物理预期） |

## 关于三个物理注意点的严重性评估

| 注意点 | 性质 | 严重性 |
|--------|------|--------|
| **多束团与非整倍 RF 谐波** | 允许的物理配置：中心相位使用 $z_{\text{lab}}=z_{\text{rel}}+z_{\text{center}}$；若腔谐波不是分组数整数倍，不同束团中心会有不同参考增益 | 取决于机器运行方案。程序不限制该配置，但使用者应确认不同束团的纵向工作点是否符合预期 |
| **低 β 束流 Qs 绝热漂移**（2048 圈下移 ~1.7%） | 真实物理：β²E 对 γ 的放大 ~1/β² | 低。分析时对比绝热平均理论 ⟨Qs(γ)⟩ 即可（0.2–0.5% 吻合）；缩短圈数或提高能量可抑制 |
| **twiss 一阶漂移在边界失效**（±1.0Δp 丢失时机不同） | 模型近似属性：一阶 −ηCδp 不含高阶项 | 中低。正常束流 δp≲1e-3 时影响小；大 δp / 边界粒子研究应使用 element 模式 |

**RFCavity 实现结论**：核心物理（能量增益、同步振荡、bucket、基于 $z_{\text{lab}}$ 的多束团相位、ramping 读取、dp 孔径、绝热阻尼）全部通过真实模拟验证，**未发现问题**。唯一已知限制：GPU 实现未完成（`execute_gpu` 抛 NotImplementedError），本 example 使用 CPU backend。

## 文件结构

```
05_rf_cavity_longitudinal/
├── make_input.py   # 单一配置源：CASES 字典 + calc_theory() + build_case()
├── run.py          # run_case(name) → PASS.main
├── analyse.py      # 9 个验证模块 + A/B 对比（import make_input 保证一致性）
├── fodo.madx/.seq/.ps/.tfs  # example 03/04 的 FODO（提供 C、γ_t）
├── rf_ramp.tfs     # ramping 波形（make_input 自动生成）
├── beam0_<case>.json
└── output/<case>/YYYY_MMDD/HHMM_SS/
```

## 使用方法

```bash
cd example/05_rf_cavity_longitudinal
python make_input.py    # 生成 4 个 beam0_<case>.json + rf_ramp.tfs
python run.py           # 跑 4 个 case（CPU backend，每个约 10 s）
python analyse.py       # 打印全部验证结果 + 交互图（plt.show()）
```

`make_input.py` / `run.py` / `analyse.py` 的 `__main__` 中均可手动取消注释选择单个 case。

## 注意事项

1. **backend 必须为 CPU**：`RFCavity.execute_gpu` 未实现。
2. **序列排序修复**：本 example 依赖 `PASS/commands/__init__.py` 中统一维护的 `COMMAND_PRIORITY`，其中 `"RFCavity": 300`。此前 RFCavity 落入 `Other=999`，会在同 s 处排到 monitors 之后（每圈记录 kick 前状态）。修复后顺序为 `Injection → RFCavity → monitors → 环传输`，turn n 记录第 n 次 kick 后的状态。
3. **turn 约定**：turn 0 即包含第一次 kick。
4. **纵向坐标**：手动指定粒子的 z 是相对该束团中心的 $z_{\text{rel}}$；需要 RF 相位时程序自动加上 $z_{\text{center}}$。
5. **K 值归一化**：`fodo.tfs` 的 K1L 等为归一化强度，能量无关（见概述），element case 可直接使用真实 FODO 元件，无需磁刚度缩放。
6. **FFT 测 Qs**：Qs ~ 3.5e-3 很小，需零填充 65536 + 抛物线插值，并与绝热平均理论 ⟨Qs(γ)⟩ 对比而非初始值。
