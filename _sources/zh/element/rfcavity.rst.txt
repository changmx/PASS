高频加速腔（RFCavity）
========================

本模块介绍 PASS 中的高频加速腔元件 **RFCavity** ，用于模拟带电粒子在射频电场中的纵向加速运动。高频加速腔是同步加速器、直线加速器和回旋加速器中最核心的元件之一，通过周期性电场为粒子提供能量增益，维持同步粒子的能量并控制纵向束流动力学（同步振荡、束团压缩、纵向接受度等）。

PASS 中的高频加速腔建模为 **薄透镜** （ ``length = 0`` ）的瞬时能量 kick ，采用 **精确相对论能量-动量关系** 和 **移动参考系** 。纵向变换弃用了传统的一阶线性化近似，改用精确的 :math:`E^2 = p^2 + m_0^2` 关系，避免了线性化带来的 :math:`O(\delta^2)` 误差和参考系变换中的 :math:`\beta_1/\beta_0` 因子问题。

**代码位置**

- 源文件： ``PASS/commands/element/rfcavity.py``
- 类名： ``RFCavity`` （继承自 ``Command`` ）
- 注册名： ``rfcavity``
- 核心特征：

  - 薄透镜模型，瞬时能量 kick
  - 精确相对论能量-动量变换（ :math:`E^2 = p^2 + m_0^2` ），无线性化近似
  - 移动参考系：每圈更新束流参考能量（ :math:`E_k, \gamma, \beta, p_0, B\rho` ）
  - 横向动量重缩放（绝热阻尼）： :math:`p_x \leftarrow p_x \cdot \beta_0\gamma_0 / (\beta_1\gamma_1)`
  - 归一化发射度 :math:`\epsilon_N = \beta\gamma\epsilon` 严格守恒
  - 支持 dp 接受度（纵向孔径）检查
  - 支持固定值和 TFS 文件（Ramping）两种参数输入方式
  - 支持多圈加速模拟
  - 谐波数是腔的状态属性，对所有束团统一


坐标约定
--------

PASS 采用归一化曲线坐标，六维相空间变量为 :math:`(x, p_x, y, p_y, \zeta, \delta)` ：

.. list-table::
  :header-rows: 1
  :widths: 15 20 65

  * - 变量
    - 符号
    - 定义
  * - ``x``
    - :math:`x`
    - 水平偏移（相对于参考轨道）
  * - ``px``
    - :math:`p_x`
    - 归一化水平动量， :math:`p_x = P_x / P_0`
  * - ``y``
    - :math:`y`
    - 垂直偏移
  * - ``py``
    - :math:`p_y`
    - 归一化垂直动量， :math:`p_y = P_y / P_0`
  * - ``z``
    - :math:`\zeta`
    - 纵向坐标， :math:`\zeta = s - \beta_0 c t`
  * - ``dp``
    - :math:`\delta`
    - 相对动量偏差， :math:`\delta = P / P_0 - 1`

其中 :math:`P_0` 为参考粒子动量， :math:`\beta_0 = v_0 / c` 为参考粒子归一化速度。


物理推导
--------

RF 腔的物理本质
~~~~~~~~~~~~~~~~

RF 腔产生纵向（沿束流方向）振荡电场 :math:`E_s(t) = E_0 \sin(\omega_{\text{rf}} t + \varphi_s)` 。粒子穿过腔体时获得的能量增益取决于粒子到达腔体的时刻，即粒子的纵向位置 :math:`\zeta` 决定了它感受到的 RF 相位。

设机器周长为 :math:`C` ，机器半径 :math:`R = C / (2\pi)` ，谐波数 :math:`h` ， RF 电压 :math:`V` ，同步相位 :math:`\varphi_s` 。粒子的纵向位置 :math:`\zeta` 对应的方位角 :math:`\theta = \zeta / R` ，其 RF 相位为：

.. math::

  \varphi_{\text{particle}} = \varphi_s - h \cdot \theta + \varphi_{\text{off}}

其中 :math:`\varphi_{\text{off}}` 为附加相位偏移（详见后文）。


phi_offset 的用途
~~~~~~~~~~~~~~~~~~

``phi_offset`` 是施加在所有粒子上的常数相位偏移，用于平移 RF 波形的时间基准。其主要用途包括：

1. **多腔相位对齐** ：当多台 RF 腔沿环分布、腔间距不是 RF 波长的整数倍时，每台腔需要独立的相位修正以保持同步。
2. **多谐波系统** ：不同谐波数的腔共享同一频率基准时，通过 :math:`\varphi_{\text{off}}` 实现各腔独立的相位调节。
3. **相位 trim** ：在运行中微调腔的相位而不改变同步粒子的定义（ ``phase`` 仍然定义同步粒子的能量增益）。

物理上， :math:`\varphi_{\text{off}}` 旋转整个 :math:`\sin` 曲线，使得粒子的实际相位变为 :math:`\varphi_s + \varphi_{\text{off}} - h\theta` 。


能量 kick
~~~~~~~~~~

每个粒子的能量增益为：

.. math::

  \Delta E_{\text{kick}} = \frac{q}{A} \cdot V \cdot \sin(\varphi_{\text{particle}})

其中 :math:`q/A` 为荷质比。同步粒子（ :math:`\zeta = 0` ）的能量增益为：

.. math::

  \Delta E_{\text{syn}} = \frac{q}{A} \cdot V \cdot \sin(\varphi_s)

:math:`\Delta E_{\text{syn}}` 体现了 RF 腔的净加速效果。


一阶线性化近似及其问题
~~~~~~~~~~~~~~~~~~~~~~~

传统方法在将能量偏差 :math:`dE` 转换为动量偏差 :math:`\delta` 时，采用一阶泰勒展开：

.. math::

  dE \approx \delta \cdot \beta^2 \cdot E_{\text{total}}

其中 :math:`dE = E - E_0` 为粒子相对参考粒子的能量偏差。这是精确关系

.. math::

  E = \sqrt{(p_0 (1+\delta) c)^2 + (m_0 c^2)^2}

在 :math:`\delta \to 0` 处的一阶近似。展开的误差为 :math:`O(\delta^2)` 。

**问题 1：大** ``delta`` **时精度差**

当 :math:`\delta` 较大（如 :math:`\pm 30\%` 的注入接收度）， :math:`O(\delta^2)` 项可达 :math:`\sim 0.01` ，远超浮点精度，引入不可忽略的系统性偏差。

**问题 2：参考系变换引入额外的** ``beta1/beta0`` **因子**

在移动参考系中，参考能量从 :math:`E_0` 变为 :math:`E_1 = E_0 + \Delta E_{\text{syn}}` 。传统方法假设参考系变换时 :math:`\delta` 不变，推导如下：

从 :math:`dE = \delta \cdot \beta^2 \cdot E_{\text{total}}` 出发，旧参考系中 :math:`dE_0 = \delta \cdot \beta_0^2 \cdot E_0` ，新参考系中 :math:`dE_1 = \delta \cdot \beta_1^2 \cdot E_1` 。若假设 :math:`\delta` 不变，则：

.. math::

  dE_1 = dE_0 \cdot \frac{\beta_1^2 E_1}{\beta_0^2 E_0}

进一步将 :math:`\beta_1^2 E_1 / (\beta_0^2 E_0)` 近似为 :math:`\beta_1 / \beta_0` （仅在 :math:`E_1 \approx E_0` 即弱加速时成立），得到：

.. math::

  dE_1 = \frac{\beta_1}{\beta_0} \cdot (dE_0 + \Delta E_{\text{non-syn}} - \Delta E_{\text{syn}})

这是一个 **三重近似** ：

1. **一阶线性化** ： :math:`dE \approx \delta \cdot \beta^2 \cdot E_{\text{total}}` （截断 :math:`O(\delta^2)` ）
2. **delta 不变假设** ：参考系变换时 :math:`\delta` 不变（实际 :math:`\delta = p/p_0 - 1` 依赖于 :math:`p_0` ）
3. **弱加速近似** ： :math:`\beta_1^2 E_1 / (\beta_0^2 E_0) \approx \beta_1/\beta_0` （仅在 :math:`E_1 \approx E_0` 时成立）

在强加速场景（如从注入到引出能量变化数倍），第 3 步近似显著失效。


精确能量-动量变换
~~~~~~~~~~~~~~~~~~~

PASS 弃用上述全部近似，直接从精确相对论关系出发。所有物理量采用自然单位制（ :math:`c = 1` ）， :math:`m_0` 、 :math:`p_0` 、 :math:`E` 均以 eV 为单位。能量-动量关系为：

.. math::

  E^2 = p^2 + m_0^2

跟踪链路如下：

**Kick 前** ：粒子动量 :math:`p_{\text{old}} = p_{0,\text{old}} \cdot (1 + \delta)` ，绝对总能量：

.. math::

  E_{\text{old}} = \sqrt{p_{\text{old}}^2 + m_0^2}

**施加 kick** ：

.. math::

  E_{\text{new}} = E_{\text{old}} + \Delta E_{\text{kick}}

**恢复动量** ：

.. math::

  p_{\text{new}} = \sqrt{E_{\text{new}}^2 - m_0^2}

**计算新** :math:`\delta` （相对于新参考动量 :math:`p_{0,\text{new}}` ）：

.. math::

  \delta_{\text{new}} = \frac{p_{\text{new}}}{p_{0,\text{new}}} - 1

**为什么精确方法不需要** ``beta1/beta0`` **因子**

精确方法直接追踪每个粒子的绝对总能量 :math:`E_{\text{particle}}` 和绝对动量 :math:`p_{\text{particle}}` 。 RF kick 改变的是能量（ :math:`E_{\text{new}} = E_{\text{old}} + \Delta E` ），然后从能量精确恢复动量（ :math:`p = \sqrt{E^2 - m_0^2}` ），最后除以新参考动量得到 :math:`\delta` 。

这个过程中：

- 不需要 :math:`dE \to \delta` 的线性化（直接用 :math:`E \to p` 的精确关系）
- 不需要假设 :math:`\delta` 不变（ :math:`\delta` 由 :math:`p_{\text{particle}} / p_{0,\text{new}}` 直接计算）
- 不需要 :math:`\beta_1/\beta_0` 缩放因子（参考系变换已隐含在 :math:`p_{0,\text{new}}` 中）

因此三重近似全部被绕过， :math:`\beta_1/\beta_0` 因子自然不出现。

此变换 **完全精确** ，无任何线性化近似，只需两次 ``sqrt`` 运算（numpy 向量化，成本可忽略），适用于大 :math:`\delta` 和强加速场景。


移动参考系
~~~~~~~~~~~

PASS 采用移动参考系（moving reference frame）：每次 RF kick 后，束流的参考能量更新为包含 :math:`\Delta E_{\text{syn}}` 的新值：

.. math::

  E_{\text{total},1} = E_{\text{total},0} + \Delta E_{\text{syn}}

.. math::

  \gamma_1 = \frac{E_{\text{total},1}}{m_0}

.. math::

  \beta_1 = \sqrt{1 - \frac{1}{\gamma_1^2}}

.. math::

  p_{0,\text{new}} = \gamma_1 m_0 \beta_1

.. math::

  E_{k,1} = E_{\text{total},1} - m_0

同步粒子的 :math:`\delta` 始终保持在 0 附近，避免了固定参考系中 :math:`\delta` 持续增长导致的数值精度问题。


横向动量重缩放与绝热阻尼
~~~~~~~~~~~~~~~~~~~~~~~~~

RF kick 是纯纵向能量增益，不改变粒子的绝对横向动量 :math:`P_x` 。但由于参考动量 :math:`P_0` 增长，归一化横向动量 :math:`p_x = P_x / P_0` 必须重缩放：

.. math::

  p_x \leftarrow p_x \cdot \frac{p_{0,\text{old}}}{p_{0,\text{new}}} = p_x \cdot \frac{\beta_0 \gamma_0}{\beta_1 \gamma_1}

.. math::

  p_y \leftarrow p_y \cdot \frac{\beta_0 \gamma_0}{\beta_1 \gamma_1}

此缩放 **精确** ，不是近似（因为 :math:`p_0 c = \beta \gamma m_0 c^2` ，所以 :math:`p_{0,\text{old}} / p_{0,\text{new}} = \beta_0 \gamma_0 / (\beta_1 \gamma_1)` ）。

**物理意义** ：归一化发射度 :math:`\epsilon_N = \beta\gamma\epsilon` 是绝热不变量（Liouville 定理）。当 :math:`p_0` 增长时，几何发射度 :math:`\epsilon` 按 :math:`1/(\beta\gamma)` 收缩，即绝热阻尼（adiabatic damping）。横向动量重缩放保证了 :math:`\epsilon_N` 严格守恒：

.. math::

  \epsilon_{\text{geom}}^{\text{new}} = \epsilon_{\text{geom}}^{\text{old}} \cdot \frac{\beta_0 \gamma_0}{\beta_1 \gamma_1} = \frac{\epsilon_N}{\beta_1 \gamma_1}


跟踪流程
--------

.. code-block:: text

  输入: z, dp(=δ), px, py, tag, bunch参数(β₀, γ₀, m₀, q/A, Ek, p₀, C)

  1. 计算 RF 相位
     θ = z / R                          (R = C/2π)
     φ_particle = phase + φ_off - h·θ

  2. 能量 kick
     ΔE_kick = (q/A)·V·sin(φ_particle)  [逐粒子]
     ΔE_syn  = (q/A)·V·sin(phase)       [标量]

  3. 更新束流参考（移动参考系）
     E_total1 = E_total0 + ΔE_syn
     γ₁ = E_total1 / m₀
     β₁ = √(1 - 1/γ₁²)
     p₀_new = γ₁·m₀·β₁
     Ek₁ = E_total1 - m₀

  4. 精确 δ 更新
     p_old = p₀_old·(1+δ)
     E_old = √(p_old² + m₀²)
     E_new = E_old + ΔE_kick
     p_new = √(E_new² - m₀²)
     δ_new = p_new / p₀_new - 1

  5. 横向动量重缩放（绝热阻尼）
     scale = β₀γ₀ / (β₁γ₁)
     px *= scale
     py *= scale

  6. dp 接受度检查（超出 → 标记丢失）

  7. z-wrap 到 [-C/2, C/2)

  8. 更新丢失粒子信息


接口参数
--------

.. list-table:: RF 参数
  :header-rows: 1
  :widths: 20 22 10 10 38

  * - 属性名
    - JSON key
    - 类型
    - 默认值
    - 说明
  * - ``voltage``
    - ``voltage (v)``
    - float
    - 0.0
    - RF 电压（V）
  * - ``harmonic``
    - ``harmonic``
    - int
    - 1
    - 谐波数 :math:`h`
  * - ``phase``
    - ``phase (rad)``
    - float
    - 0.0
    - 同步相位 :math:`\varphi_s` （rad）
  * - ``phi_offset``
    - ``phi offset (rad)``
    - float
    - 0.0
    - 附加相位偏移（rad），用于多腔相位对齐和相位 trim
  * - ``_rf_table``
    - ``rf data file``
    - str
    - None
    - Ramping 数据文件路径（TFS 格式），提供后覆盖固定值参数。每行对应一圈，所需列名： ``HARMONIC`` 、 ``VOLTAGE`` 、 ``PHASE`` 、 ``PHI_OFFSET``
  * - ``is_enabled``
    - ``is enabled``
    - bool
    - True
    - 开关

.. list-table:: 纵向孔径参数
  :header-rows: 1
  :widths: 20 22 10 10 38

  * - 属性名
    - JSON key
    - 类型
    - 默认值
    - 说明
  * - ``dp_aperture_lower``
    - ``dp aperture[0]``
    - float
    - -1.0
    - dp 接受度下限
  * - ``dp_aperture_upper``
    - ``dp aperture[1]``
    - float
    - 1.0
    - dp 接受度上限

.. list-table:: 通用参数
  :header-rows: 1
  :widths: 20 22 10 10 38

  * - 属性名
    - JSON key
    - 类型
    - 默认值
    - 说明
  * - ``s``
    - ``s (m)``
    - float
    - 必填
    - 元件在束线中的纵向位置
  * - ``cmd_name``
    - ``name``
    - str
    - 必填
    - 元件名称
  * - ``aperture_type``
    - ``aperture type``
    - str
    - ``off``
    - 横向孔径类型
  * - ``aperture_value``
    - ``aperture value``
    - list
    - ``[]``
    - 横向孔径参数


Ramping 数据文件
----------------

当 RF 参数需要随圈数变化时（如能量 ramping ），可提供 TFS 格式的数据文件。 TFS （ Table File System ）是一种带元数据的表格格式，文件中可包含标题、注释等文档信息，列通过列名识别而非位置。

所需列名（大小写不限，顺序不限）：

- ``HARMONIC`` —— 谐波数
- ``VOLTAGE`` —— RF 电压（V）
- ``PHASE`` —— 同步相位（rad）
- ``PHI_OFFSET`` —— 附加相位偏移（rad）

读取时列名自动转换为小写，因此 ``Harmonic`` 、 ``voltage`` 、 ``phase`` 等任意大小写组合均可。每行对应一圈（第 0 行 = 第 0 圈）。若圈数超出文件行数，使用最后一行数据。文件还可包含 ``TITLE`` 、 ``DATE`` 等元数据头信息，由 ``tfs-pandas`` 库自动解析。


使用示例
--------

示例 1：基本加速腔
~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
    "RFCavity_1": {
      "S (m)": 0.0,
      "Command": "RFElement",
      "voltage (v)": 100000,
      "harmonic": 1,
      "phase (rad)": 0.3,
      "is enabled": true
    }
  }

同步粒子每圈获得 :math:`\Delta E = q \cdot V \cdot \sin(0.3) \approx 29552` eV 的能量增益。

示例 2：带 dp 接受度和相位偏移的腔
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
    "RFCavity_2": {
      "S (m)": 500.0,
      "Command": "RFElement",
      "voltage (v)": 200000,
      "harmonic": 4,
      "phase (rad)": 0.5236,
      "phi offset (rad)": 0.05,
      "dp aperture": [-0.02, 0.02],
      "is enabled": true
    }
  }

:math:`\varphi_s = \pi/6 = 0.5236` rad ， 4 次谐波， dp 接受度 :math:`\pm 2\%` ，附加相位偏移 0.05 rad 。

示例 3：Ramping 腔（TFS 文件输入）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
    "RFCavity_3": {
      "S (m)": 0.0,
      "Command": "RFElement",
      "rf data file": "D:/PASS/para/rf_data.tfs",
      "is enabled": true
    }
  }

TFS 文件中每行指定该圈的 ``HARMONIC`` 、 ``VOLTAGE`` 、 ``PHASE`` 、 ``PHI_OFFSET`` 。

示例 4：禁用腔
~~~~~~~~~~~~~~~

.. code-block:: json

  {
    "RFCavity_4": {
      "S (m)": 0.0,
      "Command": "RFElement",
      "voltage (v)": 100000,
      "harmonic": 1,
      "phase (rad)": 0.3,
      "is enabled": false
    }
  }

``is_enabled = false`` 时，腔体不执行任何操作（no-op ）。


应用场景
--------

1. **同步加速器加速** ：从注入能量到引出能量的加速过程，移动参考系保持 :math:`\delta` 在小量级，数值精度好

2. **纵向束流动力学** ：同步振荡、束团拉伸/压缩、纵向 Emittance 控制

3. **多谐波加速** ：多个 RF 腔串联使用，不同谐波数实现束团塑形

4. **能量 ramping** ：通过 TFS 文件输入实现电压/相位随圈数变化，模拟真实加速器运行方案

5. **纵向接受度研究** ：通过 dp aperture 参数设定纵向接受度，研究束流损失边界


验证测试
--------

``tests/test_rf_verification.py`` —— 17 组共 25 项测试，全部通过：

1. 同步粒子 :math:`\delta \approx 0` （精度 :math:`< 10^{-12}` ）
2. 同步能量增益 :math:`\Delta E = (q/A) V \sin(\varphi_s)`
3. 精确能量-动量关系 :math:`E^2 = p^2 + m_0^2`
4. 相位依赖性（不同 :math:`\zeta` 的粒子获得不同 kick ）
5. 移动参考系（束流参考能量正确更新）
6. 绝热阻尼（ :math:`p_x` 按 :math:`\beta_0\gamma_0/(\beta_1\gamma_1)` 缩放）
7. 归一化发射度守恒（相对误差 :math:`< 10^{-15}` ）
8. 死粒子不被 kick
9. 零电压退化为 no-op
10. dp 接受度检查
11. 多圈加速（10 圈后能量正确， :math:`\delta` 仍为 :math:`\sim 0` ）
12. 与独立参考实现的全量对比（ :math:`< 10^{-10}` ）
13. 低 :math:`\gamma` （非超相对论， :math:`\beta = 0.417` ）
14. 大 :math:`\delta` （ :math:`\pm 30\%` ，非线性区）
15. 离子（ :math:`q/A \neq 1` ）
16. 相位偏移 :math:`\varphi_{\text{off}}`
17. 禁用腔（ ``is_enabled = false`` ）
