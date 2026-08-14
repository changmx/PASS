静电偏转板（ElSeparator）
=========================

本模块介绍 PASS 中的静电偏转板元件 **ElSeparator** ，用于模拟带电粒子在均匀横向电场中的偏转。静电偏转板广泛应用于束流注入与引出系统，通过 septum（切割板/丝）将孔径分为无场区（环内循环束流）和有场区（注入或引出束流），仅对越过 septum 的粒子施加电场偏转。

PASS 中的静电偏转板支持 **薄透镜** （纯动量踢角）和 **厚透镜** （DKD 精确抛物线轨迹）两种模式。用户可通过场强（ ``ex`` / ``ey`` ， V/m）或积分场（ ``exl`` / ``eyl`` ， V）两种方式输入偏转参数，与 MAD-X 的定义一致。

**代码位置**

- 源文件： ``PASS/commands/element/elseparator.py``
- 类名： ``ElSeparator`` （继承自 ``Command`` ）
- 注册名： ``elseparator``
- 核心特征：

  - 薄透镜（ ``length = 0`` ）：纯动量平移，严格辛
  - 厚透镜（ ``length > 0`` ）：DKD（Drift-Kick-Drift）2 阶辛积分，对均匀电场为精确解
  - 场强（ ``ex`` / ``ey`` ）与积分场（ ``exl`` / ``eyl`` ）两种输入方式，互相自动推导
  - septum 位置检测：自动判断粒子处于无场区、有场区或撞击极板/丝
  - 支持 ``tilt`` 绕 :math:`s` 轴的 roll 旋转（顺时针，与 MAD-X 一致）
  - 支持孔径检查


坐标约定
--------

PASS 采用的六维相空间变量为 :math:`(x, p_x, y, p_y, z, \delta)` ：

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

电场力与归一化踢角
~~~~~~~~~~~~~~~~~~

静电偏转板在极板之间产生均匀横向电场 :math:`E_x` 或 :math:`E_y` 。带电粒子在电场中受力：

.. math::

  \vec{F} = q \vec{E}

粒子以纵向速度 :math:`v = \beta_0 c` 穿过长度 :math:`L` 的偏转板，停留时间为 :math:`t = L / (\beta_0 c)` ，横向动量变化：

.. math::

  \Delta P_x = q E_x \cdot t = \frac{q E_x L}{\beta_0 c}

归一化到 PASS 坐标（ :math:`p_x = P_x / P_0` ， :math:`P_0 = q_0 B\rho` ，同种粒子 :math:`q = q_0` ）：

.. math::

  \Delta p_x = \frac{\Delta P_x}{P_0} = \frac{E_x L}{\beta_0 c \cdot B\rho} = \frac{\mathrm{exl}}{\beta_0 c \cdot B\rho}

其中 :math:`\mathrm{exl} = E_x \cdot L` 为积分电场（单位：伏特）， :math:`B\rho = P_0 / q_0` 为磁刚度。同理：

.. math::

  \Delta p_y = \frac{\mathrm{eyl}}{\beta_0 c \cdot B\rho}

量纲验证： :math:`[\mathrm{V}] / ([\mathrm{m/s}] \cdot [\mathrm{T \cdot m}]) = [\mathrm{J/C}] / [\mathrm{kg \cdot m / (C \cdot s)}] = 1` （无量纲） ✓

与磁偶极子的等效关系
~~~~~~~~~~~~~~~~~~~~~~

电场对速度 :math:`\beta_0 c` 的粒子产生的偏转，等效于磁刚度 :math:`B\rho` 的粒子穿过磁场 :math:`B` 。由 :math:`\Delta p_x = E_x L / (\beta_0 c \cdot B\rho)` 与磁偶极踢角 :math:`\Delta p_x = B L / B\rho` 等价：

.. math::

  E_x = \beta_0 c \cdot B

即 :math:`1\,\mathrm{MV/m}` 的电场在 :math:`\beta_0 \approx 1` 时等效于 :math:`B \approx 3.336\,\mathrm{mT}` 的磁场。

偏转角等效但能量变化不同
^^^^^^^^^^^^^^^^^^^^^^^^^^

上述等效仅指 **偏转角相同** 。电场与磁场在能量守恒方面有本质区别：

- **磁场不做功** ： :math:`\vec{F} = q\vec{v}\times\vec{B}` ， :math:`\vec{F} \perp \vec{v}` ，粒子总动量 :math:`P` 不变， :math:`p_x` 增加的同时 :math:`p_z` 减小（动量重新分配）， :math:`\delta` 精确不变。
- **电场做功** ： :math:`\vec{F} = q\vec{E}` ，粒子在板内有横向位移 :math:`\Delta x` ，电场做功 :math:`W = qE_x \cdot \Delta x \neq 0` ，粒子总能量增加， :math:`\delta` 变化。

对厚透镜，粒子的横向位移（ DKD 精确解）为：

.. math::

  \Delta x = \frac{p_{x0} L}{p_z} + \frac{\Delta p_x \cdot L}{2 p_z}

电场做功对应的 :math:`\delta` 变化：

.. math::

  \Delta\delta = \frac{W}{P_0 c} = \frac{E_x \cdot \Delta x}{B\rho \cdot c}

对 :math:`p_{x0} = 0` 的粒子，代入 :math:`\Delta x = \Delta p_x \cdot L / (2 p_z)` 和 :math:`\Delta p_x = E_x L / (\beta_0 c \cdot B\rho)` ：

.. math::

  \Delta\delta = \frac{\Delta p_x^2}{2\beta_0}

对 :math:`\Delta p_x = 30\,\mathrm{mrad}` 、 :math:`\beta_0 \approx 1` ， :math:`\Delta\delta \approx 4.5 \times 10^{-4}` ，比典型束流动量展宽（ :math:`10^{-3} \sim 10^{-2}` ）小一到两个量级。

.. note::

  PASS 的 DKD 实现中 kick 仅更新 :math:`p_x` / :math:`p_y` ，不更新 :math:`\delta` ，即忽略了电场做功。这是合理近似：

  - **量级可忽略** ： :math:`\Delta\delta = O(\Delta p_x^2)` ，对几十 mrad 偏转角为 :math:`10^{-4}` 量级
  - **逐粒子正确处理代价高** ： :math:`\Delta x` 依赖初始 :math:`p_{x0}` ，不同粒子不同；正确算功需要在 DKD 内部逐粒子追踪位移，将简单辛积分器变为迭代方案
  - **薄透镜自洽** ： :math:`L = 0` 时 :math:`\Delta x = 0` ， :math:`W = 0` ， :math:`\Delta\delta = 0` ，薄透镜忽略能量变化天然自洽


薄透镜模式
----------

当 ``length = 0`` 时，静电偏转板建模为薄透镜：粒子位置不变，仅动量发生瞬时跳变：

.. math::

  x \leftarrow x

.. math::

  p_x \leftarrow p_x + \frac{\mathrm{exl}}{\beta_0 c \cdot B\rho}

.. math::

  y \leftarrow y

.. math::

  p_y \leftarrow p_y + \frac{\mathrm{eyl}}{\beta_0 c \cdot B\rho}

该映射的 Jacobi 矩阵为单位矩阵，严格辛。踢角量直接由积分场 :math:`\mathrm{exl}` / :math:`\mathrm{eyl}` 计算，无需知道极板长度。


厚透镜模式（DKD）
-----------------

当 ``length > 0`` 时，采用 Drift-Kick-Drift（DKD）2 阶辛积分：

.. math::

  \mathcal{M}_{\mathrm{DKD}}(L) = \mathrm{Drift}\!\left(\frac{L}{2}\right) \circ \mathrm{Kick}(L) \circ \mathrm{Drift}\!\left(\frac{L}{2}\right)

其中 Kick 为薄透镜踢角（ :math:`\Delta p_x = \mathrm{exl} / (\beta_0 c \cdot B\rho)` ），Drift 为精确漂移映射。

每次 ``_drift_exact_cpu`` 调用执行 :math:`x \mathrel{+}= L \cdot p_x / p_z` 。 DKD 分三步：第一次 drift 用初始 :math:`p_{x0}` ， kick 后第二次 drift 用 :math:`p_{x0} + \Delta p_x` ，合并得：

.. math::

  \Delta x = \frac{p_{x0} L}{2 p_z} + \frac{(p_{x0} + \Delta p_x) L}{2 p_z} = \frac{p_{x0} L}{p_z} + \frac{\Delta p_x \cdot L}{2 p_z}

DKD 对均匀电场是精确解
~~~~~~~~~~~~~~~~~~~~~~~

均匀电场下粒子运动方程为常加速度运动。设 :math:`k = E_x / (\beta_0 c \cdot B\rho)` （常数），在 :math:`p_z` 近似不变的条件下：

.. math::

  \frac{dp_x}{ds} = k

.. math::

  \frac{dx}{ds} = \frac{p_x}{p_z}

积分得抛物线轨迹：

.. math::

  p_x(s) = p_{x0} + k \cdot s

.. math::

  x(s) = x_0 + \frac{p_{x0}}{p_z} s + \frac{k}{2 p_z} s^2

在 :math:`s = L` 处，代入 :math:`\Delta p_x = k L` ：

.. math::

  x(L) = x_0 + \frac{p_{x0} L}{p_z} + \frac{\Delta p_x \cdot L}{2 p_z}

与 DKD 结果 **完全一致** 。这不是巧合——leapfrog（DKD）对常加速度运动是精确的，因为前半段 drift 用初始 :math:`p_x` 、后半段用踢后 :math:`p_{x0} + \Delta p_x` ，平均正好给出抛物线。

.. note::

  上述精确性前提是 :math:`p_z \approx \mathrm{const}` 。对典型偏转角（几十 mrad）， :math:`p_z` 变化量 :math:`\Delta p_z \approx -\Delta p_x^2 / (2 p_z) \sim 10^{-4}` ，可忽略。因此 DKD 无需额外的 ``model="exact"`` 模式。


薄透镜与厚透镜的 kick 一致性
------------------------------

薄透镜和厚透镜的踢角 :math:`\Delta p_x` **完全相同** ，均由积分场 :math:`\mathrm{exl}` 计算。区别仅在于位置变化：

.. list-table::
  :header-rows: 1
  :widths: 20 40 40

  * -
    - 薄透镜（ :math:`L = 0` ）
    - 厚透镜 DKD（ :math:`L > 0` ）
  * - kick :math:`\Delta p_x`
    - :math:`\mathrm{exl} / (\beta_0 c \cdot B\rho)`
    - 相同
  * - 位置变化 :math:`\Delta x`
    - 0
    - :math:`p_x L / p_z + \Delta p_x \cdot L / (2 p_z)`
  * - septum 检测
    - 单点（入口位置）
    - 入口位置分类

对 :math:`\Delta p_x = 30\,\mathrm{mrad}` ， :math:`L = 0.5\,\mathrm{m}` ， :math:`p_z \approx 1` 的典型参数，位置差：

.. math::

  \frac{\Delta p_x \cdot L}{2 p_z} \approx \frac{0.03 \times 0.5}{2} = 7.5\,\mathrm{mm}

该量级在注入/引出场景中不可忽略（ septum 间隙通常为 mm 量级），因此建议使用厚透镜模式。


Septum 逻辑
-----------

静电偏转板的核心物理特征是 **并非所有粒子都感受电场** 。 septum （切割板/丝）将孔径分为：

- **无场区** ：循环束流所在区域，粒子不受电场作用，纯漂移
- **有场区** ：注入/引出束流所在区域，粒子受电场偏转
- **极板/丝区域** （ septum 厚度内）：粒子撞击切割板/丝，标记为丢失

判定规则
~~~~~~~~

septum 方向由哪个场分量非零直接决定：

- :math:`E_x \neq 0` （ ``exl`` 非零）：极板为竖直方向， septum 为竖直线，检测 :math:`x` 坐标
- :math:`E_y \neq 0` （ ``eyl`` 非零）：极板为水平方向， septum 为水平线，检测 :math:`y` 坐标

``septum_x_position`` 的符号决定哪侧有场—— **场总是在远离束流中心的一侧** ：

.. list-table::
  :header-rows: 1
  :widths: 25 25 25 25

  * - septum_x_position
    - 无场区（循环束流）
    - 极板/丝区域
    - 有场区（偏转束流）
  * - :math:`> 0`
    - :math:`x \le s_x`
    - :math:`s_x < x \le s_x + t`
    - :math:`x > s_x + t`
  * - :math:`< 0`
    - :math:`x \ge s_x`
    - :math:`s_x - t \le x < s_x`
    - :math:`x < s_x - t`

其中 :math:`s_x` 为 ``septum x position`` ， :math:`t` 为 ``septum thickness`` 。 ``septum y position`` 的规则同理，将 :math:`x` 替换为 :math:`y` 。

.. raw:: html

  <div style="text-align: center">
  <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="400" height="300" fill="#1a1a2e"/>
    <!-- axes -->
    <line x1="20" y1="150" x2="380" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="200" y1="20" x2="200" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="385" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="206" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <!-- septum plate (loss zone) -->
    <rect x="260" y="30" width="16" height="240" fill="#e94560" fill-opacity="0.3" stroke="#e94560" stroke-width="1.5"/>
    <!-- field region (right of plate) -->
    <rect x="276" y="30" width="104" height="240" fill="#00d2ff" fill-opacity="0.08" stroke="none"/>
    <!-- field-free region (left of septum) -->
    <text x="140" y="90" fill="#00d2ff" font-size="13" font-family="monospace">无场区</text>
    <text x="140" y="108" fill="#00d2ff" font-size="11" font-family="monospace">循环束流</text>
    <!-- loss zone label -->
    <text x="255" y="22" fill="#e94560" font-size="11" font-family="monospace">极板/丝</text>
    <!-- field region label -->
    <text x="305" y="90" fill="#f5a623" font-size="13" font-family="monospace">有场区</text>
    <text x="305" y="108" fill="#f5a623" font-size="11" font-family="monospace">偏转束流</text>
    <!-- septum position marker -->
    <line x1="260" y1="140" x2="260" y2="160" stroke="#e94560" stroke-width="2"/>
    <text x="245" y="175" fill="#e94560" font-size="11" font-style="italic" font-family="monospace">s</text>
    <text x="245" y="188" fill="#e94560" font-size="10" font-family="monospace">x</text>
    <!-- thickness marker -->
    <line x1="260" y1="265" x2="276" y2="265" stroke="#e94560" stroke-width="1.5" stroke-dasharray="2,2"/>
    <text x="262" y="278" fill="#e94560" font-size="10" font-style="italic" font-family="monospace">t</text>
    <!-- E field arrow in field region -->
    <line x1="310" y1="200" x2="370" y2="200" stroke="#f5a623" stroke-width="2" marker-end="url(#arrowhead)"/>
    <defs>
      <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" fill="#f5a623"/>
      </marker>
    </defs>
    <text x="330" y="195" fill="#f5a623" font-size="12" font-style="italic" font-family="monospace">E</text>
    <text x="322" y="215" fill="#f5a623" font-size="10" font-family="monospace">x</text>
  </svg>
  </div>

.. note::

  - 若 ``septum x position`` 未提供（ ``None`` ）但 ``exl`` 非零，则所有粒子都感受电场（无 septum 检测）。这对应均匀场校正元件的用法
  - ``septum thickness`` 默认为 0，此时极板/丝区域宽度为零，粒子要么在无场区要么在有场区


Tilt 旋转
---------

``tilt`` 参数实现元件绕 :math:`s` 轴的 roll 旋转。遵循 MAD-X 约定： **正角度代表顺时针旋转** （沿 :math:`+s` 方向看去）。

.. raw:: html

  <div style="text-align: center">
  <svg width="350" height="250" xmlns="http://www.w3.org/2000/svg">
    <rect width="350" height="250" fill="#1a1a2e"/>
    <!-- s axis (into page, marked as dot) -->
    <circle cx="175" cy="125" r="4" fill="#888"/>
    <text x="182" y="120" fill="#888" font-size="12" font-family="monospace">s</text>
    <!-- original (untilted) frame: x and y axes -->
    <line x1="175" y1="125" x2="315" y2="125" stroke="#555" stroke-width="1" stroke-dasharray="4,3"/>
    <line x1="175" y1="125" x2="175" y2="30" stroke="#555" stroke-width="1" stroke-dasharray="4,3"/>
    <text x="320" y="130" fill="#555" font-size="11" font-family="monospace">x</text>
    <text x="180" y="28" fill="#555" font-size="11" font-family="monospace">y</text>
    <!-- tilted frame (clockwise by tilt) -->
    <line x1="175" y1="125" x2="305" y2="65" stroke="#00d2ff" stroke-width="2"/>
    <line x1="175" y1="125" x2="95" y2="35" stroke="#00d2ff" stroke-width="2"/>
    <text x="310" y="60" fill="#00d2ff" font-size="12" font-family="monospace">x'</text>
    <text x="80" y="32" fill="#00d2ff" font-size="12" font-family="monospace">y'</text>
    <!-- rotation arc -->
    <path d="M 250 125 A 75 75 0 0 0 225 55" fill="none" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="3,2"/>
    <text x="258" y="95" fill="#f5a623" font-size="12" font-style="italic" font-family="monospace">tilt</text>
    <!-- clockwise arrow on arc -->
    <polygon points="225,55 232,58 228,50" fill="#f5a623"/>
  </svg>
  </div>

tilt 不影响积分方法的选择——它只在入口和出口做瞬时坐标变换：

::

  入口:  顺时针旋转 :math:`(x, y, p_x, p_y)` by :math:`+\varphi`  → 进入元件自然坐标系
  内部:  DKD 或薄透镜，在自然坐标系中追踪（场沿 :math:`x'` ， septum 沿 :math:`x'` ）
  出口:  逆时针旋转 :math:`(x, y, p_x, p_y)` by :math:`-\varphi`  → 回到实验室坐标系

顺时针旋转矩阵：

.. math::

  x' = x \cos\varphi - y \sin\varphi

.. math::

  y' = x \sin\varphi + y \cos\varphi

.. math::

  p_x' = p_x \cos\varphi - p_y \sin\varphi

.. math::

  p_y' = p_x \sin\varphi + p_y \cos\varphi

Drift 本身是坐标无关的（自由空间传播不依赖横向坐标系方向）， kick 在自然坐标系中沿 :math:`x'` 方向， septum 在自然坐标系中是 :math:`x' = \mathrm{const}` 的直线。所有物理都在自然坐标系中完成。


整体追踪流程
------------

::

  ====== 薄透镜 (length = 0) ======

    1. Tilt 旋转 (若有)
    2. 分类粒子: 无场区 / 有场区 / 撞击极板
    3. 无场区: 无操作 (位置和动量均不变)
    4. 有场区: 纯 kick (Δpx = exl / (β₀c·Bρ), Δpy = eyl / (β₀c·Bρ))
    5. 撞击极板: tag 取负, 记录 lost_position/lost_turn
    6. Tilt 旋转回 (若有)

  ====== 厚透镜 (length > 0) ======

    1. Tilt 旋转 (若有)
    2. 分类粒子: 无场区 / 有场区 / 撞击极板
    3. 无场区: 纯 Drift(L)
    4. 有场区: DKD
       Drift(L/2) → Kick → Drift(L/2)
    5. 撞击极板: tag 取负, 记录 lost_position/lost_turn
    6. Tilt 旋转回 (若有)

  厚透镜中的 Drift 更新连续的束团相对纵向坐标，不做环周折叠。


接口参数
--------

.. list-table::
  :header-rows: 1
  :widths: 22 28 10 10 30

  * - 属性名
    - JSON key
    - 类型
    - 单位
    - 说明
  * - ``s``
    - ``s (m)``
    - float
    - m
    - 元件在束线中的纵向位置
  * - ``name``
    - ``name``
    - str
    - -
    - 元件名称
  * - ``length``
    - ``length (m)``
    - float
    - m
    - 极板长度（ :math:`\ge 0` ； :math:`= 0` 为薄透镜）
  * - ``ex``
    - ``ex (v/m)``
    - float
    - V/m
    - 水平电场强度，默认 0
  * - ``ey``
    - ``ey (v/m)``
    - float
    - V/m
    - 垂直电场强度，默认 0
  * - ``exl``
    - ``exl (v)``
    - float
    - V
    - 水平积分场 :math:`E_x L` ，默认由 :math:`E_x \cdot L` 推导
  * - ``eyl``
    - ``eyl (v)``
    - float
    - V
    - 垂直积分场 :math:`E_y L` ，默认由 :math:`E_y \cdot L` 推导
  * - ``tilt``
    - ``tilt (rad)``
    - float
    - rad
    - 绕 :math:`s` 轴 roll 角，正值为顺时针，默认 0
  * - ``septum_x_position``
    - ``septum x position (m)``
    - float
    - m
    - :math:`x` 方向 septum 位置（ ``exl`` 非零时生效），默认 ``None``
  * - ``septum_y_position``
    - ``septum y position (m)``
    - float
    - m
    - :math:`y` 方向 septum 位置（ ``eyl`` 非零时生效），默认 ``None``
  * - ``septum_thickness``
    - ``septum thickness (m)``
    - float
    - m
    - septum 极板/丝厚度，默认 0
  * - ``aperture_type``
    - ``aperture type``
    - str
    - -
    - 孔径类型，默认 ``off``
  * - ``aperture_value``
    - ``aperture value``
    - list
    - -
    - 孔径参数值，默认 ``[]``

.. note::

  - 场强（ ``ex`` / ``ey`` ）与积分场（ ``exl`` / ``eyl`` ）两种输入方式：

    - 厚透镜：提供 ``ex`` / ``ey`` 时， :math:`\mathrm{exl} = E_x \cdot L` 自动计算；提供 ``exl`` / ``eyl`` 时， :math:`E_x = \mathrm{exl} / L` 自动反推
    - 薄透镜：直接使用 ``exl`` / ``eyl`` （ ``ex`` / ``ey`` 在 :math:`L = 0` 时无意义）

  - ``ex`` 和 ``ey`` 通常只有一个非零（水平或垂直偏转板）。若两者同时非零，将发出警告
  - ``septum x position`` / ``septum y position`` 仅在对应方向场分量非零时生效
  - ``Command`` 字段应设为 ``elseparator``


使用示例
--------

薄透镜水平偏转
~~~~~~~~~~~~~~

以下示例在 :math:`s = 10.0` m 处放置一个薄透镜静电偏转板，积分场 :math:`\mathrm{exl} = 1 \times 10^5` V：

.. code-block:: json

  {
      "ES1": {
          "S (m)": 10.0,
          "Command": "elseparator",
          "ExL (V)": 1e5
      }
  }

粒子获得水平踢角 :math:`\Delta p_x = \mathrm{exl} / (\beta_0 c \cdot B\rho)` 。位置不变。

厚透镜水平偏转
~~~~~~~~~~~~~~

以下示例在 :math:`s = 15.0` m 处放置一个厚透镜静电偏转板，长度 0.5 m，场强 :math:`E_x = 2 \times 10^5` V/m：

.. code-block:: json

  {
      "ES2": {
          "S (m)": 15.0,
          "Command": "elseparator",
          "Length (m)": 0.5,
          "Ex (V/m)": 2e5
      }
  }

积分场 :math:`\mathrm{exl} = 2 \times 10^5 \times 0.5 = 1 \times 10^5` V。采用 DKD 追踪，粒子走抛物线轨迹。

带 Septum 的注入偏转板
~~~~~~~~~~~~~~~~~~~~~~

以下示例模拟一个注入用静电偏转板， septum 位于 :math:`x = 5` mm， septum 厚度 2 mm：

.. code-block:: json

  {
      "ES3": {
          "S (m)": 20.0,
          "Command": "elseparator",
          "Length (m)": 0.3,
          "Ex (V/m)": 3e5,
          "Septum X Position (m)": 0.005,
          "Septum Thickness (m)": 0.002
      }
  }

粒子分类：

- :math:`x \le 5` mm：无场区，纯漂移（循环束流）
- :math:`5\,\mathrm{mm} < x \le 7` mm：撞击极板/丝，标记丢失
- :math:`x > 7` mm：有场区， DKD 偏转（注入束流）

负侧 Septum （引出场景）
~~~~~~~~~~~~~~~~~~~~~~~~

以下示例模拟引出场景， septum 位于 :math:`x = -5` mm，场在左侧：

.. code-block:: json

  {
      "ES4": {
          "S (m)": 25.0,
          "Command": "elseparator",
          "Length (m)": 0.3,
          "Ex (V/m)": -3e5,
          "Septum X Position (m)": -0.005
      }
  }

粒子分类：

- :math:`x \ge -5` mm：无场区，纯漂移（循环束流）
- :math:`x < -5` mm：有场区， DKD 偏转（引出束流）

垂直偏转板
~~~~~~~~~~

以下示例放置一个垂直偏转板，仅 :math:`E_y` 非零：

.. code-block:: json

  {
      "ES5": {
          "S (m)": 30.0,
          "Command": "elseparator",
          "Length (m)": 0.4,
          "Ey (V/m)": 2.5e5,
          "Septum Y Position (m)": 0.005
      }
  }

粒子在 :math:`y` 方向被偏转。 septum 检测沿 :math:`y` 方向。

带 Tilt 的偏转板
~~~~~~~~~~~~~~~~

以下示例放置一个绕 :math:`s` 轴顺时针旋转 30 度的偏转板：

.. code-block:: json

  {
      "ES6": {
          "S (m)": 35.0,
          "Command": "elseparator",
          "Length (m)": 0.5,
          "Ex (V/m)": 2e5,
          "Tilt (rad)": 0.5236,
          "Septum X Position (m)": 0.005
      }
  }

元件自然坐标系顺时针旋转 30 度，场和 septum 均在旋转后的坐标系中定义。入口旋转 → 追踪 → 出口旋转回来。

零场退化
~~~~~~~~

以下示例场强为零，退化为纯漂移（厚透镜）或标记器（薄透镜）：

.. code-block:: json

  {
      "ES7": {
          "S (m)": 40.0,
          "Command": "elseparator",
          "Length (m)": 0.5
      }
  }

:math:`\mathrm{exl} = \mathrm{eyl} = 0` 时，所有粒子纯漂移，无偏转。


应用场景
--------

- **束流注入** ：在注入段放置静电偏转板，将注入束流偏转至与主环闭合轨道匹配的方向。 septum 分隔循环束流与注入束流，仅偏转注入粒子
- **束流引出** ：在引出点放置静电偏转板，将引出束流偏转至引出通道。 septum 确保循环束流不受影响
- **轨道校正** ：无 septum 时（ ``septum x position = None`` ），静电偏转板可作为均匀场校正元件，对所有粒子施加相同踢角
- **低能束流偏转** ：低能段 :math:`\beta\gamma` 较小，静电偏转效率高于磁偏转（电场力与速度无关，磁场力正比于速度），常用于低能注入线
- **快引出系统** ：静电偏转板响应速度快（纳秒级脉冲），适用于快引出和逐束团引出


参考文献
--------

- MAD-X User's Guide, "ELSEPARATOR" section ( ``ex`` / ``ey`` / ``ex_l`` / ``ey_l`` / ``tilt`` 定义)
- Xsuite 源码： ``xtrack/mad_loader.py`` （ ``convert_elseparator = convert_drift_like`` ， xsuite 暂未实现独立 elseparator ）
- Wiedemann, H., "Particle Accelerator Physics", Ch. 4（电场偏转与磁偏转的等效关系）
- Conte, M. & MacKay, W.W., "An Introduction to the Physics of Particle Accelerators", Ch. 7（静电偏转板在注入引出中的应用）
