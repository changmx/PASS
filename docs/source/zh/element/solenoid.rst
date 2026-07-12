螺线管（Solenoid）
==================

本模块介绍 PASS 中的螺线管元件 **Solenoid** ，用于模拟带电粒子在纵向磁场中的运动。螺线管产生沿束流方向的均匀磁场 :math:`B_z`，通过 Larmor 旋转效应耦合水平与垂直平面，同时提供横向聚焦。

PASS 中的螺线管采用 **精确非线性映射** （Larmor 框架下的哈密顿量解析解）。纯螺线管（无多极场叠加）时映射零误差；叠加多极场时采用 Sol-Kick-Sol（SKS）辛积分器。

**代码位置**

- 源文件： ``PASS/commands/element/solenoid.py``
- 类名： ``Solenoid`` （继承自 ``Command`` ）
- 注册名： ``solenoid``
- 核心特征：

  - 采用精确螺线管映射（Larmor 旋转 + 聚焦， :math:`p_z` 逐粒子计算）
  - 无薄透镜模式（螺线管无薄透镜极限， :math:`L=0` 时无效应）
  - 支持多极场叠加（ ``knl`` / ``ksl`` ），使用 SKS 积分器
  - 支持 uniform（2阶蛙跳）和 yoshida4（4阶 Yoshida 组合）积分器
  - :math:`k_s = 0` 时自动退化为纯漂移
  - 色品效应通过逐粒子 :math:`p_z` 自然引入
  - 支持孔径检查


坐标约定
--------

PASS 采用归一化曲线坐标，六维相空间变量为 :math:`(x, p_x, y, p_y, z, \delta)` ：

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


螺线管磁场与归一化强度
----------------------

螺线管产生沿束流方向（ :math:`s` 轴）的均匀磁场：

.. math::

  \vec{B} = (0, \, 0, \, B_z)

归一化螺线管强度定义为：

.. math::

  k_s = \frac{q_0 B_z}{P_0}

其中 :math:`q_0` 为参考粒子电荷， :math:`P_0` 为参考粒子动量。 PASS 中用户直接指定 :math:`k_s` （ ``ks`` ）。

定义半强度：

.. math::

  \text{sk} = \frac{k_s}{2}

Larmor 旋转角为：

.. math::

  \theta = \frac{\text{sk} \cdot L}{p_z} = \frac{k_s L}{2 p_z}

其中 :math:`p_z` 为粒子的归一化纵向动量分量（逐粒子不同，见下文）， :math:`L` 为螺线管长度。


物理推导
--------

哈密顿量
~~~~~~~~

在直线坐标系中，螺线管的哈密顿量为：

.. math::

  H_{\text{sol}} = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2} + \frac{k_s^2}{8}(x^2 + y^2) - \frac{k_s}{2}(x p_y - y p_x)

其中各项物理含义：

.. list-table::
  :header-rows: 1
  :widths: 40 60

  * - 项
    - 物理含义
  * - :math:`-\sqrt{(1+\delta)^2 - p_x^2 - p_y^2}`
    - 自由传播（精确漂移）
  * - :math:`\frac{k_s^2}{8}(x^2 + y^2)`
    - 螺线管聚焦（等效四极分量）
  * - :math:`-\frac{k_s}{2}(x p_y - y p_x)`
    - Larmor 旋转（ :math:`x` - :math:`y` 耦合）

.. note::

  Larmor 旋转项 :math:`-\frac{k_s}{2}(x p_y - y p_x)` **同时依赖位置和动量** ，这是螺线管与四极铁的本质区别。四极铁的踢角项仅依赖位置，可以将哈密顿量干净地分裂为漂移和踢角两部分（DKD 积分器）。螺线管的 Larmor 旋转项不可分裂为纯位置或纯动量的部分，因此 **不能使用普通漂移做 DKD 积分** 。

Larmor 框架与精确解
~~~~~~~~~~~~~~~~~~~~~

做 Larmor 变换——将横截面坐标系绕 :math:`s` 轴旋转角度 :math:`\theta = \text{sk} \cdot s / p_z` ，定义 Larmor 框架下的正则动量：

.. math::

  p_{k1} = p_x + \text{sk} \cdot y

.. math::

  p_{k2} = p_y - \text{sk} \cdot x

在 Larmor 框架中， :math:`p_{k1}` 和 :math:`p_{k2}` 是 **守恒量** （不随 :math:`s` 变化），因此纵向动量分量：

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_{k1}^2 - p_{k2}^2}

在整个螺线管内逐粒子恒定。这使得螺线管映射可以 **精确求解** ，无需近似。

精确螺线管映射
~~~~~~~~~~~~~~~~~~

给定螺线管长度 :math:`L` ， Larmor 旋转角：

.. math::

  \theta = \frac{\text{sk} \cdot L}{p_z}

映射分为旋转和漂移两步：

**步骤 1：Larmor 旋转** （将坐标旋转到 :math:`s=L` 处的 Larmor 框架）

.. math::

  \text{rps}_0 = \cos\theta \cdot x + \sin\theta \cdot y

.. math::

  \text{rps}_1 = \cos\theta \cdot p_x + \sin\theta \cdot p_y

.. math::

  \text{rps}_2 = \cos\theta \cdot y - \sin\theta \cdot x

.. math::

  \text{rps}_3 = \cos\theta \cdot p_y - \sin\theta \cdot p_x

**步骤 2：Larmor 框架中的漂移** （等效漂移长度 :math:`\sin\theta / \text{sk}` ）

.. math::

  x' = \cos\theta \cdot \text{rps}_0 + \frac{\sin\theta}{\text{sk}} \cdot \text{rps}_1

.. math::

  p_x' = \cos\theta \cdot \text{rps}_1 - \text{sk} \cdot \sin\theta \cdot \text{rps}_0

.. math::

  y' = \cos\theta \cdot \text{rps}_2 + \frac{\sin\theta}{\text{sk}} \cdot \text{rps}_3

.. math::

  p_y' = \cos\theta \cdot \text{rps}_3 - \text{sk} \cdot \sin\theta \cdot \text{rps}_2

**纵向坐标更新** ：

.. math::

  \Delta\zeta = L \cdot \left(1 - \frac{1+\delta}{p_z \cdot \text{rvv}}\right)

其中 :math:`\text{rvv} = \beta / \beta_0` 为粒子速度与参考粒子速度之比：

.. math::

  \beta = \frac{(1+\delta) \, \beta_0 \gamma_0}{\sqrt{1 + \left[(1+\delta) \, \beta_0 \gamma_0\right]^2}}

.. note::

  - :math:`p_z` 逐粒子不同（包含 :math:`\delta` 和 Larmor 动量的贡献），因此映射是 **精确非线性** 的
  - 当 :math:`k_s \to 0` 时， :math:`\sin\theta/\text{sk} \to L/p_z` ，映射退化为精确漂移


为什么螺线管没有薄透镜模式
--------------------------

四极铁的薄透镜极限（ :math:`L \to 0` ， :math:`k_1 \to \infty` ， :math:`k_1 L = \text{const}` ）给出有限的动量踢角 :math:`\Delta p_x = -k_{1L} \cdot x` ，物理上自洽。

螺线管的薄透镜极限（ :math:`L \to 0` ， :math:`k_s \to \infty` ， :math:`k_s L = \text{const}` ）存在根本困难：

- Larmor 旋转角 :math:`\theta = k_s L / (2 p_z)` 有限 ✓
- 聚焦项 :math:`\text{sk} \cdot \sin\theta = (k_s/2) \cdot \sin\theta \to \infty` 发散 ✗

位置和动量的缩放行为不对称：旋转角有限但聚焦力发散，薄透镜极限不存在。

因此 PASS 中螺线管 :math:`L = 0` 时 **无效应** （恒等映射），不提供薄透镜模式。


多极场叠加与 SKS 积分器
-----------------------

当螺线管内部叠加横向多极场分量（ :math:`k_{nl}` / :math:`k_{sl}` ）时，总哈密顿量为：

.. math::

  H = H_{\text{sol}} + H_{\text{mult}}

其中 :math:`H_{\text{mult}}` 为多极铁踢角哈密顿量（仅依赖位置）。由于 :math:`H_{\text{sol}}` 和 :math:`H_{\text{mult}}` 不对易，需要分裂算符法。

PASS 采用 **Sol-Kick-Sol** （SKS）积分器，与四极铁的 DKD 完全平行：

.. math::

  \mathcal{M}_{\text{SKS}}(\Delta s) = \text{Sol}\!\left(\frac{\Delta s}{2}\right) \circ \text{Kick}(\Delta s) \circ \text{Sol}\!\left(\frac{\Delta s}{2}\right)

其中：

- **Sol** = 精确螺线管映射（ ``_solenoid_exact_cpu`` ），处理 :math:`B_z` 场
- **Kick** = 多极铁踢角（Horner 递归），处理横向多极场

.. list-table::
  :header-rows: 1
  :widths: 25 25 25 25

  * - 
    - 漂移算子
    - 踢角算子
    - 场景
  * - 四极铁 DKD
    - 自由漂移 ``drift_exact``
    - 四极踢角 ``quad_kick``
    - :math:`B_z = 0` ，仅横向梯度场
  * - 螺线管 SKS
    - 螺线管映射 ``solenoid_exact``
    - 多极踢角 ``multipole_kick``
    - :math:`B_z \neq 0` ，叠加横向多极场

.. note::

  SKS 中的 "Sol" 不是自由漂移，而是螺线管精确映射。螺线管内 :math:`B_z` 始终存在，粒子不是在无场空间漂移。如果错误地使用自由漂移替代螺线管映射，将丢失 Larmor 旋转效应。

uniform 积分器（2阶辛）
~~~~~~~~~~~~~~~~~~~~~~~~~~

每个切片采用 Sol-Kick-Sol 结构，即二阶蛙跳（leapfrog）：

.. math::

  S_2(\Delta s) = \text{Sol}\!\left(\frac{\Delta s}{2}\right) \circ \text{Kick}(\Delta s) \circ \text{Sol}\!\left(\frac{\Delta s}{2}\right)

每个切片误差为 :math:`O(\Delta s^3)` ，全局误差为 :math:`O(\Delta s^2)` 。

yoshida4 积分器（4阶辛）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

通过组合三个二阶 SKS 步构造四阶辛映射 [Yoshida 1990]：

.. math::

  S_4(\Delta s) = S_2(z_1 \Delta s) \circ S_2(z_0 \Delta s) \circ S_2(z_1 \Delta s)

其中 Yoshida 系数为：

.. math::

  z_1 = \frac{1}{2 - 2^{1/3}} \approx 1.3512

.. math::

  z_0 = 1 - 2 z_1 \approx -1.7024

每个切片误差为 :math:`O(\Delta s^5)` ，全局误差为 :math:`O(\Delta s^4)` 。


整体追踪流程
------------

根据是否有多极场叠加，螺线管有两种追踪路径：

::

  ====== 厚透镜 (length > 0) ======

  无多极场 (knl/ksl 全零):
    单段精确螺线管映射 Sol(L, ks)
    [零误差，无需切片]

  有多极场 (knl/ksl 非零):
    切片1 → 切片2 → ... → 切片N
    (每个切片: Sol(ds/2) → Kick(ds) → Sol(ds/2))
    其中 ds = L / N

  特殊情况:
    ks = 0 → 退化为纯漂移 Drift(L)
    L = 0 → 无效应（螺线管无薄透镜极限）

完整映射为：

无多极场：

.. math::

  \mathcal{M} = \text{Sol}(L, k_s)

有多极场（ :math:`N` 个切片）：

.. math::

  \mathcal{M} = \left[\mathcal{M}_{\text{SKS}}(\Delta s)\right]^N


色品效应
--------

螺线管的色品效应通过逐粒子 :math:`p_z` 表达式自然引入。

在 Larmor 框架中， :math:`p_z` 依赖 :math:`\delta` 和 Larmor 动量 :math:`p_{k1}, p_{k2}` ：

.. math::

  p_z = \sqrt{(1+\delta)^2 - (p_x + \text{sk} \cdot y)^2 - (p_y - \text{sk} \cdot x)^2}

不同动量偏差 :math:`\delta` 的粒子有不同的 :math:`p_z` ，因此有不同的 Larmor 旋转角 :math:`\theta = \text{sk} \cdot L / p_z` 和不同的等效漂移长度 :math:`\sin\theta / \text{sk}` 。这就是螺线管色品的物理来源——动量依赖的旋转角和聚焦强度。


接口参数
--------

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 35

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
  * - ``length``
    - ``length (m)``
    - float
    - m
    - 元件长度 （必须 :math:`\ge 0` ； :math:`= 0` 时无效应）
  * - ``name``
    - ``name``
    - str
    - -
    - 元件名称
  * - ``ks``
    - ``ks``
    - float
    - :math:`\text{m}^{-1}`
    - 螺线管归一化强度 :math:`k_s = q_0 B_z / P_0` ，默认 0
  * - ``knl``
    - ``kil``
    - list
    - :math:`\text{m}^{-n}`
    - 多极铁法向积分强度数组 :math:`K_{nL}` ，默认 ``[]``
  * - ``ksl``
    - ``kisl``
    - list
    - :math:`\text{m}^{-n}`
    - 多极铁斜向积分强度数组 :math:`K_{sL}` ，默认 ``[]``
  * - ``num_slice``
    - ``num slices``
    - int
    - -
    - 切片数，默认 1（仅多极场叠加时有效）
  * - ``integrator``
    - ``integrator``
    - str
    - -
    - 积分器，可选： ``adaptive`` （默认 ``uniform`` ）、 ``uniform`` 、 ``yoshida4``
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

  - ``knl`` / ``ksl`` 为可选参数。不指定或全零时，螺线管使用单段精确映射（零误差），忽略 ``num_slices`` 和 ``integrator``
  - 指定非零 ``knl`` / ``ksl`` 时，启用 SKS 积分器， ``num_slices`` 和 ``integrator`` 生效
  - ``ks = 0`` 且有长度时，元件退化为纯漂移
  - ``length = 0`` 时，螺线管无效应（不提供薄透镜模式）


使用示例
--------

纯螺线管（精确映射）
~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SOL1": {
          "S (m)": 10.0,
          "Command": "Solenoid",
          "Length (m)": 1.0,
          "ks": 2.0,
          "Aperture Type": "off"
      }
  }

长度 1.0 m，归一化强度 :math:`k_s = 2.0` 。使用单段精确螺线管映射，零误差。

弱螺线管
~~~~~~~~

.. code-block:: json

  {
      "SOL2": {
          "S (m)": 20.0,
          "Command": "Solenoid",
          "Length (m)": 2.0,
          "ks": 0.5,
          "Aperture Type": "off"
      }
  }

弱场螺线管， Larmor 旋转角较小。

反向磁场螺线管
~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SOL3": {
          "S (m)": 30.0,
          "Command": "Solenoid",
          "Length (m)": 1.5,
          "ks": -3.0,
          "Aperture Type": "off"
      }
  }

:math:`k_s < 0` 表示反向磁场， Larmor 旋转方向相反。

螺线管叠加四极场（SKS 积分器）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SOL4": {
          "S (m)": 40.0,
          "Command": "Solenoid",
          "Length (m)": 1.0,
          "ks": 2.0,
          "KiL": [0.0, 0.1],
          "Kisl": [],
          "Num Slices": 4,
          "Integrator": "yoshida4",
          "Aperture Type": "off"
      }
  }

螺线管（ :math:`k_s = 2.0` ）叠加四极分量（ :math:`K_{1L} = 0.1` ），4 个切片，4 阶辛积分器。 ``KiL`` 数组第 0 项为 :math:`K_{0L}` （偶极），第 1 项为 :math:`K_{1L}` （四极）。

零场退化（纯漂移）
~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SOL5": {
          "S (m)": 50.0,
          "Command": "Solenoid",
          "Length (m)": 1.0,
          "ks": 0.0,
          "Aperture Type": "off"
      }
  }

:math:`k_s = 0` 时退化为纯漂移。


应用场景
--------

- **低能束流传输线** ：低能段 :math:`\beta\gamma` 较小，螺线管聚焦效率高于四极铁，常用于注入器和低能传输线
- **电子冷却器** ：螺线管约束电子束与离子束共线运动，用于冷却横向发射度
- **对撞机探测器螺线管** ：大型实验探测器（如 CMS、ATLAS）的螺线管磁场对束流光学有显著影响，需在 lattice 模型中精确计入
- **超导螺线管** ：高场超导螺线管中的多极场误差可通过 ``knl`` / ``ksl`` 参数叠加建模
- **旋转对称束流** ：螺线管的 Larmor 旋转可用于消除 :math:`x` - :math:`y` 耦合或产生特定旋转对称的束流分布
