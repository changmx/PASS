八极铁（Octupole）
==================

本模块介绍 PASS 中的八极铁元件 **Octupole** ，用于模拟带电粒子在八极磁铁中的运动。八极铁提供三次非线性磁场，是加速器中重要的非线性元件，主要用于 Landau 阻尼、振幅依赖 tune 移和共振抑制。

PASS 中的八极铁支持 **厚元件** （ ``length > 0`` ）和 **薄透镜** （ ``length = 0`` ）两种模式，厚元件采用精确漂移-踢角-漂移（DKD-exact）辛积分方案，支持 uniform（2阶）和 yoshida4（4阶）两种辛积分器。

**代码位置**

- 源文件： ``PASS/commands/element/octupole.py``
- 类名： ``Octupole`` （继承自 ``Command`` ）
- 注册名： ``octupole``
- 核心特征：

  - 支持薄透镜模式（ ``length = 0`` ，仅施加八极踢角）
  - 支持厚透镜模式（ ``length > 0`` ，DKD-exact 辛积分）
  - 支持 uniform（2阶蛙跳）和 yoshida4（4阶 Yoshida 组合）积分器
  - 支持正常八极（ ``k3l`` ）和斜八极（ ``k3sl`` ）及其组合
  - 零场（ ``k3l = k3sl = 0`` ）时自动退化为纯漂移
  - 高阶非线性效应通过精确漂移自然引入
  - 支持孔径检查


坐标约定
--------

PASS 采用与 Xsuite 一致的归一化曲线坐标，六维相空间变量为 :math:`(x, p_x, y, p_y, z, \delta)` ：

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

其中 :math:`P_0` 为参考粒子动量， :math:`\beta_0 = v_0 / c` 为参考粒子归一化速度， :math:`s` 为沿参考轨道的弧长， :math:`t` 为时间。

纵向动量分量定义为：

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

荷质比因子：

.. math::

  \chi = \frac{q}{q_0} \cdot \frac{m_0}{m}

对于同种粒子束 :math:`\chi = 1` 。


八极磁场与归一化强度
--------------------

八极磁铁的磁场在横向平面内三次分布。用复数表示：

.. math::

  B_y + i B_x = \frac{1}{6}(B''' + i B'''_s)(x + i y)^3

其中 :math:`B'''` 为正常八极场三阶导数， :math:`B'''_s` 为斜八极场三阶导数。展开后：

.. math::

  B_y = \frac{1}{6} B''' (x^3 - 3 x y^2) - \frac{1}{6} B'''_s (3 x^2 y - y^3)

.. math::

  B_x = \frac{1}{6} B''' (3 x^2 y - y^3) + \frac{1}{6} B'''_s (x^3 - 3 x y^2)

归一化八极强度定义为：

.. math::

  K_3 = \frac{q_0 B'''}{6 P_0}

.. math::

  K_{3s} = \frac{q_0 B'''_s}{6 P_0}

积分强度为：

.. math::

  K_{3L} = K_3 \cdot L, \qquad K_{3sL} = K_{3s} \cdot L

其中 :math:`L` 为磁铁长度。PASS 中用户直接指定 :math:`K_{3L}` （ ``k3l`` ）和 :math:`K_{3sL}` （ ``k3sl`` ），厚透镜时内部自动解出 :math:`K_3 = K_{3L} / L` 和 :math:`K_{3s} = K_{3sL} / L` 。


整体追踪流程
------------

根据磁铁长度，八极铁有两种追踪模式：

**薄透镜模式** （ :math:`L = 0` ）

::

  ====== 薄透镜 (length = 0) ======

  单次八极踢角 Kick(K3L, K3sL)
  [位置不变，仅动量跳变]

**厚透镜模式** （ :math:`L > 0` ）

::

  ====== 厚透镜 (length > 0) ======

  切片1 → 切片2 → ... → 切片N
  (每个切片: Drift(ds/2) → Kick(ds) → Drift(ds/2))

  其中 ds = L / N

  若 K3L = 0 且 K3sL = 0：退化为单次精确漂移 Drift(L)

完整映射为：

薄透镜：

.. math::

  \mathcal{M}_{\text{thin}} = \text{Kick}(K_{3L}, K_{3sL})

厚透镜（N 个切片）：

.. math::

  \mathcal{M}_{\text{thick}} = \left[\mathcal{M}_{\text{DKD}}(\Delta s)\right]^N

其中每个切片的 DKD 映射为：

.. math::

  \mathcal{M}_{\text{DKD}}(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

.. note::

  - 薄透镜模式不改变粒子的位置坐标 :math:`(x, y, z)` ，仅施加动量踢角
  - 厚透镜模式的色散相关效应通过精确漂移中的 :math:`p_z` 表达式自然引入
  - 当 :math:`K_{3L} = 0` 且 :math:`K_{3sL} = 0` 时，厚透镜退化为纯漂移，避免无意义的空踢角循环


物理推导
--------

哈密顿量
~~~~~~~~

在直线坐标系中（八极铁无曲率， :math:`h = 0` ），八极铁的哈密顿量为：

.. math::

  H_{\text{oct}} = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2} + \frac{\chi}{24}\left[K_3(x^4 - 6 x^2 y^2 + y^4) + K_{3s}(4 x^3 y - 4 x y^3)\right]

将其拆分为传播部分（精确漂移 :math:`H_D` ）和踢角部分（ :math:`H_K` ）：

.. math::

  H_D = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  H_K = \frac{\chi}{24}\left[K_3(x^4 - 6 x^2 y^2 + y^4) + K_{3s}(4 x^3 y - 4 x y^3)\right]

其中 :math:`H_D` 是精确漂移哈密顿量（保留 :math:`p_z` 的根号，不做小动量展开）， :math:`H_K` 是八极踢角。这是 **分裂算符法** （split-operator）的标准做法：将哈密顿量拆分为可解析求解的部分，分别施加映射，再组合为辛积分器。

精确漂移映射 D
~~~~~~~~~~~~~~~~~~

传播部分的哈密顿方程给出精确直漂移：

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  x \leftarrow x + \frac{p_x}{p_z} \cdot L_D

.. math::

  y \leftarrow y + \frac{p_y}{p_z} \cdot L_D

.. math::

  \zeta \leftarrow \zeta + L_D \cdot \left(1 - \frac{\beta_0}{\beta} \cdot \frac{1+\delta}{p_z}\right)

其中 :math:`L_D` 为漂移长度， :math:`\beta` 为粒子实际归一化速度：

.. math::

  \beta = \frac{(1+\delta) \, \beta_0 \gamma_0}{\sqrt{1 + \left[(1+\delta) \, \beta_0 \gamma_0\right]^2}}

.. note::

  "exact" 的含义：漂移部分保留精确根号 :math:`p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}` ，不做 :math:`p_x \ll 1` 的小动量展开。近似仅在于将传播部分与踢角部分分离（分裂算符法）。该公式与漂移节（Drift）和四极铁（Quadrupole）、六极铁（Sextupole）中的精确漂移完全一致。

八极踢角映射 K
~~~~~~~~~~~~~~~~~~

踢角部分为薄透镜映射（位置不变，仅动量跳变）。由哈密顿方程 :math:`\dot{p}_x = -\partial H / \partial x` ， :math:`\dot{p}_y = -\partial H / \partial y` 得：

.. math::

  \Delta p_x = -\frac{\chi}{6} K_{3L} (x^3 - 3 x y^2) + \frac{\chi}{6} K_{3sL} (3 x^2 y - y^3)

.. math::

  \Delta p_y = \frac{\chi}{6} K_{3L} (3 x^2 y - y^3) + \frac{\chi}{6} K_{3sL} (x^3 - 3 x y^2)

其中踢角有效长度已包含在积分强度 :math:`K_{3L}` 和 :math:`K_{3sL}` 中。

复数表示验证： :math:`(x+iy)^3 = (x^3 - 3xy^2) + i(3x^2y - y^3)` ，实部对应正常八极，虚部对应斜八极。

各项物理含义：

.. list-table::
  :header-rows: 1
  :widths: 30 15 55

  * - 项
    - 来源
    - 物理含义
  * - :math:`-\frac{\chi}{6} K_{3L} (x^3 - 3xy^2)`
    - :math:`\frac{\chi K_3}{24} x^4`
    - 水平三次非线性聚焦（正比于 :math:`x^3` ）
  * - :math:`+\frac{\chi}{6} K_{3L} (3x^2y - y^3)`
    - :math:`-\frac{\chi K_3}{4} x^2 y^2`
    - 水平-垂直耦合踢角
  * - :math:`+\frac{\chi}{6} K_{3sL} (3x^2y - y^3)`
    - :math:`\frac{\chi K_{3s}}{6} x^3 y`
    - 斜八极水平耦合踢角
  * - :math:`+\frac{\chi}{6} K_{3sL} (x^3 - 3xy^2)`
    - :math:`-\frac{\chi K_{3s}}{24} y^4`
    - 斜八极垂直三次非线性聚焦

对于薄透镜模式，直接使用积分强度 :math:`K_{3L}` 和 :math:`K_{3sL}` 。对于 DKD 模式，使用 :math:`K_3 \Delta s` 和 :math:`K_{3s} \Delta s` 。

.. note::

  正常八极铁（ :math:`K_3 > 0` ）在水平方向对正偏移粒子提供恢复力（正比于 :math:`x^3` ），这是与六极铁（正比于 :math:`x^2` ）和四极铁（正比于 :math:`x` ）的关键区别。八极铁的聚焦力与位置三次方成正比，是非线性元件——远离轴线的粒子受到远强于近轴粒子的偏转。

  与四极铁、六极铁的对比：四极踢角线性依赖于 :math:`x` ，六极踢角二次依赖于 :math:`x` ，八极踢角三次依赖于 :math:`x` 。这意味着八极铁不改变参考轨道上的粒子（ :math:`x = y = 0` 时踢角为零），也不改变线性轨道（小振幅粒子的踢角极小），但对大振幅粒子产生强烈的非线性偏转。这一特性使八极铁成为 Landau 阻尼的理想元件。

  斜八极铁（ :math:`K_{3s} \neq 0` ）将八极作用旋转 :math:`\pi / 8` ，产生不同的 :math:`x` - :math:`y` 耦合模式。实际中常用于模拟安装旋转误差或驱动特定高阶耦合共振。


uniform 积分器（2阶辛）
~~~~~~~~~~~~~~~~~~~~~~~~~~

每个切片采用漂移-踢角-漂移（DKD）结构，即二阶蛙跳（leapfrog）：

.. math::

  S_2(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

每个切片误差为 :math:`O(\Delta s^3)` ，全局误差为 :math:`O(\Delta s^2)` 。二阶辛积分器，每步都是正则变换。


yoshida4 积分器（4阶辛）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

通过组合三个二阶 DKD 步构造四阶辛映射 [Yoshida 1990]：

.. math::

  S_4(\Delta s) = S_2(z_1 \Delta s) \circ S_2(z_0 \Delta s) \circ S_2(z_1 \Delta s)

其中 Yoshida 系数为：

.. math::

  z_1 = \frac{1}{2 - 2^{1/3}} \approx 1.3512

.. math::

  z_0 = 1 - 2 z_1 \approx -1.7024

.. note::

  :math:`z_0 < 0` 意味着中间一步是反向追踪（漂移和踢角的 "长度" 为负）。这是 Yoshida 组合方法的数学要求，在辛映射群中完全自洽。每个切片误差为 :math:`O(\Delta s^5)` ，全局误差为 :math:`O(\Delta s^4)` 。


振幅依赖 tune 移
----------------

八极铁最重要的物理效应是产生振幅依赖的 tune 移（amplitude-dependent tune shift, ADTS），这是 Landau 阻尼的物理基础。

物理机制
~~~~~~~~

考虑单平面运动（ :math:`y = 0` ），正常八极铁的踢角为：

.. math::

  \Delta p_x = -\frac{\chi}{6} K_{3L} \, x^3

在平滑近似下，一个回旋中的等效频率移动为：

.. math::

  \Delta Q_x = -\frac{\chi K_{3L}}{16\pi} \oint \beta_x^2 \, ds \cdot J_x

其中 :math:`J_x = \frac{1}{2\beta_x}(x^2 + (\beta_x p_x + \alpha_x x)^2)` 为作用量。tune 移正比于振幅的平方（即作用量 :math:`J_x` ），这意味着大振幅粒子的 tune 偏离小振幅粒子，使束流在 tune 空间中展开，从而实现 Landau 阻尼。

.. note::

  - 八极铁产生的 tune 移正比于 :math:`J` （作用量），即正比于振幅平方
  - 六极铁通过色散-动量偏差耦合产生的 tune 移正比于 :math:`\delta` ，即正比于动量偏差
  - 四极铁的 tune 移与振幅无关（线性元件）
  - 八极铁的 ADTS 不依赖色散（ :math:`\eta_x` ），可在无色散位置使用


自然包含的高阶效应
------------------

DKD-exact 方案中，理想八极磁铁的所有非线性效应天然包含，无需任何额外处理：

.. list-table::
  :header-rows: 1
  :widths: 30 70

  * - 效应
    - 来源
  * - 振幅依赖 tune 移
    - kick 中 :math:`x^3` 项，大振幅粒子受更强偏转
  * - 高阶色散
    - drift 中精确 :math:`p_z` 使色散演化含所有阶次的 :math:`\delta` 依赖
  * - 路径长度效应（ :math:`R_{56}` 等）
    - drift 中 :math:`\zeta` 更新包含完整的 :math:`R_{56}` , :math:`T_{566}` 等高阶项
  * - 厚透镜分布效应
    - DKD 多切片中 drift 改变 :math:`x` ，后续 kick 感受更新后的坐标
  * - :math:`x` - :math:`y` 耦合
    - kick 中 :math:`x^2 y` , :math:`xy^2` 交叉项
  * - 共振驱动
    - 四阶共振（ :math:`4Q_x` , :math:`2Q_x \pm 2Q_y` , :math:`4Q_y` 等）

.. note::

  唯一近似是 split-operator 积分器的离散化误差（uniform 为 :math:`O(\Delta s^2)` ，yoshida4 为 :math:`O(\Delta s^4)` ），可通过增加切片数控制。这是数学方法的截断误差，不是物理效应的遗漏。


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
    - 元件长度（必须 :math:`\ge 0` ； :math:`= 0` 时为薄透镜）
  * - ``name``
    - ``name``
    - str
    - -
    - 元件名称
  * - ``k3l``
    - ``k3l``
    - float
    - :math:`\text{m}^{-3}`
    - 正常八极积分强度 :math:`K_{3L}` ，默认 0
  * - ``k3sl``
    - ``k3sl``
    - float
    - :math:`\text{m}^{-3}`
    - 斜八极积分强度 :math:`K_{3sL}` ，默认 0
  * - ``num_slice``
    - ``num slices``
    - int
    - -
    - 切片数，默认 1（仅厚透镜有效）
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


使用示例
--------

厚透镜正常八极铁
~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "OCT1": {
          "S (m)": 10.0,
          "Command": "Octupole",
          "Length (m)": 0.5,
          "K3L": 500.0,
          "Num Slices": 5,
          "Integrator": "yoshida4",
          "Aperture Type": "off"
      }
  }

正常八极铁（ :math:`K_{3L} > 0` ），长度 0.5 m，5 个切片，4 阶辛积分。用于 Landau 阻尼。

薄透镜八极铁
~~~~~~~~~~~~

.. code-block:: json

  {
      "OCT2": {
          "S (m)": 20.0,
          "Command": "Octupole",
          "Length (m)": 0.0,
          "K3L": 1000.0,
          "Aperture Type": "off"
      }
  }

零长度八极铁，仅施加 :math:`K_{3L}` 薄透镜踢角，无 body 追踪。

负八极铁
~~~~~~~~

.. code-block:: json

  {
      "OCT3": {
          "S (m)": 30.0,
          "Command": "Octupole",
          "Length (m)": 0.4,
          "K3L": -500.0,
          "Num Slices": 1,
          "Integrator": "uniform",
          "Aperture Type": "off"
      }
  }

负八极铁（ :math:`K_{3L} < 0` ），提供与正八极铁相反方向的 tune 移。

斜八极铁
~~~~~~~~

.. code-block:: json

  {
      "OCT4": {
          "S (m)": 40.0,
          "Command": "Octupole",
          "Length (m)": 0.3,
          "K3L": 0.0,
          "K3SL": 300.0,
          "Num Slices": 1,
          "Integrator": "uniform",
          "Aperture Type": "off"
      }
  }

纯斜八极铁（ :math:`K_{3L} = 0` , :math:`K_{3sL} \neq 0` ），产生与正常八极铁旋转 :math:`\pi / 8` 的耦合效应。

正常 + 斜八极组合
~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "OCT5": {
          "S (m)": 50.0,
          "Command": "Octupole",
          "Length (m)": 0.5,
          "K3L": 500.0,
          "K3SL": 100.0,
          "Num Slices": 3,
          "Integrator": "yoshida4",
          "Aperture Type": "circle",
          "Aperture Value": [0.04]
      }
  }

同时含正常和斜八极分量的组合八极铁（模拟安装旋转误差），带圆形孔径检查。


应用场景
--------

- **Landau 阻尼** ：八极铁产生振幅依赖 tune 移，使大振幅粒子的 tune 偏离工作点，提供相干振荡的 Landau 阻尼，抑制束流不稳定性
- **共振抑制** ：通过调整八极铁强度，将粒子 tune 推离危险共振线，避免共振激发导致的束流损失
- **动态孔径控制** ：八极铁的三次非线性场限制稳定相空间区域，影响束流寿命和动态孔径
- **非线性耦合校正** ：使用斜八极铁（ ``k3sl`` ）控制高阶 :math:`x` - :math:`y` 耦合
- **四阶共振驱动** ：在特定相位放置八极铁驱动四阶共振（ :math:`4Q_x` , :math:`2Q_x \pm 2Q_y` 等）用于共振引出或束流刮削
- **LHC Landau 阻尼方案** ：在弧区分布八极铁家族（MO），在宽能范围内提供足够的 Landau 阻尼


参考文献
--------

- Xsuite Physics Guide, Sec 1.10.3 (精确漂移), Sec 1.10.5 (多极铁)
- Xsuite 源码： ``xtrack/beam_elements/elements_src/octupole.h`` , ``track_magnet.h`` , ``track_magnet_kick.h`` , ``track_magnet_drift.h``
- Yoshida, H., "Construction of higher order symplectic integrators", Phys. Lett. A 150 (1990)
- MAD-X 物理手册：八极磁场与非线性传输
- Wiedemann, H., "Particle Accelerator Physics", Ch. 4 (非线性束流动力学)
