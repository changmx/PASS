四极铁（Quadrupole）
====================

本模块介绍 PASS 中的四极铁元件 **Quadrupole** ，用于模拟带电粒子在四极磁铁中的运动。四极铁是加速器中最基本的聚焦元件，通过梯度磁场提供线性聚焦力。

PASS 中的四极铁支持 **厚元件** （ ``length > 0`` ）和 **薄透镜** （ ``length = 0`` ）两种模式，厚元件采用精确漂移-踢角-漂移（DKD-exact）辛积分方案，支持 uniform（2阶）和 yoshida4（4阶）两种辛积分器。

**代码位置**

- 源文件： ``PASS/commands/element/quadrupole.py``
- 类名： ``Quadrupole`` （继承自 ``Command`` ）
- 注册名： ``quadrupole``
- 核心特征：

  - 支持薄透镜模式（ ``length = 0`` ，仅施加四极踢角）
  - 支持厚透镜模式（ ``length > 0`` ，DKD-exact 辛积分）
  - 支持 uniform（2阶蛙跳）和 yoshida4（4阶 Yoshida 组合）积分器
  - 支持正常四极（ ``k1l`` ）和斜四极（ ``k1sl`` ）及其组合
  - 零场（ ``k1l = k1sl = 0`` ）时自动退化为纯漂移
  - 色品效应通过精确漂移自然引入
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


四极磁场与归一化强度
--------------------

四极磁铁的磁场在横向平面内线性分布。用复数表示：

.. math::

  B_y + i B_x = (G + i G_s)(x + i y)

其中 :math:`G` 为正常四极梯度， :math:`G_s` 为斜四极梯度。展开后：

.. math::

  B_y = G \cdot x - G_s \cdot y

.. math::

  B_x = G \cdot y + G_s \cdot x

归一化四极强度定义为：

.. math::

  K_1 = \frac{q_0 G}{P_0}

.. math::

  K_{1s} = \frac{q_0 G_s}{P_0}

积分强度为：

.. math::

  K_{1L} = K_1 \cdot L, \qquad K_{1sL} = K_{1s} \cdot L

其中 :math:`L` 为磁铁长度。PASS 中用户直接指定 :math:`K_{1L}` （ ``k1l`` ）和 :math:`K_{1sL}` （ ``k1sl`` ），厚透镜时内部自动解出 :math:`K_1 = K_{1L} / L` 和 :math:`K_{1s} = K_{1sL} / L` 。


整体追踪流程
------------

根据磁铁长度，四极铁有两种追踪模式：

**薄透镜模式** （ :math:`L = 0` ）

::

  ====== 薄透镜 (length = 0) ======

  单次四极踢角 Kick(K1L, K1sL)
  [位置不变，仅动量跳变]

**厚透镜模式** （ :math:`L > 0` ）

::

  ====== 厚透镜 (length > 0) ======

  切片1 → 切片2 → ... → 切片N
  (每个切片: Drift(ds/2) → Kick(ds) → Drift(ds/2))

  其中 ds = L / N

  若 K1L = 0 且 K1sL = 0：退化为单次精确漂移 Drift(L)

完整映射为：

薄透镜：

.. math::

  \mathcal{M}_{\text{thin}} = \text{Kick}(K_{1L}, K_{1sL})

厚透镜（N 个切片）：

.. math::

  \mathcal{M}_{\text{thick}} = \left[\mathcal{M}_{\text{DKD}}(\Delta s)\right]^N

其中每个切片的 DKD 映射为：

.. math::

  \mathcal{M}_{\text{DKD}}(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

.. note::

  - 薄透镜模式不改变粒子的位置坐标 :math:`(x, y, z)` ，仅施加动量踢角
  - 厚透镜模式的色品效应通过精确漂移中的 :math:`p_z` 表达式自然引入（见色品章节）
  - 当 :math:`K_{1L} = 0` 且 :math:`K_{1sL} = 0` 时，厚透镜退化为纯漂移，避免无意义的空踢角循环


物理推导
--------

哈密顿量
~~~~~~~~

在直线坐标系中（四极铁无曲率， :math:`h = 0` ），四极铁的哈密顿量为：

.. math::

  H_{\text{quad}} = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2} + \frac{\chi}{2}\left(K_1 x^2 - K_1 y^2 + 2 K_{1s} x y\right)

将其拆分为传播部分（精确漂移 :math:`H_D` ）和踢角部分（ :math:`H_K` ）：

.. math::

  H_D = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  H_K = \frac{\chi}{2}\left(K_1 x^2 - K_1 y^2 + 2 K_{1s} x y\right)

其中 :math:`H_D` 是精确漂移哈密顿量（保留 :math:`p_z` 的根号，不做小动量展开）， :math:`H_K` 是四极踢角。这是 **分裂算符法** （split-operator）的标准做法：将哈密顿量拆分为可解析求解的部分，分别施加映射，再组合为辛积分器。

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

  "exact" 的含义：漂移部分保留精确根号 :math:`p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}` ，不做 :math:`p_x \ll 1` 的小动量展开。近似仅在于将传播部分与踢角部分分离（分裂算符法）。该公式与漂移节（Drift）和二极铁（SBend）中的精确漂移完全一致。

四极踢角映射 K
~~~~~~~~~~~~~~~~~~

踢角部分为薄透镜映射（位置不变，仅动量跳变）。由哈密顿方程 :math:`\dot{p}_x = -\partial H / \partial x` ， :math:`\dot{p}_y = -\partial H / \partial y` 得：

.. math::

  \Delta p_x = -\chi K_1 L_K \cdot x + \chi K_{1s} L_K \cdot y

.. math::

  \Delta p_y = +\chi K_1 L_K \cdot y + \chi K_{1s} L_K \cdot x

其中 :math:`L_K` 为踢角有效长度。

各项物理含义：

.. list-table::
  :header-rows: 1
  :widths: 30 15 55

  * - 项
    - 来源
    - 物理含义
  * - :math:`-\chi K_1 L_K \cdot x`
    - :math:`\frac{\chi K_1 x^2}{2}`
    - 水平正常四极聚焦（ :math:`K_1 > 0` 时聚焦， :math:`K_1 < 0` 时散焦）
  * - :math:`+\chi K_1 L_K \cdot y`
    - :math:`-\frac{\chi K_1 y^2}{2}`
    - 垂直正常四极散焦（与水平反向）
  * - :math:`+\chi K_{1s} L_K \cdot y`
    - :math:`\chi K_{1s} x y`
    - 斜四极水平耦合踢角
  * - :math:`+\chi K_{1s} L_K \cdot x`
    - :math:`\chi K_{1s} x y`
    - 斜四极垂直耦合踢角

对于薄透镜模式， :math:`L_K = 1` ，直接使用积分强度 :math:`K_{1L}` 和 :math:`K_{1sL}` 。对于 DKD 模式， :math:`L_K = \Delta s` ，使用 :math:`K_1 \Delta s` 和 :math:`K_{1s} \Delta s` 。

.. note::

  正常四极铁（ :math:`K_1 > 0` ）在水平方向聚焦、垂直方向散焦，这是四极磁场 :math:`B_y = G \cdot x` 的直接结果：偏离轴线的粒子受到与偏移成正比的力，水平方向恢复力（聚焦），垂直方向排斥力（散焦）。要实现两个方向同时聚焦，需要交替排列聚焦四极铁（F）和散焦四极铁（D），即 FODO 结构。

  斜四极铁（ :math:`K_{1s} \neq 0` ）将聚焦作用旋转 :math:`\pi / 4` ，产生 :math:`x` - :math:`y` 耦合。实际中常用于耦合校正或模拟安装旋转误差。


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

积分器选择建议：

.. list-table::
  :header-rows: 1
  :widths: 25 20 55

  * - 场景
    - 推荐积分器
    - 原因
  * - 快速模拟
    - uniform
    - 每切片 2 次漂移 + 1 次踢角，计算量小
  * - 高精度模拟
    - yoshida4
    - 4 阶精度，但每切片 6 次漂移 + 3 次踢角
  * - 含空间电荷
    - uniform + 更多切片
    - PIC 求解开销远大于漂移，4阶 Yoshida 需要 3 次 PIC 求解


色品效应
--------

色品（chromaticity）描述了粒子 tune 对动量偏差 :math:`\delta` 的依赖。PASS 的 DKD-exact 模型通过精确漂移的 :math:`p_z` 表达式自然引入色品，无需任何额外处理。

物理机制
~~~~~~~~

在 DKD 积分中，漂移使用精确表达式 :math:`p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}` ，而踢角在 :math:`(x, p_x)` 空间中为 :math:`\Delta p_x = -\chi K_1 \Delta s \cdot x` （不除以 :math:`1+\delta` ）。

转换到 :math:`(x, x')` 空间（其中 :math:`x' = p_x / (1+\delta)` ），等效聚焦强度自动变为：

.. math::

  K_{1,\text{eff}} = \frac{K_1}{1+\delta}

这就是自然色品 :math:`Q'_x = -\frac{1}{4\pi}\oint \beta_x K_1 \, ds` 的物理来源。代码中不需要显式做任何除法——精确漂移的 :math:`p_z` 表达式自动完成了这件事。

薄透镜与厚透镜的色品对比
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 25 20 55

  * - 效应
    - 薄透镜
    - 厚透镜 DKD-exact
  * - 自然色品（正常四极）
    - 不引入
    - 引入（ :math:`K_{1,\text{eff}} = K_1/(1+\delta)` ）
  * - 耦合色品（斜四极）
    - 不引入
    - 引入（耦合传输含 :math:`\delta` 依赖）
  * - 高阶非线性色散
    - 不引入
    - 引入（ :math:`p_z` 保留完整根号）
  * - 路径长度效应（ :math:`R_{56}` 等）
    - 不引入
    - 引入

物理上，薄透镜零长度无 drift 空间，踢角 :math:`\Delta p_x = -K_{1L}\,x` 不含 :math:`\delta` 。厚透镜中粒子在磁铁内部有漂移路径，不同动量粒子走不同路径、感受不同有效聚焦——这就是色品的来源。斜四极同理，drift 中的 :math:`p_z` 依赖使耦合传输也含 :math:`\delta` 依赖，引入耦合色品。

.. note::

  - 薄透镜模式（ ``length = 0`` ）不存在路径长度效应，因此薄透镜四极铁本身 **不引入自然色品** ——无论正常四极还是斜四极
  - 厚透镜 DKD-exact 模式完整包含自然色品效应，包括高阶非线性色散项
  - 与 Xsuite 的 mat-kick-mat 模型不同（显式除以 :math:`1+\delta` ），PASS 的 DKD-exact 是通过精确 :math:`p_z` 隐式引入的，还包含了 :math:`p_z` 的高阶非线性效应
  - 在 PASS 的 Twiss 线性传输框架中，自然色品通过 ``DQx`` / ``DQy`` 参数（相移中的 :math:`\delta` 项）引入，而非通过元件本身。若在 Twiss 传输中额外插入薄透镜四极铁，不会与 ``DQx`` 重复计数色品——因为薄透镜本身不引入色品。但若插入的四极铁强度较大，显著改变了 lattice 的 tune 和 :math:`\beta` 函数，则原有 Twiss 参数（包括 ``DQx`` ）不再准确，需重新计算


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
    - 元件长度 （必须 :math:`\ge 0` ； :math:`= 0` 时为薄透镜）
  * - ``name``
    - ``name``
    - str
    - -
    - 元件名称
  * - ``k1l``
    - ``k1l``
    - float
    - :math:`\text{m}^{-1}`
    - 正常四极积分强度 :math:`K_{1L}` ，默认 0
  * - ``k1sl``
    - ``k1sl``
    - float
    - :math:`\text{m}^{-1}`
    - 斜四极积分强度 :math:`K_{1sL}` ，默认 0
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

厚透镜正常四极铁
~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "QF1": {
          "S (m)": 10.0,
          "Command": "Quadrupole",
          "Length (m)": 0.5,
          "K1L": 0.2,
          "Num Slices": 5,
          "Integrator": "yoshida4",
          "Aperture Type": "off"
      }
  }

聚焦四极铁（ :math:`K_{1L} > 0` ），长度 0.5 m，5 个切片，4 阶辛积分。

薄透镜四极铁
~~~~~~~~~~~~

.. code-block:: json

  {
      "QF2": {
          "S (m)": 20.0,
          "Command": "Quadrupole",
          "Length (m)": 0.0,
          "K1L": 0.3,
          "Aperture Type": "off"
      }
  }

零长度四极铁，仅施加 :math:`K_{1L}` 薄透镜踢角，无 body 追踪，无色品效应。

散焦四极铁
~~~~~~~~~~

.. code-block:: json

  {
      "QD1": {
          "S (m)": 30.0,
          "Command": "Quadrupole",
          "Length (m)": 0.4,
          "K1L": -0.15,
          "Num Slices": 1,
          "Integrator": "uniform",
          "Aperture Type": "off"
      }
  }

散焦四极铁（ :math:`K_{1L} < 0` ），水平散焦、垂直聚焦。

斜四极铁
~~~~~~~~

.. code-block:: json

  {
      "QS1": {
          "S (m)": 40.0,
          "Command": "Quadrupole",
          "Length (m)": 0.3,
          "K1L": 0.0,
          "K1SL": 0.1,
          "Num Slices": 1,
          "Integrator": "uniform",
          "Aperture Type": "off"
      }
  }

纯斜四极铁（ :math:`K_{1L} = 0` , :math:`K_{1sL} \neq 0` ），产生 :math:`x` - :math:`y` 耦合。

正常 + 斜四极组合
~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "QFS1": {
          "S (m)": 50.0,
          "Command": "Quadrupole",
          "Length (m)": 0.5,
          "K1L": 0.2,
          "K1SL": 0.05,
          "Num Slices": 3,
          "Integrator": "yoshida4",
          "Aperture Type": "circle",
          "Aperture Value": [0.04]
      }
  }

同时含正常和斜四极分量的组合四极铁（模拟安装旋转误差），带圆形孔径检查。

带旋转角的等效表示
~~~~~~~~~~~~~~~~~~

一个 :math:`K_{1L} = 0.2` 的正常四极铁旋转角度 :math:`\theta = 0.01` rad 后，等效为：

.. math::

  K_{1L}' = K_{1L} \cos 2\theta \approx 0.2 \times 0.9998 = 0.19996

.. math::

  K_{1sL}' = K_{1L} \sin 2\theta \approx 0.2 \times 0.02 = 0.004

.. code-block:: json

  {
      "QF_rot": {
          "S (m)": 60.0,
          "Command": "Quadrupole",
          "Length (m)": 0.5,
          "K1L": 0.19996,
          "K1SL": 0.004,
          "Num Slices": 1,
          "Integrator": "uniform"
      }
  }


应用场景
--------

- **线性聚焦** ：在 FODO 结构中交替排列聚焦（F）和散焦（D）四极铁，实现束流的横向约束
- **色品校正** ：利用四极铁的自然色品效应，通过调整六极铁补偿色品
- **耦合校正** ：使用斜四极铁（ ``k1sl`` ）控制 :math:`x` - :math:`y` 耦合，校正安装误差
- **Tune 调整** ：通过调整四极铁强度改变工作点（tune），将束流调至最佳工作区域
- **色散匹配** ：在弯铁后设置四极铁，调节色散函数 :math:`\eta(s)` 的演化
- **束流传输线** ：在注入线和引出线中使用四极铁聚焦束流，控制束流包络


参考文献
--------

- Xsuite Physics Guide, Sec 1.10.3 (精确漂移), Sec 1.10.5 (四极铁)
- Xsuite 源码： ``xtrack/beam_elements/elements_src/quadrupole.h`` , ``track_magnet.h`` , ``track_magnet_kick.h`` , ``track_magnet_drift.h``
- Yoshida, H., "Construction of higher order symplectic integrators", Phys. Lett. A 150 (1990)
- MAD-X 物理手册：四极磁场与线性传输
