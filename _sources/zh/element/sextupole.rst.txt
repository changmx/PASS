六极铁（Sextupole）
====================

本模块介绍 PASS 中的六极铁元件 **Sextupole** ，用于模拟带电粒子在六极磁铁中的运动。六极铁是加速器中最基本的非线性元件，通过二次磁场提供非线性聚焦力，主要用于色品校正和共振驱动。

PASS 中的六极铁支持 **厚元件** （ ``length > 0`` ）和 **薄透镜** （ ``length = 0`` ）两种模式，厚元件采用精确漂移-踢角-漂移（DKD-exact）辛积分方案，支持 uniform（2阶）和 yoshida4（4阶）两种辛积分器。

**代码位置**

- 源文件： ``PASS/commands/element/sextupole.py``
- 类名： ``Sextupole`` （继承自 ``Command`` ）
- 注册名： ``sextupole``
- 核心特征：

  - 支持薄透镜模式（ ``length = 0`` ，仅施加六极踢角）
  - 支持厚透镜模式（ ``length > 0`` ，DKD-exact 辛积分）
  - 支持 uniform（2阶蛙跳）和 yoshida4（4阶 Yoshida 组合）积分器
  - 支持正常六极（ ``k2l`` ）和斜六极（ ``k2sl`` ）及其组合
  - 零场（ ``k2l = k2sl = 0`` ）时自动退化为纯漂移
  - 色品校正、非线性色散等高阶效应通过精确漂移自然引入
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


六极磁场与归一化强度
--------------------

六极磁铁的磁场在横向平面内二次分布。用复数表示：

.. math::

  B_y + i B_x = \frac{1}{2}(B'' + i B''_s)(x + i y)^2

其中 :math:`B''` 为正常六极场二阶导数， :math:`B''_s` 为斜六极场二阶导数。展开后：

.. math::

  B_y = \frac{1}{2} B'' (x^2 - y^2) - B''_s x y

.. math::

  B_x = B'' x y + \frac{1}{2} B''_s (x^2 - y^2)

归一化六极强度定义为：

.. math::

  K_2 = \frac{q_0 B''}{2 P_0}

.. math::

  K_{2s} = \frac{q_0 B''_s}{2 P_0}

积分强度为：

.. math::

  K_{2L} = K_2 \cdot L, \qquad K_{2sL} = K_{2s} \cdot L

其中 :math:`L` 为磁铁长度。PASS 中用户直接指定 :math:`K_{2L}` （ ``k2l`` ）和 :math:`K_{2sL}` （ ``k2sl`` ），厚透镜时内部自动解出 :math:`K_2 = K_{2L} / L` 和 :math:`K_{2s} = K_{2sL} / L` 。


整体追踪流程
------------

根据磁铁长度，六极铁有两种追踪模式：

**薄透镜模式** （ :math:`L = 0` ）

::

  ====== 薄透镜 (length = 0) ======

  单次六极踢角 Kick(K2L, K2sL)
  [位置不变，仅动量跳变]

**厚透镜模式** （ :math:`L > 0` ）

::

  ====== 厚透镜 (length > 0) ======

  切片1 → 切片2 → ... → 切片N
  (每个切片: Drift(ds/2) → Kick(ds) → Drift(ds/2))

  其中 ds = L / N

  若 K2L = 0 且 K2sL = 0：退化为单次精确漂移 Drift(L)

完整映射为：

薄透镜：

.. math::

  \mathcal{M}_{\text{thin}} = \text{Kick}(K_{2L}, K_{2sL})

厚透镜（N 个切片）：

.. math::

  \mathcal{M}_{\text{thick}} = \left[\mathcal{M}_{\text{DKD}}(\Delta s)\right]^N

其中每个切片的 DKD 映射为：

.. math::

  \mathcal{M}_{\text{DKD}}(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

.. note::

  - 薄透镜模式不改变粒子的位置坐标 :math:`(x, y, z)` ，仅施加动量踢角
  - 厚透镜模式的色品效应通过精确漂移中的 :math:`p_z` 表达式自然引入（见色品校正章节）
  - 当 :math:`K_{2L} = 0` 且 :math:`K_{2sL} = 0` 时，厚透镜退化为纯漂移，避免无意义的空踢角循环


物理推导
--------

哈密顿量
~~~~~~~~

在直线坐标系中（六极铁无曲率， :math:`h = 0` ），六极铁的哈密顿量为：

.. math::

  H_{\text{sext}} = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2} + \frac{\chi}{6}\left[K_2(x^3 - 3 x y^2) + K_{2s}(3 x^2 y - y^3)\right]

将其拆分为传播部分（精确漂移 :math:`H_D` ）和踢角部分（ :math:`H_K` ）：

.. math::

  H_D = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  H_K = \frac{\chi}{6}\left[K_2(x^3 - 3 x y^2) + K_{2s}(3 x^2 y - y^3)\right]

其中 :math:`H_D` 是精确漂移哈密顿量（保留 :math:`p_z` 的根号，不做小动量展开）， :math:`H_K` 是六极踢角。这是 **分裂算符法** （split-operator）的标准做法：将哈密顿量拆分为可解析求解的部分，分别施加映射，再组合为辛积分器。

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

  "exact" 的含义：漂移部分保留精确根号 :math:`p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}` ，不做 :math:`p_x \ll 1` 的小动量展开。近似仅在于将传播部分与踢角部分分离（分裂算符法）。该公式与漂移节（Drift）和四极铁（Quadrupole）中的精确漂移完全一致。

六极踢角映射 K
~~~~~~~~~~~~~~~~~~

踢角部分为薄透镜映射（位置不变，仅动量跳变）。由哈密顿方程 :math:`\dot{p}_x = -\partial H / \partial x` ， :math:`\dot{p}_y = -\partial H / \partial y` 得：

.. math::

  \Delta p_x = -\frac{\chi}{2} K_{2L} (x^2 - y^2) + \chi K_{2sL} \, x y

.. math::

  \Delta p_y = \chi K_{2L} \, x y + \frac{\chi}{2} K_{2sL} (x^2 - y^2)

其中 :math:`L_K` 为踢角有效长度。

各项物理含义：

.. list-table::
  :header-rows: 1
  :widths: 30 15 55

  * - 项
    - 来源
    - 物理含义
  * - :math:`-\frac{\chi}{2} K_{2L} (x^2 - y^2)`
    - :math:`\frac{\chi K_2}{6} x^3`
    - 水平非线性聚焦（正比于 :math:`x^2` ）
  * - :math:`+\chi K_{2L} \, xy`
    - :math:`-\frac{\chi K_2}{2} x y^2`
    - 水平-垂直耦合踢角
  * - :math:`+\chi K_{2sL} \, xy`
    - :math:`\frac{\chi K_{2s}}{2} x^2 y`
    - 斜六极水平耦合踢角
  * - :math:`+\frac{\chi}{2} K_{2sL} (x^2 - y^2)`
    - :math:`-\frac{\chi K_{2s}}{6} y^3`
    - 斜六极垂直非线性聚焦

对于薄透镜模式， :math:`L_K = 1` ，直接使用积分强度 :math:`K_{2L}` 和 :math:`K_{2sL}` 。对于 DKD 模式， :math:`L_K = \Delta s` ，使用 :math:`K_2 \Delta s` 和 :math:`K_{2s} \Delta s` 。

.. note::

  正常六极铁（ :math:`K_2 > 0` ）在水平方向对正偏移粒子提供恢复力（正比于 :math:`x^2` ），垂直方向则相反。六极铁的聚焦力与位置平方成正比，是非线性元件——远离轴线的粒子受到更强的偏转。

  斜六极铁（ :math:`K_{2s} \neq 0` ）将六极作用旋转 :math:`\pi / 6` ，产生不同的 :math:`x` - :math:`y` 耦合模式。实际中常用于模拟安装旋转误差或驱动特定共振。

  与四极铁的对比：四极踢角线性依赖于 :math:`x` （ :math:`\Delta p_x \propto x` ），六极踢角二次依赖于 :math:`x` （ :math:`\Delta p_x \propto x^2` ）。这意味着六极铁不改变参考轨道上的粒子（ :math:`x = y = 0` 时踢角为零），但对偏离轴线的粒子产生非线性偏转。


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


色品校正
--------

色品（chromaticity）描述了粒子 tune 对动量偏差 :math:`\delta` 的依赖。六极铁是色品校正的核心元件。

物理机制
~~~~~~~~

粒子在六极铁处的横向位置包含色散部分：

.. math::

  x = x_\beta + \eta_x \, \delta

其中 :math:`x_\beta` 为 betatron 振荡部分， :math:`\eta_x` 为色散函数。代入六极踢角：

.. math::

  \Delta p_x = -\frac{\chi}{2} K_{2L} (x_\beta + \eta_x \delta)^2

展开后：

.. math::

  \Delta p_x = -\frac{\chi}{2} K_{2L} \, x_\beta^2 \;-\; \chi K_{2L} \, \eta_x \, \delta \, x_\beta \;-\; \frac{\chi}{2} K_{2L} \, \eta_x^2 \, \delta^2

第二项 :math:`-\chi K_{2L} \eta_x \delta \, x_\beta` 是一个等效四极踢角（线性依赖于 :math:`x_\beta` ，系数正比于 :math:`\delta` ），它改变了 tune 对 :math:`\delta` 的依赖，从而实现色品校正。在有色散的六极铁处，等效四极强度为：

.. math::

  K_{1,\text{eff}} = -K_2 \, \eta_x

对应的色品贡献为：

.. math::

  \Delta Q'_x = \frac{1}{4\pi} \oint \beta_x K_{1,\text{eff}} \, ds = -\frac{1}{4\pi} \oint \beta_x K_2 \, \eta_x \, ds

.. note::

  - 六极铁仅在有色散的位置才能校正色品（ :math:`\eta_x \neq 0` ）
  - 色品校正在踢角中自动产生——踢角作用于真实坐标 :math:`x` （包含色散），不做任何展开
  - 即使薄透镜（无 drift）也有色品校正效应
  - 第三项 :math:`-\frac{\chi}{2} K_{2L} \eta_x^2 \delta^2` 是二阶色散驱动项，也自然包含
  - 在无色散位置（ :math:`\eta_x = 0` ），六极铁不校正一阶色品，但仍保留非线性效应（三阶共振驱动、非线性耦合、动态孔径限制等）


在 Twiss 线性传输中使用六极铁
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PASS 的 Twiss 传输（ ``twiss.py`` ）在 :math:`(x, p_x)` 归一化动量坐标下工作，色散处理为"减去→线性传输→加回"，自然色品通过 ``DQx`` / ``DQy`` 参数（相移中的 :math:`\delta` 项）引入。在该框架中插入六极铁 kick 时需注意以下事项。

**坐标一致性**

Twiss 传输到达六极铁位置时，粒子的 :math:`x` 已包含色散（ :math:`x = x_\beta + \eta_x \delta` ），六极 kick 直接作用于该真实坐标，色品校正项 :math:`-K_{2L}\eta_x\delta\cdot x_\beta` 自动出现。 **踢角中不应除以** :math:`1+\delta` ——那是 :math:`(x, x')` 角度坐标体系的写法，与 PASS 的 :math:`(x, p_x)` 体系不兼容，混用会导致色品重复计数。

**避免色品重复计数**

.. list-table::
  :header-rows: 1
  :widths: 30 70

  * - 场景
    - 正确做法
  * - ``DQx`` 含总色品（含六极铁贡献）
    - 不再单独施加六极 kick，否则一阶色品被双重计数
  * - ``DQx`` 仅含自然色品（不含六极铁）
    - 施加六极 kick 补充色品校正及非线性效应，不冲突
  * - ``DQx`` 含总色品，但仍需模拟非线性效应
    - 将 ``DQx`` 减去六极铁色品贡献（ :math:`\Delta Q'_x = -\frac{1}{4\pi}\oint \beta_x K_2 \eta_x \, ds` ），再施加完整六极 kick

**薄透镜与厚透镜的差异**

.. list-table::
  :header-rows: 1
  :widths: 25 20 55

  * - 效应
    - 薄透镜
    - 厚透镜 DKD-exact
  * - 色品校正（通过色散位置）
    - 有
    - 有
  * - 元件内部 drift 色散
    - 无
    - 有
  * - 厚透镜分布效应
    - 无
    - 有
  * - 路径长度效应（ :math:`R_{56}` 等）
    - 无
    - 有

薄透镜缺少的效应是"磁铁内部漂移"带来的——零长度磁铁物理上就没有内部漂移，这是正确的物理近似，不是遗漏。若需要这些效应，使用厚透镜模式。

.. note::

  - 在逐元件追踪（element-by-element tracking）模式中，不存在 ``DQx`` 重复计数问题——所有效应由 DKD-exact 物理模拟自然产生
  - Twiss 线性传输是一阶模型，六极 kick 中除 :math:`1+\delta` 会引入与模型精度不匹配的二阶非线性色散效应，应避免
  - 若六极铁强度较大或需要精确的非线性效应模拟，建议切换到完整的逐元件 DKD-exact 追踪，而非在 Twiss 线性框架中局部引入非线性 kick


自然包含的高阶效应
------------------

DKD-exact 方案中，理想六极磁铁的所有非线性效应天然包含，无需任何额外处理：

.. list-table::
  :header-rows: 1
  :widths: 30 70

  * - 效应
    - 来源
  * - 色品校正
    - kick 作用于含色散的真实坐标 :math:`x` ，展开后自动出现等效四极项
  * - 自然色品
    - drift 中精确 :math:`p_z` 使等效聚焦强度含 :math:`1/(1+\delta)` 依赖
  * - 高阶色散
    - drift 中 :math:`p_z` 保留完整根号，色散演化含所有阶次的 :math:`\delta` 依赖
  * - 路径长度效应（ :math:`R_{56}` 等）
    - drift 中 :math:`\zeta` 更新包含完整的 :math:`R_{56}` , :math:`T_{566}` 等高阶项
  * - 厚透镜分布效应
    - DKD 多切片中 drift 改变 :math:`x` ，后续 kick 感受更新后的坐标
  * - :math:`x` - :math:`y` 耦合
    - kick 中 :math:`xy` 交叉项

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
  * - ``k2l``
    - ``k2l``
    - float
    - :math:`\text{m}^{-2}`
    - 正常六极积分强度 :math:`K_{2L}` ，默认 0
  * - ``k2sl``
    - ``k2sl``
    - float
    - :math:`\text{m}^{-2}`
    - 斜六极积分强度 :math:`K_{2sL}` ，默认 0
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

厚透镜正常六极铁
~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SF1": {
          "S (m)": 10.0,
          "Command": "Sextupole",
          "Length (m)": 0.5,
          "K2L": 5.0,
          "Num Slices": 5,
          "Integrator": "yoshida4",
          "Aperture Type": "off"
      }
  }

正常六极铁（ :math:`K_{2L} > 0` ），长度 0.5 m，5 个切片，4 阶辛积分。用于色品校正。

薄透镜六极铁
~~~~~~~~~~~~

.. code-block:: json

  {
      "SF2": {
          "S (m)": 20.0,
          "Command": "Sextupole",
          "Length (m)": 0.0,
          "K2L": 10.0,
          "Aperture Type": "off"
      }
  }

零长度六极铁，仅施加 :math:`K_{2L}` 薄透镜踢角，无 body 追踪。

负六极铁
~~~~~~~~

.. code-block:: json

  {
      "SD1": {
          "S (m)": 30.0,
          "Command": "Sextupole",
          "Length (m)": 0.4,
          "K2L": -5.0,
          "Num Slices": 1,
          "Integrator": "uniform",
          "Aperture Type": "off"
      }
  }

负六极铁（ :math:`K_{2L} < 0` ），提供与正六极铁相反的色品校正方向。

斜六极铁
~~~~~~~~

.. code-block:: json

  {
      "SS1": {
          "S (m)": 40.0,
          "Command": "Sextupole",
          "Length (m)": 0.3,
          "K2L": 0.0,
          "K2SL": 3.0,
          "Num Slices": 1,
          "Integrator": "uniform",
          "Aperture Type": "off"
      }
  }

纯斜六极铁（ :math:`K_{2L} = 0` , :math:`K_{2sL} \neq 0` ），产生与正常六极铁旋转 :math:`\pi / 6` 的耦合效应。

正常 + 斜六极组合
~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SFS1": {
          "S (m)": 50.0,
          "Command": "Sextupole",
          "Length (m)": 0.5,
          "K2L": 5.0,
          "K2SL": 1.0,
          "Num Slices": 3,
          "Integrator": "yoshida4",
          "Aperture Type": "circle",
          "Aperture Value": [0.04]
      }
  }

同时含正常和斜六极分量的组合六极铁（模拟安装旋转误差），带圆形孔径检查。


应用场景
--------

- **色品校正** ：在有色散的位置放置六极铁，补偿四极铁自然色品，使粒子 tune 对动量偏差不敏感
- **共振驱动** ：驱动三阶共振（ :math:`3Q_x` , :math:`2Q_x \pm Q_y` 等）用于共振引出或束流刮削
- **动态孔径控制** ：六极铁的非线性场限制稳定相空间区域，影响束流寿命
- **非线性耦合校正** ：使用斜六极铁（ ``k2sl`` ）控制高阶 :math:`x` - :math:`y` 耦合
- **Harmonic sextupole** ：在特定相位放置六极铁驱动或抑制特定共振项
- **LHC 色品方案** ：在弧区分布六极铁家族（SF/SD），实现宽能范围内的色品控制


参考文献
--------

- Xsuite Physics Guide, Sec 1.10.3 (精确漂移), Sec 1.10.5 (六极铁)
- Xsuite 源码： ``xtrack/beam_elements/elements_src/sextupole.h`` , ``track_magnet.h`` , ``track_magnet_kick.h`` , ``track_magnet_drift.h``
- Yoshida, H., "Construction of higher order symplectic integrators", Phys. Lett. A 150 (1990)
- MAD-X 物理手册：六极磁场与非线性传输
- Wiedemann, H., "Particle Accelerator Physics", Ch. 4 (非线性束流动力学)
