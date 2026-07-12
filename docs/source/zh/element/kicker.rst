冲击器（Kicker）
================

本模块介绍 PASS 中的冲击器元件 **Kicker** ，用于模拟带电粒子在脉冲偶极磁铁中的运动。冲击器是一种快速作用的横向偏转元件，通过施加瞬时角偏转改变束流方向，广泛应用于束流注入、引出、轨道校正和快速束流调节等场景。

PASS 中的冲击器建模为 **薄透镜** （ ``length = 0`` ），即纯动量平移映射，严格辛。这一处理与 MAD-X 默认薄透镜约定和 Xsuite 将 kicker 转为 order-0 Multipole 的方案一致。物理上，典型冲击器的踢角量级为 :math:`10^{-3}` rad，远小于 betatron 波长尺度，薄透镜近似的误差远低于机器精度，无需厚透镜 DKD 积分。

**代码位置**

- 源文件： ``PASS/commands/element/kicker.py``
- 类名： ``Kicker`` （继承自 ``Command`` ）
- 注册名： ``kicker``
- 核心特征：

  - 薄透镜模型（ ``length = 0`` ，不可配置），纯动量平移，严格辛
  - 水平踢角（ ``hkick`` ）与垂直踢角（ ``vkick`` ）独立设置
  - 支持单向踢角（仅 ``hkick`` 或仅 ``vkick`` ）和双向踢角（两者同时非零）
  - 掩码方式施加踢角，无逐粒子分支
  - 零踢角时退化为标记器（仅孔径检查）
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

其中 :math:`P_0` 为参考粒子动量， :math:`\beta_0 = v_0 / c` 为参考粒子归一化速度。


物理推导
--------

冲击器的物理本质
~~~~~~~~~~~~~~~~

冲击器在物理上是一个脉冲偶极磁铁（pulsed dipole magnet），在短暂的时间窗口内产生均匀磁场 :math:`B_0` ，对穿过其中的粒子施加横向偏转。长度为 :math:`L` 的偶极磁铁对参考粒子的偏转角为：

.. math::

  \theta = \frac{q B_0 L}{p_0} = \frac{B_0 L}{B\rho}

其中 :math:`B\rho = p_0 / q` 为磁刚度。因此冲击器在物理上等价于一个积分强度为 :math:`K_{0L} = \theta` 的 order-0 多极铁。

薄透镜近似
~~~~~~~~~~

冲击器建模为薄透镜（ :math:`\delta` -kick），即粒子位置不变，仅动量发生瞬时跳变：

.. math::

  x \leftarrow x

.. math::

  p_x \leftarrow p_x + \Delta p_x

.. math::

  y \leftarrow y

.. math::

  p_y \leftarrow p_y + \Delta p_y

其中踢角量为：

.. math::

  \Delta p_x = \text{hkick}

.. math::

  \Delta p_y = \text{vkick}

这里 ``hkick`` 和 ``vkick`` 是积分偶极强度（单位：弧度），等价于 MAD-X 的 ``hkick`` / ``vkick`` 参数，也等价于 Multipole 的 ``knl=[hkick]`` 、 ``ksl=[vkick]`` 。

薄透镜近似的误差分析
~~~~~~~~~~~~~~~~~~~~~~

薄透镜近似引入两个误差源：

1. **路径长度效应** ：粒子穿过物理长度 :math:`L` 但纵向坐标 :math:`z` 不变，误差量级 :math:`O(\text{hkick} \cdot L)` 。

2. **色差效应** ：薄 kick 对所有动量粒子施加相同的 :math:`\Delta p_x` ，但物理上动量为 :math:`p_0(1+\delta)` 的粒子实际偏转为 :math:`\Delta p_x^{\text{exact}} = \text{hkick} / (1+\delta)` ，薄透镜丢掉了 :math:`-\text{hkick} \cdot \delta` 这一项，量级 :math:`O(\text{hkick} \cdot \delta)` 。

对典型冲击器（ :math:`\text{hkick} \sim 10^{-3}` rad， :math:`\delta \sim 10^{-3}` ），总误差 :math:`\sim 10^{-6}` rad，远低于机器精度，薄透镜近似完全充分。

辛性分析
~~~~~~~~

薄透镜踢角是一个纯正则动量平移映射：

.. math::

  (x, p_x, y, p_y, z, \delta) \mapsto (x, p_x + \text{hkick}, y, p_y + \text{vkick}, z, \delta)

该映射的 Jacobi 矩阵为单位矩阵，行列式为 1，严格辛。无需 DKD 积分器或任何近似。

.. note::

  使用厚透镜 DKD 对冲击器 **没有意义** 。DKD 的 kick 步骤仍然是 :math:`\delta` 无关的薄 kick ，无法修正色差；同时 DKD 仅是偶极的 2 阶近似，精度低于精确偶极映射。若需要物理精确的厚偶极，应使用专用 Dipole 元件。


接口参数
--------

.. list-table::
  :header-rows: 1
  :widths: 20 20 10 10 40

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
  * - ``length``
    - - （固定 0.0）
    - float
    - 0.0
    - 元件长度（恒为 0，不可配置）
  * - ``hkick``
    - ``hkick``
    - float
    - 0.0
    - 水平踢角（弧度）， :math:`\Delta p_x = \text{hkick}`
  * - ``vkick``
    - ``vkick``
    - float
    - 0.0
    - 垂直踢角（弧度）， :math:`\Delta p_y = \text{vkick}`
  * - ``aperture_type``
    - ``aperture type``
    - str
    - ``off``
    - 孔径类型
  * - ``aperture_value``
    - ``aperture value``
    - list
    - ``[]``
    - 孔径参数值

.. note::

  ``hkick`` 和 ``vkick`` 均为可选参数，默认值为 0。两者独立设置：

  - 仅 ``hkick`` 非零：单向水平冲击器
  - 仅 ``vkick`` 非零：单向垂直冲击器
  - 两者均非零：双向冲击器
  - 两者均为零：退化为标记器

  ``Command`` 字段应设为 ``kicker`` 。无需提供 ``Length (m)`` 字段。


使用示例
--------

水平冲击器
~~~~~~~~~~

以下示例在 :math:`s = 10.0` m 处放置一个水平冲击器，踢角 :math:`1.5 \times 10^{-3}` rad：

.. code-block:: json

  {
      "HK1": {
          "S (m)": 10.0,
          "Command": "kicker",
          "hkick": 0.0015,
          "vkick": 0.0
      }
  }

仅施加水平踢角，垂直方向不受影响。适用于水平轨道校正或水平注入。

垂直冲击器
~~~~~~~~~~

以下示例在 :math:`s = 20.0` m 处放置一个垂直冲击器，踢角 :math:`-2.3 \times 10^{-3}` rad：

.. code-block:: json

  {
      "VK1": {
          "S (m)": 20.0,
          "Command": "kicker",
          "hkick": 0.0,
          "vkick": -0.0023
      }
  }

仅施加垂直踢角，水平方向不受影响。负号表示踢角方向向下。

双向冲击器
~~~~~~~~~~

以下示例在 :math:`s = 15.0` m 处放置一个双向冲击器，同时施加水平和垂直踢角：

.. code-block:: json

  {
      "BK1": {
          "S (m)": 15.0,
          "Command": "kicker",
          "hkick": 0.003,
          "vkick": -0.0015,
          "Aperture Type": "circle",
          "Aperture Value": [0.05]
      }
  }

水平和垂直踢角同时施加，并配置圆形孔径检查（半径 0.05 m）。适用于需要同时在两个方向偏转束流的场景。

零踢角冲击器
~~~~~~~~~~~~

以下示例在 :math:`s = 30.0` m 处放置一个零踢角冲击器（退化为标记器）：

.. code-block:: json

  {
      "K0": {
          "S (m)": 30.0,
          "Command": "kicker",
          "hkick": 0.0,
          "vkick": 0.0
      }
  }

踢角为零时，冲击器不改变任何粒子坐标，行为等同于标记器。可用于预留冲击器位置，后续通过 ramping 表动态启用。


应用场景
--------

- **束流注入** ：在注入段放置冲击器，将注入束流偏转至与主环闭合轨道匹配的方向，实现束流注入
- **束流引出** ：在引出点放置冲击器，快速将束流偏转至引出通道，实现快引出或慢引出
- **轨道校正** ：在束线关键位置放置冲击器，校正轨道偏差或实现局部凸轨（ bump ）
- **快速束流调节** ：通过时序控制冲击器的踢角，实现束流方向的快速切换或扫描
- **反馈系统** ：与 Pickup（拾取器）配合，构成逐束团横向反馈系统，抑制束流不稳定性


参考文献
--------

- MAD-X User's Guide, "Kicker" section (hkick / vkick 定义)
- Xsuite 源码： ``xtrack/mad_loader.py`` （ ``convert_kicker`` , ``_make_kicker_multipole`` ）
- Wiedemann, H., "Particle Accelerator Physics", Ch. 4 (偶极磁铁与偏转)
