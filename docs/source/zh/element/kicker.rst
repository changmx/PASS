快脉冲偏转磁铁（Kicker）
========================

本模块介绍 PASS 中的快脉冲偏转磁铁元件 **Kicker** ，用于模拟带电粒子在脉冲偶极磁铁中的运动。快脉冲偏转磁铁是一种快速作用的横向偏转元件，通过施加角偏转改变束流方向，广泛应用于束流注入、引出、轨道校正和快速束流调节等场景。

PASS 中的快脉冲偏转磁铁支持 **厚元件** （ ``length > 0`` ）和 **薄透镜** （ ``length = 0`` ）两种模式。厚元件采用精确漂移-动量更新-漂移（DKD-exact）辛积分方案，支持 uniform（2阶）和 yoshida4（4阶）两种辛积分器。物理上，快脉冲偏转磁铁等价于 order-0 多极铁（偶极铁），动量更新公式为 :math:`\Delta p_x = \text{hkick}` ， :math:`\Delta p_y = \text{vkick}` 。

- 核心特征：

  - 支持薄透镜模式（ ``length = 0`` ，仅施加偶极动量更新）
  - 支持厚透镜模式（ ``length > 0`` ，DKD-exact 辛积分）
  - 支持 uniform（2阶蛙跳）和 yoshida4（4阶 Yoshida 组合）积分器
  - 水平动量更新（ ``hkick`` ）与垂直动量更新（ ``vkick`` ）独立设置
  - 支持单向动量更新（仅 ``hkick`` 或仅 ``vkick`` ）和双向动量更新（两者同时非零）
  - 掩码方式施加动量更新，无逐粒子分支
  - 零动量更新时薄透镜退化为标记器，厚透镜退化为纯漂移
  - 支持孔径检查

以下接口中的字段用于 ``PASS.para.schema.elements.KickerItem`` 配置类；
元件名称由 ``Sequence.add(name, item)`` 的序列键给出，不是配置类字段。

接口参数
--------

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python 配置字段
     - JSON 键
     - 类型
     - 单位
     - 默认值
     - 说明
   * - ``s``
     - ``S (m)``
     - ``float``
     - m
     - ``必填``
     - 元件出口或零长度作用点的纵向位置。
   * - ``length``
     - ``Length (m)``
     - ``float``
     - m
     - ``0.0``
     - 磁铁长度， :math:`= 0` 为薄透镜， :math:`> 0` 为厚透镜
   * - ``hkick``
     - ``HKICK``
     - ``float``
     - 1
     - ``0.0``
     - 水平归一化动量增量；仅在小角度近似下对应弧度。
   * - ``vkick``
     - ``VKICK``
     - ``float``
     - 1
     - ``0.0``
     - 垂直归一化动量增量；仅在小角度近似下对应弧度。
   * - ``num_slices``
     - ``Num slices``
     - ``int``
     - 1
     - ``1``
     - 厚透镜切片数
   * - ``integrator``
     - ``Integrator``
     - ``str``
     - —
     - ``'adaptive'``
     - 积分器，可选： ``adaptive`` （默认 ``uniform`` ）、 ``uniform`` 、 ``yoshida4``
   * - ``aperture_type``
     - ``Aperture type``
     - ``str``
     - —
     - ``'off'``
     - 孔径类型
   * - ``aperture_value``
     - ``Aperture value``
     - ``list``
     - m / rad
     - ``[]``
     - 孔径参数值


.. note::

  ``hkick`` 和 ``vkick`` 均为可选参数，默认值为 0。两者独立设置：

  - 仅 ``hkick`` 非零：单向水平快脉冲偏转磁铁
  - 仅 ``vkick`` 非零：单向垂直快脉冲偏转磁铁
  - 两者均非零：双向快脉冲偏转磁铁
  - 两者均为零：薄透镜退化为标记器，厚透镜退化为纯漂移

  ``Command`` 字段应设为 ``kicker`` 。

使用示例
--------

薄透镜水平快脉冲偏转磁铁
~~~~~~~~~~~~~~~~~~~~~~~~

以下示例在 :math:`s = 10.0` m 处放置一个薄透镜水平快脉冲偏转磁铁，归一化动量增量 :math:`1.5 \times 10^{-3}`：

.. code-block:: json

   {
       "HK1": {
           "S (m)": 10.0,
           "Command": "kicker",
           "HKICK": 0.0015,
           "VKICK": 0.0
       }
   }

仅施加水平动量更新，垂直方向不受影响。适用于水平轨道校正或水平注入。

薄透镜垂直快脉冲偏转磁铁
~~~~~~~~~~~~~~~~~~~~~~~~

以下示例在 :math:`s = 20.0` m 处放置一个薄透镜垂直快脉冲偏转磁铁，归一化动量增量 :math:`-2.3 \times 10^{-3}`：

.. code-block:: json

   {
       "VK1": {
           "S (m)": 20.0,
           "Command": "kicker",
           "HKICK": 0.0,
           "VKICK": -0.0023
       }
   }

仅施加垂直动量更新，水平方向不受影响。负号表示动量更新方向向下。

厚透镜双向快脉冲偏转磁铁
~~~~~~~~~~~~~~~~~~~~~~~~

以下示例在 :math:`s = 15.0` m 处放置一个厚透镜双向快脉冲偏转磁铁，4 个切片，uniform 积分器：

.. code-block:: json

   {
       "BK1": {
           "S (m)": 15.0,
           "Command": "kicker",
           "Length (m)": 0.3,
           "HKICK": 0.003,
           "VKICK": -0.0015,
           "Num slices": 4,
           "Integrator": "uniform",
           "Aperture type": "circle",
           "Aperture value": [0.05]
       }
   }

水平和垂直动量更新同时施加，长度 0.3 m，4 个切片的 DKD-exact 辛积分，并配置圆形孔径检查（半径 0.05 m）。

厚透镜 yoshida4 积分器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下示例使用 4 阶 Yoshida 积分器，适用于对精度要求较高的场景：

.. code-block:: json

   {
       "BK2": {
           "S (m)": 25.0,
           "Command": "kicker",
           "Length (m)": 0.5,
           "HKICK": 0.002,
           "VKICK": 0.0,
           "Num slices": 2,
           "Integrator": "yoshida4"
       }
   }

2 个切片，每个切片执行 3 次 DKD 步（Yoshida 组合），截断误差 :math:`O(\Delta s^4)` 。

零动量更新快脉冲偏转磁铁
~~~~~~~~~~~~~~~~~~~~~~~~

以下示例在 :math:`s = 30.0` m 处放置一个零动量更新快脉冲偏转磁铁：

.. code-block:: json

   {
       "K0": {
           "S (m)": 30.0,
           "Command": "kicker",
           "HKICK": 0.0,
           "VKICK": 0.0
       }
   }

动量更新为零且长度为零时，快脉冲偏转磁铁退化为标记器，不改变任何粒子坐标。可用于预留元件位置并在后续配置中更改强度。磁铁 ramping 字段为预留接口，本元件不执行时变波形。

坐标约定
--------

采用 :ref:`zh-longitudinal-reference` 的连续纵向坐标。
:math:`T_b` 是理想参考粒子在当前位置的实际通过时刻，不是束团质心时刻；
局部映射中的 :math:`\beta_0` 和 :math:`P_0` 表示当前束团参考量。
:math:`p_x=P_x/P_0` 是归一化动量，不是轨迹斜率。

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
    - 纵向坐标， :math:`z=\zeta=\beta_b c(T_b-t_i)`
  * - ``dp``
    - :math:`\delta`
    - 相对动量偏差， :math:`\delta = P / P_0 - 1`

其中 :math:`P_0` 为参考粒子动量， :math:`\beta_0 = v_0 / c` 为参考粒子归一化速度， :math:`s` 为沿参考轨道的弧长， :math:`t` 为时间。

物理推导
--------

快脉冲偏转磁铁的物理本质
~~~~~~~~~~~~~~~~~~~~~~~~

快脉冲偏转磁铁在物理上是一个脉冲偶极磁铁（pulsed dipole magnet），在短暂的时间窗口内产生均匀磁场 :math:`B_0` ，对穿过其中的粒子施加横向偏转。在小角度近似下，长度为 :math:`L` 的均匀偶极场对参考粒子的偏转角大小为：

.. math::

  |\theta| \simeq \frac{|q B_0|L}{P_0} = \frac{|B_0|L}{|B\rho|}

其中 :math:`B\rho=P_0/q` 为磁刚度。``hkick``、``vkick`` 给出下述归一化横向动量增量；其符号直接指定水平、垂直动量变化的方向。它们与正多极零阶系数之间的换算还须考虑符号约定。

偶极动量更新
~~~~~~~~~~~~

快脉冲偏转磁铁的动量更新公式为：

.. math::

  \Delta p_x = \text{hkick}

.. math::

  \Delta p_y = \text{vkick}

对于薄透镜模式， ``hkick`` 和 ``vkick`` 直接作为积分强度施加。对于厚透镜 DKD 模式，每个切片的动量更新为 :math:`\text{hkick}_{\text{eff}} = \text{hk} \cdot \Delta s` ，其中 :math:`\text{hk} = \text{hkick} / L` 为单位长度强度。

.. note::

  ``hkick`` 和 ``vkick`` 直接给出 :math:`\Delta p_x` 和 :math:`\Delta p_y`，是无量纲归一化动量增量。
  只有在小角度近似下才可解释为弧度表示的偏转角。

整体跟踪流程
------------

根据磁铁长度，快脉冲偏转磁铁有两种跟踪模式：

**薄透镜模式** （ :math:`L = 0` ）

::

  ====== 薄透镜 (length = 0) ======

  单次偶极动量更新 Kick(hkick, vkick)
  [位置不变，仅动量跳变]

**厚透镜模式** （ :math:`L > 0` ）

::

  ====== 厚透镜 (length > 0) ======

  切片1 → 切片2 → ... → 切片N
  (每个切片: Drift(ds/2) → Kick(ds) → Drift(ds/2))

  其中 ds = L / N
  hkick_eff = hk * ds,  vkick_eff = vk * ds

  若 hkick=0 且 vkick=0：退化为单次精确漂移 Drift(L)

完整映射为：

薄透镜：

.. math::

  \mathcal{M}_{\text{thin}} = \text{Kick}(\text{hkick}, \text{vkick})

厚透镜（N 个切片）：

.. math::

  \mathcal{M}_{\text{thick}} = \left[\mathcal{M}_{\text{DKD}}(\Delta s)\right]^N

其中每个切片的 DKD 映射为：

.. math::

  \mathcal{M}_{\text{DKD}}(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

.. note::

  - 薄透镜模式不改变粒子的位置坐标 :math:`(x, y, z)` ，仅施加动量动量更新
  - 厚透镜模式的色品等效应通过精确漂移中的 :math:`p_z` 表达式自然引入
  - 当 ``hkick`` 和 ``vkick`` 均为零时，厚透镜退化为纯漂移，避免无意义的空动量更新循环

精确漂移映射
------------

漂移部分采用精确漂移（Table 1.1, map D, Eq. 1.86-1.88），与四极铁/六极铁/八极铁/多极铁完全相同：

.. math::

  x \mathrel{+}= \frac{p_x}{p_z} L

.. math::

  y \mathrel{+}= \frac{p_y}{p_z} L

.. math::

  z \mathrel{+}= L \left(1 - \frac{\beta_0}{\beta} \cdot \frac{1+\delta}{p_z}\right)

其中：

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  \beta = \frac{(1+\delta) \beta_0 \gamma_0}{\sqrt{1 + \left[(1+\delta) \beta_0 \gamma_0\right]^2}}

精确漂移保留了 :math:`p_z` 的完整非线性，自然引入色品、高阶色散和路径长度效应。

辛积分器
--------

Uniform（2阶蛙跳）
~~~~~~~~~~~~~~~~~~~~~~~~~~

每个切片执行 Drift-Kick-Drift：

.. math::

  \mathcal{M}_{\text{DKD}}(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

这是2阶辛积分器，截断误差 :math:`O(\Delta s^2)` 。

Yoshida4（4阶组合）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

将3个 DKD 步组合为4阶辛积分器：

.. math::

  \mathcal{M}_{\text{Y4}}(\Delta s) = \mathcal{M}_{\text{DKD}}(z_1 \Delta s) \circ \mathcal{M}_{\text{DKD}}(z_0 \Delta s) \circ \mathcal{M}_{\text{DKD}}(z_1 \Delta s)

其中 Yoshida 系数为：

.. math::

  z_1 = \frac{1}{2 - 2^{1/3}} \approx 1.3512

.. math::

  z_0 = 1 - 2 z_1 \approx -1.7024

截断误差 :math:`O(\Delta s^4)` 。


绝对正、斜多极场误差与静态 DX/DY/DPSI 准直误差使用公共 :ref:`zh-error` 接口。
准直只移动磁场，孔径和 SC 边界保持在设计坐标系。

应用场景
--------

- **束流注入** ：在注入段放置快脉冲偏转磁铁，将注入束流偏转至与主环闭合轨道匹配的方向，实现束流注入
- **束流引出** ：在引出点放置快脉冲偏转磁铁，快速将束流偏转至引出通道，实现快引出
- **轨道校正** ：在束线关键位置放置快脉冲偏转磁铁，校正轨道偏差或实现局部凸轨（ bump ）

本 Kicker 模型施加固定配置强度。给定注入波形见 :doc:`bump`，横向激励见 :doc:`exciter`；本元件不包含拾取器反馈控制器。

参考文献
--------

- MAD-X User's Guide, "Kicker" section (hkick / vkick 定义)
- Xsuite 源码： ``xtrack/mad_loader.py`` （ ``convert_kicker`` , ``_make_kicker_multipole`` ）
- Yoshida, H., "Construction of higher order symplectic integrators", Phys. Lett. A 150 (1990)
- Wiedemann, H., "Particle Accelerator Physics", Ch. 4 (偶极磁铁与偏转)

元件内部空间电荷
----------------

正长度元件可设置 ``space_charge`` （JSON ``Space charge``）为
``ElementSpaceCharge`` 对象。``Num slices`` 控制外场传输，
``Space charge.Num kicks`` 控制 SC 积分。调度规则、共享资源、
支持的后端和示例见 :ref:`zh-internal-space-charge`。
