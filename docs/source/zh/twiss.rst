Twiss 参数输运（Twiss）
==================================

``Twiss`` 根据入口和出口的光学函数及相位差计算无耦合横向输运，并提供可选的纵向线性映射和色品相位修正。坐标、参考动量及物理到达时间的约定见 :ref:`zh-longitudinal-reference`。

输入光学已包含设计聚焦；额外插入的磁铁产生附加作用，应避免重复计入。横向 beta 必须为正。``Mu`` 以圈为单位（1 对应 2π），应保留两点之间所需的完整相位差。``drift`` 模式需要束团的跃迁 Lorentz 因子；``matrix`` 模式需要正的束长和相对动量分散。

使用示例
--------

以下 JSON 片段展示了一个 ``Sequence`` 中的 ``Twiss`` 条目：

.. code-block:: json

   "Twiss1": {
       "S (m)": 10.0,
       "Command": "Twiss",
       "S previous (m)": 5.0,
       "Alpha x": 0.5,
       "Alpha y": -0.3,
       "Alpha x previous": 0.4,
       "Alpha y previous": -0.2,
       "Beta x (m)": 3.5,
       "Beta y (m)": 2.8,
       "Beta x previous (m)": 3.0,
       "Beta y previous (m)": 2.5,
       "Mu x": 0.123,
       "Mu y": 0.456,
       "Mu x previous": 0.1,
       "Mu y previous": 0.4,
       "Dx (m)": 0.5,
       "Dx previous (m)": 0.3,
       "Dpx": 0.01,
       "Dpx previous": 0.005,
       "DQx": 2.0,
       "DQy": 2.0,
       "Longitudinal transfer": "drift"
   }

接口参数
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python 属性
     - JSON 键
     - 单位
     - 默认值
     - 说明
   * - ``s``
     - ``S (m)``
     - m
     - 必填
     - 出口位置。
   * - ``command``
     - ``Command``
     - —
     - ``Twiss``
     - 命令标识。
   * - ``s_previous``
     - ``S previous (m)``
     - m
     - 必填
     - 入口位置；与出口之差为传输长度。
   * - ``alpha_x``
     - ``Alpha x``
     - —
     - 必填
     - 出口水平 Twiss α。
   * - ``alpha_y``
     - ``Alpha y``
     - —
     - 必填
     - 出口垂直 Twiss α。
   * - ``beta_x``
     - ``Beta x (m)``
     - m
     - 必填
     - 出口水平 Twiss β，必须为正。
   * - ``beta_y``
     - ``Beta y (m)``
     - m
     - 必填
     - 出口垂直 Twiss β，必须为正。
   * - ``mu_x``
     - ``Mu x``
     - 圈
     - 必填
     - 出口水平相位。
   * - ``mu_y``
     - ``Mu y``
     - 圈
     - 必填
     - 出口垂直相位。
   * - ``mu_z``
     - ``Mu z``
     - 圈
     - ``0.0``
     - 出口纵向相位。
   * - ``dx``
     - ``Dx (m)``
     - m
     - 必填
     - 出口水平位置色散。
   * - ``dpx``
     - ``Dpx``
     - —
     - 必填
     - 出口归一化水平动量色散。
   * - ``alpha_x_previous``
     - ``Alpha x previous``
     - —
     - 必填
     - 入口水平 Twiss α。
   * - ``alpha_y_previous``
     - ``Alpha y previous``
     - —
     - 必填
     - 入口垂直 Twiss α。
   * - ``beta_x_previous``
     - ``Beta x previous (m)``
     - m
     - 必填
     - 入口水平 Twiss β，必须为正。
   * - ``beta_y_previous``
     - ``Beta y previous (m)``
     - m
     - 必填
     - 入口垂直 Twiss β，必须为正。
   * - ``mu_x_previous``
     - ``Mu x previous``
     - 圈
     - 必填
     - 入口水平相位。
   * - ``mu_y_previous``
     - ``Mu y previous``
     - 圈
     - 必填
     - 入口垂直相位。
   * - ``mu_z_previous``
     - ``Mu z previous``
     - 圈
     - ``0.0``
     - 入口纵向相位。
   * - ``dx_previous``
     - ``Dx previous (m)``
     - m
     - ``0.0``
     - 入口水平位置色散。
   * - ``dpx_previous``
     - ``Dpx previous``
     - —
     - ``0.0``
     - 入口归一化水平动量色散。
   * - ``dqx``
     - ``DQx``
     - —
     - ``0.0``
     - 本段水平色品相位修正系数。
   * - ``dqy``
     - ``DQy``
     - —
     - ``0.0``
     - 本段垂直色品相位修正系数。
   * - ``longitudinal_transfer``
     - ``Longitudinal transfer``
     - —
     - ``off``
     - ``off``、``drift`` 或 ``matrix``。

输出与适用范围
------------------

该命令更新粒子坐标和参考通过时间，本身不写诊断文件；在出口布置 :doc:`monitor/index` 中的监视器保存结果。横向模型不包含 x–y 耦合或垂直色散。设置非零 ``DQx``、``DQy`` 后，相位依赖相对动量偏差，整体映射不再是固定的六维线性矩阵。

``DQx``、``DQy`` 直接用于本段相位修正，程序不会再乘本段长度占环周长的比例。它们必须与输入光学及额外非线性元件共同定义的色品一致。

运行时手写 JSON 可附加 ``Aperture type``、``Aperture value``（默认 ``off``、``[]``），但 ``TwissItem`` 当前不导出这些字段。使用参数构建器时，可在同位置放置带孔径的 ``Marker``，见 :doc:`aperture`。

物理推导
--------

纵向传输
~~~~~~~~

纵向传输由 ``Longitudinal transfer`` 参数控制，支持三种模式：

**drift 模式** ：使用跃迁 Lorentz 因子，纵向传输矩阵元为：

.. math::

   m_{12,z} = -\left(\frac{1}{\gamma_t^2} - \frac{1}{\gamma^2}\right)(s - s_\mathrm{previous})

其中 :math:`\gamma_t` 为跃迁 Lorentz 因子， :math:`\gamma` 为束团参考粒子的 Lorentz 因子， :math:`s` 为当前纵向位置， :math:`s_\mathrm{previous}` 为前一元件纵向位置。

**matrix 模式** ：使用给定的纵向相位差，传输矩阵为：

.. math::

   m_{11,z} = \cos(\phi_z)

.. math::

   m_{12,z} = \frac{\sigma_z}{\sigma_\delta} \sin(\phi_z)

.. math::

   m_{21,z} = -\frac{\sigma_\delta}{\sigma_z} \sin(\phi_z)

.. math::

   m_{22,z} = \cos(\phi_z)

其中 :math:`\phi_z` 为纵向相移， :math:`\sigma_z` 为配置的 RMS 束长， :math:`\sigma_\delta` 为配置的 RMS 相对动量分散。

**off 模式** ：纵向传输矩阵取单位矩阵。

色散处理
~~~~~~~~

由于横向传输矩阵描述的是非色散部分的运动，需要在传输前后对色散进行去除与恢复：

1. **去除前一点色散** ：

.. math::

   x_1 = x - D_{x,\mathrm{previous}} \cdot \delta

.. math::

   px_1 = px - D_{px,\mathrm{previous}} \cdot \delta

2. **线性传输** ：

.. math::

   x_\mathrm{temp} = x_1 \cdot m_{11} + px_1 \cdot m_{12}

3. **加新点色散** ：

.. math::

   x_2 = x_\mathrm{temp} + D_x \cdot \delta_2

其中 :math:`D_x` 为当前点水平色散， :math:`D_{x,\mathrm{previous}}` 为前一点水平色散， :math:`\delta` 为粒子相对动量偏差。

横向传输矩阵
~~~~~~~~~~~~

横向传输矩阵由前后两点的 Twiss 参数及相位差表示。以水平方向为例：

.. math::

   m_{11,x} = \sqrt{\frac{\beta_x}{\beta_{x,\mathrm{prev}}}}
   \left(\cos\phi_x + \alpha_{x,\mathrm{prev}} \sin\phi_x\right)

.. math::

   m_{12,x} = \sqrt{\beta_x \, \beta_{x,\mathrm{prev}}} \sin\phi_x

.. math::

   m_{21,x} = -\frac{1 + \alpha_x \, \alpha_{x,\mathrm{prev}}}
   {\sqrt{\beta_x \, \beta_{x,\mathrm{prev}}}} \sin\phi_x
   + \frac{\alpha_{x,\mathrm{prev}} - \alpha_x}
   {\sqrt{\beta_x \, \beta_{x,\mathrm{prev}}}} \cos\phi_x

.. math::

   m_{22,x} = \sqrt{\frac{\beta_{x,\mathrm{prev}}}{\beta_x}}
   \left(\cos\phi_x - \alpha_x \sin\phi_x\right)

其中 :math:`\beta_x` 、 :math:`\alpha_x` 为当前点水平 Twiss 参数， :math:`\beta_{x,\mathrm{prev}}` 、 :math:`\alpha_{x,\mathrm{prev}}` 为前一点水平 Twiss 参数， :math:`\phi_x` 为两点间的水平相移。

垂直方向（y）的传输矩阵形式完全相同，只需将下标 x 替换为 y。

色品修正
~~~~~~~~

动量偏差会引起 工作点偏移，通过色品参数对相位进行修正：

.. math::

   \phi_x = \phi_x + \delta \cdot \Delta Q_x \cdot 2\pi

.. math::

   \phi_y = \phi_y + \delta \cdot \Delta Q_y \cdot 2\pi

其中 :math:`\Delta Q_x` 、 :math:`\Delta Q_y` 分别为水平与垂直色品， :math:`\delta` 为粒子相对动量偏差。

纵向坐标连续性
~~~~~~~~~~~~~~

连续保存 z，不折叠多圈累计滑移。物理到达时间为 :math:`t_i=T_b-z_i/(\beta_b c)`，必须使用当前参考事件；名义槽位偏移不参与时间重建。

t0 更新
~~~~~~~

参考时间 t0 根据纵向位置变化更新：

.. math::

   \Delta t = \frac{s - s_\mathrm{previous}}{\beta \, c}

其中 :math:`\beta` 为束团参考粒子的归一化速度， :math:`c` 为光速。

相位与动量记号
------------------

上式的动量偏差为无量纲量 :math:`\delta=P/P_0-1`，不是动量的绝对变化。未加色品修正的相位差为 :math:`\phi_u=2\pi(\mu_u-\mu_{u,\mathrm{previous}})`。纵向 ``matrix`` 使用配置的 RMS 束长与相对动量分散，不由当前粒子统计量自动更新。
