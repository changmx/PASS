Twiss 传输（Twiss）
====================

简介
----

``Twiss`` 元件基于 Twiss 参数实现 6D 线性光学传输。它使用束流光学函数（beta、alpha、mu、色散）构建传输矩阵，对粒子进行线性光学追踪，是连接 lattice 设计与粒子追踪的核心元件之一。

- **代码位置** ： ``PASS/commands/twiss.py``
- **类名** ： ``Twiss`` ，注册名 ``"twiss"``
- **核心特征** ：

  - 基于前后两点的 Twiss 参数（beta、alpha、相位）构建横向传输矩阵；
  - 支持色散的去除与恢复，保证动量偏差粒子在色散区正确传输；
  - 支持色品修正，动量偏差引起的 tune 偏移可自动计入相位；
  - 纵向传输支持 drift、matrix 及单位矩阵三种模式；
  - 支持 z 坐标回旋折叠，保持纵向坐标在环周长范围内；
  - 支持孔径检查，与其它元件一致。

物理推导
--------

纵向传输
~~~~~~~~

纵向传输由 ``Longitudinal Transfer`` 参数控制，支持三种模式：

**drift 模式** ：使用 gamma transition 参数，纵向传输矩阵元为：

.. math::

   m_{12,z} = -\left(\frac{1}{\gamma_t^2} - \frac{1}{\gamma^2}\right)(s - s_\mathrm{previous})

其中 :math:`\gamma_t` 为过渡 gamma， :math:`\gamma` 为粒子相对论 gamma， :math:`s` 为当前纵向位置， :math:`s_\mathrm{previous}` 为前一元件纵向位置。

**matrix 模式** ：使用纵向振荡频率，传输矩阵为：

.. math::

   m_{11,z} = \cos(\phi_z)

.. math::

   m_{12,z} = \frac{\sigma_z}{\Delta p_\mathrm{bunch}} \sin(\phi_z)

.. math::

   m_{21,z} = -\frac{\Delta p_\mathrm{bunch}}{\sigma_z} \sin(\phi_z)

.. math::

   m_{22,z} = \cos(\phi_z)

其中 :math:`\phi_z` 为纵向相移， :math:`\sigma_z` 为束团纵向尺寸， :math:`\Delta p_\mathrm{bunch}` 为束团动量展宽。

**其他模式** ：纵向传输矩阵取单位矩阵。

色散处理
~~~~~~~~

由于横向传输矩阵描述的是非色散部分的运动，需要在传输前后对色散进行去除与恢复：

1. **去除前一点色散** ：

.. math::

   x_1 = x - D_{x,\mathrm{previous}} \cdot \Delta p

.. math::

   px_1 = px - D_{px,\mathrm{previous}} \cdot \Delta p

2. **线性传输** ：

.. math::

   x_\mathrm{temp} = x_1 \cdot m_{11} + px_1 \cdot m_{12}

3. **加新点色散** ：

.. math::

   x_2 = x_\mathrm{temp} + D_x \cdot \Delta p_2

其中 :math:`D_x` 为当前点水平色散， :math:`D_{x,\mathrm{previous}}` 为前一点水平色散， :math:`\Delta p` 为粒子动量偏差。

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

动量偏差会引起 tune 偏移，通过色品参数对相位进行修正：

.. math::

   \phi_x = \phi_x + \Delta p \cdot \Delta Q_x \cdot 2\pi

.. math::

   \phi_y = \phi_y + \Delta p \cdot \Delta Q_y \cdot 2\pi

其中 :math:`\Delta Q_x` 、 :math:`\Delta Q_y` 分别为水平与垂直色品， :math:`\Delta p` 为粒子动量偏差。

z 坐标回旋折叠
~~~~~~~~~~~~~~

为保持纵向坐标在环周长范围内，传输后对 z 坐标进行回旋折叠：

.. math::

   z_2 = z_2 + (\mathrm{under} - \mathrm{over}) \cdot C

其中 :math:`C` 为环周长， ``under`` 与 ``over`` 为折叠计数。

t0 更新
~~~~~~~

参考时间 t0 根据纵向位置变化更新：

.. math::

   \Delta t = \frac{s - s_\mathrm{previous}}{\beta \, c}

其中 :math:`\beta` 为粒子相对论速度， :math:`c` 为光速。

接口参数
--------

位置参数
~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - 参数名
     - 键名
     - 类型
     - 单位
     - 说明
   * - ``s``
     - ``"S (m)"``
     - float
     - m
     - 当前元件纵向位置
   * - ``s_previous``
     - ``"S Previous (m)"``
     - float
     - m
     - 前一元件纵向位置
   * - ``name``
     - ``"name"``
     - str
     - -
     - 由序列键名自动填入

横向参数
~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - 参数名
     - 键名
     - 类型
     - 单位
     - 说明
   * - ``alphax``
     - ``"Alpha X"``
     - float
     - -
     - 当前点水平 alpha
   * - ``alphay``
     - ``"Alpha Y"``
     - float
     - -
     - 当前点垂直 alpha
   * - ``alphax_previous``
     - ``"Alpha X Previous"``
     - float
     - -
     - 前一点水平 alpha
   * - ``alphay_previous``
     - ``"Alpha Y Previous"``
     - float
     - -
     - 前一点垂直 alpha
   * - ``betax``
     - ``"Beta X (m)"``
     - float
     - m
     - 当前点水平 beta
   * - ``betay``
     - ``"Beta Y (m)"``
     - float
     - m
     - 当前点垂直 beta
   * - ``betax_previous``
     - ``"Beta X Previous (m)"``
     - float
     - m
     - 前一点水平 beta
   * - ``betay_previous``
     - ``"Beta Y Previous (m)"``
     - float
     - m
     - 前一点垂直 beta
   * - ``mux``
     - ``"Mu X"``
     - float
     - -
     - 当前点水平相位
   * - ``muy``
     - ``"Mu Y"``
     - float
     - -
     - 当前点垂直相位
   * - ``mux_previous``
     - ``"Mu X Previous"``
     - float
     - -
     - 前一点水平相位
   * - ``muy_previous``
     - ``"Mu Y Previous"``
     - float
     - -
     - 前一点垂直相位

纵向参数
~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - 参数名
     - 键名
     - 类型
     - 单位
     - 说明
   * - ``longitudinal_transfer``
     - ``"Longitudinal Transfer"``
     - str
     - -
     - 纵向传输模式（drift/matrix/其他）
   * - ``muz``
     - ``"Mu Z"``
     - float
     - -
     - 当前点纵向相位（可选，默认0）
   * - ``muz_previous``
     - ``"Mu Z Previous"``
     - float
     - -
     - 前一点纵向相位（可选，默认0）

色散与色品
~~~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - 参数名
     - 键名
     - 类型
     - 单位
     - 说明
   * - ``Dx``
     - ``"Dx (m)"``
     - float
     - m
     - 当前点水平色散
   * - ``Dx_previous``
     - ``"Dx Previous (m)"``
     - float
     - m
     - 前一点水平色散
   * - ``Dpx``
     - ``"Dpx"``
     - float
     - -
     - 当前点水平色散导数
   * - ``Dpx_previous``
     - ``"Dpx Previous"``
     - float
     - -
     - 前一点水平色散导数
   * - ``DQx``
     - ``"Dqx"``
     - float
     - -
     - 水平色品
   * - ``DQy``
     - ``"Dqy"``
     - float
     - -
     - 垂直色品

孔径参数
~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - 参数名
     - 键名
     - 类型
     - 单位
     - 说明
   * - ``aperture_type``
     - ``"Aperture Type"``
     - str
     - -
     - 孔径类型（默认off）
   * - ``aperture_value``
     - ``"Aperture Value"``
     - list
     - -
     - 孔径参数（默认[]）

使用示例
--------

以下 JSON 片段展示了一个完整的 ``Twiss`` 元件定义：

.. code-block:: json

   "Twiss1": {
       "S (m)": 10.0,
       "Command": "Twiss",
       "S Previous (m)": 5.0,
       "Alpha X": 0.5,
       "Alpha Y": -0.3,
       "Alpha X Previous": 0.4,
       "Alpha Y Previous": -0.2,
       "Beta X (m)": 3.5,
       "Beta Y (m)": 2.8,
       "Beta X Previous (m)": 3.0,
       "Beta Y Previous (m)": 2.5,
       "Mu X": 0.123,
       "Mu Y": 0.456,
       "Mu X Previous": 0.1,
       "Mu Y Previous": 0.4,
       "Dx (m)": 0.5,
       "Dx Previous (m)": 0.3,
       "Dpx": 0.01,
       "Dpx Previous": 0.005,
       "Dqx": 2.0,
       "Dqy": 2.0,
       "Longitudinal Transfer": "drift",
       "Aperture Type": "off"
   }

其中：

- ``"S (m)": 10.0`` —— 当前元件纵向位置为 10.0 m；
- ``"S Previous (m)": 5.0`` —— 前一元件纵向位置为 5.0 m；
- ``"Alpha X": 0.5`` —— 当前点水平 alpha 为 0.5；
- ``"Beta X (m)": 3.5`` —— 当前点水平 beta 为 3.5 m；
- ``"Mu X": 0.123`` —— 当前点水平相位为 0.123（以 :math:`2\pi` 为单位）；
- ``"Dx (m)": 0.5`` —— 当前点水平色散为 0.5 m；
- ``"Dqx": 2.0`` —— 水平色品为 2.0；
- ``"Longitudinal Transfer": "drift"`` —— 纵向传输采用 drift 模式；
- ``"Aperture Type": "off"`` —— 关闭孔径检查。

应用场景
--------

``Twiss`` 元件适用于以下场景：

- **基于 lattice 设计的线性追踪** ：当已有 MadX、AT 等光学计算程序输出的 Twiss 参数表时，可直接使用 ``Twiss`` 元件进行粒子追踪，无需重新建模磁铁元件。
- **色散与色品研究** ： ``Twiss`` 元件内置色散去除/恢复与色品修正，适合研究动量偏差粒子的横向动力学。
- **纵向动力学模拟** ：通过选择 drift 或 matrix 纵向传输模式，可模拟不同纵向传输场景。
- **快速光学评估** ：相比逐元件建模，基于 Twiss 参数的线性传输计算量更小，适合大规模参数扫描与初步评估。
- **与非线性元件混合使用** ： ``Twiss`` 元件可与 sextupole、octupole 等非线性元件串联使用，在线性传输基础上叠加非线性效应。
