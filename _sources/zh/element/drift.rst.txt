漂移节（Drift）
==================

本模块介绍 PASS 中的漂移节元件 **Drift** ，用于模拟粒子在无场自由空间中的传输。漂移节是最基本的束线元件，粒子在其中不受任何电磁力作用，仅凭借初始动量做直线运动。

**代码位置**

- 源文件： ``PASS/commands/element/drift.py``
- 类名： ``Drift`` （继承自 ``Command`` ）
- 注册名： ``drift``
- 核心特征：

  - 厚元件（ ``length > 0`` ），改变粒子的位置和纵向坐标；
  - 使用精确的几何传输公式，考虑横向动量对纵向速度的投影；
  - 支持孔径检查，与其它元件一致。


物理推导
--------

粒子在漂移节中不受力，以恒定动量做直线运动。设漂移节长度为 :math:`L` ，粒子归一化横向动量为 :math:`p_x` 、 :math:`p_y` ，动量偏差为 :math:`\delta` 。

**粒子总动量**

归一化总动量（以参考粒子动量 :math:`P_0` 为单位）为：

.. math::

  P_{\text{tot}} = 1 + \delta

纵向动量分量（考虑横向动量的投影）为：

.. math::

  p_z = \sqrt{(1 + \delta)^2 - p_x^2 - p_y^2}

若 :math:`p_z^2 \le 0` ，粒子物理上不可能存在 （横向动量大于总动量），标记为丢失。

**粒子速度**

粒子 :math:`\beta` 值由参考粒子 :math:`\beta_0` 、 :math:`\gamma_0` 和动量偏差 :math:`\delta` 计算：

.. math::

  \beta = \frac{(1 + \delta) \, \gamma_0 \, \beta_0}{\sqrt{1 + \left[(1 + \delta) \, \gamma_0 \, \beta_0\right]^2}}

**坐标更新**

漂移节中粒子坐标更新为：

.. math::

  x \leftarrow x + L \cdot \frac{p_x}{p_z}

.. math::

  y \leftarrow y + L \cdot \frac{p_y}{p_z}

.. math::

  z \leftarrow z + L \cdot \left(1 - \frac{\beta_0}{\beta} \cdot \frac{1 + \delta}{p_z}\right)

其中 :math:`z` 的更新包含路径长度差效应：具有动量偏差的粒子速度不同，导致纵向位置发生变化。

**纵向坐标连续性**

Drift 不对更新后的 :math:`z_{\mathrm{rel}}` 做环周折叠。连续保存纵向滑移可避免丢失多圈累计信息；需要实验室坐标时使用 :math:`z_{\mathrm{lab}}=z_{\mathrm{rel}}+z_{\mathrm{center}}`。


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
    - ``S (m)``
    - float
    - m
    - 元件在束线中的纵向位置
  * - ``length``
    - ``Length (m)``
    - float
    - m
    - 元件长度 （必须 :math:`\ge 0` ）
  * - ``name``
    - ``name``
    - str
    - -
    - 元件名称 （由序列 JSON 的键名自动填入）
  * - ``aperture_type``
    - ``Aperture Type``
    - str
    - -
    - 孔径类型 （默认 ``off`` ，可选值见孔径章节）
  * - ``aperture_value``
    - ``Aperture Value``
    - list
    - -
    - 孔径参数值 （默认 ``[]`` ，含义随类型而异，详见孔径章节）


使用示例
--------

以下 JSON 片段展示了漂移节的配置方式：

**基本用法** ：

.. code-block:: json

  "Drift1": {
      "S (m)": 10.0,
      "Command": "Drift",
      "Length (m)": 0.5,
      "Aperture Type": "off"
  }

**带圆形孔径检查** ：

.. code-block:: json

  "Drift2": {
      "S (m)": 10.5,
      "Command": "Drift",
      "Length (m)": 0.3,
      "Aperture Type": "circle",
      "Aperture Value": [0.05]
  }

**带矩形孔径检查** ：

.. code-block:: json

  "Drift3": {
      "S (m)": 11.0,
      "Command": "Drift",
      "Length (m)": 0.2,
      "Aperture Type": "rectangle",
      "Aperture Value": [0.06, 0.04]
  }


应用场景
--------

- **束线连接** ：在各磁铁元件之间提供自由漂移空间，是最常用的束线元件
- **色散测量** ：在偏转磁铁后设置漂移段，利用色散效应测量束流动量分散
- **束流传输** ：在注入线和引出线中传输束流，不施加任何场
- **孔径检查** ：在关键位置设置带孔径检查的漂移节，监控束流损失
