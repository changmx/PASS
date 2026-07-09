标记（Marker）
==================

本模块介绍 PASS 中的标记器元件 **Marker** ，用于在束线中标记一个特定的纵向位置，不改变任何粒子坐标。

PASS 中的标记器为 **零长度元件** （ ``length = 0`` ），其 ``execute_cpu`` 和 ``execute_gpu`` 均为空操作。标记器的主要作用包括：

- 在束线序列中标记关键位置 （如测量点、对撞点、注入点等），便于后续分析
- 作为排序基准点，其他元件可参考标记器位置进行布局
- 在输出和日志中标识物理位置，不参与粒子追踪


物理说明
--------

标记器不产生任何电磁场，不施加任何力，也不改变粒子状态。粒子穿过标记器时，所有六个相空间坐标 （ :math:`x, p_x, y, p_y, z, \delta` ）保持不变：

.. math::

  x \leftarrow x

.. math::

  p_x \leftarrow p_x

.. math::

  y \leftarrow y

.. math::

  p_y \leftarrow p_y

.. math::

  z \leftarrow z

.. math::

  \delta \leftarrow \delta

标记器仅记录其在束线中的纵向位置 :math:`s` ，用于序列排序和位置标注。


参数列表
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
    - -
    - float
    - m
    - 元件长度 （恒为 0，不可配置）
  * - ``name``
    - ``name``
    - str
    - -
    - 元件名称 （由序列 JSON 的键名自动填入）

.. note::

  标记器无需提供 ``Length (m)`` 字段，长度在代码中固定为 0。

  ``Command`` 字段应设为 ``Marker`` 。


使用示例
--------

输入文件示例
~~~~~~~~~~~~~~~~

以下示例取自 ``input/beam0.json`` ，在 :math:`s = 12.5` m 处放置一个标记器：

.. code-block:: json

  {
      "IP": {
          "S (m)": 12.5,
          "Command": "Marker"
      }
  }

``"IP"`` 为元件名称，由 ``CommandSequence`` 自动读取并赋给 ``name`` 属性。

多个标记器可在同一束线中重复使用，例如标记对撞点、注入点和测量点：

.. code-block:: json

  {
      "IP": {
          "S (m)": 12.5,
          "Command": "Marker"
      },
      "Injection_Point": {
          "S (m)": 0.0,
          "Command": "Marker"
      },
      "Measurement_Point": {
          "S (m)": 105.3,
          "Command": "Marker"
      }
  }


应用场景
--------

- **对撞点标注** ：在对撞机中标记束流对撞位置，便于亮度计算和束束作用元件定位
- **注入点标注** ：标记注入元件的位置，便于注入器与主环的衔接
- **测量点标注** ：标记监视器 （Monitor） 所在位置，便于数据分析与物理量提取
- **序列排序基准** ：利用标记器的 ``s`` 位置作为参考，组织束线中其他元件的排列顺序
