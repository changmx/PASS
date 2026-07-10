标记（Marker）
====================

本模块介绍 PASS 中的标记器元件 **Marker** ，用于在束线中标记一个特定的纵向位置，不改变任何粒子坐标。

标记器代码位于 ``PASS/commands/element/marker.py`` ，类名为 ``Marker`` ，注册名为 ``marker`` 。标记器的核心特征如下：

- **零长度元件** （ ``length = 0.0`` ，不可配置），不占据束线物理空间
- **不做任何粒子坐标变换** ，粒子穿过标记器时所有相空间坐标保持不变
- **支持孔径检查** ，通过 ``aperture_type`` 和 ``aperture_value`` 参数控制， ``execute_cpu`` 中仅调用 ``check_aperture_cpu`` 函数
- **GPU 追踪为空操作** （ ``execute_gpu`` 为 ``pass`` ），不执行任何计算

标记器的主要作用包括：

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

当配置了孔径检查 （ ``aperture_type`` 不为 ``off`` ）时，标记器在 ``execute_cpu`` 中调用 ``check_aperture_cpu`` 函数，对存活粒子 （ :math:`\text{tag} > 0` ）进行横向坐标 （ :math:`(x, y)` ）的孔径边界检查。超出孔径边界的粒子将被标记为丢失 （ ``tag`` 取负） ，并记录丢失位置和圈数。孔径检查的详细原理与各类型定义详见 ``孔径`` 章节。


接口参数
--------

标记器的所有接口参数如下表所示：

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
    - - （固定 0.0，不可配置）
    - float
    - m
    - 元件长度 （恒为 0，不可配置）
  * - ``name``
    - ``name``
    - str
    - -
    - 元件名称 （由序列 JSON 的键名自动填入）
  * - ``aperture_type``
    - ``Aperture Type``
    - str
    - -
    - 孔径类型，默认 ``off`` ，不区分大小写
  * - ``aperture_value``
    - ``Aperture Value``
    - list
    - -
    - 孔径参数值，默认 ``[]`` ，含义随类型而异


孔径类型可选值
~~~~~~~~~~~~~~

``aperture_type`` 不区分大小写，内部统一转换为小写后匹配。可选值如下：

.. list-table::
  :header-rows: 1
  :widths: 15 25 60

  * - 类型
    - aperture_value
    - 说明
  * - ``off``
    - 忽略
    - 不做孔径检查 （默认）
  * - ``default``
    - 忽略
    - 默认 ±1m 矩形孔径
  * - ``circle``
    - ``[r]``
    - 圆形， :math:`r` 为半径
  * - ``rectangle``
    - ``[w, h]``
    - 矩形， :math:`w` 为半宽， :math:`h` 为半高
  * - ``ellipse``
    - ``[a, b]``
    - 椭圆， :math:`a` 为半长轴， :math:`b` 为半短轴
  * - ``rectcircle``
    - ``[w, h, r]``
    - 矩形与圆的交集
  * - ``rectellipse``
    - ``[w, h, a, b]``
    - 矩形与椭圆的交集
  * - ``racetrack``
    - ``[w, h, a, b]``
    - 跑道形 （矩形 + 椭圆端）
  * - ``octagon``
    - ``[w, h, d]``
    - 八角形 （矩形切 45° 角）
  * - ``polygon``
    - ``[[x1,y1], ...]``
    - 多边形顶点列表，自动闭合

.. note::

  标记器无需提供 ``Length (m)`` 字段，长度在代码中固定为 0。

  ``Command`` 字段应设为 ``Marker`` 。

  ``off`` 和 ``default`` 类型忽略 ``aperture_value`` 。各孔径类型的详细物理说明与判定条件见 ``孔径`` 章节。


使用示例
--------

基本用法
~~~~~~~~~~

以下示例在 :math:`s = 12.5` m 处放置一个标记器，标记对撞点位置，不启用孔径检查：

.. code-block:: json

  {
      "IP": {
          "S (m)": 12.5,
          "Command": "Marker"
      }
  }

``"IP"`` 为元件名称，由 ``CommandSequence`` 自动读取并赋给 ``name`` 属性。

带孔径检查的标记器
~~~~~~~~~~~~~~~~~~~~

标记器也可配置孔径检查，用于在该位置检测超出束流管边界的粒子。以下示例在 :math:`s = 12.5` m 处放置一个圆形孔径 （半径 0.05 m） 的标记器：

.. code-block:: json

  {
      "IP": {
          "S (m)": 12.5,
          "Command": "Marker",
          "Aperture Type": "circle",
          "Aperture Value": [0.05]
      }
  }

多个标记器组合
~~~~~~~~~~~~~~~~~~

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
          "Command": "Marker",
          "Aperture Type": "rectangle",
          "Aperture Value": [0.05, 0.03]
      }
  }


应用场景
--------

- **对撞点标注** ：在对撞机中标记束流对撞位置，便于亮度计算和束束作用元件定位
- **注入点标注** ：标记注入元件的位置，便于注入器与主环的衔接
- **测量点标注** ：标记监视器 （Monitor） 所在位置，便于数据分析与物理量提取
- **序列排序基准** ：利用标记器的 ``s`` 位置作为参考，组织束线中其他元件的排列顺序
- **孔径监测** ：在关键位置 （如对撞点、注入点） 配置孔径检查，监测束流损失，无需额外添加物理元件
