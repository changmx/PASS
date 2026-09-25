监视器与输出
============

根据所需分析数据选择监视器。监视器位置与晶格元件使用相同的 s 坐标，圈号从 0 开始。

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - 监视器
     - 用途
     - 记录范围
   * - :doc:`statmonitor`
     - 束团统计矩及损失
     - 逐圈
   * - :doc:`distmonitor`
     - 粒子分布快照
     - 指定圈号
   * - :doc:`particlemonitor`
     - 所选粒子的轨迹
     - [start, end) 内逐圈
   * - :doc:`phaseadvancemonitor`
     - 单粒子小数工作点
     - 完整分析窗口

.. toctree::
   :maxdepth: 1

   table_output
   statmonitor
   distmonitor
   particlemonitor
   phaseadvancemonitor

.. _zh-reference-state:

输出中的参考信息
------------------------

分布文件头保存 ``ReferenceArrivalTime``、``ReferenceBeta``、
``ReferenceMomentum``、``CoordinateDefinition`` 和元件出口事件标识。
统计监视器逐行保存相应参考参数。ParticleMonitor 仅在 ``"Include reference": true`` 时
保存这些列，默认在数据列和 headers 中均不保存参考量。必须使用同行参考信息重建存活粒子时间，
不能使用其他圈的参考值。损失坐标是冻结的诊断记录，不能用后续存活束团参考系解释。

物理坐标定义见 :ref:`zh-longitudinal-reference`，通用文件格式与读取方法见 :doc:`table_output`。
