监视器（Monitor）
==================

本模块介绍 PASS 中支持的各类束流监视器。

.. toctree::
   :maxdepth: 2

   statmonitor
   distmonitor
   phaseadvancemonitor
   particlemonitor

.. _zh-reference-state:

输出中的参考信息
------------------------

分布文件头保存 ``ReferenceArrivalTime``、``ReferenceBeta``、
``ReferenceMomentum``、``CoordinateDefinition`` 和元件出口事件标识。
统计监视器逐行保存相应参考参数。ParticleMonitor 仅在 ``"Include reference": true`` 时
保存这些列，默认在数据列和 headers 中均不保存参考量。必须使用同行参考信息重建存活粒子时间，
不能使用其他圈的参考值。损失坐标是冻结的诊断记录，不能用后续存活束团参考系解释。
