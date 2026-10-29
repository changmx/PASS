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
   * - :doc:`slow_extraction_monitor`
     - 指定引出 source 的 spill
     - 圈数和/或物理时间分箱

.. toctree::
   :maxdepth: 1

   table_output
   plotting
   statmonitor
   distmonitor
   particlemonitor
   phaseadvancemonitor
   slow_extraction_monitor

空槽与粒子记录
--------------

beam 仍保留全部谐波槽位。没有分配粒子记录的空槽会被逐束团监视器跳过，
不生成该束团的输出文件。判断依据是已分配的粒子范围，而非存活粒子数量。
为后续注入预留的记录仍然有效；粒子已经全部损失的束团也继续保留损失诊断和
存活数量为零的统计记录。重分组后如果有粒子进入空槽，监视器会在下一次计划采样时
记录该槽。ParticleMonitor 和 SlowExtractionMonitor 仍保存整个 beam 的轨迹或
引出汇总，空槽不会增加文件。

粒子注入后，其身份编号 ``abs(tag)`` 在损失和重分组过程中保持不变。
损失标记为 ``tag=-particle_id``，磁铁边缘场造成的损失也遵循这一规则，
使轨迹记录和注入快照始终关联原粒子。

.. _zh-reference-state:

输出中的参考信息
------------------------

分布文件头保存 ``ReferenceArrivalTime``、``ReferenceBeta``、
``ReferenceMomentum``、``CoordinateDefinition`` 和元件出口事件标识。
统计监视器逐行保存相应参考参数。ParticleMonitor 仅在 ``"Include reference": true`` 时
保存逐圈参考量。其 HDF5 ``initial`` 组或 TFS 初值行始终保存注入参考量，
它不能代替后续每个样本的参考量。TFS 在关闭此选项时仍保留参考列，轨迹行对应值为 NaN。
必须使用同行参考信息重建存活粒子时间，
不能使用其他圈的参考值。损失坐标是冻结的诊断记录，不能用后续存活束团参考系解释。

物理坐标定义见 :ref:`zh-longitudinal-reference`，通用文件格式与读取方法见 :doc:`table_output`。
