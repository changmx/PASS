相位推进监视器（PhaseAdvanceMonitor）
=======================================

``PhaseAdvanceMonitor`` 在固定观测位置逐粒子测量非耦合横向运动的水平和垂直小数工作点。这是唯一正式的相位推进监视器 API，取代原 ``PhaseMonitor``。

监视器先减去固定闭轨和水平色散，再以固定设计 Twiss 参数构造归一化坐标：

.. math::

   u_x = (x-x_{CO}-D_x\delta)/\sqrt{\beta_x},\qquad
   v_x = \alpha_x u_x + \sqrt{\beta_x}(p_x-p_{x,CO}-D'_x\delta).

y 平面使用同样形式但不减色散。相邻圈的有向相位推进被累加并以
``sum(dmu)/(2 pi N)`` 输出为小数工作点；不输出 phase，因为两者仅相差 ``2 pi`` 的固定关系，信息重复。

此监视器假设横向非耦合。存在 x-y 耦合时，x/y 输出为投影工作点，不是正常模工作点。

配置
----

``Turn ranges`` 使用 ``[start, end)``，表示从 0 开始的左闭右开区间
``[start, end)``，至少需要两圈。取值为 ``0``（或空列表）时不进行计算；也可用
``Enable`` 关闭监视器。超出范围的端点自动裁剪到 ``[0, num_turns]``。

.. code-block:: json

   "tune_1": {
       "S (m)": 12.5,
       "Command": "PhaseAdvanceMonitor",
       "Beta x (m)": 18.0,
       "Beta y (m)": 22.0,
       "Alpha x": -0.4,
       "Alpha y": 0.2,
       "Dx (m)": 1.1,
       "Dpx": 0.03,
       "X CO (m)": 0.0,
       "PX CO": 0.0,
       "Y CO (m)": 0.0,
       "PY CO": 0.0,
       "Enable": true,
       "Turn ranges": [[0, 1024]]
   }

``Dx``、``Dpx`` 及全部闭轨字段默认是 0。``Min action`` 可选；归一化 action 过小时不对该采样点求角度。按粒子精度，默认阈值为 float64 的 ``5e-17`` 与 float32 的 ``5e-9``。

输出
----

每个完成窗口、每个 beam、bunch 和 monitor 各写一个 HDF5 文件（可选 TFS），保存在 ``tuneSpread``。每个 bunch 独立输出，包含已损失粒子。列包含 ``tag``、``tuneXFractional``、``tuneYFractional``、两平面各自的区间数、``validX/Y``、``completeX/Y`` 和损失信息。``valid`` 表示粒子仍存活且至少有一个有效区间；``complete`` 表示窗口内所有区间均有效。损失粒子绝不参与相位累加。

文件头记录固定光学参考、窗口端点、期望区间数、后端、精度以及 ``PASSVersion``。

``output_format``（JSON ``"Output format"``）默认为 ``"hdf5-gzip1"``；
设置为 ``"hdf5"`` 使用不压缩的 HDF5，或设置为 ``"tfs"`` 使用文本输出。HDF5 结构、压缩与统一读取方式见
:doc:`table_output`。
