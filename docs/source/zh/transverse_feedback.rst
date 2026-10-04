逐束团横向反馈
==============

``TransversePickup`` 测量每个束团分组槽内活粒子的质心。
配对的 ``TransverseFeedback`` 在另一跟踪截面执行带延迟的有限冲激响应（FIR）滤波，
并对束团施加均匀、零长度横向踢。该模型用于研究相干振荡阻尼及横向不稳定性的抑制，
包括 :doc:`wake_field` 引起的不稳定性。

同一束流序列中，每个拾取器必须恰好对应一个反馈节点。
反馈的 ``Pickup`` 字段直接引用拾取器的序列名称，不需要独立控制器命令或顶层配置块。
``Enable=false`` 关闭这一对节点；独立质心输出请使用 :doc:`monitor/index`。

模型与约定
----------

第 :math:`n` 圈、分组槽 :math:`b` 的拾取信号为

.. math::

   m_{b,n}=\frac{1}{N_{b,n}}\sum_{i\in\mathrm{live}(b,n)}x_i-x_{\mathrm{ref}}.

只有 ``tag > 0`` 的粒子参与计算；当前束流模型中宏粒子权重相同。
固定参考位置的单位为米，拾取器不改变粒子，也不自动估计闭轨。
垂直方向使用相同公式，将 :math:`x` 替换为 :math:`y`。

反馈计算

.. math::

   f_{b,n}=\sum_{k=0}^{L-1}a_k m_{b,n-d-k},\qquad
   \Delta p_{x,b}=\operatorname{clip}(-G_x f_{b,n},-K_x,K_x),

其中 ``Delay turns`` 为 :math:`d\geq1`，系数按最新延迟采样在前排列，
增益 :math:`G_x` 的单位为 :math:`\mathrm{m}^{-1}`，可选限幅 :math:`K_x` 无量纲。
``Max kick x=null`` 表示不限幅。选定分组槽的所有活粒子接受相同踢。
``Harmonic IDs`` 选择分组槽，不代表粒子身份或 RF 谐波。

PASS 存储 :math:`p_x=P_x/P_0`、:math:`p_y=P_y/P_0`；这里施加的是归一化横向动量增量，
不能对所有偏动量粒子都直接解释为同样的几何斜率增量。
这是理想薄横向踢模型，暂不包含电压标定、踢器带宽、束内波形、ADC 噪声、量化或饱和恢复动力学。

刚性束团质心反馈不需要 :doc:`slicer`。
拾取器与反馈必须放在真实跟踪截面；如需放入 Twiss 映射或厚元件内部，必须先拆分传输。
同一 ``S (m)`` 处需要指定物理先后时使用 ``Order``；该位置全部命令均需提供互不重复的值。
节点必须在把粒子传输到该截面的命令之后执行。

时序、预热与分组
----------------

延迟按序列圈号定义：第 :math:`n` 圈反馈使用第 :math:`n-d` 圈拾取结果。
本实现只保存圈号，不保存硬件时间戳，也不模拟以秒为单位的固定硬件延迟。
实际物理时间间隔还包含两个跟踪位置之间的传播时间，升能时不能简单视为固定的
:math:`dT_{\mathrm{rev}}`。
粒子通过时间仍遵循 :math:`t_i=t_0-z_i/(\beta c)`；
``z_center`` 仅为分组元数据。拾取与反馈不折叠或修改纵向坐标。

两个方向共用较长 FIR 的历史长度 :math:`L`，较短系数列在末尾补零。
初始化或某槽重置后，必须获得 :math:`L` 次连续有效采样才能输出。
若从第零圈起每圈均有有效采样，最早踢圈为 :math:`L-1+d`，同时受 ``Start turn`` 限制。
即使反馈踢晚于第零圈启用，拾取历史仍从第零圈开始预热。
空槽及无效采样会禁止输出并要求重新预热。
任何新注入批次，包括向已占用槽追加粒子，都会重置受影响槽的反馈历史，避免施加过时输出。

``SortBunch`` 保留每个分组槽的历史；它替换粒子 tag 数组后，反馈缓存的粒子范围会失效，
在下一次拾取或反馈执行时重建。数组未变时仅比较对象身份，不逐圈扫描全部束团。
启用反馈的预热及活动区间从第零圈延伸到独占的反馈结束圈，期间不允许 ``ReorganizeBunch``。
第一版要求固定分组；粒子跨槽迁移时，反馈历史不会跟随单个粒子。

FIR 相位与增益选择
-------------------

产生阻尼的是传输、FIR 滤波和延迟的总相位。
滤波器能够补相时，拾取器到踢器的几何相移不必固定为 :math:`\pi/2`。
在扣除闭轨后的归一化 betatron 坐标中，阻尼应反向作用于相干动量
:math:`P=(\alpha x+\beta x')/\sqrt{\beta}`。
位置反馈相位不合适时，可能改变 tune 或激发增长。

``design_feedback_fir(tune, phase_advance, delay_turns=1, tap_count=5, *, method="minimum_norm")``
返回实数系数。这是可选的输入生成辅助函数；跟踪始终使用显式系数列表，
JSON 命令没有 ``method`` 字段。``phase_advance`` 是 **同一序列圈号下** 的
:math:`\mu_{\mathrm{feedback}}-\mu_{\mathrm{pickup}}`，单位 rad。
反馈节点位于拾取器上游时，该值可以为负；不要重复加入已由 ``delay_turns`` 表示的整圈相位。

对于 :math:`x_n=\cos(\omega n+\phi)` 和设计频率 :math:`\omega_0=2\pi Q`，两种方法均要求

.. math::

   H_d(\omega)=\sum_k a_ke^{-i\omega(d+k)},\qquad
   H_d(\omega_0)=e^{i(\Delta\mu+\pi/2)},\qquad \sum_k a_k=0.

结合命令踢公式中的负号，正增益在目标 tune 处反向作用于相干动量。
系数和为零使滤波器在预热后抑制固定拾取偏置。
两种系数设计如下：

.. list-table:: FIR 设计方法
   :header-rows: 1
   :widths: 22 15 63

   * - ``method``
     - 最少系数数目
     - 约束和目标
   * - ``"minimum_norm"`` （默认）
     - 3
     - 满足去直流和指定复响应三个实数约束，再使系数欧氏范数最小。
       原有调用保持此设计。
   * - ``"flat"``
     - 5
     - 除上述约束外，要求目标 tune 处包含延迟的复响应导数为零。
       系数多于五个时，在满足全部五个实数约束的解中选择最小范数解。

平坦响应方法增加条件

.. math::

   H'_d(\omega_0)=-i\sum_k(d+k)a_ke^{-i\omega_0(d+k)}=0.

在线性系统中增加 :math:`(d+k)\cos(k\omega_0)` 和
:math:`(d+k)\sin(k\omega_0)` 两行，目标值均为零。
这里必须包含完整延迟 :math:`d+k`，不能只对系数下标求导。
在拾取器到踢器的几何相移固定时，目标 tune 处的幅值和相位一阶导数均为零，
残余响应变化从二阶开始。这是局部平坦性，不保证某个有限带宽内的性能。

函数独立求解这些约束，不使用跟踪数据。若设计矩阵秩不足或病态，包括整数及
半整数 tune 附近的情况，则拒绝设计；条件数上限为 :math:`10^8`。
对于平坦响应方法，求解和条件数检查前，前三行除以 :math:`\sqrt L`，
导数两行除以 :math:`\sqrt{\sum_k(d+k)^2}`。正弦/余弦行使用相同尺度，
避免放大奇异 tune 附近的微小行。未缩放约束的残差还必须不超过
:math:`5\times10^{-10}`。

函数仅将一个目标 tune 处的响应幅度归一化，不保证闭环稳定性、不自动选增益、
不补偿光学幅度比，也不保证某个阻尼时间。
更平坦的响应可能需要更大的系数，也可能有更小的稳定增益范围。
设计频率应参考预计的相干质心振荡谱；尾场及其他集体作用可能使它偏离裸晶格 tune。
平坦性是在几何相移固定的条件下计算的；如果机器设置改变时该相移也改变，
扫描中应同时考虑这种变化。
应先用小增益在完整晶格与延迟下验证，再扫描增益、tune 与相位。
用户直接输入的 FIR 系数不强制和为零，因此可能对静态轨道偏置产生踢。

Python 输入
-----------

把以下两个节点加入已在目标位置设置传输边界的序列；示例相移仅作演示，实际模拟应使用晶格值：

.. code-block:: python

   import math
   from PASS.para.api import (
       TransversePickupItem, TransverseFeedbackItem, design_feedback_fir,
   )

   taps = design_feedback_fir(
       tune=0.23, phase_advance=math.pi / 3, delay_turns=1, tap_count=5, method="flat",
   )
   seq.add("pickup_x", TransversePickupItem(s=0.0, plane="x"))
   seq.add("feedback_x", TransverseFeedbackItem(
       s=10.0, pickup="pickup_x", coefficients_x=taps,
       gain_x=0.01, delay_turns=1, max_kick_x=1e-4,
       diagnostics_interval=10,
   ))

仍使用 ``generate_input(main, seq, output_path)`` 导出，不增加生成器参数。
GUI 的横向反馈组件区提供两类节点；新反馈草稿的增益及系数均为零，必须配置后才会产生阻尼。
反馈面板中的 **生成 FIR 系数…** 使用同一个辅助函数。
引用的拾取器决定生成 x、y 或两个方向的系数；各方向独立填写 tune 和同一圈号下的
有符号相移，单位 rad。GUI 不从节点位置或机器配置推断这些数值。
两个方向共用方法（默认 ``minimum_norm``）、延迟及系数数量。
GUI 允许 3–256 个系数，其中 ``flat`` 至少需要五个；辅助函数本身没有 256 个系数的上限。

先 **计算并预览**，再 **填入草稿**，替换测量方向的系数、清空未测量方向的系数并同步延迟。
增益保持原值，因此初始零增益仍需在跟踪前自行设置。
系数数组可继续手动编辑；点击 **应用** 或 **插入** 才保存草稿。
tune、相移和设计方法不会成为 JSON 字段。完整操作见 :doc:`gui`。

接口
----

两个节点都接受 ``s`` / ``S (m)``（非负有限数，单位米）及可选严格整数 ``order`` / ``Order``。
序列键提供节点名称。

.. list-table:: TransversePickupItem
   :header-rows: 1
   :widths: 22 24 14 40

   * - Python 参数
     - JSON 别名
     - 默认值
     - 含义
   * - ``plane``
     - ``Plane``
     - ``"x"``
     - ``x``、``y`` 或独立双平面 ``xy``。
   * - ``reference_x``、``reference_y``
     - ``Reference x (m)``、``Reference y (m)``
     - ``0.0``
     - 固定拾取参考位置，单位米。

.. list-table:: TransverseFeedbackItem
   :header-rows: 1
   :widths: 22 25 13 40

   * - Python 参数
     - JSON 别名
     - 默认值
     - 含义
   * - ``pickup``
     - ``Pickup``
     - 必填
     - 本束流拾取器的精确序列名称。
   * - ``enabled``
     - ``Enable``
     - ``True``
     - 严格布尔值；false 时关闭这一对节点。
   * - ``start_turn``、``end_turn``
     - ``Start turn``、``End turn``
     - ``0``、``None``
     - 踢的圈数窗口；非负严格整数，结束圈不包含在内；null 表示运行末尾。
   * - ``delay_turns``
     - ``Delay turns``
     - ``1``
     - 至少为 1 的严格整数。
   * - ``coefficients_x``、``coefficients_y``
     - ``FIR coefficients x``、``FIR coefficients y``
     - ``None``
     - 每个测量方向必须提供非空有限数列；未测量方向必须为 null。
   * - ``gain_x``、``gain_y``
     - ``Gain x (1/m)``、``Gain y (1/m)``
     - ``0.0``
     - 有符号增益，单位 1/m；未测量方向必须为零。
   * - ``max_kick_x``、``max_kick_y``
     - ``Max kick x``、``Max kick y``
     - ``None``
     - 归一化动量踢绝对值的非负上限；null 表示不限幅。
   * - ``bunch_ids``
     - ``Harmonic IDs``
     - ``None``
     - 不重复的非负严格整数；null 选择所有分组槽。
   * - ``diagnostics_interval``
     - ``Diagnostics interval (turns)``
     - ``0``
     - 非负严格整数；零表示不输出反馈诊断。

状态与数值范围
--------------

两个命令在构造时自行登记反馈对，双方就绪后直接关联，不依赖构造或跟踪先后顺序。
任一节点首次执行时初始化反馈缓冲，不需要执行器准备钩子。
注入命令在新批次进入某槽时通知反馈对，因此即使踢器位于拾取器上游，也会及时重置受影响槽的历史。
CPU 与 GPU 实现在同一跟踪模块内；GPU 路径的质心归约、FIR 历史、延迟输出及粒子踢均保留在设备端，
诊断输出按独立的可选采样周期传输。

束团粒子范围必须完整覆盖粒子且互不重叠。GPU 粒子数组必须是一维连续数组，
并满足声明的精度、长度和设备要求；这些检查不传输粒子数据。
质心或减去参考位置后的信号若为非有限数，该次采样无效，并需要重新完成滤波预热；
无效采样不会写入保存的历史。

``Diagnostics interval (turns)`` 为正时，在本次运行输出目录下写入
``feedback/beam<beam_id>_<safe_command_name>_<hash>.csv``。
每行含分组槽 ID、拾取质心、活粒子数、滤波位置、实际踢及限幅标记。
``sample_turn`` 是最新拾取采样圈；``filter_target_turn=sample_turn+d``
是该滤波位置排队等待的目标圈。
``kick_turn`` 是实际施踢圈，``requested_source_turn=kick_turn-d`` 是请求的延迟信号源圈；
预热或重置后，该圈可能没有可用采样。
因此，同一行的滤波位置通常不直接产生这一行记录的踢。
输出每 128 个采样圈批量写入，结束时写入剩余缓冲，不覆盖已有文件。

反馈状态导出并不等于完整模拟检查点：续跑还需要匹配的粒子、束团参考事件、时钟及圈号状态。
现有碰撞检查点接口暂不支持反馈对，会拒绝保存，以免静默漏掉反馈历史。
