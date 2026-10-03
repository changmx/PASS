慢引出（SlowExtraction）
========================

``SlowExtraction`` 在已有跟踪截面上收集横向切线指定一侧的存活粒子，
随后停止这些粒子的环内跟踪，并向 :doc:`monitor/slow_extraction_monitor`
提供不可变的粒子事件。六极磁铁、激励器、静电 septum 和其间的晶格仍负责
实际物理输运。本命令不施加踢、不补漂移，也不改变参考粒子。

选择判据与粒子状态
------------------

对于配置的横向滚转角 :math:`\theta`，切线坐标为

.. math::

   u=x\cos\theta-y\sin\theta.

``Side="positive"`` 选择 :math:`u>u_{\rm cut}`；
``Side="negative"`` 选择 :math:`u<u_{\rm cut}`。
``Position (m)`` 给出 :math:`u_{\rm cut}`，恰好落在切线上的粒子不被选中。
只有 ``tag > 0`` 的存活粒子可被选中。该半平面在正交横向方向上无限延伸，
不代表实体材料边界。

命令首先把每个选中粒子的坐标、身份、权重和参考参数完整复制到独立的主机事件缓冲区，
再将原环内 ``tag`` 改为 ``-tag``，并在 ``lost_turn``、``lost_position`` 中
记录停止跟踪的圈号与截面。坐标和动量不变。后续跟踪跳过这些粒子，因此一个粒子
只收集一次。CPU 与 GPU 使用同一选择逻辑；GPU 路径在修改原 tag 前先将事件复制到主机。

GPU 路径使用专门的 CUDA kernel 合并筛选、事件打包与环内停止标记，并复用设备工作缓冲区。
每个选中束团批次的身份与六维坐标通过一个连续数据块传回主机。切线判定仍使用 float64，
保留原坐标存储精度与事件顺序，并在停止环内跟踪前完成独立主机快照。
spill 监视器消费这些主机事件，因此当前实现仍需同步以发布每次调用的结果。
``Buffer size (particles)`` 控制写盘批次，不控制 GPU 回传间隔。

这里的负 tag 表示环内跟踪结束，不一定表示材料损失。现有 StatMonitor 的损失计数
会包含已引出粒子。应合并同一次运行、同束流的所有引出 source 事件表，
按 ``(RunId, beam_id, particle_id)`` 判断是否属于引出，从而与物理损失区分。
与分布快照核对时，只计入截至该快照圈号及命令位置/顺序已经执行的引出事件；
同截面在引出动作之前保存的快照仍会显示这些粒子存活。普通分布快照仍可用于查看环内状态。

位置与执行顺序
--------------

``S (m)`` 标识粒子已经到达的跟踪截面，设置该值不会自动补齐输运。
在漂移段或厚元件内部设置引出点前，必须正确拆分其传输；输入校验会拒绝位于
未拆分厚元件内部或 Twiss 映射 ``S previous (m)`` 至 ``S (m)`` 区间内部的引出面。
在传输终点，引出必须在相应传输之后执行，显式设置 ``Order`` 时也必须满足这一条件。
建议优先使用现有元件边界。所选截面和切线应代表
预期的引出通道；在环内其他位置出现大位移，本身不能证明粒子已经成功引出。

同位置的默认优先级为：``SlowExtraction`` 850，``SlowExtractionMonitor`` 860，
普通监视器 800，因此默认先保存普通分布快照，再执行引出。若需调整这一关系或安排
集体效应，请显式设置 ``Order``。同一位置只要有一条命令设置了 ``Order``，
该位置所有命令都必须设置互不相同的整数。spill 监视器必须位于同束流、同位置，
并在其指定的 source 后执行。

圈数与物理时间窗口
------------------

圈号从 0 开始。``Start turn`` 包含起点，``End turn`` 不包含终点；
省略终点表示在本次运行内不另设上限。可选时间窗口同样为左闭右开，
对每个候选粒子的物理到达时刻应用：

.. math::

   t_i=t_0-\frac{z_i}{\beta_0 c}.

同时配置圈数与时间窗口时，粒子必须同时满足两个条件。时间单位为秒，沿用现有束团
参考时钟，允许有限的负时间边界。不折叠存储的连续 ``z``，也不加入名义束团分组中心。
不使用固定回旋频率把圈数换算为时间。详见 :ref:`zh-longitudinal-reference`。

接口参数
--------

.. list-table::
   :header-rows: 1
   :widths: 20 25 20 35

   * - Python 参数
     - JSON 字段
     - 类型 / 默认值
     - 含义
   * - ``command``
     - ``Command``
     - ``"SlowExtraction"``
     - 命令类型。
   * - ``s``
     - ``S (m)``
     - 有限浮点数，必填
     - 已有跟踪截面，非负且位于环内。
   * - ``order``
     - ``Order``
     - 严格整数或 null；null
     - 显式同位置顺序；未设时使用优先级 850。
   * - ``position``
     - ``Position (m)``
     - 有限浮点数，必填
     - 横向切线位置 :math:`u_{\rm cut}`。
   * - ``side``
     - ``Side``
     - ``"positive"``
     - ``"positive"`` 或 ``"negative"``，使用严格不等式。
   * - ``tilt``
     - ``Tilt (rad)``
     - 有限浮点数；0
     - 横向滚转角，不是纵向偏航角。
   * - ``start_turn``
     - ``Start turn``
     - 严格整数；0
     - 包含的非负起始圈。
   * - ``end_turn``
     - ``End turn``
     - 严格整数或 null；null
     - 不包含的终止圈；设置时必须大于起始圈。
   * - ``start_time``
     - ``Start time (s)``
     - 有限浮点数或 null；null
     - 包含的粒子到达时间下界。
   * - ``end_time``
     - ``End time (s)``
     - 有限浮点数或 null；null
     - 不包含的时间上界；两者均设置时必须大于下界。
   * - ``buffer_size``
     - ``Buffer size (particles)``
     - 严格正整数；65536
     - 按事件行数设置的写盘阈值，不是粒子抽样上限。
   * - ``output_format``
     - ``Output format``
     - ``"hdf5-gzip1"``
     - ``"hdf5"`` 或无损压缩 ``"hdf5-gzip1"``，不支持 TFS。

配置示例
--------

以下命令添加到已经把束流输运至 ``s=12.5`` m 的现有序列中。
数值仅演示配置接口，不构成某台机器的引出设计。

.. code-block:: python

   from PASS.para.schema import SlowExtractionItem, SlowExtractionMonitorItem

   seq.add("extract", SlowExtractionItem(
       s=12.5, position=0.035, side="positive",
       start_turn=100, end_turn=10000,
   ))
   seq.add("spill", SlowExtractionMonitorItem(
       s=12.5, source="extract", bin_by="both",
       turn_bin_width=10, time_bin_width=1e-3,
   ))

监视器可独立设置额外的圈数和时间窗口。这些窗口只改变统计选择，不改变引出动作。

事件输出
--------

source 在运行输出目录下写入 ``distribution/slow_extraction/*_events.h5``。
flat-output 模式省略 ``slow_extraction`` 子目录。文件名包含运行 UUID、束流编号和
source 专用标识。各列是文件根节点下的一维 dataset，使用标准
:doc:`monitor/table_output` 格式。

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - 列名
     - 含义 / 单位
   * - ``particle_id``、``beam_id``、``bunch_id``
     - 收集时的身份；``particle_id`` 是原始正 tag。
   * - ``turn``、``s``、``time``
     - 收集圈号、截面位置（m）、物理到达时刻（s）。
   * - ``x``、``px``、``y``、``py``、``z``、``dp``
     - 收集时的 PASS 坐标：位置为 m；``px=Px/P0``、``py=Py/P0``；
       ``z`` 为连续相对时间坐标；``dp=(P-P0)/P0``。
   * - ``reference_time``、``reference_beta``、``reference_momentum``
     - 收集时的参考量：s、无量纲 beta、束团约定下的 eV/c（离子按每核子）。
   * - ``macro_weight``
     - 该宏粒子代表的真实粒子数。
   * - ``charge_number``、``proton_number``、``neutron_number``
     - 带符号电荷数与粒子组成。
   * - ``rest_energy``
     - 束团约定下的静止能量，单位 eV（离子按每核子）。

坐标保留原粒子精度；时间、参考量和权重使用 float64。即使后续存活束团参考量改变，
已保存事件的坐标和参考量仍对应收集瞬间。物理斜率应由
:math:`x'=p_x/\sqrt{(1+\delta)^2-p_x^2-p_y^2}` 计算，不能直接把归一化动量 ``px`` 当成角度。

所有收集事件都会保留，不进行抽样。完整批次可能使缓冲行数超过阈值；达到阈值时追加写盘，
finalize 时写出剩余行。没有引出事件时，finalize 也会生成具有明确列类型的空表。
内存快照先于环内停止标记，但缓冲区不保证进程崩溃后的数据持久性，当前也没有随断点恢复的
引出事件账本。输出错误会停止执行，不会静默丢弃事件或自动重试状态不明确的追加写入。

捕获或环内停止标记过程出错后，source 同样禁止继续执行。只有捕获与停止标记均成功，
才向 monitor 发布该批事件。只要输出本身没有失败，finalize 会保留缓冲的捕获记录，
包括停止标记过程中断的诊断证据。磁盘与粒子内存更新之间不保证原子事务，
因此，中止运行的表格应作为诊断数据，不能视为完整引出账本。

事件表头 ``SourceStatus`` 初始为 ``open``，finalize 时改为 ``finalized`` 或
``aborted-diagnostic``。``finalized`` 仅描述该 source，不证明已完成所有请求的仿真圈数。
中止表可能包含尚未完成环内停止标记的捕获记录。``SuccessfulExtracted``、
``SuccessfulRealExtracted``、``SuccessfulBatchSerial``、``LastSuccessfulTurn``
记录成功发布的 source 状态。写盘失败可能留下 ``open`` 或不完整文件，不能当作已完成账本。
