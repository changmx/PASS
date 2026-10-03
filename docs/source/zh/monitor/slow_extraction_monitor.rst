慢引出 spill 监视器
===================

``SlowExtractionMonitor`` 读取指定 :doc:`../slow_extraction` 命令产生的事件，
按跟踪圈数、物理到达时间或两者同时累计 spill 直方图。它不从环内选择粒子，
也不改变粒子状态。统计输入是 source 事件行，而不是反复扫描负 tag。
对同一 source 批次重复执行时，不会重复计数。

``Source`` 必须精确匹配同束流 Sequence 中一个 ``SlowExtraction`` 的名称。
监视器必须与 source 位于相同 ``S (m)``，并在 source 后执行。
默认优先级 850、860 满足这一顺序；同位置使用显式 ``Order`` 时，
应为该处每条命令设置互不相同的值。
监视器必须从 source 首个批次开始消费每次调用，包括统计窗口之外的圈。
遗漏批次会报错；监视器不会从 source 文件补读漏掉的事件。

窗口与分箱定义
--------------

圈数窗口和物理时间窗口均可独立配置，同时设置时取交集；两个窗口均为左闭右开。
监视器窗口只筛选 source 已经收集的事件，不改变 source 的引出条件，
也无法恢复 source 未曾收集的粒子。

``Bin by="turn"`` 按 ``Turn bin width`` 指定的圈数宽度分箱，以 ``Start turn``
为分箱起点。``Bin by="time"`` 按 ``Time bin width (s)`` 指定的秒数宽度分箱，
以 ``Time origin (s)`` 为时间网格原点。``"both"`` 分别输出两类直方图。
时间分箱始终采用 source 捕获的粒子到达时刻：

.. math::

   t_i=t_{0,i}-\frac{z_i}{\beta_{0,i}c}.

不使用固定回旋频率换算。同一圈的粒子可以落入不同时间 bin，较晚跟踪圈产生的事件
也可能更新更早的时间 bin。因此，运行中的时间直方图始终是可更新状态，
写出一次快照不代表关闭其时间 bin。

名义边界 ``bin_start``、``bin_end`` 在不同快照之间保持固定。
时间边界按 float64 的 ``origin + k * width`` 计算；事件与这些实际保存的边界
进行严格左闭右开比较，不用容差将相邻时刻吸附到边界。若 bin 宽度过小，
无法在观测时刻分辨相邻边界，则报错。
``observed_start``、``observed_end``、``observed_width`` 单独表示当前观测所覆盖的部分。
圈数覆盖范围是监视器圈数窗口内实际执行的圈号区间。时间覆盖范围是 source 参考时刻
与所接受事件时刻的包络，再截取到监视器时间窗口。该包络不证明其中所有粒子到达事件
都已完整结束。

观测范围内会补齐零计数 bin，但不会把它们外推至整个请求的运行区间。
``is_partial`` 标识覆盖不足一个名义 bin 的区间。若最新事件恰好位于时间 bin 边界，
可以暂时保留宽度为零的 partial bin，直到后续观测扩展覆盖范围。
比较完整 bin 的 spill 均匀性时，应排除 partial bin，并在报告纹波指标时保留所用分箱宽度。

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
     - ``"SlowExtractionMonitor"``
     - 只读 spill 监视器类型。
   * - ``s``
     - ``S (m)``
     - 有限非负浮点数，必填
     - 必须与 source 位于同一跟踪截面。
   * - ``order``
     - ``Order``
     - 严格整数或 null；null
     - 同位置顺序，默认优先级 860。
   * - ``source``
     - ``Source``
     - 非空字符串，必填
     - 本束流中 ``SlowExtraction`` 的精确 Sequence 名称。
   * - ``bin_by``
     - ``Bin by``
     - ``"both"``
     - ``"turn"``、``"time"`` 或 ``"both"``。
   * - ``turn_bin_width``
     - ``Turn bin width``
     - 严格正整数；1
     - 名义圈数 bin 宽度，以 ``Start turn`` 为原点。
   * - ``time_bin_width``
     - ``Time bin width (s)``
     - 有限正浮点数；0.001
     - 名义物理时间 bin 宽度。
   * - ``time_origin``
     - ``Time origin (s)``
     - 有限浮点数；0
     - 时间分箱网格原点，允许负 bin 索引。
   * - ``start_turn``
     - ``Start turn``
     - 严格非负整数；0
     - 包含的统计起始圈。
   * - ``end_turn``
     - ``End turn``
     - 严格整数或 null；null
     - 不包含的终止圈；设置时必须大于起始圈。
   * - ``start_time``
     - ``Start time (s)``
     - 有限浮点数或 null；null
     - 包含的事件到达时间下界。
   * - ``end_time``
     - ``End time (s)``
     - 有限浮点数或 null；null
     - 不包含的时间上界；两者均设置时必须大于下界。
   * - ``write_interval_turns``
     - ``Write interval (turns)``
     - 严格正整数；100
     - 输出写盘间隔；finalize 时写出尚未发布的更新。
   * - ``output_format``
     - ``Output format``
     - ``"hdf5-gzip1"``
     - ``"hdf5"`` 或无损压缩 ``"hdf5-gzip1"``，不支持 TFS。

配置示例
--------

以下片段假定同位置已经存在名为 ``extract`` 的引出动作。

.. code-block:: python

   from PASS.para.schema import SlowExtractionMonitorItem

   seq.add("spill", SlowExtractionMonitorItem(
       s=12.5, source="extract", bin_by="both",
       start_turn=100, end_turn=10000,
       start_time=0.10, end_time=1.20,
       turn_bin_width=10, time_bin_width=1e-3, time_origin=0.0,
       write_interval_turns=100,
   ))

只有同时满足圈数和时间边界的 source 事件才参与计数。
高层 API 的 monitors 列表也接受 ``type="SlowExtractionMonitor"``；
引出动作应单独使用 ``SlowExtractionItem``。

输出表
------

表格写在运行输出目录下的 ``slow_extraction/``；flat-output 模式省略
``slow_extraction`` 子目录。文件名由 source 和 monitor 名称、摘要、运行标识及束流编号组成，
``_turn.h5``、``_time.h5`` 后缀分别标识两类直方图。完整圈数 bin 在每次写盘时仅追加一次，
当前未完成的 partial 圈数 bin 仅在 finalize 时追加；内存中的 ``get_histogram("turn")``
也包含当前 partial bin。时间表在同一路径发布完整累计快照，允许后续事件更新早期 bin。
source 的逐粒子事件表单独保存。

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 列名
     - 含义
   * - ``bin_start``、``bin_end``
     - 固定名义边界，单位为圈或秒，左闭右开。
   * - ``observed_start``、``observed_end``、``observed_width``
     - 相同单位下的已观测覆盖范围，宽度非负。
   * - ``is_partial``
     - 是否尚未覆盖完整名义 bin 的布尔标记。
   * - ``num_extracted``
     - bin 内的宏粒子事件数。
   * - ``real_extracted``
     - ``macro_weight`` 之和，即代表的真实粒子数。
   * - ``charge_extracted``
     - 带符号的 ``charge_number * e * macro_weight`` 之和，单位 C。
   * - ``cumulative_extracted``、``cumulative_real``
     - 按 bin 递增顺序累计的宏粒子数与真实粒子数。
   * - ``particle_rate``
     - 仅时间表：``real_extracted / observed_width``，单位粒子/s。
   * - ``current``
     - 仅时间表：``charge_extracted / observed_width``，单位 A，保留电荷符号。

当 ``observed_width`` 为零时，速率列写 NaN。速率的分母为已观测宽度，不假定完整 bin
都已被覆盖。partial bin 的速率不应直接参与完整 bin 的纹波统计比较。
headers 保存 source 事件文件身份、时间定义、窗口以及参考/事件观测包络。

CPU 与 GPU 路径均消费相同的独立主机事件批次，监视器不逐圈重读事件文件。
默认每 100 圈写一次输出，并在运行结束或 finalize 时写出。
正常处理中断时使用命令统一的 finalize 流程；进程突然结束可能丢失仍在内存中的更新。
圈数表追加失败可能留下不完整表格；执行会停止，不会重试该次追加。时间快照仅替换此前完整
发布的快照。当前实现没有恢复直方图状态与 source 事件历史的断点账本。

可按 :doc:`table_output` 使用 ``PASS.utils.table_io.read_table`` 读取。
需要六维分布、逐粒子到达时间或其他离线分箱时，使用独立的 source 事件表。
