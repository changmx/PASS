切片器（Slicer）
=================

``Slicer`` 命令把每个束团中的存活宏粒子分配到纵向切片，并将结果写入束团拥有的命名 ``SliceSet``。切片是局部分类操作：不会重排任何粒子数组，也不会改变束团归属。

CUDA 对固定范围的 ``equal_length`` 网格批量执行各束团的直方图和切片表计算，
共三次核启动，随后统一传回存活数与越界数。切片编号、诊断表、边界归属及显式
更新时间与逐束团路径定义一致。其他切片模型和自动范围保留原有执行路径。
等长网格只分配直方图与几何工作区，排序缓冲区仅为等粒子数网格分配。

职责边界
--------

``SortBunch`` 和 ``ReorganizeBunch`` 负责全局bucket分组，并且可能重排全部粒子数组。由于粒子范围可能改变，它们会使已有的 ``SliceSet`` 全部失效。随后由 ``Slicer`` 在序列指定位置重新计算目标集合。空间电荷和束束效应等需要切片的效应可以引用不同名称的集合，因此可以使用不同切片网格。

.. important::

   **每次实际执行 SortBunch 或 ReorganizeBunch 后，都必须重新执行 Slicer，
   才能继续使用切片信息。** 排序或重分组会清空原有 ``slice_id`` 和
   ``slice_table``。后续命令需要使用的每个命名 ``SliceSet`` 都必须重新计算，
   即使同一圈内此前已经执行过 Slicer。

   实际执行顺序示例（每个 Slicer 更新后续命令所引用的切片集）：

   .. code-block:: text

      正确：SortBunch       -> Slicer -> SpaceCharge
      正确：ReorganizeBunch -> Slicer -> WakeField
      错误：Slicer -> SortBunch -> SpaceCharge

   最后一种顺序需要在 SortBunch 与 SpaceCharge 之间再执行一次 Slicer。
   使用切片的命令不会自动重建失效结果。此规则对 CPU 和 CUDA 均适用。
   当现有粒子顺序和束团归属没有改变时，Slicer 本身并不要求先执行 SortBunch。

切片映射按当前数组位置对齐：全局索引为 ``i`` 的粒子，其切片编号是
``slice_id[i - bunch.start_idx]``。``tag`` 在排序时随粒子一起移动，但不是
``slice_id`` 的数组索引；因此，保留相同的 tag 并不能让排序前的切片映射继续
有效。分组操作详见 :doc:`reorganize`。

粒子坐标采用连续的束团相对坐标 :math:`z_{rel}`。``z_rel`` 切片直接使用该坐标；``z_periodic`` 和 ``arrival_phase`` 使用临时投影，三种模式都不会替换粒子存储坐标。

切片坐标与复用
--------------

默认 ``Coordinate=z_rel`` 直接按连续时间坐标
:math:`z=\beta_b c(T_b-t_i)` 分箱，用于连续尾场时间。
SpaceCharge 必须使用下文的独立选项 ``z_periodic``。
独立选项 ``arrival_phase`` 为漂移束尾场构造临时周期到达投影；
``Periodic=true`` 是该选项的别名。不再支持 ``ring_position``，因为此坐标定义下
z 加名义槽位偏移不能给出严格的物理环位置。

最近一次显式 Slicer 结果是权威切片信息，其 z 区间、宽度和成员一直复用，直到
用户再次执行 Slicer。RF 不缩放、不平移、不重建这些数组。当前参考事件下，
``z_rel`` 局部宽度 :math:`\Delta z` 对应 :math:`\Delta t=\Delta z/(\beta_b c)`，
中心对应 :math:`T_b-z_{slice}/(\beta_b c)`。因此保存的是当前 z 坐标中的
区间，不是冻结的物理时间区间。用户负责控制复用误差。结构性重分组使旧的局部
索引失效，需要显式执行新的 Slicer。

尾场源在发出时保存采样的物理时间和宽度，后续参考变化不移动历史源。
周期到达相位切片另有共同观测窗口，该近似见 :doc:`wake_field`。
准静态 SpaceCharge 使用按环周折叠后的 z 区间；对于任意速度展宽，以速度缩放的时间坐标
不等于严格同时的三维空间分布。

``Z range mode`` 仍只有 ``auto`` 和 ``explicit``。粒子存储使用配置的 float32
或 float64，局部分箱中间量使用 float64。CPU 与 CUDA 保存的切片边界、中心、宽度和
密度表均使用 float64。粒子存储为 float32 时，CUDA 将局部工作坐标提升为 float64，
避免切片几何的舍入使等间距尾场网格变得不等间距；跟踪粒子坐标不变。
快照保存连续 z、参考时间、beta 和坐标
定义；周期快照还保存 ``slice_coordinate``。密度单位是每米真实粒子数，不是 C/m。

TFS 文件头 ``ZCoordinate="z_rel"`` 描述原始粒子 ``z`` 列，是输出元数据，
不是输入选项，也不是默认值赋值。``Coordinate`` 标识所选切片投影；
``CoordinateDefinition="z=beta*c*(T-t)"``、``ReferenceArrivalTime``（秒）和
``ReferenceBeta`` 定义原始粒子坐标。到达相位快照还保存 ``ObservationTime``
（秒）、``ObservationVelocity``（米/秒）及 ``SliceCoordinateDefinition``。
这些字段与 ``Circumference`` 一起定义保存的观测窗口，其时间参数可以与束团
参考参数不同。

配置格式
--------

等长切片的索引采用 FP64 运算，并在除法后核对相邻的已保存边界，避免将刚位于
边界下方的 FP32 坐标错分到相邻箱。CPU、CUDA 和 CUDA 批处理路径的分类与
输出表使用同一组已生成边界。按坐标递增方向，各区间左闭右开，最后一个区间
包含最大端点。这不改变粒子数组或局部投影的存储精度。

空间电荷的环周折叠切片
~~~~~~~~~~~~~~~~~~~~~~

``Coordinate=z_periodic`` 构造临时坐标：

.. math::

   z_{slice,i}=[(z_i+C/2)\bmod C]-C/2.

环周必须有限且为正，折叠不依赖规定时钟。CPU/GPU 均支持 ``equal_length``、
``equal_particle`` 以及 ``auto``、``explicit`` 范围。``auto`` 使用存活粒子折叠
坐标的最小值和最大值；``explicit`` 必须位于 :math:`[-C/2,C/2]` 内。整环网格
使用该完整范围，并保留空箱；较窄范围外的折叠坐标进入边界切片。束团跨越折叠
边界时，``auto`` 范围可能覆盖接近整圈。

SpaceCharge 只接受该坐标。已有 SC 输入必须在对应 Slicer 中显式增加
``"Coordinate": "z_periodic"``；SC 对缺少坐标元数据、``z_rel``、``arrival_phase``
均报错。这仍是逐束团的共同参考速度近似，折叠不会重建任意速度展宽下的严格
同时分布，也不会合并不同束团中的重叠粒子群。

WakeField 拒绝 ``z_periodic``，因为折叠中心不保留连续到达时间。SC 和尾场应
使用分别命名的切片集。旧 ``Periodic`` 标志仍只选择 ``arrival_phase``；
包括输出文件在内，具体模式由 ``Coordinate`` 标识。

漂移束尾场的周期到达切片
~~~~~~~~~~~~~~~~~~~~~~~~

``Coordinate=arrival_phase`` 要求 ``equal_length``、``explicit`` 及
``Explicit={"z min": -C, "z max": 0}``。每次显式更新选择共同观测事件
:math:`T_{obs}=\Psi^{-1}(n+s/C)`，观测速度为
:math:`v_{obs}=C f_{rev}(T_{obs})`。CPU/GPU 构造

.. math::

   u_i=\frac{v_{obs}}{C}(T_{obs}-t_i),\qquad
   z_{phase,i}=-C[(-u_i)\bmod1].

切片表示窗口 :math:`[T_{obs},T_{obs}+C/v_{obs})`，直到下次用户更新前保留该
观测事件。允许不同束团参考时间和速度；参与同一周期尾场的全部粒子群必须拥有
相同的已保存观测窗口与周长。切片 0 最早到达，损失粒子 ID 为 -1，存储 z 不折叠。
恰好为整数圈的相位映射到 :math:`z_{phase}=0`，即窗口起点和切片 0。被排除的
右端点的相位按周期等价映射到窗口起点，不归入最后一片，也不额外生成源通过事件。相位滑移诊断继续使用连续、
未取模的相位 :math:`u_i`。

使用尾场同位置 Slicer 和因果历史。复用周期快照不会推进其观测窗口；每个新的
物理源通过事件需要用户更新快照，这与复用局部 z 区间不同。SpaceCharge 使用
独立的 ``z_periodic`` SliceSet。每参考圈一次穿越近似与收敛要求见 :doc:`wake_field`。

``Max phase slip`` 默认 0.05 圈，范围 (0, 0.1]。Slicer 记录相邻观测之间的
连续相位变化而不拒绝诊断输出，WakeField 检查此限制。同圈重复更新使用相同的
上一圈基准，失效操作重置该诊断。它不能重建未采样的穿越，也不能证明短波长模收敛。

输入命令名称为 ``Slicer``，并通过用户定义的 ``slice set`` 选择集合：

.. code-block:: json

   {
       "sc_slicer": {
           "S (m)": 12.5,
           "Command": "Slicer",
           "Slice set": "space_charge",
           "Coordinate": "z_periodic",
           "Slice model": "equal_particle",
           "Number of slices": 128,
           "Z range mode": "auto",
           "Save turns": [[0], [100, 1000, 100]]
       }
   }

范围参数按模式分别放置，不接受平铺的 ``Z min``、``Z max`` 或 ``Number of sigma`` 字段。

``auto``
    直接使用当前存活粒子的实际最小值和最大值，是最宽的数据驱动范围，不会排除观测到的离群粒子。
``explicit``
    使用 ``Explicit`` 中给出的固定束团相对坐标范围：

    .. code-block:: json

       "Z range mode": "explicit",
       "Explicit": {"Z min": -0.30, "Z max": 0.30}

所有范围都在命令执行时解析。``z_rel`` 或 ``z_periodic`` 模式下，显式范围外的粒子
进入对应边界切片并记录 warning；不会静默丢弃。小于下限进入切片 N-1，
大于上限进入切片 0。

切片模型
--------

切片 ID 按选定的切片坐标从大到小排列：切片 ``0`` 是 ``z`` 最大的区间，切片
``N-1`` 是 ``z`` 最小的区间。

``equal_length`` 将范围划分为 ``N`` 个等宽切片：

.. math::

   i = N-1-\operatorname{clip}\left(\left\lfloor
       \frac{z-z_{min}}{\Delta z}\right\rfloor,0,N-1\right),
   \qquad \Delta z = \frac{z_{max}-z_{min}}{N}.

``equal_particle`` 只对临时的 ``z`` 数组和索引排序，按粒子秩分配切片 ID，再将 ID 写回原粒子顺序。粒子池本身不会改变。即使多个粒子具有相同坐标，秩分配仍能使各切片粒子数尽量均衡。``slice_table`` 中的几何边界使用 NumPy 分位数计算。

如果存活粒子数小于 ``N``，最多只有这些粒子数目的切片能够非空。其余切片仍然保留，粒子数为 0，并记录 warning。``effective_num_slices`` 保存 :math:`\min(N_{live},N)`；不会修改用户设置的网格大小。

SliceSet 数据
-------------

每个束团都有类似 ``bunch.slice_sets["space_charge"]`` 的映射。命令执行后，``slice_id`` 是与当前束团粒子区间对齐的整数数组；已损失粒子的 ID 为 ``-1``。``slice_table`` 包含每个切片的数组：

``z_min``、``z_max``、``z_center``
    所选切片坐标中的边界和中心，按从大 ``z`` 到小 ``z`` 排列（切片 ``0``
    是大 ``z`` 区间）。
``delta_z``
    每个切片的 ``z_max - z_min``。
``macro_count``
    存活宏粒子数。
``real_charge``
    等效真实粒子数，即 ``macro_count * bunch.ratio``，不是库仑。若需要物理电荷，还要乘以带符号的粒子电荷和元电荷。
``lind_density``
    线性真实粒子密度，即 ``real_charge / delta_z``。
``effective_num_slices``
    当前存活粒子群能够填充的切片数上限。

结果还记录 ``valid_turn`` 和 ``valid_s``。之后如果发生全局重分组，所有依赖粒子的字段都会清空；必须等待新的 ``Slicer`` 命令执行后，空间电荷或束束效应才能继续使用这些结果。

快照输出
--------

``Save turns`` 是可选的命令级参数，不属于共享 ``SliceSet`` 的配置。每项为
``[turn]`` 或 ``[start, end, step]``，两端均包含。切片器每圈仍会计算，只有被
选中的圈数才写出文件。每次保存会在 ``output/.../slice/`` 中写出同一时刻的粒子
TFS 文件（``tag``、``z``、``slice_id`` 和损失信息）与逐切片 TFS 汇总。两个文件的
header 都记录圈数、位置、Beam、Bunch、切片集合、模型和坐标约定。

超过总圈数的结束圈会截断到最后一圈并记录 warning；起始圈大于等于总圈数的范围会
被忽略并记录 warning。负起始圈也会截断，同时保持原始 ``start + k*step`` 的选圈
序列。``end < start``、非整数值和非正步长仍属于配置错误。

接口参数
--------

以下表格列出 ``Slicer`` 序列命令支持的参数。表格中的 JSON 键名采用输入文件的写法；PASS 内部会统一转换键名大小写。

通用命令参数
~~~~~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - 参数名
     - 键名
     - 类型
     - 默认值 / 是否必填
     - 说明
   * - ``command``
     - ``"Command"``
     - str
     - 必填
     - 必须为 ``"Slicer"``，用于选择 Slicer 命令实现。
   * - ``s``
     - ``"S (m)"``
     - float
     - 必填
     - 更新切片数据时的环上纵向位置。
   * - ``name``
     - 序列对象键名
     - str
     - 序列键必填
     - 当前命令实例名称。序列加载器会将该名称传给命令用于诊断。
   * - ``slice_set``
     - ``"Slice set"``
     - str
     - 必填
     - 要更新的束团级 ``SliceSet`` 名称，例如 ``"space_charge"`` 或 ``"beambeam_ip1"``。
   * - ``slice_model``
     - ``"Slice model"``
     - str
     - ``"equal_length"``
     - 粒子到切片的分配模型，支持 ``"equal_length"`` 和 ``"equal_particle"``。
   * - ``coordinate``
     - ``"Coordinate"``
     - str
     - ``"z_rel"``
     - ``z_rel`` 为连续时间切片；``z_periodic`` 为 SC 环周折叠切片；
       ``arrival_phase`` 使用尾场观察时钟。SC 必须显式选择 ``z_periodic``。
   * - ``num_slices``
     - ``"Number of slices"``
     - int
     - 10
     - 纵向切片数，必须至少为 1。当存活粒子数更少时，该配置值仍然保留。
   * - ``z_range_mode``
     - ``"Z range mode"``
     - str
     - ``"auto"``
     - 选择范围模式：``"auto"`` 或 ``"explicit"``。
   * - ``save_turns``
     - ``"Save turns"``
     - 整数列表的列表
     - ``[]``
     - 可选快照圈数：``[turn]`` 或 ``[start, end, step]``。
   * - ``periodic``
     - ``"Periodic"``
     - bool
     - false
     - 漂移束 WakeField 的全环到达相位投影，要求 equal_length、explicit [-C,0]。
   * - ``max_phase_slip``
     - ``"Max phase slip"``
     - float
     - 0.05
     - WakeField 允许的每参考圈最大已观测相位变化，以圈为单位，范围 (0, 0.1]；诊断切片只记录，不拒绝。

范围模式参数
~~~~~~~~~~~~

``auto`` 不需要模式专属参数块；``explicit`` 必须提供 ``Explicit`` 块。

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - 模式 / 参数名
     - 键名
     - 类型
     - 默认值 / 是否必填
     - 说明
   * - ``auto``
     - ``"Z range mode"``
     - str
     - 可选
     - 使用当前存活粒子的实际最小值和最大值，包含观测到的离群粒子。
   * - ``explicit``
     - ``"Z range mode"``
     - str
     - 可选
     - 使用固定范围；该模式必须同时提供 ``Explicit``。
   * - ``explicit`` 块
     - ``"Explicit"``
     - object
     - 显式模式必填
     - 包含 ``Z min`` 和 ``Z max`` 的模式专属对象。
   * - ``z_min``
     - ``"Z min"``
     - float
     - 显式模式必填
     - 所选切片坐标下限，必须小于 ``Z max``。
   * - ``z_max``
     - ``"Z max"``
     - float
     - 显式模式必填
     - 所选切片坐标上限，必须大于 ``Z min``。

两种范围模式的配置示例：

.. code-block:: json

   {"Z range mode": "auto"}

   {
       "Z range mode": "explicit",
       "Explicit": {"Z min": -0.30, "Z max": 0.30}
   }

SliceSet 运行时接口
~~~~~~~~~~~~~~~~~~~

命令配置会在初始化阶段转换为每个束团独立的 ``SliceSet``。以下字段可供空间电荷、束束效应及诊断模块读取；它们是 Slicer 的输出，不是额外的 JSON 输入参数。

.. list-table::
   :widths: 23 18 15 44
   :header-rows: 1

   * - 字段
     - 类型
     - 切片前是否有效
     - 说明
   * - ``name``
     - str
     - 是
     - ``Slice set`` 指定的用户名称。
   * - ``model``
     - str
     - 是
     - 规范化后的切片模型名称。
   * - ``num_slices``
     - int
     - 是
     - 配置的网格切片数。
   * - ``z_range_mode``
     - str
     - 是
     - 规范化后的范围模式。
   * - ``explicit``
     - ``ExplicitRange`` 或 None
     - 是
     - 显式模式下的规范化范围边界。
   * - ``slice_id``
     - int 数组
     - 否
     - 与当前束团粒子区间对应的切片 ID；已损失粒子为 ``-1``。
   * - ``slice_table``
     - 数组字典
     - 否
     - 每个切片的几何和粒子统计数组，详见前文 SliceSet 数据章节。
   * - ``valid_turn``
     - int 或 None
     - 否
     - 生成该结果时的模拟圈数。
   * - ``valid_s``
     - float 或 None
     - 否
     - 生成该结果时的序列位置。

配置校验
~~~~~~~~

在束流初始化阶段，引用同一个 ``Slice set`` 的多个 ``Slicer`` 命令必须具有完全一致的 ``Slice model``、``Number of slices``、范围模式及（显式模式下的）显式参数块。配置冲突会抛出 ``ValueError``，并指出两个相关的序列命令。执行时，如果显式范围不能覆盖全部存活粒子，会记录 warning；超出范围的粒子会被限制到第一个或最后一个切片。
