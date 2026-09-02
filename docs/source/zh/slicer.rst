切片器（Slicer）
=================

``Slicer`` 命令把每个束团中的存活宏粒子分配到纵向切片，并将结果写入束团拥有的命名 ``SliceSet``。切片是局部分类操作：不会重排任何粒子数组，也不会改变束团归属。

职责边界
--------

``SortBunch`` 和 ``ReorganizeBunch`` 负责全局bucket分组，并且可能重排全部粒子数组。由于粒子范围可能改变，它们会使已有的 ``SliceSet`` 全部失效。随后由 ``Slicer`` 在序列指定位置重新计算目标集合。空间电荷和束束效应等需要切片的效应可以引用不同名称的集合，因此可以使用不同切片网格。

粒子坐标采用连续的束团相对坐标 :math:`z_{rel}`。切片过程中不对坐标进行环折叠，也不使用实验室坐标 :math:`z_{lab}`。

配置格式
--------

输入命令名称为 ``Slicer``，并通过用户定义的 ``slice set`` 选择集合：

.. code-block:: json

   {
       "sc_slicer": {
           "S (m)": 12.5,
           "Command": "Slicer",
           "Slice set": "space_charge",
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

所有范围都在命令执行时解析。对于显式范围之外的粒子，小于下限的粒子进入第一个切片，大于上限的粒子进入最后一个切片，并记录 warning；不会静默丢弃。

切片模型
--------

切片 ID 按 ``z_rel`` 从大到小排列：切片 ``0`` 是 ``z`` 最大的区间，切片
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
    ``z_rel`` 坐标中的边界和中心，按从大 ``z`` 到小 ``z`` 排列（切片 ``0``
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
     - ``z_rel`` 坐标下限，必须小于 ``Z max``。
   * - ``z_max``
     - ``"Z max"``
     - float
     - 显式模式必填
     - ``z_rel`` 坐标上限，必须大于 ``Z min``。

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
