分布监视器（DistMonitor）
=========================

``DistMonitor`` 在指定位置、指定圈数保存完整的束团粒子分布，包含尚未丢失和已经丢失的所有粒子。

- **代码位置**：``PASS/commands/monitor/distribution.py``
- **注册命令**：``"distmonitor"``

圈数选择
--------

``Save turns`` 是由一个元素或三个元素组成的列表。单元素列表选择一个
从 0 开始的圈数；三元素列表 ``[start, end, step]`` 选择
``start, start + step, ...``，其中 ``end`` 为包含端点。可以给出多个
列表，重叠的选择会自动合并。

例如：

.. code-block:: json

   "distribution_1": {
       "S (m)": 12.5,
       "Command": "DistMonitor",
       "Save turns": [[0], [100, 200, 10], [500, 1000, 100]]
   }

``"Save turns": []`` 表示关闭保存，``step`` 必须为正数。如果范围超出
模拟圈数，超过最后一圈的 ``end`` 会裁剪为 ``num_turns - 1`` 并给出 warning；
``start >= num_turns`` 的范围会给出 warning 后忽略；负的 ``start`` 会裁剪为
0。范围格式错误、``step`` 非正数以及 ``end < start`` 仍视为配置错误。监视器
在初始化时将选择编译为字节表，运行时判断当前圈数只需一次边界检查和数组查找。

接口参数
--------

.. list-table::
   :header-rows: 1
   :widths: 20 22 12 10 36

   * - Python 字段
     - JSON key
     - 类型
     - 默认值
     - 说明
   * - ``s``
     - ``"S (m)"``
     - float
     - 必填
     - 监视器在束线中的纵向位置。
   * - ``command``
     - ``"Command"``
     - str
     - ``"DistMonitor"``
     - 使用 ``"DistMonitor"``；运行时不区分大小写。
   * - ``save_turns``
     - ``"Save turns"``
     - list[list[int]]
     - ``[]``
     - 单圈 ``[turn]`` 或包含端点的范围 ``[start, end, step]``。
   * - ``include_injection_metadata``
     - ``"Include injection metadata"``
     - bool
     - ``false``
     - 追加 ``particle_id``、``injection_turn`` 和 ``injection_batch``；保存时要求束流具有 Injection 状态。
   * - ``output_format``
     - ``"Output format"``
     - str
     - ``"tfs"``
     - 可选 ``"tfs"``（文本，``.tfs``）或 ``"hdf5"``（压缩数据集，``.h5``）。

序列键名会作为监视器名称。通过高层 API 可以直接使用 schema 对象：

.. code-block:: python

   from PASS.para.schema.monitors import DistMonitor

   monitor = DistMonitor(s=12.5, save_turns=[[0], [100, 200, 10]])

保存带注入信息的 HDF5 快照：

.. code-block:: python

   injection_monitor = DistMonitor(
       s=0.0,
       save_turns=[[0], [10, 100, 10]],
       include_injection_metadata=True,
       output_format="hdf5",
   )

这两个选项在生成的 JSON 中分别为 ``"Include injection metadata": true``
和 ``"Output format": "hdf5"``。

输出内容
--------

每个选中的圈数、每个束团生成一个 TFS 文件（或指定的 HDF5 文件）。文件名
包含运行时间、束流和束团编号、监视器位置、名称及圈数。全部已注入粒子都会
写出，包括损失粒子；尚未注入的 ``tag=0`` 预留位置不写出。

默认的九个数据列如下：

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - 列名
     - 单位
     - 说明
   * - ``x``、``px``、``y``、``py``
     - m 或归一化动量
     - 横向相空间坐标。
   * - ``z``
     - m
     - 跟踪使用的束团相对坐标 ``z_rel``。
   * - ``dp``
     - -
     - 相对动量偏差。
   * - ``tag``
     - -
     - 粒子标识；正值表示存活，负值表示丢失。
   * - ``lost_turn``
     - -
     - 粒子丢失圈数（未丢失为 ``-1``）。
   * - ``lost_position``
     - m
     - 丢失位置（未丢失为 ``-1``）。

TFS 文件头包含 ``S``、command 和监视器名称、束流/束团编号、``Turn``、
粒子计数、后端和精度、PASS 版本、时间，以及 ``ZCoordinate``、``ZCenter``、
``Circumference`` 等信息。保存时不会折叠或平移 ``z``；存活粒子的通过时刻由
``t = ReferenceArrivalTime - z / (ReferenceBeta*c)`` 恢复。
``ZCenter`` 仅为分组元数据，不用于恢复物理通过时刻。

CPU 与 GPU
----------

CPU 直接从 NumPy 粒子数组写出。GPU 在选中圈数将九个跟踪字段
复制到主机内存，再由指定格式的写入器输出。监视器不会跨圈保留历史 buffer，因此
内存开销与单个粒子快照成正比，而不是与总圈数快照成正比。

注入快照
--------

``Include injection metadata`` 默认 ``false``。开启后追加三个整数列，
两种输出格式均包含十二列：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 列名
     - 说明
   * - ``particle_id``
     - 束流内的粒子身份，等于 ``abs(tag)``。
   * - ``injection_turn``
     - 粒子实际注入时的模拟圈数，从 0 开始。
   * - ``injection_batch``
     - 原始注入源束团内的批次编号，从 0 开始；重新分组不会重新编号。

主机端根据 ``abs(tag)`` 和束流的 Injection 批次记录生成这些列，
``ParticlePool`` 不保存对应数组。若缺少 Injection 状态，开启该选项保存时
会抛出 ``ValueError``。若状态存在，但粒子 ID 没有匹配的批次记录，
``injection_turn`` 和 ``injection_batch`` 为 ``-1``，表示未知。
排序或损失不改变已出生粒子的身份和注入事件。不输出 ``tag=0`` 的预留位置，
其数量记录为 ``NumPending``。

``Output format`` 支持默认的 ``"tfs"`` 和 ``"hdf5"``。HDF5 使用 gzip
压缩数据集保存相同数据列，用文件属性保存头信息。可选注入列不需要额外复制
设备上的逐粒子数组。
