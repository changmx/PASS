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
   :widths: 20 20 12 48

   * - Python 字段
     - JSON key
     - 类型
     - 说明
   * - ``s``
     - ``"S (m)"``
     - float
     - 监视器在束线中的纵向位置。
   * - ``command``
     - ``"Command"``
     - str
     - 使用 ``"DistMonitor"``；运行时不区分大小写。
   * - ``save_turns``
     - ``"Save turns"``
     - list[list[int]]
     - 单圈 ``[turn]`` 或包含端点的范围 ``[start, end, step]``。

序列键名会作为监视器名称。通过高层 API 可以直接使用 schema 对象：

.. code-block:: python

   from PASS.para.schema.monitors import DistMonitor

   monitor = DistMonitor(s=12.5, save_turns=[[0], [100, 200, 10]])

输出内容
--------

每个选中的圈数、每个束团生成一个 TFS 文件。文件名包含运行时间、束流和
束团编号、监视器位置、名称及圈数。束团中的全部粒子都会写出，不根据
``tag`` 符号过滤。

数据列如下：

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
``Circumference`` 等信息。保存时不会折叠或平移 ``z``；恢复实验室坐标时
应结合 ``ZCenter`` 使用。

CPU 与 GPU
----------

CPU 直接从 NumPy 粒子数组写出。GPU 在选中圈数将输出所需的九个字段复制
到主机内存，再由主机端 TFS 写入。监视器不会跨圈保留历史 buffer，因此
内存开销与单个粒子快照成正比，而不是与总圈数快照成正比。
