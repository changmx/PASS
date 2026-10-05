粒子监视器（ParticleMonitor）
============================================

``ParticleMonitor`` 在给定圈数区间内逐圈记录所选粒子的六维相空间坐标与损失信息，适用于单粒子轨迹及频谱分析。重建物理到达时间或动量时，应启用参考量输出。

配置示例
------------

.. code-block:: python

   from PASS.para.schema.monitors import ParticleMonitorItem
   from PASS.para.schema.sequence import Sequence

   sequence = Sequence()
   sequence.add("particle1", ParticleMonitorItem(
       s=0.0, max_tag=5, start_turn=0, end_turn=64,
       include_reference=True,
   ))

本例记录绝对 tag 值为 1–5 的粒子及每行对应的参考量。使用时应将监视器加入包含这些粒子的完整序列。

接口参数
--------

.. list-table::
  :header-rows: 1
  :widths: 20 20 10 10 40

  * - Python 字段
    - JSON key
    - 类型
    - 默认值
    - 说明
  * - ``s``
    - ``"S (m)"``
    - float
    - 必填
    - 监视器在束线中的纵向位置
  * - ``command``
    - ``"Command"``
    - str
    - ``"ParticleMonitor"``
    - 命令类型标识
  * - ``max_tag``
    - ``"Max tag"``
    - int
    - 必填
    - 记录粒子的最大 tag 值，需 :math:`\geq 1`
  * - ``start_turn``
    - ``"Start turn"``
    - int
    - 0
    - 记录起始圈（含， 0-based ）
  * - ``end_turn``
    - ``"End turn"``
    - int
    - -1
    - 记录结束圈（不含， -1 表示至最后一圈含）
  * - ``include_reference``
    - ``"Include reference"``
    - bool
    - false
    - 逐行追加参考时间、beta 和动量，用于物理时间/能量分析

``output_format``（JSON 键 ``Output format``）默认为不压缩的 ``hdf5``，也可选择 ``hdf5-gzip1`` 或 ``tfs``。命令名称由序列键给定。

粒子选择机制
------------

PASS 中每个粒子拥有全局唯一的 ``tag`` （正整数），插入的测试粒子从 ``tag = 1`` 开始递增。 ``ParticleMonitor`` 通过 ``max_tag`` 参数指定记录范围：

.. math::

   \text{recorded} = \{\, i \;\mid\; 1 \leq |\mathrm{tag}_i| \leq \mathrm{max\_tag} \,\}

注意匹配条件使用的是 :math:`|\mathrm{tag}|` （绝对值），因此：

- ``tag = 1, 2, \ldots, \mathrm{max\_tag}`` ：正常存活粒子
- ``tag`` 取负 ：已丢失粒子 **同样被记录** ，其坐标保持丢失前的最后值

tag 标识在排序及损失前后保持不变。``max_tag`` 是 tag 上界，并非手动插入粒子数或单个束团的粒子数。非正的 max_tag 不记录粒子，并输出警告。

记录圈数范围
------------

通过 ``start_turn`` 和 ``end_turn`` 可指定记录的圈数范围：

.. math::

   \text{recorded turns} = \{\, n \;\mid\; \mathrm{start\_turn} \leq n < \mathrm{end\_turn} \,\}

- ``start_turn`` ：记录起始圈（含），默认 0
- ``end_turn`` ：记录结束圈（不含），默认 -1 表示最后一圈（含）

计划记录区间完整执行后，记录圈数为：

.. math::

   N_{\mathrm{record}} = \mathrm{end\_turn} - \mathrm{start\_turn}

典型用途：前 200 圈让束流稳定（不记录），从第 200 圈开始记录 1000 圈用于 FFT 分析。


输出文件
--------

每个监视器、每个束流输出 **一个文件**，包含所选粒子的全部记录圈。
通过 ``output_format`` 选择 ``hdf5-gzip1``、``hdf5`` 或 ``tfs``，无需选择布局。

* 文件名：``{hms}_beam{bid}_{monitor_name}_s{s:.3f}_particles.h5``；
  TFS 使用相同文件名主体和 ``.tfs`` 扩展名。
* 输出目录：``output_dir_particle``。
* HDF5 根属性或 TFS 文件头包含 ``Name="PASS Particle Monitor"``、
  ``Layout="single_file"``、``FormatVersion=2``、监测位置、束流编号、
  粒子 tag 上界和计划记录区间。

PM 读取器要求当前第 2 版格式。输出配置仅包含格式与缓冲设置，不设布局字段。

写入间隔只改变缓冲方式，**仍然每圈采样**。
例如 ``write_interval_turns=128`` 将 128 个记录圈一起追加到 HDF5 后复用缓冲区。
TFS 输出在追踪期间使用私有 HDF5 文件，结束清理时导出完整文本表。
记录结束或正常清理时会补写最后不足一块的数据，并非每 128 圈只保存一次采样。

.. code-block:: python

   monitor = ParticleMonitorItem(
       s=100.0, max_tag=10000, output_format="hdf5",
       write_interval_turns=128,
   )

``write_interval_turns``（JSON ``Write interval (turns)``）为正的严格整数，
默认 128，对 HDF5 和 TFS 均有效。不压缩的 HDF5 避免文本格式化和压缩计算，
并支持选择读取部分数组。Gzip1 能减少磁盘数据量，最快设置取决于存储吞吐量
与数据的可压缩程度。TFS 适合文本交换，大规模扫描需要承担文本格式化与解析成本。

记录量
------

历史数据包含以下 11 个量；HDF5 每次采样仅保存一个 ``turn``，
其他量按采样和粒子保存。TFS 还增加明确的记录类型与粒子 ID 列，见后文。


.. list-table::
  :header-rows: 1
  :widths: 20 15 65

  * - 列名
    - 单位
    - 说明
  * - ``turn``
    - -
    - 实际圈数（ :math:`\mathrm{start\_turn}` 至 :math:`\mathrm{end\_turn}-1` ）
  * - ``x``
    - m
    - 水平位置
  * - ``px``
    - -
    - 归一化水平动量
  * - ``y``
    - m
    - 垂直位置
  * - ``py``
    - -
    - 归一化垂直动量
  * - ``z``
    - m
    - 相对所属束团参考到达时间的纵向坐标 :math:`z_{\mathrm{rel}}`
  * - ``dp``
    - -
    - 相对动量偏差 :math:`\delta`
  * - ``tag``
    - -
    - 粒子标签（正=存活，负=丢失）
  * - ``lostTurn``
    - -
    - 丢失圈数（ -1 表示未丢失）
  * - ``lostPosition``
    - m
    - 丢失位置 :math:`s` （ -1 表示未丢失）
  * - ``zCenter``
    - m
    - 所属束团的名义槽位 :math:`z_{\mathrm{center}}`

默认不保存随圈变化的参考量；注入时参考量始终与历史数据分开记录。
仅在 ``"Include reference": true`` 时，每行额外保存 ``referenceTime`` （s）、
``referenceBeta`` （无量纲）、 ``referenceMomentum`` （eV/c 每核子），
历史暂存缓冲因此从 11 列增加至 14 列。
参考量与同行粒子坐标对应同一次记录事件，此时存活粒子时间为

.. math::

   t_i=referenceTime-z/(referenceBeta\,c).

``zCenter`` 仅表示名义槽位。连续 z 可以超过环周范围，不单独决定分组。
开启参考列时，损失记录的参考量为 NaN，避免将冻结损失坐标误解为当前束团坐标。
需要参考历史的分析必须在跟踪前开启此选项；加速或重分组后，最终参考快照不能用于重建此前各圈。

诊断运行可通过 schema API 开启：

.. code-block:: python

   from PASS.para.schema.monitors import ParticleMonitorItem

   monitor = ParticleMonitorItem(s=0.0, max_tag=5, include_reference=True)

或在生成的 JSON 中设置：

.. code-block:: json

   "PM_reference": {
       "S (m)": 0.0,
       "Command": "ParticleMonitor",
       "Max tag": 5,
       "Include reference": true
   }

有界缓冲与完成状态
------------------

CPU 批量选择粒子身份；GPU 使用融合采样核，暂存块保留在显存中。
float64 历史缓冲的大小上限为

.. math::

   M = \mathrm{max\_tag}\,\min(N_{\mathrm{record}},N_{\mathrm{write}})\,
       N_{\mathrm{col}}\,8\;\mathrm{bytes}.

默认 :math:`N_{\mathrm{col}}=11`，启用 ``Include reference`` 后为 14。
初值数据和索引工作区另占与 ``max_tag`` 成正比的空间。
例如 10,000 个粒子、128 圈一块时，暂存缓冲为 112.64 MB；启用逐圈参考量后
为 143.36 MB，不随更长运行的总圈数继续增大。GPU 每次写出仅传回待写块。
写出时还需要该块的临时主机内存。结束时的 TFS 导出每块最多格式化 65,536 行，
不会把完整历史读入内存。配置的历史缓冲大小不受额外自动上限调整。

``ValidSamples`` 表示已提交的记录圈数，``EndTurn`` 为最近采样圈加一，
``RequestedEndTurn`` 为计划终点（不含）。
``Completed`` 只表示监视器覆盖自己的计划区间，不代表 sequence 中后续命令完成。
收尾仅刷新已有样本，不在其他格点位置补采样。协作提前停止因此只保留已经采样的圈。
GUI 正常停止会等待整圈完成再收尾；强制终止进程可能丢失尚未写出的块。
停止操作见 :doc:`../project_files`。

``NumTurn`` 保存实际样本数。最终 TFS 文件头包含全部最终计数，
包括 ``ValidSamples``、``ValidRows``、``EndTurn`` 和 ``Completed``。
正常提前停止会为已观测区间生成完整 TFS 表；未覆盖计划区间时 ``Completed=false``。

写入失败后，收尾不会重试状态不确定的块，也不会重复提交已有样本。
原始错误继续传播，并拒绝使用该监视器继续追踪。HDF5 读取器仅使用已提交前缀。
处理追加异常时，会按实际 ``ValidSamples`` 修正 ``NumTurn``、``EndTurn`` 和
``Completed``。若底层 HDF5 错误也阻止修复，保留原异常并补充说明；
``ValidSamples`` 仍是提交标记。注入初值的 ``initial/valid`` 标记只在坐标、
注入圈和参考量全部写入后发布。
应在完成后或暂停于两次写入之间读取 HDF5；这不是 SWMR 实时读取接口。
最终 TFS 文件在导出完成后可用。

HDF5 数组与注入初值
-------------------

文件根目录包含：

* ``particle_id``：(particle)，正整数身份，等于绝对 tag；
* ``turn``：(sample)，实际采样圈；
* ``x, px, y, py, z, dp``：(sample, particle)，按粒子精度保存；
* ``zCenter``：(sample, particle)，float64 名义分组元数据；
* ``tag, lostTurn, lostPosition``：(sample, particle)，类型依次为 int32、
  int64、float32；
* 可选 ``referenceTime, referenceBeta, referenceMomentum``：
  (sample, particle)，float64。

完整块写入后才更新 ``ValidSamples``；已提交前缀之后的行不属于有效数据。

``initial`` 组为每个所选粒子保存六维坐标，以及 ``particle_id``、``valid``、
``injection_turn``。这些是 **完成参考转换后的实际注入坐标**，在后续追踪命令前
捕获，即使 PM 从较晚圈开始记录也如此。注入事件的 ``referenceTime``、
``referenceBeta`` 和 ``referenceMomentum`` 始终保存，与可选逐圈参考量独立。
未捕获注入的条目为 ``valid=false``、NaN 坐标和 ``injection_turn=-1``。
不会用监视器第一条轨迹记录代替未知初值。

TFS 长表
--------

最终 TFS 文件为标准数值表，固定列顺序为：

::

   record turn particle_id x px y py z dp tag lostTurn lostPosition zCenter
   referenceTime referenceBeta referenceMomentum

上方分两行显示的是同一个表头。``record=0`` 表示注入记录，``turn`` 为注入圈；
每个捕获到初值的粒子保存一行；全部初值行位于轨迹行之前。
``record=1`` 表示轨迹样本，每个记录圈为每个
所选粒子保存一行。即使丢失后的有符号 ``tag`` 为负数，或尚未出现粒子的 tag
为零，``particle_id`` 始终保留正的身份编号。

TFS 始终包含参考量列。注入行保留实际参考量；未启用 ``Include reference`` 时，
轨迹行的参考列为 NaN。浮点文本保留 17 位有效数字。完整文件使用固定数值表结构，
可直接通过 ``tfs.read(path)`` 或 ``PASS.utils.table_io.read_table(path)`` 读取。

追踪期间，监视器将有界数据块写入私有的 ``.pass_pm_<uuid>.h5``，采用不压缩 HDF5。
结束清理时先导出到私有 ``.tfs.partial``，关闭后再发布完整最终 TFS，且不覆盖已有目标。
仅在发布成功后移除本次运行创建的临时 HDF5 与文本文件。导出失败时两者保留，
错误信息给出可恢复的 HDF5 路径；再次结束清理可由此重试导出。
未完成的文本不会出现在最终文件名下。标准 TFS 仅包含常规文件头、列名/类型行和数据行，
不在数据中插入块提交注释。

若发布成功但临时文件清理失败，最终 TFS 仍然有效；再次结束清理仅重试移除临时文件。
清理范围仅限此监视器在本次运行中创建的临时文件。
若创建最终硬链接后才收到异常，会核对文件身份以确认发布已经成功。
再次结束清理仅重试移除临时文件；恢复输出已完成的组件快照时也适用。

读取、组件快照与结果解释
------------------------

DA 分析使用 ``PASS.analysis.read_dynamic_aperture(path)``；指定轨迹可使用
``PASS.analysis.data_io.load_signal(path, "x", object_range=[0, 10])``。
这些读取器识别实际圈号、粒子 ID 及初值与轨迹记录。频谱读取会拒绝丢失或缺失样本。
多维 PM HDF5 文件不能交给普通一维 ``read_table`` 接口。
DA 与轨迹读取器接受一个当前 PM 文件，文件自身包含捕获的注入初值。

``state_dict()`` 仅捕获当前监视器的状态：已写出的文件、待写缓冲和注入初值。
``stage_checkpoint(data, next_turn, xp)`` 校验该组件快照，返回供调用者应用的
独立状态。这些底层 API 不恢复粒子、参考量、模拟圈号或其他命令。
跟踪入口从第 0 圈开始新运行，不提供使用这些快照续跑的流程。

除常规有界缓冲外，创建快照时还需临时容纳已保存文件的字节数据。
组件数据使用 ``PASS-particle-monitor-state-2``，标记保存的字节属于运行中
HDF5 还是已完成的 TFS；仅支持此版本。监视器的输出恢复机制将此前历史保留在
新文件中，不依赖原输出路径。从 TFS 快照恢复时，先重建私有 HDF5 存储，再追加新样本。

尚未出现的粒子 tag 为零，例如尚未注入的粒子。应结合 tag 与损失字段筛选数据；
冻结的损失坐标不能用后续存活束团的参考量解释。
研究完整圈存活时，应将监视器放在所有相关效应之后。
通用格式见 :doc:`table_output`，坐标约定见 :ref:`zh-longitudinal-reference`。
