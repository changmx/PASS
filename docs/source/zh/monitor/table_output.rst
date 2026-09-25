表格输出格式
============

格式选择
--------

``output_format``（JSON ``"Output format"``）统一选择格式和压缩方式：

- ``"hdf5-gzip1"``：HDF5，使用无损 gzip level 1 和 shuffle（**默认**）。
- ``"hdf5"``：不压缩的 HDF5，也不启用 shuffle。
- ``"tfs"``：TFS 文本输出。

省略此选项时使用 ``"hdf5-gzip1"``。格式选择不启用原本关闭的输出，
也不改变配置的记录圈数。

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - 输出来源
     - 文件组织
     - 附加输出
   * - DistMonitor
     - 每个束团、每个选定圈一个分布快照
     - 无
   * - PhaseAdvanceMonitor
     - 每个束团、每个完成窗口一个表格
     - 无
   * - ParticleMonitor
     - 每个选定粒子一个逐圈历史文件
     - 无
   * - StatMonitor
     - 每个束团、每个监视器位置一个统计历史文件
     - 包含相同行的 CSV
   * - Injection
     - 启用 ``save_init_dist`` 时保存初始分布；
       在各 ``BunchConfig`` / ``bunchN`` 中设置 ``output_format``
     - 无
   * - Slicer
     - 每个束团、每个选定圈的逐粒子信息
     - 切片汇总仍为 TFS 和 CSV

HDF5 表格使用 ``.h5``，TFS 表格使用 ``.tfs``，文件名主体保持一致。
SpaceCharge 场输出使用多维 HDF5 结构，不增加 TFS 导出。

HDF5 结构与元数据
-----------------

每个数值或布尔列对应文件根目录下的一个 **一维 dataset**，保留原列名和
数据类型，各列长度相同。分布表的一行对应一个粒子；StatMonitor 的一行
对应一个记录圈；ParticleMonitor 的一行对应文件所标识粒子的一个记录圈。
例如，统计文件包含::

   /turn                  (N,)
   /xAverage              (N,)
   /sigmaX                (N,)
   /referenceTime         (N,)
   /referenceBeta         (N,)
   /referenceMomentum     (N,)
   ...

表格头信息保存为 **文件根属性**，包括名称、现有元数据中的
单位、参考约定和监视器位置。StatMonitor 随圈变化的参考量始终作为逐行列保存，
保留完整历史；快照中的参考 attributes 只描述该次快照。ParticleMonitor
仍然仅在 ``"Include reference": true`` 时保存参考量列。

另有两个保留属性描述表格格式：``_pass_table_version=1`` 和
``_pass_table_columns``（按输出顺序排列的列名 JSON 数组）。用户 headers
不能以 ``_pass_table_`` 开头。统一读取器也支持没有这些属性的旧 PASS
HDF5 分布快照。SpaceCharge 多维场文件需使用专用场数据读取器。

``"hdf5-gzip1"`` 使用无损 gzip level 1 压缩并启用 shuffle。
shuffle 在压缩前重排字节，不改变保存的数值或粒子顺序。
``"hdf5"`` 同时关闭压缩和 shuffle。两者使用相同的 ``.h5`` 扩展名、
逻辑结构和读取器，无需单独的压缩参数。此选项适用于上表中的表格输出；
CSV 保持不变，SpaceCharge 继续使用独立的场数据写入器。
例如，选择不压缩的 HDF5：

.. code-block:: python

   from PASS.para.schema.monitors import DistMonitorItem

   monitor = DistMonitorItem(s=0.0, save_turns=[[0]], output_format="hdf5")

.. code-block:: json

   {"Command": "DistMonitor", "S (m)": 0.0, "Save turns": [[0]],
    "Output format": "hdf5"}

不压缩 HDF5 可以缩短频繁保存快照时的写入耗时，代价是文件更大。
两种设置均不改变数值精度或表格逻辑结构。快照每列的 chunk 约以 64 KiB 为上限，
同时不超过该列的行数。StatMonitor 每列的 chunk 行数等于配置的批量写入间隔，
使正常的完整批次填入新块，避免重新压缩之前的块。
分块只影响存储和压缩，不改变采样间隔或数值精度。

粒子统计批量写入与运行中查看
----------------------------

StatMonitor **每圈计算并缓存一行**。可选参数 ``write_interval_turns`` /
``"Write interval (turns)"`` 必须为正整数，默认 **100**，不使用时间触发。
设置为 100 时，在零起始圈号 99、199 等处刷新；最后一圈补写剩余记录。
执行器在可捕获异常或键盘中断后的清理阶段也会补写不足一批的记录；
强制终止进程可能丢失尚未写入的一批。

.. code-block:: python

   from PASS.para.schema.monitors import StatMonitorItem

   stat = StatMonitorItem(s=0.0, output_format="hdf5-gzip1", write_interval_turns=100)

.. code-block:: json

   {
       "Command": "StatMonitor",
       "S (m)": 0.0,
       "Output format": "hdf5-gzip1",
       "Write interval (turns)": 100
   }

GPU 模式下，监视器根据指定圈数间隔和初始束团数预分配统计缓存，
中间各圈只向缓存填入结果，不将统计值复制回 CPU。
到达批次边界后，一次 GPU 到 CPU 的复制取回所有束团的待写记录；
结束清理时也会传回最后不足一批的数据。
参考量逐圈记录在 CPU 上，保留完整历史。
传回后逐条使用 CPU 公式计算派生输出列。
CPU 模式则在 CPU 内存中缓存已计算完毕的各行。

HDF5 与 CSV 写入同一批记录，每批写完关闭文件。选择 ``"tfs"`` 时，
CSV 同样按批追加，完整 TFS 在结束清理时生成。
CSV 在第一批写入后即可查看，保存全部记录，而非每批只保留一行。
两个文件依次写入，不是原子提交的一对文件。

运行中查看统计值建议使用 CSV，例如 Linux 的 ``tail -n 10 -f path.csv``，
或 PowerShell 的 ``Get-Content path.csv -Tail 10 -Wait``。
HDF5 是二进制文件，不能用 ``tail`` 直接查看有意义的数值。
应在模拟完成后，或暂停于两批写入之间时读取 HDF5。
此写入器不启用 SWMR；其他进程长期打开 HDF5 可能阻止后续写入，
写入过程中直接复制也不能保证得到一致快照。

读取表格
--------

绘图工具、GUI 结果读取、文件分布注入和示例分析脚本均支持两种格式。
统一读取器返回包含数据列与头信息 的 ``TfsDataFrame``：

.. code-block:: python

   from PASS.utils.table_io import read_table

   data = read_table("statistics.h5")  # 也支持 .tfs 和 .hdf5
   print(data[["turn", "sigmaX", "referenceTime"]].tail())
   print(data.headers)

仅需少量列或末尾几行时，可直接读取数据集：

.. code-block:: python

   import h5py

   with h5py.File("statistics.h5", "r") as data:
       turns = data["turn"][-10:]
       sigma_x = data["sigmaX"][-10:]
       metadata = dict(data.attrs)
