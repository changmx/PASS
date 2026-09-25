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

``output_format``（JSON 键 ``Output format``）默认为 ``hdf5-gzip1``，也可选择 ``hdf5`` 或 ``tfs``。命令名称由序列键给定。

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

每个粒子默认生成一个独立的 HDF5 文件：

- **文件名** ： ``{hms}_beam{bid}_{monitor_name}_s{s:.3f}_tag{tag}.h5``
- **输出目录** ： ``output_dir_particle``

元数据（HDF5 属性，文本模式下为 TFS 文件头）：

::

   @ Name             PASS Particle Monitor
   @ Time             2026-07-14 00:11:03
   @ Monitor          pm1
   @ S                0.0
   @ BeamId           0
   @ Tag              1
   @ NumTurn          1000
   @ StartTurn        0
   @ EndTurn          1000

协作提前停止时，收尾只写出监视器已经采样的圈，
不输出预分配缓冲区中尚未采样的未来行。
``NumTurn`` 和 ``EndTurn`` 描述实际保存的行，其中 ``EndTurn`` 仍为不含端点。
部分输出额外记录 ``RequestedEndTurn``，表示将 ``-1`` 解析并按仿真圈数裁剪后的计划终点。
例如从第 200 圈开始记录、完成第 499 圈后停止，若原计划终点为 1000，
则输出 ``NumTurn=300``、``EndTurn=500`` 和 ``RequestedEndTurn=1000``。
完整区间保持原有元数据，不增加 ``RequestedEndTurn``；尚未开始记录时不写粒子表。

CPU 与 GPU 使用相同的收尾规则，重复收尾不会重写已经完成的表。
GPU 在设备到主机拷贝之前先裁切缓冲区，因此传输量与得到的主机数组大小随实际记录圈数增长，
初始化时的显存预分配仍覆盖计划区间。
GUI 正常停止会等待完整一圈结束后收尾；强制结束不保证缓冲数据已经写出。
GUI 停止控件及运行记录见 :doc:`../project_files`。

默认输出列（共 11 列）：

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

默认不保存参考量，数据列和 headers 中均不写入这三个值。
仅在 ``"Include reference": true`` 时，每行额外保存 ``referenceTime`` （s）、
``referenceBeta`` （无量纲）、 ``referenceMomentum`` （eV/c 每核子），合计 14 列。
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

预分配策略
----------

``ParticleMonitor`` 在初始化时预分配完整 buffer ：

.. math::

   \mathrm{buffer} \in \mathbb{R}^{\mathrm{max\_tag} \times N_{\mathrm{record}} \times N_{\mathrm{col}}}

``Include reference`` 默认为 false，此时 :math:`N_{\mathrm{col}}=11`；
开启后 :math:`N_{\mathrm{col}}=14`。关闭时 CPU 和 GPU 均不为参考列分配缓冲区。

内存开销：

.. math::

   M = \mathrm{max\_tag} \times N_{\mathrm{record}} \times N_{\mathrm{col}} \times 8 \;\text{bytes}

典型场景（ 14 个测试粒子，记录 1000 圈） ：

.. math::

   M = 14 \times 1000 \times 11 \times 8 = 1.232 \;\text{MB}

开启参考列后，该示例占用 1.568 MB，比默认模式增加 27.3%。

buffer 使用与束流相同的数组后端（ ``beam.particles.xp`` ）， CPU 用 numpy ， GPU 用 cupy 。预分配的优势：

- 历史缓冲区在跟踪前分配，记录过程仍有计算开销；
- GPU 场景下 buffer 全程驻留 GPU 显存，每圈直接从 GPU 粒子数组写入 GPU buffer ，仅在模拟结束时做一次 D2H 拷贝；
- 固定内存布局，便于后处理分析。


结果解释与限制
--------------

尚未出现的粒子对应历史行保持零值，例如尚未注入的粒子；分析时应结合 tag 与损失信息筛选。损失坐标保持冻结，不能使用后续存活束团的参考量解释。完整历史缓冲区大小与 max_tag 和记录圈数的乘积成正比，较大规模运行前应合理设置这两个范围。通用格式及读取方法见 :doc:`table_output`，坐标定义见 :ref:`zh-longitudinal-reference`。
