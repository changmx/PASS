粒子监视器（ParticleMonitor）
==============================

简介
----

``ParticleMonitor`` 是逐圈粒子坐标监视器，在指定纵向位置记录选定粒子的 6D 相空间坐标，每圈记录一次。与 ``StatMonitor`` 记录束团整体统计量不同， ``ParticleMonitor`` 关注 **单个粒子** 的逐圈运动轨迹，是工作点（ tune ）测量、色品测量、振幅依赖效应分析等逐束团逐圈（ turn-by-turn, TBT ）诊断的核心工具。

- **代码位置** ： ``PASS/commands/monitor/particle_monitor.py``
- **类名** ： ``ParticleMonitor`` ，注册名 ``"particlemonitor"``
- **核心特征** ：

  - 通过 ``max_tag`` 参数选择记录粒子，匹配条件为 :math:`1 \leq |\mathrm{tag}| \leq \mathrm{max\_tag}` ；
  - 支持设置记录圈数范围 ``[start_turn, end_turn)`` ，不必从第 0 圈开始追踪；
  - 预分配 buffer ``（max_tag, num_record_turn, 10）`` ，避免运行时动态分配；
  - 每圈记录 10 列数据： turn + 6D 坐标 + tag + lost_turn + lost_position ；
  - 模拟结束后每个粒子单独写入一个 TFS 文件；
  - 文件名含监视器名称和纵向位置（ 3 位小数），支持多位置部署；
  - CPU 使用 numpy ， GPU 使用 cupy ， buffer 全程驻留 GPU ，仅结束时做一次 D2H 拷贝；


粒子选择机制
------------

PASS 中每个粒子拥有全局唯一的 ``tag`` （正整数），插入的测试粒子从 ``tag = 1`` 开始递增。 ``ParticleMonitor`` 通过 ``max_tag`` 参数指定记录范围：

.. math::

   \text{recorded} = \{\, i \;\mid\; 1 \leq |\mathrm{tag}_i| \leq \mathrm{max\_tag} \,\}

注意匹配条件使用的是 :math:`|\mathrm{tag}|` （绝对值），因此：

- ``tag = 1, 2, \ldots, \mathrm{max\_tag}`` ：正常存活粒子
- ``tag`` 取负 ：已丢失粒子 **同样被记录** ，其坐标保持丢失前的最后值

.. note::

  测试粒子通过 ``Injection`` 的 ``Insert Particle Coordinate`` 参数插入，插入后的粒子 ``tag`` 从 1 开始递增。 ``max_tag`` 应等于插入的测试粒子数。

  若 ``max_tag < 1`` ，监视器仅输出警告日志，不记录任何粒子，但不影响模拟运行。


记录圈数范围
------------

通过 ``start_turn`` 和 ``end_turn`` 可指定记录的圈数范围：

.. math::

   \text{recorded turns} = \{\, n \;\mid\; \mathrm{start\_turn} \leq n < \mathrm{end\_turn} \,\}

- ``start_turn`` ：记录起始圈（含），默认 0
- ``end_turn`` ：记录结束圈（不含），默认 -1 表示最后一圈（含）

实际记录的圈数为：

.. math::

   N_{\mathrm{record}} = \mathrm{end\_turn} - \mathrm{start\_turn}

典型用途：前 200 圈让束流稳定（不记录），从第 200 圈开始记录 1000 圈用于 FFT 分析。


预分配策略
----------

``ParticleMonitor`` 在初始化时预分配完整 buffer ：

.. math::

   \mathrm{buffer} \in \mathbb{R}^{\mathrm{max\_tag} \times N_{\mathrm{record}} \times 10}

内存开销：

.. math::

   M = \mathrm{max\_tag} \times N_{\mathrm{record}} \times 10 \times 8 \;\text{bytes}

典型场景（ 14 个测试粒子，记录 1000 圈） ：

.. math::

   M = 14 \times 1000 \times 10 \times 8 = 1.12 \;\text{MB}

buffer 使用与束流相同的数组后端（ ``beam.particles.xp`` ）， CPU 用 numpy ， GPU 用 cupy 。预分配的优势：

- 运行时零内存分配，不影响追踪性能；
- GPU 场景下 buffer 全程驻留 GPU 显存，每圈直接从 GPU 粒子数组写入 GPU buffer ，仅在模拟结束时做一次 D2H 拷贝；
- 固定内存布局，便于后处理分析。


接口参数
--------

.. list-table::
  :header-rows: 1
  :widths: 20 20 10 10 40

  * - 属性名
    - JSON key
    - 类型
    - 默认值
    - 说明
  * - ``s``
    - ``"S (m)"``
    - float
    - 必填
    - 监视器在束线中的纵向位置
  * - ``cmd_name``
    - ``"name"``
    - str
    - 必填
    - 监视器名称（由序列键名自动填入）
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

.. note::

  ``max_tag`` 应与 ``Injection`` 中 ``Insert Particle Coordinate`` 插入的粒子数量一致。例如插入 14 个测试粒子，则 ``max_tag = 14`` 。


输出文件
--------

每个粒子生成一个独立的 TFS 文件：

- **文件名** ： ``{hms}_particle_beam{bid}_{monitor_name}_s_{s:.3f}_tag_{tag}.tfs``
- **输出目录** ： ``output_dir_particle``

TFS 文件头：

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

输出列（共 10 列）：

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
    - 纵向位置
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


使用示例
--------

基本用法
~~~~~~~~

以下 JSON 片段在 :math:`s = 0.0` m 处放置一个粒子监视器，记录 ``tag = 1`` 至 ``tag = 3`` 的粒子：

.. code-block:: json

   "PM1": {
       "S (m)": 0.0,
       "Command": "ParticleMonitor",
       "Max tag": 3
   }

配合 ``Injection`` 中插入 3 个测试粒子：

.. code-block:: json

   "injection": {
       "S (m)": 0.0,
       "Command": "Injection",
       "bunch0": {
           "Insert Particle Coordinate": [
               [0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.001, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.001]
           ]
       }
   }

上述配置插入了 3 个测试粒子：

- ``tag = 1`` ： :math:`x = 1` mm 水平偏移粒子，用于水平工作点测量
- ``tag = 2`` ： :math:`y = 1` mm 垂直偏移粒子，用于垂直工作点测量
- ``tag = 3`` ： :math:`\delta = 10^{-3}` 动量偏移粒子，用于色散和色品测量

模拟结束后在 ``output_dir_particle`` 目录下生成 3 个 TFS 文件，每个文件包含该粒子所有记录圈的 6D 坐标。

延迟记录
~~~~~~~~

以下配置在前 200 圈不记录（让束流稳定），从第 200 圈开始记录至第 1000 圈：

.. code-block:: json

   "PM1": {
       "S (m)": 0.0,
       "Command": "ParticleMonitor",
       "Max tag": 14,
       "Start turn": 200,
       "End turn": 1000
   }

buffer 大小按 :math:`1000 - 200 = 800` 圈分配，输出的 TFS 文件中 ``turn`` 列从 200 开始。

多位置监视
~~~~~~~~~~

可在环上不同位置放置多个粒子监视器，比较粒子在不同位置的相空间坐标：

.. code-block:: json

   "PM_start": {
       "S (m)": 0.0,
       "Command": "ParticleMonitor",
       "Max tag": 14
   },
   "PM_mid": {
       "S (m)": 284.5,
       "Command": "ParticleMonitor",
       "Max tag": 14
   }


应用场景
--------

- **工作点测量** ：对 TBT 坐标做 FFT 或 NAFF ，提取 betatron 振荡频率即为工作点 :math:`Q_x` 、 :math:`Q_y`
- **色品测量** ：在不同动量偏差 :math:`\delta` 下分别测量工作点，线性拟合 :math:`Q(\delta)` 的斜率即为色品 :math:`DQ_x` 、 :math:`DQ_y`
- **振幅依赖 tune 偏移（ ADTS ）** ：以不同初始振幅的粒子测量 tune ，分析非线性 tune 随振幅的偏移
- **色散函数测量** ：对动量偏移粒子的 TBT 质心轨道取时间平均，除以 :math:`\delta` 即得色散函数 :math:`D(s)`
- **滑移因子测量** ：对动量偏移粒子的纵向坐标 :math:`z` 逐圈记录，每圈 :math:`z` 的变化率除以 :math:`\delta` 即得滑移因子 :math:`\eta`
- **闭合轨道验证** ：初始无偏移粒子的 TBT 坐标应保持不变，验证闭合轨道稳定性
- **粒子损失追踪** ：通过 ``tag`` 符号变化和 ``lostTurn`` / ``lostPosition`` 定位粒子丢失的时刻和位置
