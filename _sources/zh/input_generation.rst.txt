输入文件生成（命令行模式）
============================

简介
----

PASS 采用 **JSON 文件** 作为仿真输入。引擎（ ``Config`` 、 ``Beam`` 、 ``CommandSequence`` ）从 JSON 文件读取全部参数，包括粒子种类、束团分布、Lattice序列、监测器等。

参数系统 ``PASS/para/`` 提供了一套基于 **pydantic v2** 的 schema 定义，用户通过 Python 脚本组装参数对象，调用 ``generate_input()`` 即可输出引擎兼容的 JSON 文件。这种方式相比手写 JSON 有以下优势：

- **类型安全** ：参数类型、范围在 schema 中声明，非法值在生成时即被拦截；
- **别名映射** ：Python 代码使用简洁属性名（如 ``circumference`` ），JSON 输出自动使用引擎期望的 key（如 ``"Circumference (m)"`` ）；
- **可复用** ：schema 对象可 ``model_copy(update={...})`` 快速派生变体，适合参数扫描；
- **GUI 可扩展** ：schema 自带 JSON Schema 导出，未来 GUI 可自动渲染表单。

.. note::

    本文档介绍命令行模式下的输入文件生成方式。GUI 模式将在未来版本中提供。

架构概览
--------

参数系统分为五层，各层职责清晰、互不依赖：

.. code-block:: text

   PASS/para/
   ├── schema/       参数定义（唯一数据源）
   │   ├── main.py         MainConfig：全局仿真参数
   │   ├── bunch.py        BunchConfig + OffsetConfig + InjectionItem
   │   ├── twiss.py        TwissPoint：twiss 传输点
   │   ├── elements.py     12 种元件（Drift→RFCavity）
   │   ├── monitors.py     StatMonitor / DistMonitor / PhaseMonitor
   │   ├── space_charge.py SpaceChargeConfig
   │   └── sequence.py     Sequence：有序容器 + 自动排序
   ├── madx.py        MADX TFS → schema 对象（element / twiss / error）
   ├── smooth.py      解析平滑近似 twiss
   ├── tools/        外部数据 → PASS TFS
   │   ├── data_converter.py 通用数据转换流水线
   │   ├── ramping.py         元件 ramping 文件生成
   │   ├── rf_data.py         RF 数据文件生成
   │   └── exciter_data.py    Exciter 数据文件生成
   ├── toolkit.py    sort_sequence + class_map + apply_element_settings + build_sequence
   └── api.py        高级 API（generate_input / load_input / generate_from_tfs）

数据流如下：

.. code-block:: text

   MADX TFS / 用户参数 / 外部数据文件
              │
              ▼
        madx.py / smooth.py + tools/  → schema 对象 / TFS 文件
              │
              ▼
         schema/ (pydantic)     ← 唯一数据源：验证 + 别名
              │
              ▼
        api.py (generate_input) → beam0.json
              │
              ▼
         PASS 引擎 (Config → Beam → CommandSequence → Executor)


快速开始
--------

最简示例
~~~~~~~~

以下脚本生成一个包含注入 + 平滑近似 twiss + 统计监测器的完整输入文件：

.. code-block:: python

   from PASS.para.api import generate_input
   from PASS.para.schema.main import MainConfig
   from PASS.para.schema.bunch import BunchConfig, InjectionItem
   from PASS.para.schema.sequence import Sequence
   from PASS.para.schema.monitors import StatMonitor
   from PASS.para.smooth import generate_smooth_twiss

   # 1. 全局参数
   main = MainConfig(
       beam_name="proton",
       num_proton=1, num_neutron=0, num_electron=1,
       gamma_t=4.8, circumference=251.327,
       num_turns=1000, backend="cpu",
   )

   # 2. 束团
   bunch = BunchConfig(
       kinetic_energy=45e6,
       num_real_particles=int(1e11),
       num_macro_particles=int(1e5),
       beta_x=0.5, beta_y=0.5,
       alpha_x=-2.61, alpha_y=1.57,
       emit_x=200e-6, emit_y=100e-6,
       sigma_z=30, dp=0.005,
       dist_trans="gaussian", dist_longi="matchz",
       rf_voltage=100e3, rf_phase=0.5236,
   )

   # 3. Lattice序列
   items, circum = generate_smooth_twiss(
       circumference=main.circumference,
       qx=4.8, qy=4.4, num_points=100,
   )
   main.circumference = circum

   seq = Sequence()
   seq.add("injection", InjectionItem(s=0.0, bunches=[bunch]))
   for i, item in enumerate(items):
       seq.add(f"twiss_{i:04d}", item)
   seq.add("stat1", StatMonitor(s=0.0))

   # 4. 生成 JSON
   generate_input(main, seq, "beam0.json")

运行方式：

.. code-block:: console

   cd C:\Users\changmx\Documents\PASS
   python input/generate_beam0.py

输出文件： ``input/beam0.json``

JSON 文件结构
-------------

生成的 JSON 文件结构如下：

.. code-block:: json

   {
       "Beam Name": "proton",
       "Number of Protons": 1,
       "Number of Neutrons": 0,
       "Number of Charges": 1,
       "Transition Gamma": 4.8,
       "Circumference (m)": 251.327,
       "Number of turns": 1000,
       "Backend (gpu/cpu)": "cpu",
       "Number of GPU devices": 1,
       "Device Id": [0],
       "Output directory": "./output",
       "Is plot figure": true,
       "Is space charge": false,
       "Is beam-beam": false,
       "Sequence": {
           "injection": {
               "S (m)": 0.0,
               "Command": "Injection",
               "bunch0": {}
           },
           "twiss_0000": {
               "S (m)": 0.0,
               "Command": "Twiss",
               "S previous (m)": 0.0,
               "Beta x (m)": 8.333
           },
           "stat1": {
               "S (m)": 0.0,
               "Command": "StatMonitor"
           }
       }
   }

.. note::

    JSON 的 key 名称是引擎的硬性契约。schema 层通过 pydantic 的 ``alias`` 机制自动处理 Python 属性名到 JSON key 的映射，用户无需手写。

    引擎在读取时会先调用 ``convert_keys_to_lower()`` 将所有 key 转为小写，因此 JSON key 的大小写不影响读取。


核心组件
--------

MainConfig（全局参数）
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 10 40

   * - 属性名
     - JSON key
     - 类型
     - 说明
   * - ``beam_name``
     - ``Beam Name``
     - str
     - 束流标签
   * - ``num_proton``
     - ``Number of Protons``
     - int
     - 每粒子质子数（0 表示电子/正电子）
   * - ``num_neutron``
     - ``Number of Neutrons``
     - int
     - 每粒子中子数（>0 表示离子）
   * - ``num_electron``
     - ``Number of Charges``
     - int
     - 每粒子电荷数（可负，不可为 0）
   * - ``gamma_t``
     - ``Transition Gamma``
     - float
     - 过渡 gamma
   * - ``circumference``
     - ``Circumference (m)``
     - float
     - 环周长 (m)
   * - ``num_turns``
     - ``Number of turns``
     - int
     - 仿真圈数
   * - ``backend``
     - ``Backend (gpu/cpu)``
     - str
     - 计算后端： ``cpu`` 或 ``gpu``
   * - ``num_gpu``
     - ``Number of GPU devices``
     - int
     - GPU 数量
   * - ``gpu_id``
     - ``Device Id``
     - list[int]
     - GPU 设备 ID 列表
   * - ``output_dir``
     - ``Output directory``
     - str
     - 输出目录
   * - ``is_plot``
     - ``Is plot figure``
     - bool
     - 是否生成图表
   * - ``is_space_charge``
     - ``Is space charge``
     - bool
     - 是否启用空间电荷
   * - ``is_beambeam``
     - ``Is beam-beam``
     - bool
     - 是否启用束流-束流相互作用

BunchConfig（束团参数）
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 30 10 35

   * - 属性名
     - JSON key
     - 类型
     - 说明
   * - ``kinetic_energy``
     - ``Kinetic Energy per Nucleon (eV/u)``
     - float
     - 每核子动能 (eV/u)
   * - ``num_real_particles``
     - ``Number of Real Particles``
     - int
     - 每束团真实粒子数
   * - ``num_macro_particles``
     - ``Number of Macro Particles``
     - int
     - 每束团宏粒子数
   * - ``beta_x`` / ``beta_y``
     - ``Beta x (m)`` / ``Beta y (m)``
     - float
     - Twiss β 函数
   * - ``alpha_x`` / ``alpha_y``
     - ``Alpha x`` / ``Alpha y``
     - float
     - Twiss α 函数
   * - ``emit_x`` / ``emit_y``
     - ``Emittance x (m'rad)``
     - float
     - 发射度
   * - ``sigma_z``
     - ``Sigma z (m)``
     - float
     - 束团长度
   * - ``dp``
     - ``Sigma dp/p``
     - float
     - 动量展宽
   * - ``dist_trans``
     - ``Transverse dist``
     - str
     - 横向分布： ``kv`` / ``gaussian`` / ``uniform`` / ``waterbag`` / ``parabolic``
   * - ``dist_longi``
     - ``Longitudinal dist``
     - str
     - 纵向分布： ``gaussian`` / ``coasting`` / ``matchz`` / ``matchdp``
   * - ``rf_voltage``
     - ``RF Voltage (V)``
     - float
     - RF 电压（matchz/matchdp 模式使用）
   * - ``rf_phase``
     - ``RF Phase (rad)``
     - float
     - RF 相位

Sequence（序列容器）
~~~~~~~~~~~~~~~~~~~~

``Sequence`` 是一个有序容器，存储所有按位置 ``s`` 排列的序列项。添加顺序不影响最终结果——导出时自动按 ``(s, command priority)`` 排序。

.. code-block:: python

   seq = Sequence()
   seq.add("injection", InjectionItem(s=0.0, bunches=[bunch]))
   seq.add("qd1", QuadrupoleElement(s=1.0, k1l=0.2, length=0.5))
   seq.add("stat1", StatMonitor(s=0.0))

支持的序列项类型：

- ``InjectionItem`` — 注入点（必须 ``s=0`` ）
- ``TwissPoint`` — twiss 传输点
- ``DriftElement`` 、 ``QuadrupoleElement`` 、 ``SBendElement`` 等 — 物理元件
- ``StatMonitor`` 、 ``DistMonitor`` 、 ``PhaseMonitor`` — 监测器


Lattice来源
------------

PASS 支持三种Lattice序列生成方式，可根据需要选择或混合使用：

方式一：从 MADX twiss 文件读取
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

读取 MADX 生成的 twiss TFS 文件，每个元件转为一个 ``TwissPoint`` 传输点。适用于 **逐 twiss 传输** 模式。

.. code-block:: python

   from PASS.para.madx import read_madx_twiss

   items, circum = read_madx_twiss(
       twiss_file="lattice.tfs",
       error_file="errors.tfs",       # 可选
       muz=0.001,                      # 纵向 tune
       dqx=0.0,                        # 色品（或 "from_file"）
       dqy=0.0,
       is_field_error=False,           # 是否读取场误差
       insert_patterns=["QD.*"],      # 正则匹配，插入为薄透镜元件
   )

方式二：从 MADX twiss 文件读取为元件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

读取 twiss 文件，但每个元件转为对应的物理元件对象（ ``QuadrupoleElement`` 、 ``SBendElement`` 等）。适用于 **逐元件追踪** 模式。

.. code-block:: python

   from PASS.para.madx import read_madx_elements

   items, names, circum = read_madx_elements(
       twiss_file="lattice.tfs",
       is_merge_drift=True,            # 合并相邻漂移节
       is_field_error=True,
       error_file="errors.tfs",
   )

方式三：平滑近似 twiss
~~~~~~~~~~~~~~~~~~~~~~

无需 MADX 文件，用解析公式生成恒定 β 函数的 twiss 点。 :math:`\beta = C / (2\pi Q)` 。适用于快速测试。

.. code-block:: python

   from PASS.para.smooth import generate_smooth_twiss

   items, circum = generate_smooth_twiss(
       circumference=569.1,
       qx=9.47, qy=9.43,
       num_points=100,
       muz=0.001,
   )

混合模式
~~~~~~~~

twiss 传输点和物理元件可以在同一个序列中混合使用。例如在 twiss 序列中插入一个 RF 腔：

.. code-block:: python

   from PASS.para.schema.elements import RFCavityElement

   seq = Sequence()
   seq.add("injection", InjectionItem(s=0.0, bunches=[bunch]))

   # twiss 传输点
   for i, item in enumerate(twiss_items):
       seq.add(f"twiss_{i:04d}", item)

   # 插入 RF 腔（在 s=0 处）
   seq.add("rf1", RFCavityElement(s=0.0, voltage=100e3, harmonic=1, phase=0.5236))


外部数据文件转换
----------------

PASS 使用 **TFS 格式** 作为所有 ramping/RF/exciter 数据文件的统一格式。 ``tools/data_converter.py`` 提供了通用转换流水线，将各种外部文件（CSV/TXT/TFS）转为 PASS TFS。

四步流水线
~~~~~~~~~~

.. code-block:: text

   外部文件 → load_raw_data → time_to_turn → interpolate → write_tfs

1. **load_raw_data** ：读取外部文件，自动检测 turn/time 列
2. **time_to_turn** ：如外部文件给的是时间而非圈数，用回旋频率转换
3. **interpolate_to_continuous_turns** ：圈数不连续时自动插值
4. **write_tfs_ramping** ：写入 PASS 统一 TFS 格式

一步到位
~~~~~~~~

.. code-block:: python

   from PASS.para.tools.data_converter import convert_external_to_tfs

   convert_external_to_tfs(
       input_path="external_ramp.csv",     # 外部文件
       output_path="k1l_ramping.tfs",      # PASS TFS
       data_cols=["k1l", "k1sl"],          # 数据列名
       revolution_freq=1.76e6,             # 回旋频率 (Hz)
       num_turns=5000,                     # 目标圈数
       method="linear",                    # 插值方法
   )

预置封装
~~~~~~~~

针对常见元件类型的薄封装：

.. code-block:: python

   from PASS.para.tools.ramping import convert_k1l_ramping, convert_k2l_ramping
   from PASS.para.tools.rf_data import convert_rf_data

   # 四极铁 ramping
   convert_k1l_ramping("external.csv", "k1l_ramping.tfs", revolution_freq=1.76e6)

   # RF 数据
   convert_rf_data("llrf.csv", "rf_data.tfs", revolution_freq=1.76e6)

分步调用
~~~~~~~~

外部文件格式特殊时，可分步调用各函数：

.. code-block:: python

   from PASS.para.tools.data_converter import (
       interpolate_to_continuous_turns, write_tfs_ramping,
   )
   import numpy as np

   # 自行准备数据
   turn_arr = np.array([1, 50, 100, 500, 1000])
   k2l = np.array([0.0, 0.5, 1.0, 2.5, 4.4])

   turn_cont, data_cont = interpolate_to_continuous_turns(
       turn_arr, {"K2L": k2l},
       start_turn=1, end_turn=1000, method="linear",
   )
   write_tfs_ramping("k2l_ramping.tfs", turn_cont, None, data_cont)


API 参考
--------

.. code-block:: python

   from PASS.para.api import generate_input, load_input

   # 生成 JSON
   generate_input(
       main: MainConfig,
       sequence: Sequence,
       output_path: str,
       space_charge: SpaceChargeConfig | None = None,
       extra_modules: dict | None = None,
   ) -> str

   # 加载已有 JSON（用于修改后重新生成）
   main, seq_dict = load_input("beam0.json")

完整示例
--------

项目内置的示例脚本位于 ``input/generate_beam0.py`` ，可直接运行：

.. code-block:: console

   cd C:\Users\changmx\Documents\PASS
   python input/generate_beam0.py

该脚本演示了完整的端到端流程：全局参数 → 多束团配置 → 平滑近似 twiss → Lattice 序列组装 → JSON 输出。生成的 ``beam0.json`` 可直接被 PASS 引擎读取执行。
