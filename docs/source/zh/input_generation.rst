输入文件生成（命令行模式）
============================

PASS 从 JSON 文件读取仿真输入。使用 Python 配置类定义全局参数、束团与晶格序列，再调用 ``generate_input()`` 生成文件。图形界面配置方法见 :doc:`gui`。

配置对象在构造时检查已声明字段的类型与约束；:doc:`input_validation` 进一步检查完整输入、命令顺序及外部文件。``generate_input()`` 负责写出配置，不能代替完整输入校验。

.. _zh-minimal-input-example:

最小示例
------------

安装 PASS 后，将以下脚本保存为仓库根目录下的 ``input/generate_beam0.py``；若 ``input`` 目录不存在，先创建该目录。示例使用 2048 个宏粒子、64 圈 CPU 跟踪、平滑近似线性晶格和高斯分布，无需外部晶格文件。

.. code-block:: python

   from pathlib import Path

   from PASS.para.api import generate_input
   from PASS.para.schema.main import MainConfig
   from PASS.para.schema.bunch import BunchConfig, InjectionItem
   from PASS.para.schema.sequence import Sequence
   from PASS.para.schema.monitors import StatMonitorItem
   from PASS.para.smooth import generate_smooth_twiss
   from PASS.validation import validate_file

   main = MainConfig(
       beam_name="proton",
       num_proton=1, num_neutron=0, num_electron=1,
       gamma_t=4.8, circumference=251.327,
       num_turns=64, backend="cpu", output_dir="output", is_plot=False,
   )
   items, names, circumference = generate_smooth_twiss(
       circumference=main.circumference,
       qx=4.8, qy=4.4, num_points=17,
       longitudinal_transfer="off",
   )
   bunch = BunchConfig(
       kinetic_energy=45e6,
       num_real_particles=100_000_000_000,
       num_macro_particles=2048,
       beta_x=items[0].beta_x, beta_y=items[0].beta_y,
       alpha_x=0.0, alpha_y=0.0,
       emit_x=2e-6, emit_y=2e-6,
       sigma_z=0.1, dp=0.001,
       dist_trans="gaussian", dist_longi="gaussian",
   )
   seq = Sequence()
   seq.add("injection", InjectionItem(s=0.0, random_seed=2026, bunches=[bunch]))
   for name, item in zip(names, items):
       seq.add(name, item)
   seq.add("stat1", StatMonitorItem(s=0.0, write_interval_turns=16))

   output_path = Path(__file__).resolve().parent / "beam0.json"
   generate_input(main, seq, str(output_path))
   report = validate_file(str(output_path))
   if not report.ok:
       raise ValueError(report.text())
   print(f"Validated input: {output_path}")

在仓库根目录依次执行：

.. code-block:: console

   python input/generate_beam0.py
   python -c "from PASS.main import main; main('input/beam0.json', raise_errors=True)"

脚本生成 ``input/beam0.json``，其中的相对输出目录解析为 ``input/output``，每次运行在其下创建独立运行目录。运行目录包含 CSV 与 HDF5 统计表，共 64 行（圈号 0–63）；具体目录由日志给出。运行完成后可读取最新统计文件：

.. code-block:: python

   from pathlib import Path

   from PASS.utils.table_io import read_table

   files = list(Path("input/output").rglob("*stat*.h5"))
   latest = max(files, key=lambda path: path.stat().st_mtime)
   data = read_table(latest)
   print(latest)
   print(data[["turn", "sigmaX", "sigmaY", "xEmittance", "yEmittance"]].tail())

该非耦合线性模型中，横向 RMS 发射度应在数值舍入误差范围内保持不变；有限采样得到的初始值不必精确等于输入目标。本例关闭纵向输运，不包含同步振荡或集体效应。

JSON 文件结构
-------------

以下为省略部分字段的结构示意；空束团对象及省略参数仅用于展示层次，不能直接作为完整输入运行。

.. code-block:: json

   {
       "Beam Name": "proton",
       "Number of Protons": 1,
       "Number of Neutrons": 0,
       "Number of Charges": 1,
       "Transition Gamma": 4.8,
       "Circumference (m)": 251.327,
       "Number of turns": 64,
       "Backend (gpu/cpu)": "cpu",
       "Number of GPU devices": 1,
       "Device Id": [0],
       "Output directory": "./output",
       "Is plot figure": false,
       "Is beam-beam": false,
       "Sequence": {
           "injection": {
               "S (m)": 0.0,
               "Command": "Injection",
               "Harmonic Number": 1,
               "Random Seed": 2026,
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

    JSON 使用固定字段名。配置类通过 pydantic 的 ``alias`` 机制将 Python 字段名转换为 JSON 键。

    引擎在读取时会先调用 ``convert_keys_to_lower()`` 将所有 key 转为小写，因此 JSON key 的大小写不影响读取。


核心组件
--------

MainConfig（全局参数）
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 24 12 13 33

   * - Python 字段
     - JSON 键
     - 类型
     - 默认值
     - 说明
   * - ``beam_name``
     - ``Beam Name``
     - ``str``
     - ``'proton'``
     - 束流粒子种类的显示名称。
   * - ``num_proton``
     - ``Number of Protons``
     - ``int``
     - ``1``
     - 每个粒子的质子数；电子和正电子取 0。
   * - ``num_neutron``
     - ``Number of Neutrons``
     - ``int``
     - ``0``
     - 每个粒子的中子数。
   * - ``num_electron``
     - ``Number of Charges``
     - ``int``
     - ``1``
     - 带符号电荷数 Z（q=Z e），并非电子数；为非零整数。
   * - ``reference_clock``
     - ``Reference clock``
     - ``ReferenceClock | None``
     - ``None``
     - 给定的回转频率函数；定义与默认规则见下文参考时钟小节。
   * - ``gamma_t``
     - ``Transition Gamma``
     - ``float``
     - ``7.635``
     - 晶格的渡越伽马因子 gamma_t。
   * - ``circumference``
     - ``Circumference (m)``
     - ``float``
     - ``569.1``
     - 正的环周长（m）。
   * - ``num_turns``
     - ``Number of turns``
     - ``int``
     - ``100``
     - 仿真圈数，正整数。
   * - ``backend``
     - ``Backend (gpu/cpu)``
     - ``str``
     - ``'cpu'``
     - 计算后端：cpu 或 gpu。
   * - ``particle_precision``
     - ``Particle Precision``
     - ``str``
     - ``'float64'``
     - 六维粒子坐标存储精度：float32 或 float64。
   * - ``num_gpu``
     - ``Number of GPU devices``
     - ``int``
     - ``1``
     - 使用的 GPU 数量。
   * - ``gpu_id``
     - ``Device Id``
     - ``list[int]``
     - ``[0]``
     - GPU 设备编号列表。
   * - ``output_dir``
     - ``Output directory``
     - ``str``
     - ``'./output'``
     - 输出目录；相对路径以输入 JSON 所在目录为基准。
   * - ``is_plot``
     - ``Is plot figure``
     - ``bool``
     - ``False``
     - 是否在跟踪结束后生成图形。
   * - ``timing``
     - ``Timing``
     - ``TimingConfig``
     - ``TimingConfig()``
     - 计时配置；默认 mode=command、log_interval=10、warmup_turns=1、include_io=True。
   * - ``is_beambeam``
     - ``Is beam-beam``
     - ``bool``
     - ``False``
     - 保留开关；当前未实现束束跟踪，须保持 False。


空间电荷使用独立的顶层 ``Space charge``
配置块。命名资源 schema 和 Sequence command 引用方式参见 :doc:`space_charge`。
每个资源通过 ``Method``（``pic``、``frozen``、``quasi-frozen``）和 ``Solver``
选择计算方式；Solver 名称包含边界条件。命令中的 ``Aperture type/value``
定义粒子损失孔径，并在 Dirichlet 中同时定义导体壁，
网格输入必须完整选择全宽或半宽一组；省略命令孔径时默认使用网格同尺寸矩形。

.. _zh-reference-clock:

规定的机器时钟与初始化
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

顶层可选 ``Reference clock`` 定义正值回转频率 :math:`f_{rev}(t)` 和基准时刻
:math:`t_*`：

.. math::

   \Psi(t)=\int_{t_*}^{t} f_{rev}(u)\,du.

输入为 ``Revolution frequency (Hz)``（标量或列表）、列表对应的 ``Time (s)``，
以及默认 0 的 ``Time origin (s)``。采样值分段线性插值，区间外保持端点值，
每段积分解析计算。该规定程序独立于实际跟踪束团的能量。
未指定时，PASS 固定使用 harmonic-id-zero 束团初始参考速度除以周长；
后续加速不会自动改变这个频率。

初始 :math:`T_b=\Psi^{-1}(-h_{id}/h_{group})`，也可由 BunchConfig 的
``Reference arrival time (s)`` 指定。第 n 圈注入使用
:math:`\Psi^{-1}(n-h_{id}/h_{group})`；显式初始到达时间与名义初始时间的差
平移该注入源的日程。注入粒子的 z 和归一化动量变换到目标束团参考系，保持物理
到达时间和机械动量。

``harmonic_id``、``harmonic_number`` 表示名义槽位。
名义槽位位置 ``harmonic_id*C/harmonic_number`` 在需要输出元数据时计算；
将它加到 z 不能重建实际位置或到达时间。RF 谐波与分组数相互独立。:doc:`reorganize` 说明如何用
规定时钟相位重分组，同时保留展开的粒子时间。

InjectionItem（注入与分组）
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``InjectionItem`` 在注入层统一声明 ``harmonic_number`` （JSON 键 ``Harmonic Number`` ）。该值是束团分组数，决定：

- 一圈内建立多少个束团中心，中心间距为 :math:`C/h_{\mathrm{group}}`
- ``bunches`` 列表必须包含多少个 ``BunchConfig``
- ``harmonic_id`` 必须唯一覆盖 :math:`0,\ldots,h_{\mathrm{group}}-1`

它不限制 ``RFComponent.harmonic`` 。未填充的分组应使用 ``num_macro_particles=0`` 的空束团占位。

当需要复现生成的粒子分布时，设置整数 ``random_seed`` （JSON 键 ``Random Seed`` ）。不设置或在 JSON 中设为 ``null`` 时采用默认的非确定性种子。该种子属于整个 Injection 命令，因此所有声明的束团和注入轮次共享同一随机数流。


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
   * - ``harmonic_id``
     - ``Harmonic ID of this bunch``
     - int
     - 束团分组编号；中心为 :math:`z_{\mathrm{center}}=h_{\mathrm{id}}C/h_{\mathrm{group}}`
   * - ``rf_s_position``
     - ``RF S Position Refer to Inj. Point (m)``
     - float
     - RF 腔相对注入点的位置，用于把匹配分布线性逆传输到 :math:`s=0`
   * - ``momentum_offset_dp``
     - ``Momentum Offset dp``
     - float
     - 束团平均相对动量偏差；与动能偏差互斥
   * - ``kinetic_energy_offset``
     - ``Kinetic Energy Offset (eV)``
     - float
     - 束团平均动能偏差，内部精确转换为相对动量偏差

``BunchConfig`` 中手动输入或生成的 ``z`` 坐标都是相对束团参考粒子到达时间的 :math:`z_{\mathrm{rel}}` ，不是实验室绝对方位。

Sequence（序列容器）
~~~~~~~~~~~~~~~~~~~~

``Sequence`` 是一个有序容器，存储所有按位置 ``s`` 排列的序列项。导出时按 ``(s, command priority)`` 排序；排序键相同时保留插入次序。执行时还会按位置容差归并邻近位置。

.. code-block:: python

   from PASS.para.schema.sequence import Sequence
   from PASS.para.schema.bunch import BunchConfig, InjectionItem
   from PASS.para.schema.elements import QuadrupoleItem
   from PASS.para.schema.monitors import StatMonitorItem

   bunch = BunchConfig(kinetic_energy=45e6, num_real_particles=100000000000,
                       num_macro_particles=2048, emit_x=2e-6, emit_y=2e-6)
   seq = Sequence()
   seq.add("injection", InjectionItem(s=0.0, bunches=[bunch]))
   seq.add("qd1", QuadrupoleItem(s=1.0, k1l=0.2, length=0.5))
   seq.add("stat1", StatMonitorItem(s=0.0))

支持的序列项类型：

- ``InjectionItem`` — 注入点（必须 ``s=0`` ）
- ``TwissItem`` — twiss 传输点
- ``DriftItem`` 、 ``QuadrupoleItem`` 、 ``SBendItem`` 等 — 物理元件
- ``StatMonitorItem`` 、 ``DistMonitorItem`` 、 ``PhaseAdvanceMonitorItem`` — 监视器


晶格来源
------------

PASS 支持三种晶格序列生成方式，可根据需要选择或混合使用：

方式一：从 MADX twiss 文件读取
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

读取 MADX 生成的 twiss TFS 文件，每个元件转为一个 ``TwissItem`` 传输点。适用于 **逐 twiss 传输** 模式。

.. code-block:: python

   from PASS.para.madx import read_madx_twiss

   items, names, circum = read_madx_twiss(
       twiss_file="lattice.tfs",
       error_file="errors.tfs",       # 可选
       muz=0.001,                      # 纵向 tune
       dqx=0.0,                        # 色品（或 "from_file"）
       dqy=0.0,
       is_field_error=False,           # 是否读取场误差
       insert_patterns=["QD.*"],      # 正则匹配，插入为薄透镜元件
   )

绝对场误差的单位、实例匹配规则，以及逐元件分布误差与 Twiss 出口的集中动量增量的区别，见 :ref:`zh-error`。

如需均匀基础网格，改用重采样读取函数：

.. code-block:: python

   from PASS.para.madx import read_madx_twiss_interpolated

   items, names, circum = read_madx_twiss_interpolated(
       twiss_file="lattice.tfs",
       num_interp_slice=101,          # 100 段，包含 0 和 C 两个端点
       dqx="from_file", dqy="from_file",
       longitudinal_transfer="off",
   )

该函数使用唯一支持的 ``interp_kind="phase_hermite"``。
``num_interp_slice`` 表示 **基础点数** 而非分段数，必须是至少为二的整数。
DQx/DQy 默认 ``"from_file"``，Mu z 默认零；纵向相位仅在
``longitudinal_transfer="matrix"`` 时使用。

对长度为 :math:`h` 的每个源区间，令 :math:`t=(s-s_i)/h`，用五次 Hermite 多项式
插值 :math:`p(t)=\mu(s)-\mu(s_i)`。以周为相位单位，两端导数约束为：

.. math::

   p_t = \frac{h}{2\pi\beta}, \qquad
   p_{tt} = \frac{h^2\alpha}{\pi\beta^2}.

插值光学函数由此计算：

.. math::

   \beta(s)=\frac{h}{2\pi p_t},\qquad
   \alpha(s)=\frac{p_{tt}}{4\pi p_t^2}.

这在每个区间内保持 :math:`\beta'=-2\alpha`、:math:`\mu'=1/(2\pi\beta)`，
同时在端点匹配源 beta、alpha 和累计相位。相位导数在两端及全部内部极值点接受检查，
排除非正 beta。DX/DPX 使用成对的三次 Hermite 插值，沿用无耦合、参考轨道近轴约定
及 TFS 色散归一化，不进行耦合或闭轨坐标转换。源数据精度与间距限制插值精度。

表中必须包含 S=0、S=LENGTH，光学量有限、beta 为正、相位保持展开。
完整相位差与 Q1/Q2 的核对允许 TFS 输出舍入误差：相对容差
:math:`2\times10^{-8}`，绝对容差 :math:`2\times10^{-9}`。
输出相位保留源值，不按舍入后的头信息重新缩放。分段色品按源相位差的比例分配，
保持用户指定的 DQx/DQy 总和。

原始行不作为额外点保留。匹配的薄元件、场误差和重复 S 处的光学跳变增加必要拆分位置。
场误差名称在重采样前与原始表匹配。光学跳变通过零长度 Twiss 传输连接入端和出端状态。
同一位置的附加薄元件或场误差在 Twiss 之后执行，与 command 优先级一致。
若显式插入源光学已包含的设计聚焦，会再次叠加该作用；插入误差不会自动保持受扰机器
的工作点。对应导入控件见 :doc:`gui`。

方式二：从 MADX twiss 文件读取为元件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

读取 twiss 文件，但每个元件转为对应的物理元件对象（ ``QuadrupoleItem`` 、 ``SBendItem`` 等）。适用于 **逐元件跟踪** 模式。

.. code-block:: python

   from PASS.para.madx import read_madx_elements

   items, names, circum = read_madx_elements(
       twiss_file="lattice.tfs",
       is_merge_drift=True,            # 合并相邻漂移节
       is_field_error=True,
       is_alignment_error=True,
       error_file="errors.tfs",
   )

两个误差开关独立，均默认为 ``False``。准直从误差 TFS 读取 ``DX``、``DY``、``DPSI``，
只移动磁场，孔径和 SC 边界保持固定；非零且不支持的准直分量明确报错。
Twiss 传输及重采样 Twiss 导入拒绝准直误差，应使用这里的逐元件模式。
归一化、实例匹配和执行顺序见 :ref:`zh-error`。
``generate_from_tfs`` 同样接受 ``is_alignment_error``。

方式三：平滑近似 twiss
~~~~~~~~~~~~~~~~~~~~~~

无需 MADX 文件，用解析公式生成恒定 β 函数的 twiss 点。 :math:`\beta = C / (2\pi Q)` 。适用于快速测试。

.. code-block:: python

   from PASS.para.smooth import generate_smooth_twiss

   items, names, circum = generate_smooth_twiss(
       circumference=569.1,
       qx=9.47, qy=9.43,
       num_points=100,
       longitudinal_transfer="off",
   )

混合模式
~~~~~~~~

twiss 传输点和物理元件可以在同一个序列中混合使用。例如在 twiss 序列中插入一个 RF 腔：

.. code-block:: python

   from PASS.para.schema.elements import RFCavityItem

   from PASS.para.schema.sequence import Sequence
   from PASS.para.schema.bunch import BunchConfig, InjectionItem
   from PASS.para.schema.elements import QuadrupoleItem
   from PASS.para.schema.monitors import StatMonitorItem

   bunch = BunchConfig(kinetic_energy=45e6, num_real_particles=100000000000,
                       num_macro_particles=2048, emit_x=2e-6, emit_y=2e-6)
   seq = Sequence()
   seq.add("injection", InjectionItem(s=0.0, bunches=[bunch]))

   from PASS.para.smooth import generate_smooth_twiss
   twiss_items, twiss_names, circumference = generate_smooth_twiss(251.327, 4.8, 4.4, 17)

   # Twiss 传输点
   for i, item in enumerate(twiss_items):
       seq.add(f"twiss_{i:04d}", item)

   # 插入 RF 腔（在 s=0 处）
   seq.add("rf1", RFCavityItem(s=0.0, components=[dict(voltage=100e3, harmonic=1, phase=0.5236)]))


外部数据文件转换
----------------

PASS 使用 **TFS 格式** 作为所有 ramping/RF/exciter 数据文件的统一格式。 ``tools/data_converter.py`` 提供了通用转换流水线，将各种外部文件（CSV/TXT/TFS）转为 PASS TFS。

RF 文件保留物理秒，不使用下面磁铁 ramping 的圈号转换管线。RF 列为 ``TIME, VOLTAGE, FREQUENCY, PHASE``，接口见 :doc:`element/rfcavity`。

以下磁铁 ramping 转换函数仅用于准备数据表。当前跟踪引擎不支持启用磁铁元件的 ramping；生成数据表不会启用该功能。RFCavity 支持以物理时间为自变量的 RF 数据表。

四步流水线
~~~~~~~~~~

.. code-block:: text

   外部文件 → load_raw_data → time_to_turn → interpolate → write_tfs

1. **load_raw_data** ：读取外部文件，自动检测 turn/time 列
2. **time_to_turn** ：如外部文件给的是时间而非圈数，用回转频率转换
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
       revolution_freq=1.76e6,             # 回转频率 (Hz)
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
   convert_rf_data("llrf.csv", "rf_physical_time.tfs")

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


参数扫描与校验
--------------

``model_copy(update=...)`` 不会重新校验更新值。参数扫描应从字段字典重新构造配置对象，再对生成的完整输入执行校验：

.. code-block:: python

   from PASS.para.schema.main import MainConfig

   baseline = MainConfig(circumference=251.327)
   candidate = {**baseline.model_dump(), "num_turns": 128}
   scan_config = MainConfig.model_validate(candidate)

Python 字段 ``num_electron`` 沿用历史命名，实际表示带符号电荷数，并非束缚电子数量。因此 ``num_proton=1, num_neutron=0, num_electron=1`` 表示质子。Python 字段名用于配置对象构造，JSON 文件使用表中的别名。

架构概览
--------

参数模块分别负责输入对象的构造、校验与序列化：

.. code-block:: text

   PASS/para/
   ├── schema/       参数定义（字段定义与别名）
   │   ├── main.py         MainConfig：全局仿真参数
   │   ├── bunch.py        BunchConfig + OffsetConfig + InjectionItem
   │   ├── twiss.py        TwissItem：twiss 传输点
   │   ├── elements.py     元件配置类
   │   ├── monitors.py     StatMonitor / DistMonitor / PhaseAdvanceMonitor
   │   ├── space_charge.py SpaceChargeConfig + SpaceChargeResourceConfig + SpaceCharge
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
         schema/ (pydantic)     ← 字段定义与别名：验证 + 别名
              │
              ▼
        api.py (generate_input) → beam0.json
              │
              ▼
         PASS 引擎 (Config → Beam → CommandSequence → Executor)


API 入口
------------

``PASS.para.api`` 提供 ``generate_input``、``build_sequence``、``generate_from_tfs`` 和 ``load_input``。``load_input(path)`` 返回 ``(MainConfig, raw_sequence_dict)``，不会重建带类型的命令对象，也不返回顶层 Space charge/Wake field 配置块。编辑现有完整文件时应单独保留这些配置块。
