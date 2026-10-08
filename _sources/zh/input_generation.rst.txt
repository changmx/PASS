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
   pass-run input/beam0.json

安装 PASS 会在对应 Python 环境中注册 ``pass-run``。
安装后，``python -m PASS`` 接受完全相同的参数，并明确使用所选 Python 解释器。
两种入口都不需要 Qt 或 PySide6。安装后执行 ``pass-run --version`` 或
``python -m PASS --version`` 查看版本号并退出，无需提供输入文件。
程序根据文件后缀识别输入类型：

.. code-block:: console

   pass-run input/beam0.json
   pass-run input/beam0.json input/beam1.json
   pass-run example.passproj
   python -m PASS example.passproj

命名输入选项也支持相同的三种运行方式：

.. code-block:: console

   pass-run --beam0 input/beam0.json
   pass-run --beam0 input/beam0.json --beam1 input/beam1.json
   pass-run --passproj example.passproj

位置参数和命名输入选项只能选择其中一种，不能混用。
``--beam0`` 和 ``--beam1`` 要求 JSON 文件，``--beam1`` 必须与 ``--beam0`` 一起使用。
``--passproj`` 要求 ``.passproj`` 文件，不能与任何束流选项组合使用。
相对输入路径以终端当前工作目录为基准，包含空格的路径需要加引号。

一个或两个 JSON 分别用于单束流或双束流运行。单个 ``.passproj`` 文件使用已保存的
Beam 0 和可选 Beam 1 选择；未保存 Beam 0 时使用当前活动配置，
不会依次执行项目中的全部配置。已保存的选择失效或两束选择重复时会报错。
项目选择和输出路径规则见 :doc:`project_files`。

使用 ``--output DIR`` 指定本次运行的输出根目录，优先于 JSON 或项目保存的设置，
支持单份 JSON、双份 JSON 或一个项目。命令行指定的相对输出路径以启动时的工作目录
为基准，原始 JSON 和项目文件保持不变：

.. code-block:: console

   pass-run input/beam0.json --output results
   pass-run input/beam0.json input/beam1.json --output "results/two beams"
   python -m PASS example.passproj --output results

不使用此选项时，JSON 运行使用第一份 JSON 的 ``Output directory``，
配置内的相对路径以该 JSON 所在目录为基准。字段缺省或设为 ``default``
（不区分大小写）时，输出根目录为第一份 JSON 旁的 ``output``。
生成输入的默认值仍为 ``./output``，不再回退到仓库或安装目录。

以下两条帮助命令均显示输入方式、选项、路径规则和示例，不启动仿真：

.. code-block:: console

   pass-run --help
   pass-run -h

``--stop-file`` 指定的文件存在时，在初始化前或圈边界停止。
退出码 0 表示完成，1 表示失败，2 表示命令参数错误，3 表示请求停止，
130 表示中断。Python 集成也可直接调用
``PASS.main.main('input/beam0.json', raise_errors=True)``。
Python 集成可使用
``PASS.main.main('input/beam0.json', output_dir='results', raise_errors=True)``
覆盖输出根目录；相对 ``output_dir`` 路径以调用 ``main()`` 时的工作目录为基准。
此覆盖要求保留默认的 ``archive_inputs=True``，不能与供已有快照使用的
``archive_inputs=False`` 组合。

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

执行所用的输入快照
------------------

``PASS.main.main(beam0_path, beam1_path=None, ...)`` 默认归档输入
（``archive_inputs=True``）。检查原始配置后，复制 JSON 配置及引用的输入文件，
校验生成的快照，再从快照初始化跟踪。依赖包括粒子分布、RF 程序、偏移表、
尾场模型和磁铁 ramping 表。之后修改原始文件不会影响本次运行。

JSON、已保存项目和 GUI 运行默认统一使用按日期组织的结果目录，
输入快照保存在本次结果目录的 ``input`` 子目录中：

.. code-block:: text

   <output>/<YYYY_MMDD>/<HHMM_SS>/       # 本次仿真结果
       input/
           configuration0.json         # 原始配置值
           beam0.json                  # 执行输入；可选 beam1.json
           assets/<filename>           # 已复制的输入依赖
           run.json                    # 路径、SHA-256 校验和及运行状态

复制输入前先创建本次结果目录；必要时为时间目录添加后缀，以避免复用已有结果目录。
运行 ID 仍保存在 ``run.json`` 中，不再单独作为一层目录。
已复制的依赖直接放在 ``assets/`` 中，不创建编号子目录。保留原文件名；
检查重名时不区分大小写，发生冲突就在扩展名前依次追加 ``1``、``2`` 等编号，
例如 ``rf.tfs``、``rf1.tfs``、``rf2.tfs``。执行 JSON 和 ``run.json``
记录实际使用的文件名。

双输入运行还保存 ``configuration1.json``。执行 JSON 中的文件引用使用快照内的
相对路径 ``assets/...``，因此可以整体移动输入快照目录。
输出根目录（包括 ``--output`` 或 Python ``output_dir`` 的覆盖值）在复制前解析，
并以绝对 ``Output directory`` 保存到执行快照；移动快照不会改变结果目的地。
覆盖同时改变快照和结果的输出根目录，保留上述布局。原始 JSON 和输入文件保持不变。
``configurationN.json`` 重新序列化原始配置值；依赖文件则逐字节原样复制。
GUI 与命令行项目运行使用相同的依赖复制、哈希和结果目录规则，见 :doc:`project_files`。

结果目录中的参数 JSON 保留原有文件名，并保存绝对输入路径和输出路径。
它从已经载入的配置生成，使用解析路径后、展开命名配置前的内容，不再次读取源 JSON。
主快照中的 ``beamN.json`` 仍使用相对依赖路径。

格式版本为 1 的 ``run.json`` 记录执行 JSON、比较用配置及可用依赖的 SHA-256 校验和。
依赖记录还包括原始来源路径和复制后的字节数。
未启用资源的缺失文件保留原有校验警告，并列入 ``unavailable_dependencies``；
已启用功能所需输入缺失则阻止运行。缺失引用指向快照内未创建的路径，
之后恢复原始文件也不会使其成为未经归档的运行输入。记录随准备及执行过程更新状态。
``output_root`` 保存选定的输出根目录，``output_directory`` 保存写入执行配置的输出路径。
``results_directory`` 从准备阶段起保存已分配的结果目录，包括日期子目录。
``on_initialized(cfg)`` 回调可通过 ``cfg.input_snapshot_path`` 定位本次运行的 ``run.json``。
准备失败时，若记录已创建，则状态记为 ``preparation_failed``。

需要最终粒子状态来生成结果摘要时，``main()`` 还接受 ``on_completed(sim)``。
该回调在跟踪成功结束后、运行记录标记为完成前接收 ``Simulation``，
停止或失败的运行不会调用它。回调抛出异常时，运行记为失败，
并遵循通常的 ``raise_errors`` 行为。复用命令行解析器的集成程序可通过
``cli_main(argv, on_completed=callback)`` 传入同一回调；
例如，注入涂抹示例的运行包装脚本用它生成 ``completed.json``。

对已保存的 ``beam0.json`` 调用 ``main()`` 会从初始条件重新运行，
并创建新的输入快照及结果目录。这是输入复现，不是检查点续算。
输入快照不会恢复之前的 Python 环境、源码版本或随机数生成器状态；
``Random Seed: null`` 仍采用非确定性采样。

已有输入快照的集成程序可用 ``archive_inputs=False`` 关闭再次复制；GUI 子进程使用此设置。
依赖自动归档由 ``main()`` 负责，底层 ``Config.load_input()`` 本身不归档依赖。
Python API 显式设置 ``flat_output=True`` 时，为兼容已有集成保留原有布局：
结果仍直接写入指定输出目录，
快照则位于该目录的父目录下的 ``input_snapshots/<run-id>``。
若结果目录本身名为 ``input_snapshots``，则改用旁边的 ``input_snapshots_archive/<run-id>``，
使 flat 结果目录不包含快照子目录。

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
     - 输出目录；相对路径以输入 JSON 所在目录为基准。缺省或 ``default``（不区分大小写）时使用第一份 JSON 旁的 ``output``。
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
     - 已停用的占位字段；true 报错，生成的输入不再输出此字段。

束束相互作用使用独立顶层 ``Beam beam`` 配置块和两条 sequence 中的显式命令。
调用 ``generate_input`` 时传入 ``beam_beam=BeamBeamConfig(...)``；
源方法、执行顺序和相关元件见 :doc:`beam_beam`。


空间电荷使用独立的顶层 ``Space charge``
配置块。命名资源 schema 和 Sequence command 引用方式参见 :doc:`space_charge`。
每个资源通过 ``Method``（``pic``、``frozen``、``quasi-frozen``）和 ``Solver``
选择计算方式；Solver 名称包含边界条件。命令中的 ``Aperture type/value``
定义粒子损失孔径，并在 Dirichlet 中同时定义导体壁，
网格输入必须完整选择全宽或半宽一组；省略命令孔径时默认使用网格同尺寸矩形。

电子云使用独立顶层 ``Electron cloud`` 配置块。
调用 ``generate_input`` 时传入 ``electron_cloud=ElectronCloudConfig(...)``，
通过 ``ElectronCloudItem`` 引用命名配置。``frozen`` 不要求 Slicer；
外驱动 ``build_up`` 与 PIC ``coupled`` 均要求当前 ``z_rel`` SliceSet
和嵌套 ``ElectronCloudBuildUpConfiguration``；耦合模式包含云自场和横向束流响应。
物理约定、参数与限制见 :doc:`electron_cloud`。

束内散射使用独立顶层 ``Intrabeam scattering`` 配置块。
调用 ``generate_input`` 时传入 ``intrabeam_scattering=IBSConfig(...)``，
通过 ``IBSItem`` 引用命名模型。高斯增长率诊断和动力学踢要求显式局部
``IBSOpticsConfig``；二体碰撞使用局部三维网格。
增长率、作用时间和模型限制见 :doc:`ibs`。

.. _zh-reference-clock:

自动设计时钟与初始化
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PASS 在跟踪前，根据 harmonic ID=0 束团的初始能量、粒子质量与电荷、环周长，
以及各实际位置处启用的 RFCavity 波形，构造共同的理想纯 RF 设计轨迹。
同一位置的 RF 分量在同一个设计通过时刻采样并合计能量增量；飞行时间采用前一踢
之后的参考速度。设计回旋频率为 :math:`f_{rev}=\beta_{design}c/C`，累计相位
固定以物理时刻零为原点：

.. math::

   \Psi(t)=\int_0^t f_{rev}(u)\,du.

在圈边界和 RF 通过前记录频率节点，在物理时间上分段线性插值，区间外保持端点值，
每段积分解析计算。RF 踢只计算请求的圈数；之后以最终设计能量增加一次设计通过，
封闭插值区间，不再施加 RF 踢。没有活动 RF 电压时，设计时钟保持常频。集体效应踢角、粒子损失
及跟踪束团能量的变化不会重新定义这条理想设计时钟。输入 RF 相位仍是波形的附加
相位调制，不会被解释成要求设计粒子达到的同步相位。

不再提供公开 ``Reference clock`` 输入或 ``ReferenceClock`` schema；旧输入需删除该字段。
仅由初始能量和 RF 数据，一般无法唯一反推出任意外部规定时钟。外部信号源给定实际
RF 频率时，应使用分量的 ``Frequency (Hz)`` 或文件 ``FREQUENCY`` 列，并保留其
积分相位，包括原有非零时间原点。:doc:`element/rfcavity` 中的 HIAF 转换器执行这种迁移。
重构已有设计轨迹属于模型计算，不保证频率、能量采样逐位相同；应显式比较通过时间、
相位和能量。

初始 :math:`T_b=\Psi^{-1}(-h_{id}/h_{group})`，也可由 BunchConfig 的
``Reference arrival time (s)`` 指定。第 n 圈注入使用
:math:`\Psi^{-1}(n-h_{id}/h_{group})`；显式初始到达时间与名义初始时间的差
平移该注入源的日程。注入粒子的 z 和归一化动量变换到目标束团参考系，保持物理
到达时间和机械动量。

``harmonic_id``、``harmonic_number`` 表示名义槽位。
名义槽位位置 ``harmonic_id*C/harmonic_number`` 在需要输出元数据时计算；
将它加到 z 不能重建实际位置或到达时间。RF 谐波与分组数相互独立。:doc:`reorganize` 说明如何用
自动设计时钟相位重分组，同时保留展开的粒子时间。

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

两个 Twiss 读取函数都会将原生 MAD-X 的动量导数转换为 PASS 的
:math:`\delta=P/P_0-1` 约定。MAD-X ``TWISS`` 使用
:math:`\mathrm{PT}=(E-E_0)/(P_0c)\simeq\beta_0\delta`，因此导入的
``DX``、``DPX`` 以及从文件读取的 ``DQ1``/``DQ2`` 均乘以 :math:`\beta_0`。
显式数值参数 ``dqx``/``dqy`` 已表示 :math:`dQ/d\delta`，保持不变；
原始 TFS 表本身保留原单位。使用文件色品时应以 ``twiss, chrom`` 导出。

这些读取函数要求原生 ``TWISS`` 输出。对于可识别的 ``PTC_TWISS`` 表会明确拒绝：
其导数约定取决于 PTC 的 ``TIME`` 模式，而 TFS 头信息未记录该模式；
相位列也可能不同。此传输模型应使用原生 ``TWISS, CHROM`` 表。
该单位换算不意味着原生 TWISS 与精确 PTC 元件模型等价。

参考速度优先由 ``GAMMA`` 计算，缺少时由 ``ENERGY/MASS`` 计算。
两种信息均缺失时，为兼容旧表仍允许读取，但发出警告并采用超相对论近似
:math:`\beta_0=1`；非超相对论束流应补充参考能量头信息。
无效的参考参数会被拒绝。已有输入需要重新生成才能应用此修正；
800 MeV 动能质子的换算因子约为 0.84181。定义参见
`MAD-X 坐标与导数约定
<https://cds.cern.ch/record/2928179/files/document.pdf>`_。

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
排除非正 beta。DX/DPX 使用成对的三次 Hermite 插值，沿用无耦合、参考轨道近轴约定。
插值内部保持源单位，构建 ``TwissItem`` 时将色散转换为对 :math:`\delta` 的导数。
不进行耦合或闭轨坐标转换。源数据精度与间距限制插值精度。

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

磁铁 ramping、RF 和 exciter 输入采用 TFS 表。四极、六极、八极和多极铁读取以
物理秒为自变量的归一化强度；每个非空 bunch 在元件入口采样一次，整个元件使用该强度。
支持的列和采样模型见 :doc:`element/magnet_ramping`。
RF 使用独立的 ``TIME, VOLTAGE, FREQUENCY, PHASE`` 接口，见 :doc:`element/rfcavity`。

生成磁铁程序
~~~~~~~~~~~~

可以直接写入时间断点表，跟踪时进行分段线性插值，并在表格区间之外保持端点值：

.. code-block:: python

   from PASS.para.tools.ramping import write_magnet_ramping
   from PASS.para.schema.elements import QuadrupoleItem

   write_magnet_ramping(
       "quadrupole_ramp.tfs",
       times=[0.0, 0.05, 0.10],               # physical seconds
       columns={"K1L": [0.20, 0.25, 0.22], "K1SL": [0.01, 0.0, -0.01]},
   )
   quad = QuadrupoleItem(
       s=10.0, length=0.5, num_slices=8,
       is_ramping=True, ramping_file="quadrupole_ramp.tfs",
   )

写入器校验表格，并加入 ``TIME_UNIT="s"``、
``STRENGTH_CONVENTION="normalized"`` 和强度单位元数据。
输入是绝对强度而非倍率。非积分 ``K1``、``K2`` 等列要求正的磁铁长度；
积分 ``K1L``、``K2L`` 等列也支持薄透镜。正、斜分量可以独立指定，
同一分量不可同时提供 K 和 KL。

转换外部文件
~~~~~~~~~~~~

专用转换器读取 CSV、空白分隔 TXT 或 TFS。将源列名映射为所需的强度列名，
并明确指定时间单位换算。转换保留原始采样时刻，不转为圈号或重采样到逐圈网格：

.. code-block:: python

   from PASS.para.tools.ramping import convert_magnet_ramping

   convert_magnet_ramping(
       input_path="external_ramp.csv",
       output_path="quadrupole_ramp.tfs",
       time_column="time_ms",
       time_scale=1e-3,                       # milliseconds -> seconds
       column_mapping={"normal": "K1L", "skew": "K1SL"},
       delimiter=",",
   )

对于特殊表头或需要跳过行的文件，可使用同一模块的
``read_magnet_ramping_source(input_path, delimiter=None, header=0, skiprows=0)``，
提取所需数值列后调用 ``write_magnet_ramping``。
源 TFS 的时间单位元数据必须与选定的时间换算一致；若提供强度单位元数据，也会校验。
强度不会从实际磁场隐式换算。

GUI 在 **工具 → 数据转换 → 磁铁 ramping…** 中提供导入转换和时间断点生成，
见 :doc:`gui_tools`。导出后，在支持的元件中启用 ramping 并选取文件。

旧 ``convert_external_to_tfs``、``convert_k1l_ramping`` 等辅助函数保留历史圈号表
行为。只有包含实际物理秒的旧 ``TIME_S`` 列才可以被跟踪器接受；仅含 ``TURN``
的表不能驱动磁铁 ramping。加速过程中不能由行号或瞬时回转频率推断累计时间。
新输入应使用上述物理时间写入器或转换器。

RF 文件使用独立接口：

.. code-block:: python

   from PASS.para.tools.rf_data import convert_rf_data

   convert_rf_data("llrf.csv", "rf_physical_time.tfs")


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
   │   ├── electron_cloud.py ElectronCloudConfig + ElectronCloudConfiguration + ElectronCloudItem
   │   ├── ibs.py      IBSConfig + IBSConfiguration + IBSOpticsConfig + IBSItem
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

``PASS.para.api`` 提供 ``generate_input``、``build_sequence``、``generate_from_tfs`` 和 ``load_input``。``load_input(path)`` 返回 ``(MainConfig, raw_sequence_dict)``，不会重建带类型的命令对象，也不返回顶层 Space charge/Wake field/Electron cloud/Intrabeam scattering 配置块；返回前会校验电子云和 IBS 配置及其引用。编辑现有完整文件时应单独保留这些配置块。
