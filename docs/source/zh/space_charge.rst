空间电荷效应（SpaceCharge）
======================================

``SpaceCharge`` 根据用户提供的纵向切片计算横向空间电荷动量增量。支持 PIC、固定解析分布（``frozen``）及按当前切片统计量更新的解析分布（``quasi-frozen``），可在 CPU 或 NVIDIA GPU 上执行。模型采用准静态 2.5D 近似，计入电场力与磁场力的相对论抵消，不计算纵向空间电荷力。

配置前提
------------------

先在顶层 ``Space charge`` 启用计算并定义命名配置，再执行该配置引用的 ``Slicer``。空间电荷要求 ``Coordinate="z_periodic"``；重分组后必须重新切片。用独立命令或元件内部配置指定作用长度，独立命令本身不将粒子传输该长度。坐标约定见 :ref:`zh-longitudinal-reference`。下面为输入片段，还需提供注入及输运序列。

配置示例
--------

空间电荷配置在顶层 ``Space charge`` 块中声明一次，Sequence 命令通过名称引用：

.. code-block:: json

   {
       "Backend (gpu/cpu)": "cpu",
       "Space charge": {
           "Enabled": true,
           "Configurations": {
               "round_pipe": {
                   "Slice set": "space_charge",
                   "Nx": 129,
                   "Ny": 129,
                   "Grid Width X (m)": 0.08,
                   "Grid Width Y (m)": 0.08,
                   "Method": "pic",
                   "Solver": "fd_dirichlet",
                   "Particle Deposition Method": "CIC"
               }
           }
       },
       "Sequence": {
           "sc_slicer": {
               "S (m)": 10.0,
               "Command": "Slicer",
               "Slice set": "space_charge",
               "Coordinate": "z_periodic",
               "Slice model": "equal_length",
               "Number of slices": 64,
               "Z range mode": "auto"
           },
           "sc_kick": {
               "S (m)": 10.0,
               "Command": "SpaceCharge",
               "Configuration": "round_pipe",
               "SC length (m)": 0.10,
               "Aperture type": "circle",
               "Aperture value": [0.035],
               "Save field": false,
               "Save potential": false,
               "Save density": false,
               "Save turns": []
           }
       }
   }

程序只构建被 ``SpaceCharge`` 序列项或元件内部 ``Space charge`` 对象引用的配置。
内部配置、中点调度、支持的元件和示例见 :ref:`zh-internal-space-charge`。
同一 beam 内，相同配置名的
多个命令共享网格。Dirichlet 求解器按配置和解析后的命令孔径缓存：相同管壁复用
分解，不同管壁创建独立求解器。FFT 核不随损失孔径改变，可以共享。不同配置名
始终拥有独立资源，即使字段相同。

每个命令定义自己的孔径，例如：

.. code-block:: json

   {
       "sc_wide": {
           "Command": "SpaceCharge", "S (m)": 2.5,
           "Configuration": "default", "SC length (m)": 0.1,
           "Aperture type": "circle", "Aperture value": [0.006]
       },
       "sc_narrow": {
           "Command": "SpaceCharge", "S (m)": 7.5,
           "Configuration": "default", "SC length (m)": 0.1,
           "Aperture type": "circle", "Aperture value": [0.003]
       }
   }

以上是 Sequence 片段，两个点引用同一个已定义的 ``default`` 配置，
半径分别为 6 mm 和 3 mm。对于 ``fd_dirichlet``，它们也是不同的导体壁，因此
两个求解器共享同一网格。对于 FFT 或解析方法，它们仅影响损失。这些圆孔径
不能用于 ``dst_dirichlet``。

跟踪方法与解析分布
------------------

``Method`` 为 ``pic``（默认）、``frozen`` 或 ``quasi-frozen``。
``Solver`` 同时指定场算法或分布模型及其边界条件：

.. list-table:: 支持的组合
   :header-rows: 1
   :widths: 22 48 30

   * - Method
     - Solver
     - 模型
   * - ``pic``
     - ``fft_free_space``
     - 开放边界 Green 函数 PIC。
   * - ``pic``
     - ``fd_dirichlet``
     - 零电势导体；支持已实现的连续边界几何。
   * - ``pic``
     - ``dst_dirichlet``
     - 零电势导体；仅完整且与网格对齐的矩形。
   * - ``frozen``、``quasi-frozen``
     - ``gaussian_round_free_space``、``gaussian_ellipse_free_space``
     - 圆形高斯或椭圆高斯（Bassetti--Erskine）。
   * - ``frozen``、``quasi-frozen``
     - ``uniform_round_free_space``、``uniform_ellipse_free_space``
     - 均匀圆盘或椭圆，包括束流外部场。
   * - ``frozen``、``quasi-frozen``
     - ``parabolic_round_free_space``、``parabolic_ellipse_free_space``
     - 抛物型实空间密度，对应均匀四维水袋的投影。

``frozen`` 中，同一配置的所有切片共用固定的横向中心、尺寸和方向。
省略中心和角度时取零；对应 solver 的尺寸参数必填。不同作用位置可以引用不同配置。
切片电荷和 ``delta_z`` 仍采用当前输入，因此冻结的是横向形状，并非完整电磁场。

``quasi-frozen`` 在每次空间电荷作用前，对每个切片当前存活且具有有效切片编号的粒子，
重新统计质心和总体协方差。统计分母使用 :math:`N`，不是 :math:`N-1`；
同一束团宏粒子的物理权重相同。源电荷为 :math:`Q_k=N_k R Z e`。

椭圆模型的协方差特征向量确定主轴。高斯尺寸为特征值平方根；均匀分布半轴为
这些 RMS 尺寸的两倍；抛物型分布半轴为 RMS 尺寸的 sqrt(6) 倍。
输出的第一个主轴尺寸为较大者，主轴逆时针角度按 pi 周期
报告到 :math:`[-\pi/2,\pi/2)`。先平移、旋转到主轴坐标计算场，再转回原横向坐标。

圆形模型采用等径向二阶矩规则：

.. math::

   \sigma^2=\frac{\operatorname{Var}(x)+\operatorname{Var}(y)}{2},
   \qquad R=2\sigma.

``gaussian_round_free_space`` 使用 sigma，``uniform_round_free_space`` 使用 R。
``parabolic_round_free_space`` 使用 :math:`R=\sqrt6\,\sigma`。
此规则保留相对质心的径向二阶矩；对于非圆分布，它是用户选择的圆对称近似，
不保证重建实际电场。选择均匀模型也不表示粒子已成为 KV 分布。抛物型模型假定
投影密度保持抛物型，即使演化后的相空间分布已不再是严格水袋。程序不自动切换模型。
密度归一化及完整束内、束外抛物型场见 :doc:`field_solver`。

空切片的电荷和场为零。非空 quasi-frozen 圆形切片至少需要两个粒子且径向方差为正；
椭圆切片至少需要三个粒子且协方差非退化。具体要求为较小特征值严格大于
``64 * float64_epsilon * 较大特征值``。尺寸或协方差无效时报告切片编号，
不会静默跳过带电切片或代入最小尺寸。这只是数值有效性检查，并不保证统计采样充分。

当前仅实现自由空间解析分布和横向 2.5D 力，不包含导体壁解析修正或纵向空间电荷力。
明显的晕、多峰或非高斯结构通常不能由这些少量横向矩完整表示。

例如，可在 ``Space charge.Configurations`` 中加入：

.. code-block:: json

   {
       "fixed_gaussian": {
           "Method": "frozen",
           "Solver": "gaussian_ellipse_free_space",
           "Slice set": "space_charge",
           "Center X (m)": 0.001,
           "Center Y (m)": 0.0,
           "Sigma X (m)": 0.004,
           "Sigma Y (m)": 0.002,
           "Angle (rad)": 0.2
       },
       "updated_gaussian": {
           "Method": "quasi-frozen",
           "Solver": "gaussian_ellipse_free_space",
           "Slice set": "space_charge"
       }
   }

普通 ``SpaceCharge`` 序列命令通过 ``Configuration`` 引用其中一个配置。
解析粒子横向动量增量 不沉积电荷、不回插场。网格几何定义默认损失孔径以及可选的场和
密度采样范围。显式指定孔径或 ``off`` 后，改变诊断网格不会改变跟踪；使用默认
孔径时，改变网格范围可能改变粒子损失。

接口参数
--------

顶层 ``Space charge`` 配置块
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - 参数名
     - 键名
     - 类型
     - 默认值 / 是否必填
     - 说明
   * - ``enabled``
     - ``"Enabled"``
     - bool
     - ``false``
     - 是否为该 beam 输入启用空间电荷。为 false 时忽略配置内容和序列命令，
       且不构建资源。
   * - ``configurations``
     - ``"Configurations"``
     - object
     - ``{}``
     - 从非空用户配置名到 ``SpaceChargeResourceConfig`` 对象的映射。

命名资源配置
~~~~~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - 参数名
     - 键名
     - 类型
     - 默认值
     - 说明
   * - ``slice_set``
     - ``Slice set``
     - str
     - ``space_charge``
     - 输入的束团级 SliceSet 名称，不检查 Slicer 序列或执行历史。
   * - ``method``
     - ``Method``
     - str
     - ``pic``
     - pic、frozen 或 quasi-frozen。
   * - ``solver``
     - ``Solver``
     - str
     - ``fd_dirichlet``
     - 上述 Method/Solver 支持组合之一。
   * - ``nx / ny``
     - ``Nx / Ny``
     - int
     - ``128``
     - PIC 或诊断网格节点数，各至少为 3。
   * - ``grid_width_x / grid_width_y``
     - ``Grid Width X (m) / Grid Width Y (m)``
     - float or null
     - 两组均省略时为 ``0.02``
     - 正的网格全宽，中心为零；两个方向必须成对提供。
   * - ``grid_half_width_x / grid_half_width_y``
     - ``Grid Half Width X (m) / Grid Half Width Y (m)``
     - float or null
     - ``null``
     - 正的网格半宽；与全宽二选一，不可混用，两个方向必须成对提供。
   * - ``deposition_method``
     - ``Particle Deposition Method``
     - str or null
     - ``null``
     - 仅 PIC：null 使用 CIC；支持 CIC/TSC 及配对回插。
   * - ``center_x / center_y``
     - ``Center X (m) / Center Y (m)``
     - float or null
     - ``null``
     - 仅 frozen；null 表示零；所有切片共用固定中心。
   * - ``angle``
     - ``Angle (rad)``
     - float or null
     - ``null``
     - frozen 椭圆局部 x 轴的逆时针角度；null 表示零；圆形只允许零或 null。
   * - ``sigma``
     - ``Sigma (m)``
     - float or null
     - ``null``
     - frozen 圆形高斯必填的正 RMS 尺寸。
   * - ``sigma_x / sigma_y``
     - ``Sigma X (m) / Sigma Y (m)``
     - float or null
     - ``null``
     - frozen 椭圆高斯必填的正主轴 RMS 尺寸。
   * - ``radius``
     - ``Radius (m)``
     - float or null
     - ``null``
     - frozen 均匀或抛物型圆盘必填的正支撑半径。
   * - ``a / b``
     - ``Semi-axis A (m) / Semi-axis B (m)``
     - float or null
     - ``null``
     - frozen 均匀或抛物型椭圆必填的正支撑半轴。


``SpaceCharge`` 序列命令
~~~~~~~~~~~~~~~~~~~~~~~~

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
     - ``"SpaceCharge"``
     - 选择空间电荷命令实现。
   * - ``s``
     - ``"S (m)"``
     - float
     - 0.0 m
     - 薄透镜动量更新的环上纵向位置，必须为有限值。
   * - ``name``
     - Sequence 对象键名
     - str
     - 作为 Sequence 键必填
     - 命令实例名，用于日志和输出路径。
   * - ``configuration``
     - ``"Configuration"``
     - str
     - 必填
     - 顶层 ``Configurations`` 中的精确名称；禁止首尾空白。
   * - ``sc_length``
     - ``"SC length (m)"``
     - float
     - 0.0 m
     - 非负有效作用长度。为零时该命令不更新横向动量，也不产生快照。
       若损失孔径开启，仍进行局部损失检查；此时不需要切片数据。
   * - ``sc_start``
     - ``"SC start (m)"``
     - float 或 null
     - null
     - 代表的积分区间起点，仅用于 :ref:`zh-sc-coverage`，不改变 作用位置或粒子传输。
   * - ``aperture_type``
     - ``"Aperture type"``
     - str
     - ``"default"``
     - 默认解析为网格矩形；支持 off、circle、rectangle、ellipse、rectcircle、
       rectellipse、racetrack、octagon、polygon。Dirichlet 中同时定义导体壁，禁止 off。
   * - ``aperture_value``
     - ``"Aperture value"``
     - list
     - ``[]``
     - 单位 m；circle 为 [半径]，rectangle 为 [半宽, 半高]，ellipse 为
       [半轴 a, 半轴 b]，polygon 为顶点列表。完整格式见 :doc:`aperture`。
   * - ``save_field``
     - ``"Save field"``
     - bool
     - ``false``
     - 选中圈数时保存 ``integrated_Ex`` 和 ``integrated_Ey``。
   * - ``save_potential``
     - ``"Save potential"``
     - bool
     - ``false``
     - 仅 PIC；解析方法在初始化时拒绝此请求。
   * - ``save_density``
     - ``"Save density"``
     - bool
     - ``false``
     - 保存 PIC 沉积密度或解析模型采样密度。
   * - ``save_turns``
     - ``"Save turns"``
     - 整数列表的列表或整数列表
     - ``[]``
     - 快照圈数选择，使用 ``[turn]`` 或 ``[start, end, step]``，端点均包含；
       单个 ``[0]`` 也可作为简写。

诊断输出
--------

只有命令成功执行、至少一个保存开关为 true，且当前 turn 命中 ``Save turns``
时才写出快照。``Save turns`` 为空时不写文件。输出路径为

.. code-block:: text

   <output_dir>/space_charge/<command_name>/turn_NNNNNN/

每个束团写入一个独立 HDF5 文件，网格数据的数组顺序为
``(slice, y, x)``。

.. list-table::
   :widths: 24 18 18 40
   :header-rows: 1

   * - 数据集
     - 形状
     - 单位
     - 含义
   * - ``x``、``y``
     - ``(nx,)``、``(ny,)``
     - m
     - 水平和垂直网格节点坐标。
   * - ``slice_id``
     - ``(n_slice,)``
     - -
     - 保存的切片编号。
   * - ``delta_z``
     - ``(n_slice,)``
     - m
     - 把积分场转换为平均场时使用的切片宽度。
   * - ``slice_charge``
     - ``(n_slice,)``
     - C
     - 每切片的 PIC 沉积电荷或当前解析源电荷。
   * - ``charge_density``
     - ``(n_slice, ny, nx)``
     - C/m\ :sup:`2`
     - PIC 沉积源项或解析采样密度；``Save density`` 为 true 时存在。
   * - ``potential``
     - ``(n_slice, ny, nx)``
     - V m
     - 积分电势；``Save potential`` 为 true 时存在。
   * - ``integrated_Ex``、``integrated_Ey``
     - ``(n_slice, ny, nx)``
     - V
     - 积分场；``Save field`` 为 true 时存在。除以 ``delta_z`` 后得到 V/m
       单位的平均场。

文件 attributes 还记录求解器、沉积方法、网格范围和间距、turn、beam 与 bunch
标识、harmonic 元数据、命令位置与长度、宏粒子电荷、粒子数、精度和随机种子。
``aperture_type`` 和 JSON 编码的 ``aperture_value`` 记录解析后的孔径。
``aperture_role`` 对 Dirichlet 为 ``loss_and_conductor``，自由空间方法为 ``loss_only``。
``potential_gauge`` 对 ``fd_dirichlet`` 和 ``dst_dirichlet`` 为 ``boundary_zero``，
对 ``fft_free_space`` 为 ``kernel_reference``。

解析模型的诊断输出
--------------------

HDF5 的 ``schema_version`` 为 ``3``，attributes 包含 ``method``、完整 ``solver``
名称和 ``grid_role``（``tracking`` 或 ``diagnostic``）。解析快照额外保存每切片的
``macro_count``、``center_x``、``center_y``、``size_x``、``size_y``、``angle``。
高斯的 ``size_convention`` 为 ``principal_rms``，均匀模型为 ``uniform_semi_axes``，
抛物型模型为 ``parabolic_semi_axes``。
``parameter_source`` 为 ``configuration`` 或 ``current_slice_population_moments``。
空切片参数为 NaN，电荷和采样场、密度为零。frozen 椭圆保留用户的轴顺序，
quasi-frozen 椭圆将长轴放在首位。

解析 ``Save potential`` 尚未实现，即使未选择保存圈数也会拒绝该请求；
``potential_gauge`` 为 ``not_computed``。密度网格是解析采样，不是沉积电荷，
有限网格积分不一定等于 ``slice_charge``。诊断不会改变源电荷或粒子横向动量增量。

网格范围与孔径校验
------------------

``Grid Width X/Y (m)`` 与 ``Grid Half Width X/Y (m)`` 必须完整选一组，
另一组省略或为 null。宽度必须有限且为正；两组均省略时全宽默认为 0.02 m。
不接受 ``Dx``/``Dy`` 间距或显式上下界映射输入。网格中心固定为零。
例如两个方向半宽为 0.04 m、0.02 m，节点数均为 129，则范围分别为
[-0.04, 0.04] m、[-0.02, 0.02] m，节点间距分别为 0.000625 m、0.0003125 m。
``SpaceCharge.print()`` 输出范围、节点数、节点间距，以及解析后的孔径与作用。

初始化时对每个被引用的命令检查：

- ``fd_dirichlet``：必须提供有限的受支持孔径，且完整位于网格内。
- ``dst_dirichlet``：孔径必须为轴对齐矩形，且与完整网格范围相同；
  较小矩形和其他形状均报错。
- ``fft_free_space``：有限损失孔径必须位于网格内；允许 ``off``，
  但参与计算的粒子仍必须位于网格内。
- 解析方法：显式损失孔径不受诊断网格范围约束，允许 ``off``；
  缺省孔径仍为网格同尺寸矩形。

PIC 孔径中没有有效网格节点时在初始化阶段报错；沉积时若存活参与粒子的沉积模板
没有任何有效节点，则报告网格分辨率错误，不静默丢弃源电荷。

配置与校验规则
--------------

- 未提供 ``Space charge`` 等价于 ``"Enabled": false``。
- 禁用时不会校验嵌套配置内容，所有 ``SpaceCharge`` 序列项均不执行。
- 初始化时拒绝未定义配置；计算横向动量增量时拒绝缺失或结构无效的 SliceSet 数据。
- 未被引用的配置仅记录警告，不分配计算资源。
- 配置名和命令引用必须为非空字符串，且不得包含首尾空白。
- 网格与求解器字段仅在顶层命名配置中设置，不能内联到 ``SpaceCharge`` 序列命令中。
- ``"Backend (gpu/cpu)": "gpu"`` 支持 PIC、frozen、quasi-frozen 方法与损失孔径，
  使用配置中的粒子精度。

周期 z 切片
-----------

SpaceCharge 必须使用 ``Coordinate=z_periodic``。Slicer 将连续坐标
:math:`z=\beta_b c(T_b-t_i)` 临时折叠为
:math:`z_{slice}=[(z+C/2)\bmod C]-C/2`，不修改粒子 z，也不使用规定时钟。
CPU/GPU 的 SC 均拒绝 ``z_rel``、``arrival_phase`` 和缺少坐标元数据的切片。
SC 配置要求在 Slicer 中显式选择 ``z_periodic``。整环等长网格应使用
显式 :math:`[-C/2,C/2]` 范围；auto 使用折叠后的存活粒子最小值和最大值，见 :doc:`slicer`。
该逐束团准静态近似不重建任意速度展宽下严格同时的三维空间分布。
每个存活宏粒子按最新保存的切片成员沉积，
积分横向场仍除以保存的 ``delta_z``。RF 不缩放这些 z 区间，用户手动更新切片，
包括元件内部 SC 节点。不同束团仍分别求场，横向边界由所选求解器决定。

.. _zh-internal-space-charge:

元件切片与内部空间电荷
----------------------------------------

三个独立的分辨率
~~~~~~~~~~~~~~~~

``Num slices`` 控制元件外场传输的切片数；元件内可选的 ``Space charge``
对象控制沿元件长度的 SC 积分；命名的 ``Slicer`` 结果控制粒子的纵向分箱。
这三个数量相互独立，修改其中一个不会隐式更新另外两个。

内部节点使用
``SpaceCharge`` 的命名配置、场求解器和横向动量增量归一化。
逐 Twiss 或元件外显式布置仍可使用独立 ``SpaceCharge`` 序列命令：
它使用命令执行时的粒子状态和自身 ``SC length (m)`` 计算横向动量增量，
不会把粒子传输该长度。

支持的元件
~~~~~~~~~~

内部 SC 要求正的元件长度，支持 CPU 与 GPU 后端。支持的运行时命令为 ``Drift``、
``SBend``、``Quadrupole``、``Sextupole``、``Octupole``、``Multipole``、
``Kicker``、``Bump``、``Solenoid`` 和 ``ElSeparator``。

* 多极场强和 Kicker 的偏转角是积分强度，每个外场子步使用相应长度比例。
  Yoshida 内部带符号的子步保留其带符号的外场强度。
* Drift、矩阵四极铁和纯螺线管 可以传输部分长度。需要中点时，执行
  半个短段映射、SC、剩余半个映射。
* Solenoid 保留螺线管本体映射，其本体不替换成 drift。
  叠加多极场时使用原有 Sol-Kick-Sol 步进。
* 静电偏转板在内部空间电荷节点之间使用相对论均匀电场解析映射，
  入口和出口执行硬边电势匹配。轨迹与隔板或孔径的首次接触决定损失位置；
  SC 在束流坐标系中作用于全部存活粒子，包括无场区域的粒子。
* 弯铁入口、出口映射仅在真实边界各执行一次。

薄元件保留薄透镜动量更新，不能配置内部 SC。``RFCavity`` 和 ``Exciter``
可在附近放置独立 SC 命令。Drift、ElSeparator 的 ``Num slices`` 默认值为 1。
不启用内部 SC 时，外场切片也支持 GPU。GPU 内部节点遵循相同的物理位置
与正 SC 权重，包括带符号的 Yoshida 外场子步。FP32 粒子存储下，极坐标弯铁
计算使用双精度中间量以降低相消误差。

调度规则
~~~~~~~~

设元件长度为 :math:`L`，请求的外场切片数为 :math:`N_e`，SC 次数为
:math:`N_c`，调度器计算

.. math::

   m=\left\lceil N_e/N_c\right\rceil,\qquad
   N_{e,\mathrm{actual}}=mN_c,\qquad
   h=L/N_{e,\mathrm{actual}},\qquad H=L/N_c.

每个 SC 积分区间恰好包含 :math:`m` 个外场切片，实际外场分辨率不低于请求值。
``num_slice`` 保留请求值，``slice_plan.num_slices`` 记录实际值。
未配置或关闭 SC 时，实际切片数等于外场切片数。

``S (m)`` 表示元件出口坐标。第 :math:`j=0,\ldots,N_c-1` 个节点为

.. math::

   s_j=s_{\mathrm{exit}}-L+(j+1/2)H,\qquad L_{\mathrm{sc},j}=H>0.

所有内部 SC 权重之和为 :math:`L`。SC 权重独立于执行它的外场子步长度。
内部配置不接受单独的 ``SC length (m)``，该值由调度器推导。

* :math:`m` 为奇数：在中间那个外场切片的中心、其完整中央外场动量更新 之后执行 SC。
* :math:`m` 为偶数：在中间两个完整外场切片之间执行 SC。

.. list-table:: 示例
   :header-rows: 1
   :widths: 15 15 15 20 35

   * - 请求外场数
     - SC 次数
     - 实际外场数
     - 每个 SC 区间的切片数
     - 位置
   * - 4
     - 10
     - 10
     - 1
     - 外场切片中心
   * - 12
     - 4
     - 12
     - 3
     - 中间外场切片的中心
   * - 10
     - 3
     - 12
     - 4
     - 中间两个完整切片之间

``uniform`` 的中心回调位于完整中央外场动量更新 之后。
``yoshida4`` 的中心回调位于第二个、负长度二阶子步的中央动量更新 之后，
它对应完整正长度外场切片的代数中点。只在这一个回调执行 SC，
其他 Yoshida 子步不执行 SC。SC 权重始终是正的 :math:`H`，
绝不使用带符号的 Yoshida 子步长度，也不把中央外场动量更新 拆成两次半强度动量更新。

这是中点 SC 耦合，不能因为外场选择 ``yoshida4`` 就认为整体具有四阶精度。
应分别检查外场与 SC 分辨率的收敛；平滑、线性均匀束流验证中，
本耦合方案趋于二阶收敛。固定纵向分箱也不意味着任意外场映射与 SC 可交换。

配置与使用
~~~~~~~~~~

所有支持的元件 schema 提供 ``num_slices``（JSON ``Num slices``，默认 1）
和 ``space_charge``（JSON ``Space charge``，默认 null）。
``ElementSpaceCharge`` 接口如下：

.. list-table::
   :header-rows: 1
   :widths: 25 25 15 35

   * - Python 字段
     - JSON 键
     - 默认值
     - 含义
   * - ``configuration``
     - ``Configuration``
     - 必填
     - 顶层命名 SC 资源
   * - ``num_kicks``
     - ``Num kicks``
     - 1
     - 正整数，不接受 bool、浮点数或数字字符串
   * - ``aperture_type`` / ``aperture_value``
     - ``Aperture type`` / ``Aperture value``
     - ``default`` / []
     - 兼容原有配置；默认继承元件孔径，不一致时警告并替换
   * - ``save_field`` / ``save_potential`` / ``save_density``
     - ``Save field`` / ``Save potential`` / ``Save density``
     - false
     - 诊断支持范围与所选显式 SC 求解器相同
   * - ``save_turns``
     - ``Save turns``
     - []
     - 原有格式，例如 [[0]] 或 [[0, 100, 10]]

对于已有的 sequence 和命名配置 ``sc_default``：

.. code-block:: python

   from PASS.para.schema.elements import QuadrupoleItem
   from PASS.para.schema.space_charge import ElementSpaceCharge

   sequence.add("q1", QuadrupoleItem(
       s=1.0, length=0.4, k1l=0.12,
       num_slices=10, integrator="yoshida4",
       space_charge=ElementSpaceCharge(
           configuration="sc_default", num_kicks=3,
           save_density=True, save_turns=[[0]],
       ),
   ))

此例请求 10 个外场切片，实际使用 12 个切片、3 次 SC，每次权重为 0.4/3 m。
需要启用顶层 ``Space charge.Enabled``，并在 ``Configurations`` 定义
``sc_default``。元件执行前必须已有该配置指定的 SliceSet，通常由上游
``Slicer`` 提供。

内部节点复用已有 ``slice_id`` 和 ``slice_table.delta_z``，不重新分箱、
不修改这些宽度，也不折叠 ``p.z``。每次场计算使用当前横向粒子坐标，
具体分布处理由所选 PIC/frozen/quasi-frozen 模型决定。
元件只按总长度推进一次 ``bunch.t0``，每个节点只作用于当前跟踪的束团。

内部 SC 始终使用所属元件的孔径，粒子损失检查和 Dirichlet 导体边界均以
元件孔径为准。建议省略内部孔径字段：其 ``default`` 表示继承元件孔径。
若内部显式指定的孔径不同，会对该元件的这一处不一致警告一次，并在建立求解器资源前
替换为元件孔径。原始输入对象保持不变；打印输出实际生效值和 ``Source=element``。

元件的通用 ``default`` 仍为 +/-1 m 矩形，不是 SC 网格矩形。
元件孔径为 ``off`` 时内部 SC 也保持 off：自由空间 PIC 和解析模型允许这样设置，
Dirichlet 求解器则要求元件具有有限孔径。PIC 网格必须包含继承后的孔径，
DST 要求它恰好等于网格对齐的完整矩形。不兼容时初始化报错，不会静默更换管壁或求解器。
关闭损失孔径时，自由空间 PIC 中粒子超出网格仍会触发原有网格范围错误。
独立的显式 SC 命令 仍保留自己的孔径设置。

每个束团每次通过普通元件（包括 ``Bump``）时，若没有生效的内部 SC，仅在出口
检查一次孔径。若有 :math:`K` 个生效的内部 SC 节点，每个 SC 入口在计算源项前
检查一次，元件出口再检查一次，共 :math:`K+1` 次。其余外场切片边界不增加检查。
刚损失的粒子不参与源项和横向动量增量，首次损失记录得到保留。每个 SC 节点仍执行 PIC
计算域有效性检查。``ElSeparator`` 另行保留沿粒子轨迹的首次接触碰撞处理。
内部节点按配置名和实际孔径共享资源，快照保存在
``<space_charge_output>/<element>/internal_sc/node_000000/turn_000000/``。
文件名保留 beam 和 bunch 标识。HDF5 记录 ``parent_element``、
``internal_node_index``、``s``、``sc_length`` 和 ``sc_start``。
显式 SC 仅在填写了起点时保存 ``sc_start``；它描述代表的积分区间，不改变力的计算。

各元件的 ``print()`` 输出请求和实际外场切片数、切片长度、SC 配置名、
method、solver、SliceSet、SC 次数和放置方式、单次和总作用长度、首末节点位置、
实际孔径和保存设置。未配置内部 SC 时打印 ``off``，全局关闭时打印
``disabled by top-level Space charge.Enabled``。内部 SC 耗时仍计入母元件耗时。

导入 MAD-X 元件时，合并后的普通 drift 使用末段出口 S；带局部 SC、
非默认外场切片或孔径配置的 drift 保留原边界，不参与合并。

.. _zh-sc-coverage:

跟踪前的作用长度与覆盖检查
--------------------------

``Executor.run`` 在粒子跟踪前检查各启用 beam 的实际命令序列。
每次经过 sequence 的内部 SC 权重与显式 command 长度各累计一次，
不乘束团数，也不乘模拟圈数。除了总权重，还检查区间覆盖：
总长度相等不能排除某处遗漏、另一处重复。

以下字段位于顶层 ``Space charge`` 中，与 ``Enabled``、``Configurations``
同级，也可通过 GUI 的空间电荷表单编辑。

.. list-table:: 覆盖检查配置
   :header-rows: 1
   :widths: 25 25 15 35

   * - Python 字段
     - JSON 键
     - 默认值
     - 含义
   * - ``coverage_check``
     - ``Coverage check``
     - ``warn``
     - ``warn`` 警告后继续；``error`` 在不一致或无法完整检查时于跟踪前报错；
       ``off`` 跳过这项检查。
   * - ``coverage_mode``
     - ``Coverage mode``
     - ``full-ring``
     - 全环模式比较总权重与 ``Circumference (m)``，并检查遗漏和重复。
       ``partial`` 允许未覆盖区段，但仍检查重复。
   * - ``expected_sc_length``
     - ``Expected SC length (m)``
     - null
     - 仅 ``partial`` 可选的非负总权重目标；全环模式固定使用环长，禁止此覆盖值。

要严格检查全环，可将以下设置合入已有顶层 ``Space charge`` 块：

.. code-block:: json

   {
       "Coverage check": "error",
       "Coverage mode": "full-ring"
   }

内部 SC 的积分区间由元件本体确定。显式 SC 命令 可选填
``SC start (m)``（Python ``sc_start``），与 ``SC length (m)`` 一起表示
半开区间 :math:`[s_{start},s_{start}+L_{sc})`。起点独立于作用点的 ``S (m)``。
例如在 0.7 m 处施加代表 [0.5, 0.9) m 区间的 SC：

.. code-block:: json

   {
       "Command": "SpaceCharge",
       "S (m)": 0.7,
       "Configuration": "sc_default",
       "SC start (m)": 0.5,
       "SC length (m)": 0.4
   }

区间按环长周期比较，支持跨越 s=0 的区间和超过一圈的作用长度，不折叠粒子坐标。
绝对比较容差为 ``max(1e-12 m, 1e-10 * circumference)``。

正长度显式 command 未填写起点时，其权重仍参与求和，但检查器不会擅自假定
区间以 作用位置为中心，报告状态为 ``incomplete``。此时已知区间未覆盖的位置
不能被认定为真实遗漏。严格模式拒绝检查不完整或不一致的配置，警告模式兼容原有工作流。
关闭或零权重的显式 SC 不参与累计。

日志输出总长度、目标长度、次数、状态及前十个遗漏或重复区段。
完整区间和贡献项保存到 ``<space_charge_output>/coverage_beam0.json``，
每个启用检查的 beam 各有一份，并可从 ``sim.space_charge_coverage[beam_id]`` 获取。
严格模式失败时也先保存报告再抛出异常。partial 模式中的未覆盖区段仅作信息输出。
程序不会自动修改长度或用户区间来满足检查。环长缺失或无效时周期检查不完整；
全局关闭空间电荷时跳过检查。

精度与收敛
------------------

粒子坐标可使用 float32 或 float64。即使求场和参考量采用更高精度的中间计算，float32 存储仍可能改变边界附近的切片归属，并积累输运舍入误差。CPU/GPU 的求和顺序和稀疏分解不同，结果应按适合问题的容差比较，不要求逐位相同。分别检查宏粒子数、切片数、横向网格范围和分辨率、以及空间电荷作用点数的收敛。

物理模型
--------

宏粒子电荷与横向源项
~~~~~~~~~~~~~~~~~~~~

每个存活宏粒子代表 ``bunch.ratio`` 个真实粒子，其带符号源电荷为

.. math::

   q_{\mathrm{macro}} = R\,Z e,

其中 :math:`R` 为 ``bunch.ratio``，:math:`Z` 为带符号电荷数
``bunch.num_charge``，:math:`e` 为元电荷。对第 :math:`k` 个切片，沉积步骤构造
沿纵向积分后的横向电荷密度

.. math::

   \Sigma_k(x_i,y_j)
   = \frac{1}{\Delta x\,\Delta y}
     \sum_{n\in k} q_{\mathrm{macro},n} W_{ij,n},

单位为 C/m\ :sup:`2`。``CIC`` 使用 4 个网格节点，``TSC`` 使用 9 个节点。
已损失粒子和切片 ID 为 -1 的粒子不进入源项，其他非法切片编号报错。
命令先将孔径上或孔径外的粒子标记为损失；此后仍存活并参与 PIC 的粒子若超出网格则报错。

积分电势与积分场
~~~~~~~~~~~~~~~~

场求解器把每个切片作为二维 Poisson 问题：

.. math::

   -\nabla_\perp^2 \Psi_k = \frac{\Sigma_k}{\epsilon_0},
   \qquad
   \boldsymbol{\mathcal E}_{\perp,k} = -\nabla_\perp\Psi_k.

这里 :math:`\Psi_k` 是单位为 V m 的纵向积分电势，
:math:`\boldsymbol{\mathcal E}_{\perp,k}` 是单位为 V 的横向积分场。网格场回插到
粒子后，``SpaceCharge`` 仅在计算横向动量增量时使用相应的 ``SliceSet.delta_z``，把积分场
转换为平均场：

.. math::

   \overline{E}_{x,k}=\frac{\mathcal E_{x,k}}{\Delta z_k},
   \qquad
   \overline{E}_{y,k}=\frac{\mathcal E_{y,k}}{\Delta z_k}.

因此场求解器本身不需要 ``delta_z``。离散格式、边界条件和求解器接口参见
:doc:`field_solver`。

相对论横向动量更新
~~~~~~~~~~~~~~~~~~

PASS 使用归一化横向动量 :math:`p_x=P_x/P_0` 和 :math:`p_y=P_y/P_0`。
对有效作用长度 :math:`L_{\mathrm{sc}}`，命令施加

.. math::

   \Delta p_x =
   \frac{\operatorname{sgn}(Z)L_{\mathrm{sc}}}
        {\beta c\,B\rho\,\gamma^2}\,\overline{E}_x,
   \qquad
   \Delta p_y =
   \frac{\operatorname{sgn}(Z)L_{\mathrm{sc}}}
        {\beta c\,B\rho\,\gamma^2}\,\overline{E}_y.

电场已经包含带符号的源电荷；额外的 :math:`\operatorname{sgn}(Z)` 表示被跟踪
粒子的受力符号。:math:`1/\gamma^2` 表示共速束流横向电场力与磁场力的抵消。

横向动量更新只更新 ``px``、``py``，不修改 ``x``、``y``、``z``、``dp``。
每个 SC 命令可设置独立的 ``Aperture type`` 和 ``Aperture value``，在求场前
检查粒子损失：更新 tag、损失位置和圈数，刚损失的粒子不再参与源项和横向动量更新。
已损失粒子的损失记录不被后续计算点覆盖。也可继续使用 ``Marker`` 等上游元件处理损失。
PIC 要求损失检查后仍参与计算的粒子位于网格内，否则报错。只有严格位于孔径内部
的粒子存活，接触任意一处管壁即损失。所有方法中，省略孔径或使用 ``default``
均解析为配置网格同尺寸的矩形。解析方法显式指定孔径或 ``off`` 后，跟踪不受诊断
网格范围限制；采样不会截断源分布。

执行流程
--------

用户独立控制切片和命令顺序。SpaceCharge 不检查 Slicer 是否存在于序列、
是否执行、执行圈数、位置或粒子状态版本。它直接使用 ``slice_id`` 和
``slice_table.delta_z``，不根据 z 重新分配粒子。仅检查数据有效性：
每个束团粒子对应一个整数编号，编号为 -1 或有效范围内值，切片宽度有限且严格为正。
程序不自动切片，现有同位置命令优先级保持不变。

.. code-block:: text

   当前计算点的粒子损失孔径检查
       -> 输入切片数据 + 当前 x、y、更新后的存活标记
       -> PIC 沉积/求解/回插 或 解析粒子位置处直接求场
       -> 除以切片 delta_z
       -> 统一更新横向动量 并按需保存快照
