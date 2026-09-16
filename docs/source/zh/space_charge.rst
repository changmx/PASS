空间电荷效应（SpaceCharge）
===========================

概念介绍
--------

``SpaceCharge`` 使用 PIC、frozen 解析分布或 quasi-frozen 解析分布施加横向
2.5D 薄踢，读取用户提供的束团级切片数据。所有方法使用 :math:`1/\gamma^2`
处理电场力与磁场力的抵消，均不计算纵向空间电荷力。

- **代码位置**：``PASS/commands/space_charge.py``
- **类名**：``SpaceCharge``，注册名称为 ``"SpaceCharge"``
- **Schema 位置**：``PASS/para/schema/space_charge.py``
- **计算后端**：CPU 与 NVIDIA GPU
- **主要功能**：

  - 在每个 beam 输入文件顶层定义可复用的命名空间电荷配置；
  - 纵向切片、横向网格、粒子沉积方法和场求解器相互独立配置；
  - 支持 CIC 或 TSC 电荷沉积，并自动配对相应的场回插方法；
  - 三种场求解器分别提供导体边界或自由空间边界；
  - 可选保存电荷密度、电势和电场的 HDF5 快照。

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
粒子后，``SpaceCharge`` 仅在计算踢量时使用相应的 ``SliceSet.delta_z``，把积分场
转换为平均场：

.. math::

   \overline{E}_{x,k}=\frac{\mathcal E_{x,k}}{\Delta z_k},
   \qquad
   \overline{E}_{y,k}=\frac{\mathcal E_{y,k}}{\Delta z_k}.

因此场求解器本身不需要 ``delta_z``。离散格式、边界条件和求解器接口参见
:doc:`field_solver`。

相对论横向踢
~~~~~~~~~~~~

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

踢动只更新 ``px``、``py``，不修改 ``x``、``y``、``z``、``dp``。
每个 SC 命令可设置独立的 ``Aperture type`` 和 ``Aperture value``，在求场前
检查粒子损失：更新 tag、损失位置和圈数，刚损失的粒子不再参与源项和踢动。
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
       -> 共用横向 kick 并按需保存快照

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

``quasi-frozen`` 在每次 kick 前，对每个切片当前存活且具有有效切片编号的粒子，
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
解析粒子 kick 不沉积电荷、不回插场。网格几何定义默认损失孔径以及可选的场和
密度采样范围。显式指定孔径或 ``off`` 后，改变诊断网格不会改变跟踪；使用默认
孔径时，改变网格范围可能改变粒子损失。


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
始终拥有独立资源，即使字段相同。配置中不再包含 ``Chamber``。

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
     - 薄踢的环上纵向位置，必须为有限值。
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
     - 非负有效作用长度。为零时该命令不施加踢，也不产生快照。
       若损失孔径开启，仍进行局部损失检查；此时不需要切片数据。
   * - ``sc_start``
     - ``"SC start (m)"``
     - float 或 null
     - null
     - 代表的积分区间起点，仅用于 :ref:`zh-sc-coverage`，不改变 kick 位置或粒子传输。
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

网格范围与孔径校验
------------------

``Grid Width X/Y (m)`` 与 ``Grid Half Width X/Y (m)`` 必须完整选一组，
另一组省略或为 null。宽度必须有限且为正；两组均省略时全宽默认为 0.02 m。
不再接受 ``Dx``/``Dy`` 间距输入或显式上下界映射输入。网格中心固定为零。
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

PIC 孔径中没有有效网格节点时在初始化阶段报错；沉积时若存活参与粒子的 stencil
没有任何有效节点，则报告网格分辨率错误，不静默丢弃源电荷。

配置与校验规则
--------------

- 未提供 ``Space charge`` 等价于 ``"Enabled": false``。
- 禁用时不会校验嵌套配置内容，所有 ``SpaceCharge`` 序列项均不执行。
- 初始化时拒绝未定义配置；执行 kick 时拒绝缺失或结构无效的 SliceSet 数据。
- 未被引用的配置仅记录 warning，不分配计算资源。
- 配置名和命令引用必须为非空字符串，且不得包含首尾空白。
- 旧根级字段 ``Is space charge`` 和 ``Space-charge simulation parameters``
  会被拒绝；网格与求解器字段也不能内联到 ``SpaceCharge`` 序列命令中。
- ``"Backend (gpu/cpu)": "gpu"`` 支持 PIC、frozen、quasi-frozen 方法与损失孔径，
  使用配置中的粒子精度。

GPU 初始化与复用
----------------

GPU 复用现有命名资源配置。初始化时准备每个被引用的配置/孔径组合：
FD 持有 cuDSS 分析与数值分解；DST 持有变换数据与 plan；自由空间 FFT
持有格林函数频谱与 plan。已知 SliceSet 数量用于预分配批量工作区。
共享相同资源的命令直接复用。应在跟踪前通过序列或元件内部 SC 配置声明
不同孔径，以便初始化时分别建立缓存项。

网格数和宽度不变，并不意味着孔径变化后的 FD 算子不变：有效节点、稀疏结构
与 Shortley--Weller 壁距都可能变化，因此必须切换到对应孔径的缓存项。
即使有效节点 mask 不变，壁距改变也需要新的数值分解。
DST 要求孔径等于完整网格矩形；不能通过对矩形 DST 解加 mask 来表示变化的内部孔径。

从沉积、求场、回插到 kick，粒子与网格数组均保留在 GPU。
校验和损失处理仍会读取少量状态量，整个命令目前不是完全异步的 CUDA Graph。
选定的 HDF5 输出会将诊断数组传回 CPU。性能测量应将初始化、编译和首次
批量工作区准备与重复跟踪分开。DST 自动择优与独立 GPU 接口参数见
:doc:`field_solver`。

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

验证文件组织
------------

积分测试文件名同时写明束团分布与边界模型。``*_free_space_fft`` 覆盖圆形/椭圆
高斯与 KV 束团，使用开放边界 Green 函数；对应的
``*_rectangular_fd_dst`` 会用相同种子重新生成这四种束团，在 257×257 接地矩形
网格上让 FD 和 DST 求解完全相同的沉积电荷。每个分析目录都输出三者同图
``free_space_fft_vs_rectangular_fd_dst_field.png``、CSV 数据，以及把 FD/DST 两个
方向逐点相对差同时画出的 ``fd_dst_relative_error_same_plot.png``。FD/DST 与
FFT 的差异同时包含物理边界条件变化，不能把它解释为 FD/DST 离散误差。

三者场对比图显示 0、22.5、45、67.5 和 90 度的径向扫描。颜色区分角度；
点线、实线和虚线分别表示自由空间 FFT、矩形 FD 和矩形 DST。

仅适用于 FD 的测试在文件名中注明物理边界，包括较大圆导体内的圆 KV 束团、填满
椭圆导体的 KV 束团，以及高斯源和 KV 均匀投影源在
``PASS.utils.aperture`` 全部孔径上的求解（含椭圆、跑道、八边形和用户 polygon）。
``dst_dirichlet`` 只能对完整矩形 Dirichlet 网格对角化，因此不用于这些曲线或
不规则区域。

运行验证套件
------------

整个 ``tests/`` 目录仅在本地维护，不纳入 Git 版本控制，新克隆的仓库不包含该目录。
本页的测试命令和测试文件路径均要求本地已有对应的测试套件。

全部空间电荷积分测试默认自动执行，包括十个完整的生成输入工作流。在仓库根目录运行：

.. code-block:: console

   python -m tests.integration.space_charge
   python -m tests.integration.space_charge regression

第一条命令选择全部 25 个积分测试。第二条命令显式执行相关单元测试和本地 Codex
回归文件，包括已恢复的切片隔离以及切片宽度、作用长度缩放检查。本地 Codex 文件
必须存在；它们不属于默认 pytest 发现范围。仓库级 pytest 默认发现 ``tests/unit``
和 ``tests/integration``。

可用 ``analytic``、``fft``、``rectangle``、``aperture``、``checks`` 或 ``workflows`` 替换默认的
``all``，按类别选择测试。重复 ``--case <case_name>`` 可选择多个完整工作流；
``--collect-only`` 列出选中的 pytest 条目。不会因积分测试复杂而默认跳过。

工作流串行执行。矩形对比会在同一批运行内自动准备并复用对应 FFT 参考。每次新的
积分测试在 ``tests/codex/space_charge_runs`` 下创建唯一目录，终端汇总打印其路径；
``--output-dir`` 可指定新的或空的批次目录。运行器不会删除旧结果。
``--mode ana --output-dir <existing_batch>`` 复用保存快照并更新分析产物，不重新跟踪。
非 test 模式只处理完整工作流，不执行纯数值检查。

普通 ``python -m pytest tests/integration/space_charge`` 命令执行相同的积分测试。
原有各模块的 ``sim``、``ana``、``simana``、``--run-dir`` 接口继续保留。完整分组表、
单例调用、输出结构以及批次目录与单例目录的区别，见
``tests/integration/space_charge/README.md``。


解析诊断与验证
--------------

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
有限网格积分不一定等于 ``slice_charge``。诊断不会改变源电荷或粒子 kick。

运行 ``python -m tests.integration.space_charge analytic``，执行四个生成输入的
比较案例（每个 180,000 粒子、三个切片、全部三种方法）和两个重复 kick 参数演化测试。
独立高斯/均匀场积分核对实际动量增量，相对 L2 容差为 ``2e-8``；PIC 在有充分
分辨率的中心区域比较，相对 L2 容差为 ``0.06``，包含有限粒子噪声和离散化误差。
电荷对照采用随粒子数变化的浮点累加误差界。演化测试使用指定仿射输运，
不代表已经验证长期环跟踪稳定性。

输出包含输入 JSON、HDF5、粒子 kick NPZ、CSV/JSON 测量，以及
``pic_frozen_quasi_frozen_field_comparison.png``、
``analytic_kick_vs_independent_integral.png``、
``frozen_vs_quasi_frozen_parameter_evolution.png``。
生成输入的解析案例可调用
``tests.integration.space_charge.test_analytic_free_space_tracking`` 中的
``analyse(run_dir)``，复用保存结果更新图形。解析组目前使用 ``--mode test``；
原有 batch 非测试模式的 workflow 选择保持不变。

旧资源键 ``Field solver``、``Aperture``、``Chamber`` 和旧公开 solver 值会被拒绝。
配置使用 ``Method``、``Solver``，各命令使用 ``Aperture type/value``。固定横向参数
仅用于 ``frozen``，且必须匹配所选分布；沉积设置仅用于 PIC。

.. _zh-internal-space-charge:

元件切片与内部空间电荷
----------------------------------------

三个独立的分辨率
~~~~~~~~~~~~~~~~

``Num slices`` 控制元件外场传输的切片数；元件内可选的 ``Space charge``
对象控制沿元件长度的 SC 积分；命名的 ``Slicer`` 结果控制粒子的纵向分箱。
这三个数量相互独立，修改其中一个不会隐式更新另外两个。

实现位于 ``PASS/utils/slicing.py``。内部节点复用
``SpaceCharge`` 的命名配置、场求解器和踢量归一化。
逐 Twiss 或元件外显式布置仍可使用独立 ``SpaceCharge`` 序列命令：
它使用命令执行时的粒子状态和自身 ``SC length (m)`` 计算踢量，
不会把粒子传输该长度。

支持的元件
~~~~~~~~~~

内部 SC 要求正的元件长度，支持 CPU 与 GPU 后端。支持的运行时命令为 ``Drift``、
``SBend``、``Quadrupole``、``Sextupole``、``Octupole``、``Multipole``、
``Kicker``、``Solenoid`` 和 ``ElSeparator``。

* 多极场强和 kicker 角度是积分强度，每个外场子步使用相应长度比例。
  Yoshida 内部带符号的子步保留其带符号的外场强度。
* Drift、矩阵四极铁和纯 solenoid 可以传输部分长度。需要中点时，执行
  半个短段映射、SC、剩余半个映射。
* Solenoid 保留螺线管本体映射，其本体不替换成 drift。
  叠加多极场时使用原有 Sol-Kick-Sol 步进。
* 切片后的 ES 使用 drift、电场 kick、drift。电场 kick 按子步长度缩放，
  在每个外场切片中心重新判断 septum 区域。电场 kick 在元件倾斜坐标系中执行，
  SC 回到束流坐标系后，对包括无场区域在内的全部存活粒子计算。
  Septum 截获按这些中心位置采样，并不连续求解轨迹与 septum 的交点。
* 弯铁入口、出口映射仅在真实边界各执行一次。

薄元件保留其薄 kick，不能配置内部 SC。``RFCavity`` 和 ``Exciter``
可在附近放置独立 SC command。新增的 Drift、ElSeparator 外场切片默认值为 1。
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

* :math:`m` 为奇数：在中间那个外场切片的中心、其完整中央外场 kick 之后执行 SC。
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

``uniform`` 的中心回调位于完整中央外场 kick 之后。
``yoshida4`` 的中心回调位于第二个、负长度二阶子步的中央 kick 之后，
它对应完整正长度外场切片的代数中点。只在这一个回调执行 SC，
其他 Yoshida 子步不执行 SC。SC 权重始终是正的 :math:`H`，
绝不使用带符号的 Yoshida 子步长度，也不把中央外场 kick 拆成两个半 kick。

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

   from PASS.para.schema.elements import QuadrupoleElement
   from PASS.para.schema.space_charge import ElementSpaceCharge

   sequence.add("q1", QuadrupoleElement(
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
独立的显式 SC command 仍保留自己的孔径设置。

SC 节点和元件出口均检查元件孔径。损失粒子不参与源项和踢量，首次损失记录得到保留。
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

内部 SC 的积分区间由元件本体确定。显式 SC command 可选填
``SC start (m)``（Python ``sc_start``），与 ``SC length (m)`` 一起表示
半开区间 :math:`[s_{start},s_{start}+L_{sc})`。起点独立于 kick 的 ``S (m)``。
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
区间以 kick 位置为中心，报告状态为 ``incomplete``。此时已知区间未覆盖的位置
不能被认定为真实遗漏。严格模式拒绝检查不完整或不一致的配置，警告模式兼容原有工作流。
关闭或零权重的显式 SC 不参与累计。

日志输出总长度、目标长度、次数、状态及前十个遗漏或重复区段。
完整区间和贡献项保存到 ``<space_charge_output>/coverage_beam0.json``，
每个启用检查的 beam 各有一份，并可从 ``sim.space_charge_coverage[beam_id]`` 获取。
严格模式失败时也先保存报告再抛出异常。partial 模式中的未覆盖区段仅作信息输出。
程序不会自动修改长度或用户区间来满足检查。环长缺失或无效时周期检查不完整；
全局关闭空间电荷时跳过检查。

FODO 工作点验证
----------------

需显式运行的生成测试位于 ``tests/codex/sc_tune_fodo/``，使用
``example/03_tracking_element_by_element/fodo.tfs`` 进行带元件内部 SC 的
多圈逐元件跟踪。测试保留原文件，在独立生成的输入中关闭非线性多极强度，
使用 33.2 MeV 质子束、无 RF，并开启严格的整环 SC 覆盖检查。在仓库根目录运行：

.. code-block:: console

   python -m pytest tests/codex/sc_tune_fodo/test_benchmark.py -v
   python -m tests.codex.sc_tune_fodo.run --suite frozen --turns 256
   python -m tests.codex.sc_tune_fodo.run --suite duration --turns 128
   python -m tests.codex.sc_tune_fodo.run --suite collective --turns 256 --particles 4096

均匀 frozen 模型采用独立的连续线性矩阵作为理论参考；高斯模型采用一阶
作用量平均场积分。每组 SC 计算均有初始坐标与外场切片一致的无 SC 对照。
PhaseAdvanceMonitor 记录全体粒子，另对 36 个作用量探针的 ParticleMonitor
轨迹进行独立复信号傅里叶分析。输出包含工作点分布、束流演化、CSV/JSON
对比、覆盖报告及加密差异。各组生成输入和跟踪应串行运行。

Frozen 圆束是预设场源，并非与 FODO 匹配的自洽束流。
Quasi-frozen/PIC 使用初始匹配高斯束理论作为参考，并报告网格、粒子数、
随机种子和纵向分箱的敏感性；frozen 检查通过不代表 PIC 已收敛。
测试目录 README 说明公式、容差和纵向静止的分箱模型。

全粒子 PIC 跟踪
~~~~~~~~~~~~~~~

``tests.codex.sc_tune_fodo.long_pic`` 将验证扩展至真正的四维 KV 和高斯
粒子分布。每个粒子均参与电荷沉积，接受重新求解的 PIC 场，其四个横向
坐标逐圈保存到一个无损 HDF5 文件。例如：

.. code-block:: console

   python -m tests.codex.sc_tune_fodo.long_pic --profile kv --turns 1024 --particles 8192 --grid 129 --kicks 2 --output tests/codex/sc_tune_fodo/output/my_kv_1024
   python -m tests.codex.sc_tune_fodo.long_pic --profile gaussian --turns 1024 --particles 8192 --grid 129 --kicks 2 --output tests/codex/sc_tune_fodo/output/my_gaussian_1024
   python -m tests.codex.sc_tune_fodo.long_pic --profile gaussian --turns 512 --particles 65536 --grid 129 --kicks 2 --output tests/codex/sc_tune_fodo/output/my_gaussian_refined_512

使用新的输出目录，并串行运行。该实验通过测试局部钩子将纵向坐标固定为零，
逐圈检查动量偏差保持不变，用一个分箱表示线密度 ``N/C``。
这是横向静止切片验证，不是纵向束团模拟，也没有新增公开元件参数。
应通过上述模块运行；单独运行生成的 JSON 不会启用这些测试钩子。

独立的匹配 KV 理论在每个平面给出一个下降后的工作点；高斯理论通过
相位平均场积分计算初始弱空间电荷条件下随振幅变化的工作点。
傅里叶分析覆盖全部粒子和不同时间窗口。输出包括逐粒子 CSV 对比、
模拟与理论 tune spread 图、发射度演化、源文件哈希及整环覆盖检查。
测试目录中的 ``LONG_PIC.md`` 说明匹配、低噪声采样、公式和数值限制。
完整串行流程可通过
``python -m tests.codex.sc_tune_fodo.run_long_pic --output <新目录>`` 运行。
初始低噪声粒子采样不保证多圈后仍然低噪声；流程保留初始高斯长圈数结果，
并在经过独立检查的较短频率窗口内验证更大的粒子数。

不同边界的多圈集成测试
~~~~~~~~~~~~~~~~~~~~~~

可复用的 ``tests.integration.space_charge.multiturn`` 使用正式输入、Twiss、
Slicer 和 PIC 命令构造均匀聚焦基准环。全部带电 KV/高斯粒子追踪 512 圈，
覆盖自由空间 FFT、圆形接地边界 FD 和矩形接地边界 FD/DST。
Twiss 纵向传输关闭；一个固定纵向切片表示 ``N/C`` 线密度，
正权重中心 SC kick 的积分长度覆盖整环。

.. code-block:: console

   python -m tests.integration.space_charge.multiturn --output tests/integration/space_charge/output/my_multiturn
   python -m tests.integration.space_charge.multiturn --analyse-only --output tests/integration/space_charge/output/my_multiturn
   python -m tests.integration.space_charge multiturn

最后一个命令通过 pytest 执行追踪验收及独立理论检查；``all`` 组包含这些测试。
默认 KV 使用 32,768 个粒子，高斯使用 65,536 个粒子。
请使用新的输出目录，并串行运行生成输入的工作流。
``--resume`` 检查配置后复用已完成案例；未完成案例保留并报错。

矩形管理论由连续 Poisson 方程的本征函数展开、解析源系数及 Bessel 相位平均得到，
独立于 PIC 电荷沉积和网格求解器。圆形管内居中的圆对称源使用自由空间径向场。
高斯 tune 预测采用初始弱空间电荷近似，并非精确的非线性 Vlasov 平衡解。
测试覆盖横向准静态边界模型，不覆盖纵向空间电荷、管壁频率响应或 GPU。

输出包括完整粒子轨迹、逐粒子理论/模拟 tune、前后半段频率差、
rms 束斑及发射度演化、密度/势/场图和独立的中轴线场比较。
``tests/integration/space_charge/multiturn/README.md`` 说明公式、验收误差、边界尺寸
及复现方法。此均匀聚焦基准与上面的逐元件 FODO 测试互为补充。

强空间电荷验证与扁平算例输出
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests.integration.space_charge.strong`` 提供独立的串行验证流程，包含
匹配 KV 的 0.9、0.7、0.5 tune depression 扫描、数值加密、周期 FODO
追踪，以及开放和导体边界下的高斯演化。均匀聚焦 KV 使用完整自洽平衡解，
不采用一阶 tune shift；FODO 由独立周期 KV 包络匹配，逐晶胞相位累计确定
包含整数部分的完整 tune。高斯 rms 匹配并非精确非线性平衡，因此需要通过
数值加密及独立的无网格轴对称平均场参照比较其演化。此参照不包含非轴对称模，
本身也有有限采样和时间步误差。

周期 KV 包络匹配只保证二阶矩的周期性，并不保证所有高阶分布模稳定。
发生集体增长的算例保留静态 KV 判据的未通过记录，并提供高阶矩诊断。
可选的 ``strong.retuned`` 模块增加单独记录的 FODO 工作点控制，
不会替代原始结果。首次内部 KV 场采用独立部分四极传输后的均匀椭圆解析场比较。

.. code-block:: console

   python -m pytest tests/integration/space_charge/test_strong_references.py -v
   python -m tests.integration.space_charge.strong --output tests/integration/space_charge/output/my_strong_run
   python -m tests.integration.space_charge.strong.finish tests/integration/space_charge/output/my_strong_run

最后一个命令补齐独立参照及诊断，重新生成报告，并核查保存的轨迹及扁平输出文件，
不会重复 PIC 追踪。串行批次运行期间需要中间报告时，可运行
``tests.integration.space_charge.strong.report`` 模块并传入输出目录。

后处理还会在二维束流区域内将首次网格场与独立连续场比较。区域采用小于 4 sigma
的归一化椭圆半径，并排除网格外缘的 5%；各场分量的预设 L2 相对误差门限为 3%。
各算例中的 ``initial_field_2d`` 文件包含图片、指标、理论/PIC 网格及区域掩码。
这种区域范数不是逐网格点的误差上界，也不代表长期精度。
椭圆高斯初始场采用独立连续 Poisson 积分解；参照测试检查积分加密、圆束极限
以及轴线之外的场方程和零旋度条件。

可选的 ``strong.linear_fodo`` 诊断复用已完成 FODO 算例的粒子、电荷和 SC 位置，
以独立线性矩阵替换生产元件传输并保留非线性 PIC，用于隔离外场几何非线性和
元件调度。它本身不能确定某种已证实的物理 Vlasov 不稳定性。

流程输出中文报告和科学绘图。每个独立算例的输入、轨迹、CSV、图片和报告
直接放在同一个目录。``Config.load_input(path, flat_output=True)`` 是本流程
使用的可选 Python 运行时参数，不是 JSON 字段。调用方负责为每次运行隔离目录；
已有运行输入快照时会报错。SC 快照文件名包含命令、内部节点（如适用）、beam、
bunch 和圈数。应用程序默认输出仍采用日期目录布局。

均匀聚焦环关闭 Twiss 纵向传输；FODO 实验通过测试局部钩子在元件末尾冻结 z，
并检查 dp 不变。单独运行生成的 JSON 不会启用此钩子。这些是 CPU 横向测试，
不验证纵向空间电荷、RF、GPU 或管壁频率响应。
本轮 FODO 强流算例每次 SC 对应两个外场切片，SC 位于完整外场切片之间，
不能据此验证奇数分组的内部中心回调强流精度。
关闭 SC 的基线使用裸 tune 绝对误差；由于 SC shift 为零，按 shift 归一化的指标
记录为 ``null``。
``tests/integration/space_charge/strong/README.md`` 说明全部设置、独立参照、
预先规定的门限和扁平输出文件。

周期 z 切片
-----------

SpaceCharge 必须使用 ``Coordinate=z_periodic``。Slicer 将连续坐标
:math:`z=\beta_b c(T_b-t_i)` 临时折叠为
:math:`z_{slice}=[(z+C/2)\bmod C]-C/2`，不修改粒子 z，也不使用规定时钟。
CPU/GPU 的 SC 均拒绝 ``z_rel``、``arrival_phase`` 和缺少坐标元数据的切片。
已有 SC 输入必须在 Slicer 中显式选择 ``z_periodic``。整环等长网格应使用
显式 :math:`[-C/2,C/2]` 范围；auto 使用折叠后的存活粒子最小值和最大值，见 :doc:`slicer`。
该逐束团准静态近似不重建任意速度展宽下严格同时的三维空间分布。
每个存活宏粒子按最新保存的切片成员沉积，
积分横向场仍除以保存的 ``delta_z``。RF 不缩放这些 z 区间，用户手动更新切片，
包括元件内部 SC 节点。不同束团仍分别求场，横向边界与求解器保持原有定义。
