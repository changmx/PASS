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
- **计算后端**：仅 CPU
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

``frozen`` 中，同一配置的所有切片共用固定的横向中心、尺寸和方向。
省略中心和角度时取零；对应 solver 的尺寸参数必填。不同作用位置可以引用不同配置。
切片电荷和 ``delta_z`` 仍采用当前输入，因此冻结的是横向形状，并非完整电磁场。

``quasi-frozen`` 在每次 kick 前，对每个切片当前存活且具有有效切片编号的粒子，
重新统计质心和总体协方差。统计分母使用 :math:`N`，不是 :math:`N-1`；
同一束团宏粒子的物理权重相同。源电荷为 :math:`Q_k=N_k R Z e`。

椭圆模型的协方差特征向量确定主轴。高斯尺寸为特征值平方根；均匀分布半轴为
这些 RMS 尺寸的两倍。输出的第一个主轴尺寸为较大者，主轴逆时针角度按 pi 周期
报告到 :math:`[-\pi/2,\pi/2)`。先平移、旋转到主轴坐标计算场，再转回原横向坐标。

圆形模型采用等径向二阶矩规则：

.. math::

   \sigma^2=\frac{\operatorname{Var}(x)+\operatorname{Var}(y)}{2},
   \qquad R=2\sigma.

``gaussian_round_free_space`` 使用 sigma，``uniform_round_free_space`` 使用 R。
此规则保留相对质心的径向二阶矩；对于非圆分布，它是用户选择的圆对称近似，
不保证重建实际电场。选择均匀模型也不表示粒子已成为 KV 分布。程序不自动切换模型。

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

程序只构建被 ``SpaceCharge`` 序列项引用的配置。同一 beam 内，相同配置名的
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
     - frozen 均匀圆盘必填的正半径。
   * - ``a / b``
     - ``Semi-axis A (m) / Semi-axis B (m)``
     - float or null
     - ``null``
     - frozen 均匀椭圆必填的正半轴。


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
- GPU 后端执行已启用且有非零长度或开启损失孔径的空间电荷命令时会报错；应使用
  ``"Backend (gpu/cpu)": "cpu"``。

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
高斯的 ``size_convention`` 为 ``principal_rms``，均匀模型为 ``uniform_semi_axes``。
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
