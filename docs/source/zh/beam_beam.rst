束束相互作用（BeamBeam）
========================

``BeamBeam`` 是两条束流 sequence 中的显式会合命令，采用理想同步、同圈对撞和固定束团配对。
等待状态只协调执行顺序，不推进或修正任何束团的物理参考时间。两束的参考到达时间必须在
浮点容差内一致；多个 IP 在两条 sequence 中的访问顺序也必须相同。

配置与执行顺序
--------------

使用顶层 ``Beam beam`` 配置块。配置 ID 区分大小写，在两个输入中只能定义一次。
共享 ``Enabled`` 可由任一输入提供；两侧显式提供时必须一致，全部缺省时关闭。
``Is beam-beam`` 已停用：值为 true 时明确报错，输入生成器不再输出该旧字段。

.. code-block:: json

   {
     "Beam beam": {
       "Enabled": true,
       "Configurations": {
         "IP1": {
           "Beams": [0, 1],
           "Mode": "weak-strong",
           "Weak beam": 0,
           "Sources": {
             "0": {"Slice set": "bb_ip1"},
             "1": {
               "Slice set": "bb_ip1",
               "Method": "frozen",
               "Solver": "gaussian_round_free_space",
               "Frozen parameters": {"Sigma (m)": 0.001}
             }
           }
         }
       }
     }
   }

两条 sequence 都需要在 IP 放置 Slicer 和 BeamBeam，例如以下 :math:`s=0` 的节点。
同一位置的其他命令，包括 Injection、到达该点的 Twiss 和监测命令，也必须具有互不重复的
整数 ``Order``。某位置完全未使用 Order 时，保持原有命令类别优先级。
GUI、输入生成器、校验器与运行时使用同一排序规则。

.. code-block:: json

   {
     "slice_ip1": {
       "Command": "Slicer", "S (m)": 0, "Order": 400,
       "Purpose": "beam_beam", "Configuration": "IP1",
       "Slice set": "bb_ip1", "Coordinate": "z_rel",
       "Slice model": "equal_particle", "Number of slices": 8,
       "Z range mode": "auto"
     },
     "ip1": {
       "Command": "BeamBeam", "S (m)": 0, "Order": 500,
       "Configuration": "IP1"
     }
   }

最近一次显式 Slicer 决定成员关系；BeamBeam 不重新切片，也不重排粒子池。
显式区间必须覆盖所有存活粒子，越界时报错而不是夹到边界切片。
再次访问同一 IP 前必须重新显式执行 Slicer。保存的切片快照保持原始几何信息，
碰撞中更新的源矩另行计算。

GUI 配置
--------

GUI 左侧 **物理效应 → 束束效应** 包含全局配置、插入束束切片、插入对撞点、交叉角变换、
蟹腔和浮动束腰编辑器。全局配置支持命名 IP，并分别设置碰撞、beam0 源、beam1 源和亮度。
源编辑器只显示所选源方法适用的参数。
PIC 的 ``Propagation step (m)`` 必须手动填写，控件不能关闭，GUI 不提供默认正值。

每个共享 IP 只在一个束流输入中定义；另一输入的序列命令可填写该配置名作为引用。
共享 Enabled 控件的半选状态表示此输入不声明开关。按上述顺序与切片要求配置两条序列，
再联合校验两个输入；填写 GUI 表单不能替代另一侧匹配的序列。

接口参数
--------

顶层 ``Beam beam`` 配置块
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python 属性
     - JSON 键
     - 单位
     - 默认值
     - 说明
   * - ``enabled``
     - ``Enabled``
     - —
     - 未声明
     - 可选共享开关；两个文件显式值必须一致。全部缺省时关闭。
   * - ``configurations``
     - ``Configurations``
     - —
     - ``{}``
     - 区分大小写的名称到 IP 配置的映射；每个名称只能在两个输入中定义一次。

命名 IP 配置
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python 属性
     - JSON 键
     - 单位
     - 默认值
     - 说明
   * - ``beams``
     - ``Beams``
     - —
     - ``[0, 1]``
     - 只能为这两个 beam ID；两束的后端与粒子精度必须一致。
   * - ``mode``
     - ``Mode``
     - —
     - ``strong-strong``
     - ``strong-strong`` 更新两束；``weak-strong`` 只更新 ``weak_beam``；``weak-weak`` 不施加束束 kick。
   * - ``weak_beam``
     - ``Weak beam``
     - —
     - ``null``
     - weak-strong 必须指定 0 或 1；其他模式不能设置。
   * - ``bunch_pairs``
     - ``Bunch pairs``
     - —
     - ``null``
     - 可选完整双射 ``[[id0, id1], ...]``；省略时按相同稳定束团 ID 配对。所有 IP 共用配对关系。
   * - ``full_crossing_angle``
     - ``Full crossing angle (rad)``
     - rad
     - ``0.0``
     - 两束设计轨道相对严格反向的全偏离角，严格位于 −π 和 π 之间。
   * - ``crossing_plane``
     - ``Crossing plane (rad)``
     - rad
     - ``0.0``
     - 共同几何中的交叉平面方向角。
   * - ``interaction_map``
     - ``Interaction map``
     - —
     - ``synchro_beam_6d``
     - 支持的六维 synchro-beam 映射。
   * - ``potential_reference_length``
     - ``Potential reference length (m)``
     - m
     - ``1.0``
     - 正的固定对数势参考长度。
   * - ``sources``
     - ``Sources``
     - —
     - 必填
     - 必须恰好包含键 ``"0"``、``"1"``。两侧均需切片引用，实际供源侧还需 Method 与 Solver。
   * - ``luminosity``
     - ``Luminosity``
     - —
     - ``null``
     - 可选共享诊断配置；省略时关闭输出，生成 JSON 时也省略。

``weak-weak`` 未启用亮度时还跳过碰撞切片、坐标变换和场资源分配。
CrabCavity 与 FloatWaister 仍独立执行。固定配对不兼容 ReorganizeBunch。

BeamBeam 序列命令
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python 属性
     - JSON 键
     - 单位
     - 默认值
     - 说明
   * - ``command``
     - ``Command``
     - —
     - ``BeamBeam``
     - 命令类型。
   * - ``s``
     - ``S (m)``
     - m
     - 必填
     - 本束 sequence 中的 IP 位置。
   * - ``order``
     - ``Order``
     - —
     - ``null``
     - 遵循统一命令排序规则的可选整数顺序。
   * - ``configuration``
     - ``Configuration``
     - —
     - 必填
     - 共享 IP 配置的准确名称。

逐束源配置
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python 属性
     - JSON 键
     - 单位
     - 默认值
     - 说明
   * - ``slice_set``
     - ``Slice set``
     - —
     - 必填
     - 源束显式碰撞 Slicer 结果的非空名称。
   * - ``method``
     - ``Method``
     - —
     - ``null``
     - ``pic``、``frozen`` 或 ``quasi-frozen``；仅作为目标的一侧可只包含 Slice set。
   * - ``solver``
     - ``Solver``
     - —
     - ``null``
     - 指定 Method 时必填，组合见下方数值模型表。
   * - ``statistics_precision``
     - ``Statistics precision``
     - —
     - ``null``
     - 源矩累加精度 ``float32`` 或 ``float64``；null 跟随粒子精度。
   * - ``nx / ny``
     - ``Nx / Ny``
     - —
     - ``128``
     - 仅 PIC：每轴整数网格节点数，至少为 3。
   * - ``grid_half_width_x / grid_half_width_y``
     - ``Grid Half Width X (m) / Grid Half Width Y (m)``
     - m
     - ``null``
     - PIC 必须指定两个正半宽，定义居中的固定网格。
   * - ``deposition_method``
     - ``Particle Deposition Method``
     - —
     - ``TSC``
     - 仅 PIC：``CIC`` 或 ``TSC``，复用 SpaceCharge 沉积基础实现。
   * - ``propagation_step``
     - ``Propagation step (m)``
     - m
     - PIC 必填
     - 仅 PIC：手动指定正且有限的源势采样步长，沿碰撞距离采样；缺失或 null 均报错。
   * - ``frozen_parameters``
     - ``Frozen parameters``
     - —
     - ``null``
     - frozen 必填；按下表尺寸定义一个预设轮廓。
   * - ``slice_parameters``
     - ``Slice parameters``
     - —
     - ``{}``
     - 仅 frozen：非负切片编号映射到完整的逐片 Frozen parameters。
   * - ``frozen_optics_reference``
     - ``Frozen optics reference``
     - —
     - ``null``
     - 仅 frozen：源束在本 IP 的 Twiss 条目引用，用于沙漏传播所需的预设角向矩。
   * - ``source_center_slopes``
     - ``Source center slopes``
     - —
     - ``[0.0, 0.0]``
     - 仅 frozen：两个带符号横向中心斜率。

Frozen parameters 与 Slice parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python 属性
     - JSON 键
     - 单位
     - 默认值
     - 说明
   * - ``center_x / center_y``
     - ``Center X (m) / Center Y (m)``
     - m
     - ``0.0``
     - 预设横向中心。
   * - ``angle``
     - ``Angle (rad)``
     - rad
     - ``0.0``
     - 椭圆方向角；圆对称轮廓必须为零。
   * - ``sigma``
     - ``Sigma (m)``
     - m
     - ``null``
     - 圆 Gaussian 的正 RMS 尺寸。
   * - ``sigma_x / sigma_y``
     - ``Sigma X (m) / Sigma Y (m)``
     - m
     - ``null``
     - 椭圆 Gaussian 的正主轴 RMS 尺寸。
   * - ``radius``
     - ``Radius (m)``
     - m
     - ``null``
     - uniform/parabolic 圆盘的正支撑半径。
   * - ``a / b``
     - ``Semi-axis A (m) / Semi-axis B (m)``
     - m
     - ``null``
     - uniform/parabolic 椭圆的正支撑半轴。
   * - ``sigma_delta``
     - ``Sigma delta``
     - —
     - ``0.0``
     - 色散贡献采用的非负 RMS 动量展宽；正值要求 Frozen optics reference。

每个轮廓必须且只能指定对应的尺寸组合；没有任意协方差输入接口。

物理与数值模型
--------------

.. list-table:: 支持的源表示
   :header-rows: 1
   :widths: 20 40 40

   * - 方法
     - 求解器
     - 源表示
   * - ``pic``
     - ``fft_free_space``
     - 当前粒子通过 CIC 或 TSC 沉积到固定横向网格。
   * - ``frozen``
     - ``gaussian``、``uniform`` 或 ``parabolic``，分别加 ``round_free_space`` 或 ``ellipse_free_space`` 后缀
     - 预设空间轮廓；可选 IP 光学参数提供角向矩。
   * - ``quasi-frozen``
     - 相同的六种解析求解器
     - 每个切片对事件前，用当前切片矩重建轮廓。

PIC 源与距离插值
~~~~~~~~~~~~~~~~

``pic`` 使用 ``fft_free_space``、``Nx``、``Ny`` 以及正的
``Grid Half Width X (m)`` / ``Grid Half Width Y (m)``。
沉积复用 SpaceCharge 实现，``Particle Deposition Method`` 可选 ``CIC`` 或默认 ``TSC``。
源和目标的完整形函数支撑必须落在固定网格中；不覆盖时报数值错误，不标记物理粒子损失。
该固定网格是自由空间计算的数值区域，不表示接地导体壁。

纵向作用还需要势，不能只使用横向场。PIC 在以 :math:`S=0` 为锚点的均匀传播网格
:math:`S_k=kh` 上计算源势。目标粒子只决定需要求值的区间，其极值不决定网格原点或步长。
对于 :math:`kh\le S<(k+1)h`，四节点 Catmull--Rom 插值使用
:math:`(k-1)h`、:math:`kh`、:math:`(k+1)h`、:math:`(k+2)h` 上的源势。
横向 kick 与纵向源项均对同一个插值势求导，并采用相同的固定对数势参考长度。

每个 PIC 源都必须显式设置 ``Propagation step (m)``（Python 名为 ``propagation_step``），
其值是正且有限的步长 :math:`h`。缺失或 ``null`` 在追踪前报错；程序不自动计算或补填步长，
源斜率全部为零时也必须手动指定。该值在所选粒子精度中仍须为正且有限，
转换后下溢为零或上溢为无穷时均报错。这是沿 :math:`S` 对源势采样的间隔，
不是实际碰撞距离 :math:`S`、束团长度或沙漏效应的开关。
解析源直接计算传播量，不使用且必须省略这一 PIC 专用字段。

以下源配置示例手动选择 :math:`h=0.001\,\mathrm{m}`，该数值仅作示例，需针对具体算例检查收敛。

.. code-block:: json

   {
     "Slice set": "bb_ip1",
     "Method": "pic",
     "Solver": "fft_free_space",
     "Nx": 128, "Ny": 128,
     "Grid Half Width X (m)": 0.01,
     "Grid Half Width Y (m)": 0.01,
     "Propagation step (m)": 0.001
   }

横向网格必须在所需节点的完整外延范围内覆盖源的全部沉积支撑，即从
:math:`(k_{\min}-1)h` 到 :math:`(k_{\max}+2)h`，同时覆盖各目标粒子在相遇位置的完整取样支撑。
只覆盖目标传播距离的极值还不够。覆盖失败时应扩大横向区域或选择可分辨的传播步长；
数值网格边界不作为物理粒子损失孔径。

只对有目标粒子的传播区间求场。每个区间使用固定的四平面批次：一层基准密度和三层相对
该密度的差值。密度差求势减少纵向导数中的大数相消；横向场直接由取样势求导，
不计算未使用的网格场数组。减小 :math:`h` 可能增加需要求解的区间数。
GPU 势工作区在同一 stream 上提交对应取样后才复用；独立的源快照在整个事件中保持有效。

传播方向的插值势为分段三次、:math:`C^1` 连续，但完整粒子映射并非处处光滑：
传播节点处的二阶导数可能变化，横向沉积和取样也受网格形函数的光滑性限制。
当源势数据光滑时，Catmull--Rom 一般具有 :math:`O(h^3)` 的势误差和
:math:`O(h^2)` 的导数误差，不能称为四阶精度。应比较显式设置的 :math:`h` 与 :math:`h/2`，
分别检查纵向、横向 kick。横向网格、粒子数及切片数需要独立改变并检查收敛；
单独检查网格收敛时，应保持手动指定的步长不变。

CIC 对横向双线性插值势求导，因此力在网格线上可能跳变。位于网格线上或极近处的坐标，
在 float32 与 float64 中可能选到不同侧的梯度，不能保证这些位置的逐点结果一致。
对于固定网格势且完整插值支撑得到覆盖的情况，TSC 重建的横向势连续可微，梯度也连续。
精度对比建议使用 TSC，并同时检查横向网格与宏粒子数分辨率；仅提高算术精度不能证明 PIC 收敛。

解析源与沙漏传播
~~~~~~~~~~~~~~~~

``frozen`` 与 ``quasi-frozen`` 支持六种自由空间解析求解器：
``gaussian_round_free_space``、``gaussian_ellipse_free_space``、
``uniform_round_free_space``、``uniform_ellipse_free_space``、
``parabolic_round_free_space`` 和 ``parabolic_ellipse_free_space``。
Gaussian 跟踪在数值条件良好时使用完整闭式势导数：横向场由 Bassetti--Erskine 公式计算，
纵向源导数由 Gaussian 热方程恒等式得到，包含中心变化和全部协方差分量。
严格圆形轮廓直接使用径向 Hessian；近圆椭圆、接近奇异的协方差以及远处观察点，
保守回退到统一的共焦 Green 积分，并用两个积分阶数检查收敛。
需要标量势值时，以及 uniform/parabolic 轮廓，仍使用该积分。
CPU/GPU 闭式计算均采用 float64 中间量，返回配置的粒子精度。
两条路径均保留完整纵向导数；改变的是求值代价，不是源模型。

``Frozen parameters`` 包含默认均为零的 ``Center X (m)``、``Center Y (m)``、
``Angle (rad)``，以及对应轮廓所需的尺寸：``Sigma (m)``、
``Sigma X (m)`` / ``Sigma Y (m)``、``Radius (m)`` 或
``Semi-axis A (m)`` / ``Semi-axis B (m)``。不接受任意协方差输入。
可选 ``Slice parameters`` 按切片索引提供完整的逐片参数覆盖。
包括 frozen 在内，电荷始终由当前存活源粒子数确定。

不提供 ``Frozen optics reference`` 时，规定的宽度没有角度展宽；可选的
``Source center slopes``（两个数，默认零）决定中心传播。
需要 hourglass 时，引用源束自身位于 IP 的 Twiss 节点，角度矩由其非耦合 beta、alpha 模型构造，
再与规定的空间轮廓一起旋转。
此时 Frozen parameters 中可选的 ``Sigma delta`` 规定色散贡献；输入投影尺寸扣除色散后
必须保留正的 betatron 方差。目前普通 IP Twiss 仅提供水平色散。
内部相空间矩是派生量，不要求用户额外提供。

Quasi-frozen 在每个切片对事件前重新统计源中心和横向相空间矩。
``Statistics precision`` 可选 float32/float64，缺省跟随粒子精度。
均匀及抛物轮廓按矩重建，半轴分别为 :math:`2\sigma`、:math:`\sqrt{6}\sigma`。
圆形求解器在传播后仍采用两个横向方差的平均值。

切片事件与六维作用
~~~~~~~~~~~~~~~~~~

每个事件先准备双方的源快照，再施加两个方向的 kick。
事件按两个入口切片质心之和从大到小执行，后续事件因此能使用前一事件已更新的状态。
PIC 快照独立保存源的四个横向坐标，因此计算第二个方向时不会读取已被第一个 kick 改变的粒子。
目标坐标为 :math:`z`、对方切片质心为 :math:`\bar z_s` 时，
相遇距离为 :math:`S=(z-\bar z_s)/2`，源传播距离为 :math:`-S`。
令 :math:`X=x+Sp_x`、:math:`Y=y+Sp_y`，映射为

.. math::

   \Delta p_x=C E_x(X,Y,S),\quad \Delta p_y=C E_y(X,Y,S),
   \qquad x'=x-S\Delta p_x,\quad y'=y-S\Delta p_y,

   \eta'=\eta-\frac{C}{2}\partial_S\Psi+
   \frac{2p_x\Delta p_x+\Delta p_x^2+2p_y\Delta p_y+\Delta p_y^2}{4}.

其中 :math:`\eta=(E-E_0)/(\beta_0P_0c)`。交叉角坐标区间外，公开 ``dp`` 存储动量偏差
:math:`\delta`，边界使用稳定的相对论转换。在 PASS 能量单位中，离子的
:math:`C=\operatorname{sgn}(Z)|Z|/(A p_0)`，因为参考动量按每核子存储。
交叉角坐标区间内，分母采用该坐标系的参考动量 :math:`p_0^*=p_0\cos(\Theta/2)`。
源电荷为 :math:`N_{\rm live}wZ_s e`。不额外乘二、SpaceCharge 作用长度、切片宽度，
也不沿用 SpaceCharge 的 :math:`\gamma^{-2}` 因子。
碰撞几何采用超相对论近轴模型；精确转换 delta/eta 不表示该模型适用于一般有限速度对撞。

亮度诊断
--------

在 IP 的共享配置中启用原生亮度输出，例如：

.. code-block:: json

   {
     "Luminosity": {
       "Enabled": true,
       "Sample interval (turns)": 100,
       "Output format": "tfs"
     }
   }

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python 属性
     - JSON 键
     - 单位
     - 默认值
     - 说明
   * - ``enabled``
     - ``Enabled``
     - —
     - ``false``
     - 为每个 IP occurrence 启用一个共享亮度记录器。
   * - ``sample_interval_turns``
     - ``Sample interval (turns)``
     - turn
     - ``100``
     - 正整数；设为 10 表示每十圈采样，首末圈规则见下文。
   * - ``output_format``
     - ``Output format``
     - —
     - ``tfs``
     - ``tfs``、``hdf5`` 或 ``hdf5-gzip1``；为兼容旧输入，默认 tfs 在生成 JSON 时省略。
   * - ``reference_luminosity``
     - ``Reference luminosity (cm^-2 s^-1)``
     - cm⁻² s⁻¹
     - ``null``
     - 可选正固定参考亮度，分别用于每对束团；null 使用该对首次测量值。
   * - ``collision_frequency``
     - ``Collision frequency (Hz)``
     - Hz
     - ``null``
     - 可选正归一化频率；null 使用两束相匹配的参考周回频率。

采样与归一化
~~~~~~~~~~~~

默认使用该束团对在此 IP occurrence 的参考周回频率 :math:`\beta_0 c/C`。
两束的参考周回频率必须在运行时容差内一致，否则需要显式提供 ``Collision frequency (Hz)``。
显式值只改变诊断归一化，不会修正物理上的非同步相遇。参考 beta 用于频率换算；
碰撞平面的重叠量仍采用超相对论共同坐标模型，不表示对非相对论束流或速度明显不同的两束精确成立。

首圈、满足 ``(turn + 1) % interval == 0`` 的圈，以及配置的最后一圈都会采样。
输出圈号沿用 PASS 从零开始的约定。每行表示该次相遇的快照，不是此前采样间隔内的平均值。
两侧 BeamBeam 会合节点只生成一次物理记录，不分别重复测量。
各 IP occurrence 有独立的表格输出；改变采样间隔只改变诊断工作量，不改变碰撞 kick。
诊断本身不施加粒子 kick。GPU 浮点归约在等价运行之间可能采用不同求和顺序，
因此数值一致不意味着坐标逐位相同；重复运行同一诊断设置时也需要考虑这一点。
每次采样正式提交时，日志同时显示当前 IP 的亮度（``cm^-2 s^-1``）、因子和损失；
多对束团时显示 IP 合计，只有一对时显示该束团对的结果。
在非采样圈停止时，不会回头补算亮度。只有两束共同完成的整圈记录才正式提交，
异常中断的半圈不会作为完整圈结果写入。

对每个切片对，诊断在本事件两个 kick 之前、切片中心的碰撞平面计算横向密度重叠。
后续切片对使用当时的源状态，因此包含更早切片事件的 kick。
它与跟踪采用相同的薄切片碰撞几何，需要检查有限切片长度和所选横向密度表示的收敛性，
不能解释为任意连续六维分布的精确积分。

重叠量定义
~~~~~~~~~~

对切片对 :math:`(i,j)`，:math:`N_{A,i}`、:math:`N_{B,j}` 为真实粒子数，
:math:`\rho_{A,i}`、:math:`\rho_{B,j}` 为积分归一到一的横向密度。
下式的位置、中心及协方差均在相同碰撞平面坐标中求值，
:math:`S_{ij}=(\bar z_{A,i}-\bar z_{B,j})/2`。每对束团的重叠量为

.. math::

   \mathcal O=\sum_{i,j}\mathcal O_{ij},\qquad
   \mathcal O_{ij}=N_{A,i}N_{B,j}
      \int \rho_{A,i}(\mathbf r)\rho_{B,j}(\mathbf r)\,d^2\mathbf r.

Gaussian 切片的中心差记为 :math:`\mathbf d`，协方差之和为
:math:`\mathbf V=\mathbf C_{A,i}+\mathbf C_{B,j}`，则

.. math::

   \mathcal O_{ij}=\frac{N_{A,i}N_{B,j}}{2\pi\sqrt{\det\mathbf V}}
       \exp\!\left(-\tfrac12\mathbf d^T\mathbf V^{-1}\mathbf d\right).

PIC 中，:math:`n_{A,i,g}`、:math:`n_{B,j,g}` 为沉积到横向网格节点 :math:`g`
的真实粒子数，已包含宏粒子权重。令网格单元面积为 :math:`A_g=\Delta x\Delta y`，则

.. math::

   \mathcal O_{ij}^{\rm PIC}
   =\sum_g\frac{n_{A,i,g}n_{B,j,g}}{A_g}
   =\sum_g\nu_{A,i,g}\nu_{B,j,g}\,A_g,\qquad
   \nu_{A,i,g}=n_{A,i,g}/A_g.

因此计算需要累加 **所有切片对**，不是只取编号相同的切片。
:math:`N_{A,i}`、:math:`N_{B,j}` 已包含各片粒子数，不再乘纵向切片宽度。
两种等价 PIC 写法分别使用粒子数和数密度，不能混用归一化。

双方均为 Gaussian 解析源时，使用传播后的中心和协方差之和计算闭式重叠。
Gaussian 与 uniform/parabolic 混合时，在有界轮廓上用确定性极坐标求积积分平滑的 Gaussian 密度。
双方均为有界轮廓时，先将每条径向射线裁剪到双方支撑区域的交集，解析积分径向密度多项式的乘积，
再执行角向数值求积。角向积分和混合轮廓积分仍存在有限求积误差。

解析源与粒子表示混合时，在另一侧传播后的粒子位置上平均解析密度。
双方均为粒子表示时，在共同横向网格上沉积两侧密度，对真实粒子数密度的乘积积分。
共同区域覆盖双方已配置的 PIC 区域，间距不粗于任一输入网格；任一 PIC 源要求 TSC 时使用 TSC，
否则使用 CIC。切片对碰撞平面上的完整沉积支撑必须得到覆盖。诊断只沉积数密度，不求解势。
因此 PIC 亮度保留粒子采样和网格误差，不会将源替换成 Gaussian 拟合。
所有路径均采用真实粒子数权重，不依赖电荷符号或离子的电荷态。
CPU/GPU 均用 float64 累加；源矩和传播坐标仍含有配置跟踪精度带来的误差。

将包含真实粒子数权重的单次相遇重叠量记作 :math:`\mathcal O`，单位为
:math:`\mathrm{m}^{-2}`，输出亮度为

.. math::

   L\,[\mathrm{cm}^{-2}\mathrm{s}^{-1}]
   =10^{-4} f_{\mathrm{coll}}\,[\mathrm{Hz}]
    \mathcal O\,[\mathrm{m}^{-2}].

提供参考亮度时，每对束团的输出比值都使用该固定值；否则采用该对束团首次实际测得的亮度
作为基准。首次测量通常已经包含配置的交叉角、沙漏效应和束流状态，不能自动视为
零交叉角、无沙漏的参考亮度。研究交叉角或沙漏损失时，应提供独立测量或理论计算的适当基准。
若初始基准为零，比值无定义并输出 ``NaN``，后续非零测量不会自动替换该基准。

输出文件与列
~~~~~~~~~~~~

文件位于 ``<输出目录>/luminosity/``，文件名主体为
``{output_hms}_{configuration_with_hash}_ip{occurrence}``，扩展名为 ``.tfs`` 或 ``.h5``。
配置名会转换为安全文件名并附加哈希，以区分转换后重名的配置。
默认使用 TFS；``hdf5`` 不压缩，``hdf5-gzip1`` 使用无损 gzip level 1 和 shuffle。
各格式的列、单位和采样语义一致。每批采样结果直接追加，不需要输出完整粒子分布，
也不会重写全部历史行；不额外生成 CSV。

HDF5 遵循 :doc:`monitor/table_output` ：每列对应文件根目录下一个可扩展的一维 dataset。
``TURN``、``BUNCH_A``、``BUNCH_B`` 使用 int64，其余列使用 float64；每列 chunk 至少包含 128 行。
根属性保留对应的 TFS headers，包括配置名、IP occurrence、源模型、几何、频率约定、参考约定和单位。
保留属性为 ``_pass_table_version=1`` 及 ``_pass_table_columns`` （按输出顺序编码为 JSON 的列名列表）。

.. list-table:: 亮度表格列
   :header-rows: 1
   :widths: 35 65

   * - 列
     - 含义
   * - ``TURN`` / ``TIME``
     - 从零开始的圈号和参考相遇时间，时间单位为秒。
   * - ``BUNCH_A`` / ``BUNCH_B``
     - 实际配对的稳定束团 ID。
   * - ``FREQUENCY_HZ``
     - 该束团对亮度归一化采用的频率。
   * - ``N_A`` / ``N_B``
     - 本次相遇时存活的真实粒子数。
   * - ``OVERLAP_M2``
     - 包含粒子数权重的单次相遇重叠量，单位为平方米的倒数。
   * - ``LUMINOSITY`` / ``L_REFERENCE``
     - 测量和参考亮度，单位为每平方厘米每秒。
   * - ``FACTOR`` / ``LOSS``
     - :math:`L/L_{\rm ref}` 与 :math:`1-L/L_{\rm ref}`，不截断到 [0, 1]。

存在多对束团时，额外输出 ``BUNCH_A=BUNCH_B=-1`` 的合计行，累加各对束团的亮度和参考亮度。
合计因子是 :math:`\sum L/\sum L_{\rm ref}`，不是各束团对因子的算术平均；
``FREQUENCY_HZ`` 为 ``NaN``，因为合计行没有单一的束团对频率。
仅有一对束团时只输出该对的行，不重复输出合计。

两种格式均可使用公共表格读取器，返回带有 ``headers`` 元数据的 ``TfsDataFrame``，
分析代码无需区分文件格式。HDF5 应在运行结束后或两次写入之间读取，
不要在跟踪时持续占用文件；该写入器不启用 SWMR。

.. code-block:: python

   from PASS.utils.table_io import read_table

   luminosity = read_table("output/luminosity/run_IP1_hash_ip0.h5")
   print(luminosity.headers)
   print(luminosity[["TURN", "BUNCH_A", "BUNCH_B", "LUMINOSITY", "FACTOR", "LOSS"]])

上述文件名仅为示意，应替换为本次运行实际生成的文件；读取 TFS 时使用实际的 ``.tfs`` 文件名。

``weak-weak`` 模式可以在不施加 kick 的情况下计算亮度，但仍需匹配的碰撞 Slicer，
以及有交叉角时所需的坐标变换。至少一侧必须指定源密度方法；两侧均无密度模型时，
启用亮度诊断会报错。

CrossingAngle、CrabCavity 与 FloatWaister
-----------------------------------------

交叉角坐标变换
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python 属性
     - JSON 键
     - 单位
     - 默认值
     - 说明
   * - ``configuration``
     - ``Configuration``
     - —
     - 必填
     - 共享 IP 配置名。
   * - ``direction``
     - ``Direction``
     - —
     - 必填
     - ``forward`` 或 ``inverse``，在同一 IP 包围匹配的 Slicer 与 BeamBeam。

非零交叉角需要显式使用 ``CrossingAngle`` 包围 Slicer 与 BeamBeam，
两端的 ``Configuration`` 一致，``Direction`` 分别为 ``forward``、``inverse``。
推荐局部 Order 为到达=100、可选前置元件=200、正变换=300、Slicer=400、
BeamBeam=500、逆变换=600、可选后置元件=700、监测=800。
区间内 Slicer 必须使用 ``Coordinate: collision_z``，两次变换之间只允许匹配的
碰撞 Slicer 与 BeamBeam。区间内 ``dp`` 临时存储 eta，切片输出标明碰撞坐标系。
逆变换恢复普通 PASS 坐标，公开参考动量和参考时钟保持不变。

正碰时共同基底对 beam 0 为 :math:`(X,Y,Z)`，对 beam 1 为 :math:`(-X,Y,-Z)`；
两侧纵向坐标均以提前到达为正。无需显式交叉角变换时，碰撞计算仍包含该横向反射。
交叉角区间内的 frozen 源采用设计轨道附近的线性薄切片变换，
不将其解释为任意六维分布的精确变换。

IP 等效元件的共同参数
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python 属性
     - JSON 键
     - 单位
     - 默认值
     - 说明
   * - ``s``
     - ``S (m)``
     - m
     - 必填
     - IP 位置；碰撞相关理想元件为零长度且不施加物理孔径。
   * - ``order``
     - ``Order``
     - —
     - ``null``
     - 整数执行顺序；同一 IP 的各命令应显式区分。
   * - ``optics_reference``
     - ``Optics reference``
     - —
     - 必填
     - 本束在此 IP 的 Twiss 条目引用。
   * - ``side``
     - ``Side``
     - —
     - 必填
     - ``before`` 或 ``after``；只描述位置，不自动翻转符号。
   * - ``equivalent_dispersion``
     - ``Equivalent dispersion``
     - —
     - 零
     - 包含 Dx (m)、Dpx、Dy (m)、Dpy 的对象，各项默认零。
   * - ``longitudinal_shear``
     - ``Longitudinal shear (m)``
     - m
     - ``0.0``
     - 等效传输中的带符号纵向剪切。

``CrabCavity``、``FloatWaister`` 是普通 IP 坐标系内的独立零长度元件。
两者都需要同一 IP 的 ``Optics reference``，以及 ``Side``（``before`` 或 ``after``）。
Side 仅描述位置，不自动改变强度或相位的符号。
可选 ``Equivalent dispersion`` 包含 Dx/Dpx/Dy/Dpy，
``Longitudinal shear (m)`` 默认零。规范传输包含色散对应的纵向项，
逆传输使用 kick 后的能量。这是光学等效薄映射，不是有限长度电磁腔跟踪。

CrabCavity 需要 ``Plane`` （x/y）、带符号的 ``Phase advance (rad)`` 、
``Equivalent kick`` 和 ``Frequency (Hz)``。
FloatWaister 需要 ``Phase advance x (rad)``、``Phase advance y (rad)``；
``Mode: rfq`` 还需 ``Equivalent gx (1/m)``、``Equivalent gy (1/m)`` 和频率，
``Mode: theory`` 则需 ``Strength x``、``Strength y``。
RF 相位由 ``Phase (rad)`` （默认零）、``Phase epoch (s)`` （默认束流参考时钟原点）
及每个粒子的物理到达时间确定。等效强度已经包含电荷符号与归一化，不再重复乘电荷。
两个元件均保留与横向作用来自同一生成势的纵向 kick。

CrabCavity 参数
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python 属性
     - JSON 键
     - 单位
     - 默认值
     - 说明
   * - ``plane``
     - ``Plane``
     - —
     - 必填
     - ``x`` 或 ``y``。
   * - ``phase_advance``
     - ``Phase advance (rad)``
     - rad
     - 必填
     - 等效传输的带符号相移。
   * - ``equivalent_kick``
     - ``Equivalent kick``
     - —
     - 必填
     - 已含电荷符号的带符号归一化横向 kick 幅度。
   * - ``frequency``
     - ``Frequency (Hz)``
     - Hz
     - 必填
     - 正 RF 频率。
   * - ``phase``
     - ``Phase (rad)``
     - rad
     - ``0.0``
     - 相位历元处的 RF 相位。
   * - ``phase_epoch``
     - ``Phase epoch (s)``
     - s
     - ``null``
     - null 使用束流参考时钟原点。

FloatWaister 参数与强度约定
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python 属性
     - JSON 键
     - 单位
     - 默认值
     - 说明
   * - ``mode``
     - ``Mode``
     - —
     - 必填
     - ``rfq`` 或 ``theory``，不能混用另一模式的参数。
   * - ``phase_advance_x / phase_advance_y``
     - ``Phase advance x (rad) / Phase advance y (rad)``
     - rad
     - 必填
     - 两个平面的带符号等效相移。
   * - ``strength_x / strength_y``
     - ``Strength x / Strength y``
     - —
     - ``null``
     - theory 模式均必填，带符号强度定义见下文。
   * - ``gx / gy``
     - ``Equivalent gx (1/m) / Equivalent gy (1/m)``
     - m⁻¹
     - ``null``
     - rfq 模式均必填，表示带符号等效四极系数。
   * - ``frequency``
     - ``Frequency (Hz)``
     - Hz
     - ``null``
     - rfq 模式必填且为正；theory 模式不能设置。
   * - ``phase``
     - ``Phase (rad)``
     - rad
     - ``null``
     - 仅 RFQ；null 按零相位计算。
   * - ``phase_epoch``
     - ``Phase epoch (s)``
     - s
     - ``null``
     - 仅 RFQ；null 使用束流参考时钟原点。

``theory`` 是规定的理想行进腰点映射。在零色散、零纵向 shear、
:math:`\alpha_u^*=0` 和 :math:`\mu_u=\pi/2` 时，普通 IP 坐标中的作用为

.. math::

   \Delta u=-\frac{s_u}{2}z p_u,\qquad \Delta p_u=0,\qquad
   \Delta\eta=\frac{s_xp_x^2+s_yp_y^2}{4}.

这里 :math:`s_u` 是 ``Strength x/y``，纵向坐标正值表示提前到达。
正强度将漂移等效腰点移至局部距离 :math:`s_u z/2`。
``rfq`` 具有 RF 曲率，不能对有限束长自动等同于 theory；两个等效梯度也不能任意解释为
一只普通物理 RF 四极腔。该行进腰点模型与交叉角碰撞中的 sextupole crab-waist 方案应分别验证。

共同整圈边界上的检查点
----------------------

``PASS.commands.beam_beam`` 提供 ``capture_collision_state(sim, sequences)``
和 ``restore_collision_state(sim, sequences, state)``。检查点是含 NumPy 数组的
独立 Python 字典；这两个函数不定义文件格式。
保存要求两束完成同一整圈并回到普通 PASS 坐标、所有 Injection 批次已结束，
且监测输出已成功刷新。必须一起保存两束：状态包含粒子、束团参考量、规定时钟、
碰撞次数、已完成注入状态、尾场历史以及续跑所需的粒子/调谐监测累计量。

恢复在修改任一束前校验完整输入及 sequence 指纹、数组布局、物理参考量和命令状态，
使旧切片失效，并重新构建可丢弃的场资源。
跟踪从 ``sim.state.next_turn`` 继续至 ``cfg.num_turn``。
仅恢复协调对象的 ``state_dict()`` 不是完整仿真恢复。

检查点不包含已有输出文件，也不回退输出文件。新的输出目录不会自动补齐历史
StatMonitor 或已完成调谐窗口的文件；已有超过保存边界圈数的数据时，不能将其与续跑输出混用。
最终写入阶段若发生设备故障，不能自动回滚。

启用亮度输出后，检查点还保存各记录器的初始参考值、输出格式和最后提交圈号。
恢复要求所选格式的现有表格历史与检查点匹配，包括列定义、元数据、完整配对行和输出水位。
历史缺失、重复、截断或超前时明确报错，不会补造历史、覆盖已有运行或静默改变参考亮度。
检查点保留各亮度输出文件的路径，修改仿真的输出目录不会自动迁移这部分历史。

实现位置与验证范围
------------------

主命令与会合状态位于 ``PASS/commands/beam_beam.py``。
三个独立元件位于 ``PASS/commands/element/``。
在 ``PASS/commands/collision/`` 中，``config.py`` 定义配置，``interaction.py`` 负责源准备、
切片对顺序和六维 kick。``hourglass.py`` 负责已准备源的中心、协方差及其距离导数的传播，
并管理 PIC 源的传播采样网格。源沿 :math:`-S` 传播，势的距离导数保持碰撞平面的横向坐标不变；
亮度诊断复用同一组传播矩。``luminosity.py`` 负责密度重叠和 TFS/HDF5 追加记录，
输出生命周期由会合协调对象管理，传播模块不执行文件 I/O。

共享场求解器、沉积形函数和融合势取样核仍保留为底层数值工具。
内部传播模块不增加独立的物理沙漏元件，也不增加任意耦合 frozen 源输入。
不新增 core 碰撞模块，也不拆独立 coordinator/coordinates 文件。

CPU/GPU 支持 float32、float64，碰撞主数组跟随配置精度，时钟和 Slicer 边界保留各自所需的
较高精度。单次解析 kick、势导数、PIC 收敛、CPU/GPU 对照及真实多 IP 跟踪属于不同验证层级，
不等同于长期束流品质误差保证或 A100 性能结果。选择切片、网格及精度时，
需要将纵向误差与横向误差分开检查。

理论参考
--------

独立亮度检查可参考 `T. Sen, FERMILAB-FN-1175-AD (2022)
<https://lss.fnal.gov/archive/test-fn/1000/fermilab-fn-1175-ad.pdf>`__：
式 2.12 给出对称 Gaussian 束的交叉角与沙漏联合积分，式 2.15 为正碰沙漏极限，
式 2.17 为纯交叉角损失因子。这些公式假设两束 Gaussian 分布及 IP 光学匹配，
不能直接作为有限宏粒子或 PIC 网格结果的无误差参考。

`R. B. Palmer, SLAC-PUB-4707 (1988)
<https://inspirehep.net/files/6ca30a0993d21afd34ef80ede1e30590>`__ 提出了蟹形对撞；
`L. I. Malysheva 等，IPAC2011 TUPC004
<https://proceedings.jacow.org/IPAC2011/papers/tupc004.pdf>`__ 讨论了行进焦点及其对偏移的敏感性。
这些文献提供物理背景；PASS 的等效元件强度和纵向坐标符号由上述映射规定，
对照时必须显式匹配。特别是对较长束团，有限频率蟹腔补偿不必与理想线性倾斜产生完全相同的亮度。
