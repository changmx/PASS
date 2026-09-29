束内散射（IBS）
===============

``IBS`` 描述每个束团内部的小角度库仑散射。统一的粒子种类接口支持电子、正电子、
质子和离子，以及聚束和连续束纵向密度。三种方法分别用于局部高斯增长率诊断、
高斯动力学跟踪和局部二体碰撞。粒子更新使用 NumPy 或 CuPy；解析积分和随机数
生成在 CPU 上执行。

数值实现依据所列物理论文独立编写，没有导入、内置或翻译 Xsuite 的 IBS 实现。

方法选择
--------

.. list-table:: 可用方法
   :header-rows: 1
   :widths: 20 35 45

   * - ``Method``
     - 操作
     - 假设与限制
   * - ``bjorken_mtingwa``
     - 计算瞬时局部增长率，不踢粒子。
     - 高斯、非耦合 betatron 分布，包含水平和垂直色散；需要提供局部光学参数。测量协方差不匹配时报告 ``model_moments_matched=False``。
   * - ``kinetic``
     - 使用独立构造的张量 Ornstein--Uhlenbeck（OU）闭合，使其高斯协方差导数匹配 BM 核。
     - 要求测量协方差与指定的非耦合光学及无纵向相关性的束流矩一致；可选用已保存的纵向切片对碰撞密度加权。
   * - ``binary``
     - 在三维网格单元内随机配对粒子，散射相对动量。
     - 不假设高斯密度；需要足够的每格粒子数，并验证网格和相互作用时间间隔收敛；不接受光学参数。

所有方法要求弱库仑/等离子体耦合，并使用用户明确提供的正库仑对数；该假设
针对二体相遇，不是横向 betatron 耦合。跟踪要求束流静止系中的
热运动较小；当前实现拒绝固有速度满足 :math:`|\mathbf p_*|/(Mc)>0.05` 的情形。
参考运动可以是相对论性的，但这不是完整的相对论热碰撞算子。模型不包含大角度
散射和 Touschek 损失。辐射阻尼、量子激发和电子冷却属于独立物理过程，IBS
不会自动添加这些作用。

高斯模型不描述任意横向耦合或强非高斯分布。二体碰撞方法为非高斯研究提供局部
碰撞算子；具体应用的精度必须通过宏粒子数和网格收敛验证。此版本不包含
Nagaitsev 后端。

动力学方法匹配规定高斯模型的二阶矩，不是完整 Landau 碰撞算子，也不是文献中
修改随机踢的逐式实现。协方差匹配是必要条件，但不能证明高斯性或验证强空间
电荷工况。IBS 与空间电荷联合应用时，应针对具体束流独立验证收敛性与物理精度。

Python 配置
-----------

下列片段向已有 ``sequence`` 添加一个动力学作用点。注入和输运应按
:doc:`input_generation` 配置。局部光学必须与该位置的实际分布、输运一致，
包括下文说明的色散约定。

.. code-block:: python

   from PASS.para.api import generate_input
   from PASS.para.schema import IBSConfig, IBSConfiguration, IBSItem, IBSOpticsConfig

   ibs = IBSConfig(
       enabled=True,
       configurations={
           "CoreIBS": IBSConfiguration(
               method="kinetic", coulomb_log=12.0,
               random_seed=2026, bunched=True,
           ),
       },
   )
   sequence.add(
       "ibs_at_s10",
       IBSItem(
           s=10.0, configuration="CoreIBS", interaction_length=1.0,
           optics=IBSOpticsConfig(beta_x=8.0, beta_y=6.0, dx=0.5),
           save_diagnostics=True,
       ),
   )
   generate_input(main, sequence, "beam0.json", intrabeam_scattering=ibs)

``Coulomb log=12`` 是输入示例，不是普适物理默认值。应针对束流和适用工况选择
碰撞参数的上下限；PASS 不自动推断截断、屏蔽或强耦合修正。
``coulomb_log_note`` 可记录选取 :math:`\ln\Lambda=\ln(b_{\max}/b_{\min})`
时使用的物理截断方法、数值、单位与来源。该文字仅用于来源记录，不改变碰撞
计算。碰撞网格属于数值密度分辨率选择，单元尺寸不决定物理库仑对数。

只计算增长率时选择 ``method="bjorken_mtingwa"``，相互作用长度可以为零，仍可
计算增长率。二体碰撞跟踪选择 ``method="binary"``，需要时设置
``grid_shape=(nx, ny, nz)`` 和可选的
``grid_bounds=((xmin, ymin, zmin), (xmax, ymax, zmax))``（实验室坐标，单位 m），
并省略 ``optics``。``collision_steps`` 将作用时间分为若干步，每步重新随机配对。
配置名称区分大小写；加载器
接受不区分大小写的 schema 字段名。

输入参数
--------

顶层 JSON 配置块为 ``Intrabeam scattering``。``IBSConfig`` 包含
``enabled`` / ``Enabled``（严格布尔值，默认 ``False``）和
``configurations`` / ``Configurations``（名称到 ``IBSConfiguration`` 的映射）。
关闭全局开关时忽略配置内容，并关闭其所有 IBS 命令。

.. list-table:: ``IBSConfiguration``
   :header-rows: 1
   :widths: 22 23 20 35

   * - Python 名称
     - JSON 字段
     - 默认值 / 类型
     - 含义
   * - ``method``
     - ``Method``
     - ``"bjorken_mtingwa"``
     - 上述三种方法之一。
   * - ``coulomb_log``
     - ``Coulomb log``
     - 必填浮点数
     - 有限且严格为正，无量纲。
   * - ``coulomb_log_note``
     - ``Coulomb log note``
     - ``None``；字符串或 null
     - 可选的物理截断方法和来源。提供时必须非空且不包含首尾空白；仅记录来源，不参与物理状态哈希。
   * - ``random_seed``
     - ``Random Seed``
     - ``None``；整数或 null
     - 非负严格整数提供可复现的初始化；``None`` 使用新熵。拒绝布尔、浮点和数字字符串。
   * - ``bunched``
     - ``Bunched``
     - ``True``；严格布尔值
     - 聚束密度，或沿周长分布的连续束密度。
   * - ``slice_set``
     - ``Slice set``
     - ``None``；名称或 null
     - 动力学方法可选用已执行 Slicer 的线密度加权；其他方法要求 null。
   * - ``grid_shape``
     - ``Grid shape``
     - ``(8, 8, 8)``；三个严格正整数
     - 二体碰撞沿 x、y、纵向的网格数；非默认值仅允许用于 binary 方法。
   * - ``grid_bounds``
     - ``Grid bounds (m)``
     - ``None`` 或两个三元组
     - 实验室 ``(x, y, z_rel)`` 米坐标中的二体碰撞网格下界和上界。各轴上界必须大于下界；粒子在范围外时报告错误。
   * - ``collision_steps``
     - ``Collision steps``
     - ``1``；严格正整数
     - 二体碰撞物理作用时间的细分次数，每步重新随机配对。命令内位置保持不变；非默认值仅允许用于 binary。
   * - ``matching_tolerance``
     - ``Matching tolerance``
     - ``0.1``；(0, 0.25] 内浮点数
     - 下文定义的高斯无量纲协方差偏差上限。非默认值仅允许用于高斯方法；无法关闭动力学匹配检查。
   * - ``max_scattering``
     - ``Max scattering``
     - ``0.05``；(0, 0.05] 内浮点数
     - 动力学摩擦作用量上限；二体碰撞每个角度子步的半散射角正切方差上限。
   * - ``max_substeps``
     - ``Max substeps``
     - ``1000``；严格正整数
     - 超出时报告错误，不静默截断碰撞强度。

.. list-table:: ``IBSItem`` sequence 命令
   :header-rows: 1
   :widths: 22 23 20 35

   * - Python 名称
     - JSON 字段
     - 默认值 / 类型
     - 含义
   * - ``command``
     - ``Command``
     - ``"IBS"``
     - 注册的命令名称。
   * - ``s``
     - ``S (m)``
     - ``0.0``；有限浮点数
     - sequence 位置，单位 m。
   * - ``order``
     - ``Order``
     - ``None``；严格整数或 null
     - 同位置的可选显式执行顺序，见 :doc:`input_generation`。
   * - ``configuration``
     - ``Configuration``
     - 必填名称
     - 引用 ``Intrabeam scattering.Configurations``。
   * - ``interaction_length``
     - ``Interaction length (m)``
     - 必填非负浮点数
     - 正物理作用长度；此命令不会执行粒子输运。
   * - ``optics``
     - ``Optics``
     - ``None`` 或 ``IBSOpticsConfig``
     - 高斯方法必填；二体碰撞不接受该参数。
   * - ``is_enabled``
     - ``Is enabled``
     - ``True``；严格布尔值
     - 在全局开关开启时启用此作用点。
   * - ``save_diagnostics``
     - ``Save diagnostics``
     - ``False``；严格布尔值
     - 在 ``save_turns`` 选定的圈保存诊断 JSON。
   * - ``save_turns``
     - ``Save turns``
     - ``[]``；严格整数选择列表
     - 启用保存时，空列表表示每次执行都保存；非空使用 ``[turn]`` 或 ``[start, end, step]`` 行选择从零编号、包含结束圈的范围。

``IBSOpticsConfig`` 必须提供正且有限的 ``beta_x`` / ``Beta x (m)`` 和
``beta_y`` / ``Beta y (m)``。其余有限浮点数均默认为零：
``alpha_x`` / ``Alpha x``、``alpha_y`` / ``Alpha y``、``dx`` / ``Dx (m)``、
``dpx`` / ``Dpx``、``dy`` / ``Dy (m)``、``dpy`` / ``Dpy``。
色散导数对应 PASS 的归一化机械动量，不是精确横向斜率；高斯近似只在近轴阶次
下等同二者。此版本不会从 Twiss 命令自动推断 IBS 光学；用户应明确提供一致的
局部光学参数。

PASS 要求四个色散字段均为对 :math:`\delta=(P-P_0)/P_0` 的导数。
MAD-X 原生 ``TWISS`` 表给出的则是对 ``PT`` 的导数；参考粒子附近满足

.. math::

   \mathrm{PT}=\frac{E-E_0}{P_0c}\simeq\beta_0\delta,
   \qquad D_{a,\delta}=\beta_0 D_{a,\mathrm{PT}},
   \qquad a\in\{x,p_x,y,p_y\}.

因此将 MAD-X 原生 ``TWISS`` 的 ``DX``、``DPX``、``DY``、``DPY`` 用于此处前，
应乘以 :math:`\beta_0`；参见
`MAD-X 手册的色散定义
<https://raw.githubusercontent.com/MethodicalAcceleratorDesign/MAD-X/master/doc/usrguide/Introduction/tables.html#linear>`_。
MAD-X ``IBS`` 表中的色散已经转换为动量偏差约定，不能再次乘以 beta。
MAD-X 5.09.03 的公开 API 检查还表明，该表使用中点位置以及相邻 TWISS 值的
平均值，因此不能只按元件名称将这些行替换为端点光学参数。

``IBSOpticsConfig.from_twiss(twiss, endpoint="exit", dy=0.0, dpy=0.0)``
可从 PASS ``TwissItem`` 或其完整 JSON 映射构造这些显式值。
``endpoint="exit"`` 选择当前 beta、alpha、Dx 和 Dpx；
``endpoint="entrance"`` 选择对应的 previous 值。结果独立执行
``IBSOpticsConfig`` 校验。此辅助函数不接受任意 MAD-X 表行，不推断缺失的
必填 Twiss 数据，不选择 IBS sequence 位置，也不改变运行时输运。
``TwissItem`` 不包含垂直色散，因此非零 ``dy``、``dpy`` 需要显式提供。

.. code-block:: python

   # twiss_item is the PASS Twiss transport map associated with this location.
   optics = IBSOpticsConfig.from_twiss(twiss_item, endpoint="exit", dy=0.0, dpy=0.0)

作用时间、坐标与密度
--------------------

每个命令的实验室作用时间和束流静止系作用时间分别为

.. math::

   \Delta t=\frac{L_{\mathrm{IBS}}}{\beta_0 c},
   \qquad \Delta t_* = \frac{\Delta t}{\gamma_0}.

每次执行施加完整的规定作用量。环上分布式模型的各相互作用长度应代表相应格点
路段，不应在每个节点都使用整圈周长。每次调用重新计算系数；动力学跟踪还会
在每个自适应子步重新计算。IBS 是独立的正时间碰撞算子，不能把积分器的负子步
作为碰撞作用时间。

输入校验按束流累加已启用动力学和二体碰撞命令的非负相互作用长度，包括零长度
节点。存在此类命令且总长度在
相对容差 :math:`10^{-9}` 下不同于周长，报告 ``ibs.exposure_length`` 警告。
只做诊断的 BM 节点及关闭的命令不参与求和。局部环段研究可以有意触发该警告；
它不会修改或拒绝规定的作用量。总长度相等既不能证明空间覆盖正确，也不能在
参考 beta 变化时证明总碰撞时间正确。

IBS 保持 ``x``、``y``、连续 ``z_rel``、参考能量和 ``bunch.t0`` 不变，踢只更新
``px``、``py`` 和总相对动量偏差 ``dp``。存活宏粒子数乘以束团的宏粒子权重得到
真实粒子数。离子的经典半径使用完整离子质量 :math:`M` 和电荷 :math:`q=Ze`：

.. math::

   r_0=\frac{q^2}{4\pi\epsilon_0 M c^2}.

高斯束流矩在扣除质心与指定色散之后测量。聚束模型使用连续 ``z_rel`` 的 RMS。
连续束高斯模型将 :math:`\sigma_z` 替换为 :math:`C/(2\sqrt{\pi})`，对应周长 C
上的均匀线密度。

动力学跟踪不指定 ``Slice set`` 时使用高斯空间平均系数。指定时应先执行命名
:doc:`slicer`。聚束使用 ``Purpose="general"`` 和 ``Coordinate="z_rel"``；
连续束还允许 ``z_periodic``。严格保留已保存的成员关系和切片宽度。重组会使其
失效，需要再次显式执行 Slicer。所有存活粒子都必须具有有效的已保存切片编号。
加权系数为测得的归一化线密度除以聚束的 :math:`1/(2\sqrt{\pi}\sigma_z)` 或
连续束的 :math:`1/C`。该加权不会将横向高斯闭合变成完整的局部非高斯模型。

二体碰撞使用冻结局部快照近似 :math:`z_* = \gamma_0 z_{\mathrm{rel}}`，不会精确
重建等时粒子事件。即使分布形状为非高斯，此适配仍要求接近参考运动的近轴分布
和较窄的相对动量展宽；参考 beta 很小时，仅满足热速度上限并不能保证该近似。
连续束只在临时
碰撞数组中折叠纵向位置，绝不折叠保存的 ``z_rel``。网格覆盖粒子范围；连续束的
纵向范围使用整圈周长。显式 ``Grid bounds (m)`` 替代自动范围，并且必须覆盖
所有存活粒子，不会静默排除束晕。连续束的纵向边界必须在相对容差
:math:`10^{-12}` 内等于实验室区间 :math:`[-C/2,C/2]`；接受的舍入误差会被
归一到这两个精确端点。命令将纵向边界乘以 :math:`\gamma_0`，转换为静止系
米坐标。固定边界有助于区分网格分辨率效应和样本极值变化。
空单元或只有一个粒子的单元无法发生碰撞。诊断会报告
网格占据与未配对粒子；若过细网格的大多数单元只有单个粒子，就没有充分解析
碰撞动力学。

连续束 IBS 要求 ``Harmonic Number=1``，由单个束团分组代表整圈粒子数。
聚束 IBS 分别处理各束团，假设它们的物理分布互不重叠；不描述重叠束团分组
之间的碰撞。

高斯匹配检查
------------

每次高斯 IBS 命令求值都先对测量坐标去质心，并定义扣除色散的坐标
:math:`x_\beta=x-D_x\delta`、:math:`p_{x\beta}=p_x-D_{p_x}\delta`，
y 平面同理。使用测得的本征发射度和指定光学构造

.. math::

   u_x=\frac{x_\beta}{\sqrt{\varepsilon_x\beta_x}},\qquad
   u_{p_x}=\frac{\beta_xp_{x\beta}+\alpha_xx_\beta}
                   {\sqrt{\varepsilon_x\beta_x}},\qquad
   u_\delta=\frac{\delta}{\sigma_\delta},\qquad
   u_z=\frac{z-\langle z\rangle}{\sigma_z}.

连续束的向量顺序为 :math:`(u_x,u_{p_x},u_y,u_{p_y},u_\delta)`；聚束在末尾
增加 :math:`u_z`。规定的匹配模型满足

.. math::

   C_{\mathrm{norm}}=\langle\mathbf u\mathbf u^T\rangle=I,
   \qquad e_{\mathrm{match}}=
   \max_{i,j}|(C_{\mathrm{norm}}-I)_{ij}|.

此检查能够发现光学不匹配、不支持的 betatron 相关性，以及聚束中涉及 z 的
相关性，包括动量啁啾；不会检测正态性、束晕或高阶矩。默认 0.1 是协方差检查
阈值，不代表增长率误差保证在 10% 以内。在受控高斯包络扰动校准中，协方差
偏差 0.01、0.02、0.05、0.09 分别产生 2.21%、4.33%、10.26%、17.22% 的
增长率向量相对误差。这些是特定算例的校准结果，不是普适误差上界；耦合情形
还可能在投影增长率中隐藏张量误差。

定量计算可以从 0.01--0.02 的容差起步，前提是宏粒子数充分，并验证抽样和
阈值收敛。有限样本的协方差噪声本身可能触发拒绝；噪声占主导时应增加粒子数，
不能靠放宽阈值掩盖物理不匹配。持续存在的不匹配或耦合应改用二体碰撞模型，
并独立验证其网格、粒子数和时间步收敛。

``kinetic`` 在每个子步前以及最终暂存结果上检查匹配；失败时拒绝该命令，
不提交粒子或随机流更新。``bjorken_mtingwa`` 保留增长率诊断，并报告匹配情况
和 ``model_moments_matched``。若后者为 false，这些增长率仅描述规定的高斯模型，不能
当作测得实际分布的物理增长率。
独立系数 API 只接收束流参数和光学，不能检查粒子或执行这一匹配检查。

增长率与碰撞约定
----------------

独立 API 位于 ``PASS.commands.ibs``，提供 ``IBSBeamParameters``、
``IBSOptics``、``compute_local_coefficients``（别名 ``compute_bjorken_mtingwa``）
和 ``ring_average_growth_rates``，不依赖 Simulation 或粒子池。
全环平均要求显式非负路段长度或驻留时间权重，不会自动补充闭合格点路段。

高斯矩阵构造使用动量顺序
:math:`(P_x/P_0,P_y/P_0,\delta/\gamma_0)`，相应单位基向量为
:math:`\mathbf e_x,\mathbf e_y,\mathbf e_z`。对于横向平面
:math:`u\in\{x,y\}`，定义

.. math::

   \phi_u=D_{p_u}+\frac{\alpha_u D_u}{\beta_u},
   \qquad
   L_u=\frac{\beta_u}{\varepsilon_u}
       (\mathbf e_u-\gamma_0\phi_u\mathbf e_z)
       (\mathbf e_u-\gamma_0\phi_u\mathbf e_z)^T
       +\frac{\gamma_0^2D_u^2}{\beta_u\varepsilon_u}
        \mathbf e_z\mathbf e_z^T,

   L_z=\frac{\gamma_0^2}{\sigma_\delta^2}\mathbf e_z\mathbf e_z^T,
   \qquad L=L_x+L_y+L_z,\qquad \Sigma=L^{-1}.

这里 :math:`\Sigma` 是给定横向位置时的条件热动量协方差，不是整个束团的
投影动量协方差。独立数值求解的 Bjorken--Mtingwa 积分和归一化系数为

.. math::

   J=\int_0^\infty
      \frac{\sqrt{\lambda}\,(L+\lambda I)^{-1}}
           {\sqrt{\det(L+\lambda I)}}\,d\lambda,
   \qquad
   a=\frac{cNr_0^2\ln\Lambda}
          {8\pi\beta_0^3\gamma_0^4\varepsilon_x\varepsilon_y
           \sigma_z\sigma_\delta}.

PASS 使用的摩擦、扩散和物理协方差导数为

.. math::

   F=2aJL,\qquad D=2a[\operatorname{tr}(J)I-J],
   \qquad K=D-F\Sigma-\Sigma F^T
           =2a[\operatorname{tr}(J)I-3J].

通常的 BM 核为 :math:`K/2`。将增长率积分转为粒子扩散过程时必须保留这个
二倍因子。实际条件协方差等于匹配模型 :math:`\Sigma` 时，
迹 :math:`\operatorname{tr}(K)=0` 表示主导阶热能守恒。协方差不匹配时，
实际 OU 协方差导数的迹可以不为零。
局部增长率为

.. math::

   G_x=\tfrac12\operatorname{tr}(L_xK),\qquad
   G_y=\tfrac12\operatorname{tr}(L_yK),\qquad
   G_\delta=\frac{\gamma_0^2K_{zz}}{2\sigma_\delta^2}.

增长率单位为实验室时间的每秒，定义为

.. math::

   G_x=\frac{d\ln\varepsilon_x}{dt},\qquad
   G_y=\frac{d\ln\varepsilon_y}{dt},\qquad
   G_\delta=\frac{d\ln\sigma_\delta}{dt},\qquad
   G_{\delta^2}=2G_\delta.

横向振幅增长率为 :math:`G_x/2` 与 :math:`G_y/2`。纵向增长率描述瞬时动量踢，
没有进行同步振荡相位平均。后续 RF 和输运决定该踢如何在束长和动量展宽之间
分配。与其他程序报告的 IBS 时间比较前，必须统一这些定义。

动力学模型在 :math:`(P_x/P_0,P_y/P_0,\delta/\gamma_0)` 中构造去质心的条件
热动量残差，并求解

.. math::

   d\mathbf w=-F\mathbf w\,dt+B\,d\mathbf W,
   \qquad D=BB^T,\qquad
   \frac{d\Sigma}{dt}=D-F\Sigma-\Sigma F^T.

条件均值包含横向位置相关性和色散。扩散为半正定，但较热自由度的净增长率可以
为负。实现保留摩擦和非对角张量项。冻结系数的 Ornstein--Uhlenbeck 子步采用
精确解，自适应正子步更新束流矩；有限样本质心修正保持平均归一化动量。
当分布的条件协方差与模型匹配时，束流静止系主导阶热能守恒成立于瞬时高斯
碰撞期望；有限时间步和有限粒子样本会引入误差，需要
进行收敛验证。此闭合不强制逐对能量守恒。

二体碰撞方法采用等权重的局部随机配对与 Takizuka--Abe 小角度散射定律。
约化质量 :math:`\mu=M/2`、相对速度 g 下，半散射角正切的方差为

.. math::

   \left\langle\tan^2(\theta/2)\right\rangle
   =\frac{q^4 n\ln\Lambda\,\Delta t_*}
          {8\pi\epsilon_0^2\mu^2 g^3}.

在粒子对的零总动量系中，使用均匀方位角旋转相对动量，逐对四动量在舍入精度
内守恒。散射频率仍使用非相对论热运动近似。奇数粒子单元随机选一个粒子不参与
配对，并在期望意义下补偿配对作用量。``Collision steps`` 细分物理作用时间，
每次细分重新随机配对，位置保持固定。内部角度子步细分已选粒子对的旋转，不会
重新配对。应同时验证物理碰撞步数和外部命令时间间隔的收敛；两种细分都不会
在 IBS 内添加输运。

诊断与可复现性
--------------

``IBS.last_diagnostics`` 提供最近一次执行的诊断。启用保存时，
``Save turns=[]`` 表示每次执行都保存。例如，
``save_turns=[[0], [10, 50, 10]]`` 选择第 0、10、20、30、40、50 圈；
``save_turns=[10, 50, 10]`` 是单个范围的简写。圈数从零编号，结束圈包含在内。
``Save diagnostics=False`` 无论圈选择如何都不写文件。保存间隔不改变碰撞
作用时间、系数更新、随机抽样或 ``last_diagnostics``。JSON 文件位于
本次运行输出目录下的
``ibs/<唯一目录>/beam<id>_<命令>/turn_<圈数>_call_<次数>_<唯一后缀>.json``。
实际求值的高斯记录包含测得束流参数和增长率。BM 报告 ``matching`` 与 ``model_moments_matched``；
动力学方法报告 ``initial_matching``、``final_matching`` 和 ``max_matching_error``；
后者包含每个子步前及最终检查的最大偏差。
匹配记录包含 ``dimension``、``normalized_covariance``、``max_abs_error``、``tolerance`` 和
``matched``。

每份诊断的顶层包含 ``interaction_length_m`` 和 ``coulomb_log_note``。
每个束团记录包含 ``dt_lab_s`` 和 ``interaction_length_over_circumference``；
仅在周长非正时后者为 null。这些量公开规定的局部作用量，不会推断格点覆盖，
也不会证明库仑对数物理截断的正确性。

实际求值的二体碰撞记录包含 ``grid_shape``、``grid_bounds_m``（静止系米坐标）、
``n_grid_cells``、``n_empty_cells``、``singleton_fraction`` 和
``occupancy_histogram`` 中的 ``{"particles": count, "cells": count}`` 条目，
包括空单元。``n_pairs`` 为每个物理步可配对的粒子对数；``n_pairings`` 为已完成
``n_collision_steps`` 的配对总次数。``singleton_fraction`` 是单粒子单元中的
粒子数占总粒子数的比例；``n_unpaired_particles`` 为每个物理步未配对的数量，
不是跨步去重后的粒子数。``n_substeps`` 为最大角度细分数，
``max_scattering`` 为每个物理碰撞步在角度细分前的最大方差，因此该诊断值可以
超过输入规定的每子步上限。``n_zero_relative_pairs`` 累计
相对运动为零的粒子对数。``proper_speed_max_over_c`` 为初始状态与每个物理
碰撞步完成后的最大固有速度除以 c。诊断说明所选模型的结果，不能替代时间步、
网格和粒子数收敛检查。

随机流属于特定束流、命令和束团。固定种子在相同执行条件下提供可复现初始化。
``state_dict()`` 与 ``load_state_dict()`` 保存和恢复 IBS 随机流及执行计数。
这是 IBS 组件检查点，不是完整模拟重启：粒子、参考状态、圈数、切片及其他集体
效应状态必须匹配。CPU/GPU 的浮点计算可能产生不同轨迹，应验证后端间的统计
一致性。
解析较小 IBS 动量增量时建议使用 ``Particle Precision="float64"``；float32
存储可能将小增量舍入为零。
诊断开关、保存圈选择及库仑对数备注不参与物理配置哈希，因此修改这些输出或
来源记录设置，不会使其他条件仍匹配的 IBS 组件检查点失效。

参考文献
--------

* J. D. Bjorken and S. K. Mtingwa, *Intrabeam Scattering*, Particle Accelerators
  **13** (1983), 115--143：高斯 IBS 理论。
* M. Zampetakis et al., `Interplay of space charge and intrabeam scattering in
  the LHC ion injector chain <https://arxiv.org/abs/2310.03504>`_：高斯矩阵
  系数及摩擦/扩散背景。PASS 使用独立的完整张量闭合，瞬时增长率约定见上文。
* T. Takizuka and H. Abe, `A binary collision model for plasma simulation with
  a particle code <https://doi.org/10.1016/0021-9991(77)90099-7>`_, Journal of
  Computational Physics **25** (1977), 205--219：随机二体碰撞定律。
