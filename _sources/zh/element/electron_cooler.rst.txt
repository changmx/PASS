ElectronCooler（电子冷却）
========================================

``ElectronCooler`` 在有限长度冷却段内输运离子，并计算离子与给定电子库之间的
非相干碰撞。电子束支持直流和高斯束团模式，横向密度可选均匀圆形或高斯分布。
NumPy、CuPy 粒子数组使用相同物理模型。``ElectronBeamConfig`` 和
``ElectronCoolerItem`` 由 ``PASS.para.schema``、``PASS.para.api`` 导出，
无需新增顶层命名配置。

电子库不会响应离子束而演化。本元件不模拟阴极发射、自洽电子光学、集体屏蔽动力学、
复合、相干电子冷却或电子库耗竭。

碰撞模型
--------

.. list-table::
   :header-rows: 1
   :widths: 22 33 45

   * - ``Model``
     - 功能
     - 假设与限制
   * - ``gaussian``
     - 非磁化高斯 Landau 摩擦和动量扩散。
     - 支持任意对称正定 3×3 局部电子速度协方差。名称指速度分布，与空间密度
       形状无关。即使设置螺线管场，碰撞算子也不包含电子回旋。
   * - ``parkhomchuk``
     - 经验磁化摩擦。
     - 要求磁场非零、电子轴倾角为零且 ``Diffusion=False``。适用于经过标定的摩擦力、冷却率研究，
       不提供平衡扩散定律。
   * - ``magnetized_collision``
     - 有限作用窗磁化响应摩擦与扩散。
     - 弱响应、弱偏转、局部均匀且回旋对称电子库的数值参考模型。要求显式冲量参数
       截断、正轴向磁场、电子轴倾角为零、两个横向热方差相等且交叉协方差为零。

磁化响应后端不是覆盖高密度实际冷却器全部工况的通用模型。超出适用域或积分不收敛
时明确报错，不静默切换公式。其约束包括 :math:`\omega_pT_*\leq0.3`、
:math:`\omega_iT_*\leq0.1` 和不大于 0.1 的紫外截断偏转度量。
:math:`T_*` 是整段物理作用时间，:math:`\omega_p` 是电子等离子体频率，
:math:`\omega_i=|Z|eB/M` 是离子回旋频率。这些有限窗系数作为局部 Markov
踢作用：离子动量和局部电子库在该物理窗口内必须变化很小；程序不求解相邻踢之间
全部时间相关性。系数描述与初始无关联、局部均匀的新电子库之间的一次有限作用，
包含瞬态极化；不声称有限窗 Maxwell 细致平衡，也不代表一般屏蔽等离子体平衡。

最大电子热 RMS 速度、实际遇到的局部平均剪切速度以及参与碰撞的离子相对电子速度
均不得超过 :math:`0.05c`。实验室系参考运动可相对论化，但热碰撞模型仍为非相对论。
高斯 RMS 检查用于控制近似适用域，不截断高斯分布的数学尾部。

Python 示例
-----------

将下面元件加入含 Injection 和相邻束线输运的序列。电子能量独立于离子参考能量，
不会跟随离子质心自动变化。速度匹配要求
:math:`E_{k,e}=(\gamma_i-1)m_ec^2`。

.. code-block:: python

   from PASS.para.schema import ElectronBeamConfig, ElectronCoolerItem

   electrons = ElectronBeamConfig(
       kinetic_energy=1000.0, profile="uniform_round", radius=0.01,
       mode="dc", current=0.1,
       temperature_transverse=0.1, temperature_longitudinal=0.001,
   )
   sequence.add(
       "cooler",
       ElectronCoolerItem(
           s=12.0, length=2.0, electron_beam=electrons,
           model="gaussian", diffusion=True, coulomb_log=10.0,
           num_slices=8, random_seed=2026, save_diagnostics=True,
           save_turns=[[0, 1000, 10]],
       ),
   )

该元件占据 [10, 12] m，``S (m)`` 是出口位置。不要再用独立 Drift 或 Solenoid
重复输运同一段。

元件接口
--------

别名不区分大小写；未知字段和重复别名会报错。输入必须有限。整数控制量与布尔量
不接受字符串或可强制转换的浮点数。标准 ``Order`` 和孔径字段继承自
``ElementBase``，参见 :doc:`../input_generation`。

.. list-table::
   :header-rows: 1
   :widths: 25 29 15 31

   * - Python 字段
     - JSON 键
     - 默认值
     - 含义
   * - ``s``、``length``
     - ``S (m)``、``Length (m)``
     - 必填；0
     - 出口位置和非负物理长度。
   * - ``electron_beam``
     - ``Electron beam``
     - 必填
     - 一个 ``ElectronBeamConfig``，见下表。
   * - ``model``
     - ``Model``
     - ``gaussian``
     - 上述三种碰撞模型之一。
   * - ``collisions``、``diffusion``
     - ``Collisions``、``Diffusion``
     - True、True
     - 分别控制整体碰撞漂移与噪声、仅扩散噪声。关闭碰撞后仍保留输运和可选平滑场。
   * - ``magnetic_field``、``mean_space_charge``
     - ``Magnetic field (T)``、``Mean space charge``
     - 0、False
     - 带符号均匀轴向磁场；可选给定电子束平滑自场。
   * - ``coulomb_log``
     - ``Coulomb log``
     - None
     - 正的固定 Gaussian/Parkhomchuk 库仑对数；省略则使用有效截断。
       磁化响应不允许指定此项。
   * - ``min_impact_parameter``、``max_impact_parameter``
     - ``Min impact parameter (m)``、``Max impact parameter (m)``
     - None、None
     - 磁化响应要求显式 :math:`0<b_{\min}<b_{\max}`；最大值也可限制
       Gaussian/Parkhomchuk 自动截断，但不能与固定库仑对数同时使用。
   * - ``effective_velocity_spread``
     - ``Effective velocity spread (m/s)``
     - 0
     - Parkhomchuk 中代表未解析场误差或漂移的附加 RMS 速度。
   * - ``num_slices``
     - ``Num slices``
     - 1
     - 正整数输运/碰撞分段数。
   * - ``max_fractional_step``、``max_substeps``
     - ``Max fractional step``、``Max substeps``
     - 0.05、1000
     - 碰撞漂移/扩散自适应步长控制，以及每个输运段最大内部碰撞子步数。
   * - ``quadrature_order``
     - ``Quadrature order``
     - 64
     - Gaussian 速度积分每个积分区间的阶数，至少 8。
   * - ``radial_order``、``polar_order``、``azimuthal_order``、``time_order``
     - ``Radial order``、``Polar order``、``Azimuthal order``、``Time order``
     - 16、12、16、64
     - 磁化响应初始积分阶数；加密时四者同时增加。
   * - ``quadrature_rtol``、``max_refinements``
     - ``Quadrature relative tolerance``、``Max refinements``
     - 0.02、2
     - 磁化积分收敛控制；容差范围 [1e-6, 0.1]。不收敛时抛出错误。
   * - ``random_seed``
     - ``Random Seed``
     - None
     - 非负严格整数，或非确定性熵；JSON 保留 null。
   * - ``save_diagnostics``、``save_turns``
     - ``Save diagnostics``、``Save turns``
     - False、[]
     - 空选择表示每圈；否则使用 [turn] 或含端点的 [start, end, step] 条目。

仅属于未启用模型的数值参数不能偏离默认值。离子自身空间电荷和 IBS 使用独立显式
命令；组合格架时避免重复计算相同物理作用。

电子库接口
----------

宽度参数描述入口；可选出口宽度规定宽度自身沿程线性插值，省略则保持常数。
这是给定包络，不是自洽电子输运或连续性方程解。

.. list-table::
   :header-rows: 1
   :widths: 25 30 14 31

   * - Python 字段
     - JSON 键
     - 默认值
     - 含义
   * - ``kinetic_energy``
     - ``Kinetic energy (eV)``
     - 必填
     - 每个电子的正动能。
   * - ``profile``
     - ``Profile``
     - ``uniform_round``
     - 横向空间密度：``uniform_round`` 或 ``gaussian``。
   * - ``radius``、``radius_exit``
     - ``Radius (m)``、``Exit radius (m)``
     - None
     - 均匀圆形分布的正硬边界半径，入口半径必填。
   * - ``sigma_x``、``sigma_y``、``sigma_x_exit``、``sigma_y_exit``
     - ``Sigma x (m)``、``Sigma y (m)``、``Exit sigma x (m)``、``Exit sigma y (m)``
     - None
     - 高斯正 RMS 尺寸，两个入口尺寸均必填。
   * - ``mode``、``current``
     - ``Mode``、``Current (A)``
     - ``dc``、None
     - 直流模式要求非负电子电流幅值。
   * - ``bunch_charge``、``sigma_time``
     - ``Bunch charge (C)``、``Sigma time (s)``
     - None
     - ``gaussian_bunch`` 模式要求非负电荷幅值和正 RMS 时间宽度；
       不能同时指定电流。
   * - ``bunch_center_time``、``repetition_frequency``
     - ``Bunch center time (s)``、``Repetition frequency (Hz)``
     - 0、None
     - 入口处电子脉冲中心时刻和可选正重复频率。重复模式定义无限脉冲列；
       中心时刻是相位原点，不是开启时刻。省略重复频率时只有一个脉冲。
   * - ``center_x``、``center_y``、``angle_x``、``angle_y``
     - ``Center x (m)``、``Center y (m)``、``Angle x (rad)``、``Angle y (rad)``
     - 0
     - 电子轴入口偏移和方向；局部坐标基随电子轴旋转，离子参考轨道不随之改变。
       两个磁化碰撞模型均要求倾角为零；未解析场线缺陷可由 Parkhomchuk 的
       附加有效速度展宽近似表示。
   * - ``temperature_transverse``、``temperature_longitudinal``
     - ``Transverse temperature (eV)``、``Longitudinal temperature (eV)``
     - None
     - 电子平均静止系中正的 :math:`k_BT`，单位 eV，横向按每个笛卡尔分量。
       两个温度同时指定，或改用协方差。
   * - ``velocity_covariance``
     - ``Velocity covariance (m2/s2)``
     - None
     - 电子轴坐标基、电子平均静止系中的对称正定 3×3 条件协方差。
   * - ``velocity_gradient``
     - ``Velocity gradient (1/s)``
     - None
     - 有限 3×2 矩阵 :math:`G`，给出局部平均速度
       :math:`\bar{\mathbf u}_e=G(x_e,y_e)^T`；省略时为零。
       磁化响应只允许纵向一行非零。

温度输入给出 :math:`C=\mathrm{diag}(T_\perp,T_\perp,T_\parallel)e/m_e`。
存在剪切时，:math:`C` 是给定局部位置的条件协方差，不是对整个电子束平均后的协方差。
直流电子固有密度为

.. math::

   n_*=\frac{I}{e\beta_e c\gamma_e}g_\perp(x_e,y_e;s),
   \qquad \int g_\perp\,dx_e\,dy_e=1.

束团模式用 :math:`Q_bh(t)` 替换 :math:`I`，其中 :math:`h` 是归一化高斯到达时间
分布。重复脉冲包含相互重叠。电流和电荷输入均为非负幅值，平滑场使用负电子电荷。

碰撞方程
--------

令 :math:`M` 为完整离子质量、:math:`g=|Z|e^2/(4\pi\epsilon_0)`、
:math:`\mathbf w=\mathbf v-\mathbf u`。下式平均使用归一化局部电子速度分布。
非磁化系数为

.. math::

   \mathbf F_*=-4\pi n_*g^2\ln\Lambda
   \left(\frac1{m_e}+\frac1M\right)
   \left\langle\frac{\mathbf w}{|\mathbf w|^3}\right\rangle,\qquad
   Q_*=4\pi n_*g^2\ln\Lambda
   \left\langle\frac{\mathsf I-\hat{\mathbf w}\hat{\mathbf w}^{T}}
   {|\mathbf w|}\right\rangle.

高斯 Rosenbluth 卷积采用一维数值积分，积分区间围绕热协方差特征值尺度进行几何分区。
:math:`\mathbf F_*` 单位为 N，
:math:`Q_*=d\,\mathrm{cov}(\Delta\mathbf p_*)/dt_*` 单位为
:math:`\mathrm{kg^2\,m^2\,s^{-3}}`。随机增量是

.. math::

   \Delta\mathbf p_*=\mathbf F_*\Delta t_*+
   B_*\sqrt{\Delta t_*}\,\boldsymbol\xi,\qquad
   B_*B_*^T=Q_*,\quad \boldsymbol\xi\sim\mathcal N(0,\mathsf I).

这里没有额外的因子 2。模型保留离子反冲，也不扣除平均踢；能量失配或轴线失配可以
改变离子质心。碰撞更新使用 Itô Euler--Maruyama，弱阶为 1、强阶为 1/2；
仅摩擦的碰撞步进也是一阶。对称输运半映射不会提高碰撞积分本身的阶数。
各向同性热库的平衡验证使用固定库仑对数。自动库仑对数在微观电子速度
积分之外保持局部常数，是有效近似；不能据此声称速度相关库仑对数的精确细致平衡。

Gaussian 自动截断为

.. math::

   u^2=v^2+\mathrm{tr}\,C,\quad
   b_{\min}=\max\left(\frac{g}{\mu u^2},\frac{\hbar}{2\mu u}\right),\quad
   b_{\max}=\min(a,uT_*,u/\omega_p),\quad
   \ln\Lambda=\ln(1+b_{\max}/b_{\min}),

其中 :math:`\mu=m_eM/(m_e+M)`、:math:`\omega_p^2=n_*e^2/(\epsilon_0m_e)`；
:math:`a` 为局部硬半径或两个高斯 RMS 尺寸中较小者。
``Max impact parameter (m)`` 可以进一步限制上界。

Parkhomchuk 使用

.. math::

   \mathbf F_*=-\frac{4n_*g^2}{m_e}
   \frac{\ln\Lambda\,\mathbf v}
   {(v^2+\sigma_\parallel^2+v_{\rm extra}^2)^{3/2}},\qquad
   \ln\Lambda=\ln\left(1+\frac{b_{\max}}{b_{90}+\rho_L}\right),

其中 :math:`u^2=v^2+\sigma_\parallel^2+v_{\rm extra}^2`、
:math:`b_{90}=g/(m_eu^2)`、
:math:`\rho_L=\sqrt2m_e\sigma_\perp/(e|B|)`。
:math:`\sigma_\perp` 是单个笛卡尔分量的 RMS 速度。
该经验摩擦定律不定义扩散张量。输入完整协方差时，Parkhomchuk 将其化约为
:math:`\sigma_\parallel^2=C_{zz}` 和 :math:`\sigma_\perp^2=(C_{xx}+C_{yy})/2`，
忽略交叉相关；只有 Gaussian 模型积分使用完整各向异性协方差。

磁化响应使用 :math:`\Omega=eB/m_e`、:math:`\alpha=2g^2/\pi`，以及

.. math::

   C_k(t)=\exp\left[-\frac12\sigma_\parallel^2k_z^2t^2
     -\sigma_\perp^2k_\perp^2\frac{1-\cos\Omega t}{\Omega^2}\right],
   \qquad R_e=k_z^2t+k_\perp^2\frac{\sin\Omega t}{\Omega}.

初始均匀、相互独立且沿未扰动螺旋线运动的电子给出

.. math::

   Q_* = 2n_*\alpha\int\frac{d^3k}{k^4}\mathbf k\mathbf k^T
     \int_0^{T_*}\left(1-\frac{t}{T_*}\right)
     C_k(t)\cos(\mathbf k\cdot\mathbf v t)\,dt,

   \mathbf F_*=-n_*\alpha\int\frac{d^3k}{k^4}\mathbf k
     \int_0^{T_*}\left(1-\frac{t}{T_*}\right)
     C_k(t)\sin(\mathbf k\cdot\mathbf v t)
     \left(\frac{R_e}{m_e}+\frac{k^2t}{M}\right)\,dt.

Fourier 截断约定为 :math:`k_{\min}=1/b_{\max}`、
:math:`k_{\max}=1/b_{\min}`。反冲项是
:math:`\nabla_{\mathbf v}\cdot Q_*/(2M)`，不是人为指定的 Einstein 噪声律。
零磁场、长窗口极限恢复库仑对数为
:math:`\ln\Lambda=\ln(b_{\max}/b_{\min})` 的 Landau 系数。
底层系数函数支持零磁场验证极限，磁化命令配置则要求正磁场。
积分加密时四个阶数同时翻倍，检查力/张量相对误差、张量正性以及回旋相位分辨率。
这种受控弱响应参考计算的数值开销可能较高。

输运、时钟与平滑场
------------------

每个正长度分段采用半段输运、碰撞/平均场踢、半段输运。零磁场使用漂移，
非零轴向磁场使用均匀螺线管映射；耗散和扩散不使用负 Yoshida 子步。
螺线管内部的机械横向动量为
:math:`p_x^{\rm mech}/P_0=p_x+k_sy/2`、
:math:`p_y^{\rm mech}/P_0=p_y-k_sx/2`，碰撞速度由这些机械动量计算。

坐标 boost 使用相对论机械动量。PASS 的
:math:`dp=(|\mathbf P|-P_0)/P_0` 是总动量偏差，不是纵向动量偏差。
完整离子的 SI 冲量转换回 PASS 的每核子归一化。参考能量不会跟随冷却质心变化。
输运只把 :math:`t_0` 推进一次 :math:`L/(\beta_0c)`，
连续 :math:`z` 随飞行时间差演化。物理节点上的粒子时刻为
:math:`t_i=t_{0,\rm node}-z_i/(\beta_0c)`，束团分组元数据不参与该时钟。

改变数值分辨率时，整段物理作用时间保持不变。平行轴时
:math:`T_*=\gamma_eL(1-\beta_e\beta_0)/(\beta_0c)`。
该截断时间沿理想离子参考世界线计算，而每次踢的实际时间增量沿粒子世界线计算。
截断因此采用接近参考运动的近似；宽分布或明显失配的束流需要检查飞越时间截断的
敏感性。精确动量 boost 并不能消除这项近似。
若在截断公式中用数值分段长度代替整段物理长度，增加分段数就会改变物理模型，
参见 `JSPEC 厚冷却器研究，IPAC 2023，TUPM030
<https://inspirehep.net/files/aab738e313248932b9fad111f6c1bf7e>`_。
物理长度为零时坐标和时钟保持不变。

``Mean space charge=True`` 独立于碰撞地加入准静态、局部均匀、长束近似下的
电子横向平滑场。静止系作用力和 boost 包含实验室系磁场抵消；
平行近等速横向力含 :math:`1-\beta_e\beta_i` 因子。
这不同于静止电子云，也不同于离子自身空间电荷。

束团电子要求 :math:`\gamma_e\beta_ec\sigma_t\geq10a_{\max}`，
其中使用配置的入口/出口半径或 RMS 宽度最大值。
模型不包含端部场、导电管镜像、纵向空间电荷加速或一般三维有限束团场，
也不按空间电荷电势降自动修正电子动能。

诊断、组件状态与 IBS
--------------------

选中圈次的调用追加到每个冷却器独立的 JSONL 文件：
``output/electron_cooler/beam<id>_<name>_<unique>.jsonl``，
同时更新 ``last_diagnostics``。记录包括模型和参考系、作用长度、粒子数/损失、
子步数、密度/重叠、平均力、扩散对角项、能量交换以及模型特有数值检查。
密度、重叠与系数摘要描述最后一个作用节点；能量交换与子步数按整个元件累加。
真实束流所代表的总能量交换包含宏粒子权重。

随机流按束流、命令名和束团分离，粒子按稳定 tag 顺序汇集。
NumPy 随机数驱动两个后端，但浮点轨迹不要求逐位一致。
``state_dict()`` 保存配置身份、熵、调用计数与随机数状态；
``load_state_dict()`` 仅恢复当前冷却器的组件状态。匹配的粒子、参考状态和执行
边界仍需由调用者负责。这些 API 不提供完整模拟重启；跟踪入口从第 0 圈开始新运行。

冷却—IBS 平衡要求使用 IBS 的 ``kinetic`` 或 ``binary`` 实际跟踪；
``bjorken_mtingwa`` 只报告增长率。解释平衡结果前，应验证格架采样、碰撞步长、
宏粒子数及随机统计的收敛。仅 Parkhomchuk 摩擦不能预言电子碰撞的扩散下限。
参见 :doc:`../ibs`。

参考资料
--------

* `Y. Derbenev, Theory of Electron Cooling, arXiv:1703.09735
  <https://arxiv.org/abs/1703.09735>`_：非磁化力/扩散及磁化响应。
  上述有限窗平均与反冲归一化明确了本程序采用的数值变体。
* `A. V. Fedotov et al., Numerical Studies of the Friction Force for the RHIC
  Electron Cooler, PAC 2005, TPAT092
  <https://proceedings.jacow.org/p05/PAPERS/TPAT092.PDF>`_：经验磁化摩擦、
  有效速度与截断约定。
