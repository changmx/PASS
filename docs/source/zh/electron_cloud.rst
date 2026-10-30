电子云（ElectronCloud）
========================================

``ElectronCloud`` 提供三种模式：

* ``frozen`` 对束流施加预设静止均匀电子圆盘产生的横向薄透镜踢角，
  场采用自由空间解析式，或对宏电子采样后通过 PIC 求解一次。
* ``build_up`` 跟踪受预设束流切片及外加磁场驱动的电子，可选自场、多种初级源、多通道材料及不同真空室几何，
  包括壁面初级电子产生、吸收及简化真二次发射；不对束流施加踢角。
* ``coupled`` 加入电子云 PIC 自场和横向束流踢角，
  在接地真空室内使用已保存束流切片的真实粒子分布驱动电子，壁面事件和场边界使用同一几何。

``build_up`` 默认保留外驱动低密度近似；设置 ``Build up.Electron self fields=true`` 可加入云自场。
``coupled`` 始终计算自场。新增物理模型、响应提取、回放和分析见 :ref:`electron-cloud-models-zh`。

``coupled`` 是首个准静态横向薄透镜耦合模型，尚未完成不稳定性阈值或平衡电子云密度的验证。
这三种模式需显式选择。

配置与执行
----------

启用顶层 ``Electron cloud`` 配置块并定义命名配置。
每个 ``ElectronCloud`` 序列命令选择一个配置，同时给出自己的 ``Interaction length (m)``。
该命令不将粒子输运过这一长度，实际输运需另行添加对应命令。
``build_up`` 保留此必填字段以维持 API 兼容，但由于没有束流反作用，
它不改变电子积累动力学。

``frozen`` 不要求 ``Slicer``，因为预设场与纵向位置及时间无关。
命令保持 ``z_rel``、``dp``、``bunch.t0`` 和参考能量不变。
``build_up`` 要求当前 ``z_rel`` SliceSet，并保持全部束流粒子坐标不变。
``coupled`` 同样要求该 SliceSet，且只改变 ``px`` 和 ``py``。
坐标约定见 :doc:`injection`。
给定密度确定静态云或初始化动态云，动态模式也可直接指定初始电子线密度；
两种强度均不从束流强度或束团分组推算。

以下为输入片段，还需另行提供注入与输运：

.. code-block:: json

   {
       "Electron cloud": {
           "Enabled": true,
           "Configurations": {
               "round_cloud": {
                   "Mode": "frozen",
                   "Solver": "uniform_round_free_space",
                   "Electron density (1/m^3)": 1e12,
                   "Radius (m)": 0.01,
                   "Center X (m)": 0.0,
                   "Center Y (m)": 0.0
               }
           }
       },
       "Sequence": {
           "cloud_at_ip": {
               "Command": "ElectronCloud",
               "S (m)": 10.0,
               "Configuration": "round_cloud",
               "Interaction length (m)": 1.0,
               "Save fields": true,
               "Save turns": [[0]]
           }
       }
   }

Python 输入生成使用 ``PASS.para.schema.electron_cloud`` 中的
``ElectronCloudConfig``、``ElectronCloudConfiguration`` 和 ``ElectronCloudItem``，
通过 ``generate_input(..., electron_cloud=cloud_config)`` 传入顶层配置。
配置名称区分大小写；每个命令拥有独立的云状态与资源。
相同源参数与显式随机种子产生相同的初始宏电子样本。
省略种子或使用 JSON ``null`` 表示非确定性初始化；布尔值与非整数不属于合法整数种子。

配置参数
--------

顶层块包含 ``Enabled``（布尔值，默认 ``false``）和 ``Configurations``
（名称到下列配置对象的映射）。关闭顶层开关时跳过配置及全部电子云操作。

.. list-table:: 命名配置
   :header-rows: 1
   :widths: 30 18 52

   * - JSON 字段 / Python 字段
     - 默认值
     - 含义
   * - ``Mode`` / ``mode``
     - ``frozen``
     - ``frozen``、外驱动 ``build_up`` 或 ``coupled``。
   * - ``Solver`` / ``solver``
     - ``uniform_round_free_space``
     - 静态：自由空间解析场或 ``fft_free_space``、``fd_dirichlet``、``dst_dirichlet`` PIC；积累模式要求 ``round_gaussian_beam``，耦合模式要求 ``fd_dirichlet``。
   * - ``Electron density (1/m^3)`` / ``electron_density``
     - 必填
     - 圆盘内非负物理电子数密度；动态非圆盘初态将其作为等效参考圆盘密度。直接指定初始电子线密度时须设为零。
   * - ``Radius (m)`` / ``radius``
     - 必填
     - 均匀圆盘正半径，或动态初态默认总电子数归一化的参考半径；不是高斯 RMS 尺寸。非圆盘分布仍须填写，但不以此确定空间尺寸。
   * - ``Center X (m)``、``Center Y (m)`` / ``center_x``、``center_y``
     - ``0``、``0``
     - 电子云横向中心，单位为米。
   * - ``Number of macro electrons`` / ``n_macroparticles``
     - ``10000``
     - 静态 PIC 或初始动态云的源样本数，须为正整数，不改变物理密度。
   * - ``Random seed`` / ``random_seed``
     - ``null``
     - 云初始化和发射随机数的整数种子或 ``null``。
   * - ``Nx``、``Ny`` / ``nx``、``ny``
     - ``65``、``65``
     - 网格点数，均为至少 3 的整数；动态 PIC 要求至少 5 的奇数。
   * - ``Grid Width X (m)``、``Grid Width Y (m)`` / ``grid_width_x``、``grid_width_y``
     - ``0.1``、``0.1``
     - 以原点为中心的网格完整宽度，须为正值。
   * - ``Particle Deposition Method`` / ``deposition_method``
     - ``CIC``
     - PIC 沉积和插值采用 ``CIC`` 或 ``TSC``。
   * - ``Aperture type``、``Aperture value`` / ``aperture_type``、``aperture_value``
     - ``default``、``[]``
     - 场与壁面共用的几何；动态模式支持 circle、ellipse、rectangle、racetrack、polygon；default/off 保留 Build up.Chamber radius (m) 定义的圆壁面。
   * - ``Build up`` / ``buildup``
     - ``null``
     - 两种动态模式均必填的嵌套 ``ElectronCloudBuildUpConfiguration``；静态模式不允许设置。

网格和沉积参数用于静态 PIC、静态场诊断、耦合 PIC，以及启用自场、规定 Beam driver 或非圆真空室的 build_up。
网格须覆盖并解析完整真空室；旧圆室每轴网格间距不超过半径的一半。
这是最低可分辨性检查，不是充分收敛条件。
显式 Aperture type/value 定义壁面；default/off 使用 ``Build up.Chamber radius (m)``。

.. list-table:: 序列命令
   :header-rows: 1
   :widths: 30 18 52

   * - JSON 字段 / Python 字段
     - 默认值
     - 含义
   * - ``S (m)`` / ``s``
     - ``0``
     - 序列中薄透镜作用位置。
   * - ``Order`` / ``order``
     - ``null``
     - 同一位置的可选整数顺序，遵循共享序列排序规则。
   * - ``Configuration`` / ``configuration``
     - 必填
     - 当前束流 ``Electron cloud.Configurations`` 中的名称。
   * - ``Interaction length (m)`` / ``interaction_length``
     - 必填
     - 静态或耦合束流踢角所代表的非负机器作用长度；积累动力学与该值无关。
   * - ``Slice set`` / ``slice_set``
     - ``null``
     - 两种动态模式均要求当前 ``z_rel`` SliceSet；静态模式无需设置。
   * - ``Is enabled`` / ``is_enabled``
     - ``true``
     - 当前命令的启用开关。
   * - ``Save fields`` / ``save_fields``
     - ``false``
     - 在指定圈保存静态场，或动态粒子状态和事件历史；耦合输出还包含最终云场。
   * - ``Save turns`` / ``save_turns``
     - ``[]``
     - ``[[turn], [start, end, step], ...]``，端点均包含，圈数从零开始；空列表关闭保存。

所有数值配置必须有限。未知电子云字段会报错，因此未实现的动态模型参数不会被静默忽略。

静态云的场与动量约定
--------------------

设电子密度 :math:`n_e\ge0`、元电荷 :math:`e>0`、电荷密度
:math:`\rho_e=-e n_e`、云半径 :math:`a`，以及
:math:`\boldsymbol r=(x-x_c,y-y_c)`。
由无限长均匀带电圆柱的高斯定律得到

.. math::

   \boldsymbol E_e(\boldsymbol r)
   =-\frac{e n_e}{2\epsilon_0}\boldsymbol r
   \begin{cases}
       1, & r\le a,\\
       a^2/r^2, & r>a.
   \end{cases}

电场在中心为零，方向指向中心；解析解同时包含圆盘外径向 :math:`1/r` 的场。
选择 :math:`\phi(a)=0` 作为势参考时，

.. math::

   \phi(r)=\begin{cases}
      \dfrac{e n_e}{4\epsilon_0}(r^2-a^2), & r\le a,\\
      \dfrac{e n_e a^2}{2\epsilon_0}\ln(r/a), & r>a.
   \end{cases}

二维自由空间势具有任意加法常数；该参考选择不影响踢角。

PASS 保存归一化横向机械动量 :math:`p_x=P_x/P_0` 和 :math:`p_y=P_y/P_0`。
在小角度、参考速度薄透镜近似下，命令施加

.. math::

   \Delta p_x=\frac{q_b L_{\mathrm{int}}}{P_0\beta_0 c}E_{e,x},
   \qquad
   \Delta p_y=\frac{q_b L_{\mathrm{int}}}{P_0\beta_0 c}E_{e,y}.

其中 :math:`q_b` 带符号，:math:`P_0` 为完整粒子的 SI 单位参考动量。
内部使用等价因子 :math:`\operatorname{sign}(q_b)L_{\mathrm{int}}/(\beta_0 c B\rho)`，
其中正值 :math:`B\rho=P_0/|q_b|` 包含离子的电荷质量比归一化。
因此近轴电子云聚焦正电荷粒子、散焦电子束粒子。

该公式不含 :math:`1/\gamma_0^2`，因为静止电子云没有可抵消电力的预设纵向电流磁场。
同速自场中的相对论抵消属于 :doc:`space_charge` 的物理模型。
这里不是完整六维电磁积分：纵向电场、云磁场、电场做功，以及逐粒子速度修正均不在此近似内。

PIC 归一化与边界条件
--------------------

PIC 在圆盘面积内均匀采样，各样本具有相等的非负电子数权重。
内部源长度 :math:`L_s=1\,\mathrm{m}`、样本数 :math:`N_m` 对应

.. math::

   w=\frac{n_e\pi a^2 L_s}{N_m},\qquad Q_m=-ew.

共享 PIC 求解器沉积带符号电荷，输出积分密度 :math:`\widetilde\rho`
（C/m\ :sup:`2`）、积分势 :math:`\widetilde\phi`（V m）和积分场
:math:`\widetilde E_x,\widetilde E_y`（V）。``ElectronCloud`` 将其除以
:math:`L_s`，得到 C/m\ :sup:`3`、V 和 V/m；仅最终束流踢角乘以独立的机器作用长度
:math:`L_{\mathrm{int}}`。这两个长度都不是纵向切片宽度。

``fft_free_space`` 采用开放边界场；``fd_dirichlet`` 支持 :doc:`field_solver`
中的导体几何；``dst_dirichlet`` 要求完整网格对齐矩形，采用 ``default``
或完全匹配的矩形 aperture。Dirichlet 求解器使用零壁面势，必须提供有限边界。
其结果通常与自由空间解析圆柱具有真实的边界物理差异。
只有矩形计算区域和沉积电荷完全相同时，才应直接比较 FD 与 DST。

对于静态 PIC，完整源圆盘及沉积支撑域必须位于支持的场区域内；
无效几何会报错，不会裁掉源电荷。
此处 aperture 用于场计算域，不会执行束流粒子丢失；束流损失请配置输运元件的 :doc:`aperture` 属性。
自由空间解析场可作用于源圆盘外的粒子。PIC 要求粒子位于有效插值区域内，
不会用零场替代网格外粒子的场。

非凸多边形的源包含检查使用连续边线，因此能够发现小于网格单元的凹缺口。
对于矩形部分与端帽高度不等的非凸 racetrack，源包含检查使用每个曲线端 128 段的内接多边形。
这一保守检查可能拒绝极接近曲壁的合法源；此时需将源向内移动。

动态模型与物理时间
------------------

选择 ``Mode="build_up"``、``Solver="round_gaussian_beam"`` 并提供
``Build up`` 对象。耦合 PIC 则选择 ``Mode="coupled"``、
``Solver="fd_dirichlet"``，使用相同嵌套对象。
默认情况下，顶层电子密度、圆盘半径与中心描述均匀初始圆盘，
整个圆盘必须严格位于所选真空室内。初始方向在三维空间各向同性，
动能为给定单能值。下述可选位置与速度分布适用于两种动态模式。
初始强度可以为零，之后由初级源产生电子。

.. code-block:: python

   from PASS.para.schema.electron_cloud import (
       ElectronCloudBuildUpConfiguration, ElectronCloudConfiguration,
   )

   model = ElectronCloudConfiguration(
       mode="build_up", solver="round_gaussian_beam",
       electron_density=1e8, radius=0.01, n_macroparticles=512,
       random_seed=20260926,
       buildup=ElectronCloudBuildUpConfiguration(
           chamber_radius=0.02, beam_sigma=0.002, max_time_step=5e-11,
           primary_electrons_per_particle_per_m=1e-6,
           secondary_yield_max=1.5,
       ),
   )

.. list-table:: ``Build up`` 参数
   :header-rows: 1
   :widths: 35 15 50

   * - JSON 字段 / Python 字段
     - 默认值
     - 含义
   * - ``Chamber radius (m)`` / ``chamber_radius``
     - null
     - default/off 孔径所用正圆壁面半径，此时必填；显式非圆几何省略。
   * - ``Beam sigma (m)`` / ``beam_sigma``
     - null
     - 未设 Beam driver 的积累模式必填正圆高斯 sigma；耦合模式可省略且不使用。
   * - ``Max time step (s)`` / ``max_time_step``
     - 必填
     - 正积分步长上限；推进器还可进一步缩短。
   * - ``Magnetic field (T)`` / ``magnetic_field``
     - ``[0, 0, 0]``
     - 束流局部坐标系中外加磁场的均匀部分 ``[B0x, B0y, B0z]``。
   * - ``Magnetic gradient (T/m)`` / ``magnetic_gradient``
     - ``0``
     - 带符号有限正规四极磁场梯度；拒绝布尔值。
   * - ``Initial electron energy (eV)`` / ``initial_energy_ev``
     - ``0``
     - 默认单能速度分布下，每电子的非负初始动能；均匀或高斯速度分布要求设为零。
   * - ``Initial distribution`` / ``initial_distribution``
     - ``uniform_disk``
     - ``uniform_disk``、``uniform_chamber``、``uniform_rectangle`` 或 ``gaussian``。位置与速度分布独立选择。
   * - ``Initial position spread (m)`` / ``initial_position_spread``
     - ``null``
     - uniform_rectangle 必填正 ``[x, y]`` 半宽；gaussian 必填截断前标准差；圆盘或全腔分布不填写。
   * - ``Initial electron line density (1/m)`` / ``initial_line_density``
     - ``null``
     - 可选非负真实电子总线密度，指腔室截断后每米的总电子数。要求顶层 Electron density = 0；null 保留参考圆盘归一化。
   * - ``Initial velocity distribution`` / ``initial_velocity_distribution``
     - ``monoenergetic``
     - ``monoenergetic`` 为各向同性固定动能，``uniform`` 为独立均匀速度分量，``gaussian`` 为独立高斯速度分量。
   * - ``Initial velocity spread (m/s)`` / ``initial_velocity_spread``
     - ``null``
     - uniform 必填非负 ``[vx, vy, vz]`` 半宽，gaussian 必填截断前标准差；单能分布不填写。零宽度分量保持为零。
   * - ``Primary electrons per beam particle (1/m)`` / ``primary_electrons_per_particle_per_m``
     - ``0``
     - 每真实束流粒子、每米的非负预设初级电子产额。
   * - ``Primary macro electrons`` / ``primary_macroparticles``
     - ``64``
     - 每个非零初级发射事件产生的宏电子样本数，须为正整数。
   * - ``Secondary yield max`` / ``secondary_yield_max``
     - ``0``
     - 未经能量截断的真二次产额曲线峰值，非负；零表示吸收壁。
   * - ``Secondary peak energy (eV)`` / ``secondary_peak_energy_ev``
     - ``300``
     - 未截断曲线峰值处的入射动能，须为正值。
   * - ``Secondary shape`` / ``secondary_shape``
     - ``1.35``
     - 严格大于一的曲线形状参数。
   * - ``Emission energy (eV)`` / ``emission_energy_ev``
     - ``2``
     - 正的初级发射能量和名义二次发射能量。
   * - ``Max macro electrons`` / ``max_macroparticles``
     - ``100000``
     - 正整数粒子数上限；超过时报告错误，不丢弃电荷。
   * - ``Max steps`` / ``max_steps``
     - ``100000``
     - 每个物理时间区间内允许的积分步数，须为正整数。
   * - ``Max wall hits per step`` / ``max_wall_hits_per_step``
     - ``32``
     - 每粒子在单个积分步内允许的壁面事件数，须为正整数。

初态分布与归一化
~~~~~~~~~~~~~~~~

``uniform_chamber`` 在所选完整腔室内按面积均匀采样，要求顶层中心坐标为零。
``uniform_rectangle`` 在以 ``Center X/Y`` 为中心的矩形与腔室的交集中均匀采样。
``gaussian`` 先定义以该中心为均值、两轴独立的高斯分布，再取腔室内的条件分布。
拒绝采样保留指定宏电子数和总权重，不丢弃尾部电荷；因此截断后高斯分布的
实际均值和 RMS 可以不同于输入中心和宽度。只有 ``uniform_disk`` 使用 Radius 定义空间尺寸。
采样次数有上限：交集面积为零或达到采样上限仍未获得足够粒子时会报错，
不会无限循环或丢弃粒子后继续运行。

设顶层参考半径为 :math:`a`，电子密度为 :math:`n_e`。
未指定 Initial electron line density 时，四种分布均保留
:math:`N_e/L_s=n_e\pi a^2`。对于非圆盘分布，:math:`n_e` 是等效参考圆盘密度，
不代表局部密度或峰值密度。直接指定线密度后，总电子数与 :math:`a` 无关，
且须设置 :math:`n_e=0`，避免两套强度输入冲突。初始源长度为
:math:`L_s=1\,\mathrm m`，每个宏电子的权重均为
:math:`N_e/N_{\mathrm{macro}}`。总强度为零时创建空状态。

均匀速度的各分量独立、零均值，范围为 :math:`[-w_i,w_i]`，因此
:math:`\langle v_i^2\rangle=w_i^2/3`，其最大合成速度必须严格小于 :math:`c`。
高斯速度的各分量以零为均值、输入值为标准差，取满足
:math:`|\boldsymbol v|<c` 的条件分布；它是规定的速度分布，不是相对论热平衡分布。
两者均按 :math:`\boldsymbol u=\gamma\boldsymbol v/c` 精确转换为归一化机械动量，
初始能量账本使用实际样本的相对论动能。不进行样本均值扣除或 RMS 重缩放。
将某个宽度设为零即可保持对应速度为零，例如纵向速度为零。
高斯速度中亚光速概率过低、在有限采样预算内无法填满指定粒子数时，同样会报错。

椭圆高斯云配合独立横向高斯速度的示例：

.. code-block:: python

   model = ElectronCloudConfiguration(
       mode="build_up", solver="round_gaussian_beam",
       electron_density=0, radius=0.01, n_macroparticles=10000,
       center_x=0.001, random_seed=42,
       buildup=ElectronCloudBuildUpConfiguration(
           chamber_radius=0.02, beam_sigma=0.002, max_time_step=5e-11,
           initial_distribution="gaussian",
           initial_position_spread=[0.003, 0.002],
           initial_line_density=1e6,
           initial_velocity_distribution="gaussian",
           initial_velocity_spread=[2e5, 1e5, 0.0],
       ),
   )

CPU 与 GPU 初始化均使用相同的主机 PCG64 采样，再一次性复制到所选后端。
旧圆盘/单能默认配置保持原有随机序列；这些选项不增加推进循环中的工作。

切片时序
~~~~~~~~

在同一圈、同一位置执行指定 ``Slicer``，再调用云命令。
只支持连续 ``z_rel`` 区间，要求覆盖全部存活粒子且含粒子切片宽度为正；
零宽度空切片会被跳过。
命令消费已保存的切片归属和区间，不隐式重算或缩放。
区间 :math:`[z_{\min},z_{\max}]` 对应的物理通过时间和线电荷为

.. math::

   t_{\mathrm{start}}=t_0-\frac{z_{\max}}{\beta_0c},\qquad
   t_{\mathrm{end}}=t_0-\frac{z_{\min}}{\beta_0c},\qquad
   \lambda_b=\frac{Z_b e N_{\mathrm{slice}}}{\Delta z}.

驱动器按这些时间合并排序全部束团区间；区间可以端点相接，但不能重叠。
重复圈和物理时间倒退会被拒绝。空切片仍以零束流场推进电子，
间隙内继续计算运动、外加磁场及壁面事件。
端点舍入容限不超过八个绝对时间 ULP 与较短切片时长的 :math:`10^{-12}` 倍中的较小值。
若绝对端点相减得到的时长相对保存值 :math:`\Delta z/(\beta_0c)` 的误差超过
:math:`10^{-6}`，则拒绝执行，避免很大的绝对时钟悄悄吞掉短束流脉冲。
第一个区间起点或观测点建立初始云时钟，允许负时间，不凭空补算未定义的过去。
每次调用在最后一个保存切片的后沿或更晚的观测点结束，不自动推进至环周期末端；
下次调用首先推进中间间隙。没有存活源粒子的槽保留显式空区间；使用 ``auto`` 时，
仅在 ``bunch.t0`` 提供零电荷观测点，不把空切片的极小回退宽度当作物理脉冲。
这也适用于已分配记录全部损失或尚待注入的情况：这些记录及其统计仍保留，
但它们不能定义自动束流脉冲宽度。已经被真实区间覆盖的观测点不会再次推进时钟。
含已丢失或预留粒子记录的槽仍按通常流程切片；此规则只改变其零电荷电子云时间投影。

参考时间由输运命令推进，不由执行器的圈计数推进，因此多圈输入需要真实闭合输运路径。
必须使用 ``bunch.t0`` 与保存区间构造时间，不能用名义 ``z_center`` 或
``harmonic_id`` 替代物理到达时间。

预设驱动与电子推进
~~~~~~~~~~~~~~~~~~

在未设 Beam driver 的旧圆室 ``build_up`` 的每个切片内，横向源是位于轴心、在壁面半径 :math:`R` 截断的圆高斯，
sigma 为给定 :math:`\sigma_b`。当前切片粒子数确定带符号线电荷；
跟踪束流的横向坐标不会改变源中心和尺寸。
在 :math:`0<r<R` 内，

.. math::

   \boldsymbol E_b(\boldsymbol r)=
   \frac{\lambda_b}{2\pi\epsilon_0}
   \frac{1-\exp[-r^2/(2\sigma_b^2)]}
        {1-\exp[-R^2/(2\sigma_b^2)]}
   \frac{\boldsymbol r}{r^2},\qquad
   \boldsymbol B_b=\frac{\beta_0}{c}(-E_y,E_x,0).

中心使用连续线性极限。分母保证 ``lambda_b`` 是真空室内包含的总线电荷。
由于轴对称，接地圆壁只改变静电势参考，不改变径向电场。
束流磁场与配置的外加磁场共同作用于电子。

两种动态模式均支持均匀外场与理想正规四极场叠加，在电子横向中点求值：

.. math::

   \boldsymbol B_{\mathrm{ext}}(x,y)
   = (B_{0x}+G y,\ B_{0y}+G x,\ B_{0z}),
   \qquad G=\texttt{magnetic\_gradient}.

此外场属于局部电子云作用点，须显式配置，不从格架 ``Quadrupole`` 命令自动推断。
纯二极场使用 ``magnetic_gradient=0`` 和横向均匀磁场。
圆形真空室内的保守场强上界为 :math:`|\boldsymbol B_0|+|G|R`。
模型沿纵向均匀，可以出现横向磁镜俘获，但不包含有限磁铁长度、
端部边缘场或纵向电子损失。

电子采用两个位置坐标和三个无量纲动量分量
:math:`\boldsymbol u=\boldsymbol P/(m_ec)=\gamma_e\boldsymbol v/c`，
其中 :math:`\gamma_e=\sqrt{1+|\boldsymbol u|^2}`。
相对论 Boris 动量推进与对称半漂移组合；每个半漂移求首次圆壁交点，
在交点吸收或发射后继续推进剩余时间。不模拟不同云作用点之间的纵向电子输运。
Boris 方法参见
`WarpX 粒子推进器文档 <https://warpx.readthedocs.io/en/24.01/theory/pic.html#boris-relativistic-velocity-rotation>`_。

时间步长上限还受到保守的束流振荡频率、回旋频率和横向位移估计限制。
这些限制不能替代分辨率研究：撞壁附近的力分裂误差和 Boris 相位误差仍依赖步长。
宏粒子数与发射统计还需独立检查收敛。
标准相对论 Boris 对相对论 :math:`\boldsymbol E\times\boldsymbol B` 漂移也有已知限制，见
`Higuera 与 Cary <https://arxiv.org/pdf/1701.05605>`_。

耦合 PIC 场与束流响应
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``coupled`` 使用保存切片的存活粒子横向坐标及归属。
每个束流宏粒子沉积线电荷 :math:`q_{\ell,j}=r_b Z_b e/\Delta z`，
其中 :math:`r_b` 为真实粒子与宏粒子的数量比。
在保存的物理切片时间区间内，横向源分布保持固定，不再用高斯轮廓替代；
``beam_sigma`` 对本模式的物理计算没有影响。

接地 FD 求解器分别求束流场和云场。动态云粒子先沉积
:math:`q_{\ell,e,j}=-e w_j/L_s`（C/m），其中源长度 :math:`L_s` 初始为 1 m。
除以网格面积后得到 C/m\ :sup:`3` 的密度，Poisson 直接给出 V 的电势与 V/m 的电场，
不再额外除以源长度。恢复的快照可以包含其他源长度；电子权重随该长度同比缩放时，物理场不变。
每个电子步先执行含壁面事件的半漂移，在中点重建云场，
使用 :math:`\boldsymbol E_b+\boldsymbol E_e` 和
:math:`\boldsymbol B_{\mathrm{ext}}+\boldsymbol B_b` 执行 Boris 更新，
再执行第二次半漂移。无束流间隙内仍推进云自场。
不包含云磁场或纵向电场。

对位于切片固定横向坐标的见证粒子，以中点求积累积时长 :math:`\Delta t_s` 内的云场：

.. math::

   \overline{\boldsymbol E}_{e,j}
     =\frac{1}{\Delta t_s}\sum_k
       \boldsymbol E_e(\boldsymbol r_j,t_{k+1/2})\Delta t_k,\qquad
   \Delta\boldsymbol p_{\perp,j}
     =\frac{\operatorname{sign}(Z_b)L_{\mathrm{int}}}
            {\beta_0 c B\rho}\overline{\boldsymbol E}_{e,j}.

束流踢角只使用云场，不含束流自踢角或 :math:`1/\gamma_0^2` 折减。
此操作只改变 ``px``、``py``；``x``、``y``、``z_rel``、``dp``、
参考能量和 ``bunch.t0`` 均不变，切片内不推进束流源坐标。
作用长度缩放束流响应，不缩放电子数或电子物理时钟。

靠近圆壁时，沉积和插值将每个模板在有效内节点上重新归一化，保持沉积电荷。
此处理在壁面附近具有一阶误差，并非精确 Hamiltonian 或严格能量守恒的粒子—场离散。
步长受 ``max_time_step``、外场与束流场的回旋频率估计、
云等离子体及网格加速度频率估计（:math:`\omega\Delta t\le0.2`），
以及电子位移（:math:`v_\perp\Delta t\le0.2h`，:math:`h` 为较小网格间距）限制。
这些控制不能证明收敛。得出物理结论前，应分别检查网格、宏电子与初级发射样本数、
随机种子、切片宽度和时间步长。

初级源与二次发射
~~~~~~~~~~~~~~~~

每个切片前沿产生预设初级电子数
:math:`N_{\mathrm{primary}}=Y_p N_{\mathrm{slice}}L_s`，
使用沿壁面周向均匀采样、朝内发射的等权宏电子表示。
其中源长度 :math:`L_s` 初始为 1 m，:math:`Y_p` 单位为 1/m。
这一经验源不计算残余气体电离、同步光子输运或材料光电发射能谱。
初级产生集中在切片前沿，没有在整个切片时长内连续分布。
仅缩短电子积分步长无法消除这一源时间近似，还应检查 Slicer 宽度收敛。

设入射动能为 :math:`E_i`，:math:`x=E_i/E_{\max}` 且 :math:`s>1`。
未经截断的真二次曲线与实现的限制为

.. math::

   \delta(E_i)=\delta_{\max}\frac{s x}{s-1+x^s},\qquad
   \delta_{\mathrm{eff}}=\min\!\left(\delta(E_i),\frac{E_i}{E_{\mathrm{emit}}}\right),
   \qquad E_o=\min(E_{\mathrm{emit}},E_i).

每个入射宏电子至多产生一个宏二次电子，其权重为
:math:`w_o=w_i\delta_{\mathrm{eff}}`，每电子能量为 :math:`E_o`；
权重为零表示吸收。这样同时限制单电子能量与加权总能量不超过入射能量，
并在低能端保守地抑制发射。
曲线使用
`Furman 与 Pivi 式 31–32 <https://www.classe.cornell.edu/~critten/cesrta/ecloud/doc/furmanpivi.pdf>`_
的真二次形状，并非完整概率模型：这一旧选项不含弹性反射、再扩散、角度相关材料产额及联合二次能谱；这些能力由可选的 Surface emission 提供。
宏粒子通过权重表示倍增，不采用整数分支。

初级和二次发射方向均服从相对于朝内法向的三维余弦分布：
:math:`\mu=\cos\theta=\sqrt U`、:math:`\phi=2\pi V`，
其中 :math:`U,V` 为独立均匀随机数。
尽管只跟踪 x、y 位置，三个动量分量均保留；默认单能初态使用各向同性完整球面，
可选初态速度分布独立于壁面发射模型。

输出与状态
----------

静态模式选定圈的 HDF5 诊断写入当前 PASS 运行目录下的
``electron_cloud/<run_id>/beam<id>_<command>/turn_<turn>_call_<call>.h5``。
运行标识包含新生成的唯一标识；调用编号区分同一圈的重复执行。
命令的 ``saved_fields`` 列出已写入路径。
所需场输出写入成功后才提交暂存的束流动量踢角。写入失败时束流动量保持不变，
重试会使用新的诊断文件名。文件格式标记为
``PASS-electron-cloud-fields-1``，包含：

.. list-table:: 场快照数据集
   :header-rows: 1
   :widths: 45 20 35

   * - 数据集
     - 单位
     - 含义
   * - ``grid/x``、``grid/y``
     - m
     - 一维网格坐标轴。
   * - ``fields/charge_density``
     - C/m\ :sup:`3`
     - 带符号的物理体电荷密度。
   * - ``fields/electron_density``
     - 1/m\ :sup:`3`
     - 物理电子数密度。
   * - ``fields/potential``
     - V
     - 静电势。
   * - ``fields/ex``、``fields/ey``
     - V/m
     - 横向电场。
   * - ``source/x``、``source/y``、``source/weight``
     - m、m、1
     - PIC 源坐标与电子数权重；解析云不含这些数据集。

二维网格数组按 ``(y, x)`` 排列，各数据集属性注明单位。
文件 ``metadata_json`` 记录命令、束流、圈数、位置、作用长度、配置和踢角诊断；
PIC 源元数据含所代表的源长度与随机数状态。
执行后的 ``last_diagnostics`` 提供存活粒子数、最大电场与踢角以及逐束团记录。
解析势采用上文给出的势参考。解释结果时，应区分 PIC 采样噪声、导体壁面像场与力归一化误差。

两种动态模式采用相同的输出目录与选圈规则，保存
``source/x,y,ux,uy,uz,weight`` 数组和源元数据；默认稀薄近似的 ``build_up`` 不保存云自场图。
耦合输出还包含最终真实云场的 ``grid`` 和 ``fields`` 数据集，
采用上表 SI 单位，并保存历史与最大踢角诊断。
``ux,uy,uz`` 为无量纲动量。源元数据包含 ``source_length``、物理 ``time``、
``last_turn``、随机数状态及累计粒子数/能量计数器。
``history_json`` 逐间隙或切片记录时间、束流粒子数、带符号线电荷、初级注入、
作用前后电子数、入射/发射电子数和步数。每条记录满足

.. math::

   N_{\mathrm{after}}=N_{\mathrm{before}}+N_{\mathrm{primary}}
                      -N_{\mathrm{incident}}+N_{\mathrm{emitted}}.

累计壁面能量为全部代表电子的入射动能减发射动能，单位 eV，乘 :math:`e` 可转换为焦耳。
计数和壁面能量对应保存的源长度；若代表机器长度 :math:`L`，可乘 :math:`L/L_s` 换算。
这不是自洽的机器热负荷预测。配置的作用长度不缩放底层电子集合。

累计能量收支保存 ``initial_energy_ev``、``primary_energy_ev`` 和带符号
``field_work_ev`` （Boris 力更新时的动能变化）。以 eV 为单位计算
:math:`K=\sum w(\gamma_e-1)m_ec^2`，满足
:math:`K_{\mathrm{final}}=K_{\mathrm{initial}}+K_{\mathrm{primary}}+W_{\mathrm{field}}-E_{\mathrm{wall}}`。
该关系验证数值收支，不度量积分器的物理轨迹误差。
在耦合模式下，它也不证明准静态薄透镜近似中的束流、电子和场总能量守恒。
整个动态命令暂存电子状态、随机数状态及束流踢角，
全部演化和请求的输出成功后才一并提交；演化或输出失败时均保持不变。

命令提供 ``state_dict()`` / ``load_state_dict()`` 和
``save_state(path)`` / ``load_state(path)``，保存自身源、配置标识和随机数状态，
从而可重复使用非确定性 PIC 样本，或带时间和动量的动态电子状态。
这些 API 是云快照，不是完整模拟重启：匹配的束流粒子、束流参考状态、圈数和其他集体效应状态
仍需分别处理。静态模式没有演化时钟，两种动态模式均恢复保存的物理时钟。
耦合快照沿用动态状态格式，校验模式与配置标识，并重建场资源而不保存缓存。

跟踪期间，动态电子数组驻留于所选 CPU 或 GPU 后端。
暂存状态拥有独立粒子数据；耦合模式通过串行调用复用固定网格、场分解和数值工作区。
读取公开云状态、写入源/场快照或捕获组件快照时才构造经过校验的主机快照；
GPU 上这会增加设备到主机的数据传输。输入字段与组件快照格式保持不变。
标量诊断和发射采样仍可能引发主机与设备同步。
比较 CPU/GPU 性能时，应先完成编译和求解器初始化预热，
分别测量同步后的跟踪、初始化与快照输出耗时，并覆盖无碰壁和壁面发射负载；
少量电子使用 GPU 不一定更快。

耦合 GPU 执行结合固定次序沉积与确定性 cuDSS 求解。
高度聚集的单元使用固定归约树，因此与此前串行求和相比可能出现末位差异。
复现检查限定于相同代码、GPU 架构、SM 数量及软件栈，
不承诺跨设备一致，见 :doc:`field_solver`。

读取云快照只改变当前命令的源。跟踪入口从第 0 圈开始新运行，
这些组件 API 不提供整机重启流程。

可运行示例
----------

``example/09_electron_cloud_models`` 提供英文入门说明和两个串行工作流。
``run_models.py`` 覆盖五种真空室、自场开关、椭圆偏轴与密度图驱动、
四种初级源、三通道表面发射、宏粒子管理以及时间步长比较。
五种几何主例及椭圆半步长例均禁用宏粒子管理；另外两个椭圆例以同一个
无合并半步长结果为基线，分别比较粗、细相空间单元的合并影响。
全部八条历史均保存；报告将积分步长变化与合并近似分开，并给出有效宏粒子数、
合并次数和是否达到软目标。权重与动能守恒本身不能约束合并导致的后续轨道、
能谱及壁面热负载误差。
``run_response.py`` 从真实 coupled PIC 提取 G 和四个 K 分量，
并以独立直接跟踪检验响应重建。两者均支持 CPU/GPU，
在 ``tests/codex/ecloud_examples`` 新建目录保存 HDF5/JSON；
模型示例还生成 CSV 和 PNG。假设、单位与验证范围见 :ref:`electron-cloud-models-zh`。

.. _electron-cloud-models-zh:

电子云模型与响应工作流
----------------------

以下选项扩展 :doc:`electron_cloud`，旧输入仍使用稀薄云、圆高斯驱动和简化二次发射默认值。
电子运动仍为 2D3V：推进 x/y 并保留三个机械动量分量，不包含纵向电子输运。

规定驱动、自场与真空室
~~~~~~~~~~~~~~~~~~~~~~

``Build up.Electron self fields`` 为布尔值，默认 ``false``。在 ``build_up`` 中打开后，
受驱动区间和无束流间隙都计算电子空间电荷场；``coupled`` 始终包含自场。
求场使用接地横向静电网格，应分别检查网格和时间步长收敛。

``Build up.Beam driver`` 可选 ``ElectronCloudBeamDriver``：

.. list-table:: 规定横向驱动
   :header-rows: 1
   :widths: 35 65

   * - 参数
     - 含义
   * - ``Model``
     - ``gaussian`` 或 ``density_map``。
   * - ``Sigma X (m)``、``Sigma Y (m)``
     - 高斯 RMS 尺寸，均为正数，可以不相等。
   * - ``Center X (m)``、``Center Y (m)``
     - 规定束流中心，独立于初始电子云中心。
   * - ``X (m)``、``Y (m)``、``Relative density``
     - 严格递增坐标轴和 (y, x) 排列的非负相对密度图。

驱动按每个已保存切片的带符号物理电荷归一化。``coupled`` 使用实际束流分布，不允许另设该驱动。
为兼容旧输入，``round_gaussian_beam`` 仍是 build_up 模式的求解器标签，
即使可选驱动实际使用有限区域 PIC 求场。

动态 ``Aperture type/value`` 支持 circle、ellipse、rectangle、racetrack 和简单 polygon，
尺寸约定与 :doc:`space_charge` 一致。碰壁交点、内法向、源位置与接地 Poisson 区域共用该几何。
``default``/``off`` 保留 ``Chamber radius (m)`` 定义的圆壁面。
整个初始均匀圆盘必须严格位于真空室内，其他初始空间分布取腔室内部的条件分布。
非圆几何自动使用有限区域束流场；未设置 ``Beam driver`` 时，
将 ``Beam sigma (m)`` 定义的轴心圆高斯密度投影至该区域。
显式非圆孔径须省略 ``Chamber radius (m)``，避免冲突的几何定义。

初级电子产生
~~~~~~~~~~~~

``Build up.Primary sources`` 是 ``PrimarySourceConfiguration`` 列表，
不能与非零旧初级产额同时使用。每个源可设 ``Macro electrons`` （默认 64）、
``Mean energy (eV)`` （默认 2）、``Energy distribution`` （monoenergetic 或 gamma）
及 ``Energy shape`` （正 Gamma 形状参数，默认 2）。宏粒子权重表示真实电子数。
出生时间在有限源区间内采样，仅推进出生后的运动。

.. list-table:: 源长度 L、持续时间 dt 下的物理电子数
   :header-rows: 1
   :widths: 25 40 35

   * - ``Kind``
     - 物理电子数
     - 参数
   * - ``gas_ionization``
     - N_b P sigma L / (k_B T)
     - ``Gas pressure (Pa)``、``Gas temperature (K)``、``Ionization cross section (m^2)``；位置跟随束流体分布。
   * - ``prescribed_wall``
     - N_b Y L
     - ``Electrons per beam particle (1/m)``。
   * - ``beam_loss``
     - N_b f_loss Y_loss L
     - ``Beam loss fraction (1/m)``、``Electrons per lost particle``；使用给定损失率，不自动从跟踪丢粒子推断。
   * - ``photoelectric``
     - photon_rate QE L dt
     - ``Photon rate (1/m/s)``、``Quantum efficiency``；输入规定照射，不追踪光子路径。

壁面源的 ``Wall angle min (rad)`` 和 ``Wall angle max (rad)`` 限定发射区域。
这里是相对于真空室原点的极角范围；在该区域内按壁面弧长均匀采样，椭圆上并非极角均匀。
光照源在无束流间隙也持续产生电子；N_b 为当前区间通过的真实束流粒子数。
气压、电离截面、损失率与照射量须来自目标运行工况。

壁面发射
~~~~~~~~

``Build up.Surface emission`` 选择 ``SurfaceEmissionConfiguration``，
不能与非零旧 ``Secondary yield max`` 同时使用。这是三通道经验模型，不是完整的 Furman-Pivi 材料拟合。

* 真二次通道参数为 ``True yield max``、``True peak energy (eV)``、``True shape``、
  ``Yield angle coefficient`` 和 ``Peak angle coefficient``。
* ``True energy distribution`` 选择 gamma 或 monoenergetic；
  ``True mean energy (eV)`` 与 ``True energy shape`` 定义未加能量条件的谱。
  有限入射能量会改变加条件后的实际平均出射能量。
* ``Elastic probability at zero energy`` 和 ``Elastic energy scale (eV)`` 定义弹性支路；
  ``Rediffused probability`` 与 ``Rediffused energy shape`` 定义再扩散支路。
* ``Sampling`` 为 stochastic_event 或 weighted_channels；
  ``Max secondaries per impact`` 限制单次事件多重性。

``True yield energy table (eV)`` 和 ``True yield table`` 可替代解析真次级产额曲线，
使用线性插值；``Yield table outside`` 指定区间外为 zero（默认）或 constant。
角度系数仍作用于能轴与产额。

各通道共享入射能量预算。随机真二次支路先采样有条件总能量，再采样联合能量分配；
不把独立谱样本事后裁剪到入射能量，避免产生非物理尖峰。余弦发射保留正负两种纵向动量。
单能随机发射的 Poisson 多重性以 N 乘发射能量不超过入射能量为条件，
因此低能区实际平均产额可能不同于未加此条件的输入曲线。
实际材料、表面处理历史及能区仍需实验标定。

宏粒子管理与诊断
~~~~~~~~~~~~~~~~

``Build up.Population control`` 选择 ``PopulationControlConfiguration``。
局域相空间单元内使用两个代表动量合并，保持总权重、加权动量与相对论动能；
过重代表粒子可以拆分。参数为 ``Target macro electrons``、``Spatial cell size (m)``、
``Normalized momentum cell size``、``Max weight to mean ratio`` 和 ``Max split children``。
目标数量为软目标：不会为凑数量任意合并相隔较远的单元。
独立的 ``Max macro electrons`` 仍是硬资源上限。

守恒低阶矩不等于保持所有空间结构和高阶矩。应与更多宏粒子、更细单元下的密度及电场比较。
有效宏粒子数为 :math:`N_\mathrm{eff}=(\sum w)^2/\sum w^2`。
同状态粒子的拆分会增大这个权重指标，但不会创造统计独立的相空间样本。
``PASS.commands.ecloud.diagnostics`` 提供带权粒子数、线密度、动能、核心密度及 RMS 统计。
HDF5 ``diagnostics`` 谱各自包含带单位的 ``edges``、``centers`` 和 ``values``。
云谱以 electrons/bin 描述快照时刻；壁面谱另提供净动能沉积的 J/bin 权重，按 ``scope`` 标识的记录范围累计。
壁面位置轴为 atan2(y,x) 极角而非弧长。壁面能谱末 bin 包含全部不低于 1 MeV 的事件，
有限的 10 MeV 绘图边界不是能量上限。GUI 保留这些属性，支持读取、绘制和导出。

壁面净沉积能量为入射动能减去出射动能。``wall_power`` 再除以观测持续时间与源长度得到 W/m。
可选周长用于计算平均 W/m2，不表示已解析局部热点。

CPU/GPU 使用相同物理分布与能量约束。GPU 材料采样在设备上使用 Philox，
每次发射调用从组件已保存的 PCG64 状态取得一个种子。
同一后端与软件栈上的克隆、恢复可以重现抽样，但不要求 CPU/GPU 的具体样本相同。
随机样本可复现并不意味着 GPU 墙谱逐位一致：双精度原子加归约在权重不同时可能产生末位舍入差异，
但不会反馈到粒子轨迹或 RNG。
仅条件 CDF 下溢或极端 Dirichlet 参数下的少量样本进入 host 退路。
初级源保留 host 随机发生器，但从 GPU 仅提取被抽中的束流坐标；
宏粒子管理和壁面谱归约留在 GPU，只传回小规模诊断结果。

Sequence 命令只在启用 ``Save fields`` 时累计墙谱；始终启用时为 ``scope=cumulative_since_state_origin``。
无论是否保存输出，标量粒子数与能量收支始终保留，避免跟踪为未使用的谱付出代价。
低层 ``BuildUpCloud`` API 默认保留墙谱；显式传入 ``record_wall_spectra=False``
可关闭，恢复状态时也丢弃旧谱，不改变物理粒子与 RNG。

若快照中已经发生过碰壁但未记录墙谱，恢复时重新启用记录会开始新的累计范围，
标记为 ``scope=cumulative_since_recording_restart``。后续累计及再次保存、恢复均保留此范围。
标量粒子数与能量账本仍包含完整物理历史，因此不能将重新记录的墙谱当作全历史壁面能量。

响应提取与回放
~~~~~~~~~~~~~~

``PASS.commands.ecloud.response`` 提供 ``ResponseSlice``、``extract_response``、
``write_response``、``read_response`` 和 ``make_periodic_response``。
提取器从克隆的物理云及 RNG 状态驱动真实 ``BuildUpCloud`` 求解器，
测量空间背景踢角 G 和四个源到见证束片横向响应分量 K。
只扰动源质心，见证探针保持不变。epsilon 与 epsilon/2 的中心差分须在指定容差内一致，
并相对基准检查偶对称残差，捕获仅用中心差分可能漏掉的二次响应。
有限宽度切片在自身持续时间内可以已有响应，因此不强制因果矩阵对角元为零。

.. math::

   \Delta \mathbf p_{\perp,i}(x,y)
   = \mathbf G_i(x,y) + \sum_{j\le i}\mathbf K_{ij}\Delta\mathbf r_j.

p_perp 为横向机械动量除以参考动量；G 无量纲，K 单位为 1/m。
束流电荷、作用长度和带符号见证束参考因子已包含在文件中。
G 已给出背景聚焦，不应再另加相同冻结云踢角。
该近似描述已标定见证束分布的相干响应，不覆盖任意变化的束流分布及非线性运动。

``ElectronCloudResponseItem`` 可加入 Sequence：

.. list-table:: 响应作用点
   :header-rows: 1
   :widths: 40 20 40

   * - JSON 键
     - 默认
     - 含义
   * - ``Response file``、``Slice set``
     - 必填
     - 标定 HDF5 和在本位置刚执行的 general/z_rel 切片。
   * - ``Timeline mode``
     - finite
     - 完整已测因果历史，或经过验证的 periodic 复用。
   * - ``Maximum centroid displacement (m)``
     - null
     - 使用文件中的标定范围，或设定更严格范围。
   * - ``Profile tolerance``
     - null
     - 中心化横向协方差的相对 Frobenius 容差；使用文件值或更严格值。
   * - ``S (m)``、``Order``、``Is enabled``
     - 0、null、true
     - 标准位置、执行顺序和开关。

有限时间线可以包含多个束团与多个圈，不要求平稳核。
周期回放要求记录多个周期，并验证 G、各滞后 K 及被截断记忆尾部的一致性。
局部云状态不是整个模拟检查点；每次跟踪仍从新初始条件开始，响应文件不会恢复其他命令或时钟。

``capture_response_slices`` 可捕获实际保存切片的时间线。自动空束团可能产生零宽观测点，
这种点不是有限响应区间；含空组时应使用有限 explicit Slicer 范围，捕获器会拒绝零宽观测点。
协方差检查不能检测所有高阶矩或细结构变化；束流超出固定分布假设时须重新标定，
即使协方差恰好保持不变也不例外。

数值与物理验证
~~~~~~~~~~~~~~

应在指定小位移范围内用直接 coupled 跟踪检查响应重建。
分别细化电子时间步、网格、宏粒子数、响应探针网格和记忆长度。
工作点与指数增长率需要明确拟合区间，阈值结论取决于实际扫描参数和增长模型。
数值一致性不能替代材料与机器实测输入，也不能证明完整三维、电磁或边缘场有效性。
