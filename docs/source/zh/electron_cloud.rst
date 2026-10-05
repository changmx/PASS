电子云（ElectronCloud）
========================================

``ElectronCloud`` 提供三种模式：

* ``frozen`` 对束流施加预设静止均匀电子圆盘产生的横向薄透镜踢角，
  场采用自由空间解析式，或对宏电子采样后通过 PIC 求解一次。
* ``build_up`` 在圆形真空室中跟踪受预设束流切片及外加磁场驱动的电子，
  包括壁面初级电子产生、吸收及简化真二次发射；不对束流施加踢角。
* ``coupled`` 加入电子云 PIC 自场和横向束流踢角，
  在接地圆形真空室内使用已保存束流切片的真实粒子分布驱动电子。

``build_up`` 属于外驱动低密度近似，不包含电子云自场、束流反作用或物理空间电荷饱和机制。
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
给定密度确定静态云或初始化动态云，不从束流强度或束团分组推算。

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
     - 圆盘内非负物理电子数密度；动态模式仅用其初始化云。
   * - ``Radius (m)`` / ``radius``
     - 必填
     - 均匀圆盘的正半径，动态模式为初始云半径；不是束流 RMS 尺寸。
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
     - 网格点数，均为至少 3 的整数；耦合模式要求至少 5 的奇数。
   * - ``Grid Width X (m)``、``Grid Width Y (m)`` / ``grid_width_x``、``grid_width_y``
     - ``0.1``、``0.1``
     - 以原点为中心的网格完整宽度，须为正值。
   * - ``Particle Deposition Method`` / ``deposition_method``
     - ``CIC``
     - PIC 沉积和插值采用 ``CIC`` 或 ``TSC``。
   * - ``Aperture type``、``Aperture value`` / ``aperture_type``、``aperture_value``
     - ``default``、``[]``
     - 静态场计算区域的几何；动态模式要求 ``default`` 或 ``off`` 且参数为空，其圆形壁面由 ``Build up`` 定义。
   * - ``Build up`` / ``buildup``
     - ``null``
     - 两种动态模式均必填的嵌套 ``ElectronCloudBuildUpConfiguration``；静态模式不允许设置。

网格和沉积参数用于静态 PIC、静态场诊断和耦合 PIC；``build_up`` 不使用 PIC 网格。
耦合模式要求两个网格宽度均覆盖完整真空室直径，每轴网格间距不超过真空室半径的一半。
这是最低可分辨性检查，不是充分收敛条件。
圆形 Dirichlet 壁面由 ``Build up.Chamber radius (m)`` 定义。

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
顶层电子密度、圆盘半径与中心描述初始电子云，
整个初始圆盘必须严格位于圆形真空室内。
初始方向在三维空间各向同性，动能为给定单能值。
初始密度可以为零，之后由初级源产生电子。

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
     - 必填
     - 以束流轴为中心的正圆形壁面半径。
   * - ``Beam sigma (m)`` / ``beam_sigma``
     - 必填
     - 积累模式的正固定横向高斯 sigma；耦合模式为保持 schema 兼容仍要求填写，但其场与步长控制不使用此值。
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
     - 每电子的非负初始动能。
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
第一个区间起点建立初始云时钟，允许负时间，不凭空补算未定义的过去。
每次调用在最后一个保存切片的后沿结束，不自动推进至环周期末端；
下次调用首先推进中间间隙。显式声明的空束团贡献其保存的空区间，
缺失桶位不会被隐式补造。

参考时间由输运命令推进，不由执行器的圈计数推进，因此多圈输入需要真实闭合输运路径。
必须使用 ``bunch.t0`` 与保存区间构造时间，不能用名义 ``z_center`` 或
``harmonic_id`` 替代物理到达时间。

预设驱动与电子推进
~~~~~~~~~~~~~~~~~~

在 ``build_up`` 的每个切片内，横向源是位于轴心、在壁面半径 :math:`R` 截断的圆高斯，
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

接地圆形 FD 求解器分别求束流场和云场。动态云粒子先沉积
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
的真二次形状，并非完整概率模型：未实现弹性反射、再扩散、角度相关材料产额及联合二次能谱。
宏粒子通过权重表示倍增，不采用整数分支。

初级和二次发射方向均服从相对于朝内法向的三维余弦分布：
:math:`\mu=\cos\theta=\sqrt U`、:math:`\phi=2\pi V`，
其中 :math:`U,V` 为独立均匀随机数。
尽管只跟踪 x、y 位置，三个动量分量均保留；初始电子云方向则使用各向同性完整球面。

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
``source/x,y,ux,uy,uz,weight`` 数组和源元数据；``build_up`` 不保存云自场图。
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
读取公开云状态、写入源/场快照或捕获检查点时才构造经过校验的主机快照；
GPU 上这会增加设备到主机的数据传输。输入字段与检查点格式保持不变。
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

``example/07_electron_cloud`` 提供英文入门说明和六个串行单踢角算例：零密度、
解析密度与双倍密度、自由空间 PIC、矩形 FD 和矩形 DST。
显式探针粒子覆盖云内外区域。分析从作用前后分布快照读取踢角，与高斯定律比较，
检查密度正比关系、坐标和参考时间不变，并在相同边界条件下比较 FD/DST。
JSON、CSV 和科学绘图 PNG 输出位于 Git 忽略的
``tests/codex/electron_cloud/example_output`` 目录。

``example/08_electron_cloud_buildup`` 提供四个串行束团列算例：初始云吸收、
初级产生与吸收、初级产生与真二次发射，以及时间步长减半比较。
真实单圈 Drift 推进参考时钟，同时保持预设驱动束流分布不变。
分析检查粒子数/能量收支、时间连续性和束流坐标不变，
在 ``tests/codex/electron_cloud/buildup_example_output`` 保存 JSON、CSV 和电子数历史图。
时间步减半的差异作为分辨率比较报告，不表示物理饱和或统计收敛已被证明。
