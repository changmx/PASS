WakeField
=========

``WakeField`` 使用指定的 ``Slicer`` 结果施加积分薄元件尾场。一个命令代表
一个物理位置，各算法组分别保存源历史或模式状态。本页说明 CPU 与 CUDA 实现，
数值验证结果见仓库的尾场验证报告。

物理范围与约定
--------------

PASS 追踪解析或用户提供的响应，不求解任意三维结构的 Maxwell 方程。
响应速度耦合使用束团参考速度；横向电压到冲量的换算则使用粒子实际入射速度。
粒子连续坐标为 :math:`z_i=\beta_b c(T_b-t_i)`，当前位置的物理到达时间为

.. math::

   t_i=T_b-z_i/(\beta_b c).

不保存到达修正量。RF 改变参考能量时缩放 z 以保持此时间；重分组将 z 和动量
变换到目标参考系。尾场适配器转换用户最近提供的 z 区间：中心时间为
:math:`T_b-z_{slice}/(\beta_b c)`，时间宽度为 :math:`\Delta z/(\beta_b c)`。
RF 不改变保存的区间或成员。新发出的源永久保留当次采样的物理时间和宽度，
后续参考变换不重新解释因果历史。参见 :ref:`zh-longitudinal-reference`。

.. math::

   F(f)=\int W(t)e^{-2\pi i f t}\,dt,\qquad Z_\parallel=F,\qquad Z_\perp=iF.

源和测试横向单项式总阶数为 :math:`n` 时，积分尾函数单位为 V/C/m\ :sup:`n`，
阻抗单位为 ohm/m\ :sup:`n`。源矩使用带符号真实电荷，测试宏粒子权重约去。
纵向为正表示损失能量，横向为正表示正方向 Lorentz 力。每核子能量变化
:math:`\Delta E=-Z_{\mathrm{ion}}V_\parallel/A`，随后精确换算动量；横向
:math:`\Delta p_x=(Z_{\mathrm{ion}}/A)V_x/(\beta_i p_0)` 使用实际入射速度。
因果核零点取有限跳跃的一半。``uniform`` 对整个切片时间宽度均匀积分，
``point`` 位于切片中心。切片和参数收敛由用户扫描，没有自动切片误差控制。

响应模型
--------

.. list-table:: Model 中的 Kind
   :header-rows: 1
   :widths: 23 77

   * - Kind
     - 参数与含义
   * - ``constant``
     - ``Amplitude``、``Duration (s)``；有限因果测试响应。
   * - ``resonator``
     - 正值 ``R``、``Q``、``Frequency (Hz)``。R 为谐振阻抗实部；振幅衰减率 :math:`\pi f_r/Q`。支持欠阻尼、临界阻尼及过阻尼。
   * - ``tabulated``
     - 递增 ``Times (s)``、对应 ``Values``、``Causal`` （默认 true）。线性插值，表外为零。因果表从零开始，双侧表可含负延迟。
   * - ``file``
     - 数值 table 或 headtail 文件，显式声明列、单位、符号与归一化；只在初始化读取，保存内容指纹，详见文件输入。
   * - ``impedance``
     - 递增非负 ``Frequencies (Hz)``、对应 ``Real``/``Imag``、显式 ``Reconstruction``；保留不可变原始采样。
   * - ``fitted_impedance``
     - 频谱及 ``Initial poles (1/s)``（[实部, 虚部]）、``Optimize poles``、``Max evaluations``、``Relative floor``、``Fit tolerance``。共轭对只输入正虚部极点，实极点单独输入。
   * - ``modes``
     - ``Poles (1/s)``、``Residues`` （[实部, 虚部]），须含完整共轭对；实极点对应实留数。
   * - ``resistive_wall``
     - 有限 β 圆管良导体厚壁：``Radius (m)``、``Conductivity (S/m)``、``Length (m)``、``Beta``、``Frequencies (Hz)``。可选 ``Wall thickness (m)`` 仅检查厚壁条件，不启用有限壁厚场匹配求解器。
   * - ``ultrarelativistic_wall``
     - Bane-Sands 圆管 DC 壁：``Radius (m)``、``Conductivity (S/m)``、``Length (m)``。要求 beta >= 0.99，另须检查短束团适用性。

有限 β 壁模型采用 Stupakov，PRAB **23**, 094401 (2020)，
`doi:10.1103/PhysRevAccelBeams.23.094401 <https://doi.org/10.1103/PhysRevAccelBeams.23.094401>`_。
支持纵向与对角偶极分量，减去相同几何的理想导体响应，仅提供有限电导率修正。
缩放 Bessel 函数避免溢出并保留有限 β 依赖。DC 不在适用范围内。
``Max skin depth ratio`` 限制趋肤深度与半径（及给定壁厚）之比，
``Max surface impedance ratio`` 限制归一化表面阻抗，默认及最大值均为 0.1。
有限壁厚、磁性或色散材料及任意几何须提供相应已验证响应，不能从该模型推断。

``Field content`` 声明 ``wake``、``finite_conductivity_correction``、``pec_image``、
``direct_space_charge`` 或 ``total``。程序不自动减去空间电荷。现有 SpaceCharge
为横向计算，不能据此认为纵向或所有像场项已包含；组合前须检查几何、归一化及
物理内容，防止重复或遗漏。

频谱与拟合
----------

原始路径对分段线性频谱做振荡积分，保留非均匀频率采样，不引入 FFT 的人工
周期时间窗口。频带外明确为零，频带与间距需要独立收敛检查。有限带宽逆变换
一般为双侧函数。``two_sided`` 保留该结果；``causal_projection`` 明确截去
负时间部分，零点取跳跃一半，会改变有效频率响应，是带宽近似。

拟合采用变量投影：非线性优化用户选定的极点，以实数最小二乘确定留数，
不自动选模式数。保留原始频谱及诊断；最大相对误差须满足 ``Fit tolerance``，
分母下限为 ``Relative floor``。极点半平面受到约束，但不自动强制或认证无源性。
纵向标量无源条件不能套用于所有横向分量；缺失频带不推断为 delta 或其导数项。
右半平面极点只表示向负时间衰减的空间分支，不作为增长时间状态向前递推。
``causal_projection`` 拟合要求左半平面初始极点；一般双侧拟合选择 ``two_sided``
及明确空间边界。CST 工程/二进制解析暂缓；数值导出可使用下述显式文件约定。

速度、加速与算法组
------------------

每个分量的 ``Velocity`` 显式选择：

* ``fixed`` 加 ``Beta``：同一参考速度的稳态响应，追踪中参考速度变化会被拒绝。
* ``factorized`` 加递增 ``Betas``、实数 ``Source`` 和 ``Witness`` 表：
  :math:`W(t;\beta_s,\beta_w)=g_s(\beta_s)W_0(t)g_w(\beta_w)`，只在表内线性插值。
  历史激励保存源通过时的速度；当前观察应用测试束团因子。
* ``ideal``：显式定义速度无关点响应，不是对实际腔体渡越时间因子的推导。

有限 β 壁自动设置相同固定速度。单个稳态谱不能确定任意不同速度轨迹、复数
渡越相位或任意加速。只有已提供且适用的实因子分解可用于加速；极点频率和阻尼
保持不变，不施加通用 β 缩放。

顶层字段为 ``S (m)``、``Command="WakeField"``、``Slice set``、``Groups``、
``Is enabled``。组名 ``Name`` 必须唯一，``Components`` 不为空。分量包含
``Component``、``Model``、``Velocity``，可选 ``Scale`` 与 ``Field content``。

.. list-table:: 算法组控制
   :header-rows: 1
   :widths: 25 75

   * - 字段
     - 含义
   * - ``Solver``
     - 必填 ``direct``、``fft``、``recursive`` （谐振子）、``modal`` （极点/留数）、``partitioned_fft`` （固定带间隙束列）或 ``time_fft`` （变周期物理时间历史）。
   * - ``History``
     - 必填 ``none``、``direct``（源切片历史）、``state`` 或 ``partitioned``；recursive/modal 必须使用 state，partitioned_fft/time_fft 必须使用 partitioned。
   * - ``Source shape``
     - 各组单独选择 uniform（默认）或 point。
   * - ``Memory turns``
     - direct 历史为正整数或 null；partitioned_fft 必须为有限正整数。计数前序通过次数，当前通过另计。time_fft 必须为 null。
   * - ``Memory time (s)``
     - 正值或 null，用于 direct/partitioned 历史；time_fft 必须为有限正值。截断延迟核，包含均匀源部分积分；物理时间投影在不连续截断处存在网格误差。
   * - ``Convolution grid``
     - partitioned_fft 必填，定义固定的物理时间网格，见下文。
   * - ``Time grid``
     - 仅 time_fft 必填：正有限 ``Step (s)``，整数 ``Block size`` >= 2（默认 64），可选有限 ``Origin (s)`` （默认 null）。
   * - ``Partition``
     - 两种分块 FFT 均选择 dyadic（默认）或 uniform，不自动切换算法。
   * - ``Max workspace (MiB)``
     - 分块 FFT 保守峰值内存预算，默认 1024 MiB；分配前检查，CUDA 另检查可用显存。
   * - ``Boundary``
     - causal_passages（默认）、isolated 或 periodic。
   * - ``Period (s)``、``Periodic images``
     - periodic 必填，使用 direct 求和 -N 至 +N 镜像，不保存瞬态历史。

FFT 要求递增均匀当前网格，源宽度相同且不大于间距；History=direct 时历史
明确直接求和。没有 auto 或静默回退。模式按实际时间推进，保留无限衰减历史，
状态内存随模式数增长。因果批次须按物理时间排序且不重叠（包括均匀源支撑）；
跨圈重叠会报错，按圈追踪不能推断未来轨迹。isolated 声明完整源列已在当前
批次，periodic 声明其周期稳态；都要求固定 β 且无瞬态历史，不能表示任意
加速或瞬态环形分布。用户选择周期镜像数并检查收敛。

示例与恢复
----------

.. code-block:: python

   from PASS.para.schema import WakeField
   wake = WakeField(s=0, slice_set="wake", groups=[
       dict(name="short", solver="fft", history="none", source_shape="uniform",
            components=[dict(component="longitudinal", velocity=dict(kind="ideal"),
                model=dict(kind="resonator", r=1e3, q=1, frequency=1e8))]),
       dict(name="long", solver="recursive", history="state", source_shape="point",
            components=[dict(component="dipolar_y",
                velocity=dict(kind="factorized", betas=[0.01, 0.5, 1.0],
                              source=[0.2, 1.0, 0.8], witness=[0.1, 0.3, 1.0]),
                model=dict(kind="resonator", r=1e6, q=1e6, frequency=1e7))]),
   ])

示例耦合仅为说明，不是腔体数据；之前须执行相应 Slicer。
旧平铺 Components/Solver/Memory turns 接口必须迁移至 Groups，不解释过时字段。
命令提供 ``last_sources``、``last_coefficients``、``last_diagnostics``、
``group_states``，诊断包含算法、边界、保留次数、状态字节数及拟合误差。
``state_dict()``/``load_state_dict()`` 保存恢复各组并校验配置指纹。
这些方法仅处理尾场状态；``reset_state()`` 将该位置场状态归零。

``Executor.run(sim, sequences)`` 始终从第 0 圈开始。新运行开始时，启用的
WakeField 命令不能保留之前的历史；若在重新初始化的模拟中复用 WakeField
命令，应先调用 ``reset_state()``。

宏粒子权重由初始注入参数固定。Executor 不在 Injection 前后复制 tag 或更新
源电荷；新激活粒子保持原始权重，见 :doc:`injection`。

CPU 与 GPU 执行
---------------

CPU 与 GPU 实现放在同一文件中，CUDA 辅助函数和核函数源码位于对应 CPU 代码
下方。``PASS/commands/wake/`` 中，时域模型位于 ``wake_models.py``，频谱及
有理响应位于 ``wake_spectrum.py``，显式求解器和模式扫描位于
``wake_solvers.py``，两类历史卷积分别位于 ``convolution.py`` 和
``time_convolution.py``。投影归入 ``wake_moments.py``，时钟归入
``wake_timing.py``，源数据存储归入 ``wake_state.py``，耦合归入
``wake_components.py`` 和 ``wake_velocity.py``。融合粒子踢动位于
``PASS/commands/wake_field.py`` 的 command 实现下方。CuPy 导入和 CUDA 编译
仍按需执行，CPU 运行不依赖 CUDA 环境。

CPU 与 CUDA 使用同一模型和算法组配置，通过模拟的后端设置选择；
``execute_gpu`` 要求粒子数组已在显存。原始频谱、表格响应、谐振子、极点模式、
直接历史及空间周期求和均在 GPU 上计算。有理拟合和解析有限 β 阻抗构造属于
CPU 初始化工作，得到的响应数组按设备上传一次。数值表格/HEADTAIL 转换及分块
卷积核采样也在 CPU 初始化阶段完成。

CUDA 计算使用融合的 ``RawKernel``/``RawModule``：时间换算、源矩、响应函数及积分、
直接求和、速度耦合、事件累积、频谱乘积、插值、校验和粒子踢动均在核内完成。
CuPy 保留为显存管理及启动接口；FFT 和排序／去重继续使用专用 GPU 库。
不要求用基础核替代这些专用库。

所有有存活粒子的束团共用一次投影启动及一次踢动启动，支持不同参考时间、速度及
保存的切片宽度。空粒子群通过批量校验排除，不逐束团同步。粒子范围或 SliceSet
存储变化时重建布局和指针缓存，参考参数单独更新；各次源记录保留自己的数组。
固定范围的等长 Slicer 也批量执行各束团的直方图和切片表计算，存活数与越界数
统一传回主机。主机尾场源快照合并上传各数组，显存中的各行保留本次通过的数据，
不会被后续上传覆盖。

直接求解器由一个 warp 负责一个见证切片，在核内计算响应并归约源贡献，不分配
源切片乘见证切片大小的矩阵。物理时间近场修正对所有响应模型采用同一方式。
普通 FFT 按校验后的间距、宽度、模型配置及截断时间缓存响应频谱；参考 beta
改变几何时缓存失效。复用频谱时仍在设备上校验均匀网格。

粒子坐标支持 float32 与 float64；物理到达时间、切片电荷矩、响应、模式状态、
能量与动量换算均使用 float64。GPU 原子归约的求和顺序不同，因此 CPU/GPU 应按
误差容差比较，而不要求逐位相同。float32 存储仍可能改变边界附近的切片归属，
并在长圈数输运中积累舍入差异。

投影通过一次粒子遍历和块内直方图累积电荷及横向偶极矩。单个 CUDA 核函数
合并各分量的测试单项式与机械踢动。模式历史采用有界仿射前缀扫描，指数因子
保持衰减，支持临界及过阻尼谐振子；圈内事件使用相对通过原点的时间，减少
长时间后窄切片边界的舍入损失。
对于中心递增、各切片边界未到达相邻中心的布局，直接构造有序事件，支持相邻
切片的部分重叠及重合边界；其他布局保留通用排序／去重。模式更新只分配一次
下一状态向量，保留旧状态供回滚和检查点使用。物理时间卷积在事务前统一校验
完整沉积帧，逐子块处理时不重复同步主机；单槽位空间变换使用一维 cuFFT。
源几何、响应模型、历史截断和时间网格近似均保持原有定义。

等长局部 z 区间按当前参考速度转换为均匀时间网格。响应缓存依赖实际网格几何，beta 改变时其时间宽度也改变。用户控制 Slicer 更新；检查点序列化将显存历史转换为可移植主机数组。

小规模算例可能由 CPU 更快完成，原因是 CUDA 启动及同步开销。性能测量应先
预热、同步 CUDA 事件，并计入 Slicer、时钟、投影和踢动。生成报告
``tests/codex/wake_redesign/validation_report.md`` 记录实际硬件、规模、误差、
计时及复现命令。

自定义空间项
------------

保留现有命名分量。``Component="custom"`` 必须提供 ``Spatial``，其中
``Plane`` 为 x/y/z，``Source powers`` 为 [a,b]，``Test powers`` 为 [c,d]，
均为非负整数。投影源矩为 :math:`\sum_j q_j x_j^a y_j^b`，响应乘以测试粒子的
:math:`x_i^c y_i^d`；尾函数单位的空间阶数为 a+b+c+d。这是输入多项式响应的
表示方式，不推导未提供的多极分量，也不自动保证不同分量之间的 Maxwell 约束。
解析圆壁模型仍只支持明确规定的纵向与对角偶极分量。

CPU/CUDA 支持相同幂次。GPU 批量直方图在同一次粒子遍历中累积 Q/Qx/Qy
及所需的通用单项式；测试幂次在融合机械踢动核中按整数幂求值。

固定网格多束团与长历史
----------------------

选择 ``Solver="partitioned_fft"``、``History="partitioned"`` 和有限
``Memory turns=H``，任意因果表格均不要求模式拟合。采样中心时间定义为

.. math::

   t_{n,b,i}=t_{\rm origin}+nT+bP+i\Delta t,
   \qquad K_{\ell,d,q}=\overline W(\ell T+dP+q\Delta t).

网格字段为 ``Period (s)`` T、``Slots`` B、``Slices`` S、``Slot spacing (s)`` P、
``Slice spacing (s)`` delta-t、``Origin (s)`` （默认 0）、``Width (s)`` （默认 0）、
``Projection`` （默认 exact）。周期与间距为正；宽度不超过切片间距，槽位窗口
不能重叠或跨越一个周期。uniform 源要求正宽度，point 源要求零宽度。

槽位与槽内切片分别补零至至少 2B-1 和 2S-1，形成保留真实间隙的线性卷积，
即使 P/delta-t 不是整数也成立。空槽置零且保留其物理时间位置。源切片允许
重排、重复；重叠的独立粒子群以守恒方式累加。程序根据实际到达时间映射，
不按 bunch 枚举顺序推断。由于到达时间使用 T_b-z/(beta_b*c)，harmonic ID
递增不一定对应到达时间递增。

``exact`` 要求每次通过的中心、宽度与网格一致，只容许浮点舍入。
``linear`` 允许窗口内离网格的 **点源**，源矩线性分配到两个时间节点，场再
插值回测试位置。这引入时间网格近似，尖锐尾场尤其需要细化收敛检查。
历史算法本身不拟合尾部、不降采样、不粗化时间。周回周期变化、超出窗口或
宽度不匹配均报错；变周期因果历史可选择时间网格已收敛的 time_fft，或直接求和。
不会静默重置历史或切换算法。

uniform 分块每次通过调度全部 H 个延迟频谱；dyadic 分块覆盖 [L,2L)，
L=1,2,4,...，最后截到 H。已完成的 L 圈源块只向未来通过时刻贡献，不需要
未来束流轨迹。只有一个延迟的分块使用频谱延迟线。设补零后的空间频率单元数
为 F，在源通道与分量数量固定时，uniform 历史每圈 O(H F)，dyadic 摊销
O(F log-squared H)，另加空间 FFT 与分量因子；两者频谱存储均为 O(H F)。
分层调度在块边界存在耗时峰值，性能测量应至少覆盖一个最大块周期，并报告
平均值和最大值，不能只看中位数。

每个束团按自己的参考事件转换保存的区间。GPU 投影累积当前源矩，不保存粒子时钟数组；融合粒子踢随后使用相应响应系数。

对于 [-zmax,zmax] 等长 Slicer、共同 beta、谐波槽位 0..B-1、周长 C，且
t0=n*C/(beta*c)，精确点源网格示例如下：

.. code-block:: python

   from PASS.para.schema import WakeSolverGroup, WakeConvolutionGrid
   speed = beta * 299792458.0
   grid = WakeConvolutionGrid(
       period=C/speed, slots=B, slices=S, slot_spacing=C/(B*speed),
       slice_spacing=2*zmax/(S*speed),
       origin=-((B-1)*C/B + zmax-zmax/S)/speed, width=0)
   group = WakeSolverGroup(
       name="tabular_tail", solver="partitioned_fft", history="partitioned",
       source_shape="point", memory_turns=512, convolution_grid=grid,
       partition="dyadic", max_workspace_mb=2048, components=components)

选择正常 GPU 后端即在 CUDA 执行同一配置。粒子存储可为 float32，但声明的
物理网格和尾场算术保持双精度；边界附近的切片归属仍可能受 float32 影响。
生成测试、性能脚本与完整跟踪示例位于 ``tests/codex/wake_fast_history/``，
分别测量源沉积与历史调度、以及同步后的完整追踪。

变周期下的一般历史计算
----------------------

``Solver="time_fft"`` 使用固定物理时间节点
:math:`t_k=t_{\rm origin}+k\Delta t`，不要求固定周回周期。
选择 ``History="partitioned"``、有限 ``Memory time (s)`` 和 ``Time grid``，
不设置 ``Memory turns`` 与 ``Convolution grid``。每次通过的真实到达时间来自
物理时钟。β、束团间距、切片宽度和周回周期变化时，不重缩放或丢弃已有历史。
前文的响应模型及速度耦合近似仍适用；此算法不推导一般双时间电磁响应。

点源矩线性沉积到相邻时间节点；有限均匀源对同一帽函数基底在源区间内积分，
保持每个耦合源矩守恒。场线性插值到测试时间。在距离源中心不超过半个源宽
加两个网格步长的范围内，用精确源平均核替换该对源—目标的投影贡献。这一
局部修正保留因果跳跃和半自作用，并包含上一批次的临近源。远处的时间投影
仍是近似；点源与测试点恰在网格上时，在浮点误差内恢复对应离散直接求和。

短尺度响应和高频分量需要细化 ``Step (s)``；物理切片数、历史截断时间须分别
检查收敛。表格折点和硬截断可限制实际收敛阶，选择 time_fft 本身不保证统一
误差阈值。步长还必须能被绝对到达时间的浮点表示分辨。

已完成的物理时间块送入 uniform 或 dyadic 在线卷积；尚未完成的块，包括源
沉积超出中心的部分，跨通过保留。preview 只计算当前已知源，不要求未来束流
轨迹。预览中触及的持久环形缓冲行会保存并精确还原，所有组完成踢动校验后
才提交更新。CPU/GPU 检查点包含未完成块、近场修正源、时钟原点及已完成块
历史。空批次仍推进通过计数；间隙超过完整历史时长及投影支撑后，可精确跳过
已无响应的空时间块。

块长为 B、记忆时间为 H 秒时，历史块数 M=ceil(H/(B*delta-t))+1。
源通道和分量数固定时，dyadic 每个完成块摊销计算量为
O(B log B + B log-squared M)，另加源沉积、测试插值及局部源—目标修正；
存储为 O(B M)。一次通过可能跨越多个块。细时间网格也表示束团间的空隙，
因此固定稀疏束列使用 partitioned_fft 可能明显更省。大量宽源重叠会增加
局部修正工作量。``Block size`` 控制调度及内存访问，不改变物理分辨率。
性能测试应覆盖完整最大块周期并报告峰值耗时。

CPU 和 CUDA 实现同一投影及历史方案。CUDA 使用设备端沉积、融合的投影对
计算、共享源 FFT、持久 cuFFT 计划和显存历史；仍会同步少量校验元数据。
Slicer 保持逐束团调用，变周期支持不要求整列 Slicer 接口。

.. code-block:: python

   from PASS.para.schema import WakeSolverGroup, WakeTimeGrid
   group = WakeSolverGroup(
       name="accelerating_tail", solver="time_fft", history="partitioned",
       source_shape="uniform", memory_time=200e-6,
       time_grid=WakeTimeGrid(step=0.5e-9, block_size=1024),
       partition="dyadic", max_workspace_mb=2048, components=components)

示例网格值仅作说明，须对实际尾场检查收敛。原点为 null 时，首个源区间会
选取邻近原点；显式原点不得晚于首个源的支撑起点。不同调用的源批次仍须按
因果时间排序且支撑不重叠；同一批次内的独立重叠粒子群允许累加。
可执行验证和同步计时位于 ``tests/codex/wake_variable_period/``。

漂移束适配范围
--------------

漂移束可使用单束团组 ``harmonic_number=1``，配合覆盖整圈的等长度显式
Slicer 表示全环电荷/电流及横向源矩分布，并不要求 RF 成束。均匀电流 I 对
有限因果响应的稳态电压为 :math:`V=I\int_0^\infty W(\tau)d\tau`。
验证套件在 CPU/CUDA 上检查这一 DC 结果及方位谐波与解析卷积的收敛。
零初始历史会产生启动瞬态，应等待响应记忆填满。``Boundary="periodic"``
表示预先给定的重复稳态分布，需要镜像数收敛，不等同于演化中的瞬态历史。

演化漂移束使用 ``Coordinate=arrival_phase``（或 ``Periodic=true``）、
``equal_length`` 及 ``Explicit={"z min": -C, "z max": 0}``。显式更新时
:math:`z_{phase}=C[(u\bmod1)-1]`，其中 :math:`u=v_{obs}(T_{obs}-t_i)/C`。
规定时钟选择共同观测事件和速度，见 :doc:`slicer`。各束团的参考时间、速度可以
不同，但所有粒子群必须共享保存的观测窗口与周长，Slicer 应位于尾场位置。

切片覆盖 :math:`[T_{obs},T_{obs}+C/v_{obs})`。每个新的物理源通过事件需要用户
更新 Slicer；复用周期快照保留旧窗口，不自动重切片。演化历史使用
``Boundary="causal_passages"``。``time_fft`` 支持变化且互不重叠的通过窗口，
固定卷积网格仍要求其声明的时间结构。

这里采用 **每粒子每参考圈经过一次的近似**。在该模型内支持动量展宽及长期
累计滑移，但不调度同一参考圈内个别粒子的零次或多次通过。实际周期为 Ti
的均匀刚性流，在模型中表示的电流为 Q/T，实际为 Q/Ti；令
epsilon=1-Ti/T，其相对电流误差为 abs(epsilon)。真实 Drift 验证检查这一
一阶误差及其随动量偏差减小而收敛；全环因果电压验证在 CPU/CUDA 上检查
电荷守恒和 :math:`I\int W`。

要求单圈滑移、单圈集体作用变化足够小，并对需要解析的方位模 m 满足
:math:`2\pi |m\,\Delta u|\ll1`。切片数与时间网格应分别做收敛检查。
``Max phase slip`` 默认 0.05、最大 0.1，在 WakeField 使用切片时拒绝
过大的已观测单步变化；Slicer 仍可生成诊断投影。
不是通用精度保证。长期累计运动后，粒子存储精度仍须能够解析切片宽度，
长期漂移束推荐 float64。未启用周期到达切片的普通 explicit 模式仍保持
原有的边界截取行为。

回归与分块接口
--------------

``PartitionedConvolution.preview_block`` 接收有限的 float64 后端数组，
形状为 (源通道, 槽位, 切片)。源速度因子已包含，见证速度因子只在物理
时间采样后施加。返回的更新独立持有频谱，调用者后续修改输入数组不会
影响它。预计算结束时持久历史不变，更新只可提交一次。多块预计算通过
有界环形行事务在异常时精确回滚，同一状态禁止重叠事务，从而保证各组
历史统一接受，且不必每次复制完整长历史。

已纳入版本管理的入口为
``python -m tests.codex.wake_production.run_regression``。CPU CI 在 PR 和
推送时执行；CUDA 主机使用同一入口加 ``--require-gpu``，没有真实设备时
直接失败。生成结果与环境仍被 Git 忽略。长期验证入口
``python -m tests.codex.wake_production.long_term --backend cpu``，也可选
gpu，以变周期非模态表格验证 2048 圈自洽运动、增长率、tune 和强度阈值。
独立参考为有限历史 Floquet 矩阵，不是另一个 PASS 求解器。测试使用归一化
单位及外加的小线性阻尼映射，得到的是该基准的阈值，不是普遍机器极限。

CUDA 将保存的切片区间直接换算为物理时间。所有支持的模型均采用每个见证
切片一个 warp 计算近场精确修正，避免源—目标对列表及其计数的主机同步。性能测试包括逐束团 Slicer、参考参数变化、源投影、
历史计算与 kick，并预热且覆盖最大历史块的完整调度周期。

文件输入
--------

``Model.Kind="file"`` 接受 ``File path``、``Format`` （默认 table 或 headtail）、
``Axis column`` （默认 0）、必填 ``Value column``、可选 ``Imag column``、
``Delimiter`` （null 为空白分隔）、``Skip rows`` （默认 0）、``Causal`` （默认 true）、
``Reconstruction`` （默认 two_sided）、可选 ``Length (m)`` 和必填 ``Convention``。
列号从零起算，不能重复，必须为数值。UTF-8 文件支持 # 注释与 BOM。
一般表格分段线性插值、支撑外为零，因果表必须包含零延迟；拒绝乱序和重复
采样，反向时间/距离轴可以重排。

``Convention`` 声明 ``Data kind`` （wake_function/impedance）、``Axis``
（time/distance/frequency）、``Axis unit``、``Value unit``、``Positive trailing``、
``Longitudinal positive loss``、``Integrated``、``Reference beta``。
``Fourier exponent`` 默认 -1，``Transverse impedance factor`` 默认 i（可为 -i/1），
``Shunt impedance convention`` 默认 not_applicable，仅记录来源：数值表已归一化，
此字段不重新缩放分路阻抗。

时间单位 s/ms/us/ns/ps，距离 m/cm/mm，频率 Hz/kHz/MHz/GHz。
尾场幅值显式使用 V/kV/MV 除以 C/nC/pC 及所需空间幂次，例如 V/C/m^2 或
V/(pC*mm)。阻抗单位为 ohm/Ohm/kOhm/MOhm 附加空间幂次。单位长度数据多一个
分母长度幂次，必须给定 ``Length (m)``；已经积分的数据不允许再次乘长度。
距离轴通过 reference beta*c 换算成延迟，尾函数幅值不额外乘雅可比。
显式换算 Fourier 与纵向符号。阻抗输入必须有实部、虚部列和正延迟为尾随的
约定，变换符号差异通过 Fourier 字段声明。有限带宽因果投影仍有上述近似；
有限束长 wake potential 需要单独反卷积，不能作为点电荷尾函数直接读取。

HEADTAIL 有多种列布局，须明确列号；其单位约定为 ns，以及零阶 V/pC、一阶
V/(pC*mm)，符号仍需声明。参见 `CERN HEADTAIL 表格规范
<https://indico.cern.ch/event/178920/contributions/1446485/attachments/235706/329825/HDTL_lattice_def.pdf>`_。
PASS 自行读取与换算数值，不依赖 Xwakes 或 PyHEADTAIL。例如：

.. code-block:: python

   model = dict(kind="file", file_path="tail.dat", format="headtail",
       axis_column=0, value_column=2,
       convention=dict(data_kind="wake_function", axis="time", axis_unit="ns",
           value_unit="V/(pC*mm)", positive_trailing=True,
           longitudinal_positive_loss=True, integrated=True, reference_beta=beta))

JSON 输入将 ``File path`` 相对其所在目录解析；直接 Python 构造则相对当前
工作目录。文件只在初始化读取，模型保存内容哈希和转换元数据，检查点同时
校验文件内容与配置。尚不包含 CST 工程/二进制解析、自动单位猜测或
wake-potential 反卷积；导出的数值文件可用 table 并明确实际约定。


切片坐标与响应边界
------------------

WakeField 接受局部 ``Coordinate=z_rel`` 区间及独立的
``Coordinate=arrival_phase`` 漂移束投影。SpaceCharge 使用局部 z 区间。
最近一次显式 Slicer 结果决定成员与几何。坐标选择不改变算法组的 ``Boundary``：
``periodic`` 表示重复的稳态空间响应，演化通过历史使用 ``causal_passages``。
时间计算使用 float64 与连续存储的 z。
