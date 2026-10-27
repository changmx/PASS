ACDipole
========

``ACDipole`` 是零长度、单横向平面的激励元件，用于固定能量光学测量和规定的扫频实验。
幅度与频率使用相互独立的物理时间程序。所有 bunch 取样同一个波形；
圈数计数器和 ``harmonic_id`` 都不会重置波形相位。

物理约定
--------

对于 ``Direction = x`` （垂直激励时将 :math:`x` 换为 :math:`y`），

.. math::

   t_i=t_0-\frac{z_i}{\beta_0c},\qquad
   \Phi(t_i)=\phi_0+2\pi\int_{t_*}^{t_i} f_{\rm ac}(u)\,du,
   \qquad p_{x,i}^{+}=p_{x,i}^{-}+\theta(t_i)\sin\Phi(t_i).

``Kick angle (rad)`` 指定有符号振幅 :math:`\theta`，``Phase (rad)`` 指定
:math:`\phi_0`，``Phase epoch (s)`` 指定 :math:`t_*`。PASS 存储的是
:math:`p_x=P_x/P_0`，因此直接加上这个 kick，不再除以 :math:`1+\delta`。
只有傍轴近似下的斜率增量才满足 :math:`\Delta x'\simeq\Delta p_x/(1+\delta)`。

正 ``z`` 表示更早通过。实际 bunch 参考通过时刻 ``bunch.t0`` 和连续的束内相对
坐标 ``p.z`` 决定取样时间；名义束团槽位位置不进入这个公式。
该模型不改变位置、``dp``、参考能量和 ``bunch.t0``。
它描述理想横向激励，不包含完整时变电磁场的纵向感应项，因此不声称是完整的六维辛映射。

在右手 :math:`(x,y,s)` 坐标系中沿 :math:`+s` 运动时，洛伦兹力给出
:math:`\theta_x=-q\int B_y ds/P_0` 和 :math:`\theta_y=+q\int B_x ds/P_0`，
其中 :math:`q` 是有符号粒子电荷。这些公式在固定参考动量下将物理磁场积分转换为输入振幅。
加速过程中固定的归一化输入不等于
固定的实验室磁场；这类计算需要显式给出适当的归一化幅度程序。

输入接口
--------

配置模型为 ``PASS.para.schema.elements.ACDipoleItem``，名称由
``Sequence.add(name, item)`` 提供。所有数值都必须有限。

.. list-table::
   :header-rows: 1
   :widths: 17 21 14 8 11 29

   * - Python 字段
     - JSON 键
     - 类型
     - 单位
     - 默认值
     - 含义
   * - ``s``
     - ``S (m)``
     - ``float``
     - m
     - 必填
     - 作用位置。
   * - ``length``
     - ``Length (m)``
     - ``float``
     - m
     - ``0.0``
     - 必须为零。
   * - ``direction``
     - ``Direction``
     - ``'x' | 'y'``
     - —
     - 必填
     - 激励横向平面。
   * - ``kick_angle``
     - ``Kick angle (rad)``
     - ``float | list[float]``
     - rad
     - 必填
     - 有符号归一化动量振幅。
   * - ``kick_times``
     - ``Kick time (s)``
     - ``list[float] | None``
     - s
     - ``None``
     - 幅度采样时刻及有效时间区间。
   * - ``frequency``
     - ``Frequency (Hz)``
     - ``float | list[float]``
     - Hz
     - 必填
     - 非负载波频率，允许为零。
   * - ``frequency_times``
     - ``Frequency time (s)``
     - ``list[float] | None``
     - s
     - ``None``
     - 独立的频率采样时刻。
   * - ``phase``
     - ``Phase (rad)``
     - ``float``
     - rad
     - ``0.0``
     - 相位原点时刻的载波相位。
   * - ``phase_epoch``
     - ``Phase epoch (s)``
     - ``float``
     - s
     - ``0.0``
     - 频率积分的下限。
   * - ``is_enabled``
     - ``Is enabled``
     - ``bool``
     - —
     - ``True``
     - 启用激励。
   * - ``cpu_chunk_size``
     - ``CPU chunk size``
     - ``int | None``
     - 粒子数
     - ``65536``
     - 每个 CPU 批次的最大粒子数，必须为正整数。``None`` 表示整束一次处理。GPU 忽略此参数。
   * - ``order``
     - ``Order``
     - ``int | None``
     - —
     - ``None``
     - 同一位置命令的显式顺序。
   * - ``aperture_type``
     - ``Aperture type``
     - ``str``
     - —
     - ``'off'``
     - 标准横向损失孔径。
   * - ``aperture_value``
     - ``Aperture value``
     - ``list``
     - m / rad
     - ``[]``
     - 标准孔径尺寸。

时间表与相位
------------

每个提供的时间数组都必须非空且严格递增。包含多个数值的数组必须具有等长的对应时间数组，
采样点之间采用线性插值。标量或只有一个数值的数组表示常量，也可以同时指定时间采样点。

提供 ``Kick time (s)`` 时，首尾采样点构成的闭区间之外 kick 为零。
两个端点仍有效；单时间点区间仅在该时刻有效。不提供 kick 时间表时，标量幅度连续有效。
每个粒子按自己的到达时刻判断有效区间，因此一个 bunch 可以跨越边界。
频率表在区间外保持端点值，相位在所有频率节点及幅度边界处连续。

``LinearProgram`` 精确积分每个线性频率段。在某一段内若
:math:`f(t)=f_a+\dot f(t-t_a)`，则

.. math::

   \Phi(t)-\Phi(t_a)=2\pi\left[f_a(t-t_a)+\tfrac12\dot f(t-t_a)^2\right].

使用 :math:`2\pi f(t)(t-t_*)` 会产生错误的扫频相位。
实现分别保留参考时刻和粒子时间偏移，并在计算正弦前按周期约化相位。
输入程序是不可变快照，修改波形需要重建命令。这里没有需要在新运行开始时恢复的逐圈相位累加状态。

CPU 和 CUDA 的波形中间量使用 float64，粒子数组保持配置的存储精度。
分别保留微小偏移可以避免将其直接加到大时间原点时丢失信息，但不能消除浮点输入
或局部相位乘积的舍入。时间与频率应处于有实际物理意义的范围，不能假定任意数量级
都保留任意相位精度。

CPU 分块大小
------------

默认 ``CPU chunk size = 65536`` 每个 NumPy 批次处理至多 65,536 个粒子。
显式设为 ``None`` 时整束一次处理。正整数限制每批粒子数，从而约束临时数组内存。修改此参数
不改变粒子映射，也不会创建工作线程或将 bunch 分配到多个 CPU 核心；GPU 执行不受影响。

在实际服务器上，应使用代表性的粒子数和波形，测试例如 16,384、65,536、262,144、
1,048,576 粒子每块，必要时测试更大的块，并与 ``None`` 比较。更大内存允许更大的
数组，但更大的块也可能超过 CPU 缓存容量并增加内存访问量。核数、内存带宽和
NUMA 数据位置是不同因素；分块大小参数本身不提供多核或 NUMA 调度。应依据目标
服务器上测得的吞吐量与峰值内存选择，而不是套用另一台机器的计时结果。

固定频率光学测量示例
--------------------

固定能量环中设 :math:`f_{\rm rev}=\beta_0c/C`，并使用
:math:`f_{\rm ac}=q_d f_{\rm rev}`。需要选择实际物理激励频率所在的支路：
给 :math:`q_d` 加整数时，单个理想参考粒子的逐圈取样相同，但其他 bunch 及不同到达时刻
的粒子所见相位一般不同。

以下例子假设 :math:`f_{\rm rev}=10^6` Hz，且参考粒子首次通过元件的时刻为 :math:`t=0`。
实际使用时替换为所配置环的规定时钟和首次通过时刻。

.. code-block:: python

   from PASS.para.schema import ACDipoleItem

   f_rev = 1.0e6
   first_passage_time = 0.0
   ac = ACDipoleItem(
       s=0.0,
       direction="x",
       kick_angle=[0.0, 1.0e-6, 1.0e-6, 0.0],
       kick_times=[first_passage_time + n / f_rev for n in (0, 2000, 6000, 8000)],
       frequency=0.31 * f_rev,
       phase=0.0,
       phase_epoch=first_passage_time,
   )
   sequence.add("ac_x", ac)

测量平台期应选择与束流 tune 分布有间隔的激励 tune。幅度升降时间（以圈计）应明显长于
该 tune 间隔的倒数。在将受迫轨迹用于线性光学分析前，应检查关断后的自由振荡残余和
响应的幅度依赖。实正弦激励包含两个旋转分量，受迫光学需要考虑双边带响应，不能直接等同于自由 betatron 运动。
受迫响应的解释依据 R. Miyamoto 等人的
`Parametrization of the Driven Betatron Oscillation
<https://arxiv.org/abs/0709.4192>`_，Phys. Rev. ST Accel. Beams 11, 084002 (2008)。

扫频时，可以将标量频率替换为 ``frequency=[0.30*f_rev, 0.32*f_rev]``，并设置
``frequency_times=[first_passage_time, first_passage_time + 8000/f_rev]``。
幅度程序保持原样。扫过束流 tune 分布的实验与稳态光学测量平台期属于不同的激励过程。

与 MAD-X 的比较
---------------

MAD-X ``TRACK`` 中的 ``HACDIPOLE``/``VACDIPOLE`` 按圈取样，``FREQ`` 的单位是周/圈。
匹配参考粒子的采样值时使用 :math:`f_{\rm ac}=\mathtt{FREQ}\,f_{\rm rev}`，
振幅归一化采用 :math:`\theta_0=0.3\,\mathtt{VOLT}/\mathtt{PC}`（``PC`` 单位 GeV/c）。
初相取 :math:`\phi_0=2\pi\,\mathtt{LAG}`。MAD-X 将 ``START`` 状态输出为第零圈，
然后在第一圈施加首次 kick。该对照应使 PASS 在物理时刻 :math:`t_n=n/f_{\rm rev}`
取样，并使用 ``Phase epoch (s) = 0``；仅凭 PASS 圈数计数器不能确定这个时间原点。
若 ``RAMP1`` 至 ``RAMP4`` 严格递增，可将其除以 :math:`f_{\rm rev}` 后作为 kick
采样时刻，振幅取 :math:`[0,\theta_0,\theta_0,0]`。
PASS 进一步按各粒子的实际物理到达时刻取样，并支持连续频率程序。
MAD-X ``PTC`` 的 AC dipole 约定不同，不属于上述 ``TRACK`` 对应关系。

逐圈幅度升降参数及 ``TRACK`` 与 ``PTC`` 的差异见
`MAD-X AC dipole 手册源码
<https://github.com/MethodicalAcceleratorDesign/MAD-X-User-Manual/blob/master/elements.tex>`_。
