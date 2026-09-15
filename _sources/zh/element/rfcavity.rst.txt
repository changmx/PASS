RF 腔（RFCavity）
========================================

``RFCavity`` 表示同一位置同时作用的有效电压薄踢。所有束团采样同一组物理波形，CPU 与 CUDA 使用相同算法及 float64 中间量。

.. math::

   t_i=T_b-\frac{z_i}{\beta_b c},\qquad
   U(t)=\sum_k V_k(t)\sin\!\left[2\pi\int_{t_*}^{t}f_k(u)du+\phi_k(t)\right].

频率必须积分；变频时不能使用 ``2*pi*f(t)*t``。``Phase (rad)`` 是未折叠的附加相位调制，总瞬时频率为载波频率加相位调制导数除以 :math:`2\pi`。``harmonic_id`` 和派生的名义槽位位置不进入运行中的相位公式。

能量踢与参考更新
----------------

离子的能量、静质量能和动量乘 c 按 PASS 的每核子 eV 约定，电荷因子为
带符号的 :math:`q=Z/A`；这里 Z 是电荷数，而不是质子数。电子、正电子
使用每粒子量，电荷因子分别为 -1、+1，不除以为零的核子数。
运行时采用 ``sign(bunch.num_charge) * bunch.qm_ratio``，其中
``qm_ratio`` 保持电荷质量比幅值的既有含义。同一 command 的全部分量
先在共同入口事件上求和，再进行一次更新：


.. math::

   E_i'=\sqrt{[P_{0,b}(1+\delta_i)]^2+m^2}+qU(t_i),\qquad
   E_{0,b}'=E_{0,b}+qU(T_b),

.. math::

   P_i'=\sqrt{(E_i'-m)(E_i'+m)},\quad
   P_{0,b}'=\sqrt{(E_{0,b}'-m)(E_{0,b}'+m)},\quad
   \delta_i'=P_i'/P_{0,b}'-1,

.. math::

   p_{x,y}'=p_{x,y}P_{0,b}/P_{0,b}',\qquad
   T_b'=T_b,\qquad z_i'=z_i\beta_b'/\beta_b.

纵向电场保持机械横向动量。零长度踢不增加通过时间；z 缩放保证时间连续，不额外缩放实际能量偏差。无效参考能量报错，停止或无法向前传播的粒子标记损失。动量接受度在总踢之后检查。零电压仍检查孔径，禁用 command 则不执行。保存的切片区间、宽度和成员保持原值，由用户重新执行 Slicer 更新。

具体地，存储的 :math:`p_x=P_x/P_{0,b}` 是归一化动量，不是轨迹斜率。
参考动量更新后，:math:`P_x'=P_{0,b}'p_x'=P_{0,b}p_x=P_x`，y 方向同理。
而轨迹斜率 :math:`dx/ds=P_x/P_s` 可以随纵向动量改变；加速时横向角度
减小并不意味着机械横向动量减小。这些关系适用于这里的理想纵向薄踢，
有限腔的横向电磁场与 RF 聚焦不包含在此模型内。

弱踢的稳定计算
--------------

实现通过避免相近动量相减，计算同一个精确动量映射。记
:math:`g_i=qU(t_i)`、:math:`g_0=qU(T_b)`、:math:`r=P_{0,b}/P_{0,b}'`，
将动量差有理化后得到：

.. math::

   P_i'-P_i=\frac{g_i(2E_i+g_i)}{P_i'+P_i},\qquad
   \delta_i'=r\delta_i
   -\frac{g_0(2E_{0,b}+g_0)}{P_{0,b}'(P_{0,b}'+P_{0,b})}
   +\frac{g_i(2E_i+g_i)}{P_{0,b}'(P_i'+P_i)}.

直接计算 :math:`P_i'/P_{0,b}'-1` 时，弱 RF 踢可能被舍入误差淹没。
上述形式保留这些小量，是精确公式的代数重排，没有对踢角作线性化。

GPU 实现
--------

CPU 使用 NumPy 数组。GPU 使用 ``PASS/commands/element/rfcavity.py`` 中的
融合 ``RawKernel``：每个存活粒子在同一内核中计算所有波形分量、执行精确能量
与参考量变换，并记录纵向损失。每个非空束团启动一次 RF 内核；启用横向孔径时，
随后单独执行标准孔径检查。

粒子存储可为 float32 或 float64，时间、相位、能量和动量中间量始终保持
float64。RF 内核不分配与粒子数等长的能量增益、时间、掩码或能量临时数组。
常值波形使用专门路径；时变波形使用缓存的设备端时间表，其线性插值、频率积分
和端点保持规则与 CPU 一致。查表时比较局部时间偏移与平移后的节点，避免在
较大参考时刻下丢失束内微小到达时间差。

分量数不超过 32 时，波形描述参数按值传入；更多分量使用打包的设备缓冲区，
避免超过可移植的内核参数大小限制。设备时间表和编译内核按 command、设备和
粒子精度缓存。稳态性能测量应排除首次编译和上传。旧单分量 RawKernel 使用
不同的坐标与精度规则，不能替代当前物理映射。

CPU 标量与数组时间表求值统一由 ``PASS.utils.program.LinearProgram`` 管理；CPU/GPU 共用
参考踢的主机计算。程序持有输入时间与数值数组的独立只读副本，避免外部
修改输入后使插值系数、积分或设备缓存失效。更换规定波形时应重新构造
程序及使用该程序的 command，不在跟踪中原地修改程序数组。

输入接口
--------

.. list-table::
   :header-rows: 1

   * - JSON key
     - Type
     - Default
     - Meaning
   * - S (m)
     - float
     - Required
     - 物理位置
   * - Length (m)
     - float
     - 0
     - 必须为零
   * - Is enabled
     - bool
     - true
     - 是否执行
   * - Components
     - list
     - Required
     - 一个或多个 RFComponent
   * - Dp aperture
     - [float,float]
     - None（关闭）
     - 总踢后动量偏差边界
   * - Aperture type / Aperture value
     - str / list
     - off / []
     - 标准横向孔径

每个分量选择以下两种频率定义之一：

* ``Frequency (Hz)``：规定的实际载波频率，正标量或列表。
* ``Harmonic``：正整数，乘以共同 ``Reference clock`` 的回旋频率；不依赖当前束团能量，也不受分组谐波整除限制。

``Voltage (V)`` 与 ``Phase (rad)`` 默认为 0，可为标量或列表。列表共享严格递增的有限 ``Time (s)``。所有程序分段线性插值，区间外保持端点值。参考时钟与默认值见 :ref:`zh-reference-clock`。


.. code-block:: json

   {
     "Command": "RFCavity", "S (m)": 0.0,
     "Components": [
       {"Voltage (V)": 100000.0, "Frequency (Hz)": 5000000.0, "Phase (rad)": 0.3},
       {"Voltage (V)": 20000.0, "Frequency (Hz)": 10000000.0, "Phase (rad)": 1.2}
     ]
   }

.. code-block:: python

   from PASS.para.schema import RFCavityElement, RFComponent

   rf = RFCavityElement(s=0.0, components=[
       RFComponent(voltage=100e3, harmonic=1, phase=0.3),
       RFComponent(voltage=[0., 20e3], frequency=[10e6, 10.1e6],
                   times=[0., 0.01], phase=1.2),
   ])

文件与同步程序
--------------

使用 ``{"Program file": "rf.tfs"}`` 读取 ``TIME, VOLTAGE, FREQUENCY, PHASE`` 列，单位依次为 s、V、Hz、rad。使用 ``{"Program file": "rf.tfs", "Harmonic": 2}`` 时文件应只有 ``TIME, VOLTAGE, PHASE``，禁止同时提供 FREQUENCY。文件模式不能混用内联波形值。

旧腔级 ``Voltage (V)``、``Harmonic``、``Phase (rad)``、``Phi offset (rad)``、``RF data file`` 及按圈索引的 RF 表已移除。``convert_rf_data(input_path, output_path)`` 仅转换物理时间表，不再把秒转换为圈号。

``PASS.para.tools.rf_data.synchronous_rf_program`` 从电压、目标通过相位及明确的设计粒子能量/飞行时间构造规定波形；它只生成输入，运行时不重置实际束团相位或能量。变频载波积分和附加相位程序共同保证设计采样点相位，采样点之间采用声明的线性插值。

物理范围
--------

模型采用有效电压、理想纵向薄踢与零长度腔。不额外建模有限间隙渡越、RF 横向聚焦或腔内轨迹；若输入已包含渡越时间因子，不重复乘入。准确的 RF 踢不消除其他输运映射或准静态集体效应的近似。

物理 RF 表使用 tfs-pandas 的 ``colwidth=25, headerswidth=25`` 保存，保留 float64 时间精度。同步输入生成器中的 ``origin`` 是首次通过腔的时刻，``time_origin`` 是共同波形基准时刻，两者可以不同。
