射频腔（RFCavity）
========================================

``RFCavity`` 将射频腔表示为零长度纵向作用，以有效电压计算粒子能量增量。
同一位置的各电压分量同时作用，所有束团按各自的到达时间采样同一组波形。
CPU 与 CUDA 使用相同算法，计算中间量采用 float64。

.. math::

   t_i=T_b-\frac{z_i}{\beta_b c},\qquad
   U(t)=\sum_k V_k(t)\sin\!\left[2\pi\int_{t_*}^{t}f_k(u)du+\phi_k(t)\right].

频率必须积分；变频时不能使用 ``2*pi*f(t)*t``。``Phase (rad)`` 是未折叠的附加相位调制，总瞬时频率为载波频率加相位调制导数除以 :math:`2\pi`。``harmonic_id`` 和派生的名义槽位位置不进入运行中的相位公式。

以下接口中的字段用于 ``PASS.para.schema.elements.RFCavityItem`` 配置类；
元件名称由 ``Sequence.add(name, item)`` 的序列键给出，不是配置类字段。

输入接口
--------

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python 配置字段
     - JSON 键
     - 类型
     - 单位
     - 默认值
     - 说明
   * - ``s``
     - ``S (m)``
     - ``float``
     - m
     - ``必填``
     - 元件出口或零长度作用点的纵向位置。
   * - ``length``
     - ``Length (m)``
     - ``float``
     - m
     - ``0.0``
     - 必须为零
   * - ``is_enabled``
     - ``Is enabled``
     - ``bool``
     - —
     - ``True``
     - 是否执行
   * - ``components``
     - ``Components``
     - ``list[RFComponent]``
     - —
     - ``必填``
     - 一个或多个 RFComponent
   * - ``dp_aperture``
     - ``Dp aperture``
     - ``list[float] | None``
     - 1
     - ``None``
     - 总能量更新后动量偏差边界
   * - ``aperture_type``
     - ``Aperture type``
     - ``str``
     - —
     - ``'off'``
     - 标准横向孔径
   * - ``aperture_value``
     - ``Aperture value``
     - ``list``
     - m / rad
     - ``[]``
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

   from PASS.para.schema import RFCavityItem, RFComponent

   rf = RFCavityItem(s=0.0, components=[
       RFComponent(voltage=100e3, harmonic=1, phase=0.3),
       RFComponent(voltage=[0., 20e3], frequency=[10e6, 10.1e6],
                   times=[0., 0.01], phase=1.2),
   ])

能量增量与参考量更新
--------------------

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

纵向电场保持机械横向动量。零长度作用不增加通过时间；z 缩放保证时间连续，不额外缩放实际能量偏差。无效参考能量报错，停止或无法向前传播的粒子标记损失。动量接受度在总能量更新之后检查。零电压仍检查孔径，禁用 command 则不执行。保存的切片区间、宽度和成员保持原值，由用户重新执行 Slicer 更新。

具体地，存储的 :math:`p_x=P_x/P_{0,b}` 是归一化动量，不是轨迹斜率。
参考动量更新后，:math:`P_x'=P_{0,b}'p_x'=P_{0,b}p_x=P_x`，y 方向同理。
而轨迹斜率 :math:`dx/ds=P_x/P_s` 可以随纵向动量改变；加速时横向角度
减小并不意味着机械横向动量减小。这些关系适用于这里的理想纵向薄透镜作用，
有限腔的横向电磁场与 RF 聚焦不包含在此模型内。

小能量增量的稳定计算
--------------------

实现通过避免相近动量相减，计算同一个精确动量映射。记
:math:`g_i=qU(t_i)`、:math:`g_0=qU(T_b)`、:math:`r=P_{0,b}/P_{0,b}'`，
将动量差有理化后得到：

.. math::

   P_i'-P_i=\frac{g_i(2E_i+g_i)}{P_i'+P_i},\qquad
   \delta_i'=r\delta_i
   -\frac{g_0(2E_{0,b}+g_0)}{P_{0,b}'(P_{0,b}'+P_{0,b})}
   +\frac{g_i(2E_i+g_i)}{P_{0,b}'(P_i'+P_i)}.

直接计算 :math:`P_i'/P_{0,b}'-1` 时，弱 RF 能量增量可能被舍入误差淹没。
上述形式保留这些小量，是精确公式的代数重排，没有对能量增量作线性化。

数值精度
--------

CPU 与 GPU 使用相同的能量和参考量变换。粒子坐标可采用 float32 或 float64，
时间、相位、能量和动量的中间运算使用 float64。相位围绕参考事件计算局部频率积分，
避免将束内微小到达时间差直接加到很大的累计相位上。
该方法不能恢复输入参考时刻中已丢失的信息；小增量写回 float32 坐标时仍可能发生舍入。

波形时间节点和值作为固定输入使用。改变规定波形时应重新构造相应配置和命令，
不要在跟踪过程中原地修改输入数组。

文件与同步程序
--------------

使用 ``{"Program file": "rf.tfs"}`` 读取 ``TIME, VOLTAGE, FREQUENCY, PHASE`` 列，单位依次为 s、V、Hz、rad。使用 ``{"Program file": "rf.tfs", "Harmonic": 2}`` 时文件应只有 ``TIME, VOLTAGE, PHASE``，禁止同时提供 FREQUENCY。文件模式不能混用内联波形值。

RF 表按物理时间给出；``convert_rf_data(input_path, output_path)`` 转换时间表格式，不把秒转换为圈号。

``PASS.para.tools.rf_data.synchronous_rf_program`` 从电压、目标通过相位及明确的设计粒子能量/飞行时间构造规定波形；它只生成输入，运行时不重置实际束团相位或能量。变频载波积分和附加相位程序共同保证设计采样点相位，采样点之间采用声明的线性插值。

物理范围
--------

模型采用有效电压、理想纵向薄透镜作用与零长度腔。不额外建模有限间隙渡越、RF 横向聚焦或腔内轨迹；若输入已包含渡越时间因子，不重复乘入。准确的 RF 能量增量不消除其他输运映射或准静态集体效应的近似。

物理 RF 表使用 tfs-pandas 的 ``colwidth=25, headerswidth=25`` 保存，保留 float64 时间精度。同步输入生成器中的 ``origin`` 是首次通过腔的时刻，``time_origin`` 是共同波形基准时刻，两者可以不同。
