射频腔（RFCavity）
========================================

``RFCavity`` 将射频腔表示为零长度纵向作用，以有效电压计算粒子能量增量。
同一位置的各电压分量同时作用，所有束团按各自的到达时间采样同一组波形。
CPU 与 CUDA 使用相同算法，计算中间量采用 float64。

.. math::

   t_i=T_b-\frac{z_i}{\beta_b c},\qquad
   U(t)=\sum_k V_k(t)\sin\!\left[2\pi\int_0^{t}f_k(u)du+\phi_k(t)\right].

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
* ``Harmonic``：正整数，乘以自动派生的设计回旋频率；不随集体效应引起的跟踪束团能量变化而改变，也不受分组谐波整除限制。

``Voltage (V)`` 与 ``Phase (rad)`` 默认为 0，可为标量或列表。
列表共享严格递增的有限 ``Time (s)``，并采用分段线性插值。
凡是提供时间采样的分量，包括在时间网格上指定标量值的情况，其电压在
首末采样时刻构成的闭区间外严格为零；两个端点均保留给定值。
只有一个时间节点的表仅在该时刻有效。未提供 ``Time (s)`` 的标量分量连续工作。
文件输入遵循相同规则；RF 表电压不在数据范围外保持端点值。

每个粒子按自身到达时间 :math:`t_i` 判断，参考粒子按 :math:`T_b` 判断，
因此同一束团内的粒子可以分别位于数据边界两侧。频率程序、积分得到的载波相位
及共享参考时钟保持连续，并保留原有的端点外推；仅分量电压受数据时间域限制。
自动纯 RF 设计时钟及其限制见 :ref:`zh-reference-clock`。


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

文件首末 ``TIME`` 采样点直接确定分量电压的有效时间域；
域外电压为零，无需额外配置参数。

RF 表按物理时间给出；``convert_rf_data(input_path, output_path)`` 转换时间表格式，不把秒转换为圈号。

``PASS.para.tools.rf_data.synchronous_rf_program`` 从电压、目标通过相位及明确的设计粒子能量/飞行时间构造规定波形；它只生成输入，运行时不重置实际束团相位或能量。变频载波积分和附加相位程序共同保证设计采样点相位，采样点之间采用声明的线性插值。

HIAF RF 图表导出
~~~~~~~~~~~~~~~~

``PASS.para.tools.hiaf_rf`` 读取 HIAF 图表导出目录中的 14 个双列
``#RF...PlotData`` 文件。无后缀名称表示通道 0，后缀 ``1``、``2`` 分别表示
通道 1、2。转换器检查有限值、所有文件共用的严格递增时间网格及整数谐波标签。
源文件单位为 ms、kV、kHz、rad，输出单位为 s、V、Hz、rad，并保留非均匀时间间隔。
转换器不依据离子质量修正频率，也不推断源数据采用的质量约定。

不预设相位约定、仅检查数据时，可运行：

.. code-block:: console

   python -m PASS.para.tools.hiaf_rf input/hiaf_export runs/rf_import

输出为 ``hiaf_rf_normalized.tfs`` 和 ``conversion_report.json``，包含源文件名、
SHA-256 校验值、单位和时间范围。默认保护已有输出；只有显式指定 ``--overwrite``
才允许替换同名生成文件。

生成可执行 RF 需要明确相位映射和时钟基准时刻。JSON 文件以
``通道:谐波`` 为键，给出各导出相位列的线性组合系数；可选 ``offset``
表示以 rad 为单位的常数。例如，BRing 捕获及合束前加速段的一种明确解释为：

.. code-block:: json

   {
     "0:4": {"Phase": 1},
     "1:8": {"Phase": 2, "DeltaPhi1": 1}
   }

将它保存为 ``phase_rules.json`` 后，选择适用的时间范围：

.. code-block:: console

   python -m PASS.para.tools.hiaf_rf input/hiaf_export runs/rf_import_mapped --phase-rules phase_rules.json --phase-origin 0.045006 --end-time 0.312614

这些时间是 2026-10-05 BRing 导出数据的示例，并非通用机器常数。
``--start-time`` 和 ``--end-time`` 以物理秒选择源数据内的范围；新增端点的模拟量
采用线性插值，谐波标签保持离散。``--phase-origin`` 指定原信号源时钟积分相位为零的物理时刻，
不平移时间列。即使截取分量时间域，仍使用完整源频率前史计算各段的相位修正。

上述示例明确规定 :math:`\psi_4=\mathrm{Phase}`、
:math:`\psi_8=2\mathrm{Phase}+\mathrm{DeltaPhi1}`。
数据中的 :math:`\mathrm{DeltaPhi1}=\pi-2\mathrm{Phase}` 关系因此给出
:math:`\psi_8=\pi`。该数值关系支持这种解释，但不能证明控制系统的相位语义；
将结果作为实机过程复现前仍需确认。相位列必须已展开，转换器不会自行展开相位，
也不会隐式叠加 ``Phase1``。后续 :math:`h=2,1` 合束阶段必须提供各自的明确映射。

指定映射后输出仅包含 ``Components`` 的 ``rf_config.json``，将该列表填入 RFCavity。
每个分量只指定 ``Program file``，文件包含 TIME、VOLTAGE、FREQUENCY、PHASE，
不再输出 ``Harmonic`` 字段；源谐波标签保留在文件名及转换元数据中。
显式 RF 频率等于该段源谐波数乘以基础通道频率，再除以 ``--base-harmonic`` （默认 4）。
活动通道频率必须在报告的容差内满足此关系。相邻禁用零电压节点继续采用该频率程序，
不使用禁用通道的零频率标记，从而保留旧共享源时钟转换实际生成的物理波形。

对于从 :math:`a` 开始的分量段，转换器增加常量相位
:math:`2\pi[h\int_{t_*}^{a}f_{base}(u)/h_{base}\,du-\int_0^a f_{segment}(u)\,du]`
并对 :math:`2\pi` 取模，使运行时从零开始积分仍保留原始时间原点与频率前史。
这种外部 RF 波形不会按自动设计时钟重新标定。相邻零电压节点保留线性开关过程，
分量时间域外电压为零。没有零电压分隔的谐波切换、缺失的活动相位映射、频率不一致均在写出前拒绝。

Python 接口 ``load_hiaf_rf(source_directory)`` 返回归一化数组；
``convert_hiaf_rf(source_directory, output_directory, phase_rules=...,
phase_origin=..., base_harmonic=4, start_time=None, end_time=None)`` 生成输出。
依据统一质量重新构建设计波形属于需要单独记录的步骤；普通格式转换保留源频率程序。

物理范围
--------

模型采用有效电压、理想纵向薄透镜作用与零长度腔。不额外建模有限间隙渡越、RF 横向聚焦或腔内轨迹；若输入已包含渡越时间因子，不重复乘入。准确的 RF 能量增量不消除其他输运映射或准静态集体效应的近似。

物理 RF 表使用 tfs-pandas 的 ``colwidth=25, headerswidth=25`` 保存，保留 float64 时间精度。
同步输入生成器中的 ``origin`` 仍是首次通过腔的时刻。旧 ``time_origin`` 参数继续接受并记录，
但其常量载波相位已吸收到输出 PHASE，使表格采用运行时的零时间原点。这保留要求的通过相位
与物理波形，不平移时间列，也不恢复公开机器时钟输入。
