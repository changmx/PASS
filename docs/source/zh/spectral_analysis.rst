频谱分析与频率图
================

**分析** 工作区提供基础 FFT、精细 FFT 和频率图分析（FMA），与 Python 脚本
调用相同的数组计算函数。无需打开跟踪输入或工程即可使用。虚拟 BPM、OMC3 接入
和光学重建不属于此工作流。

独立 Python 函数
----------------

三个公开函数分别为 ``PASS/analysis/fft.py`` 的 ``compute_fft``、
``PASS/analysis/refined_fft.py`` 的 ``compute_refined_fft`` 和
``PASS/analysis/fma.py`` 的 ``compute_fma``。完整复制任意一个函数定义到其他
Python 文件后，安装 NumPy 即可使用。必要导入和辅助函数均在函数内部；三个函数
不依赖 PASS、Qt、文件读取器或绘图模块，不修改输入，返回包含数组与元数据的字典。

.. code-block:: python

   import numpy as np
   from PASS.analysis import compute_fft, compute_refined_fft, compute_fma

   turns = np.arange(2048)
   x = 0.002 * np.cos(2 * np.pi * 0.2317 * turns + 0.3)
   y = 0.001 * np.cos(2 * np.pi * 0.3172 * turns - 0.2)

   spectrum = compute_fft(x, sample_spacing=1.0)
   peaks = compute_refined_fft(
       x, window="hann", padding_factor=8,
       frequency_range=(0.20, 0.27), n_peaks=1,
   )
   print(peaks["peak_frequency"], peaks["peak_amplitude"])

   # Rows identify particles; the last axis contains consecutive samples.
   fma = compute_fma(
       x[None, :], y[None, :],
       windows=((0, 1024), (1024, 2048)),
       frequency_range_x=(0.20, 0.27),
       frequency_range_y=(0.29, 0.35),
   )
   print(fma["qx_first"], fma["qx_second"], fma["drift"])

FFT 约定
--------

``sample_spacing`` 必须有限且为正。每圈一次、间隔为 1 时频率单位为 cycles/turn；
间隔以秒计时频率单位为 Hz。每若干圈采样一次必须填写实际圈数间隔，对应的 Nyquist
频带也会缩小。FFT 不恢复整数 tune，也不自动消除混叠。
对于实数逐圈信号，仅凭正频率谱不能区分小数 tune ``Q`` 与 ``1-Q``。

``compute_fft`` 接受实数或复数数组、``axis``（默认 -1）和 ``remove_mean``
（默认 false）。输出将频率放在最后一个轴。实信号使用单边谱，复信号使用排序后的
正负频率双边谱。``coefficients`` 是按幅值归一化的复系数，``amplitude`` 是其模，
``phase`` 以弧度计，相位参考为所分析区间的第一个样本。实信号正频率系数乘以 2，
但 DC 和偶数长度的 Nyquist 点不加倍。输出为幅度谱，不是功率谱密度。

精细 FFT 接受 ``window``（``rectangle``、``hann``、``hamming``、``blackman``）、
整数 ``padding_factor``、``interpolation``（``none`` 或 ``parabolic``）、
``frequency_range`` 和 ``n_peaks``。默认使用 Hann 窗、8 倍补零、抛物线插值并提取
一个谱峰。算法校正窗函数的相干增益，在估计的频率处重新计算复系数。补零使频率
网格更密，并不增加观测圈数或实际分辨能力。精度取决于记录长度、窗函数、噪声、
邻近谱线和时间变化。输出包含峰值有效性和质量标记；无效结果不代表测得零频率。
仅需峰值时可用 ``return_spectrum=False`` 避免返回频谱数组。

FMA 约定
--------

FMA 比较两个时间窗的频率，本身不是另一种 FFT 方法。``x`` 和 ``y`` 的形状必须
一致，最后一个轴为采样轴。单条轨迹使用一维数组，粒子群使用前置对象维度。
显式窗口为 ``((start1, end1), (start2, end2))``，结束索引不包含在内；两窗口必须
等长、依次排列且不重叠。默认使用前后两个等长相邻区间，奇数长度时舍去末尾一个
样本。元数据记录实际使用的窗口。

``method`` 可选 ``fft`` 或 ``refined_fft``，FMA 默认去均值。独立函数内部包含
默认频率估计器，也允许通过可选 ``frequency_estimator`` 显式提供自定义估计器，
不要求调用者额外复制其他函数。回调契约见函数文档字符串；默认估计器与独立
频率计算函数采用相同的数值约定。

两个横向信号的输出定义为

.. math::

   \Delta Q_x = Q_{x,2}-Q_{x,1},\qquad
   \Delta Q_y = Q_{y,2}-Q_{y,1},\qquad
   d_Q=\sqrt{(\Delta Q_x)^2+(\Delta Q_y)^2},\qquad
   D=\log_{10}d_Q.

对应键为 ``qx_first``、``qy_first``、``qx_second``、``qy_second``、
``delta_qx``、``delta_qy``、``drift`` 和 ``diffusion_log10``。漂移精确为零时
保留 ``D=-inf``，绘图色标下限不改变数值结果。差值不自动周期折叠；两个窗口应
选择一致的频率分支与搜索区间。窗口含非有限值或频率估计无效时，通过 ``valid``
和 ``quality`` 标记该对象。程序调谐、加速、噪声和集体演化也可能造成频率变化，
该指标本身不能证明混沌运动。

文件与采样
----------

``PASS.analysis.data_io.inspect_data`` 列出可选数值列或数组；``load_signal``
读取显式指定的数据，返回信号、采样坐标、间隔和来源元数据。数值函数直接接受
数组，不依赖文件格式。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 格式
     - 数据选择
   * - CSV / TSV
     - 命名数值列，可配置表头和跳过行数。
   * - TXT / DAT
     - 分隔的数值文本，可选择分隔符、表头处理及跳过行数。
   * - TFS
     - 数值列及 TFS 头部元数据。
   * - HDF5（``.h5``、``.hdf5``）
     - 显式选择数值数据集路径与采样轴，保留文件及数据集属性。
   * - NPY
     - ``data`` 数组及其显式采样轴。
   * - NPZ
     - 命名数值数组及其显式采样轴。

不载入 object 数组或 pickle 数据。多维数组需要明确采样轴，``sample_range``
和 ``object_range`` 均采用左闭右开的索引区间。显式指定 ``object_range`` 时，前置
对象维度按 C 顺序展平；省略时保留原有对象维度。返回的 ``object_ids`` 是原数组
中的位置索引，不是物理粒子身份；调用者须保证 X/Y 数组采用相同的粒子顺序。
若提供采样坐标列或数据集，其值
必须严格递增且等间隔；使用坐标推导的间隔替代手工填写值。缺圈、重复坐标和不等
间隔采样不会被静默删除、排序、插值或补零。已识别的 PASS ParticleMonitor 文件
自动使用 ``turn`` 列；其他情况下没有坐标时，调用者负责给出正确间隔。

CSV 默认逗号分隔，TSV 默认制表符，TXT/DAT 默认空白分隔。文本表头可选
``auto``、``present``、``absent``；无表头时列名为 ``column_0``、``column_1``
等。HDF5/NPY 在复制前切片；NPZ 读取所选的完整成员，文本读取整张表。

PASS ParticleMonitor 会冻结已损失粒子的记录。数据适配器对已识别的
ParticleMonitor 文件使用带符号 tag，也识别保留 PASS 元数据的转换文件；如果选择
中含有已损失或非有限样本，会拒绝该选择。外部文件可以显式选择存活状态列，正值
表示存活。应选择完整的存活区间，不能把冻结尾段当成振荡。
同一表中若不同粒子使用重复圈号，应先整理为独立轨迹。

GUI 使用流程
------------

1. 打开 **分析**，选择 **频谱分析** 或 **频率图分析（FMA）**，载入数据文件。
2. 选择信号列或数据集、采样轴、坐标列或间隔，以及样本与对象范围。
3. 频谱分析选择基础或精细 FFT；精细 FFT 可配置窗函数、补零、插值和搜索频段。
4. FMA 选择成对信号、两个等长窗口及频率估计设置。窗口索引相对于已选择的信号区间。
5. 执行后查看图形和数值结果，导出数据或图片。**复制 Python 调用** 将所选数据和
   计算参数写成可用于脚本的调用代码。

文件读取和计算在后台执行。显示预览可以限制点数，计算使用选定的完整分辨率
数组。取消会等待当前数值操作结束并丢弃结果。普通数据文件的载入不改变跟踪配置。

NPZ 导出保留结果数组、所选输入数组、坐标和元数据。CSV 导出完整数值结果，并以
注释保存元数据；频谱行使用 ``record_type=spectrum``，精细峰值行使用
``record_type=peak``，包含有效性和质量字段。PNG/SVG 导出保存当前显示的图形，
包括其预览点数限制。
