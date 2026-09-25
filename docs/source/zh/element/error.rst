.. _zh-error:

元件误差
========

本页统一介绍元件误差模型及其配置。CPU、GPU 均支持绝对磁场误差和静态磁场准直误差
（``DX``、``DY``、``DPSI``）。

场误差
------

SBend、Quadrupole、Sextupole、Octupole、Multipole、Solenoid 和 Kicker
在 CPU、GPU 上均支持相同的附加正、斜多极积分场误差。误差系数在跟踪期间保持固定。
本接口只支持绝对磁场误差；相对误差生成、孔径偏移和 BPM 读数误差不在模型范围内。

参数与归一化
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python 配置字段
     - JSON 键
     - 类型
     - 单位
     - 默认值
     - 说明
   * - ``is_field_error``
     - ``Is field error``
     - ``bool``
     - —
     - ``False``
     - 启用附加场误差动量更新。
   * - ``field_error_knl``
     - ``Field error KNL``
     - ``list[float]``
     - m^-n
     - ``[]``
     - 正多极积分误差，从第 0 阶偶极项开始。
   * - ``field_error_ksl``
     - ``Field error KSL``
     - ``list[float]``
     - m^-n
     - ``[]``
     - 斜多极积分误差，从第 0 阶偶极项开始。


数组第 :math:`n` 项单位为 :math:`\mathrm{m}^{-n}`；第 0、1、2、3 项依次为偶极、四极、六极、八极。
系数必须有限。缺失阶数补零，较短数组补齐；全零误差无效应，不按幅值阈值丢弃有限小量。
开关关闭时，即使数组非零也保持名义映射。

在 :math:`p_x=P_x/P_0`、:math:`p_y=P_y/P_0` 约定下，完整积分误差动量更新为

.. math::

   F(x,y) = \sum_{n=0}^{N}
   \frac{\Delta K_nL+i\Delta K_{ns}L}{n!}(x+iy)^n,
   \qquad \Delta p_x=-\Re F,\quad \Delta p_y=\Im F.

输入系数不含阶乘；归一化动量动量更新也不额外乘 :math:`1/(1+\delta)`。
因此正偶极误差为正时减小 ``px``，与正 ``HKICK`` 符号相反；
斜偶极误差为正时增大 ``py``，与正 ``VKICK`` 一致。

Multipole 和 Solenoid 的 ``KiL`` / ``KiSL`` 表示名义横向多极分量，启用的误差只与它们相加一次。
Solenoid 的 ``KS`` 是独立的纵向场强 :math:`B_z/(B\rho)`，不是斜多极数组。
螺线管纵向主场变化应通过 ``KS`` 配置。

跟踪模型
~~~~~~~~

薄元件执行一次完整积分误差动量更新。厚元件将误差沿长度均匀分布：
每个带符号积分步 :math:`\Delta s` 使用 :math:`\Delta s/L` 的积分误差。
Yoshida 子步保留负权重，因此增加切片不会增加总误差强度。

保留原有名义传输。DKD/RKR/SKS 将误差加在中心动量更新处；
四极矩阵模型采用“半段名义矩阵、误差动量更新、半段名义矩阵”。
内部空间电荷仍在计划节点执行，使用自身的正积分权重。



模型限制
^^^^^^^^

SBend 保留名义曲率及入、出口映射，在弯曲名义传输之间加入局部直线多极动量更新。
这是局部薄误差近似，不包含完整的曲线多极场或与场误差相关的端部磁场。

Solenoid 长度为零时纵向场映射无效应，但横向积分多极分量和启用的场误差仍执行薄动量更新。

.. code-block:: json

   {
     "Command": "Quadrupole",
     "S (m)": 1.0,
     "Length (m)": 0.3,
     "K1L": 0.15,
     "Num slices": 16,
     "Integrator": "yoshida4",
     "Is field error": true,
     "Field error KNL": [0.00001, 0.002, 0.3],
     "Field error KSL": [-0.00002, -0.001]
   }

MAD-X 导入
~~~~~~~~~~~~~~~~

将名义光学和强度导出为 Twiss TFS，再使用 ``EFCOMP, DKN=..., DKS=...``
设置绝对积分误差并通过 ``ESAVE`` 导出独立误差 TFS。
PASS 从误差表读取 ``K0L``、``K1L`` 等正分量和 ``K0SL``、``K1SL`` 等斜分量。
这些是附加强度；不要同时把同一份误差写进名义强度或源光学，避免重复计入。
PASS 不生成相对误差。

.. code-block:: python

   from PASS.para.madx import read_madx_elements

   items, names, circumference = read_madx_elements(
       "ideal.tfs", error_file="errors.tfs", is_field_error=True)

读取器在漂移合并或重采样之前，根据原始 Twiss 的 ``NAME`` 列匹配实例。
唯一名称可直接匹配，不区分大小写；重复名称必须明确指定 ``Q[2]`` 或 ``Q:2`` 等序号。
稀疏误差表的行次序不能用于推断实例。名称歧义、目标不存在、重复非零记录及非有限系数均报错。
至少需要一个支持的系数列，其余缺失阶数补零；接受格式有效的空表和全零误差。
准直列仅在 ``is_alignment_error=True`` 时读取；孔径和监视器误差列不参与导入。

逐元件导入将误差附加到物理磁铁，不支持的元件出现非零场误差时直接报错。
Twiss 传输导入在原始元件出口 ``S`` 处插入一个薄 Multipole，在 Twiss 映射之后执行；
合并和重采样保留这些必要位置。这种集中误差与逐元件跟踪中的分布误差是不同近似。

螺线管逐元件导入从 MAD-X 标准 Twiss 的 ``KSI`` 列计算 ``KS = KSI/L``；
若没有 ``KSI``，也接受自定义 ``KS`` 列。两列均缺失，或零长度时 ``KSI`` 非零，均报错。

准直误差
--------

上述七类磁性元件均支持固定横向偏移 ``DX``、``DY`` 和绕纵轴转角 ``DPSI``。
这些参数使名义磁场与附加场误差多极分量一起移动；元件孔径和空间电荷导体边界保持在设计位置。
准直误差不会移动束流或初始粒子分布。

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python 配置字段
     - JSON 键
     - 类型
     - 单位
     - 默认值
     - 说明
   * - ``is_alignment_error``
     - ``Is alignment error``
     - ``bool``
     - —
     - ``False``
     - 启用磁场偏移和转角。
   * - ``alignment_dx``
     - ``Alignment DX (m)``
     - ``float``
     - m
     - ``0.0``
     - 在理想入口坐标系中指定的水平偏移。
   * - ``alignment_dy``
     - ``Alignment DY (m)``
     - ``float``
     - m
     - ``0.0``
     - 在理想入口坐标系中指定的垂直偏移。
   * - ``alignment_dpsi``
     - ``Alignment DPSI (rad)``
     - ``float``
     - rad
     - ``0.0``
     - 绕理想入口纵轴、遵循右手定则的转角。


所有参数必须有限。准直与场误差开关彼此独立。关闭准直开关或参数全为精确零时，
直接使用原有跟踪路径；不按阈值丢弃有限小量。以准直分量形式传入的非零 ``DS``、
``DPHI``、``DTHETA`` 明确报错，包括 schema 输入及已开启的 MAD-X 准直导入。
其他元件不支持准直误差。本版不实现随机抽样、随时间变化的准直误差、孔径偏移
（``AREX`` / ``AREY``）以及 BPM 读数或刻度误差。

坐标变换
~~~~~~~~

对于直线元件，入口变换为

.. math::

   \begin{pmatrix}x_m\\y_m\end{pmatrix}
   = R(-\psi)\left[\begin{pmatrix}x\\y\end{pmatrix}
   -\begin{pmatrix}DX\\DY\end{pmatrix}\right],\qquad
   \begin{pmatrix}p_{xm}\\p_{ym}\end{pmatrix}
   = R(-\psi)\begin{pmatrix}p_x\\p_y\end{pmatrix}.

名义映射和已启用的场误差动量更新在磁场坐标系中计算。
出口通过逆变换恢复设计坐标，再检查固定孔径。直线元件的坐标补丁不改变 ``z``、``dp``
和参考时钟。旋转直接使用三角函数，不采用小角近似。薄元件的入、出口位于同一平面。

有限长度 SBend 将整个弯曲磁场绕理想入口刚性平移、旋转。
在出口和内部 SC 节点，磁场截面与设计截面不再重合；PASS 先旋转三维归一化动量，
再将粒子射线投影到目标平面。若变换后位置为 :math:`r'`、动量为 :math:`u'`，则

.. math::

   \lambda=-r'_z/u'_z,\qquad
   r_{\perp,\mathrm{new}}=r'_\perp+\lambda u'_\perp,\qquad
   \Delta z=-\lambda\sqrt{1-\beta_0^2+\beta_0^2(1+\delta)^2}.

旋转前 :math:`u_z=\sqrt{(1+\delta)^2-p_x^2-p_y^2}`。
坐标补丁保持 ``dp``，不推进 ``bunch.t0``；元件的 ``execute_cpu/gpu`` 在跟踪和
出口孔径检查后统一推进一次参考时钟。
这样保留连续束团相对时间，并计入两个截面之间的粒子飞行时间差。
不能正向到达目标平面或无效的交点记为损失。零长度 SBend 沿用现有的直线薄动量更新模型。

空间电荷、孔径与损失
~~~~~~~~~~~~~~~~~~~~

同时启用两种误差和内部 SC 时，执行顺序为：

#. 从设计入口变换到磁场入口坐标系。
#. 执行名义磁场分片传输和分布场误差动量更新。
#. 每到计划的 SC 节点，将存活粒子转回设计坐标，用原网格和边界计算 SC，再转回磁场坐标。
#. 在出口恢复设计坐标，检查固定元件孔径。
#. 按元件长度推进一次参考时钟。

SC 的位置和积分权重不变。关闭 SC 时没有第 3 步；关闭准直误差时没有坐标补丁。
在上游已损失的粒子保持冻结；元件内部损失的粒子最终恢复到所记录损失平面的设计坐标，
不会继续传输或复活。损失平面的精度沿用现有损失位置存储精度。
元件孔径仍是出口检查，不是沿程连续碰壁检测。

MAD-X 准直导入
~~~~~~~~~~~~~~~~~~~~

用 ``EALIGN, DX=..., DY=..., DPSI=...`` 设置准直，再由 ``ESAVE`` 导出。
同一误差 TFS 可以同时含准直和绝对场误差：

.. code-block:: python

   items, names, circumference = read_madx_elements(
       "ideal.tfs", error_file="errors.tfs",
       is_field_error=True, is_alignment_error=True)

逐元件导入的两个开关均默认为 ``False``；只读取准直表时仅开启准直开关。
``DX``、``DY``、``DPSI`` 来自 **误差表**；名义 Twiss 的 ``DX`` / ``DY`` 是色散，
绝不会作为准直偏移。所选误差种类的缺失分量补零。
非零且不支持的准直分量、不支持的元件上的非零误差均报错。
实例匹配与重复检查和场误差共用，在漂移合并前完成。

``generate_from_tfs`` 同样提供 ``is_alignment_error`` 参数，GUI 逐元件导入提供准直复选框。
``read_madx_twiss`` 和 ``read_madx_twiss_interpolated`` 对此选项明确报错：
Twiss 传输映射不能重建移动后的实际磁场。准直误差应使用逐元件跟踪。
关闭准直导入时，共享误差文件中的准直列被忽略。
