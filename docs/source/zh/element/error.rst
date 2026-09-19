.. _zh-error:

元件误差
========

本页统一介绍元件误差模型及其配置。目前已实现绝对磁场误差，准直误差尚未实现。

场误差
------

SBend、Quadrupole、Sextupole、Octupole、Multipole、Solenoid 和 Kicker
在 CPU、GPU 上均支持相同的附加正、斜多极积分场误差。误差系数在追踪期间保持固定。
本接口只支持绝对磁场误差；相对误差生成、准直误差、孔径偏移和 BPM 读数误差不在本次范围内。

参数与归一化
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 15 35

   * - Python schema 字段
     - JSON 键
     - 默认值
     - 含义
   * - ``is_field_error``
     - ``Is field error``
     - ``false``
     - 启用附加场误差踢角。
   * - ``field_error_knl``
     - ``Field error KNL``
     - ``[]``
     - 正多极积分误差，从第 0 阶偶极项开始。
   * - ``field_error_ksl``
     - ``Field error KSL``
     - ``[]``
     - 斜多极积分误差，从第 0 阶偶极项开始。

数组第 :math:`n` 项单位为 :math:`\mathrm{m}^{-n}`；第 0、1、2、3 项依次为偶极、四极、六极、八极。
系数必须有限。缺失阶数补零，较短数组补齐；全零误差无效应，不按幅值阈值丢弃有限小量。
开关关闭时，即使数组非零也保持名义映射。

在 :math:`p_x=P_x/P_0`、:math:`p_y=P_y/P_0` 约定下，完整积分误差踢角为

.. math::

   F(x,y) = \sum_{n=0}^{N}
   \frac{\Delta K_nL+i\Delta K_{ns}L}{n!}(x+iy)^n,
   \qquad \Delta p_x=-\Re F,\quad \Delta p_y=\Im F.

输入系数不含阶乘；归一化动量踢角也不额外乘 :math:`1/(1+\delta)`。
因此正偶极误差为正时减小 ``px``，与正 ``HKICK`` 符号相反；
斜偶极误差为正时增大 ``py``，与正 ``VKICK`` 一致。

Multipole 和 Solenoid 的 ``KiL`` / ``KiSL`` 表示名义横向多极分量，启用的误差只与它们相加一次。
Solenoid 的 ``KS`` 是独立的纵向场强 :math:`B_z/(B\rho)`，不是斜多极数组。
螺线管纵向主场变化应通过 ``KS`` 配置。

追踪模型
~~~~~~~~

薄元件执行一次完整积分误差踢角。厚元件将误差沿长度均匀分布：
每个带符号积分步 :math:`\Delta s` 使用 :math:`\Delta s/L` 的积分误差。
Yoshida 子步保留负权重，因此增加切片不会增加总误差强度。

保留原有名义传输。DKD/RKR/SKS 将误差加在中心踢角处；
四极矩阵模型采用“半段名义矩阵、误差踢角、半段名义矩阵”。
内部空间电荷仍在计划节点执行，使用自身的正积分权重。

代码组织
^^^^^^^^

``PASS/commands/element/error.py`` 保留 ``FieldErrors``、GPU 误差调度
``_track_field_errors_gpu`` 和矩阵误差组合 ``_transport_matrix_errors``。
``PASS/utils/slicing.py`` 中的 ``execute_element_body_gpu`` 负责本体分片传输，
只在已配置的节点调用 SC；没有 SC 时，它也用于矩阵模型和弯铁的误差追踪。
矩阵、弯铁的本体传输与误差踢角仍使用独立的 GPU 调用。

固定的多极系数、阶乘倒数及子步参数准备一次后复用，GPU 数组按精度和设备缓存。
粒子坐标和束团参考量仍从当前状态读取。
输入开关 ``Is field error`` 保留；运行状态由 ``FieldErrors.enabled`` 和
``FieldErrors.active`` 管理，元件不再保存重复的开关变量。模块改名不代表已实现准直误差。

模型限制
^^^^^^^^

SBend 保留名义曲率及入、出口映射，在弯曲名义传输之间加入局部直线多极踢角。
这是明确的薄误差近似，不是完整的曲线多极场或误差相关端场模型，
因而不保证与 PTC 原生厚弯铁误差模型完全一致。
PTC 某些原生元件命令也可能只处理部分误差阶数；验证本模型时应同时比较等价的显式薄 Multipole 误差格架。

Solenoid 长度为零时纵向场映射无效应，但横向积分多极分量和启用的场误差仍执行薄踢角。

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
~~~~~~~~~~~

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
准直、孔径和监测器误差列不参与导入。

逐元件导入将误差附加到物理磁铁，不支持的元件出现非零场误差时直接报错。
Twiss 传输导入在原始元件出口 ``S`` 处插入一个薄 Multipole，在 Twiss 映射之后执行；
合并和重采样保留这些必要位置。这种集中误差与逐元件追踪中的分布误差是不同近似。

螺线管逐元件导入从 MAD-X 标准 Twiss 的 ``KSI`` 列计算 ``KS = KSI/L``；
若没有 ``KSI``，也接受自定义 ``KS`` 列。两列均缺失，或零长度时 ``KSI`` 非零，均报错。
