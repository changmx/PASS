漂移节（Drift）
==================

本模块介绍 PASS 中的漂移节元件 **Drift** ，用于模拟粒子在无场自由空间中的传输。漂移节是最基本的束线元件，粒子在其中不受任何电磁力作用，仅凭借初始动量做直线运动。

**代码位置**

- 源文件： ``PASS/commands/element/drift.py``
- 类名： ``Drift`` （继承自 ``Command`` ）
- 注册名： ``drift``
- 核心特征：

  - 厚元件（ ``length > 0`` ），改变粒子的位置和纵向坐标；
  - 使用精确的几何传输公式，考虑横向动量对纵向速度的投影；
  - 支持孔径检查，与其它元件一致。


物理推导
--------

四极铁、六极铁、八极铁、一般多极铁、弯铁、踢轨器及螺线管内部的直线漂移阶段，
在 CPU 和 GPU 上也使用下面的有理化纵向更新式。令
:math:`g=\gamma_0^{-2}`、:math:`q_\perp=p_x^2+p_y^2`，以及
:math:`R=\sqrt{g+(1-g)(1+\delta)^2}`，则

.. math::

   \Delta z=L\frac{\delta(2+\delta)g-q_\perp}{p_z(p_z+R)}.

这避免了高能下两个相近数相减，并使轴线上、动量无偏差的参考粒子得到严格为零的滑移。
螺线管精确映射使用同一表达式，其中横向机械动量为
:math:`p_x+k_s y/2` 和 :math:`p_y-k_s x/2`。
该公式不能消除增量写入粒子存储坐标时的舍入误差。

粒子在漂移节中不受力，以恒定动量做直线运动。设漂移节长度为 :math:`L` ，粒子归一化横向动量为 :math:`p_x` 、 :math:`p_y` ，动量偏差为 :math:`\delta` 。

**粒子总动量**

归一化总动量（以参考粒子动量 :math:`P_0` 为单位）为：

.. math::

  P_{\text{tot}} = 1 + \delta

纵向动量分量（考虑横向动量的投影）为：

.. math::

  p_z = \sqrt{(1 + \delta)^2 - p_x^2 - p_y^2}

向前传播要求 :math:`1+\delta>0`，且 :math:`p_z^2` 有限并严格大于零。
不满足这些条件的存活粒子标记为丢失；已丢失粒子的坐标和首次损失记录保持不变。

**粒子速度**

粒子 :math:`\beta` 值与参考粒子 :math:`\beta_0` 、 :math:`\gamma_0` 和动量偏差 :math:`\delta` 的关系为：

.. math::

  \beta = \frac{(1 + \delta) \, \gamma_0 \, \beta_0}{\sqrt{1 + \left[(1 + \delta) \, \gamma_0 \, \beta_0\right]^2}}

**坐标更新**

漂移节中粒子坐标更新为：

.. math::

  x \leftarrow x + L \cdot \frac{p_x}{p_z}

.. math::

  y \leftarrow y + L \cdot \frac{p_y}{p_z}

.. math::

  z \leftarrow z + L \cdot \left(1 - \frac{\beta_0}{\beta} \cdot \frac{1 + \delta}{p_z}\right)

上面的纵向更新式是原实现直接计算的公式。它同时包含动量偏差造成的速度差，
以及横向运动造成的实际飞行路程增加。

从飞行时间推导原公式
~~~~~~~~~~~~~~~~~~~~

这里 :math:`p_x=P_x/P_0`、:math:`p_y=P_y/P_0`，
:math:`\delta=(P-P_0)/P_0`。上面的 :math:`p_z` 是归一化纵向动量，
而保存的 ``p.z`` 是连续的时间坐标：

.. math::

  z=\beta_0c(t_0-t_i).

推导假定粒子向前运动，满足 :math:`1+\delta>0` 和 :math:`p_z>0`。
无场漂移中，参考速度和各粒子的动量保持不变。参考粒子的飞行时间和实际粒子的
纵向速度分别为：

.. math::

  \Delta t_0=\frac{L}{\beta_0c},\qquad
  v_s=\beta c\frac{p_z}{1+\delta},\qquad
  \Delta t_i=\frac{L}{v_s}.

因此：

.. math::

  \Delta z=\beta_0c(\Delta t_0-\Delta t_i)
  =L\left(1-\frac{\beta_0}{\beta}\frac{1+\delta}{p_z}\right).

参考事件的时间推进 :math:`\Delta t_0`；正的 :math:`\Delta z` 表示粒子相对于
参考粒子获得了到达时间上的提前量。

有理化得到当前公式
~~~~~~~~~~~~~~~~~~

令 :math:`a=\gamma_0^{-2}`、:math:`u=1+\delta`、
:math:`q_\perp=p_x^2+p_y^2`。利用 :math:`\beta_0^2=1-a` 和相对论能量—动量关系：

.. math::

  A\equiv\frac{\beta_0}{\beta}u
  =\frac{\sqrt{1+(\gamma_0\beta_0u)^2}}{\gamma_0}
  =\sqrt{a+(1-a)u^2}=\frac{E_i}{E_0}.

其中 :math:`E_i` 和 :math:`E_0` 是静止质量相同的粒子的总能量。
当前 ``drift_factors`` 中的 ``energy`` 对应 :math:`A`。
由 :math:`p_z^2=u^2-q_\perp`，将差值的分子有理化：

.. math::

  \frac{\Delta z}{L}
  =\frac{p_z-A}{p_z}
  =\frac{p_z^2-A^2}{p_z(p_z+A)},

.. math::

  p_z^2-A^2
  =u^2-q_\perp-\left[a+(1-a)u^2\right]
  =a(u^2-1)-q_\perp
  =\frac{\delta(2+\delta)}{\gamma_0^2}-p_x^2-p_y^2.

CPU 和 GPU 当前均计算：

.. math::

  \Delta z = L\frac{\delta(2+\delta)/\gamma_0^2-p_x^2-p_y^2}
  {p_z\left[p_z+\sqrt{\gamma_0^{-2}+(1-\gamma_0^{-2})(1+\delta)^2}\right]}.

这是代数恒等变换，没有对动量偏差或横向角度作截断展开，保留了原来的精确几何映射。
带内部空间电荷的 GPU 漂移分段也使用同一个内核。

修改目的与数值限制
~~~~~~~~~~~~~~~~~~

滑移很小时，原表达式需要将两个接近 1 的数相减，容易损失有效数字。
FP32 在 1 右侧的相邻浮点数间隔约为 :math:`1.19\times10^{-7}`，
所以真实值为 :math:`10^{-8}` 或 :math:`10^{-9}` 量级的滑移因子，在直接计算时
可能产生很大的相对误差，甚至被舍入为零。

当前表达式直接构造小量分子。尤其是 ``dp * (2 + dp)``，即使 ``1 + dp``
被舍入为 1，仍能保留很小的动量偏差；若写成 ``(1 + dp)**2 - 1``，则会再次引入
相近数相减的问题。

仅为理解物理含义，保留动量偏差与横向运动的主导项可得：

.. math::

  \frac{\Delta z}{L}\simeq
  \frac{\delta}{\gamma_0^2}-\frac{p_x^2+p_y^2}{2}.

因此，轴上粒子的正动量偏差使其提前到达；总动量不变时，横向运动使其推迟到达；
理想参考粒子的滑移为零。跟踪实际计算的是前面的完整有理化公式，而非这个主导项表达式。

例如 :math:`L=1\,\mathrm{m}`、:math:`\gamma_0=2`、:math:`\delta=10^{-8}`，
且 :math:`p_x=p_y=0` 时，精确滑移约为 :math:`2.5\times10^{-9}\,\mathrm{m}`。
初始 z 为零时，有理化公式在 FP32 中能够保留这一增量。六维粒子坐标仍统一使用所选的
FP32 或 FP64：若将同一增量加到已有的 FP32 :math:`z=1\,\mathrm{m}` 上，结果仍会舍入为 1。
因此，本次修改改善增量的计算精度，但不能消除反复累加中的舍入误差，也不能消除两个
物理贡献几乎抵消时的相消误差。

精确推导还假定计算 :math:`p_z` 时没有设置人为下限。旧 CPU 实现在判断丢失后
使用 :math:`p_z=\sqrt{\max(p_z^2,10^{-10})}`，会改变极小正纵向动量粒子的
飞行时间，且与 GPU 不一致。当前 CPU 和 GPU 对有效粒子均直接计算
:math:`p_z=\sqrt{p_z^2}`，不再夹紧正值。CPU 仅为无效行设置安全的计算占位值，
这些行不参与坐标更新；这也避免了用零掩码乘 NaN 时污染已丢失粒子的记录。

CPU 的 ``drift_factors`` 与 CUDA 的 ``pass_drift_factors`` 集中实现同一组
前向条件和稳定公式，供 Drift 及有限长度静电元件的直线传输复用。
这是计算逻辑的整理，未增加新的漂移力或动量踢。

计算开销
~~~~~~~~

两种写法对 :math:`N` 个粒子的计算量均为 :math:`O(N)`。CPU 新写法去掉了显式的
粒子速度数组及其除法，同时增加了有理化分子和分母的运算；两者每粒子仍需计算两次平方根。
NumPy 临时数组和内存读写也会影响耗时，因此不能仅凭表达式长短判断是否提速。

GPU 继续在同一个融合内核中完成计算，每粒子仍计算两次平方根。相对于原内核，
新分母增加了一次浮点除法，并增加了分子的算术运算。坐标存储、全局数组访问和内核启动次数
保持不变。运算吞吐受限时，耗时可能增加，FP64 尤其需要关注；小批量可能主要受启动开销
影响，大批量也可能受显存带宽限制。本次修改的目的是可靠地计算小滑移，具体耗时影响应按
硬件、精度、粒子数和分段配置实测。

GPU 计时应排除首次编译，并通过同步或 CUDA 事件处理异步执行，参见
`CuPy 性能说明 <https://docs.cupy.dev/en/stable/user_guide/performance.html>`_。
稳态内核计时本身不能代表包含空间电荷、监测器和数据传输的完整模拟耗时。

纵向坐标连续性
~~~~~~~~~~~~~~

连续保存 z，不折叠多圈累计滑移。物理到达时间为 :math:`t_i=T_b-z_i/(\beta_b c)`，必须使用当前参考事件；名义槽位偏移不参与时间重建。


接口参数
--------

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 35

  * - 属性名
    - JSON key
    - 类型
    - 单位
    - 说明
  * - ``s``
    - ``S (m)``
    - float
    - m
    - 元件在束线中的纵向位置
  * - ``length``
    - ``Length (m)``
    - float
    - m
    - 元件长度 （必须 :math:`\ge 0` ）
  * - ``name``
    - ``name``
    - str
    - -
    - 元件名称 （由序列 JSON 的键名自动填入）
  * - ``aperture_type``
    - ``Aperture Type``
    - str
    - -
    - 孔径类型 （默认 ``off`` ，可选值见孔径章节）
  * - ``aperture_value``
    - ``Aperture Value``
    - list
    - -
    - 孔径参数值 （默认 ``[]`` ，含义随类型而异，详见孔径章节）


使用示例
--------

以下 JSON 片段展示了漂移节的配置方式：

**基本用法** ：

.. code-block:: json

  "Drift1": {
      "S (m)": 10.0,
      "Command": "Drift",
      "Length (m)": 0.5,
      "Aperture Type": "off"
  }

**带圆形孔径检查** ：

.. code-block:: json

  "Drift2": {
      "S (m)": 10.5,
      "Command": "Drift",
      "Length (m)": 0.3,
      "Aperture Type": "circle",
      "Aperture Value": [0.05]
  }

**带矩形孔径检查** ：

.. code-block:: json

  "Drift3": {
      "S (m)": 11.0,
      "Command": "Drift",
      "Length (m)": 0.2,
      "Aperture Type": "rectangle",
      "Aperture Value": [0.06, 0.04]
  }


应用场景
--------

- **束线连接** ：在各磁铁元件之间提供自由漂移空间，是最常用的束线元件
- **色散测量** ：在偏转磁铁后设置漂移段，利用色散效应测量束流动量分散
- **束流传输** ：在注入线和引出线中传输束流，不施加任何场
- **孔径检查** ：在关键位置设置带孔径检查的漂移节，监控束流损失

元件内部空间电荷
----------------

正长度元件可设置 ``space_charge``（JSON ``Space charge``）为
``ElementSpaceCharge`` 对象。``Num slices`` 控制外场传输，
``Space charge.Num kicks`` 控制 SC 积分。调度规则、共享资源、
支持的后端和示例见 :ref:`zh-internal-space-charge`。

``num_slices``（JSON ``Num slices``）为正整数，默认 1。
不启用内部 SC 时，CPU 与 GPU 均使用该数量的本体切片。

共享数值漂移因子
----------------

Drift 与有限长度 ElSeparator 共享 CPU 漂移因子和 CUDA 内联映射。
纵向滑移采用有理化表达式，保留 FP32 的微小增量。
非正总动量（delta <= -1）、非正纵向动量及非有限纵向动量判损，
不再把正的纵向动量截断到任意 epsilon。已有损失记录保持不变。
共享映射不改变 Drift 的出口/节点孔径取样；ElSeparator 额外求解沿直线子段的管壁交点。
