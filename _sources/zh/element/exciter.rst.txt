激励器（Exciter）
====================

本模块介绍 PASS 中的横向激励器元件 **Exciter** ，用于通过时变电场对束流施加横向动量扰动。激励器在 tune 测量、束流不稳定性研究、发射度增长等场景中广泛应用。

PASS 中的激励器为 **薄透镜元件** （ ``length = 0`` ），仅改变粒子横向动量（ :math:`p_x` 或 :math:`p_y` ），不改变位置坐标。

**代码位置**

- 源文件： ``PASS/commands/element/exciter.py``
- 类名： ``Exciter`` （继承自 ``Command`` ）
- 注册名： ``exciter``
- 核心特征：

  - 薄透镜元件（ ``length = 0`` ），仅改变粒子横向动量，不改变位置坐标；
  - 支持 4 种激励模式（ ``single_fm`` 、 ``single_fm_am`` 、 ``dual_fm`` 、 ``dual_fm_am`` ）；
  - 频率参数支持工作点模式和频率模式两种输入方式；
  - 支持孔径检查，与其它元件一致。


物理推导
--------

激励器由一对平行极板组成，极板间施加电压 :math:`V` ，极板间距 :math:`d` ，极板有效长度 :math:`L` 。

电场强度为：

.. math::

  E = \frac{V}{d}

粒子（电荷量 :math:`Q = Z \cdot e` ，其中 :math:`Z` 为电荷数， :math:`e` 为元电荷）受到的力为：

.. math::

  F = Q \cdot E = Z \cdot e \cdot \frac{V}{d}

粒子以速度 :math:`v = \beta c` 穿过极板，作用时间为：

.. math::

  \Delta t = \frac{L}{\beta c}

因此动量增量为：

.. math::

  \Delta P_x = F \cdot \Delta t = \frac{Z \cdot e \cdot V \cdot L}{d \cdot \beta c}

归一化 kick （除以参考粒子总动量 :math:`P_0` ）为：

.. math::

  \Delta p_x = \frac{\Delta P_x}{P_0} = \frac{Z \cdot e \cdot V \cdot L}{d \cdot \beta c \cdot P_0}

利用磁刚度 :math:`B\rho = P_0 / Q` ，可简化为：

.. math::

  \Delta p_x = \frac{V \cdot L}{d \cdot \beta c \cdot B\rho}

该形式对质子束 （ :math:`Z=1, A=1` ）和离子束 （ :math:`Z \neq A` ）统一适用，因为 :math:`B\rho` 已包含荷质比信息。


粒子到达时间
------------

粒子数组中的纵向坐标 :math:`z_{\mathrm{rel}}` 是相对各自束团中心的坐标。对于束团中心位置 :math:`z_{\mathrm{center}}`，粒子的实验室系纵向位置为：

.. math::

  z_{\mathrm{lab}} = z_{\mathrm{rel}} + z_{\mathrm{center}}

因此，粒子到达激励器的时刻为：

.. math::

  t_{\text{arrive}} = t_0 - \frac{z_{\mathrm{lab}}}{\beta c}

其中 :math:`t_0` 是机器坐标原点处参考粒子的时钟， :math:`z_{\mathrm{lab}} > 0` 表示粒子位于该参考点前方，因而更早到达。Exciter 直接使用 :math:`z_{\mathrm{lab}}` 计算信号相位，不会对 :math:`z_{\mathrm{rel}}` 进行折叠或回绕。这一到达时间差异使不同粒子看到不同相位的激励信号，是纵向-横向耦合的来源。

回旋频率为：

.. math::

  f_0 = \frac{\beta c}{C}

其中 :math:`C` 为环周长。回旋频率用于将圈数转换为真实时间。


频率输入模式
------------

激励器的中心频率 :math:`f_c` 和扫频宽度 :math:`\Delta f` 支持两种输入方式：

**工作点模式** （推荐）

直接输入激励工作点 :math:`Q_{\text{excite}}` 和扫频工作点 :math:`\Delta Q` ，程序在运行时根据束流参数自动计算频率：

.. math::

  f_c = Q_{\text{excite}} \cdot f_0

.. math::

  \Delta f = \Delta Q \cdot f_0

此模式下无需手动计算频率，且自动适应不同能量和周长的束流。需成对提供 ``excite tune`` 和 ``sweep tune`` 。

**频率模式**

直接输入中心频率和扫频宽度 （单位 Hz），适用于需要精确控制频率的场景。需成对提供 ``central frequency (hz)`` 和 ``sweep width (hz)`` 。

.. note::

  两种模式二选一。若提供了 ``excite tune`` 则使用工作点模式，否则使用频率模式。工作点模式中 ``excite tune`` 和 ``sweep tune`` 必须成对提供。


激励模式
--------

激励器有 4 种工作模式，由频率调制 （FM）方式和幅度调制 （AM）方式两个维度组合而成：

.. list-table::
  :header-rows: 1
  :widths: 20 15 15 50

  * - 模式
    - FM 方式
    - AM 方式
    - 说明
  * - ``single_fm``
    - 单段扫频
    - 常值幅度
    - 最基本的线性 chirp
  * - ``single_fm_am``
    - 单段扫频
    - 时变幅度
    - 扫频 + 幅度绝热增长
  * - ``dual_fm``
    - 双段扫频
    - 常值幅度
    - 复杂频谱覆盖
  * - ``dual_fm_am``
    - 双段扫频
    - 时变幅度
    - 最复杂的激励模式


频率调制（FM）维度
~~~~~~~~~~~~~~~~~~~~~~

**单段线性扫频（single）**

在一个周期 :math:`T` 内，相位为：

.. math::

  \theta(\tau) = 2\pi f_c \cdot \tau + \frac{\pi \Delta f}{T} \cdot \tau (\tau - T)

其中 :math:`\tau = t \bmod T` 为周期内时间， :math:`f_c` 为中心频率， :math:`\Delta f` 为扫频宽度。

瞬时频率为：

.. math::

  f(t) = f_c + \frac{\Delta f}{T}\left(\tau - \frac{T}{2}\right)

- 当 :math:`\tau = 0` 时， :math:`f = f_c - \Delta f / 2` （起始频率）
- 当 :math:`\tau = T/2` 时， :math:`f = f_c` （中心频率）
- 当 :math:`\tau = T` 时， :math:`f = f_c + \Delta f / 2` （终止频率）

频率在 :math:`[f_c - \Delta f/2,\; f_c + \Delta f/2]` 范围内线性扫描，每 :math:`T` 秒重复一次。中心频率 :math:`f_c` 应接近 :math:`Q \cdot f_0` （工作点乘回旋频率），以覆盖束流的共振频率。

**双段扫频（dual）**

一个周期分为前后两半，各使用不同的相位公式，同时引入余弦包络 :math:`2\cos(\frac{\pi}{2}\Delta f \cdot \tau)` ：

前半周期 :math:`[0,\; T/2]` ：

.. math::

  \theta_1(\tau) = 2\pi f_c \cdot \tau + \pi \Delta f \cdot (f_d \cdot \tau - 0.5) \cdot \tau

后半周期 :math:`[T/2,\; T]` ：

.. math::

  \theta_2(\tau) = 2\pi f_c \cdot \tau + \pi \Delta f \cdot (\tau - T/2) \cdot (f_d \cdot \tau - 1.0)

其中 :math:`f_d` 为双频频率参数。余弦包络在 :math:`\tau = 0` 时最大 （ :math:`2A` ），随时间衰减，减少周期边界处的不连续性。双段相位公式产生更复杂的频谱结构，可同时覆盖多个 tune 峰。


幅度调制（AM）维度
~~~~~~~~~~~~~~~~~~~~~~

**常值幅度**

.. math::

  A(t) = A_0 = \Delta p_{x,\text{amplitude}}

即直接使用由电压参数计算的 kick 幅度，不随时间变化。

**时变幅度（am）**

基于束流扩散/增长模型，激励幅度随时间增长：

.. math::

  A(t) = A_0 \cdot \text{am\_factor}(t)

其中 :math:`\text{am\_factor}(t)` 是无量纲的时变缩放因子：

.. math::

  \text{am\_factor}(t) = \sqrt{\frac{\delta^2(t)}{f_0 \cdot k_{\text{const}}}}

其中 :math:`t = n_{\text{eff}} / f_0` 为从激励开始的真实时间 （秒）， :math:`n_{\text{eff}}` 为有效激励圈数。

初始发射度占比：

.. math::

  \varepsilon = \exp\!\left(-\frac{r_0^2}{\delta_0^2}\right)

时变发射度平方：

.. math::

  \delta^2(t) = \frac{r_0^2 (1 - \varepsilon)}{L^2 \cdot D}

其中：

.. math::

  L = \ln\!\left(\frac{t}{t_{\text{ext}}}(1 - \varepsilon) + \varepsilon\right)

.. math::

  D = t_{\text{ext}} \cdot \varepsilon + t (1 - \varepsilon)

物理意义：

- :math:`r_0` ：初始束流尺寸
- :math:`\delta_0` ：初始束流扩散范围
- :math:`t_{\text{ext}}` ：束流扩散特征时间
- :math:`k_{\text{const}}` ：发射度增长系数
- :math:`\varepsilon` ：初始发射度占比 （ :math:`r_0 / \delta_0` 比值的度量）

激励器持续给束流注入能量，束流振荡幅度增大，发射度增长，需要更大的激励幅度维持相对驱动效果。对数项使得增长开始快 （陡峭段），后期减缓 （平缓段），符合绝热增长过程的物理特征。


各模式完整公式
--------------

1. **single_fm** （单段扫频 + 常值幅度）

.. math::

  \text{kick}(\tau) = A_0 \cdot \sin\!\left(2\pi f_c \cdot \tau + \frac{\pi \Delta f}{T} \cdot \tau (\tau - T)\right)

2. **single_fm_am** （单段扫频 + 时变幅度）

.. math::

  \text{kick}(\tau) = A_0 \cdot \text{am\_factor}(t) \cdot \sin\!\left(2\pi f_c \cdot \tau + \frac{\pi \Delta f}{T} \cdot \tau (\tau - T)\right)

3. **dual_fm** （双段扫频 + 常值幅度）

前半周期 （ :math:`0 \le \tau \le T/2` ）：

.. math::

  \text{kick} = 2 A_0 \cos\!\left(\frac{\pi}{2} \Delta f \cdot \tau\right) \sin\!\left(2\pi f_c \cdot \tau + \pi \Delta f (f_d \cdot \tau - 0.5) \tau\right)

后半周期 （ :math:`T/2 < \tau \le T` ）：

.. math::

  \text{kick} = 2 A_0 \cos\!\left(\frac{\pi}{2} \Delta f \cdot \tau\right) \sin\!\left(2\pi f_c \cdot \tau + \pi \Delta f (\tau - T/2)(f_d \cdot \tau - 1.0)\right)

4. **dual_fm_am** （双段扫频 + 时变幅度）

前半周期 （ :math:`0 \le \tau \le T/2` ）：

.. math::

  \text{kick} = 2 A_0 \cdot \text{am\_factor}(t) \cos\!\left(\frac{\pi}{2} \Delta f \cdot \tau\right) \sin\!\left(2\pi f_c \cdot \tau + \pi \Delta f (f_d \cdot \tau - 0.5) \tau\right)

后半周期 （ :math:`T/2 < \tau \le T` ）：

.. math::

  \text{kick} = 2 A_0 \cdot \text{am\_factor}(t) \cos\!\left(\frac{\pi}{2} \Delta f \cdot \tau\right) \sin\!\left(2\pi f_c \cdot \tau + \pi \Delta f (\tau - T/2)(f_d \cdot \tau - 1.0)\right)

其中 :math:`\tau = t \bmod T` ， :math:`A_0 = \frac{V \cdot L}{d \cdot \beta c \cdot B\rho}` 。


Kick 的施加
-----------

激励器是薄透镜元件，kick 直接加到对应方向的归一化动量上：

.. math::

  p_x \leftarrow p_x + \text{kick} \quad (\text{direction} = x)

.. math::

  p_y \leftarrow p_y + \text{kick} \quad (\text{direction} = y)

仅对存活粒子 （ ``tag > 0`` ）施加 kick，已丢失粒子不受影响。

kick 施加后，激励器会根据孔径参数 （ ``aperture_type`` ）对粒子进行孔径检查：若孔径类型不为 ``off`` ，则超出孔径范围的粒子将被标记为丢失 （ ``tag`` 置为负值）；若孔径类型为 ``off`` ，则不进行孔径检查。


参数列表
--------

通用参数
~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 35

  * - 属性名
    - JSON key
    - 类型
    - 单位
    - 说明
  * - ``s``
    - ``s (m)``
    - float
    - m
    - 元件在束线中的纵向位置
  * - ``length``
    - ``length (m)``
    - float
    - m
    - 元件长度 （必须为 0）
  * - ``name``
    - ``name``
    - str
    - -
    - 元件名称
  * - ``is_enabled``
    - ``enable``
    - bool
    - -
    - 激励器开关，可选： ``true`` 、 ``false``
  * - ``mode``
    - ``mode``
    - str
    - -
    - 激励模式，可选： ``single_fm`` 、 ``single_fm_am`` 、 ``dual_fm`` 、 ``dual_fm_am``
  * - ``direction``
    - ``direction``
    - str
    - -
    - 激励方向，可选： ``x`` 、 ``y``
  * - ``start_turn``
    - ``start turn``
    - int
    - -
    - 激励起始圈数 （含）
  * - ``end_turn``
    - ``end turn``
    - int
    - -
    - 激励结束圈数 （不含）
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

硬件参数
~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 35

  * - 属性名
    - JSON key
    - 类型
    - 单位
    - 说明
  * - ``voltage``
    - ``voltage (v)``
    - float
    - V
    - 极板峰值电压
  * - ``gap``
    - ``gap (m)``
    - float
    - m
    - 极板间距
  * - ``plate_length``
    - ``plate length (m)``
    - float
    - m
    - 极板有效长度

频率参数
~~~~~~~~~~

频率参数支持两种输入模式，二选一。

**工作点模式** （推荐）：

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 35

  * - 属性名
    - JSON key
    - 类型
    - 单位
    - 说明
  * - ``excite_tune``
    - ``excite tune``
    - float
    - -
    - 激励工作点 :math:`Q_{\text{excite}}` ，运行时自动计算 :math:`f_c = Q_{\text{excite}} \cdot f_0`
  * - ``sweep_tune``
    - ``sweep tune``
    - float
    - -
    - 扫频工作点 :math:`\Delta Q` ，运行时自动计算 :math:`\Delta f = \Delta Q \cdot f_0`

**频率模式** ：

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 35

  * - 属性名
    - JSON key
    - 类型
    - 单位
    - 说明
  * - ``cf``
    - ``central frequency (hz)``
    - float
    - Hz
    - 中心频率 :math:`f_c`
  * - ``cfw``
    - ``sweep width (hz)``
    - float
    - Hz
    - 扫频宽度 :math:`\Delta f`

**通用频率参数** （两种模式均需提供）：

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 20 15

  * - 属性名
    - JSON key
    - 类型
    - 单位
    - 适用模式
    - 说明
  * - ``period``
    - ``period (s)``
    - float
    - s
    - 所有模式
    - 扫频周期 :math:`T`
  * - ``fm_dual_frequency``
    - ``fm dual frequency (hz)``
    - float
    - Hz
    - dual_fm / dual_fm_am
    - 双频频率参数 :math:`f_d`

幅度调制（AM）参数
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 20 20 10 10 25 15

  * - 属性名
    - JSON key
    - 类型
    - 单位
    - 适用模式
    - 说明
  * - ``am_t_ext``
    - ``am t ext (s)``
    - float
    - s
    - single_fm_am / dual_fm_am
    - 束流扩散特征时间
  * - ``am_r0``
    - ``am r0 (m)``
    - float
    - m
    - single_fm_am / dual_fm_am
    - 初始束流尺寸
  * - ``am_delta0``
    - ``am delta0``
    - float
    - -
    - single_fm_am / dual_fm_am
    - 初始束流扩散范围
  * - ``am_k_const``
    - ``am k const``
    - float
    - -
    - single_fm_am / dual_fm_am
    - 发射度增长系数

.. note::

  ``am_r0`` 与 ``am_delta0`` 应为同量级，否则 :math:`\exp(-r_0^2/\delta_0^2)` 可能数值下溢。

  常值幅度模式 （ ``single_fm`` 、 ``dual_fm`` ）下 AM 参数不参与计算，可填 0。


使用示例
--------

输入文件示例
~~~~~~~~~~~~~~~~

以下示例取自 ``input/beam0.json`` ，使用工作点模式：

.. code-block:: json

  {
      "Exciter_x": {
          "S (m)": 0.0,
          "Command": "Exciter",
          "Length (m)": 0.0,
          "Enable": false,
          "Mode": "single_fm",
          "Direction": "x",
          "Start Turn": 100,
          "End Turn": 1000,
          "Voltage (V)": 1000.0,
          "Gap (m)": 0.1,
          "Plate length (m)": 0.3,
          "Excite tune": 0.44,
          "Sweep tune": 0.02,
          "Period (s)": 1e-3,
          "Fm Dual Frequency (Hz)": 0.0,
          "Am t ext (s)": 0.0,
          "Am r0 (m)": 0.0,
          "Am delta0": 0.0,
          "Am k const": 0.0,
          "Aperture Type": "off"
      }
  }

若使用频率模式，将 ``Excite tune`` 和 ``Sweep tune`` 替换为：

.. code-block:: json

  "Central Frequency (Hz)": 1743.0,
  "Sweep Width (Hz)": 79.2,

模式选择指南
~~~~~~~~~~~~~~~~

- **tune 测量** ：推荐 ``single_fm`` ，简单有效，扫频覆盖工作点
- **发射度增长研究** ：推荐 ``single_fm_am`` ，时变幅度模拟绝热增长
- **多 tune 峰覆盖** ：推荐 ``dual_fm`` ，双段扫频产生复杂频谱
- **复杂不稳定性研究** ：推荐 ``dual_fm_am`` ，最完整的激励模式

参数选择建议
~~~~~~~~~~~~~~~~

- **激励工作点** ：设为束流工作点 :math:`Q_x` （水平）或 :math:`Q_y` （垂直）
- **扫频工作点** ：取决于色散和 tune 展宽，通常为 0.01~0.05
- **扫频周期** ：应远大于回旋周期 :math:`1/f_0` ，保证足够的频率分辨率
- **电压** ：根据所需 kick 幅度反推，典型值为百伏至千伏量级
- **AM 参数** ： :math:`r_0` 与 :math:`\delta_0` 取同量级， :math:`t_{\text{ext}}` 根据束流扩散时间尺度设定
