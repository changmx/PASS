激励器（Exciter）
========================

本模块介绍 PASS 中的横向激励器元件 **Exciter** ，用于通过规定的横向踢角波形对束流施加动量扰动。激励器在 工作点 测量、束流不稳定性研究、发射度增长等场景中广泛应用。

PASS 中的激励器为 **薄透镜元件** （ ``length = 0`` ），仅改变粒子横向动量（ :math:`p_x` 或 :math:`p_y` ），不改变位置坐标。
长度在内部固定，不是输入参数。

- 核心特征：

  - 薄透镜元件（ ``length = 0`` ），仅改变粒子横向动量，不改变位置坐标；
  - 支持 4 种激励模式（ ``single_fm`` 、 ``single_fm_am`` 、 ``dual_fm`` 、 ``dual_fm_am`` ）；
  - 频率参数支持工作点模式和频率模式两种输入方式；
  - 支持孔径检查，与其它元件一致。

以下接口中的字段用于 ``PASS.para.schema.elements.ExciterItem`` 配置类；
元件名称由 ``Sequence.add(name, item)`` 的序列键给出，不是配置类字段。

参数列表
--------

通用参数
~~~~~~~~~~

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
   * - ``is_enabled``
     - ``Enable``
     - ``bool``
     - —
     - ``True``
     - 激励器开关，可选： ``true`` 、 ``false``
   * - ``mode``
     - ``Mode``
     - ``str``
     - -
     - ``必填``
     - 激励模式，可选： ``single_fm`` 、 ``single_fm_am`` 、 ``dual_fm`` 、 ``dual_fm_am``
   * - ``direction``
     - ``Direction``
     - ``str``
     - -
     - ``必填``
     - 激励方向，可选： ``x`` 、 ``y``
   * - ``start_turn``
     - ``Start turn``
     - ``int``
     - -
     - ``必填``
     - 激励起始圈数 （含）
   * - ``end_turn``
     - ``End turn``
     - ``int``
     - -
     - ``必填``
     - 激励结束圈数 （不含）
   * - ``aperture_type``
     - ``Aperture type``
     - ``str``
     - —
     - ``'off'``
     - 孔径类型 （默认 ``off`` ，可选值见孔径章节）
   * - ``aperture_value``
     - ``Aperture value``
     - ``list``
     - m / rad
     - ``[]``
     - 孔径参数值 （默认 ``[]`` ，含义随类型而异，详见孔径章节）


踢角幅度
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python 配置字段
     - JSON 键
     - 类型
     - 单位
     - 默认值
     - 说明
   * - ``kick_angle``
     - ``Kick angle (rad)``
     - ``float``
     - rad
     - ``必填``
     - 两个 DDS 信号共用的有限带符号名义踢角幅度，作为归一化横向动量增量施加；双频将两路信号相加而不除以二


频率参数
~~~~~~~~~~

频率参数支持两种输入模式，二选一。

**工作点模式** （推荐）：

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python 配置字段
     - JSON 键
     - 类型
     - 单位
     - 默认值
     - 说明
   * - ``excite_tune``
     - ``Excite tune``
     - ``float | None``
     - -
     - ``None``
     - 激励工作点 :math:`Q_{\text{excite}}` ，使用共享规定参考钟计算 :math:`f_c(t) = Q_{\text{excite}} f_0(t)`
   * - ``sweep_tune``
     - ``Sweep tune``
     - ``float | None``
     - -
     - ``None``
     - 归一化扫频宽度 :math:`\Delta Q` ，使用共享规定参考钟计算 :math:`\Delta f(t) = \Delta Q f_0(t)`


**频率模式** ：

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python 配置字段
     - JSON 键
     - 类型
     - 单位
     - 默认值
     - 说明
   * - ``central_frequency``
     - ``Central frequency (Hz)``
     - ``float | None``
     - Hz
     - ``None``
     - 中心频率 :math:`f_c`
   * - ``sweep_width``
     - ``Sweep width (Hz)``
     - ``float | None``
     - Hz
     - ``None``
     - 扫频宽度 :math:`\Delta f`


**通用频率参数** ：

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python 配置字段
     - JSON 键
     - 类型
     - 单位
     - 默认值
     - 说明
   * - ``period``
     - ``Period (s)``
     - ``float``
     - s
     - ``必填``
     - 扫频周期 :math:`T` 适用模式：所有模式.
   * - ``dual_sweep_offset``
     - ``Dual sweep offset``
     - ``float``
     - T 的比例
     - ``0.5``
     - DDS1 相对 DDS2 的扫频进度提前量，范围 [0, 1]。非 0.5 时警告并按填写值计算；不是正弦载波相位差。
   * - ``fm_dual_frequency``
     - ``FM dual frequency (Hz)``
     - ``float | None``
     - Hz
     - ``None``
     - 已停用，仅接受旧输入兼容，不参与计算；双频模式下填写值与 :math:`1/T` 不一致时警告。


幅度调制（AM）参数
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python 配置字段
     - JSON 键
     - 类型
     - 单位
     - 默认值
     - 说明
   * - ``am_t_ext``
     - ``AM t ext (s)``
     - ``float``
     - s
     - ``必填``
     - 束流扩散特征时间 适用模式：single_fm_am / dual_fm_am.
   * - ``am_r0``
     - ``AM r0 (m)``
     - ``float``
     - m
     - ``必填``
     - 初始束流尺寸 适用模式：single_fm_am / dual_fm_am.
   * - ``am_delta0``
     - ``AM delta0``
     - ``float``
     - m
     - ``必填``
     - 初始束流扩散范围 适用模式：single_fm_am / dual_fm_am.
   * - ``am_k_const``
     - ``AM k const``
     - ``float``
     - :math:`\mathrm{m}^2`
     - ``必填``
     - 模型归一化系数 适用模式：single_fm_am / dual_fm_am.


.. note::

  ``am_r0`` 与 ``am_delta0`` 应为同量级，否则 :math:`\exp(-r_0^2/\delta_0^2)` 可能数值下溢。

  常值幅度模式 （ ``single_fm`` 、 ``dual_fm`` ）下 AM 参数不参与计算，可填 0。

使用示例
--------

输入文件示例
~~~~~~~~~~~~~~~~

以下示例使用工作点模式：

.. code-block:: json

   {
       "Exciter_x": {
           "S (m)": 0.0,
           "Command": "Exciter",
           "Enable": false,
           "Mode": "single_fm",
           "Direction": "x",
           "Start turn": 100,
           "End turn": 1000,
           "Kick angle (rad)": 1e-4,
           "Excite tune": 0.44,
           "Sweep tune": 0.02,
           "Period (s)": 1e-3,
           "Dual sweep offset": 0.5,
           "AM t ext (s)": 0.0,
           "AM r0 (m)": 0.0,
           "AM delta0": 0.0,
           "AM k const": 0.0,
           "Aperture type": "off"
       }
   }

若使用频率模式，将 ``Excite tune`` 和 ``Sweep tune`` 替换为：

.. code-block:: json

   {
   "Central frequency (Hz)": 1743.0,
   "Sweep width (Hz)": 79.2
   }

模式选择指南
~~~~~~~~~~~~~~~~

- **tune 测量** ：推荐 ``single_fm`` ，简单有效，扫频覆盖工作点
- **发射度增长研究** ： ``single_fm_am`` 提供随时间变化的激励幅度
- **两个 DDS 通道** ： ``dual_fm`` 将两路各自连续积累相位的扫频信号相加，默认错开半个扫频周期
- **复杂不稳定性研究** ：推荐 ``dual_fm_am`` ，最完整的激励模式

参数选择建议
~~~~~~~~~~~~~~~~

- **激励工作点** ：设为束流工作点 :math:`Q_x` （水平）或 :math:`Q_y` （垂直）
- **归一化扫频宽度** ：取决于色散和 工作点 展宽，通常为 0.01~0.05
- **扫频周期** ：应远大于回旋周期 :math:`1/f_0` ，保证足够的频率分辨率
- **踢角** ：直接填写以 rad 为单位的带符号名义幅度，或在 :doc:`../gui_tools` 中按指定参考粒子和能量将电压换算为踢角
- **AM 参数** ： :math:`r_0` 与 :math:`\delta_0` 取同量级， :math:`t_{\text{ext}}` 根据束流扩散时间尺度设定

踢角约定
--------

令 :math:`\theta_0=\text{Kick angle (rad)}`。元件施加规定的归一化横向动量增量，
采用与 :doc:`kicker` 相同的名义小角度约定：

.. math::

   \Delta p_{u,i}=\theta_0 F(t_i),\qquad p_u=\frac{P_u}{P_0},\qquad u=x\ \text{or}\ y.

对处于参考动量的近轴粒子，:math:`\Delta u'\simeq\Delta p_u`，因此输入使用角度单位。
这不是让所有偏离参考动量或非近轴粒子都严格转过相同的几何角。
:math:`\theta_0` 的正负直接指定所选横向的踢角方向，已经包含所需的电荷符号约定；
跟踪时不再乘电荷符号、磁刚度或逐粒子速度因子。

束流能量变化时，配置的系数保持不变。到达时间相同的粒子获得相同的规定动量增量，
并按模式乘共同的 AM 因子；不同到达时间采样不同相位和 AM 值。
工作点模式使用共享参考钟，不为每个粒子定义独立的激励源频率。

这是零长度冲量，x、y、z、dp、t0 均不改变，不模拟穿过电极的过程、纵向电磁力、
能量交换、边缘场或传输线传播。FM 相位跨扫频回扫点保持连续，AM 使用同一个粒子
物理时间并扣除共同启动时刻。入口无效状态或冲量后不能正向运动的状态在该平面移出，
首次损失记录不被覆盖。CPU 和 GPU 使用相同方程。

迁移原电压输入
~~~~~~~~~~~~~~

``Voltage (V)``、``Gap (m)`` 和 ``Plate length (m)`` 不再是 Exciter 输入。
在 :doc:`../gui_tools` 中按所需参考粒子和能量进行电压到踢角换算，将带符号结果填入
``Kick angle (rad)``，并从 Exciter 配置中移除硬件字段和 ``Length (m)``。
换算器给出旧模型的名义参考系数。固定输入踢角不会复现旧电场模型的逐粒子速度因子，
也不随参考能量自动改变幅度。

粒子到达时间
------------

``bunch.t0`` 是参考粒子在当前激励位置的实际时刻，连续坐标满足
:math:`z=\beta_b c(T_b-t_i)`。激励器直接计算
:math:`t_i=T_b-z_i/(\beta_b c)`，不叠加名义槽位偏移，也不折叠存储 z。
不同到达时间采样不同信号相位。RF 参考速度变化时的 z 缩放保持该时间连续，
见 :ref:`zh-longitudinal-reference`。

所有束团采样同一个波形，共享启动时刻 :math:`t_*`。
由规定参考钟 :math:`f_0(t)` 的累计圈数确定启动时刻：

.. math::

   \int_{t_{\rm origin}}^{t_*} f_0(t)\,dt=n_{\rm start}+\frac{s}{C},
   \qquad u_i=t_i-t_*.

两个 DDS 在 :math:`u=0` 时都从零相位启动；启动前到达的粒子受到零激励。
原有 ``Start turn`` / ``End turn`` 命令执行区间仍然生效。
只有扫频进度对 :math:`T` 取余，累计相位不清零；改变绝对时刻原点不会改变该波形。

频率输入模式
------------

激励器的中心频率 :math:`f_c` 和扫频宽度 :math:`\Delta f` 支持两种输入方式：

**工作点模式** （推荐）

直接输入激励工作点 :math:`Q_{\text{excite}}` 和归一化扫频宽度 :math:`\Delta Q` ，程序使用束流共享的规定参考钟频率：

.. math::

  f_c(t) = Q_{\text{excite}} \cdot f_0(t)

.. math::

  \Delta f(t) = \Delta Q \cdot f_0(t)

共同参考钟采用自动生成的理想纯 RF 设计轨迹，不随集体效应引起的跟踪束团能量变化而改变。
没有活动 RF 电压时保持常频，见 :ref:`zh-reference-clock`。相位对真实时间上的瞬时频率积分，
包含参考钟的变化，不能用当前频率乘经过时间代替。需成对提供 ``excite tune`` 和 ``sweep tune`` 。

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
    - 扫频 + 时变幅度
  * - ``dual_fm``
    - 两个 DDS 扫频
    - 常值幅度
    - 两路各自连续相位的信号相加
  * - ``dual_fm_am``
    - 两个 DDS 扫频
    - 时变幅度
    - 两路信号乘相同 AM 因子


频率调制（FM）维度
~~~~~~~~~~~~~~~~~~~~~~

**单段线性扫频（single）**

令 :math:`u=t-t_*` 为共同启动后的经过时间， :math:`\tau=u\bmod T`。
中心频率和扫频宽度为常量时，相位为：

.. math::

  \phi_2(u)=2\pi f_cu+\frac{\pi\Delta f}{T}\tau(\tau-T).

载波项使用完整经过时间 :math:`u`，跨过扫频边界时不清零相位。

瞬时频率为：

.. math::

  f_s(u)=f_c+\Delta f\left(\frac{u\bmod T}{T}-\frac12\right).

- 当 :math:`\tau = 0` 时， :math:`f = f_c - \Delta f / 2` （起始频率）
- 当 :math:`\tau = T/2` 时， :math:`f = f_c` （中心频率）
- 当 :math:`\tau\to T^-` 时， :math:`f\to f_c+\Delta f/2`；回扫点跳回起始频率

频率在 :math:`[f_c - \Delta f/2,\; f_c + \Delta f/2]` 范围内线性扫描，每 :math:`T` 秒重复一次。中心频率 :math:`f_c` 应接近 :math:`Q \cdot f_0` （工作点乘回旋频率），以覆盖束流的共振频率。

**两个 DDS 扫频（dual）**

每个 DDS 都扫描完整频率范围，频率交叉后仍保持各自通道身份。
令 :math:`\alpha=\text{Dual sweep offset}`， :math:`\delta=\alpha T`：

.. math::

   f_1(u)=f_s(u+\delta),\qquad f_2(u)=f_s(u),
   \qquad S(x)=f_cx+\frac{\Delta f}{2T}(x\bmod T)((x\bmod T)-T),

.. math::

   \phi_1(u)=2\pi[S(u+\delta)-S(\delta)],\qquad
   \phi_2(u)=2\pi S(u).

减去 :math:`S(\delta)` 保证任意扫频偏移下，两个 DDS 的初始相位都为零。
默认 :math:`\alpha=0.5` 时，DDS1 在半周期回扫，DDS2 在整周期回扫，
并非让两个正弦载波相差半个周期。修改该偏移会给出警告，仍按所填值计算。
两路信号直接相加，无 AM 时峰值上界为 :math:`2|\theta_0|`，没有额外施加的余弦包络。
两个信号共用一个 ``Kick angle (rad)`` 设置和同一个 AM 因子，没有分路幅度输入，
也不自动除以二。

工作点模式的参考钟随时间变化时，使用一般积分定义：

.. math::

   f_j(t_*+u)=f_0(t_*+u)
      \left[Q_{\rm excite}+\Delta Q
      \left(\frac{(u+\delta_j)\bmod T}{T}-\frac12\right)\right],
   \qquad \phi_j(u)=2\pi\int_0^u f_j(t_*+v)\,dv,
   \qquad (\delta_1,\delta_2)=(\alpha T,0).

此处仍是连续波形模型，不模拟硬件的采样保持台阶。


幅度调制（AM）维度
~~~~~~~~~~~~~~~~~~~~~~

**常值幅度**

.. math::

  A(u) = \theta_0

不施加随时间变化的 AM 包络。带符号系数就是固定配置的踢角，
不随粒子速度或参考能量变化。

**时变幅度（am）**

基于束流扩散/增长模型，激励幅度随时间变化，不一定单调增加：

.. math::

  A(u) = \theta_0 \cdot \text{am\_factor}(u)

其中 :math:`\text{am\_factor}(u)` 是无量纲的时变缩放因子：

.. math::

  \text{am\_factor}(u) = \sqrt{\frac{\delta^2(u)}{f_{0,*} \cdot k_{\text{const}}}}

AM 与 FM 使用同一物理经过时间 :math:`u_i=t_i-t_*`，包含各粒子的到达偏移，
连续求值而不取整到圈。固定归一化频率 :math:`f_{0,*}=f_0(t_*)` 来自共同启动时
的规定参考钟，不跟随被跟踪束团的能量变化或后续参考钟频率变化。启动前 AM 为零。

初始发射度占比：

.. math::

  \varepsilon = \exp\!\left(-\frac{r_0^2}{\delta_0^2}\right)

AM 模型的辅助扩散量：

.. math::

  \delta^2(u) = \frac{r_0^2 (1 - \varepsilon)}{L^2 \cdot D}

其中：

.. math::

  L = \ln\!\left(\frac{u}{t_{\text{ext}}}(1 - \varepsilon) + \varepsilon\right)

.. math::

  D = t_{\text{ext}} \cdot \varepsilon + u (1 - \varepsilon)

长度使用米、时间使用秒时，:math:`\delta^2(u)` 的单位为
:math:`\mathrm{m}^2/\mathrm{s}`，:math:`k_{\text{const}}` 的单位为
:math:`\mathrm{m}^2`，从而使 :math:`\text{am\_factor}` 无量纲。

扩散公式保持原样，在 :math:`u=t_{\text{ext}}` 发散。应选择激励窗口，使粒子的
经过时间小于该值。GUI 拒绝达到这一边界的预览，输入校验对达到边界的参考时间窗口
给出警告；跟踪中不新增幅度截断。

物理意义：

- :math:`r_0` ：初始束流尺寸
- :math:`\delta_0` ：初始束流扩散范围
- :math:`t_{\text{ext}}` ：束流扩散特征时间
- :math:`k_{\text{const}}` ：模型归一化系数
- :math:`\varepsilon` ：初始发射度占比 （ :math:`r_0 / \delta_0` 比值的度量）

这一规定的 AM 包络用于横向激励和扩散研究，本身不计算发射度增长或机械能增益。
薄冲量保持 ``dp`` 不变，实际束流响应取决于晶格和采样的激励相位。

各模式完整公式
--------------

1. **single_fm** （一个 DDS + 常值幅度）

.. math::

  \text{kick}_i=\theta_0\sin\phi_2(u_i).

2. **single_fm_am** （一个 DDS + 时变幅度）

.. math::

  \text{kick}_i=\theta_0\,\text{am\_factor}(u_i)\sin\phi_2(u_i).

3. **dual_fm** （两个 DDS + 常值幅度）

.. math::

  \text{kick}_i=\theta_0[\sin\phi_1(u_i)+\sin\phi_2(u_i)].

4. **dual_fm_am** （两个 DDS + 时变幅度）

.. math::

  \text{kick}_i=\theta_0\,\text{am\_factor}(u_i)[\sin\phi_1(u_i)+\sin\phi_2(u_i)].

对粒子 :math:`i`，:math:`u_i=t_i-t_*`，:math:`\theta_0` 为配置的带符号踢角；
:math:`u_i<0` 时激励为零。FM 和 AM 使用同一个物理经过时间 :math:`u_i`。
上述公式给出最终归一化动量增量，跟踪时不再进行额外的幅度换算。

横向动量更新
------------

激励器是薄透镜元件，动量更新 直接加到对应方向的归一化动量上：

.. math::

  p_x \leftarrow p_x + \text{kick} \quad (\text{direction} = x)

.. math::

  p_y \leftarrow p_y + \text{kick} \quad (\text{direction} = y)

仅对存活粒子 （ ``tag > 0`` ）施加动量增量，已丢失粒子不受影响。

动量更新后，激励器会根据孔径参数 （ ``aperture_type`` ）对粒子进行孔径检查：若孔径类型不为 ``off`` ，则超出孔径范围的粒子将被标记为丢失 （ ``tag`` 置为负值）；若孔径类型为 ``off`` ，则不进行孔径检查。
