二极铁（Dipole / SBend）
========================

本模块介绍 PASS 中的二极铁元件 **SBend** ，用于模拟带电粒子在扇形弯转磁铁中的运动。二极铁是加速器中最基本的弯转元件，通过均匀磁场使粒子轨道偏转。

PASS 中的二极铁为 **厚元件** （ ``length > 0`` ），支持完整的非线性追踪，包括边缘角效应、边缘场效应、以及多种辛积分方案。

**代码位置**

- 源文件： ``PASS/commands/element/dipole.py``
- 类名： ``SBend`` （继承自 ``Command`` ）
- 注册名： ``sbend``
- 核心特征：

  - 支持 rot-kick-rot (RKR) 和 drift-kick-drift-exact (DKD-exact) 两种主体模型，默认 rot-kick-rot
  - 支持 uniform (2阶) 和 yoshida4 (4阶) 两种辛积分器
  - 支持非线性边缘角 (wedge) 效应
  - 支持非线性边缘场 (fringe field) 效应
  - 支持孔径检查


坐标约定
--------

PASS 采用与 Xsuite 一致的归一化曲线坐标，六维相空间变量为 :math:`(x, p_x, y, p_y, z, \delta)` ：

.. list-table::
  :header-rows: 1
  :widths: 15 20 65

  * - 变量
    - 符号
    - 定义
  * - ``x``
    - :math:`x`
    - 水平偏移（相对于参考轨道）
  * - ``px``
    - :math:`p_x`
    - 归一化水平动量， :math:`p_x = P_x / P_0`
  * - ``y``
    - :math:`y`
    - 垂直偏移
  * - ``py``
    - :math:`p_y`
    - 归一化垂直动量， :math:`p_y = P_y / P_0`
  * - ``z``
    - :math:`\zeta`
    - 纵向坐标， :math:`\zeta = s - \beta_0 c t`
  * - ``dp``
    - :math:`\delta`
    - 相对动量偏差， :math:`\delta = P / P_0 - 1`

其中 :math:`P_0` 为参考粒子动量， :math:`\beta_0 = v_0 / c` 为参考粒子归一化速度， :math:`s` 为沿参考轨道的弧长， :math:`t` 为时间。

纵向动量分量定义为：

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

其他常用物理量：

.. math::

  h = \frac{1}{\rho} = \frac{K_0 L}{L} = \frac{K_{0L}}{L}

.. math::

  K_0 = \frac{q_0 B_0}{P_0} = \frac{K_{0L}}{L}

.. math::

  \chi = \frac{q}{q_0} \cdot \frac{m_0}{m}

其中 :math:`h` 为参考轨道曲率， :math:`\rho` 为弯转半径， :math:`K_0` 为归一化二极场强， :math:`\chi` 为荷质比因子（同种粒子束 :math:`\chi = 1` ）。对于扇形弯铁 :math:`h = K_0` 。


整体追踪流程
------------

一个完整的二极铁由 **入口边缘** 、 **主体** 和 **出口边缘** 三部分组成，追踪顺序为：

::

  ====== 入口边缘 (B=0 → B=B0) ======

  YRotation(-e1)  →  Fringe Field  →  Wedge(-e1, K0)
  [纯几何旋转]       [非线性边缘场]     [旋转+聚焦踢腿]
  [B=0]             [B: 0→B0]         [B=B0]

         ↓

  ====== 主体 Body (B=B0) ======

  切片1 → 切片2 → ... → 切片N
  (参考轨迹坐标系, 辛积分器)

         ↓

  ====== 出口边缘 (B=B0 → B=0) ======

  Wedge(-e2, K0)   →  Fringe Field  →  YRotation(-e2)
  [旋转+聚焦踢腿]     [非线性边缘场]     [纯几何旋转]
  [B=B0]             [B: B0→0]         [B=0]

  入口净旋转: (-e1) + (+e1) = 0
  出口净旋转: (+e2) + (-e2) = 0
  → Body 在参考轨迹坐标系中运行

完整映射为：

.. math::

  \mathcal{M}_{\text{bend}} = \mathcal{M}_{\text{exit}} \circ \mathcal{M}_{\text{body}} \circ \mathcal{M}_{\text{entry}}

**入口边缘** ：

.. math::

  \mathcal{M}_{\text{entry}} = \text{Wedge}(-e_1, K_0) \circ \text{Fringe}(e_1) \circ \text{YRotation}(-e_1)

**出口边缘** （与入口镜像对称）：

.. math::

  \mathcal{M}_{\text{exit}} = \text{YRotation}(-e_2) \circ \text{Fringe}(e_2) \circ \text{Wedge}(-e_2, K_0)

.. note::

  - 当 :math:`e_1 = 0` 时，YRotation 和 Wedge 均跳过（无边缘角效应）
  - 当 ``fint`` = 0 或 ``hgap`` = 0 时，Fringe 跳过（无边缘场效应）
  - 当 :math:`K_0 = 0` 时，Fringe 和 Wedge 均跳过
  - 入口和出口的执行顺序互为镜像
  - **出口处** :math:`K_0` **部分取反** ：Xsuite 在出口处将局部变量 :math:`K_0` 取反（ ``if (is_exit) k0 = -k0`` ），但仅 **DipoleFringe** 使用取反后的 :math:`-K_0` ，因为出口边缘场是磁场从 :math:`B_0` 下降到 0（与入口的 0 上升到 :math:`B_0` 方向相反）。 **Wedge** 直接使用原始 ``knorm[0]`` （不取反），因为 Wedge 描述的是均匀磁场 :math:`B_0` 中的旋转，磁场方向在入口和出口一致。PASS 在 ``_edge_exit_cpu`` 中使用 ``k0_fringe = -k0`` （仅 Fringe）和 ``k0`` （Wedge）实现此行为。


为什么是这个顺序
~~~~~~~~~~~~~~~~

**YRotation 的作用** ：纯几何坐标旋转，将粒子坐标从参考轨迹坐标系变换到磁铁端面坐标系。不施加任何力，仅改变观察参考系。

**Fringe Field 的作用** ：在端面参考系中计算非线性边缘场效应。Fringe 公式中的粒子斜率 :math:`x' = p_x / p_z` 必须是相对于磁铁端面的斜率，因此必须先做 YRotation。

**Wedge 的作用** ：在均匀磁场 :math:`B_0` 中旋转观测平面，同时施加边缘角聚焦踢角。Wedge 包含几何旋转（将坐标系转回参考轨迹系）和磁场踢角两部分。

YRotation 和 Wedge 的几何旋转方向相反，净旋转为零：

.. math::

  \text{YRotation}(-e_1) \text{ 的旋转} + \text{Wedge}(-e_1) \text{ 的几何旋转} = (-e_1) + (+e_1) = 0

因此 body 在参考轨迹坐标系中运行，不需要任何额外旋转。


主体：DKD-exact 模型
--------------------

哈密顿量
~~~~~~~~

在曲线坐标系中，二极铁的完整哈密顿量为：

.. math::

  H_{\text{bend}} = \frac{p_\tau}{\beta_0} - (1+hx)\sqrt{(1+\delta)^2 - p_x^2 - p_y^2} + \chi K_0\!\left(x + \frac{h x^2}{2}\right)

将其拆分为传播部分（精确直漂移 :math:`H_D` ）和踢角部分（ :math:`H_h` , :math:`H_{K_0}` , :math:`H_{K_0 h}` ）：

.. math::

  H_D = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  H_h = -h x (1+\delta)

.. math::

  H_{K_0} = \chi K_0 x

.. math::

  H_{K_0 h} = \frac{\chi K_0 h x^2}{2}

其中 :math:`H_D` 是精确直漂移哈密顿量（保留 :math:`p_z` 的根号，不做小动量展开），后三项是薄透镜踢角。

精确漂移映射 D
~~~~~~~~~~~~~~~~~~

传播部分的哈密顿方程给出精确直漂移：

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  x \leftarrow x + \frac{p_x}{p_z} \cdot L_D

.. math::

  y \leftarrow y + \frac{p_y}{p_z} \cdot L_D

.. math::

  \zeta \leftarrow \zeta + L_D \cdot \left(1 - \frac{\beta_0}{\beta} \cdot \frac{1+\delta}{p_z}\right)

其中 :math:`L_D` 为漂移长度， :math:`\beta` 为粒子实际归一化速度：

.. math::

  \beta = \frac{(1+\delta) \beta_0 \gamma_0}{\sqrt{1 + \left[(1+\delta) \beta_0 \gamma_0\right]^2}}

.. note::

  "exact" 的含义：漂移部分保留精确根号 :math:`p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}` ，不做 :math:`p_x \ll 1` 的小动量展开。近似仅在于将传播部分与踢角部分分离（分裂算符法）。

二极踢角映射 K
~~~~~~~~~~~~~~~~~~~~

踢角部分合并为单个薄透镜 kick（位置不变，仅动量跳变）：

.. math::

  p_x \leftarrow p_x + L_K \cdot \left[h(1+\delta) - \chi K_0 - \chi K_0 h x\right]

.. math::

  \zeta \leftarrow \zeta - \frac{\beta_0}{\beta} \cdot h x L_K

各项物理含义：

.. list-table::
  :header-rows: 1
  :widths: 30 15 55

  * - 项
    - 来源
    - 物理含义
  * - :math:`h(1+\delta) L_K`
    - :math:`H_h`
    - 曲率踢角（参考轨道弯转）
  * - :math:`-\chi K_0 L_K`
    - :math:`H_{K_0}`
    - 主二极弯转
  * - :math:`-\chi K_0 h x L_K`
    - :math:`H_{K_0 h}`
    - 弱聚焦（曲率与二极场耦合）
  * - :math:`-(\beta_0/\beta) \cdot h x L_K`
    - :math:`H_h`
    - 路径长度效应（纵向）

.. note::

  对于扇形弯铁 :math:`h = K_0` ，参考粒子（ :math:`x=0, \delta=0, \chi=1` ）的净踢角为 :math:`h L_K - K_0 L_K = 0` 。这是正确的：在曲线坐标系中，参考粒子沿参考轨道运动， :math:`p_x` 始终为 0。弯转效应已编码在曲线坐标系本身中。

uniform 积分器（2阶辛）
~~~~~~~~~~~~~~~~~~~~~~~~~~

每个切片采用漂移-踢角-漂移（DKD）结构，即二阶蛙跳（leapfrog）：

.. math::

  S_2(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

每个切片误差为 :math:`O(\Delta s^3)` ，全局误差为 :math:`O(\Delta s^2)` 。二阶辛积分器，每步都是正则变换。

yoshida4 积分器（4阶辛）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

通过组合三个二阶 DKD 步构造四阶辛映射 [Yoshida 1990]：

.. math::

  S_4(\Delta s) = S_2(z_1 \Delta s) \circ S_2(z_0 \Delta s) \circ S_2(z_1 \Delta s)

其中 Yoshida 系数为：

.. math::

  z_1 = \frac{1}{2 - 2^{1/3}} \approx 1.3512

.. math::

  z_0 = 1 - 2 z_1 \approx -1.7024

.. note::

  :math:`z_0 < 0` 意味着中间一步是反向追踪（漂移和踢角的 "长度" 为负）。这是 Yoshida 组合方法的数学要求，在辛映射群中完全自洽。每个切片误差为 :math:`O(\Delta s^5)` ，全局误差为 :math:`O(\Delta s^4)` 。

积分器选择建议：

.. list-table::
  :header-rows: 1
  :widths: 25 20 55

  * - 场景
    - 推荐积分器
    - 原因
  * - 快速模拟
    - uniform
    - 每切片 2 次漂移 + 1 次踢角，计算量小
  * - 高精度模拟
    - yoshida4
    - 4 阶精度，但每切片 6 次漂移 + 3 次踢角
  * - 含空间电荷
    - uniform + 更多切片
    - PIC 求解开销远大于漂移，4阶 Yoshida 需要 3 次 PIC 求解


主体模型对比
-----------

PASS 二极铁主体支持两种物理模型，由 ``model`` 参数选择。

模型说明
~~~~~~~~

- **rot-kick-rot (RKR)** ：偶极场为常数场（不依赖 :math:`x` ），薄透镜踢角本身是精确的。drift 步使用极坐标漂移（polar drift），处理曲率效应。1个切片即可达到高精度。默认模型。
- **drift-kick-drift-exact (DKD-exact)** ：drift 步使用直线精确漂移，曲率作为薄透镜 kick 处理。对于偏角较大的弯铁，切片数不足时色品会有误差。

特性对比
~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 25 35 40

  * - 特性
    - rot-kick-rot
    - drift-kick-drift-exact
  * - drift类型
    - 极坐标漂移（含曲率）
    - 直线精确漂移
  * - k0处理
    - 在drift步内部（interleaved k0_kick）
    - 在kick步（薄透镜）
  * - 曲率Jacobian
    - 包含（(1+h·x)修正）
    - 不包含（需多切片弥补）
  * - 1切片精度
    - 高（常数场kick精确）
    - 有限（曲率近似）
  * - 色品精度
    - 1切片即精确
    - 需足够切片数

切片数推荐
~~~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 25 15 20 40

  * - 模型
    - 推荐切片数
    - 推荐积分器
    - 说明
  * - rot-kick-rot
    - 1
    - yoshida4
    - 偶极常数场使1切片kick精确，极坐标漂移处理曲率，1切片即达高精度
  * - drift-kick-drift-exact
    - 5~10
    - yoshida4
    - 偏角较大时需多切片弥补曲率Jacobian缺失导致的色品误差

.. note::

  DKD-exact模型中，曲率h作为薄透镜kick处理，drift步不含(1+h·x) Jacobian修正。当弯铁偏角较大时，不同动量粒子在drift中走过的路径差异未被正确计入，导致色品出现偏差。增加切片数可缓解此问题（误差按 :math:`O(1/N^2)` 收敛），但RKR模型从根本上避免了此问题——极坐标漂移在drift步内部处理曲率，无需多切片。


入口边缘：YRotation
--------------------

物理目的
~~~~~~~~

YRotation 是纯几何坐标旋转，不涉及任何磁场。其目的是将粒子坐标从参考轨迹坐标系变换到磁铁端面坐标系，使后续的边缘场计算能在正确的参考系中进行。

当磁铁端面与参考轨迹法线有夹角 :math:`e_1` 时，粒子相对于端面的入射斜率不等于 :math:`p_x / p_z` 。YRotation 将坐标系旋转 :math:`-e_1` ，使得旋转后的 :math:`p_x / p_z` 成为相对于端面的斜率。

完整推导
~~~~~~~~

::

  YRotation(-e1)                    Wedge(-e1, K0)
  ─────────────────                 ──────────────────
  坐标系: 轨迹系 → 端面系            坐标系: 端面系 → 轨迹系
  磁场: B = 0 (无场区)              磁场: B = B0 (有场区)
  旋转: -e1                         旋转: +e1 (与 YRotation 反向)
  踢腿: 无                          踢腿: Δpx = K0·x·sin(e1) (聚焦)
  ─────────────────                 ──────────────────

  净旋转: (-e1) + (+e1) = 0
  净效果: 边缘场效应 + 聚焦踢腿

**第一步：动量旋转**

参考系绕 :math:`y` 轴旋转角度 :math:`\theta` 。这个旋转在三维空间的 :math:`(x, z)` 平面内进行， :math:`y` 方向不受影响。因此被混合的是动量的 :math:`x` 分量 :math:`p_x` 和 :math:`z` 分量 :math:`p_z` ，而 :math:`p_y` 不变。动量的标准旋转变换为：

.. math::

  \begin{pmatrix} p_x' \\ p_z' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} p_x \\ p_z \end{pmatrix}

因此：

.. math::

  p_x' = \cos\theta \cdot p_x - \sin\theta \cdot p_z

.. math::

  p_z' = \sin\theta \cdot p_x + \cos\theta \cdot p_z

这是纯动量旋转，不涉及力。

**第二步：位置投影**

关键在于曲线坐标系中 :math:`x` 的定义： **x 是在参考位置** :math:`s` **处的横向平面上测量的横向偏移** 。当参考系旋转 :math:`\theta` 后，"横向平面" 的方向变了，同一物理位置在新坐标系中的 :math:`x'` 值不同。

粒子在旧参考系中的轨迹为 :math:`\vec{r}(\lambda) = (x + \lambda p_x/p_z,\; 0,\; \lambda)` 。新参考系的 :math:`z'` 轴方向为 :math:`\hat{z}' = (\sin\theta, 0, \cos\theta)` ，横向方向为 :math:`\hat{x}' = (\cos\theta, 0, -\sin\theta)` 。

在新参考系中，纵向位置为零意味着：

.. math::

  \vec{r} \cdot \hat{z}' = (x + \lambda p_x/p_z)\sin\theta + \lambda\cos\theta = 0

解出 :math:`\lambda` ：

.. math::

  \lambda = -\frac{x \sin\theta \cdot p_z}{p_x \sin\theta + p_z \cos\theta} = -\frac{x \sin\theta \cdot p_z}{p_z'}

新横向位置为：

.. math::

  x' = \vec{r} \cdot \hat{x}' = (x + \lambda p_x/p_z)\cos\theta - \lambda\sin\theta

代入 :math:`\lambda` 并化简，利用旋转的正交性 :math:`p_z'\cos\theta - p_x'\sin\theta = p_z` ，最终得到：

.. math::

  x' = \frac{x \cdot p_z}{p_z'} = \frac{x \cdot p_z}{p_x \sin\theta + p_z \cos\theta}

**验证与 Xsuite 公式的一致性** ：

Xsuite 写作 ``x_hat = x / (cos_angle * ptt)`` ，其中 ``ptt = 1 + tan_angle * px / pz`` ：

.. math::

  \cos\theta \cdot p_{tt} = \cos\theta + \sin\theta \cdot \frac{p_x}{p_z} = \frac{p_z'}{p_z}

因此 :math:`x' = x / (\cos\theta \cdot p_{tt}) = x \cdot p_z / p_z'` ，与推导一致。

**第三步： y 方向**

:math:`y` 方向不直接参与旋转，但由于横向平面倾斜，粒子在走到新横向平面期间， :math:`y` 方向因轨迹斜率产生额外偏移：

.. math::

  y' = y - \sin\theta \cdot \frac{x \cdot p_y}{p_z'} = y - \tan\theta \cdot \frac{x \cdot p_y}{p_z \cdot p_{tt}}

**第四步：** :math:`\zeta` **方向**

纵向坐标 :math:`\zeta = s - \beta_0 c t` 。旋转参考系后， :math:`\sin\theta \cdot x` 是横向位置在新纵向方向上的投影（因参考系旋转导致的额外路径长度），需转换为时间增量：

.. math::

  \Delta\zeta = \beta_0 \cdot \tan\theta \cdot \frac{x \cdot \text{time\_fac}}{p_z \cdot p_{tt}}

其中 :math:`\text{time\_fac} = 1/\beta_0 + p_\tau` 是时间-路径转换因子（见变量转换说明）。

最终公式
~~~~~~~~

定义：

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  p_{tt} = 1 + \tan(\theta) \cdot \frac{p_x}{p_z}

其中 :math:`\theta` 为旋转角度（入口 :math:`\theta = -e_1` ，出口 :math:`\theta = -e_2` ）。

时间因子：

.. math::

  \text{time\_fac} = \frac{1}{\beta_0} + p_\tau = \sqrt{(1+\delta)^2 + \frac{1}{\beta_0^2 \gamma_0^2}}

六维映射：

.. math::

  x \leftarrow \frac{x}{\cos\theta \cdot p_{tt}}

.. math::

  p_x \leftarrow \cos\theta \cdot p_x - \sin\theta \cdot p_z

.. math::

  y \leftarrow y - \tan\theta \cdot \frac{x \cdot p_y}{p_z \cdot p_{tt}}

.. math::

  \zeta \leftarrow \zeta + \beta_0 \cdot \tan\theta \cdot \frac{x \cdot \text{time\_fac}}{p_z \cdot p_{tt}}

.. math::

  p_y, \delta \text{ 不变}

变量转换说明
~~~~~~~~~~~~~~~~~~~~

Xsuite 中存储 :math:`p_\tau` （归一化纵向动量偏差），而 PASS 存储 :math:`\delta` （归一化总动量偏差）。两者的精确关系为：

.. math::

  (1+\delta)^2 = \left(\frac{1}{\beta_0} + p_\tau\right)^2 - \frac{1}{\beta_0^2 \gamma_0^2}

因此：

.. math::

  \frac{1}{\beta_0} + p_\tau = \sqrt{(1+\delta)^2 + \frac{1}{\beta_0^2 \gamma_0^2}}

**推导** ：粒子总能量 :math:`E = E_0(1 + p_\tau)` ，由 :math:`E^2 - E_0^2 = (Pc)^2 - (P_0 c)^2` 得：

.. math::

  (1+\delta)^2 = (P/P_0)^2 = \frac{(1+p_\tau)^2 - 1/\gamma_0^2}{\beta_0^2}

展开并利用 :math:`1 - 1/\gamma_0^2 = \beta_0^2` ：

.. math::

  (1+\delta)^2 = 1 + \frac{2 p_\tau}{\beta_0^2} + \frac{p_\tau^2}{\beta_0^2}

而 Xsuite 中 :math:`p_z` 的表达式为 ``sqrt(1 + 2*pt/beta0 + pt*pt - px*px - py*py)`` ，代入上述关系后恰好等于 :math:`\sqrt{(1+\delta)^2 - p_x^2 - p_y^2}` 。


入口边缘：边缘场（Fringe Field）
--------------------------------

物理目的
~~~~~~~~

真实磁铁的磁场在端面处不是阶跃函数，而是有一个渐变区域。这个渐变区域产生额外的非线性效应，主要是垂直方向的聚焦。

物理推导
~~~~~~~~

::

  B_y                                          B_y = B0
  ↑                              ┌──────────────
  │                             /
  │         Fringe             /  end face
  │         Field             /
  │       (B: 0→B0)          /
  │                        /
  │  B = 0               /
  └────────────┬────────┘──────────────────────── → s
               ↑        ↑
               YRotation Fringe    Wedge / Body
               (B=0)    (0→B0)     (B=B0)

  hgap = 半气隙
  Δp_y = 垂直聚焦踢腿

**边缘场分布**

真实磁铁端面处的磁场渐变为：

.. math::

  B_y(s) = B_0 \cdot b(s), \quad b(s): 0 \to 1

其中 :math:`b(s)` 是归一化边缘场分布函数。这个渐变场对粒子产生非线性效应，主要是垂直方向的聚焦。

**边缘场积分**

Forest 定义边缘场积分：

.. math::

  F = \int \frac{b(s)\bigl(K_0 - b(s)K_0\bigr)}{g_{\text{full}} \cdot K_0^2} \, ds

其中 :math:`g_{\text{full}}` 为磁铁全气隙（ :math:`g_{\text{full}} = 2 \times \text{hgap}` ）。 :math:`F` 描述边缘场的"强度"： :math:`F=0` 对应硬边缘， :math:`F` 越大边缘场效应越强。

.. warning::

  **hgap 与 g 的命名关系**

  Xsuite 和 PASS 的 ``hgap`` 参数是磁铁的 **半气隙** （half gap），即上下极板间距的一半。

  - ``hgap`` = 半气隙 = :math:`g_{\text{half}}`
  - 磁铁全气隙 :math:`g_{\text{full}} = 2 \times \text{hgap}`

  在边缘场代码中，辅助量 :math:`f_h` 直接使用 ``hgap`` （半气隙）：

  .. math::

    f_h = \text{hgap} \times \text{fint}

  这与 Xsuite 源码 ``track_dipole_fringe.h`` 第 37 行 ``fh = hgap * fint`` 完全一致。Xsuite 的 ``hgap`` 参数也是半气隙。因此 :math:`f_h` 的物理含义是"半气隙 x 边缘场积分"，而非"全气隙 x 边缘场积分"。

  在物理文献中（如 Forest 的原始论文），边缘场积分公式中的 :math:`g` 通常指全气隙。Xsuite/MAD-NG 的实现中已将 :math:`g` 替换为半气隙 ``hgap`` ，相应的系数已做调整（如 :math:`f_{\text{sad}} = 1/(72 \cdot f_h)` 中的因子 72 即来自此调整）。PASS 严格遵循 Xsuite 的实现，不做额外换算。

**生成函数**

边缘场映射是一个正则变换（保辛映射），由生成函数 :math:`\Phi_0` 生成：

.. math::

  \Phi_0 = \arctan\!\left(\frac{x'}{1+y'^2}\right) - c_2 \left(1 + x'^2(1+y'^2)\right) p_z

其中 :math:`x' = p_x/p_z` , :math:`y' = p_y/p_z` 是粒子斜率， :math:`c_2 = 2 K_0 \chi \cdot f_h` 是线性边缘场强度参数。

- **第一项** :math:`\arctan(x'/(1+y'^2))` ：粒子在端面处的入射角修正。 :math:`x'` 是水平斜率， :math:`1+y'^2` 反映垂直运动对水平入射角的几何修正（三维方向余弦）。
- **第二项** :math:`-c_2(1 + x'^2(1+y'^2))p_z` ：边缘场积分效应。 :math:`c_2` 是边缘场强度， :math:`(1 + x'^2(1+y'^2))` 是斜率的高阶修正。

**偏导数与力**

从 :math:`\Phi_0` 对斜率 :math:`(x', y', p_z)` 求偏导，再通过链式法则转换为对 :math:`(p_x, p_y, \delta)` 的偏导，得到力的分量。

引入中间变量 :math:`c_{o2} = b_0 / \cos^2\Phi_0` , :math:`c_{o1}`, :math:`c_{o3}` （详见公式部分），偏导数为：

.. math::

  \phi_1 = \frac{\partial \Phi_0}{\partial x'}, \quad \phi_2 = \frac{\partial \Phi_0}{\partial y'}, \quad \phi_3 = \frac{\partial \Phi_0}{\partial p_z}

力的分量（链式法则 :math:`k_i = \phi_1 \partial x'/\partial p_i + \phi_2 \partial y'/\partial p_i + \phi_3 \partial p_z/\partial p_i` ）：

.. math::

  k_x = \phi_1 \frac{1+x'^2}{p_z} + \phi_2 \frac{x'y'}{p_z} - \phi_3 x'

.. math::

  k_y = \phi_1 \frac{x'y'}{p_z} + \phi_2 \frac{1+y'^2}{p_z} - \phi_3 y'

.. math::

  k_z = \phi_1 \frac{\text{tfac} \cdot x'}{p_z^2} + \phi_2 \frac{\text{tfac} \cdot y'}{p_z^2} - \phi_3 \frac{\text{tfac}}{p_z}

其中 :math:`\text{tfac} = -(1/\beta_0 + p_\tau)` 来自 :math:`\zeta = s - \beta_0 c t` 中 :math:`t` 对 :math:`p_z` 的依赖。

**隐式方程**

边缘场映射不是简单的 kick（位置不变、动量跳变），而是一个 **隐式映射** 。原因是边缘场效应与粒子的 :math:`y` 坐标非线性耦合：粒子在穿过边缘场时， :math:`y` 坐标本身也在变化，因此力（依赖 :math:`y` ）和位移（依赖力）相互耦合。

隐式解来自生成函数展开到二阶：

.. math::

  y_f = \frac{2y}{1 + \sqrt{1 - 2 k_y y}}

这个形式保证了正则性（保辛）：当 :math:`k_y y` 很小时， :math:`y_f \approx y + k_y y^2/2` ，即二阶展开。

参数
~~~~

.. list-table::
  :header-rows: 1
  :widths: 15 15 70

  * - 参数
    - 符号
    - 说明
  * - ``fint``
    - :math:`F`
    - 边缘场积分（fringe field integral）， :math:`F=0` 为硬边缘
  * - ``hgap``
    - :math:`g_{\text{half}}`
    - 磁铁 **半气隙** （half gap），全气隙 :math:`g_{\text{full}} = 2 \cdot \text{hgap}`
  * - ``k0``
    - :math:`K_0`
    - 归一化二极场强

完整公式
~~~~~~~~

定义辅助量：

.. math::

  f_h = \text{hgap} \cdot F

.. math::

  f_{\text{sad}} = \frac{1}{72 \cdot f_h} \quad (f_h > 0 \text{ 时，否则为 } 0)

.. math::

  b_0 = K_0 \cdot \chi

.. math::

  \text{relp} = \frac{1}{\sqrt{(1+\delta)^2}}

.. math::

  \text{tfac} = -\left(\frac{1}{\beta_0} + p_\tau\right) = -\sqrt{(1+\delta)^2 + \frac{1}{\beta_0^2 \gamma_0^2}}

.. math::

  c_2 = b_0 \cdot f_h \cdot 2

.. math::

  c_3 = b_0^2 \cdot f_{\text{sad}} \cdot \text{relp}

其中 :math:`c_2` 是线性边缘场强度， :math:`c_3` 是六阶非线性修正（来自 :math:`f_{\text{sad}} = 1/(72 f_h)` ，称为 "sixth-order achromatic detuning" 项）。

粒子斜率：

.. math::

  x' = \frac{p_x}{p_z}, \quad y' = \frac{p_y}{p_z}

特征函数及偏导数：

.. math::

  \phi_0 = \arctan\!\left(\frac{x'}{1 + y'^2}\right) - c_2 \left(1 + x'^2(1+y'^2)\right) p_z

.. math::

  c_{o2} = \frac{b_0}{\cos^2\phi_0}

.. math::

  c_{o1} = \frac{c_{o2}}{1 + \left(\frac{x'}{1+y'^2}\right)^2} \cdot \frac{1}{1+y'^2}

.. math::

  c_{o3} = c_{o2} \cdot c_2

.. math::

  \phi_1 = c_{o1} - c_{o3} \cdot 2 x'(1+y'^2) p_z

.. math::

  \phi_2 = -2 c_{o1} \cdot x' y' \cdot \frac{1}{1+y'^2} - c_{o3} \cdot 2 x' y' \cdot p_z

.. math::

  \phi_3 = -c_{o3} \left(1 + x'^2(1+y'^2)\right)

力的分量：

.. math::

  k_x = \phi_1 \frac{1+x'^2}{p_z} + \phi_2 \frac{x' y'}{p_z} - \phi_3 x'

.. math::

  k_y = \phi_1 \frac{x' y'}{p_z} + \phi_2 \frac{1+y'^2}{p_z} - \phi_3 y'

.. math::

  k_z = \phi_1 \frac{\text{tfac} \cdot x'}{p_z^2} + \phi_2 \frac{\text{tfac} \cdot y'}{p_z^2} - \phi_3 \frac{\text{tfac}}{p_z}

六维映射：

.. math::

  y_f = \frac{2y}{1 + \sqrt{1 - 2 k_y y}}

.. math::

  x \leftarrow x + \frac{1}{2} k_x y_f^2

.. math::

  p_y \leftarrow p_y - 4 c_3 y_f^3 - b_0 \tan(\phi_0) \cdot y_f

.. math::

  \zeta \leftarrow \zeta + \beta_0 \left(\frac{1}{2} k_z y_f^2 + c_3 y_f^4 \cdot \text{relp}^2 \cdot \text{tfac}\right)

.. math::

  p_x, \delta \text{ 不变}

.. note::

  - :math:`p_x` 和 :math:`\delta` 不变：边缘场是静磁场，不做功
  - :math:`p_y` 的变化中的 :math:`-b_0 \tan(\phi_0) y_f` 项是主要的垂直聚焦
  - :math:`-4 c_3 y_f^3` 是六阶非线性修正
  - :math:`y_f` 通过隐式方程求解，保证了非线性效应的精确处理和保辛性
  - :math:`x` 的变化是 :math:`O(y^2)` 量级的水平-垂直耦合


入口边缘：Wedge（边缘角）
--------------------------

物理目的
~~~~~~~~

Wedge 描述粒子在均匀磁场 :math:`B_0` 中穿过倾斜端面时的坐标变换。它同时完成两件事：

1. **几何旋转** ：将坐标系从端面参考系转回参考轨迹参考系（与 YRotation 的旋转方向相反，互相抵消）
2. **磁场踢角** ：施加边缘角聚焦效应 :math:`\Delta p_x \propto -x`

当 :math:`K_0 = 0` （无磁场）时，Wedge 退化为纯 YRotation（证明见下文）。

计算公式
~~~~~~~~

参数： :math:`\theta` （楔形角度，入口 :math:`\theta = -e_1` ）， :math:`K_0` ， :math:`\chi`

定义：

.. math::

  b_1 = K_0 \cdot \chi

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  A = \frac{1}{\sqrt{(1+\delta)^2 - p_y^2}}

.. math::

  \text{rvv} = \frac{\beta}{\beta_0}

其中 :math:`\beta` 为粒子实际归一化速度（见漂移映射中的公式）。

六维映射：

.. math::

  p_x' = p_x \cos\theta + (p_z - b_1 x) \sin\theta

.. math::

  p_z' = \sqrt{(1+\delta)^2 - p_x'^2 - p_y^2}

.. math::

  x' = x \cos\theta + \frac{x p_x \sin(2\theta) + \sin^2\theta \cdot (2 x p_z - b_1 x^2)}{p_z' + p_z \cos\theta - p_x \sin\theta}

.. math::

  D = \arcsin(A \cdot p_x) - \arcsin(A \cdot p_x')

.. math::

  \Delta y = \frac{p_y (\theta + D)}{b_1}

.. math::

  \Delta \ell = \frac{(1+\delta) (\theta + D)}{b_1}

最终更新：

.. math::

  x \leftarrow x'

.. math::

  p_x \leftarrow p_x'

.. math::

  y \leftarrow y + \Delta y

.. math::

  \zeta \leftarrow \zeta - \frac{\Delta \ell}{\text{rvv}}

.. math::

  p_y, \delta \text{ 不变}

踢角项的物理含义
~~~~~~~~~~~~~~~~~~~~~~~~~~

将 :math:`p_x'` 的公式展开：

.. math::

  p_x' = \underbrace{p_x \cos\theta + p_z \sin\theta}_{\text{几何旋转}} \;\underbrace{-\; b_1 x \sin\theta}_{\text{磁场踢角}}

- **几何旋转部分** ： :math:`p_x \cos\theta + p_z \sin\theta` ，与 YRotation 的 :math:`\cos\theta \cdot p_x - \sin\theta \cdot p_z` 方向相反，互相抵消
- **磁场踢角部分** ： :math:`-b_1 x \sin\theta = -K_0 \chi x \sin\theta` ，正比于 :math:`-x` ，即边缘角聚焦

对于入口（ :math:`\theta = -e_1` ），踢角为：

.. math::

  \Delta p_x = -K_0 \chi x \sin(-e_1) = K_0 \chi x \sin(e_1) > 0 \quad (x > 0 \text{ 时})

这使轨道外侧（ :math:`x > 0` ）的粒子获得向内的动量，即聚焦效应。等效焦距为 :math:`f = \rho / \sin(e_1)` 。


Delta-ell 与 zeta 的关系
~~~~~~~~~~~~~~~~~~~~~~~~

公式 Eq. 1.201 给出的是 **路径长度** :math:`\Delta\ell` ，而代码更新的是 **纵向坐标** :math:`\zeta` 。两者不同：

.. math::

  \Delta\ell = \frac{(1+\delta)(\theta + D)}{b_1} \quad \text{(路径长度)}

.. math::

  \Delta\zeta = -\frac{\Delta\ell}{\text{rvv}} \quad (\zeta \text{ 坐标更新})

**物理原因** ：

1. :math:`\Delta\ell` 是粒子在 wedge 中走过的路径长度
2. :math:`\zeta = s - \beta_0 c t` ，更新 :math:`\zeta` 需要时间： :math:`\Delta t = \Delta\ell / v = \Delta\ell / (\text{rvv} \cdot \beta_0 c)`
3. :math:`\Delta s` 已在 :math:`x'` 和 :math:`\Delta y` 的几何变换中处理了，所以 :math:`\zeta` 只需要 **时间修正** 部分：

.. math::

  \Delta\zeta = -\beta_0 c \cdot \Delta t = -\beta_0 c \cdot \frac{\Delta\ell}{\text{rvv} \cdot \beta_0 c} = -\frac{\Delta\ell}{\text{rvv}}

- **负号** ： :math:`\zeta = s - \beta_0 c t` ，时间增加导致 :math:`\zeta` 减小
- **除以 rvv** ： :math:`\text{rvv} = v/v_0 = \beta/\beta_0` ，将路径长度转换为时间时需要除以粒子实际速度

.. math::

  \zeta \leftarrow \zeta - \frac{\Delta \ell}{\text{rvv}}

与 Xsuite 源码 ``add_to_zeta(-delta_ell / rvv)`` 完全一致。

.. warning::

  不要将 :math:`\Delta\ell` 直接加到 :math:`\zeta` 上。 :math:`\Delta\ell` 是路径长度， :math:`\zeta` 是时间相关坐标，两者通过 :math:`-\Delta\ell/\text{rvv}` 联系。


arcsin 的 clip 处理
~~~~~~~~~~~~~~~~~~~~~

Wedge 中 :math:`D` 的计算用到 :math:`\arcsin` ：

.. math::

  D = \arcsin(A \cdot p_x) - \arcsin(A \cdot p_x')

其中 :math:`A = 1/\sqrt{(1+\delta)^2 - p_y^2}` 。理论上 :math:`|A \cdot p_x| \leq 1` （因为 :math:`|p_x| \leq \sqrt{(1+\delta)^2 - p_y^2}` ），但 **浮点误差** 可能导致 :math:`A \cdot p_x = 1.0000000001` ，此时 Python 的 ``np.arcsin`` 返回 NaN 并触发 RuntimeWarning。

代码中使用 ``np.clip`` 将参数限制在 :math:`[-1, 1]` 范围内：

.. code-block:: python

  arg_px = np.clip(arg_px, -1.0, 1.0)
  arg_new_px = np.clip(arg_new_px, -1.0, 1.0)

这是 **纯数值安全措施** ，不改变物理结果。在精确算术下 :math:`A \cdot p_x` 严格在 :math:`[-1, 1]` 内， clip 不会触发。Xsuite 是 C 代码， ``asin(1.0000000001)`` 在 C 中返回有限值不抛异常，但 Python 需要显式保护。


K0=0 时 Wedge 退化为 YRotation 的证明
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Wedge 的** :math:`p_x'` **公式** ：

.. math::

  p_x' = p_x \cos\theta + (p_z - b_1 x) \sin\theta

令 :math:`b_1 = K_0 \chi \to 0` ：

.. math::

  p_x' = p_x \cos\theta + p_z \sin\theta

YRotation 的 :math:`p_x'` 公式为 :math:`p_x' = \cos\alpha \cdot p_x - \sin\alpha \cdot p_z` 。令 :math:`\alpha = -\theta` ：

.. math::

  p_x' = \cos\theta \cdot p_x + \sin\theta \cdot p_z

动量公式一致。

**位置** :math:`x'` ：

令 :math:`b_1 = 0` ， Wedge 的 :math:`x'` 化简为：

.. math::

  x' = x \cos\theta + \frac{2 x \sin\theta \cdot p_x'}{p_z' + p_z \cos\theta - p_x \sin\theta}

利用旋转正交性 :math:`p_z' = p_x \sin\theta + p_z \cos\theta` ，分母变为 :math:`2 p_z \cos\theta` ，最终得到：

.. math::

  x' = \frac{x \cdot p_z}{p_z'} = \frac{x \cdot p_z}{p_x \sin\theta + p_z \cos\theta}

与 YRotation 的位置公式一致。

**y 和** :math:`\zeta` ：

当 :math:`b_1 \to 0` 时， :math:`D \to 0` ， :math:`\theta + D \to \theta` ，所以 :math:`\Delta y = p_y \theta / b_1 \to \infty` 。

这意味着 :math:`y` 和 :math:`\zeta` 的公式在 :math:`b_1 \to 0` 时发散。因此 Wedge 的 :math:`b_1 = 0` 极限不是连续的——当 :math:`|b_1| < \epsilon` 时，代码直接跳转到 YRotation 分支：

.. code-block:: python

  if abs(b1) < const.eps:
      # 直接调用 YRotation，跳过含 1/b1 的计算
      self._y_rotation_cpu(x, px, y, py, z, dp, tag, mask, theta, beta0)
      return

.. note::

  Wedge 在 :math:`b_1 = 0` 时通过 **代码分支结构** 退化为 YRotation。动量和位置的公式在 :math:`b_1 \to 0` 极限下一致，但 :math:`y` 和 :math:`\zeta` 含有 :math:`1/b_1` 因子，极限不连续，必须用分支处理。


边缘角正负号定义
----------------

::

  e1 > 0 (聚焦):

                    端面法线
                      ↗  e1
  轨迹法线  ↑        ↗
            |      ╱  端面
            |    ╱
  ──────────┼──╱──────────→ s (参考轨迹)
            |  ╱
            |╱

  端面法线相对于轨迹法线向弯转外侧倾斜


  e1 < 0 (散焦):

            |╲
            |  ╲
  ──────────┼──╲──────────→ s (参考轨迹)
            |    ╲  端面
            |      ╲
  轨迹法线  ↓        ╲  e1
                     ╲ 端面法线

  端面法线相对于轨迹法线向弯转内侧倾斜

边缘角 :math:`e_1` , :math:`e_2` 的正负号采用 MAD-X / Xsuite 约定：

- :math:`e_1 > 0` ：入口端面法线相对于参考轨迹法线向弯转外侧倾斜
- :math:`e_2 > 0` ：出口端面法线相对于参考轨迹法线向弯转外侧倾斜

常见磁铁类型：

.. list-table::
  :header-rows: 1
  :widths: 30 15 15 40

  * - 磁铁类型
    - :math:`e_1`
    - :math:`e_2`
    - 说明
  * - 扇形弯铁
    - 0
    - 0
    - 端面垂直于参考轨迹
  * - 矩形弯铁
    - :math:`\alpha/2`
    - :math:`\alpha/2`
    - :math:`\alpha` 为弯转角
  * - 通用
    - 任意
    - 任意
    - 用户指定


参数列表
--------

通用参数
~~~~~~~~

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
    - 元件长度（ :math:`L` ）
  * - ``name``
    - ``name``
    - str
    - -
    - 元件名称
  * - ``k0l``
    - ``k0l``
    - float
    - -
    - 归一化二极场积分（ :math:`K_{0L}` ）
  * - ``e1``
    - ``e1 (rad)``
    - float
    - rad
    - 入口边缘角（ :math:`e_1` ），默认 0
  * - ``e2``
    - ``e2 (rad)``
    - float
    - rad
    - 出口边缘角（ :math:`e_2` ），默认 0
  * - ``hgap``
    - ``hgap (m)``
    - float
    - m
    - 磁铁半气隙（ :math:`g_{\text{half}}` ），默认 0
  * - ``fint``
    - ``fint``
    - float
    - -
    - 入口边缘场积分（ :math:`F` ），默认 0
  * - ``fintx``
    - ``fintx``
    - float
    - -
    - 出口边缘场积分，默认 0（ :math:`\leq 0` 时自动设为 ``fint`` ）
  * - ``num_slice``
    - ``num slices``
    - int
    - -
    - 切片数，默认 1
  * - ``model``
    - ``model``
    - str
    - -
    - 物理模型，可选： ``adaptive`` （默认，自动选 ``rot-kick-rot`` ）、 ``rot-kick-rot`` 、 ``drift-kick-drift-exact``
  * - ``integrator``
    - ``integrator``
    - str
    - -
    - 积分器，可选： ``adaptive`` （默认 ``uniform`` ）、 ``uniform`` 、 ``yoshida4``
  * - ``aperture_type``
    - ``aperture type``
    - str
    - -
    - 孔径类型，默认 ``off``
  * - ``aperture_value``
    - ``aperture value``
    - list
    - -
    - 孔径参数值，默认 ``[]``

扩展参数（预留）
~~~~~~~~~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 35

  * - 属性名
    - JSON key
    - 类型
    - 单位
    - 说明
  * - ``is_field_error``
    - ``is field error``
    - bool
    - -
    - 是否启用磁场误差，默认 ``false``
  * - ``is_ramping``
    - ``is ramping``
    - bool
    - -
    - 是否启用磁场斜坡，默认 ``false``
  * - ``k0l_ramping_filepath``
    - ``k0l ramping filepath``
    - str
    - -
    - 磁场斜坡数据文件路径


使用示例
--------

扇形弯铁
~~~~~~~~

.. code-block:: json

  {
      "BEND1": {
          "S (m)": 10.0,
          "Command": "SBend",
          "Length (m)": 1.5,
          "K0L": 0.05,
          "Num Slices": 5,
          "Integrator": "yoshida4",
          "Aperture Type": "off"
      }
  }

扇形弯铁，端面垂直于参考轨道，无边缘效应。

矩形弯铁
~~~~~~~~

.. code-block:: json

  {
      "BEND2": {
          "S (m)": 20.0,
          "Command": "SBend",
          "Length (m)": 2.0,
          "K0L": 0.1,
          "E1 (rad)": 0.05,
          "E2 (rad)": 0.05,
          "HGap (m)": 0.02,
          "FInt": 0.5,
          "Num Slices": 10,
          "Integrator": "yoshida4",
          "Aperture Type": "off"
      }
  }

矩形弯铁，含边缘角和边缘场效应。弯转角 :math:`\alpha = K_{0L} = 0.1` rad，边缘角 :math:`e_1 = e_2 = \alpha/2 = 0.05` rad。

薄透镜弯转
~~~~~~~~~~

.. code-block:: json

  {
      "BEND3": {
          "S (m)": 30.0,
          "Command": "SBend",
          "Length (m)": 0.0,
          "K0L": 0.02,
          "Aperture Type": "off"
      }
  }

零长度二极铁，仅施加 :math:`K_{0L}` 薄透镜踢角，无 body 追踪，无边缘效应。


参考文献
--------

- Xsuite Physics Guide, Sec 1.10.3 (精确弯铁), Sec 1.10.9 (边缘场), Sec 1.10.10 (楔形), Sec 1.10.12 (四极楔形修正)
- Forest, E. et al., "Edge Focusing Effects in Sector Bending Magnets"
- Yoshida, H., "Construction of higher order symplectic integrators", Phys. Lett. A 150 (1990)
- MAD-NG fringe field implementation: https://github.com/MethodicalAcceleratorDesign/MAD
