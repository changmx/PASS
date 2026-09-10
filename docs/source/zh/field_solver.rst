横向场求解器
============

概念介绍
--------

``PASS.commands.solver`` 提供 ``SpaceCharge`` 使用的数值计算层。它刻意不依赖
``Simulation`` 或 PASS 粒子类：PIC 调用者提供横向粒子数组、切片 ID、均匀网格和
单个宏粒子的电荷；解析求场则直接使用坐标、电荷与分布参数。因此同一套接口可以用于命令执行、自动测试和独立场计算。

- **代码位置**：``PASS/commands/solver/``
- **PIC 入口**：``PASS/commands/solver/pic.py``
- **底层 PIC 求解器标识**：``fd``、``dst_rectangle``、``fft_free_space``
- **解析跟踪入口**：``PASS/commands/solver/analytic.py``
- **数组顺序**：批量网格数据使用 ``(slice, y, x)``
- **计算后端**：CPU 使用 NumPy/SciPy；GPU 使用 CuPy 与 CUDA 库

数值层只负责横向场问题，不读取作用长度、束流磁刚度、相对论踢因子或模拟圈数；
这些属于 :doc:`space_charge` 的职责。

公开配置使用 ``Method`` 和 ``Solver``。``Method="pic"`` 时，
``fd_dirichlet``、``dst_dirichlet``、``fft_free_space`` 分别分派到底层
``fd``、``dst_rectangle``、``fft_free_space``。下文数值 API 中的标识是
``build_pic_resources`` 的底层参数。其中 ``fft_free_space`` 与公开 JSON 的
``Solver`` 值相同；FD 和 DST 在这两个接口中的标识不同。
``frozen`` 和 ``quasi-frozen`` 使用后文介绍的解析分布。

PIC 数据流
----------

一次调用依次执行：

.. code-block:: text

   粒子的 x、y、tag 和 slice_id
                |
                v
        CIC 或 TSC 电荷沉积
                |
                v
      Sigma[slice, y, x]，单位 C/m^2
                |
                v
          批量 Poisson 场求解
                |
                v
      Psi 单位 V m，积分 Ex/Ey 单位 V
                |
                v
        使用配对的 CIC 或 TSC 权重回插

全部纵向切片作为前导右端项一起求解。几何 mask、稀疏矩阵分解、谱特征值或 FFT
核在 ``PICResources`` 中只构建一次，之后重复使用。

网格与源项
----------

均匀节点网格
~~~~~~~~~~~~

``GridGeometry`` 描述均匀的节点型矩形网格。顶层空间电荷输入通过全宽
:math:`W_x` 和 :math:`W_y` 定义

.. math::

   x_{\min}=-\frac{W_x}{2},\quad x_{\max}=\frac{W_x}{2},
   \qquad
   \Delta x=\frac{W_x}{N_x-1},

.. math::

   y_{\min}=-\frac{W_y}{2},\quad y_{\max}=\frac{W_y}{2},
   \qquad
   \Delta y=\frac{W_y}{N_y-1}.

``Nx`` 和 ``Ny`` 是节点数而不是网格单元数，且都必须至少为 3。配置和
``build_grid_geometry`` 均只接受完整全宽或 ``Grid Half Width X/Y (m)`` 半宽
输入，半宽满足 :math:`W_x=2H_x,W_y=2H_y`。两组不能混用；拒绝 ``Dx``/``Dy``
间距或显式上下界映射输入。数值代码仍可直接构造 ``GridGeometry(...)``。

电荷沉积
~~~~~~~~

对每个切片 ID 有效的存活粒子，沉积方法把带符号电荷分配到有效场节点：

.. list-table::
   :widths: 18 20 25 37
   :header-rows: 1

   * - 方法
     - 每粒子节点数
     - 配对回插
     - 特点
   * - ``CIC``
     - 2 x 2
     - 双线性
     - 根据粒子所在网格单元使用分段线性权重。
   * - ``TSC``
     - 3 x 3
     - 二次
     - 使用更宽的二次权重，粒子—网格耦合更平滑。

沉积 stencil 会去除导体和边界节点，再逐粒子归一化剩余权重，从而守恒该粒子的
沉积电荷；回插过程使用相同的有效节点归一化。边界修改 stencil 时会记录 warning。
如果区域内粒子找不到任何有效节点，本次 PIC 调用将忽略它并回插零场，但不会修改
PASS 粒子 tag。

``tag <= 0``、切片 ID 无效，或位于网格/物理孔径之外的粒子也会被忽略。因此网格
和孔径尺寸应覆盖需要跟踪的粒子分布。
这些独立 PIC 函数不修改粒子 tag。``SpaceCharge`` 先调用共享孔径损失函数，
在沉积前将命令孔径上和孔径外的粒子标记为损失。此后存活的参与粒子若超出网格，
或没有有效 stencil 节点，则报错。初始化校验和命令约定见 :doc:`space_charge`。

Poisson 方程与单位
------------------

每个求解器读取单位为 C/m\ :sup:`2` 的沉积面密度 :math:`\Sigma_k`，并求解

.. math::

   -\nabla_\perp^2\Psi_k=\frac{\Sigma_k}{\epsilon_0},
   \qquad
   \mathcal E_{x,k}=-\frac{\partial\Psi_k}{\partial x},
   \qquad
   \mathcal E_{y,k}=-\frac{\partial\Psi_k}{\partial y}.

由于切片电荷已沿纵向积分，:math:`\Psi` 的单位是 V m，
:math:`\mathcal E_x,\mathcal E_y` 的单位是 V。这里返回的是积分场而不是 V/m
单位的平均场；``SpaceCharge`` 在计算踢之前用回插场除以该切片的 ``delta_z``。

场求解器选择
------------

.. list-table::
   :widths: 18 22 25 35
   :header-rows: 1

   * - 底层 ``field_solver``
     - 边界模型
     - 支持的孔径
     - 数值方法与适用场景
   * - ``fd``
     - 零 Dirichlet 导体
     - 完整网格矩形或任意受支持的连续孔径
     - 缓存稀疏 LU 分解。完整矩形使用规则五点差分；曲线或斜边界附近使用
       Shortley--Weller 距离。
   * - ``dst_rectangle``
     - 零 Dirichlet 导体
     - 仅完整且与网格对齐的矩形
     - 使用缓存特征值的 I 型离散正弦变换，是 ``fd`` 的矩形导体专用替代方案。
   * - ``fft_free_space``
     - 开放自由空间
     - 仅完整网格，不允许导体孔径
     - 使用缓存 Green 函数核的零填充 Hockney 线性卷积；适用于不考虑导体镜像
       电荷的情况。

有限差分：``fd``
~~~~~~~~~~~~~~~~~

完整矩形区域使用规则五点差分：

.. math::

   \left(\frac{2}{\Delta x^2}+\frac{2}{\Delta y^2}\right)\Psi_{i,j}
   -\frac{\Psi_{i-1,j}+\Psi_{i+1,j}}{\Delta x^2}
   -\frac{\Psi_{i,j-1}+\Psi_{i,j+1}}{\Delta y^2}
   =\frac{\Sigma_{i,j}}{\epsilon_0}.

外层网格节点固定为 :math:`\Psi=0`。显式孔径与完整网格相同时仍使用矩形求解器。
其他连续孔径使用
Shortley--Weller 求解器。如果相邻节点位于孔径外，规则步长会替换为有效节点沿
网格线到物理壁交点的实际距离，因此物理边界不是简单的阶梯状节点 mask。

稀疏矩阵和 LU 分解只构建一次；所有切片作为多列右端项交给同一个分解求解。

正弦变换：``dst_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dst_rectangle`` 在四条外网格边界上施加零电势。I 型离散正弦变换将同一个矩形
有限差分算子对角化。水平模 :math:`m` 和垂直模 :math:`n` 的特征值为

.. math::

   \lambda_{m,n}
   =\frac{4}{\Delta x^2}\sin^2\!\left(
      \frac{\pi m}{2(N_x-1)}\right)
    +\frac{4}{\Delta y^2}\sin^2\!\left(
      \frac{\pi n}{2(N_y-1)}\right).

变换只作用于两个横向轴，保留前导切片轴。该求解器无法表示曲线导体或网格内部的
较小导体边界。

自由空间 Green 函数：``fft_free_space``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fft_free_space`` 进行零填充线性卷积，从而避免未填充 FFT 的周期回绕。除自单元外，
卷积核为

.. math::

   G_\Psi(\mathbf r)
   =-\frac{1}{2\pi\epsilon_0}\ln\!\left(\frac{r}{r_0}\right),
   \qquad r_0=\sqrt{\Delta x\Delta y},

.. math::

   G_x(\mathbf r)=\frac{x}{2\pi\epsilon_0 r^2},
   \qquad
   G_y(\mathbf r)=\frac{y}{2\pi\epsilon_0 r^2}.

自单元的核值设为零。电势采用核参考，只在加性常数意义下确定；横向场才是主要
物理输出。该求解器描述开放自由空间，不描述接地束流管。

源电荷做一次正向实数 FFT；每个请求的输出做一次逆变换，因此只计算电场需要
两次逆变换，再计算电势则需要第三次。``FFTFreeSpaceSolver.solve`` 默认返回全部
三个输出；设置 ``compute_potential=False`` 时返回 ``potential=None`` 并跳过
电势变换。电势核仅在首次请求电势时建立并缓存。逆变换共用一个频谱工作数组；
返回数组只保留实际网格，释放较大的补零数组。

``solve_pic``、``pic_cpu`` 和 ``solve_poisson_fft_free_space`` 也提供这个可选参数。
FD 和 DST 仍计算电势，因为其电场需要电势梯度。``SpaceCharge`` 命令只在启用
``Save potential`` 且当前圈被选中输出时，请求 FFT 电势。

场梯度
~~~~~~

完整矩形 ``fd`` 和 ``dst_rectangle`` 通过网格有限差分计算
``E = -grad(Psi)``。Shortley--Weller ``fd`` 在壁面附近的有效节点使用不等距导数
系数；``fft_free_space`` 则直接与解析场核卷积。

孔径接口
--------

SC 命令定义损失孔径，并在 FD/DST 中同时定义导体壁，配置不再包含 ``Chamber``。
例如在命令内填写：

.. code-block:: json

   "Aperture type": "ellipse",
   "Aperture value": [0.04, 0.02]

内部转换为 ``{"Type": "ellipse", "Value": [0.04, 0.02]}`` 传给 FD 的
``aperture`` 参数。FFT 不接收导体孔径，命令孔径仅用于粒子损失。下表描述共享
底层几何构造器，尺寸单位均为 m。通用 ``default`` 与 SC 不同：SC 调用构造器前
将 default 解析为实际网格矩形，并禁止 Dirichlet 使用 ``off``。

.. list-table::
   :widths: 18 27 55
   :header-rows: 1

   * - 类型
     - ``Aperture Value``
     - 几何定义
   * - ``off``
     - 省略
     - 不设置独立物理孔径；对 ``fd``，外网格仍作为零电势边界。
   * - ``default``
     - 省略
     - 跟踪默认矩形 :math:`|x|\leq1`、:math:`|y|\leq1`。
   * - ``circle``
     - ``[R]``
     - 半径为 :math:`R` 的圆。
   * - ``rectangle``
     - ``[A, B]``
     - :math:`|x|\leq A`、:math:`|y|\leq B` 的矩形。
   * - ``ellipse``
     - ``[A, B]``
     - 椭圆 :math:`x^2/A^2+y^2/B^2\leq1`。
   * - ``rectcircle``
     - ``[W, H, R]``
     - 半宽/半高为 :math:`(W,H)` 的矩形与半径 :math:`R` 的圆的交集。
   * - ``rectellipse``
     - ``[W, H, A, B]``
     - 矩形 :math:`(W,H)` 与椭圆 :math:`(A,B)` 的交集。
   * - ``racetrack``
     - ``[W, H, A, B]``
     - 中央半宽/半高为 :math:`(W,H)`，两侧为半轴 :math:`(A,B)` 的水平
       椭圆端帽。
   * - ``octagon``
     - ``[W, H, D]``
     - 满足 :math:`|x|\leq W`、:math:`|y|\leq H` 和
       :math:`|x|+|y|\leq W+H-D` 的对称八边形。
   * - ``polygon``
     - ``[[x1,y1], ...]``
     - 至少三个有限顶点且面积非零的多边形。

底层还接受 ``circular``、``elliptic`` 和 ``rectangular`` 别名。Python 孔径
构造器也支持命名参数，但生成的输入文件建议使用上表格式。

对 ``dst_rectangle`` 和 ``fft_free_space``，孔径必须精确等价于完整且与网格对齐的
矩形，通常应设置为 ``null``。对 ``fd``，物理孔径应在所选网格中得到充分表示；
SC 初始化器会拒绝超出网格的有限 PIC 孔径，并拒绝与完整矩形不同的 DST 孔径，
因此不会将过大的命令孔径静默截断为网格导体。

Python 接口
-----------

网格与 PIC 流水线
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 26 27 47
   :header-rows: 1

   * - 接口
     - 主要参数
     - 返回值与行为
   * - ``GridGeometry(...)``
     - ``nx, ny, x_min, x_max, y_min, y_max``
     - 不可变均匀网格描述，提供 ``dx``、``dy``、``x`` 和 ``y`` 属性。
   * - ``build_grid_geometry(config, **kwargs)``
     - 节点数，加完整全宽或半宽输入
     - 构建中心为零的 ``GridGeometry``。
   * - ``build_aperture_mask(geometry, aperture)``
     - 网格和连续孔径映射
     - 返回节点是否属于孔径的布尔 mask。
   * - ``build_pic_resources(...)``
     - ``geometry``、可选 ``aperture``、``field_solver``
     - 返回可复用 ``PICResources``，包括网格、孔径、有效 mask 和缓存求解器。
   * - ``deposit_particles(...)``
     - particles、``slice_id``、网格、资源、方法和电荷
     - 分派到 CIC 或 TSC，返回 ``DepositResult``。
   * - ``solve_pic(...)``
     - particles、``slice_id``、网格、资源、方法、
       ``charge_per_macro``、``num_slices``、``compute_potential=True``
     - 沉积并批量求解全部切片，返回 ``PICResult``。
   * - ``gather_bilinear(...)``
     - 场、粒子、网格、资源、``slice_id``
     - 使用 CIC 权重回插一个或多个场。
   * - ``gather_quadratic(...)``
     - 场、粒子、网格、资源、``slice_id``
     - 使用 TSC 权重回插一个或多个场。
   * - ``pic_cpu(...)``
     - 数组 ``x``、``y``、``slice_id``、电荷、geometry 或 mesh，
       ``compute_potential=True``
     - ``solve_pic`` 的数组便捷接口。``delta_z`` 只用于推断切片数，不缩放场。

``particles`` 可以是映射，也可以是具有同形 ``x``、``y`` 数组及可选 ``tag``
的对象。``charge_per_macro`` 可以是有限标量或可广播到粒子形状的数组。
``slice_id`` 必须是每粒子一个元素的整数数组。显式提供 ``num_slices`` 可在结果中
保留尾部空切片。

求解器构建器与一次性封装
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 34 30 36
   :header-rows: 1

   * - 构建器
     - 可复用求解器
     - 一次性封装
   * - ``build_fd_resources(geometry)``
     - ``FDSolver``
     - ``solve_poisson_fd(...)``
   * - ``build_fd_arbitrary_resources(geometry, aperture)``
     - ``ArbitraryFDSolver``
     - ``solve_poisson_fd_arbitrary(...)``
   * - ``build_dst_rectangle_resources(geometry)``
     - ``DSTRectangleSolver``
     - 调用 ``solver.solve(density)``
   * - ``build_fft_free_space_resources(geometry)``
     - ``FFTFreeSpaceSolver``
     - ``solve_poisson_fft_free_space(...)``

重复计算时应优先使用构建器加 ``solver.solve``，以便复用缓存资源。

结果对象
--------

.. list-table::
   :widths: 22 22 18 38
   :header-rows: 1

   * - 对象 / 字段
     - 形状
     - 单位
     - 说明
   * - ``DepositResult.density``
     - ``(n_slice, ny, nx)``
     - C/m\ :sup:`2`
     - 沉积电荷密度。
   * - ``DepositResult.deposited_charge``
     - ``(n_slice,)``
     - C
     - 每个切片保留的电荷。
   * - ``DepositResult.deposited_count``
     - ``(n_slice,)``
     - -
     - 每个切片成功沉积的宏粒子数。
   * - ``FieldResult.potential``
     - 二维、三维或 ``None``
     - V m
     - 积分电势；FFT 只求电场而省略电势时为 ``None``。
   * - ``FieldResult.integrated_ex``、``integrated_ey``
     - 二维或三维
     - V
     - 横向积分场；``ex`` 和 ``ey`` 为兼容别名。
   * - ``PICResult``
     - 批量
     - 混合
     - 汇总密度、电势、场、网格、沉积电荷及沉积诊断。

场求解器接受 ``(ny, nx)`` 单切片数组，或 ``(n_slice, ny, nx)`` 批量数组。
所有值必须有限，末两维必须与求解器网格一致。

解析跟踪与参考场
----------------

``formula_*`` 模块提供自由空间解析积分场，用于 ``frozen``、``quasi-frozen``
跟踪，同时保留参考计算用途。公开 solver 名称为 ``gaussian_round_free_space``、
``gaussian_ellipse_free_space``、``uniform_round_free_space``、
``uniform_ellipse_free_space``。直接在粒子位置计算，不经过 PIC 流水线。

源电荷、坐标与单位
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

四种分布都求解单个带电切片的横向自由空间问题。以 Q 表示切片带符号总电荷（C），
(u, v) 表示源分布主轴坐标。命令先处理孔径损失，由当前存活且已分配到切片的粒子
计算 Q，再进行中心平移和旋转：

.. math::

   Q_k=N_k R Z e,\qquad
   \begin{pmatrix}u\\v\end{pmatrix}
   =\begin{pmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{pmatrix}
   \begin{pmatrix}x-c_x\\y-c_y\end{pmatrix},\qquad
   \begin{pmatrix}\mathcal E_x\\\mathcal E_y\end{pmatrix}
   =\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}
   \begin{pmatrix}\mathcal E_u\\\mathcal E_v\end{pmatrix}.

其中 R 为 ``bunch.ratio``，Z 为带符号电荷数，N_k 为切片存活宏粒子数。
下列密度均为纵向积分后的面电荷密度，单位 C/m\ :sup:`2`，在整个横向平面的积分
等于 Q。场是单位为 V 的积分场；``SpaceCharge`` 除以 ``delta_z`` 得到 V/m，
再单独施加相对论踢。公式函数本身不包含 ``delta_z`` 或 1/gamma² 因子。

分布中心和尺寸描述束流，不是管壁。自由空间公式不包含导体镜像场；命令孔径仅判断
粒子损失，缺省值为网格矩形。诊断采样不会截断或重新归一化解析模型。即使发生损失，
frozen 高斯仍使用指定的完整高斯形状与更新后的 Q，并非孔径截断高斯的精确场。

圆高斯：gaussian_round_free_space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``formula_gaussian_round.gaussian_round_field`` 使用单轴 RMS 尺寸 sigma，
对应 frozen 配置的 ``Sigma (m)``：

.. math::

   r^2=u^2+v^2,\qquad
   \Sigma(u,v)=\frac{Q}{2\pi\sigma^2}\exp\!\left(-\frac{r^2}{2\sigma^2}\right),
   \qquad
   \begin{pmatrix}\mathcal E_u\\\mathcal E_v\end{pmatrix}
   =\frac{Q}{2\pi\epsilon_0}
   \frac{1-\exp[-r^2/(2\sigma^2)]}{r^2}
   \begin{pmatrix}u\\v\end{pmatrix}.

原点处两个场分量均为零；乘在 (u, v) 前的标量因子极限为
Q/(4 pi epsilon_0 sigma²)，因此束心附近场为线性。实现使用
``-expm1(-r²/(2 sigma²))`` 避免相近数相减。远场按 1/r 衰减，径向场趋于
Q/(2 pi epsilon_0 r)，方向由 Q 的符号决定。

椭圆高斯：gaussian_ellipse_free_space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``formula_gaussian_ellipse.gaussian_elliptic_field`` 使用两个主轴方向的单轴
RMS 尺寸 sigma_u、sigma_v，对应 frozen 的 ``Sigma X/Y (m)``：

.. math::

   \Sigma(u,v)=\frac{Q}{2\pi\sigma_u\sigma_v}
   \exp\!\left(-\frac{u^2}{2\sigma_u^2}-\frac{v^2}{2\sigma_v^2}\right).

当 sigma_u > sigma_v 时，PASS 在第一象限计算 Bassetti--Erskine 公式，
其中 Faddeeva 函数使用 ``scipy.special.wofz``：

.. math::

   D=\sigma_u^2-\sigma_v^2,\quad
   U=|u|,\quad V=|v|,\quad
   z_1=\frac{U+iV}{\sqrt{2D}},\quad
   z_2=\frac{U\sigma_v/\sigma_u+iV\sigma_u/\sigma_v}{\sqrt{2D}},


.. math::

   g=\exp\!\left(-\frac{u^2}{2\sigma_u^2}-\frac{v^2}{2\sigma_v^2}\right),\qquad
   F=\frac{iQ}{2\epsilon_0\sqrt{2\pi D}}\,[w(z_1)-g\,w(z_2)],
   \qquad w(z)=e^{-z^2}\operatorname{erfc}(-iz),


.. math::

   \mathcal E_u=-\operatorname{sgn}(u)\operatorname{Re}F,\qquad
   \mathcal E_v=\operatorname{sgn}(v)\operatorname{Im}F.

sigma_u < sigma_v 时交换坐标、尺寸及返回的场分量；尺寸相等时退化为圆高斯，
数值上使用 ``np.isclose(sigma_u, sigma_v, rtol=const.eps, atol=0)`` 判断。
束心附近的一阶项为

.. math::

   \mathcal E_u\simeq\frac{Q u}{2\pi\epsilon_0\sigma_u(\sigma_u+\sigma_v)},\qquad
   \mathcal E_v\simeq\frac{Q v}{2\pi\epsilon_0\sigma_v(\sigma_u+\sigma_v)}.

实现还包含束心附近的三次修正；数值稳定性处理见下方 Python 接口表后的说明。

均匀圆盘：uniform_round_free_space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``formula_uniform_round.uniform_round_field`` 的 R_b 是束流外半径，
对应 ``Radius (m)``，不是 RMS 尺寸：

.. math::

   \Sigma(u,v)=
   \begin{cases}Q/(\pi R_b^2),&r\le R_b,\\0,&r>R_b,\end{cases}
   \qquad
   \begin{pmatrix}\mathcal E_u\\\mathcal E_v\end{pmatrix}
   =\frac{Q}{2\pi\epsilon_0}
   \begin{cases}
   R_b^{-2}\begin{pmatrix}u\\v\end{pmatrix},&r\le R_b,\\
   r^{-2}\begin{pmatrix}u\\v\end{pmatrix},&r>R_b.
   \end{cases}

束内场为线性，束外场按 1/r 衰减，在 r=R_b 处连续。该均匀圆盘的单轴 RMS
尺寸为 R_b/2。源分布边缘与命令物理孔径壁是不同概念；公式在分布边缘求值本身
不会将粒子标记为损失。

均匀椭圆：uniform_ellipse_free_space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``formula_uniform_ellipse.uniform_elliptic_field`` 使用束流半轴 a、b，
对应 ``Semi-axis A/B (m)``，单轴 RMS 尺寸分别为 a/2、b/2：

.. math::

   \eta=\frac{u^2}{a^2}+\frac{v^2}{b^2},\qquad
   \Sigma(u,v)=\begin{cases}Q/(\pi ab),&\eta\le1,\\0,&\eta>1.\end{cases}


.. math::

   \lambda=0\quad(\eta\le1),\qquad
   \frac{u^2}{a^2+\lambda}+\frac{v^2}{b^2+\lambda}=1,\quad\lambda>0\quad(\eta>1),


.. math::

   A=\sqrt{a^2+\lambda},\quad B=\sqrt{b^2+\lambda},\qquad
   \mathcal E_u=\frac{Q u}{\pi\epsilon_0 A(A+B)},\qquad
   \mathcal E_v=\frac{Q v}{\pi\epsilon_0 B(A+B)}.

束内 A=a、B=b，场为线性；束外 lambda 为共焦椭圆方程的正根。
实现使用避免相消的二次方程求根表达式，并采用上述有理化场表达式，保证 a、b
接近时的数值稳定性。场在分布边缘连续，a=b 时退化为均匀圆盘。

frozen 与 quasi-frozen 的参数选择
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``frozen`` 中，引用同一配置的全部切片共用固定中心、方向和尺寸；中心与角度
缺省为零，对应 solver 的尺寸必须填写。Q 和 ``delta_z`` 始终采用当前值，
所以冻结横向形状不等于冻结场强。

``quasi-frozen`` 在每次 kick 时重新计算各切片的总体矩：

.. math::

   \mathbf c_k=\frac{1}{N_k}\sum_{n\in k}\mathbf r_n,\qquad
   C_k=\frac{1}{N_k}\sum_{n\in k}(\mathbf r_n-\mathbf c_k)
   (\mathbf r_n-\mathbf c_k)^{\mathsf T}.


.. math::

   \sigma_u=\sqrt{\nu_1},\quad\sigma_v=\sqrt{\nu_2}
   \quad\text{(Gaussian ellipse)},\qquad
   a=2\sqrt{\nu_1},\quad b=2\sqrt{\nu_2}\quad\text{(uniform ellipse)},


.. math::

   \sigma=\sqrt{\frac{\operatorname{tr}C_k}{2}},\qquad
   R_b=2\sigma\quad\text{(round profiles)}.

特征值满足 nu_1 >= nu_2，nu_1 对应的特征向量确定长轴角度。统计分母使用
N_k 而非 N_k-1。圆化规则保留径向二阶矩，非圆粒子也可采用，但不会精确重建
非圆源的场。均匀模型近似均匀横向投影密度，并不把实际粒子变成 KV 分布。

空切片返回零场。非空 quasi-frozen 圆切片至少需要两个粒子且径向方差为正；
椭圆切片至少需要三个粒子，且 ``nu_2 > 64 * float64_epsilon * nu_1``。
无效统计量会报错。切片编号与宽度由用户独立提供，不检查 Slicer 执行或圈数历史。

直接调用公式示例
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下例先计算积分场，再换算切片平均场。输入坐标已位于源分布坐标系：

.. code-block:: python

   import numpy as np
   from PASS.commands.solver.formula_gaussian_ellipse import gaussian_elliptic_field

   x = np.linspace(-0.02, 0.02, 201)  # m, relative to the source center
   ex_integrated, ey_integrated = gaussian_elliptic_field(
       x, np.zeros_like(x), slice_charge=1e-9,
       sigma_x=0.003, sigma_y=0.002,
   )
   delta_z = 0.01  # m
   ex_average = ex_integrated / delta_z  # V/m

用于追踪时，以 ``Method="frozen"`` 或 ``"quasi-frozen"`` 配合对应公开
``Solver``；JSON 示例及命令孔径默认值见 :doc:`space_charge`。
``tests/integration/space_charge/test_analytic_free_space_tracking.py`` 使用
独立场积分核对真实粒子踢量。运行
``python -m tests.integration.space_charge analytic`` 可执行这些比较和重复 kick
参数演化检查，并生成对应图片。

Python 接口与数值稳定性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``solve_analytic(x, y, slice_id, valid, num_slices, charge_per_macro,
configuration)`` 按切片组织已分配且存活的粒子。frozen 参数来自配置，
quasi-frozen 参数来自各切片当前总体矩。``AnalyticResult`` 包含粒子长度的
``integrated_ex``/``integrated_ey``、切片电荷与计数，以及 ``(n_slice, 5)``
参数数组；列依次为中心 x、中心 y、尺寸 x、尺寸 y、角度。
尺寸是高斯 RMS 或均匀分布半轴。空切片参数为 NaN，场和电荷为零。
不读取模拟圈数或 Slicer 执行元数据。具体矩匹配规则见 :doc:`space_charge`。

``sample_analytic_grid(result, configuration, geometry)`` 在粒子场计算完成后，
采样诊断网格上的密度和场，返回 ``potential=None`` 的网格结果；当前命令拒绝解析
电势输出。采样网格不决定粒子 kick，也不截断解析电荷分布。

.. list-table::
   :widths: 32 34 34
   :header-rows: 1

   * - 函数
     - 分布参数
     - 返回值
   * - ``gaussian_round_field``
     - ``x, y, slice_charge, sigma``
     - 圆高斯分布的积分 ``(Ex, Ey)``，单位 V。
   * - ``gaussian_elliptic_field``
     - ``x, y, slice_charge, sigma_x, sigma_y``
     - Bassetti--Erskine 积分场，单位 V；
       ``gaussian_ellipse_field`` 为其别名。
   * - ``uniform_round_field``
     - ``x, y, slice_charge, radius``
     - 均匀圆切片束内外的积分场。
   * - ``uniform_elliptic_field``
     - ``x, y, slice_charge, a, b``
     - 均匀椭圆切片束内外的积分场。
   * - ``macro_charge_to_physical``
     - 真实粒子数、带符号电荷数
     - 带符号物理电荷，单位 C。

所有解析函数都支持标量或可广播坐标数组，尺寸参数必须为正有限值；未显式覆盖时
使用 PASS 常量中的 ``epsilon_0``。

均匀椭圆场使用共焦半轴 :math:`A=\sqrt{a^2+\lambda}`、
:math:`B=\sqrt{b^2+\lambda}`，按等价表达式
:math:`\mathcal E_x=Qx/[\pi\epsilon_0 A(A+B)]`、
:math:`\mathcal E_y=Qy/[\pi\epsilon_0 B(A+B)]` 计算。
椭圆内部 lambda 为零，外部使用非负共焦参数。这一有理化形式避免两半轴趋于相等时
的相消误差，也适用于 quasi-frozen 中接近各向同性的切片。

椭圆高斯公式在第一象限计算，再利用反射对称性恢复场分量符号，以避免下半平面
Faddeeva 函数的指数增长及大数相减，近圆束尤其需要这一处理。宽度平方差写为
``(sigma_x - sigma_y) * (sigma_x + sigma_y)``，保留已有圆束极限及坐标轴交换约定。
在束心附近，稳定半平面的两项仍可能几乎抵消；当
``(x/sigma_x)**2 + (y/sigma_y)**2 <= 1e-6`` 时使用场的三次展开，其相对截断误差
为该归一化半径平方的平方量级。

高效网格数建议
--------------

这里 ``N`` 是包含两端点的节点数，宽度 ``W`` 对应间距 ``h = W/(N-1)``。
首先确定物理范围和所需分辨率。下表是在各个规模附近便于选择的起点，并非所有
硬件上的绝对最快值；两个方向分别应用相应规则。

.. list-table:: 节点数建议
   :header-rows: 1
   :widths: 16 24 20 20 20

   * - 目标规模
     - FD 基准
     - DST 节点数
     - FFT 节点数
     - FFT 补零尺寸
   * - 128
     - 约 128
     - 129
     - 128
     - 256
   * - 256
     - 约 256
     - 257
     - 256
     - 512
   * - 512
     - 约 512
     - 513
     - 512
     - 1024
   * - 1024
     - 约 1024
     - 1025
     - 1024
     - 2048
   * - 2048
     - 约 2048
     - 2049
     - 2048
     - 4096

DST-I 的内部长度为 ``N-2``，对应逻辑变换长度 ``2*(N-1)``，因此
``N = 2**k + 1`` 是便利的取值系列。更一般地，``N-1`` 只有较小质因子时通常
更有利；二次幂不是唯一的快速长度。

FFT 格林函数卷积对每个方向补零至
``P = scipy.fft.next_fast_len(2*N-1)``。在表中规模下，选择 ``N = 2**k``
对应 ``P = 2**(k+1)``。邻近节点数也可能较快，应在分辨率相近时实测比较。
补零用于避免循环卷积回绕，不扩大物理网格范围。

FD 使用稀疏矩阵分解，没有二次幂尺寸的特殊优势。应选择满足几何与收敛要求的
最小节点数，例如 ``N >= ceil(W/h_max) + 1``；需要中心线上恰好有节点时可选
奇数。表中的 FD 列只是分辨率基准，2048×2048 的稀疏分解可能需要很大内存。
对于完整的接地矩形，DST 求解相同的离散泊松系统，且无需稀疏 LU 因子。
PASS 保留用户显式配置的节点数。

选择建议与限制
--------------

- 接地导体腔体使用 ``fd``，曲线、多边形或复合孔径尤其应选此方法。
- 完整矩形网格恰好就是接地导体腔体时，可使用 ``dst_rectangle``。
- 不需要导体镜像电荷的开放边界近似使用 ``fft_free_space``。
- 开放边界计算应增加网格范围，直到场对截断不敏感；所有求解器都应增加分辨率，
  直到场和踢的观测量收敛。
- CIC 更局部、计算量较小；TSC 使用更宽 stencil，耦合更平滑。沉积与回插方法
  必须配对。
- CPU 和 GPU 使用相同的边界模型与单位；浮点归约和稀疏分解不保证逐位一致。

GPU 资源与执行
--------------

在具备兼容 CUDA 工具包和驱动的环境中，运行
``python -m pip install --editable ".[cuda]"`` 安装依赖。
GPU 模块采用延迟导入，CPU 使用不依赖 CuPy。粒子沉积、回插与场处理使用
CuPy 设备数组和 ``RawKernel``；Python 主机端调用 cuDSS 绑定与 cuFFT plan，
不会在 CUDA kernel 内调用这些主机库接口。

每项功能的 CPU 和 GPU 实现位于同一个模块中。CUDA 源码内嵌在对应模块，
首次使用时编译；这些求解器不再使用独立的 ``gpu_*.py`` 实现文件或外部
``.cu`` 源码文件。原有 GPU 模块导入路径已移除。

.. list-table:: ``PASS.commands.solver`` 下的源码模块
   :header-rows: 1
   :widths: 25 75

   * - 模块
     - CPU 与 GPU 入口
   * - ``pic.py``
     - ``pic_cpu`` / ``pic_gpu``，资源构建、沉积与回插。
   * - ``fd_rectangle.py``
     - ``FDSolver`` / ``GPUFDSolver(geometry, dtype="float64")``。
   * - ``fd_arbitrary.py``
     - ``ArbitraryFDSolver`` / ``GPUArbitraryFDSolver(geometry, aperture, dtype="float64")``。
   * - ``dst_rectangle.py``
     - ``DSTRectangleSolver`` / ``GPUDSTRectangleSolver``，包括 cuFFTDx 编译。
   * - ``fft_free_space.py``
     - ``FFTFreeSpaceSolver`` / ``GPUFFTFreeSpaceSolver``。
   * - ``analytic.py``
     - ``solve_analytic`` / ``solve_analytic_gpu``。

``field_result.py`` 提供共享结果类型以及 GPU 缓冲区、编译工具。
空间电荷的两条执行路径都位于 ``PASS.commands.space_charge``，元件内部切片调度
位于 ``PASS.utils.slicing``。GPU 库仅在 GPU 入口内部导入，代码合并不会让
CPU 执行依赖 CUDA。

.. list-table:: GPU 求解实现
   :header-rows: 1
   :widths: 22 78

   * - 求解器
     - 常驻 GPU 的计算
   * - ``fd``
     - 初始化执行 cuDSS 分解，跟踪时批量求解多个稠密右端。
       完整矩形使用 SPD 模式；Shortley--Weller 的不等壁距可破坏矩阵对称性，
       因此使用 general 模式。
   * - ``dst_rectangle``
     - 通用 DST-I 使用奇延拓、实数 cuFFT，以及融合的打包、转置和归一化 kernel。
       合适的二次幂延拓还支持 cuFFTDx，将 y 方向正逆变换与特征值除法融合。
   * - ``fft_free_space``
     - 使用批量实数 cuFFT 线性卷积，缓存格林函数频谱，以小素因子尺寸补零，
       按有限切片批量控制工作区。一次源变换复用于两个场分量和可选电势。

``build_pic_resources_gpu`` 使用与 ``build_pic_resources`` 相同的几何和底层
求解器名称，另提供 ``dtype``、``num_slices``、``dst_implementation``、
``fft_batch_size`` 与 ``deposition_strategy``。默认精度为 ``float64``，
DST 自动择优，FFT 每批 16 个切片，沉积使用直接原子加。
建议初始化时提供 ``num_slices``。假定 ``x``、``y`` 和整数 ``slice_id``
已经是设备数组：

.. code-block:: python

   from PASS.commands.solver import (
       GridGeometry, build_pic_resources_gpu, pic_gpu, gather_fields_gpu,
   )

   grid = GridGeometry(513, 513, -0.02, 0.02, -0.02, 0.02)
   resources = build_pic_resources_gpu(
       grid, field_solver="dst_rectangle", dtype=x.dtype, num_slices=100,
   )
   result = pic_gpu(x, y, slice_id, 1.0e-15, geometry=grid,
                    resources=resources, num_slices=100, method="CIC")
   ex, ey = gather_fields_gpu(result.ex, result.ey, {"x": x, "y": y},
                             grid, resources, slice_id, method="CIC")
   resources.close()

数值数组和归约诊断量保留在 GPU。默认返回的网格数组由结果独立持有；
``copy=False`` 借用资源工作区，在相同资源的下一次调用后失效。
对于已校验输入，``validate=False`` 可跳过有限值检查，仍保留元数据检查。
显式指定 ``num_slices`` 可以避免读取设备上的最大切片编号。
复用要求串行调用、创建时的 device 和 stream，以及相同的几何、边界算子和精度。
``close()`` 释放工作区与 cuDSS 句柄，关闭后再次使用会报错。
切片数改变会重建批量工作区，但保留 FD 分解。

DST 自动模式先校验 cuFFT/cuFFTDx 数值一致性，再交错进行 CUDA event 计时；
cuFFTDx 中位耗时至少低 5% 才选用它。选择结果按设备、精度、网格尺寸与切片数
缓存在进程中。择优与首次 NVRTC 编译属于初始化工作。
也可显式选择 ``cufft``、``cufftdx`` 或普通 CUDA 二次幂实现 ``fused``。
cuFFTDx 头文件由 ``nvidia-mathdx`` 提供。自动模式遇到该可选实现的依赖或编译
问题时警告并保留 cuFFT；若数值检查失败则报错。
显式请求 cuFFTDx 时直接报告依赖或编译错误。

节点数包含边界：513 个节点对应 511 个内部节点和 1024 点 DST-I 延拓，
512 个节点对应 1022 点延拓。可优先比较 ``2**k + 1`` 节点。
融合实现目前支持 8 至 2048 点的二次幂延拓；其他合法尺寸使用 cuFFT，
不会修改配置中的网格数。应在实际目标显卡和精度下测量性能。

``deposition_strategy="warp"`` 合并 warp 内指向同一节点的贡献；
``sorted_warp`` 先按切片和网格单元排序临时粒子索引，再进行相同的合并。
两者均不改变粒子池顺序。性能比较必须包含排序成本，粒子集中本身并不保证排序
有收益。CIC/TSC 都对壁面保留 stencil 归一化，并采用匹配的回插权重。
即使粒子采用 FP32，几何和孔径判断仍使用双精度中间量，避免近壁粒子因舍入
落入不同节点。密度、电势、场与踢量保留配置指定的精度。

``analytic.solve_analytic_gpu`` 提供四种圆形/椭圆 Gaussian/uniform
分布的 frozen 与 quasi-frozen GPU 跟踪。切片中心矩和特殊函数中间量使用 FP64，
粒子场遵循配置精度。选定输出圈的解析场诊断网格采样可使用 CPU。
