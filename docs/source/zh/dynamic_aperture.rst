动力学孔径扫描
==============

源码目录中的 ``example/07_dynamic_aperture/README.md`` 提供完整的输入生成、
追踪和绘图工作流。

动力学孔径（DA）由粒子追踪结果分析得到，不替换用户配置的 sequence，
也不自动插入、删除或调整物理效应。扫描可以包含元件、RF、集体效应和用户
配置的孔径。存活表示粒子在指定监测事件仍被观测为存活，有限圈追踪不能证明
无限时间稳定性。

初始坐标
--------

第一版生成器采用物理坐标的直角网格：

.. math::

   (x_i,\;0,\;y_j,\;0,\;z_0,\;\delta_k),\qquad
   N=N_xN_yN_{\delta}.

每个初始 ``dp`` 对应完整的 x-y 网格，所有组合在同一次模拟中追踪。
``px``、``py`` 和 ``z`` 默认均为零；Python 生成器和 schema 也允许指定固定值。
位置单位为米；``px``、``py`` 是以参考动量归一化的机械动量，
``dp=(P-P0)/P0``。这定义了横向相空间的一个二维切片，并不覆盖全部横向相位，
也不假定象限对称。

``PASS.utils.scan_grid.generate_scan_grid(x_values, y_values, dp_values,
px=0, py=0, z=0)`` 返回 ``coordinates``，列顺序为 x、px、y、py、z、dp；
``indices`` 保存 dp、y、x 索引。排列顺序为 dp 最外层、y 次之、x 变化最快。
每个轴的数值必须有限且不重复，纵向机械动量必须为实数且为正。

Injection 的束团可使用紧凑的 ``Scan Grid`` 配置：

.. code-block:: json

   {
     "X range (m)": [-0.01, 0.01],
     "Y range (m)": [-0.01, 0.01],
     "Number of x points": 41,
     "Number of y points": 41,
     "dp values": [-0.01, 0.0, 0.01],
     "px": 0.0,
     "py": 0.0,
     "z (m)": 0.0
   }

该对象放在束团的 ``Scan Grid`` 字段中，不属于 ParticleMonitor 参数。
它与 ``Insert Particle Coordinate``、``Insert Particle File`` 互斥。
明确坐标占用第一批注入中已经配置的宏粒子名额，不隐式增加束流粒子数。
第一批的数量至少应等于扫描点数；纯扫描时将其设为扫描点总数。

明确坐标在普通分布的偏移和色散处理之后写入，以入射参考定义。
之后的参考转换保持物理动量和到达时间，因此目标参考不同时，最终存储的
坐标数字可能改变。ParticleMonitor 单独记录注入时捕获的实际初值、
注入参考和注入圈数。DA 按保存的初始 ``dp`` 分组，不按 RF 等效应作用后的
最终 ``dp`` 分组。直接比较网格时应使用一致的注入参考。

ParticleMonitor 用粒子 ID 关联每一条捕获的初值。HDF5 的 ``initial`` 组中包含
``initial/dp``，TFS 则使用初始记录。DA 按保存的初始数值精确分组，不根据粒子
排列位置猜测，也不通过舍入划分区间。粒子重排、丢失后的负 tag 以及后续动量变化
不会改变其初始分组。因此连续随机的动量分布不会自动变成几个离散的 dp 扫描组。

所有宏粒子均参与配置的物理作用。有集体效应时，各 dp 组共同构成一个源分布，
所得存活区对应这个混合分布，不等同于分别模拟的单能束流。
改变网格密度或范围也可能改变源分布。``z=0`` 将粒子放在同一纵向坐标；
若该模型不适合所研究问题，应另行配置纵向分布。

记录与读取
----------

使用 :doc:`monitor/particlemonitor` 保留逐圈轨迹。
每个 beam、每个监测器写一个 HDF5 或 TFS 文件，使用有界写入缓冲。
大规模扫描推荐 HDF5：它保留数值类型，并能选择读取特定记录，不需要解析整个文本文件。
初始坐标与第一条轨迹记录分开保存，因为监测器前可能已经执行了物理元件。
对于 HDF5，DA 读取器只读取所需的轨迹行和初值数组，不将所有圈的坐标整体载入内存。
流式 HDF5 的 ``ValidSamples`` 指定已提交的前缀；未提交的尾部行被忽略。
任一必需数据集短于该前缀时，读取器会拒绝该不一致文件。

.. code-block:: python

   from PASS.analysis import read_dynamic_aperture, export_dynamic_aperture
   from PASS.plot.plot_dynamic_aperture import plot_dynamic_aperture

   result = read_dynamic_aperture("output/particle/run_particles.h5")
   ax = plot_dynamic_aperture(result, dp=0.0, mode="status", boundary=True)
   ax.figure.savefig("dynamic_aperture.png", dpi=180, bbox_inches="tight")
   export_dynamic_aperture(result, "dynamic_aperture.npz")

上例文件名是实际监测文件的占位示例。
``read_dynamic_aperture(path, requested_turn=None, cancel=None, clamp_to_available=False)`` 接受一个当前
ParticleMonitor HDF5 或完成导出的 TFS 文件，标识为 ``Layout="single_file"``、
``FormatVersion=2``。同一文件包含粒子 ID、捕获的注入坐标和注入圈，以及轨迹样本。
在脚本中明确指定此文件，GUI 则通过 **ParticleMonitor 文件** 选择它。

读取器不会把 PM 第一条轨迹记录当成注入初值。匹配使用 beam 范围内 ``abs(tag)``；
缺少初值或结果的粒子保持不可用。粒子记录和丢失不得早于已知注入圈。
这些检查拒绝相互矛盾的事件，不推断未知注入时长或同一圈内的事件顺序。

Injection 单独保存的分布是最后一次注入事件时的当前束团快照；多圈注入中，
早期粒子可能已经运动。因此 DA 使用 PM 文件内捕获的出生坐标，
不需要另行提供初始快照，也不通过 DistMonitor 文件分析。

``requested_turn`` 是所选监测器处包含在内的模拟圈索引，默认依次取
``RequestedEndTurn-1``、``EndTurn-1`` 或最后实际记录的圈。
这使提前结束的运行被标为覆盖不足，而不会静默降低目标追踪范围。
监测位置和执行顺序仍决定该样本之前已经发生的物理作用。例如，最后一圈开头
的监测器尚未观测到该圈剩余的作用。结束清理不补造样本，也不自动把圈索引 N
解释为已经完成 N 个回旋周期。

分析、绘图与导出
----------------

数组函数 ``compute_dynamic_aperture`` 接受形状为 ``(particles, 6)`` 的初值、
已对齐的 ``tag`` 数组 ``(samples, particles)`` 和整数 ``sample_turn``。
可选的采样坐标为 ``(samples, particles, 6)``。其他参数包括 ``particle_id``、
``lost_turn``、``lost_position``、``injection_turn``、``initial_valid``、
``requested_turn`` 和 ``metadata``。函数在不晚于目标圈的最后样本上分类：

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 状态
     - 含义
   * - ``survived``
     - 目标圈的样本中粒子仍存活。
   * - ``lost``
     - 负 tag 表示粒子在采样事件或之前已丢失。
   * - ``invalid``
     - 提供的采样坐标非有限，或纵向机械动量非实数/非正。
   * - ``incomplete``
     - 有存活样本，但尚未覆盖目标圈。
   * - ``unavailable``
     - 缺少可用初值或粒子状态，或注入晚于目标圈。

数值有效性检查仅针对所选样本，不检查完整历史轨迹。
不提供采样坐标时，数组函数不能检查数值有效性。分析不修改 tag，也不添加
丢失模型。丢失圈数仍是模拟的绝对索引；不同圈注入的粒子不自动具有相同追踪时长。

结果保留所有初始点、ID、状态、观测圈、丢失圈和位置、所选终态坐标、
初始 dp 分组和元数据。``plot_dynamic_aperture(result, dp=None,
mode="status", ax=None, boundary=True)`` 按一个精确的初始 dp 值选择分组，
以 mm 显示初始 x-y。``loss_turn`` 模式按绝对丢失圈着色，并单独区分其他状态。
不指定 dp 时选择第一个可用组。

默认在散点上叠加虚线孔径边界；``boundary=False`` 可关闭边界显示。
边界要求所选 dp 组构成完整、唯一的直角网格，且初始 ``px``、``py``、``z`` 固定。
按相邻存活/丢失值的中点插值过渡线，同时保留稳定岛及存活区内部的丢失点。
未知状态被遮罩，到达扫描边缘的曲线保持开放。图例标识采样孔径边界，不进行平滑或凸包包络。
这是所选网格分辨率和追踪范围下的边界估计，不是额外拟合出的物理接受度边界。
全部存活、全部丢失或没有可绘制的有效存活/丢失过渡时，图中说明原因；
不会把扫描矩形外沿当成 DA 边界。

完整直角网格可以只覆盖 ``y >= 0`` 的半平面或某个象限；“完整”指所选 x/y
数值的组合，不要求覆盖整个平面。每个轴至少需要两个不同数值才能绘制二维轮廓。
到达 ``y=0`` 的边界保持开放，不沿横轴补线闭合，也不镜像生成未追踪的负 y 点。
扫描边缘提示表示区域以外尚未分类；若 ``y=0`` 是有意选择的截取边缘，仅该边缘
有存活点不要求扩大扫描。推断未扫描的另一半平面需要单独确认模型的对称性。

``plot_dynamic_aperture_boundaries(result, dp_values=None, ax=None)`` 将全部初始
dp 组的边界叠加，也可以选择其中一部分。每组使用独立颜色、线型和图例；无法绘制
边界的组会明确标注。叠加模式只显示曲线，查看个别粒子的状态时切回单 dp 散点图。
外轮廓、内部丢失孔洞及分离的稳定岛均保留；存活区内的孤立丢失点仍形成孔洞，
不会被填补或作为噪声删除。所有边界对应同一个目标监测事件。

.. code-block:: python

   from PASS.plot.plot_dynamic_aperture import plot_dynamic_aperture_boundaries

   ax = plot_dynamic_aperture_boundaries(result)
   ax.figure.savefig("dynamic_aperture_dp_overlay.png", dpi=180, bbox_inches="tight")
   # 从结果中的精确 dp 值选择部分组。
   ax = plot_dynamic_aperture_boundaries(result, dp_values=result["dp_values"][:2])

``export_dynamic_aperture`` 导出所有粒子、所有初始 dp 组的分析结果，
不受 GUI 当前显示的分组限制：

* CSV 每个粒子一行，包含 ID、六维初值、初值有效性、注入圈、分类、观测圈、
  丢失圈及丢失位置。同名 JSON 文件保存分析元数据。
* NPZ 保存全部结果数组，包括所选终态坐标、dp 列表及 ``metadata_json``。
  使用 ``numpy.load`` 并设置 ``allow_pickle=False`` 即可读取。
  这是用于后续 Python 分析的未压缩 NumPy 归档。

这些文件保存分类后的点数据，不是全部逐圈轨迹的另一份副本，也不是边界顶点表。
分析其他圈仍需保留 ParticleMonitor 文件。图形导出保存当前 Matplotlib 图，
包含选定的 dp 分组和显示设置，格式为 PNG、PDF 或 SVG；不导出 Python 源码，
也不是 GUI 截图。大量散点在 PDF/SVG 中可能以栅格图层保存。
比较结果时应同时保存 sequence、参考、孔径、注入条件和目标追踪范围。

独立 Python 文件
----------------

将 ``example/07_dynamic_aperture/standalone/dynamic_aperture.py`` 复制到其他
目录或电脑即可使用。该文件需要 Python 3.11 及以上版本、NumPy、h5py 和
Matplotlib，不需要安装 PASS、Qt 或模拟后端。它支持相同的当前
ParticleMonitor HDF5 和完成导出的 TFS 文件、状态分类、孔洞、稳定岛、
半平面网格以及多 dp 边界。

.. code-block:: python

   from dynamic_aperture import read_dynamic_aperture, plot_dynamic_aperture_boundaries

   result = read_dynamic_aperture("run_particles.h5")
   ax = plot_dynamic_aperture_boundaries(result)
   ax.figure.savefig("da_overlay.png", dpi=180, bbox_inches="tight")

该模块也提供 ``compute_dynamic_aperture``、``plot_dynamic_aperture`` 和
``export_dynamic_aperture``，参数与原接口一致。导入模块不会选择或更改
Matplotlib 后端。单个初始 dp 可用 ``plot_dynamic_aperture(result, dp=0.0)``。
也可以直接运行：

.. code-block:: console

   python dynamic_aperture.py run_particles.h5 --overlay --output da_overlay.png

此文件从 GUI 使用的同一套读取、分析和绘图源码生成。维护者修改这些源码后，
运行 ``python tools/export_dynamic_aperture.py`` 更新，再用
``python tools/export_dynamic_aperture.py --check`` 检查一致性。
已复制到其他位置的文件是当时的版本，需要重新复制才能更新。

数值收敛
--------

长期 DA 对比应使用 ``float64``，并记录计算后端和粒子精度。
跨程序比较前，应对齐坐标归一化、元件顺序与强度、孔径边界约定以及监测事件。

混沌轨迹可能放大浮点舍入，进而改变丢失圈数，甚至有限圈数下的存活状态；
CPU 与 GPU 之间也可能出现这种差异。应先检查短程轨迹和单个元件映射，再用
微小初值扰动或独立的更高精度参考复核敏感点，并检查追踪圈数和网格分辨率。
``survived``、``lost`` 描述本次运行的实际观测，不估计数值不确定性。
平滑的绘图边界不会消除这种敏感性。

GUI 工作流
----------

在选定的 **Injection** 束团中配置 x-y 范围、点数、初始 dp 列表及固定坐标。
ParticleMonitor 只记录已配置的粒子；通过现有编辑器配置监测器和 sequence。
准备网格不会自动启动模拟。

在 **分析 → 动力学孔径** 中选择一个当前 ParticleMonitor 文件。
**终点圈数（-1 自动）** 按该监测器处从 0 开始的文件圈数填写，不是新的追踪时长。
例如记录圈号为 0 至 999 时，选 999 分析计划末次事件，选 499 分析较早的事件。
自动模式采用上文所述的计划截止圈；存活记录未覆盖该事件时仍标为覆盖不足。
手动输入超过文件最后已提交采样圈时，GUI 使用最后采样圈分析，同时修改输入框
并显示提示：例如文件最后记录为 999，输入 1100 会改为 999。图形和导出元数据中的
``requested_turn`` 使用实际分析圈，``requested_turn_input`` 保留原始输入，
``last_sample_turn`` 保存文件的最后已提交采样圈（空历史为 ``None``）。
此调整不需要再次读取粒子历史。
自动模式不进行此下调，避免将中断运行误认为完成；记录范围内缺失目标圈时仍按
覆盖不足处理。空历史没有可供调整的终点。
Python 和独立模块默认保留严格的指定圈语义；传入 ``clamp_to_available=True``
即可启用与 GUI 相同的手动输入上限处理，并从返回的元数据查看调整情况。
再选择初始 dp、
状态/丢失圈着色；**显示孔径边界** 默认开启，也可关闭。数值与图片导出复用 Python 脚本的同一组分析和
绘图函数。读取在后台执行，支持取消。
在 **绘图方式** 中选择 **多个 dp：边界叠加** 后，勾选需要比较的 dp 组；初始全选，
并提供全选与清空按钮。单 dp 的着色和边界开关仅用于单 dp 视图。
