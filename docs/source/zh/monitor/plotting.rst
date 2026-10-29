自动绘图
========

将顶层 ``"Is plot figure": true``（Python 字段 ``is_plot=True``）开启后，
PASS 会在跟踪正常结束、监视器完成文件写入后读取结果并生成图片。默认值为
``false``。停止或失败的运行不会启动自动绘图。

绘图使用已保存的数据，同一入口也可以处理历史结果，无须重新跟踪。
这一步不需要 GUI、Qt 或 GPU。

对已有结果绘图
--------------

传入包含统计文件、``distribution/`` 和 ``particle/`` 子目录的单次结果目录，
而不是输出根目录：

.. code-block:: console

   python -m PASS.plot output/2026_1029/1036_59
   python -m PASS.plot output/2026_1029/1036_59 --format png pdf --dpi 300 --bins 80
   python -m PASS.plot output/2026_1029/1036_59 --output figures/review

相对路径以终端当前工作目录为基准，包含空格的路径需要加引号。
默认图片保存在所选运行目录内的 ``plot/``；``--output`` 只修改图片输出位置。

.. list-table:: 命令行选项
   :header-rows: 1
   :widths: 25 20 55

   * - 选项
     - 默认值
     - 含义
   * - ``--format``
     - ``png``
     - ``png``、``pdf`` 中的一种或多种。
   * - ``--dpi``
     - ``300``
     - PNG 分辨率，以及 PDF 中密集数据栅格图层的分辨率。
   * - ``--bins``
     - ``80``
     - 分布投影每个坐标轴上的分箱数。
   * - ``--output``
     - ``<run>/plot``
     - 图片及绘图记录的输出目录。

对应的 Python 调用返回绘图记录：

.. code-block:: python

   from PASS.plot.plot_main import plot_run

   manifest = plot_run(
       "output/2026_1029/1036_59",
       save_dir=None,
       formats=("png", "pdf"),
       dpi=300,
       bins=80,
   )

图片布局
--------

每份 :doc:`statmonitor` 历史数据按束流、束团、监视器生成一张 3 × 3 九联图。
每个已保存的 BPM 都有自己的统计图，不把多个 BPM 叠加成一张拥挤的曲线图。
九个子图各自保留 ``Turn`` 横轴标签及横轴刻度文字。

.. list-table:: StatMonitor 子图，按行排列
   :header-rows: 1
   :widths: 34 33 33

   * - 左
     - 中
     - 右
   * - :math:`\langle x\rangle,\langle y\rangle`
     - :math:`\sigma_x,\sigma_y`
     - :math:`\varepsilon_x,\varepsilon_y`
   * - :math:`\langle p_x\rangle,\langle p_y\rangle`
     - :math:`\langle z_{\mathrm{fold}}\rangle`
     - :math:`\sigma_{z,\mathrm{fold}}`
   * - :math:`\langle\delta\rangle`
     - :math:`\sigma_\delta`
     - 损失百分比

图片直接使用已保存的统计矩及损失百分比，不从部分粒子重新计算。
纵轴范围按各监视器数据自适应。全程为零的损失率仍正常绘制。

每份 :doc:`distmonitor` 快照按束流、束团、监视器、保存圈生成一张 2 × 3 六联图。
同时支持已保存的注入分布。

.. list-table:: 分布子图，按行排列
   :header-rows: 1
   :widths: 34 33 33

   * - 左
     - 中
     - 右
   * - :math:`x-p_x`
     - :math:`y-p_y`
     - :math:`z-\delta`
   * - :math:`x-y`
     - :math:`z-x`
     - :math:`z-y`

六个投影采用存活粒子（``tag > 0``）的二维直方图，共享色条表示每个网格内的
宏粒子数，不是物理电荷密度。每张图内的六个子图共享一个计数色标，
上限取这六个子图中的最大网格计数；不同保存圈各自确定色标。
同一束流、束团、监视器的不同保存圈仍共享坐标范围和分箱边界。
注入分布单独分组进行比较。
空快照及没有存活粒子的快照记为跳过。

每份 :doc:`particlemonitor` 文件生成一张 4 × 3 十二联图：

.. list-table:: ParticleMonitor 子图，按行排列
   :header-rows: 1
   :widths: 34 33 33

   * - 左
     - 中
     - 右
   * - :math:`x` 随圈数变化
     - :math:`y` 随圈数变化
     - :math:`z` 随圈数变化
   * - :math:`p_x` 随圈数变化
     - :math:`p_y` 随圈数变化
     - :math:`\delta` 随圈数变化
   * - :math:`x-p_x`
     - :math:`y-p_y`
     - :math:`z-\delta`
   * - :math:`x-y`
     - :math:`z-x`
     - :math:`z-y`

所有已记录的粒子 ID 和全部有效轨迹样本均参与绘制，不限制粒子数，也不对历史记录抽样。
前两行使用细线，投影使用小散点。通过 ``tag > 0`` 排除尚未注入及已损失记录；
无效样本会断开逐圈曲线。不生成冗长的逐粒子图例。
这些投影叠加了保存圈范围内的轨迹，而 DistMonitor 投影是某个时刻的分布快照。

坐标与风格
----------

图片采用白色背景、内向刻度，保留坐标标签和单位、必要的短图例及分布计数色条。
不显示标题、参数说明框，也不自动叠加 Twiss 椭圆或 RF bucket。
同图中的水平／垂直统计曲线采用蓝色／橙色；分布计数使用 ``jet`` 色图，空网格留白。

横向位置使用 mm，纵向位置使用 m，RMS 发射度使用 mm mrad。
已保存的横向动量为 :math:`p_x=P_x/P_0`、:math:`p_y=P_y/P_0`，
``dp`` 为 :math:`\delta=(P-P_0)/P_0`。这些量按 :math:`10^{-3}` 缩放显示，
不标成斜率或实际纵向动量。

StatMonitor 的 ``zAverage``、``sigmaZ`` 使用按周长折叠的坐标。
源文件元数据明确标识该约定时，轴标签包含 ``fold``；缺少这类元数据的独立 CSV
使用通用 ``z`` 标签。分布和粒子图保留已保存的连续束团相对坐标 :math:`z`；
绘图不加上 ``zCenter``，也不折叠这些坐标。
坐标约定见 :ref:`zh-longitudinal-reference`。

保留完整有效坐标范围，不平滑数据，也不按百分位裁剪。
PDF 中密集数据图层栅格化，坐标文字和刻度保留为矢量元素；
栅格化控制文件大小，不减少参与绘图的数据。

文件及诊断
----------

默认目录布局为：

.. code-block:: text

   <run>/plot/
       statistics/
       distribution/
       particle/
       manifest.json

图片文件名保留来源标识，包括源文件中的束流、束团、监视器及圈号等信息。
已有图片予以保留，再次绘图时使用未占用的文件名后缀，绘图记录也采用该规则
（``manifest_2.json`` 等）。
绘图记录包含输入来源、生成文件、绘图设置、跳过记录和失败信息。

支持已保存的 HDF5、TFS 表格，统计结果还支持 CSV。
同名文件仅处理一份，优先使用 HDF5 而不是配套文本副本。
绘图只读取原始结果，不修改它们，也不重新运行模拟。
单份输入绘图失败时会报告错误并继续处理其他输入。
自动绘图失败与跟踪成功状态分别报告；图片输出不完整时应查看绘图记录和日志。
