JSON 输入文件检测
=================

一键全面检测
------------

在 **配置 → 执行序列** 中点击 **校验**，检查当前完整 JSON，包含全部命令、
命名资源和输入数据表。检测前先提交属性表单和源码修改；JSON 语法错误不会替换
当前配置，而是在源码中定位行列。全面检测在工作线程中执行，不生成粒子、
不构造泊松矩阵、不启动跟踪，也不创建仿真输出目录。

报告逐项显示 **错误**、**警告**、精确 JSON 路径、原因和稳定的规则编号。
可以按级别筛选，双击打开对应命令、bunch 或空间电荷资源，也可以复制或导出报告。
导出文件为普通 UTF-8 JSON，包含 ``valid``、``errors``、``warnings``、
``commands``、``checked_files`` 和 ``diagnostics``。路径采用 JSON Pointer 转义，
命令名称含标点时也能正确定位。

**错误会阻止运行。** 警告说明合法但可能不符合预期的行为，例如保存范围被裁剪、
文件或资源未启用、RF 表耗尽后沿用最后一行，以及监视器缓冲区较大。
警告不会额外要求确认，也不会阻止运行。关闭的配置内容也会检查：
已声明的参数不合法仍为错误；未启用功能所对应的文件缺失或内容不合法则为警告。

编辑过程中显示的是 **参数预检**，不会读取文件内容。
修改后，之前的全面检测状态会更新为参数预检；再次点击校验可获得最新完整报告。
运行前会重新检查所选的全部输入。双束流输入的圈数、后端、精度、GPU 设置和
Timing 必须一致，因为引擎共享这些设置。直接调用 ``PASS.main.main`` 也会在
初始化之前执行同一套全面检测。

检测范围
--------

* JSON 语法、对象根节点、重复键（包括大小写冲突）、有限数值、严格的数值/布尔/
  列表类型、未知参数和引擎必需字段。请使用当前导出的 schema 别名和 Command
  拼写；旧名称、Python 属性名和可被强制转换的字符串均不接受。
* 全局粒子种类、非零电荷、正的周长及过渡 gamma、圈数、后端、粒子精度、
  Timing 和 GPU 设备编号。
  显式规定时钟要求正频率，设计速度（回旋频率乘环长）小于光速；嵌套错误可定位到
  时钟/RF 时间表或 WakeField 求解组、分量字段。
* ``Sequence.injection`` 且 :math:`S=0`、连续 bunch 编号、分组数和唯一谐波 ID、
  正的动能、粒子数量、发射度、Twiss 参数、分布类型和动量/动能偏移互斥关系。
  同一束流的非空束团必须具有相同的“真实粒子数 / 宏粒子数”，空束团继承此固定权重。
  RF 谐波仍独立于 bunch 分组谐波，不要求相等或整数倍。
* 手动粒子的行形状和动量定义域。注入窗口为 ``T``、间隔为 ``I`` 时，
  注入次数为 :math:`M=\lceil T/I\rceil`，首次注入数量为
  :math:`\lfloor N/M\rfloor + N\bmod M`。手动坐标替换首次注入块内的粒子。
  还会报告在运行结束前无法完成的注入计划。
* 全部已注册元件、排序/重组命令和监视器：位置、长度、点操作长度要求、
  积分器/模型、多极系数、孔径尺寸和多边形自交、RF 接受范围、Exciter 的两种
  频率输入及调制分母、正的光学 beta、注入未结束时发生重组，以及监视器圈数窗口。
* Slicer 模型、显式范围、同名切片集配置冲突、空间电荷引用，以及按位置容差、
  命令优先级和稳定插入次序确定的实际执行顺序。SC 之前必须先执行 Slicer；
  若中间的 SortBunch/ReorganizeBunch 使切片结果失效，则必须重新执行 Slicer。
* 显式及元件内部 SC：方法/求解器/分布模型组合、成对网格范围、网格尺寸、
  孔径包含关系、DST 的完整矩形边界、Dirichlet 孔径内有效网格节点、
  解析方法不支持保存电势、仅 CPU 支持，以及内部 SC 的厚元件要求。
  覆盖检查复用跟踪阶段的周期区间算法，遵循 ``Coverage check`` 和 ``Coverage mode``。
* 分布表格（HDF5 或 TFS）、RF 和偏移 TFS 文件：存在性、解析、必需列、数值列类型、有限数据、
  分布行数和动量定义域、RF 正整数谐波、偏移时间严格递增以及圈数为整数。
  同一输入中的共享文件只读取一次。相对路径在校验和执行时都以 JSON 所在目录为基准；
  JSON 接受带 BOM 的 UTF-8 编码。

当前引擎尚未实现磁铁元件的 ramping 和 BeamBeam 命令。启用这些功能会报告错误，
避免设置被静默忽略；已实现的 RFCavity RF 数据表继续支持。
ElSeparator 必须在有限的 ``V (V)`` 与 ``VL (V m)`` 中恰好提供一个，
并提供正的间隙和有限隔板位置。零强度合法，非零 V 要求正长度，VL 也支持零长度薄冲量。
旧 ``Voltage (V)``、电极高度/中心、模式及独立 EX/EY/EXL/EYL 参数会报错，不推测缺失几何。

静态检测无法证明长期束流稳定性，也无法预先保证随跟踪变化的粒子状态。
例如粒子可能在后续运行中超出 PIC 网格，quasi-frozen 束流可能退化，
RF 跟踪也可能改变参考能量，因此运行时检查仍然保留。
GPU 驱动、实际可用内存、文件系统权限及数值收敛还取决于执行环境。
大缓冲区警告只是估算，不代表资源预留；匹配分布检查建立 RF 稳定性的必要条件，
采样及收敛仍须通过实际运行验证。

命令行与 Python
---------------

不需要安装 Qt 或打开 GUI：

.. code-block:: console

   python -m PASS.validation beam.json
   python -m PASS.validation beam0.json beam1.json --report validation-report.json

无错误时退出码为 ``0``，存在警告仍为 ``0``；检测失败时为 ``1``。
``--report`` 只写入指定的报告文件，不创建仿真输出。

.. code-block:: python

   from PASS.validation import validate_file, validate_input

   report = validate_file("beam.json")
   for issue in report.diagnostics:
       print(issue.severity, issue.code, issue.pointer, issue.message)
   if not report.ok:
       raise ValueError(report.text())

   # 对编辑器中的字典进行参数预检：
   report = validate_input(data, base_dir="inputs", check_files=False)

检测不会修改传入的字典，也不会重写输入文件。
