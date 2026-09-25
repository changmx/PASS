统计监视器（StatMonitor）
====================================

``StatMonitor`` 在指定晶格位置逐圈记录各束团的质心、标准差、RMS 发射度、统计 Twiss 参数及粒子损失。统计矩仅使用存活粒子（``tag > 0``）；每个监视器为各束团输出 CSV 及所选格式的统计历史。

配置示例
------------

.. code-block:: python

   from PASS.para.schema.monitors import StatMonitorItem
   from PASS.para.schema.sequence import Sequence

   sequence = Sequence()
   sequence.add("stat1", StatMonitorItem(s=0.0, write_interval_turns=100))

将该命令加入完整跟踪序列。``write_interval_turns`` 控制文件写入间隔，统计量仍逐圈记录。命令名称由序列键给定。

接口参数
--------

.. list-table::
  :header-rows: 1
  :widths: 20 20 10 10 40

  * - Python 字段
    - JSON key
    - 类型
    - 默认值
    - 说明
  * - ``s``
    - ``"S (m)"``
    - float
    - 必填
    - 监视器在束线中的纵向位置
  * - ``command``
    - ``"Command"``
    - str
    - ``"StatMonitor"``
    - 命令类型标识

  * - ``output_format``
    - ``"Output format"``
    - str
    - ``"hdf5-gzip1"``
    - ``"hdf5-gzip1"``（gzip-1 + shuffle）、``"hdf5"``（不压缩）或 ``"tfs"``；始终额外提供 CSV
  * - ``write_interval_turns``
    - ``"Write interval (turns)"``
    - 正整数
    - 100
    - 批量写入的圈数间隔，保留期间全部逐圈记录

.. note::

  统计对象为该位置处束团内的所有存活粒子（ ``tag > 0`` ），无需指定粒子编号。


输出文件
--------

每个束团每个监视器位置生成一对文件：

- **CSV** （分批追加） ： ``{hms}_stat_beam{bid}_bunch{bid}_Np_{Np}_s_{s:.4f}.csv``
- **HDF5** （默认，与 CSV 同批追加）： ``{hms}_stat_beam{bid}_bunch{bid}_Np_{Np}_s_{s:.4f}.h5``
- **TFS** （选择后代替 HDF5，结束时生成） ： ``{hms}_stat_beam{bid}_bunch{bid}_Np_{Np}_s_{s:.4f}.tfs``

输出目录为 ``output_dir_stat`` 。

元数据示例（HDF5 属性，文本模式下为 TFS 文件头）：

::

   @ Name             PASS Statistic Data
   @ Time             2026-07-14 00:11:03

输出列：

.. list-table::
  :header-rows: 1
  :widths: 25 15 60

  * - 列名
    - 分组
    - 说明
  * - ``turn``
    - 基本
    - 圈数
  * - ``xAverage``
    - 质心
    - 水平位置均值 :math:`\langle x \rangle`
  * - ``pxAverage``
    - 质心
    - 水平动量均值 :math:`\langle p_x \rangle`
  * - ``sigmaX``
    - 束流尺寸
    - 水平位置标准差 :math:`\sigma_x`
  * - ``sigmaPx``
    - 束流尺寸
    - 水平动量标准差 :math:`\sigma_{p_x}`
  * - ``yAverage``
    - 质心
    - 垂直位置均值
  * - ``pyAverage``
    - 质心
    - 垂直动量均值
  * - ``sigmaY``
    - 束流尺寸
    - 垂直位置标准差
  * - ``sigmaPy``
    - 束流尺寸
    - 垂直动量标准差
  * - ``zAverage``
    - 质心
    - 折叠后的束团相对纵向坐标均值 :math:`\langle z_{\mathrm{rel}}\rangle`
  * - ``dpAverage``
    - 质心
    - 动量偏差均值
  * - ``sigmaZ``
    - 束流尺寸
    - 折叠后的束团相对纵向坐标标准差
  * - ``sigmadp``
    - 束流尺寸
    - 动量偏差标准差
  * - ``xEmittance``
    - 发射度
    - 水平 2D 发射度 :math:`\varepsilon_x`
  * - ``yEmittance``
    - 发射度
    - 垂直 2D 发射度 :math:`\varepsilon_y`
  * - ``betax``
    - Twiss
    - 水平 beta 函数
  * - ``betay``
    - Twiss
    - 垂直 beta 函数
  * - ``alphax``
    - Twiss
    - 水平 alpha 函数
  * - ``alphay``
    - Twiss
    - 垂直 alpha 函数
  * - ``gammax``
    - Twiss
    - 水平 gamma 函数
  * - ``gammay``
    - Twiss
    - 垂直 gamma 函数
  * - ``invariantx``
    - 校验
    - 水平不变量 :math:`\gamma_x \beta_x - \alpha_x^2` （应等于 1）
  * - ``invarianty``
    - 校验
    - 垂直不变量（应等于 1）
  * - ``zCenter``
    - 纵向参考
    - 束团名义分组位置（并非物理质心） :math:`z_{\mathrm{center}}`
  * - ``referenceTime``
    - 参考量
    - 此次观测的参考通过时间（s）
  * - ``referenceBeta``
    - 参考量
    - 此次观测的参考速度与 c 的比值
  * - ``referenceMomentum``
    - 参考量
    - 此次观测的参考机械动量（离子按 eV/c/u 计）
  * - ``sigmaTime``
    - 束流尺寸
    - 由连续 z 计算的通过时间标准差（s）
  * - ``xzAverage``
    - 关联
    - :math:`\langle x \, z \rangle`
  * - ``xyAverage``
    - 关联
    - :math:`\langle x \, y \rangle`
  * - ``yzAverage``
    - 关联
    - :math:`\langle y \, z \rangle`
  * - ``xzDevideSigmaxSigmaz``
    - 关联
    - :math:`\langle x \, z \rangle / (\sigma_x \, \sigma_z)` 归一化未中心化交叉矩，并非 Pearson 相关系数
  * - ``beamLossTotal``
    - 损失
    - 丢失粒子数
  * - ``lossPercent``
    - 损失
    - 损失百分比
  * - ``xSkewness``
    - 高阶矩
    - 水平偏度
  * - ``xKurtosis``
    - 高阶矩
    - 水平峰度
  * - ``ySkewness``
    - 高阶矩
    - 垂直偏度
  * - ``yKurtosis``
    - 高阶矩
    - 垂直峰度
  * - ``Ek``
    - 能量
    - 束团动能


输出中的纵向坐标
----------------

CPU 和 GPU 的纵向矩均使用束团相对 z 在 ``[-C/2,C/2)`` 的临时整环代表值，
不会修改存储的粒子 z。HDF5 属性和 TFS header 记录 ``ZCoordinate=z_rel_folded_by_ring``
和 ``ZInterval``。这些量是所选代表值的矩，不是展开滑移统计或圆周统计；
分布跨越区间切口时，报告的宽度可能较大。需要累计纵向滑移时，可分析
ParticleMonitor 和 Distribution 保存的连续 z_rel。

注入粒子数
----------

numAlive、numInjected、numPending 分别记录存活、已注入（含损失）及预留宏粒子数。
beamLossTotal 不包含预留位置，lossPercent 以已注入粒子数为分母。
已分配的粒子群若无存活粒子，CPU/GPU 都输出零矩与明确的零存活计数；
声明为空的 bunch 仍沿用不输出行的行为。

批量写入、最后不足一批的处理、运行中查看 CSV 和 HDF5 结构见
:doc:`table_output`。
工作原理
--------

统计量计算
~~~~~~~~~~

对于束团中的 :math:`N` 个存活粒子（ :math:`\text{tag} > 0` ），各阶矩定义为：

一阶矩（质心）：

.. math::

   \langle x \rangle = \frac{1}{N} \sum_{i=1}^{N} x_i

二阶矩：

.. math::

   \langle x^2 \rangle = \frac{1}{N} \sum_{i=1}^{N} x_i^2

未中心化混合二阶矩：

.. math::

   \langle x \, p_x \rangle = \frac{1}{N} \sum_{i=1}^{N} x_i \, p_{x,i}

束流尺寸（标准差）：

.. math::

   \sigma_x = \sqrt{\langle x^2 \rangle - \langle x \rangle^2}

同理计算 :math:`\sigma_{p_x}`, :math:`\sigma_y`, :math:`\sigma_{p_y}`, :math:`\sigma_z`, :math:`\sigma_{\delta}` 。

实现中计算等价的中心矩，避免两个较大的原点矩相减。对于 FP32 和 FP64
粒子，CPU 与 GPU 均使用 FP64 累加：先以一个存活粒子为基准求质心，
再围绕质心累计各阶矩。这也改善了窄束团或偏心束团的协方差、发射度、
偏度与峰度的数值稳定性。粒子存储精度不变，z 统计的临时环周投影使用 FP64。

``sigmaZ`` 和 z 矩仍按临时环周代表值计算，不回写连续存储 z。``sigmaTime`` 使用未折叠 z 的标准差除以 :math:`\beta_b c`，表示实际通过时间展宽。输出逐行包含 ``referenceTime``、``referenceBeta``、``referenceMomentum``；名义 zCenter 不能用来重建实验室质心。

发射度与 Twiss 参数
~~~~~~~~~~~~~~~~~~~

由二阶矩导出 2D 发射度：

.. math::

   \varepsilon_x = \sqrt{\sigma_x^2 \, \sigma_{p_x}^2 - \sigma_{x,p_x}^2}

其中 :math:`\sigma_{x,p_x} = \langle x \, p_x \rangle - \langle x \rangle \langle p_x \rangle` 为协方差。

Twiss 参数：

.. math::

   \beta_x = \frac{\sigma_x^2}{\varepsilon_x}

.. math::

   \alpha_x = -\frac{\sigma_{x,p_x}}{\varepsilon_x}

.. math::

   \gamma_x = \frac{\sigma_{p_x}^2}{\varepsilon_x}

不变量校验：

.. math::

   \gamma_x \, \beta_x - \alpha_x^2 = 1

垂直方向（ y ）的公式形式完全相同，将下标 x 替换为 y 即可。

高阶矩
~~~~~~

偏度（三阶标准化矩）：

.. math::

   S_x = \frac{\langle x^3 \rangle - 3 \langle x \rangle \sigma_x^2 - \langle x \rangle^3}{\sigma_x^3}

峰度（四阶标准化矩）：

.. math::

   K_x = \frac{\langle x^4 \rangle - 4 \langle x \rangle \langle x^3 \rangle + 2 \langle x \rangle^2 \langle x^2 \rangle + 4 \langle x \rangle^2 \sigma_x^2 + \langle x \rangle^4}{\sigma_x^4}

束流损失
~~~~~~~~

.. math::

   N_{\text{loss}} = N_{\text{injected}} - N_{\text{alive}}

.. math::

   \text{loss\%} = \frac{N_{\text{loss}}}{N_{\text{injected}}} \times 100\%

其中 :math:`N_{\text{injected}}` 为已注入宏粒子数（存活与已损失粒子之和）， :math:`N_{\text{alive}}` 为当前存活粒子数。

数值精度与结果解释
------------------

CPU 与 GPU 均以 float64 累加中心矩，包括粒子坐标以 float32 存储的情况；粒子存储精度不变。这可减轻窄束团或偏心束团统计中的消减误差，但不能消除有限采样误差及跟踪误差。GPU 统计记录按给定写入间隔批量传回主机。

发射度为零时，对应的统计 Twiss 参数与不变量输出为零。发射度非零时，gamma*beta-alpha²=1 是定义导致的恒等式，不能独立证明跟踪正确。偏度与峰度使用中心化标准矩；这里的峰度不是减去 3 的超额峰度。非零质心会对 xzAverage 及其归一化输出产生贡献，因此它们本身不是中心化协方差。

参考时间与 z 定义见 :ref:`zh-longitudinal-reference`。批量写入、运行结束处理及实时查看方法见 :doc:`table_output`。
