统计监视器（StatMonitor）
==========================

简介
----

``StatMonitor`` 是束流统计监视器，在指定纵向位置逐圈记录束团的统计量，包括质心位置、束流尺寸、发射度、 Twiss 参数、高阶矩及束流损失等。它是评估束流品质演变和诊断束流动力学行为的核心工具。

- **代码位置** ： ``PASS/commands/monitor/statistic.py``
- **类名** ： ``StatMonitor`` ，注册名 ``"statmonitor"``
- **核心特征** ：

  - 逐圈计算束团在 6D 相空间的统计量（一阶至四阶矩）；
  - 由二阶矩导出发射度和 Twiss 参数（beta、alpha、gamma）；
  - 记录束流损失数与损失百分比；
  - CPU 使用 numpy 向量化计算， GPU 使用 CUDA 核函数 + warp 归约；
  - 每圈数据追加写入 CSV ，最后一圈统一转换为 TFS 格式；
  - 仅统计存活粒子（ ``tag > 0`` ），已丢失粒子不计入。


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

协方差：

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

``sigmaZ`` 和 z 矩仍按临时环周代表值计算，不回写连续存储 z。新增 ``sigmaTime`` 使用未折叠 z 的标准差除以 :math:`\beta_b c`，表示实际通过时间展宽。输出逐行包含 ``referenceTime``、``referenceBeta``、``referenceMomentum``；名义 zCenter 不能用来重建实验室质心。

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

   N_{\text{loss}} = N_{\text{total}} - N_{\text{alive}}

.. math::

   \text{loss\%} = \frac{N_{\text{loss}}}{N_{\text{total}}} \times 100\%

其中 :math:`N_{\text{total}}` 为束团初始宏粒子数， :math:`N_{\text{alive}}` 为当前存活粒子数。

GPU 实现
~~~~~~~~

GPU 版本使用 CUDA 核函数 ``calc_all_stats`` ，采用 grid stride loop 遍历粒子，每个线程在寄存器中累加 22 个统计量，经 warp 归约（ ``__shfl_down_sync`` ）和 block 归约后，通过 ``atomicAdd`` 写入全局结果。块数上限为 512 （因 ``atomicAdd`` 竞争开销）。


接口参数
--------

.. list-table::
  :header-rows: 1
  :widths: 20 20 10 10 40

  * - 属性名
    - JSON key
    - 类型
    - 默认值
    - 说明
  * - ``s``
    - ``"S (m)"``
    - float
    - 必填
    - 监视器在束线中的纵向位置
  * - ``cmd_name``
    - ``"name"``
    - str
    - 必填
    - 监视器名称（由序列键名自动填入）
  * - ``command``
    - ``"Command"``
    - str
    - ``"StatMonitor"``
    - 命令类型标识

.. note::

  ``StatMonitor`` 无额外配置参数。统计对象为该位置处束团内的所有存活粒子（ ``tag > 0`` ），无需指定粒子编号。


输出文件
--------

每个束团每个监视器位置生成一对文件：

- **CSV** （逐圈追加） ： ``{hms}_stat_beam{bid}_bunch{bid}_Np_{Np}_s_{s:.4f}.csv``
- **TFS** （最后一圈由 CSV 转换） ： ``{hms}_stat_beam{bid}_bunch{bid}_Np_{Np}_s_{s:.4f}.tfs``

输出目录为 ``output_dir_stat`` 。

TFS 文件头：

::

   @ Name             PASS Statistic Data
   @ Time             2026-07-14 00:11:03

输出列（共 35 列）：

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
    - 束团实验室纵向中心 :math:`z_{\mathrm{center}}`
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
    - :math:`\langle x \, z \rangle / (\sigma_x \, \sigma_z)` 归一化关联
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


使用示例
--------

以下 JSON 片段在 :math:`s = 0.0` m 处放置一个统计监视器：

.. code-block:: json

   "SM1": {
       "S (m)": 0.0,
       "Command": "StatMonitor"
   }

统计监视器不需要额外参数，只需指定位置和命令类型。模拟运行过程中会逐圈记录该位置处束团的统计量。

多位置监视
~~~~~~~~~~

可在不同位置放置多个统计监视器，比较束流沿束线的统计量变化：

.. code-block:: json

   "SM_start": {
       "S (m)": 0.0,
       "Command": "StatMonitor"
   },
   "SM_mid": {
       "S (m)": 100.0,
       "Command": "StatMonitor"
   },
   "SM_end": {
       "S (m)": 250.0,
       "Command": "StatMonitor"
   }


应用场景
--------

- **束流品质评估** ：逐圈监测发射度、束流尺寸、质心位置的变化，评估束流品质是否稳定或退化
- **发射度测量** ：由二阶矩计算发射度和 Twiss 参数，与设计值对比验证
- **束流损失诊断** ：通过 ``beamLossTotal`` 和 ``lossPercent`` 监测束流损失率，定位损失发生的圈数和位置
- **非线性效应识别** ：通过偏度和峰度的高阶矩信息，判断束流分布偏离高斯分布的程度，识别非线性共振或色散耦合
- **动量展宽监测** ： ``sigmadp`` 和 ``sigmaZ`` 反映束团内部的纵向束流品质；跨束团比较绝对方位时还应使用 ``zCenter``
- **关联诊断** ： ``xzAverage`` 等关联量可用于诊断色散耦合或横向-纵向耦合

输出中的纵向坐标
----------------

CPU 和 GPU 的纵向矩均使用束团相对 z 在 ``[-C/2,C/2)`` 的临时整环代表值，
不会修改存储的粒子 z。TFS header 记录 ``ZCoordinate=z_rel_folded_by_ring``
和 ``ZInterval``。这些量是所选代表值的矩，不是展开滑移统计或圆周统计；
分布跨越区间切口时，报告的宽度可能较大。需要累计纵向滑移时，可分析
ParticleMonitor 和 Distribution 保存的连续 z_rel。

注入粒子数
----------

numAlive、numInjected、numPending 分别记录存活、已注入（含损失）及预留宏粒子数。
beamLossTotal 不包含预留位置，lossPercent 以已注入粒子数为分母。
已分配的粒子群若无存活粒子，CPU/GPU 都输出零矩与明确的零存活计数；
声明为空的 bunch 仍沿用不输出行的行为。
