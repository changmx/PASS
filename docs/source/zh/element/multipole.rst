多极铁（Multipole）
====================

本模块介绍 PASS 中的通用多极铁元件 **Multipole** ，用于模拟带电粒子在任意阶多极磁铁中的运动。与四极铁、六极铁、八极铁等单阶元件不同，多极铁通过 ``knl`` / ``ksl`` 数组同时支持任意阶（含混合阶）多极分量，适用于场误差注入、组合多极元件、高阶多极铁等场景。

PASS 中的多极铁支持 **厚元件** （ ``length > 0`` ）和 **薄透镜** （ ``length = 0`` ）两种模式，厚元件采用精确漂移-踢角-漂移（DKD-exact）辛积分方案，支持 uniform（2阶）和 yoshida4（4阶）两种辛积分器。踢角采用 Horner 嵌套求值，与 Xsuite ``kick_simple_single_coordinates`` 在公式层面完全一致。

**代码位置**

- 源文件： ``PASS/commands/element/multipole.py``
- 类名： ``Multipole`` （继承自 ``Command`` ）
- 注册名： ``multipole``
- 核心特征：

  - 支持任意阶多极分量（ ``knl`` / ``ksl`` 数组，最高阶由数组长度决定）
  - 支持正常分量（ ``knl`` ）和斜分量（ ``ksl`` ）及其组合
  - 支持薄透镜模式（ ``length = 0`` ，仅施加多极踢角）
  - 支持厚透镜模式（ ``length > 0`` ，DKD-exact 辛积分）
  - 支持 uniform（2阶蛙跳）和 yoshida4（4阶 Yoshida 组合）积分器
  - Horner 嵌套求值，向量化实现，无逐粒子分支
  - 零场（所有 ``knl`` / ``ksl`` 分量为零）时自动退化为纯漂移
  - 支持孔径检查
  - 单阶退化与四极铁/六极铁/八极铁逐粒子一致


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

荷质比因子：

.. math::

  \chi = \frac{q}{q_0} \cdot \frac{m_0}{m}

对于同种粒子束 :math:`\chi = 1` 。


多极磁场与归一化强度
--------------------

通用多极磁铁的磁场在横向平面内展开为泰勒级数。用复数表示：

.. math::

  B_y + i B_x = \frac{P_0}{q_0} \sum_{n=0}^{N} \frac{K_n}{n!} (x + i y)^n

其中 :math:`K_n` 为 :math:`n` 阶归一化多极强度（单位 :math:`\text{m}^{-n-1}` ）， :math:`N` 为最高阶数。 :math:`1/n!` 是泰勒展开的自然系数。

展开前几阶：

.. list-table::
  :header-rows: 1
  :widths: 10 15 25 50

  * - 阶数 :math:`n`
    - :math:`n!`
    - 元件类型
    - 磁场表达式
  * - 0
    - 1
    - 二极铁
    - :math:`B_y + i B_x = \frac{P_0}{q_0} K_0`
  * - 1
    - 1
    - 四极铁
    - :math:`B_y + i B_x = \frac{P_0}{q_0} K_1 (x + i y)`
  * - 2
    - 2
    - 六极铁
    - :math:`B_y + i B_x = \frac{P_0}{q_0} \frac{K_2}{2} (x + i y)^2`
  * - 3
    - 6
    - 八极铁
    - :math:`B_y + i B_x = \frac{P_0}{q_0} \frac{K_3}{6} (x + i y)^3`

归一化积分强度定义为：

.. math::

  K_{nL} = K_n \cdot L, \qquad K_{nsL} = K_{ns} \cdot L

其中 :math:`L` 为磁铁长度， :math:`K_{nL}` 为正常分量， :math:`K_{nsL}` 为斜分量。PASS 中用户通过 ``knl`` 数组指定 :math:`[K_{0L}, K_{1L}, K_{2L}, \ldots]` ，通过 ``ksl`` 数组指定 :math:`[K_{0sL}, K_{1sL}, K_{2sL}, \ldots]` 。

.. note::

  MAD-X 导出的 ``KNL`` / ``KSL`` 值与 PASS 的 ``knl`` / ``ksl`` 定义完全一致，均为积分强度 :math:`K_{nL}` ，可直接填入，无需手动计算阶乘。 :math:`1/n!` 由代码内部的 Horner 递推自动处理。


整体追踪流程
------------

根据磁铁长度，多极铁有两种追踪模式：

**薄透镜模式** （ :math:`L = 0` ）

::

  ====== 薄透镜 (length = 0) ======

  单次多极踢角 Kick(knl, ksl)
  [位置不变，仅动量跳变]

**厚透镜模式** （ :math:`L > 0` ）

::

  ====== 厚透镜 (length > 0) ======

  切片1 → 切片2 → ... → 切片N
  (每个切片: Drift(ds/2) → Kick(ds) → Drift(ds/2))

  其中 ds = L / N
  knl_eff = kn * ds, ksl_eff = ks * ds

  若所有 knl/ksl 分量为零：退化为单次精确漂移 Drift(L)

完整映射为：

薄透镜：

.. math::

  \mathcal{M}_{\text{thin}} = \text{Kick}(K_{nL}, K_{nsL})

厚透镜（N 个切片）：

.. math::

  \mathcal{M}_{\text{thick}} = \left[\mathcal{M}_{\text{DKD}}(\Delta s)\right]^N

其中每个切片的 DKD 映射为：

.. math::

  \mathcal{M}_{\text{DKD}}(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

.. note::

  - 薄透镜模式不改变粒子的位置坐标 :math:`(x, y, z)` ，仅施加动量踢角
  - 厚透镜模式的色品等效应通过精确漂移中的 :math:`p_z` 表达式自然引入
  - 当所有 ``knl`` / ``ksl`` 分量为零时，厚透镜退化为纯漂移，避免无意义的空踢角循环


物理推导
--------

哈密顿量
~~~~~~~~

在直线坐标系中（多极铁无曲率， :math:`h = 0` ），通用多极铁的哈密顿量为：

.. math::

  H_{\text{mult}} = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2} + \chi \sum_{n=0}^{N} \frac{K_n}{n!} \operatorname{Re}\left[(x - i y)^n\right]

其中求和项为势能部分。将其拆分为传播部分（精确漂移 :math:`H_D` ）和踢角部分（ :math:`H_K` ）：

.. math::

  H_D = \frac{p_\tau}{\beta_0} - p_z

.. math::

  H_K = \chi \sum_{n=0}^{N} \frac{K_n}{n!} \operatorname{Re}\left[(x - i y)^n\right]

踢角映射
~~~~~~~~

由哈密顿方程 :math:`\Delta p_x = -\frac{\partial H_K}{\partial x} \Delta s` ， :math:`\Delta p_y = -\frac{\partial H_K}{\partial y} \Delta s` ，对积分强度 :math:`K_{nL} = K_n \cdot \Delta s` 求得：

.. math::

  \Delta p_x = -\chi \sum_{n=0}^{N} \frac{K_{nL}}{n!} \operatorname{Re}\left[(x + i y)^n\right]

.. math::

  \Delta p_y = +\chi \sum_{n=0}^{N} \frac{K_{nsL}}{n!} \operatorname{Im}\left[(x + i y)^n\right]

其中 :math:`(x+iy)^n` 的实部对应正常分量，虚部对应斜分量。

.. note::

  复数场约定为 :math:`B_y + i B_x = \frac{P_0}{q_0} \sum_n \frac{K_n}{n!} (x+iy)^n` （ **不含共轭** ）。使用共轭 :math:`\overline{(x+iy)^n}` 会导致 :math:`\Delta p_y` 符号错误，此约定已通过六极铁交叉验证确认。


Horner 嵌套求值
----------------

多极踢角的核心是计算多项式：

.. math::

  P(z) = \sum_{n=0}^{N} c_n z^n, \qquad z = x + i y

其中 :math:`c_n = \chi \cdot K_{nL} / n!` 。直接展开高阶项计算量大且数值不稳定。PASS 采用 Horner 嵌套求值，与 Xsuite ``kick_simple_single_coordinates`` （ ``track_magnet_kick.h:182-228`` ）在算法层面一致。

Horner 递推从最高阶系数开始，逐步向下：

::

  index = order
  dpx_mul = chi * knl[order] / order!     # 最高阶系数
  dpy_mul = chi * ksl[order] / order!

  while index > 0:
      zre = dpx_mul * x - dpy_mul * y      # Re[(dpx_mul + i*dpy_mul) * (x + iy)]
      zim = dpx_mul * y + dpy_mul * x      # Im[(dpx_mul + i*dpy_mul) * (x + iy)]
      index -= 1
      dpx_mul = chi * knl[index] / index! + zre
      dpy_mul = chi * ksl[index] / index! + zim

  dpx = -dpx_mul    # px 取负（弧度约定）
  dpy = +dpy_mul    # py 不取负

其中 ``zre`` 和 ``zim`` 是复数乘法 :math:`(\text{dpx\_mul} + i \cdot \text{dpy\_mul}) \cdot (x + i y)` 的实部和虚部。

最终踢角为：

.. math::

  \Delta p_x = -\text{dpx\_mul}

.. math::

  \Delta p_y = +\text{dpy\_mul}

注意 :math:`\Delta p_x` 取负（弧度约定）， :math:`\Delta p_y` 不取负。

各阶展开结果
~~~~~~~~~~~~

将 Horner 递推展开，前几阶结果为：

.. list-table::
  :header-rows: 1
  :widths: 10 50 40

  * - 阶数
    - :math:`\Delta p_x` （正常分量）
    - :math:`\Delta p_y` （正常分量）
  * - :math:`n=0`
    - :math:`-\chi K_{0L}`
    - :math:`0`
  * - :math:`n=1`
    - :math:`-\chi K_{1L} \cdot x`
    - :math:`+\chi K_{1L} \cdot y`
  * - :math:`n=2`
    - :math:`-\chi K_{2L}/2 \cdot (x^2 - y^2)`
    - :math:`+\chi K_{2L} \cdot x y`
  * - :math:`n=3`
    - :math:`-\chi K_{3L}/6 \cdot (x^3 - 3xy^2)`
    - :math:`+\chi K_{3L}/6 \cdot (3x^2 y - y^3)`

斜分量的踢角通过复数乘法 :math:`i \cdot z^n` 自然交换实虚部：将上表中 :math:`\Delta p_x` 的正常分量公式移至 :math:`\Delta p_y` ，将 :math:`\Delta p_y` 的正常分量公式移至 :math:`\Delta p_x` 并取负。

.. note::

  Horner 递推对任意阶 :math:`N` 通用。当 ``knl`` / ``ksl`` 数组只有单阶非零分量时，多极铁退化为对应的单阶元件（四极铁/六极铁/八极铁等），踢角公式与硬编码版本逐粒子一致。


精确漂移映射
------------

漂移部分采用精确漂移（Table 1.1, map D, Eq. 1.86-1.88），与四极铁/六极铁/八极铁完全相同：

.. math::

  x \mathrel{+}= \frac{p_x}{p_z} L

.. math::

  y \mathrel{+}= \frac{p_y}{p_z} L

.. math::

  z \mathrel{+}= L \left(1 - \frac{\beta_0}{\beta} \cdot \frac{1+\delta}{p_z}\right)

其中：

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  \beta = \frac{(1+\delta) \beta_0 \gamma_0}{\sqrt{1 + \left[(1+\delta) \beta_0 \gamma_0\right]^2}}

精确漂移保留了 :math:`p_z` 的完整非线性，自然引入色品、高阶色散和路径长度效应。


辛积分器
--------

Uniform（2阶蛙跳）
~~~~~~~~~~~~~~~~~~

每个切片执行 Drift-Kick-Drift：

.. math::

  \mathcal{M}_{\text{DKD}}(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

这是2阶辛积分器，截断误差 :math:`O(\Delta s^2)` 。

Yoshida4（4阶组合）
~~~~~~~~~~~~~~~~~~~

将3个 DKD 步组合为4阶辛积分器：

.. math::

  \mathcal{M}_{\text{Y4}}(\Delta s) = \mathcal{M}_{\text{DKD}}(z_1 \Delta s) \circ \mathcal{M}_{\text{DKD}}(z_0 \Delta s) \circ \mathcal{M}_{\text{DKD}}(z_1 \Delta s)

其中 Yoshida 系数为：

.. math::

  z_1 = \frac{1}{2 - 2^{1/3}} \approx 1.3512

.. math::

  z_0 = 1 - 2 z_1 \approx -1.7024

截断误差 :math:`O(\Delta s^4)` 。


自然包含的效应
--------------

DKD-exact 对理想多极铁自然包含所有非线性效应，无需额外项：

.. list-table::
  :header-rows: 1
  :widths: 40 60

  * - 效应
    - 来源
  * - 自然色品
    - 精确漂移中 :math:`p_z` 的 :math:`\delta` 依赖性
  * - 高阶非线性色散
    - :math:`p_z` 的完整平方根表达式
  * - 路径长度效应（ :math:`R_{56}` 等）
    - 精确漂移的 :math:`z` 更新
  * - 各阶多极踢角的完整非线性
    - Horner 递推保留 :math:`(x+iy)^n` 的所有项

唯一近似来源是积分器的截断误差（uniform 为 :math:`O(\Delta s^2)` ，yoshida4 为 :math:`O(\Delta s^4)` ）。


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
    - ``s (m)``
    - float
    - 必填
    - 元件在束线中的纵向位置
  * - ``cmd_name``
    - ``name``
    - str
    - 必填
    - 元件名称
  * - ``length``
    - ``length (m)``
    - float
    - 必填
    - 磁铁长度， :math:`= 0` 为薄透镜
  * - ``knl``
    - ``KiL``
    - list
    - ``[]``
    - 正常分量积分强度数组 :math:`[K_{0L}, K_{1L}, \ldots]`
  * - ``ksl``
    - ``KiSL``
    - list
    - ``[]``
    - 斜分量积分强度数组 :math:`[K_{0sL}, K_{1sL}, \ldots]`
  * - ``num_slice``
    - ``num slices``
    - int
    - 1
    - 厚透镜切片数
  * - ``integrator``
    - ``integrator``
    - str
    - ``adaptive``
    - 积分器，可选： ``adaptive`` （默认 ``uniform`` ）、 ``uniform`` 、 ``yoshida4``
  * - ``aperture_type``
    - ``aperture type``
    - str
    - ``off``
    - 孔径类型
  * - ``aperture_value``
    - ``aperture value``
    - list
    - ``[]``
    - 孔径参数值

.. note::

  ``knl`` 和 ``ksl`` 数组长度不需要相同，短的数组自动补零。最高阶数 :math:`N` 由较长数组的长度决定（ :math:`N = \max(\text{len}) - 1` ）。


使用示例
--------

薄透镜多极铁（场误差注入）
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "MPE1": {
          "S (m)": 10.0,
          "Command": "multipole",
          "Length (m)": 0.0,
          "KiL": [0.0, 0.0, 0.001, 0.0005],
          "KiSL": [0.0, 0.0, 0.0003, 0.0001],
          "Aperture Type": "off"
      }
  }

零长度多极铁，含二阶和三阶场误差分量。用于模拟磁铁安装误差或加工误差对束流的影响。

厚透镜多极铁（组合元件）
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "MP1": {
          "S (m)": 20.0,
          "Command": "multipole",
          "Length (m)": 0.5,
          "KiL": [0.0, 0.3, 5.0, 200.0],
          "KiSL": [0.0, 0.0, 0.0, 0.0],
          "Num Slices": 5,
          "Integrator": "yoshida4",
          "Aperture Type": "off"
      }
  }

厚透镜组合多极铁，同时含四极、六极、八极正常分量，5 个切片，4 阶辛积分。

单阶多极铁（等价于八极铁）
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "MP2": {
          "S (m)": 30.0,
          "Command": "multipole",
          "Length (m)": 0.0,
          "KiL": [0.0, 0.0, 0.0, 500.0],
          "KiSL": [0.0, 0.0, 0.0, 200.0],
          "Aperture Type": "off"
      }
  }

仅含三阶分量（ ``knl=[0,0,0,500]`` ， ``ksl=[0,0,0,200]`` ），等价于一个正常+斜八极铁薄透镜。与 ``Octupole`` 元件逐粒子一致。

高阶多极铁（十极铁）
~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "MP3": {
          "S (m)": 40.0,
          "Command": "multipole",
          "Length (m)": 0.0,
          "KiL": [0.0, 0.0, 0.0, 0.0, 10000.0],
          "KiSL": [0.0, 0.0, 0.0, 0.0, 0.0],
          "Aperture Type": "off"
      }
  }

四阶多极铁（十极铁）， ``knl=[0,0,0,0,10000]`` 。专用元件只支持到八极（三阶），多极铁可支持任意阶。


应用场景
--------

- **场误差注入** ：将 MAD-X 导出的磁铁场误差以多极铁形式插入束线，模拟安装误差和加工偏差
- **组合多极元件** ：同一位置同时施加多个阶数的多极踢角（如四极+六极+八极组合）
- **高阶多极铁** ：十极铁（ :math:`n=4` ）、十二极铁（ :math:`n=5` ）等超出专用元件范围的高阶元件
- **非线性效应研究** ：研究高阶多极场对束流动力学的影响，如动态孔径、共振驱动
- **MAD-X 兼容** ： ``knl`` / ``ksl`` 定义与 MAD-X 完全一致，可直接导入 MAD-X 序列


参考文献
--------

- Xsuite Physics Guide, Sec 1.10.3 (精确漂移), Sec 1.10.5 (多极铁)
- Xsuite 源码： ``xtrack/beam_elements/elements_src/multipole.h`` , ``track_magnet.h`` , ``track_magnet_kick.h`` , ``track_magnet_drift.h``
- Yoshida, H., "Construction of higher order symplectic integrators", Phys. Lett. A 150 (1990)
- MAD-X 物理手册：多极磁场与非线性传输
- Wiedemann, H., "Particle Accelerator Physics", Ch. 4 (非线性束流动力学)
