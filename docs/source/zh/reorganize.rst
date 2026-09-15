束团重分组（ReorganizeBunch）
==============================

本模块介绍 PASS 中的 **ReorganizeBunch** 命令。该命令在指定圈数切换束流的束团分组数，并按照粒子的物理到达相位重新建立束团结构。

**代码位置**

- 源文件： ``PASS/commands/reorganize.py``
- 分组算法： ``PASS/commands/sort_bunch.py``
- 类名： ``ReorganizeBunch`` （继承自 ``Command`` ）
- 注册名： ``reorganizebunch``
- Schema 类： ``ReorganizeBunchElement`` （ ``PASS/para/schema/elements.py`` ）


分组操作与边界
--------------

在 ``Start turn`` 将分组数改为 h。规定时钟相位
:math:`\Psi(t)=\int_{t_*}^t f_{rev}(u)du` 给出位置 s 处排序键：

.. math::

   k_i=\left[-\Psi(t_i)+s/C+1/(2h)\right]\bmod1,\qquad
   t_i=T_b-z_i/(\beta_b c).

分组 j 包含 :math:`j/h\le k_i<(j+1)/h` 的粒子，半槽位平移对奇偶分组数
采用同一规则。所有粒子数组一起重排。周期键只负责分类，展开的物理时间始终保留。

新参考事件由规定时钟、当前通过次数及新槽位 ID 确定。参考速度为
:math:`C f_{rev}(T_b)`，必须小于光速；参考能量由此速度与静质量确定。
参考能量改变属于坐标选择，不是 RF 踢：

.. math::

   z_i'=\beta_b'c(T_b'-t_i),\qquad
   p_{x,y}'=p_{x,y}P_{0,b}/P_{0,b}',\qquad
   1+\delta_i'=(1+\delta_i)P_{0,b}/P_{0,b}'.

物理时间、能量和机械动量保持不变。``SortBunch`` 使用同样的分类键与参考变换，
但保留已有束团参考参数。两者均使旧 SliceSet 失效，因为局部粒子索引已变化；
随后由用户显式执行 Slicer。不自动按质心重新居中，也不产生物理散束、合束或压缩。

接口参数
--------

.. list-table::
  :header-rows: 1
  :widths: 22 30 12 12 24

  * - 属性名
    - JSON key
    - 类型
    - 默认值
    - 说明
  * - ``s``
    - ``S (m)``
    - float
    - 必填
    - 命令在环中的纵向位置
  * - ``name``
    - ``name``
    - str
    - 自动填入
    - 命令名称
  * - ``start_turn``
    - ``Start turn``
    - int
    - 0
    - 执行圈数（含，0-based）；命令只执行一次
  * - ``new_harmonic``
    - ``New harmonic number``
    - int
    - 必填
    - 新的束团分组数，必须 :math:`\ge 1`


使用示例
--------

以下示例在第 500 圈把束流切换到 1 个纵向分组：

.. code-block:: json

  {
      "ReorganizeBunch1": {
          "S (m)": 0.0,
          "Command": "ReorganizeBunch",
          "Start turn": 500,
          "New harmonic number": 1
      }
  }


应用场景
--------

- 在 RF 操作改变纵向分布后，按新的束团中心网格更新诊断分组
- 在不同模拟阶段切换束团分组数
- 将跨越旧分组边界的粒子按当前机器时钟相位重新归类

.. note::

   ReorganizeBunch 只改变 PASS 的束团分组参考系，不会替代 RF 腔产生的散束、俘获、合束或束团压缩过程。应先通过相应物理元件形成所需纵向分布，再在合适圈数执行重分组。


参见 :ref:`zh-longitudinal-reference` 和 :doc:`slicer`。
