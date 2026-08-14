束团重分组（ReorganizeBunch）
==============================

本模块介绍 PASS 中的 **ReorganizeBunch** 命令。该命令在指定圈数切换束流的束团分组数，并按照粒子的实验室纵向位置重新建立束团结构。

**代码位置**

- 源文件： ``PASS/commands/reorganize.py``
- 分组算法： ``PASS/commands/sort_bunch.py``
- 类名： ``ReorganizeBunch`` （继承自 ``Command`` ）
- 注册名： ``reorganizebunch``
- Schema 类： ``ReorganizeBunchElement`` （ ``PASS/para/schema/elements.py`` ）


功能说明
--------

设原束团分组数为 :math:`h_{\mathrm{old}}` ，新分组数为 :math:`h_{\mathrm{new}}` 。命令在 ``Start turn`` 指定的圈数执行一次，并完成：

1. 根据旧束团中心恢复每个粒子的实验室纵向位置

   .. math::

      z_{\mathrm{lab}} = z_{\mathrm{rel}} + z_{\mathrm{center,old}}.

2. 建立间隔为 :math:`C/h_{\mathrm{new}}` 的新束团中心网格

   .. math::

      z_{\mathrm{center},k} = k\frac{C}{h_{\mathrm{new}}},
      \qquad k=0,1,\ldots,h_{\mathrm{new}}-1.

3. 按环上的最近分组中心为粒子分组，并重排所有粒子数组，使每个新束团对应连续的索引范围。
4. 将实验室坐标重新写为新束团的相对坐标

   .. math::

      z_{\mathrm{rel,new}}
      = \operatorname{fold}_C
        \left(z_{\mathrm{lab}}-z_{\mathrm{center,new}}\right).

5. 更新束流的 ``harmonic_number`` 、每个束团的 ``harmonic_id`` 、 ``z_center`` 、粒子数和索引范围。
6. 若新束团继承了不同的参考动量，则重新归一化 :math:`p_x` 、 :math:`p_y` 和 :math:`\delta` ，以保持每个粒子的绝对机械动量不变。

因此，ReorganizeBunch 不只是修改索引，也不是直接执行物理意义上的散束、合束或压缩。粒子的实验室位置保持不变，但束团参考中心、相对纵向坐标和归一化动量可能改变。


分组边界
--------

算法使用以下环方位排序键：

.. math::

   k_z = \left(z_{\mathrm{lab}}+\frac{C}{2h_{\mathrm{new}}}\right)\bmod C.

第 :math:`j` 个分组包含满足

.. math::

   j\frac{C}{h_{\mathrm{new}}}
   \le k_z
   < (j+1)\frac{C}{h_{\mathrm{new}}}

的粒子。半个分组宽度的平移使边界位于相邻中心的中点，并对奇数、偶数分组数采用完全相同的规则。


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
- 将跨越旧分组边界的粒子按当前实验室方位重新归类

.. note::

   ReorganizeBunch 只改变 PASS 的束团分组参考系，不会替代 RF 腔产生的散束、俘获、合束或束团压缩过程。应先通过相应物理元件形成所需纵向分布，再在合适圈数执行重分组。
