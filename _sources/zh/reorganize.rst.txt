束团重整（ReorganizeBunch）
==============================

本模块介绍 PASS 中的束团重整命令 **ReorganizeBunch** ，用于在模拟过程中动态调整束团的分组方式。

**功能说明**

ReorganizeBunch 命令 **只修改束团的索引范围** （ ``start_idx`` 和 ``end_idx`` ）， **不修改任何粒子坐标** （x, px, y, py, z, dp, tag 等）。

.. note::

   本命令仅重整束团的索引分配，不涉及物理上的散束或合束。真正的散束（bunch lengthening）和合束（bunch compression）是通过调节高频腔的电压和相位等参数实现的物理过程，而非索引重分配。

适用场景：

  - **合并** ：将多个束团的索引合并为更少的束团。粒子的 z 坐标保持不变，只是索引范围重新分配。例如注入时谐波数 :math:`h=4` 产生 4 个束团，合并为 1 个束团
  - **拆分** ：将现有束团的索引重新分组为更多束团

**代码位置**

  - 源文件： ``PASS/commands/reorganize.py``
  - 类名： ``ReorganizeBunch`` （继承自 ``Command`` ）
  - 注册名： ``reorganizebunch``
  - Schema 类： ``ReorganizeBunchElement`` （ ``PASS/para/schema/elements.py`` ）


接口参数
--------

.. list-table::
  :header-rows: 1
  :widths: 20 30 10 10 30

  * - 属性名
    - JSON key
    - 类型
    - 单位
    - 说明
  * - ``s``
    - ``S (m)``
    - float
    - m
    - 命令在环中的 s 位置
  * - ``name``
    - ``name``
    - str
    - -
    - 命令名称，由序列键名自动填入
  * - ``mode``
    - ``Mode``
    - str
    - -
    - 操作模式，可选： ``merge`` （合并束团）、 ``split`` （拆分束团）
  * - ``start_turn``
    - ``Start turn``
    - int
    - -
    - 生效起始圈数（含，从 0 开始计数）
  * - ``end_turn``
    - ``End turn``
    - int
    - -
    - 生效结束圈数（不含）。设为 -1 表示不限制上限，持续到模拟结束
  * - ``new_num_bunch``
    - ``New num bunch``
    - int
    - -
    - 新的束团数量（必须 :math:`\ge 1` ）


物理说明
--------

ReorganizeBunch 命令的设计理念是： **粒子的物理位置由 tracking 过程决定，束团身份只是索引标签。**

粒子的 z 坐标随 tracking 自然演化（通过 RF 腔的能量调制和漂移中的滑相效应），不需要人为调整。ReorganizeBunch 只负责重新分配索引范围，使后续的诊断、切片等操作能正确识别新的束团结构。

索引分配方式
~~~~~~~~~~~~

总粒子数 :math:`N_{\text{total}}` 被尽可能均匀地分配到 ``new_num_bunch`` 个束团中：

.. math::

  N_k = \left\lfloor \frac{N_{\text{total}}}{n} \right\rfloor + \begin{cases} 1 & k < N_{\text{total}} \bmod n \\ 0 & \text{otherwise} \end{cases}

其中 :math:`n` 为新束团数， :math:`k = 0, 1, \ldots, n-1` 。前 :math:`N_{\text{total}} \bmod n` 个束团各多分一个粒子。


使用示例
--------

以下示例展示如何在第 500 圈将 4 个束团合并为 1 个束团：

.. code-block:: json

  {
      "ReorganizeBunch1": {
          "S (m)": 0.0,
          "Command": "ReorganizeBunch",
          "Mode": "merge",
          "Start turn": 500,
          "End turn": -1,
          "New num bunch": 1
      }
  }


应用场景
--------

  - **散束后合并索引** ：注入时使用高谐波数（如 :math:`h=4` ）产生多个束团，然后关闭或降低 RF 电压使束团自然散开。在散束完成后使用 ReorganizeBunch 将索引合并为 1 个束团
  - **再聚束后重新分组** ：散束后重新施加 RF 电压使粒子重新聚束，使用 ReorganizeBunch 将索引重新分组
  - **束团填充方案调整** ：在不同模拟阶段使用不同的束团分组方式
