孔径（Aperture）
====================

本模块介绍 PASS 中的 **孔径检查系统** （ Aperture ），用于在粒子追踪过程中检查粒子是否超出束流管的横向孔径边界。孔径检查是束流损失模拟的核心环节，能够标识并记录因横向坐标超出物理管道限制而丢失的粒子。

孔径模块位于 ``PASS/utils/aperture.py`` ，提供 CPU 和 GPU 两种实现 （分别通过 ``check_aperture_cpu`` 和 ``check_aperture_gpu`` 函数调用），在每次元件追踪后被自动调用。调用时传入当前元件的纵向位置 :math:`s` 和当前圈数。

孔径检查仅在粒子的横向坐标 :math:`(x, y)` 上进行，不涉及纵向坐标。每个元件可独立设置孔径类型和参数，支持 10 种孔径几何形状。


接口参数
--------

孔径系统通过两个参数控制：

.. list-table::
  :header-rows: 1
  :widths: 20 25 15 40

  * - 属性名
    - JSON key
    - 类型
    - 说明
  * - ``aperture_type``
    - ``Aperture Type``
    - str
    - 孔径类型，不区分大小写，可选值见下文
  * - ``aperture_value``
    - ``Aperture Value``
    - list
    - 孔径参数值，含义随类型而异

.. note::

  ``aperture_type`` 不区分大小写，内部统一转换为小写后匹配。 ``off`` 和 ``default`` 类型忽略 ``aperture_value`` 。


丢失粒子处理
------------

当粒子被判定为丢失时，系统执行以下操作：

- **tag 取负** ： :math:`\text{tag} \leftarrow -|\text{tag}|` ，保留粒子 ID 信息，仅将符号取负以标记为丢失
- **lost_position** ：记录丢失位置的纵向坐标 :math:`s` ，即当前元件的纵向位置
- **lost_turn** ：记录丢失时的圈数

已丢失的粒子 （ :math:`\text{tag} < 0` ）在后续孔径检查中将被跳过，不再重复标记。仅对存活粒子 （ :math:`\text{tag} > 0` ）执行孔径检查。


各孔径类型详解
--------------

以下逐一介绍 10 种孔径类型的参数定义与丢失判定条件。

off（关闭）
~~~~~~~~~~~~~~

**参数** ：无 （ ``aperture_value`` 被忽略）

**说明** ：不做任何孔径检查，所有粒子均保留。

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <circle cx="150" cy="150" r="60" stroke="#e94560" stroke-width="3" fill="none"/>
    <line x1="108" y1="108" x2="192" y2="192" stroke="#e94560" stroke-width="3"/>
    <text x="132" y="248" fill="#e94560" font-size="16" font-weight="bold" font-family="monospace">OFF</text>
  </svg>
  </div>


default（默认矩形）
~~~~~~~~~~~~~~~~~~~~~~

**参数** ：无 （ ``aperture_value`` 被忽略）

**说明** ：使用默认的 ±1m 矩形孔径，等价于 ``rectangle`` 类型且 ``aperture_value = [1.0, 1.0]`` 。

**丢失条件** ：

.. math::

  |x| > 1.0 \quad \text{或} \quad |y| > 1.0

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <rect x="90" y="90" width="120" height="120" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <text x="125" y="84" fill="#00d2ff" font-size="11" font-family="monospace">+1m</text>
    <text x="125" y="228" fill="#00d2ff" font-size="11" font-family="monospace">-1m</text>
  </svg>
  </div>


circle（圆形）
~~~~~~~~~~~~~~~~~

**参数** ： ``aperture_value = [r]`` ，其中 :math:`r` 为圆半径。

**丢失条件** ：

.. math::

  x^2 + y^2 > r^2

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <circle cx="150" cy="150" r="108" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <line x1="150" y1="150" x2="258" y2="150" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="195" y="142" fill="#f5a623" font-size="13" font-style="italic" font-family="monospace">r</text>
  </svg>
  </div>


rectangle（矩形）
~~~~~~~~~~~~~~~~~~~

**参数** ： ``aperture_value = [w, h]`` ，其中 :math:`w` 为半宽， :math:`h` 为半高。

**丢失条件** （满足任一即丢失）：

.. math::

  |x| > w

.. math::

  |y| > h

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <rect x="30" y="66" width="240" height="168" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <line x1="150" y1="150" x2="270" y2="150" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="200" y="142" fill="#f5a623" font-size="13" font-style="italic" font-family="monospace">w</text>
    <line x1="150" y1="150" x2="150" y2="66" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="156" y="112" fill="#f5a623" font-size="13" font-style="italic" font-family="monospace">h</text>
  </svg>
  </div>


ellipse（椭圆）
~~~~~~~~~~~~~~~~~

**参数** ： ``aperture_value = [a, b]`` ，其中 :math:`a` 为半长轴 （ x 方向）， :math:`b` 为半短轴 （ y 方向）。

**丢失条件** ：

.. math::

  \left(\frac{x}{a}\right)^2 + \left(\frac{y}{b}\right)^2 > 1

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <ellipse cx="150" cy="150" rx="120" ry="84" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <line x1="150" y1="150" x2="270" y2="150" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="200" y="142" fill="#f5a623" font-size="13" font-style="italic" font-family="monospace">a</text>
    <line x1="150" y1="150" x2="150" y2="66" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="156" y="112" fill="#f5a623" font-size="13" font-style="italic" font-family="monospace">b</text>
  </svg>
  </div>


rectcircle（矩形内切圆）
~~~~~~~~~~~~~~~~~~~~~~~~~~

**参数** ： ``aperture_value = [w, h, r]`` ，其中 :math:`w` 为矩形半宽， :math:`h` 为矩形半高， :math:`r` 为圆半径。

孔径区域为矩形与圆的 **交集** （粒子需同时在矩形和圆内才存活）。

**丢失条件** （满足任一即丢失）：

.. math::

  |x| > w \quad \text{或} \quad |y| > h

.. math::

  x^2 + y^2 > r^2

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <rect x="90" y="60" width="120" height="180" stroke="#e94560" stroke-width="1.5" fill="none" stroke-dasharray="5,3"/>
    <circle cx="150" cy="150" r="78" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <line x1="150" y1="150" x2="210" y2="150" stroke="#e94560" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="172" y="145" fill="#e94560" font-size="11" font-style="italic" font-family="monospace">w</text>
    <line x1="150" y1="150" x2="150" y2="60" stroke="#e94560" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="135" y="108" fill="#e94560" font-size="11" font-style="italic" font-family="monospace">h</text>
    <line x1="150" y1="150" x2="205" y2="107" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="165" y="128" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">r</text>
  </svg>
  </div>

.. note::

  红色虚线为矩形边界，蓝色实线为圆边界。孔径区域为两者的交集。


rectellipse（矩形内切椭圆）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**参数** ： ``aperture_value = [w, h, a, b]`` ，其中 :math:`w` 为矩形半宽， :math:`h` 为矩形半高， :math:`a` 为椭圆半长轴 （ x 方向）， :math:`b` 为椭圆半短轴 （ y 方向）。

孔径区域为矩形与椭圆的 **交集** （粒子需同时在矩形和椭圆内才存活）。

**丢失条件** （满足任一即丢失）：

.. math::

  |x| > w \quad \text{或} \quad |y| > h

.. math::

  \left(\frac{x}{a}\right)^2 + \left(\frac{y}{b}\right)^2 > 1

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <rect x="90" y="60" width="120" height="180" stroke="#e94560" stroke-width="1.5" fill="none" stroke-dasharray="5,3"/>
    <ellipse cx="150" cy="150" rx="90" ry="78" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <line x1="150" y1="150" x2="210" y2="150" stroke="#e94560" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="172" y="145" fill="#e94560" font-size="11" font-style="italic" font-family="monospace">w</text>
    <line x1="150" y1="150" x2="150" y2="60" stroke="#e94560" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="135" y="108" fill="#e94560" font-size="11" font-style="italic" font-family="monospace">h</text>
    <line x1="150" y1="155" x2="240" y2="155" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="188" y="170" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">a</text>
    <line x1="155" y1="150" x2="155" y2="72" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="160" y="90" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">b</text>
  </svg>
  </div>

.. note::

  红色虚线为矩形边界，蓝色实线为椭圆边界。孔径区域为两者的交集。


racetrack（跑道形）
~~~~~~~~~~~~~~~~~~~~~

**参数** ： ``aperture_value = [w, h, a, b]`` ，其中 :math:`w` 为矩形半宽， :math:`h` 为矩形半高， :math:`a` 为椭圆端 x 方向半轴， :math:`b` 为椭圆端 y 方向半轴。

跑道形孔径由中间矩形和两端的半椭圆组成。椭圆端中心位于 :math:`(\pm w, 0)` 。

**生存条件** （满足任一即存活）：

矩形区域内：

.. math::

  |x| \le w \quad \text{且} \quad |y| \le h

椭圆端区域内 （ :math:`|x| > w` 时）：

.. math::

  \left(\frac{|x| - w}{a}\right)^2 + \left(\frac{y}{b}\right)^2 \le 1

**丢失条件** ：不在矩形内且不在椭圆端内。

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <path d="M 102 60 L 198 60 A 60 72 0 0 1 198 240 L 102 240 A 60 72 0 0 1 102 60 Z" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <circle cx="198" cy="150" r="3" fill="#f5a623"/>
    <circle cx="102" cy="150" r="3" fill="#f5a623"/>
    <line x1="150" y1="150" x2="198" y2="150" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="168" y="145" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">w</text>
    <line x1="150" y1="150" x2="150" y2="60" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="135" y="108" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">h</text>
    <line x1="198" y1="150" x2="258" y2="150" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="220" y="165" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">a</text>
    <line x1="198" y1="150" x2="198" y2="78" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="203" y="118" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">b</text>
  </svg>
  </div>

.. note::

  橙色圆点标记椭圆端中心位置 :math:`(\pm w, 0)` 。


octagon（八角形）
~~~~~~~~~~~~~~~~~~~

**参数** ： ``aperture_value = [w, h, d]`` ，其中 :math:`w` 为半宽， :math:`h` 为半高， :math:`d` 为半对角间隙 （切角距离）。

八角形为矩形切去 45° 角后得到的形状。 :math:`d` 越大，切角越大； :math:`d = 0` 时退化为矩形。

**丢失条件** （满足任一即丢失）：

.. math::

  |x| > w \quad \text{或} \quad |y| > h

.. math::

  |x| + |y| > w + h - d

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <polygon points="60,60 240,60 270,90 270,210 240,240 60,240 30,210 30,90" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <line x1="150" y1="150" x2="270" y2="150" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="200" y="145" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">w</text>
    <line x1="150" y1="150" x2="150" y2="60" stroke="#f5a623" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="156" y="108" fill="#f5a623" font-size="11" font-style="italic" font-family="monospace">h</text>
    <line x1="240" y1="60" x2="240" y2="90" stroke="#e94560" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="244" y="80" fill="#e94560" font-size="11" font-style="italic" font-family="monospace">d</text>
  </svg>
  </div>


polygon（多边形）
~~~~~~~~~~~~~~~~~~~

**参数** ： ``aperture_value = [[x1, y1], [x2, y2], ...]`` ，顶点列表，自动闭合 （最后一个顶点连接回第一个顶点）。

**丢失条件** ：点在多边形外部。

使用 **射线法** （ ray casting ）判断点是否在多边形内部：从待测点出发沿水平方向发射射线，统计与多边形边的交点数：

- 交点数为奇数 → 点在多边形内部 （存活）
- 交点数为偶数 → 点在多边形外部 （丢失）

.. raw:: html

  <div style="text-align: center">
  <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="300" fill="#1a1a2e"/>
    <line x1="20" y1="150" x2="280" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="150" y1="20" x2="150" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="285" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="156" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <polygon points="270,150 210,46 90,46 30,150 90,254 210,254" stroke="#00d2ff" stroke-width="2" fill="#00d2ff22"/>
    <circle cx="270" cy="150" r="4" fill="#f5a623"/>
    <circle cx="210" cy="46" r="4" fill="#f5a623"/>
    <circle cx="90" cy="46" r="4" fill="#f5a623"/>
    <circle cx="30" cy="150" r="4" fill="#f5a623"/>
    <circle cx="90" cy="254" r="4" fill="#f5a623"/>
    <circle cx="210" cy="254" r="4" fill="#f5a623"/>
    <text x="248" y="170" fill="#f5a623" font-size="10" font-family="monospace">P1</text>
    <text x="215" y="40" fill="#f5a623" font-size="10" font-family="monospace">P2</text>
    <text x="72" y="40" fill="#f5a623" font-size="10" font-family="monospace">P3</text>
    <text x="10" y="170" fill="#f5a623" font-size="10" font-family="monospace">P4</text>
    <text x="72" y="270" fill="#f5a623" font-size="10" font-family="monospace">P5</text>
    <text x="215" y="270" fill="#f5a623" font-size="10" font-family="monospace">P6</text>
  </svg>
  </div>


参数总览表
----------

.. list-table::
  :header-rows: 1
  :widths: 15 25 60

  * - 类型
    - aperture_value
    - 说明
  * - ``off``
    - 忽略
    - 不做孔径检查
  * - ``default``
    - 忽略
    - 默认 ±1m 矩形孔径
  * - ``circle``
    - ``[r]``
    - 圆形， :math:`r` 为半径
  * - ``rectangle``
    - ``[w, h]``
    - 矩形， :math:`w` 为半宽， :math:`h` 为半高
  * - ``ellipse``
    - ``[a, b]``
    - 椭圆， :math:`a` 为半长轴， :math:`b` 为半短轴
  * - ``rectcircle``
    - ``[w, h, r]``
    - 矩形与圆的交集
  * - ``rectellipse``
    - ``[w, h, a, b]``
    - 矩形与椭圆的交集
  * - ``racetrack``
    - ``[w, h, a, b]``
    - 跑道形 （矩形 + 椭圆端）
  * - ``octagon``
    - ``[w, h, d]``
    - 八角形 （矩形切 45° 角）
  * - ``polygon``
    - ``[[x1,y1], ...]``
    - 多边形顶点列表，自动闭合


使用示例
--------

以下 JSON 片段展示了各孔径类型的配置方式。孔径参数作为元件属性,
与 ``S (m)`` 、 ``Command`` 、 ``Length (m)`` 等字段并列:

**圆形孔径** :

.. code-block:: json

  "Drift1": {
      "S (m)": 10.0,
      "Command": "Drift",
      "Length (m)": 0.5,
      "Aperture Type": "circle",
      "Aperture Value": [0.1]
  }

**矩形孔径** :

.. code-block:: json

  "Drift2": {
      "S (m)": 10.5,
      "Command": "Drift",
      "Length (m)": 0.3,
      "Aperture Type": "rectangle",
      "Aperture Value": [0.06, 0.04]
  }

**椭圆孔径** :

.. code-block:: json

  "Drift3": {
      "S (m)": 11.0,
      "Command": "Drift",
      "Length (m)": 0.2,
      "Aperture Type": "ellipse",
      "Aperture Value": [0.06, 0.04]
  }

**跑道形孔径** :

.. code-block:: json

  "Drift4": {
      "S (m)": 11.5,
      "Command": "Drift",
      "Length (m)": 0.4,
      "Aperture Type": "racetrack",
      "Aperture Value": [0.03, 0.05, 0.02, 0.05]
  }

**八角形孔径** :

.. code-block:: json

  "Drift5": {
      "S (m)": 12.0,
      "Command": "Drift",
      "Length (m)": 0.3,
      "Aperture Type": "octagon",
      "Aperture Value": [0.05, 0.03, 0.01]
  }

**多边形孔径** :

.. code-block:: json

  "Drift6": {
      "S (m)": 12.5,
      "Command": "Drift",
      "Length (m)": 0.2,
      "Aperture Type": "polygon",
      "Aperture Value": [[0.05, 0.0], [0.025, 0.043], [-0.025, 0.043], [-0.05, 0.0], [-0.025, -0.043], [0.025, -0.043]]
  }

**关闭孔径检查** :

.. code-block:: json

  "Drift7": {
      "S (m)": 13.0,
      "Command": "Drift",
      "Length (m)": 0.5,
      "Aperture Type": "off"
  }