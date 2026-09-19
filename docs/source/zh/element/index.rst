元件（Element）
================

本模块介绍 PASS 中支持的各类束线元件。

.. _zh-element-integration-precision:

积分计算精度
------------

对称四阶组合采用

.. math::

   S_4(h) = S_2(w_1 h) S_2(w_0 h) S_2(w_1 h), \qquad
   w_1 = \frac{1}{2-2^{1/3}}, \quad w_0 = 1-2w_1.

``PASS.utils.constants.const`` 统一定义 ``yoshida_z1`` 和 ``yoshida_z0``。
Python 使用双精度系数；CUDA 接收由同一来源生成的双精度编译期常量，
不为每个粒子重复计算立方根。GPU 分裂映射中的磁铁长度、组合子步长度及相应设备函数
参数均保持双精度，包括内部空间电荷节点之间的传输。多极踢动的增量累加
也保持双精度，避免长度在坐标更新之前被舍入为单精度。

该约定适用于弯铁、四极铁、六极铁、八极铁、多极铁、踢轨器，以及叠加
多极场的螺线管分裂映射。纯均匀螺线管使用均匀场精确映射。这些调整保持
原有映射、场强归一化和积分阶数。

``Particle Precision`` 仍决定六维坐标的存储精度，因此 FP32 表示存储精度，
不表示所有运算均为单精度。其余中间量类型取决于元件实现：例如静态 RKR
弯铁采用双精度工作坐标，而带内部空间电荷节点的 CPU FP32 RKR 传输仍将
各子映射结果舍入到实时单精度数组。CPU/GPU 结果不要求逐位相同。双精度
子步系数减少了一项可避免的误差来源，但不能消除坐标舍入和积分截断误差；
增加切片数不保证提高 FP32 精度。

.. toctree::
   :maxdepth: 2

   error
   marker
   drift
   dipole
   quadrupole
   sextupole
   octupole
   multipole
   solenoid
   kicker
   bump
   elseparator
   exciter
   rfcavity
