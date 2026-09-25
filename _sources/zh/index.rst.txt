PASS 使用手册
=============

PASS（Particle Accelerator Simulation Studio）用于六维粒子跟踪与束流动力学分析，
支持基于元件或 Twiss 传输映射的跟踪，以及注入、射频系统、集体效应与束流诊断。

使用 Python 时，从 :doc:`input_generation` 开始；使用图形界面时，参阅 :doc:`gui`。
设置束流参数前，请先阅读 :doc:`injection` 中的坐标与参考粒子约定。
各组件的用途、配置、物理模型和适用条件均在对应页面中说明。

.. toctree::
   :maxdepth: 1
   :caption: 入门与操作

   input_generation
   input_validation
   gui
   project_files
   gui_tools

.. toctree::
   :maxdepth: 1
   :caption: 束流与传输

   injection
   element/index
   twiss
   aperture
   reorganize

.. toctree::
   :maxdepth: 1
   :caption: 集体效应

   slicer
   space_charge
   field_solver
   wake_field

.. toctree::
   :maxdepth: 1
   :caption: 诊断与输出

   monitor/index
