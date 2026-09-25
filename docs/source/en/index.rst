PASS User Manual
================

PASS (Particle Accelerator Simulation Studio) provides six-dimensional particle
tracking using accelerator elements or Twiss transfer maps, with injection,
RF systems, collective effects, and beam diagnostics.

Start with :doc:`input_generation` for a Python workflow or :doc:`gui` for the
graphical interface. Before setting beam parameters, read the coordinate and
reference conventions in :doc:`injection`. Each component page describes its
purpose, configuration, physical model, and limits in one place.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   input_generation
   input_validation
   gui
   project_files
   gui_tools

.. toctree::
   :maxdepth: 1
   :caption: Beam and transport

   injection
   element/index
   twiss
   aperture
   reorganize

.. toctree::
   :maxdepth: 1
   :caption: Collective effects

   slicer
   space_charge
   field_solver
   wake_field

.. toctree::
   :maxdepth: 1
   :caption: Diagnostics and output

   monitor/index
