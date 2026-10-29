Monitors and output
======================================

Select a monitor by the data needed for analysis. Monitor positions use the same s coordinate as lattice elements, and turn indices start at zero.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Monitor
     - Purpose
     - Recording interval
   * - :doc:`statmonitor`
     - Bunch moments and losses
     - Every turn
   * - :doc:`distmonitor`
     - Particle-distribution snapshots
     - Selected turns
   * - :doc:`particlemonitor`
     - Selected particle trajectories
     - Every turn in [start, end)
   * - :doc:`phaseadvancemonitor`
     - Single-particle fractional tunes
     - Completed analysis windows
   * - :doc:`slow_extraction_monitor`
     - Spill from named extraction events
     - Turn and/or physical-time bins

.. toctree::
   :maxdepth: 1

   table_output
   plotting
   statmonitor
   distmonitor
   particlemonitor
   phaseadvancemonitor
   slow_extraction_monitor

Empty slots and particle records
--------------------------------

The beam retains all harmonic slots. A slot with no allocated particle records
is skipped by bunch monitors and produces no bunch-specific output file.
This decision uses the allocated particle range, not the number of live particles.
Reserved records for later injection remain active; a bunch whose particles
have all been lost also keeps its loss diagnostics and zero-survival statistics.
If regrouping later fills an empty slot, monitors include it at their next
scheduled sample. ParticleMonitor and SlowExtractionMonitor keep their
beam-wide trajectory and extraction summaries; empty slots do not add files.

Once injected, a particle retains its identity ``abs(tag)`` through loss and
regrouping. Loss uses ``tag=-particle_id``, including at magnet fringes, so
trajectory records and injection snapshots remain associated with the original
particle.

.. _en-reference-state:

Output reference information
----------------------------

Distribution headers record ``ReferenceArrivalTime``, ``ReferenceBeta``,
``ReferenceMomentum``, ``CoordinateDefinition`` and the element-exit event.
Statistical monitors include the corresponding per-row reference values.
ParticleMonitor saves turn-by-turn reference values only with
``"Include reference": true``. Its HDF5 ``initial`` group or TFS initial rows
always capture the injection reference; those values cannot replace later
per-sample references. TFS keeps reference columns present with NaN values in
trajectory rows when this option is disabled.
Reconstruct live-particle time using the reference saved with the row,
not the reference at another turn. Loss coordinates are frozen diagnostic
records and must not be interpreted with a later live reference.

Use :ref:`en-longitudinal-reference` for the physical coordinate definitions and :doc:`table_output` for file formats and readers.
