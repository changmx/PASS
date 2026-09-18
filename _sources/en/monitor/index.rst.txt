Monitor
==================

This module introduces the various beam monitors supported in PASS.

.. toctree::
   :maxdepth: 2

   table_output
   statmonitor
   distmonitor
   phaseadvancemonitor
   particlemonitor

.. _en-reference-state:

Output reference information
----------------------------

Distribution headers record ``ReferenceArrivalTime``, ``ReferenceBeta``,
``ReferenceMomentum``, ``CoordinateDefinition`` and the element-exit event.
Statistical monitors include the corresponding per-row reference values.
ParticleMonitor saves these columns only with ``"Include reference": true``;
by default it stores no reference values in either columns or headers.
Reconstruct live-particle time using the reference saved with the row,
not the reference at another turn. Loss coordinates are frozen diagnostic
records and must not be interpreted with a later live reference.
