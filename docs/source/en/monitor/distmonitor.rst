DistMonitor
===========

``DistMonitor`` saves a complete particle snapshot at selected turns and at
one lattice position. It is intended for post-processing the full bunch
distribution, including particles that have already been lost.

- **Code location**: ``PASS/commands/monitor/distribution.py``
- **Registered command**: ``"distmonitor"``
- **Output directory**: ``output_dir_dist`` (the ``distribution`` subdirectory
  of the run output)

Turn selection
--------------

``Save turns`` is a list of one-element or three-element lists. A one-element
list selects one zero-based turn. A three-element list is
``[start, end, step]`` and selects ``start, start + step, ...`` up to and
including ``end``. Multiple entries may be supplied; overlapping entries are
merged automatically.

For example:

.. code-block:: json

   "distribution_1": {
       "S (m)": 12.5,
       "Command": "DistMonitor",
       "Save turns": [[0], [100, 200, 10], [500, 1000, 100]]
   }

An empty list, ``"Save turns": []``, disables saving. Turns must be within
``[0, num_turns)`` and ``step`` must be positive. The monitor compiles the
selection into a byte table during initialization; checking the current turn
is a single bounds check and table lookup.

Interface
---------

.. list-table::
   :header-rows: 1
   :widths: 20 20 12 48

   * - Python field
     - JSON key
     - Type
     - Description
   * - ``s``
     - ``"S (m)"``
     - float
     - Longitudinal position of the monitor.
   * - ``command``
     - ``"Command"``
     - str
     - Must be ``"DistMonitor"`` (case-insensitive at runtime).
   * - ``save_turns``
     - ``"Save turns"``
     - list[list[int]]
     - Single turns ``[turn]`` or inclusive ranges ``[start, end, step]``.

The sequence key supplies the monitor name. With the high-level API, the
schema object can be used directly:

.. code-block:: python

   from PASS.para.schema.monitors import DistMonitor

   monitor = DistMonitor(s=12.5, save_turns=[[0], [100, 200, 10]])

Output
------

One TFS file is written for each selected turn and each bunch. The filename
contains the run time, beam and bunch identifiers, monitor position, monitor
name, and turn number. All particles in the bunch are written, regardless of
the sign of ``tag``.

The data columns are:

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Column
     - Unit
     - Description
   * - ``x``, ``px``, ``y``, ``py``
     - m or normalized momentum
     - Transverse phase-space coordinates.
   * - ``z``
     - m
     - Tracked bunch-relative coordinate ``z_rel``.
   * - ``dp``
     - -
     - Relative momentum deviation.
   * - ``tag``
     - -
     - Particle identifier; positive means alive and negative means lost.
   * - ``lost_turn``
     - -
     - Turn at which the particle was lost (``-1`` if it was not lost).
   * - ``lost_position``
     - m
     - Longitudinal loss position (``-1`` if it was not lost).

Headers include ``S``, command and monitor names, beam/bunch identifiers,
``Turn``, particle counts, backend and precision, PASS version, timestamp,
``ZCoordinate``, ``ZCenter``, and ``Circumference``. The ``z`` column is not
folded or shifted while saving; use ``ZCenter`` when reconstructing a lab-frame
coordinate.

CPU and GPU behavior
--------------------

On CPU, the monitor writes directly from the NumPy particle arrays. On GPU,
only the nine output fields are copied to host memory at a selected turn and
the host copy is passed to the TFS writer. No history buffer is retained
between turns, so memory use is proportional to one particle snapshot rather
than to ``num_turns`` snapshots.
