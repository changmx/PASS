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

An empty list, ``"Save turns": []``, disables saving. ``step`` must be positive.
If a range extends beyond the simulation, an end above the last turn is clipped
to ``num_turns - 1`` with a warning. A range whose start is at or beyond
``num_turns`` is ignored with a warning; a negative start is clipped to zero.
Malformed ranges and non-positive steps remain errors. The monitor compiles the
selection into a byte table during initialization; checking the current turn is
a single bounds check and table lookup.

Interface
---------

.. list-table::
   :header-rows: 1
   :widths: 20 22 12 10 36

   * - Python field
     - JSON key
     - Type
     - Default
     - Description
   * - ``s``
     - ``"S (m)"``
     - float
     - Required
     - Longitudinal position of the monitor.
   * - ``command``
     - ``"Command"``
     - str
     - ``"DistMonitor"``
     - Must be ``"DistMonitor"`` (case-insensitive at runtime).
   * - ``save_turns``
     - ``"Save turns"``
     - list[list[int]]
     - ``[]``
     - Single turns ``[turn]`` or inclusive ranges ``[start, end, step]``.
   * - ``include_injection_metadata``
     - ``"Include injection metadata"``
     - bool
     - ``false``
     - Append ``particle_id``, ``injection_turn`` and ``injection_batch``; requires the beam's Injection state when saving.
   * - ``output_format``
     - ``"Output format"``
     - str
     - ``"tfs"``
     - Accepts ``"tfs"`` (text, ``.tfs``) or ``"hdf5"`` (compressed datasets, ``.h5``).

The sequence key supplies the monitor name. With the high-level API, the
schema object can be used directly:

.. code-block:: python

   from PASS.para.schema.monitors import DistMonitor

   monitor = DistMonitor(s=12.5, save_turns=[[0], [100, 200, 10]])

To save injection information in HDF5 snapshots:

.. code-block:: python

   injection_monitor = DistMonitor(
       s=0.0,
       save_turns=[[0], [10, 100, 10]],
       include_injection_metadata=True,
       output_format="hdf5",
   )

The two options serialize as ``"Include injection metadata": true`` and
``"Output format": "hdf5"`` in the generated JSON.

Output
------

One TFS file (or HDF5 file when selected) is written per selected turn and bunch. The filename
contains the run time, beam and bunch identifiers, monitor position, monitor
name, and turn number. All born particles in the bunch are written, including
lost particles. Reserved slots with ``tag=0`` are omitted.

The nine default data columns are:

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
folded or shifted while saving. For live particles reconstruct passage time as
``t = ReferenceArrivalTime - z / (ReferenceBeta*c)``; ``ZCenter`` is grouping
metadata and does not reconstruct physical arrival time.

CPU and GPU behavior
--------------------

On CPU, the monitor writes directly from the NumPy particle arrays. On GPU,
the nine tracking fields are copied to host memory
and passed to the selected writer. No history buffer is retained
between turns, so memory use is proportional to one particle snapshot rather
than to ``num_turns`` snapshots.

Injection snapshots
-------------------

``Include injection metadata`` defaults to ``false``. Enabling it appends
three integer columns, giving twelve columns in either output format:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Description
   * - ``particle_id``
     - Particle identity within the beam, equal to ``abs(tag)``.
   * - ``injection_turn``
     - Zero-based simulation turn when the particle was injected.
   * - ``injection_batch``
     - Zero-based batch index within the original injection source bunch; regrouping does not renumber it.

These columns are generated on the host from ``abs(tag)`` and the beam's
Injection batch history; ``ParticlePool`` has no corresponding arrays.
If the Injection state is absent, saving with this option raises ``ValueError``.
If the state exists but a particle ID has no matching batch record,
``injection_turn`` and ``injection_batch`` are ``-1`` (unknown).
Sorting or loss does not change a born particle's identity or recorded birth
event. Pending slots (``tag=0``) are omitted, with their count stored as
``NumPending``.

``Output format`` accepts ``"tfs"`` (default) or ``"hdf5"``. HDF5 uses
gzip-compressed datasets and file attributes for the same columns and headers.
The optional birth columns require no additional device-to-host particle-array
copies.
