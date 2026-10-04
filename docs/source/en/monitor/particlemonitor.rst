ParticleMonitor
==============================

``ParticleMonitor`` records the six phase-space coordinates and loss metadata of selected particles on every turn in a specified interval. Use it for individual trajectories and frequency analysis; enable reference columns when reconstructing physical passage times or momenta.

Configuration example
------------------------------------------

.. code-block:: python

   from PASS.para.schema.monitors import ParticleMonitorItem
   from PASS.para.schema.sequence import Sequence

   sequence = Sequence()
   sequence.add("particle1", ParticleMonitorItem(
       s=0.0, max_tag=5, start_turn=0, end_turn=64,
       include_reference=True,
   ))

The example records particles whose absolute tag is 1–5, including their per-row reference quantities. Add the monitor to a complete sequence containing those particles.

Interface Parameters
--------------------

.. list-table::
  :header-rows: 1
  :widths: 20 20 10 10 40

  * - Python field
    - JSON key
    - Type
    - Default
    - Description
  * - ``s``
    - ``"S (m)"``
    - float
    - Required
    - Longitudinal position of the monitor in the beamline
  * - ``command``
    - ``"Command"``
    - str
    - ``"ParticleMonitor"``
    - Command type identifier
  * - ``max_tag``
    - ``"Max tag"``
    - int
    - Required
    - Maximum tag value of recorded particles, must be :math:`\geq 1`
  * - ``start_turn``
    - ``"Start turn"``
    - int
    - 0
    - Starting turn for recording (inclusive, 0-based)
  * - ``end_turn``
    - ``"End turn"``
    - int
    - -1
    - Ending turn for recording (exclusive, -1 means up to and including the last turn)
  * - ``include_reference``
    - ``"Include reference"``
    - bool
    - false
    - Append per-row reference time, beta and momentum for physical-time/energy analysis

``output_format`` (JSON ``Output format``) defaults to uncompressed ``hdf5``; ``hdf5-gzip1`` and ``tfs`` are also supported. The sequence key supplies the monitor name.

Particle Selection Mechanism
----------------------------

Each particle in PASS has a globally unique ``tag`` (positive integer), and inserted test particles are incremented starting from ``tag = 1``. ``ParticleMonitor`` specifies the recording range via the ``max_tag`` parameter:

.. math::

   \text{recorded} = \{\, i \;\mid\; 1 \leq |\mathrm{tag}_i| \leq \mathrm{max\_tag} \,\}

Note that the matching condition uses :math:`|\mathrm{tag}|` (absolute value), therefore:

- ``tag = 1, 2, \ldots, \mathrm{max\_tag}``: normal surviving particles
- Negative ``tag``: lost particles are **also recorded**, with their coordinates retaining the last values before loss

Tags identify particles across sorting and loss. ``max_tag`` is an upper tag bound, not the number of manually inserted particles or a per-bunch count. A nonpositive max_tag records no particles and logs a warning.

Recorded Turn Range
-------------------

The recorded turn range can be specified via ``start_turn`` and ``end_turn``:

.. math::

   \text{recorded turns} = \{\, n \;\mid\; \mathrm{start\_turn} \leq n < \mathrm{end\_turn} \,\}

- ``start_turn``: starting turn for recording (inclusive), default 0
- ``end_turn``: ending turn for recording (exclusive), default -1 meaning up to and including the last turn

When the requested interval completes, the number of recorded turns is:

.. math::

   N_{\mathrm{record}} = \mathrm{end\_turn} - \mathrm{start\_turn}

Typical use: let the beam stabilize for the first 200 turns (not recorded), then record 1000 turns starting from turn 200 for FFT analysis.


Output files
------------

Each monitor writes **one file per beam**, containing all selected particles
and all turns in the configured recording interval. Choose ``hdf5-gzip1``,
``hdf5`` or ``tfs`` through ``output_format``; there is no layout selector.

* Filename: ``{hms}_beam{bid}_{monitor_name}_s{s:.3f}_particles.h5``;
  TFS uses the same stem and ``.tfs``.
* Output directory: ``output_dir_particle``.
* Root attributes or TFS headers include ``Name="PASS Particle Monitor"``,
  ``Layout="single_file"``, ``FormatVersion=2``, monitor position, beam ID,
  particle tag bound and requested recording interval.

The PM readers require this current version-2 format. Output configuration
contains only the format and buffering settings; it has no layout field.

The write interval changes buffering only: **every turn is still sampled**.
For example, ``write_interval_turns=128`` appends 128 recorded turns together
to HDF5, then reuses the buffer. TFS output uses a private HDF5 file during
tracking and exports the complete text table at finalization. The last partial
block is written when recording ends or during normal cleanup. It does not
save only every 128th turn.

.. code-block:: python

   monitor = ParticleMonitorItem(
       s=100.0, max_tag=10000, output_format="hdf5",
       write_interval_turns=128,
   )

``write_interval_turns`` (JSON ``Write interval (turns)``) is a positive strict
integer with default 128, for both HDF5 and TFS. Uncompressed HDF5 avoids text
formatting and compression work and supports selected-array reads. Gzip1 can
reduce disk traffic; the fastest choice depends on storage throughput and data
compressibility. TFS is useful for text exchange, but larger scans incur text
formatting and parsing costs.

Recorded quantities
-------------------

The history records the following 11 quantities; HDF5 stores ``turn`` once per
sample and the other quantities per sample and particle. TFS adds explicit
record-type and particle-ID columns, as described below.


.. list-table::
  :header-rows: 1
  :widths: 20 15 65

  * - Column name
    - Unit
    - Description
  * - ``turn``
    - -
    - Actual turn number (:math:`\mathrm{start\_turn}` to :math:`\mathrm{end\_turn}-1`)
  * - ``x``
    - m
    - Horizontal position
  * - ``px``
    - -
    - Normalized horizontal momentum
  * - ``y``
    - m
    - Vertical position
  * - ``py``
    - -
    - Normalized vertical momentum
  * - ``z``
    - m
    - Longitudinal coordinate relative to the owning bunch reference passage time, :math:`z_{\mathrm{rel}}`
  * - ``dp``
    - -
    - Relative momentum deviation :math:`\delta`
  * - ``tag``
    - -
    - Particle tag (positive = surviving, negative = lost)
  * - ``lostTurn``
    - -
    - Loss turn (-1 means not lost)
  * - ``lostPosition``
    - m
    - Loss position :math:`s` (-1 means not lost)
  * - ``zCenter``
    - m
    - Nominal grouping slot of the owning bunch, :math:`z_{\mathrm{center}}`

Turn-by-turn reference values are not saved by default.
Injection reference values are always captured separately from the history.
With ``"Include reference": true``, each row additionally saves
``referenceTime`` (s), ``referenceBeta`` (dimensionless) and
``referenceMomentum`` (eV/c per nucleon), increasing the history staging buffer from 11 to 14 columns.
These values belong to the same recording event as the particle coordinates.
Live-particle arrival time is then

.. math::

   t_i=referenceTime-z/(referenceBeta\,c).

``zCenter`` is nominal slot metadata only. Continuous z may extend beyond one
circumference and does not alone determine group membership. Reference columns
are NaN for loss records when enabled, preventing reinterpretation of frozen
loss coordinates using the current bunch reference. Analyses requiring reference
history must enable this option before tracking; a final reference snapshot
cannot reconstruct earlier turns during acceleration or regrouping.

Enable reference output for a diagnostic run with the schema API:

.. code-block:: python

   from PASS.para.schema.monitors import ParticleMonitorItem

   monitor = ParticleMonitorItem(s=0.0, max_tag=5, include_reference=True)

or in generated JSON:

.. code-block:: json

   "PM_reference": {
       "S (m)": 0.0,
       "Command": "ParticleMonitor",
       "Max tag": 5,
       "Include reference": true
   }

Bounded buffering and completion
---------------------------------

CPU sampling batch-selects particle identities; GPU sampling uses fused
kernels and keeps the staging block on the device. The float64 history buffer
is bounded by

.. math::

   M = \mathrm{max\_tag}\,\min(N_{\mathrm{record}},N_{\mathrm{write}})\,
       N_{\mathrm{col}}\,8\;\mathrm{bytes}.

Here :math:`N_{\mathrm{col}}=11` by default, or 14 with ``Include reference``.
Initial data and lookup workspace additionally scale with ``max_tag``. For
10,000 particles and a 128-turn block, the staging buffer is 112.64 MB without
reference history or 143.36 MB with it, independently of a longer run's total
turn count. Each write transfers only the pending block from GPU to CPU.
Writing needs additional temporary host memory for that block. TFS export
formats at most 65,536 rows per block during finalization, without loading
the full history into memory. No automatic cap changes the configured history
buffer size.

``ValidSamples`` counts committed recorded turns, ``EndTurn`` is one beyond the latest recorded turn,
and ``RequestedEndTurn`` is the planned exclusive endpoint. ``Completed``
means the monitor covered its requested interval; it does not mean later
commands in the sequence finished. Finalization flushes existing samples and
never adds a sample at a different lattice position. A cooperative early stop
therefore preserves only the already sampled turns. GUI normal stopping waits
for a complete turn before finalizing; forced process termination may lose the
pending block. See :doc:`../project_files` for stop controls.

``NumTurn`` stores the actual sample count. Final TFS headers contain all
final counts, including ``ValidSamples``, ``ValidRows``, ``EndTurn`` and
``Completed``. Normal early stopping produces a complete TFS table for the
observed interval, with ``Completed=false`` when the requested interval was
not covered.

After a write failure, cleanup does not retry an uncertain block or duplicate
committed samples. The original error is propagated and further tracking with
that monitor is rejected. HDF5 readers use only the committed prefix.
Summary attributes ``NumTurn``, ``EndTurn`` and ``Completed`` are reconciled
with ``ValidSamples`` when handling an append failure. If the underlying HDF5
error also prevents that repair, the original exception is retained with an
additional explanation; ``ValidSamples`` remains the commit marker.
Injection ``initial/valid`` flags are published only after their coordinate,
injection-turn and reference fields have been written completely. Read
HDF5 after completion or while tracking is paused between writes; this is not
a SWMR live-reader interface. A final TFS file becomes available after export.

HDF5 arrays and injection data
---------------------------------

The file contains these root datasets:

* ``particle_id``: (particle), positive integer IDs equal to absolute tags;
* ``turn``: (sample), actual sampled turns;
* ``x, px, y, py, z, dp``: (sample, particle), in particle precision;
* ``zCenter``: (sample, particle), float64 nominal grouping metadata;
* ``tag, lostTurn, lostPosition``: (sample, particle), respectively int32,
  int64 and float32;
* optional ``referenceTime, referenceBeta, referenceMomentum``:
  (sample, particle), float64.

``ValidSamples`` is updated after a complete block is written. Any rows beyond
that committed prefix are not valid data.

The ``initial`` group stores one six-coordinate record per selected particle,
plus ``particle_id``, ``valid`` and ``injection_turn``. These are the **actual
injected coordinates after reference conversion**, captured before subsequent
tracking commands, even when PM recording starts later. Injection-event
``referenceTime``, ``referenceBeta`` and ``referenceMomentum`` are always saved,
independently of the optional history reference columns. Missing injection
records have ``valid=false``, NaN coordinates and ``injection_turn=-1``. The
first monitor sample never substitutes for an unknown initial condition.

TFS long table
--------------

The final TFS file is a standard numeric table with this fixed column order::

   record turn particle_id x px y py z dp tag lostTurn lostPosition zCenter
   referenceTime referenceBeta referenceMomentum

The wrapped listing above represents one table header. ``record=0`` marks an
injection record and ``turn`` gives its injection turn; each captured particle
gets one such row. All injection rows precede the trajectory rows.
``record=1`` marks a trajectory sample, with one row for
every selected particle at every recorded turn. ``particle_id`` remains the
positive identity even if the signed ``tag`` becomes negative after loss or
is zero for a particle absent at that sample.

Reference columns always exist in TFS. Injection rows retain their reference
values; trajectory rows contain NaN there when ``Include reference`` is false.
Floating-point text uses 17 significant digits. The complete table keeps one
numeric schema and can be read directly with ``tfs.read(path)`` or
``PASS.utils.table_io.read_table(path)``.

During tracking, the monitor writes its bounded blocks to a private
``.pass_pm_<uuid>.h5`` file using uncompressed HDF5. Finalization exports to a
private ``.tfs.partial`` file, closes it, and publishes the complete final TFS
without overwriting an existing target. Only after successful publication are
this run's temporary HDF5 and text files removed. If export fails, both are
retained and the error identifies the recoverable HDF5 path; finalization can
retry export from it. No incomplete text file is published under the final
name. The standard TFS contains only its normal headers, column/type lines and
data rows, with no block-commit comments.

If publication succeeds but temporary-file removal fails, the final TFS remains
valid; a repeated finalization retries cleanup only. Cleanup is restricted to
temporary files created by this monitor in the current run.
If an exception arrives after the final hard link was created, file identity
is checked to recognize the successful publication. Repeated finalization then
retries cleanup only, including when restoring an already completed checkpoint.

Reading, checkpoints and interpretation
------------------------------------------

Use ``PASS.analysis.read_dynamic_aperture(path)`` for DA, or
``PASS.analysis.data_io.load_signal(path, "x", object_range=[0, 10])`` for
selected trajectories. These readers identify actual turns, particle IDs and
initial versus trajectory records. Spectral loading rejects lost or unavailable
samples. Multidimensional PM HDF5 files cannot be passed to the ordinary
one-dimensional ``read_table`` interface. DA and trajectory readers accept
one current PM file, containing its own captured injection coordinates.

Collision checkpoints capture both the already written monitor file and the
pending block, together with injection data. Resuming writes a new output file
containing the preceding history; it does not depend on the old output path.
Checkpoint creation temporarily needs memory for the saved file bytes in
addition to the normal bounded history buffer.
PM checkpoint payloads use ``PASS-particle-monitor-state-2`` and identify
whether their saved file bytes are runtime HDF5 or finalized TFS. Only this
current checkpoint version is supported.
If an early-stopped run has already exported a partial TFS history, resuming
reconstructs the private HDF5 history before appending new samples. The final
TFS then contains both intervals, including new samples buffered until the
last requested turn.

Absent particles have zero tags, for example before injection. Select samples
using their tag and loss fields; frozen loss coordinates cannot be interpreted
using a later live-bunch reference. Place the monitor after all relevant
effects when measuring complete-turn survival. For common formats, see
:doc:`table_output`; for coordinates, see :ref:`en-longitudinal-reference`.
