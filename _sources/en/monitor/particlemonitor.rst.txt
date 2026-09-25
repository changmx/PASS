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

``output_format`` (JSON ``Output format``) defaults to ``hdf5-gzip1``; ``hdf5`` and ``tfs`` are also supported. The sequence key supplies the monitor name.

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


Output Files
------------

Each particle generates an independent HDF5 file by default:

- **Filename**: ``{hms}_beam{bid}_{monitor_name}_s{s:.3f}_tag{tag}.h5``
- **Output directory**: ``output_dir_particle``

Metadata (HDF5 attributes, or TFS headers in text mode):

::

   @ Name             PASS Particle Monitor
   @ Time             2026-07-14 00:11:03
   @ Monitor          pm1
   @ S                0.0
   @ BeamId           0
   @ Tag              1
   @ NumTurn          1000
   @ StartTurn        0
   @ EndTurn          1000

On a cooperative early stop, finalization writes only the turns already sampled;
unused future rows from the preallocated buffer are omitted. ``NumTurn`` and
``EndTurn`` describe the saved rows, with ``EndTurn`` still exclusive. A partial
file additionally records ``RequestedEndTurn`` for the planned endpoint after
normalizing ``-1`` or clipping to the simulation length. For example, recording
from turn 200 and stopping after turn 499 gives ``NumTurn=300``, ``EndTurn=500``,
and ``RequestedEndTurn=1000`` if the planned endpoint was 1000. A completed
interval keeps its existing metadata without ``RequestedEndTurn``. If recording
has not started, no particle table is written.

This behavior is shared by CPU and GPU output. Repeated finalization does not
rewrite an already completed table. On GPU, the buffer is sliced before the
device-to-host copy: transfer size and the resulting host array scale with the
recorded turns, while the initial device allocation still covers the planned interval.
GUI normal stopping waits for a complete
turn before finalizing; force stopping cannot guarantee that buffered data are
written. See :doc:`../project_files` for the GUI stop controls and run records.

Default output columns (11 columns total):

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

Reference values are not saved by default, either in data columns or in headers.
With ``"Include reference": true``, each row additionally saves
``referenceTime`` (s), ``referenceBeta`` (dimensionless) and
``referenceMomentum`` (eV/c per nucleon), giving 14 columns in total.
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

Pre-allocation Strategy
-----------------------

``ParticleMonitor`` pre-allocates the complete buffer at initialization:

.. math::

   \mathrm{buffer} \in \mathbb{R}^{\mathrm{max\_tag} \times N_{\mathrm{record}} \times N_{\mathrm{col}}}

``Include reference`` defaults to false: :math:`N_{\mathrm{col}}=11`.
When enabled, :math:`N_{\mathrm{col}}=14`. Disabled reference columns have no
buffer allocation on either CPU or GPU.

Memory overhead:

.. math::

   M = \mathrm{max\_tag} \times N_{\mathrm{record}} \times N_{\mathrm{col}} \times 8 \;\text{bytes}

Typical scenario (14 test particles, recording 1000 turns):

.. math::

   M = 14 \times 1000 \times 11 \times 8 = 1.232 \;\text{MB}

Enabling reference columns increases this example to 1.568 MB (27.3% more).

The buffer uses the same array backend as the beam (``beam.particles.xp``), numpy on CPU, cupy on GPU. Advantages of pre-allocation:

- The history buffer is allocated before tracking; recording still consumes processing time;
- In GPU scenarios, the buffer resides in GPU memory throughout; each turn writes directly from the GPU particle array to the GPU buffer, with only a single D2H copy performed at the end of simulation;
- Fixed memory layout, facilitating post-processing analysis.


Interpretation and limits
--------------------------------------------------

Absent particles leave zero-filled history rows. A particle may be absent before injection; use the tag and loss fields when selecting data. Lost coordinates remain frozen and must not be interpreted using a later live-bunch reference. The complete history buffer scales with max_tag times the number of recorded turns, so select those bounds before a large run. For common formats and readers, see :doc:`table_output`; for coordinate definitions, see :ref:`en-longitudinal-reference`.
