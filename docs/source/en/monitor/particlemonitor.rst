ParticleMonitor
==============================

Introduction
------------

``ParticleMonitor`` is a turn-by-turn particle coordinate monitor that records the 6D phase space coordinates of selected particles at a specified longitudinal position, once per turn. Unlike ``StatMonitor`` which records overall bunch statistics, ``ParticleMonitor`` focuses on the turn-by-turn trajectory of **individual particles**, and is the core tool for turn-by-turn (TBT) diagnostics such as tune measurement, chromaticity measurement, and amplitude-dependent effect analysis.

- **Code location**: ``PASS/commands/monitor/particle_monitor.py``
- **Class name**: ``ParticleMonitor``, registered name ``"particlemonitor"``
- **Key features**:

  - Selects recorded particles via the ``max_tag`` parameter, with the matching condition :math:`1 \leq |\mathrm{tag}| \leq \mathrm{max\_tag}`;
  - Supports setting a recorded turn range ``[start_turn, end_turn)``, without needing to start tracking from turn 0;
  - Pre-allocates buffer ``(max_tag, num_record_turn, num_columns)``, avoiding runtime dynamic allocation;
  - Records 11 columns per turn by default: turn + 6D coordinates + tag + lost_turn + lost_position + zCenter; ``Include reference`` adds three optional reference columns;
  - After simulation, each particle is written to a separate HDF5 file (or TFS file when selected);
  - Filenames include the monitor name and longitudinal position (3 decimal places), supporting multi-position deployment;
  - CPU uses numpy, GPU uses cupy, with the buffer residing on GPU throughout; only a single D2H copy is performed at the end;


Particle Selection Mechanism
----------------------------

Each particle in PASS has a globally unique ``tag`` (positive integer), and inserted test particles are incremented starting from ``tag = 1``. ``ParticleMonitor`` specifies the recording range via the ``max_tag`` parameter:

.. math::

   \text{recorded} = \{\, i \;\mid\; 1 \leq |\mathrm{tag}_i| \leq \mathrm{max\_tag} \,\}

Note that the matching condition uses :math:`|\mathrm{tag}|` (absolute value), therefore:

- ``tag = 1, 2, \ldots, \mathrm{max\_tag}``: normal surviving particles
- Negative ``tag``: lost particles are **also recorded**, with their coordinates retaining the last values before loss

.. note::

  Test particles are inserted via the ``Insert Particle Coordinate`` parameter of ``Injection``. After insertion, particle ``tag`` values increment starting from 1. ``max_tag`` should equal the number of inserted test particles.

  If ``max_tag < 1``, the monitor only outputs a warning log and records no particles, but does not affect the simulation run.


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

- Zero memory allocation at runtime, no impact on tracking performance;
- In GPU scenarios, the buffer resides in GPU memory throughout; each turn writes directly from the GPU particle array to the GPU buffer, with only a single D2H copy performed at the end of simulation;
- Fixed memory layout, facilitating post-processing analysis.


Interface Parameters
--------------------

.. list-table::
  :header-rows: 1
  :widths: 20 20 10 10 40

  * - Property
    - JSON key
    - Type
    - Default
    - Description
  * - ``s``
    - ``"S (m)"``
    - float
    - Required
    - Longitudinal position of the monitor in the beamline
  * - ``cmd_name``
    - ``"name"``
    - str
    - Required
    - Monitor name (automatically filled from the sequence key name)
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

.. note::

  ``max_tag`` should be consistent with the number of particles inserted via ``Insert Particle Coordinate`` in ``Injection``. For example, if 14 test particles are inserted, then ``max_tag = 14``.


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
    - Longitudinal coordinate relative to the owning bunch center, :math:`z_{\mathrm{rel}}`
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

   ParticleMonitor(s=0.0, max_tag=5, include_reference=True)

or in generated JSON:

.. code-block:: json

   "PM_reference": {
       "S (m)": 0.0,
       "Command": "ParticleMonitor",
       "Max tag": 5,
       "Include reference": true
   }

Usage Example
-------------

Basic Usage
~~~~~~~~~~~

The following JSON snippet places a particle monitor at :math:`s = 0.0` m, recording particles with ``tag = 1`` through ``tag = 3``:

.. code-block:: json

   "PM1": {
       "S (m)": 0.0,
       "Command": "ParticleMonitor",
       "Max tag": 3
   }

Combined with inserting 3 test particles in ``Injection``:

.. code-block:: json

   "injection": {
       "S (m)": 0.0,
       "Command": "Injection",
       "bunch0": {
           "Insert Particle Coordinate": [
               [0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.001, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.001]
           ]
       }
   }

The above configuration inserts 3 test particles:

- ``tag = 1``: :math:`x = 1` mm horizontal offset particle, for horizontal tune measurement
- ``tag = 2``: :math:`y = 1` mm vertical offset particle, for vertical tune measurement
- ``tag = 3``: :math:`\delta = 10^{-3}` momentum offset particle, for dispersion and chromaticity measurement

After simulation, 3 HDF5 files by default are generated in the ``output_dir_particle`` directory, each containing the 6D coordinates of that particle for all recorded turns.

Delayed Recording
~~~~~~~~~~~~~~~~~

The following configuration does not record for the first 200 turns (to let the beam stabilize), then records from turn 200 to turn 1000:

.. code-block:: json

   "PM1": {
       "S (m)": 0.0,
       "Command": "ParticleMonitor",
       "Max tag": 14,
       "Start turn": 200,
       "End turn": 1000
   }

The buffer size is allocated for :math:`1000 - 200 = 800` turns, and the ``turn`` column in the output table starts from 200.

Multi-position Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple particle monitors can be placed at different positions on the ring to compare the phase space coordinates of particles at different locations:

.. code-block:: json

   "PM_start": {
       "S (m)": 0.0,
       "Command": "ParticleMonitor",
       "Max tag": 14
   },
   "PM_mid": {
       "S (m)": 284.5,
       "Command": "ParticleMonitor",
       "Max tag": 14
   }


Application Scenarios
---------------------

- **Tune measurement**: Perform FFT or NAFF on TBT coordinates to extract the betatron oscillation frequencies, which are the tunes :math:`Q_x`, :math:`Q_y`
- **Chromaticity measurement**: Measure the tune at different momentum deviations :math:`\delta`; the slope of the linear fit of :math:`Q(\delta)` gives the chromaticity :math:`DQ_x`, :math:`DQ_y`
- **Amplitude-dependent tune shift (ADTS)**: Measure the tune for particles with different initial amplitudes to analyze the nonlinear tune shift with amplitude
- **Dispersion function measurement**: Take the time average of the TBT centroid orbit of the momentum-offset particle, divided by :math:`\delta`, to obtain the dispersion function :math:`D(s)`
- **Slip-factor measurement**: Record the bunch-relative coordinate :math:`z_{\mathrm{rel}}` of a momentum-offset particle turn-by-turn. For comparisons across bunches or after regrouping, enable ``Include reference`` and use the saved referenceTime and referenceBeta to reconstruct physical arrival times
- **Closed orbit verification**: The TBT coordinates of an initially un-offset particle should remain unchanged, verifying closed orbit stability
- **Particle loss tracking**: Locate the time and position of particle loss through ``tag`` sign changes and ``lostTurn`` / ``lostPosition``

``output_format`` (JSON ``"Output format"``) defaults to ``"hdf5-gzip1"``;
Use ``"hdf5"`` for uncompressed HDF5 or ``"tfs"`` for text output. See :doc:`table_output` for
the HDF5 layout, compression and common reader.
