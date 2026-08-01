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
  - Pre-allocates buffer ``(max_tag, num_record_turn, 10)``, avoiding runtime dynamic allocation;
  - Records 10 columns of data per turn: turn + 6D coordinates + tag + lost_turn + lost_position;
  - After simulation, each particle is written to a separate TFS file;
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

The actual number of recorded turns is:

.. math::

   N_{\mathrm{record}} = \mathrm{end\_turn} - \mathrm{start\_turn}

Typical use: let the beam stabilize for the first 200 turns (not recorded), then record 1000 turns starting from turn 200 for FFT analysis.


Pre-allocation Strategy
-----------------------

``ParticleMonitor`` pre-allocates the complete buffer at initialization:

.. math::

   \mathrm{buffer} \in \mathbb{R}^{\mathrm{max\_tag} \times N_{\mathrm{record}} \times 10}

Memory overhead:

.. math::

   M = \mathrm{max\_tag} \times N_{\mathrm{record}} \times 10 \times 8 \;\text{bytes}

Typical scenario (14 test particles, recording 1000 turns):

.. math::

   M = 14 \times 1000 \times 10 \times 8 = 1.12 \;\text{MB}

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

.. note::

  ``max_tag`` should be consistent with the number of particles inserted via ``Insert Particle Coordinate`` in ``Injection``. For example, if 14 test particles are inserted, then ``max_tag = 14``.


Output Files
------------

Each particle generates an independent TFS file:

- **Filename**: ``{hms}_particle_beam{bid}_{monitor_name}_s_{s:.3f}_tag_{tag}.tfs``
- **Output directory**: ``output_dir_particle``

TFS file header:

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

Output columns (10 columns total):

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
    - Longitudinal position
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

After simulation, 3 TFS files are generated in the ``output_dir_particle`` directory, each containing the 6D coordinates of that particle for all recorded turns.

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

The buffer size is allocated for :math:`1000 - 200 = 800` turns, and the ``turn`` column in the output TFS file starts from 200.

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
- **Slip factor measurement**: Record the longitudinal coordinate :math:`z` of the momentum-offset particle turn-by-turn; the rate of change of :math:`z` per turn divided by :math:`\delta` gives the slip factor :math:`\eta`
- **Closed orbit verification**: The TBT coordinates of an initially un-offset particle should remain unchanged, verifying closed orbit stability
- **Particle loss tracking**: Locate the time and position of particle loss through ``tag`` sign changes and ``lostTurn`` / ``lostPosition``
