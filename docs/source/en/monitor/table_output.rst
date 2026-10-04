Table output formats
====================

Format selection
----------------

``output_format`` (JSON ``"Output format"``) is the single format and
compression selector:

- ``"hdf5-gzip1"``: HDF5 with lossless gzip level 1 and shuffle.
- ``"hdf5"``: uncompressed HDF5, with no shuffle.
- ``"tfs"``: TFS text output.

Omitting the option selects uncompressed ``"hdf5"`` for ParticleMonitor and
Injection, and ``"hdf5-gzip1"`` for the other monitors and Slicer.
:doc:`BeamBeam luminosity <../beam_beam>` defaults to ``"tfs"``
in its shared ``Luminosity`` configuration. Selecting a format does not
enable a disabled output or change the configured recording turns.

:doc:`SlowExtraction <../slow_extraction>` and
:doc:`SlowExtractionMonitor <slow_extraction_monitor>` support only the two
HDF5 options; their event and histogram tables do not offer TFS output.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Producer
     - Output organization
     - Additional output
   * - DistMonitor
     - One snapshot per bunch and selected turn
     - None
   * - PhaseAdvanceMonitor
     - One table per bunch and completed window
     - None
   * - ParticleMonitor
     - One file per monitor and beam, containing all selected particles and recorded turns
     - HDF5 arrays or a TFS long table; both include injection coordinates
   * - StatMonitor
     - One statistics history per bunch and monitor position
     - CSV with the same rows
   * - SlowExtraction
     - One append-only particle-event table per source and run
     - HDF5 only
   * - SlowExtractionMonitor
     - One turn and/or physical-time histogram per monitor and source
     - HDF5 only
   * - Injection
     - Initial distributions when ``save_init_dist`` is enabled;
       set ``output_format`` in each ``BunchConfig`` / ``bunchN`` block
     - None
   * - Slicer
     - Particle details per bunch and selected turn
     - Slice summaries remain TFS and CSV
   * - BeamBeam luminosity
     - One sampled encounter history per IP configuration and occurrence
     - None; defaults to TFS

HDF5 tables use ``.h5``; TFS tables use ``.tfs`` with the same filename stem.
SpaceCharge field output uses a multidimensional HDF5 layout
and does not offer TFS export.

HDF5 layout and metadata
------------------------

For the ordinary table layout, each numeric or boolean column is a separate **one-dimensional
dataset at the file root**, with the original column name and data type.
All columns have the same length. A distribution row represents one
particle; a StatMonitor row represents one recorded turn.
For example, a statistics file contains::

   /turn                  (N,)
   /xAverage              (N,)
   /sigmaX                (N,)
   /referenceTime         (N,)
   /referenceBeta         (N,)
   /referenceMomentum     (N,)
   ...

Table headers are stored as **root attributes**, including
names, units embedded in existing metadata, reference conventions and
monitor positions. Varying StatMonitor reference values remain per-row
columns, preserving their full history. Snapshot reference attributes
describe only that snapshot. ParticleMonitor saves turn-by-turn reference columns
only when ``"Include reference": true``; its HDF5 ``initial`` group
always includes injection reference values.

The :doc:`ParticleMonitor <particlemonitor>` HDF5 file
uses two-dimensional (sample, particle) datasets, one-dimensional
``turn`` and ``particle_id`` axes, and an ``initial`` group. It uses
``FormatVersion=2`` and ``ValidSamples`` metadata and dedicated DA/spectral readers;
the ordinary ``read_table`` function is not its reader. Its TFS output instead
uses a long table with explicit ``record``, ``turn`` and ``particle_id`` columns.
Record 0 contains injection coordinates and record 1 contains a trajectory
sample. Both preserve every recorded turn with bounded buffers. HDF5 appends
blocks during tracking; TFS uses private HDF5 storage during tracking and
exports a standard text table at finalization. The PM readers require one
current version-2 file. See the ParticleMonitor page for completion metadata
and format-specific readers.

Two reserved root attributes describe the table format:
``_pass_table_version=1`` and ``_pass_table_columns`` (a JSON array of column
names in output order). User headers must not start with ``_pass_table_``.
Multidimensional SpaceCharge files require a field-specific reader.

``"hdf5-gzip1"`` uses lossless gzip level 1 compression with shuffle.
Shuffle rearranges bytes before compression without changing stored values
or particle order. ``"hdf5"`` disables both compression and shuffle.
Both choices use the same ``.h5`` extension, logical layout and reader;
no separate compression parameter is needed. The choice applies to the
table producers above. CSV is unchanged, and SpaceCharge uses its
separate field writer. For example, to request uncompressed HDF5:

.. code-block:: python

   from PASS.para.schema.monitors import DistMonitorItem

   monitor = DistMonitorItem(s=0.0, save_turns=[[0]], output_format="hdf5")

.. code-block:: json

   {"Command": "DistMonitor", "S (m)": 0.0, "Save turns": [[0]],
    "Output format": "hdf5"}

Uncompressed HDF5 can shorten frequent snapshot writes at the cost of larger
files. Neither choice changes numerical precision or the logical table layout.
Snapshot chunks hold up to
approximately 64 KiB per column, bounded by the number of rows. StatMonitor
uses one configured write interval per column chunk. Thus a normal full
batch fills a new chunk without rewriting earlier compressed chunks.
Chunking controls storage and compression, not sampling or numerical precision.

Statistics batching and live inspection
---------------------------------------

StatMonitor computes and retains a row on **every turn**. Its optional
``write_interval_turns`` / ``"Write interval (turns)"`` is a positive integer
with default **100**. There is no elapsed-time trigger. A value of 100 flushes
at zero-based turns 99, 199, and so on. The final turn flushes any remaining
rows. Normal executor cleanup also flushes a partial batch after a catchable
exception or keyboard interruption; forced process termination can lose the
unwritten batch.

BeamBeam luminosity uses a separate **sample interval**, with first/final-turn
handling. It computes only the selected encounters and appends their records
after a complete common turn; it does not buffer luminosity for every turn.

.. code-block:: python

   from PASS.para.schema.monitors import StatMonitorItem

   stat = StatMonitorItem(s=0.0, output_format="hdf5-gzip1", write_interval_turns=100)

.. code-block:: json

   {
       "Command": "StatMonitor",
       "S (m)": 0.0,
       "Output format": "hdf5-gzip1",
       "Write interval (turns)": 100
   }

In GPU mode, the monitor preallocates a statistics buffer using the configured
turn interval and the initial bunch count. Intermediate turns fill this buffer
without copying statistics to the CPU. At a batch boundary, one GPU-to-CPU copy
collects all pending bunch records; finalization also transfers any partial batch.
Reference quantities are recorded on the CPU every turn, preserving their full
history. After transfer, the CPU formulas calculate the derived output
columns for each record. CPU mode buffers its completed rows in CPU memory.

HDF5 and CSV receive the same batch and are closed after each write.
With ``"tfs"``, CSV is still appended in batches, and the complete TFS is
generated during finalization. CSV is available during tracking once the
first batch has been written. It contains all recorded rows, not a sampled
subset. The two files are written sequentially, not as an atomic pair.

Use CSV for live text inspection, for example ``tail -n 10 -f path.csv`` on
Linux or ``Get-Content path.csv -Tail 10 -Wait`` in PowerShell. HDF5 is binary
and cannot be inspected meaningfully with ``tail``. Read HDF5 after completion
or while the simulation is paused between writes. This writer does not use
SWMR; holding the HDF5 file open in another process can block a later write,
and copying it during a write is not a consistent snapshot.

Reading tables
--------------

The plotting tools, GUI result reader, distribution-file injection and
example analysis scripts accept both formats. A shared reader returns the
columns and original headers as a ``TfsDataFrame``:

.. code-block:: python

   from PASS.utils.table_io import read_table

   data = read_table("statistics.h5")  # also accepts .tfs and .hdf5
   print(data[["turn", "sigmaX", "referenceTime"]].tail())
   print(data.headers)

For a small selection without reading every dataset:

.. code-block:: python

   import h5py

   with h5py.File("statistics.h5", "r") as data:
       turns = data["turn"][-10:]
       sigma_x = data["sigmaX"][-10:]
       metadata = dict(data.attrs)
