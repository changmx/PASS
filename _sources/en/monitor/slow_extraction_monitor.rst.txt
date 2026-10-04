Slow-extraction spill monitor
=============================

``SlowExtractionMonitor`` reads events from one named :doc:`../slow_extraction`
command and accumulates spill histograms by tracking turn, physical arrival
time, or both. It never selects particles from the ring or changes particle
state. Source event rows, rather than a repeated scan of negative tags, are
the counting input. Repeated execution for the same source batch does not
count its events twice.

The ``Source`` value must exactly match a ``SlowExtraction`` sequence name
in the same beam. The monitor must execute after that source at the same
``S (m)``. Default priorities 850 and 860 provide this order; if explicit
``Order`` is used at that position, set distinct values on every node there.
The monitor must consume every source invocation from its first batch, including
turns outside its statistical window. Missing a batch raises an error; the monitor
does not recover skipped events from the source file.

Windows and bin definitions
---------------------------

Turn and physical-time windows are independently optional and intersect
when both are configured. Each is left-closed and right-open. Monitor
windows filter the source's already captured events; they do not change the
source's extraction conditions. A monitor cannot recover particles that its
source never captured.

With ``Bin by="turn"``, bins of ``Turn bin width`` turns are anchored at
``Start turn``. With ``Bin by="time"``, bins of ``Time bin width (s)`` seconds
are anchored at ``Time origin (s)``. ``"both"`` writes the two histograms
separately. Time bins always use the source's captured particle times:

.. math::

   t_i=t_{0,i}-\frac{z_i}{\beta_{0,i}c}.

No fixed revolution-frequency conversion is used. Particles from one turn
can occupy different time bins, and later tracking turns can add events to
earlier time bins. Therefore time histograms remain provisional during a
run; writing a snapshot does not close their bins.

Nominal ``bin_start`` and ``bin_end`` stay fixed between snapshots.
Time edges are evaluated in float64 as ``origin + k * width``. Assignment uses
exact left-closed, right-open comparisons against these same stored edges,
without snapping nearby times to a boundary. A width too small to resolve
distinct edges at the observed times raises an error.
``observed_start``, ``observed_end`` and ``observed_width`` separately describe
the portion covered by the observations so far. Turn coverage is the range
of executed turns inside the monitor's turn window. Time coverage is the
envelope of the source reference times and accepted particle event times,
clipped to the monitor's time window. This envelope does not prove that all
particle arrivals in it are complete.

Zero-count bins are included within the observed range, not extrapolated
over the entire requested run. ``is_partial`` marks bins whose observed
range is smaller than their nominal range. An event exactly on the newest
time boundary can produce a zero-width partial bin until later observations
extend its coverage. Exclude partial bins when comparing full-bin spill
uniformity, and retain the chosen bin width with any reported ripple metric.

Interface parameters
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 20 35

   * - Python parameter
     - JSON key
     - Type / default
     - Meaning
   * - ``command``
     - ``Command``
     - ``"SlowExtractionMonitor"``
     - Read-only spill monitor type.
   * - ``s``
     - ``S (m)``
     - finite nonnegative float, required
     - Same tracking plane as the source.
   * - ``order``
     - ``Order``
     - strict int or null; null
     - Same-position order; default priority 860.
   * - ``source``
     - ``Source``
     - nonempty str, required
     - Exact sequence name of a ``SlowExtraction`` in this beam.
   * - ``bin_by``
     - ``Bin by``
     - ``"both"``
     - ``"turn"``, ``"time"`` or ``"both"``.
   * - ``turn_bin_width``
     - ``Turn bin width``
     - strict positive int; 1
     - Nominal turn-bin width; origin is ``Start turn``.
   * - ``time_bin_width``
     - ``Time bin width (s)``
     - finite positive float; 0.001
     - Nominal physical-time bin width.
   * - ``time_origin``
     - ``Time origin (s)``
     - finite float; 0
     - Anchor of the time-bin grid; negative bin indices are allowed.
   * - ``start_turn``
     - ``Start turn``
     - strict nonnegative int; 0
     - Inclusive statistical starting turn.
   * - ``end_turn``
     - ``End turn``
     - strict int or null; null
     - Exclusive ending turn, greater than the start when supplied.
   * - ``start_time``
     - ``Start time (s)``
     - finite float or null; null
     - Inclusive event arrival-time bound.
   * - ``end_time``
     - ``End time (s)``
     - finite float or null; null
     - Exclusive arrival-time bound, greater than the start if both exist.
   * - ``write_interval_turns``
     - ``Write interval (turns)``
     - strict positive int; 100
     - Output cadence; finalization writes pending updates.
   * - ``output_format``
     - ``Output format``
     - ``"hdf5-gzip1"``
     - ``"hdf5"`` or lossless ``"hdf5-gzip1"``; no TFS output.

Example
-------

This snippet assumes an existing ``extract`` action at the same plane.

.. code-block:: python

   from PASS.para.schema import SlowExtractionMonitorItem

   seq.add("spill", SlowExtractionMonitorItem(
       s=12.5, source="extract", bin_by="both",
       start_turn=100, end_turn=10000,
       start_time=0.10, end_time=1.20,
       turn_bin_width=10, time_bin_width=1e-3, time_origin=0.0,
       write_interval_turns=100,
   ))

Counts include only source events satisfying both the turn and time bounds.
``SlowExtractionMonitor`` is also accepted as the ``type`` in the high-level
API's monitor list. Use ``SlowExtractionItem`` separately for the action.

Output tables
-------------

Tables are written under ``slow_extraction/`` in the run output directory;
flat-output mode omits the ``slow_extraction`` subdirectory. Source and monitor
names, a digest, run identity and beam identifier form the filename stem.
Suffixes ``_turn.h5`` and ``_time.h5`` identify the selected histogram types.
Completed turn bins are appended once at each write interval; the active
partial turn bin is appended only at finalization. The in-memory
``get_histogram("turn")`` result also includes the current partial bin.
Time output publishes a complete cumulative snapshot at its file path,
allowing later events to revise earlier bins. The source's particle-event
table remains separate.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Column
     - Meaning
   * - ``bin_start``, ``bin_end``
     - Fixed nominal edges, in turns or seconds; left-closed, right-open.
   * - ``observed_start``, ``observed_end``, ``observed_width``
     - Observed coverage in the same units; width is nonnegative.
   * - ``is_partial``
     - Boolean indicator of incomplete nominal-bin coverage.
   * - ``num_extracted``
     - Number of macro-particle events in the bin.
   * - ``real_extracted``
     - Sum of ``macro_weight``: represented real particles.
   * - ``charge_extracted``
     - Sum of signed ``charge_number * e * macro_weight``, in coulombs.
   * - ``cumulative_extracted``, ``cumulative_real``
     - Cumulative macro-particle and real-particle counts in increasing bin order.
   * - ``particle_rate``
     - Time table only: ``real_extracted / observed_width``, in particles/s.
   * - ``current``
     - Time table only: ``charge_extracted / observed_width``, in amperes;
       the charge sign is retained.

Rates are NaN when ``observed_width`` is zero. They use the observed width,
not an assumed full-bin exposure. A partial-bin rate should not be compared
directly with a full-bin ripple statistic. Headers record source event-file
identity, time definition, windows and observed reference/event envelopes.

The CPU and GPU paths consume the same owned host event batches. The monitor
does not reread event files on every turn. Writes normally occur every 100
turns and at run completion or finalization; an interrupted
run uses the normal command-finalization mechanism. Hard process failure can
lose updates still held in memory. A failed turn-table append can leave an
incomplete table; execution stops without retrying that append. Time
snapshots replace only previously completed snapshots. Histogram state and source event history
are not restored by a resume ledger in this implementation.

Read tables with ``PASS.utils.table_io.read_table`` as described in
:doc:`table_output`. Use the separate source event table for six-dimensional
distributions, particle-level arrival times or different offline binning.
