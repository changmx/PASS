Slow extraction (SlowExtraction)
================================

``SlowExtraction`` captures live particles on a selected side of a transverse
cut at an existing tracking plane, then stops their subsequent ring tracking.
It supplies immutable particle events to :doc:`monitor/slow_extraction_monitor`.
The sextupoles, excitation, electrostatic septum and intervening lattice still
provide the physical transport. This command applies no kick, drift or change
of reference.

Selection and particle state
----------------------------

For the configured roll angle :math:`\theta`, the cut coordinate is

.. math::

   u=x\cos\theta-y\sin\theta.

``Side="positive"`` selects :math:`u>u_{\rm cut}`;
``Side="negative"`` selects :math:`u<u_{\rm cut}`, with
``Position (m)`` defining :math:`u_{\rm cut}`. Equality is excluded. Only
particles with ``tag > 0`` can be selected. The cut is infinite in the
orthogonal transverse direction; it does not model a material surface.

The command first copies every selected particle's coordinates, identity,
weight and reference into an owned host event buffer. It then changes the
original ring ``tag`` to ``-tag`` and records the termination turn and plane
in ``lost_turn`` and ``lost_position``. Coordinates and momenta are unchanged.
Later tracking skips the retired particles, so a particle is captured once.
CPU and GPU execution use the same selection logic; GPU events are copied to
host memory before the original tags change.

These negative tags indicate termination of ring tracking, not necessarily
material loss. Existing StatMonitor loss counts include extracted particles.
Use the union of all extraction source event tables for the same run and
beam, identified by ``(RunId, beam_id, particle_id)``, to distinguish
extraction from physical loss. When reconciling a snapshot, include only
captures already executed at its turn and command position/order; a
same-plane snapshot before the action still shows those particles live.
Ordinary distribution snapshots remain available for the ring state.

Position and execution order
----------------------------

``S (m)`` labels an already reached tracking plane. Setting it does not insert
missing transport. Split a drift or thick element correctly before placing
an extraction plane inside it; input validation rejects a cut inside an
unsplit element body. Prefer existing element boundaries. Choose a plane and
cut that represent the intended extraction channel: a large displacement
elsewhere in the ring alone does not establish successful extraction.

Default same-position priorities are 850 for ``SlowExtraction`` and 860 for
``SlowExtractionMonitor``; ordinary monitors have priority 800. Thus an
ordinary same-position distribution snapshot precedes extraction by default.
Use explicit ``Order`` to change this relationship or arrange collective
effects. If any command at that position specifies ``Order``, every command
there must specify a distinct integer. The spill monitor must follow its
named source at the same position and in the same beam.

Turn and physical-time windows
------------------------------

Turn indices are zero-based. ``Start turn`` is inclusive and ``End turn`` is
exclusive. Omitted end bounds are unbounded within the configured run.
Optional time bounds are also left-closed and right-open, and apply to each
selected particle's physical arrival time:

.. math::

   t_i=t_0-\frac{z_i}{\beta_0 c}.

Both conditions must hold when turn and time bounds are configured together.
Time is in seconds on the existing bunch reference clock; negative finite
bounds are allowed. The stored continuous ``z`` is not folded, and nominal
bunch-group centres are not added. No fixed revolution frequency converts
turns to time. See :ref:`en-longitudinal-reference`.

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
     - ``"SlowExtraction"``
     - Command type.
   * - ``s``
     - ``S (m)``
     - finite float, required
     - Existing tracking plane; nonnegative and within the ring.
   * - ``order``
     - ``Order``
     - strict int or null; null
     - Explicit same-position order; otherwise priority 850.
   * - ``position``
     - ``Position (m)``
     - finite float, required
     - Transverse cut coordinate :math:`u_{\rm cut}`.
   * - ``side``
     - ``Side``
     - ``"positive"``
     - ``"positive"`` or ``"negative"``; strict side of the cut.
   * - ``tilt``
     - ``Tilt (rad)``
     - finite float; 0
     - Transverse roll angle, not a longitudinal yaw.
   * - ``start_turn``
     - ``Start turn``
     - strict int; 0
     - Inclusive nonnegative starting turn.
   * - ``end_turn``
     - ``End turn``
     - strict int or null; null
     - Exclusive ending turn, greater than the start when supplied.
   * - ``start_time``
     - ``Start time (s)``
     - finite float or null; null
     - Inclusive particle arrival-time bound.
   * - ``end_time``
     - ``End time (s)``
     - finite float or null; null
     - Exclusive particle arrival-time bound; greater than the start if both exist.
   * - ``buffer_size``
     - ``Buffer size (particles)``
     - strict positive int; 65536
     - Flush threshold in event rows, not a particle sampling limit.
   * - ``output_format``
     - ``Output format``
     - ``"hdf5-gzip1"``
     - ``"hdf5"`` or lossless ``"hdf5-gzip1"``; TFS is not supported.

Example
-------

Append these commands to an existing sequence that already transports the
beam to ``s=12.5`` m. The values illustrate configuration, not a machine design.

.. code-block:: python

   from PASS.para.schema import SlowExtractionItem, SlowExtractionMonitorItem

   seq.add("extract", SlowExtractionItem(
       s=12.5, position=0.035, side="positive",
       start_turn=100, end_turn=10000,
   ))
   seq.add("spill", SlowExtractionMonitorItem(
       s=12.5, source="extract", bin_by="both",
       turn_bin_width=10, time_bin_width=1e-3,
   ))

The monitor may specify additional turn and time windows independently. Its
window changes statistical selection only; it never changes extraction.

Event output
------------

The source writes ``distribution/slow_extraction/*_events.h5`` beneath the run output
directory. Flat-output mode omits the ``slow_extraction`` subdirectory.
Filenames include a run UUID, beam identifier and source-specific identifier.
Each dataset is a one-dimensional column in the standard
:doc:`monitor/table_output` layout.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Columns
     - Meaning / units
   * - ``particle_id``, ``beam_id``, ``bunch_id``
     - Identity at capture; ``particle_id`` is the positive original tag.
   * - ``turn``, ``s``, ``time``
     - Capture turn, plane in metres, and physical arrival time in seconds.
   * - ``x``, ``px``, ``y``, ``py``, ``z``, ``dp``
     - Captured PASS coordinates: positions in metres, ``px=Px/P0``,
       ``py=Py/P0``, continuous relative-time coordinate ``z``, and ``dp=(P-P0)/P0``.
   * - ``reference_time``, ``reference_beta``, ``reference_momentum``
     - Reference at capture: seconds, dimensionless beta, and eV/c in the
       bunch convention (per nucleon for ions).
   * - ``macro_weight``
     - Number of real particles represented by the event.
   * - ``charge_number``, ``proton_number``, ``neutron_number``
     - Signed charge number and species composition.
   * - ``rest_energy``
     - Rest energy in eV in the bunch convention (per nucleon for ions).

The original coordinate precision is retained; time, reference quantities
and weights are stored in float64. Event coordinates and reference values
remain valid even after the live bunch reference changes. For a physical
slope, use :math:`x'=p_x/\sqrt{(1+\delta)^2-p_x^2-p_y^2}` rather than treating
the normalized momentum ``px`` as an angle.

All captured events are retained. A complete batch can take the buffer above
its threshold; reaching the threshold triggers an append, and finalization
flushes the remaining rows. An empty run still produces a typed empty event
table at finalization. The in-memory capture precedes ring retirement, but
buffering is not crash-durable storage. No extraction ledger is restored by
a simulation restart. Output errors stop execution rather than silently
discarding events or automatically retrying an ambiguous append.
