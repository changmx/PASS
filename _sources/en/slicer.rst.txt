Slicer
======

The ``Slicer`` command assigns every live macro-particle in each bunch to a
longitudinal slice.  It stores the result in a named ``SliceSet`` owned by the
bunch.  Slicing is a local classification operation: it does not reorder any
particle array and it does not change bunch membership.

On CUDA, fixed-range ``equal_length`` grids batch histogram and table generation
across bunches in three kernel launches, then transfer alive/outside counts
together. Slice IDs, diagnostic tables, boundary assignment and explicit update
timing retain the same definitions as the per-bunch path. Other grid models and
automatic ranges retain their existing execution paths.
Equal-length grids allocate only histogram and geometry workspaces; sorting
buffers are allocated only for equal-particle grids.

Responsibilities
----------------

``SortBunch`` and ``ReorganizeBunch`` perform global bucket assignment and may
reorder all particle arrays.  They invalidate every existing ``SliceSet``
because particle ranges can change.  ``Slicer`` then recomputes the selected
set at its position in the sequence.  Space-charge and beam-beam modules can
refer to different named sets and therefore use different slice meshes.

.. important::

   **After every executed SortBunch or ReorganizeBunch, run Slicer again before
   using slice information.** Sorting/regrouping clears the previous
   ``slice_id`` and ``slice_table``. Recompute each named ``SliceSet`` required
   by subsequent commands, even if Slicer already ran earlier in the same turn.

   Examples of actual execution order (each Slicer updates the set used by
   the following consumer):

   .. code-block:: text

      Correct:   SortBunch       -> Slicer -> SpaceCharge
      Correct:   ReorganizeBunch -> Slicer -> WakeField
      Incorrect: Slicer -> SortBunch -> SpaceCharge

   The last sequence requires another Slicer between SortBunch and SpaceCharge.
   Consumers do not automatically rebuild invalidated slices. This rule applies
   on both CPU and CUDA. Slicer itself does not require a preceding SortBunch
   when the existing particle order and bunch grouping are unchanged.

The mapping uses current array positions: global particle index ``i`` belongs
to ``slice_id[i - bunch.start_idx]``. ``tag`` moves with its particle during
sorting, but is not an index into ``slice_id``. Keeping the same tags therefore
does not make an old slice mapping reusable after sorting. See :doc:`reorganize`
for the grouping operation.

The stored particle coordinate is the continuous bunch-relative coordinate
:math:`z_{rel}`. ``z_rel`` slicing uses it directly; ``z_periodic`` and
``arrival_phase`` use temporary projections. None replaces the stored coordinate.

Slice coordinate and reuse
--------------------------

``Coordinate=z_rel`` is the default: bin the continuous time-scaled coordinate
:math:`z=\beta_b c(T_b-t_i)` directly for continuous wake timing.
SpaceCharge requires the separate ``z_periodic`` option described below.
The separate ``arrival_phase`` option forms a temporary periodic arrival
projection for coasting wakes. ``Periodic=true`` is an alias for that option.
``ring_position`` is not supported: z plus a nominal slot offset cannot supply
an exact physical ring position under this coordinate definition.

The latest explicit Slicer result is authoritative. Its saved z intervals,
widths and memberships are reused until the user executes Slicer again.
RF never rescales, recenters or recomputes these arrays. A local interval of
width :math:`\Delta z` in ``z_rel`` corresponds, at the current reference event, to
:math:`\Delta t=\Delta z/(\beta_b c)` and center
:math:`T_b-z_{slice}/(\beta_b c)`. Thus saved intervals remain intervals in
the current z coordinate, not frozen physical time intervals. Reuse accuracy
is the user's responsibility. Structural regrouping invalidates old local
indices and requires an explicit new Slicer execution.

Wake sources store their sampled physical times and widths when emitted.
Subsequent reference changes cannot move historical sources. Periodic
arrival-phase slices have a separately recorded common observation window;
see :doc:`wake_field` for that approximation. Quasi-static
SpaceCharge uses circumference-folded z intervals; time-scaled z is not an exact simultaneous
three-dimensional spatial distribution for arbitrary velocity spread.

``Z range mode`` remains ``auto`` or ``explicit``. Particle storage stays in
the configured float32 or float64 precision; local bin arithmetic uses float64.
Saved slice boundaries, centers, widths and density tables use float64 on both
CPU and CUDA. CUDA promotes a local working coordinate when particle storage
is float32, so rounding the slice geometry cannot turn a uniform wake grid
into a nonuniform one. The tracked particle coordinates remain unchanged.
Snapshots preserve continuous z and record the reference time, beta and
coordinate definition. Periodic snapshots also include ``slice_coordinate``.
The density is real-particle count per metre, not coulombs per metre.

The output metadata ``ZCoordinate="z_rel"`` describes the original particle ``z``
column; it is output metadata, not an input option or a default assignment.
``Coordinate`` identifies the selected slicing projection, while
``CoordinateDefinition="z=beta*c*(T-t)"``, ``ReferenceArrivalTime`` (seconds)
and ``ReferenceBeta`` define the original particle coordinate. Arrival-phase
snapshots additionally save ``ObservationTime`` (seconds),
``ObservationVelocity`` (metres per second), and ``SliceCoordinateDefinition``.
Together with ``Circumference``, these specify the saved observation window;
its time parameters can differ from the bunch reference parameters.

Configuration
-------------

An input entry uses the command name ``Slicer`` and a user-defined ``slice
set`` key:

.. code-block:: json

   {
       "sc_slicer": {
           "S (m)": 12.5,
           "Command": "Slicer",
           "Slice set": "space_charge",
           "Coordinate": "z_periodic",
           "Slice model": "equal_particle",
           "Number of slices": 128,
           "Z range mode": "auto",
           "Save turns": [[0], [100, 1000, 100]]
       }
   }

The range configuration is mode-specific.  No flat ``Z min``, ``Z max`` or
``Number of sigma`` fields are accepted.

``auto``
    Uses the actual minimum and maximum of the current live distribution.  It
    is the widest data-driven range and never excludes an observed outlier.
``explicit``
    Uses the fixed local interval supplied by ``Explicit``:

    .. code-block:: json

       "Z range mode": "explicit",
       "Explicit": {"Z min": -0.30, "Z max": 0.30}

All ranges are resolved at command execution. With ``z_rel`` or ``z_periodic``, values outside an explicit
interval are clipped to the first or last slice and a warning is logged; they
are never silently dropped.

Slice IDs are ordered from high to low selected slice coordinate: slice ``0`` is the
largest-z interval and slice ``N-1`` is the smallest-z interval.
Uniform-bin indexing uses FP64 arithmetic and checks the neighboring saved edge
after division, so a stored FP32 value just below a boundary keeps its correct
owner. Classification and the output table use the same generated edge array
on CPU, CUDA and the CUDA batch path. In increasing coordinate order, bins are
left-closed and right-open, except that the final bin includes the maximum.
This does not change the precision of the particle arrays or local projections.

Circumference-folded slicing for SpaceCharge
-----------------------------------------------

``Coordinate=z_periodic`` forms the temporary coordinate

.. math::

   z_{slice,i}=[(z_i+C/2)\bmod C]-C/2.

It uses the finite positive ring circumference, independently of the prescribed
clock. Both CPU and GPU support ``equal_length`` and ``equal_particle``, with
``auto`` or ``explicit`` ranges. ``auto`` uses the live folded-coordinate
minimum and maximum. ``explicit`` must lie within :math:`[-C/2,C/2]`; use that
full interval for a whole-ring mesh, retaining empty bins. A narrower interval
clips out-of-range folded coordinates to boundary slices. A bunch straddling
the seam can give an ``auto`` range spanning nearly the whole ring.

SpaceCharge accepts only this coordinate. Existing SC inputs must explicitly
add ``"Coordinate": "z_periodic"`` to their Slicer. Missing coordinate metadata,
``z_rel`` and ``arrival_phase`` are rejected by SC. This is the per-bunch
common-reference-velocity approximation; folding does not reconstruct an exact
simultaneous distribution for arbitrary velocity spread or combine overlapping
populations belonging to different bunches.

WakeField rejects ``z_periodic`` because its centers do not retain continuous
arrival times. Use separate named slice sets for SC and wakes. The legacy
``Periodic`` flag still selects only ``arrival_phase``; ``Coordinate`` identifies
the mode, including in saved output.

Periodic arrival slicing for coasting wakes
-------------------------------------------

``Coordinate=arrival_phase`` requires ``equal_length``, ``explicit`` and
``Explicit={"z min": -C, "z max": 0}``. At each explicit Slicer update, the
common observation event is :math:`T_{obs}=\Psi^{-1}(n+s/C)` and
:math:`v_{obs}=C f_{rev}(T_{obs})`. CPU and GPU form

.. math::

   u_i=\frac{v_{obs}}{C}(T_{obs}-t_i),\qquad
   z_{phase,i}=-C[(-u_i)\bmod1].

The bins represent the window :math:`[T_{obs},T_{obs}+C/v_{obs})`, and retain
that observation event until the next user update. Different bunch references
are allowed; all populations contributing to one periodic wake must use the
same saved observation window and circumference. Source bin 0 is the earliest
arrival. Lost particles have ID -1; stored z is never folded.
An exact integer phase maps to :math:`z_{phase}=0`, the start of the window
and slice 0. The right-endpoint phase is periodically identified with the window
start, rather than the last slice; no extra source passage is emitted. Slip diagnostics continue to use
the continuous, unreduced phase :math:`u_i`.

Use a Slicer at the wake location with causal history. Reusing a periodic
snapshot does not advance its observation window; the user must update it for
each new physical source passage. This differs from reusing local z intervals.
SpaceCharge requires its own ``z_periodic`` SliceSet. The one-passage-per-reference-
turn approximation and convergence requirements are explained in :doc:`wake_field`.

``Max phase slip`` defaults to 0.05 revolutions, with allowed values (0, 0.1].
Slicer records the continuous phase change at consecutive observations without
rejecting diagnostic output; WakeField checks the bound. Same-turn updates use
the same previous-turn baseline. Invalidation resets this diagnostic. It cannot
reconstruct unsampled crossings or prove convergence of short-wavelength modes.

Slice models
------------

``equal_length`` divides the resolved interval into ``N`` equal-width bins:

.. math::

   i = N-1-\operatorname{clip}\left(\left\lfloor
       \frac{z-z_{min}}{\Delta z}\right\rfloor,0,N-1\right),
   \qquad \Delta z = \frac{z_{max}-z_{min}}{N}.

``equal_particle`` sorts only temporary ``z`` values and indices, assigns IDs
by rank, and scatters those IDs back to the original particle order.  The
particle pool itself is unchanged.  Rank assignment keeps populations nearly
equal, including when several particles have identical coordinates.  Quantile
boundaries are computed with NumPy for the diagnostic geometry in
``slice_table``.

If the number of live particles is smaller than ``N``, only that many slices
can be populated.  The remaining slices are retained with zero count and a
warning is emitted.  ``effective_num_slices`` records
:math:`\min(N_{live},N)`; the configured mesh size is not changed.

SliceSet data
-------------

Each bunch has a mapping such as ``bunch.slice_sets["space_charge"]``.  After
execution, ``slice_id`` is an integer array aligned with the bunch's current
particle range; lost particles have ID ``-1``.  ``slice_table`` contains one
array per slice:

``z_min``, ``z_max``, ``z_center``
    Boundaries and center in the selected slice coordinate, listed from high-z to
    low-z (slice ``0`` is the high-z interval).
``delta_z``
    ``z_max - z_min`` for each slice.
``macro_count``
    Number of live macro-particles.
``real_charge``
    Equivalent number of real particles, ``macro_count * bunch.ratio``.  It is
    not Coulombs; multiply by the signed particle charge and elementary charge
    when physical charge is required.
``lind_density``
    Linear real-particle density, ``real_charge / delta_z``.
``effective_num_slices``
    Number of bins that can be populated by the current live population.

Results also record ``valid_turn`` and ``valid_s``.  A subsequent global
regrouping clears all particle-dependent fields and consumers must wait for a
new ``Slicer`` execution.

Snapshots
---------

``Save turns`` is optional and belongs to the command instance, not the shared
``SliceSet`` configuration.  Each item is either ``[turn]`` or
``[start, end, step]`` with inclusive endpoints.  Slicing still runs on every
turn; only selected turns write files.  Each selected execution writes a
same-instant particle HDF5 file (TFS when selected) (``tag``, ``z``, ``slice_id``, loss data) and a
per-slice TFS summary, also exported to CSV, to ``output/.../slice/``.  Particle and TFS summary files carry metadata that
identify the turn, position, beam, bunch, slice set, model and coordinate
convention.

An overlong end turn is clipped to the final simulated turn with a warning. A
range whose start lies outside the simulated turn range is ignored with a warning.
Negative starts are similarly clipped while preserving the configured step
sequence; malformed ranges (an end before its start), non-integer values and
non-positive steps remain errors.

Interface Parameters
--------------------

The following table lists the parameters accepted by a ``Slicer`` sequence
entry.  JSON keys are shown in the spelling used by generated input files;
PASS normalizes key case internally.

Common command parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - Parameter
     - Key
     - Type
     - Default / required
     - Description
   * - ``command``
     - ``"Command"``
     - str
     - Required
     - Must be ``"Slicer"``.  This selects the Slicer command implementation.
   * - ``s``
     - ``"S (m)"``
     - float
     - Required
     - Longitudinal machine position at which the slice data are updated.
   * - ``name``
     - sequence object key
     - str
     - Required as sequence key
     - Name of this command instance.  The sequence loader also passes it to
       the command for diagnostics.
   * - ``slice_set``
     - ``"Slice set"``
     - str
     - Required
     - Name of the bunch-owned ``SliceSet`` to update, for example
       ``"space_charge"`` or ``"beambeam_ip1"``.
   * - ``slice_model``
     - ``"Slice model"``
     - str
     - ``"equal_length"``
     - Particle-to-slice mapping model.  Supported values are
       ``"equal_length"`` and ``"equal_particle"``.
   * - ``coordinate``
     - ``"Coordinate"``
     - str
     - ``"z_rel"``
     - ``z_rel`` for continuous time slicing; ``z_periodic`` for circumference-folded SC slicing;
       ``arrival_phase`` for the wake observation clock. SC requires an explicit ``z_periodic`` selection.
   * - ``num_slices``
     - ``"Number of slices"``
     - int
     - 10
     - Configured number of longitudinal bins.  Must be at least 1; the value
       is retained even when the live population is smaller.
   * - ``z_range_mode``
     - ``"Z range mode"``
     - str
     - ``"auto"``
     - Selects ``"auto"`` or ``"explicit"``.
   * - ``save_turns``
     - ``"Save turns"``
     - list of int lists
     - ``[]``
     - Optional snapshot selections: ``[turn]`` or ``[start, end, step]``.
   * - ``periodic``
     - ``"Periodic"``
     - bool
     - false
     - Whole-ring arrival-phase projection for coasting WakeField; requires equal_length, explicit [-C,0].
   * - ``max_phase_slip``
     - ``"Max phase slip"``
     - float
     - 0.05
     - Maximum observed phase change per reference turn accepted by WakeField, in revolutions; (0, 0.1]. Diagnostic slicing records but does not enforce it.

Range-mode parameters
~~~~~~~~~~~~~~~~~~~~~

Only the block corresponding to ``Z range mode`` is used.  ``auto`` has no
mode-specific block.  ``explicit`` requires the ``Explicit`` block.

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - Mode / parameter
     - Key
     - Type
     - Default / required
     - Description
   * - ``auto``
     - ``"Z range mode"``
     - str
     - Optional
     - Uses the actual minimum and maximum of the current live distribution;
       observed outliers are included.
   * - ``explicit``
     - ``"Z range mode"``
     - str
     - Optional
     - Uses a fixed interval.  ``Explicit`` is required with this mode.
   * - ``explicit`` block
     - ``"Explicit"``
     - object
     - Required for explicit mode
     - Mode-specific object containing ``Z min`` and ``Z max``.
   * - ``z_min``
     - ``"Z min"``
     - float
     - Required for explicit mode
     - Lower bound in the selected slice coordinate; must be smaller than
       ``Z max``.
   * - ``z_max``
     - ``"Z max"``
     - float
     - Required for explicit mode
     - Upper bound in the selected slice coordinate; must be larger than
       ``Z min``.

Example configurations for the two range modes are:

.. code-block:: json

   {"Z range mode": "auto"}

   {
       "Z range mode": "explicit",
       "Explicit": {"Z min": -0.30, "Z max": 0.30}
   }

SliceSet runtime interface
~~~~~~~~~~~~~~~~~~~~~~~~~~

The command configuration is converted to one ``SliceSet`` per bunch.  The
following fields are available to SpaceCharge, BeamBeam and diagnostic code;
they are outputs of Slicer rather than additional JSON input parameters.

.. list-table::
   :widths: 23 18 15 44
   :header-rows: 1

   * - Field
     - Type
     - Valid before slicing
     - Description
   * - ``name``
     - str
     - Yes
     - User-defined lookup key from ``Slice set``.
   * - ``model``
     - str
     - Yes
     - Normalized slice model name.
   * - ``num_slices``
     - int
     - Yes
     - Configured mesh size.
   * - ``z_range_mode``
     - str
     - Yes
     - Normalized range mode.
   * - ``explicit``
     - ``ExplicitRange`` or None
     - Yes
     - Canonical explicit bounds when explicit mode is selected.
   * - ``slice_id``
     - int array
     - No
     - One ID per particle in the bunch's current particle range; lost
       particles have ``-1``.
   * - ``slice_table``
     - dict of arrays
     - No
     - Per-slice geometry and population arrays listed in the SliceSet data
       section above.
   * - ``valid_turn``
     - int or None
     - No
     - Simulation turn at which the result was generated.
   * - ``valid_s``
     - float or None
     - No
     - Sequence position at which the result was generated.

Configuration validation
~~~~~~~~~~~~~~~~~~~~~~~~

At beam initialization, repeated ``Slicer`` entries referring to the same
``Slice set`` must have identical ``Slice model``, ``Number of slices``, range
mode and (for explicit mode) the explicit block.  A conflicting definition raises
``ValueError`` and identifies both sequence entries.  At execution, an
explicit range that does not cover all live particles produces a warning;
out-of-range particles are clipped to the first or last slice.

``output_format`` (JSON ``"Output format"``) defaults to ``"hdf5-gzip1"`` and
accepts ``"hdf5"`` (uncompressed) and ``"tfs"``. The default uses gzip-1
with shuffle. The choice affects particle details only; slice summaries remain
TFS and CSV. See :doc:`monitor/table_output` for the HDF5 layout and reader.
