Slicer
======

The ``Slicer`` command assigns every live macro-particle in each bunch to a
longitudinal slice.  It stores the result in a named ``SliceSet`` owned by the
bunch.  Slicing is a local classification operation: it does not reorder any
particle array and it does not change bunch membership.

Responsibilities
----------------

``SortBunch`` and ``ReorganizeBunch`` perform global bucket assignment and may
reorder all particle arrays.  They invalidate every existing ``SliceSet``
because particle ranges can change.  ``Slicer`` then recomputes the selected
set at its position in the sequence.  Space-charge and beam-beam modules can
refer to different named sets and therefore use different slice meshes.

The stored particle coordinate is the continuous bunch-relative coordinate
:math:`z_{rel}`.  Slicing never folds it around the ring and never uses the
laboratory coordinate :math:`z_{lab}`.

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

All ranges are resolved at command execution.  Values outside an explicit
interval are clipped to the first or last slice and a warning is logged; they
are never silently dropped.

Slice IDs are ordered from high ``z_rel`` to low ``z_rel``: slice ``0`` is the
largest-z interval and slice ``N-1`` is the smallest-z interval.

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
    Geometric boundaries and center in ``z_rel``, listed from high-z to
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
same-instant particle TFS file (``tag``, ``z``, ``slice_id``, loss data) and a
per-slice TFS summary to ``output/.../slice/``.  Both files carry headers that
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
     - Lower bound in local ``z_rel`` coordinates; must be smaller than
       ``Z max``.
   * - ``z_max``
     - ``"Z max"``
     - float
     - Required for explicit mode
     - Upper bound in local ``z_rel`` coordinates; must be larger than
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
