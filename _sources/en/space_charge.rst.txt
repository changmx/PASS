Space-Charge Effect (SpaceCharge)
==================================================================

``SpaceCharge`` computes transverse space-charge momentum increments from user-provided longitudinal slices. It supports PIC, fixed analytic profiles (``frozen``), and profiles updated from current slice moments (``quasi-frozen``), on CPU or NVIDIA GPU. The quasi-static 2.5-D model includes relativistic electric/magnetic self-force cancellation and does not compute a longitudinal space-charge force.

Configuration prerequisites
------------------------------------------------------

Enable the top-level ``Space charge`` block and define a named configuration, then execute its named ``Slicer``. Space charge requires ``Coordinate="z_periodic"``; regenerate slices after regrouping. Specify interaction length through an independent command or internal element configuration. An independent command does not transport particles over that length. See :ref:`en-longitudinal-reference` for coordinates. The example below is an input fragment; injection and transport must also be supplied.

Configuration Example
---------------------

The configurations are declared once in the top-level ``Space charge`` block.
Sequence commands refer to one configuration by name:

.. code-block:: json

   {
       "Backend (gpu/cpu)": "cpu",
       "Space charge": {
           "Enabled": true,
           "Configurations": {
               "round_pipe": {
                   "Slice set": "space_charge",
                   "Nx": 129,
                   "Ny": 129,
                   "Grid Width X (m)": 0.08,
                   "Grid Width Y (m)": 0.08,
                   "Method": "pic",
                   "Solver": "fd_dirichlet",
                   "Particle Deposition Method": "CIC"
               }
           }
       },
       "Sequence": {
           "sc_slicer": {
               "S (m)": 10.0,
               "Command": "Slicer",
               "Slice set": "space_charge",
               "Coordinate": "z_periodic",
               "Slice model": "equal_length",
               "Number of slices": 64,
               "Z range mode": "auto"
           },
           "sc_kick": {
               "S (m)": 10.0,
               "Command": "SpaceCharge",
               "Configuration": "round_pipe",
               "SC length (m)": 0.10,
               "Aperture type": "circle",
               "Aperture value": [0.035],
               "Save field": false,
               "Save potential": false,
               "Save density": false,
               "Save turns": []
           }
       }
   }

Only configurations referenced by a ``SpaceCharge`` sequence entry or an
element's internal ``Space charge`` object are built. See :ref:`en-internal-space-charge`
for internal configuration, midpoint scheduling, supported elements and examples.
Within one beam, commands that reference the same configuration name share
one grid. Dirichlet solver resources are cached by configuration and resolved
command aperture: equal walls reuse the factorization, different walls require
separate resources. FFT kernels are shared regardless of the loss aperture.
Different configuration names own independent resources even when their
values are identical.

Each command defines its own aperture. For example:

.. code-block:: json

   {
       "sc_wide": {
           "Command": "SpaceCharge", "S (m)": 2.5,
           "Configuration": "default", "SC length (m)": 0.1,
           "Aperture type": "circle", "Aperture value": [0.006]
       },
       "sc_narrow": {
           "Command": "SpaceCharge", "S (m)": 7.5,
           "Configuration": "default", "SC length (m)": 0.1,
           "Aperture type": "circle", "Aperture value": [0.003]
       }
   }

This Sequence fragment references one already defined ``default`` configuration
at both points, with radii of 6 mm and 3 mm. For ``fd_dirichlet`` these are
also different conducting walls, so two solvers share the same grid. For
``fft_free_space`` or analytic methods they affect losses only. These circular
apertures are invalid for ``dst_dirichlet``.

Tracking Methods and Analytic Profiles
------------------------------------------

``Method`` is ``pic`` (default), ``frozen``, or ``quasi-frozen``. ``Solver``
specifies both the field algorithm/profile and its boundary condition:

.. list-table:: Supported combinations
   :header-rows: 1
   :widths: 22 48 30

   * - Method
     - Solver
     - Model
   * - ``pic``
     - ``fft_free_space``
     - Open-boundary Green-function PIC.
   * - ``pic``
     - ``fd_dirichlet``
     - Zero-potential conductor; supported continuous chamber geometries.
   * - ``pic``
     - ``dst_dirichlet``
     - Zero-potential conductor; full grid-aligned rectangle only.
   * - ``frozen``, ``quasi-frozen``
     - ``gaussian_round_free_space``, ``gaussian_ellipse_free_space``
     - Round Gaussian or elliptic Gaussian (Bassetti--Erskine).
   * - ``frozen``, ``quasi-frozen``
     - ``uniform_round_free_space``, ``uniform_ellipse_free_space``
     - Uniform disk or ellipse, including the exterior field.
   * - ``frozen``, ``quasi-frozen``
     - ``parabolic_round_free_space``, ``parabolic_ellipse_free_space``
     - Parabolic spatial density, the projection of a uniform 4D waterbag.

In ``frozen``, all slices using one configuration share fixed transverse
center, sizes and orientation. Defaults for omitted center and angle are zero;
the solver-specific sizes are required. Different lattice positions may use
different configurations. Slice charge and ``delta_z`` remain current inputs:
the transverse shape is frozen, not the complete electromagnetic field.

In ``quasi-frozen``, each kick recomputes the centroid and population covariance
of each slice's current live, assigned particles. Moments use denominator
:math:`N`, not :math:`N-1`; all macroparticles within a bunch have the same
physical weight. The source charge is :math:`Q_k=N_k R Z e`.

For an elliptic profile, covariance eigenvectors define the principal axes.
Gaussian sizes are the square roots of the eigenvalues; uniform semi-axes
are twice these RMS sizes; parabolic semi-axes are sqrt(6) times the RMS sizes.
The first reported principal size is the larger.
The counterclockwise major-axis angle is reported modulo pi in
:math:`[-\pi/2,\pi/2)`. Fields are evaluated after translation/rotation and
rotated back to the original transverse axes.

For a round profile the moment-matched radius rule is

.. math::

   \sigma^2=\frac{\operatorname{Var}(x)+\operatorname{Var}(y)}{2},
   \qquad R=2\sigma.

``gaussian_round_free_space`` uses sigma; ``uniform_round_free_space`` uses R.
``parabolic_round_free_space`` uses :math:`R=\sqrt6\,\sigma`.
This preserves the centroid-relative radial second moment. It is a deliberate
round approximation for non-round populations, not an exact reconstruction of
their field. Choosing a uniform profile does not turn the particles into a KV
distribution. Likewise, the parabolic model assumes a parabolic projected
density even if the evolved phase-space distribution is no longer a waterbag.
No automatic profile switching occurs. Density normalization and the exact
interior/exterior parabolic field are given in :doc:`field_solver`.

Empty slices produce zero field and charge. Nonempty quasi-frozen round slices
require at least two particles and positive radial variance; elliptic slices
require at least three particles and a nondegenerate covariance. Specifically,
the smaller eigenvalue must exceed ``64 * float64_epsilon * larger_eigenvalue``.
Invalid sizes or covariance produce a slice-specific error, without silently
skipping a charged slice or substituting a minimum size. This is a numerical
validity check, not a guarantee of good statistical sampling.

Only free-space analytic profiles and transverse 2.5-D forces are implemented.
There is no conducting-wall analytic correction or longitudinal space-charge
force in these methods. Significant halo, multiple peaks or non-Gaussian
structure generally cannot be represented by their few transverse moments.

For example, named analytic configurations can be written as:

.. code-block:: json

   {
       "fixed_gaussian": {
           "Method": "frozen",
           "Solver": "gaussian_ellipse_free_space",
           "Slice set": "space_charge",
           "Center X (m)": 0.001,
           "Center Y (m)": 0.0,
           "Sigma X (m)": 0.004,
           "Sigma Y (m)": 0.002,
           "Angle (rad)": 0.2
       },
       "updated_gaussian": {
           "Method": "quasi-frozen",
           "Solver": "gaussian_ellipse_free_space",
           "Slice set": "space_charge"
       }
   }

These objects belong inside ``Space charge.Configurations``. An ordinary
``SpaceCharge`` sequence command selects one by ``Configuration``. The analytic
particle kick does not deposit charge or interpolate fields. Grid geometry
defines the default loss aperture and optional field/density sampling.
With an explicit aperture or ``off``, changing diagnostic grid extent cannot
change tracking; changing the default aperture can change particle losses.

Interface Parameters
--------------------

Top-level ``Space charge`` block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - Parameter
     - Key
     - Type
     - Default / required
     - Description
   * - ``enabled``
     - ``"Enabled"``
     - bool
     - ``false``
     - Enables space charge for this beam input.  When false, configuration
       contents and sequence commands are ignored and no resources are built.
   * - ``configurations``
     - ``"Configurations"``
     - object
     - ``{}``
     - Mapping from a non-empty user-defined name to one
       ``SpaceChargeResourceConfig`` object.

Named resource configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - Parameter
     - Key
     - Type
     - Default
     - Description
   * - ``slice_set``
     - ``Slice set``
     - str
     - ``space_charge``
     - Name of supplied bunch-local SliceSet; no Slicer sequence/history check.
   * - ``method``
     - ``Method``
     - str
     - ``pic``
     - pic, frozen, or quasi-frozen.
   * - ``solver``
     - ``Solver``
     - str
     - ``fd_dirichlet``
     - One supported method/solver combination listed above.
   * - ``nx / ny``
     - ``Nx / Ny``
     - int
     - ``128``
     - PIC or diagnostic grid nodes, each at least 3.
   * - ``grid_width_x / grid_width_y``
     - ``Grid Width X (m) / Grid Width Y (m)``
     - float or null
     - ``0.02`` if neither pair is supplied
     - Positive full widths, centered at zero; supply both axes together.
   * - ``grid_half_width_x / grid_half_width_y``
     - ``Grid Half Width X (m) / Grid Half Width Y (m)``
     - float or null
     - ``null``
     - Positive half widths; alternative to the full-width pair, never mixed.
   * - ``deposition_method``
     - ``Particle Deposition Method``
     - str or null
     - ``null``
     - PIC only: null selects CIC; CIC or TSC with matching gather.
   * - ``center_x / center_y``
     - ``Center X (m) / Center Y (m)``
     - float or null
     - ``null``
     - Frozen only; null means zero. Fixed for all slices.
   * - ``angle``
     - ``Angle (rad)``
     - float or null
     - ``null``
     - Frozen ellipse only: counterclockwise local-x axis angle; null means zero. Round requires zero or null.
   * - ``sigma``
     - ``Sigma (m)``
     - float or null
     - ``null``
     - Required positive RMS size for frozen round Gaussian.
   * - ``sigma_x / sigma_y``
     - ``Sigma X (m) / Sigma Y (m)``
     - float or null
     - ``null``
     - Required positive principal RMS sizes for frozen elliptic Gaussian.
   * - ``radius``
     - ``Radius (m)``
     - float or null
     - ``null``
     - Required positive support radius for frozen uniform or parabolic disk.
   * - ``a / b``
     - ``Semi-axis A (m) / Semi-axis B (m)``
     - float or null
     - ``null``
     - Required positive support semi-axes for frozen uniform or parabolic ellipse.


``SpaceCharge`` sequence command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
     - ``"SpaceCharge"``
     - Selects the space-charge command implementation.
   * - ``s``
     - ``"S (m)"``
     - float
     - 0.0 m
     - Longitudinal machine position of the thin kick; must be finite.
   * - ``name``
     - sequence object key
     - str
     - Required as sequence key
     - Command instance name used in logging and the output path.
   * - ``configuration``
     - ``"Configuration"``
     - str
     - Required
     - Exact name in the top-level ``Configurations`` mapping.  Leading or
       trailing whitespace is rejected.
   * - ``sc_length``
     - ``"SC length (m)"``
     - float
     - 0.0 m
     - Non-negative effective interaction length.  Zero disables the kick at
       this command and produces no snapshot.
       An enabled local loss aperture is still checked without requiring slices.
   * - ``sc_start``
     - ``"SC start (m)"``
     - float or null
     - null
     - Start of the represented integration interval, used only by
       :ref:`en-sc-coverage`. It does not move the kick or transport particles.
   * - ``aperture_type``
     - ``"Aperture type"``
     - str
     - ``"default"``
     - Default resolves to the grid rectangle. Supports off, circle, rectangle,
       ellipse, rectcircle, rectellipse, racetrack, octagon and polygon.
       For Dirichlet solvers this also defines the conducting wall; off is rejected.
   * - ``aperture_value``
     - ``"Aperture value"``
     - list
     - ``[]``
     - Dimensions in meters: [radius] for circle, [half-width, half-height] for
       rectangle, [a, b] for ellipse, or a vertex list for polygon. See :doc:`aperture`.
   * - ``save_field``
     - ``"Save field"``
     - bool
     - ``false``
     - Saves ``integrated_Ex`` and ``integrated_Ey`` when the turn is selected.
   * - ``save_potential``
     - ``"Save potential"``
     - bool
     - ``false``
     - PIC only. Analytic methods reject this request at initialization.
   * - ``save_density``
     - ``"Save density"``
     - bool
     - ``false``
     - Saves deposited PIC density or sampled analytic model density.
   * - ``save_turns``
     - ``"Save turns"``
     - list of int lists or int list
     - ``[]``
     - Snapshot selections ``[turn]`` or ``[start, end, step]`` with inclusive
       endpoints.  A single ``[0]`` is accepted as shorthand.

Diagnostic Output
-----------------

A snapshot is written only after a successful command execution when at least
one save flag is true and the current turn matches ``Save turns``.  An empty
``Save turns`` writes nothing.  Files are placed under

.. code-block:: text

   <output_dir>/space_charge/<command_name>/turn_NNNNNN/

Each bunch is stored in a separate HDF5 file.  Grid datasets use array order
``(slice, y, x)``.

.. list-table::
   :widths: 24 18 18 40
   :header-rows: 1

   * - Dataset
     - Shape
     - Unit
     - Meaning
   * - ``x``, ``y``
     - ``(nx,)``, ``(ny,)``
     - m
     - Horizontal and vertical grid-node coordinates.
   * - ``slice_id``
     - ``(n_slice,)``
     - -
     - Stored slice indices.
   * - ``delta_z``
     - ``(n_slice,)``
     - m
     - Slice widths used later to convert integrated fields to average fields.
   * - ``slice_charge``
     - ``(n_slice,)``
     - C
     - Deposited PIC charge or current analytic source charge in each slice.
   * - ``charge_density``
     - ``(n_slice, ny, nx)``
     - C/m\ :sup:`2`
     - Deposited PIC source or sampled analytic density when requested.
   * - ``potential``
     - ``(n_slice, ny, nx)``
     - V m
     - Integrated potential; present when ``Save potential`` is true.
   * - ``integrated_Ex``, ``integrated_Ey``
     - ``(n_slice, ny, nx)``
     - V
     - Integrated fields; present when ``Save field`` is true.  Divide by
       ``delta_z`` for the average field in V/m.

File attributes record the solver and deposition method, grid bounds and
spacing, turn, beam and bunch identity, harmonic metadata, command position
and length, charge per macroparticle, particle counts, precision, and random
seed. ``aperture_type`` and the JSON-encoded ``aperture_value`` record the local
resolved geometry. ``aperture_role`` is ``loss_and_conductor`` for Dirichlet
or ``loss_only`` for free-space methods. ``potential_gauge`` is ``boundary_zero`` for ``fd_dirichlet`` and
``dst_dirichlet`` and ``kernel_reference`` for ``fft_free_space``.

Analytic-profile diagnostics
----------------------------

HDF5 ``schema_version`` is ``3``. Attributes include ``method``, the full
``solver`` name and ``grid_role`` (``tracking`` or ``diagnostic``). Analytic
snapshots additionally store ``macro_count``, ``center_x``, ``center_y``,
``size_x``, ``size_y`` and ``angle`` per slice. ``size_convention`` is
``principal_rms`` for Gaussian profiles, ``parabolic_semi_axes`` for parabolic
profiles, or ``uniform_semi_axes`` for uniform
profiles. ``parameter_source`` is ``configuration`` or
``current_slice_population_moments``. Empty-slice parameters are NaN; their
charge and sampled fields/density are zero. A frozen ellipse retains the user's
axis order, while quasi-frozen ellipses report the major axis first.

Analytic ``Save potential`` is not implemented and is rejected even if no save
turns are selected; ``potential_gauge`` is ``not_computed``. The density grid is
an analytic sample, not deposited charge: its finite-grid quadrature need not
equal ``slice_charge``. Diagnostics never alter the source or particle kick.

Grid Extent and Aperture Validation
-----------------------------------

Supply either ``Grid Width X/Y (m)`` or ``Grid Half Width X/Y (m)`` as a complete
pair; the unused pair must be absent or null. All widths must be positive and
finite. Omitting both pairs selects full widths of 0.02 m. Spacing inputs
``Dx``/``Dy`` and explicit-bound mapping inputs are rejected. The grid is
centered at zero. For half widths 0.04 m and 0.02 m with 129 nodes per axis,
the ranges are [-0.04, 0.04] m and [-0.02, 0.02] m, and the nodal spacings
are 0.000625 m and 0.0003125 m. ``SpaceCharge.print()`` reports both ranges,
node counts, spacings, and the resolved aperture and its role.

Initialization checks every referenced command before tracking:

- ``fd_dirichlet``: a finite supported aperture must fit completely in the grid.
- ``dst_dirichlet``: the aperture must be an axis-aligned rectangle identical
  to the complete grid. A smaller rectangle or any other shape is rejected.
- ``fft_free_space``: a finite loss aperture must fit in the grid; ``off`` is
  allowed, but participating particles still must lie in the grid.
- Analytic solvers: an explicit loss aperture is independent of diagnostic
  grid extent; ``off`` is allowed. The default still resolves to the grid rectangle.

A PIC aperture containing no active nodes is rejected during initialization.
During deposition, a surviving participant with no active stencil node raises
a mesh-resolution error rather than silently losing source charge.

Configuration Rules
-------------------

- Omitting ``Space charge`` is equivalent to ``"Enabled": false``.
- When disabled, nested configuration contents are intentionally not
  validated; all ``SpaceCharge`` sequence entries are inert.
- Undefined configurations are rejected during initialization. Missing or
  structurally invalid SliceSet data are rejected when the kick consumes them.
- Unreferenced configurations produce a warning and allocate no resources.
- Configuration names and command references must be non-empty and may not
  contain surrounding whitespace.
- Set mesh and solver fields in top-level named configurations, not inline in
  ``SpaceCharge`` sequence commands.
- ``"Backend (gpu/cpu)": "gpu"`` supports PIC, frozen and quasi-frozen methods,
  including loss apertures, with the configured particle precision.

Periodic z slicing
------------------

SpaceCharge requires ``Coordinate=z_periodic``. Slicer temporarily folds the
continuous coordinate :math:`z=\beta_b c(T_b-t_i)` as
:math:`z_{slice}=[(z+C/2)\bmod C]-C/2`, without changing particle z. The
prescribed clock does not enter this projection. ``z_rel``, ``arrival_phase``
and missing coordinate metadata are rejected in both CPU and GPU SC paths.
SC configuration requires selecting ``z_periodic`` in their Slicer.
For a whole-ring equal-length mesh, select explicit :math:`[-C/2,C/2]` limits;
auto ranges use the folded live minimum and maximum. See :doc:`slicer`.

This per-bunch common-reference-velocity approximation does not reconstruct an
exact simultaneous 3D distribution for arbitrary velocity spread.
Each live macro particle is deposited using the latest saved
membership, and integrated transverse fields are divided by saved ``delta_z``.
RF does not rescale these z intervals. Users update slices manually, including
for internal SC nodes. Different bunches are still solved separately, with
the existing transverse boundary conditions and solvers.

.. _en-internal-space-charge:

Element Slicing and Internal Space Charge
------------------------------------------------------------

Three independent resolutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Num slices`` controls external-field body transport. The optional element
``Space charge`` object controls SC integration along that body. The named
``Slicer`` result controls longitudinal particle bins. These are three distinct
quantities; changing one does not implicitly update either of the others.

Internal nodes
reuse the same named configurations, field solvers and kick normalization as
``SpaceCharge``. An independent ``SpaceCharge`` sequence command remains
available for Twiss-point tracking and explicit placement outside an element.
That command uses the particle state when it executes and its own ``SC length (m)``;
it does not transport particles through that length.

Supported elements
~~~~~~~~~~~~~~~~~~

Internal SC requires positive body length and supports CPU and GPU. Supported runtime
commands are ``Drift``, ``SBend``, ``Quadrupole``, ``Sextupole``, ``Octupole``,
``Multipole``, ``Kicker``, ``Bump``, ``Solenoid`` and ``ElSeparator``.

* Multipole strengths and kicker angles are integrated strengths: each external
  substep uses its fraction of the original length. Signed Yoshida stages retain
  their signed external strengths.
* Drift, matrix quadrupole and pure-solenoid maps can advance through a partial
  body length. They execute half a short map, SC, then the remaining half when
  a midpoint is requested.
* A solenoid retains its solenoidal body map; its body is not replaced by drift.
  A solenoid with superimposed multipoles uses its existing Sol-Kick-Sol steps.
* An electrostatic separator uses the relativistic uniform-electric-field analytic
  map between internal SC nodes, with hard-edge potential matching at entry and
  exit. First contact of the trajectory with the septum or aperture determines
  loss. SC acts in the beam frame on all surviving particles, including those
  in the field-free region.
* Bend entrance and exit maps are executed once at the physical boundaries.

Thin elements retain their thin kicks and cannot request internal SC. Place
explicit SC commands nearby for ``RFCavity`` and ``Exciter``. ``Num slices`` for Drift and ElSeparator defaults to one. Without
internal SC, external slicing also works on the GPU. GPU internal nodes follow
the same physical placements and positive SC weights, including signed Yoshida
external stages. Polar bend calculations use double intermediates with FP32
particle storage to reduce cancellation error.

Scheduling rule
~~~~~~~~~~~~~~~

For body length :math:`L`, requested external count :math:`N_e` and SC kick count
:math:`N_c`, the scheduler computes

.. math::

   m=\left\lceil N_e/N_c\right\rceil,\qquad
   N_{e,\mathrm{actual}}=mN_c,\qquad
   h=L/N_{e,\mathrm{actual}},\qquad H=L/N_c.

Thus each SC integration interval contains exactly :math:`m` external slices.
The requested external resolution is never reduced. ``num_slice`` retains the
requested count, while ``slice_plan.num_slices`` records the actual count.
With SC disabled or absent, the actual count equals the external count.

``S (m)`` is the element exit coordinate. Node :math:`j=0,\ldots,N_c-1` has

.. math::

   s_j=s_{\mathrm{exit}}-L+(j+1/2)H,\qquad L_{\mathrm{sc},j}=H>0.

The sum of internal SC weights is :math:`L`. The SC weight is independent of
the length of the external substep where it is evaluated. Internal configuration
does not accept a separate ``SC length (m)``; the scheduler derives that value.

* Odd :math:`m`: execute SC at the center of the middle external slice, after
  its full central external kick.
* Even :math:`m`: execute SC between the two middle complete external slices.

.. list-table:: Examples
   :header-rows: 1
   :widths: 15 15 15 20 35

   * - Requested external
     - SC kicks
     - Actual external
     - Slices per SC
     - Location
   * - 4
     - 10
     - 10
     - 1
     - External slice center
   * - 12
     - 4
     - 12
     - 3
     - Middle external slice center
   * - 10
     - 3
     - 12
     - 4
     - Between middle complete slices

For ``uniform``, the center hook follows the full central external kick. For
``yoshida4``, the hook follows the central kick of the second, negative-length
second-order stage, which is the algebraic midpoint of the complete positive
external slice. Only this hook executes SC; the other Yoshida stages do not.
Its SC weight is still positive :math:`H`, never a signed Yoshida stage length.
The external central kick is not split into two half kicks.

This is a midpoint SC coupling. Do not infer fourth-order accuracy of the coupled
map from the external ``yoshida4`` setting. Validate external resolution and SC
resolution separately; the smooth linear uniform-beam verification approaches
second-order convergence for this coupling. Frozen longitudinal membership also
does not imply arbitrary external and SC maps commute.

Configuration and use
~~~~~~~~~~~~~~~~~~~~~

All supported element schemas expose ``num_slices`` (JSON ``Num slices``,
default 1) and ``space_charge`` (JSON ``Space charge``, default null).
``ElementSpaceCharge`` has the following interface:

.. list-table::
   :header-rows: 1
   :widths: 25 25 15 35

   * - Python field
     - JSON key
     - Default
     - Meaning
   * - ``configuration``
     - ``Configuration``
     - Required
     - Top-level named SC resource
   * - ``num_kicks``
     - ``Num kicks``
     - 1
     - Positive integer; bool, float and numeric strings are rejected
   * - ``aperture_type`` / ``aperture_value``
     - ``Aperture type`` / ``Aperture value``
     - ``default`` / []
     - Defaults to the parent aperture; conflicting requests are overridden with a warning
   * - ``save_field`` / ``save_potential`` / ``save_density``
     - ``Save field`` / ``Save potential`` / ``Save density``
     - false
     - Same diagnostic support as the selected explicit SC solver
   * - ``save_turns``
     - ``Save turns``
     - []
     - Existing format: e.g. [[0]] or [[0, 100, 10]]

For an existing sequence and named configuration ``sc_default``:

.. code-block:: python

   from PASS.para.schema.elements import QuadrupoleItem
   from PASS.para.schema.space_charge import ElementSpaceCharge

   sequence.add("q1", QuadrupoleItem(
       s=1.0, length=0.4, k1l=0.12,
       num_slices=10, integrator="yoshida4",
       space_charge=ElementSpaceCharge(
           configuration="sc_default", num_kicks=3,
           save_density=True, save_turns=[[0]],
       ),
   ))

This requests 10 external slices; tracking uses 12 slices and three SC kicks
with weight 0.4/3 m. Enable the top-level ``Space charge.Enabled`` switch and
define ``sc_default`` under ``Configurations``. Supply its named SliceSet
before the element, normally by placing ``Slicer`` upstream.

Internal nodes reuse existing ``slice_id`` and ``slice_table.delta_z`` without
rebinning, modifying those widths, or folding ``p.z``. Fields use current
transverse particle coordinates at each invocation, subject to the selected
PIC/frozen/quasi-frozen model. The owning element advances ``bunch.t0`` once
for its full length. Each node acts only on the bunch currently being tracked.

Internal SC always uses the owning element's aperture, for both particle
losses and Dirichlet conducting walls. Omit the nested aperture fields: their
``default`` means inherit the element. Conflicting explicit nested aperture
values produce one warning per element/configuration mismatch and are replaced
by the element aperture before solver resources are built. The input object is
preserved; printing reports the effective values with ``Source=element``.

The generic element ``default`` remains the +/-1 m rectangle, not the SC grid
rectangle. An element aperture of ``off`` stays off: free-space PIC and analytic
methods permit it, while Dirichlet solvers require a finite element aperture.
PIC grids must contain the inherited aperture, and DST requires exactly the
full grid-aligned rectangle. Incompatible geometry is an initialization error;
the program does not silently choose a different wall or solver. Particles
outside a free-space PIC grid with losses disabled still cause the existing
grid-domain error. Independent explicit SC commands retain their own apertures.

For each bunch passage, ordinary elements, including ``Bump``, check the aperture
once at the exit when no internal SC is active. With :math:`K` active internal SC
nodes, each SC entry point checks once before source evaluation, and the element
checks once more at the exit: :math:`K+1` checks in total. Other external slice
boundaries do not add checks. Newly lost particles are excluded from sources and
kicks; first recorded loss positions are preserved. PIC field-domain validation
remains active at every SC node. ``ElSeparator`` additionally retains its separate
first-contact collision handling along each drift subsegment.
Nodes share resources by configuration and effective aperture. Their snapshots
are stored under
``<space_charge_output>/<element>/internal_sc/node_000000/turn_000000/``.
Filenames retain beam and bunch identifiers. HDF5 records ``parent_element``,
``internal_node_index``, ``s``, ``sc_length`` and ``sc_start``. For explicit SC,
``sc_start`` is written only when supplied. It describes the represented
integration interval and does not modify the force calculation.

Each supported element's ``print()`` includes requested/actual external slice
counts, slice length, SC configuration/method/solver/SliceSet, kick count and
placement, per-kick/total SC lengths, first/last node positions, effective
aperture and output settings. Missing internal SC prints ``off``; global
disabling prints ``disabled by top-level Space charge.Enabled``. Internal SC
execution time remains included in the parent element timing.

When importing MAD-X elements, merged plain drifts retain the final exit S;
drifts with local SC, non-default slicing or apertures are not merged.

.. _en-sc-coverage:

Pre-tracking integration-length and coverage checks
---------------------------------------------------

``Executor.run`` checks each enabled beam's actual command sequence before any
particle tracking. Internal weights and explicit command lengths are added once
per pass, without multiplication by bunch count or number of simulation turns.
Both total weight and interval coverage are checked: equal total length does
not rule out a gap compensated by an overlapping interval.

These fields belong to the top-level ``Space charge`` block, alongside
``Enabled`` and ``Configurations``. They are also editable in the GUI SC form.

.. list-table:: Coverage configuration
   :header-rows: 1
   :widths: 25 25 15 35

   * - Python field
     - JSON key
     - Default
     - Meaning
   * - ``coverage_check``
     - ``Coverage check``
     - ``warn``
     - ``warn`` reports issues and continues; ``error`` rejects mismatches or
       incomplete checks before tracking; ``off`` skips this check.
   * - ``coverage_mode``
     - ``Coverage mode``
     - ``full-ring``
     - Full-ring mode compares weight with ``Circumference (m)`` and checks
       gaps and overlaps. ``partial`` permits uncovered regions but checks overlaps.
   * - ``expected_sc_length``
     - ``Expected SC length (m)``
     - null
     - Optional non-negative total-weight target for ``partial`` only.
       Full-ring mode always uses circumference and rejects this override.

For strict full-ring validation, merge these settings into the existing
top-level ``Space charge`` block:

.. code-block:: json

   {
       "Coverage check": "error",
       "Coverage mode": "full-ring"
   }

Internal intervals are known from their element bodies. An explicit SC command
may supply optional ``SC start (m)`` (Python ``sc_start``): together with
``SC length (m)`` it represents the half-open interval
:math:`[s_{start},s_{start}+L_{sc})`. This start is independent of the kick's
``S (m)``. For example, a kick at 0.7 m representing [0.5, 0.9) m can use:

.. code-block:: json

   {
       "Command": "SpaceCharge",
       "S (m)": 0.7,
       "Configuration": "sc_default",
       "SC start (m)": 0.5,
       "SC length (m)": 0.4
   }

Intervals are compared periodically using the ring circumference, including
intervals crossing s=0 and lengths covering multiple turns. Particle coordinates
are not wrapped. The absolute comparison tolerance is
``max(1e-12 m, 1e-10 * circumference)``.

If ``SC start (m)`` is omitted for a positive-length explicit command, its
weight still counts, but the checker does not invent an interval centered on
the kick. The report is ``incomplete``. Uncovered regions of the known intervals
then cannot be certified as physical gaps. Strict mode rejects incomplete
coverage as well as mismatches; warning mode preserves existing workflows.
Disabled and zero-weight explicit commands do not contribute.

The log reports total and target lengths, kick counts, status and the first ten
gap/overlap regions. Complete contributions and regions are stored in
``<space_charge_output>/coverage_beam0.json`` (one file per enabled checked beam),
and in ``sim.space_charge_coverage[beam_id]``. A strict failure writes the report
before raising. Partial mode reports uncovered regions for information without
treating them as errors. No lengths or user intervals are automatically adjusted
to make the check pass. An invalid/missing circumference makes the periodic
check incomplete. A globally disabled SC module bypasses these checks.

Precision and convergence
--------------------------------------------------

Particle coordinates may use float32 or float64. Higher-precision field or reference intermediates do not remove float32 storage effects near slice boundaries or accumulated transport rounding. CPU/GPU summation order and sparse factorization can differ; compare results with problem-appropriate tolerances rather than bitwise equality. Check convergence independently in macro-particle count, longitudinal slices, transverse extent and resolution, and space-charge interaction-point count.

Physical Model
--------------

Macroparticle charge and transverse source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For one bunch, each live macroparticle represents ``bunch.ratio`` real
particles.  Its signed source charge is

.. math::

   q_{\mathrm{macro}} = R\,Z e,

where :math:`R` is ``bunch.ratio``, :math:`Z` is the signed charge number
``bunch.num_charge``, and :math:`e` is the elementary charge.  For slice
:math:`k`, the deposition step constructs the longitudinally integrated
transverse charge density

.. math::

   \Sigma_k(x_i,y_j)
   = \frac{1}{\Delta x\,\Delta y}
     \sum_{n\in k} q_{\mathrm{macro},n} W_{ij,n},

with units C/m\ :sup:`2`.  ``CIC`` uses four grid nodes and ``TSC`` uses nine.
Lost particles and particles assigned slice ID -1 do not contribute. Other
invalid slice IDs are errors. The command first marks particles on or outside
its aperture as lost. Surviving participating PIC particles outside the grid
are a configuration error.

Integrated potential and field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The field solver treats every slice as a two-dimensional Poisson problem,

.. math::

   -\nabla_\perp^2 \Psi_k = \frac{\Sigma_k}{\epsilon_0},
   \qquad
   \boldsymbol{\mathcal E}_{\perp,k} = -\nabla_\perp\Psi_k.

Here :math:`\Psi_k` is the longitudinally integrated potential in V m and
:math:`\boldsymbol{\mathcal E}_{\perp,k}` is the integrated transverse field
in V.  After gathering the grid field to each particle, ``SpaceCharge`` uses
the corresponding ``SliceSet.delta_z`` only to obtain the average field,

.. math::

   \overline{E}_{x,k}=\frac{\mathcal E_{x,k}}{\Delta z_k},
   \qquad
   \overline{E}_{y,k}=\frac{\mathcal E_{y,k}}{\Delta z_k}.

Consequently, the field solver itself does not need ``delta_z``.  See
:doc:`field_solver` for the discretization, boundary conditions, and solver
interfaces.

Relativistic transverse kick
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PASS stores normalized transverse momenta
:math:`p_x=P_x/P_0` and :math:`p_y=P_y/P_0`.  For an effective interaction
length :math:`L_{\mathrm{sc}}`, the command applies

.. math::

   \Delta p_x =
   \frac{\operatorname{sgn}(Z)L_{\mathrm{sc}}}
        {\beta c\,B\rho\,\gamma^2}\,\overline{E}_x,
   \qquad
   \Delta p_y =
   \frac{\operatorname{sgn}(Z)L_{\mathrm{sc}}}
        {\beta c\,B\rho\,\gamma^2}\,\overline{E}_y.

The electric field already contains the signed source charge.  The additional
:math:`\operatorname{sgn}(Z)` is the sign of the force on the tracked
particle.  The :math:`1/\gamma^2` term represents the cancellation between
the transverse electric and magnetic self-forces of a co-moving beam.

The kick updates only ``px`` and ``py``, leaving ``x``, ``y``, ``z`` and ``dp``
unchanged. Each SC command may set its own ``Aperture type`` and ``Aperture value``.
This local loss check precedes the field calculation: it updates tags, loss
positions and turns, excluding newly lost particles from both sources and kicks.
Later points preserve existing loss records. Upstream element apertures such as
``Marker`` may also handle losses. Participating PIC particles surviving the
loss check must still lie inside the grid, or an error is raised. Only strict
aperture interiors survive; touching any wall is sufficient for loss.
A missing aperture (or ``default``) becomes the configuration's grid rectangle
for every method. With an explicit aperture or ``off``, analytic tracking is
independent of diagnostic grid extent; sampling does not truncate the source.

Execution Workflow
------------------

The user controls slicing and command ordering. SpaceCharge does not check
Slicer presence, execution history, turn, position or particle-state versions.
It consumes ``slice_id`` and ``slice_table.delta_z`` as supplied and does not
reassign particles based on z. Only data validity is checked: one integer ID
per bunch particle, IDs -1 or in range, and finite strictly positive widths.
No automatic slicing is performed. Existing same-s command priorities are
unchanged.

.. code-block:: text

   point-local particle loss aperture check
       -> supplied slice data + current x, y, updated live tags
       -> PIC deposition/solve/gather OR analytic per-particle evaluation
       -> divide by slice delta_z
       -> shared transverse kick and optional snapshot
