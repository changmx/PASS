Space-Charge Effect (SpaceCharge)
=================================

Introduction
------------

The ``SpaceCharge`` command applies a transverse 2.5-D thin kick using
PIC, frozen analytic profiles, or quasi-frozen analytic profiles. It consumes
bunch-local slice data supplied by the user. All methods include electric and
magnetic self-force cancellation through :math:`1/\gamma^2` and do not
calculate a longitudinal force.

- **Code location**: ``PASS/commands/space_charge.py``
- **Class name**: ``SpaceCharge``, registered name ``"SpaceCharge"``
- **Schema location**: ``PASS/para/schema/space_charge.py``
- **Execution backend**: CPU only
- **Main features**:

  - named, reusable space-charge configurations at the top level of each beam
    input;
  - independent longitudinal slicing, mesh, deposition, and field-solver
    choices;
  - CIC or TSC charge deposition with the matching field-gather method;
  - conducting or free-space boundary models supplied by three field solvers;
  - optional HDF5 snapshots of charge density, potential, and field.

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
are twice these RMS sizes. The first reported principal size is the larger.
The counterclockwise major-axis angle is reported modulo pi in
:math:`[-\pi/2,\pi/2)`. Fields are evaluated after translation/rotation and
rotated back to the original transverse axes.

For a round profile the moment-matched radius rule is

.. math::

   \sigma^2=\frac{\operatorname{Var}(x)+\operatorname{Var}(y)}{2},
   \qquad R=2\sigma.

``gaussian_round_free_space`` uses sigma; ``uniform_round_free_space`` uses R.
This preserves the centroid-relative radial second moment. It is a deliberate
round approximation for non-round populations, not an exact reconstruction of
their field. Choosing a uniform profile does not turn the particles into a KV
distribution. No automatic profile switching occurs.

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

Only configurations referenced by a ``SpaceCharge`` sequence entry are built.
Within one beam, commands that reference the same configuration name share
one grid. Dirichlet solver resources are cached by configuration and resolved
command aperture: equal walls reuse the factorization, different walls require
separate resources. FFT kernels are shared regardless of the loss aperture.
Different configuration names own independent resources even when their
values are identical. Configurations no longer contain ``Chamber``.

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
     - Required positive radius for frozen uniform disk.
   * - ``a / b``
     - ``Semi-axis A (m) / Semi-axis B (m)``
     - float or null
     - ``null``
     - Required positive semi-axes for frozen uniform ellipse.


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
- Legacy root keys ``Is space charge`` and
  ``Space-charge simulation parameters`` are rejected.  Mesh and solver
  fields are not accepted inline in a ``SpaceCharge`` sequence command.
- GPU execution with an enabled module and either nonzero length or an active loss aperture raises an error;
  use ``"Backend (gpu/cpu)": "cpu"``.

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

Validation Layout
-----------------

The integration-test names state both the source and boundary model.  The
``*_free_space_fft`` workflows cover round/elliptic Gaussian and KV sources
with the open-boundary Green-function solver.  The corresponding
``*_rectangular_fd_dst`` workflows regenerate the same four fixed-seed
sources and solve the same deposited charge with FD and DST on a 257 by 257
grounded rectangular grid.  Their analysis directories contain a three-way
``free_space_fft_vs_rectangular_fd_dst_field.png`` comparison, a CSV table,
and ``fd_dst_relative_error_same_plot.png`` with both directions of the
pointwise FD/DST relative difference.  Differences from FFT include the
physical boundary-condition change and are not an FD/DST discretization error.

FD-only aperture tests are named by their physical boundary.  They include a
round KV beam inside a larger circular conductor, a KV beam filling an
elliptic conductor, and Gaussian plus KV-uniform projected sources across all
apertures supported by ``PASS.utils.aperture`` (including ellipse, racetrack,
octagon, and user polygon).  DST is not applied to those curved or irregular
domains because ``dst_dirichlet`` diagonalizes only the full rectangular
Dirichlet grid.

Running the Validation Suite
----------------------------

All space-charge integration cases run automatically, including the ten full
generated-input workflows. From the repository root, run:

.. code-block:: console

   python -m tests.integration.space_charge
   python -m tests.integration.space_charge regression

The first command selects all 25 integration tests. The second explicitly runs
the related unit and local Codex regression files, including the restored
slice-isolation and slice-width/interaction-length scaling checks. The local
Codex files must be present; they are not part of default pytest discovery.
Repository-wide pytest discovers ``tests/unit`` and ``tests/integration``.

Use ``analytic``, ``fft``, ``rectangle``, ``aperture``, ``checks``, or ``workflows`` instead of
the default ``all`` to select a category. Repeat ``--case <case_name>`` to
select individual full workflows. ``--collect-only`` lists the selected pytest
items. No complex integration case is skipped by default.

Execution is serial. Rectangle comparisons automatically prepare and reuse the
matching FFT workflow within the same batch. Each new integration run creates
a unique directory under ``tests/codex/space_charge_runs``; its path is printed
in the terminal summary. ``--output-dir`` selects a new or empty batch directory.
Existing results are never deleted by the runner. ``--mode ana --output-dir
<existing_batch>`` reruns analysis of saved snapshots and updates analysis
products without tracking. Non-test modes operate only on full workflows;
they do not run numerical-only checks.

The normal ``python -m pytest tests/integration/space_charge`` command executes
the same integration cases. Original per-module ``sim``, ``ana``, ``simana``
and ``--run-dir`` interfaces remain available. See
``tests/integration/space_charge/README.md`` for the group table, single-case
commands, output layout, and the distinction between batch and per-case paths.


Analytic Diagnostics and Validation
------------------------------------------

HDF5 ``schema_version`` is ``3``. Attributes include ``method``, the full
``solver`` name and ``grid_role`` (``tracking`` or ``diagnostic``). Analytic
snapshots additionally store ``macro_count``, ``center_x``, ``center_y``,
``size_x``, ``size_y`` and ``angle`` per slice. ``size_convention`` is
``principal_rms`` for Gaussian profiles or ``uniform_semi_axes`` for uniform
profiles. ``parameter_source`` is ``configuration`` or
``current_slice_population_moments``. Empty-slice parameters are NaN; their
charge and sampled fields/density are zero. A frozen ellipse retains the user's
axis order, while quasi-frozen ellipses report the major axis first.

Analytic ``Save potential`` is not implemented and is rejected even if no save
turns are selected; ``potential_gauge`` is ``not_computed``. The density grid is
an analytic sample, not deposited charge: its finite-grid quadrature need not
equal ``slice_charge``. Diagnostics never alter the source or particle kick.

Run ``python -m tests.integration.space_charge analytic`` for four generated-input
comparisons (180,000 particles each, three slices, all three methods) and two
repeated-kick parameter-evolution tests. Independent Gaussian/uniform field
integrals verify the actual momentum increments with relative L2 tolerance
``2e-8``. PIC comparisons use ``0.06`` relative L2 in a resolved central region;
finite-particle noise and discretization are included. Charge comparison uses
a particle-count-dependent floating-summation bound. The evolution tests use
prescribed affine transport; they do not establish long-term ring stability.

Outputs include input JSON, HDF5, particle-kick NPZ, CSV/JSON measurements,
``pic_frozen_quasi_frozen_field_comparison.png``,
``analytic_kick_vs_independent_integral.png`` and
``frozen_vs_quasi_frozen_parameter_evolution.png``. For generated analytic
cases, call ``analyse(run_dir)`` from
``tests.integration.space_charge.test_analytic_free_space_tracking`` to update
plots from saved artifacts without tracking. This analytic test group currently
uses ``--mode test``; the older batch non-test workflow selection is unchanged.

The old resource keys ``Field solver``, ``Aperture`` and ``Chamber`` and old
public solver values are rejected. Use ``Method`` and ``Solver`` in the
configuration and ``Aperture type/value`` in each command. Fixed transverse
parameters are accepted only by ``frozen`` and must match the selected profile;
deposition settings are PIC-only.
