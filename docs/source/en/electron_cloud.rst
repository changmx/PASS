Electron Cloud (ElectronCloud)
==============================

``ElectronCloud`` provides three modes:

* ``frozen`` applies a transverse thin kick from a prescribed stationary
  uniform electron disk. Its field is analytic in free space or calculated
  once from sampled macro electrons with a PIC solver.
* ``build_up`` tracks electrons driven by prescribed beam slices and external
  magnetic fields in a circular chamber, including primary wall emission,
  absorption and simplified true-secondary emission. It applies no beam kick.
* ``coupled`` adds electron-cloud PIC self-fields and a transverse beam kick.
  The actual saved beam-slice particles drive the electrons in a grounded
  circular chamber.

The ``build_up`` model is an externally driven, low-density approximation. It
does not include electron-cloud self-fields, beam feedback or physical
space-charge saturation. ``coupled`` is a first quasistatic, transverse thin-lens
coupling model; it has not been validated for instability thresholds or
equilibrium cloud densities. These modes are selected explicitly.

Configuration and execution
---------------------------

Enable the top-level ``Electron cloud`` block and define named configurations.
Each ``ElectronCloud`` sequence command selects a configuration and supplies
its own ``Interaction length (m)``. The command does not transport particles
over that length. Insert actual transport commands separately. In ``build_up``
this required field is retained for API compatibility but does not change
electron dynamics, because no beam feedback is applied.

In ``frozen``, no ``Slicer`` is required: the prescribed field is independent of longitudinal
position and time. The command leaves ``z_rel``, ``dp``, ``bunch.t0`` and the
reference energy unchanged. In ``build_up``, the command requires a current
``z_rel`` SliceSet and leaves all beam particle coordinates unchanged.
``coupled`` also requires that SliceSet and changes only ``px`` and ``py``.
See :doc:`injection` for the coordinate convention.
The configured density prescribes the frozen cloud or initializes the dynamic
cloud; it is not inferred from beam intensity or bunch grouping.

The following is an input fragment; injection and transport must also be
supplied:

.. code-block:: json

   {
       "Electron cloud": {
           "Enabled": true,
           "Configurations": {
               "round_cloud": {
                   "Mode": "frozen",
                   "Solver": "uniform_round_free_space",
                   "Electron density (1/m^3)": 1e12,
                   "Radius (m)": 0.01,
                   "Center X (m)": 0.0,
                   "Center Y (m)": 0.0
               }
           }
       },
       "Sequence": {
           "cloud_at_ip": {
               "Command": "ElectronCloud",
               "S (m)": 10.0,
               "Configuration": "round_cloud",
               "Interaction length (m)": 1.0,
               "Save fields": true,
               "Save turns": [[0]]
           }
       }
   }

Python input generation uses ``ElectronCloudConfig``,
``ElectronCloudConfiguration`` and ``ElectronCloudItem`` from
``PASS.para.schema.electron_cloud``. Pass the top-level model to
``generate_input(..., electron_cloud=cloud_config)``. Configuration names are
case-sensitive; each command owns its own cloud state and resources.
The same explicit random seed gives the same initial macro-electron
sample for equal source parameters. An omitted seed or JSON ``null`` is
nondeterministic. Booleans and fractional values are not valid integer seeds.

Configuration parameters
------------------------

The top-level block contains ``Enabled`` (boolean, default ``false``) and
``Configurations`` (mapping of names to the following configuration objects).
Disabling the block skips its configurations and all electron-cloud actions.

.. list-table:: Named configuration
   :header-rows: 1
   :widths: 30 18 52

   * - JSON key / Python field
     - Default
     - Meaning
   * - ``Mode`` / ``mode``
     - ``frozen``
     - ``frozen``, externally driven ``build_up``, or ``coupled``.
   * - ``Solver`` / ``solver``
     - ``uniform_round_free_space``
     - Frozen: analytic free space or ``fft_free_space``, ``fd_dirichlet``, ``dst_dirichlet`` PIC. Build-up requires ``round_gaussian_beam``; coupled requires ``fd_dirichlet``.
   * - ``Electron density (1/m^3)`` / ``electron_density``
     - Required
     - Nonnegative physical electron number density inside the disk. In dynamic modes this initializes the cloud only.
   * - ``Radius (m)`` / ``radius``
     - Required
     - Positive uniform-disk radius; for dynamic modes, the initial disk radius. This is not an RMS beam size.
   * - ``Center X (m)``, ``Center Y (m)`` / ``center_x``, ``center_y``
     - ``0``, ``0``
     - Transverse cloud center, in metres.
   * - ``Number of macro electrons`` / ``n_macroparticles``
     - ``10000``
     - Positive integer source sample size for frozen PIC or the initial dynamic cloud. It does not alter physical density.
   * - ``Random seed`` / ``random_seed``
     - ``null``
     - Integer or ``null`` for cloud initialization and emission randomness.
   * - ``Nx``, ``Ny`` / ``nx``, ``ny``
     - ``65``, ``65``
     - Grid point counts, integers at least 3; coupled requires odd counts of at least 5.
   * - ``Grid Width X (m)``, ``Grid Width Y (m)`` / ``grid_width_x``, ``grid_width_y``
     - ``0.1``, ``0.1``
     - Full positive widths of the grid, centered at the origin.
   * - ``Particle Deposition Method`` / ``deposition_method``
     - ``CIC``
     - ``CIC`` or ``TSC`` for PIC deposition and interpolation.
   * - ``Aperture type``, ``Aperture value`` / ``aperture_type``, ``aperture_value``
     - ``default``, ``[]``
     - Frozen field-domain geometry. Dynamic modes require ``default`` or ``off`` and an empty value; their circular wall comes from ``Build up``.
   * - ``Build up`` / ``buildup``
     - ``null``
     - Required nested ``ElectronCloudBuildUpConfiguration`` in both dynamic modes; forbidden in frozen mode.

The grid and deposition fields apply to frozen PIC, frozen field diagnostics
and coupled PIC. ``build_up`` does not use the PIC grid. In coupled mode,
both grid widths must cover the complete chamber diameter and each grid
spacing must be at most half the chamber radius. This is a minimum
resolvability check, not a convergence criterion. The circular Dirichlet
wall is defined by ``Build up.Chamber radius (m)``.

.. list-table:: Sequence command
   :header-rows: 1
   :widths: 30 18 52

   * - JSON key / Python field
     - Default
     - Meaning
   * - ``S (m)`` / ``s``
     - ``0``
     - Thin-kick location in the sequence.
   * - ``Order`` / ``order``
     - ``null``
     - Optional integer order at the same position, following the shared sequence rules.
   * - ``Configuration`` / ``configuration``
     - Required
     - Name in this beam's ``Electron cloud.Configurations``.
   * - ``Interaction length (m)`` / ``interaction_length``
     - Required
     - Nonnegative represented machine length for frozen/coupled beam kicks. Build-up dynamics are independent of this value.
   * - ``Slice set`` / ``slice_set``
     - ``null``
     - Required current ``z_rel`` SliceSet in both dynamic modes; unnecessary for frozen clouds.
   * - ``Is enabled`` / ``is_enabled``
     - ``true``
     - Per-command switch.
   * - ``Save fields`` / ``save_fields``
     - ``false``
     - Save frozen fields, or dynamic particle state/history at selected turns. Coupled output also contains the final cloud fields.
   * - ``Save turns`` / ``save_turns``
     - ``[]``
     - ``[[turn], [start, end, step], ...]`` with inclusive endpoints and zero-based turns. Empty disables saving.

All numeric configuration values must be finite. Unknown electron-cloud fields
are rejected, so unsupported dynamic-model parameters cannot silently enable
physics that is absent.

Frozen-cloud field and momentum conventions
------------------------------------------------------------

Let :math:`n_e\ge0` be the electron density, :math:`e>0` the elementary charge,
:math:`\rho_e=-e n_e`, :math:`a` the cloud radius and
:math:`\boldsymbol r=(x-x_c,y-y_c)`. Gauss's law for an infinitely long,
uniform charged cylinder gives

.. math::

   \boldsymbol E_e(\boldsymbol r)
   =-\frac{e n_e}{2\epsilon_0}\boldsymbol r
   \begin{cases}
       1, & r\le a,\\
       a^2/r^2, & r>a.
   \end{cases}

The field is zero at the center and points toward it. The analytic field
includes the exterior :math:`1/r` radial dependence. A convenient potential
reference is :math:`\phi(a)=0`:

.. math::

   \phi(r)=\begin{cases}
      \dfrac{e n_e}{4\epsilon_0}(r^2-a^2), & r\le a,\\
      \dfrac{e n_e a^2}{2\epsilon_0}\ln(r/a), & r>a.
   \end{cases}

Free-space two-dimensional potentials have an arbitrary additive constant;
this choice does not affect the kick.

PASS stores normalized transverse mechanical momenta
:math:`p_x=P_x/P_0` and :math:`p_y=P_y/P_0`. Under the small-angle,
reference-speed thin-kick approximation, the command applies

.. math::

   \Delta p_x=\frac{q_b L_{\mathrm{int}}}{P_0\beta_0 c}E_{e,x},
   \qquad
   \Delta p_y=\frac{q_b L_{\mathrm{int}}}{P_0\beta_0 c}E_{e,y}.

Here :math:`q_b` is signed and :math:`P_0` is the full particle reference
momentum in SI units. Internally the equivalent factor is
:math:`\operatorname{sign}(q_b)L_{\mathrm{int}}/(\beta_0 c B\rho)` with
positive :math:`B\rho=P_0/|q_b|`, including the ion charge-to-mass
normalization. The resulting kick focuses positively charged particles and
defocuses electrons near the cloud center.

There is no :math:`1/\gamma_0^2` factor: the stationary electron cloud has
no prescribed longitudinal current whose magnetic force cancels its electric
force. That cancellation belongs to the co-moving self-field model in
:doc:`space_charge`. This is not a full six-dimensional electromagnetic
integrator: longitudinal electric fields, cloud magnetic fields, energy work,
and corrections for individual-particle speed are outside the approximation.

PIC normalization and boundary conditions
-----------------------------------------

PIC samples uniformly in disk area, with equal nonnegative electron-number
weights. Using an internal source length :math:`L_s=1\,\mathrm{m}` and
:math:`N_m` samples,

.. math::

   w=\frac{n_e\pi a^2 L_s}{N_m},\qquad Q_m=-ew.

The shared PIC solver deposits these signed charges and returns the integrated
density :math:`\widetilde\rho` in C/m\ :sup:`2`, potential
:math:`\widetilde\phi` in V m and fields :math:`\widetilde E_x,\widetilde E_y`
in V. ``ElectronCloud`` divides them by :math:`L_s` to obtain C/m\ :sup:`3`, V
and V/m. Only the final beam kick multiplies by the separate machine length
:math:`L_{\mathrm{int}}`. Neither length is a longitudinal slice width.

``fft_free_space`` uses open-boundary fields. ``fd_dirichlet`` supports the
conducting geometries of :doc:`field_solver`; ``dst_dirichlet`` requires the
full grid-aligned rectangle, using ``default`` or a matching rectangular
aperture. Dirichlet solvers impose zero wall potential and require a finite
aperture. Their result generally differs physically from the free-space
analytic cylinder. FD and DST can be compared directly only on the same
rectangular domain with identical deposited charge.

For frozen PIC, the complete source disk and deposition support must fit within the
supported field domain; the command rejects invalid geometry instead of
clipping electron charge. The aperture configures the field domain rather
than a particle-loss operation. Use transport-element :doc:`aperture` settings
for beam losses. Free-space analytic evaluation includes points outside the source
disk. PIC evaluation requires particles to lie in its supported interpolation
domain; it does not substitute a zero field for out-of-grid particles.

Nonconvex polygon source containment is checked against the continuous edges,
including concave features smaller than a grid cell. For a nonconvex racetrack
with unequal rectangle and end-cap heights, source containment uses an inner
polygon with 128 segments per curved end. This conservative check can reject a
source extremely close to a curved wall; move the source inward in that case.

Dynamic models and physical time
--------------------------------

Set ``Mode="build_up"``, ``Solver="round_gaussian_beam"`` and provide a
``Build up`` object. For coupled PIC instead select ``Mode="coupled"`` and
``Solver="fd_dirichlet"`` with the same nested object.
The top-level electron density, disk radius and center
describe the initial electron cloud; the entire disk must lie strictly inside
the circular chamber. Initial directions are isotropic in three dimensions,
with the specified monoenergetic kinetic energy. A zero initial density is
allowed; primary sources can subsequently create electrons.

.. code-block:: python

   from PASS.para.schema.electron_cloud import (
       ElectronCloudBuildUpConfiguration, ElectronCloudConfiguration,
   )

   model = ElectronCloudConfiguration(
       mode="build_up", solver="round_gaussian_beam",
       electron_density=1e8, radius=0.01, n_macroparticles=512,
       random_seed=20260926,
       buildup=ElectronCloudBuildUpConfiguration(
           chamber_radius=0.02, beam_sigma=0.002, max_time_step=5e-11,
           primary_electrons_per_particle_per_m=1e-6,
           secondary_yield_max=1.5,
       ),
   )

.. list-table:: ``Build up`` parameters
   :header-rows: 1
   :widths: 35 15 50

   * - JSON key / Python field
     - Default
     - Meaning
   * - ``Chamber radius (m)`` / ``chamber_radius``
     - Required
     - Positive circular wall radius, centered on the beam axis.
   * - ``Beam sigma (m)`` / ``beam_sigma``
     - Required
     - Positive fixed transverse Gaussian sigma for build-up. Still required for schema compatibility in coupled mode, but unused by its fields and time-step controls.
   * - ``Max time step (s)`` / ``max_time_step``
     - Required
     - Positive integration-step ceiling; the pusher may shorten it further.
   * - ``Magnetic field (T)`` / ``magnetic_field``
     - ``[0, 0, 0]``
     - Uniform part ``[B0x, B0y, B0z]`` of the external field in local beam coordinates.
   * - ``Magnetic gradient (T/m)`` / ``magnetic_gradient``
     - ``0``
     - Signed finite normal-quadrupole gradient; Boolean values are rejected.
   * - ``Initial electron energy (eV)`` / ``initial_energy_ev``
     - ``0``
     - Nonnegative initial kinetic energy per electron.
   * - ``Primary electrons per beam particle (1/m)`` / ``primary_electrons_per_particle_per_m``
     - ``0``
     - Nonnegative prescribed primary yield per real beam particle per metre.
   * - ``Primary macro electrons`` / ``primary_macroparticles``
     - ``64``
     - Positive integer samples created at each nonzero primary-emission event.
   * - ``Secondary yield max`` / ``secondary_yield_max``
     - ``0``
     - Nonnegative peak of the uncapped true-secondary yield curve; zero is an absorbing wall.
   * - ``Secondary peak energy (eV)`` / ``secondary_peak_energy_ev``
     - ``300``
     - Positive incident energy at the uncapped yield maximum.
   * - ``Secondary shape`` / ``secondary_shape``
     - ``1.35``
     - Yield-curve shape parameter, strictly greater than one.
   * - ``Emission energy (eV)`` / ``emission_energy_ev``
     - ``2``
     - Positive primary emission energy and nominal secondary emission energy.
   * - ``Max macro electrons`` / ``max_macroparticles``
     - ``100000``
     - Positive integer population limit; exceeding it raises an error instead of discarding charge.
   * - ``Max steps`` / ``max_steps``
     - ``100000``
     - Positive integer integration-step limit per physical interval.
   * - ``Max wall hits per step`` / ``max_wall_hits_per_step``
     - ``32``
     - Positive integer per-particle wall-event limit within one integration step.

Execute the selected ``Slicer`` at the same position and turn before the
cloud. Only continuous ``z_rel`` intervals are supported; complete live-particle
coverage and positive occupied-bin widths are required. Zero-width empty bins
are skipped. The saved Slicer memberships and
intervals are consumed without silently recomputing or rescaling them.
For interval :math:`[z_{\min},z_{\max}]`, the physical passage interval is

.. math::

   t_{\mathrm{start}}=t_0-\frac{z_{\max}}{\beta_0c},\qquad
   t_{\mathrm{end}}=t_0-\frac{z_{\min}}{\beta_0c},\qquad
   \lambda_b=\frac{Z_b e N_{\mathrm{slice}}}{\Delta z}.

The driver sorts intervals from all bunches by these times. Intervals may
touch but may not overlap; repeated turns and backwards physical time are
rejected. Empty bins still advance the cloud with zero beam field. Gaps also
advance electron motion, external-field effects and wall events.
Endpoint roundoff tolerance is bounded by the smaller of eight absolute-time
ULPs and :math:`10^{-12}` times the shorter slice duration. If subtracting
absolute endpoints changes a saved slice duration by more than :math:`10^{-6}`
relative to :math:`\Delta z/(\beta_0c)`, execution is rejected. This prevents
large absolute clocks from silently erasing short bunch pulses.
The first interval start establishes the initial cloud time, including negative
times; no unspecified earlier history is invented. Each command call ends at
the last saved bin's trailing edge, not automatically at the end of the ring
period. The following call advances the intervening gap. Declared empty
bunches contribute their saved empty intervals; missing buckets are not
created implicitly.

Reference time advances through transport commands, not through the executor's
turn counter. Multi-turn input therefore needs an actual closed transport path.
Use ``bunch.t0`` and saved intervals for timing, never nominal ``z_center`` or
``harmonic_id`` as a substitute for physical passage time.

Prescribed drive and electron integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``build_up``, during each slice the transverse source is a fixed, axis-centered round
Gaussian truncated at chamber radius :math:`R`, with configured sigma
:math:`\sigma_b`. The current slice population sets its signed line charge;
tracked transverse beam coordinates do not change its center or size.
For :math:`0<r<R`,

.. math::

   \boldsymbol E_b(\boldsymbol r)=
   \frac{\lambda_b}{2\pi\epsilon_0}
   \frac{1-\exp[-r^2/(2\sigma_b^2)]}
        {1-\exp[-R^2/(2\sigma_b^2)]}
   \frac{\boldsymbol r}{r^2},\qquad
   \boldsymbol B_b=\frac{\beta_0}{c}(-E_y,E_x,0).

The center uses the continuous linear limit. The denominator ensures that
``lambda_b`` is the line charge contained within the chamber. Axial symmetry
makes the grounded circular wall affect the electrostatic potential reference,
not the radial electric field. The beam magnetic field and configured external
field both act on the electrons.

Both dynamic modes support a uniform external field plus an ideal normal
quadrupole, evaluated at the electron's transverse midpoint:

.. math::

   \boldsymbol B_{\mathrm{ext}}(x,y)
   = (B_{0x}+G y,\ B_{0y}+G x,\ B_{0z}),
   \qquad G=\texttt{magnetic\_gradient}.

This field belongs to the local electron-cloud station. It is configured
explicitly and is not inferred from lattice ``Quadrupole`` commands.
A pure dipole uses ``magnetic_gradient=0`` and a transverse uniform field.
For the circular chamber the conservative field bound is
:math:`|\boldsymbol B_0|+|G|R`. The model is longitudinally uniform:
transverse magnetic-mirror trapping can occur, but finite magnet length,
fringe fields and longitudinal electron losses are not represented.

Electrons use two position coordinates and three dimensionless momentum components,
:math:`\boldsymbol u=\boldsymbol P/(m_ec)=\gamma_e\boldsymbol v/c`,
with :math:`\gamma_e=\sqrt{1+|\boldsymbol u|^2}`. A relativistic Boris
momentum update is combined with symmetric half drifts. Each half drift finds
the first circular-wall intersection, emits or absorbs there, then advances
the remaining time. No longitudinal transport between cloud stations is modeled.
The Boris update is described in the
`WarpX particle-pusher documentation <https://warpx.readthedocs.io/en/24.01/theory/pic.html#boris-relativistic-velocity-rotation>`_.

The configured time-step ceiling is additionally restricted using conservative
beam-oscillation, cyclotron and transverse-displacement estimates. These
controls do not replace resolution studies: force splitting near wall impacts
and Boris phase error still depend on the time step. Particle count and
emission statistics need independent convergence checks. The standard
relativistic Boris method also has known limitations for relativistic
:math:`\boldsymbol E\times\boldsymbol B` drift; see
`Higuera and Cary <https://arxiv.org/pdf/1701.05605>`_.

Coupled PIC fields and beam response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``coupled``, the beam source uses the saved slice's live transverse
coordinates and memberships. Each beam macro particle deposits line charge
:math:`q_{\ell,j}=r_b Z_b e/\Delta z`, where :math:`r_b` is the real-to-macro
particle ratio. Its transverse source distribution is held fixed throughout
the saved physical slice interval. It is not replaced by the Gaussian profile;
``beam_sigma`` has no physical effect in this mode.

The grounded circular FD solver separately solves the beam field and cloud
field. Dynamic cloud particles deposit line charges
:math:`q_{\ell,e,j}=-e w_j/L_s` in C/m, with source length :math:`L_s`
initially 1 m. Dividing by cell area gives density in C/m\ :sup:`3`;
Poisson then returns potential in V and field in V/m, with no further
source-length division. Restored snapshots may carry another source length;
scaling electron weights with that length preserves the physical field.
Each electron step performs a half drift with wall events,
rebuilds the cloud field at the midpoint, applies the Boris update using
:math:`\boldsymbol E_b+\boldsymbol E_e` and
:math:`\boldsymbol B_{\mathrm{ext}}+\boldsymbol B_b`, then performs the second
half drift. Empty-beam gaps still evolve cloud self-fields.
There is no cloud magnetic field or longitudinal electric field.

For a witness at the slice's fixed transverse position, midpoint quadrature
accumulates the cloud field over the slice duration :math:`\Delta t_s`:

.. math::

   \overline{\boldsymbol E}_{e,j}
     =\frac{1}{\Delta t_s}\sum_k
       \boldsymbol E_e(\boldsymbol r_j,t_{k+1/2})\Delta t_k,\qquad
   \Delta\boldsymbol p_{\perp,j}
     =\frac{\operatorname{sign}(Z_b)L_{\mathrm{int}}}
            {\beta_0 c B\rho}\overline{\boldsymbol E}_{e,j}.

Only the cloud field contributes to the beam kick. There is no beam self-kick
or :math:`1/\gamma_0^2` reduction. The operation changes only ``px`` and ``py``;
``x``, ``y``, ``z_rel``, ``dp``, reference energy and ``bunch.t0`` stay fixed.
Beam source coordinates are not advanced during a slice. The interaction
length scales this beam response, not electron counts or their physical clock.

Near the circular wall, deposition and interpolation renormalize each stencil
over its supported interior nodes to conserve deposited charge. This treatment
has first-order wall error and is not an exact Hamiltonian or energy-conserving
particle-field discretization. Time steps are limited by ``max_time_step``,
external/beam cyclotron estimates, cloud plasma and cell-acceleration frequency
estimates (:math:`\omega\Delta t\le0.2`), and electron displacement
(:math:`v_\perp\Delta t\le0.2h`, with the smaller grid spacing :math:`h`).
These controls are safeguards, not a convergence proof. Independently vary
grid size, macro-electron/primary sample counts, random seeds, slice widths
and time steps before drawing physical conclusions.

Primary and secondary emission
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At a slice's leading edge, a prescribed primary source creates
:math:`N_{\mathrm{primary}}=Y_p N_{\mathrm{slice}}L_s` electrons, represented
by equal-weight macro electrons sampled uniformly around the wall and directed
inward. Here :math:`L_s` is the source length (initially 1 m) and :math:`Y_p`
has units 1/m. This phenomenological source does not calculate gas ionization,
synchrotron-photon transport or a material photoelectric spectrum.
Primary production is concentrated at the slice front rather than distributed
continuously across its duration. Reducing only the electron time step does
not remove that source-time approximation; also refine the Slicer intervals.

For incident kinetic energy :math:`E_i`, let :math:`x=E_i/E_{\max}` and
:math:`s>1`. The uncapped true-secondary curve and implemented limits are

.. math::

   \delta(E_i)=\delta_{\max}\frac{s x}{s-1+x^s},\qquad
   \delta_{\mathrm{eff}}=\min\!\left(\delta(E_i),\frac{E_i}{E_{\mathrm{emit}}}\right),
   \qquad E_o=\min(E_{\mathrm{emit}},E_i).

Each incident macro electron creates at most one macro secondary, with
weight :math:`w_o=w_i\delta_{\mathrm{eff}}` and per-electron energy :math:`E_o`.
Zero weight means absorption. This enforces both the per-electron and weighted
aggregate incident-energy bound, conservatively suppressing low-energy emission.
The curve uses the true-secondary shape of
`Furman and Pivi, Eqs. 31-32 <https://www.classe.cornell.edu/~critten/cesrta/ecloud/doc/furmanpivi.pdf>`_.
It is not their full probabilistic model: elastic/reflected and rediffused
components, angle-dependent material yield and a joint secondary energy
spectrum are absent. Macro-particle multiplication is represented by weights,
not integer branching.

Primary and secondary directions obey a three-dimensional cosine distribution
about the inward normal: :math:`\mu=\cos\theta=\sqrt U` and
:math:`\phi=2\pi V`, for independent uniform random values :math:`U,V`.
All three momentum components are retained although only x and y are tracked.
Initial cloud directions instead use an isotropic full sphere.

Output and state
----------------

Frozen selected-turn HDF5 diagnostics are written below the current PASS run directory
as ``electron_cloud/<run_id>/beam<id>_<command>/turn_<turn>_call_<call>.h5``.
The run identifier contains a fresh token; call numbers distinguish repeated
executions at the same turn. The command's ``saved_fields`` lists written paths.
Required field output completes before staged beam kicks are committed. A failed
write leaves beam momenta unchanged; a retry reserves a new diagnostic filename.
Files have format marker ``PASS-electron-cloud-fields-1`` and contain:

.. list-table:: Field snapshot datasets
   :header-rows: 1
   :widths: 45 20 35

   * - Dataset
     - Unit
     - Meaning
   * - ``grid/x``, ``grid/y``
     - m
     - One-dimensional grid axes.
   * - ``fields/charge_density``
     - C/m\ :sup:`3`
     - Signed physical volume charge density.
   * - ``fields/electron_density``
     - 1/m\ :sup:`3`
     - Physical electron number density.
   * - ``fields/potential``
     - V
     - Electrostatic potential.
   * - ``fields/ex``, ``fields/ey``
     - V/m
     - Transverse electric fields.
   * - ``source/x``, ``source/y``, ``source/weight``
     - m, m, 1
     - PIC source coordinates and electron-number weights; absent for analytic clouds.

Two-dimensional grid arrays use ``(y, x)`` ordering. Dataset attributes specify
units. File ``metadata_json`` records the command, beam, turn, position,
interaction length, configuration and kick diagnostics; PIC source metadata
includes the represented source length and random state. After execution,
``last_diagnostics`` gives live-particle counts and maximum fields/kicks, with
per-bunch records. The analytic potential uses the reference stated above.
PIC sampling noise and conducting-wall image fields should be distinguished
from force-normalization errors when interpreting these plots.

Both dynamic modes use the same output-directory and selection rules and save
``source/x,y,ux,uy,uz,weight`` arrays and source metadata. ``build_up`` saves no
cloud self-field maps. Coupled output additionally stores the actual final
cloud ``grid`` and ``fields`` datasets with the SI units above, plus history
and maximum-kick diagnostics. The momentum components
``ux, uy, uz`` are dimensionless. Source metadata
contains ``source_length``, physical ``time``, ``last_turn``, RNG state and
cumulative number/energy counters. ``history_json`` records each gap or slice,
including times, beam population, signed line charge, primary input, electron
populations before/after, incident/emitted populations and step count.
Each record satisfies

.. math::

   N_{\mathrm{after}}=N_{\mathrm{before}}+N_{\mathrm{primary}}
                      -N_{\mathrm{incident}}+N_{\mathrm{emitted}}.

Accumulated wall energy is incident minus emitted kinetic energy, in eV for
all represented electrons. Multiply by :math:`e` for joules. Counts and wall
energies refer to the saved source length; scale by :math:`L/L_s` for a
station representing machine length :math:`L`. This is not a self-consistent
machine heat-load prediction. The configured interaction length does not
rescale the underlying electron ensemble.

The cumulative energy ledger stores ``initial_energy_ev``,
``primary_energy_ev`` and signed ``field_work_ev`` (the kinetic-energy change
at Boris force updates). With :math:`K=\sum w(\gamma_e-1)m_ec^2` in eV,
it obeys
:math:`K_{\mathrm{final}}=K_{\mathrm{initial}}+K_{\mathrm{primary}}+W_{\mathrm{field}}-E_{\mathrm{wall}}`.
This checks numerical bookkeeping; it does not measure the integrator's
physical trajectory error.
In coupled mode it also does not establish conservation of total beam,
electron and field energy under the quasistatic thin-lens approximation.
The entire dynamic command stages electron state, RNG state and beam kicks;
all are committed only after successful evolution and requested output.
An evolution or output failure leaves them unchanged.

The command provides ``state_dict()`` / ``load_state_dict()`` and
``save_state(path)`` / ``load_state(path)`` for its own source,
configuration identity and random state. This preserves a nondeterministic
PIC realization, or a dynamic population with time and momenta, for reuse.
These APIs are cloud snapshots, not a complete
simulation restart: matching beam particles, beam reference state, turn and
any other collective-effect state must be handled separately. Frozen mode
has no evolving cloud clock; both dynamic modes restore their saved physical
clock. Coupled snapshots use the same dynamic-state format, validate mode and
configuration identity, and rebuild field resources rather than saving caches.

During tracking, dynamic electron arrays remain on the selected CPU or GPU
backend. Staged states own detached particle data; coupled mode reuses the fixed
grid, field factorization and numerical workspace through serial calls. Reading a
public cloud state, writing source/field snapshots or capturing a checkpoint materializes
a validated host snapshot; on GPU this adds a device-to-host transfer.
Input fields and checkpoint formats are unchanged. Scalar diagnostics and
emission sampling can still synchronize the host and device.
When comparing CPU/GPU performance, warm compilation and solver setup first,
measure synchronized tracking separately from setup and snapshot output,
and include both wall-free and wall-emission workloads. Small clouds need
not run faster on GPU.

Coupled GPU execution combines fixed-order deposition with deterministic
cuDSS solves. Highly occupied cells use a fixed reduction tree, so the last
bits may differ from earlier serial summation. Repeatability checks apply
to the same code, GPU architecture, SM count and software stack, not across
devices. See :doc:`field_solver`.

For simulations already using the joint two-beam checkpoint interface,
``capture_collision_state`` / ``restore_collision_state`` also capture and
restore enabled electron-cloud realizations. They validate the cloud
identity and rebuild its field resources before committing the joint restore.
The existing common-turn and output requirements still apply; see
:doc:`beam_beam`. This integration does not introduce a separate single-beam
machine-restart interface.

Runnable example
----------------

``example/07_electron_cloud`` contains an English walkthrough and six serial
one-kick cases: zero density, analytic density and doubled density, free-space
PIC, rectangular FD and rectangular DST. Explicit probe particles sample the
interior and exterior fields. Analysis reads before/after distribution
snapshots, compares with Gauss's law, checks density proportionality and
unchanged coordinates/reference time, and compares FD/DST under identical
boundary conditions. It writes JSON, CSV and a scientific PNG figure under
the Git-ignored ``tests/codex/electron_cloud/example_output`` directory.

``example/08_electron_cloud_buildup`` adds four serial bunch-train cases:
initial-cloud absorption, primary production with absorption, primary plus
true-secondary emission, and a halved time-step comparison. A real one-turn
Drift advances the reference clocks while the prescribed driver distribution
stays fixed. Its analysis verifies number/energy bookkeeping, timeline
continuity and unchanged beam coordinates, and writes JSON, CSV and population
histories under ``tests/codex/electron_cloud/buildup_example_output``.
Changes under time-step halving are reported as a resolution comparison,
not as proof of physical saturation or statistical convergence.
