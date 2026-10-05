Beam-Beam Interaction (BeamBeam)
================================

``BeamBeam`` is an explicit rendezvous command on both beam sequences. It
implements ideal synchronous, same-turn collisions with fixed bunch pairing.
The waiting state coordinates execution; it does not advance or correct either
bunch's physical reference time. Reference encounters must agree within floating
point tolerance. Different IPs must occur in the same order on both sequences.

Configuration and ordering
--------------------------

Use the top-level ``Beam beam`` block. Configuration IDs are case sensitive and
must be defined exactly once across the two inputs. Either file can supply the
shared ``Enabled`` switch; explicitly supplied switches must agree. If neither
file supplies it, collisions are disabled. ``Is beam-beam`` is obsolete: a true
value is rejected, and the input generator no longer emits it.

.. code-block:: json

   {
     "Beam beam": {
       "Enabled": true,
       "Configurations": {
         "IP1": {
           "Beams": [0, 1],
           "Mode": "weak-strong",
           "Weak beam": 0,
           "Sources": {
             "0": {"Slice set": "bb_ip1"},
             "1": {
               "Slice set": "bb_ip1",
               "Method": "frozen",
               "Solver": "gaussian_round_free_space",
               "Frozen parameters": {"Sigma (m)": 0.001}
             }
           }
         }
       }
     }
   }

Both sequences need a Slicer and BeamBeam at the IP, for example the following
entries at :math:`s=0`. Other commands at that position, including Injection,
the arriving Twiss map and monitors, must also have distinct integer ``Order``
values. With no explicit Order at a position, the existing command priorities
remain in use. The GUI, input generator, validator and runtime share this rule.

.. code-block:: json

   {
     "slice_ip1": {
       "Command": "Slicer", "S (m)": 0, "Order": 400,
       "Purpose": "beam_beam", "Configuration": "IP1",
       "Slice set": "bb_ip1", "Coordinate": "z_rel",
       "Slice model": "equal_particle", "Number of slices": 8,
       "Z range mode": "auto"
     },
     "ip1": {
       "Command": "BeamBeam", "S (m)": 0, "Order": 500,
       "Configuration": "IP1"
     }
   }

The latest explicit Slicer defines membership. BeamBeam neither reslices nor
reorders the particle pool. Particle longitudinal coordinates and live membership
must remain unchanged between the collision Slicer and BeamBeam; rerun Slicer
after any change. BeamBeam rejects a nonempty bunch whose per-slice live counts
have changed, preventing use of stale head/tail data.
An explicit range must cover every live particle;
out-of-range particles cause an error instead of being clipped. A repeated IP
visit requires another explicit Slicer execution. Saved slice snapshots retain
their original geometry; source moments used during collisions are separate.

GUI configuration
-----------------

The GUI's **Physics Effects → BeamBeam** module contains the shared configuration
editor and sequence editors for the collision Slicer, BeamBeam, CrossingAngle,
CrabCavity and FloatWaister. The configuration editor supports named IPs with
separate Collision, beam0 Source, beam1 Source and Luminosity tabs. Source
controls show only the parameters applicable to the selected source method.
PIC uses the actual particle head and tail recorded by the collision Slicer.
Set the slice count and slicing mode there; no separate propagation step is used.

Define each shared IP in one beam input only; the other beam's sequence can
reference that configuration name. The shared Enabled control has an unset
state for an input that does not declare the switch. Configure both sequences
using the ordering and slicing requirements above, then validate the two inputs
together. A GUI form does not remove the need for a matching second sequence.

Interface Parameters
--------------------

Top-level ``Beam beam`` block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python attribute
     - JSON key
     - Unit
     - Default
     - Description
   * - ``enabled``
     - ``Enabled``
     - —
     - unset
     - Optional shared switch; explicit values in the two files must agree. No supplied switch means disabled.
   * - ``configurations``
     - ``Configurations``
     - —
     - ``{}``
     - Mapping of case-sensitive names to IP configurations, each defined exactly once across the inputs.

Named IP configuration
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python attribute
     - JSON key
     - Unit
     - Default
     - Description
   * - ``beams``
     - ``Beams``
     - —
     - ``[0, 1]``
     - Exactly these two beam IDs; their backend and particle precision must agree.
   * - ``mode``
     - ``Mode``
     - —
     - ``strong-strong``
     - ``strong-strong`` kicks both beams; ``weak-strong`` kicks only ``weak_beam``; ``weak-weak`` applies no beam-beam kick.
   * - ``weak_beam``
     - ``Weak beam``
     - —
     - ``null``
     - Required 0 or 1 for weak-strong; forbidden in other modes.
   * - ``bunch_pairs``
     - ``Bunch pairs``
     - —
     - ``null``
     - Optional complete bijection ``[[id0, id1], ...]``; omission pairs equal stable bunch IDs. All IPs share the pairing.
   * - ``full_crossing_angle``
     - ``Full crossing angle (rad)``
     - rad
     - ``0.0``
     - Full deviation from antiparallel design trajectories; strictly between −π and π.
   * - ``crossing_plane``
     - ``Crossing plane (rad)``
     - rad
     - ``0.0``
     - Crossing-plane orientation in the common geometry.
   * - ``interaction_map``
     - ``Interaction map``
     - —
     - ``synchro_beam_6d``
     - The supported six-dimensional synchro-beam map.
   * - ``potential_reference_length``
     - ``Potential reference length (m)``
     - m
     - ``1.0``
     - Positive fixed logarithmic-potential reference length.
   * - ``sources``
     - ``Sources``
     - —
     - required
     - Exactly keys ``"0"`` and ``"1"``. Both require a slice reference; each emitting side requires Method and Solver.
   * - ``luminosity``
     - ``Luminosity``
     - —
     - ``null``
     - Optional shared diagnostic block; omission disables output and is omitted from generated JSON.

Without luminosity, ``weak-weak`` also skips collision slicing, coordinate
charts and field allocation. CrabCavity and FloatWaister remain independent
elements. Fixed pairing is incompatible with ReorganizeBunch.

BeamBeam sequence command
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python attribute
     - JSON key
     - Unit
     - Default
     - Description
   * - ``command``
     - ``Command``
     - —
     - ``BeamBeam``
     - Command type.
   * - ``s``
     - ``S (m)``
     - m
     - required
     - IP position on this beam's sequence.
   * - ``order``
     - ``Order``
     - —
     - ``null``
     - Optional integer order under the shared command-ordering rules.
   * - ``configuration``
     - ``Configuration``
     - —
     - required
     - Exact name of the shared IP configuration.

Per-beam source configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python attribute
     - JSON key
     - Unit
     - Default
     - Description
   * - ``slice_set``
     - ``Slice set``
     - —
     - required
     - Nonempty name of the source beam's explicit collision Slicer result.
   * - ``method``
     - ``Method``
     - —
     - ``null``
     - ``pic``, ``frozen`` or ``quasi-frozen``. A target-only entry may contain only Slice set.
   * - ``solver``
     - ``Solver``
     - —
     - ``null``
     - Required with Method; use a supported combination in the numerical-model table below.
   * - ``statistics_precision``
     - ``Statistics precision``
     - —
     - ``null``
     - ``float32`` or ``float64`` source-moment accumulation; null follows particle precision.
   * - ``nx / ny``
     - ``Nx / Ny``
     - —
     - ``128``
     - PIC only: integer mesh nodes per axis, at least 3.
   * - ``grid_half_width_x / grid_half_width_y``
     - ``Grid Half Width X (m) / Grid Half Width Y (m)``
     - m
     - ``null``
     - PIC requires both positive half widths of the fixed, centered mesh.
   * - ``deposition_method``
     - ``Particle Deposition Method``
     - —
     - ``TSC``
     - PIC only: ``CIC`` or ``TSC``; shares the SpaceCharge deposition primitives.
   * - ``frozen_parameters``
     - ``Frozen parameters``
     - —
     - ``null``
     - Required for frozen; one prescribed profile using the dimensions below.
   * - ``slice_parameters``
     - ``Slice parameters``
     - —
     - ``{}``
     - Frozen only: nonnegative slice indices mapped to complete per-slice Frozen parameters.
   * - ``frozen_optics_reference``
     - ``Frozen optics reference``
     - —
     - ``null``
     - Frozen only: Twiss entry at this IP on the source's own beam; enables the prescribed angular moments for hourglass.
   * - ``source_center_slopes``
     - ``Source center slopes``
     - —
     - ``[0.0, 0.0]``
     - Frozen only: two signed transverse center slopes.

Frozen parameters and Slice parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python attribute
     - JSON key
     - Unit
     - Default
     - Description
   * - ``center_x / center_y``
     - ``Center X (m) / Center Y (m)``
     - m
     - ``0.0``
     - Prescribed transverse center.
   * - ``angle``
     - ``Angle (rad)``
     - rad
     - ``0.0``
     - Ellipse orientation; must remain zero for a round profile.
   * - ``sigma``
     - ``Sigma (m)``
     - m
     - ``null``
     - Positive RMS size for a round Gaussian.
   * - ``sigma_x / sigma_y``
     - ``Sigma X (m) / Sigma Y (m)``
     - m
     - ``null``
     - Positive principal RMS sizes for an elliptic Gaussian.
   * - ``radius``
     - ``Radius (m)``
     - m
     - ``null``
     - Positive support radius for a uniform/parabolic disk.
   * - ``a / b``
     - ``Semi-axis A (m) / Semi-axis B (m)``
     - m
     - ``null``
     - Positive support semi-axes for a uniform/parabolic ellipse.
   * - ``sigma_delta``
     - ``Sigma delta``
     - —
     - ``0.0``
     - Nonnegative RMS momentum spread for the dispersion contribution; requires Frozen optics reference when positive.

Supply exactly the dimensions required by the chosen profile; arbitrary
covariance input is not supported.

Physics and Numerical Model
---------------------------

.. list-table:: Supported source representations
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Solver
     - Source representation
   * - ``pic``
     - ``fft_free_space``
     - Current particles deposited with CIC or TSC on a fixed transverse grid.
   * - ``frozen``
     - ``gaussian``, ``uniform`` or ``parabolic``, each with ``round_free_space`` or ``ellipse_free_space`` suffix
     - Prescribed spatial profile; optional IP optics supplies angular moments.
   * - ``quasi-frozen``
     - The same six analytic solvers
     - Profile reconstructed from the current source slice moments used by each slice-pair event.

PIC sources and distance interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pic`` uses ``fft_free_space``, ``Nx``, ``Ny``, and positive
``Grid Half Width X (m)`` / ``Grid Half Width Y (m)``. Deposition reuses the
SpaceCharge implementation: ``Particle Deposition Method`` is ``CIC`` or
``TSC`` (default). Complete source and target stencils must fit in the fixed
grid; this numerical coverage error does not mark physical particle loss.
The fixed grid is a numerical free-space domain, not a grounded conducting wall.

The longitudinal force needs the potential as well as transverse fields.
For every slice pair and source direction, PIC uses two propagation positions
determined by the target slice's actual live-particle head and tail. The Slicer
stores these extrema as ``z_particle_min`` and ``z_particle_max``, separately
from the existing interval boundaries ``z_min`` and ``z_max``. For a source
slice with particle mean :math:`\bar z_s`,

.. math::

   S_0=\frac{z_{\mathrm{particle\ min}}-\bar z_s}{2},\qquad
   S_1=\frac{z_{\mathrm{particle\ max}}-\bar z_s}{2},\qquad
   u=\frac{S-S_0}{S_1-S_0},\qquad 0\le u\le1.

The source center is its particle mean, not the midpoint of its head and tail.
Collision-frame slicing supplies all three quantities in the same coordinate
frame. Equal-length and equal-particle slicing both use their saved memberships;
BeamBeam neither repartitions the slices nor introduces a second distance grid.
Let :math:`\Phi_0` be the potential of the density at :math:`S_0`, and
:math:`\Delta\Phi` the potential obtained by solving the difference between
the densities at :math:`S_1` and :math:`S_0`. With the same transverse
CIC/TSC reconstruction applied to both planes,

.. math::

   \Phi(x,y,S)=\Phi_0(x,y)+u\,\Delta\Phi(x,y),\qquad
   \left.\frac{\partial\Phi}{\partial S}\right|_{x,y}
       =\frac{\Delta\Phi(x,y)}{S_1-S_0},
   \qquad E_x=-\partial_x\Phi,\quad E_y=-\partial_y\Phi.

Transverse kicks and the longitudinal source term differentiate this same
interpolated potential, using one fixed logarithmic-potential reference.
The complete longitudinal kick of the six-dimensional map below remains active.

There is no separate propagation-step input. Remove the obsolete
``Propagation step (m)`` / ``propagation_step`` field from older inputs; unknown
fields are rejected. Control longitudinal resolution with the Slicer's slice
count and mode. The actual head-to-tail width can differ between equal-particle
slices, particularly in the tails; check slice-count convergence for the chosen
mode. Analytic sources continue to evaluate their propagation directly.

A PIC source entry is shown below.

.. code-block:: json

   {
     "Slice set": "bb_ip1",
     "Method": "pic",
     "Solver": "fft_free_space",
     "Nx": 128, "Ny": 128,
     "Grid Half Width X (m)": 0.01,
     "Grid Half Width Y (m)": 0.01
   }

Grid coverage must include the complete source deposition stencil at both
head/tail propagation positions and every target gather stencil at its actual
collision position. Enlarge the transverse domain when coverage fails; the
numerical boundary is not a particle-loss aperture.

Every nonempty slice pair normally solves two potential planes per source
direction: the first-endpoint density and the head-to-tail density difference.
With :math:`N_A,N_B` nonempty slices, this is normally :math:`2N_AN_B` planes
for weak--strong and :math:`4N_AN_B` for strong--strong. These are Poisson
solution planes, not FFT or kernel launch counts. Empty slices are skipped.
For a zero-width target slice, the source density and its propagation derivative
are evaluated at the same :math:`S`; the longitudinal derivative is not set to
zero and no finite auxiliary distance is introduced. Solving density differences
before the FFT reduces cancellation in the longitudinal derivative. Transverse
fields differentiate the gathered potential without computing unused grid-field
arrays. The CIC gather directly expands the four-node shape derivatives and
reuses the same potential samples for all kick components.

The propagation coordinates and density sums are evaluated in float64 from the
tracked particle values and the head/tail positions, including during float32
tracking. The head-to-tail density difference is formed from stable paired
particle contributions, avoiding subtraction of two nearly equal completed
density grids. Density and density difference are converted to the configured
precision for the FFT; particle and returned field arrays retain that precision.

All particles of the target slice use the same pair of source planes; no
distance-interval sorting or interval-by-interval host schedule is needed.
GPU potential workspace is reused only after its gathers have been enqueued on
the same stream; the independent source snapshot remains valid for the event.
Each GPU FFT solver retains plans and workspaces for at most two recently used
batch sizes, avoiding repeated plan creation when one- and two-plane solves
are requested. This cache belongs to the solver's device and stream and is released
when its resources close.

Within a slice pair, the propagation potential is linear between the actual
head and tail; its :math:`S` derivative is constant at fixed transverse position.
Transverse deposition/gather has its own mesh regularity. For smooth
source-potential data and :math:`\Delta S=S_1-S_0`, linear interpolation generally
gives :math:`O(\Delta S^2)` potential error and :math:`O(\Delta S)` pointwise
:math:`S`-derivative error. Increase the slice count and inspect longitudinal
and transverse kicks separately. Vary transverse mesh and particle count
independently; slice-count convergence alone does not establish transverse
field convergence.

CIC differentiates a bilinear transverse potential, so its force can jump at
mesh lines. Coordinates on or very near a line can select different one-sided
gradients in float32 and float64; pointwise agreement is not guaranteed there.
Differencing the grid potential first and then interpolating the grid field is
a different discretization: it can have better pointwise transverse accuracy,
but does not generally equal the gradient of the same CIC interpolated potential.
Compare accuracy as well as timing rather than treating the routes as identical.
TSC reconstructs a continuously differentiable transverse potential and hence
a continuous gradient for a fixed grid potential with full stencil coverage.
Use TSC for precision comparisons, and check transverse mesh and macro-particle
resolution together; higher arithmetic precision alone does not establish PIC
convergence.

Comparison with Athena's head/tail PIC path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This comparison refers to the head/tail path in Athena's ``src/simulator.cu``,
``transfer_headAndTail`` and ``cal_beamKick_interpolation`` in
``src/collision.cu``, and ``calElectricField`` in ``src/pic.cu``.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Item
     - Athena head/tail path
     - PASS
   * - Distance samples
     - For each slice pair and source direction, two collision locations are formed from the source slice center and the target slice head/tail; their separation is half the target slice width.
     - The same head/tail geometry, using the actual particle extrema saved by Slicer and the source particle mean. Two potential planes per source direction; no independent distance step.
   * - Interpolated quantity
     - Centered differences of grid potential give :math:`E_x,E_y`; these fields are interpolated transversely and between the head/tail samples.
     - The reconstructed potential is interpolated along :math:`S`; transverse fields and the source-distance derivative come from that same potential.
   * - Collision kick
     - The inspected head-on interpolated-kick kernel updates transverse :math:`p_x,p_y`. The inverse crossing-angle transformation also produces a longitudinal beam-beam change in the laboratory frame.
     - The complete six-dimensional kick includes the longitudinal source derivative and kinematic terms.

In Athena's horizontal crossing convention with half angle :math:`\theta`,
the inverse transformation gives
:math:`\Delta p_{z,\mathrm{lab}}=\sin\theta\,\Delta p_x^*` for a head-on transverse
kick. Thus Athena does include crossing-induced longitudinal beam-beam effects;
the inspected PIC kick does not include a separate head-on
:math:`\partial_S\Phi` term. PASS retains both its crossing transform and its
explicit six-dimensional source-distance kick. Two-point interpolation therefore
does not make the complete maps identical. Compare wall time with the same
particle count, slicing, mesh, precision and requested diagnostics; the plane
count alone does not establish a speedup.

Analytic sources and hourglass propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``frozen`` and ``quasi-frozen`` support the six free-space analytic solvers
``gaussian_round_free_space``, ``gaussian_ellipse_free_space``,
``uniform_round_free_space``, ``uniform_ellipse_free_space``,
``parabolic_round_free_space`` and ``parabolic_ellipse_free_space``.
Gaussian tracking uses a complete closed-form jet when well conditioned:
Bassetti--Erskine transverse fields and the Gaussian heat identity provide the
longitudinal source derivative, including changing centers and every covariance
component. Exact round profiles use the radial Hessian. Near-round ellipses,
nearly singular covariance and remote points conservatively retain the common
confocal Green integral, with two quadrature orders for convergence checking.
Requests for the scalar potential and uniform/parabolic profiles also use that
integral. CPU and GPU closed-form calculations use float64 intermediates and
return the configured particle precision. This changes evaluation cost, not
the source model; longitudinal derivatives are retained on both paths.

``Frozen parameters`` supplies ``Center X (m)``, ``Center Y (m)`` and
``Angle (rad)`` (all default zero), plus the appropriate dimensions:
``Sigma (m)``, ``Sigma X (m)`` / ``Sigma Y (m)``, ``Radius (m)``, or
``Semi-axis A (m)`` / ``Semi-axis B (m)``. No arbitrary covariance input is
accepted. ``Slice parameters`` optionally maps slice indices to complete
per-slice parameter sets. Charge always comes from the current live source
population, including for frozen sources.

Without ``Frozen optics reference``, prescribed widths have no angular spread;
the optional ``Source center slopes`` (two numbers, default zero) propagate the
center. For hourglass, reference a Twiss entry at the IP on the source's own
beam. Angular moments follow the prescribed uncoupled Twiss model and are then
rotated with the spatial profile. Optional
``Sigma delta`` in Frozen parameters then defines the dispersion contribution;
input projected widths must leave positive betatron variance after subtracting
that contribution. Ordinary IP Twiss currently supplies horizontal dispersion
only. The internal moments are derived quantities, not additional user inputs.

Quasi-frozen uses the current source centers and transverse phase-space moments
at every slice-pair event. A source that receives kicks is recomputed before
each event; an un-kicked source can reuse its moments within the same bunch-pair
encounter. ``Statistics precision`` optionally selects float32 or float64
accumulation; by default it follows particle precision. Uniform and
parabolic profiles are reconstructed from moments, with semi-axes respectively
:math:`2\sigma` and :math:`\sqrt{6}\sigma`. A round solver uses the average
transverse variance, including after propagation.

Slice events and the six-dimensional kick
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both directions prepare their source snapshots before either kick. Events run
in descending sum of the two entrance slice centroids; a later event therefore
sees the earlier physical update. A PIC snapshot owns the source's four
transverse coordinates, so evaluating the second direction cannot observe
particles already changed by the first kick. Within one bunch-pair encounter,
an un-kicked source slice can reuse its snapshot, PIC transverse coordinates,
and analytic moments across slice-pair events. This includes the strong source
in weak-strong tracking. A source that receives kicks is rebuilt before each
event, as required for strong-strong updates; the source cache does not survive
the encounter.

With target coordinate :math:`z` and opposing
slice centroid :math:`\bar z_s`, the collision distance is
:math:`S=(z-\bar z_s)/2` and the source propagates by :math:`-S`.
Writing :math:`X=x+Sp_x`, :math:`Y=y+Sp_y`, the map is

.. math::

   \Delta p_x=C E_x(X,Y,S),\quad \Delta p_y=C E_y(X,Y,S),
   \qquad x'=x-S\Delta p_x,\quad y'=y-S\Delta p_y,

   \eta'=\eta-\frac{C}{2}\partial_S\Psi+
   \frac{2p_x\Delta p_x+\Delta p_x^2+2p_y\Delta p_y+\Delta p_y^2}{4}.

Here :math:`\eta=(E-E_0)/(\beta_0P_0c)`. Public ``dp`` stores momentum deviation
:math:`\delta` outside the crossing chart; stable relativistic conversions are
used at the boundaries. In PASS energy units, :math:`C=\operatorname{sgn}(Z)
|Z|/(A p_0)` for ions because their reference momentum is stored per nucleon.
Inside a crossing chart this denominator is the frame reference momentum
:math:`p_0^*=p_0\cos(\Theta/2)`.
The source charge is :math:`N_{\rm live} w Z_s e`. There is no extra factor of
two, SpaceCharge length, slice width or SpaceCharge :math:`\gamma^{-2}` factor.
The collision geometry is the ultrarelativistic paraxial model; exact
delta/eta conversion does not turn it into a general finite-velocity collision model.

Luminosity diagnostics
----------------------

Enable native luminosity output inside the shared IP configuration, for example:

.. code-block:: json

   {
     "Luminosity": {
       "Enabled": true,
       "Sample interval (turns)": 100,
       "Output format": "tfs"
     }
   }

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python attribute
     - JSON key
     - Unit
     - Default
     - Description
   * - ``enabled``
     - ``Enabled``
     - —
     - ``false``
     - Enable one shared luminosity recorder per IP occurrence.
   * - ``sample_interval_turns``
     - ``Sample interval (turns)``
     - turn
     - ``100``
     - Positive integer; set 10 to sample every ten turns, with first/final-turn handling below.
   * - ``output_format``
     - ``Output format``
     - —
     - ``tfs``
     - ``tfs``, ``hdf5`` or ``hdf5-gzip1``. The TFS default is omitted from generated JSON for compatibility.
   * - ``reference_luminosity``
     - ``Reference luminosity (cm^-2 s^-1)``
     - cm⁻² s⁻¹
     - ``null``
     - Optional positive fixed reference applied separately to each bunch pair; null uses its first measured value.
   * - ``collision_frequency``
     - ``Collision frequency (Hz)``
     - Hz
     - ``null``
     - Optional positive normalization frequency; null uses the matching reference revolution frequencies.

Sampling and normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, each bunch pair uses its reference revolution frequency
:math:`\beta_0 c/C` for this IP occurrence. The two beams' reference revolution
frequencies must agree within the runtime tolerance; otherwise provide an
explicit ``Collision frequency (Hz)``. This override changes only the diagnostic
normalization and does not repair asynchronous physical encounters. Reference
beta enters this frequency conversion; the collision-plane overlap still uses
the ultrarelativistic common-coordinate model and is not an exact treatment of
nonrelativistic beams or substantially different beam velocities.

The first turn, turns satisfying ``(turn + 1) % interval == 0``, and the final
configured turn are sampled. The stored turn index follows PASS's zero-based
convention. A row is a snapshot of that encounter, not a mean over the preceding
sampling interval. The two BeamBeam rendezvous nodes produce one physical
record, not two independent measurements. Each IP occurrence has its own table
output; changing the sample interval changes diagnostic work, not collision kicks.
The diagnostic applies no particle kick. GPU floating-point reductions can
change their summation order between otherwise equivalent runs, so numerical
agreement does not imply bitwise-identical coordinates; this is also relevant
when repeating a run with the same diagnostic setting.
At each committed sample, the log also reports the current IP luminosity in
``cm^-2 s^-1``, factor and loss. It uses the IP total for multiple bunch pairs
and the pair result when only one pair is present.
Stopping at an unsampled turn does not retrospectively calculate a luminosity
sample. Only records from completed common turns are committed; an interrupted
partial turn does not produce a complete-turn record.

For each slice pair, the diagnostic evaluates the transverse density overlap
at the slice-center collision plane before either slice receives that event's
kick. Later slice pairs use their then-current source state, including earlier
kicks. This is the same thin-slice collision geometry as tracking; finite slice
length and the chosen transverse density representation require convergence
checks. It is not an exact integral over an arbitrary continuous six-dimensional
distribution.

Overlap definition
~~~~~~~~~~~~~~~~~~

For slice pair :math:`(i,j)`, let :math:`N_{A,i}` and :math:`N_{B,j}` be
real-particle populations and let :math:`\rho_{A,i}`, :math:`\rho_{B,j}` be
transverse densities normalized to unit integral. All positions, centers and
covariances below are evaluated in the same collision-plane coordinates at
:math:`S_{ij}=(\bar z_{A,i}-\bar z_{B,j})/2`. The per-bunch-pair overlap is

.. math::

   \mathcal O=\sum_{i,j}\mathcal O_{ij},\qquad
   \mathcal O_{ij}=N_{A,i}N_{B,j}
      \int \rho_{A,i}(\mathbf r)\rho_{B,j}(\mathbf r)\,d^2\mathbf r.

For Gaussian slices, define the center difference :math:`\mathbf d` and
covariance sum :math:`\mathbf V=\mathbf C_{A,i}+\mathbf C_{B,j}`. Then

.. math::

   \mathcal O_{ij}=\frac{N_{A,i}N_{B,j}}{2\pi\sqrt{\det\mathbf V}}
       \exp\!\left(-\tfrac12\mathbf d^T\mathbf V^{-1}\mathbf d\right).

For PIC, let :math:`n_{A,i,g}` and :math:`n_{B,j,g}` be the deposited
real-particle counts assigned to transverse mesh node :math:`g`, including
macro-particle weights. With cell area :math:`A_g=\Delta x\Delta y`,

.. math::

   \mathcal O_{ij}^{\rm PIC}
   =\sum_g\frac{n_{A,i,g}n_{B,j,g}}{A_g}
   =\sum_g\nu_{A,i,g}\nu_{B,j,g}\,A_g,\qquad
   \nu_{A,i,g}=n_{A,i,g}/A_g.

Thus the calculation sums **all slice pairs**, not only equally numbered
slices. No longitudinal bin-width factor is added: :math:`N_{A,i}` and
:math:`N_{B,j}` already contain the slice populations. The two equivalent PIC
forms use counts and number densities respectively; they must not be mixed.

Two Gaussian analytic sources use the closed overlap of their propagated
centers and covariance sum. For Gaussian/uniform or Gaussian/parabolic pairs,
deterministic polar quadrature integrates the bounded profile against the
smooth Gaussian density. For two bounded profiles, each radial ray is clipped
to the intersection of their supports; its polynomial density product is
integrated analytically, followed by angular quadrature. These numerical angular
and mixed-profile integrals retain finite quadrature error.

For an analytic/particle pair, the analytic density is averaged over the other
side's propagated particles. For two particle representations, both are
deposited on a common transverse mesh and the product of the number densities
is integrated. The common domain covers both configured PIC domains and uses
spacing no coarser than either grid; TSC is used if either PIC source requests
TSC, otherwise CIC. Each deposition stencil must remain covered at the slice-pair
collision plane. This diagnostic deposits number density without solving a
potential. PIC luminosity therefore retains its particle and mesh dependence
instead of replacing the source by a Gaussian fit. All branches use real
particle populations, independently of electric charge sign or ion charge
state. Accumulation uses float64 on both backends; source moments and propagated
coordinates still carry their configured tracking-precision error.

Writing the population-weighted overlap per encounter as
:math:`\mathcal O` in :math:`\mathrm{m}^{-2}`, the reported luminosity is

.. math::

   L\,[\mathrm{cm}^{-2}\mathrm{s}^{-1}]
   =10^{-4} f_{\mathrm{coll}}\,[\mathrm{Hz}]
    \mathcal O\,[\mathrm{m}^{-2}].

When a reference is supplied, the reported ratio uses that fixed value for
each bunch pair. Otherwise its first measured luminosity defines the pair's
baseline. This measured baseline generally includes the configured crossing
angle, hourglass effect and beam state; it is not automatically a head-on,
zero-hourglass reference. A crossing-angle or hourglass loss study must supply
the appropriate independently measured or theoretical reference.
If the initial baseline is zero, the ratio remains undefined and is written as
``NaN``; a later nonzero measurement does not silently replace the baseline.

Output files and columns
~~~~~~~~~~~~~~~~~~~~~~~~

Files are written under ``<output directory>/luminosity/`` with the stem
``{output_hms}_{configuration_with_hash}_ip{occurrence}`` and the extension
``.tfs`` or ``.h5``. The configuration name is made safe for filenames and
includes a hash to distinguish names that sanitize identically. TFS is the
default; ``hdf5`` is uncompressed and ``hdf5-gzip1`` uses lossless gzip level 1
with shuffle. All formats have the same columns, units and sampling semantics.
A sampled batch appends rows; it does not require a full particle dump or
rewrite all earlier rows. No additional CSV is produced.

HDF5 follows :doc:`monitor/table_output`: each column is a one-dimensional,
extendible root dataset. ``TURN``, ``BUNCH_A`` and ``BUNCH_B`` use int64; all
other columns use float64. Column chunks contain at least 128 rows. Root
attributes preserve the TFS headers, including configuration, IP occurrence,
source models, geometry, frequency convention, reference convention and units.
The reserved attributes are ``_pass_table_version=1`` and
``_pass_table_columns`` (the ordered column-name list encoded as JSON).

.. list-table:: Luminosity table columns
   :header-rows: 1
   :widths: 35 65

   * - Column
     - Meaning
   * - ``TURN`` / ``TIME``
     - Zero-based turn and reference encounter time in seconds.
   * - ``BUNCH_A`` / ``BUNCH_B``
     - Stable bunch IDs of the physical pair.
   * - ``FREQUENCY_HZ``
     - Frequency used to normalize this pair's luminosity.
   * - ``N_A`` / ``N_B``
     - Live real-particle populations for the encounter.
   * - ``OVERLAP_M2``
     - Population-weighted per-encounter overlap in inverse square metres.
   * - ``LUMINOSITY`` / ``L_REFERENCE``
     - Measured and reference luminosities in inverse square centimetres per second.
   * - ``FACTOR`` / ``LOSS``
     - :math:`L/L_{\rm ref}` and :math:`1-L/L_{\rm ref}`; no clipping to [0, 1].

With multiple bunch pairs, an additional row with ``BUNCH_A=BUNCH_B=-1`` sums
their luminosities and reference luminosities. Its factor is
:math:`\sum L/\sum L_{\rm ref}`, rather than the arithmetic mean of pair factors;
``FREQUENCY_HZ`` is ``NaN`` because the total has no single assigned pair frequency.
One bunch pair produces only its pair row.

Read either format with the shared table reader. It returns a ``TfsDataFrame``
with metadata in ``headers``; no format-specific analysis code is needed.
Read HDF5 after completion or between writes, without holding it open during
tracking; this writer does not enable SWMR.

.. code-block:: python

   from PASS.utils.table_io import read_table

   luminosity = read_table("output/luminosity/run_IP1_hash_ip0.h5")
   print(luminosity.headers)
   print(luminosity[["TURN", "BUNCH_A", "BUNCH_B", "LUMINOSITY", "FACTOR", "LOSS"]])

The filename above is illustrative; use the file produced by your run. Replace
``.h5`` with the actual ``.tfs`` filename for TFS input.

Luminosity diagnostics can run in ``weak-weak`` mode without applying a kick,
but still need matching collision Slicers and the required crossing-coordinate
charts. At least one side must specify a source density method; enabling the
diagnostic with no density model on either side is rejected.

CrossingAngle, CrabCavity and FloatWaister
------------------------------------------

Crossing-angle coordinate charts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python attribute
     - JSON key
     - Unit
     - Default
     - Description
   * - ``configuration``
     - ``Configuration``
     - —
     - required
     - Name of the shared IP configuration.
   * - ``direction``
     - ``Direction``
     - —
     - required
     - ``forward`` or ``inverse``; bracket the matching Slicer and BeamBeam at the same IP.

For nonzero crossing angle, explicitly bracket Slicer and BeamBeam with
``CrossingAngle`` commands having matching ``Configuration`` and
``Direction: forward`` / ``inverse``. A typical local Order is arrival=100,
optional pre-elements=200, forward=300, Slicer=400, BeamBeam=500, inverse=600,
optional post-elements=700, monitor=800. The Slicer must use
``Coordinate: collision_z`` inside this interval. Only that Slicer and BeamBeam
may run between the two charts. Within it ``dp`` temporarily stores eta and
the coordinate frame is labeled in slice output. The inverse restores ordinary
PASS coordinates without changing the public reference momentum or clock.

At head-on geometry the common basis is :math:`(X,Y,Z)` for beam 0 and
:math:`(-X,Y,-Z)` for beam 1. Both longitudinal coordinates remain positive
for early passage. The collision calculation applies this transverse reflection
also when no explicit crossing chart is needed. Frozen sources in a crossing
chart use a linear, thin-slice transform about the design orbit; they do not
claim an exact transformation of an arbitrary six-dimensional distribution.

Common IP-equivalent element parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python attribute
     - JSON key
     - Unit
     - Default
     - Description
   * - ``s``
     - ``S (m)``
     - m
     - required
     - IP position; collision-adjacent elements have zero length and no physical aperture.
   * - ``order``
     - ``Order``
     - —
     - ``null``
     - Integer sequence order; explicitly distinguish commands sharing the IP.
   * - ``optics_reference``
     - ``Optics reference``
     - —
     - required
     - Twiss entry at this IP on this beam.
   * - ``side``
     - ``Side``
     - —
     - required
     - ``before`` or ``after``; descriptive, without an automatic sign change.
   * - ``equivalent_dispersion``
     - ``Equivalent dispersion``
     - —
     - zero
     - Object with Dx (m), Dpx, Dy (m), Dpy, each default zero.
   * - ``longitudinal_shear``
     - ``Longitudinal shear (m)``
     - m
     - ``0.0``
     - Signed longitudinal shear in the equivalent transport.

``CrabCavity`` and ``FloatWaister`` are independent zero-length elements in the
ordinary IP frame. Both require ``Optics reference`` at the same IP and
``Side`` (``before`` or ``after``). Side is descriptive and does not negate
strength or phase. Optional ``Equivalent dispersion`` contains Dx/Dpx/Dy/Dpy,
and ``Longitudinal shear (m)`` defaults to zero. Their canonical transport
includes the longitudinal companion of dispersion and uses the updated energy
on inverse transport. These are optics-equivalent thin maps, not finite-length
electromagnetic cavity tracking.

CrabCavity requires ``Plane`` (x/y), signed ``Phase advance (rad)``,
``Equivalent kick`` and ``Frequency (Hz)``. FloatWaister requires
``Phase advance x (rad)`` and ``Phase advance y (rad)``; ``Mode: rfq`` also
requires ``Equivalent gx (1/m)``, ``Equivalent gy (1/m)`` and frequency,
whereas ``Mode: theory`` requires ``Strength x`` and ``Strength y``.
RF phase uses ``Phase (rad)`` (default zero), ``Phase epoch (s)`` (default
the beam reference clock origin) and each particle's physical arrival time.
Equivalent strengths already include the sign and normalization of charge;
they are not multiplied by charge a second time. Both elements retain the
longitudinal kick from the same generating potential as the transverse kick.

CrabCavity parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python attribute
     - JSON key
     - Unit
     - Default
     - Description
   * - ``plane``
     - ``Plane``
     - —
     - required
     - ``x`` or ``y``.
   * - ``phase_advance``
     - ``Phase advance (rad)``
     - rad
     - required
     - Signed phase advance of the equivalent transport.
   * - ``equivalent_kick``
     - ``Equivalent kick``
     - —
     - required
     - Signed normalized transverse kick amplitude, including charge sign.
   * - ``frequency``
     - ``Frequency (Hz)``
     - Hz
     - required
     - Positive RF frequency.
   * - ``phase``
     - ``Phase (rad)``
     - rad
     - ``0.0``
     - RF phase at the epoch.
   * - ``phase_epoch``
     - ``Phase epoch (s)``
     - s
     - ``null``
     - Null selects the beam reference clock origin.

FloatWaister parameters and strength convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python attribute
     - JSON key
     - Unit
     - Default
     - Description
   * - ``mode``
     - ``Mode``
     - —
     - required
     - ``rfq`` or ``theory``; parameters of the other mode are rejected.
   * - ``phase_advance_x / phase_advance_y``
     - ``Phase advance x (rad) / Phase advance y (rad)``
     - rad
     - required
     - Signed equivalent phase advances in both planes.
   * - ``strength_x / strength_y``
     - ``Strength x / Strength y``
     - —
     - ``null``
     - Both required in theory mode; signed strengths with the convention below.
   * - ``gx / gy``
     - ``Equivalent gx (1/m) / Equivalent gy (1/m)``
     - m⁻¹
     - ``null``
     - Both required in rfq mode; signed equivalent quadrupole coefficients.
   * - ``frequency``
     - ``Frequency (Hz)``
     - Hz
     - ``null``
     - Positive and required in rfq mode; forbidden in theory mode.
   * - ``phase``
     - ``Phase (rad)``
     - rad
     - ``null``
     - RFQ only; null evaluates as zero phase.
   * - ``phase_epoch``
     - ``Phase epoch (s)``
     - s
     - ``null``
     - RFQ only; null uses the beam reference clock origin.

``theory`` is a prescribed ideal traveling-waist map. With zero dispersion
and longitudinal shear, :math:`\alpha_u^*=0` and :math:`\mu_u=\pi/2`,
its action in ordinary IP coordinates is

.. math::

   \Delta u=-\frac{s_u}{2}z p_u,\qquad \Delta p_u=0,\qquad
   \Delta\eta=\frac{s_xp_x^2+s_yp_y^2}{4}.

Here :math:`s_u` is ``Strength x/y`` and positive z means early passage.
Positive strength shifts the drift-equivalent waist to local distance
:math:`s_u z/2`. The ``rfq`` model retains RF curvature and is not automatically
equivalent to theory at finite bunch length; arbitrary pairs of equivalent
gradients also need not represent one physical RF quadrupole. Validate this
traveling-waist model separately from the sextupole crab-waist scheme used
with crossing-angle collisions.

Run initialization and stopping
-------------------------------

Each tracking run starts at turn 0 from the configured initial conditions.
A stop request is honored at a common completed-turn boundary, then buffered
output is finalized. A new run requires a newly initialized simulation and
commands. Beam-beam tracking does not provide a joint checkpoint or a workflow
for continuing a stopped run. Component state snapshots exposed by other
commands do not restore the complete simulation.

Implementation and validation scope
-----------------------------------

The command and rendezvous state live in ``PASS/commands/beam_beam.py``.
The three independent elements live in ``PASS/commands/element/``.
Within ``PASS/commands/collision/``, ``config.py`` defines the schema and
``interaction.py`` owns source preparation, slice-pair order and the six-dimensional
kick. ``hourglass.py`` propagates prepared source centers and covariances, their
distance derivatives, and the PIC source sampling lattice. Source propagation
uses :math:`-S`; its potential derivative holds collision-plane transverse
coordinates fixed. The luminosity diagnostic reuses those propagated moments.
``luminosity.py`` owns density overlap and append-only TFS/HDF5 records. The coordinator
owns output lifetime; the propagation module performs no file I/O.

Existing field solvers, deposition shapes and fused potential-gather kernels
remain shared numerical primitives. The internal propagation module neither
adds a separate physical hourglass element nor introduces arbitrary coupled
frozen-source inputs. No core collision module or separate
coordinator/coordinates module is added.

CPU and GPU support float32 and float64. Main collision arrays follow the
configured precision; clocks and Slicer boundaries retain their appropriate
higher precision. Single-kick analytical checks, potential derivative checks,
PIC convergence, CPU/GPU comparisons, and actual multi-IP tracking are distinct
validation layers. They do not establish a many-turn beam-quality error bound
or A100 throughput. Recheck longitudinal errors separately from transverse
errors when selecting slicing, mesh size and precision.

Theory references
-----------------

For independent luminosity checks, `T. Sen, FERMILAB-FN-1175-AD (2022)
<https://lss.fnal.gov/archive/test-fn/1000/fermilab-fn-1175-ad.pdf>`__ gives
the symmetric Gaussian crossing-angle/hourglass integral (Eq. 2.12), the
head-on hourglass limit (Eq. 2.15), and the crossing-only reduction
(Eq. 2.17). These formulas assume matched Gaussian beams and the specified
IP optics; they are not finite-particle or PIC-grid references.

`R. B. Palmer, SLAC-PUB-4707 (1988)
<https://inspirehep.net/files/6ca30a0993d21afd34ef80ede1e30590>`__ introduces
crab crossing. `L. I. Malysheva et al., IPAC2011 TUPC004
<https://proceedings.jacow.org/IPAC2011/papers/tupc004.pdf>`__ discusses
traveling focus and its offset sensitivity. These papers provide physical
context; the equivalent element strengths and longitudinal-coordinate signs
in PASS are defined by the maps above and must be matched explicitly in a
comparison. In particular, a compensated finite-frequency crab cavity and an
ideal linear tilt need not give identical luminosity for a long bunch.
