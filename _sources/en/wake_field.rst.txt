WakeField
=========

``WakeField`` applies an integrated thin-element wake to particles using a named
``Slicer`` result. One command represents one physical location. Its algorithm
groups have independent source histories or mode states. CPU and CUDA execution
are described here; numerical validation is recorded in the
repository's wake validation report.

Shared configuration and wake points
------------------------------------

The optional root-level ``Wake field`` block contains ``Enabled`` (default true)
and ``Configurations``, a mapping from unique names to objects containing
``Groups``. A wake point supplies either its inline ``Groups`` or a
``Configuration`` reference, exclusively. ``S (m)``, ``Slice set`` and
``Is enabled`` remain per-point parameters. Both enable switches must be true
for a point to execute. Without a root block, existing inline commands behave
as before.

Configurations share input parameters only. Loading the input expands each
reference into an independent definition; each physical point constructs its
own models, histories and solver state. Disabling the module does not permit
invalid configurations or missing references. File paths in shared definitions
are resolved relative to the input JSON, as for inline definitions.

For example, a root block and a corresponding entry inside ``Sequence`` are::

   "Wake field": {
       "Enabled": true,
       "Configurations": {
           "pipe": {"Groups": [{
               "Name": "longitudinal", "Solver": "direct", "History": "none",
               "Components": [{
                   "Component": "longitudinal", "Velocity": {"Kind": "ideal"},
                   "Model": {"Kind": "constant", "Amplitude": 1e12, "Duration (s)": 1e-6}
               }]
           }]}
       }
   },
   "Sequence": {
       "wake_1": {"Command": "WakeField", "S (m)": 1.0,
                  "Slice set": "wake", "Configuration": "pipe", "Is enabled": true}
   }

This is an interface fragment: supply Injection, optics and the matching Slicer
before the wake point. The constant model is illustrative, not a prescribed
machine impedance. The Python API exports ``WakeFieldConfig`` and
``WakeResourceConfig``; pass ``wake_field=WakeFieldConfig(...)`` to
``generate_input``. ``load_input`` returns an expanded inline sequence so that
its two-value return contract does not lose the shared definitions. Direct
low-level command construction requires an inline definition; use
``resolve_wake_point`` when starting from a named configuration.

Physical scope and conventions
------------------------------

PASS tracks supplied or analytic response models. It does not derive the
response of arbitrary three-dimensional structures by solving Maxwell's
equations. The reference-velocity approximation within each bunch is retained
when evaluating a response's velocity coupling. This is distinct from the exact
incoming particle speed used to convert transverse voltage to momentum impulse.

The continuous particle coordinate is :math:`z_i=\beta_b c(T_b-t_i)`.
At this location, the physical arrival time is

.. math::

   t_i=T_b-z_i/(\beta_b c).

No arrival correction is stored. RF reference-energy changes scale z to
preserve this time; regrouping transforms z and momenta into the destination
reference. The wake adapter converts the latest user-supplied z intervals:
centers are :math:`T_b-z_{slice}/(\beta_b c)` and widths are
:math:`\Delta z/(\beta_b c)`. RF does not alter saved intervals or membership.
Newly emitted sources retain their sampled physical times and widths forever;
later reference changes do not reinterpret causal history. See :ref:`en-longitudinal-reference`.

The canonical frequency convention is

.. math::

   F(f)=\int W(t)e^{-2\pi i f t}\,dt,\qquad
   Z_\parallel=F,\qquad Z_\perp=iF.

For source/test monomial powers of total order :math:`n`, integrated wake units
are V/C/m\ :sup:`n` and impedance units are ohm/m\ :sup:`n`. Source moments use
signed real charge represented by each macro particle; witness macro weight
cancels. Positive longitudinal wake means energy loss. Positive transverse
voltage means positive Lorentz force. PASS updates energy per nucleon by
:math:`\Delta E=-Z_{\mathrm{ion}}V_\parallel/A`, then computes momentum exactly.
The transverse kick uses incoming particle :math:`\beta_i`:
:math:`\Delta p_x=(Z_{\mathrm{ion}}/A)V_x/(\beta_i p_0)`.

Causal kernels use half of a finite jump at zero delay. A ``uniform`` source
represents constant charge density over the slice's full time width; its kernel
is integrated analytically or through its primitive. ``point`` sources are at
the slice centers. Slicing convergence and parameter scans are user-controlled;
there is no automatic slice-error controller.

Response models
---------------

.. list-table:: Models (``Kind`` in each component's ``Model``)
   :header-rows: 1
   :widths: 23 77

   * - Kind
     - Parameters and interpretation
   * - ``constant``
     - ``Amplitude``, ``Duration (s)``. Finite causal test response.
   * - ``resonator``
     - Positive ``R``, ``Q``, ``Frequency (Hz)``. R is the real impedance at resonance; amplitude decay rate is :math:`\pi f_r/Q`. Under-, critical- and over-damped cases are supported.
   * - ``tabulated``
     - Increasing ``Times (s)``, matching ``Values``, ``Causal`` (default true). Linear interpolation, zero outside the table. Causal tables start at zero; two-sided tables can include negative delays.
   * - ``file``
     - Numeric ``table`` or ``headtail`` input with explicit column indices, units, signs and normalization. See File input below. Loaded once, with content fingerprinting.
   * - ``impedance``
     - Increasing nonnegative ``Frequencies (Hz)``, matching ``Real``/``Imag``, explicit ``Reconstruction``. Original samples remain immutable.
   * - ``fitted_impedance``
     - The same spectrum plus ``Initial poles (1/s)`` as [real, imaginary] pairs, ``Optimize poles``, ``Max evaluations``, ``Relative floor``, ``Fit tolerance``. Supply only the positive-imaginary pole of each conjugate pair, or real poles.
   * - ``modes``
     - ``Poles (1/s)`` and ``Residues`` as [real, imaginary] pairs, with both members of every conjugate pair explicitly present. Real poles require real residues.
   * - ``resistive_wall``
     - Finite-beta round good-conductor thick wall: ``Radius (m)``, ``Conductivity (S/m)``, ``Length (m)``, ``Beta``, ``Frequencies (Hz)``. Optional ``Wall thickness (m)`` checks the thick-wall condition; it does not activate a finite-wall field-matching solver.
   * - ``ultrarelativistic_wall``
     - Bane-Sands round DC wall, ``Radius (m)``, ``Conductivity (S/m)``, ``Length (m)``. Requires beta >= 0.99; additional short-bunch validity must also be assessed.

The finite-beta wall implements Stupakov, PRAB **23**, 094401 (2020),
`doi:10.1103/PhysRevAccelBeams.23.094401 <https://doi.org/10.1103/PhysRevAccelBeams.23.094401>`_.
It supports longitudinal and diagonal dipolar components. It subtracts the
perfect-conductor response for the identical round geometry and supplies only
the finite-conductivity correction. Scaled Bessel functions retain the finite
beta dependence and avoid overflow. DC is outside this model's validity.
``Max skin depth ratio`` bounds skin depth divided by radius, and also thickness
when supplied. ``Max surface impedance ratio`` bounds the normalized surface
impedance. Both default to 0.1 and cannot exceed 0.1. Finite wall thickness,
magnetic/dispersive materials and arbitrary geometry require an appropriate
validated response; this model does not infer them.

``Field content`` declares ``wake``, ``finite_conductivity_correction``,
``pec_image``, ``direct_space_charge`` or ``total``. PASS does not automatically
subtract space charge. Existing SpaceCharge calculations are transverse and do
not establish that longitudinal or all image terms are present. Combine terms
only after checking geometry, normalization and physical content.

Raw spectra and fitting
-----------------------

The raw route integrates the piecewise-linear spectrum with an oscillatory
quadrature, preserving nonuniform frequency samples and avoiding an artificial
periodic FFT time window. Values outside the measured band are explicitly zero.
A finite-band inverse is two-sided even for a causal underlying system.
``Reconstruction="two_sided"`` retains it. ``causal_projection`` explicitly
sets its negative-time part to zero and uses half the projected jump at zero;
this changes the effective frequency response and is a bandwidth approximation.
Frequency bandwidth and spacing require independent convergence checks.

Fitting uses variable projection: nonlinear optimization of user-selected poles
and real least squares for residues. It does not select the number of modes.
The original spectrum and fit diagnostics remain available on the fitted model.
The fit must satisfy the specified maximum relative error, normalized with the
specified relative floor. Stability follows the selected pole half-planes;
passivity is not automatically enforced or certified. Scalar longitudinal
passivity tests cannot be applied indiscriminately to transverse components.
No instantaneous delta/derivative feedthrough is inferred from missing bandwidth.

Right-half-plane poles describe a decaying backward spatial branch. They are
never advanced as a growing temporal state. ``causal_projection`` fits require
left-half-plane initial poles. General two-sided fits use ``two_sided`` and an
explicit spatial boundary. CST project/binary parsing remains deferred;
numeric exports can use the explicit file convention described below.

Velocity and acceleration
-------------------------

Each component declares ``Velocity``:

* ``Kind="fixed"``, ``Beta``: stationary response at one common reference speed.
  A different reference speed is rejected, including changes during tracking.
* ``Kind="factorized"``, increasing ``Betas`` and matching real ``Source`` and
  ``Witness`` tables: :math:`W(t;\beta_s,\beta_w)=g_s(\beta_s)W_0(t)g_w(\beta_w)`.
  Linear interpolation is restricted to the supplied beta interval. Historical
  excitation uses the source speed at that passage; the current witness factor
  is applied when observing the stored field.
* ``Kind="ideal"`` explicitly defines a velocity-independent point-response
  coupling. This is a model assumption, not an inferred transit-time correction
  for a physical cavity.

The finite-beta wall sets its matching fixed velocity automatically. A single
stationary spectrum cannot determine arbitrary unequal-velocity trajectories,
complex transit phases or arbitrary acceleration. Factorized real coupling
supports acceleration only when that factorization is supplied and valid; pole
frequencies and damping remain fixed. No universal beta multiplier is applied.

Explicit algorithm groups
-------------------------

Command fields are ``S (m)``, ``Command="WakeField"``, ``Slice set``,
``Groups`` or ``Configuration``, and ``Is enabled``. Groups have unique ``Name`` values and nonempty
``Components`` lists. Each component selects ``Component``, ``Model``,
``Velocity``, optional ``Scale`` and ``Field content``.

.. list-table:: Group execution controls
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``Solver``
     - Required: ``direct``, ``fft``, ``recursive`` (resonator), ``modal`` (pole/residue), ``partitioned_fft`` (stationary gapped train), or ``time_fft`` (variable-period physical-time history).
   * - ``History``
     - Required: ``none``, ``direct`` (stored source bins), ``state`` (resonator/modal state), or ``partitioned``. The last requires ``partitioned_fft`` or ``time_fft``.
   * - ``Source shape``
     - ``uniform`` (default) or ``point``; independently selected per group.
   * - ``Memory turns``
     - Positive integer or null for direct history; required finite positive integer for ``partitioned_fft``. Counts preceding passages, with the current passage also included. Must be null for ``time_fft``.
   * - ``Memory time (s)``
     - Positive value or null for direct/partitioned history; required finite positive value for ``time_fft``. Clips the kernel at this delay, including a partial uniform source bin. Physical-time projection introduces mesh error at a discontinuous cutoff.
   * - ``Convolution grid``
     - Required for ``partitioned_fft``; stationary physical time layout described below.
   * - ``Time grid``
     - Required only for ``time_fft``: positive finite ``Step (s)``, integer ``Block size`` >= 2 (default 64), and optional finite ``Origin (s)`` (default null).
   * - ``Partition``
     - ``dyadic`` (default for partitioned FFT) or ``uniform``. No automatic algorithm switch.
   * - ``Max workspace (MiB)``
     - Partitioned FFT conservative peak allocation budget, default 1024 MiB. Checked before allocating history; CUDA also checks available device memory.
   * - ``Boundary``
     - ``causal_passages`` (default), ``isolated``, or ``periodic``.
   * - ``Period (s)``, ``Periodic images``
     - Both required for periodic response; use direct solver and no transient history. Images -N through +N are summed explicitly.

``fft`` requires an increasing uniform current grid with equal source widths no
larger than the spacing. With ``History="direct"``, previous passages are
explicitly calculated directly. No ``auto`` option or silent solver fallback is
provided. Modes retain unlimited decay history with memory proportional to mode
count. Gaps between bunches use physical elapsed time.

Causal passages must be ordered and nonoverlapping in physical time, including
uniform-bin support. Overlap across tracking turns is rejected: turn-batched
tracking cannot infer future trajectories. A spatial ``isolated`` boundary
declares that the complete source train is present in the current batch;
``periodic`` declares its repeated steady state. Both require fixed beta and no
transient history. They cannot be used to represent arbitrary accelerating or
transient ring distributions. The user chooses the number of periodic images
and checks convergence.

Example
-------

.. code-block:: python

   from PASS.para.schema import WakeField

   wake = WakeField(s=0, slice_set="wake", groups=[
       dict(name="short", solver="fft", history="none", source_shape="uniform",
            components=[dict(component="longitudinal", velocity=dict(kind="ideal"),
                model=dict(kind="resonator", r=1e3, q=1, frequency=1e8))]),
       dict(name="long", solver="recursive", history="state", source_shape="point",
            components=[dict(component="dipolar_y",
                velocity=dict(kind="factorized", betas=[0.01, 0.5, 1.0],
                              source=[0.2, 1.0, 0.8], witness=[0.1, 0.3, 1.0]),
                model=dict(kind="resonator", r=1e6, q=1e6, frequency=1e7))]),
   ])

The coupling numbers in this example are illustrative, not cavity field data.
Configure and execute a named Slicer before WakeField. The old flat
``Components``/``Solver``/``Memory turns`` interface must migrate to ``Groups``;
there is no compatibility interpretation of obsolete fields.

Diagnostics and state
---------------------

The command exposes ``last_sources``, ``last_coefficients``, ``last_diagnostics``
and ``group_states``. Diagnostics include each selected algorithm, boundary,
retained passage count, state bytes and fit errors. ``state_dict()`` and
``load_state_dict()`` serialize/restore all groups and check a configuration
fingerprint. These methods cover wake state only. ``reset_state()`` starts
the location with zero field.

``Executor.run(sim, sequences)`` always starts at turn 0. Enabled WakeField
commands must have no retained history at the start of a new run; call
``reset_state()`` before reusing a WakeField command for a newly initialized
simulation.

All macro particles in a beam share one fixed weight from the initial injection
inputs. Source projection uses ``bunch.ratio * bunch.num_charge * e`` as the
charge per live macro particle on CPU and GPU, without an individual-charge
array. The Executor
does not copy tags or update source charges around Injection commands; newly
activated particles retain their original weights. See :doc:`injection`.

CPU and GPU execution
---------------------

CPU and GPU implementations share files, with CUDA helpers and kernel source
below the corresponding CPU code. Under ``PASS/commands/wake/``, temporal
models live in ``wake_models.py``, spectral and rational responses in
``wake_spectrum.py``, explicit solvers and modal scans in ``wake_solvers.py``,
and the two history convolutions in ``convolution.py`` and
``time_convolution.py``. Projection belongs to ``wake_moments.py``, timing to
``wake_timing.py``, source storage to ``wake_state.py``, and coupling to
``wake_components.py`` and ``wake_velocity.py``. Fused particle kicks follow
the command in ``PASS/commands/wake_field.py``. CuPy imports and CUDA compilation
remain lazy, so CPU execution does not require CUDA dependencies.

The same model and group configuration runs on CPU and CUDA. Select the normal
simulation backend; ``execute_gpu`` requires device particle arrays. Original
spectra, tabulated responses, resonators, rational modes, direct history and
spatial periodic sums are evaluated on the GPU. Rational fitting and analytic
finite-beta impedance construction are initialization work on the CPU; their
resulting response arrays are uploaded once per device. Numeric table/HEADTAIL
conversion and partitioned-kernel sampling are also initialization work on CPU.

CUDA execution uses fused ``RawKernel``/``RawModule`` kernels for time conversion,
source moments, model evaluation and integration, exact pair sums, velocity
coupling, event accumulation, spectral products, interpolation, validation and
particle kicks. CuPy remains the device-memory and launch interface; cuFFT and
the device sort/unique primitives remain specialized library operations. Replacing
those libraries with elementary kernels is not a performance requirement.

All populated bunches share one projection launch and one kick launch, including
unequal reference times, velocities and saved slice widths. Empty populations
are removed without a separate device-to-host check for each bunch. Layout and
pointer caches are rebuilt when the particle ranges or SliceSet storage change;
reference parameters update independently. Source records retain their own arrays.
Fixed-range equal-length Slicer grids also batch their histogram and table work
across bunches, with one transfer for alive/outside diagnostics. Host wake-source
snapshots upload their arrays together; the device rows retain ownership of that
passage and are not overwritten by later uploads.

The direct solver assigns one warp to a witness and reduces the scalar response
over source slices. It does not allocate a source-by-witness matrix. All response
types also use this approach for physical-time near-field corrections. Ordinary
FFT response spectra are cached by validated spacing, width, model configuration
and cutoff; changing reference beta invalidates the cache when geometry changes.
Uniform-grid validity is checked on device even when the spectra are reused.

Particle coordinates may use float32 or float64. Physical arrival times, slice
charge moments, response calculations, modal states and energy/momentum
conversions use float64 in both cases. GPU atomic reductions change summation
order, so CPU/GPU results need tolerance-based comparisons, not bitwise equality.
Float32 particle storage can still change slice membership near a boundary and
accumulate transport rounding over many turns.

Projection accumulates charge and transverse dipole moments in a single particle
pass using block-local histograms. The particle kick combines all component
monomials and applies the mechanical update in one CUDA kernel. Modal history
uses bounded affine prefix scans with decaying exponential factors, including
critical and overdamped resonators. Within-passage event times are relative to
the passage origin to resolve narrow bins after a long elapsed time.
For increasing centers whose bin edges do not reach neighboring centers, event
ordering is constructed directly, including adjacent-bin overlap and coincident
edges. Other layouts retain general sort/unique handling. Mode updates allocate
their next vectors once and preserve the previous state for rollback/checkpoints.
Physical-time convolution validates each complete deposited frame before its
transaction, then processes its sub-blocks without repeated host checks. A
single-slot spatial transform uses a one-dimensional cuFFT; source geometry,
response models, history cutoffs and the time-mesh approximation are unchanged.

Equal-length local z intervals convert to a uniform time grid at the current reference velocity. Response caches depend on this actual grid geometry; a change of beta changes its widths. The user controls Slicer updates. Checkpoint serialization transfers device history to portable host arrays.

Small problems can run faster on CPU because CUDA launch and synchronization
costs dominate. Measure a representative workload after warmup, synchronize
CUDA events, and include Slicer, clock, projection and kick costs when comparing
complete tracking. The generated validation report under
``tests/codex/wake_redesign/validation_report.md`` records the actual device,
problem sizes, numerical comparisons, timings and reproducible commands.

Custom spatial terms
--------------------

Existing named components remain supported. ``Component="custom"`` requires
``Spatial`` containing ``Plane`` (x/y/z), ``Source powers`` [a,b] and
``Test powers`` [c,d], all nonnegative integers. The projected source moment is
:math:`\sum_j q_j x_j^a y_j^b`; its response is multiplied by witness
:math:`x_i^c y_i^d`. The kernel unit order is a+b+c+d. A custom term specifies
one supplied polynomial response; it does not derive unprovided multipoles or
enforce cross-component Maxwell constraints. The round-wall analytic model
retains its explicitly supported longitudinal/diagonal dipolar components.

CPU and CUDA support the same powers. The batch CUDA histogram accumulates
Q/Qx/Qy and arbitrary requested monomials in the same particle pass.
Integer witness powers are evaluated in the fused mechanical kick.

Stationary multibunch and long history
--------------------------------------

Set ``Solver="partitioned_fft"``, ``History="partitioned"`` and finite
``Memory turns=H``. No modal fit is required for an arbitrary causal table.
Define the sample-center times by

.. math::

   t_{n,b,i}=t_{\rm origin}+nT+bP+i\Delta t,
   \qquad K_{\ell,d,q}=\overline W(\ell T+dP+q\Delta t).

The grid fields are ``Period (s)`` T, ``Slots`` B, ``Slices`` S,
``Slot spacing (s)`` P, ``Slice spacing (s)`` delta-t, ``Origin (s)``
(default 0), ``Width (s)`` (default 0), and ``Projection`` (default ``exact``).
Period/spacings must be positive. Width must not exceed slice spacing; slot
windows cannot overlap or span more than one period. Uniform source shape
requires positive width; point shape requires zero width.

Slot and intra-slot indices are padded independently to at least 2B-1 and
2S-1. This is a linear, gapped convolution, even if P/delta-t is not an integer.
Empty slots are zero; they still occupy their physical time positions. Source
bins may be reordered or duplicated, and independent overlapping populations
are summed conservatively. Times are mapped from the physical arrival clock,
not from the enumeration order of bunches. Since arrival uses T_b-z/(beta_b*c),
increasing harmonic IDs need not have increasing arrival times.

``Projection="exact"`` requires the observed centers and widths to match this
grid at every passage, up to floating-point roundoff. ``linear`` permits
off-grid **point** sources within the declared windows: charge moments are
distributed to two time nodes and fields interpolated back. This introduces a
time-mesh approximation; refine the grid, especially near sharp wake features.
The history algorithm adds no tail fitting, temporal decimation or coarsening.
Changing revolution period, drifting outside the windows or incompatible
widths raises an error. Choose ``time_fft`` for variable-period causal history
with a converged physical-time mesh, or direct summation. There is no silent reset
or change of history algorithm.

Uniform partitions schedule all H lag spectra each passage. Dyadic partitions
cover lags [L,2L) for L=1,2,4,..., truncated at H. Each completed L-passage source
block produces contributions only to future passages; no future beam trajectory
is used. Single-lag partitions use spectral delay lines. With F padded spatial
frequency cells, the history work is O(H F) per turn for uniform partitions and
O(F log-squared H) amortized for dyadic partitions, in addition to spatial FFTs
and component factors. Both retain O(H F) spectral storage. Dyadic scheduling
has bursts at block boundaries; benchmark mean and maximum latency over at
least one complete largest-block cycle, as well as the median.

These bounds assume a fixed number of source channels/components. Components
with identical source powers and source-speed laws share a forward spatial FFT.
Kernels and FFT resources belong to ``GroupExecution``; ring buffers and pending
fields belong to ``ConvolutionState``. They are not stored in component objects.
CUDA plans retain their cuFFT handles across long scheduling intervals; temporal
transforms use contiguous time batches. Uniform scheduling fuses spectral
multiply/add without allocating a full-history temporary. Projection and kicks are batched across the train for all solver groups,
using the latest explicitly generated SliceSets; multibunch clock evaluation is
also batched. Float64 is retained throughout.
All groups preview their new fields before the command applies kicks and commits
history. Persistent spectra are not copied every turn. Portable checkpoints
include the sampled-kernel/grid fingerprint and are restored on the selected
backend; spectral shapes and values are checked before resumed tracking.
Passage counters must be consecutive, including empty turns. ``reset_state()``
is required for a fresh run; initial past history is zero.

For an equal-length Slicer on [-zmax,zmax], common beta, harmonic slots 0..B-1,
circumference C, and t0=n*C/(beta*c), an exact point grid is:

.. code-block:: python

   from PASS.para.schema import WakeSolverGroup, WakeConvolutionGrid
   speed = beta * 299792458.0
   grid = WakeConvolutionGrid(
       period=C/speed, slots=B, slices=S, slot_spacing=C/(B*speed),
       slice_spacing=2*zmax/(S*speed),
       origin=-((B-1)*C/B + zmax-zmax/S)/speed, width=0)
   group = WakeSolverGroup(
       name="tabular_tail", solver="partitioned_fft", history="partitioned",
       source_shape="point", memory_turns=512, convolution_grid=grid,
       partition="dyadic", max_workspace_mb=2048, components=components)

Use the normal GPU backend to run the same configuration on CUDA. Particle
storage may be float32; the declared physical mesh and wake arithmetic remain
double precision. Projection membership near a slice edge can still differ
with float32. The generated suite, benchmark and full tracking example are in
``tests/codex/wake_fast_history/``; the benchmark includes source deposition,
history scheduling and synchronized end-to-end tracking separately.

Variable-period general history
-------------------------------

``Solver="time_fft"`` uses fixed physical-time nodes
:math:`t_k=t_{\rm origin}+k\Delta t`, independently of revolution periods.
Set ``History="partitioned"``, finite ``Memory time (s)`` and ``Time grid``;
leave ``Memory turns`` and ``Convolution grid`` unset. The input arrival clock
supplies each passage's actual times. Changing beta, bunch spacing, bin width
or revolution period does not rescale or discard the existing history.
The previously stated response/velocity approximations still apply: this
algorithm does not derive a general two-time electromagnetic response.

Point source moments are deposited linearly onto neighboring time nodes.
For finite uniform bins, deposition integrates the same hat basis over the
source interval, conserving each coupled source moment. Fields are linearly
interpolated to witness times. Within half the source width plus two mesh steps
of a source center, the exact source-averaged response replaces its projected
pair contribution. This local correction preserves the causal jump and half
self-wake, including sources from the preceding passage. Far-field projection
remains an approximation. Point sources and witnesses exactly on the mesh
recover the corresponding discrete direct sum up to floating-point error.

Resolve short response features and high frequencies by refining ``Step (s)``.
Converge the physical slice count and memory duration separately. Table knots
and a sharp memory cutoff can limit the observed order of convergence; no
universal error tolerance follows just from selecting ``time_fft``. The step
also must be resolvable by the absolute floating-point arrival clock.

Completed physical-time blocks feed uniform or dyadic online convolution.
The unfinished block, including source deposition beyond its center, remains
open across passages. One node before the passage end is also kept writable,
so roundoff at touching passage boundaries cannot prematurely seal a source
deposition node. This changes the physical-time checkpoint plan version;
checkpoints produced before this guard must be regenerated from the initial
beam state. Preview evaluates the currently available sources;
no future beam trajectory is requested. Persistent ring rows touched during
preview are saved and restored exactly, and updates are committed only after
all groups validate their kicks. CPU/GPU checkpoints include the open block,
near-correction sources, clock origin and completed-block history. Empty
passages still advance the passage counter. Gaps longer than the complete
memory horizon and projection support can skip expired empty blocks exactly.

For block size B, horizon H seconds and M=ceil(H/(B*delta-t))+1 history
blocks, dyadic work per completed block is O(B log B + B log-squared M)
amortized, plus source deposition, witness interpolation and local pair
correction; storage is O(B M) for a fixed number of channels/components.
One passage may span several blocks. Fine meshes also represent empty time
between bunches, so ``partitioned_fft`` can be substantially cheaper for a
stationary sparse train. Large overlapping source widths can increase local
pair work. ``Block size`` changes scheduling and memory traffic, not physical
resolution. Benchmark full largest-block cycles including peak latency.

CPU and CUDA implement the same projection and history scheme. CUDA uses
device deposition and fused projected-pair evaluation, shares source FFTs,
retains cuFFT plans and keeps spectral history on the device. Small validation
metadata still synchronizes with the host. Slicer continues to be invoked per
bunch; variable-period support does not require a train-wide Slicer API.

.. code-block:: python

   from PASS.para.schema import WakeSolverGroup, WakeTimeGrid
   group = WakeSolverGroup(
       name="accelerating_tail", solver="time_fft", history="partitioned",
       source_shape="uniform", memory_time=200e-6,
       time_grid=WakeTimeGrid(step=0.5e-9, block_size=1024),
       partition="dyadic", max_workspace_mb=2048, components=components)

These mesh values are illustrative and require convergence for the supplied
wake. With a null origin, the first source interval chooses a nearby origin;
an explicit origin must not follow the first source support. Source batches
must remain causally ordered and nonoverlapping across calls, including bin
widths. Independent overlapping populations inside one batch are permitted.
The executable validation and synchronized benchmarks are under
``tests/codex/wake_variable_period/``.

Coasting beams
--------------

A coasting beam can be represented by one bunch group with
``harmonic_number=1`` and an equal-length explicit Slicer covering a full
circumference. This permits full-ring charge/current and transverse-moment
profiles; it does not require RF bunching. For uniform current I and a finite
causal response, the settled voltage is :math:`V=I\int_0^\infty W(\tau)d\tau`.
The validation suite checks this DC result and convergence of an azimuthal
harmonic against its analytic convolution on CPU and CUDA. Zero initial
history produces a startup transient; allow the response memory to fill.
``Boundary="periodic"`` is a prescribed repeated steady distribution, with
image-count convergence, rather than evolving transient history.

For evolving coasting profiles, use ``Coordinate=arrival_phase`` (or
``Periodic=true``), ``equal_length`` and ``Explicit={"z min": -C, "z max": 0}``.
At an explicit Slicer update, :math:`z_{phase}=-C[(-u)\bmod1]` with
:math:`u=v_{obs}(T_{obs}-t_i)/C`. The prescribed clock selects the common
observation event and velocity; see :doc:`slicer`. Bunch reference times and
velocities may differ. All populations must share the saved observation window
and circumference, and Slicer must be at the wake location.

The bins cover :math:`[T_{obs},T_{obs}+C/v_{obs})`. An exact integer phase maps
to the window start and slice 0, rather than its excluded right endpoint.
A new physical source passage
requires a user Slicer update; reusing a periodic snapshot retains its old
window. No automatic reslicing is performed. Use ``Boundary="causal_passages"``
for evolving history. Variable, non-overlapping passage windows are supported
by ``time_fft``; a fixed convolution grid still requires its documented timing.

This is the **one passage per particle per reference turn approximation**.
It supports long accumulated slip and momentum spread within this model,
but does not schedule zero or multiple individual crossings during one turn.
For a uniform rigid stream with actual period Ti, the represented current
is Q/T instead of Q/Ti. If epsilon=1-Ti/T, the relative current error is
abs(epsilon); the included actual-Drift test verifies this first-order error
and its reduction with momentum spread. A full-ring causal voltage test
checks charge conservation and :math:`I\int W` on CPU and CUDA.

Require small per-turn slip, small collective change within a turn, and
:math:`2\pi |m\,\Delta u|\ll1` for each resolved azimuthal mode m. Converge
slice count and the time mesh separately. ``Max phase slip`` (default 0.05,
at most 0.1) rejects large observed single-step phase changes when WakeField
consumes the slices; diagnostic Slicer projections remain available. It is a guard,
not a universal accuracy tolerance. Floating particle storage must also
resolve the slice width after accumulated motion; prefer float64 for long
coasting runs. An ordinary explicit Slicer keeps its original boundary
clipping behavior unless periodic arrival slicing is enabled.

Regression and block contracts
------------------------------

``PartitionedConvolution.preview_block`` accepts finite float64 backend
arrays with shape (source channels, slots, slices). Source velocity factors
are already included; witness factors are applied only after physical-time
gathering. The returned update owns its spectrum independently of the caller
array. Preview leaves persistent history unchanged, and an update can be
committed once. Multi-block previews use a bounded ring-row transaction with
exception-safe rollback; overlapping transactions on one state are rejected.
These interfaces preserve group-level atomic history acceptance without
copying the entire long history on every passage.

The versioned runner is ``python -m tests.codex.wake_production.run_regression``.
CPU CI runs on pull requests and pushes; CUDA hosts run the same command with
``--require-gpu``, which fails if no real device is available. Generated
results and environments remain ignored by Git. The long-term benchmark
``python -m tests.codex.wake_production.long_term --backend cpu`` (or gpu)
checks 2048-turn self-consistent modes, growth, tune and an intensity threshold
for a nonmodal table with varying periods. Its independent reference is a
finite-memory Floquet matrix, not a second PASS solver. It uses normalized
units and a small external linear damping map; the threshold is specific to
that benchmark and is not a general machine instability limit.

CUDA converts the saved slice intervals directly to physical time.
All supported models use a warp per witness for the exact near-field correction,
without allocating source-target pair lists or synchronizing pair counts. Performance benchmarks
include the per-bunch Slicer, reference changes, projection, history and kick,
with warm-up and at least one complete largest-block scheduling cycle.

File input
----------

``Model.Kind="file"`` takes ``File path``, ``Format`` (``table`` default or
``headtail``), ``Axis column`` (default 0), required ``Value column``, optional
``Imag column``, ``Delimiter`` (null for whitespace), ``Skip rows`` (default 0),
``Causal`` (default true), ``Reconstruction`` (``two_sided`` default), optional
``Length (m)``, and required ``Convention``. Columns are zero-based, distinct,
and numeric. UTF-8 files may contain # comments and a byte-order mark. General
tables are linearly interpolated with zero outside their supplied support;
causal tables must include zero delay. Unordered/duplicate samples are rejected;
a reversed time/distance axis is accepted and reordered.

``Convention`` declares ``Data kind`` (``wake_function``/``impedance``),
``Axis`` (``time``/``distance``/``frequency``), ``Axis unit``, ``Value unit``,
``Positive trailing``, ``Longitudinal positive loss``, ``Integrated``, and
``Reference beta``. ``Fourier exponent`` defaults to -1,
``Transverse impedance factor`` to i (also -i or 1), and
``Shunt impedance convention`` to ``not_applicable`` (provenance only: numeric
samples are already normalized; this field does not rescale a shunt impedance).

Time units: s/ms/us/ns/ps; distance: m/cm/mm; frequency: Hz/kHz/MHz/GHz.
Wake amplitudes explicitly use V/kV/MV divided by C/nC/pC and the required
spatial powers, for example ``V/C/m^2`` or ``V/(pC*mm)``. Impedance units use
ohm/Ohm/kOhm/MOhm with spatial powers. Per-length data add one denominator
length power and require physical ``Length (m)``; integrated data forbid it.
Distance coordinates convert to delay through reference beta*c, without an
additional wake-amplitude Jacobian. Fourier and longitudinal signs are
converted explicitly. Impedance files require both real and imaginary columns
and positive-trailing delay convention; use the Fourier sign field for
opposite transform signs. Finite-band causal projection retains its documented
approximation. A finite-bunch wake potential requires separate deconvolution
and is rejected as a point-charge wake.

HEADTAIL supports several column layouts, so select the columns explicitly.
Its contract is ns and integrated V/pC for order zero, V/(pC*mm) for order one;
signs remain explicit. See the `CERN HEADTAIL table specification
<https://indico.cern.ch/event/178920/contributions/1446485/attachments/235706/329825/HDTL_lattice_def.pdf>`_.
PASS reads these numeric units directly; it does not depend on Xwakes or
PyHEADTAIL. For example:

.. code-block:: python

   model = dict(kind="file", file_path="tail.dat", format="headtail",
       axis_column=0, value_column=2,
       convention=dict(data_kind="wake_function", axis="time", axis_unit="ns",
           value_unit="V/(pC*mm)", positive_trailing=True,
           longitudinal_positive_loss=True, integrated=True, reference_beta=beta))

Input JSON resolves ``File path`` relative to its directory. Direct Python
construction uses the supplied path relative to the current directory. Files
are read only at construction; source hash and conversion metadata remain on
the model. Checkpoints verify file-content hashes as well as configuration.
CST project/binary parsing, automatic unit detection and wake-potential
deconvolution are not included; exported numeric files can use ``table`` with
their actual conventions explicitly provided.


Slice coordinates and response boundaries
-----------------------------------------

WakeField accepts local ``Coordinate=z_rel`` intervals and the separate
``Coordinate=arrival_phase`` coasting projection. WakeField rejects the
``Coordinate=z_periodic`` circumference-folded slices required by SpaceCharge,
because their centers do not preserve continuous arrival times. Use separate
named slice sets. The latest explicit Slicer result controls membership and geometry.
Coordinate selection does not change the algorithm group's ``Boundary``:
``periodic`` is a repeated steady spatial response; evolving passage history
uses ``causal_passages``. Timing uses float64 and the stored continuous z.
