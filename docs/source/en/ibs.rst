Intrabeam Scattering (IBS)
================================================================================

``IBS`` models small-angle Coulomb scattering within each bunch. The same
species-aware interface supports electrons, positrons, protons and ions,
with bunched or coasting longitudinal density. Three methods cover local
Gaussian growth-rate diagnostics, Gaussian kinetic tracking, and local
binary collisions. Particle updates use NumPy or CuPy; analytical quadrature
and random-number generation run on the CPU.

The implementations are derived independently from the cited physics papers.
They do not import, vendor or translate Xsuite's IBS implementation.

Choosing a method
--------------------------------------------------------------------------------

.. list-table:: Available methods
   :header-rows: 1
   :widths: 20 35 45

   * - ``Method``
     - Operation
     - Assumptions and limits
   * - ``bjorken_mtingwa``
     - Compute instantaneous local growth rates; do not kick particles.
     - Gaussian, uncoupled betatron distribution; both horizontal and vertical dispersion are included. Supply local optics. Mismatched measured covariance is reported with ``model_moments_matched=False``.
   * - ``kinetic``
     - Apply an independently constructed tensor Ornstein--Uhlenbeck (OU) closure whose Gaussian covariance derivative matches the BM kernel.
     - Requires measured covariance consistent with supplied uncoupled optics and uncorrelated longitudinal moments. Optional saved longitudinal slices weight the collision density.
   * - ``binary``
     - Randomly pair particles within three-dimensional cells and scatter their relative momenta.
     - Does not assume a Gaussian density; requires enough particles per cell and convergence in grid size and interaction interval. Do not supply optics.

All methods assume weak Coulomb/plasma coupling and a prescribed positive
Coulomb logarithm; this assumption concerns binary encounters, not transverse
betatron coupling.
Tracking requires small thermal motion in the beam rest frame; the current
implementation rejects proper speed :math:`|\mathbf p_*|/(Mc)>0.05`.
This permits relativistic reference motion, but does not implement a fully
relativistic thermal collision operator. Large-angle scattering and Touschek
losses are outside this model. Radiation damping, quantum excitation and
electron cooling are separate physical processes and are not added by IBS.

The Gaussian models do not represent arbitrary transverse coupling or a
strongly non-Gaussian distribution. The binary model provides a local
collision operator for non-Gaussian studies; macroparticle and mesh convergence
must establish the accuracy of a particular application. No Nagaitsev backend
is included in this version.

The kinetic method matches second moments of a prescribed Gaussian model.
It is not the full Landau collision operator or a literal implementation of
the published modified stochastic kick. Covariance matching is necessary but
does not establish Gaussianity or validate strong-space-charge operation.
Combining IBS with space charge requires independent convergence and physical
validation for the beam being studied.

Python configuration
--------------------------------------------------------------------------------

This fragment adds one kinetic interaction point to an existing ``sequence``.
Supply injection and lattice transport as described in :doc:`input_generation`.
The local optics must describe the actual distribution and transport at this
point, including the dispersion convention below.

.. code-block:: python

   from PASS.para.api import generate_input
   from PASS.para.schema import IBSConfig, IBSConfiguration, IBSItem, IBSOpticsConfig

   ibs = IBSConfig(
       enabled=True,
       configurations={
           "CoreIBS": IBSConfiguration(
               method="kinetic", coulomb_log=12.0,
               random_seed=2026, bunched=True,
           ),
       },
   )
   sequence.add(
       "ibs_at_s10",
       IBSItem(
           s=10.0, configuration="CoreIBS", interaction_length=1.0,
           optics=IBSOpticsConfig(beta_x=8.0, beta_y=6.0, dx=0.5),
           save_diagnostics=True,
       ),
   )
   generate_input(main, sequence, "beam0.json", intrabeam_scattering=ibs)

``Coulomb log=12`` is an example input, not a universal physical default.
Choose the impact-parameter cutoffs for the beam and regime under study.
PASS does not infer cutoffs, screening, or a strong-coupling correction.
``coulomb_log_note`` can record the physical cutoff prescription, values,
units and source used to choose :math:`\ln\Lambda=\ln(b_{\max}/b_{\min})`.
It is provenance text only and does not change the collision calculation.
The collision mesh is a numerical density-resolution choice; choosing its
cell size does not determine the physical Coulomb logarithm.

For rate diagnostics, select ``method="bjorken_mtingwa"``. An interaction
length of zero is valid and still permits rate evaluation. For binary tracking,
select ``method="binary"``, supply ``grid_shape=(nx, ny, nz)`` and optional
``grid_bounds=((xmin, ymin, zmin), (xmax, ymax, zmax))`` in laboratory metres,
and omit ``optics``. ``collision_steps`` divides the exposure into steps with
new random partners in each step. Configuration names are case-sensitive; schema field names
are accepted case-insensitively by the loader.

Input fields
--------------------------------------------------------------------------------

The top-level JSON block is ``Intrabeam scattering``. ``IBSConfig`` contains
``enabled`` / ``Enabled`` (strict boolean, default ``False``) and
``configurations`` / ``Configurations`` (a mapping of names to
``IBSConfiguration``). A disabled block ignores configuration contents and
disables all its IBS commands.

.. list-table:: ``IBSConfiguration``
   :header-rows: 1
   :widths: 22 23 20 35

   * - Python name
     - JSON field
     - Default / type
     - Meaning
   * - ``method``
     - ``Method``
     - ``"bjorken_mtingwa"``
     - One of the three methods above.
   * - ``coulomb_log``
     - ``Coulomb log``
     - Required float
     - Finite and strictly positive, dimensionless.
   * - ``coulomb_log_note``
     - ``Coulomb log note``
     - ``None``; string or null
     - Optional physical-cutoff prescription and source. If present, must be nonempty without surrounding whitespace. Provenance only; excluded from the physics state hash.
   * - ``random_seed``
     - ``Random Seed``
     - ``None``; integer or null
     - Nonnegative strict integer gives reproducible initialization; ``None`` requests fresh entropy. Booleans, floats and numeric strings are rejected.
   * - ``bunched``
     - ``Bunched``
     - ``True``; strict boolean
     - Bunched density or coasting density over the circumference.
   * - ``slice_set``
     - ``Slice set``
     - ``None``; name or null
     - Optional kinetic line-density weighting from an executed Slicer. Other methods require null.
   * - ``grid_shape``
     - ``Grid shape``
     - ``(8, 8, 8)``; three positive strict integers
     - Binary cell counts along x, y and longitudinal position. Nondefault values require the binary method.
   * - ``grid_bounds``
     - ``Grid bounds (m)``
     - ``None`` or two triples
     - Binary lower and upper bounds in laboratory ``(x, y, z_rel)`` metres. Every upper bound must exceed its lower bound; outside particles cause an error.
   * - ``collision_steps``
     - ``Collision steps``
     - ``1``; positive strict integer
     - Binary physical exposure subdivisions with fresh random partners. Positions remain fixed within the command. Nondefault values require binary.
   * - ``matching_tolerance``
     - ``Matching tolerance``
     - ``0.1``; float in (0, 0.25]
     - Gaussian dimensionless covariance discrepancy limit, defined below. Nondefault values require a Gaussian method; there is no option to disable the kinetic guard.
   * - ``max_scattering``
     - ``Max scattering``
     - ``0.05``; float in (0, 0.05]
     - Kinetic friction-exposure bound; binary bound on the variance of the half-angle tangent per angular substep.
   * - ``max_substeps``
     - ``Max substeps``
     - ``1000``; positive strict integer
     - Exceeding this limit raises an error; collision strength is not silently clipped.

.. list-table:: ``IBSItem`` sequence command
   :header-rows: 1
   :widths: 22 23 20 35

   * - Python name
     - JSON field
     - Default / type
     - Meaning
   * - ``command``
     - ``Command``
     - ``"IBS"``
     - Registered command name.
   * - ``s``
     - ``S (m)``
     - ``0.0``; finite float
     - Sequence location in metres.
   * - ``order``
     - ``Order``
     - ``None``; strict integer or null
     - Optional explicit order at a shared position; see :doc:`input_generation`.
   * - ``configuration``
     - ``Configuration``
     - Required name
     - Reference to ``Intrabeam scattering.Configurations``.
   * - ``interaction_length``
     - ``Interaction length (m)``
     - Required nonnegative float
     - Positive physical exposure; this command does not transport particles.
   * - ``optics``
     - ``Optics``
     - ``None`` or ``IBSOpticsConfig``
     - Required for Gaussian methods, rejected for binary collisions.
   * - ``is_enabled``
     - ``Is enabled``
     - ``True``; strict boolean
     - Enable this point under the global switch.
   * - ``save_diagnostics``
     - ``Save diagnostics``
     - ``False``; strict boolean
     - Save diagnostic JSON records on turns selected by ``save_turns``.
   * - ``save_turns``
     - ``Save turns``
     - ``[]``; list of strict-integer selectors
     - Empty means every execution when saving is enabled; otherwise select inclusive zero-based turns with ``[turn]`` or ``[start, end, step]`` rows.

``IBSOpticsConfig`` requires positive finite ``beta_x`` / ``Beta x (m)`` and
``beta_y`` / ``Beta y (m)``. Its remaining finite floats default to zero:
``alpha_x`` / ``Alpha x``, ``alpha_y`` / ``Alpha y``, ``dx`` / ``Dx (m)``,
``dpx`` / ``Dpx``, ``dy`` / ``Dy (m)``, and ``dpy`` / ``Dpy``.
The dispersion derivatives refer to PASS normalized mechanical momenta,
not exact transverse slopes. The Gaussian approximation identifies them
only to paraxial order. This version does not infer IBS optics from a Twiss
command; the user supplies consistent local optics explicitly.

PASS expects all four dispersion fields as derivatives with respect to
:math:`\delta=(P-P_0)/P_0`. Native MAD-X ``TWISS`` instead reports derivatives
with respect to ``PT``; near the reference particle,

.. math::

   \mathrm{PT}=\frac{E-E_0}{P_0c}\simeq\beta_0\delta,
   \qquad D_{a,\delta}=\beta_0 D_{a,\mathrm{PT}},
   \qquad a\in\{x,p_x,y,p_y\}.

Thus multiply native MAD-X ``TWISS`` values ``DX``, ``DPX``, ``DY`` and
``DPY`` by :math:`\beta_0` before supplying them here; see the
`MAD-X manual's dispersion definitions
<https://raw.githubusercontent.com/MethodicalAcceleratorDesign/MAD-X/master/doc/usrguide/Introduction/tables.html#linear>`_.
MAD-X ``IBS``-table dispersion is already converted to the momentum
convention: do not multiply it by beta again. Public-API checks with MAD-X
5.09.03 also show midpoint positions and averaged adjacent TWISS values in
that table, so its rows cannot be substituted for endpoint optics solely by
matching element names.

``IBSOpticsConfig.from_twiss(twiss, endpoint="exit", dy=0.0, dpy=0.0)``
constructs those explicit values from a PASS ``TwissItem`` or its complete
JSON mapping. ``endpoint="exit"`` selects current beta, alpha, Dx and Dpx;
``endpoint="entrance"`` selects the corresponding previous values. The
result is independently validated as ``IBSOpticsConfig``. This helper does
not accept arbitrary MAD-X rows, infer missing required Twiss data, choose
the IBS sequence position, or change runtime transport. Because ``TwissItem``
does not store vertical dispersion, supply ``dy`` and ``dpy`` explicitly when
nonzero.

.. code-block:: python

   # twiss_item is the PASS Twiss transport map associated with this location.
   optics = IBSOpticsConfig.from_twiss(twiss_item, endpoint="exit", dy=0.0, dpy=0.0)

Exposure, coordinates and density
--------------------------------------------------------------------------------

At each command the laboratory exposure and beam-rest exposure are

.. math::

   \Delta t=\frac{L_{\mathrm{IBS}}}{\beta_0 c},
   \qquad \Delta t_* = \frac{\Delta t}{\gamma_0}.

Each execution applies its full specified exposure. For a distributed ring
model, interaction lengths should represent the intended lattice segments.
Do not apply a full circumference at every node. Coefficients are recomputed
on every call and, for kinetic tracking, at each adaptive substep. IBS is an
independent positive-time collision operator; negative integration substeps
are not accepted as collision exposures.

Input validation sums the nonnegative interaction lengths of enabled kinetic
and binary commands in each beam, including zero-length points. If any such
command exists and the sum differs from the circumference
with relative tolerance :math:`10^{-9}`, it emits the warning
``ibs.exposure_length``. Diagnostic-only BM points and disabled commands are
excluded. Partial-ring studies can intentionally produce this warning; it
does not change or reject the exposure. Equal total length does not establish
spatial coverage, nor correct total collision time when reference beta varies.

IBS leaves ``x``, ``y``, continuous ``z_rel``, the reference energy and
``bunch.t0`` unchanged. Kicks update ``px``, ``py`` and total relative momentum
deviation ``dp``. Live macroparticle count times the bunch's macroparticle
weight gives the real population. Ion classical radius uses the complete
ion mass :math:`M` and charge :math:`q=Ze`:

.. math::

   r_0=\frac{q^2}{4\pi\epsilon_0 M c^2}.

Gaussian beam moments are measured after centroid and specified dispersion
subtraction. A bunched model uses the RMS continuous ``z_rel``. The coasting
Gaussian model replaces :math:`\sigma_z` by
:math:`C/(2\sqrt{\pi})`, giving uniform line density over circumference C.

Kinetic tracking without ``Slice set`` uses Gaussian spatially averaged
coefficients. With ``Slice set``, execute the named :doc:`slicer` first.
Use ``Purpose="general"`` and ``Coordinate="z_rel"`` for bunched beams;
coasting beams also permit ``z_periodic``. The saved memberships and widths
are retained exactly. Regrouping invalidates them; another explicit Slicer
execution is required. All live particles must have a valid saved membership.
The weighting is the measured normalized line density divided by
:math:`1/(2\sqrt{\pi}\sigma_z)` for bunched beams or :math:`1/C` for coasting
beams. It does not turn the transverse Gaussian closure into a fully local
non-Gaussian model.

Binary collisions use a frozen local snapshot approximation
:math:`z_* = \gamma_0 z_{\mathrm{rel}}`, not exact reconstruction of equal-time
particle events. This adapter requires a near-reference, paraxial distribution
with narrow relative momentum spread, even when its shape is non-Gaussian.
The thermal-speed cap alone does not guarantee this at a very small reference
beta. Coasting positions are folded only
in a temporary collision array; stored ``z_rel`` is never wrapped. Cells span
the particle extent, with the full circumference used longitudinally for a
coasting beam. Explicit ``Grid bounds (m)`` replaces automatic bounds and
must contain all live particles; tails are never silently excluded. Its
longitudinal bounds for coasting beams must be exactly the laboratory interval
:math:`[-C/2,C/2]`, within relative tolerance :math:`10^{-12}`; accepted
roundoff is canonicalized to these exact endpoints. The command converts the
longitudinal bounds to rest-frame metres by multiplying by :math:`\gamma_0`.
Fixed bounds help separate mesh-resolution effects from changing sample extrema.
Empty or singleton cells cannot collide. Diagnostics report
occupancy and unpaired particles; a fine mesh with mostly singleton cells
does not represent resolved collision kinetics.

Coasting IBS requires ``Harmonic Number=1``: one bunch group must represent
the complete ring population. Bunched IBS processes bunches independently
and assumes their physical distributions do not overlap. It does not model
collisions between overlapping bunch groups.

Gaussian matching guard
--------------------------------------------------------------------------------

For every Gaussian IBS command evaluation, center the measured coordinates and define
dispersion-subtracted coordinates :math:`x_\beta=x-D_x\delta` and
:math:`p_{x\beta}=p_x-D_{p_x}\delta`, with the same construction in y.
Using the measured intrinsic emittances and supplied optics, form

.. math::

   u_x=\frac{x_\beta}{\sqrt{\varepsilon_x\beta_x}},\qquad
   u_{p_x}=\frac{\beta_xp_{x\beta}+\alpha_xx_\beta}
                   {\sqrt{\varepsilon_x\beta_x}},\qquad
   u_\delta=\frac{\delta}{\sigma_\delta},\qquad
   u_z=\frac{z-\langle z\rangle}{\sigma_z}.

The vector order is :math:`(u_x,u_{p_x},u_y,u_{p_y},u_\delta)` for coasting
beams; bunched beams append :math:`u_z`. The prescribed matched model has

.. math::

   C_{\mathrm{norm}}=\langle\mathbf u\mathbf u^T\rangle=I,
   \qquad e_{\mathrm{match}}=
   \max_{i,j}|(C_{\mathrm{norm}}-I)_{ij}|.

This check detects optics mismatch, unsupported betatron correlations and,
for bunched beams, correlations involving z, including momentum chirp.
It does not test normality, tails or higher-order moments. The default 0.1
is a covariance guard, not a 10% growth-rate accuracy guarantee. In a controlled
Gaussian-envelope perturbation check, covariance errors of 0.01, 0.02, 0.05
and 0.09 produced relative growth-rate-vector errors of 2.21%, 4.33%, 10.26%
and 17.22%, respectively. These are case-specific calibration results, not
universal error bounds; coupled cases can also hide tensor errors in projected
rates.

For quantitative work, 0.01--0.02 is a starting tolerance only when sufficient
macroparticles are available and sampling and threshold convergence have been
checked. Finite-sample covariance noise can itself trigger rejection. Increase
particle count when noise dominates instead of loosening the threshold to hide
a physical mismatch. Persistent mismatch or coupling calls for the binary
model with its own mesh, particle-count and time-step convergence checks.

``kinetic`` checks matching before every substep and on the final staged
result. Failure rejects the command without committing particle or random-stream
updates. ``bjorken_mtingwa`` retains rate diagnostics and reports matching plus
``model_moments_matched``. If false, those rates describe the prescribed Gaussian model;
they must not be interpreted as the measured distribution's physical growth rates.
The standalone coefficient API receives beam parameters and optics only;
it cannot inspect particles or perform this matching check.

Growth-rate and collision conventions
--------------------------------------------------------------------------------

The standalone API in ``PASS.commands.ibs`` provides
``IBSBeamParameters``, ``IBSOptics``, ``compute_local_coefficients``
(``compute_bjorken_mtingwa`` is an alias), and
``ring_average_growth_rates``. It needs no Simulation or particle pool.
Ring averaging takes explicit nonnegative segment-length or residence-time
weights; it does not invent a closing lattice segment.

The Gaussian matrix construction uses momentum order
:math:`(P_x/P_0,P_y/P_0,\delta/\gamma_0)`. Let
:math:`\mathbf e_x,\mathbf e_y,\mathbf e_z` be its unit basis vectors.
For transverse plane :math:`u\in\{x,y\}`, define

.. math::

   \phi_u=D_{p_u}+\frac{\alpha_u D_u}{\beta_u},
   \qquad
   L_u=\frac{\beta_u}{\varepsilon_u}
       (\mathbf e_u-\gamma_0\phi_u\mathbf e_z)
       (\mathbf e_u-\gamma_0\phi_u\mathbf e_z)^T
       +\frac{\gamma_0^2D_u^2}{\beta_u\varepsilon_u}
        \mathbf e_z\mathbf e_z^T,

   L_z=\frac{\gamma_0^2}{\sigma_\delta^2}\mathbf e_z\mathbf e_z^T,
   \qquad L=L_x+L_y+L_z,\qquad \Sigma=L^{-1}.

Here :math:`\Sigma` is the thermal momentum covariance conditional on
transverse position, not the projected whole-bunch momentum covariance.
The independently evaluated Bjorken--Mtingwa integral and normalization are

.. math::

   J=\int_0^\infty
      \frac{\sqrt{\lambda}\,(L+\lambda I)^{-1}}
           {\sqrt{\det(L+\lambda I)}}\,d\lambda,
   \qquad
   a=\frac{cNr_0^2\ln\Lambda}
          {8\pi\beta_0^3\gamma_0^4\varepsilon_x\varepsilon_y
           \sigma_z\sigma_\delta}.

The friction, diffusion and physical covariance derivative used by PASS are

.. math::

   F=2aJL,\qquad D=2a[\operatorname{tr}(J)I-J],
   \qquad K=D-F\Sigma-\Sigma F^T
           =2a[\operatorname{tr}(J)I-3J].

The conventional BM kernel is :math:`K/2`. Keeping this factor of two is
necessary when turning the rate integral into a particle diffusion process.
The trace :math:`\operatorname{tr}(K)=0` expresses leading-order thermal
energy conservation when the actual conditional covariance equals the matched
model covariance :math:`\Sigma`. With an unmatched covariance, the actual
OU covariance derivative can have nonzero trace. The local rates are

.. math::

   G_x=\tfrac12\operatorname{tr}(L_xK),\qquad
   G_y=\tfrac12\operatorname{tr}(L_yK),\qquad
   G_\delta=\frac{\gamma_0^2K_{zz}}{2\sigma_\delta^2}.

Rates are per laboratory second and mean

.. math::

   G_x=\frac{d\ln\varepsilon_x}{dt},\qquad
   G_y=\frac{d\ln\varepsilon_y}{dt},\qquad
   G_\delta=\frac{d\ln\sigma_\delta}{dt},\qquad
   G_{\delta^2}=2G_\delta.

Transverse amplitude rates are :math:`G_x/2` and :math:`G_y/2`.
The longitudinal rates describe an instantaneous momentum kick, without
synchrotron phase averaging. Subsequent RF and transport determine how that
kick is shared between bunch length and momentum spread. These conventions
must match before comparing another program's reported IBS times.

The kinetic model uses centered, conditional thermal residuals in
:math:`(P_x/P_0,P_y/P_0,\delta/\gamma_0)` and evolves

.. math::

   d\mathbf w=-F\mathbf w\,dt+B\,d\mathbf W,
   \qquad D=BB^T,\qquad
   \frac{d\Sigma}{dt}=D-F\Sigma-\Sigma F^T.

The conditional mean accounts for transverse position correlations and
dispersion. Diffusion is positive semidefinite; the net growth of a hot
degree of freedom may be negative. The implementation retains friction and
off-diagonal tensor contributions. A frozen-coefficient Ornstein--Uhlenbeck
step is evaluated exactly; adaptive positive substeps update the moments.
The finite-sample centroid correction preserves mean normalized momenta.
For a distribution with the matched conditional model covariance,
leading-order rest-frame thermal energy is conserved in the instantaneous
Gaussian collision expectation;
finite time steps and finite particle samples introduce errors that require
convergence checks. This closure does not enforce pairwise energy conservation.

The binary method uses equal-weight random local partners and the
Takizuka--Abe small-angle scattering law. With reduced mass :math:`\mu=M/2`
and relative speed g, the half-angle tangent has variance

.. math::

   \left\langle\tan^2(\theta/2)\right\rangle
   =\frac{q^4 n\ln\Lambda\,\Delta t_*}
          {8\pi\epsilon_0^2\mu^2 g^3}.

An isotropic azimuth rotates the relative momentum in the pair centre-of-momentum
frame; pair four-momentum is preserved up to rounding. The scattering
frequency still uses the nonrelativistic thermal approximation. One randomly
selected particle sits out in an odd cell, and paired exposure is corrected
in expectation. ``Collision steps`` divides the physical exposure
and resamples partners at each division, while keeping positions fixed.
Internal angular substeps subdivide each selected pair's rotation and do not
resample partners. Check convergence in both physical collision steps and
the outer command interval; neither refinement adds transport inside IBS.

Diagnostics and reproducibility
--------------------------------------------------------------------------------

``IBS.last_diagnostics`` exposes the most recent execution. With saving enabled,
``Save turns=[]`` saves every execution. For example,
``save_turns=[[0], [10, 50, 10]]`` selects turns 0, 10, 20, 30, 40 and 50;
``save_turns=[10, 50, 10]`` is the flat shorthand for the single range.
Endpoints are inclusive and turns are zero-based. ``Save diagnostics=False``
suppresses files regardless of the selection. Saving cadence does not change
collision exposure, coefficient updates, random draws or ``last_diagnostics``.
JSON records are placed under
``ibs/<unique directory>/beam<id>_<command>/turn_<turn>_call_<call>_<unique suffix>.json``
in the run output directory. Active Gaussian
records include measured beam parameters and growth rates. BM reports
``matching`` and ``model_moments_matched``; kinetic reports ``initial_matching``,
``final_matching`` and ``max_matching_error`` (the maximum over all pre-substep
checks and the final check). A matching record contains
``dimension``, ``normalized_covariance``, ``max_abs_error``,
``tolerance`` and ``matched``.

Each diagnostic record includes top-level ``interaction_length_m`` and
``coulomb_log_note``. Each bunch record includes ``dt_lab_s`` and
``interaction_length_over_circumference`` (null only for a nonpositive
circumference). These values expose the prescribed local exposure; they
do not infer lattice coverage or justify the physical Coulomb-log cutoffs.

Active binary records include ``grid_shape``, ``grid_bounds_m`` (rest-frame metres),
``n_grid_cells``, ``n_empty_cells``, ``singleton_fraction`` and
``occupancy_histogram`` entries ``{"particles": count, "cells": count}``,
including empty cells. ``n_pairs`` counts eligible pairs per physical step;
``n_pairings`` sums selected pairs over all completed ``n_collision_steps``.
``singleton_fraction`` counts particles in singleton cells divided by the
particle count; ``n_unpaired_particles`` counts omissions per physical step,
not unique identities across steps.
``n_substeps`` is the maximum angular subdivision count, ``max_scattering``
is the maximum variance per physical collision step before angular subdivision
(so this diagnostic can exceed the per-substep input bound), and
``n_zero_relative_pairs`` sums pairs with zero relative motion.
``proper_speed_max_over_c`` is the maximum proper speed divided by c over
the initial state and the result of every physical collision step.
Diagnostics describe
the selected model and are not a substitute for time-step, grid and particle
number convergence.

Random streams belong to a particular beam, command and bunch. A fixed seed
provides reproducible initialization under the same execution conditions.
``state_dict()`` and ``load_state_dict()`` save and restore IBS random-stream
state and execution counters. These component APIs require matching particles,
reference state, turn, slicing and other collective state; they do not provide
a complete simulation restart. The tracking entry starts a new run from turn 0.
CPU/GPU floating-point arithmetic may produce different trajectories; validate
statistical agreement between backends.
Use ``Particle Precision="float64"`` when resolving small IBS momentum
increments; float32 storage can round small updates away.
The diagnostic flags, turn selection and Coulomb-log note are excluded from
the physics configuration hash, so changing those output/provenance settings
does not invalidate an otherwise matching IBS component checkpoint.

References
--------------------------------------------------------------------------------

* J. D. Bjorken and S. K. Mtingwa, *Intrabeam Scattering*, Particle Accelerators
  **13** (1983), 115--143: Gaussian IBS theory.
* M. Zampetakis et al., `Interplay of space charge and intrabeam scattering in
  the LHC ion injector chain <https://arxiv.org/abs/2310.03504>`_: Gaussian
  matrix coefficients and friction/diffusion context. PASS uses its own full
  tensor closure and states its instantaneous growth-rate convention above.
* T. Takizuka and H. Abe, `A binary collision model for plasma simulation with
  a particle code <https://doi.org/10.1016/0021-9991(77)90099-7>`_, Journal of
  Computational Physics **25** (1977), 205--219: random binary collision law.
