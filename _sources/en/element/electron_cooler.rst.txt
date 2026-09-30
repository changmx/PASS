ElectronCooler
==============

``ElectronCooler`` transports ions through a finite cooling section and applies
incoherent electron--ion collisions from a prescribed electron reservoir. DC
electron beams and Gaussian electron bunches can have uniform round or Gaussian
transverse density. NumPy and CuPy particle arrays use the same physics.
``ElectronBeamConfig`` and ``ElectronCoolerItem`` are exported by
``PASS.para.schema`` and ``PASS.para.api``; no top-level named configuration is needed.

The electron reservoir does not evolve in response to the ions. Its parameters
do not simulate cathode emission, self-consistent electron optics, collective
screening dynamics, recombination, coherent electron cooling or depletion.

Collision models
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 33 45

   * - ``Model``
     - Operation
     - Assumptions and limits
   * - ``gaussian``
     - Nonmagnetized Gaussian Landau friction and momentum diffusion.
     - Any symmetric positive-definite 3 by 3 local electron velocity covariance.
       The name describes velocity distribution, not spatial profile. The
       collision operator omits gyromotion even if a solenoid field is configured.
   * - ``parkhomchuk``
     - Empirical magnetized friction.
     - Requires nonzero magnetic field, zero electron-axis angles and
       ``Diffusion=False``. Suitable for
       calibrated friction/rate studies; it supplies no equilibrium diffusion law.
   * - ``magnetized_collision``
     - Finite-window magnetic-response friction and diffusion.
     - A numerical reference model for weak response, weak deflection and a
       locally uniform gyrotropic reservoir. Requires explicit impact cutoffs,
       positive axial field, zero electron-axis angles, equal transverse thermal
       variances and zero cross-covariances.

The magnetic-response backend is not an all-regime model for a dense practical
cooler. It rejects out-of-domain or unconverged calculations rather than falling
back to another formula. Its bounds include :math:`\omega_pT_*\leq0.3`,
:math:`\omega_iT_*\leq0.1` and an ultraviolet-cutoff deflection measure no larger
than 0.1. Here :math:`T_*` is the physical full-section interaction time,
:math:`\omega_p` is the electron plasma frequency and
:math:`\omega_i=|Z|eB/M` is the ion cyclotron frequency. Finite-window coefficients
are applied as local Markov kick coefficients: ion momentum and the local
reservoir must vary little during that window. The implementation does not
resolve all temporal correlations between successive kicks. The coefficients
describe a single finite interaction with an initially uncorrelated, locally
uniform fresh reservoir, including transient polarization. They do not assert
finite-window Maxwell detailed balance or a general screened-plasma equilibrium.

The largest electron thermal RMS speed, encountered local mean velocity shear,
and participating ion--electron relative speeds must not exceed :math:`0.05c`.
Relativistic laboratory reference motion is allowed; relativistic thermal
collisions are not implemented. The Gaussian RMS criterion is an approximation
check, not a truncation of the Gaussian's mathematical tails.

Python example
--------------

Add this section to a sequence with injection and surrounding lattice transport.
The electron energy is independent of the ion reference and does not follow its
centroid. Velocity matching requires :math:`E_{k,e}=(\gamma_i-1)m_ec^2`.

.. code-block:: python

   from PASS.para.schema import ElectronBeamConfig, ElectronCoolerItem

   electrons = ElectronBeamConfig(
       kinetic_energy=1000.0, profile="uniform_round", radius=0.01,
       mode="dc", current=0.1,
       temperature_transverse=0.1, temperature_longitudinal=0.001,
   )
   sequence.add(
       "cooler",
       ElectronCoolerItem(
           s=12.0, length=2.0, electron_beam=electrons,
           model="gaussian", diffusion=True, coulomb_log=10.0,
           num_slices=8, random_seed=2026, save_diagnostics=True,
           save_turns=[[0, 1000, 10]],
       ),
   )

This element occupies [10, 12] m; do not also transport that interval with a
separate Drift or Solenoid. ``S (m)`` is its exit position.

Element interface
-----------------

Aliases are case-insensitive. Unknown fields and duplicate aliases are rejected.
Inputs must be finite. Integer controls and booleans do not accept strings or
coercible floating values. Standard ``Order`` and aperture fields are inherited
from ``ElementBase``; see :doc:`../input_generation`.

.. list-table::
   :header-rows: 1
   :widths: 25 29 15 31

   * - Python field
     - JSON key
     - Default
     - Meaning
   * - ``s``, ``length``
     - ``S (m)``, ``Length (m)``
     - Required; 0
     - Exit position and nonnegative physical section length.
   * - ``electron_beam``
     - ``Electron beam``
     - Required
     - One ``ElectronBeamConfig``, described below.
   * - ``model``
     - ``Model``
     - ``gaussian``
     - One of the three models above.
   * - ``collisions``, ``diffusion``
     - ``Collisions``, ``Diffusion``
     - True, True
     - Disable collisional drift and noise together, or only noise. Transport
       and optional smooth fields remain active.
   * - ``magnetic_field``, ``mean_space_charge``
     - ``Magnetic field (T)``, ``Mean space charge``
     - 0, False
     - Signed uniform axial field; optional prescribed electron smooth field.
   * - ``coulomb_log``
     - ``Coulomb log``
     - None
     - Positive fixed Gaussian/Parkhomchuk logarithm; omission selects effective
       cutoffs. Not allowed for magnetic response.
   * - ``min_impact_parameter``, ``max_impact_parameter``
     - ``Min impact parameter (m)``, ``Max impact parameter (m)``
     - None, None
     - Magnetic response requires explicit :math:`0<b_{\min}<b_{\max}`. The maximum
       can cap automatic Gaussian/Parkhomchuk cutoffs, but not a fixed logarithm.
   * - ``effective_velocity_spread``
     - ``Effective velocity spread (m/s)``
     - 0
     - Extra Parkhomchuk RMS speed from unresolved field errors or drift.
   * - ``num_slices``
     - ``Num slices``
     - 1
     - Positive number of transport/collision sections.
   * - ``max_fractional_step``, ``max_substeps``
     - ``Max fractional step``, ``Max substeps``
     - 0.05, 1000
     - Adaptive collision drift/noise step control and maximum internal
       collision substeps per transport section.
   * - ``quadrature_order``
     - ``Quadrature order``
     - 64
     - Gaussian velocity-integral order per integration interval, at least 8.
   * - ``radial_order``, ``polar_order``, ``azimuthal_order``, ``time_order``
     - ``Radial order``, ``Polar order``, ``Azimuthal order``, ``Time order``
     - 16, 12, 16, 64
     - Initial magnetic-response quadrature orders; refinement increases all four.
   * - ``quadrature_rtol``, ``max_refinements``
     - ``Quadrature relative tolerance``, ``Max refinements``
     - 0.02, 2
     - Magnetic-response convergence controls. Tolerance is in [1e-6, 0.1];
       unconverged coefficients raise an error.
   * - ``random_seed``
     - ``Random Seed``
     - None
     - Nonnegative strict integer, or nondeterministic entropy. JSON keeps null.
   * - ``save_diagnostics``, ``save_turns``
     - ``Save diagnostics``, ``Save turns``
     - False, []
     - Empty selection means every turn; otherwise [turn] or inclusive
       [start, end, step] entries.

Numerical controls belonging only to an inactive model cannot be changed from
their defaults. Ion self-space-charge and IBS are separate explicit commands;
avoid duplicate physical exposure when composing the lattice.

Electron reservoir interface
----------------------------

Widths describe the entrance. Optional exit widths prescribe linear interpolation
of the width itself; omission makes it constant. This is a prescribed envelope,
not a self-consistent electron transport or continuity solution.

.. list-table::
   :header-rows: 1
   :widths: 25 30 14 31

   * - Python field
     - JSON key
     - Default
     - Meaning
   * - ``kinetic_energy``
     - ``Kinetic energy (eV)``
     - Required
     - Positive energy per electron.
   * - ``profile``
     - ``Profile``
     - ``uniform_round``
     - ``uniform_round`` or ``gaussian`` transverse spatial density.
   * - ``radius``, ``radius_exit``
     - ``Radius (m)``, ``Exit radius (m)``
     - None
     - Positive hard radius for uniform round density; entrance radius required.
   * - ``sigma_x``, ``sigma_y``, ``sigma_x_exit``, ``sigma_y_exit``
     - ``Sigma x (m)``, ``Sigma y (m)``, ``Exit sigma x (m)``, ``Exit sigma y (m)``
     - None
     - Positive Gaussian RMS sizes; both entrance sizes required.
   * - ``mode``, ``current``
     - ``Mode``, ``Current (A)``
     - ``dc``, None
     - DC mode requires a nonnegative electron-current magnitude.
   * - ``bunch_charge``, ``sigma_time``
     - ``Bunch charge (C)``, ``Sigma time (s)``
     - None
     - Mode ``gaussian_bunch`` requires nonnegative charge magnitude and positive
       RMS duration; do not also supply current.
   * - ``bunch_center_time``, ``repetition_frequency``
     - ``Bunch center time (s)``, ``Repetition frequency (Hz)``
     - 0, None
     - Pulse centre at the entrance and optional positive repetition frequency.
       Repetition defines an infinite train; centre time is a phase origin,
       not turn-on. Without repetition there is a single pulse.
   * - ``center_x``, ``center_y``, ``angle_x``, ``angle_y``
     - ``Center x (m)``, ``Center y (m)``, ``Angle x (rad)``, ``Angle y (rad)``
     - 0
     - Entrance offset and direction of the electron axis. The local basis
       rotates with this axis; the ion reference does not follow it.
       Both magnetized collision models require zero angles; use the Parkhomchuk
       effective velocity spread for unresolved field-line imperfections.
   * - ``temperature_transverse``, ``temperature_longitudinal``
     - ``Transverse temperature (eV)``, ``Longitudinal temperature (eV)``
     - None
     - Positive rest-frame :math:`k_BT` in eV, transverse per Cartesian component.
       Supply both temperatures or a covariance.
   * - ``velocity_covariance``
     - ``Velocity covariance (m2/s2)``
     - None
     - Symmetric positive-definite conditional 3 by 3 covariance, in the
       electron-axis basis and mean electron rest frame.
   * - ``velocity_gradient``
     - ``Velocity gradient (1/s)``
     - None
     - Finite 3 by 2 matrix :math:`G` defining local mean velocity
       :math:`\bar{\mathbf u}_e=G(x_e,y_e)^T`; zero when omitted. Magnetic response
       permits only its longitudinal row.

Temperature inputs give :math:`C=\mathrm{diag}(T_\perp,T_\perp,T_\parallel)e/m_e`.
With shear, :math:`C` is conditioned on local position, not the covariance after
averaging over the whole electron beam.
DC proper density is

.. math::

   n_*=\frac{I}{e\beta_e c\gamma_e}g_\perp(x_e,y_e;s),
   \qquad \int g_\perp\,dx_e\,dy_e=1.

For a bunch replace :math:`I` by :math:`Q_bh(t)`, where :math:`h` is its normalized
Gaussian arrival-time profile. Repeated profiles include pulse overlap.
Current and charge inputs are positive magnitudes; smooth fields use negative
electron charge.

Collision equations
-------------------

Let :math:`M` be complete ion mass, :math:`g=|Z|e^2/(4\pi\epsilon_0)`, and
:math:`\mathbf w=\mathbf v-\mathbf u`. Averages below use the normalized local
electron velocity distribution. The nonmagnetized coefficients are

.. math::

   \mathbf F_*=-4\pi n_*g^2\ln\Lambda
   \left(\frac1{m_e}+\frac1M\right)
   \left\langle\frac{\mathbf w}{|\mathbf w|^3}\right\rangle,\qquad
   Q_*=4\pi n_*g^2\ln\Lambda
   \left\langle\frac{\mathsf I-\hat{\mathbf w}\hat{\mathbf w}^{T}}
   {|\mathbf w|}\right\rangle.

Gaussian Rosenbluth convolutions are evaluated by one-dimensional quadrature
partitioned geometrically around the thermal covariance eigenvalue scales.
:math:`\mathbf F_*` is in N;
:math:`Q_*=d\,\mathrm{cov}(\Delta\mathbf p_*)/dt_*` is in
:math:`\mathrm{kg^2\,m^2\,s^{-3}}`. The stochastic increment is

.. math::

   \Delta\mathbf p_*=\mathbf F_*\Delta t_*+
   B_*\sqrt{\Delta t_*}\,\boldsymbol\xi,\qquad
   B_*B_*^T=Q_*,\quad \boldsymbol\xi\sim\mathcal N(0,\mathsf I).

There is no extra factor of two. Ion recoil is retained. The mean kick is not
removed: energy mismatch and misalignment can drag the ion centroid.
The collision update is Itô Euler--Maruyama (weak order 1, strong order 1/2);
drift-only collision stepping is also first order. Symmetric transport half maps
do not raise the collision integrator's order.
Isotropic-bath equilibrium verification uses a fixed Coulomb logarithm.
An automatic logarithm is an effective local approximation held outside the
microscopic electron-velocity integral; it does not establish exact detailed
balance for a velocity-dependent logarithm.

Gaussian automatic cutoffs are

.. math::

   u^2=v^2+\mathrm{tr}\,C,\quad
   b_{\min}=\max\left(\frac{g}{\mu u^2},\frac{\hbar}{2\mu u}\right),\quad
   b_{\max}=\min(a,uT_*,u/\omega_p),\quad
   \ln\Lambda=\ln(1+b_{\max}/b_{\min}),

where :math:`\mu=m_eM/(m_e+M)`, :math:`\omega_p^2=n_*e^2/(\epsilon_0m_e)`, and
:math:`a` is the local hard radius or smaller Gaussian RMS size.
``Max impact parameter (m)`` can impose a further upper cap.

Parkhomchuk uses

.. math::

   \mathbf F_*=-\frac{4n_*g^2}{m_e}
   \frac{\ln\Lambda\,\mathbf v}
   {(v^2+\sigma_\parallel^2+v_{\rm extra}^2)^{3/2}},\qquad
   \ln\Lambda=\ln\left(1+\frac{b_{\max}}{b_{90}+\rho_L}\right),

with :math:`u^2=v^2+\sigma_\parallel^2+v_{\rm extra}^2`,
:math:`b_{90}=g/(m_eu^2)`, and
:math:`\rho_L=\sqrt2m_e\sigma_\perp/(e|B|)`.
:math:`\sigma_\perp` is the RMS of one Cartesian component. This empirical
friction law does not define a diffusion tensor. For a full covariance input,
Parkhomchuk reduces it to :math:`\sigma_\parallel^2=C_{zz}` and
:math:`\sigma_\perp^2=(C_{xx}+C_{yy})/2` and omits its cross-correlations;
only the Gaussian model integrates the full anisotropic covariance.

For magnetic response set :math:`\Omega=eB/m_e`, :math:`\alpha=2g^2/\pi` and

.. math::

   C_k(t)=\exp\left[-\frac12\sigma_\parallel^2k_z^2t^2
     -\sigma_\perp^2k_\perp^2\frac{1-\cos\Omega t}{\Omega^2}\right],
   \qquad R_e=k_z^2t+k_\perp^2\frac{\sin\Omega t}{\Omega}.

Independent initially uniform electrons on unperturbed helices give

.. math::

   Q_* = 2n_*\alpha\int\frac{d^3k}{k^4}\mathbf k\mathbf k^T
     \int_0^{T_*}\left(1-\frac{t}{T_*}\right)
     C_k(t)\cos(\mathbf k\cdot\mathbf v t)\,dt,

   \mathbf F_*=-n_*\alpha\int\frac{d^3k}{k^4}\mathbf k
     \int_0^{T_*}\left(1-\frac{t}{T_*}\right)
     C_k(t)\sin(\mathbf k\cdot\mathbf v t)
     \left(\frac{R_e}{m_e}+\frac{k^2t}{M}\right)\,dt.

The Fourier convention is :math:`k_{\min}=1/b_{\max}`,
:math:`k_{\max}=1/b_{\min}`. Recoil is
:math:`\nabla_{\mathbf v}\cdot Q_*/(2M)`, not an assigned Einstein noise law.
The zero-field, long-window limit gives Landau coefficients with
:math:`\ln\Lambda=\ln(b_{\max}/b_{\min})`. The low-level coefficient function
supports that zero-field verification limit; the magnetic command configuration
requires positive field. Refinement doubles all four quadrature orders and
checks force/tensor relative error, tensor positivity and cyclotron phase
resolution. This controlled weak-response benchmark can be expensive.

Transport, timing and smooth fields
-----------------------------------

Each positive-length section uses half transport, a collision/mean-field kick,
then half transport. Zero field gives drift transport; nonzero axial field gives
the uniform-solenoid map. Dissipation and diffusion never use negative Yoshida
stages. Inside the solenoid, mechanical transverse momentum is
:math:`p_x^{\rm mech}/P_0=p_x+k_sy/2`,
:math:`p_y^{\rm mech}/P_0=p_y-k_sx/2`. Collision velocities use those quantities.

Boosts use relativistic mechanical momenta. PASS stores
:math:`dp=(|\mathbf P|-P_0)/P_0`, not longitudinal momentum deviation.
Complete-ion SI impulses are converted back to PASS's per-nucleon normalization.
The reference energy does not follow the cooling centroid. Transport advances
:math:`t_0` by :math:`L/(\beta_0c)` once; continuous :math:`z` changes through time
slip. At a physical node, :math:`t_i=t_{0,\rm node}-z_i/(\beta_0c)`.
Bunch grouping metadata does not enter this clock.

Full-section interaction time remains fixed when numerical resolution changes.
For parallel axes,
:math:`T_*=\gamma_eL(1-\beta_e\beta_0)/(\beta_0c)`.
This cutoff time follows the ideal ion reference worldline, whereas each kick's
elapsed time follows the particle worldline. The cutoff therefore assumes motion
near the reference; broad or strongly detuned beams require a transit-time cutoff
sensitivity study. Exact momentum boosts do not remove this approximation.
Substituting numerical section length into this cutoff would change the physics
with resolution, as discussed in the
`JSPEC thick-cooler study, IPAC 2023, TUPM030
<https://inspirehep.net/files/aab738e313248932b9fad111f6c1bf7e>`_.
Zero physical length leaves coordinates and clock unchanged.

``Mean space charge=True`` adds a quasistatic, locally uniform, long-beam
transverse electron field, separately from collisions. Rest-frame force and
boosts include laboratory magnetic cancellation; parallel equal-speed transverse
forces contain :math:`1-\beta_e\beta_i`. This differs from a stationary electron
cloud and from ion self-space-charge.

For bunched electrons this approximation requires
:math:`\gamma_e\beta_ec\sigma_t\geq10a_{\max}`, using the largest configured
entrance/exit radius or RMS width. End fields, conducting-pipe images,
longitudinal space-charge acceleration and general 3D finite-bunch fields are
not included. Electron kinetic energy is not adjusted for space-charge
voltage depression.

Diagnostics, restart and IBS
----------------------------

Selected calls append to one JSONL file per cooler under
``output/electron_cooler/beam<id>_<name>_<unique>.jsonl`` and update
``last_diagnostics``. Records contain model and frames, interaction length,
populations/losses, substeps, density/overlap, mean force, diffusion diagonal,
energy exchange and model-specific numerical checks. Density, overlap and
coefficient summaries describe the last interaction node; energy exchange and
substeps accumulate across the element. Total represented-beam energy exchange
includes macroparticle weight.

Random streams are separated by beam, command name and bunch. Particles are
gathered in stable tag order. NumPy random draws drive both backends, but
floating-point trajectories need not be bit-identical.
``state_dict()`` contains configuration identity, entropy, calls and RNG states.
This is not a complete checkpoint: restore matching particles, reference state
and execution boundary as well. Supported full joint checkpoints preserve
cooler and IBS random state together.

Cooling--IBS balance requires IBS ``kinetic`` or ``binary`` tracking;
``bjorken_mtingwa`` only reports rates. Establish convergence in lattice
sampling, collision steps, particles and random statistics before interpreting
equilibrium. Parkhomchuk alone does not predict an electron-collision diffusion
floor. See :doc:`../ibs`.

References
----------

* `Y. Derbenev, Theory of Electron Cooling, arXiv:1703.09735
  <https://arxiv.org/abs/1703.09735>`_: nonmagnetized force/diffusion and magnetic
  response. The finite-window average and recoil normalization above specify
  the numerical variant implemented here.
* `A. V. Fedotov et al., Numerical Studies of the Friction Force for the RHIC
  Electron Cooler, PAC 2005, TPAT092
  <https://proceedings.jacow.org/p05/PAPERS/TPAT092.PDF>`_: empirical magnetic
  friction and effective speed/cutoff conventions.
