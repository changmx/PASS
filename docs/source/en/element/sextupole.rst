Sextupole
=========

This module describes the PASS sextupole element **Sextupole**, used to simulate the motion of charged particles in a sextupole magnet. The sextupole is the most fundamental nonlinear element in accelerators, providing nonlinear focusing force through a quadratic magnetic field, primarily used for chromaticity correction and resonance driving.

The PASS sextupole supports both **thick element** (``length > 0``) and **thin lens** (``length = 0``) modes. The thick element uses the exact drift-kick-drift (DKD-exact) symplectic integration scheme, supporting both uniform (2nd-order) and yoshida4 (4th-order) symplectic integrators.

**Code Location**

- Source file: ``PASS/commands/element/sextupole.py``
- Class name: ``Sextupole`` (inherits from ``Command``)
- Registration name: ``sextupole``
- Key features:

  - Supports thin lens mode (``length = 0``, applies only a sextupole kick)
  - Supports thick lens mode (``length > 0``, DKD-exact symplectic integration)
  - Supports uniform (2nd-order leapfrog) and yoshida4 (4th-order Yoshida composition) integrators
  - Supports normal sextupole (``k2l``) and skew sextupole (``k2sl``) and their combinations
  - Zero field (``k2l = k2sl = 0``) automatically degenerates to a pure drift
  - Chromaticity correction, nonlinear dispersion, and other higher-order effects naturally introduced through exact drift
  - Supports aperture check


Coordinate Convention
---------------------

PASS uses normalized curvilinear coordinates consistent with Xsuite. The six-dimensional phase-space variables are :math:`(x, p_x, y, p_y, z, \delta)`:

.. list-table::
  :header-rows: 1
  :widths: 15 20 65

  * - Variable
    - Symbol
    - Definition
  * - ``x``
    - :math:`x`
    - Horizontal offset (relative to the reference orbit)
  * - ``px``
    - :math:`p_x`
    - Normalized horizontal momentum, :math:`p_x = P_x / P_0`
  * - ``y``
    - :math:`y`
    - Vertical offset
  * - ``py``
    - :math:`p_y`
    - Normalized vertical momentum, :math:`p_y = P_y / P_0`
  * - ``z``
    - :math:`\zeta`
    - Longitudinal coordinate, :math:`\zeta = s - \beta_0 c t`
  * - ``dp``
    - :math:`\delta`
    - Relative momentum deviation, :math:`\delta = P / P_0 - 1`

where :math:`P_0` is the reference particle momentum, :math:`\beta_0 = v_0 / c` is the reference particle normalized velocity, :math:`s` is the arc length along the reference orbit, and :math:`t` is time.

The longitudinal momentum component is defined as:

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

Charge-to-mass ratio factor:

.. math::

  \chi = \frac{q}{q_0} \cdot \frac{m_0}{m}

For a beam of identical particle species, :math:`\chi = 1`.


Sextupole Field and Normalized Strength
---------------------------------------

The magnetic field of a sextupole magnet has a quadratic distribution in the transverse plane. In complex notation:

.. math::

  B_y + i B_x = \frac{1}{2}(B'' + i B''_s)(x + i y)^2

where :math:`B''` is the normal sextupole field second derivative and :math:`B''_s` is the skew sextupole field second derivative. Expanding:

.. math::

  B_y = \frac{1}{2} B'' (x^2 - y^2) - B''_s x y

.. math::

  B_x = B'' x y + \frac{1}{2} B''_s (x^2 - y^2)

The normalized sextupole strength is defined as:

.. math::

  K_2 = \frac{q_0 B''}{2 P_0}

.. math::

  K_{2s} = \frac{q_0 B''_s}{2 P_0}

The integrated strength is:

.. math::

  K_{2L} = K_2 \cdot L, \qquad K_{2sL} = K_{2s} \cdot L

where :math:`L` is the magnet length. In PASS, the user directly specifies :math:`K_{2L}` (``k2l``) and :math:`K_{2sL}` (``k2sl``); for thick lenses, :math:`K_2 = K_{2L} / L` and :math:`K_{2s} = K_{2sL} / L` are solved internally.


Overall Tracking Flow
---------------------

Depending on the magnet length, the sextupole has two tracking modes:

**Thin lens mode** (:math:`L = 0`)

::

  ====== Thin lens (length = 0) ======

  Single sextupole kick Kick(K2L, K2sL)
  [Position unchanged, momentum jump only]

**Thick lens mode** (:math:`L > 0`)

::

  ====== Thick lens (length > 0) ======

  Slice 1 → Slice 2 → ... → Slice N
  (Each slice: Drift(ds/2) → Kick(ds) → Drift(ds/2))

  where ds = L / N

  If K2L = 0 and K2sL = 0: degenerates to a single exact drift Drift(L)

The complete map is:

Thin lens:

.. math::

  \mathcal{M}_{\text{thin}} = \text{Kick}(K_{2L}, K_{2sL})

Thick lens (N slices):

.. math::

  \mathcal{M}_{\text{thick}} = \left[\mathcal{M}_{\text{DKD}}(\Delta s)\right]^N

where the DKD map for each slice is:

.. math::

  \mathcal{M}_{\text{DKD}}(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

.. note::

  - Thin lens mode does not change the particle position coordinates :math:`(x, y, z)`, only applies momentum kicks
  - Chromaticity effects in thick lens mode are naturally introduced through the :math:`p_z` expression in exact drift (see chromaticity correction section)
  - When :math:`K_{2L} = 0` and :math:`K_{2sL} = 0`, the thick lens degenerates to a pure drift, avoiding meaningless empty kick loops


Physical Derivation
--------------------

Hamiltonian
~~~~~~~~~~~

In the Cartesian coordinate system (sextupole has no curvature, :math:`h = 0`), the sextupole Hamiltonian is:

.. math::

  H_{\text{sext}} = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2} + \frac{\chi}{6}\left[K_2(x^3 - 3 x y^2) + K_{2s}(3 x^2 y - y^3)\right]

Splitting it into the propagation part (exact drift :math:`H_D`) and the kick part (:math:`H_K`):

.. math::

  H_D = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  H_K = \frac{\chi}{6}\left[K_2(x^3 - 3 x y^2) + K_{2s}(3 x^2 y - y^3)\right]

where :math:`H_D` is the exact drift Hamiltonian (preserving the :math:`p_z` square root without small-momentum expansion), and :math:`H_K` is the sextupole kick. This is the standard **split-operator** method: the Hamiltonian is split into analytically solvable parts, maps are applied separately, and then combined into a symplectic integrator.

Exact Drift Map D
~~~~~~~~~~~~~~~~~

The Hamilton's equations of the propagation part give the exact drift:

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  x \leftarrow x + \frac{p_x}{p_z} \cdot L_D

.. math::

  y \leftarrow y + \frac{p_y}{p_z} \cdot L_D

.. math::

  \zeta \leftarrow \zeta + L_D \cdot \left(1 - \frac{\beta_0}{\beta} \cdot \frac{1+\delta}{p_z}\right)

where :math:`L_D` is the drift length, and :math:`\beta` is the particle's actual normalized velocity:

.. math::

  \beta = \frac{(1+\delta) \, \beta_0 \gamma_0}{\sqrt{1 + \left[(1+\delta) \, \beta_0 \gamma_0\right]^2}}

.. note::

  The meaning of "exact": the drift part preserves the exact square root :math:`p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}` without small-momentum expansion :math:`p_x \ll 1`. The approximation lies only in separating the propagation part from the kick part (split-operator method). This formula is identical to the exact drift in the Drift element and the Quadrupole element.

Sextupole Kick Map K
~~~~~~~~~~~~~~~~~~~~

The kick part is a thin lens map (position unchanged, momentum jump only). From Hamilton's equations :math:`\dot{p}_x = -\partial H / \partial x`, :math:`\dot{p}_y = -\partial H / \partial y`:

.. math::

  \Delta p_x = -\frac{\chi}{2} K_{2L} (x^2 - y^2) + \chi K_{2sL} \, x y

.. math::

  \Delta p_y = \chi K_{2L} \, x y + \frac{\chi}{2} K_{2sL} (x^2 - y^2)

where :math:`L_K` is the kick effective length.

Physical meaning of each term:

.. list-table::
  :header-rows: 1
  :widths: 30 15 55

  * - Term
    - Source
    - Physical Meaning
  * - :math:`-\frac{\chi}{2} K_{2L} (x^2 - y^2)`
    - :math:`\frac{\chi K_2}{6} x^3`
    - Horizontal nonlinear focusing (proportional to :math:`x^2`)
  * - :math:`+\chi K_{2L} \, xy`
    - :math:`-\frac{\chi K_2}{2} x y^2`
    - Horizontal-vertical coupling kick
  * - :math:`+\chi K_{2sL} \, xy`
    - :math:`\frac{\chi K_{2s}}{2} x^2 y`
    - Skew sextupole horizontal coupling kick
  * - :math:`+\frac{\chi}{2} K_{2sL} (x^2 - y^2)`
    - :math:`-\frac{\chi K_{2s}}{6} y^3`
    - Skew sextupole vertical nonlinear focusing

For thin lens mode, :math:`L_K = 1`, using the integrated strengths :math:`K_{2L}` and :math:`K_{2sL}` directly. For DKD mode, :math:`L_K = \Delta s`, using :math:`K_2 \Delta s` and :math:`K_{2s} \Delta s`.

.. note::

  A normal sextupole (:math:`K_2 > 0`) provides a restoring force proportional to :math:`x^2` for particles with positive offset in the horizontal direction, and the opposite in the vertical direction. The sextupole focusing force is proportional to the square of the position, making it a nonlinear element—particles farther from the axis experience stronger deflection.

  A skew sextupole (:math:`K_{2s} \neq 0`) rotates the sextupole action by :math:`\pi / 6`, producing a different :math:`x`-:math:`y` coupling pattern. In practice, it is often used to simulate installation rotation errors or drive specific resonances.

  Comparison with the quadrupole: the quadrupole kick depends linearly on :math:`x` (:math:`\Delta p_x \propto x`), while the sextupole kick depends quadratically on :math:`x` (:math:`\Delta p_x \propto x^2`). This means the sextupole does not affect particles on the reference orbit (kick is zero when :math:`x = y = 0`), but produces nonlinear deflection for particles deviating from the axis.


Uniform Integrator (2nd-order symplectic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each slice uses the drift-kick-drift (DKD) structure, i.e., 2nd-order leapfrog:

.. math::

  S_2(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

The per-slice error is :math:`O(\Delta s^3)`, and the global error is :math:`O(\Delta s^2)`. A 2nd-order symplectic integrator where every step is a canonical transformation.


yoshida4 Integrator (4th-order symplectic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A 4th-order symplectic map is constructed by composing three 2nd-order DKD steps [Yoshida 1990]:

.. math::

  S_4(\Delta s) = S_2(z_1 \Delta s) \circ S_2(z_0 \Delta s) \circ S_2(z_1 \Delta s)

where the Yoshida coefficients are:

.. math::

  z_1 = \frac{1}{2 - 2^{1/3}} \approx 1.3512

.. math::

  z_0 = 1 - 2 z_1 \approx -1.7024

.. note::

  :math:`z_0 < 0` means the middle step is a backward tracking (the drift and kick "lengths" are negative). This is a mathematical requirement of the Yoshida composition method and is fully self-consistent in the symplectic map group. The per-slice error is :math:`O(\Delta s^5)`, and the global error is :math:`O(\Delta s^4)`.


Chromaticity Correction
-----------------------

Chromaticity describes the dependence of the particle tune on the momentum deviation :math:`\delta`. The sextupole is the core element for chromaticity correction.

Physical Mechanism
~~~~~~~~~~~~~~~~~~

The transverse position of a particle at the sextupole includes a dispersive component:

.. math::

  x = x_\beta + \eta_x \, \delta

where :math:`x_\beta` is the betatron oscillation part and :math:`\eta_x` is the dispersion function. Substituting into the sextupole kick:

.. math::

  \Delta p_x = -\frac{\chi}{2} K_{2L} (x_\beta + \eta_x \delta)^2

Expanding:

.. math::

  \Delta p_x = -\frac{\chi}{2} K_{2L} \, x_\beta^2 \;-\; \chi K_{2L} \, \eta_x \, \delta \, x_\beta \;-\; \frac{\chi}{2} K_{2L} \, \eta_x^2 \, \delta^2

The second term :math:`-\chi K_{2L} \eta_x \delta \, x_\beta` is an equivalent quadrupole kick (linearly dependent on :math:`x_\beta` with a coefficient proportional to :math:`\delta`), which changes the tune dependence on :math:`\delta`, thereby achieving chromaticity correction. At a sextupole with dispersion, the equivalent quadrupole strength is:

.. math::

  K_{1,\text{eff}} = -K_2 \, \eta_x

The corresponding chromaticity contribution is:

.. math::

  \Delta Q'_x = \frac{1}{4\pi} \oint \beta_x K_{1,\text{eff}} \, ds = -\frac{1}{4\pi} \oint \beta_x K_2 \, \eta_x \, ds

.. note::

  - The sextupole can only correct chromaticity at locations with dispersion (:math:`\eta_x \neq 0`)
  - Chromaticity correction arises automatically in the kick—the kick acts on the true coordinate :math:`x` (including dispersion), without any expansion
  - Even a thin lens (no drift) has a chromaticity correction effect
  - The third term :math:`-\frac{\chi}{2} K_{2L} \eta_x^2 \delta^2` is a second-order dispersion driving term, also naturally included
  - At dispersion-free locations (:math:`\eta_x = 0`), the sextupole does not correct first-order chromaticity but still retains nonlinear effects (3rd-order resonance driving, nonlinear coupling, dynamic aperture limitation, etc.)


Using Sextupoles in Twiss Linear Transport
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PASS's Twiss transport (``twiss.py``) operates in :math:`(x, p_x)` normalized momentum coordinates, with dispersion handled as "subtract → linear transport → add back". Natural chromaticity is introduced through the ``DQx`` / ``DQy`` parameters (:math:`\delta` terms in the phase advance). When inserting a sextupole kick in this framework, the following should be noted.

**Coordinate Consistency**

When Twiss transport reaches the sextupole position, the particle's :math:`x` already includes dispersion (:math:`x = x_\beta + \eta_x \delta`). The sextupole kick acts directly on this true coordinate, and the chromaticity correction term :math:`-K_{2L}\eta_x\delta\cdot x_\beta` appears automatically. **The kick should not be divided by** :math:`1+\delta`—that is the notation for the :math:`(x, x')` angular coordinate system, which is incompatible with PASS's :math:`(x, p_x)` system. Mixing them would lead to double-counting of chromaticity.

**Avoiding Chromaticity Double-Counting**

.. list-table::
  :header-rows: 1
  :widths: 30 70

  * - Scenario
    - Correct Approach
  * - ``DQx`` contains total chromaticity (including sextupole contribution)
    - Do not apply an additional sextupole kick, otherwise first-order chromaticity is double-counted
  * - ``DQx`` contains only natural chromaticity (excluding sextupole)
    - Apply the sextupole kick to supplement chromaticity correction and nonlinear effects, no conflict
  * - ``DQx`` contains total chromaticity, but nonlinear effects still need to be simulated
    - Subtract the sextupole chromaticity contribution from ``DQx`` (:math:`\Delta Q'_x = -\frac{1}{4\pi}\oint \beta_x K_2 \eta_x \, ds`), then apply the full sextupole kick

**Differences Between Thin Lens and Thick Lens**

.. list-table::
  :header-rows: 1
  :widths: 25 20 55

  * - Effect
    - Thin Lens
    - Thick Lens DKD-exact
  * - Chromaticity correction (via dispersive location)
    - Yes
    - Yes
  * - In-element drift dispersion
    - No
    - Yes
  * - Thick-lens distribution effects
    - No
    - Yes
  * - Path-length effects (:math:`R_{56}`, etc.)
    - No
    - Yes

The effects missing from the thin lens arise from "internal drift within the magnet"—a zero-length magnet physically has no internal drift, which is a correct physical approximation, not an omission. If these effects are needed, use thick lens mode.

.. note::

  - In element-by-element tracking mode, there is no ``DQx`` double-counting issue—all effects are naturally produced by the DKD-exact physics simulation
  - Twiss linear transport is a first-order model; dividing by :math:`1+\delta` in the sextupole kick would introduce second-order nonlinear dispersion effects inconsistent with the model's precision, and should be avoided
  - If the sextupole strength is large or precise nonlinear effect simulation is needed, it is recommended to switch to full element-by-element DKD-exact tracking rather than locally introducing nonlinear kicks in the Twiss linear framework


Naturally Included Higher-Order Effects
---------------------------------------

In the DKD-exact scheme, all nonlinear effects of an ideal sextupole magnet are naturally included without any additional treatment:

.. list-table::
  :header-rows: 1
  :widths: 30 70

  * - Effect
    - Source
  * - Chromaticity correction
    - Kick acts on the true coordinate :math:`x` containing dispersion; expansion automatically produces the equivalent quadrupole term
  * - Natural chromaticity
    - Exact :math:`p_z` in drift makes the equivalent focusing strength contain :math:`1/(1+\delta)` dependence
  * - Higher-order dispersion
    - Exact :math:`p_z` in drift preserves the full square root; dispersion evolution contains all orders of :math:`\delta` dependence
  * - Path-length effects (:math:`R_{56}`, etc.)
    - :math:`\zeta` update in drift contains the complete :math:`R_{56}`, :math:`T_{566}`, and higher-order terms
  * - Thick-lens distribution effects
    - In DKD multi-slice, drift changes :math:`x`, and subsequent kicks act on updated coordinates
  * - :math:`x`-:math:`y` coupling
    - :math:`xy` cross terms in the kick

.. note::

  The only approximation is the discretization error of the split-operator integrator (:math:`O(\Delta s^2)` for uniform, :math:`O(\Delta s^4)` for yoshida4), which can be controlled by increasing the number of slices. This is a truncation error of the mathematical method, not an omission of physical effects.


Interface Parameters
--------------------

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 35

  * - Property
    - JSON key
    - Type
    - Unit
    - Description
  * - ``s``
    - ``s (m)``
    - float
    - m
    - Longitudinal position of the element in the beamline
  * - ``length``
    - ``length (m)``
    - float
    - m
    - Element length (must be :math:`\ge 0`; :math:`= 0` for thin lens)
  * - ``name``
    - ``name``
    - str
    - -
    - Element name
  * - ``k2l``
    - ``k2l``
    - float
    - :math:`\text{m}^{-2}`
    - Normal sextupole integrated strength :math:`K_{2L}`, default 0
  * - ``k2sl``
    - ``k2sl``
    - float
    - :math:`\text{m}^{-2}`
    - Skew sextupole integrated strength :math:`K_{2sL}`, default 0
  * - ``num_slice``
    - ``num slices``
    - int
    - -
    - Number of slices, default 1 (effective only for thick lens)
  * - ``integrator``
    - ``integrator``
    - str
    - -
    - Integrator, options: ``adaptive`` (default ``uniform``), ``uniform``, ``yoshida4``
  * - ``aperture_type``
    - ``aperture type``
    - str
    - -
    - Aperture type, default ``off``
  * - ``aperture_value``
    - ``aperture value``
    - list
    - -
    - Aperture parameter values, default ``[]``


Usage Examples
--------------

Thick Lens Normal Sextupole
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SF1": {
          "S (m)": 10.0,
          "Command": "Sextupole",
          "Length (m)": 0.5,
          "K2L": 5.0,
          "Num Slices": 5,
          "Integrator": "yoshida4",
          "Aperture Type": "off"
      }
  }

Normal sextupole (:math:`K_{2L} > 0`), length 0.5 m, 5 slices, 4th-order symplectic integration. Used for chromaticity correction.

Thin Lens Sextupole
~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SF2": {
          "S (m)": 20.0,
          "Command": "Sextupole",
          "Length (m)": 0.0,
          "K2L": 10.0,
          "Aperture Type": "off"
      }
  }

Zero-length sextupole, applying only the :math:`K_{2L}` thin lens kick, no body tracking.

Negative Sextupole
~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SD1": {
          "S (m)": 30.0,
          "Command": "Sextupole",
          "Length (m)": 0.4,
          "K2L": -5.0,
          "Num Slices": 1,
          "Integrator": "uniform",
          "Aperture Type": "off"
      }
  }

Negative sextupole (:math:`K_{2L} < 0`), providing chromaticity correction in the opposite direction to a positive sextupole.

Skew Sextupole
~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SS1": {
          "S (m)": 40.0,
          "Command": "Sextupole",
          "Length (m)": 0.3,
          "K2L": 0.0,
          "K2SL": 3.0,
          "Num Slices": 1,
          "Integrator": "uniform",
          "Aperture Type": "off"
      }
  }

Pure skew sextupole (:math:`K_{2L} = 0`, :math:`K_{2sL} \neq 0`), producing a coupling effect equivalent to rotating the normal sextupole by :math:`\pi / 6`.

Normal + Skew Sextupole Combination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SFS1": {
          "S (m)": 50.0,
          "Command": "Sextupole",
          "Length (m)": 0.5,
          "K2L": 5.0,
          "K2SL": 1.0,
          "Num Slices": 3,
          "Integrator": "yoshida4",
          "Aperture Type": "circle",
          "Aperture Value": [0.04]
      }
  }

Combined sextupole with both normal and skew components (simulating installation rotation error), with a circular aperture check.


Application Scenarios
---------------------

- **Chromaticity correction**: Place sextupoles at locations with dispersion to compensate for the natural chromaticity of quadrupoles, making the particle tune insensitive to momentum deviation
- **Resonance driving**: Drive 3rd-order resonances (:math:`3Q_x`, :math:`2Q_x \pm Q_y`, etc.) for resonance extraction or beam scraping
- **Dynamic aperture control**: The nonlinear field of the sextupole limits the stable phase-space region, affecting beam lifetime
- **Nonlinear coupling correction**: Using skew sextupoles (``k2sl``) to control higher-order :math:`x`-:math:`y` coupling
- **Harmonic sextupole**: Place sextupoles at specific phases to drive or suppress specific resonance terms
- **LHC chromaticity scheme**: Distribute sextupole families (SF/SD) in the arc region to achieve chromaticity control over a wide energy range


References
----------

- Xsuite Physics Guide, Sec 1.10.3 (exact drift), Sec 1.10.5 (sextupole)
- Xsuite source code: ``xtrack/beam_elements/elements_src/sextupole.h``, ``track_magnet.h``, ``track_magnet_kick.h``, ``track_magnet_drift.h``
- Yoshida, H., "Construction of higher order symplectic integrators", Phys. Lett. A 150 (1990)
- MAD-X Physics Manual: sextupole field and nonlinear transport
- Wiedemann, H., "Particle Accelerator Physics", Ch. 4 (nonlinear beam dynamics)
