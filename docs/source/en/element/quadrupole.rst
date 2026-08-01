Quadrupole
==========

This module describes the PASS quadrupole element **Quadrupole**, used to simulate the motion of charged particles in a quadrupole magnet. The quadrupole is the most fundamental focusing element in accelerators, providing linear focusing force through a gradient magnetic field.

The PASS quadrupole supports both **thick element** (``length > 0``) and **thin lens** (``length = 0``) modes. The thick element provides two tracking models: **drift-kick-drift-exact** (DKD-exact) uses the exact drift-kick-drift symplectic integration scheme, supporting both uniform (2nd-order) and yoshida4 (4th-order) symplectic integrators; **mat-kick-mat** (MKM, default) uses the exact linear transport matrix scheme, which is exact for a purely linear field with a single slice.

**Code Location**

- Source file: ``PASS/commands/element/quadrupole.py``
- Class name: ``Quadrupole`` (inherits from ``Command``)
- Registration name: ``quadrupole``
- Key features:

  - Supports thin lens mode (``length = 0``, applies only a quadrupole kick)
  - Supports thick lens mode (``length > 0``, DKD-exact symplectic integration or MKM exact linear matrix)
  - Supports uniform (2nd-order leapfrog) and yoshida4 (4th-order Yoshida composition) integrators
  - Supports mat-kick-mat (MKM) model (exact linear transport matrix, including chromaticity)
  - MKM model handles k1+k1s combinations via rotational diagonalization
  - Supports normal quadrupole (``k1l``) and skew quadrupole (``k1sl``) and their combinations
  - Zero field (``k1l = k1sl = 0``) automatically degenerates to a pure drift
  - Chromaticity effects naturally introduced through exact drift
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


Quadrupole Field and Normalized Strength
----------------------------------------

The magnetic field of a quadrupole magnet is linearly distributed in the transverse plane. In complex notation:

.. math::

  B_y + i B_x = (G + i G_s)(x + i y)

where :math:`G` is the normal quadrupole gradient and :math:`G_s` is the skew quadrupole gradient. Expanding:

.. math::

  B_y = G \cdot x - G_s \cdot y

.. math::

  B_x = G \cdot y + G_s \cdot x

The normalized quadrupole strength is defined as:

.. math::

  K_1 = \frac{q_0 G}{P_0}

.. math::

  K_{1s} = \frac{q_0 G_s}{P_0}

The integrated strength is:

.. math::

  K_{1L} = K_1 \cdot L, \qquad K_{1sL} = K_{1s} \cdot L

where :math:`L` is the magnet length. In PASS, the user directly specifies :math:`K_{1L}` (``k1l``) and :math:`K_{1sL}` (``k1sl``); for thick lenses, :math:`K_1 = K_{1L} / L` and :math:`K_{1s} = K_{1sL} / L` are solved internally.


Overall Tracking Flow
---------------------

Depending on the magnet length, the quadrupole has two tracking modes:

**Thin lens mode** (:math:`L = 0`)

::

  ====== Thin lens (length = 0) ======

  Single quadrupole kick Kick(K1L, K1sL)
  [Position unchanged, momentum jump only]

**Thick lens mode** (:math:`L > 0`)

DKD-exact model:

::

  ====== Thick lens DKD-exact (length > 0) ======

  Slice 1 → Slice 2 → ... → Slice N
  (Each slice: Drift(ds/2) → Kick(ds) → Drift(ds/2))

  where ds = L / N

  If K1L = 0 and K1sL = 0: degenerates to a single exact drift Drift(L)

MKM model:

::

  ====== Thick lens mat-kick-mat (length > 0) ======

  Slice 1 → Slice 2 → ... → Slice N
  (Each slice: M(ds), exact linear transport matrix)

  For pure k1 + k1s (no higher-order multipoles): M(L) = M(ds)^N
  Therefore num_slice = 1 is sufficient; multiple slices do not change the result.

The complete map is:

Thin lens:

.. math::

  \mathcal{M}_{\text{thin}} = \text{Kick}(K_{1L}, K_{1sL})

Thick lens (N slices):

.. math::

  \mathcal{M}_{\text{thick}} = \left[\mathcal{M}_{\text{DKD}}(\Delta s)\right]^N

where the DKD map for each slice is:

.. math::

  \mathcal{M}_{\text{DKD}}(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

.. note::

  - Thin lens mode does not change the particle position coordinates :math:`(x, y, z)`, only applies momentum kicks
  - Chromaticity effects in thick lens mode are naturally introduced through the :math:`p_z` expression in exact drift (see chromaticity section)
  - When :math:`K_{1L} = 0` and :math:`K_{1sL} = 0`, the thick lens degenerates to a pure drift, avoiding meaningless empty kick loops


Physical Derivation
--------------------

Hamiltonian
~~~~~~~~~~~

In the Cartesian coordinate system (quadrupole has no curvature, :math:`h = 0`), the quadrupole Hamiltonian is:

.. math::

  H_{\text{quad}} = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2} + \frac{\chi}{2}\left(K_1 x^2 - K_1 y^2 + 2 K_{1s} x y\right)

Splitting it into the propagation part (exact drift :math:`H_D`) and the kick part (:math:`H_K`):

.. math::

  H_D = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  H_K = \frac{\chi}{2}\left(K_1 x^2 - K_1 y^2 + 2 K_{1s} x y\right)

where :math:`H_D` is the exact drift Hamiltonian (preserving the :math:`p_z` square root without small-momentum expansion), and :math:`H_K` is the quadrupole kick. This is the standard **split-operator** method: the Hamiltonian is split into analytically solvable parts, maps are applied separately, and then combined into a symplectic integrator.

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

  The meaning of "exact": the drift part preserves the exact square root :math:`p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}` without small-momentum expansion :math:`p_x \ll 1`. The approximation lies only in separating the propagation part from the kick part (split-operator method). This formula is identical to the exact drift in the Drift element and the SBend element.

Quadrupole Kick Map K
~~~~~~~~~~~~~~~~~~~~~

The kick part is a thin lens map (position unchanged, momentum jump only). From Hamilton's equations :math:`\dot{p}_x = -\partial H / \partial x`, :math:`\dot{p}_y = -\partial H / \partial y`:

.. math::

  \Delta p_x = -\chi K_1 L_K \cdot x + \chi K_{1s} L_K \cdot y

.. math::

  \Delta p_y = +\chi K_1 L_K \cdot y + \chi K_{1s} L_K \cdot x

where :math:`L_K` is the kick effective length.

Physical meaning of each term:

.. list-table::
  :header-rows: 1
  :widths: 30 15 55

  * - Term
    - Source
    - Physical Meaning
  * - :math:`-\chi K_1 L_K \cdot x`
    - :math:`\frac{\chi K_1 x^2}{2}`
    - Horizontal normal quadrupole focusing (focusing when :math:`K_1 > 0`, defocusing when :math:`K_1 < 0`)
  * - :math:`+\chi K_1 L_K \cdot y`
    - :math:`-\frac{\chi K_1 y^2}{2}`
    - Vertical normal quadrupole defocusing (opposite to horizontal)
  * - :math:`+\chi K_{1s} L_K \cdot y`
    - :math:`\chi K_{1s} x y`
    - Skew quadrupole horizontal coupling kick
  * - :math:`+\chi K_{1s} L_K \cdot x`
    - :math:`\chi K_{1s} x y`
    - Skew quadrupole vertical coupling kick

For thin lens mode, :math:`L_K = 1`, using the integrated strengths :math:`K_{1L}` and :math:`K_{1sL}` directly. For DKD mode, :math:`L_K = \Delta s`, using :math:`K_1 \Delta s` and :math:`K_{1s} \Delta s`.

.. note::

  A normal quadrupole (:math:`K_1 > 0`) focuses in the horizontal direction and defocuses in the vertical direction. This is a direct result of the quadrupole field :math:`B_y = G \cdot x`: particles deviating from the axis experience a force proportional to their offset, with a restoring force (focusing) in the horizontal direction and a repulsive force (defocusing) in the vertical direction. To achieve focusing in both directions simultaneously, focusing quadrupoles (F) and defocusing quadrupoles (D) must be alternately arranged, i.e., the FODO structure.

  A skew quadrupole (:math:`K_{1s} \neq 0`) rotates the focusing action by :math:`\pi / 4`, producing :math:`x`-:math:`y` coupling. In practice, it is often used for coupling correction or simulating installation rotation errors.


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

Integrator Selection Recommendations:

.. list-table::
  :header-rows: 1
  :widths: 25 20 55

  * - Scenario
    - Recommended Integrator
    - Reason
  * - Fast simulation
    - uniform
    - 2 drifts + 1 kick per slice, low computational cost
  * - High-precision simulation
    - yoshida4
    - 4th-order accuracy, but 6 drifts + 3 kicks per slice
  * - With space charge
    - uniform + more slices
    - PIC solve cost far exceeds drift; 4th-order Yoshida requires 3 PIC solves


Chromaticity Effects
--------------------

Chromaticity describes the dependence of the particle tune on the momentum deviation :math:`\delta`. The PASS DKD-exact model naturally introduces chromaticity through the exact drift's :math:`p_z` expression, without any additional treatment.

Physical Mechanism
~~~~~~~~~~~~~~~~~~

In DKD integration, the drift uses the exact expression :math:`p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}`, while the kick in :math:`(x, p_x)` space is :math:`\Delta p_x = -\chi K_1 \Delta s \cdot x` (not divided by :math:`1+\delta`).

Transforming to :math:`(x, x')` space (where :math:`x' = p_x / (1+\delta)`), the equivalent focusing strength automatically becomes:

.. math::

  K_{1,\text{eff}} = \frac{K_1}{1+\delta}

This is the physical origin of the natural chromaticity :math:`Q'_x = -\frac{1}{4\pi}\oint \beta_x K_1 \, ds`. No explicit division is needed in the code—the exact drift's :math:`p_z` expression automatically accomplishes this.

Thin Lens vs. Thick Lens Chromaticity Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 25 20 55

  * - Effect
    - Thin Lens
    - Thick Lens DKD-exact
  * - Natural chromaticity (normal quadrupole)
    - Not introduced
    - Introduced (:math:`K_{1,\text{eff}} = K_1/(1+\delta)`)
  * - Coupling chromaticity (skew quadrupole)
    - Not introduced
    - Introduced (coupling transport contains :math:`\delta` dependence)
  * - Higher-order nonlinear dispersion
    - Not introduced
    - Introduced (:math:`p_z` preserves full square root)
  * - Path-length effects (:math:`R_{56}`, etc.)
    - Not introduced
    - Introduced

Physically, a thin lens has zero length with no drift space, and the kick :math:`\Delta p_x = -K_{1L}\,x` does not contain :math:`\delta`. In a thick lens, particles have a drift path inside the magnet, and particles with different momenta travel different paths and experience different effective focusing—this is the source of chromaticity. The same applies to skew quadrupoles: the :math:`p_z` dependence in drift makes the coupling transport also contain :math:`\delta` dependence, introducing coupling chromaticity.

.. note::

  - Thin lens mode (``length = 0``) has no path-length effects, so the thin lens quadrupole itself **does not introduce natural chromaticity**—whether normal or skew quadrupole
  - Thick lens DKD-exact mode fully includes natural chromaticity effects, including higher-order nonlinear dispersion terms
  - Unlike the mat-kick-mat model (which explicitly divides by :math:`1+\delta`), PASS's DKD-exact introduces chromaticity implicitly through exact :math:`p_z`, and also includes the higher-order nonlinear effects of :math:`p_z`
  - In PASS's Twiss linear transport framework, natural chromaticity is introduced through ``DQx`` / ``DQy`` parameters (:math:`\delta` terms in the phase advance), not through the element itself. If a thin lens quadrupole is additionally inserted in Twiss transport, it will not double-count chromaticity with ``DQx``—because the thin lens itself does not introduce chromaticity. However, if the inserted quadrupole strength is large enough to significantly change the lattice tune and :math:`\beta` functions, the original Twiss parameters (including ``DQx``) are no longer accurate and need to be recomputed


Tracking Model Comparison
-------------------------

The PASS quadrupole supports two thick lens body models, suitable for different precision and speed requirements.

Model Overview
~~~~~~~~~~~~~~

**drift-kick-drift-exact (DKD-exact)**: Splits the Hamiltonian into exact drift and thin lens kick, combined via symplectic splitting. Preserves the full :math:`p_z` nonlinear kinematics, but linear focusing and chromaticity have symplectic splitting errors (controllable by increasing the number of slices or using higher-order integrators).

**mat-kick-mat (MKM)**: Solves the analytical exact solution of the linearized equation of motion :math:`u'' + K_{\text{eff}} \chi/(1+\delta) \cdot u = 0`, constructing the transport matrix using trigonometric functions (focusing plane) and hyperbolic functions (defocusing plane). Linear focusing, linear chromaticity, and :math:`R_{56}` are all exact solutions, but higher-order nonlinear terms of :math:`p_z` are not included.

Comparison Table
~~~~~~~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 25 35 40

  * - Feature
    - drift-kick-drift-exact
    - mat-kick-mat
  * - Linear focusing
    - 2nd-order symplectic splitting approximation
    - Exact (analytical matrix)
  * - Linear chromaticity
    - Approximate (:math:`O(1/N^2)` splitting error)
    - Exact (:math:`K_1/(1+\delta)` explicit)
  * - Nonlinear kinematics
    - Preserved (full :math:`p_z` square root)
    - Not included (linearized :math:`x' = p_x/(1+\delta)`)
  * - Longitudinal :math:`R_{56}`
    - Approximate
    - Exact (analytical formula)
  * - Nonlinear path length
    - Preserved
    - Not included
  * - Symplecticity
    - Strictly symplectic
    - Symplectic (matrix is symplectic)
  * - Computation speed
    - Slower (:math:`N \times 3` DKD steps)
    - Fast (matrix multiplication)

MKM Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~~

MKM solves the exact solution of the linearized equation. For a pure :math:`K_1` quadrupole, the u plane (focusing) uses sin/cos matrices, the v plane (defocusing) uses sinh/cosh matrices, with equivalent strength :math:`K = K_1 \chi / (1+\delta)` computed per particle.

For :math:`K_1 + K_{1s}` combined quadrupoles, **rotational diagonalization** is used:

.. math::

  \theta = \frac{1}{2}\arctan\frac{-K_{1s}}{K_1}, \quad K_{\text{eff}} = \sqrt{K_1^2 + K_{1s}^2}

After rotating to the principal axis frame, the matrix is applied, then rotated back. Key property: :math:`\theta` is independent of :math:`\delta` (:math:`K_1` and :math:`K_{1s}` scale proportionally by :math:`\chi/(1+\delta)`, so the ratio is unchanged), therefore :math:`\theta`, :math:`\cos\theta`, and :math:`\sin\theta` can be precomputed once in ``__init__``.

Special cases:

.. list-table::
  :header-rows: 1
  :widths: 20 15 15 50

  * - Physical State
    - :math:`K_1`
    - :math:`K_{1s}`
    - :math:`\theta`
  * - Normal quadrupole
    - :math:`K_{\text{eff}}`
    - 0
    - 0 (no rotation)
  * - Pure skew quadrupole
    - 0
    - :math:`K_{\text{eff}}`
    - :math:`\pi/4` (rotate 45°)
  * - Combined quadrupole
    - Nonzero
    - Nonzero
    - :math:`\frac{1}{2}\arctan(-K_{1s}/K_1)`

For pure :math:`K_1 + K_{1s}` (no higher-order multipoles), the matrix is exact for any slice length :math:`\Delta s`, so ``num_slice = 1`` is sufficient. Multiple slices are only needed in the future when supporting :math:`K_2` / :math:`K_{2s}` multipole kicks.

MKM Limitations
~~~~~~~~~~~~~~~

MKM linearizes the kinematics: :math:`x' = p_x / (1+\delta)`, rather than the exact :math:`x' = p_x / p_z`. Expanding:

.. math::

  \frac{p_x}{p_z} = \frac{p_x}{1+\delta}\left(1 + \frac{p_x^2 + p_y^2}{2(1+\delta)^2} + \cdots\right)

MKM retains only the first term; the lost higher-order terms lead to:

- **Amplitude-dependent tune shift** (geometric nonlinearity): from :math:`p_x^3` and similar terms; zero in MKM
- **Higher-order chromaticity** (:math:`Q''` and above): from :math:`\delta \cdot p_x^2` cross terms; lost in MKM
- **Nonlinear path length**: contributions of :math:`p_x^2`, :math:`p_x^4`, etc. to :math:`\Delta z`; lost in MKM

For typical storage ring parameters (:math:`p_x \sim 10^{-4}`), the lost effects are on the order of :math:`10^{-8}` per magnet, but may be amplified under full-ring accumulation and multi-turn effects. If studying nonlinear beam dynamics problems such as dynamic aperture and tune footprint, the DKD-exact model should be used.

Slice Count and Integrator Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 25 15 15 45

  * - Model
    - Recommended Slices
    - Recommended Integrator
    - Description
  * - mat-kick-mat
    - 1
    - — ¹
    - Matrix is exact for any :math:`\Delta s`; 1 slice is sufficient. Fast, suitable for linear optics calculations
  * - drift-kick-drift-exact
    - Needs testing ²
    - yoshida4
    - Slice count needs to be determined through convergence testing

¹ The MKM model does not use the integrator parameter.

² DKD-exact slice count selection recommendations:

  - Slice count should be determined through **convergence testing**: compare tune and chromaticity under different slice counts to confirm convergence
  - For HIAF-BRing (quadrupole length approximately 1 m, :math:`K_1` approximately 0.2), a slice count of 5 with yoshida4 integrator is recommended
  - Insufficient slice count will lead to :math:`O(1/N^2)` error in linear chromaticity; amplitude-dependent tune shift will have larger deviations
  - The yoshida4 integrator has per-slice error :math:`O(\Delta s^5)` and global error :math:`O(\Delta s^4)`, with precision far superior to uniform


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
  * - ``k1l``
    - ``k1l``
    - float
    - :math:`\text{m}^{-1}`
    - Normal quadrupole integrated strength :math:`K_{1L}`, default 0
  * - ``k1sl``
    - ``k1sl``
    - float
    - :math:`\text{m}^{-1}`
    - Skew quadrupole integrated strength :math:`K_{1sL}`, default 0
  * - ``model``
    - ``model``
    - str
    - -
    - Physical model, options: ``adaptive`` (default ``mat-kick-mat``), ``drift-kick-drift-exact``, ``mat-kick-mat``
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

Thick Lens Normal Quadrupole
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "QF1": {
          "S (m)": 10.0,
          "Command": "Quadrupole",
          "Length (m)": 0.5,
          "K1L": 0.2,
          "Num Slices": 5,
          "Integrator": "yoshida4",
          "Aperture Type": "off"
      }
  }

Focusing quadrupole (:math:`K_{1L} > 0`), length 0.5 m, 5 slices, 4th-order symplectic integration.

MKM Model Quadrupole
~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "QF1": {
          "S (m)": 10.0,
          "Command": "Quadrupole",
          "Length (m)": 0.5,
          "K1L": 0.2,
          "Model": "mat-kick-mat",
          "Num Slices": 1,
          "Aperture Type": "off"
      }
  }

MKM model, exact linear transport, 1 slice is sufficient. Faster than DKD-exact, suitable for linear optics calculations.

Thin Lens Quadrupole
~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "QF2": {
          "S (m)": 20.0,
          "Command": "Quadrupole",
          "Length (m)": 0.0,
          "K1L": 0.3,
          "Aperture Type": "off"
      }
  }

Zero-length quadrupole, applying only the :math:`K_{1L}` thin lens kick, no body tracking, no chromaticity effects.

Defocusing Quadrupole
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "QD1": {
          "S (m)": 30.0,
          "Command": "Quadrupole",
          "Length (m)": 0.4,
          "K1L": -0.15,
          "Num Slices": 1,
          "Integrator": "uniform",
          "Aperture Type": "off"
      }
  }

Defocusing quadrupole (:math:`K_{1L} < 0`), horizontal defocusing, vertical focusing.

Skew Quadrupole
~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "QS1": {
          "S (m)": 40.0,
          "Command": "Quadrupole",
          "Length (m)": 0.3,
          "K1L": 0.0,
          "K1SL": 0.1,
          "Num Slices": 1,
          "Integrator": "uniform",
          "Aperture Type": "off"
      }
  }

Pure skew quadrupole (:math:`K_{1L} = 0`, :math:`K_{1sL} \neq 0`), producing :math:`x`-:math:`y` coupling.

Normal + Skew Quadrupole Combination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "QFS1": {
          "S (m)": 50.0,
          "Command": "Quadrupole",
          "Length (m)": 0.5,
          "K1L": 0.2,
          "K1SL": 0.05,
          "Num Slices": 3,
          "Integrator": "yoshida4",
          "Aperture Type": "circle",
          "Aperture Value": [0.04]
      }
  }

Combined quadrupole with both normal and skew components (simulating installation rotation error), with a circular aperture check.

Equivalent Representation with Rotation Angle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A normal quadrupole with :math:`K_{1L} = 0.2` rotated by angle :math:`\theta = 0.01` rad is equivalent to:

.. math::

  K_{1L}' = K_{1L} \cos 2\theta \approx 0.2 \times 0.9998 = 0.19996

.. math::

  K_{1sL}' = K_{1L} \sin 2\theta \approx 0.2 \times 0.02 = 0.004

.. code-block:: json

  {
      "QF_rot": {
          "S (m)": 60.0,
          "Command": "Quadrupole",
          "Length (m)": 0.5,
          "K1L": 0.19996,
          "K1SL": 0.004,
          "Num Slices": 1,
          "Integrator": "uniform"
      }
  }


Application Scenarios
---------------------

- **Linear focusing**: Alternately arrange focusing (F) and defocusing (D) quadrupoles in a FODO structure to achieve transverse beam confinement
- **Chromaticity correction**: Utilize the natural chromaticity effect of quadrupoles, compensating chromaticity by adjusting sextupoles
- **Coupling correction**: Use skew quadrupoles (``k1sl``) to control :math:`x`-:math:`y` coupling and correct installation errors
- **Tune adjustment**: Change the working point (tune) by adjusting quadrupole strength, tuning the beam to the optimal working region
- **Dispersion matching**: Place quadrupoles after bending magnets to control the evolution of the dispersion function :math:`\eta(s)`
- **Beam transport lines**: Use quadrupoles in injection and extraction lines to focus the beam and control the beam envelope


References
----------

- Xsuite Physics Guide, Sec 1.10.3 (exact drift), Sec 1.10.5 (quadrupole)
- Xsuite source code: ``xtrack/beam_elements/elements_src/quadrupole.h``, ``track_magnet.h``, ``track_magnet_kick.h``, ``track_magnet_drift.h``
- Yoshida, H., "Construction of higher order symplectic integrators", Phys. Lett. A 150 (1990)
- MAD-X Physics Manual: quadrupole field and linear transport
