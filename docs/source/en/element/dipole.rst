Dipole (SBend)
==============

This module describes the PASS dipole element **SBend**, used to simulate the motion of charged particles in a sector bending magnet. The dipole is the most fundamental bending element in accelerators, deflecting the particle orbit through a uniform magnetic field.

The PASS dipole is a **thick element** (``length > 0``), supporting full nonlinear tracking including edge angle effects, fringe field effects, and multiple symplectic integration schemes.

**Code Location**

- Source file: ``PASS/commands/element/dipole.py``
- Class name: ``SBend`` (inherits from ``Command``)
- Registration name: ``sbend``
- Key features:

  - Supports rot-kick-rot (RKR) and drift-kick-drift-exact (DKD-exact) body models, default rot-kick-rot
  - Supports uniform (2nd-order) and yoshida4 (4th-order) symplectic integrators
  - Supports nonlinear edge angle (wedge) effects
  - Supports nonlinear fringe field effects
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

Other commonly used physical quantities:

.. math::

  h = \frac{1}{\rho} = \frac{K_0 L}{L} = \frac{K_{0L}}{L}

.. math::

  K_0 = \frac{q_0 B_0}{P_0} = \frac{K_{0L}}{L}

.. math::

  \chi = \frac{q}{q_0} \cdot \frac{m_0}{m}

where :math:`h` is the reference orbit curvature, :math:`\rho` is the bending radius, :math:`K_0` is the normalized dipole field strength, and :math:`\chi` is the charge-to-mass ratio factor (for a beam of identical particle species, :math:`\chi = 1`). For a sector bend, :math:`h = K_0`.


Overall Tracking Flow
----------------------

A complete dipole consists of the **entrance edge**, **body**, and **exit edge**. The tracking order is:

::

  ====== Entrance edge (B=0 → B=B0) ======

  YRotation(-e1)  →  Fringe Field  →  Wedge(-e1, K0)
  [Pure geometric rotation]  [Nonlinear fringe field]  [Rotation + focusing kick]
  [B=0]                      [B: 0→B0]                 [B=B0]

         ↓

  ====== Body (B=B0) ======

  Slice 1 → Slice 2 → ... → Slice N
  (Reference trajectory coordinate system, symplectic integrator)

         ↓

  ====== Exit edge (B=B0 → B=0) ======

  Wedge(-e2, K0)   →  Fringe Field  →  YRotation(-e2)
  [Rotation + focusing kick]  [Nonlinear fringe field]  [Pure geometric rotation]
  [B=B0]                      [B: B0→0]                 [B=0]

  Entrance net rotation: (-e1) + (+e1) = 0
  Exit net rotation: (+e2) + (-e2) = 0
  → Body operates in the reference trajectory coordinate system

The complete map is:

.. math::

  \mathcal{M}_{\text{bend}} = \mathcal{M}_{\text{exit}} \circ \mathcal{M}_{\text{body}} \circ \mathcal{M}_{\text{entry}}

**Entrance edge**:

.. math::

  \mathcal{M}_{\text{entry}} = \text{Wedge}(-e_1, K_0) \circ \text{Fringe}(e_1) \circ \text{YRotation}(-e_1)

**Exit edge** (mirror-symmetric to the entrance):

.. math::

  \mathcal{M}_{\text{exit}} = \text{YRotation}(-e_2) \circ \text{Fringe}(e_2) \circ \text{Wedge}(-e_2, K_0)

.. note::

  - When :math:`e_1 = 0`, both YRotation and Wedge are skipped (no edge angle effect)
  - When ``fint`` = 0 or ``hgap`` = 0, Fringe is skipped (no fringe field effect)
  - When :math:`K_0 = 0`, both Fringe and Wedge are skipped
  - The execution order of entrance and exit are mirror images of each other
  - **At the exit**, :math:`K_0` **is partially negated**: Xsuite negates the local variable :math:`K_0` at the exit (``if (is_exit) k0 = -k0``), but only **DipoleFringe** uses the negated :math:`-K_0`, because the exit fringe field is the magnetic field decreasing from :math:`B_0` to 0 (opposite direction to the entrance's 0 rising to :math:`B_0`). **Wedge** directly uses the original ``knorm[0]`` (not negated), because Wedge describes rotation in the uniform magnetic field :math:`B_0`, and the field direction is the same at entrance and exit. PASS implements this behavior in ``_edge_exit_cpu`` using ``k0_fringe = -k0`` (Fringe only) and ``k0`` (Wedge).


Why This Order
~~~~~~~~~~~~~~

**Role of YRotation**: Pure geometric coordinate rotation, transforming particle coordinates from the reference trajectory coordinate system to the magnet end-face coordinate system. It applies no force and only changes the observational reference frame.

**Role of Fringe Field**: Computes nonlinear fringe field effects in the end-face reference frame. The particle slope :math:`x' = p_x / p_z` in the Fringe formula must be the slope relative to the magnet end face, so YRotation must be performed first.

**Role of Wedge**: Rotates the observation plane in the uniform magnetic field :math:`B_0`, while applying the edge angle focusing kick. Wedge includes both geometric rotation (rotating the coordinate system back to the reference trajectory system) and magnetic field kick.

The geometric rotations of YRotation and Wedge are in opposite directions, with zero net rotation:

.. math::

  \text{Rotation of YRotation}(-e_1) + \text{Geometric rotation of Wedge}(-e_1) = (-e_1) + (+e_1) = 0

Therefore the body operates in the reference trajectory coordinate system, requiring no additional rotation.


Body: DKD-exact Model
---------------------

Hamiltonian
~~~~~~~~~~~

In curvilinear coordinates, the complete dipole Hamiltonian is:

.. math::

  H_{\text{bend}} = \frac{p_\tau}{\beta_0} - (1+hx)\sqrt{(1+\delta)^2 - p_x^2 - p_y^2} + \chi K_0\!\left(x + \frac{h x^2}{2}\right)

Splitting it into the propagation part (exact straight drift :math:`H_D`) and the kick parts (:math:`H_h`, :math:`H_{K_0}`, :math:`H_{K_0 h}`):

.. math::

  H_D = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  H_h = -h x (1+\delta)

.. math::

  H_{K_0} = \chi K_0 x

.. math::

  H_{K_0 h} = \frac{\chi K_0 h x^2}{2}

where :math:`H_D` is the exact straight drift Hamiltonian (preserving the :math:`p_z` square root without small-momentum expansion), and the remaining three terms are thin lens kicks.

Exact Drift Map D
~~~~~~~~~~~~~~~~~

The Hamilton's equations of the propagation part give the exact straight drift:

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

  \beta = \frac{(1+\delta) \beta_0 \gamma_0}{\sqrt{1 + \left[(1+\delta) \beta_0 \gamma_0\right]^2}}

.. note::

  The meaning of "exact": the drift part preserves the exact square root :math:`p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}` without small-momentum expansion :math:`p_x \ll 1`. The approximation lies only in separating the propagation part from the kick part (split-operator method).

Dipole Kick Map K
~~~~~~~~~~~~~~~~~

The kick parts are combined into a single thin lens kick (position unchanged, momentum jump only):

.. math::

  p_x \leftarrow p_x + L_K \cdot \left[h(1+\delta) - \chi K_0 - \chi K_0 h x\right]

.. math::

  \zeta \leftarrow \zeta - \frac{\beta_0}{\beta} \cdot h x L_K

Physical meaning of each term:

.. list-table::
  :header-rows: 1
  :widths: 30 15 55

  * - Term
    - Source
    - Physical Meaning
  * - :math:`h(1+\delta) L_K`
    - :math:`H_h`
    - Curvature kick (reference orbit bending)
  * - :math:`-\chi K_0 L_K`
    - :math:`H_{K_0}`
    - Main dipole bending
  * - :math:`-\chi K_0 h x L_K`
    - :math:`H_{K_0 h}`
    - Weak focusing (curvature-dipole field coupling)
  * - :math:`-(\beta_0/\beta) \cdot h x L_K`
    - :math:`H_h`
    - Path-length effect (longitudinal)

.. note::

  For a sector bend :math:`h = K_0`, the net kick of the reference particle (:math:`x=0, \delta=0, \chi=1`) is :math:`h L_K - K_0 L_K = 0`. This is correct: in curvilinear coordinates, the reference particle moves along the reference orbit, and :math:`p_x` is always 0. The bending effect is already encoded in the curvilinear coordinate system itself.

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


Body Model Comparison
---------------------

The PASS dipole body supports two physical models, selected by the ``model`` parameter.

Model Description
~~~~~~~~~~~~~~~~~

- **rot-kick-rot (RKR)**: The dipole field is a constant field (independent of :math:`x`), and the thin lens kick itself is exact. The drift step uses polar drift, handling curvature effects. A single slice achieves high precision. Default model.
- **drift-kick-drift-exact (DKD-exact)**: The drift step uses straight-line exact drift, and curvature is treated as a thin lens kick. For bends with large deflection angles, insufficient slice count leads to chromaticity errors.

Feature Comparison
~~~~~~~~~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 25 35 40

  * - Feature
    - rot-kick-rot
    - drift-kick-drift-exact
  * - Drift type
    - Polar drift (including curvature)
    - Straight-line exact drift
  * - k0 handling
    - Within the drift step (interleaved k0_kick)
    - In the kick step (thin lens)
  * - Curvature Jacobian
    - Included ((1+h·x) correction)
    - Not included (requires more slices to compensate)
  * - 1-slice precision
    - High (constant field kick is exact)
    - Limited (curvature approximation)
  * - Chromaticity precision
    - Exact with 1 slice
    - Requires sufficient slice count

Slice Count Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 25 15 20 40

  * - Model
    - Recommended Slices
    - Recommended Integrator
    - Description
  * - rot-kick-rot
    - 1
    - yoshida4
    - The constant dipole field makes the 1-slice kick exact; polar drift handles curvature; 1 slice achieves high precision
  * - drift-kick-drift-exact
    - 5~10
    - yoshida4
    - Large deflection angles require more slices to compensate for chromaticity errors due to missing curvature Jacobian

.. note::

  In the DKD-exact model, curvature h is treated as a thin lens kick, and the drift step does not include the (1+h·x) Jacobian correction. When the bend deflection angle is large, the path differences of particles with different momenta in the drift are not correctly accounted for, leading to chromaticity deviations. Increasing the slice count can mitigate this issue (error converges as :math:`O(1/N^2)`), but the RKR model fundamentally avoids this problem—polar drift handles curvature within the drift step, requiring no additional slices.


Entrance Edge: YRotation
------------------------

Physical Purpose
~~~~~~~~~~~~~~~~

YRotation is a pure geometric coordinate rotation that does not involve any magnetic field. Its purpose is to transform particle coordinates from the reference trajectory coordinate system to the magnet end-face coordinate system, so that subsequent fringe field calculations can be performed in the correct reference frame.

When the magnet end face has an angle :math:`e_1` with respect to the reference trajectory normal, the particle's incident slope relative to the end face is not equal to :math:`p_x / p_z`. YRotation rotates the coordinate system by :math:`-e_1`, so that the rotated :math:`p_x / p_z` becomes the slope relative to the end face.

Complete Derivation
~~~~~~~~~~~~~~~~~~~

::

  YRotation(-e1)                    Wedge(-e1, K0)
  ─────────────────                 ──────────────────
  Coordinate system: trajectory → end-face    Coordinate system: end-face → trajectory
  Magnetic field: B = 0 (field-free region)   Magnetic field: B = B0 (field region)
  Rotation: -e1                                Rotation: +e1 (opposite to YRotation)
  Kick: none                                   Kick: Δpx = K0·x·sin(e1) (focusing)
  ─────────────────                 ──────────────────

  Net rotation: (-e1) + (+e1) = 0
  Net effect: fringe field effect + focusing kick

**Step 1: Momentum rotation**

The reference frame is rotated about the :math:`y` axis by angle :math:`\theta`. This rotation occurs in the :math:`(x, z)` plane of 3D space, with the :math:`y` direction unaffected. Therefore, the :math:`x` component of momentum :math:`p_x` and the :math:`z` component :math:`p_z` are mixed, while :math:`p_y` is unchanged. The standard rotation transformation of momentum is:

.. math::

  \begin{pmatrix} p_x' \\ p_z' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} p_x \\ p_z \end{pmatrix}

Therefore:

.. math::

  p_x' = \cos\theta \cdot p_x - \sin\theta \cdot p_z

.. math::

  p_z' = \sin\theta \cdot p_x + \cos\theta \cdot p_z

This is a pure momentum rotation with no force involved.

**Step 2: Position projection**

The key is the definition of :math:`x` in curvilinear coordinates: **x is the transverse offset measured on the transverse plane at the reference position** :math:`s. When the reference frame is rotated by :math:`\theta`, the direction of the "transverse plane" changes, and the same physical position has a different :math:`x'` value in the new coordinate system.

The particle's trajectory in the old frame is :math:`\vec{r}(\lambda) = (x + \lambda p_x/p_z,\; 0,\; \lambda)`. The :math:`z'` axis direction of the new frame is :math:`\hat{z}' = (\sin\theta, 0, \cos\theta)`, and the transverse direction is :math:`\hat{x}' = (\cos\theta, 0, -\sin\theta)`.

In the new frame, zero longitudinal position means:

.. math::

  \vec{r} \cdot \hat{z}' = (x + \lambda p_x/p_z)\sin\theta + \lambda\cos\theta = 0

Solving for :math:`\lambda`:

.. math::

  \lambda = -\frac{x \sin\theta \cdot p_z}{p_x \sin\theta + p_z \cos\theta} = -\frac{x \sin\theta \cdot p_z}{p_z'}

The new transverse position is:

.. math::

  x' = \vec{r} \cdot \hat{x}' = (x + \lambda p_x/p_z)\cos\theta - \lambda\sin\theta

Substituting :math:`\lambda` and simplifying, using the orthogonality of rotation :math:`p_z'\cos\theta - p_x'\sin\theta = p_z`, we finally obtain:

.. math::

  x' = \frac{x \cdot p_z}{p_z'} = \frac{x \cdot p_z}{p_x \sin\theta + p_z \cos\theta}

**Verification of consistency with the Xsuite formula**:

Xsuite writes ``x_hat = x / (cos_angle * ptt)``, where ``ptt = 1 + tan_angle * px / pz``:

.. math::

  \cos\theta \cdot p_{tt} = \cos\theta + \sin\theta \cdot \frac{p_x}{p_z} = \frac{p_z'}{p_z}

Therefore :math:`x' = x / (\cos\theta \cdot p_{tt}) = x \cdot p_z / p_z'`, consistent with the derivation.

**Step 3: y direction**

The :math:`y` direction does not directly participate in the rotation, but due to the tilting of the transverse plane, the particle acquires an additional offset in the :math:`y` direction due to trajectory slope while traveling to the new transverse plane:

.. math::

  y' = y - \sin\theta \cdot \frac{x \cdot p_y}{p_z'} = y - \tan\theta \cdot \frac{x \cdot p_y}{p_z \cdot p_{tt}}

**Step 4:** :math:`\zeta` **direction**

The longitudinal coordinate :math:`\zeta = s - \beta_0 c t`. After rotating the reference frame, :math:`\sin\theta \cdot x` is the projection of the transverse position onto the new longitudinal direction (additional path length due to reference frame rotation), which needs to be converted to a time increment:

.. math::

  \Delta\zeta = \beta_0 \cdot \tan\theta \cdot \frac{x \cdot \text{time\_fac}}{p_z \cdot p_{tt}}

where :math:`\text{time\_fac} = 1/\beta_0 + p_\tau` is the time-path conversion factor (see variable conversion section).

Final Formulas
~~~~~~~~~~~~~~

Definitions:

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  p_{tt} = 1 + \tan(\theta) \cdot \frac{p_x}{p_z}

where :math:`\theta` is the rotation angle (entrance :math:`\theta = -e_1`, exit :math:`\theta = -e_2`).

Time factor:

.. math::

  \text{time\_fac} = \frac{1}{\beta_0} + p_\tau = \sqrt{(1+\delta)^2 + \frac{1}{\beta_0^2 \gamma_0^2}}

Six-dimensional map:

.. math::

  x \leftarrow \frac{x}{\cos\theta \cdot p_{tt}}

.. math::

  p_x \leftarrow \cos\theta \cdot p_x - \sin\theta \cdot p_z

.. math::

  y \leftarrow y - \tan\theta \cdot \frac{x \cdot p_y}{p_z \cdot p_{tt}}

.. math::

  \zeta \leftarrow \zeta + \beta_0 \cdot \tan\theta \cdot \frac{x \cdot \text{time\_fac}}{p_z \cdot p_{tt}}

.. math::

  p_y, \delta \text{ unchanged}

Variable Conversion Notes
~~~~~~~~~~~~~~~~~~~~~~~~~

Xsuite stores :math:`p_\tau` (normalized longitudinal momentum deviation), while PASS stores :math:`\delta` (normalized total momentum deviation). The exact relationship between them is:

.. math::

  (1+\delta)^2 = \left(\frac{1}{\beta_0} + p_\tau\right)^2 - \frac{1}{\beta_0^2 \gamma_0^2}

Therefore:

.. math::

  \frac{1}{\beta_0} + p_\tau = \sqrt{(1+\delta)^2 + \frac{1}{\beta_0^2 \gamma_0^2}}

**Derivation**: The total particle energy :math:`E = E_0(1 + p_\tau)`. From :math:`E^2 - E_0^2 = (Pc)^2 - (P_0 c)^2`:

.. math::

  (1+\delta)^2 = (P/P_0)^2 = \frac{(1+p_\tau)^2 - 1/\gamma_0^2}{\beta_0^2}

Expanding and using :math:`1 - 1/\gamma_0^2 = \beta_0^2`:

.. math::

  (1+\delta)^2 = 1 + \frac{2 p_\tau}{\beta_0^2} + \frac{p_\tau^2}{\beta_0^2}

And the Xsuite expression for :math:`p_z` is ``sqrt(1 + 2*pt/beta0 + pt*pt - px*px - py*py)``, which after substituting the above relation equals exactly :math:`\sqrt{(1+\delta)^2 - p_x^2 - p_y^2}`.


Entrance Edge: Fringe Field
---------------------------

Physical Purpose
~~~~~~~~~~~~~~~~

The magnetic field of a real magnet is not a step function at the end face, but has a gradual transition region. This gradual region produces additional nonlinear effects, mainly vertical focusing.

Physical Derivation
~~~~~~~~~~~~~~~~~~~

::

  B_y                                          B_y = B0
  ↑                              ┌──────────────
  │                             /
  │         Fringe             /  end face
  │         Field             /
  │       (B: 0→B0)          /
  │                        /
  │  B = 0               /
  └────────────┬────────┘──────────────────────── → s
               ↑        ↑
               YRotation Fringe    Wedge / Body
               (B=0)    (0→B0)     (B=B0)

  hgap = half gap
  Δp_y = vertical focusing kick

**Fringe field distribution**

The magnetic field at the end face of a real magnet transitions as:

.. math::

  B_y(s) = B_0 \cdot b(s), \quad b(s): 0 \to 1

where :math:`b(s)` is the normalized fringe field distribution function. This gradual field produces nonlinear effects on particles, mainly vertical focusing.

**Fringe field integral**

Forest defines the fringe field integral:

.. math::

  F = \int \frac{b(s)\bigl(K_0 - b(s)K_0\bigr)}{g_{\text{full}} \cdot K_0^2} \, ds

where :math:`g_{\text{full}}` is the magnet full gap (:math:`g_{\text{full}} = 2 \times \text{hgap}`). :math:`F` describes the "strength" of the fringe field: :math:`F=0` corresponds to a hard edge, and larger :math:`F` means stronger fringe field effects.

.. warning::

  **Naming relationship between hgap and g**

  The ``hgap`` parameter in Xsuite and PASS is the magnet **half gap**, i.e., half the distance between the upper and lower pole faces.

  - ``hgap`` = half gap = :math:`g_{\text{half}}`
  - Magnet full gap :math:`g_{\text{full}} = 2 \times \text{hgap}`

  In the fringe field code, the auxiliary quantity :math:`f_h` directly uses ``hgap`` (half gap):

  .. math::

    f_h = \text{hgap} \times \text{fint}

  This is fully consistent with Xsuite source code ``track_dipole_fringe.h`` line 37 ``fh = hgap * fint``. The ``hgap`` parameter in Xsuite is also the half gap. Therefore the physical meaning of :math:`f_h` is "half gap × fringe field integral", not "full gap × fringe field integral".

  In the physics literature (e.g., Forest's original paper), :math:`g` in the fringe field integral formula typically refers to the full gap. The Xsuite/MAD-NG geometry uses the half gap ``hgap``, with corresponding coefficient adjustments (e.g., the factor 72 in :math:`f_{\text{sad}} = 1/(72 \cdot f_h)` comes from this adjustment). PASS keeps this half-gap convention while using the PTC-compatible generating-function form for comparison with MAD-X PTC.

**Generating function**

The fringe field map is a canonical transformation (symplectic map), generated by the generating function :math:`\Phi_0`:

.. math::

  \Phi_0 = \arctan\!\left(\frac{x'}{1+y'^2}\right) - c_2 \left(1 + x'^2(1+y'^2)\right) p_z

where :math:`x' = p_x/p_z`, :math:`y' = p_y/p_z` are the particle slopes, and :math:`c_2 = 2 K_0 \chi \cdot f_h` is the linear fringe field strength parameter.

- **First term** :math:`\arctan(x'/(1+y'^2))`: Incident angle correction of the particle at the end face. :math:`x'` is the horizontal slope, and :math:`1+y'^2` reflects the geometric correction of vertical motion on the horizontal incident angle (3D direction cosines).
- **Second term** :math:`-c_2(1 + x'^2(1+y'^2))p_z`: Fringe field integral effect. :math:`c_2` is the fringe field strength, and :math:`(1 + x'^2(1+y'^2))` is the higher-order slope correction.

**Partial derivatives and forces**

Taking partial derivatives of :math:`\Phi_0` with respect to slopes :math:`(x', y', p_z)`, then converting to partial derivatives with respect to :math:`(p_x, p_y, \delta)` via the chain rule, gives the force components.

Introducing intermediate variables :math:`c_{o2} = b_0 / \cos^2\Phi_0`, :math:`c_{o1}`, :math:`c_{o3}` (see formula section for details), the partial derivatives are:

.. math::

  \phi_1 = \frac{\partial \Phi_0}{\partial x'}, \quad \phi_2 = \frac{\partial \Phi_0}{\partial y'}, \quad \phi_3 = \frac{\partial \Phi_0}{\partial p_z}

Force components (chain rule :math:`k_i = \phi_1 \partial x'/\partial p_i + \phi_2 \partial y'/\partial p_i + \phi_3 \partial p_z/\partial p_i`):

.. math::

  k_x = \phi_1 \frac{1+x'^2}{p_z} + \phi_2 \frac{x'y'}{p_z} - \phi_3 x'

.. math::

  k_y = \phi_1 \frac{x'y'}{p_z} + \phi_2 \frac{1+y'^2}{p_z} - \phi_3 y'

.. math::

  k_z = \phi_1 \frac{\text{tfac} \cdot x'}{p_z^2} + \phi_2 \frac{\text{tfac} \cdot y'}{p_z^2} - \phi_3 \frac{\text{tfac}}{p_z}

where :math:`\text{tfac} = -(1/\beta_0 + p_\tau)` comes from the dependence of :math:`t` on :math:`p_z` in :math:`\zeta = s - \beta_0 c t`.

**Implicit equation**

The fringe field map is not a simple kick (position unchanged, momentum jump), but an **implicit map**. The reason is that the fringe field effect is nonlinearly coupled with the particle's :math:`y` coordinate: as the particle passes through the fringe field, the :math:`y` coordinate itself is changing, so the force (depending on :math:`y`) and the displacement (depending on the force) are mutually coupled.

The implicit solution comes from the generating function expanded to second order:

.. math::

  y_f = \frac{2y}{1 + \sqrt{1 - 2 k_y y}}

This form guarantees canonicity (symplecticity): when :math:`k_y y` is small, :math:`y_f \approx y + k_y y^2/2`, i.e., the second-order expansion.

Parameters
~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 15 15 70

  * - Parameter
    - Symbol
    - Description
  * - ``fint``
    - :math:`F`
    - Fringe field integral, :math:`F=0` for hard edge
  * - ``hgap``
    - :math:`g_{\text{half}}`
    - Magnet **half gap**, full gap :math:`g_{\text{full}} = 2 \cdot \text{hgap}`
  * - ``k0``
    - :math:`K_0`
    - Normalized dipole field strength

Complete Formulas
~~~~~~~~~~~~~~~~~

Define auxiliary quantities:

.. math::

  f_h = \text{hgap} \cdot F

.. math::

  f_{\text{sad}} = \frac{1}{72 \cdot f_h} \quad (f_h > 0 \text{, otherwise } 0)

.. math::

  b_0 = K_0 \cdot \chi

.. math::

  \text{relp} = \frac{1}{\sqrt{(1+\delta)^2}}

.. math::

  \text{tfac} = -\left(\frac{1}{\beta_0} + p_\tau\right) = -\sqrt{(1+\delta)^2 + \frac{1}{\beta_0^2 \gamma_0^2}}

.. math::

  c_2 = b_0 \cdot f_h \cdot 2

.. math::

  c_3 = b_0^2 \cdot f_{\text{sad}} \cdot \text{relp}

where :math:`c_2` is the linear fringe field strength, and :math:`c_3` is the 6th-order nonlinear correction (from :math:`f_{\text{sad}} = 1/(72 f_h)`, called the "sixth-order achromatic detuning" term).

Particle slopes:

.. math::

  x' = \frac{p_x}{p_z}, \quad y' = \frac{p_y}{p_z}

Characteristic functions and partial derivatives:

.. math::

  \phi_0 = \arctan\!\left(\frac{x'}{1 + y'^2}\right) - c_2 \left(1 + x'^2(1+y'^2)\right) p_z

.. math::

  c_{o2} = \frac{b_0}{\cos^2\phi_0}

.. math::

  c_{o1} = \frac{c_{o2}}{1 + \left(\frac{x'}{1+y'^2}\right)^2} \cdot \frac{1}{1+y'^2}

.. math::

  c_{o3} = c_{o2} \cdot c_2

.. math::

  \phi_1 = c_{o1} - c_{o3} \cdot 2 x'(1+y'^2) p_z

.. math::

  \phi_2 = -2 c_{o1} \cdot x' y' \cdot \frac{1}{1+y'^2} - c_{o3} \cdot 2 x' y' \cdot p_z

.. math::

  \phi_3 = -c_{o3} \left(1 + x'^2(1+y'^2)\right)

Force components:

.. math::

  k_x = \phi_1 \frac{1+x'^2}{p_z} + \phi_2 \frac{x' y'}{p_z} - \phi_3 x'

.. math::

  k_y = \phi_1 \frac{x' y'}{p_z} + \phi_2 \frac{1+y'^2}{p_z} - \phi_3 y'

.. math::

  k_z = \phi_1 \frac{\text{tfac} \cdot x'}{p_z^2} + \phi_2 \frac{\text{tfac} \cdot y'}{p_z^2} - \phi_3 \frac{\text{tfac}}{p_z}

Six-dimensional map:

.. math::

  y_f = \frac{2y}{1 + \sqrt{1 - 2 k_y y}}

.. math::

  x \leftarrow x + \frac{1}{2} k_x y_f^2

.. math::

  p_y \leftarrow p_y - 4 c_3 y_f^3 - b_0 \tan(\phi_0) \cdot y_f

.. math::

  \zeta \leftarrow \zeta + \beta_0 \left(\frac{1}{2} k_z y_f^2 + c_3 y_f^4 \cdot \text{relp}^2 \cdot \text{tfac}\right)

.. math::

  p_x, \delta \text{ unchanged}

.. note::

  - :math:`p_x` and :math:`\delta` are unchanged: the fringe field is a static magnetic field and does no work
  - The :math:`-b_0 \tan(\phi_0) y_f` term in the :math:`p_y` change is the main vertical focusing
  - :math:`-4 c_3 y_f^3` is the 6th-order nonlinear correction
  - :math:`y_f` is solved through the implicit equation, ensuring precise treatment and symplecticity of nonlinear effects
  - The :math:`x` change is an :math:`O(y^2)` order horizontal-vertical coupling


Entrance Edge: Wedge (Edge Angle)
---------------------------------

Physical Purpose
~~~~~~~~~~~~~~~~

Wedge describes the coordinate transformation of a particle passing through a tilted end face in a uniform magnetic field :math:`B_0`. It accomplishes two things simultaneously:

1. **Geometric rotation**: Rotates the coordinate system from the end-face reference frame back to the reference trajectory reference frame (opposite rotation direction to YRotation, canceling each other)
2. **Magnetic field kick**: Applies the edge angle focusing effect :math:`\Delta p_x \propto -x`

When :math:`K_0 = 0` (no magnetic field), Wedge degenerates to pure YRotation (proof below).

Computation Formulas
~~~~~~~~~~~~~~~~~~~~

Parameters: :math:`\theta` (wedge angle, entrance :math:`\theta = -e_1`), :math:`K_0`, :math:`\chi`

Definitions:

.. math::

  b_1 = K_0 \cdot \chi

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  A = \frac{1}{\sqrt{(1+\delta)^2 - p_y^2}}

.. math::

  \text{rvv} = \frac{\beta}{\beta_0}

where :math:`\beta` is the particle's actual normalized velocity (see the formula in the drift map).

Six-dimensional map:

.. math::

  p_x' = p_x \cos\theta + (p_z - b_1 x) \sin\theta

.. math::

  p_z' = \sqrt{(1+\delta)^2 - p_x'^2 - p_y^2}

.. math::

  x' = x \cos\theta + \frac{x p_x \sin(2\theta) + \sin^2\theta \cdot (2 x p_z - b_1 x^2)}{p_z' + p_z \cos\theta - p_x \sin\theta}

.. math::

  D = \arcsin(A \cdot p_x) - \arcsin(A \cdot p_x')

.. math::

  \Delta y = \frac{p_y (\theta + D)}{b_1}

.. math::

  \Delta \ell = \frac{(1+\delta) (\theta + D)}{b_1}

Final updates:

.. math::

  x \leftarrow x'

.. math::

  p_x \leftarrow p_x'

.. math::

  y \leftarrow y + \Delta y

.. math::

  \zeta \leftarrow \zeta - \frac{\Delta \ell}{\text{rvv}}

.. math::

  p_y, \delta \text{ unchanged}

Physical Meaning of the Kick Term
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expanding the :math:`p_x'` formula:

.. math::

  p_x' = \underbrace{p_x \cos\theta + p_z \sin\theta}_{\text{geometric rotation}} \;\underbrace{-\; b_1 x \sin\theta}_{\text{magnetic field kick}}

- **Geometric rotation part**: :math:`p_x \cos\theta + p_z \sin\theta`, opposite in direction to YRotation's :math:`\cos\theta \cdot p_x - \sin\theta \cdot p_z`, canceling each other
- **Magnetic field kick part**: :math:`-b_1 x \sin\theta = -K_0 \chi x \sin\theta`, proportional to :math:`-x`, i.e., edge angle focusing

For the entrance (:math:`\theta = -e_1`), the kick is:

.. math::

  \Delta p_x = -K_0 \chi x \sin(-e_1) = K_0 \chi x \sin(e_1) > 0 \quad (\text{when } x > 0)

This gives particles on the outside of the orbit (:math:`x > 0`) an inward momentum, i.e., a focusing effect. The equivalent focal length is :math:`f = \rho / \sin(e_1)`.


Relationship Between Delta-ell and zeta
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Equation Eq. 1.201 gives the **path length** :math:`\Delta\ell`, while the code updates the **longitudinal coordinate** :math:`\zeta`. The two are different:

.. math::

  \Delta\ell = \frac{(1+\delta)(\theta + D)}{b_1} \quad \text{(path length)}

.. math::

  \Delta\zeta = -\frac{\Delta\ell}{\text{rvv}} \quad (\zeta \text{ coordinate update})

**Physical reason**:

1. :math:`\Delta\ell` is the path length the particle travels in the wedge
2. :math:`\zeta = s - \beta_0 c t`, updating :math:`\zeta` requires time: :math:`\Delta t = \Delta\ell / v = \Delta\ell / (\text{rvv} \cdot \beta_0 c)`
3. :math:`\Delta s` is already handled in the geometric transformations of :math:`x'` and :math:`\Delta y`, so :math:`\zeta` only needs the **time correction** part:

.. math::

  \Delta\zeta = -\beta_0 c \cdot \Delta t = -\beta_0 c \cdot \frac{\Delta\ell}{\text{rvv} \cdot \beta_0 c} = -\frac{\Delta\ell}{\text{rvv}}

- **Negative sign**: :math:`\zeta = s - \beta_0 c t`, time increase causes :math:`\zeta` to decrease
- **Division by rvv**: :math:`\text{rvv} = v/v_0 = \beta/\beta_0`, converting path length to time requires dividing by the particle's actual velocity

.. math::

  \zeta \leftarrow \zeta - \frac{\Delta \ell}{\text{rvv}}

Fully consistent with Xsuite source code ``add_to_zeta(-delta_ell / rvv)``.

.. warning::

  Do not add :math:`\Delta\ell` directly to :math:`\zeta`. :math:`\Delta\ell` is path length, :math:`\zeta` is a time-related coordinate, and the two are related through :math:`-\Delta\ell/\text{rvv}`.


arcsin Clipping
~~~~~~~~~~~~~~~

The computation of :math:`D` in Wedge uses :math:`\arcsin`:

.. math::

  D = \arcsin(A \cdot p_x) - \arcsin(A \cdot p_x')

where :math:`A = 1/\sqrt{(1+\delta)^2 - p_y^2}`. Theoretically :math:`|A \cdot p_x| \leq 1` (because :math:`|p_x| \leq \sqrt{(1+\delta)^2 - p_y^2}`), but **floating-point errors** may cause :math:`A \cdot p_x = 1.0000000001`, in which case Python's ``np.arcsin`` returns NaN and triggers a RuntimeWarning.

The code uses ``np.clip`` to limit the argument to the :math:`[-1, 1]` range:

.. code-block:: python

  arg_px = np.clip(arg_px, -1.0, 1.0)
  arg_new_px = np.clip(arg_new_px, -1.0, 1.0)

This is a **pure numerical safety measure** that does not change the physical result. In exact arithmetic, :math:`A \cdot p_x` is strictly within :math:`[-1, 1]`, and the clip is never triggered. Xsuite is C code, where ``asin(1.0000000001)`` returns a finite value without raising an exception, but Python requires explicit protection.


Proof That Wedge Degenerates to YRotation When K0=0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Wedge's** :math:`p_x'` **formula**:

.. math::

  p_x' = p_x \cos\theta + (p_z - b_1 x) \sin\theta

Setting :math:`b_1 = K_0 \chi \to 0`:

.. math::

  p_x' = p_x \cos\theta + p_z \sin\theta

YRotation's :math:`p_x'` formula is :math:`p_x' = \cos\alpha \cdot p_x - \sin\alpha \cdot p_z`. Setting :math:`\alpha = -\theta`:

.. math::

  p_x' = \cos\theta \cdot p_x + \sin\theta \cdot p_z

The momentum formulas are consistent.

**Position** :math:`x'`:

Setting :math:`b_1 = 0`, Wedge's :math:`x'` simplifies to:

.. math::

  x' = x \cos\theta + \frac{2 x \sin\theta \cdot p_x'}{p_z' + p_z \cos\theta - p_x \sin\theta}

Using the rotation orthogonality :math:`p_z' = p_x \sin\theta + p_z \cos\theta`, the denominator becomes :math:`2 p_z \cos\theta`, and we finally obtain:

.. math::

  x' = \frac{x \cdot p_z}{p_z'} = \frac{x \cdot p_z}{p_x \sin\theta + p_z \cos\theta}

Consistent with YRotation's position formula.

**y and** :math:`\zeta`:

When :math:`b_1 \to 0`, :math:`D \to 0`, :math:`\theta + D \to \theta`, so :math:`\Delta y = p_y \theta / b_1 \to \infty`.

This means the formulas for :math:`y` and :math:`\zeta` diverge as :math:`b_1 \to 0`. Therefore, the :math:`b_1 = 0` limit of Wedge is not continuous—when :math:`|b_1| < \epsilon`, the code directly branches to YRotation:

.. code-block:: python

  if abs(b1) < const.eps:
      # Directly call YRotation, skipping computation containing 1/b1
      self._y_rotation_cpu(x, px, y, py, z, dp, tag, mask, theta, beta0)
      return

.. note::

  Wedge degenerates to YRotation at :math:`b_1 = 0` through a **code branch structure**. The momentum and position formulas are consistent in the :math:`b_1 \to 0` limit, but :math:`y` and :math:`\zeta` contain a :math:`1/b_1` factor, making the limit discontinuous, so a branch is necessary.


Edge Angle Sign Convention
--------------------------

::

  e1 > 0 (focusing):

                    End face normal
                      ↗  e1
  Trajectory normal  ↑        ↗
            |      ╱  End face
            |    ╱
  ──────────┼──╱──────────→ s (reference trajectory)
            |  ╱
            |╱

  End face normal tilts toward the bending outside relative to trajectory normal


  e1 < 0 (defocusing):

            |╲
            |  ╲
  ──────────┼──╲──────────→ s (reference trajectory)
            |    ╲  End face
            |      ╲
  Trajectory normal  ↓        ╲  e1
                     ╲ End face normal

  End face normal tilts toward the bending inside relative to trajectory normal

The sign convention for edge angles :math:`e_1`, :math:`e_2` follows the MAD-X / Xsuite convention:

- :math:`e_1 > 0`: The entrance end face normal tilts toward the bending outside relative to the reference trajectory normal
- :math:`e_2 > 0`: The exit end face normal tilts toward the bending outside relative to the reference trajectory normal

Common magnet types:

.. list-table::
  :header-rows: 1
  :widths: 30 15 15 40

  * - Magnet Type
    - :math:`e_1`
    - :math:`e_2`
    - Description
  * - Sector bend
    - 0
    - 0
    - End face perpendicular to the reference trajectory
  * - Rectangular bend
    - :math:`\alpha/2`
    - :math:`\alpha/2`
    - :math:`\alpha` is the bending angle
  * - General
    - Arbitrary
    - Arbitrary
    - User-specified


Parameter List
--------------

General Parameters
~~~~~~~~~~~~~~~~~~

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
    - Element length (:math:`L`)
  * - ``name``
    - ``name``
    - str
    - -
    - Element name
  * - ``k0l``
    - ``k0l``
    - float
    - -
    - Normalized dipole field integral (:math:`K_{0L}`)
  * - ``e1``
    - ``e1 (rad)``
    - float
    - rad
    - Entrance edge angle (:math:`e_1`), default 0
  * - ``e2``
    - ``e2 (rad)``
    - float
    - rad
    - Exit edge angle (:math:`e_2`), default 0
  * - ``hgap``
    - ``hgap (m)``
    - float
    - m
    - Magnet half gap (:math:`g_{\text{half}}`), default 0
  * - ``fint``
    - ``fint``
    - float
    - -
    - Entrance fringe field integral (:math:`F`), default 0
  * - ``fintx``
    - ``fintx``
    - float
    - -
    - Exit fringe field integral, default 0 (auto-set to ``fint`` when :math:`\leq 0`)
  * - ``num_slice``
    - ``num slices``
    - int
    - -
    - Number of slices, default 1
  * - ``model``
    - ``model``
    - str
    - -
    - Physical model, options: ``adaptive`` (default, auto-selects ``rot-kick-rot``), ``rot-kick-rot``, ``drift-kick-drift-exact``
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

Extended Parameters (Reserved)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 35

  * - Property
    - JSON key
    - Type
    - Unit
    - Description
  * - ``is_field_error``
    - ``is field error``
    - bool
    - -
    - Whether field error is enabled, default ``false``
  * - ``is_ramping``
    - ``is ramping``
    - bool
    - -
    - Whether magnetic field ramping is enabled, default ``false``
  * - ``k0l_ramping_filepath``
    - ``k0l ramping filepath``
    - str
    - -
    - Magnetic field ramping data file path


Usage Examples
--------------

Sector Bend
~~~~~~~~~~~

.. code-block:: json

  {
      "BEND1": {
          "S (m)": 10.0,
          "Command": "SBend",
          "Length (m)": 1.5,
          "K0L": 0.05,
          "Num Slices": 5,
          "Integrator": "yoshida4",
          "Aperture Type": "off"
      }
  }

Sector bend with end faces perpendicular to the reference trajectory, no edge effects.

Rectangular Bend
~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "BEND2": {
          "S (m)": 20.0,
          "Command": "SBend",
          "Length (m)": 2.0,
          "K0L": 0.1,
          "E1 (rad)": 0.05,
          "E2 (rad)": 0.05,
          "HGap (m)": 0.02,
          "FInt": 0.5,
          "Num Slices": 10,
          "Integrator": "yoshida4",
          "Aperture Type": "off"
      }
  }

Rectangular bend with edge angle and fringe field effects. Bending angle :math:`\alpha = K_{0L} = 0.1` rad, edge angles :math:`e_1 = e_2 = \alpha/2 = 0.05` rad.

Thin Lens Bend
~~~~~~~~~~~~~~

.. code-block:: json

  {
      "BEND3": {
          "S (m)": 30.0,
          "Command": "SBend",
          "Length (m)": 0.0,
          "K0L": 0.02,
          "Aperture Type": "off"
      }
  }

Zero-length dipole, applying only the :math:`K_{0L}` thin lens kick, no body tracking, no edge effects.


References
----------

- Xsuite Physics Guide, Sec 1.10.3 (exact bend), Sec 1.10.9 (fringe field), Sec 1.10.10 (wedge), Sec 1.10.12 (quadrupole wedge correction)
- Forest, E. et al., "Edge Focusing Effects in Sector Bending Magnets"
- Yoshida, H., "Construction of higher order symplectic integrators", Phys. Lett. A 150 (1990)
- MAD-NG fringe field implementation: https://github.com/MethodicalAcceleratorDesign/MAD
