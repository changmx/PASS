Solenoid
========

This module describes the PASS solenoid element **Solenoid**, used to simulate the motion of charged particles in a longitudinal magnetic field. The solenoid produces a uniform magnetic field :math:`B_z` along the beam direction, coupling the horizontal and vertical planes through the Larmor rotation effect while providing transverse focusing.

The PASS solenoid uses an **exact nonlinear map** (analytical solution of the Hamiltonian in the Larmor framework). For a pure solenoid (without multipole field overlay), the map has zero error; when multipole fields are superimposed, a Sol-Kick-Sol (SKS) symplectic integrator is used.

**Code Location**

- Source file: ``PASS/commands/element/solenoid.py``
- Class name: ``Solenoid`` (inherits from ``Command``)
- Registration name: ``solenoid``
- Key features:

  - Uses exact solenoid map (Larmor rotation + focusing, :math:`p_z` computed per particle)
  - No thin lens mode (solenoid has no thin lens limit; :math:`L=0` produces no effect)
  - Supports multipole field overlay (``knl`` / ``ksl``), using SKS integrator
  - Supports uniform (2nd-order leapfrog) and yoshida4 (4th-order Yoshida composition) integrators
  - :math:`k_s = 0` automatically degenerates to a pure drift
  - Chromaticity effects naturally introduced through per-particle :math:`p_z`
  - Supports aperture check


Coordinate Convention
---------------------

PASS uses normalized curvilinear coordinates. The six-dimensional phase-space variables are :math:`(x, p_x, y, p_y, z, \delta)`:

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

where :math:`P_0` is the reference particle momentum, :math:`\beta_0 = v_0 / c` is the reference particle normalized velocity.


Solenoid Field and Normalized Strength
--------------------------------------

The solenoid produces a uniform magnetic field along the beam direction (:math:`s` axis):

.. math::

  \vec{B} = (0, \, 0, \, B_z)

The normalized solenoid strength is defined as:

.. math::

  k_s = \frac{q_0 B_z}{P_0}

where :math:`q_0` is the reference particle charge and :math:`P_0` is the reference particle momentum. In PASS, the user directly specifies :math:`k_s` (``ks``).

Define the half-strength:

.. math::

  \text{sk} = \frac{k_s}{2}

The Larmor rotation angle is:

.. math::

  \theta = \frac{\text{sk} \cdot L}{p_z} = \frac{k_s L}{2 p_z}

where :math:`p_z` is the particle's normalized longitudinal momentum component (different for each particle, see below), and :math:`L` is the solenoid length.


Physical Derivation
--------------------

Hamiltonian
~~~~~~~~~~~

In the Cartesian coordinate system, the solenoid Hamiltonian is:

.. math::

  H_{\text{sol}} = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2} + \frac{k_s^2}{8}(x^2 + y^2) - \frac{k_s}{2}(x p_y - y p_x)

The physical meaning of each term:

.. list-table::
  :header-rows: 1
  :widths: 40 60

  * - Term
    - Physical Meaning
  * - :math:`-\sqrt{(1+\delta)^2 - p_x^2 - p_y^2}`
    - Free propagation (exact drift)
  * - :math:`\frac{k_s^2}{8}(x^2 + y^2)`
    - Solenoid focusing (equivalent quadrupole component)
  * - :math:`-\frac{k_s}{2}(x p_y - y p_x)`
    - Larmor rotation (:math:`x`-:math:`y` coupling)

.. note::

  The Larmor rotation term :math:`-\frac{k_s}{2}(x p_y - y p_x)` **depends on both position and momentum simultaneously**, which is the fundamental difference between the solenoid and the quadrupole. The quadrupole kick term depends only on position, allowing the Hamiltonian to be cleanly split into drift and kick parts (DKD integrator). The Larmor rotation term of the solenoid cannot be split into purely position-dependent or purely momentum-dependent parts, and therefore **cannot use ordinary drift for DKD integration**.

Larmor Framework and Exact Solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Performing the Larmor transformation—rotating the transverse coordinate system about the :math:`s` axis by angle :math:`\theta = \text{sk} \cdot s / p_z`—defines the canonical momenta in the Larmor framework:

.. math::

  p_{k1} = p_x + \text{sk} \cdot y

.. math::

  p_{k2} = p_y - \text{sk} \cdot x

In the Larmor framework, :math:`p_{k1}` and :math:`p_{k2}` are **conserved quantities** (do not vary with :math:`s`), so the longitudinal momentum component:

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_{k1}^2 - p_{k2}^2}

is constant per particle throughout the solenoid. This allows the solenoid map to be **solved exactly** without approximation.

Exact Solenoid Map
~~~~~~~~~~~~~~~~~~

Given the solenoid length :math:`L`, the Larmor rotation angle:

.. math::

  \theta = \frac{\text{sk} \cdot L}{p_z}

The map is divided into two steps: rotation and drift.

**Step 1: Larmor rotation** (rotating coordinates to the Larmor framework at :math:`s=L`)

.. math::

  \text{rps}_0 = \cos\theta \cdot x + \sin\theta \cdot y

.. math::

  \text{rps}_1 = \cos\theta \cdot p_x + \sin\theta \cdot p_y

.. math::

  \text{rps}_2 = \cos\theta \cdot y - \sin\theta \cdot x

.. math::

  \text{rps}_3 = \cos\theta \cdot p_y - \sin\theta \cdot p_x

**Step 2: Drift in the Larmor framework** (equivalent drift length :math:`\sin\theta / \text{sk}`)

.. math::

  x' = \cos\theta \cdot \text{rps}_0 + \frac{\sin\theta}{\text{sk}} \cdot \text{rps}_1

.. math::

  p_x' = \cos\theta \cdot \text{rps}_1 - \text{sk} \cdot \sin\theta \cdot \text{rps}_0

.. math::

  y' = \cos\theta \cdot \text{rps}_2 + \frac{\sin\theta}{\text{sk}} \cdot \text{rps}_3

.. math::

  p_y' = \cos\theta \cdot \text{rps}_3 - \text{sk} \cdot \sin\theta \cdot \text{rps}_2

**Longitudinal coordinate update**:

.. math::

  \Delta\zeta = L \cdot \left(1 - \frac{1+\delta}{p_z \cdot \text{rvv}}\right)

where :math:`\text{rvv} = \beta / \beta_0` is the ratio of the particle velocity to the reference particle velocity:

.. math::

  \beta = \frac{(1+\delta) \, \beta_0 \gamma_0}{\sqrt{1 + \left[(1+\delta) \, \beta_0 \gamma_0\right]^2}}

.. note::

  - :math:`p_z` is different for each particle (including contributions from :math:`\delta` and Larmor momenta), so the map is **exactly nonlinear**
  - When :math:`k_s \to 0`, :math:`\sin\theta/\text{sk} \to L/p_z`, and the map degenerates to exact drift


Why the Solenoid Has No Thin Lens Mode
--------------------------------------

The thin lens limit of a quadrupole (:math:`L \to 0`, :math:`k_1 \to \infty`, :math:`k_1 L = \text{const}`) gives a finite momentum kick :math:`\Delta p_x = -k_{1L} \cdot x`, which is physically self-consistent.

The thin lens limit of a solenoid (:math:`L \to 0`, :math:`k_s \to \infty`, :math:`k_s L = \text{const}`) has a fundamental difficulty:

- Larmor rotation angle :math:`\theta = k_s L / (2 p_z)` is finite ✓
- Focusing term :math:`\text{sk} \cdot \sin\theta = (k_s/2) \cdot \sin\theta \to \infty` diverges ✗

The scaling behavior of position and momentum is asymmetric: the rotation angle is finite but the focusing force diverges, so the thin lens limit does not exist.

Therefore, in PASS, the solenoid with :math:`L = 0` has **no effect** (identity map) and does not provide a thin lens mode.


Multipole Field Overlay and SKS Integrator
------------------------------------------

When transverse multipole field components (:math:`k_{nl}` / :math:`k_{sl}`) are superimposed inside the solenoid, the total Hamiltonian is:

.. math::

  H = H_{\text{sol}} + H_{\text{mult}}

where :math:`H_{\text{mult}}` is the multipole kick Hamiltonian (depending only on position). Since :math:`H_{\text{sol}}` and :math:`H_{\text{mult}}` do not commute, a split-operator method is needed.

PASS uses the **Sol-Kick-Sol** (SKS) integrator, fully parallel to the quadrupole's DKD:

.. math::

  \mathcal{M}_{\text{SKS}}(\Delta s) = \text{Sol}\!\left(\frac{\Delta s}{2}\right) \circ \text{Kick}(\Delta s) \circ \text{Sol}\!\left(\frac{\Delta s}{2}\right)

where:

- **Sol** = exact solenoid map (``_solenoid_exact_cpu``), handling the :math:`B_z` field
- **Kick** = multipole kick (Horner recursion), handling transverse multipole fields

.. list-table::
  :header-rows: 1
  :widths: 25 25 25 25

  * -
    - Drift Operator
    - Kick Operator
    - Scenario
  * - Quadrupole DKD
    - Free drift ``drift_exact``
    - Quadrupole kick ``quad_kick``
    - :math:`B_z = 0`, only transverse gradient field
  * - Solenoid SKS
    - Solenoid map ``solenoid_exact``
    - Multipole kick ``multipole_kick``
    - :math:`B_z \neq 0`, superimposed transverse multipole fields

.. note::

  The "Sol" in SKS is not a free drift but the exact solenoid map. Inside the solenoid, :math:`B_z` is always present, and particles do not drift in field-free space. If a free drift is incorrectly used instead of the solenoid map, the Larmor rotation effect would be lost.

Uniform Integrator (2nd-order symplectic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each slice uses the Sol-Kick-Sol structure, i.e., 2nd-order leapfrog:

.. math::

  S_2(\Delta s) = \text{Sol}\!\left(\frac{\Delta s}{2}\right) \circ \text{Kick}(\Delta s) \circ \text{Sol}\!\left(\frac{\Delta s}{2}\right)

The per-slice error is :math:`O(\Delta s^3)`, and the global error is :math:`O(\Delta s^2)`.

yoshida4 Integrator (4th-order symplectic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A 4th-order symplectic map is constructed by composing three 2nd-order SKS steps [Yoshida 1990]:

.. math::

  S_4(\Delta s) = S_2(z_1 \Delta s) \circ S_2(z_0 \Delta s) \circ S_2(z_1 \Delta s)

where the Yoshida coefficients are:

.. math::

  z_1 = \frac{1}{2 - 2^{1/3}} \approx 1.3512

.. math::

  z_0 = 1 - 2 z_1 \approx -1.7024

The per-slice error is :math:`O(\Delta s^5)`, and the global error is :math:`O(\Delta s^4)`.


Overall Tracking Flow
---------------------

Depending on whether multipole fields are superimposed, the solenoid has two tracking paths:

::

  ====== Thick lens (length > 0) ======

  No multipole field (knl/ksl all zero):
    Single-segment exact solenoid map Sol(L, ks)
    [Zero error, no slicing needed]

  With multipole field (knl/ksl nonzero):
    Slice 1 → Slice 2 → ... → Slice N
    (Each slice: Sol(ds/2) → Kick(ds) → Sol(ds/2))
    where ds = L / N

  Special cases:
    ks = 0 → degenerates to pure drift Drift(L)
    L = 0 → no effect (solenoid has no thin lens limit)

The complete map is:

Without multipole field:

.. math::

  \mathcal{M} = \text{Sol}(L, k_s)

With multipole field (:math:`N` slices):

.. math::

  \mathcal{M} = \left[\mathcal{M}_{\text{SKS}}(\Delta s)\right]^N


Chromaticity Effects
--------------------

The chromaticity effects of the solenoid are naturally introduced through the per-particle :math:`p_z` expression.

In the Larmor framework, :math:`p_z` depends on :math:`\delta` and the Larmor momenta :math:`p_{k1}, p_{k2}`:

.. math::

  p_z = \sqrt{(1+\delta)^2 - (p_x + \text{sk} \cdot y)^2 - (p_y - \text{sk} \cdot x)^2}

Particles with different momentum deviations :math:`\delta` have different :math:`p_z`, and therefore different Larmor rotation angles :math:`\theta = \text{sk} \cdot L / p_z` and different equivalent drift lengths :math:`\sin\theta / \text{sk}`. This is the physical origin of solenoid chromaticity—momentum-dependent rotation angle and focusing strength.


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
    - Element length (must be :math:`\ge 0`; :math:`= 0` produces no effect)
  * - ``name``
    - ``name``
    - str
    - -
    - Element name
  * - ``ks``
    - ``ks``
    - float
    - :math:`\text{m}^{-1}`
    - Solenoid normalized strength :math:`k_s = q_0 B_z / P_0`, default 0
  * - ``knl``
    - ``kil``
    - list
    - :math:`\text{m}^{-n}`
    - Multipole normal integrated strength array :math:`K_{nL}`, default ``[]``
  * - ``ksl``
    - ``kisl``
    - list
    - :math:`\text{m}^{-n}`
    - Multipole skew integrated strength array :math:`K_{sL}`, default ``[]``
  * - ``num_slice``
    - ``num slices``
    - int
    - -
    - Number of slices, default 1 (effective only with multipole field overlay)
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

.. note::

  - ``knl`` / ``ksl`` are optional parameters. When not specified or all zero, the solenoid uses a single-segment exact map (zero error), ignoring ``num_slices`` and ``integrator``
  - When nonzero ``knl`` / ``ksl`` are specified, the SKS integrator is enabled, and ``num_slices`` and ``integrator`` take effect
  - When ``ks = 0`` and the element has length, it degenerates to a pure drift
  - When ``length = 0``, the solenoid has no effect (thin lens mode is not provided)


Usage Examples
--------------

Pure Solenoid (Exact Map)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SOL1": {
          "S (m)": 10.0,
          "Command": "Solenoid",
          "Length (m)": 1.0,
          "ks": 2.0,
          "Aperture Type": "off"
      }
  }

Length 1.0 m, normalized strength :math:`k_s = 2.0`. Uses a single-segment exact solenoid map with zero error.

Weak Solenoid
~~~~~~~~~~~~~

.. code-block:: json

  {
      "SOL2": {
          "S (m)": 20.0,
          "Command": "Solenoid",
          "Length (m)": 2.0,
          "ks": 0.5,
          "Aperture Type": "off"
      }
  }

Weak-field solenoid with a small Larmor rotation angle.

Reverse-Field Solenoid
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SOL3": {
          "S (m)": 30.0,
          "Command": "Solenoid",
          "Length (m)": 1.5,
          "ks": -3.0,
          "Aperture Type": "off"
      }
  }

:math:`k_s < 0` indicates a reverse magnetic field, with the Larmor rotation in the opposite direction.

Solenoid with Quadrupole Overlay (SKS Integrator)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SOL4": {
          "S (m)": 40.0,
          "Command": "Solenoid",
          "Length (m)": 1.0,
          "ks": 2.0,
          "KiL": [0.0, 0.1],
          "Kisl": [],
          "Num Slices": 4,
          "Integrator": "yoshida4",
          "Aperture Type": "off"
      }
  }

Solenoid (:math:`k_s = 2.0`) with superimposed quadrupole component (:math:`K_{1L} = 0.1`), 4 slices, 4th-order symplectic integrator. The ``KiL`` array index 0 is :math:`K_{0L}` (dipole), and index 1 is :math:`K_{1L}` (quadrupole).

Zero-Field Degeneration (Pure Drift)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "SOL5": {
          "S (m)": 50.0,
          "Command": "Solenoid",
          "Length (m)": 1.0,
          "ks": 0.0,
          "Aperture Type": "off"
      }
  }

When :math:`k_s = 0`, degenerates to a pure drift.


Application Scenarios
---------------------

- **Low-energy beam transport lines**: In low-energy sections where :math:`\beta\gamma` is small, solenoid focusing is more efficient than quadrupoles, and is commonly used in injectors and low-energy transport lines
- **Electron coolers**: Solenoids confine the electron beam to co-move with the ion beam, used for cooling transverse emittance
- **Collider detector solenoids**: The solenoid magnetic fields of large experimental detectors (e.g., CMS, ATLAS) have a significant impact on beam optics and must be accurately accounted for in the lattice model
- **Superconducting solenoids**: Multipole field errors in high-field superconducting solenoids can be modeled through ``knl`` / ``ksl`` parameter overlay
- **Rotationally symmetric beams**: The Larmor rotation of the solenoid can be used to eliminate :math:`x`-:math:`y` coupling or produce specific rotationally symmetric beam distributions
