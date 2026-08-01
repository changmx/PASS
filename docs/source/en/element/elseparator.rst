ElSeparator
===========

This module describes the PASS electrostatic separator element **ElSeparator**, used to simulate the deflection of charged particles in a uniform transverse electric field. Electrostatic separators are widely used in beam injection and extraction systems, using a septum (cutting plate/wire) to divide the aperture into a field-free region (circulating beam in the ring) and a field region (injected or extracted beam), applying electric field deflection only to particles that cross the septum.

The PASS electrostatic separator supports both **thin lens** (pure momentum kick) and **thick lens** (DKD exact parabolic trajectory) modes. The user can input deflection parameters via either field strength (``ex`` / ``ey``, V/m) or integrated field (``exl`` / ``eyl``, V), consistent with MAD-X definitions.

**Code Location**

- Source file: ``PASS/commands/element/elseparator.py``
- Class name: ``ElSeparator`` (inherits from ``Command``)
- Registration name: ``elseparator``
- Key features:

  - Thin lens (``length = 0``): pure momentum translation, strictly symplectic
  - Thick lens (``length > 0``): DKD (Drift-Kick-Drift) 2nd-order symplectic integration, exact solution for uniform electric field
  - Two input methods: field strength (``ex`` / ``ey``) and integrated field (``exl`` / ``eyl``), with automatic mutual derivation
  - Septum position detection: automatically determines whether a particle is in the field-free region, field region, or striking the plate/wire
  - Supports ``tilt`` roll rotation about the :math:`s` axis (clockwise, consistent with MAD-X)
  - Supports aperture check


Coordinate Convention
---------------------

PASS uses the six-dimensional phase-space variables :math:`(x, p_x, y, p_y, z, \delta)`:

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


Physical Derivation
--------------------

Electric Field Force and Normalized Kick
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The electrostatic separator produces a uniform transverse electric field :math:`E_x` or :math:`E_y` between the plates. A charged particle in the electric field experiences the force:

.. math::

  \vec{F} = q \vec{E}

The particle traverses the separator of length :math:`L` with longitudinal velocity :math:`v = \beta_0 c`, with a residence time of :math:`t = L / (\beta_0 c)`. The transverse momentum change is:

.. math::

  \Delta P_x = q E_x \cdot t = \frac{q E_x L}{\beta_0 c}

Normalizing to PASS coordinates (:math:`p_x = P_x / P_0`, :math:`P_0 = q_0 B\rho`, same species :math:`q = q_0`):

.. math::

  \Delta p_x = \frac{\Delta P_x}{P_0} = \frac{E_x L}{\beta_0 c \cdot B\rho} = \frac{\mathrm{exl}}{\beta_0 c \cdot B\rho}

where :math:`\mathrm{exl} = E_x \cdot L` is the integrated electric field (unit: volts), and :math:`B\rho = P_0 / q_0` is the magnetic rigidity. Similarly:

.. math::

  \Delta p_y = \frac{\mathrm{eyl}}{\beta_0 c \cdot B\rho}

Dimensional verification: :math:`[\mathrm{V}] / ([\mathrm{m/s}] \cdot [\mathrm{T \cdot m}]) = [\mathrm{J/C}] / [\mathrm{kg \cdot m / (C \cdot s)}] = 1` (dimensionless) ✓

Equivalence with Magnetic Dipole
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The deflection produced by an electric field on a particle with velocity :math:`\beta_0 c` is equivalent to a particle with magnetic rigidity :math:`B\rho` passing through a magnetic field :math:`B`. From :math:`\Delta p_x = E_x L / (\beta_0 c \cdot B\rho)` and the magnetic dipole kick :math:`\Delta p_x = B L / B\rho` being equivalent:

.. math::

  E_x = \beta_0 c \cdot B

That is, an electric field of :math:`1\,\mathrm{MV/m}` at :math:`\beta_0 \approx 1` is equivalent to a magnetic field of :math:`B \approx 3.336\,\mathrm{mT}`.

Same Deflection Angle but Different Energy Change
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The above equivalence refers only to the **same deflection angle**. Electric and magnetic fields differ fundamentally in energy conservation:

- **Magnetic field does no work**: :math:`\vec{F} = q\vec{v}\times\vec{B}`, :math:`\vec{F} \perp \vec{v}`, the total particle momentum :math:`P` is unchanged; as :math:`p_x` increases, :math:`p_z` decreases (momentum redistribution), and :math:`\delta` is exactly unchanged.
- **Electric field does work**: :math:`\vec{F} = q\vec{E}`, the particle has a transverse displacement :math:`\Delta x` within the plates, the electric field does work :math:`W = qE_x \cdot \Delta x \neq 0`, and the total particle energy increases, so :math:`\delta` changes.

For a thick lens, the transverse displacement of the particle (DKD exact solution) is:

.. math::

  \Delta x = \frac{p_{x0} L}{p_z} + \frac{\Delta p_x \cdot L}{2 p_z}

The :math:`\delta` change corresponding to the electric field work:

.. math::

  \Delta\delta = \frac{W}{P_0 c} = \frac{E_x \cdot \Delta x}{B\rho \cdot c}

For a particle with :math:`p_{x0} = 0`, substituting :math:`\Delta x = \Delta p_x \cdot L / (2 p_z)` and :math:`\Delta p_x = E_x L / (\beta_0 c \cdot B\rho)`:

.. math::

  \Delta\delta = \frac{\Delta p_x^2}{2\beta_0}

For :math:`\Delta p_x = 30\,\mathrm{mrad}`, :math:`\beta_0 \approx 1`, :math:`\Delta\delta \approx 4.5 \times 10^{-4}`, which is one to two orders of magnitude smaller than typical beam momentum spread (:math:`10^{-3} \sim 10^{-2}`).

.. note::

  In the PASS DKD implementation, the kick only updates :math:`p_x` / :math:`p_y`, not :math:`\delta`, i.e., the electric field work is ignored. This is a reasonable approximation:

  - **Magnitude is negligible**: :math:`\Delta\delta = O(\Delta p_x^2)`, on the order of :math:`10^{-4}` for deflection angles of tens of mrad
  - **Per-particle correct handling is costly**: :math:`\Delta x` depends on the initial :math:`p_{x0}`, which differs for each particle; correctly computing the work requires tracking displacement per particle within the DKD, turning a simple symplectic integrator into an iterative scheme
  - **Thin lens self-consistency**: When :math:`L = 0`, :math:`\Delta x = 0`, :math:`W = 0`, :math:`\Delta\delta = 0`; the thin lens ignoring energy change is naturally self-consistent


Thin Lens Mode
--------------

When ``length = 0``, the electrostatic separator is modeled as a thin lens: the particle position is unchanged, and only the momentum undergoes an instantaneous jump:

.. math::

  x \leftarrow x

.. math::

  p_x \leftarrow p_x + \frac{\mathrm{exl}}{\beta_0 c \cdot B\rho}

.. math::

  y \leftarrow y

.. math::

  p_y \leftarrow p_y + \frac{\mathrm{eyl}}{\beta_0 c \cdot B\rho}

The Jacobian of this map is the identity matrix, which is strictly symplectic. The kick is computed directly from the integrated field :math:`\mathrm{exl}` / :math:`\mathrm{eyl}` without needing to know the plate length.


Thick Lens Mode (DKD)
---------------------

When ``length > 0``, Drift-Kick-Drift (DKD) 2nd-order symplectic integration is used:

.. math::

  \mathcal{M}_{\mathrm{DKD}}(L) = \mathrm{Drift}\!\left(\frac{L}{2}\right) \circ \mathrm{Kick}(L) \circ \mathrm{Drift}\!\left(\frac{L}{2}\right)

where Kick is the thin lens kick (:math:`\Delta p_x = \mathrm{exl} / (\beta_0 c \cdot B\rho)`), and Drift is the exact drift map.

Each ``_drift_exact_cpu`` call performs :math:`x \mathrel{+}= L \cdot p_x / p_z`. The DKD has three steps: the first drift uses the initial :math:`p_{x0}`, after the kick the second drift uses :math:`p_{x0} + \Delta p_x`. Combining:

.. math::

  \Delta x = \frac{p_{x0} L}{2 p_z} + \frac{(p_{x0} + \Delta p_x) L}{2 p_z} = \frac{p_{x0} L}{p_z} + \frac{\Delta p_x \cdot L}{2 p_z}

DKD Is Exact for Uniform Electric Field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under a uniform electric field, the particle equation of motion is constant-acceleration motion. Let :math:`k = E_x / (\beta_0 c \cdot B\rho)` (constant), with :math:`p_z` approximately unchanged:

.. math::

  \frac{dp_x}{ds} = k

.. math::

  \frac{dx}{ds} = \frac{p_x}{p_z}

Integrating gives a parabolic trajectory:

.. math::

  p_x(s) = p_{x0} + k \cdot s

.. math::

  x(s) = x_0 + \frac{p_{x0}}{p_z} s + \frac{k}{2 p_z} s^2

At :math:`s = L`, substituting :math:`\Delta p_x = k L`:

.. math::

  x(L) = x_0 + \frac{p_{x0} L}{p_z} + \frac{\Delta p_x \cdot L}{2 p_z}

This is **exactly consistent** with the DKD result. This is not a coincidence—leapfrog (DKD) is exact for constant-acceleration motion, because the first half drift uses the initial :math:`p_x` and the second half uses the kicked :math:`p_{x0} + \Delta p_x`; the average exactly gives the parabola.

.. note::

  The above exactness assumes :math:`p_z \approx \mathrm{const}`. For typical deflection angles (tens of mrad), the :math:`p_z` change :math:`\Delta p_z \approx -\Delta p_x^2 / (2 p_z) \sim 10^{-4}` is negligible. Therefore, DKD does not need an additional ``model="exact"`` mode.


Kick Consistency Between Thin Lens and Thick Lens
--------------------------------------------------

The kick :math:`\Delta p_x` is **exactly the same** for both thin lens and thick lens, both computed from the integrated field :math:`\mathrm{exl}`. The difference is only in the position change:

.. list-table::
  :header-rows: 1
  :widths: 20 40 40

  * -
    - Thin lens (:math:`L = 0`)
    - Thick lens DKD (:math:`L > 0`)
  * - kick :math:`\Delta p_x`
    - :math:`\mathrm{exl} / (\beta_0 c \cdot B\rho)`
    - Same
  * - Position change :math:`\Delta x`
    - 0
    - :math:`p_x L / p_z + \Delta p_x \cdot L / (2 p_z)`
  * - Septum detection
    - Single point (entrance position)
    - Entrance position classification

For :math:`\Delta p_x = 30\,\mathrm{mrad}`, :math:`L = 0.5\,\mathrm{m}`, :math:`p_z \approx 1` (typical parameters), the position difference:

.. math::

  \frac{\Delta p_x \cdot L}{2 p_z} \approx \frac{0.03 \times 0.5}{2} = 7.5\,\mathrm{mm}

This magnitude is not negligible in injection/extraction scenarios (septum gaps are typically on the order of mm), so the thick lens mode is recommended.


Septum Logic
------------

The core physical characteristic of the electrostatic separator is that **not all particles experience the electric field**. The septum (cutting plate/wire) divides the aperture into:

- **Field-free region**: the region where the circulating beam resides; particles are unaffected by the electric field and undergo pure drift
- **Field region**: the region where the injected/extracted beam resides; particles are deflected by the electric field
- **Plate/wire region** (within septum thickness): particles strike the cutting plate/wire and are marked as lost

Determination Rules
~~~~~~~~~~~~~~~~~~~

The septum direction is directly determined by which field component is nonzero:

- :math:`E_x \neq 0` (``exl`` nonzero): plates are vertical, the septum is a vertical line, and the :math:`x` coordinate is checked
- :math:`E_y \neq 0` (``eyl`` nonzero): plates are horizontal, the septum is a horizontal line, and the :math:`y` coordinate is checked

The sign of ``septum_x_position`` determines which side has the field—**the field is always on the side away from the beam center**:

.. list-table::
  :header-rows: 1
  :widths: 25 25 25 25

  * - septum_x_position
    - Field-free region (circulating beam)
    - Plate/wire region
    - Field region (deflected beam)
  * - :math:`> 0`
    - :math:`x \le s_x`
    - :math:`s_x < x \le s_x + t`
    - :math:`x > s_x + t`
  * - :math:`< 0`
    - :math:`x \ge s_x`
    - :math:`s_x - t \le x < s_x`
    - :math:`x < s_x - t`

where :math:`s_x` is ``septum x position`` and :math:`t` is ``septum thickness``. The rules for ``septum y position`` are analogous, replacing :math:`x` with :math:`y`.

.. raw:: html

  <div style="text-align: center">
  <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="400" height="300" fill="#1a1a2e"/>
    <!-- axes -->
    <line x1="20" y1="150" x2="380" y2="150" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <line x1="200" y1="20" x2="200" y2="280" stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="385" y="165" fill="#888" font-size="12" font-family="monospace">x</text>
    <text x="206" y="18" fill="#888" font-size="12" font-family="monospace">y</text>
    <!-- septum plate (loss zone) -->
    <rect x="260" y="30" width="16" height="240" fill="#e94560" fill-opacity="0.3" stroke="#e94560" stroke-width="1.5"/>
    <!-- field region (right of plate) -->
    <rect x="276" y="30" width="104" height="240" fill="#00d2ff" fill-opacity="0.08" stroke="none"/>
    <!-- field-free region (left of septum) -->
    <text x="140" y="90" fill="#00d2ff" font-size="13" font-family="monospace">Field-free</text>
    <text x="140" y="108" fill="#00d2ff" font-size="11" font-family="monospace">Circulating beam</text>
    <!-- loss zone label -->
    <text x="255" y="22" fill="#e94560" font-size="11" font-family="monospace">Plate/Wire</text>
    <!-- field region label -->
    <text x="305" y="90" fill="#f5a623" font-size="13" font-family="monospace">Field region</text>
    <text x="305" y="108" fill="#f5a623" font-size="11" font-family="monospace">Deflected beam</text>
    <!-- septum position marker -->
    <line x1="260" y1="140" x2="260" y2="160" stroke="#e94560" stroke-width="2"/>
    <text x="245" y="175" fill="#e94560" font-size="11" font-style="italic" font-family="monospace">s</text>
    <text x="245" y="188" fill="#e94560" font-size="10" font-family="monospace">x</text>
    <!-- thickness marker -->
    <line x1="260" y1="265" x2="276" y2="265" stroke="#e94560" stroke-width="1.5" stroke-dasharray="2,2"/>
    <text x="262" y="278" fill="#e94560" font-size="10" font-style="italic" font-family="monospace">t</text>
    <!-- E field arrow in field region -->
    <line x1="310" y1="200" x2="370" y2="200" stroke="#f5a623" stroke-width="2" marker-end="url(#arrowhead)"/>
    <defs>
      <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" fill="#f5a623"/>
      </marker>
    </defs>
    <text x="330" y="195" fill="#f5a623" font-size="12" font-style="italic" font-family="monospace">E</text>
    <text x="322" y="215" fill="#f5a623" font-size="10" font-family="monospace">x</text>
  </svg>
  </div>

.. note::

  - If ``septum x position`` is not provided (``None``) but ``exl`` is nonzero, all particles experience the electric field (no septum detection). This corresponds to the use case of a uniform-field correction element
  - ``septum thickness`` defaults to 0, in which case the plate/wire region width is zero, and particles are either in the field-free region or the field region


Tilt Rotation
-------------

The ``tilt`` parameter implements a roll rotation of the element about the :math:`s` axis. It follows the MAD-X convention: **positive angle represents clockwise rotation** (looking along the :math:`+s` direction).

.. raw:: html

  <div style="text-align: center">
  <svg width="350" height="250" xmlns="http://www.w3.org/2000/svg">
    <rect width="350" height="250" fill="#1a1a2e"/>
    <!-- s axis (into page, marked as dot) -->
    <circle cx="175" cy="125" r="4" fill="#888"/>
    <text x="182" y="120" fill="#888" font-size="12" font-family="monospace">s</text>
    <!-- original (untilted) frame: x and y axes -->
    <line x1="175" y1="125" x2="315" y2="125" stroke="#555" stroke-width="1" stroke-dasharray="4,3"/>
    <line x1="175" y1="125" x2="175" y2="30" stroke="#555" stroke-width="1" stroke-dasharray="4,3"/>
    <text x="320" y="130" fill="#555" font-size="11" font-family="monospace">x</text>
    <text x="180" y="28" fill="#555" font-size="11" font-family="monospace">y</text>
    <!-- tilted frame (clockwise by tilt) -->
    <line x1="175" y1="125" x2="305" y2="65" stroke="#00d2ff" stroke-width="2"/>
    <line x1="175" y1="125" x2="95" y2="35" stroke="#00d2ff" stroke-width="2"/>
    <text x="310" y="60" fill="#00d2ff" font-size="12" font-family="monospace">x'</text>
    <text x="80" y="32" fill="#00d2ff" font-size="12" font-family="monospace">y'</text>
    <!-- rotation arc -->
    <path d="M 250 125 A 75 75 0 0 0 225 55" fill="none" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="3,2"/>
    <text x="258" y="95" fill="#f5a623" font-size="12" font-style="italic" font-family="monospace">tilt</text>
    <!-- clockwise arrow on arc -->
    <polygon points="225,55 232,58 228,50" fill="#f5a623"/>
  </svg>
  </div>

Tilt does not affect the choice of integration method—it only performs an instantaneous coordinate transformation at the entrance and exit:

::

  Entrance:  Clockwise rotation of :math:`(x, y, p_x, p_y)` by :math:`+\varphi`  → Enter element natural coordinate system
  Interior:  DKD or thin lens, tracked in the natural coordinate system (field along :math:`x'`, septum along :math:`x'`)
  Exit:      Counterclockwise rotation of :math:`(x, y, p_x, p_y)` by :math:`-\varphi`  → Return to laboratory coordinate system

Clockwise rotation matrix:

.. math::

  x' = x \cos\varphi - y \sin\varphi

.. math::

  y' = x \sin\varphi + y \cos\varphi

.. math::

  p_x' = p_x \cos\varphi - p_y \sin\varphi

.. math::

  p_y' = p_x \sin\varphi + p_y \cos\varphi

Drift itself is coordinate-independent (free-space propagation does not depend on the transverse coordinate direction), the kick is along :math:`x'` in the natural coordinate system, and the septum is a straight line at :math:`x' = \mathrm{const}` in the natural coordinate system. All physics is completed in the natural coordinate system.


Overall Tracking Flow
---------------------

::

  ====== Thin lens (length = 0) ======

    1. Tilt rotation (if any)
    2. Classify particles: field-free / field region / striking plate
    3. Field-free region: no operation (position and momentum unchanged)
    4. Field region: pure kick (Δpx = exl / (β₀c·Bρ), Δpy = eyl / (β₀c·Bρ))
    5. Striking plate: tag set negative, record lost_position/lost_turn
    6. Tilt rotation back (if any)

  ====== Thick lens (length > 0) ======

    1. Tilt rotation (if any)
    2. Classify particles: field-free / field region / striking plate
    3. Field-free region: pure Drift(L)
    4. Field region: DKD
       Drift(L/2) → Kick → Drift(L/2)
    5. Striking plate: tag set negative, record lost_position/lost_turn
    6. Tilt rotation back (if any)
    7. z coordinate wrap to [-C/2, C/2)


Interface Parameters
--------------------

.. list-table::
  :header-rows: 1
  :widths: 22 28 10 10 30

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
  * - ``name``
    - ``name``
    - str
    - -
    - Element name
  * - ``length``
    - ``length (m)``
    - float
    - m
    - Plate length (:math:`\ge 0`; :math:`= 0` for thin lens)
  * - ``ex``
    - ``ex (v/m)``
    - float
    - V/m
    - Horizontal electric field strength, default 0
  * - ``ey``
    - ``ey (v/m)``
    - float
    - V/m
    - Vertical electric field strength, default 0
  * - ``exl``
    - ``exl (v)``
    - float
    - V
    - Horizontal integrated field :math:`E_x L`, default derived from :math:`E_x \cdot L`
  * - ``eyl``
    - ``eyl (v)``
    - float
    - V
    - Vertical integrated field :math:`E_y L`, default derived from :math:`E_y \cdot L`
  * - ``tilt``
    - ``tilt (rad)``
    - float
    - rad
    - Roll angle about the :math:`s` axis, positive is clockwise, default 0
  * - ``septum_x_position``
    - ``septum x position (m)``
    - float
    - m
    - Septum position in :math:`x` direction (effective when ``exl`` is nonzero), default ``None``
  * - ``septum_y_position``
    - ``septum y position (m)``
    - float
    - m
    - Septum position in :math:`y` direction (effective when ``eyl`` is nonzero), default ``None``
  * - ``septum_thickness``
    - ``septum thickness (m)``
    - float
    - m
    - Septum plate/wire thickness, default 0
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

  - Two input methods for field strength (``ex`` / ``ey``) and integrated field (``exl`` / ``eyl``):

    - Thick lens: when ``ex`` / ``ey`` are provided, :math:`\mathrm{exl} = E_x \cdot L` is automatically computed; when ``exl`` / ``eyl`` are provided, :math:`E_x = \mathrm{exl} / L` is automatically derived
    - Thin lens: ``exl`` / ``eyl`` are used directly (``ex`` / ``ey`` are meaningless when :math:`L = 0`)

  - ``ex`` and ``ey`` typically have only one nonzero (horizontal or vertical deflection plate). If both are nonzero simultaneously, a warning is issued
  - ``septum x position`` / ``septum y position`` take effect only when the corresponding field component is nonzero
  - The ``Command`` field should be set to ``elseparator``


Usage Examples
--------------

Thin Lens Horizontal Deflection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example places a thin lens electrostatic separator at :math:`s = 10.0` m with integrated field :math:`\mathrm{exl} = 1 \times 10^5` V:

.. code-block:: json

  {
      "ES1": {
          "S (m)": 10.0,
          "Command": "elseparator",
          "ExL (V)": 1e5
      }
  }

The particle receives a horizontal kick :math:`\Delta p_x = \mathrm{exl} / (\beta_0 c \cdot B\rho)`. Position is unchanged.

Thick Lens Horizontal Deflection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example places a thick lens electrostatic separator at :math:`s = 15.0` m, length 0.5 m, field strength :math:`E_x = 2 \times 10^5` V/m:

.. code-block:: json

  {
      "ES2": {
          "S (m)": 15.0,
          "Command": "elseparator",
          "Length (m)": 0.5,
          "Ex (V/m)": 2e5
      }
  }

The integrated field :math:`\mathrm{exl} = 2 \times 10^5 \times 0.5 = 1 \times 10^5` V. DKD tracking is used; the particle follows a parabolic trajectory.

Injection Separator with Septum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example simulates an injection electrostatic separator with the septum at :math:`x = 5` mm and septum thickness 2 mm:

.. code-block:: json

  {
      "ES3": {
          "S (m)": 20.0,
          "Command": "elseparator",
          "Length (m)": 0.3,
          "Ex (V/m)": 3e5,
          "Septum X Position (m)": 0.005,
          "Septum Thickness (m)": 0.002
      }
  }

Particle classification:

- :math:`x \le 5` mm: field-free region, pure drift (circulating beam)
- :math:`5\,\mathrm{mm} < x \le 7` mm: striking plate/wire, marked as lost
- :math:`x > 7` mm: field region, DKD deflection (injection beam)

Negative-Side Septum (Extraction Scenario)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example simulates an extraction scenario with the septum at :math:`x = -5` mm and the field on the left side:

.. code-block:: json

  {
      "ES4": {
          "S (m)": 25.0,
          "Command": "elseparator",
          "Length (m)": 0.3,
          "Ex (V/m)": -3e5,
          "Septum X Position (m)": -0.005
      }
  }

Particle classification:

- :math:`x \ge -5` mm: field-free region, pure drift (circulating beam)
- :math:`x < -5` mm: field region, DKD deflection (extraction beam)

Vertical Deflection Plate
~~~~~~~~~~~~~~~~~~~~~~~~~

The following example places a vertical deflection plate with only :math:`E_y` nonzero:

.. code-block:: json

  {
      "ES5": {
          "S (m)": 30.0,
          "Command": "elseparator",
          "Length (m)": 0.4,
          "Ey (V/m)": 2.5e5,
          "Septum Y Position (m)": 0.005
      }
  }

Particles are deflected in the :math:`y` direction. Septum detection is along the :math:`y` direction.

Deflection Plate with Tilt
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example places a deflection plate rotated 30 degrees clockwise about the :math:`s` axis:

.. code-block:: json

  {
      "ES6": {
          "S (m)": 35.0,
          "Command": "elseparator",
          "Length (m)": 0.5,
          "Ex (V/m)": 2e5,
          "Tilt (rad)": 0.5236,
          "Septum X Position (m)": 0.005
      }
  }

The element's natural coordinate system is rotated 30 degrees clockwise, and both the field and septum are defined in the rotated coordinate system. Entrance rotation → tracking → exit rotation back.

Zero-Field Degeneration
~~~~~~~~~~~~~~~~~~~~~~~

The following example has zero field strength, degenerating to a pure drift (thick lens) or marker (thin lens):

.. code-block:: json

  {
      "ES7": {
          "S (m)": 40.0,
          "Command": "elseparator",
          "Length (m)": 0.5
      }
  }

When :math:`\mathrm{exl} = \mathrm{eyl} = 0`, all particles undergo pure drift with no deflection.


Application Scenarios
---------------------

- **Beam injection**: Place electrostatic separators in the injection section to deflect the injected beam to match the main ring closed orbit. The septum separates the circulating beam from the injected beam, deflecting only the injected particles
- **Beam extraction**: Place electrostatic separators at the extraction point to deflect the extracted beam into the extraction channel. The septum ensures the circulating beam is unaffected
- **Orbit correction**: Without a septum (``septum x position = None``), the electrostatic separator can serve as a uniform-field correction element, applying the same kick to all particles
- **Low-energy beam deflection**: In low-energy sections where :math:`\beta\gamma` is small, electrostatic deflection is more efficient than magnetic deflection (electric force is independent of velocity, magnetic force is proportional to velocity), commonly used in low-energy injection lines
- **Fast extraction systems**: Electrostatic separators have fast response times (nanosecond-level pulses), suitable for fast extraction and bunch-by-bunch extraction


References
----------

- MAD-X User's Guide, "ELSEPARATOR" section (``ex`` / ``ey`` / ``ex_l`` / ``ey_l`` / ``tilt`` definitions)
- Xsuite source code: ``xtrack/mad_loader.py`` (``convert_elseparator = convert_drift_like``, xsuite does not yet implement an independent elseparator)
- Wiedemann, H., "Particle Accelerator Physics", Ch. 4 (equivalence between electric and magnetic deflection)
- Conte, M. & MacKay, W.W., "An Introduction to the Physics of Particle Accelerators", Ch. 7 (electrostatic separators in injection and extraction)
