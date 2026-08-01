RFCavity
========

This module describes the PASS RF cavity element **RFCavity**, used to simulate the longitudinal acceleration of charged particles in a radio-frequency electric field. The RF cavity is one of the most critical elements in synchrotrons, linear accelerators, and cyclotrons, providing energy gain to particles through a periodic electric field, maintaining the synchronous particle energy and controlling longitudinal beam dynamics (synchrotron oscillation, bunch compression, longitudinal acceptance, etc.).

The PASS RF cavity is modeled as a **thin lens** (``length = 0``) instantaneous energy kick, using the **exact relativistic energy-momentum relation** and a **moving reference frame**. The longitudinal transformation abandons the traditional first-order linearization approximation in favor of the exact :math:`E^2 = p^2 + m_0^2` relation, avoiding the :math:`O(\delta^2)` error from linearization and the :math:`\beta_1/\beta_0` factor issue in reference frame transformation.

**Code Location**

- Source file: ``PASS/commands/element/rfcavity.py``
- Class name: ``RFCavity`` (inherits from ``Command``)
- Registration name: ``rfcavity``
- Key features:

  - Thin lens model, instantaneous energy kick
  - Exact relativistic energy-momentum transformation (:math:`E^2 = p^2 + m_0^2`), no linearization approximation
  - Moving reference frame: beam reference energy updated each turn (:math:`E_k, \gamma, \beta, p_0, B\rho`)
  - Transverse momentum rescaling (adiabatic damping): :math:`p_x \leftarrow p_x \cdot \beta_0\gamma_0 / (\beta_1\gamma_1)`
  - Normalized emittance :math:`\epsilon_N = \beta\gamma\epsilon` strictly conserved
  - Supports dp acceptance (longitudinal aperture) check
  - Supports both fixed-value and TFS file (Ramping) parameter input methods
  - Supports multi-turn acceleration simulation
  - Harmonic number is a property of the cavity state, unified for all bunches


Coordinate Convention
---------------------

PASS uses normalized curvilinear coordinates. The six-dimensional phase-space variables are :math:`(x, p_x, y, p_y, \zeta, \delta)`:

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

Physical Nature of the RF Cavity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RF cavity produces a longitudinal (along the beam direction) oscillating electric field :math:`E_s(t) = E_0 \sin(\omega_{\text{rf}} t + \varphi_s)`. The energy gain a particle receives when passing through the cavity depends on the moment the particle arrives at the cavity, i.e., the particle's longitudinal position :math:`\zeta` determines the RF phase it experiences.

Let the machine circumference be :math:`C`, machine radius :math:`R = C / (2\pi)`, harmonic number :math:`h`, RF voltage :math:`V`, synchronous phase :math:`\varphi_s`. The azimuthal angle corresponding to the particle's longitudinal position :math:`\zeta` is :math:`\theta = \zeta / R`, and its RF phase is:

.. math::

  \varphi_{\text{particle}} = \varphi_s - h \cdot \theta + \varphi_{\text{off}}

where :math:`\varphi_{\text{off}}` is an additional phase offset (see below).


Purpose of phi_offset
~~~~~~~~~~~~~~~~~~~~~

``phi_offset`` is a constant phase offset applied to all particles, used to shift the time reference of the RF waveform. Its main uses include:

1. **Multi-cavity phase alignment**: When multiple RF cavities are distributed around the ring and the cavity spacing is not an integer multiple of the RF wavelength, each cavity needs an independent phase correction to maintain synchronism.
2. **Multi-harmonic systems**: When cavities with different harmonic numbers share the same frequency reference, :math:`\varphi_{\text{off}}` enables independent phase adjustment for each cavity.
3. **Phase trim**: Fine-tuning the cavity phase during operation without changing the definition of the synchronous particle (``phase`` still defines the energy gain of the synchronous particle).

Physically, :math:`\varphi_{\text{off}}` rotates the entire :math:`\sin` curve so that the particle's actual phase becomes :math:`\varphi_s + \varphi_{\text{off}} - h\theta`.


Even Harmonic Compensation
~~~~~~~~~~~~~~~~~~~~~~~~~~

When injecting multiple bunches, PASS uses a symmetric offset formula to place each bucket symmetrically about :math:`z=0` (see the multi-bunch longitudinal offset description in :doc:`../injection`). For even harmonic number :math:`h`, this symmetric offset introduces an equivalent :math:`180^\circ` phase flip.

To compensate for this flip, the RF cavity automatically applies a :math:`C/(2h)` offset to :math:`z` when computing the particle phase for even :math:`h`:

.. math::

  z_{\text{eff}} = \begin{cases} z + \frac{C}{2h} & h \text{ is even} \\ z & h \text{ is odd} \end{cases}

  \varphi_{\text{particle}} = \varphi_s + \varphi_{\text{off}} - h \cdot \frac{z_{\text{eff}}}{R}

This compensation ensures that the synchronous particle in all buckets correctly sees the synchronous phase :math:`\varphi_s`. Odd :math:`h` does not need compensation (the offset is an integer multiple of :math:`2\pi`, and the :math:`\sin` function is automatically unchanged).


Energy Kick
~~~~~~~~~~~

The energy gain for each particle is:

.. math::

  \Delta E_{\text{kick}} = \frac{q}{A} \cdot V \cdot \sin(\varphi_{\text{particle}})

where :math:`q/A` is the charge-to-mass ratio. The energy gain of the synchronous particle (:math:`\zeta = 0`) is:

.. math::

  \Delta E_{\text{syn}} = \frac{q}{A} \cdot V \cdot \sin(\varphi_s)

:math:`\Delta E_{\text{syn}}` reflects the net acceleration effect of the RF cavity.


First-Order Linearization Approximation and Its Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The traditional method uses a first-order Taylor expansion when converting the energy deviation :math:`dE` to momentum deviation :math:`\delta`:

.. math::

  dE \approx \delta \cdot \beta^2 \cdot E_{\text{total}}

where :math:`dE = E - E_0` is the energy deviation of the particle relative to the reference particle. This is the first-order approximation of the exact relation

.. math::

  E = \sqrt{(p_0 (1+\delta) c)^2 + (m_0 c^2)^2}

at :math:`\delta \to 0`. The expansion error is :math:`O(\delta^2)`.

**Problem 1: Poor precision at large** ``delta``

When :math:`\delta` is large (e.g., :math:`\pm 30\%` injection acceptance), the :math:`O(\delta^2)` term can reach :math:`\sim 0.01`, far exceeding floating-point precision and introducing a non-negligible systematic bias.

**Problem 2: Reference frame transformation introduces an extra** ``beta1/beta0`` **factor**

In the moving reference frame, the reference energy changes from :math:`E_0` to :math:`E_1 = E_0 + \Delta E_{\text{syn}}`. The traditional method assumes :math:`\delta` is invariant under the reference frame transformation, deriving as follows:

Starting from :math:`dE = \delta \cdot \beta^2 \cdot E_{\text{total}}`, in the old frame :math:`dE_0 = \delta \cdot \beta_0^2 \cdot E_0`, and in the new frame :math:`dE_1 = \delta \cdot \beta_1^2 \cdot E_1`. If :math:`\delta` is assumed invariant, then:

.. math::

  dE_1 = dE_0 \cdot \frac{\beta_1^2 E_1}{\beta_0^2 E_0}

Further approximating :math:`\beta_1^2 E_1 / (\beta_0^2 E_0)` as :math:`\beta_1 / \beta_0` (valid only when :math:`E_1 \approx E_0`, i.e., weak acceleration):

.. math::

  dE_1 = \frac{\beta_1}{\beta_0} \cdot (dE_0 + \Delta E_{\text{non-syn}} - \Delta E_{\text{syn}})

This is a **triple approximation**:

1. **First-order linearization**: :math:`dE \approx \delta \cdot \beta^2 \cdot E_{\text{total}}` (truncating :math:`O(\delta^2)`)
2. **delta invariance assumption**: :math:`\delta` is invariant under reference frame transformation (in reality :math:`\delta = p/p_0 - 1` depends on :math:`p_0`)
3. **Weak acceleration approximation**: :math:`\beta_1^2 E_1 / (\beta_0^2 E_0) \approx \beta_1/\beta_0` (valid only when :math:`E_1 \approx E_0`)

In strong acceleration scenarios (e.g., energy changing by several times from injection to extraction), the 3rd approximation significantly fails.


Exact Energy-Momentum Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PASS abandons all the above approximations and starts directly from the exact relativistic relation. All physical quantities use natural units (:math:`c = 1`); :math:`m_0`, :math:`p_0`, and :math:`E` are all in eV. The energy-momentum relation is:

.. math::

  E^2 = p^2 + m_0^2

The tracking chain is as follows:

**Before kick**: particle momentum :math:`p_{\text{old}} = p_{0,\text{old}} \cdot (1 + \delta)`, absolute total energy:

.. math::

  E_{\text{old}} = \sqrt{p_{\text{old}}^2 + m_0^2}

**Apply kick**:

.. math::

  E_{\text{new}} = E_{\text{old}} + \Delta E_{\text{kick}}

**Recover momentum**:

.. math::

  p_{\text{new}} = \sqrt{E_{\text{new}}^2 - m_0^2}

**Compute new** :math:`\delta` (relative to the new reference momentum :math:`p_{0,\text{new}}`):

.. math::

  \delta_{\text{new}} = \frac{p_{\text{new}}}{p_{0,\text{new}}} - 1

**Why the exact method does not need the** ``beta1/beta0`` **factor**

The exact method directly tracks the absolute total energy :math:`E_{\text{particle}}` and absolute momentum :math:`p_{\text{particle}}` of each particle. The RF kick changes the energy (:math:`E_{\text{new}} = E_{\text{old}} + \Delta E`), then momentum is exactly recovered from energy (:math:`p = \sqrt{E^2 - m_0^2}`), and finally divided by the new reference momentum to obtain :math:`\delta`.

In this process:

- No :math:`dE \to \delta` linearization is needed (the exact :math:`E \to p` relation is used directly)
- No assumption of :math:`\delta` invariance is needed (:math:`\delta` is computed directly from :math:`p_{\text{particle}} / p_{0,\text{new}}`)
- No :math:`\beta_1/\beta_0` scaling factor is needed (the reference frame transformation is implicitly included in :math:`p_{0,\text{new}}`)

Thus all three approximations are bypassed, and the :math:`\beta_1/\beta_0` factor naturally does not appear.

This transformation is **completely exact**, with no linearization approximation. It requires only two ``sqrt`` operations (numpy vectorized, negligible cost), and is suitable for large :math:`\delta` and strong acceleration scenarios.


Moving Reference Frame
~~~~~~~~~~~~~~~~~~~~~~

PASS uses a moving reference frame: after each RF kick, the beam reference energy is updated to a new value including :math:`\Delta E_{\text{syn}}`:

.. math::

  E_{\text{total},1} = E_{\text{total},0} + \Delta E_{\text{syn}}

.. math::

  \gamma_1 = \frac{E_{\text{total},1}}{m_0}

.. math::

  \beta_1 = \sqrt{1 - \frac{1}{\gamma_1^2}}

.. math::

  p_{0,\text{new}} = \gamma_1 m_0 \beta_1

.. math::

  E_{k,1} = E_{\text{total},1} - m_0

The :math:`\delta` of the synchronous particle is always maintained near 0, avoiding the numerical precision issues caused by continuously growing :math:`\delta` in a fixed reference frame.


Transverse Momentum Rescaling and Adiabatic Damping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RF kick is a purely longitudinal energy gain and does not change the absolute transverse momentum :math:`P_x` of the particle. However, since the reference momentum :math:`P_0` increases, the normalized transverse momentum :math:`p_x = P_x / P_0` must be rescaled:

.. math::

  p_x \leftarrow p_x \cdot \frac{p_{0,\text{old}}}{p_{0,\text{new}}} = p_x \cdot \frac{\beta_0 \gamma_0}{\beta_1 \gamma_1}

.. math::

  p_y \leftarrow p_y \cdot \frac{\beta_0 \gamma_0}{\beta_1 \gamma_1}

This scaling is **exact**, not an approximation (since :math:`p_0 c = \beta \gamma m_0 c^2`, so :math:`p_{0,\text{old}} / p_{0,\text{new}} = \beta_0 \gamma_0 / (\beta_1 \gamma_1)`).

**Physical meaning**: The normalized emittance :math:`\epsilon_N = \beta\gamma\epsilon` is an adiabatic invariant (Liouville's theorem). When :math:`p_0` increases, the geometric emittance :math:`\epsilon` shrinks as :math:`1/(\beta\gamma)`, i.e., adiabatic damping. The transverse momentum rescaling ensures that :math:`\epsilon_N` is strictly conserved:

.. math::

  \epsilon_{\text{geom}}^{\text{new}} = \epsilon_{\text{geom}}^{\text{old}} \cdot \frac{\beta_0 \gamma_0}{\beta_1 \gamma_1} = \frac{\epsilon_N}{\beta_1 \gamma_1}


Tracking Flow
-------------

.. code-block:: text

  Input: z, dp(=δ), px, py, tag, bunch parameters (β₀, γ₀, m₀, q/A, Ek, p₀, C)

  1. Compute RF phase
     If h is even: z_eff = z + C/(2h)   (even harmonic compensation)
     If h is odd:  z_eff = z
     θ = z_eff / R                       (R = C/2π)
     φ_particle = phase + φ_off - h·θ

  2. Energy kick
     ΔE_kick = (q/A)·V·sin(φ_particle)  [per particle]
     ΔE_syn  = (q/A)·V·sin(phase)       [scalar]

  3. Update beam reference (moving reference frame)
     E_total1 = E_total0 + ΔE_syn
     γ₁ = E_total1 / m₀
     β₁ = √(1 - 1/γ₁²)
     p₀_new = γ₁·m₀·β₁
     Ek₁ = E_total1 - m₀

  4. Exact δ update
     p_old = p₀_old·(1+δ)
     E_old = √(p_old² + m₀²)
     E_new = E_old + ΔE_kick
     p_new = √(E_new² - m₀²)
     δ_new = p_new / p₀_new - 1

  5. Transverse momentum rescaling (adiabatic damping)
     scale = β₀γ₀ / (β₁γ₁)
     px *= scale
     py *= scale

  6. dp acceptance check (exceeds → mark as lost)

  7. z-wrap to [-C/2, C/2)

  8. Update lost particle information


Interface Parameters
--------------------

.. list-table:: RF Parameters
  :header-rows: 1
  :widths: 20 22 10 10 38

  * - Property
    - JSON key
    - Type
    - Default
    - Description
  * - ``voltage``
    - ``voltage (v)``
    - float
    - 0.0
    - RF voltage (V)
  * - ``harmonic``
    - ``harmonic``
    - int
    - 1
    - Harmonic number :math:`h`
  * - ``phase``
    - ``phase (rad)``
    - float
    - 0.0
    - Synchronous phase :math:`\varphi_s` (rad)
  * - ``phi_offset``
    - ``phi offset (rad)``
    - float
    - 0.0
    - Additional phase offset (rad), used for multi-cavity phase alignment and phase trim
  * - ``_rf_table``
    - ``rf data file``
    - str
    - None
    - Ramping data file path (TFS format); when provided, overrides fixed-value parameters. Each row corresponds to one turn; required column names: ``HARMONIC``, ``VOLTAGE``, ``PHASE``, ``PHI_OFFSET``
  * - ``is_enabled``
    - ``is enabled``
    - bool
    - True
    - On/off switch

.. list-table:: Longitudinal Aperture Parameters
  :header-rows: 1
  :widths: 20 22 10 10 38

  * - Property
    - JSON key
    - Type
    - Default
    - Description
  * - ``dp_aperture_lower``
    - ``dp aperture[0]``
    - float
    - -1.0
    - dp acceptance lower bound
  * - ``dp_aperture_upper``
    - ``dp aperture[1]``
    - float
    - 1.0
    - dp acceptance upper bound

.. list-table:: General Parameters
  :header-rows: 1
  :widths: 20 22 10 10 38

  * - Property
    - JSON key
    - Type
    - Default
    - Description
  * - ``s``
    - ``s (m)``
    - float
    - Required
    - Longitudinal position of the element in the beamline
  * - ``cmd_name``
    - ``name``
    - str
    - Required
    - Element name
  * - ``aperture_type``
    - ``aperture type``
    - str
    - ``off``
    - Transverse aperture type
  * - ``aperture_value``
    - ``aperture value``
    - list
    - ``[]``
    - Transverse aperture parameters


Ramping Data File
-----------------

When RF parameters need to vary with turn number (e.g., energy ramping), a TFS-format data file can be provided. TFS (Table File System) is a tabular format with metadata; the file can contain headers, comments, and other documentation. Columns are identified by column name rather than position.

Required column names (case-insensitive, order-insensitive):

- ``HARMONIC`` — harmonic number
- ``VOLTAGE`` — RF voltage (V)
- ``PHASE`` — synchronous phase (rad)
- ``PHI_OFFSET`` — additional phase offset (rad)

Column names are automatically converted to lowercase when reading, so any case combination such as ``Harmonic``, ``voltage``, ``phase``, etc. works. Each row corresponds to one turn (row 0 = turn 0). If the turn number exceeds the number of file rows, the last row's data is used. The file can also contain metadata header information such as ``TITLE``, ``DATE``, etc., which is automatically parsed by the ``tfs-pandas`` library.


Usage Examples
--------------

Example 1: Basic Acceleration Cavity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
    "RFCavity_1": {
      "S (m)": 0.0,
      "Command": "RFElement",
      "voltage (v)": 100000,
      "harmonic": 1,
      "phase (rad)": 0.3,
      "is enabled": true
    }
  }

The synchronous particle gains :math:`\Delta E = q \cdot V \cdot \sin(0.3) \approx 29552` eV of energy per turn.

Example 2: Cavity with dp Acceptance and Phase Offset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
    "RFCavity_2": {
      "S (m)": 500.0,
      "Command": "RFElement",
      "voltage (v)": 200000,
      "harmonic": 4,
      "phase (rad)": 0.5236,
      "phi offset (rad)": 0.05,
      "dp aperture": [-0.02, 0.02],
      "is enabled": true
    }
  }

:math:`\varphi_s = \pi/6 = 0.5236` rad, 4th harmonic, dp acceptance :math:`\pm 2\%`, additional phase offset 0.05 rad.

Example 3: Ramping Cavity (TFS File Input)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
    "RFCavity_3": {
      "S (m)": 0.0,
      "Command": "RFElement",
      "rf data file": "D:/PASS/para/rf_data.tfs",
      "is enabled": true
    }
  }

Each row in the TFS file specifies the ``HARMONIC``, ``VOLTAGE``, ``PHASE``, and ``PHI_OFFSET`` for that turn.

Example 4: Disabled Cavity
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
    "RFCavity_4": {
      "S (m)": 0.0,
      "Command": "RFElement",
      "voltage (v)": 100000,
      "harmonic": 1,
      "phase (rad)": 0.3,
      "is enabled": false
    }
  }

When ``is_enabled = false``, the cavity performs no operation (no-op).


Application Scenarios
---------------------

1. **Synchrotron acceleration**: The acceleration process from injection energy to extraction energy; the moving reference frame keeps :math:`\delta` at a small magnitude, providing good numerical precision

2. **Longitudinal beam dynamics**: Synchrotron oscillation, bunch stretching/compression, longitudinal emittance control

3. **Multi-harmonic acceleration**: Multiple RF cavities in series with different harmonic numbers for bunch shaping

4. **Energy ramping**: Voltage/phase variation with turn number via TFS file input, simulating real accelerator operation scenarios

5. **Longitudinal acceptance studies**: Setting longitudinal acceptance through dp aperture parameters, studying beam loss boundaries


Verification Tests
------------------

``tests/test_rf_verification.py`` — 17 groups totaling 25 tests, all passing:

1. Synchronous particle :math:`\delta \approx 0` (precision :math:`< 10^{-12}`)
2. Synchronous energy gain :math:`\Delta E = (q/A) V \sin(\varphi_s)`
3. Exact energy-momentum relation :math:`E^2 = p^2 + m_0^2`
4. Phase dependence (particles with different :math:`\zeta` receive different kicks)
5. Moving reference frame (beam reference energy correctly updated)
6. Adiabatic damping (:math:`p_x` scaled by :math:`\beta_0\gamma_0/(\beta_1\gamma_1)`)
7. Normalized emittance conservation (relative error :math:`< 10^{-15}`)
8. Dead particles not kicked
9. Zero voltage degenerates to no-op
10. dp acceptance check
11. Multi-turn acceleration (10 turns, energy correct, :math:`\delta` still :math:`\sim 0`)
12. Full comparison with independent reference implementation (:math:`< 10^{-10}`)
13. Low :math:`\gamma` (non-ultrarelativistic, :math:`\beta = 0.417`)
14. Large :math:`\delta` (:math:`\pm 30\%`, nonlinear regime)
15. Ions (:math:`q/A \neq 1`)
16. Phase offset :math:`\varphi_{\text{off}}`
17. Disabled cavity (``is_enabled = false``)
