Multipole
=========

This module describes the PASS general multipole element **Multipole**, used to simulate the motion of charged particles in an arbitrary-order multipole magnet. Unlike single-order elements such as the quadrupole, sextupole, and octupole, the multipole uses ``knl`` / ``ksl`` arrays to simultaneously support arbitrary-order (including mixed-order) multipole components, suitable for field error injection, combined multipole elements, higher-order multipoles, and other scenarios.

The PASS multipole supports both **thick element** (``length > 0``) and **thin lens** (``length = 0``) modes. The thick element uses the exact drift-kick-drift (DKD-exact) symplectic integration scheme, supporting both uniform (2nd-order) and yoshida4 (4th-order) symplectic integrators. The kick uses Horner nested evaluation, which is fully consistent with Xsuite's ``kick_simple_single_coordinates`` at the formula level.

**Code Location**

- Source file: ``PASS/commands/element/multipole.py``
- Class name: ``Multipole`` (inherits from ``Command``)
- Registration name: ``multipole``
- Key features:

  - Supports arbitrary-order multipole components (``knl`` / ``ksl`` arrays, maximum order determined by array length)
  - Supports normal components (``knl``) and skew components (``ksl``) and their combinations
  - Supports thin lens mode (``length = 0``, applies only a multipole kick)
  - Supports thick lens mode (``length > 0``, DKD-exact symplectic integration)
  - Supports uniform (2nd-order leapfrog) and yoshida4 (4th-order Yoshida composition) integrators
  - Horner nested evaluation, vectorized implementation, no per-particle branching
  - Zero field (all ``knl`` / ``ksl`` components zero) automatically degenerates to a pure drift
  - Supports aperture check
  - Single-order degeneration is particle-by-particle consistent with quadrupole/sextupole/octupole


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


Multipole Field and Normalized Strength
---------------------------------------

The magnetic field of a general multipole magnet is expanded as a Taylor series in the transverse plane. In complex notation:

.. math::

  B_y + i B_x = \frac{P_0}{q_0} \sum_{n=0}^{N} \frac{K_n}{n!} (x + i y)^n

where :math:`K_n` is the :math:`n`-th order normalized multipole strength (unit :math:`\text{m}^{-n-1}`), :math:`N` is the maximum order, and :math:`1/n!` is the natural coefficient of the Taylor expansion.

Expanding the first few orders:

.. list-table::
  :header-rows: 1
  :widths: 10 15 25 50

  * - Order :math:`n`
    - :math:`n!`
    - Element Type
    - Field Expression
  * - 0
    - 1
    - Dipole
    - :math:`B_y + i B_x = \frac{P_0}{q_0} K_0`
  * - 1
    - 1
    - Quadrupole
    - :math:`B_y + i B_x = \frac{P_0}{q_0} K_1 (x + i y)`
  * - 2
    - 2
    - Sextupole
    - :math:`B_y + i B_x = \frac{P_0}{q_0} \frac{K_2}{2} (x + i y)^2`
  * - 3
    - 6
    - Octupole
    - :math:`B_y + i B_x = \frac{P_0}{q_0} \frac{K_3}{6} (x + i y)^3`

The normalized integrated strength is defined as:

.. math::

  K_{nL} = K_n \cdot L, \qquad K_{nsL} = K_{ns} \cdot L

where :math:`L` is the magnet length, :math:`K_{nL}` is the normal component, and :math:`K_{nsL}` is the skew component. In PASS, the user specifies :math:`[K_{0L}, K_{1L}, K_{2L}, \ldots]` through the ``knl`` array and :math:`[K_{0sL}, K_{1sL}, K_{2sL}, \ldots]` through the ``ksl`` array.

.. note::

  The ``KNL`` / ``KSL`` values exported by MAD-X are fully consistent with PASS's ``knl`` / ``ksl`` definitions, both being integrated strengths :math:`K_{nL}`, and can be used directly without manually computing factorials. The :math:`1/n!` is handled automatically by the Horner recursion inside the code.


Overall Tracking Flow
---------------------

Depending on the magnet length, the multipole has two tracking modes:

**Thin lens mode** (:math:`L = 0`)

::

  ====== Thin lens (length = 0) ======

  Single multipole kick Kick(knl, ksl)
  [Position unchanged, momentum jump only]

**Thick lens mode** (:math:`L > 0`)

::

  ====== Thick lens (length > 0) ======

  Slice 1 → Slice 2 → ... → Slice N
  (Each slice: Drift(ds/2) → Kick(ds) → Drift(ds/2))

  where ds = L / N
  knl_eff = kn * ds, ksl_eff = ks * ds

  If all knl/ksl components are zero: degenerates to a single exact drift Drift(L)

The complete map is:

Thin lens:

.. math::

  \mathcal{M}_{\text{thin}} = \text{Kick}(K_{nL}, K_{nsL})

Thick lens (N slices):

.. math::

  \mathcal{M}_{\text{thick}} = \left[\mathcal{M}_{\text{DKD}}(\Delta s)\right]^N

where the DKD map for each slice is:

.. math::

  \mathcal{M}_{\text{DKD}}(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

.. note::

  - Thin lens mode does not change the particle position coordinates :math:`(x, y, z)`, only applies momentum kicks
  - Chromaticity and other effects in thick lens mode are naturally introduced through the :math:`p_z` expression in exact drift
  - When all ``knl`` / ``ksl`` components are zero, the thick lens degenerates to a pure drift, avoiding meaningless empty kick loops


Physical Derivation
--------------------

Hamiltonian
~~~~~~~~~~~

In the Cartesian coordinate system (multipole has no curvature, :math:`h = 0`), the general multipole Hamiltonian is:

.. math::

  H_{\text{mult}} = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2} + \chi \sum_{n=0}^{N} \frac{K_n}{n!} \operatorname{Re}\left[(x - i y)^n\right]

where the summation term is the potential energy part. Splitting it into the propagation part (exact drift :math:`H_D`) and the kick part (:math:`H_K`):

.. math::

  H_D = \frac{p_\tau}{\beta_0} - p_z

.. math::

  H_K = \chi \sum_{n=0}^{N} \frac{K_n}{n!} \operatorname{Re}\left[(x - i y)^n\right]

Kick Map
~~~~~~~~

From Hamilton's equations :math:`\Delta p_x = -\frac{\partial H_K}{\partial x} \Delta s`, :math:`\Delta p_y = -\frac{\partial H_K}{\partial y} \Delta s`, for the integrated strength :math:`K_{nL} = K_n \cdot \Delta s`:

.. math::

  \Delta p_x = -\chi \sum_{n=0}^{N} \frac{K_{nL}}{n!} \operatorname{Re}\left[(x + i y)^n\right]

.. math::

  \Delta p_y = +\chi \sum_{n=0}^{N} \frac{K_{nsL}}{n!} \operatorname{Im}\left[(x + i y)^n\right]

where the real part of :math:`(x+iy)^n` corresponds to the normal component and the imaginary part corresponds to the skew component.

.. note::

  The complex field convention is :math:`B_y + i B_x = \frac{P_0}{q_0} \sum_n \frac{K_n}{n!} (x+iy)^n` (**without conjugation**). Using the conjugate :math:`\overline{(x+iy)^n}` would lead to a sign error in :math:`\Delta p_y`. This convention has been verified through sextupole cross-validation.


Horner Nested Evaluation
------------------------

The core of the multipole kick is to evaluate the polynomial:

.. math::

  P(z) = \sum_{n=0}^{N} c_n z^n, \qquad z = x + i y

where :math:`c_n = \chi \cdot K_{nL} / n!`. Direct expansion of higher-order terms is computationally expensive and numerically unstable. PASS uses Horner nested evaluation, which is algorithmically consistent with Xsuite's ``kick_simple_single_coordinates`` (``track_magnet_kick.h:182-228``).

The Horner recursion starts from the highest-order coefficient and works downward:

::

  index = order
  dpx_mul = chi * knl[order] / order!     # Highest-order coefficient
  dpy_mul = chi * ksl[order] / order!

  while index > 0:
      zre = dpx_mul * x - dpy_mul * y      # Re[(dpx_mul + i*dpy_mul) * (x + iy)]
      zim = dpx_mul * y + dpy_mul * x      # Im[(dpx_mul + i*dpy_mul) * (x + iy)]
      index -= 1
      dpx_mul = chi * knl[index] / index! + zre
      dpy_mul = chi * ksl[index] / index! + zim

  dpx = -dpx_mul    # px is negated (radian convention)
  dpy = +dpy_mul    # py is not negated

where ``zre`` and ``zim`` are the real and imaginary parts of the complex multiplication :math:`(\text{dpx\_mul} + i \cdot \text{dpy\_mul}) \cdot (x + i y)`.

The final kick is:

.. math::

  \Delta p_x = -\text{dpx\_mul}

.. math::

  \Delta p_y = +\text{dpy\_mul}

Note that :math:`\Delta p_x` is negated (radian convention), and :math:`\Delta p_y` is not.

Expansion Results for Each Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expanding the Horner recursion, the first few orders are:

.. list-table::
  :header-rows: 1
  :widths: 10 50 40

  * - Order
    - :math:`\Delta p_x` (normal component)
    - :math:`\Delta p_y` (normal component)
  * - :math:`n=0`
    - :math:`-\chi K_{0L}`
    - :math:`0`
  * - :math:`n=1`
    - :math:`-\chi K_{1L} \cdot x`
    - :math:`+\chi K_{1L} \cdot y`
  * - :math:`n=2`
    - :math:`-\chi K_{2L}/2 \cdot (x^2 - y^2)`
    - :math:`+\chi K_{2L} \cdot x y`
  * - :math:`n=3`
    - :math:`-\chi K_{3L}/6 \cdot (x^3 - 3xy^2)`
    - :math:`+\chi K_{3L}/6 \cdot (3x^2 y - y^3)`

The skew component kick naturally swaps real and imaginary parts through complex multiplication :math:`i \cdot z^n`: the normal component formula for :math:`\Delta p_x` in the table above is moved to :math:`\Delta p_y`, and the normal component formula for :math:`\Delta p_y` is moved to :math:`\Delta p_x` and negated.

.. note::

  The Horner recursion is general for any order :math:`N`. When the ``knl`` / ``ksl`` arrays have only a single nonzero order component, the multipole degenerates to the corresponding single-order element (quadrupole/sextupole/octupole, etc.), and the kick formula is particle-by-particle consistent with the hardcoded version.


Exact Drift Map
---------------

The drift part uses the exact drift (Table 1.1, map D, Eq. 1.86-1.88), identical to that of the quadrupole/sextupole/octupole:

.. math::

  x \mathrel{+}= \frac{p_x}{p_z} L

.. math::

  y \mathrel{+}= \frac{p_y}{p_z} L

.. math::

  z \mathrel{+}= L \left(1 - \frac{\beta_0}{\beta} \cdot \frac{1+\delta}{p_z}\right)

where:

.. math::

  p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  \beta = \frac{(1+\delta) \beta_0 \gamma_0}{\sqrt{1 + \left[(1+\delta) \beta_0 \gamma_0\right]^2}}

The exact drift preserves the full nonlinearity of :math:`p_z`, naturally introducing chromaticity, higher-order dispersion, and path-length effects.


Symplectic Integrators
----------------------

Uniform (2nd-order leapfrog)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each slice performs Drift-Kick-Drift:

.. math::

  \mathcal{M}_{\text{DKD}}(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

This is a 2nd-order symplectic integrator with truncation error :math:`O(\Delta s^2)`.

Yoshida4 (4th-order composition)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three DKD steps are combined into a 4th-order symplectic integrator:

.. math::

  \mathcal{M}_{\text{Y4}}(\Delta s) = \mathcal{M}_{\text{DKD}}(z_1 \Delta s) \circ \mathcal{M}_{\text{DKD}}(z_0 \Delta s) \circ \mathcal{M}_{\text{DKD}}(z_1 \Delta s)

where the Yoshida coefficients are:

.. math::

  z_1 = \frac{1}{2 - 2^{1/3}} \approx 1.3512

.. math::

  z_0 = 1 - 2 z_1 \approx -1.7024

Truncation error :math:`O(\Delta s^4)`.


Naturally Included Effects
--------------------------

DKD-exact naturally includes all nonlinear effects for an ideal multipole without additional terms:

.. list-table::
  :header-rows: 1
  :widths: 40 60

  * - Effect
    - Source
  * - Natural chromaticity
    - :math:`\delta` dependence of :math:`p_z` in exact drift
  * - Higher-order nonlinear dispersion
    - Complete square root expression of :math:`p_z`
  * - Path-length effects (:math:`R_{56}`, etc.)
    - :math:`z` update in exact drift
  * - Full nonlinearity of each order multipole kick
    - Horner recursion preserves all terms of :math:`(x+iy)^n`

The only approximation source is the integrator truncation error (:math:`O(\Delta s^2)` for uniform, :math:`O(\Delta s^4)` for yoshida4).


Difference from Xsuite: hxl Curvature Correction
------------------------------------------------

The Xsuite ``Multipole`` element supports the ``hxl`` parameter (horizontal reference orbit rotation angle), used to describe **combined-function magnets**—multipole elements where the reference orbit is bent within the magnet. PASS currently does not implement ``hxl`` and only supports straight magnets (``hxl = 0``).

In Xsuite, ``hxl`` produces three sets of corrections (source code ``track_magnet_kick.h:97-143``):

.. list-table::
  :header-rows: 1
  :widths: 25 30 45

  * - Correction Term
    - Trigger Condition
    - Expression
  * - rot_frame
    - :math:`h_{xl} \neq 0` (independent of knl)
    - :math:`\Delta p_x \mathrel{+}= h_{xl}(1+\delta)`, :math:`\Delta \zeta \mathrel{+}= -\frac{\beta_0}{\beta} h_{xl} x`
  * - k0h correction
    - :math:`h_{xl} \neq 0` and :math:`k_{0L} \neq 0`
    - :math:`\Delta p_x \mathrel{+}= -\chi \, k_{0L} \cdot \frac{h_{xl}}{L} \cdot x`
  * - k1h correction
    - :math:`h_{xl} \neq 0` and :math:`k_{1L} \neq 0`
    - :math:`\Delta p_x \mathrel{+}= \chi \, k_{1L} \cdot \frac{h_{xl}}{L} \cdot (-x^2 + \frac{1}{2}y^2)`, :math:`\Delta p_y \mathrel{+}= \chi \, k_{1L} \cdot \frac{h_{xl}}{L} \cdot xy`

The ``rot_frame`` correction describes the geometric effect of reference orbit deflection and **is independent of field components**—it is triggered whenever :math:`h_{xl} \neq 0`, even if knl/ksl are all zero (pure drift). The ``k0h`` and ``k1h`` corrections are coupling terms between curvature and multipole components, requiring both ``hxl`` and the corresponding knl component to be nonzero.

.. note::

  - PASS multipole sets :math:`h_{xl} = 0`, so all three correction sets are zero, fully consistent with Xsuite's straight magnet (``hxl=0``) at the kick formula level
  - When :math:`h_{xl} = 0`, regardless of knl/ksl values, PASS and Xsuite results are particle-by-particle consistent (verified, precision :math:`< 10^{-12}`)
  - To simulate combined-function magnets (multipole elements with a bent reference orbit), ``hxl`` support needs to be added to PASS; this is a future extension item


Interface Parameters
--------------------

.. list-table::
  :header-rows: 1
  :widths: 20 20 10 10 40

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
  * - ``length``
    - ``length (m)``
    - float
    - Required
    - Magnet length, :math:`= 0` for thin lens
  * - ``knl``
    - ``KiL``
    - list
    - ``[]``
    - Normal component integrated strength array :math:`[K_{0L}, K_{1L}, \ldots]`
  * - ``ksl``
    - ``KiSL``
    - list
    - ``[]``
    - Skew component integrated strength array :math:`[K_{0sL}, K_{1sL}, \ldots]`
  * - ``num_slice``
    - ``num slices``
    - int
    - 1
    - Number of slices for thick lens
  * - ``integrator``
    - ``integrator``
    - str
    - ``adaptive``
    - Integrator, options: ``adaptive`` (default ``uniform``), ``uniform``, ``yoshida4``
  * - ``aperture_type``
    - ``aperture type``
    - str
    - ``off``
    - Aperture type
  * - ``aperture_value``
    - ``aperture value``
    - list
    - ``[]``
    - Aperture parameter values

.. note::

  The ``knl`` and ``ksl`` arrays do not need to have the same length; the shorter array is automatically zero-padded. The maximum order :math:`N` is determined by the longer array length (:math:`N = \max(\text{len}) - 1`).


Usage Examples
--------------

Thin Lens Multipole (Field Error Injection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "MPE1": {
          "S (m)": 10.0,
          "Command": "multipole",
          "Length (m)": 0.0,
          "KiL": [0.0, 0.0, 0.001, 0.0005],
          "KiSL": [0.0, 0.0, 0.0003, 0.0001],
          "Aperture Type": "off"
      }
  }

Zero-length multipole with 2nd and 3rd order field error components. Used to simulate the effect of magnet installation errors or manufacturing errors on the beam.

Thick Lens Multipole (Combined Element)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "MP1": {
          "S (m)": 20.0,
          "Command": "multipole",
          "Length (m)": 0.5,
          "KiL": [0.0, 0.3, 5.0, 200.0],
          "KiSL": [0.0, 0.0, 0.0, 0.0],
          "Num Slices": 5,
          "Integrator": "yoshida4",
          "Aperture Type": "off"
      }
  }

Thick lens combined multipole with simultaneous quadrupole, sextupole, and octupole normal components, 5 slices, 4th-order symplectic integration.

Single-Order Multipole (Equivalent to Octupole)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "MP2": {
          "S (m)": 30.0,
          "Command": "multipole",
          "Length (m)": 0.0,
          "KiL": [0.0, 0.0, 0.0, 500.0],
          "KiSL": [0.0, 0.0, 0.0, 200.0],
          "Aperture Type": "off"
      }
  }

Contains only the 3rd-order component (``knl=[0,0,0,500]``, ``ksl=[0,0,0,200]``), equivalent to a normal+skew octupole thin lens. Particle-by-particle consistent with the ``Octupole`` element.

Higher-Order Multipole (Decapole)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

  {
      "MP3": {
          "S (m)": 40.0,
          "Command": "multipole",
          "Length (m)": 0.0,
          "KiL": [0.0, 0.0, 0.0, 0.0, 10000.0],
          "KiSL": [0.0, 0.0, 0.0, 0.0, 0.0],
          "Aperture Type": "off"
      }
  }

4th-order multipole (decapole), ``knl=[0,0,0,0,10000]``. Dedicated elements only support up to octupole (3rd order); the multipole supports arbitrary order.


Application Scenarios
---------------------

- **Field error injection**: Insert magnet field errors exported from MAD-X as multipoles into the beamline, simulating installation errors and manufacturing deviations
- **Combined multipole elements**: Simultaneously apply multipole kicks of multiple orders at the same location (e.g., quadrupole + sextupole + octupole combination)
- **Higher-order multipoles**: Decapoles (:math:`n=4`), dodecapoles (:math:`n=5`), and other higher-order elements beyond the range of dedicated elements
- **Nonlinear effect studies**: Study the impact of higher-order multipole fields on beam dynamics, such as dynamic aperture and resonance driving
- **MAD-X compatibility**: The ``knl`` / ``ksl`` definitions are fully consistent with MAD-X, allowing direct import of MAD-X sequences


References
----------

- Xsuite Physics Guide, Sec 1.10.3 (exact drift), Sec 1.10.5 (multipole)
- Xsuite source code: ``xtrack/beam_elements/elements_src/multipole.h``, ``track_magnet.h``, ``track_magnet_kick.h``, ``track_magnet_drift.h``
- Yoshida, H., "Construction of higher order symplectic integrators", Phys. Lett. A 150 (1990)
- MAD-X Physics Manual: multipole field and nonlinear transport
- Wiedemann, H., "Particle Accelerator Physics", Ch. 4 (nonlinear beam dynamics)
