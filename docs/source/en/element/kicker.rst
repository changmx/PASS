Kicker
======

This module describes the PASS kicker element **Kicker**, used to simulate the motion of charged particles in a pulsed dipole magnet. A kicker is a fast-acting transverse deflection element that changes the beam direction by applying an angular deflection. It is widely used in beam injection, extraction, orbit correction, and fast beam manipulation.

The PASS kicker supports both **thick element** (``length > 0``) and **thin lens** (``length = 0``) modes. The thick element uses the exact drift-kick-drift (DKD-exact) symplectic integration scheme, supporting both uniform (2nd-order) and yoshida4 (4th-order) symplectic integrators. Physically, a kicker is equivalent to an order-0 multipole (dipole), with kick formulas :math:`\Delta p_x = \text{hkick}` and :math:`\Delta p_y = \text{vkick}`.

**Code Location**

- Source file: ``PASS/commands/element/kicker.py``
- Class name: ``Kicker`` (inherits from ``Command``)
- Registration name: ``kicker``
- Key features:

  - Supports thin lens mode (``length = 0``, applies only a dipole kick)
  - Supports thick lens mode (``length > 0``, DKD-exact symplectic integration)
  - Supports uniform (2nd-order leapfrog) and yoshida4 (4th-order Yoshida composition) integrators
  - Horizontal kick (``hkick``) and vertical kick (``vkick``) set independently
  - Supports unidirectional kick (only ``hkick`` or only ``vkick``) and bidirectional kick (both nonzero)
  - Mask-based kick application, no per-particle branching
  - Zero kick: thin lens degenerates to a marker, thick lens degenerates to a pure drift
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


Physical Derivation
--------------------

Physical Nature of the Kicker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A kicker is physically a pulsed dipole magnet that produces a uniform magnetic field :math:`B_0` within a brief time window, applying a transverse deflection to particles passing through it. The deflection angle of a dipole magnet of length :math:`L` for the reference particle is:

.. math::

  \theta = \frac{q B_0 L}{p_0} = \frac{B_0 L}{B\rho}

where :math:`B\rho = p_0 / q` is the magnetic rigidity. Therefore, a kicker is physically equivalent to an order-0 multipole with integrated strength :math:`K_{0L} = \theta`. In PASS, ``hkick`` and ``vkick`` are this integrated strength (in radians).

Dipole Kick
~~~~~~~~~~~

The kicker kick formulas are:

.. math::

  \Delta p_x = \text{hkick}

.. math::

  \Delta p_y = \text{vkick}

For thin lens mode, ``hkick`` and ``vkick`` are applied directly as integrated strengths. For thick lens DKD mode, the per-slice kick is :math:`\text{hkick}_{\text{eff}} = \text{hk} \cdot \Delta s`, where :math:`\text{hk} = \text{hkick} / L` is the per-unit-length strength.

.. note::

  ``hkick`` and ``vkick`` are integrated dipole strengths (in radians), equivalent to the ``hkick`` / ``vkick`` parameters in MAD-X, and also equivalent to Multipole's ``knl=[hkick]`` and ``ksl=[vkick]``.


Overall Tracking Flow
---------------------

Depending on the magnet length, the kicker has two tracking modes:

**Thin lens mode** (:math:`L = 0`)

::

  ====== Thin lens (length = 0) ======

  Single dipole kick Kick(hkick, vkick)
  [Position unchanged, momentum jump only]

**Thick lens mode** (:math:`L > 0`)

::

  ====== Thick lens (length > 0) ======

  Slice 1 → Slice 2 → ... → Slice N
  (Each slice: Drift(ds/2) → Kick(ds) → Drift(ds/2))

  where ds = L / N
  hkick_eff = hk * ds,  vkick_eff = vk * ds

  If hkick=0 and vkick=0: degenerates to a single exact drift Drift(L)

The complete map is:

Thin lens:

.. math::

  \mathcal{M}_{\text{thin}} = \text{Kick}(\text{hkick}, \text{vkick})

Thick lens (N slices):

.. math::

  \mathcal{M}_{\text{thick}} = \left[\mathcal{M}_{\text{DKD}}(\Delta s)\right]^N

where the DKD map for each slice is:

.. math::

  \mathcal{M}_{\text{DKD}}(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

.. note::

  - Thin lens mode does not change the particle position coordinates :math:`(x, y, z)`, only applies momentum kicks
  - Chromaticity and other effects in thick lens mode are naturally introduced through the :math:`p_z` expression in exact drift
  - When both ``hkick`` and ``vkick`` are zero, the thick lens degenerates to a pure drift, avoiding meaningless empty kick loops


Exact Drift Map
---------------

The drift part uses the exact drift (Table 1.1, map D, Eq. 1.86-1.88), identical to that of the quadrupole/sextupole/octupole/multipole:

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
    - 0.0
    - Magnet length, :math:`= 0` for thin lens, :math:`> 0` for thick lens
  * - ``hkick``
    - ``hkick``
    - float
    - 0.0
    - Horizontal kick (radians), :math:`\Delta p_x = \text{hkick}`
  * - ``vkick``
    - ``vkick``
    - float
    - 0.0
    - Vertical kick (radians), :math:`\Delta p_y = \text{vkick}`
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

  Both ``hkick`` and ``vkick`` are optional parameters with default value 0. They are set independently:

  - Only ``hkick`` nonzero: unidirectional horizontal kicker
  - Only ``vkick`` nonzero: unidirectional vertical kicker
  - Both nonzero: bidirectional kicker
  - Both zero: thin lens degenerates to a marker, thick lens degenerates to a pure drift

  The ``Command`` field should be set to ``kicker``.


Usage Examples
--------------

Thin Lens Horizontal Kicker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example places a thin lens horizontal kicker at :math:`s = 10.0` m with a kick of :math:`1.5 \times 10^{-3}` rad:

.. code-block:: json

  {
      "HK1": {
          "S (m)": 10.0,
          "Command": "kicker",
          "hkick": 0.0015,
          "vkick": 0.0
      }
  }

Only a horizontal kick is applied; the vertical direction is unaffected. Suitable for horizontal orbit correction or horizontal injection.

Thin Lens Vertical Kicker
~~~~~~~~~~~~~~~~~~~~~~~~~

The following example places a thin lens vertical kicker at :math:`s = 20.0` m with a kick of :math:`-2.3 \times 10^{-3}` rad:

.. code-block:: json

  {
      "VK1": {
          "S (m)": 20.0,
          "Command": "kicker",
          "hkick": 0.0,
          "vkick": -0.0023
      }
  }

Only a vertical kick is applied; the horizontal direction is unaffected. The negative sign indicates a downward kick direction.

Thick Lens Bidirectional Kicker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example places a thick lens bidirectional kicker at :math:`s = 15.0` m with 4 slices and a uniform integrator:

.. code-block:: json

  {
      "BK1": {
          "S (m)": 15.0,
          "Command": "kicker",
          "Length (m)": 0.3,
          "hkick": 0.003,
          "vkick": -0.0015,
          "Num Slices": 4,
          "Integrator": "uniform",
          "Aperture Type": "circle",
          "Aperture Value": [0.05]
      }
  }

Both horizontal and vertical kicks are applied simultaneously, with length 0.3 m, 4-slice DKD-exact symplectic integration, and a circular aperture check (radius 0.05 m).

Thick Lens with yoshida4 Integrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example uses a 4th-order Yoshida integrator, suitable for scenarios requiring high precision:

.. code-block:: json

  {
      "BK2": {
          "S (m)": 25.0,
          "Command": "kicker",
          "Length (m)": 0.5,
          "hkick": 0.002,
          "vkick": 0.0,
          "Num Slices": 2,
          "Integrator": "yoshida4"
      }
  }

2 slices, each performing 3 DKD steps (Yoshida composition), with truncation error :math:`O(\Delta s^4)`.

Zero-Kick Kicker
~~~~~~~~~~~~~~~~

The following example places a zero-kick kicker at :math:`s = 30.0` m:

.. code-block:: json

  {
      "K0": {
          "S (m)": 30.0,
          "Command": "kicker",
          "hkick": 0.0,
          "vkick": 0.0
      }
  }

When both kicks are zero and the length is zero, the kicker degenerates to a marker, leaving all particle coordinates unchanged. This can be used to reserve a kicker position for later activation via a ramping table.


Application Scenarios
---------------------

- **Beam injection**: Place kickers in the injection section to deflect the injected beam to match the main ring closed orbit, achieving beam injection
- **Beam extraction**: Place kickers at the extraction point to rapidly deflect the beam into the extraction channel, achieving fast or slow extraction
- **Orbit correction**: Place kickers at key beamline positions to correct orbit deviations or create local orbit bumps
- **Fast beam manipulation**: Control the kicker kick through timing to achieve rapid beam direction switching or scanning
- **Feedback systems**: Combined with pickups, form a bunch-by-bunch transverse feedback system to suppress beam instabilities


References
----------

- MAD-X User's Guide, "Kicker" section (hkick / vkick definitions)
- Xsuite source code: ``xtrack/mad_loader.py`` (``convert_kicker``, ``_make_kicker_multipole``)
- Yoshida, H., "Construction of higher order symplectic integrators", Phys. Lett. A 150 (1990)
- Wiedemann, H., "Particle Accelerator Physics", Ch. 4 (dipole magnets and deflection)
