Drift
==================

This module introduces the drift element **Drift** in PASS, used to simulate particle transport in field-free free space. The drift is the most basic beamline element; particles experience no electromagnetic forces within it and move in straight lines solely by their initial momentum.

**Code Location**

- Source file: ``PASS/commands/element/drift.py``
- Class name: ``Drift`` (inherits from ``Command``)
- Registration name: ``drift``
- Core features:

  - Thick element (``length > 0``), changes the particle's position and longitudinal coordinate;
  - Uses exact geometric transport formulae, accounting for the projection of transverse momentum onto longitudinal velocity;
  - Supports aperture checking, consistent with other elements.


Physical Derivation
-------------------

Particles experience no force in the drift and move in a straight line with constant momentum. Let the drift length be :math:`L`, the particle's normalized transverse momenta be :math:`p_x` and :math:`p_y`, and the momentum deviation be :math:`\delta`.

**Total Particle Momentum**

The normalized total momentum (in units of the reference particle momentum :math:`P_0`) is:

.. math::

  P_{\text{tot}} = 1 + \delta

The longitudinal momentum component (accounting for the projection of transverse momentum) is:

.. math::

  p_z = \sqrt{(1 + \delta)^2 - p_x^2 - p_y^2}

If :math:`p_z^2 \le 0`, the particle is physically impossible (transverse momentum exceeds total momentum) and is marked as lost.

**Particle Velocity**

The particle's :math:`\beta` value is computed from the reference particle's :math:`\beta_0`, :math:`\gamma_0`, and the momentum deviation :math:`\delta`:

.. math::

  \beta = \frac{(1 + \delta) \, \gamma_0 \, \beta_0}{\sqrt{1 + \left[(1 + \delta) \, \gamma_0 \, \beta_0\right]^2}}

**Coordinate Update**

The particle coordinates in the drift are updated as:

.. math::

  x \leftarrow x + L \cdot \frac{p_x}{p_z}

.. math::

  y \leftarrow y + L \cdot \frac{p_y}{p_z}

.. math::

  z \leftarrow z + L \cdot \left(1 - \frac{\beta_0}{\beta} \cdot \frac{1 + \delta}{p_z}\right)

where the :math:`z` update includes the path length difference effect: particles with momentum deviations have different velocities, causing a change in longitudinal position.

**z Coordinate Revolution Folding**

The updated :math:`z` coordinate is folded into the range :math:`[-C/2, \; C/2]`:

.. math::

  z \leftarrow z + \left\lfloor \frac{C/2 - z}{C} \right\rfloor \cdot C

where :math:`C` is the ring circumference. This ensures that :math:`z` always remains within the ring circumference range.


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
    - ``S (m)``
    - float
    - m
    - Longitudinal position of the element in the beamline
  * - ``length``
    - ``Length (m)``
    - float
    - m
    - Element length (must be :math:`\ge 0`)
  * - ``name``
    - ``name``
    - str
    - -
    - Element name (automatically filled from the key name of the sequence JSON)
  * - ``aperture_type``
    - ``Aperture Type``
    - str
    - -
    - Aperture type (default ``off``, available values in the Aperture chapter)
  * - ``aperture_value``
    - ``Aperture Value``
    - list
    - -
    - Aperture parameter values (default ``[]``, meaning varies by type, see the Aperture chapter)


Usage Examples
--------------

The following JSON snippet demonstrates the configuration of a drift:

**Basic usage**:

.. code-block:: json

  "Drift1": {
      "S (m)": 10.0,
      "Command": "Drift",
      "Length (m)": 0.5,
      "Aperture Type": "off"
  }

**With circular aperture checking**:

.. code-block:: json

  "Drift2": {
      "S (m)": 10.5,
      "Command": "Drift",
      "Length (m)": 0.3,
      "Aperture Type": "circle",
      "Aperture Value": [0.05]
  }

**With rectangular aperture checking**:

.. code-block:: json

  "Drift3": {
      "S (m)": 11.0,
      "Command": "Drift",
      "Length (m)": 0.2,
      "Aperture Type": "rectangle",
      "Aperture Value": [0.06, 0.04]
  }


Application Scenarios
---------------------

- **Beamline connection**: Provides free drift space between magnet elements, the most commonly used beamline element
- **Dispersion measurement**: Sets up a drift section after a dipole magnet to measure beam momentum spread using the dispersion effect
- **Beam transport**: Transports the beam in injection and extraction lines without applying any field
- **Aperture checking**: Sets up drifts with aperture checking at key positions to monitor beam loss
