Twiss Transport (Twiss)
========================

Introduction
------------

The ``Twiss`` element implements 6D linear optical transport based on Twiss parameters. It uses beam optical functions (beta, alpha, mu, dispersion) to construct the transport matrix and performs linear optical tracking of particles. It is one of the core elements connecting lattice design with particle tracking.

- **Code location** : ``PASS/commands/twiss.py``
- **Class name** : ``Twiss`` , registered name ``"twiss"``
- **Key features** :

  - Constructs the transverse transport matrix based on the Twiss parameters (beta, alpha, phase) at the previous and current points;
  - Supports dispersion removal and restoration, ensuring correct transport of particles with momentum deviation in dispersive regions;
  - Supports chromaticity correction; tune shifts caused by momentum deviation are automatically incorporated into the phase;
  - Longitudinal transport supports three modes: drift, matrix, and identity matrix;
  - Supports z-coordinate wrapping/folding, keeping the longitudinal coordinate within the ring circumference;
  - Supports aperture checking, consistent with other elements.

Physics Derivation
-------------------

Longitudinal Transport
~~~~~~~~~~~~~~~~~~~~~~

Longitudinal transport is controlled by the ``Longitudinal Transfer`` parameter and supports three modes:

**drift mode** : Uses the gamma transition parameter. The longitudinal transport matrix element is:

.. math::

   m_{12,z} = -\left(\frac{1}{\gamma_t^2} - \frac{1}{\gamma^2}\right)(s - s_\mathrm{previous})

where :math:`\gamma_t` is the transition gamma, :math:`\gamma` is the relativistic gamma of the particle, :math:`s` is the current longitudinal position, and :math:`s_\mathrm{previous}` is the longitudinal position of the previous element.

**matrix mode** : Uses the longitudinal oscillation frequency. The transport matrix is:

.. math::

   m_{11,z} = \cos(\phi_z)

.. math::

   m_{12,z} = \frac{\sigma_z}{\Delta p_\mathrm{bunch}} \sin(\phi_z)

.. math::

   m_{21,z} = -\frac{\Delta p_\mathrm{bunch}}{\sigma_z} \sin(\phi_z)

.. math::

   m_{22,z} = \cos(\phi_z)

where :math:`\phi_z` is the longitudinal phase advance, :math:`\sigma_z` is the bunch longitudinal size, and :math:`\Delta p_\mathrm{bunch}` is the bunch momentum spread.

**Other modes** : The longitudinal transport matrix is the identity matrix.

Dispersion Handling
~~~~~~~~~~~~~~~~~~~

Since the transverse transport matrix describes only the non-dispersive part of the motion, dispersion must be removed before transport and restored after:

1. **Remove dispersion at the previous point** :

.. math::

   x_1 = x - D_{x,\mathrm{previous}} \cdot \Delta p

.. math::

   px_1 = px - D_{px,\mathrm{previous}} \cdot \Delta p

2. **Linear transport** :

.. math::

   x_\mathrm{temp} = x_1 \cdot m_{11} + px_1 \cdot m_{12}

3. **Add dispersion at the new point** :

.. math::

   x_2 = x_\mathrm{temp} + D_x \cdot \Delta p_2

where :math:`D_x` is the horizontal dispersion at the current point, :math:`D_{x,\mathrm{previous}}` is the horizontal dispersion at the previous point, and :math:`\Delta p` is the particle momentum deviation.

Transverse Transport Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transverse transport matrix is expressed in terms of the Twiss parameters and phase advance at the previous and current points. Taking the horizontal direction as an example:

.. math::

   m_{11,x} = \sqrt{\frac{\beta_x}{\beta_{x,\mathrm{prev}}}}
   \left(\cos\phi_x + \alpha_{x,\mathrm{prev}} \sin\phi_x\right)

.. math::

   m_{12,x} = \sqrt{\beta_x \, \beta_{x,\mathrm{prev}}} \sin\phi_x

.. math::

   m_{21,x} = -\frac{1 + \alpha_x \, \alpha_{x,\mathrm{prev}}}
   {\sqrt{\beta_x \, \beta_{x,\mathrm{prev}}}} \sin\phi_x
   + \frac{\alpha_{x,\mathrm{prev}} - \alpha_x}
   {\sqrt{\beta_x \, \beta_{x,\mathrm{prev}}}} \cos\phi_x

.. math::

   m_{22,x} = \sqrt{\frac{\beta_{x,\mathrm{prev}}}{\beta_x}}
   \left(\cos\phi_x - \alpha_x \sin\phi_x\right)

where :math:`\beta_x` and :math:`\alpha_x` are the horizontal Twiss parameters at the current point, :math:`\beta_{x,\mathrm{prev}}` and :math:`\alpha_{x,\mathrm{prev}}` are the horizontal Twiss parameters at the previous point, and :math:`\phi_x` is the horizontal phase advance between the two points.

The transport matrix for the vertical direction (y) has exactly the same form; simply replace the subscript x with y.

Chromaticity Correction
~~~~~~~~~~~~~~~~~~~~~~~

Momentum deviation causes tune shifts, which are corrected through the chromaticity parameters:

.. math::

   \phi_x = \phi_x + \Delta p \cdot \Delta Q_x \cdot 2\pi

.. math::

   \phi_y = \phi_y + \Delta p \cdot \Delta Q_y \cdot 2\pi

where :math:`\Delta Q_x` and :math:`\Delta Q_y` are the horizontal and vertical chromaticities, respectively, and :math:`\Delta p` is the particle momentum deviation.

z-Coordinate Wrapping/Folding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To keep the longitudinal coordinate within the ring circumference, the z-coordinate is wrapped after transport:

.. math::

   z_2 = z_2 + (\mathrm{under} - \mathrm{over}) \cdot C

where :math:`C` is the ring circumference, and ``under`` and ``over`` are folding counters.

t0 Update
~~~~~~~~~

The reference time t0 is updated based on the longitudinal position change:

.. math::

   \Delta t = \frac{s - s_\mathrm{previous}}{\beta \, c}

where :math:`\beta` is the relativistic velocity of the particle and :math:`c` is the speed of light.

Interface Parameters
--------------------

Position Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - Parameter
     - Key
     - Type
     - Unit
     - Description
   * - ``s``
     - ``"S (m)"``
     - float
     - m
     - Longitudinal position of the current element
   * - ``s_previous``
     - ``"S Previous (m)"``
     - float
     - m
     - Longitudinal position of the previous element
   * - ``name``
     - ``"name"``
     - str
     - -
     - Automatically filled from the sequence key name

Transverse Parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - Parameter
     - Key
     - Type
     - Unit
     - Description
   * - ``alphax``
     - ``"Alpha X"``
     - float
     - -
     - Horizontal alpha at the current point
   * - ``alphay``
     - ``"Alpha Y"``
     - float
     - -
     - Vertical alpha at the current point
   * - ``alphax_previous``
     - ``"Alpha X Previous"``
     - float
     - -
     - Horizontal alpha at the previous point
   * - ``alphay_previous``
     - ``"Alpha Y Previous"``
     - float
     - -
     - Vertical alpha at the previous point
   * - ``betax``
     - ``"Beta X (m)"``
     - float
     - m
     - Horizontal beta at the current point
   * - ``betay``
     - ``"Beta Y (m)"``
     - float
     - m
     - Vertical beta at the current point
   * - ``betax_previous``
     - ``"Beta X Previous (m)"``
     - float
     - m
     - Horizontal beta at the previous point
   * - ``betay_previous``
     - ``"Beta Y Previous (m)"``
     - float
     - m
     - Vertical beta at the previous point
   * - ``mux``
     - ``"Mu X"``
     - float
     - -
     - Horizontal phase at the current point
   * - ``muy``
     - ``"Mu Y"``
     - float
     - -
     - Vertical phase at the current point
   * - ``mux_previous``
     - ``"Mu X Previous"``
     - float
     - -
     - Horizontal phase at the previous point
   * - ``muy_previous``
     - ``"Mu Y Previous"``
     - float
     - -
     - Vertical phase at the previous point

Longitudinal Parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - Parameter
     - Key
     - Type
     - Unit
     - Description
   * - ``longitudinal_transfer``
     - ``"Longitudinal Transfer"``
     - str
     - -
     - Longitudinal transport mode (drift/matrix/other)
   * - ``muz``
     - ``"Mu Z"``
     - float
     - -
     - Longitudinal phase at the current point (optional, default 0)
   * - ``muz_previous``
     - ``"Mu Z Previous"``
     - float
     - -
     - Longitudinal phase at the previous point (optional, default 0)

Dispersion and Chromaticity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - Parameter
     - Key
     - Type
     - Unit
     - Description
   * - ``Dx``
     - ``"Dx (m)"``
     - float
     - m
     - Horizontal dispersion at the current point
   * - ``Dx_previous``
     - ``"Dx Previous (m)"``
     - float
     - m
     - Horizontal dispersion at the previous point
   * - ``Dpx``
     - ``"Dpx"``
     - float
     - -
     - Horizontal dispersion derivative at the current point
   * - ``Dpx_previous``
     - ``"Dpx Previous"``
     - float
     - -
     - Horizontal dispersion derivative at the previous point
   * - ``DQx``
     - ``"Dqx"``
     - float
     - -
     - Horizontal chromaticity
   * - ``DQy``
     - ``"Dqy"``
     - float
     - -
     - Vertical chromaticity

Aperture Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - Parameter
     - Key
     - Type
     - Unit
     - Description
   * - ``aperture_type``
     - ``"Aperture Type"``
     - str
     - -
     - Aperture type (default off)
   * - ``aperture_value``
     - ``"Aperture Value"``
     - list
     - -
     - Aperture parameters (default [])

Usage Example
-------------

The following JSON snippet shows a complete ``Twiss`` element definition:

.. code-block:: json

   "Twiss1": {
       "S (m)": 10.0,
       "Command": "Twiss",
       "S Previous (m)": 5.0,
       "Alpha X": 0.5,
       "Alpha Y": -0.3,
       "Alpha X Previous": 0.4,
       "Alpha Y Previous": -0.2,
       "Beta X (m)": 3.5,
       "Beta Y (m)": 2.8,
       "Beta X Previous (m)": 3.0,
       "Beta Y Previous (m)": 2.5,
       "Mu X": 0.123,
       "Mu Y": 0.456,
       "Mu X Previous": 0.1,
       "Mu Y Previous": 0.4,
       "Dx (m)": 0.5,
       "Dx Previous (m)": 0.3,
       "Dpx": 0.01,
       "Dpx Previous": 0.005,
       "Dqx": 2.0,
       "Dqy": 2.0,
       "Longitudinal Transfer": "drift",
       "Aperture Type": "off"
   }

Where:

- ``"S (m)": 10.0`` — the longitudinal position of the current element is 10.0 m;
- ``"S Previous (m)": 5.0`` — the longitudinal position of the previous element is 5.0 m;
- ``"Alpha X": 0.5`` — the horizontal alpha at the current point is 0.5;
- ``"Beta X (m)": 3.5`` — the horizontal beta at the current point is 3.5 m;
- ``"Mu X": 0.123`` — the horizontal phase at the current point is 0.123 (in units of :math:`2\pi` );
- ``"Dx (m)": 0.5`` — the horizontal dispersion at the current point is 0.5 m;
- ``"Dqx": 2.0`` — the horizontal chromaticity is 2.0;
- ``"Longitudinal Transfer": "drift"`` — the longitudinal transport uses drift mode;
- ``"Aperture Type": "off"`` — aperture checking is disabled.

Application Scenarios
---------------------

The ``Twiss`` element is suitable for the following scenarios:

- **Linear tracking based on lattice design** : When a Twiss parameter table from optical calculation programs such as MadX or AT is already available, the ``Twiss`` element can be used directly for particle tracking without re-modeling magnet elements.
- **Dispersion and chromaticity studies** : The ``Twiss`` element has built-in dispersion removal/restoration and chromaticity correction, making it suitable for studying the transverse dynamics of particles with momentum deviation.
- **Longitudinal dynamics simulation** : By selecting drift or matrix longitudinal transport modes, different longitudinal transport scenarios can be simulated.
- **Fast optical evaluation** : Compared to element-by-element modeling, linear transport based on Twiss parameters requires less computation, making it suitable for large-scale parameter scans and preliminary evaluations.
- **Combined use with nonlinear elements** : The ``Twiss`` element can be used in series with nonlinear elements such as sextupoles and octupoles to superimpose nonlinear effects on top of linear transport.
