Twiss Transport (Twiss)
==============================================

``Twiss`` computes uncoupled transverse transport from entrance and exit optical functions and phase advances, with optional longitudinal linear maps and chromatic phase corrections. Coordinate, reference-momentum, and arrival-time conventions are defined in :ref:`en-longitudinal-reference`.

The supplied optics already contains design focusing; additional magnets apply additional effects and must not double-count it. Transverse beta functions must be positive. ``Mu`` is in cycles (1 corresponds to 2π); retain the full required phase difference. The ``drift`` mode needs the bunch transition gamma, while ``matrix`` requires positive bunch length and relative momentum spread.

Usage Example
-------------

The following JSON snippet shows a ``Twiss`` entry inside ``Sequence``:

.. code-block:: json

   "Twiss1": {
       "S (m)": 10.0,
       "Command": "Twiss",
       "S previous (m)": 5.0,
       "Alpha x": 0.5,
       "Alpha y": -0.3,
       "Alpha x previous": 0.4,
       "Alpha y previous": -0.2,
       "Beta x (m)": 3.5,
       "Beta y (m)": 2.8,
       "Beta x previous (m)": 3.0,
       "Beta y previous (m)": 2.5,
       "Mu x": 0.123,
       "Mu y": 0.456,
       "Mu x previous": 0.1,
       "Mu y previous": 0.4,
       "Dx (m)": 0.5,
       "Dx previous (m)": 0.3,
       "Dpx": 0.01,
       "Dpx previous": 0.005,
       "DQx": 2.0,
       "DQy": 2.0,
       "Longitudinal transfer": "drift"
   }

Interface parameters
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 25 10 13 30

   * - Python attribute
     - JSON key
     - Unit
     - Default
     - Description
   * - ``s``
     - ``S (m)``
     - m
     - Required
     - Exit position.
   * - ``command``
     - ``Command``
     - —
     - ``Twiss``
     - Command identifier.
   * - ``s_previous``
     - ``S previous (m)``
     - m
     - Required
     - Entrance position; the difference gives transport length.
   * - ``alpha_x``
     - ``Alpha x``
     - —
     - Required
     - Exit horizontal Twiss α.
   * - ``alpha_y``
     - ``Alpha y``
     - —
     - Required
     - Exit vertical Twiss α.
   * - ``beta_x``
     - ``Beta x (m)``
     - m
     - Required
     - Positive exit horizontal Twiss β.
   * - ``beta_y``
     - ``Beta y (m)``
     - m
     - Required
     - Positive exit vertical Twiss β.
   * - ``mu_x``
     - ``Mu x``
     - cycles
     - Required
     - Exit horizontal phase.
   * - ``mu_y``
     - ``Mu y``
     - cycles
     - Required
     - Exit vertical phase.
   * - ``mu_z``
     - ``Mu z``
     - cycles
     - ``0.0``
     - Exit longitudinal phase.
   * - ``dx``
     - ``Dx (m)``
     - m
     - Required
     - Exit horizontal position dispersion.
   * - ``dpx``
     - ``Dpx``
     - —
     - Required
     - Exit normalized horizontal momentum dispersion.
   * - ``alpha_x_previous``
     - ``Alpha x previous``
     - —
     - Required
     - Entrance horizontal Twiss α.
   * - ``alpha_y_previous``
     - ``Alpha y previous``
     - —
     - Required
     - Entrance vertical Twiss α.
   * - ``beta_x_previous``
     - ``Beta x previous (m)``
     - m
     - Required
     - Positive entrance horizontal Twiss β.
   * - ``beta_y_previous``
     - ``Beta y previous (m)``
     - m
     - Required
     - Positive entrance vertical Twiss β.
   * - ``mu_x_previous``
     - ``Mu x previous``
     - cycles
     - Required
     - Entrance horizontal phase.
   * - ``mu_y_previous``
     - ``Mu y previous``
     - cycles
     - Required
     - Entrance vertical phase.
   * - ``mu_z_previous``
     - ``Mu z previous``
     - cycles
     - ``0.0``
     - Entrance longitudinal phase.
   * - ``dx_previous``
     - ``Dx previous (m)``
     - m
     - ``0.0``
     - Entrance horizontal position dispersion.
   * - ``dpx_previous``
     - ``Dpx previous``
     - —
     - ``0.0``
     - Entrance normalized horizontal momentum dispersion.
   * - ``dqx``
     - ``DQx``
     - —
     - ``0.0``
     - Horizontal chromatic phase coefficient for this segment.
   * - ``dqy``
     - ``DQy``
     - —
     - ``0.0``
     - Vertical chromatic phase coefficient for this segment.
   * - ``longitudinal_transfer``
     - ``Longitudinal transfer``
     - —
     - ``off``
     - ``off``, ``drift``, or ``matrix``.

Output and scope
--------------------------------

The command updates particle coordinates and reference passage time but writes no diagnostic file. Place a monitor from :doc:`monitor/index` at its exit to record the result. The transverse model includes neither x–y coupling nor vertical dispersion. Nonzero ``DQx`` or ``DQy`` makes phase depend on relative momentum deviation, so the full map is not a fixed six-dimensional linear matrix.

``DQx`` and ``DQy`` are applied directly to this segment; no additional segment-length/circumference factor is applied. They must be consistent with the supplied optics and any additional nonlinear elements.

Hand-written runtime JSON may include ``Aperture type`` and ``Aperture value`` (defaults ``off`` and ``[]``), but ``TwissItem`` does not currently export these fields. When using the parameter builder, place an aperture-bearing ``Marker`` at the same location; see :doc:`aperture`.

Physics Derivation
-------------------

Longitudinal Transport
~~~~~~~~~~~~~~~~~~~~~~

Longitudinal transport is controlled by the ``Longitudinal transfer`` parameter and supports three modes:

**drift mode** : Uses the gamma transition parameter. The longitudinal transport matrix element is:

.. math::

   m_{12,z} = -\left(\frac{1}{\gamma_t^2} - \frac{1}{\gamma^2}\right)(s - s_\mathrm{previous})

where :math:`\gamma_t` is the transition gamma, :math:`\gamma` is the bunch reference-particle Lorentz factor, :math:`s` is the current longitudinal position, and :math:`s_\mathrm{previous}` is the longitudinal position of the previous element.

**matrix mode** : Uses the prescribed longitudinal phase advance. The transport matrix is:

.. math::

   m_{11,z} = \cos(\phi_z)

.. math::

   m_{12,z} = \frac{\sigma_z}{\sigma_\delta} \sin(\phi_z)

.. math::

   m_{21,z} = -\frac{\sigma_\delta}{\sigma_z} \sin(\phi_z)

.. math::

   m_{22,z} = \cos(\phi_z)

where :math:`\phi_z` is the longitudinal phase advance, :math:`\sigma_z` is the bunch longitudinal size, and :math:`\sigma_\delta` is the configured RMS relative momentum spread.

**off mode** : The longitudinal transport matrix is the identity matrix.

Dispersion Handling
~~~~~~~~~~~~~~~~~~~

Since the transverse transport matrix describes only the non-dispersive part of the motion, dispersion must be removed before transport and restored after:

1. **Remove dispersion at the previous point** :

.. math::

   x_1 = x - D_{x,\mathrm{previous}} \cdot \delta

.. math::

   px_1 = px - D_{px,\mathrm{previous}} \cdot \delta

2. **Linear transport** :

.. math::

   x_\mathrm{temp} = x_1 \cdot m_{11} + px_1 \cdot m_{12}

3. **Add dispersion at the new point** :

.. math::

   x_2 = x_\mathrm{temp} + D_x \cdot \delta_2

where :math:`D_x` is the horizontal dispersion at the current point, :math:`D_{x,\mathrm{previous}}` is the horizontal dispersion at the previous point, and :math:`\delta` is the relative particle momentum deviation.

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

   \phi_x = \phi_x + \delta \cdot \Delta Q_x \cdot 2\pi

.. math::

   \phi_y = \phi_y + \delta \cdot \Delta Q_y \cdot 2\pi

where :math:`\Delta Q_x` and :math:`\Delta Q_y` are the horizontal and vertical chromaticities, respectively, and :math:`\delta` is the relative particle momentum deviation.

Longitudinal Coordinate Continuity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stored z remains continuous, retaining multi-turn slip. Physical arrival time is :math:`t_i=T_b-z_i/(\beta_b c)` using the current reference event; nominal slot offsets do not enter this reconstruction.

t0 Update
~~~~~~~~~

The reference time t0 is updated based on the longitudinal position change:

.. math::

   \Delta t = \frac{s - s_\mathrm{previous}}{\beta \, c}

where :math:`\beta` is the normalized reference-particle speed and :math:`c` is the speed of light.

Phase and momentum notation
------------------------------------------------------

Momentum deviation in these equations is the dimensionless :math:`\delta=P/P_0-1`, not an absolute momentum increment. Before chromatic correction, :math:`\phi_u=2\pi(\mu_u-\mu_{u,\mathrm{previous}})`. The longitudinal ``matrix`` map uses the configured RMS bunch length and relative momentum spread; it does not recompute them from the current particle distribution.
