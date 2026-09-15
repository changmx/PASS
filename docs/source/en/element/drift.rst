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

If :math:`p_z^2 \le 0`, there is no positive longitudinal momentum for forward
transport through the element, and the particle is marked as lost.

**Particle Velocity**

The particle's :math:`\beta` value is related to the reference particle's :math:`\beta_0`, :math:`\gamma_0`, and the momentum deviation :math:`\delta` by:

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

The longitudinal update above is the formula evaluated directly by the original
implementation. It includes both the speed difference caused by momentum
deviation and the longer flight path caused by transverse motion.

Original Formula from Flight Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here :math:`p_x=P_x/P_0`, :math:`p_y=P_y/P_0`, and
:math:`\delta=(P-P_0)/P_0`. The quantity :math:`p_z` above is normalized
longitudinal momentum; the stored coordinate ``p.z`` is instead the continuous
time coordinate

.. math::

  z=\beta_0c(t_0-t_i).

The derivation assumes forward motion with :math:`1+\delta>0` and
:math:`p_z>0`. A field-free drift keeps the reference speed and each particle's
momentum constant. The reference flight time and the particle's longitudinal
velocity are

.. math::

  \Delta t_0=\frac{L}{\beta_0c},\qquad
  v_s=\beta c\frac{p_z}{1+\delta},\qquad
  \Delta t_i=\frac{L}{v_s}.

Consequently,

.. math::

  \Delta z=\beta_0c(\Delta t_0-\Delta t_i)
  =L\left(1-\frac{\beta_0}{\beta}\frac{1+\delta}{p_z}\right).

The reference event advances by :math:`\Delta t_0`; positive :math:`\Delta z`
means that the particle gains an arrival-time lead over the reference.

Current Formula by Rationalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set :math:`a=\gamma_0^{-2}`, :math:`u=1+\delta`, and
:math:`q_\perp=p_x^2+p_y^2`. Using :math:`\beta_0^2=1-a` and the
relativistic energy-momentum relation gives

.. math::

  A\equiv\frac{\beta_0}{\beta}u
  =\frac{\sqrt{1+(\gamma_0\beta_0u)^2}}{\gamma_0}
  =\sqrt{a+(1-a)u^2}=\frac{E_i}{E_0}.

Here :math:`E_i` and :math:`E_0` are total energies for particles of the same
rest mass. ``energy_over_gamma`` in the implementation represents :math:`A`.
Since :math:`p_z^2=u^2-q_\perp`, rationalizing the difference gives

.. math::

  \frac{\Delta z}{L}
  =\frac{p_z-A}{p_z}
  =\frac{p_z^2-A^2}{p_z(p_z+A)},

.. math::

  p_z^2-A^2
  =u^2-q_\perp-\left[a+(1-a)u^2\right]
  =a(u^2-1)-q_\perp
  =\frac{\delta(2+\delta)}{\gamma_0^2}-p_x^2-p_y^2.

Both CPU and GPU now evaluate

.. math::

  \Delta z = L\frac{\delta(2+\delta)/\gamma_0^2-p_x^2-p_y^2}
  {p_z\left[p_z+\sqrt{\gamma_0^{-2}+(1-\gamma_0^{-2})(1+\delta)^2}\right]}.

This is an algebraic identity, with no expansion in momentum deviation or
transverse angle. It retains the original exact geometric map. The same kernel
is used for GPU drift segments with internal space charge.

Purpose and Numerical Limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original expression subtracts two numbers close to one when the slip is
small, losing significant digits. The FP32 spacing immediately above one is
approximately :math:`1.19\times10^{-7}`. A true slip factor of order
:math:`10^{-8}` or :math:`10^{-9}` can therefore have a large relative error or
round to zero in the direct expression.

The current expression constructs the small numerator explicitly. In particular,
``dp * (2 + dp)`` retains a small momentum deviation even if ``1 + dp`` rounds to
one; evaluating ``(1 + dp)**2 - 1`` would reintroduce cancellation.

For interpretation only, retaining the leading momentum and transverse terms gives

.. math::

  \frac{\Delta z}{L}\simeq
  \frac{\delta}{\gamma_0^2}-\frac{p_x^2+p_y^2}{2}.

Thus an on-axis particle with positive momentum deviation arrives earlier,
while transverse motion at fixed total momentum delays arrival. The ideal
reference particle has zero slip. Tracking evaluates the full rationalized
formula above, rather than this leading-order expression.

For example, with :math:`L=1\,\mathrm{m}`, :math:`\gamma_0=2`,
:math:`\delta=10^{-8}`, and :math:`p_x=p_y=0`, the exact slip is approximately
:math:`2.5\times10^{-9}\,\mathrm{m}`. The rationalized expression retains this
increment in FP32 when the initial z is zero. All six stored particle coordinates
still use the selected FP32 or FP64 precision: adding the same increment to an
existing FP32 :math:`z=1\,\mathrm{m}` rounds back to one. The rewrite improves
increment evaluation; it does not remove rounding during repeated accumulation
or cancellation when the two physical contributions nearly balance.

The exact derivation also presumes that :math:`p_z` is evaluated without a floor.
The CPU currently applies :math:`p_z=\sqrt{\max(p_z^2,10^{-10})}` after checking
for loss, while the GPU directly takes the square root for valid particles.
Their results can therefore differ for :math:`0<p_z^2<10^{-10}`. This pre-existing
boundary treatment is separate from the formula rewrite.

Computational Cost
~~~~~~~~~~~~~~~~~~

Both forms have :math:`O(N)` work for :math:`N` particles. On the CPU, the new
form removes the explicit particle-speed array and its divisions, while adding
the rationalized numerator and denominator; both forms still evaluate two
square roots per particle. NumPy temporary arrays and memory traffic also affect
runtime, so the expression length alone does not predict a speedup.

The GPU keeps the calculation in the same fused kernel and retains two square
roots per particle. The new denominator adds a floating-point division relative
to the original kernel, plus arithmetic for the numerator. Coordinate storage,
global array accesses, and the number of kernel launches are unchanged. The
cost can increase for arithmetic-limited workloads, particularly with FP64;
small batches may instead be dominated by launch overhead, and large batches
may be limited by memory throughput. The purpose of this change is reliable
small-slip evaluation. Its runtime impact must be measured for the chosen
hardware, precision, particle count, and slicing configuration.

GPU timing must exclude initial compilation and account for asynchronous
execution, using synchronization or CUDA events; see
`CuPy performance guidance <https://docs.cupy.dev/en/stable/user_guide/performance.html>`_.
Steady-state kernel timings do not by themselves measure the cost of a complete
simulation with space charge, monitors, and data transfers.

Longitudinal Coordinate Continuity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stored z remains continuous, retaining multi-turn slip. Physical arrival time is :math:`t_i=T_b-z_i/(\beta_b c)` using the current reference event; nominal slot offsets do not enter this reconstruction.


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

Internal Space Charge
------------------------------------------

A positive-length element may set ``space_charge`` (JSON ``Space charge``)
to an ``ElementSpaceCharge`` object. ``Num slices`` controls external transport,
while ``Space charge.Num kicks`` controls SC integration. See :ref:`en-internal-space-charge`
for scheduling, shared resources, supported backends and examples.

``num_slices`` (JSON ``Num slices``) is a positive integer, default 1.
Without internal SC, that many body slices are used on both CPU and GPU.
