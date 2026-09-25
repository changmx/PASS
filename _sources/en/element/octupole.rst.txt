Octupole
========

This module describes the PASS octupole element **Octupole**, used to simulate the motion of charged particles in an octupole magnet. The octupole provides a cubic nonlinear magnetic field and is an important nonlinear element in accelerators, primarily used for Landau damping, amplitude-dependent tune shift (ADTS), and resonance suppression.

The PASS octupole supports both **thick element** (``length > 0``) and **thin lens** (``length = 0``) modes. The thick element uses the exact drift-kick-drift (DKD-exact) symplectic integration scheme, supporting both uniform (2nd-order) and yoshida4 (4th-order) symplectic integrators.

- Registration name: ``octupole``
- Key features:

  - Supports thin lens mode (``length = 0``, applies only an octupole kick)
  - Supports thick lens mode (``length > 0``, DKD-exact symplectic integration)
  - Supports uniform (2nd-order leapfrog) and yoshida4 (4th-order Yoshida composition) integrators
  - Supports normal octupole (``k3l``) and skew octupole (``k3sl``) and their combinations
  - Zero field (``k3l = k3sl = 0``) automatically degenerates to a pure drift
  - Higher-order nonlinear effects naturally introduced through exact drift
  - Supports aperture check

The fields below configure ``PASS.para.schema.elements.OctupoleItem``.
The key in ``Sequence.add(name, item)`` supplies the element name; it is not a configuration-model field.

Interface Parameters
--------------------

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python configuration field
     - JSON key
     - Type
     - Unit
     - Default
     - Description
   * - ``s``
     - ``S (m)``
     - ``float``
     - m
     - ``Required``
     - Longitudinal position of the element exit or zero-length action point.
   * - ``length``
     - ``Length (m)``
     - ``float``
     - m
     - ``0.0``
     - Element length (must be :math:`\ge 0`; :math:`= 0` for thin lens)
   * - ``k3l``
     - ``K3L``
     - ``float``
     - :math:`\text{m}^{-3}`
     - ``0.0``
     - Normal octupole integrated strength :math:`K_{3L}`, default 0
   * - ``k3sl``
     - ``K3SL``
     - ``float``
     - :math:`\text{m}^{-3}`
     - ``0.0``
     - Skew octupole integrated strength :math:`K_{3sL}`, default 0
   * - ``num_slices``
     - ``Num slices``
     - ``int``
     - 1
     - ``1``
     - Number of slices, default 1 (effective only for thick lens)
   * - ``integrator``
     - ``Integrator``
     - ``str``
     - -
     - ``'adaptive'``
     - Integrator, options: ``adaptive`` (default ``uniform``), ``uniform``, ``yoshida4``
   * - ``aperture_type``
     - ``Aperture type``
     - ``str``
     - —
     - ``'off'``
     - Aperture type, default ``off``
   * - ``aperture_value``
     - ``Aperture value``
     - ``list``
     - m / rad
     - ``[]``
     - Aperture parameter values, default ``[]``

Usage Examples
--------------

Thick Lens Normal Octupole
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
       "OCT1": {
           "S (m)": 10.0,
           "Command": "Octupole",
           "Length (m)": 0.5,
           "K3L": 500.0,
           "Num slices": 5,
           "Integrator": "yoshida4",
           "Aperture type": "off"
       }
   }

Normal octupole (:math:`K_{3L} > 0`), length 0.5 m, 5 slices, 4th-order symplectic integration. Used for Landau damping.

Thin Lens Octupole
~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
       "OCT2": {
           "S (m)": 20.0,
           "Command": "Octupole",
           "Length (m)": 0.0,
           "K3L": 1000.0,
           "Aperture type": "off"
       }
   }

Zero-length octupole, applying only the :math:`K_{3L}` thin lens kick, no body tracking.

Negative Octupole
~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
       "OCT3": {
           "S (m)": 30.0,
           "Command": "Octupole",
           "Length (m)": 0.4,
           "K3L": -500.0,
           "Num slices": 1,
           "Integrator": "uniform",
           "Aperture type": "off"
       }
   }

Negative octupole (:math:`K_{3L} < 0`), providing a tune shift in the opposite direction to a positive octupole.

Skew Octupole
~~~~~~~~~~~~~

.. code-block:: json

   {
       "OCT4": {
           "S (m)": 40.0,
           "Command": "Octupole",
           "Length (m)": 0.3,
           "K3L": 0.0,
           "K3SL": 300.0,
           "Num slices": 1,
           "Integrator": "uniform",
           "Aperture type": "off"
       }
   }

Pure skew octupole (:math:`K_{3L} = 0`, :math:`K_{3sL} \neq 0`), producing a coupling effect equivalent to rotating the normal octupole by :math:`\pi / 8`.

Normal + Skew Octupole Combination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
       "OCT5": {
           "S (m)": 50.0,
           "Command": "Octupole",
           "Length (m)": 0.5,
           "K3L": 500.0,
           "K3SL": 100.0,
           "Num slices": 3,
           "Integrator": "yoshida4",
           "Aperture type": "circle",
           "Aperture value": [0.04]
       }
   }

Combined octupole with both normal and skew components (simulating installation rotation error), with a circular aperture check.

Coordinate Convention
---------------------

Use the continuous longitudinal coordinate in :ref:`en-longitudinal-reference`.
:math:`T_b` is the ideal reference particle passage time at this position, not a measured bunch-centroid time.
The local-map symbols :math:`\beta_0` and :math:`P_0` refer to the current bunch reference.
:math:`p_x=P_x/P_0` is normalized momentum, not a trajectory slope.

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
    - Longitudinal coordinate, :math:`z=\zeta=\beta_b c(T_b-t_i)`
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

Octupole Field and Normalized Strength
--------------------------------------

The magnetic field of an octupole magnet has a cubic distribution in the transverse plane. In complex notation:

.. math::

  B_y + i B_x = \frac{1}{6}(B''' + i B'''_s)(x + i y)^3

where :math:`B'''` is the normal octupole field third derivative and :math:`B'''_s` is the skew octupole field third derivative. Expanding:

.. math::

  B_y = \frac{1}{6} B''' (x^3 - 3 x y^2) - \frac{1}{6} B'''_s (3 x^2 y - y^3)

.. math::

  B_x = \frac{1}{6} B''' (3 x^2 y - y^3) + \frac{1}{6} B'''_s (x^3 - 3 x y^2)

The normalized octupole strength is defined as:

.. math::

  K_3 = \frac{q_0 B'''}{6 P_0}

.. math::

  K_{3s} = \frac{q_0 B'''_s}{6 P_0}

The integrated strength is:

.. math::

  K_{3L} = K_3 \cdot L, \qquad K_{3sL} = K_{3s} \cdot L

where :math:`L` is the magnet length. In PASS, the user directly specifies :math:`K_{3L}` (``k3l``) and :math:`K_{3sL}` (``k3sl``); for thick lenses, :math:`K_3 = K_{3L} / L` and :math:`K_{3s} = K_{3sL} / L` are solved internally.

Overall Tracking Flow
---------------------

Depending on the magnet length, the octupole has two tracking modes:

**Thin lens mode** (:math:`L = 0`)

::

  ====== Thin lens (length = 0) ======

  Single octupole kick Kick(K3L, K3sL)
  [Position unchanged, momentum jump only]

**Thick lens mode** (:math:`L > 0`)

::

  ====== Thick lens (length > 0) ======

  Slice 1 → Slice 2 → ... → Slice N
  (Each slice: Drift(ds/2) → Kick(ds) → Drift(ds/2))

  where ds = L / N

  If K3L = 0 and K3sL = 0: degenerates to a single exact drift Drift(L)

The complete map is:

Thin lens:

.. math::

  \mathcal{M}_{\text{thin}} = \text{Kick}(K_{3L}, K_{3sL})

Thick lens (N slices):

.. math::

  \mathcal{M}_{\text{thick}} = \left[\mathcal{M}_{\text{DKD}}(\Delta s)\right]^N

where the DKD map for each slice is:

.. math::

  \mathcal{M}_{\text{DKD}}(\Delta s) = D\!\left(\frac{\Delta s}{2}\right) \circ K(\Delta s) \circ D\!\left(\frac{\Delta s}{2}\right)

.. note::

  - Thin lens mode does not change the particle position coordinates :math:`(x, y, z)`, only applies momentum kicks
  - Dispersion-related effects in thick lens mode are naturally introduced through the :math:`p_z` expression in exact drift
  - When :math:`K_{3L} = 0` and :math:`K_{3sL} = 0`, the thick lens degenerates to a pure drift, avoiding meaningless empty kick loops

Physical Derivation
--------------------

Hamiltonian
~~~~~~~~~~~

In the Cartesian coordinate system (octupole has no curvature, :math:`h = 0`), the octupole Hamiltonian is:

.. math::

  H_{\text{oct}} = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2} + \frac{\chi}{24}\left[K_3(x^4 - 6 x^2 y^2 + y^4) + K_{3s}(4 x^3 y - 4 x y^3)\right]

Splitting it into the propagation part (exact drift :math:`H_D`) and the kick part (:math:`H_K`):

.. math::

  H_D = \frac{p_\tau}{\beta_0} - \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}

.. math::

  H_K = \frac{\chi}{24}\left[K_3(x^4 - 6 x^2 y^2 + y^4) + K_{3s}(4 x^3 y - 4 x y^3)\right]

where :math:`H_D` is the exact drift Hamiltonian (preserving the :math:`p_z` square root without small-momentum expansion), and :math:`H_K` is the octupole kick. This is the standard **split-operator** method: the Hamiltonian is split into analytically solvable parts, maps are applied separately, and then combined into a symplectic integrator.

Exact Drift Map D
~~~~~~~~~~~~~~~~~

The Hamilton's equations of the propagation part give the exact drift:

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

  \beta = \frac{(1+\delta) \, \beta_0 \gamma_0}{\sqrt{1 + \left[(1+\delta) \, \beta_0 \gamma_0\right]^2}}

.. note::

  The meaning of "exact": the drift part preserves the exact square root :math:`p_z = \sqrt{(1+\delta)^2 - p_x^2 - p_y^2}` without small-momentum expansion :math:`p_x \ll 1`. The approximation lies only in separating the propagation part from the kick part (split-operator method). This formula is identical to the exact drift in the Drift element and the Quadrupole and Sextupole elements.

Octupole Kick Map K
~~~~~~~~~~~~~~~~~~~

The kick part is a thin lens map (position unchanged, momentum jump only). From Hamilton's equations :math:`\dot{p}_x = -\partial H / \partial x`, :math:`\dot{p}_y = -\partial H / \partial y`:

.. math::

  \Delta p_x = -\frac{\chi}{6} K_{3L} (x^3 - 3 x y^2) + \frac{\chi}{6} K_{3sL} (3 x^2 y - y^3)

.. math::

  \Delta p_y = \frac{\chi}{6} K_{3L} (3 x^2 y - y^3) + \frac{\chi}{6} K_{3sL} (x^3 - 3 x y^2)

where the kick effective length is already included in the integrated strengths :math:`K_{3L}` and :math:`K_{3sL}`.

Complex notation verification: :math:`(x+iy)^3 = (x^3 - 3xy^2) + i(3x^2y - y^3)`, the real part corresponds to the normal octupole, and the imaginary part corresponds to the skew octupole.

Physical meaning of each term:

.. list-table::
  :header-rows: 1
  :widths: 30 15 55

  * - Term
    - Source
    - Physical Meaning
  * - :math:`-\frac{\chi}{6} K_{3L} (x^3 - 3xy^2)`
    - :math:`\frac{\chi K_3}{24} x^4`
    - Horizontal cubic nonlinear focusing (proportional to :math:`x^3`)
  * - :math:`+\frac{\chi}{6} K_{3L} (3x^2y - y^3)`
    - :math:`-\frac{\chi K_3}{4} x^2 y^2`
    - Horizontal-vertical coupling kick
  * - :math:`+\frac{\chi}{6} K_{3sL} (3x^2y - y^3)`
    - :math:`\frac{\chi K_{3s}}{6} x^3 y`
    - Skew octupole horizontal coupling kick
  * - :math:`+\frac{\chi}{6} K_{3sL} (x^3 - 3xy^2)`
    - :math:`-\frac{\chi K_{3s}}{24} y^4`
    - Skew octupole vertical cubic nonlinear focusing

For thin lens mode, the integrated strengths :math:`K_{3L}` and :math:`K_{3sL}` are used directly. For DKD mode, :math:`K_3 \Delta s` and :math:`K_{3s} \Delta s` are used.

.. note::

  A normal octupole (:math:`K_3 > 0`) provides a restoring force proportional to :math:`x^3` for particles with positive offset in the horizontal direction. This is the key difference from the sextupole (proportional to :math:`x^2`) and the quadrupole (proportional to :math:`x`). The octupole focusing force is proportional to the cube of the position, making it a nonlinear element—particles far from the axis experience much stronger deflection than near-axis particles.

  Comparison with quadrupole and sextupole: the quadrupole kick depends linearly on :math:`x`, the sextupole kick depends quadratically on :math:`x`, and the octupole kick depends cubically on :math:`x`. This means the octupole does not affect particles on the reference orbit (kick is zero when :math:`x = y = 0`), and its linearization about that orbit vanishes. At finite amplitude it produces nonlinear deflection and amplitude-dependent tune shifts.

  A skew octupole (:math:`K_{3s} \neq 0`) rotates the octupole action by :math:`\pi / 8`, producing a different :math:`x`-:math:`y` coupling pattern. In practice, it is often used to simulate installation rotation errors or drive specific higher-order coupling resonances.


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

Amplitude-Dependent Tune Shift
------------------------------

The most important physical effect of the octupole is the amplitude-dependent tune shift (ADTS), which is the physical basis of Landau damping.

Physical Mechanism
~~~~~~~~~~~~~~~~~~

Considering single-plane motion (:math:`y = 0`), the normal octupole kick is:

.. math::

  \Delta p_x = -\frac{\chi}{6} K_{3L} \, x^3

For uncoupled optics at the reference momentum, :math:`J_y=0` and a weak octupole perturbation, first-order phase averaging gives:

.. math::

  \Delta Q_x = \frac{\chi J_x}{16\pi}\sum_j K_{3L,j}\beta_{x,j}^2 = \frac{\chi J_x}{16\pi}\oint K_3(s)\beta_x(s)^2\,ds

where :math:`J_x = \frac{1}{2\beta_x}(x^2 + (\beta_x p_x + \alpha_x x)^2)` is the action. The tune shift is proportional to the square of the amplitude (i.e., the action :math:`J_x`), meaning that large-amplitude particles have tunes deviating from small-amplitude particles. This produces an incoherent tune spread that can support Landau damping; actual damping also depends on the distribution and collective mode.

.. note::

  - The tune shift produced by the octupole is proportional to :math:`J` (action), i.e., proportional to the square of the amplitude
  - The tune shift produced by the sextupole through dispersion-momentum deviation coupling is proportional to :math:`\delta`, i.e., proportional to the momentum deviation
  - The tune shift of the quadrupole is independent of amplitude (linear element)
  - The octupole ADTS does not depend on dispersion (:math:`\eta_x`) and can be used at dispersion-free locations

Naturally Included Higher-Order Effects
---------------------------------------

The stated ideal octupole-field and drift model includes the following effects:

.. list-table::
  :header-rows: 1
  :widths: 30 70

  * - Effect
    - Source
  * - Amplitude-dependent tune shift
    - :math:`x^3` term in the kick; large-amplitude particles experience stronger deflection
  * - Higher-order dispersion
    - Exact :math:`p_z` in drift makes dispersion evolution contain all orders of :math:`\delta` dependence
  * - Path-length effects (:math:`R_{56}`, etc.)
    - :math:`\zeta` update in drift contains the complete :math:`R_{56}`, :math:`T_{566}`, and higher-order terms
  * - Thick-lens distribution effects
    - In DKD multi-slice, drift changes :math:`x`, and subsequent kicks act on updated coordinates
  * - :math:`x`-:math:`y` coupling
    - :math:`x^2 y`, :math:`xy^2` cross terms in the kick
  * - Resonance driving
    - 4th-order resonances (:math:`4Q_x`, :math:`2Q_x \pm 2Q_y`, :math:`4Q_y`, etc.)

.. note::

  Integration error is second order for uniform and fourth order for yoshida4 under the smooth-field assumptions of the splitting method. Check slice convergence and floating-point error. Refining the integrator does not recover fringe fields or other effects omitted by the stated field model.


Absolute normal/skew field errors and static DX/DY/DPSI alignment use the
common :ref:`en-error` interface. Alignment moves only the magnetic field;
apertures and SC boundaries remain in the design frame.

Application Scenarios
---------------------

- **Landau damping**: The octupole produces an amplitude-dependent tune shift, causing the tunes of large-amplitude particles to deviate from the working point, providing Landau damping for coherent oscillations and suppressing beam instabilities
- **Resonance suppression**: By adjusting the octupole strength, particle tunes are pushed away from dangerous resonance lines, avoiding beam loss due to resonance excitation
- **Dynamic aperture control**: The cubic nonlinear field of the octupole limits the stable phase-space region, affecting beam lifetime and dynamic aperture
- **Nonlinear coupling correction**: Using skew octupoles (``k3sl``) to control higher-order :math:`x`-:math:`y` coupling
- **4th-order resonance driving**: Placing octupoles at specific phases to drive 4th-order resonances (:math:`4Q_x`, :math:`2Q_x \pm 2Q_y`, etc.) for resonance extraction or beam scraping
- **LHC Landau damping scheme**: Distributing octupole families (MO) in the arc region to provide sufficient Landau damping over a wide energy range

References
----------

- Xsuite Physics Guide, Sec 1.10.3 (exact drift), Sec 1.10.5 (multipole)
- Xsuite source code: ``xtrack/beam_elements/elements_src/octupole.h``, ``track_magnet.h``, ``track_magnet_kick.h``, ``track_magnet_drift.h``
- Yoshida, H., "Construction of higher order symplectic integrators", Phys. Lett. A 150 (1990)
- MAD-X Physics Manual: octupole field and nonlinear transport
- Wiedemann, H., "Particle Accelerator Physics", Ch. 4 (nonlinear beam dynamics)

Internal Space Charge
------------------------------------------

A positive-length element may set ``space_charge`` (JSON ``Space charge``)
to an ``ElementSpaceCharge`` object. ``Num slices`` controls external transport,
while ``Space charge.Num kicks`` controls SC integration. See :ref:`en-internal-space-charge`
for scheduling, shared resources, supported backends and examples.
