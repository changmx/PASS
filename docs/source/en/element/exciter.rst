Exciter
====================

This module introduces the transverse exciter element **Exciter** in PASS, used to apply a prescribed transverse kick waveform to the beam. Exciters are widely used in tune measurement, beam instability studies, emittance growth, and other scenarios.

The exciter in PASS is a **thin lens element** (``length = 0``), changing only the particle's transverse momentum (:math:`p_x` or :math:`p_y`), without changing position coordinates.
Its length is fixed internally and is not an input parameter.

- Registration name: ``exciter``
- Core features:

  - Thin lens element (``length = 0``), changes only the particle's transverse momentum, without changing position coordinates;
  - Supports 4 excitation modes (``single_fm``, ``single_fm_am``, ``dual_fm``, ``dual_fm_am``);
  - Frequency parameters support both tune mode and frequency mode input methods;
  - Supports aperture checking, consistent with other elements.

The fields below configure ``PASS.para.schema.elements.ExciterItem``.
The key in ``Sequence.add(name, item)`` supplies the element name; it is not a configuration-model field.

Parameter List
--------------

General Parameters
~~~~~~~~~~~~~~~~~~

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
   * - ``is_enabled``
     - ``Enable``
     - ``bool``
     - —
     - ``True``
     - Exciter switch, options: ``true``, ``false``
   * - ``mode``
     - ``Mode``
     - ``str``
     - -
     - ``Required``
     - Excitation mode, options: ``single_fm``, ``single_fm_am``, ``dual_fm``, ``dual_fm_am``
   * - ``direction``
     - ``Direction``
     - ``str``
     - -
     - ``Required``
     - Excitation direction, options: ``x``, ``y``
   * - ``start_turn``
     - ``Start turn``
     - ``int``
     - -
     - ``Required``
     - Excitation start turn (inclusive)
   * - ``end_turn``
     - ``End turn``
     - ``int``
     - -
     - ``Required``
     - Excitation end turn (exclusive)
   * - ``aperture_type``
     - ``Aperture type``
     - ``str``
     - —
     - ``'off'``
     - Aperture type (default ``off``, available values in the Aperture chapter)
   * - ``aperture_value``
     - ``Aperture value``
     - ``list``
     - m / rad
     - ``[]``
     - Aperture parameter values (default ``[]``, meaning varies by type, see the Aperture chapter)


Kick Amplitude
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python configuration field
     - JSON key
     - Type
     - Unit
     - Default
     - Description
   * - ``kick_angle``
     - ``Kick angle (rad)``
     - ``float``
     - rad
     - ``Required``
     - Finite signed nominal kick amplitude shared by both DDS signals, applied as a normalized transverse momentum increment. Dual mode sums the signals without dividing by two.


Frequency Parameters
~~~~~~~~~~~~~~~~~~~~

Frequency parameters support two input modes, choose one.

**Tune mode** (recommended):

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python configuration field
     - JSON key
     - Type
     - Unit
     - Default
     - Description
   * - ``excite_tune``
     - ``Excite tune``
     - ``float | None``
     - -
     - ``None``
     - Excitation tune :math:`Q_{\text{excite}}`; :math:`f_c(t) = Q_{\text{excite}} f_0(t)` uses the shared prescribed reference clock
   * - ``sweep_tune``
     - ``Sweep tune``
     - ``float | None``
     - -
     - ``None``
     - Sweep tune :math:`\Delta Q`; :math:`\Delta f(t) = \Delta Q f_0(t)` uses the shared prescribed reference clock


**Frequency mode**:

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python configuration field
     - JSON key
     - Type
     - Unit
     - Default
     - Description
   * - ``central_frequency``
     - ``Central frequency (Hz)``
     - ``float | None``
     - Hz
     - ``None``
     - Center frequency :math:`f_c`
   * - ``sweep_width``
     - ``Sweep width (Hz)``
     - ``float | None``
     - Hz
     - ``None``
     - Sweep width :math:`\Delta f`


**Common frequency parameters**:

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python configuration field
     - JSON key
     - Type
     - Unit
     - Default
     - Description
   * - ``period``
     - ``Period (s)``
     - ``float``
     - s
     - ``Required``
     - Sweep period :math:`T`
   * - ``dual_sweep_offset``
     - ``Dual sweep offset``
     - ``float``
     - fraction of T
     - ``0.5``
     - DDS1 sweep lead relative to DDS2, in [0, 1]. A value other than 0.5 produces a warning and is still used; this is not a sine phase offset.
   * - ``fm_dual_frequency``
     - ``FM dual frequency (Hz)``
     - ``float | None``
     - Hz
     - ``None``
     - Obsolete compatibility input, ignored. Dual mode warns if supplied and inconsistent with :math:`1/T`.


Amplitude Modulation (AM) Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python configuration field
     - JSON key
     - Type
     - Unit
     - Default
     - Description
   * - ``am_t_ext``
     - ``AM t ext (s)``
     - ``float``
     - s
     - ``Required``
     - Beam diffusion characteristic time
   * - ``am_r0``
     - ``AM r0 (m)``
     - ``float``
     - m
     - ``Required``
     - Initial beam size
   * - ``am_delta0``
     - ``AM delta0``
     - ``float``
     - m
     - ``Required``
     - Initial beam diffusion range
   * - ``am_k_const``
     - ``AM k const``
     - ``float``
     - :math:`\mathrm{m}^2`
     - ``Required``
     - Model normalization coefficient


.. note::

  ``am_r0`` and ``am_delta0`` should be of the same order of magnitude; otherwise :math:`\exp(-r_0^2/\delta_0^2)` may suffer numerical underflow.

  In constant amplitude modes (``single_fm``, ``dual_fm``), the AM parameters do not participate in the computation and can be set to 0.

Usage Examples
--------------

Input File Example
~~~~~~~~~~~~~~~~~~

The following example uses tune mode:

.. code-block:: json

   {
       "Exciter_x": {
           "S (m)": 0.0,
           "Command": "Exciter",
           "Enable": false,
           "Mode": "single_fm",
           "Direction": "x",
           "Start turn": 100,
           "End turn": 1000,
           "Kick angle (rad)": 1e-4,
           "Excite tune": 0.44,
           "Sweep tune": 0.02,
           "Period (s)": 1e-3,
           "Dual sweep offset": 0.5,
           "AM t ext (s)": 0.0,
           "AM r0 (m)": 0.0,
           "AM delta0": 0.0,
           "AM k const": 0.0,
           "Aperture type": "off"
       }
   }

If using frequency mode, replace ``Excite tune`` and ``Sweep tune`` with:

.. code-block:: json

   {
   "Central frequency (Hz)": 1743.0,
   "Sweep width (Hz)": 79.2
   }

Mode Selection Guide
~~~~~~~~~~~~~~~~~~~~

- **Tune measurement**: ``single_fm`` is recommended; simple and effective, sweep covers the working point
- **Emittance growth study**: ``single_fm_am`` supplies a time-varying excitation amplitude
- **Two DDS channels**: ``dual_fm`` sums two independent continuous-phase sweeps, offset by half a sweep period by default
- **Complex instability study**: ``dual_fm_am`` is recommended; the most complete excitation mode

Parameter Selection Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Excitation tune**: Set to the beam working point :math:`Q_x` (horizontal) or :math:`Q_y` (vertical)
- **Sweep tune**: Depends on dispersion and tune spread; typically 0.01~0.05
- **Sweep period**: Should be much larger than the revolution period :math:`1/f_0` to ensure sufficient frequency resolution
- **Kick angle**: Set the signed nominal amplitude in radians directly, or use the voltage-to-kick converter in :doc:`../gui_tools` for a specified reference particle and energy
- **AM parameters**: :math:`r_0` and :math:`\delta_0` should be of the same order of magnitude; :math:`t_{\text{ext}}` is set according to the beam diffusion time scale

Kick Convention
---------------

Let :math:`\theta_0=\text{Kick angle (rad)}`. The element applies a prescribed
normalized transverse momentum increment, using the same nominal small-angle
convention as :doc:`kicker`:

.. math::

   \Delta p_{u,i}=\theta_0 F(t_i),\qquad p_u=\frac{P_u}{P_0},\qquad u=x\ \text{or}\ y.

For an on-momentum paraxial reference particle, :math:`\Delta u'\simeq\Delta p_u`,
which gives the input its angle unit. This is not an exact geometric rotation
by the same angle for every off-momentum or non-paraxial particle. The sign of
:math:`\theta_0` specifies the selected transverse kick direction. It already
contains any intended charge-sign convention; tracking does not multiply it by
charge sign, rigidity or a particle velocity factor.

The configured coefficient stays fixed when beam energy changes. Particles
with the same arrival time receive the same prescribed momentum increment,
subject to the mode's shared AM factor. Different arrival times sample different
waveform phases and AM values. Tune-mode frequencies use the shared reference
clock, not a separate oscillator frequency for each particle.

This is a zero-length kick: x, y, z, dp and t0 do not change. It does not model
transit through electrodes, longitudinal electromagnetic forces, energy exchange,
fringe fields or transmission-line propagation. FM phases remain continuous
across sweep resets. AM uses the same physical particle time measured from the
common trigger. Invalid incident states or non-forward post-kick states are
removed at this plane, preserving previous loss records. CPU and GPU use the
same equations.

Migrating Voltage-Based Inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Voltage (V)``, ``Gap (m)`` and ``Plate length (m)`` are no longer Exciter
inputs. Use the voltage-to-kick converter in :doc:`../gui_tools` with the desired
reference particle and energy, copy its signed result to ``Kick angle (rad)``,
and remove the hardware fields and ``Length (m)`` from the Exciter configuration.
The converter supplies the former nominal reference coefficient. A fixed input
angle does not reproduce the old electric-field model's per-particle velocity
factor or its automatic amplitude change with reference energy.

Particle Arrival Time
---------------------

``bunch.t0`` is the reference particle's actual arrival at the current exciter
location. The continuous coordinate is :math:`z=\beta_b c(T_b-t_i)`.
Exciter directly evaluates :math:`t_i=T_b-z_i/(\beta_b c)` without adding
nominal slot offsets or folding stored z. Different arrival times sample
different signal phases. Scaling z when RF changes the reference velocity
preserves this time; see :ref:`en-longitudinal-reference`.

All bunches sample one waveform with a common trigger time :math:`t_*`.
For the shared prescribed clock :math:`f_0(t)`, the trigger is the time at which
its accumulated turns reach :math:`n_{\rm start}+s/C`:

.. math::

   \int_{t_{\rm origin}}^{t_*} f_0(t)\,dt=n_{\rm start}+\frac{s}{C},
   \qquad u_i=t_i-t_*.

Both DDS phases start at zero at :math:`u=0`; arrivals before the trigger receive
zero excitation. The existing ``Start turn`` / ``End turn`` command gate also
applies. Only the sweep position is reduced modulo :math:`T`; accumulated phase
is not reset. Changing the absolute time origin does not change this waveform.

Frequency Input Modes
---------------------

The exciter's center frequency :math:`f_c` and sweep width :math:`\Delta f` support two input methods:

**Tune mode** (recommended)

Directly input the excitation tune :math:`Q_{\text{excite}}` and sweep tune :math:`\Delta Q`.
The program uses the beam's shared prescribed reference-clock frequency:

.. math::

  f_c(t) = Q_{\text{excite}} \cdot f_0(t)

.. math::

  \Delta f(t) = \Delta Q \cdot f_0(t)

The shared clock follows the automatic ideal RF-only design trajectory, not
collective changes to tracked bunch energies. Without active RF voltage it is
constant; see :ref:`en-reference-clock`. Phase integrates the instantaneous
frequency over physical time, including clock ramps; it is not computed as the
current frequency times elapsed time. ``excite tune`` and ``sweep tune`` must be
provided as a pair.

**Frequency mode**

Directly input the center frequency and sweep width (in Hz), suitable for scenarios requiring precise frequency control. ``central frequency (hz)`` and ``sweep width (hz)`` must be provided as a pair.

.. note::

  Choose one of the two modes. If ``excite tune`` is provided, tune mode is used; otherwise, frequency mode is used. In tune mode, ``excite tune`` and ``sweep tune`` must be provided as a pair.

Excitation Modes
----------------

The exciter has 4 operating modes, formed by combining two dimensions: frequency modulation (FM) method and amplitude modulation (AM) method:

.. list-table::
  :header-rows: 1
  :widths: 20 15 15 50

  * - Mode
    - FM method
    - AM method
    - Description
  * - ``single_fm``
    - Single-segment sweep
    - Constant amplitude
    - The most basic linear chirp
  * - ``single_fm_am``
    - Single-segment sweep
    - Time-varying amplitude
    - Sweep + time-varying amplitude
  * - ``dual_fm``
    - Two DDS sweeps
    - Constant amplitude
    - Sum of two independent continuous phases
  * - ``dual_fm_am``
    - Two DDS sweeps
    - Time-varying amplitude
    - Two DDS signals with the same AM factor


Frequency Modulation (FM) Dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Single-segment linear sweep (single)**

Let :math:`u=t-t_*` be elapsed time from the common trigger and
:math:`\tau=u\bmod T`. For constant center frequency and width, the phase is:

.. math::

  \phi_2(u) = 2\pi f_c u + \frac{\pi \Delta f}{T}\tau(\tau-T).

The carrier term uses the full elapsed time :math:`u`, so crossing a sweep
boundary does not clear phase.

The instantaneous frequency is:

.. math::

  f_s(u)=f_c+\Delta f\left(\frac{u\bmod T}{T}-\frac12\right).

- At :math:`\tau = 0`: :math:`f = f_c - \Delta f / 2` (start frequency)
- At :math:`\tau = T/2`: :math:`f = f_c` (center frequency)
- As :math:`\tau\to T^-`: :math:`f\to f_c+\Delta f/2`; at the reset it returns to the start frequency

The frequency sweeps linearly over :math:`[f_c - \Delta f/2,\; f_c + \Delta f/2]`, repeating every :math:`T` seconds. The center frequency :math:`f_c` should be close to :math:`Q \cdot f_0` (tune times revolution frequency) to cover the beam's resonance frequency.

**Two DDS sweeps (dual)**

Each DDS traverses the full sweep width and keeps its identity after frequencies
cross. With :math:`\alpha=\text{Dual sweep offset}` and :math:`\delta=\alpha T`:

.. math::

   f_1(u)=f_s(u+\delta),\qquad f_2(u)=f_s(u),
   \qquad S(x)=f_cx+\frac{\Delta f}{2T}(x\bmod T)((x\bmod T)-T),

.. math::

   \phi_1(u)=2\pi[S(u+\delta)-S(\delta)],\qquad
   \phi_2(u)=2\pi S(u).

The subtraction makes both initial phases zero for any sweep offset. The default
:math:`\alpha=0.5` means DDS1 resets at half-period and DDS2 at full-period; it
does not impose a half-cycle sine phase difference. A different offset produces
a warning and remains valid. The two signals are added directly, with a peak
bound of :math:`2|\theta_0|` before AM. There is no separate imposed cosine envelope.
Both signals use the single ``Kick angle (rad)`` setting and the same AM factor;
there are no separate channel amplitudes or automatic division by two.

For a time-dependent tune-mode clock the general definition is used instead:

.. math::

   f_j(t_*+u)=f_0(t_*+u)
      \left[Q_{\rm excite}+\Delta Q
      \left(\frac{(u+\delta_j)\bmod T}{T}-\frac12\right)\right],
   \qquad \phi_j(u)=2\pi\int_0^u f_j(t_*+v)\,dv,
   \qquad (\delta_1,\delta_2)=(\alpha T,0).

This continuous waveform model does not implement hardware sample-and-hold steps.


Amplitude Modulation (AM) Dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Constant amplitude**

.. math::

  A(u) = \theta_0

No time-varying AM envelope is applied. The signed coefficient is the fixed
configured kick angle, independent of particle velocity or reference energy.

**Time-varying amplitude (am)**

Based on a beam diffusion/growth model, the excitation amplitude varies over time; it need not increase monotonically:

.. math::

  A(u) = \theta_0 \cdot \text{am\_factor}(u)

where :math:`\text{am\_factor}(u)` is a dimensionless time-varying scaling factor:

.. math::

  \text{am\_factor}(u) = \sqrt{\frac{\delta^2(u)}{f_{0,*} \cdot k_{\text{const}}}}

The AM argument is the same physical elapsed time as FM,
:math:`u_i=t_i-t_*`. It is evaluated continuously, including each particle's
arrival offset, without rounding to a turn. The fixed normalization frequency
:math:`f_{0,*}=f_0(t_*)` comes from the shared prescribed reference clock at
startup; it does not follow tracked bunch energy changes or subsequent clock
ramps. AM is zero before startup.

Initial emittance fraction:

.. math::

  \varepsilon = \exp\!\left(-\frac{r_0^2}{\delta_0^2}\right)

Auxiliary diffusion quantity in the AM model:

.. math::

  \delta^2(u) = \frac{r_0^2 (1 - \varepsilon)}{L^2 \cdot D}

where:

.. math::

  L = \ln\!\left(\frac{u}{t_{\text{ext}}}(1 - \varepsilon) + \varepsilon\right)

.. math::

  D = t_{\text{ext}} \cdot \varepsilon + u (1 - \varepsilon)

With lengths in metres and times in seconds, :math:`\delta^2(u)` has units
:math:`\mathrm{m}^2/\mathrm{s}` and :math:`k_{\text{const}}` has units
:math:`\mathrm{m}^2`, making :math:`\text{am\_factor}` dimensionless.

The diffusion law is unchanged and diverges at :math:`u=t_{\text{ext}}`.
Choose an excitation window that keeps particle elapsed times below this limit.
The GUI rejects previews reaching it; input validation warns about reference
windows reaching it. Tracking does not add an amplitude cutoff.

Physical meaning:

- :math:`r_0`: Initial beam size
- :math:`\delta_0`: Initial beam diffusion range
- :math:`t_{\text{ext}}`: Beam diffusion characteristic time
- :math:`k_{\text{const}}`: Model normalization coefficient
- :math:`\varepsilon`: Initial emittance fraction (a measure of the :math:`r_0 / \delta_0` ratio)

This prescribed AM envelope is intended for transverse excitation and diffusion
studies. It does not itself calculate emittance growth or mechanical energy
gain: the thin kick leaves ``dp`` unchanged, and the beam response depends on
the lattice and the sampled excitation phases.

Complete Formulas for Each Mode
-------------------------------

1. **single_fm** (one DDS + constant amplitude)

.. math::

  \text{kick}_i=\theta_0\sin\phi_2(u_i).

2. **single_fm_am** (one DDS + time-varying amplitude)

.. math::

  \text{kick}_i=\theta_0\,\text{am\_factor}(u_i)\sin\phi_2(u_i).

3. **dual_fm** (two DDS channels + constant amplitude)

.. math::

  \text{kick}_i=\theta_0[\sin\phi_1(u_i)+\sin\phi_2(u_i)].

4. **dual_fm_am** (two DDS channels + time-varying amplitude)

.. math::

  \text{kick}_i=\theta_0\,\text{am\_factor}(u_i)[\sin\phi_1(u_i)+\sin\phi_2(u_i)].

For particle :math:`i`, :math:`u_i=t_i-t_*` and :math:`\theta_0` is the
configured signed kick angle. The kick is zero for :math:`u_i<0`. FM and AM
sample the same physical elapsed time :math:`u_i`. These are the final normalized
momentum increments; tracking applies no additional amplitude conversion.

Kick Application
----------------

The exciter is a thin lens element; the kick is directly added to the normalized momentum in the corresponding direction:

.. math::

  p_x \leftarrow p_x + \text{kick} \quad (\text{direction} = x)

.. math::

  p_y \leftarrow p_y + \text{kick} \quad (\text{direction} = y)

The kick is applied only to alive particles (``tag > 0``); lost particles are unaffected.

After the kick is applied, the exciter performs aperture checking on particles based on the aperture parameters (``aperture_type``): if the aperture type is not ``off``, particles exceeding the aperture range are marked as lost (``tag`` set to negative); if the aperture type is ``off``, no aperture checking is performed.
