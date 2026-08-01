Exciter
====================

This module introduces the transverse exciter element **Exciter** in PASS, used to apply transverse momentum perturbations to the beam through time-varying electric fields. Exciters are widely used in tune measurement, beam instability studies, emittance growth, and other scenarios.

The exciter in PASS is a **thin lens element** (``length = 0``), changing only the particle's transverse momentum (:math:`p_x` or :math:`p_y`), without changing position coordinates.

**Code Location**

- Source file: ``PASS/commands/element/exciter.py``
- Class name: ``Exciter`` (inherits from ``Command``)
- Registration name: ``exciter``
- Core features:

  - Thin lens element (``length = 0``), changes only the particle's transverse momentum, without changing position coordinates;
  - Supports 4 excitation modes (``single_fm``, ``single_fm_am``, ``dual_fm``, ``dual_fm_am``);
  - Frequency parameters support both tune mode and frequency mode input methods;
  - Supports aperture checking, consistent with other elements.


Physical Derivation
-------------------

The exciter consists of a pair of parallel plates with voltage :math:`V` applied across them, plate gap :math:`d`, and plate effective length :math:`L`.

The electric field strength is:

.. math::

  E = \frac{V}{d}

The force on a particle (charge :math:`Q = Z \cdot e`, where :math:`Z` is the charge number and :math:`e` is the elementary charge) is:

.. math::

  F = Q \cdot E = Z \cdot e \cdot \frac{V}{d}

The particle traverses the plates at velocity :math:`v = \beta c`, with an interaction time of:

.. math::

  \Delta t = \frac{L}{\beta c}

Therefore the momentum increment is:

.. math::

  \Delta P_x = F \cdot \Delta t = \frac{Z \cdot e \cdot V \cdot L}{d \cdot \beta c}

The normalized kick (divided by the reference particle total momentum :math:`P_0`) is:

.. math::

  \Delta p_x = \frac{\Delta P_x}{P_0} = \frac{Z \cdot e \cdot V \cdot L}{d \cdot \beta c \cdot P_0}

Using the magnetic rigidity :math:`B\rho = P_0 / Q`, this simplifies to:

.. math::

  \Delta p_x = \frac{V \cdot L}{d \cdot \beta c \cdot B\rho}

This form is uniformly applicable to proton beams (:math:`Z=1, A=1`) and ion beams (:math:`Z \neq A`), since :math:`B\rho` already contains the charge-to-mass ratio information.


Particle Arrival Time
---------------------

Different particles arrive at the exciter at different times due to their longitudinal coordinate :math:`z`:

.. math::

  t_{\text{arrive}} = t_0 - \frac{z}{\beta c}

where :math:`t_0` is the arrival time of the reference particle, and :math:`z > 0` means the particle is ahead of the reference particle (arrives earlier). This time difference causes different particles to see different phases of the excitation signal, which is the source of longitudinal-transverse coupling.

The revolution frequency is:

.. math::

  f_0 = \frac{\beta c}{C}

where :math:`C` is the ring circumference. The revolution frequency is used to convert turn number to real time.


Frequency Input Modes
---------------------

The exciter's center frequency :math:`f_c` and sweep width :math:`\Delta f` support two input methods:

**Tune mode** (recommended)

Directly input the excitation tune :math:`Q_{\text{excite}}` and sweep tune :math:`\Delta Q`; the program automatically computes the frequencies at runtime based on beam parameters:

.. math::

  f_c = Q_{\text{excite}} \cdot f_0

.. math::

  \Delta f = \Delta Q \cdot f_0

In this mode, there is no need to manually compute frequencies, and it automatically adapts to beams of different energies and circumferences. ``excite tune`` and ``sweep tune`` must be provided as a pair.

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
    - Sweep + amplitude adiabatic growth
  * - ``dual_fm``
    - Dual-segment sweep
    - Constant amplitude
    - Complex spectral coverage
  * - ``dual_fm_am``
    - Dual-segment sweep
    - Time-varying amplitude
    - The most complex excitation mode


Frequency Modulation (FM) Dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Single-segment linear sweep (single)**

Within one period :math:`T`, the phase is:

.. math::

  \theta(\tau) = 2\pi f_c \cdot \tau + \frac{\pi \Delta f}{T} \cdot \tau (\tau - T)

where :math:`\tau = t \bmod T` is the intra-period time, :math:`f_c` is the center frequency, and :math:`\Delta f` is the sweep width.

The instantaneous frequency is:

.. math::

  f(t) = f_c + \frac{\Delta f}{T}\left(\tau - \frac{T}{2}\right)

- At :math:`\tau = 0`: :math:`f = f_c - \Delta f / 2` (start frequency)
- At :math:`\tau = T/2`: :math:`f = f_c` (center frequency)
- At :math:`\tau = T`: :math:`f = f_c + \Delta f / 2` (end frequency)

The frequency sweeps linearly over :math:`[f_c - \Delta f/2,\; f_c + \Delta f/2]`, repeating every :math:`T` seconds. The center frequency :math:`f_c` should be close to :math:`Q \cdot f_0` (tune times revolution frequency) to cover the beam's resonance frequency.

**Dual-segment sweep (dual)**

One period is divided into first and second halves, each using a different phase formula, and a cosine envelope :math:`2\cos(\frac{\pi}{2}\Delta f \cdot \tau)` is introduced:

First half :math:`[0,\; T/2]`:

.. math::

  \theta_1(\tau) = 2\pi f_c \cdot \tau + \pi \Delta f \cdot (f_d \cdot \tau - 0.5) \cdot \tau

Second half :math:`[T/2,\; T]`:

.. math::

  \theta_2(\tau) = 2\pi f_c \cdot \tau + \pi \Delta f \cdot (\tau - T/2) \cdot (f_d \cdot \tau - 1.0)

where :math:`f_d` is the dual-frequency parameter. The cosine envelope is maximum (:math:`2A`) at :math:`\tau = 0` and decays over time, reducing discontinuities at period boundaries. The dual-segment phase formula produces a more complex spectral structure, capable of simultaneously covering multiple tune peaks.


Amplitude Modulation (AM) Dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Constant amplitude**

.. math::

  A(t) = A_0 = \Delta p_{x,\text{amplitude}}

i.e., the kick amplitude computed from the voltage parameters is used directly, without time variation.

**Time-varying amplitude (am)**

Based on a beam diffusion/growth model, the excitation amplitude grows over time:

.. math::

  A(t) = A_0 \cdot \text{am\_factor}(t)

where :math:`\text{am\_factor}(t)` is a dimensionless time-varying scaling factor:

.. math::

  \text{am\_factor}(t) = \sqrt{\frac{\delta^2(t)}{f_0 \cdot k_{\text{const}}}}

where :math:`t = n_{\text{eff}} / f_0` is the real time (seconds) since the start of excitation, and :math:`n_{\text{eff}}` is the effective excitation turn number.

Initial emittance fraction:

.. math::

  \varepsilon = \exp\!\left(-\frac{r_0^2}{\delta_0^2}\right)

Time-varying emittance squared:

.. math::

  \delta^2(t) = \frac{r_0^2 (1 - \varepsilon)}{L^2 \cdot D}

where:

.. math::

  L = \ln\!\left(\frac{t}{t_{\text{ext}}}(1 - \varepsilon) + \varepsilon\right)

.. math::

  D = t_{\text{ext}} \cdot \varepsilon + t (1 - \varepsilon)

Physical meaning:

- :math:`r_0`: Initial beam size
- :math:`\delta_0`: Initial beam diffusion range
- :math:`t_{\text{ext}}`: Beam diffusion characteristic time
- :math:`k_{\text{const}}`: Emittance growth coefficient
- :math:`\varepsilon`: Initial emittance fraction (a measure of the :math:`r_0 / \delta_0` ratio)

The exciter continuously injects energy into the beam; the beam oscillation amplitude increases, the emittance grows, and a larger excitation amplitude is needed to maintain the relative driving effect. The logarithmic term makes the growth start fast (steep segment) and slow down later (gentle segment), consistent with the physical characteristics of an adiabatic growth process.


Complete Formulas for Each Mode
-------------------------------

1. **single_fm** (single-segment sweep + constant amplitude)

.. math::

  \text{kick}(\tau) = A_0 \cdot \sin\!\left(2\pi f_c \cdot \tau + \frac{\pi \Delta f}{T} \cdot \tau (\tau - T)\right)

2. **single_fm_am** (single-segment sweep + time-varying amplitude)

.. math::

  \text{kick}(\tau) = A_0 \cdot \text{am\_factor}(t) \cdot \sin\!\left(2\pi f_c \cdot \tau + \frac{\pi \Delta f}{T} \cdot \tau (\tau - T)\right)

3. **dual_fm** (dual-segment sweep + constant amplitude)

First half (:math:`0 \le \tau \le T/2`):

.. math::

  \text{kick} = 2 A_0 \cos\!\left(\frac{\pi}{2} \Delta f \cdot \tau\right) \sin\!\left(2\pi f_c \cdot \tau + \pi \Delta f (f_d \cdot \tau - 0.5) \tau\right)

Second half (:math:`T/2 < \tau \le T`):

.. math::

  \text{kick} = 2 A_0 \cos\!\left(\frac{\pi}{2} \Delta f \cdot \tau\right) \sin\!\left(2\pi f_c \cdot \tau + \pi \Delta f (\tau - T/2)(f_d \cdot \tau - 1.0)\right)

4. **dual_fm_am** (dual-segment sweep + time-varying amplitude)

First half (:math:`0 \le \tau \le T/2`):

.. math::

  \text{kick} = 2 A_0 \cdot \text{am\_factor}(t) \cos\!\left(\frac{\pi}{2} \Delta f \cdot \tau\right) \sin\!\left(2\pi f_c \cdot \tau + \pi \Delta f (f_d \cdot \tau - 0.5) \tau\right)

Second half (:math:`T/2 < \tau \le T`):

.. math::

  \text{kick} = 2 A_0 \cdot \text{am\_factor}(t) \cos\!\left(\frac{\pi}{2} \Delta f \cdot \tau\right) \sin\!\left(2\pi f_c \cdot \tau + \pi \Delta f (\tau - T/2)(f_d \cdot \tau - 1.0)\right)

where :math:`\tau = t \bmod T`, :math:`A_0 = \frac{V \cdot L}{d \cdot \beta c \cdot B\rho}`.


Kick Application
----------------

The exciter is a thin lens element; the kick is directly added to the normalized momentum in the corresponding direction:

.. math::

  p_x \leftarrow p_x + \text{kick} \quad (\text{direction} = x)

.. math::

  p_y \leftarrow p_y + \text{kick} \quad (\text{direction} = y)

The kick is applied only to alive particles (``tag > 0``); lost particles are unaffected.

After the kick is applied, the exciter performs aperture checking on particles based on the aperture parameters (``aperture_type``): if the aperture type is not ``off``, particles exceeding the aperture range are marked as lost (``tag`` set to negative); if the aperture type is ``off``, no aperture checking is performed.


Parameter List
--------------

General Parameters
~~~~~~~~~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 35

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
  * - ``length``
    - ``length (m)``
    - float
    - m
    - Element length (must be 0)
  * - ``name``
    - ``name``
    - str
    - -
    - Element name
  * - ``is_enabled``
    - ``enable``
    - bool
    - -
    - Exciter switch, options: ``true``, ``false``
  * - ``mode``
    - ``mode``
    - str
    - -
    - Excitation mode, options: ``single_fm``, ``single_fm_am``, ``dual_fm``, ``dual_fm_am``
  * - ``direction``
    - ``direction``
    - str
    - -
    - Excitation direction, options: ``x``, ``y``
  * - ``start_turn``
    - ``start turn``
    - int
    - -
    - Excitation start turn (inclusive)
  * - ``end_turn``
    - ``end turn``
    - int
    - -
    - Excitation end turn (exclusive)
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

Hardware Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 35

  * - Property
    - JSON key
    - Type
    - Unit
    - Description
  * - ``voltage``
    - ``voltage (v)``
    - float
    - V
    - Plate peak voltage
  * - ``gap``
    - ``gap (m)``
    - float
    - m
    - Plate gap
  * - ``plate_length``
    - ``plate length (m)``
    - float
    - m
    - Plate effective length

Frequency Parameters
~~~~~~~~~~~~~~~~~~~~

Frequency parameters support two input modes, choose one.

**Tune mode** (recommended):

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 35

  * - Property
    - JSON key
    - Type
    - Unit
    - Description
  * - ``excite_tune``
    - ``excite tune``
    - float
    - -
    - Excitation tune :math:`Q_{\text{excite}}`; :math:`f_c = Q_{\text{excite}} \cdot f_0` is automatically computed at runtime
  * - ``sweep_tune``
    - ``sweep tune``
    - float
    - -
    - Sweep tune :math:`\Delta Q`; :math:`\Delta f = \Delta Q \cdot f_0` is automatically computed at runtime

**Frequency mode**:

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 35

  * - Property
    - JSON key
    - Type
    - Unit
    - Description
  * - ``cf``
    - ``central frequency (hz)``
    - float
    - Hz
    - Center frequency :math:`f_c`
  * - ``cfw``
    - ``sweep width (hz)``
    - float
    - Hz
    - Sweep width :math:`\Delta f`

**Common frequency parameters** (required for both modes):

.. list-table::
  :header-rows: 1
  :widths: 20 25 10 10 20 15

  * - Property
    - JSON key
    - Type
    - Unit
    - Applicable modes
    - Description
  * - ``period``
    - ``period (s)``
    - float
    - s
    - All modes
    - Sweep period :math:`T`
  * - ``fm_dual_frequency``
    - ``fm dual frequency (hz)``
    - float
    - Hz
    - dual_fm / dual_fm_am
    - Dual-frequency parameter :math:`f_d`

Amplitude Modulation (AM) Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
  :header-rows: 1
  :widths: 20 20 10 10 25 15

  * - Property
    - JSON key
    - Type
    - Unit
    - Applicable modes
    - Description
  * - ``am_t_ext``
    - ``am t ext (s)``
    - float
    - s
    - single_fm_am / dual_fm_am
    - Beam diffusion characteristic time
  * - ``am_r0``
    - ``am r0 (m)``
    - float
    - m
    - single_fm_am / dual_fm_am
    - Initial beam size
  * - ``am_delta0``
    - ``am delta0``
    - float
    - -
    - single_fm_am / dual_fm_am
    - Initial beam diffusion range
  * - ``am_k_const``
    - ``am k const``
    - float
    - -
    - single_fm_am / dual_fm_am
    - Emittance growth coefficient

.. note::

  ``am_r0`` and ``am_delta0`` should be of the same order of magnitude; otherwise :math:`\exp(-r_0^2/\delta_0^2)` may suffer numerical underflow.

  In constant amplitude modes (``single_fm``, ``dual_fm``), the AM parameters do not participate in the computation and can be set to 0.


Usage Examples
--------------

Input File Example
~~~~~~~~~~~~~~~~~~

The following example is taken from ``input/beam0.json``, using tune mode:

.. code-block:: json

  {
      "Exciter_x": {
          "S (m)": 0.0,
          "Command": "Exciter",
          "Length (m)": 0.0,
          "Enable": false,
          "Mode": "single_fm",
          "Direction": "x",
          "Start Turn": 100,
          "End Turn": 1000,
          "Voltage (V)": 1000.0,
          "Gap (m)": 0.1,
          "Plate length (m)": 0.3,
          "Excite tune": 0.44,
          "Sweep tune": 0.02,
          "Period (s)": 1e-3,
          "Fm Dual Frequency (Hz)": 0.0,
          "Am t ext (s)": 0.0,
          "Am r0 (m)": 0.0,
          "Am delta0": 0.0,
          "Am k const": 0.0,
          "Aperture Type": "off"
      }
  }

If using frequency mode, replace ``Excite tune`` and ``Sweep tune`` with:

.. code-block:: json

  "Central Frequency (Hz)": 1743.0,
  "Sweep Width (Hz)": 79.2,

Mode Selection Guide
~~~~~~~~~~~~~~~~~~~~

- **Tune measurement**: ``single_fm`` is recommended; simple and effective, sweep covers the working point
- **Emittance growth study**: ``single_fm_am`` is recommended; time-varying amplitude simulates adiabatic growth
- **Multi-tune-peak coverage**: ``dual_fm`` is recommended; dual-segment sweep produces a complex spectrum
- **Complex instability study**: ``dual_fm_am`` is recommended; the most complete excitation mode

Parameter Selection Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Excitation tune**: Set to the beam working point :math:`Q_x` (horizontal) or :math:`Q_y` (vertical)
- **Sweep tune**: Depends on dispersion and tune spread; typically 0.01~0.05
- **Sweep period**: Should be much larger than the revolution period :math:`1/f_0` to ensure sufficient frequency resolution
- **Voltage**: Determined by back-calculating from the required kick amplitude; typical values are in the hundreds to thousands of volts
- **AM parameters**: :math:`r_0` and :math:`\delta_0` should be of the same order of magnitude; :math:`t_{\text{ext}}` is set according to the beam diffusion time scale
