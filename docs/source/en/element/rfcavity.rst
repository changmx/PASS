RFCavity
========================================

``RFCavity`` applies simultaneous effective-voltage components at one location. All bunches sample the same physical waveforms. CPU and CUDA share the algorithm and float64 intermediates.

.. math::

   t_i=T_b-\frac{z_i}{\beta_b c},\qquad
   U(t)=\sum_k V_k(t)\sin\!\left[2\pi\int_{t_*}^{t}f_k(u)du+\phi_k(t)\right].

Frequency must be integrated; ``2*pi*f(t)*t`` is incorrect for a chirp. ``Phase (rad)`` is an unwrapped additive phase modulation: total instantaneous frequency is the carrier frequency plus its modulation derivative divided by :math:`2\pi`. ``harmonic_id`` and the derived nominal slot position do not enter the tracking phase formula.

Energy kick and reference update
--------------------------------

For ions, energy, rest energy and momentum times c follow PASS's per-nucleon
eV convention, with signed charge factor :math:`q=Z/A`; Z denotes the charge
number, not the proton count. Electrons and positrons use per-particle quantities
and charge factors -1 and +1, without dividing by their zero nucleon count.
The implementation uses ``sign(bunch.num_charge) * bunch.qm_ratio``, retaining
the existing magnitude convention for ``qm_ratio``. All components in one
command are sampled at the same entry event and summed before one update:


.. math::

   E_i'=\sqrt{[P_{0,b}(1+\delta_i)]^2+m^2}+qU(t_i),\qquad
   E_{0,b}'=E_{0,b}+qU(T_b),

.. math::

   P_i'=\sqrt{(E_i'-m)(E_i'+m)},\quad
   P_{0,b}'=\sqrt{(E_{0,b}'-m)(E_{0,b}'+m)},\quad
   \delta_i'=P_i'/P_{0,b}'-1,

.. math::

   p_{x,y}'=p_{x,y}P_{0,b}/P_{0,b}',\qquad
   T_b'=T_b,\qquad z_i'=z_i\beta_b'/\beta_b.

A longitudinal field preserves mechanical transverse momentum. A zero-length kick adds no passage time; scaling z preserves time continuously and does not rescale physical energy deviations. An invalid reference energy raises an error. Stopped or non-forward particles are marked lost. Momentum acceptance follows the total kick. Zero voltage still checks apertures; a disabled command does nothing. Saved slice intervals, widths and membership stay unchanged until the user executes Slicer.

In particular, stored :math:`p_x=P_x/P_{0,b}` is normalized momentum, not a
trajectory slope. After the reference update,
:math:`P_x'=P_{0,b}'p_x'=P_{0,b}p_x=P_x`, and similarly for y. The slope
:math:`dx/ds=P_x/P_s` can change with longitudinal momentum: a smaller angle
during acceleration does not imply a smaller mechanical transverse momentum.
These statements apply to the ideal longitudinal thin kick used here; transverse
electromagnetic fields and RF focusing in a finite cavity are outside this model.

Stable weak kicks
-----------------

The implementation evaluates the same exact momentum map without subtracting
nearly equal momenta. With :math:`g_i=qU(t_i)`, :math:`g_0=qU(T_b)`, and
:math:`r=P_{0,b}/P_{0,b}'`, rationalizing the momentum difference gives:

.. math::

   P_i'-P_i=\frac{g_i(2E_i+g_i)}{P_i'+P_i},\qquad
   \delta_i'=r\delta_i
   -\frac{g_0(2E_{0,b}+g_0)}{P_{0,b}'(P_{0,b}'+P_{0,b})}
   +\frac{g_i(2E_i+g_i)}{P_{0,b}'(P_i'+P_i)}.

This retains small RF kicks that direct evaluation of
:math:`P_i'/P_{0,b}'-1` can lose to rounding. It is an algebraic rearrangement,
not a linearized kick.

GPU implementation
------------------

CPU uses NumPy arrays. GPU uses a fused ``RawKernel`` in
``PASS/commands/element/rfcavity.py``: each live particle evaluates all waveform
components, applies the exact energy/reference transformation, and records
longitudinal losses in one kernel launch per nonempty bunch. The standard
transverse aperture check follows separately when enabled.

Time, phase, energy and momentum intermediates remain float64 with either
float32 or float64 particle storage. No full-size temporary gain, time, mask or
energy arrays are allocated by the RF kernel. Constant waveforms have a
specialized path; programmed waveforms use cached device tables with the same
linear interpolation, integrated frequency and held endpoints as CPU.
Table lookup compares local offsets to shifted knots, preserving small
intra-bunch arrival differences at large reference times.

The scalar accumulated phase is reduced modulo one cycle using high-precision
host arithmetic before conversion to float64. Each particle then adds its local
frequency integral about that reference event. This avoids rounding a large
``frequency * elapsed_time`` product or adding a tiny particle offset to a long
table interval. CPU and CUDA use the same cached reference phase; particle
calculations and storage retain the precision described above. This does not
recover timing information already lost in the supplied float64 reference time.

Waveform descriptors are passed by value for up to 32 components. Larger lists
use a packed device buffer to stay within the portable kernel argument limit.
Device tables and compiled kernels are cached per command, device and particle
precision. Initial compilation and uploads should be excluded from steady-state
benchmarks. The old single-component RawKernel implements different coordinate
and precision rules and cannot be substituted for the current physical map.

Python scalar and array program evaluation are both owned by ``PASS.utils.program.LinearProgram``;
CPU and GPU also share the host reference-kick calculation. Programs hold owned,
read-only copies of their time and value arrays, preventing external input edits
from invalidating interpolation coefficients, integrals or device caches. To
replace a prescribed waveform, construct a new program and its consuming command
instead of editing program arrays in place during tracking.

Input interface
---------------

.. list-table::
   :header-rows: 1

   * - JSON key
     - Type
     - Default
     - Meaning
   * - S (m)
     - float
     - Required
     - Physical location
   * - Length (m)
     - float
     - 0
     - Must be zero
   * - Is enabled
     - bool
     - true
     - Enable execution
   * - Components
     - list
     - Required
     - One or more RFComponent objects
   * - Dp aperture
     - [float,float]
     - None (off)
     - Ordered final delta bounds
   * - Aperture type / Aperture value
     - str / list
     - off / []
     - Standard transverse aperture

Each component selects one frequency definition:

* ``Frequency (Hz)``: prescribed positive carrier frequency, scalar or list.
* ``Harmonic``: positive integer multiplying the shared ``Reference clock`` revolution frequency. It neither follows current bunch energy nor needs to be divisible by the grouping harmonic.

``Voltage (V)`` and ``Phase (rad)`` default to zero and accept scalars or lists. Lists share finite, strictly increasing ``Time (s)`` samples. Programs are piecewise linear with held endpoints. See :ref:`en-reference-clock` for the reference clock and defaults.


.. code-block:: json

   {
     "Command": "RFCavity", "S (m)": 0.0,
     "Components": [
       {"Voltage (V)": 100000.0, "Frequency (Hz)": 5000000.0, "Phase (rad)": 0.3},
       {"Voltage (V)": 20000.0, "Frequency (Hz)": 10000000.0, "Phase (rad)": 1.2}
     ]
   }

.. code-block:: python

   from PASS.para.schema import RFCavityItem, RFComponent

   rf = RFCavityItem(s=0.0, components=[
       RFComponent(voltage=100e3, harmonic=1, phase=0.3),
       RFComponent(voltage=[0., 20e3], frequency=[10e6, 10.1e6],
                   times=[0., 0.01], phase=1.2),
   ])

Files and synchronous programs
------------------------------

Use ``{"Program file": "rf.tfs"}`` for ``TIME, VOLTAGE, FREQUENCY, PHASE`` columns in s, V, Hz and rad. With ``{"Program file": "rf.tfs", "Harmonic": 2}``, the file contains ``TIME, VOLTAGE, PHASE`` and must not also define FREQUENCY. File mode cannot be mixed with inline waveform data.

The old cavity-level ``Voltage (V)``, ``Harmonic``, ``Phase (rad)``, ``Phi offset (rad)``, ``RF data file`` and turn-indexed RF tables are removed. ``convert_rf_data(input_path, output_path)`` converts physical-time tables without converting seconds to turns.

``PASS.para.tools.rf_data.synchronous_rf_program`` builds a prescribed waveform from voltage, target passage phases and an explicit design-particle energy/flight-time trajectory. It generates input only; tracking does not reset actual bunch phases or energies. The integrated carrier and additive phase program jointly hit the requested design sample phases, with declared linear interpolation between samples.

Physical scope
--------------

The model is an ideal longitudinal, zero-length kick with effective voltage. It does not add finite-gap transit dynamics, RF transverse focusing or cavity trajectories. Do not multiply a transit-time factor twice if the supplied voltage already includes it. Exact RF kinematics do not remove approximations in other transport maps or quasi-static collective effects.

Physical RF tables are written with ``colwidth=25, headerswidth=25`` in tfs-pandas to retain float64 timing. In the synchronous input generator, ``origin`` is the first cavity passage and ``time_origin`` is the shared waveform epoch; these need not be equal.
