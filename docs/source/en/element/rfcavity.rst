RFCavity
========================================

``RFCavity`` applies simultaneous effective-voltage components at one location. All bunches sample the same physical waveforms. CPU and CUDA share the algorithm and float64 intermediates.

.. math::

   t_i=T_b-\frac{z_i}{\beta_b c},\qquad
   U(t)=\sum_k V_k(t)\sin\!\left[2\pi\int_{t_*}^{t}f_k(u)du+\phi_k(t)\right].

Frequency must be integrated; ``2*pi*f(t)*t`` is incorrect for a chirp. ``Phase (rad)`` is an unwrapped additive phase modulation: total instantaneous frequency is the carrier frequency plus its modulation derivative divided by :math:`2\pi`. ``harmonic_id`` and the derived nominal slot position do not enter the tracking phase formula.

The fields below configure ``PASS.para.schema.elements.RFCavityItem``.
The key in ``Sequence.add(name, item)`` supplies the element name; it is not a configuration-model field.

Input interface
---------------

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
     - Must be zero
   * - ``is_enabled``
     - ``Is enabled``
     - ``bool``
     - —
     - ``True``
     - Enable execution
   * - ``components``
     - ``Components``
     - ``list[RFComponent]``
     - —
     - ``Required``
     - One or more RFComponent objects
   * - ``dp_aperture``
     - ``Dp aperture``
     - ``list[float] | None``
     - 1
     - ``None``
     - Ordered final delta bounds
   * - ``aperture_type``
     - ``Aperture type``
     - ``str``
     - —
     - ``'off'``
     - Standard transverse aperture
   * - ``aperture_value``
     - ``Aperture value``
     - ``list``
     - m / rad
     - ``[]``
     - Standard transverse aperture


Each component selects one frequency definition:

* ``Frequency (Hz)``: prescribed positive carrier frequency, scalar or list.
* ``Harmonic``: positive integer multiplying the shared ``Reference clock`` revolution frequency. It neither follows current bunch energy nor needs to be divisible by the grouping harmonic.

``Voltage (V)`` and ``Phase (rad)`` default to zero and accept scalars or lists.
Lists share finite, strictly increasing ``Time (s)`` samples and use piecewise
linear interpolation. For every component with supplied time samples, including
scalar values on a time grid, its voltage is exactly zero outside the closed
interval from the first to the last sample. Both endpoints retain their supplied
values; a one-sample table is active only at that instant. A scalar component
without ``Time (s)`` operates continuously. This also applies to program files;
RF table voltages are not held outside their data range.

Each particle uses its own arrival time :math:`t_i`, and the reference particle
uses :math:`T_b`; particles in the same bunch can therefore lie on opposite
sides of a data boundary. The frequency program, its integrated carrier phase,
and the shared reference clock remain continuous and retain their existing
endpoint extrapolation. Only the component voltage is gated by the data domain.
See :ref:`en-reference-clock` for the reference clock and defaults.


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

Numerical precision
-------------------

CPU and GPU use the same energy and reference transformations. Particle storage
may be float32 or float64; time, phase, energy and momentum intermediates use
float64. Local frequency integrals around the reference event preserve small
arrival-time differences without adding them directly to a large accumulated phase.
This cannot recover information already lost in the supplied reference time.
Writing small increments back to float32 coordinates can still cause rounding.

Time nodes and waveform values are fixed inputs. Rebuild the configuration and
its command when changing a prescribed waveform; do not mutate input arrays
in place during tracking.

Files and synchronous programs
------------------------------

Use ``{"Program file": "rf.tfs"}`` for ``TIME, VOLTAGE, FREQUENCY, PHASE`` columns in s, V, Hz and rad. With ``{"Program file": "rf.tfs", "Harmonic": 2}``, the file contains ``TIME, VOLTAGE, PHASE`` and must not also define FREQUENCY. File mode cannot be mixed with inline waveform data.

The file's first and last ``TIME`` samples define the component's voltage domain;
outside it the voltage is zero, without any additional configuration parameter.

RF tables use physical time. ``convert_rf_data(input_path, output_path)`` converts the table format without turning seconds into turn numbers.

``PASS.para.tools.rf_data.synchronous_rf_program`` builds a prescribed waveform from voltage, target passage phases and an explicit design-particle energy/flight-time trajectory. It generates input only; tracking does not reset actual bunch phases or energies. The integrated carrier and additive phase program jointly hit the requested design sample phases, with declared linear interpolation between samples.

HIAF RF chart exports
~~~~~~~~~~~~~~~~~~~~~

``PASS.para.tools.hiaf_rf`` reads the 14 two-column ``#RF...PlotData`` files
in a HIAF chart-export directory. The unsuffixed names describe channel 0;
suffixes ``1`` and ``2`` describe channels 1 and 2. It checks finite values,
one strictly increasing common time grid, and integer harmonic labels.
The source units are ms, kV, kHz and rad; output uses s, V, Hz and rad.
Nonuniform time spacing is retained. The converter does not adjust frequencies
for an ion mass or infer the source's mass convention.

For inspection without assuming a phase convention, run:

.. code-block:: console

   python -m PASS.para.tools.hiaf_rf input/hiaf_export runs/rf_import

This writes ``hiaf_rf_normalized.tfs`` and ``conversion_report.json`` with
source filenames, SHA-256 hashes, units and time range. Existing outputs are
protected; ``--overwrite`` explicitly allows replacing matching generated files.

Executable RF requires an explicit phase mapping and clock epoch. A JSON file
maps each ``channel:harmonic`` to coefficients of exported phase columns;
``offset`` optionally adds a constant in radians. For example, one declared
interpretation of BRing capture and acceleration before bunch merging is:

.. code-block:: json

   {
     "0:4": {"Phase": 1},
     "1:8": {"Phase": 2, "DeltaPhi1": 1}
   }

Save this as ``phase_rules.json`` and convert an applicable interval:

.. code-block:: console

   python -m PASS.para.tools.hiaf_rf input/hiaf_export runs/rf_import_mapped --phase-rules phase_rules.json --phase-origin 0.045006 --end-time 0.312614

Those times are an example for the 2026-10-05 BRing export, not universal machine
constants. ``--start-time`` and ``--end-time`` select physical seconds within
the source range. Analog values are interpolated at a new endpoint; harmonic
labels are discrete. The full exported clock history is retained when cropping
component domains so their integrated phase does not change. ``--phase-origin`` sets the physical epoch of zero
integrated reference-clock phase and does not shift the exported timestamps.

The example explicitly sets :math:`\psi_4=\mathrm{Phase}` and
:math:`\psi_8=2\mathrm{Phase}+\mathrm{DeltaPhi1}`. The observed relation
:math:`\mathrm{DeltaPhi1}=\pi-2\mathrm{Phase}` then gives
:math:`\psi_8=\pi`. This relation supports this interpretation but does not
establish the control system's phase semantics. Confirm them before treating
the output as a reproduction of machine operation. Phase columns must already
be unwrapped; the converter neither unwraps nor silently adds ``Phase1``.
The later :math:`h=2,1` bunch-merging stages need their own explicit rules.

Mapped conversion additionally writes ``rf_config.json`` with ``Reference clock``
and ``Components`` entries, plus one TFS per constant-harmonic segment. Copy
``Reference clock`` into the root-level MainConfig settings and ``Components`` into
the RFCavity settings. The clock frequency is the base channel's exported
frequency divided by ``--base-harmonic`` (default 4). Each active channel must
agree with its harmonic times that shared frequency. PASS integrates this
prescribed clock in physical time; particle energies do not redefine it.
The nearest zero-voltage nodes preserve the exported linear on/off ramps,
and voltage vanishes outside a component's domain. A harmonic switch without
a zero-voltage separator is rejected. Unknown active phase rules and frequency
inconsistencies are also rejected before output is written.

The Python interface is ``load_hiaf_rf(source_directory)`` for normalized arrays,
or ``convert_hiaf_rf(source_directory, output_directory, phase_rules=...,
phase_origin=..., base_harmonic=4, start_time=None, end_time=None)`` for output.
Rebuilding a mass-consistent design waveform is a separate, explicitly recorded
operation; ordinary file conversion preserves the supplied frequency program.

Physical scope
--------------

The model is an ideal longitudinal, zero-length kick with effective voltage. It does not add finite-gap transit dynamics, RF transverse focusing or cavity trajectories. Do not multiply a transit-time factor twice if the supplied voltage already includes it. Exact RF kinematics do not remove approximations in other transport maps or quasi-static collective effects.

Physical RF tables are written with ``colwidth=25, headerswidth=25`` in tfs-pandas to retain float64 timing. In the synchronous input generator, ``origin`` is the first cavity passage and ``time_origin`` is the shared waveform epoch; these need not be equal.
