ACDipole
========

``ACDipole`` is a zero-length, single-plane transverse excitation for fixed-energy
optics measurements and prescribed frequency scans. Its amplitude and frequency
use independent physical-time programs. All bunches sample the same waveform;
neither the turn counter nor ``harmonic_id`` resets its phase.

Physics convention
------------------

For ``Direction = x`` (replace :math:`x` by :math:`y` for vertical excitation),

.. math::

   t_i=t_0-\frac{z_i}{\beta_0c},\qquad
   \Phi(t_i)=\phi_0+2\pi\int_{t_*}^{t_i} f_{\rm ac}(u)\,du,
   \qquad p_{x,i}^{+}=p_{x,i}^{-}+\theta(t_i)\sin\Phi(t_i).

Here ``Kick angle (rad)`` specifies the signed amplitude :math:`\theta`,
``Phase (rad)`` specifies :math:`\phi_0`, and ``Phase epoch (s)`` specifies
:math:`t_*`. PASS stores :math:`p_x=P_x/P_0`, so the kick is added directly.
Do not divide it by :math:`1+\delta`. Only the paraxial slope increment is
:math:`\Delta x'\simeq\Delta p_x/(1+\delta)`.

Positive ``z`` means earlier passage. The actual bunch reference passage time
``bunch.t0`` and continuous bunch-relative ``p.z`` determine the sampled time;
nominal bunch-slot positions are not part of this formula.
The model leaves positions, ``dp``, the reference energy and ``bunch.t0`` unchanged.
It models the ideal transverse excitation, without the longitudinal induction
terms of a complete time-dependent electromagnetic field, and therefore does
not claim a full six-dimensional symplectic map.

For motion along :math:`+s` in a right-handed :math:`(x,y,s)` frame, the Lorentz
force gives :math:`\theta_x=-q\int B_y ds/P_0` and
:math:`\theta_y=+q\int B_x ds/P_0`, with signed particle charge :math:`q`.
These convert physical magnetic-field integrals at fixed reference momentum.
A fixed normalized input is
not a fixed laboratory field during acceleration; such a calculation must
prescribe the appropriate normalized amplitude explicitly.

Input interface
---------------

The model is ``PASS.para.schema.elements.ACDipoleItem``. The name is supplied by
``Sequence.add(name, item)``. All numerical values must be finite.

.. list-table::
   :header-rows: 1
   :widths: 17 21 14 8 11 29

   * - Python field
     - JSON key
     - Type
     - Unit
     - Default
     - Meaning
   * - ``s``
     - ``S (m)``
     - ``float``
     - m
     - Required
     - Action position.
   * - ``length``
     - ``Length (m)``
     - ``float``
     - m
     - ``0.0``
     - Must be zero.
   * - ``direction``
     - ``Direction``
     - ``'x' | 'y'``
     - —
     - Required
     - Excited transverse plane.
   * - ``kick_angle``
     - ``Kick angle (rad)``
     - ``float | list[float]``
     - rad
     - Required
     - Signed normalized momentum amplitude.
   * - ``kick_times``
     - ``Kick time (s)``
     - ``list[float] | None``
     - s
     - ``None``
     - Amplitude sample times and active time interval.
   * - ``frequency``
     - ``Frequency (Hz)``
     - ``float | list[float]``
     - Hz
     - Required
     - Nonnegative carrier frequency; zero is allowed.
   * - ``frequency_times``
     - ``Frequency time (s)``
     - ``list[float] | None``
     - s
     - ``None``
     - Independent frequency sample times.
   * - ``phase``
     - ``Phase (rad)``
     - ``float``
     - rad
     - ``0.0``
     - Carrier phase at the phase epoch.
   * - ``phase_epoch``
     - ``Phase epoch (s)``
     - ``float``
     - s
     - ``0.0``
     - Lower limit of the frequency integral.
   * - ``is_enabled``
     - ``Is enabled``
     - ``bool``
     - —
     - ``True``
     - Enable the excitation.
   * - ``cpu_chunk_size``
     - ``CPU chunk size``
     - ``int | None``
     - particles
     - ``65536``
     - Maximum particles per CPU batch; a positive integer. ``None`` processes the whole bunch. Ignored on GPU.
   * - ``order``
     - ``Order``
     - ``int | None``
     - —
     - ``None``
     - Explicit order among commands at the same position.
   * - ``aperture_type``
     - ``Aperture type``
     - ``str``
     - —
     - ``'off'``
     - Standard transverse loss aperture.
   * - ``aperture_value``
     - ``Aperture value``
     - ``list``
     - m / rad
     - ``[]``
     - Standard aperture dimensions.

Time tables and phase
---------------------

Each supplied time array must be nonempty and strictly increasing. A value array
with multiple samples requires a matching time array of equal length. Values
between samples are interpolated linearly. A scalar or one-value array denotes
a constant value; it may also be paired with time samples.

With ``Kick time (s)``, the kick is zero outside the closed interval from the
first to last sample. Both endpoints remain active; a one-time-point interval
is active only at that instant. Without kick times, a scalar amplitude operates
continuously. Each particle is gated at its own arrival time, so a bunch can
straddle a boundary. Frequency samples hold their endpoint values outside their
interval, and phase remains continuous through every knot and amplitude boundary.

``LinearProgram`` integrates each linear frequency segment exactly. For a sweep
:math:`f(t)=f_a+\dot f(t-t_a)` within one segment,

.. math::

   \Phi(t)-\Phi(t_a)=2\pi\left[f_a(t-t_a)+\tfrac12\dot f(t-t_a)^2\right].

Using :math:`2\pi f(t)(t-t_*)` would give the wrong sweep phase. The implementation
keeps the reference time and particle offset separate and reduces phase in
cycles before evaluating the sine. Input programs are immutable snapshots;
reconstruct the command to change them. There is no turn-by-turn phase state
to reset at the start of a new tracking run.

CPU and CUDA use float64 waveform intermediates; particle arrays retain their
configured storage precision. Keeping small offsets separate avoids loss from
adding them to large epochs, but does not remove floating-point input or local
phase-product rounding. Use physically meaningful times and frequencies rather
than assuming arbitrary magnitudes retain arbitrary phase accuracy.

CPU batch size
--------------

The default ``CPU chunk size = 65536`` processes at most 65,536 particles per
NumPy batch. Explicitly setting ``None`` processes the whole bunch in one batch.
A positive integer caps each batch, bounding temporary-array memory. Changing
this setting preserves the particle map and does not create worker threads or
distribute bunches across CPU cores. GPU execution is unaffected.

For production servers, benchmark representative particle counts and waveforms
with, for example, 16,384, 65,536, 262,144, 1,048,576 particles per batch, larger
batches if appropriate, and ``None``. More RAM permits larger arrays, but larger
batches can exceed CPU cache capacity and increase memory traffic. Core count,
memory bandwidth and NUMA placement are separate considerations; a batch-size
setting alone does not provide multicore or NUMA scheduling. Choose from measured
throughput and peak memory on the target server, not a timing from another machine.

Fixed-frequency optics example
------------------------------

For a fixed-energy ring set :math:`f_{\rm rev}=\beta_0c/C` and
:math:`f_{\rm ac}=q_d f_{\rm rev}`. Choose the physical drive-frequency branch:
adding an integer to :math:`q_d` produces identical samples for a single ideal
reference passage per turn, but generally changes the phase seen by other
bunches and by particles with different arrival times.

The following example assumes :math:`f_{\rm rev}=10^6` Hz and a first reference
passage at the element at :math:`t=0`. Substitute the actual prescribed clock
and first-passage time for the configured ring.

.. code-block:: python

   from PASS.para.schema import ACDipoleItem

   f_rev = 1.0e6
   first_passage_time = 0.0
   ac = ACDipoleItem(
       s=0.0,
       direction="x",
       kick_angle=[0.0, 1.0e-6, 1.0e-6, 0.0],
       kick_times=[first_passage_time + n / f_rev for n in (0, 2000, 6000, 8000)],
       frequency=0.31 * f_rev,
       phase=0.0,
       phase_epoch=first_passage_time,
   )
   sequence.add("ac_x", ac)

Use a drive tune separated from the beam tune footprint for the measurement
plateau, and ramp the amplitude slowly compared with the inverse tune separation
in turns. Check residual free oscillations after ramp-down and amplitude
dependence before interpreting the driven trajectory as linear optics. A real
sinusoidal drive contains both rotating components; driven optics require their
two-sideband response, not an identification with unforced betatron motion.
The driven-response interpretation follows R. Miyamoto et al.,
`Parametrization of the Driven Betatron Oscillation
<https://arxiv.org/abs/0709.4192>`_, Phys. Rev. ST Accel. Beams 11, 084002 (2008).

To prescribe a sweep, replace the scalar frequency with, for example,
``frequency=[0.30*f_rev, 0.32*f_rev]`` and
``frequency_times=[first_passage_time, first_passage_time + 8000/f_rev]``.
The amplitude program is unchanged. A sweep through the tune footprint is a
different excitation experiment from a stationary optics plateau.

Comparison with MAD-X
---------------------

MAD-X ``TRACK`` implements ``HACDIPOLE``/``VACDIPOLE`` as a turn-sampled kick;
its ``FREQ`` is cycles per turn. For matching reference samples use
:math:`f_{\rm ac}=\mathtt{FREQ}\,f_{\rm rev}` and normalize the amplitude as
:math:`\theta_0=0.3\,\mathtt{VOLT}/\mathtt{PC}` (``PC`` in GeV/c).
Use :math:`\phi_0=2\pi\,\mathtt{LAG}`. MAD-X outputs the ``START`` state as turn
zero, then applies its first kick at turn one. Align PASS physical sampling at
:math:`t_n=n/f_{\rm rev}` with ``Phase epoch (s) = 0`` for that comparison;
the PASS turn counter alone does not establish this time origin.
For strictly increasing ``RAMP1`` through ``RAMP4``, use their values divided by
:math:`f_{\rm rev}` as kick times and amplitudes
:math:`[0,\theta_0,\theta_0,0]`. PASS additionally
samples individual physical arrival times and supports continuous frequency
programs. MAD-X ``PTC`` has different AC-dipole conventions and is not covered
by this ``TRACK`` mapping.

See the `MAD-X AC-dipole manual source
<https://github.com/MethodicalAcceleratorDesign/MAD-X-User-Manual/blob/master/elements.tex>`_
for its turn ramps and the distinction between ``TRACK`` and ``PTC``.
