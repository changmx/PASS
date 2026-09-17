Bump
====

A Bump is a two-plane pulsed dipole in fixed machine coordinates. Every live
particle passing through it receives the kick, regardless of injection batch.
Incoming coordinates and magnetic kicks use the same fixed machine coordinate
system. Injection offsets specify the incoming beam at the injection plane;
see :ref:`en-multiturn-injection` for their role in transverse painting.
This ideal model is intended for fixed-energy injection painting. Kicks use
the injection reference momentum; acceleration scaling, magnetic fringe fields,
pole-face focusing and field errors are outside this model.

.. list-table:: Interface
   :header-rows: 1
   :widths: 30 20 50

   * - JSON field
     - Default / unit
     - Meaning
   * - Waveform file
     - Required TFS
     - Real numeric TIME, HKICK, VKICK columns; at least two finite rows, strictly increasing time.
   * - Time mode
     - particle
     - particle uses laboratory arrival time; reference uses a turn-based clock.
   * - Time offset (s)
     - 0 s
     - Added to sampling time; all batches share one global origin.
   * - Length (m)
     - 0 m
     - Thin kick at zero length; forward DKD otherwise.
   * - Num slices
     - 1
     - Strict positive integer; each slice receives its integral fraction.
   * - Enable
     - true
     - Enables kicks; drift and aperture checks remain when false.
   * - Space charge
     - None
     - Existing internal midpoint scheduling, for positive length.

The strengths are dimensionless integrated kicks::

    Kx = Delta Px / P0; Ky = Delta Py / P0
    px += Kx; py += Ky
    each slice: D(L/(2*N)) -> K(Kx/N, Ky/N) -> D(L/(2*N))

These are not exact geometric slopes. Do not divide by (1+delta) again.
Particle mode samples every kick center using the local reference passage::

    t = t0 - z/(beta0*c)

Coordinates remain continuous. Reference mode uses the prescribed machine clock inverse at the invocation turn. It omits particle z and local flight time; particle mode samples the actual local kick-center event.

Here ``t0`` is the ideal reference particle's passage time, and
``z = beta0*c*(t0-t)``. The nominal grouping slot position is not added to z;
there is no separate particle or bunch arrival correction. The waveform's
``Time offset (s)`` is added to the resulting query time. Thin Bump kicks keep
``t0``, z and the total momentum deviation unchanged. Thick elements advance
``t0`` by each reference half-drift's flight time.

Transport and loss handling
---------------------------

For positive length, every slice consists of a half drift, the transverse kick
and a second half drift. Internal space charge acts at the midpoints of equal
longitudinal subintervals. Depending on the external slicing, a space-charge
node falls at a magnetic kick center, after the magnetic kick, or at a slice
boundary. It uses the particle state and reference time at that location.
Without active internal space charge, the aperture is checked once per bunch at
the element exit. With :math:`K` internal SC nodes, each SC entry point checks
once before evaluating the source, and the exit adds one check, for :math:`K+1`
checks per passage. These checks remain active when magnetic kicks are disabled.
A thin Bump checks at its own position. Other external slice boundaries do not
add checks. Excursions that leave and re-enter the aperture between check
positions are not recorded. Newly lost particles do not enter the SC source or
receive its kick. PIC field-domain validation remains active for participating
particles at every node. See :doc:`../space_charge` for internal scheduling.

The forward map requires positive longitudinal momentum:

.. math::

   1+\delta>0,\qquad (1+\delta)^2-p_x^2-p_y^2>0.

These conditions are checked before the first kick, including a thin Bump,
and after a magnetic kick before any internal space-charge evaluation or
further drift. Particles violating them are marked lost and stop moving; a
later opposing kick cannot restore them. These momentum checks are separate
from aperture checks at SC nodes and the exit. CPU and GPU execution follow the same
transport and observation sequence. Particle
coordinates support float32 and float64; waveform times and interpolation use
float64.

Waveform boundaries
-------------------

Interpolation is linear and includes both endpoints. Before a plane's first
supplied time, its first kick value is held; after its last time, its last
value is held. An out-of-range live-particle query emits one warning per
element over its lifetime, including the supplied range of each plane.
Pending and previously lost particles do not trigger this warning.

Negative times are allowed. Supply them when the pre-injection field changes.
If the first supplied time is zero, an earlier arrival receives the value at
zero and triggers the warning. A normal ramp-down must explicitly end at zero;
a nonzero final value continues to act after the table ends.

Horizontal and vertical CSV inputs to ``convert_cisp_bump`` may have different
row counts, time nodes and time ranges, including disjoint ranges. Conversion
uses the union of all time nodes and independently holds each plane's endpoint
values outside its supplied range. It never crops the longer waveform.
The resulting TFS still uses ``TIME``, ``HKICK`` and ``VKICK`` columns.
Conversion and reading preserve float64 precision, including distinct adjacent
time nodes. Complex, boolean and string columns are rejected rather than
converted to real kick values.

Optional paired TFS headers ``HKICK_START``/``HKICK_END`` and
``VKICK_START``/``VKICK_END`` record the original supplied intervals in seconds.
They preserve per-plane out-of-range warnings after conversion, even inside
the merged table's time range. Each interval must begin and end at ``TIME`` nodes and its
column must hold its endpoint values outside that interval. Without these
headers, both planes use the full table range.

Optional TFS headers are TIME_UNIT="s" and
KICK_CONVENTION="delta_p_over_p0"; conflicting values are rejected.

