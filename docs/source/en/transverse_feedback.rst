Bunch-by-bunch transverse feedback
===================================

``TransversePickup`` measures the live-particle centroid of each bunch grouping
slot. Its paired ``TransverseFeedback`` applies a delayed finite impulse response
(FIR) filter and a flat, zero-length transverse kick at another tracking boundary.
Use this model to study coherent oscillation damping and suppression of transverse
instabilities, including those driven by :doc:`wake_field`.

Each pickup must have exactly one feedback command in the same beam sequence.
The feedback's ``Pickup`` field references the pickup command's sequence name.
There is no separate controller command or top-level configuration block.
``Enable=false`` disables the pair. Independent centroid output remains the role
of :doc:`monitor/index`.

Model and conventions
---------------------

For a measured plane, the pickup signal in slot :math:`b` on turn :math:`n` is

.. math::

   m_{b,n}=\frac{1}{N_{b,n}}\sum_{i\in\mathrm{live}(b,n)}x_i-x_{\mathrm{ref}}.

Only particles with ``tag > 0`` contribute. The current model has equal macro
weights within a beam. The fixed reference is a pickup offset in metres; the
pickup neither changes the particles nor estimates an orbit automatically.
The vertical plane uses the same formula with :math:`y`.

The feedback computes

.. math::

   f_{b,n}=\sum_{k=0}^{L-1}a_k m_{b,n-d-k},\qquad
   \Delta p_{x,b}=\operatorname{clip}(-G_x f_{b,n},-K_x,K_x),

where ``Delay turns`` is :math:`d\geq1`, coefficients are ordered with the newest
delayed sample first, :math:`G_x` has units :math:`\mathrm{m}^{-1}`, and
:math:`K_x` is the optional dimensionless kick limit. ``Max kick x=null`` removes
that limit. The same kick is applied to every live particle in the selected slot.
``Harmonic IDs`` selects grouping slots, not particle identities or RF harmonics.

PASS stores :math:`p_x=P_x/P_0` and :math:`p_y=P_y/P_0`. These increments are
normalized transverse momentum kicks, not a direct change in geometrical slope
for every off-momentum particle. This command is an ideal thin transverse kicker;
it does not implement a voltage calibration, kicker bandwidth, intrabunch shape,
ADC noise, quantization, or saturation recovery dynamics.

The rigid-bunch measurement requires no :doc:`slicer`. Pickup and feedback must
lie at actual tracking boundaries: split a Twiss map or thick element before
placing a node inside its transport interval. At a shared ``S (m)``, use explicit
``Order`` when the physical order matters; then every command at that position
must have a distinct ``Order``. A same-position node must execute after the
transport that reaches its boundary.

Timing, startup, and grouping
-------------------------------

Delay refers to sequence turn numbers: a feedback kick on turn :math:`n` consumes
the pickup result from turn :math:`n-d`. This implementation stores turn indices,
not hardware timestamps, and does not simulate a fixed latency in seconds.
The physical interval also includes propagation between the two tracking
positions and need not equal a constant :math:`dT_{\mathrm{rev}}` during ramps.
Particle passage times retain the existing convention
:math:`t_i=t_0-z_i/(\beta c)`; ``z_center`` remains grouping metadata. No
longitudinal coordinate is folded or modified by the pickup or feedback.

Both planes share the history depth :math:`L`, the longer configured FIR length;
the shorter coefficient list is padded with zeros. After initialization or a
slot reset, :math:`L` consecutive valid samples are required before output.
For a valid sample on every turn beginning at turn zero, the earliest possible
kick is turn :math:`L-1+d`, also subject to ``Start turn``. Pickup history warms up
from turn zero even when the feedback kick starts later. Empty slots and invalid
samples suppress output and require renewed warmup. Any new injection batch,
including particles appended to an occupied slot, resets the affected slot's
feedback history before a stale output can be applied.

``SortBunch`` preserves the history of each grouping slot. Its replacement of the
particle tag array invalidates the feedback's cached particle ranges, which are
rebuilt at the next pickup or feedback execution. An unchanged array is detected
by an object-identity comparison rather than a per-turn scan of all bunches.
``ReorganizeBunch`` is incompatible with an enabled
pair during its warmup and activity interval, from turn zero through the
exclusive feedback end turn. This first implementation requires fixed grouping.
It does not follow individual particles when they migrate between slots.

Choosing FIR phase and gain
-----------------------------

The damping phase is the total phase of transport, FIR filtering, and
delay. A geometric pickup-to-kicker phase of :math:`\pi/2` is not a universal
requirement when the filter supplies phase compensation. In normalized betatron
coordinates, damping opposes the coherent momentum
:math:`P=(\alpha x+\beta x')/\sqrt{\beta}`, after subtracting the closed orbit.
Position-only feedback with the wrong phase can shift the tune or drive growth.

``design_feedback_fir(tune, phase_advance, delay_turns=1, tap_count=5, *, method="minimum_norm")``
returns real coefficients. It is an optional input-generation helper: tracking
always uses the explicit coefficient lists, and the JSON command has no
``method`` field. ``phase_advance`` is
:math:`\mu_{\mathrm{feedback}}-\mu_{\mathrm{pickup}}` in radians **within the same
sequence turn numbering**. It may be negative for a feedback node upstream of
the pickup. Do not add a further turn phase already represented by ``delay_turns``.

For :math:`x_n=\cos(\omega n+\phi)` and the design frequency :math:`\omega_0=2\pi Q`,
both methods impose

.. math::

   H_d(\omega)=\sum_k a_ke^{-i\omega(d+k)},\qquad
   H_d(\omega_0)=e^{i(\Delta\mu+\pi/2)},\qquad \sum_k a_k=0.

With the command's explicit minus sign, positive gain then opposes the coherent
momentum at the target tune. The zero-sum constraint rejects a constant pickup
offset after warmup. The two coefficient designs are:

.. list-table:: FIR design methods
   :header-rows: 1
   :widths: 22 15 63

   * - ``method``
     - Minimum taps
     - Constraints and objective
   * - ``"minimum_norm"`` (default)
     - 3
     - Three real constraints: zero DC and the specified complex response.
       Choose the smallest coefficient Euclidean norm. Existing calls retain
       this design.
   * - ``"flat"``
     - 5
     - The same constraints plus zero derivative of the delayed complex
       response at the target tune. With more than five taps, choose the
       minimum-norm solution satisfying all five real constraints.

The flat method adds

.. math::

   H'_d(\omega_0)=-i\sum_k(d+k)a_ke^{-i\omega_0(d+k)}=0.

Its linear system appends the real rows :math:`(d+k)\cos(k\omega_0)` and
:math:`(d+k)\sin(k\omega_0)` with zero targets. The factors include the full
delay :math:`d+k`, not only the tap index. At fixed pickup-to-kicker phase,
both magnitude and phase then have zero first derivative at the design tune;
the residual response variation starts at second order. This is local
flatness, not a guarantee of a specified bandwidth.

The helper derives these constraints without tracking data. It rejects
rank-deficient or ill-conditioned systems, including tunes close to integers
and half-integers, using a condition-number limit of :math:`10^8`. For the flat
method the first three rows are scaled by :math:`\sqrt L` and the derivative
rows by :math:`\sqrt{\sum_k(d+k)^2}` before solving and checking conditioning.
Sine/cosine pairs share a scale so that small rows near singular tunes are not
amplified. The unscaled constraints must also have residual at most
:math:`5\times10^{-10}`.

The helper normalizes the response at one tune. It does not establish closed-loop
stability, select a gain, compensate optics amplitude ratios, or guarantee a
required damping time. A flatter response may need larger coefficients and can
have a smaller stable gain range. Choose the design frequency with the expected
coherent centroid spectrum in mind: wakes and collective forces can shift it
away from the bare lattice tune. Flatness is computed at fixed geometric phase;
if that phase changes with machine settings, include the change in the scan.
Check the complete lattice and delay with a small gain, then scan gain, tune
and phase. Arbitrary user-supplied FIR coefficients need not
sum to zero; such a filter may kick a static orbit offset.

Python input
------------

Add the two nodes to an existing sequence whose transport has boundaries at the
chosen positions. The following phase is an example; use the actual lattice
phase in a simulation:

.. code-block:: python

   import math
   from PASS.para.api import (
       TransversePickupItem, TransverseFeedbackItem, design_feedback_fir,
   )

   taps = design_feedback_fir(
       tune=0.23, phase_advance=math.pi / 3, delay_turns=1, tap_count=5, method="flat",
   )
   seq.add("pickup_x", TransversePickupItem(s=0.0, plane="x"))
   seq.add("feedback_x", TransverseFeedbackItem(
       s=10.0, pickup="pickup_x", coefficients_x=taps,
       gain_x=0.01, delay_turns=1, max_kick_x=1e-4,
       diagnostics_interval=10,
   ))

Export with the existing ``generate_input(main, seq, output_path)`` API. There
is no new generator argument. In the GUI, the transverse-feedback component
section provides both node types; the initial feedback draft has zero gain and
zero coefficients, which must be configured before it can damp a beam.
The feedback form's **Generate FIR coefficients…** (``生成 FIR 系数…``)
uses the same helper. The referenced pickup selects x, y or both planes;
enter tune and signed same-turn phase in radians separately for each plane.
The GUI does not infer these values from node positions or machine settings.
Both planes share the method (``minimum_norm`` by default), delay and tap
count. The GUI permits 3–256 taps, at least five for ``flat``; the helper
itself has no 256-tap limit.

Use **Calculate and preview** (``计算并预览``), then **Fill draft**
(``填入草稿``) to replace measured-plane coefficients, clear unmeasured-plane
coefficients and synchronize the delay. Gains are unchanged, so an initial
zero gain still needs a deliberate choice before tracking. Coefficient arrays
remain editable; **Apply** or **Insert** saves the draft. Tune, phase and
method do not become JSON fields. See :doc:`gui` for the complete workflow.

Interface
---------

Both nodes accept ``s`` / ``S (m)`` (finite, nonnegative metres) and optional
strict-integer ``order`` / ``Order``. The sequence key supplies their name.

.. list-table:: TransversePickupItem
   :header-rows: 1
   :widths: 22 24 14 40

   * - Python parameter
     - JSON alias
     - Default
     - Meaning
   * - ``plane``
     - ``Plane``
     - ``"x"``
     - ``x``, ``y``, or independent ``xy`` measurements.
   * - ``reference_x``, ``reference_y``
     - ``Reference x (m)``, ``Reference y (m)``
     - ``0.0``
     - Fixed pickup references in metres.

.. list-table:: TransverseFeedbackItem
   :header-rows: 1
   :widths: 22 25 13 40

   * - Python parameter
     - JSON alias
     - Default
     - Meaning
   * - ``pickup``
     - ``Pickup``
     - Required
     - Exact pickup sequence name in this beam.
   * - ``enabled``
     - ``Enable``
     - ``True``
     - Strict Boolean; disables both nodes when false.
   * - ``start_turn``, ``end_turn``
     - ``Start turn``, ``End turn``
     - ``0``, ``None``
     - Kick window; nonnegative strict integers, exclusive end. Null means run end.
   * - ``delay_turns``
     - ``Delay turns``
     - ``1``
     - Strict integer of at least one.
   * - ``coefficients_x``, ``coefficients_y``
     - ``FIR coefficients x``, ``FIR coefficients y``
     - ``None``
     - Nonempty finite lists required for each measured plane; other planes must be null.
   * - ``gain_x``, ``gain_y``
     - ``Gain x (1/m)``, ``Gain y (1/m)``
     - ``0.0``
     - Signed gain in inverse metres; unmeasured planes require zero.
   * - ``max_kick_x``, ``max_kick_y``
     - ``Max kick x``, ``Max kick y``
     - ``None``
     - Nonnegative limit on absolute normalized momentum kick; null is unlimited.
   * - ``bunch_ids``
     - ``Harmonic IDs``
     - ``None``
     - Unique nonnegative strict integers; null selects all grouping slots.
   * - ``diagnostics_interval``
     - ``Diagnostics interval (turns)``
     - ``0``
     - Nonnegative strict integer; zero disables feedback diagnostics.

State and numerical scope
-------------------------

The commands register their own pair during construction and link directly when
both nodes exist, regardless of their construction or tracking order. Feedback
buffers are initialized when either node first executes; no executor preparation
hook is required. Injection commands notify the pair when a new batch enters a
slot, so the affected history is reset even if the kicker precedes the pickup.
CPU and GPU implementations reside in the same tracking module. The GPU path
keeps centroid reduction, FIR history, delayed output and particle kicks on the
device; diagnostic output is an optional, separately sampled transfer.

Particle ranges must form a complete, non-overlapping partition. GPU particle
arrays must be contiguous one-dimensional arrays with the declared precision,
length and device. These requirements are checked without transferring particle
data. A nonfinite centroid or reference-subtracted signal invalidates that sample
and requires renewed filter warmup; invalid samples do not enter the saved history.

With a positive ``Diagnostics interval (turns)``, the feedback writes
``feedback/beam<beam_id>_<safe_command_name>_<hash>.csv`` under the run output
directory. Rows contain the slot ID, pickup centroid, live count, filtered
position, applied kick and clipping flags. ``sample_turn`` identifies the latest
pickup measurement; ``filter_target_turn=sample_turn+d`` identifies the turn for
which that filtered position is queued. ``kick_turn`` identifies the applied
kick, and ``requested_source_turn=kick_turn-d`` identifies the requested delayed
source turn; a usable sample need not exist during warmup or after a reset.
The filtered position in a row generally does not generate the kick in that
same row. Output is buffered in batches of 128 sampled turns and flushed at
finalization. Existing output files are never overwritten.

``state_dict()`` and ``load_state_dict()`` save and restore only the feedback
controller's state. Matching particles, bunch reference events, clocks and turn
state remain the caller's responsibility. These APIs do not provide a complete
simulation restart; the tracking entry starts a new run from turn 0.
