ElSeparator
===========

``PASS.commands.element.elseparator.ElSeparator`` models an electrostatic
septum with a circulating-beam field-free region, a thin separating electrode,
a field region and a high-voltage electrode. The registered command is
``ElSeparator``. CPU and GPU implement a relativistic uniform electrostatic
body map with hard-edge potential matching for positive length. Zero length
uses a single effective impulse evaluated with each particle's incident
longitudinal velocity.
Entering the field region does not itself lose a particle. Survivors continue into
the downstream lattice, where the actual later apertures determine later losses.

The fields below configure ``PASS.para.schema.elements.ElSeparatorItem``.
The key in ``Sequence.add(name, item)`` supplies the element name; it is not a configuration-model field.

Input parameters
----------------

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
     - Nonnegative supplied length, independent of roll.
   * - ``voltage``
     - ``V (V)``
     - ``float | None``
     - V
     - ``None``
     - Signed interplate voltage difference; choose V or VL. Nonzero V requires positive length.
   * - ``voltage_length``
     - ``VL (V m)``
     - ``float | None``
     - V m
     - ``None``
     - Signed longitudinal voltage integral; supports positive or zero tracking length.
   * - ``gap``
     - ``Gap (m)``
     - ``float``
     - m
     - ``Required``
     - Positive clear electrode gap.
   * - ``septum_position``
     - ``Septum position (m)``
     - ``float``
     - m
     - ``Required``
     - Surface d facing the circulating-beam field-free region, along u.
   * - ``septum_thickness``
     - ``Septum thickness (m)``
     - ``float``
     - m
     - ``0.0``
     - Nonnegative effective material thickness.
   * - ``tilt``
     - ``Tilt (rad)``
     - ``float``
     - rad
     - ``0.0``
     - Roll used in the projection and kick equations above.
   * - ``aperture_type``
     - ``Aperture type``
     - ``str``
     - —
     - ``'off'``
     - Existing vacuum aperture geometry.
   * - ``aperture_value``
     - ``Aperture value``
     - ``list``
     - m / rad
     - ``[]``
     - Existing vacuum aperture geometry.
   * - ``num_slices``
     - ``Num slices``
     - ``int``
     - 1
     - ``1``
     - Strict positive integer; sets internal SC scheduling. The isolated analytic body does not require subdivision.
   * - ``space_charge``
     - ``Space charge``
     - ``ElementSpaceCharge | None``
     - —
     - ``None``
     - Optional internal SC configuration; requires positive length.

Input configuration
-------------------

Specify V or VL, gap and septum geometry for the modeled device. This synthetic
example uses a manual polygon with ``-0.05 < x < 0.0201`` and ``|y| < 0.02``.
The internal septum remains absorbing. Replace the values with device parameters;
the selected strength input is constant on every pass.

.. code-block:: json

   {
     "Command": "ElSeparator",
     "S (m)": 1.0,
     "Length (m)": 1.0,
     "V (V)": 1000.0,
     "Gap (m)": 0.01,
     "Septum position (m)": 0.01,
     "Septum thickness (m)": 0.0001,
     "Tilt (rad)": 0.0,
     "Num slices": 16,
     "Aperture type": "polygon",
     "Aperture value": [[-0.05, -0.02], [0.0201, -0.02], [0.0201, 0.02], [-0.05, 0.02]]
   }

To use the same integrated strength in this 1 m element, replace ``V (V)`` with
``"VL (V m)": 1000.0``. To use a thin model at S=1 m, also set ``Length (m)``
to 0. The same input field integral is used at that plane, with the incident particle
speed. Its effective impulse need not equal the full thick map exactly; it omits
the thick model's transverse displacement and flight time. For example,
(x,y)=(0.005,0) survives without a kick, (0.01005,0) is lost in septum material,
and (0.015,0) survives the entry check and receives the field kick.

Injection timing is described in :ref:`en-multiturn-injection`. New particles
specified at the ES exit start tracking at that plane. Injection generates or
loads their coordinates without an additional geometric acceptance cut.
ElSeparator evaluates deflection and losses when particles subsequently pass
through the element, according to its thick or thin model.

Length, orientation and time
----------------------------

``Length (m)`` is the supplied longitudinal transport length L. ``S (m)`` is
the exit; the entrance is S-L. ``Tilt (rad)`` is a roll about the longitudinal
axis, not a pitch or yaw of the element axis. **Never multiply or divide L by
cos(tilt)** for transport, field integration, loss positions or reference time.
Horizontal, vertical and inclined separators share one implementation.

With the existing PASS roll convention, temporary geometric coordinates are

.. math::

   u=x\cos\theta-y\sin\theta,\qquad
   v=x\sin\theta+y\cos\theta.

The particle arrays remain in the beam frame. A local normal kick is applied as

.. math::

   \Delta p_x=K_u\cos\theta,\qquad
   \Delta p_y=-K_u\sin\theta.

Thus tilt=0 selects horizontal deflection and tilt=+/-pi/2 selects vertical
deflection; the sign of the voltage and the particle charge determine polarity.
Adding pi to the roll reverses the geometric normal. The sign of the septum
position does not independently select a field side.

For every traversed reference segment ds, including empty bunches,

.. math::

   t_0\leftarrow t_0+\frac{ds}{\beta_0 c},\qquad
   z_{\mathrm{rel}}=\beta_0c(t_0-t_i).

The design reference clock retains the field-free reference transit convention.
Individual flight-time differences are computed by the exact field body map or
the standard drift in the circulating-beam field-free region. There is no additional tilt time correction or arrival state.
A zero-length instance checks geometry and momentum at its plane, applies a
``VL`` kick to surviving particles in the field region, then checks momentum
validity again. It changes neither x, y, z nor reference time. It cannot locate
collisions that would occur inside the omitted physical length; use the thick
model to resolve those trajectories and loss positions. Num slices does not
repeat the thin kick, and internal space charge requires positive length.

Geometry and voltage
--------------------

Let d be ``Septum position (m)``, t be ``Septum thickness (m)`` and g be
``Gap (m)``. The high-voltage electrode's inner surface is uc=d+t+g.
These positions are relative to the reference orbit, not the ring's geometric
center. The regions, for every local v, are

* circulating-beam field-free region: u < d;
* septum material: d <= u <= d+t;
* field region: d+t < u < uc;
* high-voltage electrode material: u >= uc.

The electrode surfaces are absorbing material boundaries.
A zero-thickness septum is a closed plane: touching or crossing it still loses a
particle. This is an effective material-loss model, not a discrete-wire or
multiple-scattering simulation.

For entry contact tests, projected coordinates within
``4*epsilon*(abs(x*cos(tilt))+abs(y*sin(tilt)))`` of a material surface count as
touching; epsilon is the particle dtype's machine precision. This prevents
rotation/storage roundoff from letting a point on a zero-thickness septum pass.
Future ray intersections still use the specified surfaces without shifting them.

The plates and field are idealized as covering all local v. Use the separate
vacuum aperture for the opposite wall and transverse clearance, including
symmetric limits in v. This assumes the modeled acceptance lies within the
electrodes' good-field coverage; it does not model finite-height fringe fields.
The polygon is an outer acceptance boundary, not a replacement for the internal
septum material. ``Aperture type: off`` disables only the vacuum-wall check.

``V (V)`` is the signed septum-minus-high-voltage-electrode potential difference.
``VL (V m)`` is its longitudinal integral, not the electric-field integral:

.. math::

   V=\phi_{\mathrm{septum}}-\phi_{\mathrm{HV}},\qquad
   VL=\int V(s)\,ds,\qquad
   \int E_u(s)\,ds=\frac{VL}{g}.

For a thick uniform-field model, V input gives Eu=V/g and VL=V*L;
VL input gives Eu=VL/(g*L). For a thin model, VL is an independent input
representing the real device integral, even though the tracking length is zero.
For zero length the effective normalized impulse is

.. math::

   K_u=\operatorname{sgn}(q)\frac{VL/g}{\beta_0 c B\rho}
       \frac{A}{p_s},\qquad
   A=\sqrt{\gamma_0^{-2}+\beta_0^2(1+\delta)^2},\qquad
   p_s=\sqrt{(1+\delta)^2-p_x^2-p_y^2}.

Here beta0*c*ps/A is the incident longitudinal speed. The incident state is
frozen during this lumped impulse. It keeps dp fixed and is not the exact map
of a finite electrode. PASS stores px=Px/P0 and py=Py/P0, so this is a
normalized mechanical momentum increment, not an angular kick. Do not divide
it by an additional (1+delta). Finite length uses the energy-consistent map below.

PASS stores the positive reference rigidity magnitude. The charge sign is
applied explicitly. The current bunch beta and rigidity are read each execution;
RF updates them through ``set_reference_energy``. A constant applied voltage
therefore does not imply a constant kick during acceleration.

Specify exactly one non-null V or VL; zero is valid. Both or neither are errors.
For zero length, nonzero V is rejected with a request to use VL; V=0 permits a
pure geometry check. Gap and septum position remain required for either input.
The old ``Voltage (V)``, electrode-height/center and EX/EY/EXL/EYL fields are
not accepted. Material and vacuum checks remain active at zero strength.

Propagation and collisions
--------------------------

The septum and external field-free regions have zero potential. Within the gap,
with uf=d+t, the ideal body potential is Phi=-Eu*(u-uf). The high-voltage plate
therefore has potential -V. Define W=energy/(P0*c), mu=mc/P0 and k=q*Eu/(P0*c).
The entrance edge changes mechanical energy to W+k*(u-uf), keeping transverse
momenta and coordinates fixed. The exit edge subtracts k*(u-uf). These are
ideal zero-width longitudinal edge impulses, not measured fringe profiles.

In the uniform body, ps and pv are constant. For a forward distance ds and
a=k*ds/ps, the analytic map is

.. math::

   p_{u,2}=p_{u,1}\cosh a+W_1\sinh a,\qquad
   W_2=W_1\cosh a+p_{u,1}\sinh a,

   u_2=u_1+\frac{W_2-W_1}{k},\qquad
   v_2=v_1+\frac{p_v}{p_s}ds,\qquad
   c\Delta t=\frac{p_{u,2}-p_{u,1}}{k},

   \delta_2=\sqrt{W_2^2-\mu^2}-1,\qquad
   z_2=z_1+ds-\beta_0c\Delta t.

The implementation evaluates cancellation-free increments and the zero-field
limit, rather than subtracting nearly equal hyperbolic functions. The short
series used near zero is a floating-point evaluation technique, not a paraxial
approximation. Across the complete static element, surviving particles recover
their entry mechanical energy when no other energy-changing interaction acts.
Particles without a positive forward momentum at an edge are removed at that
edge; this represents leaving the forward tracking model, not material absorption.

Entrance and exit matching occur once per element, never at internal slice or
SC boundaries. A center SC callback receives the state after half a body slice;
a boundary callback receives the full body-slice state. Both see the correct
design reference time and mechanical particle momenta in the beam frame.
Survivors in the circulating-beam field-free region also participate.
See :ref:`en-internal-space-charge` for node placement and weights. With no
internal SC, one analytic body propagation suffices: Num slices does not change
the external-field result. With SC, convergence of the split collective map
must still be checked.

Electrode contacts are solved from the body's energy-position relation, retaining
both momentum branches so that turns and grazing contacts are included. Vacuum
wall contacts use conservative interval bounds on the analytic curve. Line and
coordinate extrema include interior turning points; ellipse bounds enclose the
whole interval. This detects non-convex excursions even with both endpoints
inside. The earliest candidate is isolated to a longitudinal interval of
1e-12*max(1, segment length) metres, with a separate 64-double-epsilon scaled
geometry tolerance. Near tangencies that spatial tolerance can produce a larger
uncertainty in the longitudinal contact location. Stored loss positions use the
existing float32 array. Field-free motion retains analytic straight intersections.

Particles stop at first contact and keep their mechanical state at that point;
an exit edge is not applied to a particle lost inside the body. Subsequent calls
preserve the first loss record. Material surfaces and vacuum walls compete for
the earliest contact. All existing aperture types remain supported in the beam
frame: off, default, rectangle, circle, ellipse, rectcircle, rectellipse,
racetrack, octagon and polygon. Tilt rotates the electrodes and field, not the
separately specified vacuum chamber.

For a manually entered local polygon, transform each vertex into beam coordinates:

.. math::

   x=u\cos\theta+v\sin\theta,\qquad y=-u\sin\theta+v\cos\theta.

Enter vertices in boundary order without repeating the first vertex to close
the polygon. Self-intersections, non-adjacent edge contacts, repeated vertices
and malformed aperture dimensions are rejected during schema validation as
well as input preflight.

Tilt=0 puts the high-voltage electrode on local +x, while tilt=pi reverses
the layout. Changing the voltage sign changes force polarity, not the geometry.

Drift and approximation limits
------------------------------

ES and :doc:`drift` share the CPU factors and CUDA inline drift function. With
T=px^2+py^2 and ps=sqrt((1+delta)^2-T), define

.. math::

   A=\sqrt{\gamma_0^{-2}+(1-\gamma_0^{-2})(1+\delta)^2},
   \qquad
   \frac{\Delta z}{ds}=
   \frac{\delta(2+\delta)\gamma_0^{-2}-T}{p_s(p_s+A)}.

This rationalized form retains small longitudinal slips in single precision.
Nonpositive total or longitudinal momentum and nonfinite longitudinal momentum
are rejected; positive longitudinal momentum is not artificially clamped.
Tracking never wraps the stored z coordinate.

The thick map is exact for the declared ideal uniform body and hard edges,
up to floating-point and collision-location tolerances. It does not include a
three-dimensional fringe map, finite-height fields, discrete septum wires,
scattering or secondary particles. Thin tracking additionally omits the actual
body trajectory, energy exchange and transit-time change, and freezes the
incident longitudinal speed during the effective impulse. A thin VL model is
not asserted to be the exact L-to-zero limit of the finite hard-edge model.

CPU and CUDA implementations reside in the same source file. GPU kernels use
double precision for analytic field trajectories and curved collision bounds;
particle arrays retain their configured float32 or float64 storage. Internal SC
nodes materialize the particle state; compiled kernels are cached per element,
device and storage dtype. No particle coordinate array is rotated in place.
