Element
================

This module introduces the various beamline elements supported in PASS.

.. _en-element-integration-precision:

Integration precision
---------------------

The symmetric fourth-order composition uses

.. math::

   S_4(h) = S_2(w_1 h) S_2(w_0 h) S_2(w_1 h), \qquad
   w_1 = \frac{1}{2-2^{1/3}}, \quad w_0 = 1-2w_1.

``PASS.utils.constants.const`` is the common source of ``yoshida_z1`` and
``yoshida_z0``. Python uses binary64 coefficients. CUDA receives binary64
compile-time constants generated from that source; it does not calculate a
cube root per particle. In the GPU split maps, magnet lengths, composed substep lengths and the
corresponding device-function arguments remain binary64, including transport
between internal space-charge nodes. Multipole kick accumulation also retains
binary64 so that the length is not rounded before the coordinate update.

This applies to the split maps of bends, quadrupoles, sextupoles, octupoles,
multipoles, kickers and solenoids with superposed multipole fields. A pure
uniform solenoid uses its exact uniform-field map. These changes preserve the
maps, field normalization and integration order.

``Particle Precision`` still controls the stored six-dimensional coordinates.
FP32 therefore denotes storage precision, not exclusively FP32 arithmetic.
Remaining intermediate types depend on the element: for example, static RKR
bends use binary64 working coordinates, whereas CPU FP32 RKR transport with
internal space-charge nodes still rounds its submaps into the live FP32 arrays.
CPU/GPU results are not required to be bit-identical. Binary64 step coefficients
reduce an avoidable source of error, but do not remove coordinate rounding or
integrator truncation error; more slices need not improve FP32 accuracy.

.. toctree::
   :maxdepth: 2

   error
   marker
   drift
   dipole
   quadrupole
   sextupole
   octupole
   multipole
   solenoid
   kicker
   bump
   elseparator
   exciter
   rfcavity
