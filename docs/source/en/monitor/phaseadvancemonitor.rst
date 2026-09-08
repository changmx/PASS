PhaseAdvanceMonitor
===================

``PhaseAdvanceMonitor`` measures the uncoupled horizontal and vertical
fractional tune of each macro particle at one fixed lattice location.  It is
the only phase-advance monitor API and replaces the former ``PhaseMonitor``.

The monitor subtracts the configured closed orbit and horizontal dispersion,
then uses fixed design optics to form

.. math::

   u_x = (x-x_{CO}-D_x\delta)/\sqrt{\beta_x},\qquad
   v_x = \alpha_x u_x + \sqrt{\beta_x}(p_x-p_{x,CO}-D'_x\delta).

The same construction is used in y without dispersion. From consecutive turns
it accumulates the directed phase advance and writes only its equivalent
fractional tune, ``sum(dmu)/(2 pi N)``. Phase itself is deliberately not an
output field because it carries no additional information.

This monitor assumes uncoupled transverse optics. With x-y coupling its x/y
values are projected tunes, not normal-mode tunes.

Configuration
-------------

``Turn ranges`` accepts ``[start, end)`` pairs representing the zero-based
left-closed, right-open interval ``[start, end)``. At least two turns are
required. The value ``0`` (or an empty list) disables analysis, and ``Enable``
can disable the monitor without removing it from the sequence. Out-of-range
endpoints are clipped to ``[0, num_turns]``.

.. code-block:: json

   "tune_1": {
       "S (m)": 12.5,
       "Command": "PhaseAdvanceMonitor",
       "Beta x (m)": 18.0,
       "Beta y (m)": 22.0,
       "Alpha x": -0.4,
       "Alpha y": 0.2,
       "Dx (m)": 1.1,
       "Dpx": 0.03,
       "X CO (m)": 0.0,
       "PX CO": 0.0,
       "Y CO (m)": 0.0,
       "PY CO": 0.0,
       "Enable": true,
       "Turn ranges": [[0, 1024]]
   }

``Dx``, ``Dpx`` and all closed-orbit fields default to zero. ``Min action``
is optional; it excludes a sample when its normalized action is too small for
a stable angle. The precision-specific default is ``5e-17`` for float64 and
``5e-9`` for float32.

Output
------

One TFS file is written for each beam, bunch, monitor, and completed window in
the ``tuneSpread`` output directory. Rows remain separate by bunch and include
lost particles. Columns are ``tag``, ``tuneXFractional``,
``tuneYFractional``, per-plane interval counts, ``validX/Y``, ``completeX/Y``,
and loss metadata. ``valid`` means at least one accepted interval while the
particle remains alive; ``complete`` means every interval in the window was
accepted. Lost particles are never included in phase accumulation.

Headers record the complete fixed optical reference, window endpoints,
expected interval count, backend, precision, and ``PASSVersion``.
