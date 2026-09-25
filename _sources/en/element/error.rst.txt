.. _en-error:

Element errors
==============

Element error models and their configuration are documented together here.
Absolute magnetic field errors and static magnetic alignment errors
(``DX``, ``DY``, ``DPSI``) are supported on CPU and GPU.

Field errors
------------

SBend, Quadrupole, Sextupole, Octupole, Multipole, Solenoid and Kicker support
the same additional normal and skew integrated multipoles on CPU and GPU.
The coefficients remain fixed during tracking. Only absolute magnetic errors
are supported; relative error generation, aperture offsets and BPM
measurement errors are outside this interface.

Parameters and normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python configuration field
     - JSON key
     - Type
     - Unit
     - Default
     - Description
   * - ``is_field_error``
     - ``Is field error``
     - ``bool``
     - —
     - ``False``
     - Enable the additional error kick.
   * - ``field_error_knl``
     - ``Field error KNL``
     - ``list[float]``
     - m^-n
     - ``[]``
     - Normal integrated errors, starting at dipole order zero.
   * - ``field_error_ksl``
     - ``Field error KSL``
     - ``list[float]``
     - m^-n
     - ``[]``
     - Skew integrated errors, starting at dipole order zero.


Array entry :math:`n` has units :math:`\mathrm{m}^{-n}`: dipole, quadrupole,
sextupole and octupole are entries 0, 1, 2 and 3. Values must be finite.
Missing orders are zero; the shorter array is padded. Exact zero errors have
no effect, and finite small values are not discarded by a magnitude threshold.
Disabling the flag preserves the nominal map even when arrays are populated.

With :math:`p_x=P_x/P_0`, :math:`p_y=P_y/P_0`, a full integrated error kick is

.. math::

   F(x,y) = \sum_{n=0}^{N}
   \frac{\Delta K_nL+i\Delta K_{ns}L}{n!}(x+iy)^n,
   \qquad \Delta p_x=-\Re F,\quad \Delta p_y=\Im F.

The input coefficients do not contain the factorial. No additional
:math:`1/(1+\delta)` factor multiplies the normalized momentum kick.
Consequently a positive normal dipole error decreases ``px``; its sign is
opposite to a positive Kicker ``HKICK``. A positive skew dipole error increases
``py``, as does ``VKICK``.

``KiL`` / ``KiSL`` in Multipole and Solenoid are nominal transverse multipoles.
Enabled error arrays are added to them once. Solenoid ``KS`` is the separate
longitudinal strength :math:`B_z/(B\rho)`; it is not a skew multipole array.
An axial solenoid strength change must be specified through ``KS``.

Tracking model
~~~~~~~~~~~~~~

A thin element applies one full integrated error kick. A thick element
distributes the integrated errors uniformly over its length: each signed
integration step :math:`\Delta s` uses the fraction :math:`\Delta s/L`.
Yoshida substeps retain their negative weights. Increasing the slice count
therefore does not increase the total error strength.

The existing nominal transport is retained. Errors share the central kick
of DKD/RKR/SKS steps; the quadrupole matrix model uses half a nominal matrix,
an error kick, and the other half. Internal space charge remains at its
scheduled node and receives its own positive integration weight.

Model limits
^^^^^^^^^^^^

SBend retains its nominal curvature and entrance/exit maps. Errors are local
straight-multipole kicks between curved nominal transport steps. This is a
specified thin-error approximation, not a complete curved multipole field or
a model of error-dependent fringe fields.

At zero Solenoid length, the axial map has no effect, but its integrated
transverse multipoles and enabled field errors still apply a thin kick.

.. code-block:: json

   {
     "Command": "Quadrupole",
     "S (m)": 1.0,
     "Length (m)": 0.3,
     "K1L": 0.15,
     "Num slices": 16,
     "Integrator": "yoshida4",
     "Is field error": true,
     "Field error KNL": [0.00001, 0.002, 0.3],
     "Field error KSL": [-0.00002, -0.001]
   }

MAD-X import
~~~~~~~~~~~~

Export nominal optics/strengths to the Twiss TFS and absolute integrated
errors to a separate ``ESAVE`` TFS after ``EFCOMP, DKN=..., DKS=...``.
PASS reads ``K0L``, ``K1L``, ... and ``K0SL``, ``K1SL``, ... from the error
table. These are additional strengths; do not also bake the same errors into
the nominal strengths or source optics. PASS does not sample relative errors.

.. code-block:: python

   from PASS.para.madx import read_madx_elements

   items, names, circumference = read_madx_elements(
       "ideal.tfs", error_file="errors.tfs", is_field_error=True)

The reader matches against the original Twiss ``NAME`` column before drift
merging or resampling. A unique name is sufficient (case-insensitive).
Repeated names require explicit occurrences such as ``Q[2]`` or ``Q:2``;
the order of rows in a sparse error table cannot identify an occurrence.
Ambiguous, missing, duplicate nonzero targets and nonfinite values raise an
error. Missing coefficient columns are zero, provided at least one supported
coefficient column exists. An empty valid table or all-zero errors is accepted.
Alignment columns are read only when ``is_alignment_error=True``; aperture
and monitor error columns are ignored.

Element import attaches errors to the physical magnet and rejects a nonzero
error on an unsupported element. Twiss transfer import places one thin
Multipole at the original element exit ``S``, after the Twiss map. Required
positions survive merging/resampling. This concentrated representation differs
from the distributed errors used in element tracking.

Solenoid element import reads ``KS = KSI/L`` from the standard MAD-X Twiss
``KSI`` column. A custom ``KS`` column is accepted if ``KSI`` is absent.
Missing both columns, or nonzero ``KSI`` at zero length, raises an error.

Alignment errors
----------------

The same seven magnetic elements support a fixed transverse displacement
``DX``, ``DY`` and roll ``DPSI``. These parameters move the nominal magnetic
field and its field-error multipoles together. Element apertures and space-charge
conducting boundaries stay at their design positions. Alignment does not move
the beam or its initial distribution.

.. list-table::
   :header-rows: 1
   :widths: 17 21 12 9 12 29

   * - Python configuration field
     - JSON key
     - Type
     - Unit
     - Default
     - Description
   * - ``is_alignment_error``
     - ``Is alignment error``
     - ``bool``
     - —
     - ``False``
     - Enable the magnetic displacement and roll.
   * - ``alignment_dx``
     - ``Alignment DX (m)``
     - ``float``
     - m
     - ``0.0``
     - Horizontal displacement in the ideal entrance frame.
   * - ``alignment_dy``
     - ``Alignment DY (m)``
     - ``float``
     - m
     - ``0.0``
     - Vertical displacement in the ideal entrance frame.
   * - ``alignment_dpsi``
     - ``Alignment DPSI (rad)``
     - ``float``
     - rad
     - ``0.0``
     - Right-handed roll about the ideal entrance longitudinal axis.


All values must be finite. The alignment and field-error switches are
independent. Disabled or exactly zero alignment takes the original tracking
path; finite small values are not discarded. Nonzero ``DS``, ``DPHI`` or
``DTHETA`` supplied as alignment components raise an error, including through
the schema and selected MAD-X alignment import. Other element types do not
support alignment errors. Random sampling, time-dependent alignment, aperture
offsets (``AREX`` / ``AREY``) and BPM measurement/calibration errors are not
implemented.

Coordinate transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~

For a straight element, entrance coordinates are transformed as

.. math::

   \begin{pmatrix}x_m\\y_m\end{pmatrix}
   = R(-\psi)\left[\begin{pmatrix}x\\y\end{pmatrix}
   -\begin{pmatrix}DX\\DY\end{pmatrix}\right],\qquad
   \begin{pmatrix}p_{xm}\\p_{ym}\end{pmatrix}
   = R(-\psi)\begin{pmatrix}p_x\\p_y\end{pmatrix}.

The nominal map and enabled field-error kicks run in this magnetic frame.
The inverse transformation restores design coordinates before the exit
aperture check. Straight-element patches leave ``z``, ``dp`` and the reference
clock unchanged. Trigonometric rotations are evaluated without a small-angle
approximation. A thin element uses the same entrance/exit plane.

A finite-length SBend uses a rigid displacement and roll of the entire
curved field about its ideal entrance. Its magnetic and design planes differ
at the exit and at internal SC nodes. At each such plane, PASS rotates the
three-dimensional normalized momentum, then projects the particle ray onto
the target plane. If the transformed position is :math:`r'` and momentum is
:math:`u'`, the intersection parameter and longitudinal time correction are

.. math::

   \lambda=-r'_z/u'_z,\qquad
   r_{\perp,\mathrm{new}}=r'_\perp+\lambda u'_\perp,\qquad
   \Delta z=-\lambda\sqrt{1-\beta_0^2+\beta_0^2(1+\delta)^2}.

Here :math:`u_z=\sqrt{(1+\delta)^2-p_x^2-p_y^2}` before rotation.
The patch preserves ``dp`` and does not advance ``bunch.t0``; the element's
``execute_cpu/gpu`` advances the reference clock once after tracking and the
exit aperture check. This preserves continuous bunch-relative
time, including the change in particle flight time between the two planes.
A non-forward or invalid intersection is recorded as a loss. Zero-length
SBend retains its existing straight thin-kick model.

Space charge, apertures and losses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With both error types and internal SC enabled, the order is:

#. Transform from the design entrance to the magnetic entrance frame.
#. Advance nominal magnetic slices and apply distributed field-error kicks.
#. At each scheduled SC node, transform live particles to the design frame,
   apply SC with its original grid and boundary, then transform back.
#. Restore the design exit frame and check the fixed element aperture.
#. Advance the reference clock once for the element length.

SC positions and integration weights are unchanged. With SC disabled, step 3
is absent; with alignment disabled, coordinate patches are absent. Upstream
lost particles remain frozen. Particles lost inside the element are restored
to design coordinates at their recorded loss plane, without further tracking
or revival. Loss-plane accuracy follows the existing stored loss-position
precision. Element aperture checks remain exit checks, not continuous wall
collision detection.

MAD-X alignment import
~~~~~~~~~~~~~~~~~~~~~~

Export ``EALIGN, DX=..., DY=..., DPSI=...`` with ``ESAVE``. The same error
TFS may contain both alignment and absolute field errors:

.. code-block:: python

   items, names, circumference = read_madx_elements(
       "ideal.tfs", error_file="errors.tfs",
       is_field_error=True, is_alignment_error=True)

Both switches default to ``False`` in element import. Set only the alignment
switch to import an alignment-only table. ``DX``, ``DY`` and ``DPSI`` are read
from the **error table**; the nominal Twiss ``DX`` / ``DY`` columns describe
dispersion and are never used as alignment offsets. Missing selected components
are zero. Nonzero unsupported alignment components and nonzero errors targeting
an unsupported element raise an error. Instance matching and duplicate checks
are shared with field-error import before drift merging.

``generate_from_tfs`` exposes the same ``is_alignment_error`` option, and the
GUI element import offers an alignment checkbox. ``read_madx_twiss`` and
``read_madx_twiss_interpolated`` explicitly reject this option: a Twiss transfer
map cannot reconstruct the displaced physical field. Use element tracking for
alignment errors. With alignment import disabled, alignment columns in a shared
error file are ignored.
