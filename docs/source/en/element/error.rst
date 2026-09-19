.. _en-error:

Element errors
==============

Element error models and their configuration are documented together here.
Currently only absolute magnetic field errors are implemented; alignment
errors are not yet supported.

Field errors
------------

SBend, Quadrupole, Sextupole, Octupole, Multipole, Solenoid and Kicker support
the same additional normal and skew integrated multipoles on CPU and GPU.
The coefficients remain fixed during tracking. Only absolute magnetic errors
are supported; relative error generation, alignment, aperture offsets and BPM
measurement errors are outside this interface.

Parameters and normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 15 35

   * - Python schema field
     - JSON key
     - Default
     - Meaning
   * - ``is_field_error``
     - ``Is field error``
     - ``false``
     - Enable the additional error kick.
   * - ``field_error_knl``
     - ``Field error KNL``
     - ``[]``
     - Normal integrated errors, starting at dipole order zero.
   * - ``field_error_ksl``
     - ``Field error KSL``
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

Implementation layout
^^^^^^^^^^^^^^^^^^^^^

``PASS/commands/element/error.py`` contains ``FieldErrors``, GPU error dispatch
(``_track_field_errors_gpu``) and matrix-error composition
(``_transport_matrix_errors``). The shared ``execute_element_body_gpu`` in
``PASS/utils/slicing.py`` advances the body slices and invokes SC only at
configured nodes; it also serves matrix/bend error tracking without SC.
Matrix/bend transport and error kicks remain separate GPU calls.

Fixed multipole coefficients, inverse factorials and stage parameters are
prepared once and reused; GPU arrays are cached per precision and device.
Particle coordinates and bunch reference quantities remain live inputs.
The input flag ``Is field error`` is retained; the runtime state is held by
``FieldErrors.enabled`` and ``FieldErrors.active`` without a duplicate
element-level flag. Renaming the module does not add alignment errors.

Model limits
^^^^^^^^^^^^

SBend retains its nominal curvature and entrance/exit maps. Errors are local
straight-multipole kicks between curved nominal transport steps. This is a
specified thin-error approximation, not a complete curved multipole field or
a model of error-dependent fringe fields. It need not reproduce a native PTC
thick-bend field-error model exactly. The same caution applies to PTC commands
that handle only selected error orders on a native element; compare an
equivalent explicit thin-Multipole lattice when testing this model.

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
Alignment, aperture and monitor columns are ignored.

Element import attaches errors to the physical magnet and rejects a nonzero
error on an unsupported element. Twiss transfer import places one thin
Multipole at the original element exit ``S``, after the Twiss map. Required
positions survive merging/resampling. This concentrated representation differs
from the distributed errors used in element tracking.

Solenoid element import reads ``KS = KSI/L`` from the standard MAD-X Twiss
``KSI`` column. A custom ``KS`` column is accepted if ``KSI`` is absent.
Missing both columns, or nonzero ``KSI`` at zero length, raises an error.
