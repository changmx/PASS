Magnet ramping
==============

``Quadrupole``, ``Sextupole``, ``Octupole`` and ``Multipole`` support prescribed,
piecewise-linear programs of normalized magnetic strength. Both normal and skew
components can vary independently. ``SBend``, ``Kicker`` and ``Solenoid`` do not
support this interface.

Strength convention
-------------------

The program supplies the same normalized coefficients as the static element:
:math:`K_n` has units :math:`\mathrm{m}^{-(n+1)}` and :math:`K_nL` has units
:math:`\mathrm{m}^{-n}`. Here :math:`n=1,2,3` denotes quadrupole, sextupole and
octupole order. For a magnet of length :math:`L`, a ``K`` column is converted to
an integrated strength by multiplying by :math:`L`; a ``KL`` column already
contains that integral. Zero-length elements require integrated columns.

Values are **absolute strengths**, not scale factors. A specified program
component replaces its current nominal value; an omitted component retains its
current nominal value. Enabled field errors remain separate absolute additive
coefficients. They are added once during tracking and are not written into the
nominal strengths or scaled with the ramp.

Strengths are normalized to the current bunch reference momentum :math:`P_0`, as
in static tracking. A program does not supply a physical magnetic field or
convert a fixed field using a changing magnetic rigidity. For example,
:math:`\Delta p_x=-K_1L(t)x` for a normal thin quadrupole; no additional
:math:`1/(1+\delta)` factor multiplies this kick because :math:`p_x=P_x/P_0`.

Current strength state
----------------------

The runtime element maintains one current set of nominal strengths. For a
quadrupole these are ``k1l``/``k1sl`` and the derived per-unit-length
``k1``/``k1s``; sextupoles and octupoles use the corresponding order-2 and
order-3 attributes. ``Multipole`` uses ``knl``/``ksl`` and the derived
``kn``/``ks`` arrays. For thick elements, the per-unit-length coefficients are
the integrated coefficients divided by the full element length.

Tracking updates the programmed channels once at the entrance of each nonempty
bunch. After execution, these attributes retain the entrance values of the last
executed nonempty bunch. An empty bunch does not sample the program or change
the current strengths. No original-strength backup is kept, and removing the
runtime program does not restore the initial strengths.
Channels absent from the program retain their current values, including a value
changed since element construction. Input dictionaries, configuration objects
and source TFS files are not rewritten by these runtime updates.

Each of the four runtime element classes also provides
``update_strengths(reference_time, offset=0.0)``. It samples the program at the
specified physical reference time plus a separate local offset, updates the
current nominal and derived strengths, and invalidates affected coefficient
caches. It does not track particles or advance ``bunch.t0``. If the element has
no active program, the call leaves its current strengths unchanged. For example,
given a runtime quadrupole object:

.. code-block:: python

   element.update_strengths(reference_time=0.05, offset=0.0)
   print(element.k1l, element.k1sl)

This method belongs to the tracking element, not the ``QuadrupoleItem`` input
configuration model. The program table remains prescribed input data; it is
not a backup of the element's initial strengths. Automatic tracking calls this
update with the bunch entrance ``t0`` and zero offset.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 22 22 14 14 28

   * - Python field
     - JSON key
     - Type
     - Default
     - Description
   * - ``is_ramping``
     - ``Is ramping``
     - ``bool``
     - ``False``
     - Enable a prescribed strength program.
   * - ``ramping_file``
     - ``Ramping file``
     - ``str``
     - ``""``
     - TFS file containing ``TIME`` and strength columns.

Relative paths resolve against the input JSON directory. An element constructed
with ramping disabled uses the strengths supplied in its configuration. Legacy
per-component file fields, such as
``K1L ramping file`` and ``K1SL ramping file``, remain accepted; use the single
``Ramping file`` for new inputs. The unified file cannot be combined with any
legacy file field. Do not define one component twice through multiple legacy
files or through both ``K`` and ``KL`` columns. Legacy ``TIME_S`` is accepted
when it contains physical seconds.

.. code-block:: json

   {
     "Command": "Quadrupole",
     "S (m)": 10.0,
     "Length (m)": 0.5,
     "K1L": 0.2,
     "K1SL": 0.01,
     "Num slices": 8,
     "Is ramping": true,
     "Ramping file": "quadrupole_ramp.tfs"
   }

TFS format
----------

The independent column is ``TIME`` in physical seconds. Times must be finite
and strictly increasing; all strength values must be finite. Values between
rows are interpolated linearly. Before the first row and after the last row,
the corresponding endpoint value is held. A one-row file represents a constant
program. A ``TURN`` column alone is not a time program. The headers below are
optional for manually authored files, but are validated when present. The writer
also records column units (for example ``K1L_UNIT="m^-1"``).

.. code-block:: text

   @ TIME_UNIT %s "s"
   @ STRENGTH_CONVENTION %s "normalized"
   * TIME K1L K1SL
   $ %le %le %le
   0.00 0.20  0.01
   0.05 0.25  0.00
   0.10 0.22 -0.01

.. list-table:: Supported strength columns
   :header-rows: 1
   :widths: 22 39 39

   * - Element
     - Per-unit-length columns
     - Integrated columns
   * - ``Quadrupole``
     - ``K1``, ``K1S``
     - ``K1L``, ``K1SL``
   * - ``Sextupole``
     - ``K2``, ``K2S``
     - ``K2L``, ``K2SL``
   * - ``Octupole``
     - ``K3``, ``K3S``
     - ``K3L``, ``K3SL``
   * - ``Multipole``
     - ``K0``, ``K0S``, ``K1``, ``K1S``, ...
     - ``K0L``, ``K0SL``, ``K1L``, ``K1SL``, ...

For ``Multipole``, the order in a column name indexes the nominal ``KiL`` or
``KiSL`` input array and the runtime ``knl`` or ``ksl`` array. A program can
introduce a higher-order component absent from those arrays. Its zero-order
component is an ordinary multipole kick; it does not enable bend geometry or
ramping of the ``SBend`` element.

Entry sampling and frozen-element tracking
------------------------------------------

This interface uses a quasi-static magnet model with one prescribed strength
per bunch passage. For each nonempty bunch, both a thick magnet and a thin kick
sample the program once at the bunch reference particle's entrance time:

.. math::

   t_{\mathrm{sample}}=t_{0,\mathrm{entry}}.

The resulting nominal coefficients remain fixed throughout that element's
tracking for this bunch. Body slices, internal space-charge nodes and Yoshida
substeps all use the same entrance strengths; none resamples the program.
Different bunches use their own actual entrance ``bunch.t0``. All particles of
one bunch share those strengths: particle ``z`` does not correct the sampling
time, and ``harmonic_id``/``z_center`` grouping metadata does not shift it.

``Num slices`` controls spatial integration of the magnetic map and internal
space-charge scheduling. Increasing it can improve that integration, but does
not change the ramp sampling time or resolve strength variation during transit
through a single element. The chosen integrator's order applies to the map with
these frozen coefficients. The model assumes the prescribed strength changes
slowly enough during the element traversal for one entrance value to represent
that passage. Fast variation within an element or across a bunch is outside
this approximation.

This is an intentional change to the ramping model: tracking now freezes one
entrance value over the whole element. Results can therefore change when a
program varies appreciably during the element flight time. The configuration
schema, TFS format and GUI workflow are unchanged.

The ramp does not change reference energy, normalize particle momenta to a new
reference, or introduce an additional clock. A thick magnet advances ``t0`` by
its existing flight time :math:`L/(\beta_0c)` after tracking. Continuous particle
``z`` follows the existing magnetic transport map. Induced electric fields from
rapidly varying magnets are not included.

Evaluation and diagnostics
--------------------------

Programs are read once into owned snapshots. A component is constant only when
every stored value is exactly equal; a one-row column is also constant. Small
nonzero changes are preserved. Constant components bypass interval searches.
Other components are interpolated at the bunch entrance, with the endpoint
holding described above. Components sharing a time grid share its interval
lookup; legacy files with different grids retain independent interpolation.
A single query uses binary search without copying a complete long time table.

CPU and GPU tracking use the current nominal strengths and the existing
magnetic maps. The program utility binds and samples the prescribed data,
updates nominal and derived strengths, and coordinates coefficient-cache
invalidation; it does not implement separate particle maps. Field errors remain
independent and are applied once. GPU coefficient caches are derived from the
current strengths and refreshed when those strengths change.

Internal space-charge nodes retain their original order and integration
weights. Their callbacks observe the same entrance-frozen nominal strengths
throughout a bunch's passage. After tracking, the runtime element retains those
entrance values until an explicit update or the next nonempty bunch's entry.

Element logging reports whether the program is constant or dynamic, its supplied
components, file time range, endpoint behavior, entrance sampling and effective
body-slice count. The slice count describes spatial transport only. The time
range describes the table knots, not a tracking window: values outside that
range are still held at their endpoints.

Creating files
--------------

Use **Tools → Data conversion → Magnet ramping** (``工具 → 数据转换 → 磁铁 ramping…``)
to import CSV/TXT/TFS data or generate a table from time breakpoints. Select the
time column and units, map source columns to magnetic strength names, inspect
the preview, and export the TFS file. In the element editor, enable ramping and
select that file. See :doc:`../gui_tools` and :doc:`../input_generation` for the
GUI workflow and Python file-generation API.
