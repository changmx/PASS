Graphical interface
===================

Install the interface with ``python -m pip install --editable ".[gui]"`` and
launch it with ``python -m PASS.gui`` or ``pass-gui``.

1. Open a PASS input JSON or a ``.passproj`` project, or configure a new input.
2. Edit the global parameters, injection, and execution sequence. Select **Apply**
   to commit each property form; unapplied forms and JSON text are drafts.
3. Select **Validate** in the sequence toolbar and resolve errors. See
   :doc:`input_validation` for validation scope and reports.
4. Save the input or project, select inputs on **Run**, and start the simulation.
5. Open the result directory or load outputs on **Plot**.

The configuration page edits the same input used by the tracking engine.
For project packaging, run snapshots, stopping, and rerunning, see :doc:`project_files`.
Standalone calculations are available under **Tools**; see :doc:`gui_tools`.

Startup and file drops
----------------------

Tools are prepared automatically after the main window opens. The status bar
shows progress; selecting an unfinished tool prioritizes it. Tool values persist
when switching pages in the same window.

Drop one local ``.passproj`` or PASS input JSON onto the window, including child
editors. File contents are checked before replacing the current document. A
replacement requires confirmation; unsaved changes offer Save/Discard/Cancel.
Invalid files or cancellation keep the current document. With a project open,
an input JSON can replace the active input, be added as another configuration,
or open as a standalone document. The configuration selector displays the input
count, and **导入配置…** (Import configuration) adds inputs to the same project.
These configurations are independent alternatives; **Beam 1** on the run page
specifically means the optional second beam in a two-beam simulation.

OMC3 SDDS/HDF5 drops open the conversion tool (see :ref:`gui-data-conversion-en`) without replacing beam parameters.
CSV/TFS drops offer conversion or plotting; dropping them directly into the
converter selects conversion. Only one file is accepted per drop. For another
project window, launch another ``pass-gui`` process from a terminal.

Property ordering and sections
------------------------------

Property forms and the JSON source have separate drafts. Use **Apply source** or
**Discard source changes** for JSON edits. If both contain pending changes,
discard one draft before applying the other. Saving requires resolving drafts.
Injection, Space charge, and WakeField forms apply their changes together as one
undo step. **Cancel** restores applied parameters. Drafts are not automatically
backed up; see :doc:`project_files` for manual saving and recovery.

The read-only ``Command`` type appears beside the property heading. Editable
parameters start with name, position ``s``, and length, followed by the element's
field strengths and geometry, then ``Model``, ``Integrator``, and slice count.
Presentation order is independent of schema inheritance and JSON key order.

For example, ``SBend`` displays name, ``S (m)``, ``Length (m)``, ``K0L``,
``E1 (rad)``, ``E2 (rad)``, ``Hgap (m)``, ``Fint``, ``Fintx``, ``Model``,
``Integrator``, and ``Num slices``, followed by **Field errors**, **Alignment errors**, **Aperture**,
**Ramping**, and **Internal space charge** sections. Each section keeps its
switch and related parameters together inside a subtle border. All sections
remain expanded, with no collapse controls. Disabled internal space-charge
parameters stay visible.

Alignment groups its switch with ``DX``, ``DY`` and ``DPSI``. MAD-X element
import also offers an alignment checkbox using the same error TFS as field
errors. It moves only the magnetic field; apertures and SC boundaries remain
fixed. Twiss transfer import does not offer alignment. See :ref:`en-error`.

Other magnets use the same ordering for their supported parameters. RF data,
Exciter frequency and amplitude modulation, Slicer ranges, and monitor reference
optics and analysis ranges are grouped by purpose. Field, potential, density,
and turn-selection outputs share a **Diagnostic output** section. Twiss retains
its start/end optics and transfer settings sections; Injection retains its
bunch editor.

Structured array parameters
---------------------------

Array and nested parameters have dedicated controls in the property pane.
Use these controls to edit rows and dimensions directly.
The JSON source tab remains available for complete input editing.

* **Save turns** uses rows with start, inclusive end, and positive integer step.
  **Add single turn** creates a row whose start and end match; **Add interval**
  creates a range. **All turns** fills the current simulation range, while an
  empty table disables saving. Turn numbering starts at zero. Single rows are
  written as ``[turn]`` and intervals as ``[start, end, step]``.
* **Turn ranges** for PhaseAdvanceMonitor uses start and **exclusive** end.
  A range must contain at least two turns; an empty table disables analysis.
* **Insert Particle Coordinate** uses six numeric columns: x, px, y, py,
  bunch-relative z, and dp/p. Each row is one manually specified particle.
  The row count is displayed, and cannot exceed the first injection's macro
  particle count. These coordinates replace particles inside that block;
  they do not add particles beyond the configured total.
* **Aperture value** changes with the selected shape. Circle, rectangle,
  ellipse, rectangle-circle, rectangle-ellipse, racetrack, and octagon have
  labeled dimensions in meters. Polygon uses an ordered x/y vertex table.
  Radius and half-width/half-axis meanings are explicit. Shape switching keeps
  each shape's current dimensions while the form remains open. Invalid sizes,
  degenerate polygons, and invalid octagon corner cuts are rejected on apply.
* **Dp aperture** has a custom-range switch and lower/upper dp/p controls.
  With customization disabled, the engine uses its default -1 to 1 range.
* **KiL**, **KiSL**, **Field error KNL**, and **Field error KSL** use explicit
  multipole-order/value rows. Order zero is dipole, one quadrupole, and two
  sextupole. Missing orders become zero; deleting a row does not shift higher
  orders. Duplicate orders are rejected. Integrated order-n strengths have
  units of inverse meters to power n.
* **Device Id** uses integer GPU-ID rows with duplicate checking. The GPU
  device count is derived from this table, which is disabled for CPU inputs.
* Slicer's **Explicit** range has z-min/z-max controls, enabled by ``explicit``
  range mode. ``auto`` writes no explicit range. Element **Space charge** has
  a switch, named-configuration selector, kick count, aperture controls, output
  switches, and its own saving schedule.

Tables support numeric cell editors, adding/removing rows, and a larger
**Table window**. Right-click offers spreadsheet-row paste, copy, row movement,
and clearing. ``Ctrl+V`` also pastes rows when the table itself has focus.
Tab-separated or comma-separated numeric rows can be pasted without converting
them to JSON; a malformed paste leaves existing rows unchanged. The larger
window adds multiple rows at once and keeps edits private until **Apply table**.
Apply the property form afterwards to update the input. Numeric table editors
accept scientific notation and preserve small floating-point values.
Saving also commits the numeric cell currently being edited. Optional bunch
fields are shown with defaults even when omitted from the opened JSON.

Space charge
------------

Use **SC slicing** to create a ``z_periodic`` Slicer. General-purpose slicing
defaults to ``z_rel``; WakeField slicing accepts ``z_rel`` or ``arrival_phase``.
``Periodic`` is derived from ``Coordinate``; legacy ``Periodic=true`` without a
coordinate retains its arrival-phase meaning. Selecting ``arrival_phase`` sets
equal-length bins and the explicit interval ``[-C, 0]``. Changing circumference
later requires checking that interval again. Slicing never rewrites particle z.
RF does not re-bin saved intervals; regrouping requires another explicit Slicer.

Under **Physics effects**, click the **Space charge** heading to expand or
collapse its submenu, just like the outer module section. Collapsing it does
not change the project or discard the active form; there is no back-menu item.

* **Global configuration** (全局配置) manages the module switch, named
  configurations, slice-set references, PIC grids, and solvers.
* **Insert calculation point** (插入计算点) manually inserts a ``SpaceCharge``
  command at a specified position, referencing a named calculation configuration.
  Point parameters, including the interaction length and output options, retain
  their existing meanings. See :doc:`space_charge`.

Calculation options depend on ``Method`` and ``Solver``:

* ``pic`` lists only ``fft_free_space``, ``fd_dirichlet``, and ``dst_dirichlet``.
  Deposition defaults to **CIC**, with TSC also available.
* ``frozen`` and ``quasi-frozen`` list only the four free-space analytic solvers:
  round Gaussian, elliptic Gaussian, uniform disk, and uniform ellipse.
  Deposition remains visible but disabled.
* ``frozen`` enables the fixed source centroid ``Center X/Y`` and only the
  selected formula's sizes: ``Sigma`` for round Gaussian, ``Sigma X/Y`` for
  elliptic Gaussian, ``Radius`` for a uniform disk, or ``Semi-axis A/B`` for a
  uniform ellipse. Only elliptic formulas enable ``Angle``.
* ``quasi-frozen`` derives the centroid, sizes, and orientation from each
  slice's current particle moments, so fixed profile inputs are disabled.
  Grid settings in analytic modes define diagnostic sampling and the default aperture.

``Center X/Y`` describes the source centroid; ``Angle`` is the counterclockwise
rotation of its local x axis in radians. ``Sigma`` is a Gaussian single-axis
RMS size, ``Radius`` the uniform disk's outer radius, and ``Semi-axis A/B`` the
uniform ellipse's semi-axes. Centroids and sizes are in meters. These describe
the charge distribution, not the wall or particle loss aperture. Inapplicable
fields remain visible, are disabled, and are written as null on saving, avoiding parameters from a
different formula in the saved configuration.

**Grid extent input** (网格范围输入) selects full widths (全宽) or half widths
(半宽); enter both axes in meters. The inactive pair is saved as null.
Node spacings are calculated from extent and node counts.

Each SC point sets ``Aperture
type`` and ``Aperture value``; default resolves to the grid rectangle. The
aperture handles losses and also defines the FD/DST conductor. Different FD
walls use separate cached solvers with one shared grid; FFT shares kernels
and uses apertures for losses only. Initialization checks finite PIC apertures
fit in the grid and requires DST to use the complete grid rectangle.
Wall and outside particles are lost before field calculation.

Analytic solvers cannot save potential. The GUI prevents enabling this option;
an imported enabled option can be cleared. Internal SC inherits its parent
element's aperture; conflicting legacy child apertures are overridden by the
engine. ``Num kicks`` sets the internal SC nodes, independently of the parent's
transport ``Num slices``.

Injection, clocks, RF and pulsed elements
-----------------------------------------

Injection preserves valid harmonic-ID permutations when applying a form.
Adding/copying bunches assigns a new slot; deleting a slot compresses higher IDs.
The batch summary shows the planned total, first/subsequent batch sizes, last
injection turn and macro-particle weight. New bunches retain the existing weight.
Distribution input offers ``sequential`` or ``repeat``. Random seeds and
reference arrival times can be cleared to null. Integer fields accept large
particle counts without a signed-32-bit limit.

Global configuration includes ``Reference clock`` even for older JSON files
that omitted the key. Disable custom input for the default clock derived from
the initial harmonic-ID-zero bunch. Otherwise enter time origin and constant
revolution frequency, or increasing time/frequency tables. The frequency times
circumference must remain below the speed of light. This clock does not follow
the instantaneous energy of a tracked bunch.

RFCavity components support inline/file and harmonic/direct-frequency modes.
Voltage, phase and direct frequency can be scalars or tables sharing ``Time (s)``.
File input requires TIME/VOLTAGE/PHASE, plus FREQUENCY for direct-frequency mode;
a harmonic file must omit FREQUENCY. Inactive alternatives are cleared on apply.
The physical-voltage preview evaluates the tracking waveform's exact frequency
integral, shows each component and their sum, and allows an explicit time window.
It is not a multi-harmonic bucket calculation. Legacy cavity-level parameters
and turn-indexed RF files need a physical-time migration, not a renamed key.

The library includes Bump and the current voltage/geometry ElSeparator interface.
Required quantities remain blank until supplied; incomplete drafts cannot be applied.
Bump previews TIME/HKICK/VKICK and can convert two CISP CSV files into one TFS
using the union of their time nodes with each plane's endpoint values held outside its range. Kicks are integrated delta-P/P0;
the preview includes time offset and endpoint holds outside each plane's supplied
range. The CSV time grids and ranges may differ; conversion retains their union.
**Preview ES** (预览ES) previews
the tilted electrodes, circulating-beam field-free region and field region,
with the independent vacuum aperture drawn as a dashed outline in beam coordinates.
For rectcircle/rectellipse, the outline is clipped to the actual intersection
rather than showing both complete component boundaries.
Electrodes cover all local v; the drawing is cropped for display.
S is its exit and S-Length its entrance. Zero strength retains material losses;
zero length supports one VL kick. V is the interplate voltage difference in V,
and VL its longitudinal integral in V m; applying/running requires exactly one.
Gap and septum position must be filled before previewing; missing or invalid
values identify the offending field. Leaving both V and VL empty permits a
geometry-only preview without computing the field. Distribution File Mode and its popup fit both sequential/repeat
options without truncating their labels.
The ES **Hardware parameters** (硬件参数) group labels V, VL, gap width,
septum position/thickness and tilt in Chinese, retaining
their units and JSON keys. Bump preview validation and file errors are explained
in Chinese, including path, TFS format, numeric data and unit errors.

Exciter exposes tune/frequency selection and FM/AM-dependent fields. Ordinary
magnet ramping remains unavailable in tracking and cannot be newly enabled.

Exciter previews evaluate particle time as ``t = T_start + elapsed - z/(beta*c)``
using continuous bunch-relative z and the local reference arrival time. Nominal
bunch slots do not shift the preview.

WakeField and result files
--------------------------

**Wakefields → Global configuration** manages the global enable switch and named
model/solver configurations. Opening it collects existing inline point definitions
into separate named configurations, without merging them. Applying a configuration
rename updates its point references; configurations still in use cannot be deleted.
**Wake point** selects a configuration, position, slice set and local enable switch.
The global and local switches must both be on. Each point owns independent history
even when it uses the same configuration. Inline ``Groups`` remain supported.
See :doc:`wake_field` for the JSON/API representation.

Groups and components can be added, copied and removed. Solver
selection covers direct, fft, recursive, modal, partitioned_fft and time_fft;
history, grids, memory and periodic-boundary controls follow the selected mode.
The editor exposes all nine model families, spatial powers for custom components,
and fixed/factorized/ideal velocity laws. The shared schema validates combinations.
File models require explicit units, axes, sign conventions, integration convention
and reference beta; unknown external-file conventions are not guessed.

The plot page loads CSV, TFS, and one-dimensional DistMonitor HDF5 columns.
Load multiple files to select beam/bunch/turn snapshots, filter live/lost particles
and injection batches, and inspect reference metadata. ``arrival_time_s`` is
computed only for live particles with matching row or snapshot reference time
and beta. Lost coordinates cannot use a later live reference. Missing values
preserve row alignment. Multidimensional SC files use a separate field view.

Choose automatic, line, or scatter display. Automatic mode uses scatter for
particle tables with ``tag`` or ``particle_id``, and lines for other tables.
Missing samples interrupt lines. Large previews reduce displayed points without
changing the data. Each loaded file retains its axes and filter settings.
The summary reports total, selected, and finite rows. **Close current data**
unloads the current file; file loading can be cancelled.

Phase-space presets select x–px, y–py, z–dp, x–y, or arrival-time–dp when their
columns are present. Axes use supplied units and reference metadata; units are
not inferred for plain CSV. **Density and projections** bins all finite X/Y pairs
into 8–256 bins and displays counts and marginal projections. Counts represent
macro-particle rows, without charge weighting.

Select a baseline to overlay data or compute A−B or (A−B)/B. X values and row
order must be identical and finite, with compatible units, coordinates, and
reference conventions; particle IDs must also match when provided. There is
no implicit sorting, interpolation, or unit conversion. Relative differences
are NaN where the baseline is zero or nonfinite. Switch density plots to line
or scatter mode before comparing data.

**Export data** writes all selected rows and numeric columns, preserving NaN
and numeric types, with a hash-checked ``.csv.metadata.json`` sidecar in the
``pass-table-metadata-v1`` format. Keep both files together; stale or unsupported
metadata is rejected when reloaded. **Export image** generates PNG/SVG/PDF from
the complete selected data and range. Advanced plots have a Matplotlib toolbar.
Exports fix the current selection and report progress; cancellation may leave
already completed files. CSV and metadata must be retained as a pair.

SC field arrays use ``(slice, y, x)`` with matching x/y coordinates, ``slice_id``,
and ``delta_z``. Select a field and slice, and optionally overlay the saved
aperture. Raw density, potential, and integrated Ex/Ey have units C/m², V·m, and V.
**Divide by Δz: slice averages** gives C/m³, V, and V/m, requiring known units
and positive finite widths. The view displays saved solver, potential-reference,
and boundary metadata. Field export flattens the selected slice in y/x order,
retains raw fields, and adds averages when selected; it does not recompute fields.

Twiss and optics
----------------

The library entries appear in this order:

1. **Import Twiss points from MAD-X** (从 MAD-X 文件导入 Twiss 点).
2. **Insert one-turn transfer matrix** (插入单圈传输矩阵).
3. **Generate a smooth-approximation Twiss sequence** (生成平滑近似 Twiss 序列).
4. **Insert a Twiss transport point** (插入 Twiss 传输点).

The MAD-X entry supports the sampling modes described below. The two generators
use **Preview generation**, followed by **Insert into Sequence**. Previewing
does not change the project. The preview shows the number of commands, position
range, derived beta functions, step length, total phase advances, and the first
100 points. All generated points are inserted, including those beyond the table
preview. Names that already exist receive a numeric suffix, and the preview
displays the resolved names. Existing optical transport in the same ring range
is flagged because adding another map can duplicate transport.

Changing generator inputs invalidates the preview and disables insertion until
another preview is generated. Changes to the sequence, global circumference, or
unconfirmed JSON also prevent insertion of an outdated preview. Circumference
defaults to the global value; insertion synchronizes the global circumference
to the displayed generator value. Existing commands are preserved. Leaving an
edited generator asks whether to discard its uninserted draft.

MAD-X sampling
~~~~~~~~~~~~~~

**Sampling mode** defaults to **Keep original positions**. Selecting **Uniform
interpolation** enables **Segments per turn N**, a positive integer. The base
grid has N+1 points from 0 to C, with spacing C/N. **Merge consecutive Drift**
is disabled and ignored in this mode. There is no advanced interpolation menu.

Click **Preview import** to see the actual spacing, base point count, additional
split locations, Twiss command count, and full horizontal/vertical phase spans.
Then use **Import into Sequence**. Only resampled points and necessary split
points are imported; the source Twiss rows are not also imported. Import still
appends to the current Sequence, so remove any earlier transport that this
import is intended to replace. Changes to sampling inputs or the source file
invalidate the cached import result.

The interpolation uses source phase, beta and alpha together: a quintic Hermite
polynomial represents cumulative phase on each source interval. Its first and
second derivatives match those implied by beta and alpha at both endpoints.
Beta and alpha are then derived from that same polynomial. Dispersion uses
cubic Hermite interpolation of DX with DPX as its slope in the supported
uncoupled, on-reference paraxial convention. See :doc:`input_generation` for
the equations and Python API.

The source must cover 0 through LENGTH, have positive beta functions and
unwrapped increasing phases consistent with Q1/Q2. Missing endpoints are not
extrapolated. The entire phase polynomial is checked for positive slope; if
this fails, export denser source data from MAD-X. Merely increasing N cannot
recover information absent from the source. Source values and total phase
span are preserved to floating-point precision, with a tolerance when checking
rounded TFS header tunes.

Thin elements and field errors retain their original positions, splitting the
Twiss transport there. Distinct optical states at repeated S are preserved as
incoming/outgoing states with a zero-length Twiss map for the optical jump.
Consequently the final positions need not all be equally spaced. At a common
position, Twiss maps execute before additional kicks/errors. The source optics
already include its design linear focusing: explicitly inserted kicks are
additional effects, not a subtraction or replacement of that focusing. Added
linear errors can therefore change the tracked tune even though the resampled
base optics preserve the source tune.

One-turn transfer
~~~~~~~~~~~~~~~~~

Enter a name, circumference :math:`C>0`, tunes :math:`Q_x,Q_y`, and one set of
periodic optical parameters in the order :math:`\alpha_x,\alpha_y,\beta_x,\beta_y`.
Beta functions must be positive and are independently specified; they are not
derived from the smooth-approximation formula. The horizontal
dispersion :math:`D_x`, its derivative :math:`D'_x`, and full-turn chromaticities
``DQx`` and ``DQy`` appear directly in the same form, without an advanced
collapsible section. All numeric inputs must be finite.

The generator copies alpha, beta, and dispersion to both endpoints. It sets
``S previous (m) = 0``, ``S (m) = C``, the previous horizontal and vertical
``Mu`` values to zero, and their endpoint values to :math:`Q_x,Q_y`.
Thus the map has length :math:`C` and phase advance :math:`2\pi Q` even though
the periodic optical functions agree at both endpoints.

Longitudinal transfer supports ``off``, ``drift``, and ``matrix``. Only
``matrix`` enables the longitudinal tune :math:`Q_s`; its endpoint ``Mu z``
is :math:`Q_s` and its previous value is zero. Other modes generate zero
longitudinal phase advance. The engine's matrix mode also uses the bunch's
longitudinal size and momentum spread; drift uses the slip factor and path
length. See :doc:`twiss` for the tracking equations.

Smooth approximation
~~~~~~~~~~~~~~~~~~~~

Enter :math:`C>0`, positive full tunes :math:`Q_x,Q_y`, and a positive integer
**number of segments per turn**, :math:`N`. Do not supply only the fractional
tunes when deriving the smooth beta functions (enter 9.47, for example, rather
than 0.47). The form labels identify these as full tunes:

.. math::

   \beta_x = \frac{C}{2\pi Q_x},\qquad
   \beta_y = \frac{C}{2\pi Q_y},\qquad
   \Delta s = \frac{C}{N}.

The generator calls ``generate_smooth_twiss`` with ``num_points=N+1``. The
result contains an initial identity point at :math:`s=0` and :math:`N` transport
segments ending at :math:`C`. The positions are :math:`s_i=iC/N`; phase advances
are distributed linearly, and each segment receives ``DQx/N`` and ``DQy/N``.
The initial point has zero chromaticity. Names use the chosen prefix and point
index, so short steps cannot produce duplicate names through position rounding.

Alpha defaults to zero. Constant alpha, dispersion, dispersion derivative, and
full-turn chromaticity appear directly in the form, without an advanced section.
These retain the existing generator's behavior.
Standard smooth optics uses zero alpha; nonzero constants define a
custom sequence of transfer maps. Longitudinal modes follow the one-turn form.
The workflow generates only ``Twiss`` commands, not ``SpaceCharge`` points.

Manual transport points and subsequent editing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select **Insert a Twiss transport point** to add a manual map. A command
describes transfer between two endpoints, so the form groups fields into
**Start optics**, **End optics**, and **Transport settings**. ``Mu`` values are
cumulative phases in cycles (:math:`2\pi`), not radians.

Both generators write ordinary ``Twiss`` commands. Projects additionally preserve
their generation settings and MAD-X import sources. After insertion or loading a
JSON, select a generated command to edit its two endpoints individually. Use
**Save JSON** for an independent input or **Save project** for the complete project.
Generation settings can be restored from **Project contents** for another preview
and insertion; this does not automatically replace manually edited commands.

Tools
-----

The **Tools** workspace provides a beam calculator, tune diagram, RF bucket,
phase-space plotting and emittance calculation, magnet conversion, exciter
preview, and data format conversion. Calculations are independent of the active
simulation input. Ions and atoms use kinetic energy per nucleon (AMeV); species
with A=0 use MeV per particle. Particle masses come from the bundled evaluated
mass catalog. See :doc:`gui_tools` for units, formulas, data sources, and workflows.

Appearance and controls
-----------------------

Select a dark, light, or system theme in the top bar. Window and pane sizes,
theme, and sequence-column preferences are remembered locally. Drag column
boundaries to resize them; right-click the header to show additional columns.
Name, Command, and position remain visible. The sequence follows the engine's
position bins and command priorities; hover over a position to inspect its full value.

The left library starts with **Input configuration (required)** expanded. Under
**Physics effects**, Space charge and Wakefields have independent submenus.
Beam-beam effects and Electron cloud are disabled placeholders.

Property fields for the selected mode remain expanded. Long forms scroll, and
**Expand** gives the editor the full workspace; **Restore** returns to the previous
layout while retaining drafts. Dropdowns change selection by clicking an item;
scrolling an open list only browses its options.

Help and local documentation
----------------------------

The **Help** menu next to the theme selector provides **Online documentation**
(Chinese / English), **Source code**, **Read local documentation** (Chinese /
English), **Rebuild local documentation**, and **About PASS**. Web links and local
HTML pages open in the system's default browser.

To enable local builds, install the documentation dependencies in the same Python
environment used to launch the GUI, from a complete PASS source checkout::

   python -m pip install --editable ".[gui,docs]"

Every click on **Rebuild local documentation** starts a full Sphinx HTML build of
both languages, using the GUI's Python interpreter. The build runs in a separate
process; the interface remains usable. The progress window shows live logs and a
stop button. Closing that window leaves the build running; **View build log** in
the Help menu reopens it. Repeated builds are disabled while one is running.
Closing PASS during a build asks whether to stop it before exiting.

Each build writes to a new ``docs/build/gui/<timestamp-id>/`` directory. Only a
successful build with no Sphinx warnings and all language home pages present
updates ``docs/build/gui/latest.json``, which identifies the result used for
reading. Failed or stopped builds leave the previous successful result available.
Build directories are retained; rebuilding does not delete older results.
**Read local documentation** opens the saved result without rebuilding and falls
back to the conventional ``docs/build/html/`` output if no saved GUI result is
available. If no local page exists, it offers to compile or open online documentation.
Missing sources or dependencies and filesystem errors are reported with guidance;
the GUI does not install dependencies automatically. A regular package installation
without the source checkout can use online documentation but cannot build locally.

**About PASS** displays the installed version, project description, authors,
institution, copyright and Apache License 2.0. Documentation, repository and issue
tracker links are clickable. The license opens in a read-only window when available
locally, with an online fallback. **Copy version and environment information**
copies the PASS, Python and PySide6 versions, operating system and architecture.
