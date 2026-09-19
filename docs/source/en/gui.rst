Graphical configuration workflow
================================

Install the optional interface with ``python -m pip install --editable ".[gui]"``
and launch it with ``python -m PASS.gui`` or ``pass-gui``. The configuration page
edits the same input used by the tracking engine; the run and plotting pages
remain separate.

The top bar contains File, Configuration, Run, Plot, Tools, Help, and the dark/light/system
theme selector. The initial window is 1200 by 760 logical pixels; window and pane
sizes, theme, and column preferences are remembered locally. The sequence is the
central overview, ordered by the engine's position bins and command priorities.
Name, Command, and position are mandatory columns. Drag header
dividers to resize columns; right-click to select additional columns. Validation
is in the sequence toolbar and checks the complete active input and dependencies.
See :doc:`input_validation` for the full report, rules and command-line checker.
Applicable property fields remain expanded, with mode-dependent alternatives hidden or disabled and long
forms scrolling within the pane. See :doc:`project_files` for JSON/project saving,
source-file packaging, parameter reuse, and fixed input snapshots for running.

The left library sections use only the height needed by their entries, with
unused space below the entire list. Only **Input configuration (required)** is
expanded at startup; all other main sections and the Space charge submenu start
collapsed. **Physics effects** lists **Space charge**,
**Wakefields**, **Beam-beam effects**, and **Electron cloud**, in that order.
Space charge and **Wakefields** (尾场) expand independently. The latter contains
**Global configuration**, **Insert wake slicer**, and **Wake point**;
Beam-beam effects and Electron cloud remain disabled placeholders.

Startup and file drops
----------------------

The main configuration workspace appears first. Numerical dependencies are then
prepared in a worker thread, and tool pages are constructed one at a time on the
GUI thread. All tools are prepared automatically; clicking an unfinished tool
prioritizes it. The status bar reports progress. Page construction pauses while
a simulation, conversion, or modal dialog is active. Once prepared, pages retain
their values when switching. This reduces initial work; cold disk caches and
individual page construction can still affect responsiveness.

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

Tools
-----

**Tools** is a peer of Configuration, Run and Plot. Its left navigation contains
the beam calculator, tune diagram, RF bucket, phase-space plotting and emittance calculation,
magnet converter, exciter preview, and data format conversion. Calculations are independent of the active simulation
input. Reference masses use fixed AME2020, NIST/CODATA and PDG data. Ek uses MeV/u for every species, normalized by actual rest mass in u.
Formula references and clickable source websites open in separate windows. See :doc:`gui_tools` for units, conventions and workflows.

Appearance and controls
-----------------------

The dark theme uses `One Dark Pro colors <https://github.com/Binaryify/OneDark-Pro>`_,
with charcoal backgrounds, blue accents, and soft gray text. The light theme
shares the same controls and hierarchy. Interface text uses 12 logical pixels,
preferring Segoe UI Variable Text with Microsoft YaHei UI for Chinese fallback.
JSON source and run logs use JetBrains Mono when available, otherwise Consolas
or Cascadia Mono. Source keys, strings, numbers, and booleans have distinct
syntax colors. No additional font installation is required.

**Input configuration (required)** identifies the input configuration entry.
Headings and entries align left, with successive child indentation, thin vertical
guides, and subtle main-heading backgrounds. Hierarchy does not depend on color.
Dropdowns always show a right-hand arrow and divider. Checkboxes show a checkmark
when selected, including when disabled.
Dropdowns throughout all pages and dialogs ignore mouse-wheel changes to the current
selection, even when focused. To select with the mouse, click to open the list and
then click an item. The open list can still be scrolled to browse options.

Validation, table editing, message, and file selection windows follow the current
theme. Windows 11 uses system APIs to color native title-bar backgrounds and text;
Windows 10 uses the supported dark-frame flag. Native moving, resizing, and
snapping remain available. File selection uses Qt dialogs so their content also
follows the theme.

Property ordering and sections
------------------------------

The read-only ``Command`` type appears beside the property heading. Editable
parameters start with name, position ``s``, and length, followed by the element's
field strengths and geometry, then ``Model``, ``Integrator``, and slice count.
Presentation order is independent of schema inheritance and JSON key order.

For example, ``SBend`` displays name, ``S (m)``, ``Length (m)``, ``K0L``,
``E1 (rad)``, ``E2 (rad)``, ``Hgap (m)``, ``Fint``, ``Fintx``, ``Model``,
``Integrator``, and ``Num slices``, followed by **Field errors**, **Aperture**,
**Ramping**, and **Internal space charge** sections. Each section keeps its
switch and related parameters together inside a subtle border. All sections
remain expanded, with no collapse controls. Disabled internal space-charge
parameters stay visible.

Other magnets use the same ordering for their supported parameters. RF data,
Exciter frequency and amplitude modulation, Slicer ranges, and monitor reference
optics and analysis ranges are grouped by purpose. Field, potential, density,
and turn-selection outputs share a **Diagnostic output** section. Twiss retains
its start/end optics and transfer settings sections; Injection retains its
bunch editor.

Module responsibilities
-----------------------

``PASS/validation`` is shared by the GUI, the ``python -m PASS.validation`` CLI,
and simulation preflight, and has no Qt dependency. ``PASS/gui/validation.py``
only provides the worker thread and report window. ``app.py`` assembles the UI;
``appearance.py`` owns themes, fonts, icons, and window appearance;
``structured.py`` provides typed parameter controls; ``project.py`` handles
project packaging and resources; ``workspace.py`` handles document workflows;
``optics.py`` supplies optics configuration helpers; and ``runner.py`` provides
the independent execution entry point. These responsibilities stay separate
instead of placing file operations, validation, and widget implementations in
the main window module.

Structured array parameters
---------------------------

Array and nested parameters have dedicated controls in the property pane.
They no longer require JSON brackets, commas, or knowledge of array shapes.
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

Configurations no longer contain ``Chamber``. Each SC point sets ``Aperture
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
------------------------------------------

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
using their common physical-time interval. Kicks are integrated delta-P/P0;
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

The plot page loads CSV, TFS and one-dimensional DistMonitor HDF5 columns.
Load multiple files to select beam/bunch/turn snapshots; filter live/lost particles
and injection batches, and inspect reference and pending-particle metadata.
``arrival_time_s`` is derived only for live particles with matching row or snapshot
reference time and beta. Lost coordinates cannot use a later live reference.
Missing numeric cells remain aligned across columns. Multidimensional SC field
files require their dedicated analysis and are not treated as particle tables.

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

The former **Twiss** entry is now **Insert a Twiss transport point**. A command
describes transfer between two endpoints, so the form groups fields into
**Start optics**, **End optics**, and **Transport settings**. ``Mu`` values are
cumulative phases in cycles (:math:`2\pi`), not radians.

Both generators write ordinary ``Twiss`` commands. Projects additionally preserve
their generation settings and MAD-X import sources. After insertion or loading a
JSON, select a generated command to edit its two endpoints individually. Use
**Save JSON** for an independent input or **Save project** for the complete project.
Generation settings can be restored from **Project contents** for another preview
and insertion; this does not automatically replace manually edited commands.

The Exciter preview uses the actual local reference arrival time and continuous z: ``t = T_start + elapsed - z/(beta*c)``. Nominal bunch slots do not shift this preview. RF command properties now contain a Components list of prescribed physical-time waveforms.
