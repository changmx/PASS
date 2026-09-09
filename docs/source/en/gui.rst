Graphical configuration workflow
================================

Install the optional interface with ``python -m pip install --editable ".[gui]"``
and launch it with ``python -m PASS.gui`` or ``pass-gui``. The configuration page
edits the same input used by the tracking engine; the run and plotting pages
remain separate.

In the central configuration overview, the global configuration and Sequence
detail panels stack at the top when both are collapsed. Expanding either panel
lets its contents use the available vertical space.

The left library sections use only the height needed by their entries, with
unused space below the entire list. **Physics modules** lists **Space charge**,
**Wakefields**, **Beam-beam effects**, and **Electron cloud**, in that order.
Each is independently collapsible. The last three currently contain no entries;
expanding an empty section does not reserve blank space.

Space charge
------------

Under **Physics modules**, click the **Space charge** heading to expand or
collapse its submenu, just like the outer module section. Collapsing it does
not change the project or discard the active form; there is no back-menu item.

* **Calculation configuration** (计算配置) manages the module switch, named
  configurations, slice-set references, PIC grids, and solvers. This is the
  former global space-charge editor, with unchanged functionality.
* **Insert calculation point** (插入计算点) manually inserts a ``SpaceCharge``
  command at a specified position, referencing a named calculation configuration.
  Point parameters, including the interaction length and output options, retain
  their existing meanings. See :doc:`space_charge`.

Calculation options depend on ``Method`` and ``Solver``:

* ``pic`` lists only ``fft_free_space``, ``fd_dirichlet``, and ``dst_dirichlet``.
  Deposition defaults to **CIC**, with TSC also available.
* ``frozen`` and ``quasi-frozen`` list only the four free-space analytic solvers:
  round Gaussian, elliptic Gaussian, uniform disk, and uniform ellipse.
  Deposition is hidden.
* ``frozen`` displays the fixed source centroid ``Center X/Y`` and only the
  selected formula's sizes: ``Sigma`` for round Gaussian, ``Sigma X/Y`` for
  elliptic Gaussian, ``Radius`` for a uniform disk, or ``Semi-axis A/B`` for a
  uniform ellipse. Only elliptic formulas display ``Angle``.
* ``quasi-frozen`` derives the centroid, sizes, and orientation from each
  slice's current particle moments, so fixed profile inputs are hidden.
  Grid settings in analytic modes define diagnostic sampling and the default aperture.

``Center X/Y`` describes the source centroid; ``Angle`` is the counterclockwise
rotation of its local x axis in radians. ``Sigma`` is a Gaussian single-axis
RMS size, ``Radius`` the uniform disk's outer radius, and ``Semi-axis A/B`` the
uniform ellipse's semi-axes. Centroids and sizes are in meters. These describe
the charge distribution, not the wall or particle loss aperture. Inapplicable
fields are hidden and written as null on saving, avoiding parameters from a
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
interpolation** displays **Segments per turn N**, a positive integer. The base
grid has N+1 points from 0 to C, with spacing C/N. **Merge consecutive Drift**
is hidden and ignored in this mode. There is no advanced interpolation menu.

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
``matrix`` displays the longitudinal tune :math:`Q_s`; its endpoint ``Mu z``
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

Both generators write ordinary ``Twiss`` commands without storing a separate
generator configuration. After insertion or loading a saved JSON file, select
a generated command in the sequence to edit its two endpoints in this manual
form. Generated points can be edited individually. Save the project using
**Save JSON** or **Save as** after inserting or confirming edits.
