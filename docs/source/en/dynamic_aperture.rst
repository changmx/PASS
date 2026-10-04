Dynamic-aperture scans
======================

A complete input-generation, tracking and plotting workflow is available in
``example/07_dynamic_aperture/README.md`` in the source checkout.

Dynamic aperture (DA) is analyzed from tracked particles. It does not replace
the user-defined sequence or insert, remove, or reorder physical effects.
The same scan may therefore include lattice elements, RF, collective effects,
and any explicitly configured apertures. A surviving particle is one observed
alive at the requested monitor event; finite tracking does not establish
indefinite stability.

Initial coordinates
-------------------

The first scan generator uses a Cartesian grid of physical coordinates:

.. math::

   (x_i,\;0,\;y_j,\;0,\;z_0,\;\delta_k),\qquad
   N=N_xN_yN_{\delta}.

Every initial ``dp`` receives the full x-y grid in the same simulation.
``px`` and ``py`` default to zero, and ``z`` defaults to zero. The Python
generator and schema also accept explicit fixed values. Positions use metres;
``px`` and ``py`` are mechanical momenta divided by reference momentum, and
``dp=(P-P0)/P0``. This is a two-dimensional transverse phase-space slice,
not a scan over all transverse phases. No quadrant symmetry is assumed.

``PASS.utils.scan_grid.generate_scan_grid(x_values, y_values, dp_values,
px=0, py=0, z=0)`` returns ``coordinates`` with columns x, px, y, py, z, dp,
and ``indices`` containing dp, y, x indices. Ordering is dp-major, then y,
with x changing fastest. Axis values must be finite and unique, and the
longitudinal mechanical momentum must be real and positive.

An Injection bunch can use the compact ``Scan Grid`` configuration:

.. code-block:: json

   {
     "X range (m)": [-0.01, 0.01],
     "Y range (m)": [-0.01, 0.01],
     "Number of x points": 41,
     "Number of y points": 41,
     "dp values": [-0.01, 0.0, 0.01],
     "px": 0.0,
     "py": 0.0,
     "z (m)": 0.0
   }

This belongs inside the bunch's ``Scan Grid`` field, not in ParticleMonitor.
It is mutually exclusive with ``Insert Particle Coordinate`` and
``Insert Particle File``. Explicit coordinates occupy configured macroparticle
slots in the first injection batch; they do not increase the beam population.
The batch must have at least the number of scan points. For a pure scan, set
its population to exactly that number.

Explicit coordinates are written after ordinary distribution offsets and
dispersion. They are defined in the incoming reference; the subsequent
reference transformation preserves physical momenta and arrival times.
Consequently a different target reference can change the stored coordinate
numbers. ParticleMonitor records the actual coordinates captured
at injection, with their injection reference and turn. DA grouping uses this
saved initial ``dp``, never the final ``dp`` after RF or other effects.
Use a consistent injection reference for a directly comparable grid.

ParticleMonitor associates each captured coordinate row with its particle ID.
HDF5 stores these rows in ``initial``, including ``initial/dp``; TFS uses its
initial records. DA groups the saved initial values exactly, without inferring
groups from particle order or rounding values into bins. Particle reordering,
negative loss tags and later momentum changes retain the same initial group.
A continuous random momentum distribution therefore does not automatically
become a few discrete dp scans.

All macroparticles participate in the configured effects. With collective
effects, the combined dp groups form one source distribution. Their survival
regions describe that distribution; they are not equivalent to independently
simulated monoenergetic beams. Changing grid density or range may also change
the source distribution. ``z=0`` places the particles at one longitudinal
coordinate; configure a different distribution when that model is unsuitable.

Recording and reading
---------------------

Use :doc:`monitor/particlemonitor` to retain turn-by-turn trajectories.
It stores one HDF5 or TFS file per beam and monitor with bounded write buffers.
HDF5 is recommended for large scans: it preserves typed arrays and permits
selected-row reads without parsing an entire text file. Initial coordinates
are separate from the first trajectory
sample: the monitor may be downstream of physical elements.
For HDF5, the DA reader reads only the required trajectory row and initial
arrays, not every recorded coordinate on every turn.
For streamed HDF5, ``ValidSamples`` identifies the committed prefix. Trailing
uncommitted rows are ignored; any required dataset shorter than that prefix
is rejected as an inconsistent file.

.. code-block:: python

   from PASS.analysis import read_dynamic_aperture, export_dynamic_aperture
   from PASS.plot.plot_dynamic_aperture import plot_dynamic_aperture

   result = read_dynamic_aperture("output/particle/run_particles.h5")
   ax = plot_dynamic_aperture(result, dp=0.0, mode="status", boundary=True)
   ax.figure.savefig("dynamic_aperture.png", dpi=180, bbox_inches="tight")
   export_dynamic_aperture(result, "dynamic_aperture.npz")

The filename above is a placeholder for the actual monitor file.
``read_dynamic_aperture(path, requested_turn=None, cancel=None, clamp_to_available=False)`` accepts
one current ParticleMonitor HDF5 or finalized TFS file, identified by
``Layout="single_file"`` and ``FormatVersion=2``. The same file contains
particle IDs, captured injection coordinates and turns, and trajectory samples.
Select that file explicitly in scripts or in the GUI's **ParticleMonitor file** field.

The reader never treats the first trajectory row as injection data. IDs are
matched by absolute signed tag within the beam; missing initial records or
results remain unavailable. Recorded states and losses cannot precede a known
injection turn. These checks reject contradictory events without inferring
unknown injection ages or same-turn event ordering.

Injection's separate saved distribution is a current bunch snapshot at the
last injection event. During multi-turn injection, earlier particles may
already have moved. DA therefore uses the birth coordinates captured inside
the PM file, without a separate initial snapshot or DistMonitor input.

``requested_turn`` is an inclusive simulation turn index at the selected
monitor. Its default is ``RequestedEndTurn-1``, otherwise ``EndTurn-1``,
otherwise the last observed turn. This flags unfinished runs without silently
reducing the target horizon. The location and order of the monitor still
determine which physical actions have occurred. For example, a monitor at the
beginning of the last turn has not observed the remaining actions on that turn.
There is no artificial sample on finalization and no implicit claim that
turn index N means N completed revolutions.

Analysis, plots and export
--------------------------

The array function ``compute_dynamic_aperture`` accepts initial coordinates
``(particles, 6)``, aligned signed ``tag`` values ``(samples, particles)``,
and integer ``sample_turn`` labels. Optional sampled coordinates have shape
``(samples, particles, 6)``. Additional arguments include ``particle_id``,
``lost_turn``, ``lost_position``, ``injection_turn``, ``initial_valid``,
``requested_turn`` and ``metadata``. It classifies the last supplied sample
at or before the requested turn:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Status
     - Meaning
   * - ``survived``
     - Alive in a sample on the requested turn.
   * - ``lost``
     - A negative tag records loss at or before the sampled event.
   * - ``invalid``
     - Supplied sampled coordinates are nonfinite or have nonpositive/nonreal longitudinal momentum.
   * - ``incomplete``
     - A live sample exists, but does not reach the requested turn.
   * - ``unavailable``
     - No usable initial coordinates or observed particle state is available, or injection lies after the target.

Numerical validity refers only to the selected sample, and is not a scan of
the full trajectory history. Without sampled coordinates the array function
cannot check numerical validity. It does not modify tags or add a loss model.
Loss turns remain absolute simulation indices, so particles injected on
different turns do not automatically have the same tracking age.

Results retain all initial points, IDs, status, observed turn, loss turn and
position, selected final coordinates, distinct initial dp groups, and metadata.
``plot_dynamic_aperture(result, dp=None, mode="status", ax=None,
boundary=True)`` selects one exact initial dp value and displays initial x-y
in mm. ``loss_turn`` colors known losses by simulation turn; other states
remain explicitly distinguished. The default dp is the first available group.

Dashed aperture contours are enabled by default; ``boundary=False`` retains
only the sampled points. A contour requires a complete, unique Cartesian grid
with fixed initial ``px``, ``py`` and ``z`` within the selected dp group.
Contours are interpolated halfway between adjacent survived/lost values.
The scatter points remain visible, including stable islands and losses inside
the surviving region. Unknown states are masked, and curves reaching the scan
edge remain open. The legend identifies the sampled aperture boundary;
no smoothing or convex hull is applied. This is an estimate at the chosen grid
resolution and tracking horizon, not a separately fitted physical acceptance
boundary. If all points survive, all are lost, or no valid survived/lost
transition can be drawn, the plot explains why it has no contour.
The outer scan rectangle is never substituted for the DA boundary.

A complete Cartesian grid can cover only a half-plane, such as ``y >= 0``,
or a quadrant. Completeness refers to the selected x/y values, not to the
whole plane. At least two distinct values on each axis are required to draw
a two-dimensional contour. A contour reaching ``y=0`` stays open: no segment
is added along that axis and no untracked negative-y points are mirrored.
The scan-edge message describes unclassified space outside the scan; survival
on the intentionally selected ``y=0`` edge alone does not require extending
the scan. Interpreting the untracked half-plane would require a separately
justified symmetry assumption.

``plot_dynamic_aperture_boundaries(result, dp_values=None, ax=None)`` overlays
the boundaries of all initial dp groups, or a selected subset. Each group has
its own colour, line style and legend entry; groups with no drawable boundary
are identified explicitly. This view shows curves without superposing all
groups' scatter points. Switch to a single-dp plot to inspect individual losses.
Outer contours, internal loss holes and disconnected stable islands are all
retained. An isolated loss inside a surviving region remains a hole; it is not
filled in or treated as noise. All boundaries use the same target monitor event.

.. code-block:: python

   from PASS.plot.plot_dynamic_aperture import plot_dynamic_aperture_boundaries

   ax = plot_dynamic_aperture_boundaries(result)
   ax.figure.savefig("dynamic_aperture_dp_overlay.png", dpi=180, bbox_inches="tight")
   # A subset must use exact values from result["dp_values"].
   ax = plot_dynamic_aperture_boundaries(result, dp_values=result["dp_values"][:2])

``export_dynamic_aperture`` exports the analysis for all particles and initial
dp groups, independently of the groups currently displayed in the GUI:

* CSV is a per-particle table containing ID, six initial coordinates, initial
  validity, injection turn, classification, observed turn, loss turn and loss
  position. A same-stem JSON file stores the analysis metadata.
* NPZ stores every result array, including the selected final coordinates and
  dp values, plus ``metadata_json``. Load it with ``numpy.load`` using
  ``allow_pickle=False``. It is an uncompressed NumPy archive for further
  Python analysis.

These are classified point results, not a second copy of the full turn history
or exported contour vertices. Keep the ParticleMonitor file to analyze another
turn. Figure export saves the current Matplotlib figure, including the chosen
dp groups and display settings, as PNG, PDF or SVG; it does not export Python
source code or a GUI screenshot. Large scatter layers may be rasterized inside
PDF/SVG. Preserve the sequence, reference, apertures, injection conditions and
requested horizon alongside comparisons.

Portable Python file
--------------------

Copy ``example/07_dynamic_aperture/standalone/dynamic_aperture.py`` to another
directory or computer. The copied file requires Python 3.11 or newer, NumPy,
h5py and Matplotlib; it does not require PASS, Qt or a simulation backend.
It supports the same current ParticleMonitor HDF5 and finalized TFS files,
classification, holes, islands, half-plane grids and multi-dp boundaries.

.. code-block:: python

   from dynamic_aperture import read_dynamic_aperture, plot_dynamic_aperture_boundaries

   result = read_dynamic_aperture("run_particles.h5")
   ax = plot_dynamic_aperture_boundaries(result)
   ax.figure.savefig("da_overlay.png", dpi=180, bbox_inches="tight")

The module also exports ``compute_dynamic_aperture``, ``plot_dynamic_aperture``
and ``export_dynamic_aperture`` with their normal signatures. Importing it does
not select a Matplotlib backend. For a single initial dp, call
``plot_dynamic_aperture(result, dp=0.0)``. It can also run directly:

.. code-block:: console

   python dynamic_aperture.py run_particles.h5 --overlay --output da_overlay.png

The distributed file is generated from the same reader, analysis and plotting
sources used by the GUI. After changing these sources, maintainers regenerate
it with ``python tools/export_dynamic_aperture.py`` and verify it with
``python tools/export_dynamic_aperture.py --check``. Copies already taken to
another location remain snapshots until replaced.

Numerical convergence
---------------------

Use ``float64`` for long-term DA comparisons and record the backend and particle
precision. Match coordinate normalization, element order and strengths, aperture
boundary conventions, and the monitor event before comparing programs.

Chaotic trajectories can amplify floating-point rounding enough to change a
loss turn or even finite-turn survival, including between CPU and GPU tracking.
Check short trajectories and individual maps first, then test sensitive points
with small initial-coordinate perturbations or an independent higher-precision
reference. Also assess the chosen tracking horizon and grid resolution.
``survived`` and ``lost`` describe the observed run; they do not estimate
numerical uncertainty. A smooth plotted contour does not remove this sensitivity.

GUI workflow
------------

Configure x-y ranges, point counts, initial dp values and fixed coordinates in
the selected **Injection** bunch. ParticleMonitor only records the configured
particles. Configure it and the sequence through the existing editors.
Preparing a grid does not launch tracking.

For results, open **Analysis → Dynamic aperture** and select one current
ParticleMonitor file. **End turn (-1 automatic)** chooses the
zero-based simulation turn at this monitor, not a new tracking length. For a
run recording turns 0 through 999, use 999 to analyze its planned end or 499
to analyze the earlier monitor event. Automatic mode uses the planned horizon
described above; a live record short of that event remains incomplete.
If a manually entered turn exceeds the last committed sample, the GUI analyzes
that last sample, updates the input and displays a notice: entering 1100 for a
file ending at 999 uses 999. The figure and exported ``requested_turn`` use the
effective turn; ``requested_turn_input`` retains the original entry and
``last_sample_turn`` records the file's last committed sample (or ``None`` for
an empty history). No second read of the particle history is needed.
Automatic mode is not clamped, so an interrupted run cannot silently appear
complete. A missing sample within the recorded range also remains incomplete;
an empty history supplies no endpoint to clamp to.
Python and the portable module preserve strict requested-turn behavior by
default; pass ``clamp_to_available=True`` to opt into the GUI's manual upper
limit behavior and inspect the returned metadata for the adjustment.
Choose the initial dp and status/loss
coloring. **Show aperture boundary** is on by default and can be disabled.
Choose **Multiple dp: boundary overlay** under **Plot mode** to compare groups.
The checkable dp list initially selects all groups and provides Select all/Clear;
single-dp colour and boundary controls apply only to the single-dp view.
Data and figure exports use the
same analysis and plotting functions as Python scripts. Reading runs in the
background and supports cancellation.
