Automatic plotting
==================

Set the top-level ``"Is plot figure": true`` (Python ``is_plot=True``) to
generate monitor figures after a successful tracking run. PASS finishes writing
the monitor files before reading them for plotting. The default is ``false``.
Stopped or failed runs do not start automatic plotting.

The plots use saved data, so the same plotting entry point can process an earlier
run without repeating tracking. No GUI, Qt or GPU is required for this step.

Plot an existing run
--------------------

Pass the individual result directory containing the statistic files and the
``distribution/`` and ``particle/`` subdirectories, rather than the output root:

.. code-block:: console

   python -m PASS.plot output/2026_1029/1036_59
   python -m PASS.plot output/2026_1029/1036_59 --format png pdf --dpi 300 --bins 80
   python -m PASS.plot output/2026_1029/1036_59 --output figures/review

Relative paths resolve against the terminal's working directory. Quote paths
containing spaces. The default destination is ``plot/`` within the selected run;
``--output`` changes the figure destination only.

.. list-table:: Command-line options
   :header-rows: 1
   :widths: 25 20 55

   * - Option
     - Default
     - Meaning
   * - ``--format``
     - ``png``
     - One or more of ``png`` and ``pdf``.
   * - ``--dpi``
     - ``300``
     - PNG resolution and resolution of dense rasterized layers in PDF.
   * - ``--bins``
     - ``80``
     - Number of bins per axis in distribution projections.
   * - ``--output``
     - ``<run>/plot``
     - Destination directory for figures and the plotting manifest.

The equivalent Python call returns the plotting manifest:

.. code-block:: python

   from PASS.plot.plot_main import plot_run

   manifest = plot_run(
       "output/2026_1029/1036_59",
       save_dir=None,
       formats=("png", "pdf"),
       dpi=300,
       bins=80,
   )

Figure layouts
--------------

Each :doc:`statmonitor` history produces one 3-by-3 figure for its beam,
bunch and monitor. This includes every saved BPM history; BPMs are not merged
into a single crowded curve plot. All nine panels have their own ``Turn``
axis label and horizontal tick labels.

.. list-table:: StatMonitor panels, in row order
   :header-rows: 1
   :widths: 34 33 33

   * - Left
     - Centre
     - Right
   * - :math:`\langle x\rangle,\langle y\rangle`
     - :math:`\sigma_x,\sigma_y`
     - :math:`\varepsilon_x,\varepsilon_y`
   * - :math:`\langle p_x\rangle,\langle p_y\rangle`
     - :math:`\langle z_{\mathrm{fold}}\rangle`
     - :math:`\sigma_{z,\mathrm{fold}}`
   * - :math:`\langle\delta\rangle`
     - :math:`\sigma_\delta`
     - Loss percentage

The figure uses the saved moments and loss percentage rather than recomputing
them from a particle subset. Vertical ranges adapt to each monitor's data.
A zero loss history is still plotted.

Each :doc:`distmonitor` snapshot produces one 2-by-3 figure per beam, bunch,
monitor and saved turn. Saved injection distributions are also supported.

.. list-table:: Distribution panels, in row order
   :header-rows: 1
   :widths: 34 33 33

   * - Left
     - Centre
     - Right
   * - :math:`x-p_x`
     - :math:`y-p_y`
     - :math:`z-\delta`
   * - :math:`x-y`
     - :math:`z-x`
     - :math:`z-y`

The six projections are two-dimensional histograms of live particles
(``tag > 0``). Their shared colour bar gives the number of macro-particles per
bin, not a physical charge density. Within each figure, all six panels share
one count scale, whose upper limit is the largest bin count in those panels.
Each saved turn has its own colour scale. Snapshots belonging to the same beam,
bunch and monitor still share coordinate ranges and bin edges across saved
turns. Injection distributions form separate comparison groups.
Empty snapshots and snapshots without live particles are recorded as skipped.

Each :doc:`particlemonitor` file produces one 4-by-3 figure:

.. list-table:: ParticleMonitor panels, in row order
   :header-rows: 1
   :widths: 34 33 33

   * - Left
     - Centre
     - Right
   * - :math:`x` versus turn
     - :math:`y` versus turn
     - :math:`z` versus turn
   * - :math:`p_x` versus turn
     - :math:`p_y` versus turn
     - :math:`\delta` versus turn
   * - :math:`x-p_x`
     - :math:`y-p_y`
     - :math:`z-\delta`
   * - :math:`x-y`
     - :math:`z-x`
     - :math:`z-y`

All recorded particle IDs and all valid trajectory samples participate; plotting
does not impose a particle cap or downsample the history. The first two rows
use thin lines and the projection rows use small points. Pending and lost
records are excluded using ``tag > 0``; invalid samples break the time traces.
There is no long per-particle legend. These projections accumulate trajectories
over saved turns, whereas a DistMonitor projection is an instantaneous snapshot.

Coordinates and appearance
--------------------------

Figures have a white background, inward ticks, axis labels and units, short
legends where needed, and a colour bar for distribution counts. There are no
titles, parameter text boxes or automatic Twiss-ellipse/RF-bucket overlays.
Paired horizontal/vertical statistic curves use blue/orange; distribution
counts use the ``jet`` colour map with empty bins left white.

Transverse positions are shown in mm, longitudinal positions in m and RMS
emittances in mm mrad. The saved transverse momenta are
:math:`p_x=P_x/P_0` and :math:`p_y=P_y/P_0`, and ``dp`` is
:math:`\delta=(P-P_0)/P_0`. Their displayed scale is :math:`10^{-3}`;
they are not labelled as slopes or physical longitudinal momentum.

StatMonitor's saved ``zAverage`` and ``sigmaZ`` use the ring-folded coordinate.
When the source metadata identifies this convention, the axis labels include
``fold``; a standalone CSV without that metadata uses a generic ``z`` label.
Distribution and particle plots
retain the continuous saved bunch-relative :math:`z`; plotting does not add
``zCenter`` or fold these coordinates. See :ref:`en-longitudinal-reference`
for the coordinate convention.

The full valid coordinate range is retained without smoothing or percentile
clipping. Dense artists are rasterized in PDF while axis labels and ticks
remain vector elements. Rasterization controls file size without reducing the
data included in the figure.

Files and diagnostics
---------------------

The default output layout is:

.. code-block:: text

   <run>/plot/
       statistics/
       distribution/
       particle/
       manifest.json

Figure filenames preserve the source identity, including beam, bunch,
monitor and turn information where present. Existing figures are preserved;
repeated plotting chooses unused filename suffixes, including for the manifest
(``manifest_2.json``, etc.). The plotting manifest
records the sources, generated files, settings, skipped inputs and failures.

Saved HDF5 and TFS tables are supported, with CSV also available for statistics.
Matching file stems are processed once, preferring HDF5 over text copies.
Plotting reads the original results without changing them or rerunning the
simulation. A failure for one input is reported and does not prevent processing
other inputs. Automatic plotting failures are reported separately from tracking
success; inspect the plotting manifest and log for incomplete figure output.
