# Dynamic aperture from a Cartesian scan

This example generates every combination of an x-y grid and three initial
momentum deviations in **one simulation**. The initial values are px = py = z = 0.
The scan replaces the first injection batch **after** its dispersion and offset
operations. The usual conversion from the incoming reference to the receiving
bunch reference still preserves physical momentum and arrival time.

The sequence is yours to edit. The supplied example combines a periodic linear
Twiss rotation with an explicit thin sextupole kick (K2L = 10 m^-2) and an 80 mm
half-aperture. The aperture is checked after the rotation and before the kick,
so particles that have escaped are removed before evaluating the nonlinear field.
It demonstrates the workflow; it is not a calibrated machine model or evidence
for any machine's dynamic aperture. Set `--k2l 0` for the linear baseline.
Add your own element lattice and supported effects using the normal sequence.
There is no DA tracking command or passive test-particle model.

Install PASS following the [installation instructions](../../README.md#install).
Run these commands from the repository root, in order:

```powershell
python example/07_dynamic_aperture/generate_input.py
pass-run --beam0 example/07_dynamic_aperture/beam0.json
```

The generator accepts `--backend cpu|gpu`, `--turns`, `--points` (per transverse
axis), `--k2l`, and `--output`. The default scan is 31 x 31 x 3 = 2,883 particles,
x/y from -20 to +20 mm, dp = -0.003, 0, +0.003, and 256 turns. A different output
input path places its simulation output folder next to that input.

`pass-run` saves results beneath that output folder in `YYYY_MMDD/HHMM_SS/`,
with the execution input and dependencies archived in `input/`. Use
`pass-run --help` for CLI options. `run_simulation.py` remains a compatibility
wrapper, accepting `--input` (or `--beam0`), `--output`, and `--stop-file`.

The monitor is ordered after all physical operations at s = 100 m. Its turn 0
sample is after the first complete map; its final turn 255 sample is after 256
maps. The file's `initial` group is captured at Injection, before that first map.
The uncompressed HDF5 file stores every turn with a 32-turn buffer.
It retains integer tags/loss data, actual sample turns and injected coordinates.
It does not allocate the entire planned history in RAM or GPU memory.
The buffer is appended to the same file after every 32 recorded turns; the
remaining turns are written when recording finishes or on normal cleanup.
This is a write interval, not a sampling interval: no intermediate turns are
discarded. With `output_format="tfs"`, tracking uses a private uncompressed
HDF5 file, then finalization exports one standard TFS table with explicit turn
and particle-ID columns. Initial records precede all trajectory rows. The final
table has ordinary headers and numeric rows and can be read with `tfs.read`.
The run's temporary files are removed only after the final TFS is published
successfully; an export failure retains the recoverable HDF5 and partial text.
Uncompressed HDF5 is recommended for large scans to avoid text formatting and
compression costs while permitting selected-row reads during analysis.

Find the `*_da_s100.000_particles.h5` file in the chosen run's `particle` folder,
then analyze **that exact file**:

```powershell
python example/07_dynamic_aperture/analyze_results.py path/to/da_particles.h5 --output path/to/analysis
```

To analyze files on another computer without installing PASS, copy the single
[`standalone/dynamic_aperture.py`](standalone/dynamic_aperture.py) file and install
NumPy, h5py and Matplotlib. It provides the same importable reading/plotting
functions and a command-line interface. See the
[standalone instructions](standalone/README.md) for examples and dependencies.

Outputs include the complete per-particle CSV and NPZ results, plus status and
loss-turn PNG/PDF plots for each **initial** dp. An additional PNG/PDF overlays
all dp boundaries with distinct colours and line styles. Inner loss holes and
disconnected surviving regions remain separate contours. In the GUI, choose
**Plot mode → Multiple dp: boundary overlay** and check the dp groups to compare.
The initial values are associated with particle IDs in the PM file, so changes
in tracked momentum or particle storage order do not move a particle into a
different initial group. Missing observations are not
declared stable; NaN/Inf or invalid longitudinal momentum are distinguished from
ordinary survival. Plots show scatter points and dashed aperture boundaries by
default; use `boundary=False` in Python or clear **Show aperture boundary** in the
GUI to show only the points. Contours preserve stable islands, internal losses
and gaps in known states; curves reaching the scan edge are not closed artificially.
A displayed contour is only a transition on the sampled grid,
not a fitted smooth physical boundary. If all sampled points survive, enlarge
the scan before claiming a boundary. Increase both the grid resolution and turn
count to investigate convergence. Thick elements also need integration-step
convergence checks.

Use `float64` for long-term comparisons and retain the backend and precision in
the run record. Chaotic trajectories can amplify rounding enough to change loss
times or finite-turn survival across programs or CPU/GPU backends. Compare short
trajectories first, then check sensitive points using small initial-coordinate
perturbations or an independent higher-precision reference. A single run's
survival label does not estimate this numerical uncertainty.

When collective effects are enabled, all grid particles participate in the
source calculation. The combined dp distribution is one physical input ensemble,
not several independent monoenergetic experiments. Its particle density and
intensity must be chosen deliberately.

In the GUI, select the bunch in **Injection** and configure its scan grid in
**Insert particles**, alongside the manual-coordinate and particle-file inputs.
The save switch and output format share **Distribution output**.
Every dp value receives the complete x-y grid; fixed px, py and z default to
zero. Review the displayed total particle count and apply the bunch edit.
Configure a ParticleMonitor covering the required IDs and place it after the
desired final physical operation. ParticleMonitor only records particles; it
does not generate the scan. After tracking, open **Analysis → Dynamic aperture**
to select one current PM file and plot the results. DA reads its captured
injection coordinates and trajectory samples together. PM readers require
`Layout="single_file"` and `FormatVersion=2`; both HDF5 and finalized TFS contain
all selected particles and recorded turns in one file per monitor and beam.
