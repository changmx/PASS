# Electron-cloud models and response extraction

These examples call PASS's production electron-cloud solvers directly. They are small, reproducible workflows for learning the new model inputs and checking numerical outputs. Run them from the repository root after installing PASS and its dependencies.

```powershell
python example/09_electron_cloud_models/run_models.py
python example/09_electron_cloud_models/run_response.py
```

Each invocation creates a new timestamped directory under `tests/codex/ecloud_examples/`. Existing results are preserved. To choose a directory, pass `--output path/to/a/new/directory`. The directory must not already exist. Cases run serially.

Both scripts accept `--backend gpu`. This requires an available CUDA device and the GPU dependencies used by PASS's finite-difference field solver. CPU is the default baseline. The GPU keeps material sampling, population control and wall-spectrum reductions on the device; rare conditional-distribution underflow or extreme Dirichlet parameters use a small host fallback. Material draws are reproducible after cloning or restoring state on the same backend and software stack, but individual CPU/GPU samples need not match. GPU wall spectra use double-precision atomic reductions, so variable-weight bins can differ in their last bits even when random draws reproduce; these reductions do not feed back into particle trajectories or RNG. Primary sources gather only sampled beam coordinates instead of copying the full beam to the host.

## Chamber, source and material examples

`run_models.py` exercises circle, ellipse, rectangle, racetrack and polygon chambers with the same geometry for electron wall events and the grounded electrostatic boundary. The initial cloud is a uniform disk with density `2e9 /m^3`, radius `3 mm`, 128 representatives and seed 123. Its initial electron energy is 20 eV. These small sample sizes keep the demonstration fast and are not production convergence settings.

All cases use externally prescribed beam slices and electron self-fields. The beam is an off-axis Gaussian with unequal RMS widths. The rectangle case supplies the same type of distribution as a relative density map, demonstrating its independent axes and `(y, x)` ordering. Two finite beam intervals and their physical empty gaps drive the cloud. No beam backreaction is applied in this `build_up` workflow.

The configurations enable four explicit primary sources: residual-gas ionization, a prescribed wall yield, a prescribed beam-loss rate, and photoelectric production from a given photon flux. Source weights use physical electron number; the photon source also operates during empty gaps. The gas and material parameters are demonstration inputs, not calibrated values for a specific machine.

The optional surface model includes true-secondary, elastic and rediffused channels with a shared incident-energy budget. It is a phenomenological model, not a full Furman-Pivi fit. The five geometry cases and the half-step comparison leave population control disabled, so their time-step comparison does not mix integration error with a changing merge approximation.

Two separate ellipse cases demonstrate population control at the same `25 ps` maximum time step as the unmerged reference. The coarse case uses spatial cells of `2 mm` and normalized-momentum cells of `0.01`; the finer case uses `1 mm` and `0.003`. Both request a soft target of 100 representatives. These are sensitivity examples, not high-precision defaults. Local merging preserves weight, momentum and kinetic energy but changes higher moments and fine spatial structure, which can change later trajectories, wall impacts and spectra. A smaller maximum time step does not remove that approximation.

Outputs include:

- `<case>.h5`: production HDF5 snapshots containing final electron phase space, field maps, interval histories and weighted diagnostic spectra. Open these in the PASS results GUI to inspect or export individual spectra. Field arrays use `(y, x)` ordering and datasets carry units.
- `<case>_configuration.json`: the exact named configuration with public JSON aliases. To use it in normal tracking, put it under `Electron cloud.Configurations`, enable that module, then execute a general-purpose `z_rel` Slicer at the same position before `ElectronCloud`.
- `summary.json` and `summary.csv`: final weighted electron number, effective count, source line density, kinetic energy, core density, spatial/momentum moments, merge counts and whether the requested population target was reached.
- `population_histories.png`: all eight histories, including the five geometries, the half-step case and both population-control cases.
- `ellipse_half_step.h5`: a second unmerged ellipse calculation using half the maximum electron time step. `time_step_comparison` gives both measured values and relative changes; it does not label a single pair of runs as convergence.
- `ellipse_population_control.h5` and `ellipse_fine_population_control.h5`: coarse and finer merging at the half-step setting. `population_control_comparison` compares each with `ellipse_half_step`, including signed changes and the effective representative count.

These direct `BuildUpCloud` examples retain wall-spectrum collection by default. In normal Sequence tracking, `ElectronCloud` collects wall spectra only when `Save fields` is enabled; scalar number and energy accounting stays active regardless. For direct calls, `record_wall_spectra=False` disables collection and discards previously collected spectra when restoring a state, without changing particles or RNG state.

Continuous collection uses `scope=cumulative_since_state_origin`. Restoring a snapshot that contains earlier wall impacts without recorded spectra, then enabling collection, starts a new history marked `scope=cumulative_since_recording_restart`. Later accumulation and save/restore retain that scope. The scalar number and energy ledgers still include the complete physical history, so restarted spectra do not represent the full-history wall energy.

Check mesh spacing, time step, population and sampling scatter independently before interpreting saturation or heat load. Compare merging with an unmerged reference at the same step and report both its accuracy and compression. Reaching the soft target is not evidence of accuracy, and effective count is a weight statistic, not a count of independent samples after splitting.

The earlier coarse-merge example gave an `8.85%` change in vertical RMS size when its time step was halved. Repeating that 8 ns calculation across six seeds and disabling merging reduced the mean paired step difference to `-0.0047% ± 0.0751%`. At the finer step, coarse merging changed mean vertical RMS size by `-5.47% ± 2.95%` relative to the unmerged runs. These are nominal 95% Student-t intervals of the paired differences, expressed relative to the mean reference. This distinction is why the script now reports time-step and merge effects separately. It does not establish a universal error bound for either method.

Longer 40 ns tracking with 2048 initial representatives also showed that source sampling limits the final effective population: eight representatives per source/interval gave about 683 effective representatives, while 32 gave about 1317. Four-seed confidence intervals for the small `1 mm` core density remained about 70–93% of its mean. These studies held the mesh fixed at `33 x 33`, so their statistical intervals do not include mesh bias. Global population or wall-energy precision does not imply comparable precision for a sparsely populated central region.

## Measured response and direct reconstruction

`run_response.py` uses a real coupled PIC cloud with 64 initial representatives, a `33 x 33` mesh and four finite intervals over two labelled turns. It extracts:

- The spatial background kick `G` on a fixed probe grid.
- Every `xx`, `xy`, `yx`, and `yy` source-to-witness component of the causal matrix `K`.
- A same-slice response and a response carried across the turn boundary.

Each derivative uses positive/negative source-only perturbations from cloned cloud and RNG states. Fixed witness probes prevent source derivatives from accidentally including witness motion. The extractor compares `10 micrometre` and `5 micrometre` perturbations and checks the even residual against the unperturbed baseline to detect quadratic response. A separate direct calculation then tests smaller displacements that were not used to form the central difference.

`finite_response.h5` contains the response, slice timeline, charge/reference normalization, geometry/configuration identity and calibration limits. `comparison.json` stores measured and predicted kicks, relative errors, and explicit acceptance limits: 1% for the full kick and 3% for the smaller dynamic source response. The script fails if these limits are exceeded.

The full kick is `G(actual witness position) + K * source centroid displacement`. Charge, interaction length and signed witness reference factors are already included in the file. Do not multiply them in again or add another copy of the background cloud focusing.

For normal Sequence tracking, import `ElectronCloudResponseItem` from `PASS.para.api` and use `ElectronCloudResponseItem(response_file=..., slice_set=..., s=...)` after the matching Slicer. The default `Timeline mode` is `finite`. The actual slice times, charges, reference factors and profile must match the calibrated file. Do not reuse these four intervals indefinitely. `make_periodic_response` requires multiple measured periods and checks the repeated baseline, lag kernels and omitted memory before periodic replay is allowed.

## Growth, mode and threshold analysis

`PASS.commands.ecloud.response` provides `analyze_instability`, `analyze_slice_modes`, `scan_instability` and `bracket_instability_threshold`. Supply a physical-time axis to obtain growth in `1/s` and frequency in Hz, or a turn axis to obtain growth per turn and frequency per turn. Supply an explicit fit window when transients are present. Longitudinal mode projections accept explicit mode shapes and slice weights, including nonuniform physical slices.

The scan runs cases serially and combines seed scatter with fit uncertainty. A single seed cannot establish a confidence interval and is reported as unresolved. A response calibrated at one beam/cloud setting must be recalibrated when that setting changes; multiplying an existing kernel is not a substitute. The examples validate finite response reconstruction and model bookkeeping, not an experimental instability threshold, full 3D electron transport or electromagnetic fringe fields.

The replay profile guard checks centered transverse covariance. It cannot detect every higher-moment change at fixed covariance; such distribution changes need a new calibration. Empty bunches require finite explicit Slicer ranges when capturing response intervals. Wall spectra in the model snapshots accumulate from the component state origin. Their position coordinate is polar angle, and the last incident-energy bin includes all energies at or above 1 MeV despite its finite plotting edge.
