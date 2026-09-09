# Space-charge integration validation

All tests in this directory run automatically, including the full million-macroparticle simulations. There is no default slow-test exclusion. The suite checks analytic fields, convergence, PIC deposition/gathering, command equivalence, and complete PASS input-to-HDF5 workflows.

Run the commands below from the PASS repository root, using the Python environment that has PASS dependencies and `pytest` installed. These cases use the CPU backend; CUDA is not required.

## The two routine commands

Run every integration case:

```powershell
python -m tests.integration.space_charge
```

This is the same selection as `python -m tests.integration.space_charge all`: **25 tests**, consisting of the original 9 numerical checks, 10 complete PIC simulation/analysis workflows, 4 generated-input analytic comparisons, and 2 parameter-evolution checks. The original PIC particle counts, grid sizes and numerical tolerances are unchanged.

Run the related command, configuration, and PIC regression tests, including the restored slice-isolation tests:

```powershell
python -m tests.integration.space_charge regression
```

This group currently contains **89 tests** from `tests/unit/test_space_charge.py` and the local Codex files listed below. It explicitly includes `tests/codex`, which ordinary repository-wide pytest discovery excludes. Those local exploratory files are Git-ignored; the regression command requires them to be present and reports a missing-file error if they are unavailable. It does not silently skip missing coverage.

- `tests/codex/test_space_charge_stage1.py`: field references, slice isolation, separate slice-width and interaction-length scaling, and particle-state preservation. The fixture uses the current named `SpaceChargeConfig` configuration interface.
- `tests/codex/test_space_charge_configuration.py`: named resource configuration and invalid-input handling.
- `tests/codex/test_pic.py`: deposition, gathering, batched solvers, and boundary handling.
- `tests/codex/test_pic_cpu_field_validation.py`: gathered PIC fields against round-uniform theory.
- `tests/codex/test_space_charge_requested_fixes.py`: Gaussian-reference stability and command-level domain-error checks that preserve particle/loss state.
- `tests/codex/test_space_charge_analytic.py`: analytic method/schema validation, moment updates, independent round-field kicks, coordinate transformations, diagnostic-grid independence and invalid-slice handling.

Run both commands when changing shared space-charge behavior. For a focused edit, run its group first, then the complete integration suite. The integration suite typically takes a few minutes on a CPU; plotting and sparse factorization affect the duration. A passing regression group does not replace the full integration run.

## Run one category

```powershell
python -m tests.integration.space_charge fft
python -m tests.integration.space_charge rectangle
python -m tests.integration.space_charge aperture
python -m tests.integration.space_charge checks
python -m tests.integration.space_charge workflows
```

| Group | Automatically selected tests | Count |
| --- | --- | --- |
| `all` (default) | Every integration case below | 25 |
| `analytic` | Four generated-input comparisons and two repeated-kick evolution tests | 6 |
| `fft` | Four Gaussian/KV full workflows, analytic-density FFT convergence, CIC/TSC Gaussian deposition | 6 |
| `rectangle` | Four full FD/DST workflows, sinusoidal convergence, discrete-Poisson equivalence, PIC equivalence, command-kick equivalence | 8 |
| `aperture` | Circular/elliptic KV full workflows, uniform elliptic analytic solution, quartic Shortley–Weller convergence, all supported aperture geometries | 5 |
| `checks` | The nine numerical checks, without the ten full generated-input workflows | 9 |
| `workflows` | Only the ten complete generated-input workflows | 10 |
| `regression` | Related unit and explicitly selected local Codex tests | 89 currently |

Counts refer to pytest test items, not the number of parameter combinations inside a test. For example, the all-apertures test covers two sources, ten aperture types, and three grids.

Rectangle workflows need the corresponding free-space FFT output for their three-way plots. The batch runner prepares that reference automatically, using the same batch output root. Selecting `rectangle` therefore executes four FFT prerequisite workflows in addition to its eight selected pytest items. Each full workflow runs at most once per successful batch, regardless of collection order. A failed prerequisite fails the dependent test. Old FFT results from another batch are never silently substituted.

List targets or verify actual pytest collection without running simulations:

```powershell
python -m tests.integration.space_charge rectangle --list
python -m tests.integration.space_charge all --collect-only
```

## Run one or several full cases

Use `--case` with the filename stem after `test_`, without `.py`:

```powershell
python -m tests.integration.space_charge --case round_gaussian_free_space_fft
python -m tests.integration.space_charge --case round_kv_free_space_fft --case elliptic_kv_free_space_fft
python -m tests.integration.space_charge --case round_gaussian_rectangular_fd_dst
```

`--case` overrides the positional group; repeat it to select multiple workflows. Names are validated, and duplicate selections are executed once. Available full cases are:

| Free space | Grounded rectangle |
| --- | --- |
| `round_gaussian_free_space_fft` | `round_gaussian_rectangular_fd_dst` |
| `round_kv_free_space_fft` | `round_kv_rectangular_fd_dst` |
| `elliptic_gaussian_free_space_fft` | `elliptic_gaussian_rectangular_fd_dst` |
| `elliptic_kv_free_space_fft` | `elliptic_kv_rectangular_fd_dst` |

The two additional cases are `round_kv_circular_boundary_fd` and `elliptic_kv_elliptic_boundary_fd`.

For an individual numerical check, retain the normal pytest file/function interface:

```powershell
python -m pytest tests/integration/space_charge/test_deposition_and_field_solver_crosschecks.py -v
python -m pytest tests/integration/space_charge/test_deposition_and_field_solver_crosschecks.py::test_fd_and_dst_rectangle_solve_the_same_discrete_poisson_problem -v
```

Additional pytest arguments can follow `--`:

```powershell
python -m tests.integration.space_charge all -- -x -v
python -m tests.integration.space_charge checks -- -k convergence
```

The batch command preserves pytest's exit status, including test failure and an empty selection. Tests run serially; do not use `pytest-xdist -n`. The suite rejects parallel execution because generated-input workflows may share process and input state.

## Output locations and preserving old results

Without an output argument, each integration invocation creates a unique directory:

```text
tests/codex/space_charge_runs/<timestamp>_<unique-id>/
    logs/<case>_simana.log
    workflow_results_simana.json
    <full-case>/run_001/
        input/beam0.json
        parameters.json
        simulation_manifest.json
        simulation/.../*.h5
        analysis/summary.json
        analysis/field_scan.csv
        analysis/field_data.npz
        analysis/figures/*.png
    <numerical-case>/analysis/...
```

Exact artifacts vary by case; convergence checks also produce CSV and PNG results. The terminal summary prints the absolute batch directory. Full workflow logs include both PASS output and analysis failures; `workflow_results_simana.json` records executed cases, exit codes, timings, and paths. Prerequisites are included in this manifest, so its case count can differ from the selected pytest item count.

Choose an explicit **new or empty** directory when you want a memorable run name:

```powershell
python -m tests.integration.space_charge all --output-dir tests/codex/space_charge_runs/baseline_001
```

The runner refuses a nonempty directory for a new test or simulation run. It never deletes previous output. Choose another name for a new run. Analysis-only mode deliberately reuses an existing batch and rewrites its analysis products and analysis log; it does not rerun tracking.

The regression group uses the existing tests' `tmp_path` behavior and does not accept `--output-dir`. This output option applies to integration cases.

## Simulation-only and analysis-only batch modes

These modes are for diagnostic work and figure regeneration. Routine validation should use the default `test` mode above.

```powershell
python -m tests.integration.space_charge fft --mode sim --output-dir tests/codex/space_charge_runs/fft_study_001
python -m tests.integration.space_charge fft --mode ana --output-dir tests/codex/space_charge_runs/fft_study_001
python -m tests.integration.space_charge workflows --mode simana --output-dir tests/codex/space_charge_runs/full_study_001
```

| Mode | Behavior |
| --- | --- |
| `test` (default) | Run every selected pytest check, including fresh simulation and analysis for full workflows |
| `sim` | Generate input and run PASS for the selected full workflows and required FFT references; field-analysis checks are not run |
| `ana` | Reuse the selected full workflows' HDF5 snapshots, regenerate analysis, and enforce their analysis tolerances; requires `--output-dir` |
| `simana` | Simulate and analyze the selected full workflows without invoking pytest |

In non-`test` modes, only full workflows in the group run. Numerical-only checks are not selected; `checks` and `regression` reject these modes. For example, `all --mode ana` analyzes ten workflows, rather than running all nineteen integration tests. Missing required snapshots cause failure. Rectangle prerequisites follow the same mode: `sim` creates their snapshots, and `ana` only analyzes existing ones.

## Original single-module and Python interfaces

The original CLI modes and `--run-dir` option remain available:

```powershell
python -m tests.integration.space_charge.test_round_gaussian_free_space_fft simana --run-dir tests/codex/space_charge_runs/single_gaussian_001
python -m tests.integration.space_charge.test_round_gaussian_free_space_fft ana --run-dir tests/codex/space_charge_runs/single_gaussian_001
```

`--run-dir` is the individual case directory, whereas the batch `--output-dir` is its parent batch root. Omitting `--run-dir` retains the original module's default under `tests/integration/space_charge/output/<case>/run_001`; repeated direct calls can overwrite analysis there. Use an explicit new directory when preserving previous results matters.

The original functions also remain callable:

```python
from pathlib import Path
from tests.integration.space_charge.test_round_gaussian_free_space_fft import simulate, analyse

run_dir = Path("tests/codex/space_charge_runs/python_gaussian_001").resolve()
simulate(run_dir)
summary = analyse(run_dir)
```

Legacy direct rectangular modules retain their optional lookup of FFT results in the old default tree. Prefer the batch `--case ..._rectangular_fd_dst` interface when you want automatic creation and reuse of a matching FFT reference. The shared rectangular analysis also accepts an explicit `fft_run_dir` keyword for programmatic use.

## Standard pytest discovery

This command executes all 25 integration tests, with the same prerequisites and isolated outputs:

```powershell
python -m pytest tests/integration/space_charge -v
```

For an explicit batch output path, use pytest's local option:

```powershell
python -m pytest tests/integration/space_charge -v --sc-output-dir tests/codex/space_charge_runs/pytest_001
```

Repository-wide `python -m pytest` discovers `tests/unit` and `tests/integration`. It does not discover exploratory `tests/codex` files by default; use the explicit `regression` command for the restored and related local checks.

## Interpreting the results

- FD/DST equality tests compare the same grounded rectangular discrete problem. They do not independently validate factors shared by both command paths.
- Free-space FFT and grounded FD/DST use different physical boundaries; their difference is not an FD/DST accuracy failure.
- Analytic convergence checks and million-particle field comparisons measure different combinations of discretization and sampling error. This automation preserves their existing tolerances.
- The analytic tests independently verify absolute kicks against Gaussian/uniform field integrals. Prescribed affine transport verifies repeated parameter updates, but does not establish long-term self-consistent ring stability.

## Analytic tracking validation

```powershell
python -m tests.integration.space_charge analytic
```

The four generated-input cases cover round/elliptic Gaussian and uniform spatial
profiles. Uniform profiles use the existing KV phase-space generator. Each case
generates 180,000 particles, slices them into three bins, and executes `pic`,
`frozen`, and `quasi-frozen` at the same position. Since kicks change only px/py,
all methods see identical source coordinates. Analytic kicks are checked against
independent quadrature with relative L2 tolerance `2e-8`; FFT field comparisons
use `0.06` over the resolved central region and include particle sampling noise.

The two evolution cases apply nine prescribed affine maps to two slices with
different sizes. The same command instances are reused. Frozen centers, sizes
and angles must remain fixed, while quasi-frozen parameters must follow the
known transforms. Actual kicks are checked against independent integrals at
every step. Slicer timestamps are deliberately unrelated to the execution.

Outputs under `<batch>/<profile>_analytic_free_space/run_001/` include:

- `input/beam0.json`, `simulation_manifest.json`, and field HDF5 files;
- `<method>_particle_kicks.npz` with observed momentum changes and independent statistics;
- `analysis/particle_kick_comparison.csv`, `field_scan.csv`, and `summary.json`;
- `analysis/figures/pic_frozen_quasi_frozen_field_comparison.png`;
- `analysis/figures/analytic_kick_vs_independent_integral.png`.

Evolution outputs are in `<batch>/<profile>_parameter_evolution/`, including
per-step HDF5, `parameter_evolution.csv`, `summary.json`, and
`frozen_vs_quasi_frozen_parameter_evolution.png`.

To regenerate generated-case analysis without tracking:

```python
from pathlib import Path
from tests.integration.space_charge.test_analytic_free_space_tracking import analyse

analyse(Path("tests/codex/space_charge_runs/my_batch/gaussian_ellipse_analytic_free_space/run_001"))
```

The `analytic` group uses the default `--mode test`. Existing batch non-test
modes and `--case` still select the original ten PIC workflows.

## Configuration and loss-policy migration

Public resource configurations now use `Method` and `Solver`. PIC solver values
are `fft_free_space`, `fd_dirichlet`, and `dst_dirichlet`; analytic methods share
the four `gaussian_*_free_space` / `uniform_*_free_space` profiles. Each command's
`Aperture type` / `Aperture value` defines losses and the FD/DST conductor.
Old resource `Field solver`, `Aperture`, and `Chamber` keys are rejected.
Grid inputs select a complete full-width or half-width pair; spacing inputs
are rejected. The default command aperture is the complete grid rectangle.
Low-level numerical `build_pic_resources(..., field_solver=...)` identifiers
are `fft_free_space`, `fd`, and `dst_rectangle`.

SpaceCharge applies its shared aperture check before deposition: wall and
outside particles are lost, with first-loss records preserved. Surviving PIC
participants outside the grid or without active stencil nodes cause an error.
Initialization rejects finite PIC apertures larger than the grid and requires
DST apertures to equal the complete grid rectangle. Existing Gaussian
comparisons also retain their upstream +/-0.1 m `MarkerElement` acceptance.
With an explicit aperture or `off`, analytic diagnostic grids do not clip
particles or source charge. Analytic `Save potential` remains rejected.

The local regression group includes full/half-width equivalence, GUI saving,
default losses, command-specific FD resource caching, invalid geometry,
CPU/GPU wall classification, and explicit-rectangle FD/DST field agreement.
HDF5 schema version 3 records the resolved aperture and its role.
