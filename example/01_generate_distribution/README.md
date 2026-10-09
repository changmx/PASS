# Example 01 - Particle Distribution Generation

Use `generate_input.py`, `pass-run`, and `analyze_results.py` as the
workflow entry points for input generation, tracking, and result analysis.

This is the first PASS distribution-generation example. It introduces the
three basic steps used by the other examples:

1. `generate_input.py` generates `beam0_*.json` using the PASS Python API.
2. `pass-run` reads a JSON input, performs one injection, and saves the initial
   particle distribution.
3. `analyze_results.py` reads the generated HDF5 files (or TFS files), validates the distribution
   types, calculates statistics, and compares them with theory.

This example contains no transport elements. `Num Turns = 1` lets PASS finish
the injection and then stop. It is therefore intended to verify that input
parameters reach the distribution generators correctly, rather than to study
transport through magnets, RF cavities, or space charge.

`generate_input.py` fixes the Injection random seed to `2026`, so regenerating and
running the same case produces the same initial particle distribution.

## Run the example

Install PASS following the [installation instructions](../../README.md#install).
From the repository root, run:

```powershell
cd example/01_generate_distribution
python generate_input.py --case transverse
pass-run --beam0 beam0_transverse.json
python analyze_results.py --case transverse
```

These commands generate, run, and analyse the case in that order. In normal use,
do not edit the generated JSON directly. Modify constants or `CASES` in
`generate_input.py`, then regenerate the input.

To process every predefined case, use the batch helper:

```powershell
python generate_input.py --case all
python run_simulation.py --case all
python analyze_results.py --case all
```

`run_simulation.py --case all` invokes the same CLI for each case serially and
stops if a run fails or is stopped. The helper also accepts a single `--case`
and defaults to `transverse` when that option is omitted.

An existing input can also be run directly, without the case mapping:

```powershell
pass-run --beam0 C:\path\to\beam0.json
```

Use `pass-run --help` for input, output, and stop-file options. The batch helper
also accepts `--output` and `--stop-file`. By default, each case uses
`output/<case>/YYYY_MMDD/HHMM_SS/`, with its execution input and dependencies
archived in the run's `input/` folder. To keep a separate set of inputs and
results together, pass the same `--work-dir` to all three scripts:

```powershell
python generate_input.py --case transverse --work-dir C:\path\to\distribution-study
python run_simulation.py --case transverse --work-dir C:\path\to\distribution-study
python analyze_results.py --case transverse --work-dir C:\path\to\distribution-study
```

Inputs go directly in that directory; results go in its `output/<case>/`
subdirectories. Input generation rewrites the selected JSON files, so use a new
work directory to preserve an earlier configuration. Each simulation creates a
timestamped run. A raw `--output` override on the runner still changes the output
root directly; the analyzer expects the `output/<case>/` layout.

## Cases

| Case | Input file | Bunches | Purpose |
|------|------------|---------|---------|
| `transverse` | `beam0_transverse.json` | 6 | Gaussian, KV, waterbag, parabolic, uniform-real, and uniform-phase transverse distributions; all use longitudinal Gaussian |
| `longi-gaussian` | `beam0_longi_gaussian.json` | 1 | Ordinary Gaussian longitudinal distribution |
| `longi-matchz` | `beam0_longi_matchz.json` | 1 | RF-matched longitudinal distribution specified by target `Sigma z` |
| `longi-matchdp` | `beam0_longi_matchdp.json` | 1 | RF-matched longitudinal distribution specified by target `Sigma dp/p` |
| `coasting` | `beam0_coasting.json` | 1 | Coasting beam with longitudinal particles uniformly distributed around the ring |

The transverse test uses six bunches so that several transverse sampling
methods can be compared in one input. They all use ordinary longitudinal
Gaussian distributions, preventing RF-matching parameters from affecting the
transverse comparison.

`matchz` and `matchdp` require an RF harmonic number, while the injection
harmonic number is determined by the number of declared bunches. These cases
therefore use separate one-bunch inputs, which gives `h=1`. `coasting` is also
separate because its longitudinal coordinate covers the whole ring instead of
a local bunch.

## Input generation

The important sections in `generate_input.py` are:

- `CASES`: declares the bunch count and transverse/longitudinal distribution
  types for every case.
- `make_main()`: sets ring and run settings such as circumference, transition
  gamma, particle count, and output directory.
- `make_bunch()`: sets a bunch's energy, Twiss parameters, emittances, RMS
  sizes, and RF parameters.

The main parameters in this example are:

```text
Circumference = 251.327 m
Gamma T = 4.8
Kinetic energy = 45 MeV/u
Macro particles per bunch = 100000
RMS geometric emittance x / y = 200e-6 / 100e-6 m rad
Beta x / Beta y = 0.5 / 0.5 m
```

The Python arguments remain `emit_x` and `emit_y`. Generated JSON uses
`RMS geometric emittance x (m'rad)` and `RMS geometric emittance y (m'rad)`.
The former `Emittance x (m'rad)` / `Emittance y (m'rad)` keys are not supported.
These values specify the intrinsic RMS geometric emittance in each transverse
plane. All six distributions use this same convention.
The saved distribution headers retain `Emit x` and `Emit y` with this meaning.

The ordinary longitudinal Gaussian bunch uses:

```text
Sigma z = 5 m
Sigma dp/p = 1e-3
```

`matchz` and `matchdp` retain the RF-related settings from the original
`beam0` input:

```text
Sigma z = 30 m
Sigma dp/p = 5e-3
RF voltage = 100 kV
RF phase = pi/6
RF matching harmonic = 1
```

For matched distributions, `Sigma z` and `Sigma dp/p` are two different
control variables. `matchz` is constrained by bunch length, while `matchdp`
is constrained by momentum spread. The other value remains present for a
consistent input structure, but it is not a simultaneous matching target.

## Analysis output

`analyze_results.py` finds the latest completed run for each selected case and writes:

```text
output/<case>/<date>/<time>/
    distribution/*_injection.h5
    analysis/<case>_summary.csv
    analysis/<case>_distributions.png
    analysis/<case>_joint_actions.png
    analysis/<case>_joint_actions.pdf
    analysis/<case>_physics_checks.csv
    analysis/<case>_physics_checks.json
```

The CSV contains particle count, measured RMS values, transverse RMS
emittances and Twiss parameters, plus theoretical values and relative errors.
The transverse theoretical values are calculated from the Twiss parameters in
the HDF5 attributes (or TFS headers):

```text
sigma_x = sqrt(beta_x * emit_x)
sigma_px = sqrt(gamma_x * emit_x)
gamma_x = (1 + alpha_x^2) / beta_x
```

The same relations apply in the vertical plane. The longitudinal theory for an
ordinary Gaussian distribution is the configured `Sigma z` and `Sigma dp/p`.
For a coasting beam, the theoretical `sigma_z` is `C / sqrt(12)`, the RMS of a
uniform distribution over one circumference.

The action plots use the configured, uncentered Courant-Snyder invariants:

```text
I_x = x^2/beta_x + (alpha_x*x + beta_x*px)^2/beta_x
a_x = I_x / (4*emit_x)
```

The vertical formula is identical. This example has zero centroid offsets and
zero dispersion, so these invariants describe the intrinsic distribution
directly. If you introduce either, subtract the corresponding centroid and
dispersion before comparing intrinsic emittances or actions.

| Distribution | Joint normalized actions | Ideal correlation of a_x and a_y | Spatial x-y projection |
|---|---|---:|---|
| `gaussian` | Independent exponential variables, apart from the position cuts | 0 | Gaussian |
| `kv` | Uniform along `a_x + a_y = 1` | -1 | Uniform ellipse |
| `waterbag` | Uniform inside `a_x + a_y <= 3/2` | -1/2 | Parabolic ellipse |
| `parabolic` | Density proportional to `1 - (a_x + a_y)/2` inside `a_x + a_y <= 2` | -1/3 | Squared-parabolic ellipse |
| `uniform-real` | Independent actions from square phase-space samples; each is bounded by 3/2 | 0 | Uniform rectangle |
| `uniform-phase` | Independent Uniform[0, 1] variables | 0 | Product of two semicircle profiles inside a rectangle |

The checks verify RMS emittances, Twiss parameters, action means and correlation,
and each bounded distribution's support. For `uniform-phase` they also test both
uniform action CDFs and a 10-by-10 joint action histogram. The CSV and JSON record
the measured value, theory, signed error, tolerance, comparison rule, and pass
status for every check; analysis raises an error after saving results if any
check fails. Moment ratios use `max(0.02, 8/sqrt(N))`, correlations use
`max(0.025, 8/sqrt(N))`, and action CDFs use `max(0.012, 4/sqrt(N))`.
At 100000 particles these are approximately 2.53%, 0.0253, and 0.012.
Statistical tolerances expand with `1/sqrt(N)` for smaller samples. Support
checks propagate the stored coordinate precision through the Twiss action map,
including cancellation in `alpha*x + beta*px`; their tolerance has a `1e-10`
floor for float64 samples and expands for float32 rounding. Gaussian sampling
retains its existing position cuts at
4 sigma, so its target covariance differs very slightly from the untruncated
Gaussian theory.

`uniform-phase` fills each transverse phase-space ellipse independently. Its
single-plane boundary emittance is four times the RMS value; a stable sampling
formula is `r = sqrt(U)`, `phase = 2*pi*V`, with independent uniform draws in
each plane. This is a useful idealized initial state after two-plane painting
when the final transverse actions are approximately independent. A particular
painting history may instead correlate those actions or change the density.
This initial-state example does not simulate injection turns, moving closed
orbits, foil scattering, losses, or collective evolution.

KV and `uniform-phase` have the same uniformly filled single-plane phase-space
ellipses. Their joint action plots and spatial x-y projections distinguish them:
KV links the actions through a fixed sum, while `uniform-phase` fills an action
rectangle. Uniform single-plane phase space does not imply uniform x-y density.

For `matchz` and `matchdp`, the analysis also calculates first-order RF bucket
limits, `dp_max`, synchrotron tune `Qs`, and the slip factor from the output
headers: energy, `Gamma T`, RF voltage, RF phase, and harmonic number. These
values are printed to the terminal and written to the summary CSV.

The configured `matchz` setting, `Sigma z = 30 m`, is larger than the maximum
matched bunch length supported by the current RF bucket. PASS automatically
reduces the target to approximately `0.99` of its limit during generation.
For this case, `analyze_results.py` reports the measured `sigma_z`, the requested
value, and the RF bucket boundaries rather than treating 30 m as the final
theoretical RMS. This is expected behavior, not a failed run. Reduce
`MATCH_SIGMA_Z` to avoid clipping.

## Modify a case

For example, add another transverse Gaussian / longitudinal Gaussian bunch by
adding this item to `CASES["transverse"]["bunches"]`:

```python
{"transverse": "gaussian", "longitudinal": "gaussian"}
```

Adding a bunch also increases the injection harmonic number because it always
equals the declared bunch count. For longitudinal RF-matching tests that need
a specific harmonic number, keep the separate one-case, one-bunch structure.

Supported transverse distributions:

```text
gaussian, kv, waterbag, parabolic, uniform-real, uniform-phase
```

The former `uniform` name is no longer accepted. Use `uniform-real` for its
existing independent normalized-square sampling, or `uniform-phase` for
independent uniformly filled phase-space ellipses. Regenerate older inputs to
update both the distribution name and the RMS geometric emittance JSON keys.

Supported longitudinal distributions:

```text
gaussian, coasting, matchz, matchdp
```

Changing `NUM_MACRO_PARTICLES` changes both statistical error and runtime.
Changing `EMIT_*`, `BETA_*`, or `ALPHA_*` changes the transverse theoretical
RMS values. Changing the RF parameters changes both the matched distributions
and the RF bucket theory.

## Troubleshooting

- `Input file does not exist`: run `python generate_input.py --case <case>` first.
- `No completed run found`: run `pass-run --beam0 <case-input.json>` with the input filename from the table above before analysis.
- Results do not change after editing `generate_input.py`: regenerate the JSON
  because it is a generated file.
- Measured `sigma_z` is smaller than requested for `matchz`: check whether the
  RF bucket is large enough. The retained original parameters intentionally
  trigger the expected clipping behavior.
- Generated JSON, HDF5/TFS files, images, and CSV files are ignored by the local
  `.gitignore`.

ParticleMonitor and Injection default to uncompressed HDF5 (`.h5`);
other diagnostic tables default to gzip-1 + shuffle HDF5. The analysis scripts
accept current HDF5 and TFS output. Set `output_format="tfs"` on the monitor
(or initial-distribution `BunchConfig`) for text output, or
`output_format="hdf5-gzip1"` to enable gzip level 1 and shuffle.
Both HDF5 settings use `.h5` files.
StatMonitor also writes every recorded row to CSV in batches of 100 turns
by default, configurable with `write_interval_turns`. Slicer slice summaries
remain TFS/CSV. See [table output formats](../../docs/source/en/monitor/table_output.rst).
