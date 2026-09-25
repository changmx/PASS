# Example 01 - Particle Distribution Generation

Use `generate_input.py`, `run_simulation.py`, and `analyze_results.py` as the
workflow entry points for input generation, tracking, and result analysis.

This is the first PASS distribution-generation example. It introduces the
three basic steps used by the other examples:

1. `generate_input.py` generates `beam0_*.json` using the PASS Python API.
2. `run_simulation.py` reads a JSON input, performs one injection, and saves the initial
   particle distribution.
3. `analyze_results.py` reads the generated HDF5 files (or legacy TFS files), validates the distribution
   types, calculates statistics, and compares them with theory.

This example contains no transport elements. `Num Turns = 1` lets PASS finish
the injection and then stop. It is therefore intended to verify that input
parameters reach the distribution generators correctly, rather than to study
transport through magnets, RF cavities, or space charge.

`generate_input.py` fixes the Injection random seed to `2026`, so regenerating and
running the same case produces the same initial particle distribution.

## Run the example

From the repository root, run:

```powershell
cd example/01_generate_distribution
python generate_input.py --case transverse
python run_simulation.py --case transverse
python analyze_results.py --case transverse
```

The scripts generate, run, and analyse the case in that order. In normal use,
do not edit the generated JSON directly. Modify constants or `CASES` in
`generate_input.py`, then regenerate the input.

To process every predefined case:

```powershell
python generate_input.py --case all
python run_simulation.py --case all
python analyze_results.py --case all
```

`run_simulation.py` defaults to `transverse`, so this is equivalent to
`python run_simulation.py --case transverse`:

```powershell
python run_simulation.py
```

An existing input can also be run directly, without the case mapping:

```powershell
python run_simulation.py --beam0 C:\path\to\beam0.json
```

## Cases

| Case | Input file | Bunches | Purpose |
|------|------------|---------|---------|
| `transverse` | `beam0_transverse.json` | 5 | Gaussian, KV, waterbag, parabolic, and uniform transverse distributions; all use longitudinal Gaussian |
| `longi-gaussian` | `beam0_longi_gaussian.json` | 1 | Ordinary Gaussian longitudinal distribution |
| `longi-matchz` | `beam0_longi_matchz.json` | 1 | RF-matched longitudinal distribution specified by target `Sigma z` |
| `longi-matchdp` | `beam0_longi_matchdp.json` | 1 | RF-matched longitudinal distribution specified by target `Sigma dp/p` |
| `coasting` | `beam0_coasting.json` | 1 | Coasting beam with longitudinal particles uniformly distributed around the ring |

The transverse test uses five bunches so that several transverse sampling
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
Emit x / Emit y = 200e-6 / 100e-6 m rad
Beta x / Beta y = 0.5 / 0.5 m
```

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
gaussian, kv, waterbag, parabolic, uniform
```

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
- `No completed run found`: run `python run_simulation.py --case <case>` before analysis.
- Results do not change after editing `generate_input.py`: regenerate the JSON
  because it is a generated file.
- Measured `sigma_z` is smaller than requested for `matchz`: check whether the
  RF bucket is large enough. The retained original parameters intentionally
  trigger the expected clipping behavior.
- Generated JSON, HDF5/TFS files, images, and CSV files are ignored by the local
  `.gitignore`.

Diagnostic tables now default to gzip-1 + shuffle HDF5 (`.h5`). The analysis scripts
accept both HDF5 and legacy TFS output; set `output_format="tfs"` on the
monitor (or initial-distribution `BunchConfig`) to request TFS explicitly.
Set `output_format="hdf5"` to write uncompressed HDF5; the default
`output_format="hdf5-gzip1"` enables gzip level 1 and shuffle. Both use `.h5` files.
StatMonitor also writes every recorded row to CSV in batches of 100 turns
by default, configurable with `write_interval_turns`. Slicer slice summaries
remain TFS/CSV. See [table output formats](../../docs/source/en/monitor/table_output.rst).
