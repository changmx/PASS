# Example 01 - Particle Distribution Generation

This is the first PASS distribution-generation example. It introduces the
three basic steps used by the other examples:

1. `make_input.py` generates `beam0_*.json` using the PASS Python API.
2. `run.py` reads a JSON input, performs one injection, and saves the initial
   particle distribution.
3. `analyse.py` reads the generated TFS files, validates the distribution
   types, calculates statistics, and compares them with theory.

This example contains no transport elements. `Num Turns = 1` lets PASS finish
the injection and then stop. It is therefore intended to verify that input
parameters reach the distribution generators correctly, rather than to study
transport through magnets, RF cavities, or space charge.

## Run A Minimal Workflow

From the repository root, run:

```powershell
cd C:\Users\changmx\Documents\PASS\example\01_generate_distribution
python make_input.py --case transverse
python run.py --case transverse
python analyse.py --case transverse
```

The scripts generate, run, and analyse the case in that order. In normal use,
do not edit the generated JSON directly. Modify constants or `CASES` in
`make_input.py`, then regenerate the input.

To process every predefined case:

```powershell
python make_input.py --case all
python run.py --case all
python analyse.py --case all
```

`run.py` defaults to `transverse`, so this is equivalent to
`python run.py --case transverse`:

```powershell
python run.py
```

An existing input can also be run directly, without the case mapping:

```powershell
python run.py --beam0 C:\path\to\beam0.json
```

## Case Design

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

## How The Input Is Generated

The important sections in `make_input.py` are:

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

## Analysis Output

`analyse.py` finds the latest completed run for each selected case and writes:

```text
output/<case>/<date>/<time>/
    distribution/*_injection.tfs
    analysis/<case>_summary.csv
    analysis/<case>_distributions.png
```

The CSV contains particle count, measured RMS values, transverse RMS
emittances and Twiss parameters, plus theoretical values and relative errors.
The transverse theoretical values are calculated from the Twiss parameters in
the TFS headers:

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

The original `matchz` setting, `Sigma z = 30 m`, is larger than the maximum
matched bunch length supported by the current RF bucket. PASS automatically
reduces the target to approximately `0.99` of its limit during generation.
For this case, `analyse.py` reports the measured `sigma_z`, the requested
value, and the RF bucket boundaries rather than treating 30 m as the final
theoretical RMS. This is expected behavior, not a failed run. Reduce
`MATCH_SIGMA_Z` or increase RF voltage to avoid clipping.

## Add Or Modify Tests

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

## Common Issues

- `Input file does not exist`: run `python make_input.py --case <case>` first.
- `No completed run found`: run `python run.py --case <case>` before analysis.
- Results do not change after editing `make_input.py`: regenerate the JSON
  because it is a generated file.
- Measured `sigma_z` is smaller than requested for `matchz`: check whether the
  RF bucket is large enough. The retained original parameters intentionally
  trigger the expected clipping behavior.
- Generated JSON, TFS files, images, and CSV files are ignored by the local
  `.gitignore`.
