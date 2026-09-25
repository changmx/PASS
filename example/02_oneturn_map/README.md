# Example 02 — Single-Turn Twiss Map Tracking

Use `generate_input.py`, `run_simulation.py`, and `analyze_results.py` as the
workflow entry points for input generation, tracking, and result analysis.

## Overview

This example validates the single-turn Twiss transfer matrix in PASS. The lattice consists of a single Twiss point at s = C (one-turn map) with periodic optical parameters (β_prev = β, α_prev = α), so the map is a pure rotation in normalized phase space.

The workflow consists of three steps:

1. **Generate input** (`generate_input.py`) — write `beam0.json` with 12 prescribed particles and 10000 sampled particles
2. **Run simulation** (`run_simulation.py`) — execute PASS tracking for 1024 turns
3. **Analyze results** (`analyze_results.py`) — five verification modules: tune FFT, CS invariant, analytic matrix comparison, chromaticity, beam statistics

`generate_input.py` fixes the Injection random seed to `2026`, making the generated
distribution particles reproducible between runs.

## Run the example

After installing PASS from the repository root, enter this example directory:

```bash
cd example/02_oneturn_map
```

### Prerequisites

- PASS installed: `pip install -e .` (from project root)

### 1. Generate input

```bash
python generate_input.py
```

### 2. Run simulation

```bash
python run_simulation.py
```

Output is saved to `output/YYYY_MMDD/HHMM_SS/`.

### 3. Analyze results

```bash
python analyze_results.py
```

Auto-detects the latest output directory. Prints all five verification results and displays plots via `plt.show()`.

Diagnostic tables default to gzip-1 + shuffle HDF5 (`.h5`). The analysis scripts
accept both HDF5 and legacy TFS output; set `output_format="tfs"` on the
monitor (or initial-distribution `BunchConfig`) to request TFS explicitly.
Set `output_format="hdf5"` to write uncompressed HDF5; the default
`output_format="hdf5-gzip1"` enables gzip level 1 and shuffle. Both use `.h5` files.
StatMonitor also writes every recorded row to CSV in batches of 100 turns
by default, configurable with `write_interval_turns`. Slicer slice summaries
remain TFS/CSV. See [table output formats](../../docs/source/en/monitor/table_output.rst).

## Lattice

A single Twiss point acting as a one-turn map (no element-by-element tracking):

| Parameter | Value |
|-----------|-------|
| Circumference | 251.327 m |
| βx = βy | 0.5 m |
| αx | −2.6143 |
| αy | 1.5744 |
| Qx | 0.47 |
| Qy | 0.43 |
| Qs | 0.01 |
| DQx = DQy | −2.0 |
| Dx = Dpx | 0 |
| γt | 4.8 |

## Prescribed particles

12 tagged particles cover all verification targets:

| Group | Tags | Purpose | Initial coordinates |
|-------|------|---------|-------------------|
| A | 1–2 | Linear tune | x=2mm or y=2mm, dp=0 |
| B | 3–8 | Chromaticity | x=y=1mm, dp = ±1e-4, ±5e-4, ±1e-3 |
| C | 9 | Longitudinal | z=0.1m, dp=0 |
| D | 10 | Reference | origin (0,0,0,0,0,0) |
| E | 11–12 | Large amplitude | x=5mm or y=5mm, dp=0 |

The remaining 10000 particles follow a KV distribution and contribute to beam statistics. All live particles have positive tags; tag 0 denotes a particle reserved for future injection.

## Longitudinal transport

The `LONGI_TRANSFER` parameter in `generate_input.py` controls the longitudinal transport and determines which FFT measurements are possible:

| Mode | dp behavior | Qs (FFT) | Chromaticity (FFT) | Matrix comparison |
|------|------------|----------|-------------------|-------------------|
| `"off"` | dp fixed (identity matrix) | ✗ (z constant) | ✓ (tune shift is constant) | ✓ coordinate residuals |
| `"matrix"` | dp oscillates with Qs | ✓ (z oscillates) | ✗ (tune is phase-modulated, chromaticity hidden in sidebands) | ✓ coordinate residuals |

With `"matrix"`, dp oscillates as $dp(n) = dp_0 \cos(2\pi Q_s n)$. The chromatic tune shift $Q_x(n) = Q_x + DQ_x \cdot dp(n)$ becomes a phase-modulated signal. FFT decomposes it into a carrier at $Q_x$ plus sidebands at $Q_x \pm k Q_s$ with amplitudes given by Bessel functions $J_k(\beta)$ where $\beta = |DQ_x| \cdot dp_0 / Q_s$. The carrier stays at the original tune — chromaticity is invisible to simple peak-finding.

The analytic matrix calculation checks all six coordinates against the configured
map. Its errors are reported for the current run; agreement with this map does
not establish validity outside its assumptions.

Default: `"off"` (chromaticity measurable).

## Interpreting the diagnostics

- **Tune:** the analyzer applies a Hann window and FFT to turn-by-turn coordinates,
  then interpolates the spectral peak. The unpadded frequency spacing is `1/N`
  cycles per turn. Interpolation estimates a sub-bin peak but does not provide a
  universal accuracy guarantee.
- **Courant–Snyder invariant:** for the uncoupled linear map, evaluate
  $J=\gamma x^2+2\alpha x p_x+\beta p_x^2$ at fixed optical parameters.
  Inspect its relative variation over the run.
- **Analytic map:** iterate the configured transverse rotations, dispersion
  transformations, and longitudinal map, and inspect coordinate residuals.
- **Chromaticity:** with fixed momentum deviation, use symmetric pairs,
  $Q'_x=[Q_x(+\delta)-Q_x(-\delta)]/(2\delta)$, and fit several values of delta.
  Small shifts can be obscured by tune-estimation error.
- **Beam statistics:** inspect RMS emittances and reconstructed optical functions.
  Finite particle sampling introduces statistical variation.

When changing `LONGI_TRANSFER` or optical parameters, keep the corresponding
constants in `analyze_results.py` synchronized with `generate_input.py`.

## Files

| File | Description |
|------|-------------|
| `generate_input.py` | Generate `beam0.json` with PASS Python API |
| `run_simulation.py` | Run PASS simulation |
| `analyze_results.py` | Five verification modules + plots |
| `beam0.json` | Generated PASS input (overwritten by the generator) |
