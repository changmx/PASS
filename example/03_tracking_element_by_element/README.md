# Example 03 — Element-by-Element Tracking

Use `generate_input.py`, `pass-run`, and `analyze_results.py` as the
workflow entry points for input generation, tracking, and result analysis.

## Overview

This example tracks particles through a FODO ring element by element, then extracts tunes, chromaticity, dispersion, and amplitude-dependent tune shifts. The lattice is imported from a Twiss TFS file.

The workflow consists of four steps:

1. **Run MADX** (`fodo.madx`) — generate a MADX Twiss TFS and SEQUENCE files
2. **Generate input** (`generate_input.py`) — read a MADX Twiss TFS, produce `beam0.json` with 17 test particles
3. **Run simulation** (`pass-run`) — execute PASS tracking for 1024 turns
4. **Analyze results** (`analyze_results.py`) — extract tunes using a Hann window, FFT and peak interpolation; fit chromaticity and dispersion

`generate_input.py` fixes the Injection random seed to `2026`, making the generated
distribution particles reproducible between runs.

## Run the example

After installing PASS from the repository root, enter this example directory:

```bash
cd example/03_tracking_element_by_element
```

### Prerequisites

- PASS installed: `python -m pip install --editable .` (from project root), or use the [PyPI installation instructions](../../README.md#install-from-pypi).
- MAD-X executable on PATH, to generate the required TFS input files

### 1. Generate MADX Twiss files

Run the supplied `fodo.madx` to generate the TFS and sequence files. These generated files are not included in a fresh clone.

```bash
madx fodo.madx
```

### 2. Generate PASS input

```bash
python generate_input.py
```

This reads `fodo.tfs` and writes `beam0.json` with 17 test particles, 1024 turns, CPU backend.

### 3. Run PASS tracking

```bash
pass-run --beam0 beam0.json
```

Output is saved to `output/YYYY_MMDD/HHMM_SS/` with one TBT particle-monitor file per monitor and beam.
The run's `input/` folder archives its execution input and dependencies.
Use `pass-run --help` for CLI options. `run_simulation.py` remains a compatibility
wrapper around the same CLI, with `--beam0`, optional `--beam1`, `--output`, and
`--stop-file` arguments.

### 4. Analyze results

```bash
python analyze_results.py
```

Auto-detects the latest output directory. Prints:

- Linear tunes (Group A)
- Linear chromaticity (Group B) from ±dp scan
- Dispersion $D_x$, $D_{px}$ at s=0 (Group B)
- Nonlinear chromaticity (Group C) at large dp
- Amplitude-dependent tune shift (Group D)
- Coupling (Group E)

**Dispersion extraction** uses symmetric ±dp particle pairs:

$$D_x = \frac{\bar{x}(+\delta) - \bar{x}(-\delta)}{2\delta}, \quad D_{px} = \frac{\bar{p}_x(+\delta) - \bar{p}_x(-\delta)}{2\delta}$$

where $\bar{x}$ is the turn-averaged TBT coordinate. The betatron oscillation (independent of $\delta$) cancels in the difference, leaving the dispersion offset. Four ±dp pairs (δ = 5e-5, 1e-4, 5e-4, 1e-3) provide redundant measurements that should agree if the dispersion is linear. The initial dispersion values are read from `fodo.tfs`. Finite turn averaging and nonlinear motion can leave residual betatron contributions.

Options:

```bash
python analyze_results.py --output-dir output/YYYY_MMDD/HHMM_SS
python analyze_results.py --twiss fodo_ptc.tfs
python analyze_results.py --dp-list 5e-5,1e-4,5e-4,1e-3
python analyze_results.py --adts-x 5e-3,10e-3 --adts-y 5e-3,10e-3
```

The analyzer compares derivatives with respect to `delta=P/P0-1`. It converts
native MAD-X `TWISS` chromaticities and dispersion by multiplying by `beta0`.
The bundled `fodo_ptc.tfs` is produced with PTC `TIME=false`, so its
chromaticities already use delta. For an external PTC file, explicitly pass
`--chromaticity-coordinate delta` for `TIME=false` or
`--chromaticity-coordinate pt` for `TIME=true`; the TFS creation-time header
does not identify that physics setting. If regenerating the bundled file with
`TIME=true`, use the same explicit `pt` option.

ParticleMonitor and Injection default to uncompressed HDF5 (`.h5`);
other diagnostic tables default to gzip-1 + shuffle HDF5. The analysis scripts
accept current HDF5 and TFS output. Set `output_format="tfs"` on the monitor
(or initial-distribution `BunchConfig`) for text output, or
`output_format="hdf5-gzip1"` to enable gzip level 1 and shuffle.
Both HDF5 settings use `.h5` files.
Each ParticleMonitor writes one `*_beam*_particles.h5` (or `.tfs`) containing all
selected particles and recorded turns. `Write interval (turns)` controls
buffering and retains every turn. HDF5 appends blocks during tracking;
TFS uses private HDF5 storage during tracking and exports a standard text table
at finalization. The trajectory reader selects individual IDs from one current
PM file with `Layout="single_file"` and `FormatVersion=2`.
StatMonitor also writes every recorded row to CSV in batches of 100 turns
by default, configurable with `write_interval_turns`. Slicer slice summaries
remain TFS/CSV. See [table output formats](../../docs/source/en/monitor/table_output.rst).

## Lattice

The FODO ring (`fodo.seq`) contains:

- 20 FODO cells with focusing/defocusing quadrupoles (QF1, QD1)
- Sextupoles (SF1, SD1) for chromaticity correction
- Optional octupoles OF1 and OD1 are defined in a second cell type, which the default ring does not include
- 40 dipoles (MB) providing horizontal bending

The ring circumference is 234.4 m, with design tunes Qx = 3.47, Qy = 3.43.

## Prescribed particles

17 single-particle test particles are defined in `generate_input.py`:

| Group | Tags | Purpose | Initial coordinates |
|-------|------|---------|-------------------|
| A | 1–2 | Linear tune | x=2mm or y=2mm, dp=0 |
| B | 3–10 | Linear chromaticity | x=y=1mm, dp = ±5e-5, ±1e-4, ±5e-4, ±1e-3 |
| C | 11–12 | Nonlinear chromaticity | x=y=1mm, dp = ±3e-3 |
| D | 13–16 | Amplitude-dependent tune shift | x=5/10mm (y=0) or y=5/10mm (x=0), dp=0 |
| E | 17 | Coupling | x=y=3mm, dp=0 |

Group D uses single-plane excitation (y=0 for x-scan, x=0 for y-scan) to measure each plane separately. The leading perturbative tune shift from an octupole scales with squared amplitude; higher-order dynamics and spectral resolution can change this scaling.

## Files

| File | Description |
|------|-------------|
| `fodo.seq` | MADX sequence file (lattice definition) |
| `fodo.tfs` | MADX Twiss table (linear optics) |
| `fodo_ptc.tfs` | Additional optics table produced by `fodo.madx` and read by the analyzer |
| `generate_input.py` | Generate `beam0.json` from Twiss TFS |
| `run_simulation.py` | Optional compatibility wrapper for the PASS CLI |
| `analyze_results.py` | Analyze PASS output (FFT, chromaticity, ADTS) |
| `beam0.json` | Generated PASS input (overwritten each run) |
