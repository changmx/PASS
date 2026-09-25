# Example 04 — Twiss-by-Twiss Tracking

Use `generate_input.py`, `run_simulation.py`, and `analyze_results.py` as the
workflow entry points for input generation, tracking, and result analysis.

## Overview

This example demonstrates **twiss-by-twiss tracking**: particles are transported through the ring using a sequence of Twiss transfer matrices (one per MADX TFS row), with thin-lens multipoles (sextupole component K2L) inserted at their s-positions for chromaticity control.

The transport model:

- Uses **pure linear optics** between Twiss points (no nonlinear element bodies)
- Inserts **thin-lens multipoles** (length=0) at their s-positions for nonlinear effects
- Distributes the **natural chromaticity** across all Twiss points proportional to phase advance
- The total chromaticity = natural (from TwissItem DQx) + multipole correction (from kicks)

`generate_input.py` fixes the Injection random seed to `2026`, making the generated
distribution particles reproducible between runs.

## Run the example

After installing PASS from the repository root, enter this example directory:

```bash
cd example/04_tracking_twiss_by_twiss
```

### Prerequisites

- PASS installed: `pip install -e .` (from project root)
- MADX executable available in PATH

### 1. Generate the TFS inputs

The required TFS files are generated from the supplied `fodo.madx`; they are not included in a fresh clone.

```bash
madx fodo.madx
```

### 2. Generate PASS input

```bash
python generate_input.py
```

### 3. Run simulation

```bash
python run_simulation.py
```

### 4. Analyze results

```bash
python analyze_results.py
```

Plots are displayed interactively via `plt.show()` (not saved to disk).

Diagnostic tables default to gzip-1 + shuffle HDF5 (`.h5`). The analysis scripts
accept both HDF5 and legacy TFS output; set `output_format="tfs"` on the
monitor (or initial-distribution `BunchConfig`) to request TFS explicitly.
Set `output_format="hdf5"` to write uncompressed HDF5; the default
`output_format="hdf5-gzip1"` enables gzip level 1 and shuffle. Both use `.h5` files.
StatMonitor also writes every recorded row to CSV in batches of 100 turns
by default, configurable with `write_interval_turns`. Slicer slice summaries
remain TFS/CSV. See [table output formats](../../docs/source/en/monitor/table_output.rst).

## Method

### Two-step MADX workflow

| MADX run | K2 | DQ1 (header) | Purpose |
|----------|-----|-------------|---------|
| `fodo_natural.tfs` | 0 (TSF1/TSD1 KNL = 0) | -2.9488 | Read natural chromaticity |
| `fodo.tfs` | ≠0 (TSF1/TSD1 KNL) | -0.8767 | Read Twiss parameters + multipole K2L |

The Twiss parameters (β, α, μ, Dx, Dpx) are identical in both files because multipoles do not affect linear optics at dp=0.

Note: the corrected chromaticity is set manually in `fodo.madx` via `TSF1: MULTIPOLE, KNL = {0, 0, 0.039}` and `TSD1: MULTIPOLE, KNL = {0, 0, -0.06}` (not via a MATCH,CHROM fit).

### Chromaticity decomposition

The total chromaticity is split into two contributions:

1. **Natural chromaticity** (from quadrupole $K_1 \propto 1/p$): distributed to each TwissItem as `DQx_i = DQ1_nat × Δμ_x_i / Q_x`. This is a first-order approximation — the exact distribution would require per-element $\beta K_1$ data.

2. **Multipole correction** (from thin-lens kicks): the sextupole component kick $\Delta p_x = -\frac{k_{2l}}{2}(x^2 - y^2)$ with $x = x_\beta + D_x \cdot dp$ produces an effective quadrupole $K_{1,\text{eff}}L = k_{2l} D_x \cdot dp$, generating chromaticity $\Delta Q'_x = \frac{k_{2l} D_x \beta_x}{4\pi}$.

## Lattice

FODO ring (20 repetitions of one arc cell):

| Parameter | Value |
|-----------|-------|
| Circumference | 234.4 m |
| Qx | 3.47 |
| Qy | 3.43 |
| DQ1 (natural) | -2.9488 |
| DQ2 (natural) | -3.2450 |
| DQ1 (corrected) | -0.8767 |
| DQ2 (corrected) | -0.7046 |
| TSF1 K2L | +0.039 |
| TSD1 K2L | -0.060 |
| Dx at s=0 | 4.128 m |

Note: SF1/SD1 (SEXTUPOLE, K2 = 0) exist in the lattice but carry zero strength; the sextupole field is provided by the TSF1/TSD1 multipoles.

## Prescribed particles

12 tagged particles + 10000 KV-distributed particles:

| Group | Tags | Purpose | Initial coordinates |
|-------|------|---------|-------------------|
| A | 1–2 | Linear tune | x=2mm or y=2mm, dp=0 |
| B | 3–8 | Chromaticity | x=y=1mm, dp = ±1e-5, ±5e-5, ±1e-4 |
| C | 9 | Longitudinal | z=0.1m, dp=0 |
| D | 10 | Reference | origin |
| E | 11–12 | Large amplitude | x=5mm or y=5mm, dp=0 |

Small dp values (1e-5 to 1e-4) are used because the lattice has large dispersion (Dx=4.13m), causing significant nonlinear sextupole effects at higher dp.

## Interpreting the diagnostics

The analyzer reports tunes, Courant–Snyder invariants, chromaticity, dispersion,
and beam moments for the current run. The imported optical functions set the
linear transport, while the inserted multipoles add nonlinear motion.
Invariants need not remain constant once nonlinear kicks are present.
Check sensitivity to particle amplitude, momentum offset, and tracking length
before interpreting fitted chromaticity or dispersion.

## Files

| File | Description |
|------|-------------|
| `fodo.madx` | MADX lattice file (run manually via `madx fodo.madx`, outputs natural + corrected TFS) |
| `fodo_natural.tfs` | MADX Twiss with K2=0 (natural chromaticity) |
| `fodo.tfs` | MADX Twiss with K2≠0 (corrected chromaticity) |
| `generate_input.py` | Generate `beam0.json` from TFS files |
| `run_simulation.py` | Run PASS simulation |
| `analyze_results.py` | Five verification modules + interactive plots |
| `beam0.json` | Generated PASS input |
