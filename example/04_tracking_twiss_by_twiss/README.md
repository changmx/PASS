# Example 04 — Twiss-by-Twiss Tracking

## Overview

This example demonstrates **twiss-by-twiss tracking**: particles are transported through the ring using a sequence of Twiss transfer matrices (one per MADX TFS row), with thin-lens multipoles (sextupole component K2L) inserted at their s-positions for chromaticity control.

Compared to Example 03 (element-by-element tracking with drift-kick-drift integration), this approach:

- Uses **pure linear optics** between Twiss points (no nonlinear element bodies)
- Inserts **thin-lens multipoles** (length=0) at their s-positions for nonlinear effects
- Distributes the **natural chromaticity** across all Twiss points proportional to phase advance
- The total chromaticity = natural (from TwissPoint DQx) + multipole correction (from kicks)

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

1. **Natural chromaticity** (from quadrupole $K_1 \propto 1/p$): distributed to each TwissPoint as `DQx_i = DQ1_nat × Δμ_x_i / Q_x`. This is a first-order approximation — the exact distribution would require per-element $\beta K_1$ data.

2. **Multipole correction** (from thin-lens kicks): the sextupole component kick $\Delta p_x = -\frac{k_{2l}}{2}(x^2 - y^2)$ with $x = x_\beta + D_x \cdot dp$ produces an effective quadrupole $K_{1,\text{eff}} = -k_{2l} D_x \cdot dp$, generating chromaticity $\Delta Q'_x = \frac{k_{2l} D_x \beta_x}{4\pi}$.

### Name collision fix

TwissPoint names are prefixed with `twiss_` (e.g. `twiss_TSF1_s1.500`) to avoid collision with inserted element names (e.g. `TSF1_s1.500`) in the Sequence's OrderedDict.

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

## Test Particles

12 tagged particles + 10000 KV-distributed particles:

| Group | Tags | Purpose | Initial coordinates |
|-------|------|---------|-------------------|
| A | 1–2 | Linear tune | x=2mm or y=2mm, dp=0 |
| B | 3–8 | Chromaticity | x=y=1mm, dp = ±1e-5, ±5e-5, ±1e-4 |
| C | 9 | Longitudinal | z=0.1m, dp=0 |
| D | 10 | Reference | origin |
| E | 11–12 | Large amplitude | x=5mm or y=5mm, dp=0 |

Small dp values (1e-5 to 1e-4) are used because the lattice has large dispersion (Dx=4.13m), causing significant nonlinear sextupole effects at higher dp.

## Verification Results

### 1. Tune Measurement

| Tune | Expected | Measured | Error |
|------|----------|----------|-------|
| Qx | 0.4700 | 0.469949 | -5.1e-5 (FFT resolution) |
| Qy | 0.4300 | 0.429949 | -5.1e-5 (FFT resolution) |

### 2. Courant-Snyder Invariant

CS invariants are nearly constant for dp=0 particles (std/mean ~1e-4). Small variation for large-amplitude particles is from multipole nonlinear kicks (expected). Jx/εx = Jy/εy = 1.0 to machine precision.

### 3. Chromaticity

Expected (fodo.tfs headers): DQx = -0.8767, DQy = -0.7046. The PASS measurement should reproduce these corrected values (linear fit over dp = ±1e-5, ±5e-5, ±1e-4).

### 4. Dispersion

| dp | Dx | Dpx |
|----|----|-----|
| 1e-5 | 4.124 | 0.385 |
| 1e-4 | 4.124 | 0.385 |

Expected: Dx=4.128, Dpx=0.385. ✅ (0.1% accuracy)

### 5. Beam Statistics

| Parameter | Measured | Expected |
|-----------|----------|----------|
| Jx/εx | 1.0000 | 1.0 (machine precision) |
| Jy/εy | 1.0000 | 1.0 (machine precision) |

## Files

| File | Description |
|------|-------------|
| `fodo.madx` | MADX lattice file (run manually via `madx fodo.madx`, outputs natural + corrected TFS) |
| `fodo_natural.tfs` | MADX Twiss with K2=0 (natural chromaticity) |
| `fodo.tfs` | MADX Twiss with K2≠0 (corrected chromaticity) |
| `make_input.py` | Generate `beam0.json` from TFS files |
| `run.py` | Run PASS simulation |
| `analyse.py` | Five verification modules + interactive plots |
| `beam0.json` | Generated PASS input |

## How to Run

### Prerequisites

- PASS installed: `pip install -e .` (from project root)
- MADX executable available in PATH
- tfs-pandas: `pip install tfs`

### 1. Generate MADX TFS files

```bash
madx fodo.madx
```

### 2. Generate PASS input

```bash
python make_input.py
```

### 3. Run simulation

```bash
python run.py
```

### 4. Analyze results

```bash
python analyse.py
```

Plots are displayed interactively via `plt.show()` (not saved to disk).
