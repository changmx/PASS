# Example 03 — Element-by-Element Tracking

## Overview

This example demonstrates full-ring element-by-element particle tracking in PASS, validated against MADX PTC tracking. A FODO lattice with sextupoles is used to test linear optics and chromaticity.

The workflow consists of four steps:

1. **Run MADX** (`fodo.madx`) — generate a MADX Twiss TFS and SEQUENCE files
2. **Generate input** (`make_input.py`) — read a MADX Twiss TFS, produce `beam0.json` with 17 test particles
3. **Run simulation** (`run.py`) — execute PASS tracking for 1024 turns
4. **Analyze results** (`analyze.py`) — extract tunes via FFT (Hanning window + zero padding + parabolic interpolation), fit chromaticity, compare with MADX PTC reference
5. **Compare with PTC** (`compare_ptc_tracking.py`) — run MADX PTC tracking with identical initial coordinates and produce comparison plots

## Lattice

The FODO ring (`fodo.seq`) contains:

- 20 FODO cells with focusing/defocusing quadrupoles (QF1, QD1)
- Sextupoles (SF1, SD1) for chromaticity correction
- Two octupoles (OF1, OF2) for amplitude-dependent tune shift (k3=0)
- 40 dipoles (MB) providing horizontal bending

The ring circumference is 234.4 m, with design tunes Qx = 3.47, Qy = 3.43.

## Test Particles

17 single-particle test particles are defined in `make_input.py`:

| Group | Tags | Purpose | Initial coordinates |
|-------|------|---------|-------------------|
| A | 1–2 | Linear tune | x=2mm or y=2mm, dp=0 |
| B | 3–10 | Linear chromaticity | x=y=1mm, dp = ±5e-5, ±1e-4, ±5e-4, ±1e-3 |
| C | 11–12 | Nonlinear chromaticity | x=y=1mm, dp = ±3e-3 |
| D | 13–16 | Amplitude-dependent tune shift | x=5/10mm (y=0) or y=5/10mm (x=0), dp=0 |
| E | 17 | Coupling | x=y=3mm, dp=0 |

Group D uses single-plane excitation (y=0 for x-scan, x=0 for y-scan) so that the ADTS follows a strict $A^2$ scaling — the 10 mm particle should show exactly 4× the tune shift of the 5 mm particle.

## Files

| File | Description |
|------|-------------|
| `fodo.seq` | MADX sequence file (lattice definition) |
| `fodo.tfs` | MADX Twiss table (linear optics) |
| `fodo_ptc.tfs` | PTC Twiss table (reference for tune/chromaticity) |
| `make_input.py` | Generate `beam0.json` from Twiss TFS |
| `run.py` | Run PASS simulation |
| `analyze.py` | Analyze PASS output (FFT, chromaticity, ADTS) |
| `compare_ptc_tracking.py` | Compare PASS vs PTC tracking |
| `beam0.json` | Generated PASS input (overwritten each run) |

## How to Run

### Prerequisites

- PASS installed: `pip install -e .` (from project root)
- MADX with cpymad: `pip install cpymad`
- tfs-pandas: `pip install tfs`

### 1. Generate MADX Twiss files

Run MADX to generate Twiss and Sequence files.

```bash
MADX fodo.madx
```

### 2. Generate PASS input

```bash
python make_input.py
```

This reads `fodo.tfs` and writes `beam0.json` with 17 test particles, 1024 turns, CPU backend.

### 3. Run PASS tracking

```bash
python run.py
```

Output is saved to `output/YYYY_MMDD/HHMM_SS/` with per-tag TBT particle monitor files.

### 4. Analyze results

```bash
python analyze.py
```

Auto-detects the latest output directory. Prints:

- Linear tune (Group A) vs MADX PTC reference
- Linear chromaticity (Group B) from ±dp scan
- Dispersion $D_x$, $D_{px}$ at s=0 (Group B) vs MADX Twiss
- Nonlinear chromaticity (Group C) at large dp
- Amplitude-dependent tune shift (Group D)
- Coupling (Group E)

**Dispersion extraction** uses symmetric ±dp particle pairs:

$$D_x = \frac{\bar{x}(+\delta) - \bar{x}(-\delta)}{2\delta}, \quad D_{px} = \frac{\bar{p}_x(+\delta) - \bar{p}_x(-\delta)}{2\delta}$$

where $\bar{x}$ is the turn-averaged TBT coordinate. The betatron oscillation (independent of $\delta$) cancels in the difference, leaving the dispersion offset. Four ±dp pairs (δ = 5e-5, 1e-4, 5e-4, 1e-3) provide redundant measurements that should agree if the dispersion is linear. The MADX reference is read from `fodo.tfs` (not `fodo_ptc.tfs`, which reports DX=0 at s=0 due to PTC output convention).

Options:

```bash
python analyze.py --output-dir output/2026_0731/1642_34
python analyze.py --twiss fodo_ptc.tfs
python analyze.py --dp-list 5e-5,1e-4,5e-4,1e-3
python analyze.py --adts-x 5e-3,10e-3 --adts-y 5e-3,10e-3
```

### 5. Compare with PTC tracking

```bash
python compare_ptc_tracking.py
```

Runs MADX PTC tracking with the same 17 particles and 1024 turns, then produces three comparison figures in `output/comparison/`:

- `phase_space_comparison.png` — transverse phase space scatter
- `relative_difference_comparison.png` — (PASS − PTC) / amplitude vs turn
- `tbt_trajectory_comparison.png` — TBT waveform overlay

Options:

```bash
python compare_ptc_tracking.py --output-dir output/2026_0731/1642_34
```

## Note on MADX PTC

- The chromaticities calculated by the three MADX commands - `TWISS; TWISS, CHROM; PTC_TWISS;` are all different. Here, we will use the results from PTC_TWISS as the standard.

