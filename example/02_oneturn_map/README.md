# Example 02 — Single-Turn Twiss Map Tracking

## Overview

This example validates the single-turn Twiss transfer matrix in PASS. The lattice consists of a single Twiss point at s = C (one-turn map) with periodic optical parameters (β_prev = β, α_prev = α), so the map is a pure rotation in normalized phase space.

The workflow consists of three steps:

1. **Generate input** (`make_input.py`) — write `beam0.json` with 12 test particles + 10000 distribution particles
2. **Run simulation** (`run.py`) — execute PASS tracking for 1024 turns
3. **Analyze results** (`analyse.py`) — five verification modules: tune FFT, CS invariant, analytic matrix comparison, chromaticity, beam statistics

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

## Test Particles

12 tagged particles cover all verification targets:

| Group | Tags | Purpose | Initial coordinates |
|-------|------|---------|-------------------|
| A | 1–2 | Linear tune | x=2mm or y=2mm, dp=0 |
| B | 3–8 | Chromaticity | x=y=1mm, dp = ±1e-4, ±5e-4, ±1e-3 |
| C | 9 | Longitudinal | z=0.1m, dp=0 |
| D | 10 | Reference | origin (0,0,0,0,0,0) |
| E | 11–12 | Large amplitude | x=5mm or y=5mm, dp=0 |

Plus 10000 KV-distributed particles (tag=0) for beam statistics.

## Longitudinal Transfer Mode

The `LONGI_TRANSFER` parameter in `make_input.py` controls the longitudinal transport and determines which FFT measurements are possible:

| Mode | dp behavior | Qs (FFT) | Chromaticity (FFT) | Matrix comparison |
|------|------------|----------|-------------------|-------------------|
| `"off"` | dp fixed (identity matrix) | ✗ (z constant) | ✓ (tune shift is constant) | ✓ machine precision |
| `"matrix"` | dp oscillates with Qs | ✓ (z oscillates) | ✗ (tune is phase-modulated, chromaticity hidden in sidebands) | ✓ machine precision |

**Why can't both be measured simultaneously?** With `"matrix"`, dp oscillates as $dp(n) = dp_0 \cos(2\pi Q_s n)$. The chromatic tune shift $Q_x(n) = Q_x + DQ_x \cdot dp(n)$ becomes a phase-modulated signal. FFT decomposes it into a carrier at $Q_x$ plus sidebands at $Q_x \pm k Q_s$ with amplitudes given by Bessel functions $J_k(\beta)$ where $\beta = |DQ_x| \cdot dp_0 / Q_s$. The carrier stays at the original tune — chromaticity is invisible to simple peak-finding.

The **analytic matrix comparison** in `analyse.py` verifies the full 6D transport (including chromaticity and longitudinal) to machine precision (~1e-15) regardless of the transfer mode. It is the definitive verification.

Default: `"off"` (chromaticity measurable).

## Verification Modules

### 1. Tune Measurement (FFT)

Single-particle TBT signal → Hann window → FFT → parabolic interpolation.

| Tune | Expected | Measured | Error |
|------|----------|----------|-------|
| Qx | 0.4700 | 0.469949 | −5.1e-5 (FFT resolution) |
| Qy | 0.4300 | 0.429949 | −5.1e-5 (FFT resolution) |

FFT resolution = 1/N = 1/1024 ≈ 9.77e-4. The measured shift is exactly one bin — expected for 1024 turns.

### 2. Courant-Snyder Invariant

For each particle, $J = \gamma x^2 + 2\alpha x p_x + \beta p_x^2$ should be constant turn-by-turn.

Result: std(J)/mean(J) ≈ 1e-16 (machine precision) for all 12 particles. The map is symplectic.

### 3. Analytic Matrix Comparison

Compares PASS TBT output with a hand-computed 6D one-turn matrix applied iteratively. The analytic matrix includes:

- Twiss rotation (with chromatic phase advance $\mu_x + dp \cdot DQ_x$)
- Dispersion removal/addition
- Longitudinal transport (identity for `"off"`, rotation for `"matrix"`)

Result: max|Δ| ≈ 5e-15 for all particles, all 6 coordinates, all 1024 turns. This is the **definitive verification** — it confirms the Twiss transfer implementation is correct to machine precision.

### 4. Chromaticity

Symmetric ±dp particle pairs: $DQ_x = (Q_x^{+dp} - Q_x^{-dp}) / (2 \cdot dp)$.

Requires `LONGI_TRANSFER = "off"` (dp fixed).

| dp | Qx(+dp) | Qx(−dp) | DQx | Qy(+dp) | Qy(−dp) | DQy |
|----|---------|---------|-----|---------|---------|-----|
| 1e-4 | 0.4698 | 0.4702 | −2.04 | 0.4298 | 0.4302 | −2.21 |
| 5e-4 | 0.4690 | 0.4710 | −2.00 | 0.4289 | 0.4310 | −2.00 |
| 1e-3 | 0.4680 | 0.4720 | −2.00 | 0.4280 | 0.4320 | −2.00 |

Linear fit: DQx = −1.998, DQy = −2.003 (expected −2.0). ✅

### 5. Beam Statistics

From 10000 KV-distributed particles via StatMonitor:

| Parameter | Measured | Expected | Relative error |
|-----------|----------|----------|---------------|
| βx | 0.50006 | 0.5 | 0.01% |
| βy | 0.50003 | 0.5 | 0.006% |
| αx | −2.6154 | −2.6143 | 0.04% |
| αy | 1.5743 | 1.5744 | 0.006% |
| εx | 2.004e-4 | 2.0e-4 | 0.2% |
| Jx/εx | 1.0000 | 1.0 | ~1e-15 |

Statistical fluctuations ~0.1–1% are consistent with $\sigma \propto 1/\sqrt{N}$ for N=10000.

## Files

| File | Description |
|------|-------------|
| `make_input.py` | Generate `beam0.json` with PASS Python API |
| `run.py` | Run PASS simulation |
| `analyse.py` | Five verification modules + plots |
| `beam0.json` | Generated PASS input (overwritten each run) |

## How to Run

### Prerequisites

- PASS installed: `pip install -e .` (from project root)
- tfs-pandas: `pip install tfs`

### 1. Generate input

```bash
python make_input.py
```

### 2. Run simulation

```bash
python run.py
```

Output is saved to `output/YYYY_MMDD/HHMM_SS/`.

### 3. Analyze results

```bash
python analyse.py
```

Auto-detects the latest output directory. Prints all five verification results and displays plots via `plt.show()`.
