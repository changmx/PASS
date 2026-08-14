# Example 05 - RF Cavity Longitudinal Dynamics Test

## Overview

This example verifies the physical correctness and functionality of PASS's RF cavity element `RFCavity` (`PASS/commands/element/rfcavity.py`). The beam is a low-energy heavy ion, 238U35+ at 17 MeV/u, and the ring optics use the FODO lattice from examples 03/04 (`fodo.tfs`, with headers C = 234.4 m and gamma_t = 3.3746).

Four cases are covered. They are driven from a single `CASES` source in `make_input.py`, and `analyse.py` imports that source to avoid duplicate parameter definitions and theory values.

| case | lattice | RF mode | harmonic | focus |
|------|---------|---------|----------|-------|
| `twiss_h1_fixed` | single Twiss point one-turn map (`longitudinal_transfer="drift"`) | fixed scalar | h=1 | energy gain / Qs / bucket / damping / loss |
| `twiss_h2_fixed` | same | fixed scalar | h=2 | RF phase symmetry with period C/h |
| `twiss_h1_ramping` | same | TFS waveform file (one row per turn) | h=1 | turn-by-turn parameter loading + clamping |
| `element_h1_fixed` | real FODO ring (`fodo.tfs` elements) | fixed scalar | h=1 | exact element-by-element longitudinal transport / momentum compaction emergence |

### Optics and transition

For the FODO ring with gamma_t = 3.3746, the 17 MeV/u ion has gamma = 1.01825, far below gamma_t. The slip factor is eta = 1/gamma_t^2 - 1/gamma^2 = -0.8767, so the machine is below transition and the stable accelerating phase satisfies 0 < phi_s < pi/2. After 2048 turns, gamma only rises to about 1.019, still far from transition. Crossing would require about 2.2 GeV/u and a phase shift into (pi/2, pi), which is a different test scenario.

### K-value normalization

The K1L/K0L values in `fodo.tfs` are normalized strengths, i.e. divided by reference magnetic rigidity Brho. PASS and MADX use them directly as energy-independent optics parameters. In PASS's quadrupole implementation, `Delta p_x = -K1L * x` with no extra scaling. This matches the standard synchrotron operating mode where magnet gradients scale with particle rigidity. Therefore the K values calibrated for 2 TeV protons can be used directly for the 17 MeV/u ion, and the transverse optics remain unchanged. The element case uses the real `fodo.tfs` elements, and its longitudinal momentum compaction (gamma_t=3.3746) should emerge naturally from the dipole geometry; this is the dedicated PASS dipole longitudinal-map check.

## Physics Model

### RF kick

Each turn applies a longitudinal kick at s=0:

$$dE = (q/A)\,V\,\sin\left(\phi_s + \phi_{\text{off}} - \frac{2\pi h}{C}z_{\text{lab}}\right),\qquad z_{\text{lab}}=z_{\text{rel}}+z_{\text{center}}$$

- PASS stores particle coordinates relative to the bunch center as `z_rel`; `z_center = h_id * C / h_group` is the fixed bunch label in the ring. The RF phase is therefore always evaluated with the laboratory coordinate `z_lab`, not with any special even-harmonic correction.
- This example has only one bunch, so `z_center = 0` and the synchronous particle is `z_rel = 0`, independent of harmonic parity. Particles separated by one RF period C/h receive the same kick; for h=2, `z = +-C/2` is in phase with `z = 0`.
- When the cavity harmonic matches the bunch grouping number, or is an integer multiple of it, all bunch centers get the same reference gain. Otherwise the code still computes phase and reference energy per bunch using each bunch's `z_center`, so different bunches can receive different gains.
- Energy -> momentum -> delta conversion is fully relativistic, with no linearization. Each bunch's reference frame is updated by its own center particle `dE_ref`, and transverse momenta are rescaled as `p_x,y <- p_x,y * (p0_old / p0_new)` for adiabatic damping.

### One-turn map and synchrotron motion

Twiss case: `Injection -> RFCavity(s=0) -> monitors(s=0) -> Twiss(s=C, s_prev=0)`. The Twiss point provides full-ring transport, and the longitudinal motion is a first-order drift `z <- z - eta * C * delta p` (`twiss.py` "drift" mode, using the real gamma_t and gamma).

Element case: real `fodo.tfs` elements are tracked one by one, and the longitudinal motion comes from the exact drift/dipole formulas and dipole geometry. The momentum compaction (gamma_t=3.3746) emerges from the dipole mapping and should match the first-order eta from the Twiss case.

The small-amplitude linearized one-turn map gives the synchrotron tune and bucket parameters:

$$Q_s = \sqrt{\frac{-(q/A)\,h\,V\,\eta\,\cos\phi_s}{2\pi\beta^2 E}},\qquad
\Delta p_{\max} = \sqrt{\frac{-(q/A)\,V\big[2\cos\phi_s - (\pi-2\phi_s)\sin\phi_s\big]}{\pi\beta^2 E\, h\, \eta}},\qquad
z_{\max} = \frac{R(\pi-2\phi_s)}{h}$$

The separatrix is obtained numerically from the longitudinal Hamiltonian contour (`analyse.bucket_separatrix()`).

## Beam and Parameters

| parameter | value | note |
|------|-----|------|
| ion | 238U35+ (92p, 146n, q=35) | q/A = 0.14706 |
| kinetic energy | 17 MeV/u | gamma=1.01825, beta=0.18847 |
| circumference C | 234.4 m (`fodo.tfs`) | R=37.31 m |
| gamma_t | 3.3746 (`fodo.tfs`) | eta = -0.87667 |
| cavity voltage V | 20 kV | |
| synchronous phase phi_s | 0.1 rad | eta < 0 -> stable accelerating phase 0 < phi_s < pi/2 |
| turns | 2048 (ramping: 200) | |

**Why 2048 turns instead of 1024?** Qs is about 3.5e-3, so the synchrotron period is about 287 turns. With only 1024 turns, the FFT sees just 3.6 periods and the frequency resolution is poor. With 2048 turns, the tune resolution improves to about 4.9e-4, and zero-padding plus parabolic interpolation reaches about 1e-5. In addition, the bucket-edge particle (tag 12) needs about 500 turns to slip into loss, so 1024 turns leaves too little margin.

**Theory values** (`make_input.calc_theory()` computes these automatically):

| quantity | h=1 | h=2 |
|----|-----|-----|
| dE_syn | 293.628 eV/u/turn | 293.628 eV/u/turn |
| Q_s | 3.4811e-3 (period 287 turns) | 4.9230e-3 (period 203 turns) |
| Delta p_max | 7.332e-3 | 5.185e-3 |
| z_max | 109.74 m (< C/2 = 117.2 m) | 54.87 m |

**Distribution particles** (5000, KV transverse / Gaussian longitudinal): sigma_z = 5 m, sigma_dp = 1e-3. The longitudinal margin is 22x in z and 7.3x in dp, so dp is the limiting factor. The matched bunch length is about 9.4 m; this test intentionally uses an under-matched distribution dominated by dp spread.

## Test Particles

The reference synchronous position is `z_rel = 0` (and `z_center = 0` for this example). There are 13+2 tagged particles:

| tag | coordinates (relative to z_sync) | purpose |
|-----|---------------------|------|
| 1 | z=0, dp=0 | synchronous particle -> energy gain / reference tracking |
| 2-3 | z=+-3 m | Qs from z oscillation |
| 4-5 | dp=+-1e-3 | Qs from dp oscillation |
| 6-7 | dp=+-0.5*Delta p_max | bucket scan |
| 8-9 | dp=+-0.8*Delta p_max | same |
| 10-11 | dp=+-1.0*Delta p_max | boundary particles |
| 12 | dp=+1.2*Delta p_max | outside bucket -> dp aperture loss |
| 13 | x=3 mm, px=1e-4 | adiabatic damping (bunch-level check) |
| 14-15 | z=+-C/2 (h=2 only) | same RF phase separated by one period |

The dp aperture is `+-1.08 * Delta p_max` (computed automatically per case). Tag 12 is clipped on the first turn, while tags 10/11 probe the bucket edge.

## Verification Results

### 1. Energy Gain (all 4 cases)

| quantity | measured | theory | error |
|----|------|------|------|
| dE/dn (Ek slope) | 293.627696 eV/u/turn | 293.627696 eV/u/turn | 0.0000 % |

### 2. Synchrotron Tune Qs (FFT, Hann window + zero padding 65536 + parabolic interpolation)

| case | measured Qs | theory Qs(gamma_0) | rel. | theory <Qs(gamma)> | rel. |
|------|---------|-------------|------|--------------|------|
| twiss_h1 | 3.4422e-3 | 3.4811e-3 | 1.12 % | 3.4499e-3 | 0.22 % |
| twiss_h2 | 4.8553e-3 | 4.9230e-3 | 1.38 % | 4.8789e-3 | 0.48 % |
| element_h1 | 3.4418e-3 | 3.4811e-3 | 1.13 % | 3.4499e-3 | 0.24 % |

**Adiabatic drift:** after 2048 turns, the reference energy rises by about 0.60 MeV/u, but at low beta the denominator beta^2 E is highly sensitive to gamma, so Qs drifts downward by about 1.7%. The FFT measures the full-run average, which agrees with the adiabatic-average theory <Qs(gamma)> to within 0.2-0.5%.

**Momentum compaction check in the element case:** the real FODO element-by-element simulation (exact drifts + dipole geometry) matches the Twiss first-order simulation in Qs to 0.012%, confirming that PASS's dipole longitudinal map and momentum compaction (gamma_t=3.3746) are correct.

### 3. Bucket Edge Scan (dp aperture +-1.08*Delta p_max)

| dp0/Delta p_max | twiss_h1 | twiss_h2 | element_h1 |
|------------|----------|----------|------------|
| +-0.5 | stable | stable | stable |
| +-0.8 | stable | stable | stable |
| +1.0 | phase slip loss | phase slip loss | stable |
| -1.0 | stable | stable | stable |
| +1.2 | turn 0 loss | turn 0 loss | turn 0 loss |

Tag 12 is clipped by the dp aperture on the first turn (`lost_turn=0`). The boundary particles at `+-1.0 Delta p_max` are stable in the exact element model, but the +1.0 case slips out in the first-order Twiss model. That mismatch is a first-order approximation effect, not a physics error. The asymmetry between positive and negative dp comes from the bucket asymmetry when phi_s != 0.

### 4. h=2 RF Phase Symmetry

| tag | initial z | max|dp| | max|z-z0| |
|-----|--------|-----------|-------------|
| 14 | +117.2 m (= +C/2) | 6.2e-15 | 0.0 m |
| 15 | -117.2 m (= -C/2) | 6.2e-15 | 0.0 m |

Both particles are one RF period C/h = C/2 away from z=0, so they receive the same kick and remain phase-locked. This does not depend on coordinate folding or any special even-harmonic correction.

### 5. Energy Ramp (rf data file, V(n) = 20 kV * (1 + 0.02n), 50 rows)

| check | result |
|------|------|
| E(n) vs sum of V(k) sin(phi_s) | max relative error 1.5e-11 |
| slope after row 50 vs expected frozen slope | 0.000 % |

### 6. Twiss vs Element: First-Order Drift Error

Both cases use the same FODO ring (gamma_t=3.3746): the Twiss case uses the first-order longitudinal drift `z <- z - eta * C * delta p`, while the element case uses exact element-by-element mapping and dipole geometry.

**(a) Qs consistency**

| quantity | value |
|----|-----|
| Qs(twiss) measured | 3.4484e-3 |
| Qs(element) measured | 3.4480e-3 |
| ratio | 0.999879 (theory 1.0, same gamma_t) |

The first-order drift and exact element transport agree in Qs to 0.012%.

**(b) Trajectory difference vs dp**

| dp0 | max|Delta z| |
|-----|----------|
| +-1e-3 | 0.05 m |
| +-3.7e-3 | 0.10-0.36 m |
| +-5.9e-3 | 2.4-2.8 m |
| +-7.3e-3 | 172-218 m (branching at the bucket edge: tag 10 is lost in the Twiss approximation but remains stable in the exact element case) |

The first-order error grows with |delta p|. For ordinary beam conditions (delta p <= 1e-3), the error is about 5 cm, roughly 0.5% of the z oscillation amplitude. Near the bucket edge, the first-order approximation breaks down.

### 7. Adiabatic Damping (bunch level)

| quantity | relative change |
|----|----------|
| sigma_py * p0 (conserved quantity) | 7.2e-3 (about the statistical noise level for 5000 particles) |
| tag 13 Jx * p0^2 (single particle) | about 5e-3 (the x^2/beta term is not scaled by the kick, which is expected) |

## Three Important Physical Notes

| note | nature | severity |
|--------|------|--------|
| multi-bunch operation with non-integer RF harmonic ratios | allowed configuration: the phase is evaluated with `z_lab = z_rel + z_center`; if the cavity harmonic is not an integer multiple of the bunch grouping number, different bunch centers receive different reference gains | depends on the machine operating scheme. The code does not forbid it, but the user should verify the resulting longitudinal working point |
| low-beta Qs adiabatic drift (about -1.7% over 2048 turns) | real physical effect: beta^2 E is strongly amplified by 1/beta^2 at low gamma | low. Compare against the adiabatic-average theory <Qs(gamma)>; shortening the run or increasing energy reduces it |
| first-order Twiss drift failure near the bucket edge | model approximation: the first-order `-eta*C*delta p` drift omits higher-order terms | medium-low. For normal beam conditions (delta p <= 1e-3) the effect is small; use the element mode for large-dp or edge studies |

**RFCavity conclusion:** In the normal physics range covered by this example, energy gain, synchrotron motion, bucket structure, multi-bunch phase handling based on `z_lab`, ramp-file reading, dp aperture, and adiabatic damping are all validated by simulation. Two limits remain: the GPU backend is not implemented (`execute_gpu` raises `NotImplementedError`), and if an unphysical over-decelerating kick drives the total energy below rest energy, the current implementation does not yet mark that particle as lost automatically.

## File Layout

```text
05_rf_cavity_longitudinal/
├── make_input.py   # single source of truth: CASES + calc_theory() + build_case()
├── run.py          # --case/--beam0 -> PASS.main
├── analyse.py      # verification modules + A/B comparison (imports make_input)
├── fodo.madx/.seq/.ps/.tfs  # FODO lattice from examples 03/04 (provides C and gamma_t)
├── rf_ramp.tfs     # ramp waveform (generated automatically by make_input)
├── beam0_<case>.json
└── output/<case>/YYYY_MMDD/HHMM_SS/
```

## Usage

```bash
cd example/05_rf_cavity_longitudinal
python make_input.py    # generate 4 beam0_<case>.json files + rf_ramp.tfs
python run.py           # run the 4 cases in sequence
python analyse.py       # print all verification results + interactive plots
```

You can also generate or run a single case:

```bash
python make_input.py --case twiss_h1_fixed
python run.py --case twiss_h1_fixed
python run.py --case all
python run.py --beam0 beam0_twiss_h1_fixed.json
```

## Notes

1. **CPU backend required**: `RFCavity.execute_gpu` is not implemented.
2. **Sequence ordering fix**: this example depends on the shared `COMMAND_PRIORITY` in `PASS/commands/__init__.py`, where `"RFCavity": 300`. Before that fix, RFCavity fell into `Other=999` and was sorted after monitors at the same s, which caused the monitors to record the pre-kick state. The corrected order is `Injection -> RFCavity -> monitors -> ring transport`, so turn n records the state after the n-th kick.
3. **Turn convention**: turn 0 includes the first kick.
4. **Longitudinal coordinate**: manually specified particle z is `z_rel` relative to the bunch center; the code adds `z_center` automatically when computing RF phase.
5. **K-value normalization**: K1L in `fodo.tfs` is normalized strength and does not depend on beam energy. The element case can therefore use the real FODO elements directly without rigidity scaling.
6. **FFT Qs measurement**: Qs is very small, so zero padding to 65536 plus parabolic interpolation is needed, and the result should be compared against the adiabatic-average theory <Qs(gamma)> rather than the initial value.
