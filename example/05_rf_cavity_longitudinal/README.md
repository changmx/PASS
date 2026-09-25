# Example 05 - RF Cavity Longitudinal Dynamics Test

Use `generate_input.py`, `run_simulation.py`, and `analyze_results.py` as the
workflow entry points for input generation, tracking, and result analysis.

## Overview

This example demonstrates longitudinal tracking with PASS's RF cavity element `RFCavity` (`PASS/commands/element/rfcavity.py`). The beam is a low-energy heavy ion, 238U35+ at 17 MeV/u, and the ring optics use the FODO lattice from examples 03/04 (`fodo.tfs`, with headers C = 234.4 m and gamma_t = 3.3746).

Five cases are covered. They are driven from a single `CASES` source in `generate_input.py`, and `analyze_results.py` imports that source to avoid duplicate parameter definitions and theory values.

| case | lattice | RF mode | harmonic | focus |
|------|---------|---------|----------|-------|
| `twiss_h1_fixed` | single Twiss point one-turn map (`longitudinal_transfer="drift"`) | physical-time program with constant design voltage | h=1 | energy gain / Qs / bucket / damping / loss |
| `twiss_h2_fixed` | same | physical-time program with constant design voltage | h=2 | RF phase symmetry with period C/h |
| `twiss_h1_ramping` | same | physical-time TFS with design passage samples | h=1 | voltage ramp + held final voltage |
| `element_h1_fixed` | real FODO ring (`fodo.tfs` elements) | physical-time program with constant design voltage | h=1 | element-by-element longitudinal transport and momentum compaction |

The additional `twiss_h1_waveform` case prescribes a sinusoidally modulated design voltage. Every case generates `rf_physical_h<h>_<lattice>_<mode>.tfs`.

`generate_input.py` fixes the Injection random seed to `2026`, so every case uses a
reproducible generated particle distribution.

## Run the example

Install PASS and run commands from this example directory. A fresh clone needs
`fodo.tfs`; generate it with the supplied `fodo.madx` using a MAD-X executable
(`madx fodo.madx`). The table supplies the lattice and reference optical parameters.

```bash
cd example/05_rf_cavity_longitudinal
python generate_input.py    # generate 5 JSON inputs and physical-time RF tables
python run_simulation.py           # run the 5 cases serially
python analyze_results.py       # print all verification results + interactive plots
```

You can also generate or run a single case:

```bash
python generate_input.py --case twiss_h1_fixed
python run_simulation.py --case twiss_h1_fixed
python run_simulation.py --case all
python run_simulation.py --beam0 beam0_twiss_h1_fixed.json
```


### Optics and transition

For the FODO ring with gamma_t = 3.3746, the 17 MeV/u ion has gamma = 1.01825, far below gamma_t. The slip factor is eta = 1/gamma_t^2 - 1/gamma^2 = -0.8767, so the machine is below transition and the stable accelerating phase satisfies 0 < phi_s < pi/2. After 2048 turns, gamma only rises to about 1.019, still far from transition. Crossing would require about 2.2 GeV/u and a phase shift into (pi/2, pi), which is a different test scenario.

### Magnet normalization

`K1L` and `K0L` in the imported optics are strengths normalized by reference
magnetic rigidity. Keeping them fixed while changing reference energy describes
magnetic fields scaled with that rigidity. It does not describe fixed physical
magnet fields. The element case uses the geometry and normalized strengths from
`fodo.tfs`; longitudinal transport includes the chosen element maps and their
integration approximations.

## Physics model

### RF kick

Each turn samples a common physical waveform at the actual passage time:

$$t_i=T_b-\frac{z_i}{\beta_b c},\qquad
\Delta E_i=\frac{Z}{A}V(t_i)\sin\left(2\pi\int_0^{t_i} f(u)\,du+\phi(t_i)\right).$$

`z` is a continuous time difference expressed in metres. A thin cavity keeps `T_b` and every live particle's time unchanged while scaling `z` by `beta_new/beta_old`. It updates reference and particle energies using the same waveform, converts energy to momentum exactly, and preserves mechanical transverse momentum by rebasing `px` and `py`. No arrival-correction arrays are stored. `z_center` is nominal grouping metadata.

The input generator explicitly constructs a synchronous design trajectory from the requested voltages and passage phases. The TFS columns are `TIME, VOLTAGE, FREQUENCY, PHASE` in seconds, volts, Hz and radians. Frequency is integrated; PHASE is an unwrapped additive modulation. Samples are linearly interpolated with held endpoints. Tracked bunches do not reset their phase to the design phase each turn. For a time-varying waveform, particles separated by the instantaneous RF wavelength need not receive exactly identical kicks: they sample different physical times.

RF components are supplied through `RFCavityItem(components=[dict(program_file=...)])`. For a fixed hardware frequency, use an inline `frequency` value instead of this synchronous design generator. A component's `harmonic` multiplies a prescribed shared reference-clock frequency, which is constant at its initial value unless an explicit clock program is provided. Grouping harmonics impose no divisibility restriction.

CPU and CUDA implement the same thin-kick model. Effective voltage excludes additional finite-gap and transverse RF focusing models. The exact kick does not remove the separate approximations in a Twiss map or quasi-static space charge. User-controlled saved z slice intervals, widths and memberships remain unchanged across RF; only an explicit Slicer updates them.

### One-turn map and synchrotron motion

Twiss case: `Injection -> RFCavity(s=0) -> monitors(s=0) -> Twiss(s=C, s_prev=0)`. The Twiss point provides full-ring transport, and the longitudinal motion is a first-order drift `z <- z - eta * C * delta p` (`twiss.py` "drift" mode, using the real gamma_t and gamma).

Element case: real `fodo.tfs` elements are tracked one by one, and the longitudinal motion comes from the drift/dipole maps and dipole geometry. The momentum compaction (gamma_t=3.3746) emerges from the dipole mapping and should match the first-order eta from the Twiss case.

The small-amplitude linearized one-turn map gives the synchrotron tune and bucket parameters:

$$Q_s = \sqrt{\frac{-(q/A)\,h\,V\,\eta\,\cos\phi_s}{2\pi\beta^2 E}},\qquad
\Delta p_{\max} = \sqrt{\frac{-(q/A)\,V\big[2\cos\phi_s - (\pi-2\phi_s)\sin\phi_s\big]}{\pi\beta^2 E\, h\, \eta}},\qquad
z_{\max} = \frac{R(\pi-2\phi_s)}{h}$$

The separatrix is obtained numerically from the longitudinal Hamiltonian contour (`analyze_results.bucket_separatrix()`).

## Beam and parameters

| parameter | value | note |
|------|-----|------|
| ion | 238U35+ (92p, 146n, q=35) | q/A = 0.14706 |
| kinetic energy | 17 MeV/u | gamma=1.01825, beta=0.18847 |
| circumference C | 234.4 m (`fodo.tfs`) | R=37.31 m |
| gamma_t | 3.3746 (`fodo.tfs`) | eta = -0.87667 |
| cavity voltage V | 20 kV | |
| synchronous phase phi_s | 0.1 rad | eta < 0 -> stable accelerating phase 0 < phi_s < pi/2 |
| turns | 2048 (ramping: 200) | |

The fixed-voltage cases use 2048 turns to cover several synchrotron periods. For an initial Qs of about 3.5e-3, one period is about 287 turns. The unpadded FFT spacing is about 4.9e-4 cycles per turn; interpolation improves peak estimation without removing finite-record or time-dependent tune effects.

**Theory values** (`generate_input.calc_theory()` computes these automatically):

| quantity | h=1 | h=2 |
|----|-----|-----|
| dE_syn | 293.628 eV/u/turn | 293.628 eV/u/turn |
| Q_s | 3.4811e-3 (period 287 turns) | 4.9230e-3 (period 203 turns) |
| Delta p_max | 7.332e-3 | 5.185e-3 |
| z_max | 109.74 m (< C/2 = 117.2 m) | 54.87 m |

**Distribution particles** (5000, KV transverse / Gaussian longitudinal): sigma_z = 5 m, sigma_dp = 1e-3. The longitudinal margin is 22x in z and 7.3x in dp, so dp is the limiting factor. The matched bunch length is about 9.4 m; this test intentionally uses an under-matched distribution dominated by dp spread.

## Prescribed particles

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

## Interpreting the diagnostics

Energy-gain, tune, bucket, and phase-symmetry plots are computed from the current
run. The small-amplitude formulas describe a locally stationary RF system.
Acceleration changes the reference energy and synchrotron tune; a finite-record
FFT therefore need not return the tune evaluated at the initial energy.
A time-dependent RF program has no exact stationary separatrix, and particles
separated by an instantaneous RF wavelength can sample different waveform values.

The first-order longitudinal Twiss drift omits higher-order momentum and path-length
terms. Examine convergence and model sensitivity near the bucket boundary.
The bucket scan and loss aperture are separate: a particle can be removed by the
configured momentum aperture independently of its longitudinal trapping condition.

Bunch grouping and RF harmonics are independent. Each bunch samples the common
waveform at its physical arrival times, so bunches can receive different gains.
Use the saved reference state when reconstructing those times.

## Files

```text
05_rf_cavity_longitudinal/
├── generate_input.py   # single source of truth: CASES + calc_theory() + build_case()
├── run_simulation.py          # --case/--beam0 -> PASS.main
├── analyze_results.py      # verification modules + A/B comparison (imports generate_input)
├── fodo.madx/.seq/.ps/.tfs  # FODO lattice from examples 03/04 (provides C and gamma_t)
├── rf_physical_h<h>_<lattice>_<mode>.tfs  # physical-time programs
├── beam0_<case>.json
└── output/<case>/YYYY_MMDD/HHMM_SS/
```

## Notes

1. **Backend**: generated inputs default to CPU; the current RFCavity also supports CUDA.
2. **Sequence:** `Injection -> RFCavity -> monitors -> ring transport`; turn n records the state after the n-th kick.
3. **Turn convention**: turn 0 includes the first kick.
4. **Longitudinal coordinate**: manually specified z is `beta*c*(T-t)` relative to its source reference. Use the reference time and beta saved with each monitor row; no nominal-center offset enters the RF phase.
5. **K-value normalization**: Fixed normalized strengths represent magnets scaled with reference rigidity; see the normalization section above.
6. **FFT Qs measurement**: Qs is very small, so the analyzer uses zero padding and parabolic interpolation. Interpret the fitted peak with the evolving reference energy and the finite tracking interval.

RF tables and monitor metadata are written with sufficient significant digits to retain float64 timing. The design generator uses the same rest-mass constant as tracking. The approximate tune, bucket and h=2 symmetry plots in `analyze_results.py` are diagnostics; accelerating, time-varying programs do not have an exact stationary separatrix or exact instantaneous-wavelength symmetry.

ParticleMonitor reference output is disabled by default. To save the reference
state needed to reconstruct physical arrival times and energies, enable it before
tracking:

```bash
python generate_input.py --case twiss_h1_fixed --include-reference
python run_simulation.py --case twiss_h1_fixed
```

This adds `referenceTime`, `referenceBeta`, and `referenceMomentum` to each row.
Enabling the option after a run cannot recover an unrecorded reference history.

Diagnostic tables default to gzip-1 + shuffle HDF5 (`.h5`). The analysis scripts
accept both HDF5 and legacy TFS output; set `output_format="tfs"` on the
monitor (or initial-distribution `BunchConfig`) to request TFS explicitly.
Set `output_format="hdf5"` to write uncompressed HDF5; the default
`output_format="hdf5-gzip1"` enables gzip level 1 and shuffle. Both use `.h5` files.
StatMonitor also writes every recorded row to CSV in batches of 100 turns
by default, configurable with `write_interval_turns`. Slicer slice summaries
remain TFS/CSV. See [table output formats](../../docs/source/en/monitor/table_output.rst).
