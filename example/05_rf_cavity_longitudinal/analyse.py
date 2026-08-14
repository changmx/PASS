"""Analyse RF cavity longitudinal tracking results (Example 05).

Reads PASS ParticleMonitor TBT + StatMonitor CSV for one or more cases and
verifies, against the theory in make_input.calc_theory():

  energy_gain    - synchronous particle Ek(n) slope vs (q/A) V sin(phi_s)
  qs_fft         - synchrotron tune Qs from z / dp TBT FFT (Hann + zero pad)
  bucket_scan    - stability of dp = +-{0.5,0.8,1.0,1.2}*dpmax particles
  bucket_plot    - (z, dp) phase space + theory separatrix (Hamiltonian)
  damping        - adiabatic damping: Jx * p0^2 conserved (twiss case only)
  loss           - dp aperture loss: tag 12 lost at turn ~0
  h2_symmetry    - h=2: z = +-C/2 and z = 0 have the same RF phase
  ramping_gain   - Ek(n) vs integral of V(turn) sin(phi_s) (ramping case)
  ramping_clamp  - V clamped after last ramp row (n_turns > n_rows)

  compare        - twiss vs element: first-order drift error vs dp

Turn convention (after RFCavity priority fix in PASS.core.sequence):
  sequence order at s=0: Injection -> RFCavity -> monitors -> ring transport.
  So turn n records the state AFTER kick n (before transport n+1).

Usage:
    python analyse.py
    python analyse.py --case twiss_h1_fixed
    python analyse.py --case all
    python analyse.py --no-plot
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from make_input import (CASES, CASE_PAIRS, calc_theory, CIRCUM, RADIUS,
                        BETA_0, GAMMA_0, ETA_0, E_TOTAL_0, QM_RATIO, M0,
                        GAMMA_T, SIGMA_Z, SIGMA_DP, selected_cases)

SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================================
# I/O
# ============================================================

def find_latest_output(case_name: str):
    """Most recent complete output/<case_name>/YYYY_MMDD/HHMM_SS directory."""
    root = SCRIPT_DIR / "output" / case_name
    if not root.exists():
        return None
    for date_dir in sorted(root.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for time_dir in sorted(date_dir.iterdir(), reverse=True):
            particle_dir = time_dir / "particle"
            if (time_dir.is_dir()
                    and particle_dir.is_dir()
                    and any(particle_dir.glob("*.tfs"))
                    and any(time_dir.glob("*_stat_*.csv"))):
                return time_dir
    return None


def read_pass_tbt(output_dir, max_tag=20):
    """Read PASS ParticleMonitor TFS files -> {tag: {turn,x,px,y,py,z,dp}}."""
    particle_dir = output_dir / "particle"
    data = {}
    for f in sorted(particle_dir.glob("*_beam*_tag*.tfs")):
        tag = int(f.stem.split("_tag")[-1].lstrip("_"))
        if tag > max_tag:
            continue
        import tfs as tfs_lib
        df = tfs_lib.read(str(f))
        data[tag] = {k: df[k].to_numpy()
                     for k in ["turn", "x", "px", "y", "py", "z", "dp"]}
    return data


def read_pass_stat(output_dir):
    """Read PASS StatMonitor CSV -> {column: np.array}."""
    csv_files = list(output_dir.glob("*_stat_*.csv"))
    if not csv_files:
        return None
    with open(csv_files[0], "r") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


# ============================================================
# FFT tune measurement (Hann window + zero padding + parabolic peak)
# ============================================================

def measure_tune(signal, n_pad=65536):
    """Fractional tune from a TBT signal (same method as example 03/04)."""
    n = len(signal)
    if n < 4:
        return 0.0, 0.0
    sig = signal - np.mean(signal)
    win = np.hanning(n)
    padded = np.zeros(max(n, n_pad))
    padded[:n] = sig * win
    spectrum = np.abs(np.fft.rfft(padded))
    freqs = np.fft.rfftfreq(len(padded))
    spectrum[:2] = 0
    peak = np.argmax(spectrum)
    tune = freqs[peak]
    if 0 < peak < len(spectrum) - 1:
        a, b, c = spectrum[peak - 1], spectrum[peak], spectrum[peak + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-30:
            tune = (peak + 0.5 * (a - c) / denom) / len(padded)
    return tune, spectrum[peak]


# ============================================================
# Theory helpers (same formulas as PASS RFCavity / injection)
# ============================================================

def bucket_separatrix(voltage, harmonic, phase, n_z=400):
    """Numerical (z, dp) separatrix: H(z, dp) = H_sep (first-order theory).

    H(z, dp) = 1/2 h w0 eta dp^2
             + w0 (q/A) V / (2 pi beta^2 E) [cos(phi) - cos(phi_s)
                                             + (phi - phi_s) sin(phi_s)]
    phi = phi_s - h z / R,  w0 = 2 pi beta c / C.

    Returns (z_grid, dp_upper, dp_lower) over the bucket interior.
    """
    w0 = 2.0 * math.pi * BETA_0 * 2.99792458e8 / CIRCUM
    coeff = w0 * QM_RATIO * voltage / (2.0 * math.pi * BETA_0**2 * E_TOTAL_0)

    def pot(z):
        phi = phase - harmonic * z / RADIUS
        return coeff * (np.cos(phi) - math.cos(phase)
                        + (phi - phase) * math.sin(phase))

    z_max = RADIUS * (math.pi - 2.0 * phase) / harmonic
    phi_ufp = math.pi - phase
    h_sep = coeff * (math.cos(phi_ufp) - math.cos(phase)
                     + (phi_ufp - phase) * math.sin(phase))

    z = np.linspace(-z_max, z_max, n_z)
    arg = 2.0 * (h_sep - pot(z)) / (harmonic * w0 * ETA_0)
    arg = np.maximum(arg, 0.0)
    dp_sep = np.sqrt(arg)
    return z, dp_sep, -dp_sep


def freeze_turn(arr, tol=0.0):
    """Turn at which a particle stops changing (lost); None if alive to the end.

    After loss the engine keeps the last coordinates, so the series is
    constant from the loss turn onward.
    """
    diff = np.abs(np.diff(arr))
    changed = np.where(diff > tol)[0]
    if len(changed) == 0:
        return 0
    last = changed[-1]
    if last >= len(arr) - 2:
        return None
    return last + 1


# ============================================================
# Verification modules
# ============================================================

def check_energy_gain(case, data, stat, theory, ax=None):
    print("--- energy_gain ---")
    ek = stat["Ek"]                       # eV/u, after kick n
    turn = np.arange(len(ek))
    slope = np.polyfit(turn, ek, 1)[0]    # dE per turn
    rel = abs(slope - theory["dE_syn"]) / theory["dE_syn"]
    print(f"  Ek slope      = {slope:12.6f} eV/u/turn")
    print(f"  theory dE_syn = {theory['dE_syn']:12.6f} eV/u/turn")
    print(f"  relative err  = {rel*100:.4f} %")
    if ax is not None:
        ax.plot(turn, (ek - ek[0]) / 1e3, "b-", label="PASS Ek")
        model = theory["dE_syn"] * turn
        ax.plot(turn, model / 1e3, "r--", label=f"theory {theory['dE_syn']:.1f} eV/u")
        ax.set_xlabel("turn"); ax.set_ylabel(r"$\Delta E_k$ (keV/u)")
        ax.set_title("energy gain"); ax.legend(); ax.grid(alpha=0.3)
    return rel


def qs_theory_series(case, stat, n_pad=None):
    """Adiabatic-average theory Qs: Qs(gamma(n)) averaged over the run.

    At low beta the denominator beta^2 E grows ~ (1+1/gamma^2)/(gamma-1/gamma)
    times faster than E, so the synchrotron tune drifts noticeably over a
    long run.  The measured FFT tune is the average over the run, which is
    compared against both Qs(gamma_0) and <Qs(gamma(n))>.
    """
    ek = stat["Ek"]
    gamma = 1.0 + ek / M0
    beta2 = 1.0 - 1.0 / gamma**2
    eta = 1.0 / GAMMA_T**2 - 1.0 / gamma**2
    qs = np.sqrt(-(QM_RATIO * case["harmonic"] * case["voltage"]
                   * eta * math.cos(case["phase"]))
                 / (2.0 * math.pi * beta2 * gamma * M0))
    return qs


def check_qs_fft(case, data, stat, theory, ax=None):
    print("--- qs_fft ---")
    measurements = []
    for tag, plane in [(2, "z"), (3, "z"), (4, "dp"), (5, "dp")]:
        sig = data[tag][plane]
        qs, amp = measure_tune(sig)
        measurements.append(qs)
        print(f"  tag {tag:2d} {plane:2s}: Qs = {qs:.6e}")
    qs_mean = float(np.mean(measurements))
    rel_init = abs(qs_mean - theory["Qs"]) / theory["Qs"]
    qs_series = qs_theory_series(case, stat)
    qs_avg = float(np.mean(qs_series))
    rel_avg = abs(qs_mean - qs_avg) / qs_avg
    print(f"  Qs mean (FFT)        = {qs_mean:.6e}")
    print(f"  theory Qs(gamma_0)   = {theory['Qs']:.6e}   (rel. {rel_init*100:.3f} %)")
    print(f"  theory <Qs(gamma)>   = {qs_avg:.6e}   (rel. {rel_avg*100:.3f} %)")
    if ax is not None:
        for tag, plane, color in [(2, "z", "b"), (4, "dp", "g")]:
            sig = data[tag][plane]
            n = len(sig)
            padded = np.zeros(65536)
            padded[:n] = (sig - np.mean(sig)) * np.hanning(n)
            spec = np.abs(np.fft.rfft(padded))
            freqs = np.fft.rfftfreq(65536)
            ax.plot(freqs, spec / spec.max(), color, lw=1,
                    label=f"tag {tag} ({plane})")
        ax.axvline(theory["Qs"], color="r", ls="--", lw=1.5,
                   label=f"theory Qs={theory['Qs']:.5f}")
        ax.set_xlim(0, 0.02)
        ax.set_xlabel("tune"); ax.set_ylabel("normalized amplitude")
        ax.set_title("synchrotron tune (FFT)"); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    return rel_avg


def check_bucket_scan(case, data, theory, ax=None):
    print("--- bucket_scan ---")
    dpmax = theory["dpmax"]
    rows = []
    for tag in range(6, 13):
        dp = data[tag]["dp"]
        z = data[tag]["z"]
        ft = freeze_turn(dp)
        alive = ft is None
        rows.append((tag, dp[0] / dpmax, alive,
                     np.max(np.abs(dp)), np.max(np.abs(z)),
                     -1 if alive else ft))
    print(f"  {'tag':>3s} {'dp0/dpmax':>9s} {'alive':>5s} {'max|dp|':>10s} "
          f"{'max|z|(m)':>10s} {'lost_turn':>9s}")
    for tag, frac, alive, md, mz, lt in rows:
        lt_str = "-" if alive else str(lt)
        print(f"  {tag:3d} {frac:+9.3f} {str(alive):>5s} {md:10.3e} {mz:10.2f} "
              f"{lt_str:>9s}")
    return rows


def plot_bucket(case, data, theory, ax=None):
    print("--- bucket_plot ---")
    if ax is None:
        return
    z, dp_up, dp_lo = bucket_separatrix(case["voltage"], case["harmonic"],
                                        case["phase"])
    colors = {2: "C0", 3: "C0", 4: "C1", 5: "C1", 6: "C2", 7: "C2",
              8: "C3", 9: "C3", 10: "C4", 11: "C4", 12: "C5"}
    for tag in range(2, 13):
        zz = data[tag]["z"]
        ddp = data[tag]["dp"]
        ax.plot(zz, ddp, color=colors[tag], lw=0.6, alpha=0.9,
                label=f"tag {tag}" if tag in (4, 12) else None)
    ax.plot(z, dp_up, "k--", lw=1.5, label="separatrix (theory)")
    ax.plot(z, dp_lo, "k--", lw=1.5)
    ax.axhline(+theory["dp_aperture"], color="m", ls=":", lw=1,
               label=f"dp aperture +-{theory['dp_aperture']:.3e}")
    ax.axhline(-theory["dp_aperture"], color="m", ls=":", lw=1)
    ax.set_xlabel(r"$z$ (m)"); ax.set_ylabel(r"$\delta p$")
    ax.set_title("longitudinal phase space + separatrix")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def check_damping(case, data, stat, theory, ax=None):
    print("--- damping (adiabatic, bunch level: sigma_py * p0 conserved) ---")
    ek = stat["Ek"]
    p0 = np.sqrt((ek + M0)**2 - M0**2)          # reference momentum eV/c
    sp = stat["sigmaPy"]                        # y momentum spread (KV dist.)
    inv = sp * p0                               # conserved by px rescale
    rel = np.std(inv) / np.mean(inv)
    print(f"  std(sigma_py*p0)/mean = {rel:.3e}  (statistical noise ~1% "
          f"for N=5000)")
    # Single-particle info: Jx does NOT scale as p0^-2 exactly because the
    # x^2/beta term is unchanged by the kick (only px is rescaled).
    x = data[13]["x"]; px = data[13]["px"]
    jx = x**2 / 10.0 + 10.0 * px**2
    rel_jx = np.std(jx * p0**2) / np.mean(jx * p0**2)
    print(f"  tag 13 Jx*p0^2 rel. var = {rel_jx:.3e} "
          f"(x-term does not scale, expected ~0.5%)")
    print(f"  dE/E over run = {(ek[-1]-ek[0])/ek[0]*100:.4f} %")
    if ax is not None:
        ax.plot(np.arange(len(ek)), inv / inv[0], "b-")
        ax.set_xlabel("turn")
        ax.set_ylabel(r"$\sigma_{py}\,p_0 / (\sigma_{py}\,p_0)_0$")
        ax.set_title(f"adiabatic damping (rel. var. {rel:.1e})")
        ax.grid(alpha=0.3)
    return rel


def check_loss(case, data, stat, theory):
    print("--- loss (dp aperture) ---")
    dp12 = data[12]["dp"]
    ft = freeze_turn(dp12)
    loss = stat.get("lossPercent", np.zeros(len(dp12)))
    print(f"  tag 12 (dp0=1.2*dpmax): lost_turn = {ft}, "
          f"dp aperture = +-{theory['dp_aperture']:.4e}")
    print(f"  max final lossPercent = {np.max(loss):.2f} %")
    return ft


def check_h2_symmetry(case, data, theory):
    print("--- h2_symmetry (RF phase periodic mod C/h) ---")
    for tag, z0 in [(14, +CIRCUM / 2.0), (15, -CIRCUM / 2.0)]:
        dp = data[tag]["dp"]
        z = data[tag]["z"]
        max_dp = np.max(np.abs(dp))
        z_excursion = np.max(np.abs(z - z0))
        print(f"  tag {tag}: z0={z0:+.2f} m, max|dp|={max_dp:.2e}, "
              f"max|z-z0|={z_excursion:.2e} m (bucket half-width "
              f"{CIRCUM/(2.0*case['harmonic']):.2f} m)")
    return max_dp


def check_ramping_gain(case, data, stat, theory, ax=None):
    print("--- ramping_gain ---")
    v0 = case["voltage"]
    slope = case["ramp_slope"]
    n_rows = case["ramp_rows"]
    ek = stat["Ek"]
    n = len(ek)
    v_k = v0 * (1.0 + slope * np.minimum(np.arange(n), n_rows - 1))
    de_k = QM_RATIO * v_k * math.sin(case["phase"])      # eV/u per kick
    model = ek[0] - de_k[0] + np.cumsum(de_k)            # E0 + sum_{k=0..n} dE
    rel = np.max(np.abs(ek - model)) / np.abs(ek[0] - ek[-1] + 1e-30)
    print(f"  max|Ek - model| / |dE_total| = {rel:.3e}")
    print(f"  Ek(0)={ek[0]:.6e}, Ek({n-1})={ek[-1]:.6e} eV/u, "
          f"model dE_total={model[-1]-model[0]:.6e} eV/u")
    if ax is not None:
        ax.plot(np.arange(n), ek / 1e3, "b-", label="PASS Ek")
        ax.plot(np.arange(n), model / 1e3, "r--", label="ramp model")
        ax.set_xlabel("turn"); ax.set_ylabel(r"$E_k$ (keV/u)")
        ax.set_title("ramping energy gain"); ax.legend(); ax.grid(alpha=0.3)
    return rel


def check_ramping_clamp(case, data, stat, theory):
    print("--- ramping_clamp (V frozen after last row) ---")
    n_rows = case["ramp_rows"]
    ek = stat["Ek"]
    n = len(ek)
    if n <= n_rows:
        print(f"  n_turns={n} <= n_rows={n_rows}: clamp not exercised")
        return None
    slope_last = np.polyfit(np.arange(n_rows, n), ek[n_rows:], 1)[0]
    de_last = (QM_RATIO * case["voltage"]
               * (1.0 + case["ramp_slope"] * (n_rows - 1))
               * math.sin(case["phase"]))
    rel = abs(slope_last - de_last) / de_last
    print(f"  slope after row {n_rows-1} = {slope_last:.6f} eV/u/turn")
    print(f"  expected frozen dE       = {de_last:.6f} eV/u/turn")
    print(f"  relative err             = {rel*100:.3f} %")
    return rel


def compare_twiss_element(pair, ax=None):
    """Twiss vs element: first-order drift approximation error.

    Both cases now use the same fodo ring (gamma_t = 3.3746): the twiss
    case applies the first-order longitudinal drift z -= eta*C*dp, the
    element case the exact per-element mapping (momentum compaction
    emerges from the dipole geometry).  Their Qs must agree to ~1e-3
    (first-order error), and the z-trajectory difference grows with dp.
    """
    print("--- compare (twiss vs element first-order drift) ---")
    dirs = {}
    for name in pair:
        out = find_latest_output(name)
        if out is None:
            print(f"  no output for {name}")
            return
        dirs[name] = read_pass_tbt(out)

    # (a) Qs consistency (same gamma_t in both implementations)
    qs_tw = np.mean([measure_tune(dirs[pair[0]][t]["z"])[0] for t in (2, 3)])
    qs_el = np.mean([measure_tune(dirs[pair[1]][t]["z"])[0] for t in (2, 3)])
    print(f"  Qs(twiss)={qs_tw:.6e}, Qs(element)={qs_el:.6e}, "
          f"ratio={qs_el/qs_tw:.6f} (theory 1.0, same gamma_t={GAMMA_T:.4f})")

    # (b) trajectory difference vs dp (first-order drift error)
    print(f"  {'tag':>3s} {'dp0':>10s} {'max|dz| (m)':>12s} {'max|ddp|':>10s}")
    rows = []
    for tag in range(4, 12):
        zt = dirs[pair[0]][tag]["z"]
        ze = dirs[pair[1]][tag]["z"]
        dpt = dirs[pair[0]][tag]["dp"]
        dpe = dirs[pair[1]][tag]["dp"]
        dz = np.max(np.abs(zt - ze))
        ddp = np.max(np.abs(dpt - dpe))
        dp0 = dpt[0]
        rows.append((tag, dp0, dz, ddp))
        print(f"  {tag:3d} {dp0:+10.3e} {dz:12.4f} {ddp:10.3e}")
    if ax is not None:
        dp0s = [r[1] for r in rows]
        dzs = [r[2] for r in rows]
        ax.semilogy(dp0s, dzs, "bo-")
        ax.set_xlabel(r"initial $\delta p$"); ax.set_ylabel(r"$\max|\Delta z|$ (m)")
        ax.set_title("twiss vs element drift difference")
        ax.grid(alpha=0.3, which="both")
    return rows


# ============================================================
# Case dispatcher
# ============================================================

def analyse_case(name, output_dir=None, is_plot=True):
    case = CASES[name]
    theory = calc_theory(case["voltage"], case["harmonic"], case["phase"])
    out = Path(output_dir) if output_dir else find_latest_output(name)
    if out is None:
        print(f"[{name}] no output directory found (run run.py first)")
        return

    data = read_pass_tbt(out)
    stat = read_pass_stat(out)
    print(f"\n{'='*60}\n[{name}]  {out}\n{'='*60}")

    if is_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Example 05 — {name}")

    for i, check in enumerate(case["checks"]):
        ax = None
        if is_plot:
            ax = axes[i // 3, i % 3]
        if check == "energy_gain":
            check_energy_gain(case, data, stat, theory, ax)
        elif check == "qs_fft":
            check_qs_fft(case, data, stat, theory, ax)
        elif check == "bucket_scan":
            check_bucket_scan(case, data, theory)
        elif check == "bucket_plot":
            plot_bucket(case, data, theory, ax)
        elif check == "damping":
            check_damping(case, data, stat, theory, ax)
        elif check == "loss":
            check_loss(case, data, stat, theory)
        elif check == "h2_symmetry":
            check_h2_symmetry(case, data, theory)
        elif check == "ramping_gain":
            check_ramping_gain(case, data, stat, theory, ax)
        elif check == "ramping_clamp":
            check_ramping_clamp(case, data, stat, theory)

    if is_plot:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse Example 05 results.")
    parser.add_argument(
        "--case",
        choices=["all", *CASES],
        default="all",
        help="Case to analyse (default: all).",
    )
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable interactive plots.")
    args = parser.parse_args()

    case_names = selected_cases(args.case)
    for case_name in case_names:
        analyse_case(case_name, is_plot=not args.no_plot)

    for pair in CASE_PAIRS:
        if all(name in case_names for name in pair):
            compare_twiss_element(pair)
