"""Analyse twiss-by-twiss tracking results.

Reads PASS ParticleMonitor TBT data and StatMonitor CSV, then verifies:

    1. Tune measurement       — FFT of single-particle TBT → Qx, Qy
    2. Courant-Snyder invariant — Jx, Jy should be constant per particle
    3. Chromaticity           — tune shift vs dp → DQx, DQy
    4. Dispersion             — Dx, Dpx from ±dp particle pairs
    5. Beam statistics        — β, α, ε from 10000 distribution particles

Expected values are read from the MADX TFS files (fodo.tfs headers + first row).
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tfs as tfs_lib

# ============================================================
# Read expected values from TFS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
TFS_FILE = str(SCRIPT_DIR / "fodo.tfs")
TFS_NATURAL = str(SCRIPT_DIR / "fodo_natural.tfs")

_tfs = tfs_lib.read(TFS_FILE)
_nat = tfs_lib.read(TFS_NATURAL)

# Tunes (fractional)
QX_EXPECTED = _tfs.headers["Q1"] % 1
QY_EXPECTED = _tfs.headers["Q2"] % 1

# Chromaticities
# Note: MATCH,CHROM targets DQ1=DQ2=-1.0, but TWISS,CHROM reports different
# values due to MADX's internal algorithm differences. We compare against
# the MATCH target (the design intent), not the TWISS,CHROM header.
DQX_NATURAL = _nat.headers["DQ1"]
DQY_NATURAL = _nat.headers["DQ2"]
DQX_CORRECTED = _tfs.headers["DQ1"]
DQY_CORRECTED = _tfs.headers["DQ2"]

# Twiss at s=0 (first row)
_row0 = _tfs.iloc[0]
ALPHA_X = _row0["ALFX"]
ALPHA_Y = _row0["ALFY"]
BETA_X = _row0["BETX"]
BETA_Y = _row0["BETY"]
DX_EXPECTED = _row0["DX"]
DPX_EXPECTED = _row0["DPX"]
CIRCUM = _tfs.headers["LENGTH"]

# Distribution parameters (match make_input.py)
EMIT_X = 200e-6
EMIT_Y = 100e-6

# Test particle definitions (match make_input.py)
TEST_PARTICLES = [
    [2e-3, 0, 0, 0, 0, 0],          # tag 1: Qx
    [0, 0, 2e-3, 0, 0, 0],          # tag 2: Qy
    [1e-3, 0, 1e-3, 0, 0, +1e-5],   # tag 3
    [1e-3, 0, 1e-3, 0, 0, -1e-5],   # tag 4
    [1e-3, 0, 1e-3, 0, 0, +5e-5],   # tag 5
    [1e-3, 0, 1e-3, 0, 0, -5e-5],   # tag 6
    [1e-3, 0, 1e-3, 0, 0, +1e-4],   # tag 7
    [1e-3, 0, 1e-3, 0, 0, -1e-4],   # tag 8
    [0, 0, 0, 0, 0.1, 0],           # tag 9
    [0, 0, 0, 0, 0, 0],             # tag 10
    [5e-3, 0, 0, 0, 0, 0],          # tag 11
    [0, 0, 5e-3, 0, 0, 0],          # tag 12
]

# Chromaticity pairs: (tag_pos, tag_neg, dp_value)
CHROM_PAIRS = [
    (3, 4, 1e-5),
    (5, 6, 5e-5),
    (7, 8, 1e-4),
]

TAG_INFO = {
    1: "Qx (x=2mm)",
    2: "Qy (y=2mm)",
    3: "chrom +1e-5",
    4: "chrom -1e-5",
    5: "chrom +5e-5",
    6: "chrom -5e-5",
    7: "chrom +1e-4",
    8: "chrom -1e-4",
    9: "longitudinal",
    10: "reference",
    11: "large amp x",
    12: "large amp y",
}


# ============================================================
# FFT tune measurement
# ============================================================

def measure_tune(signal, n_turns=None):
    """Measure tune from a TBT signal via FFT with Hann window.

    Uses Hann window + zero-padding to 65536 for spectral interpolation,
    same method as example/03 extract_tune().
    Returns (fractional_tune, peak_amplitude).
    """
    if len(signal) < 4:
        return 0.0, 0.0

    n = len(signal)
    n_pad = max(n, 65536)

    signal = signal - np.mean(signal)

    win = np.hanning(n)
    sig_padded = np.zeros(n_pad)
    sig_padded[:n] = signal * win

    spectrum = np.abs(np.fft.rfft(sig_padded))
    freqs = np.fft.rfftfreq(n_pad)

    spectrum[:2] = 0
    peak_idx = np.argmax(spectrum)
    tune = freqs[peak_idx]

    if 0 < peak_idx < len(spectrum) - 1:
        a = spectrum[peak_idx - 1]
        b = spectrum[peak_idx]
        c = spectrum[peak_idx + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-30:
            shift = 0.5 * (a - c) / denom
            tune = (peak_idx + shift) / n_pad

    return tune, spectrum[peak_idx]


# ============================================================
# Courant-Snyder invariant
# ============================================================

def cs_invariant(x, px, alpha, beta):
    """Compute CS invariant: gamma*x^2 + 2*alpha*x*px + beta*px^2."""
    gamma = (1.0 + alpha * alpha) / beta
    return gamma * x * x + 2.0 * alpha * x * px + beta * px * px


# ============================================================
# I/O
# ============================================================

def find_latest_output(script_dir):
    """Find the most recent output/YYYY_MMDD/HHMM_SS directory."""
    output_root = script_dir / "output"
    if not output_root.exists():
        return None
    date_dirs = sorted(output_root.iterdir())
    for date_dir in reversed(date_dirs):
        if not date_dir.is_dir():
            continue
        time_dirs = sorted(date_dir.iterdir())
        for time_dir in reversed(time_dirs):
            if time_dir.is_dir():
                return time_dir
    return None


def read_pass_tbt(output_dir, max_tag=12):
    """Read PASS ParticleMonitor TFS files."""
    particle_dir = output_dir / "particle"
    if not particle_dir.exists():
        raise FileNotFoundError(f"Particle directory not found: {particle_dir}")

    tfs_files = sorted(particle_dir.glob("*_beam*_tag*.tfs"))
    if not tfs_files:
        raise FileNotFoundError(f"No particle TFS files found in {particle_dir}")

    data = {}
    for f in tfs_files:
        tag_str = f.stem.split("_tag")[-1].lstrip("_")
        tag = int(tag_str)
        if tag > max_tag:
            continue

        df = tfs_lib.read(str(f))
        data[tag] = {
            "turn": df["turn"].to_numpy(),
            "x": df["x"].to_numpy(),
            "px": df["px"].to_numpy(),
            "y": df["y"].to_numpy(),
            "py": df["py"].to_numpy(),
            "z": df["z"].to_numpy(),
            "dp": df["dp"].to_numpy(),
        }
    return data


def read_pass_stat(output_dir):
    """Read PASS StatMonitor CSV."""
    csv_files = list(output_dir.glob("*_stat_*.csv"))
    if not csv_files:
        return None

    with open(csv_files[0], "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    result = {}
    for key in rows[0]:
        result[key] = np.array([float(r[key]) for r in rows])
    return result


# ============================================================
# Plotting helpers
# ============================================================

def plot_tune_fft(pass_data, tag, plane, expected_tune, ax):
    """Plot FFT spectrum for a given tag and plane."""
    signal = pass_data[tag][plane]
    n = len(signal)
    if n < 4:
        return

    n_pad = max(n, 65536)
    win = np.hanning(n)
    sig_padded = np.zeros(n_pad)
    sig_padded[:n] = (signal - np.mean(signal)) * win

    spectrum = np.abs(np.fft.rfft(sig_padded))
    freqs = np.fft.rfftfreq(n_pad)

    spectrum[:2] = 0
    ax.plot(freqs, spectrum, "b-", linewidth=1)
    ax.axvline(expected_tune, color="r", linestyle="--", linewidth=1,
               label=f"expected Q={expected_tune:.4f}")

    measured, _ = measure_tune(signal, n)
    ax.axvline(measured, color="g", linestyle=":", linewidth=1,
               label=f"measured Q={measured:.6f}")

    ax.set_xlabel("tune")
    ax.set_ylabel("amplitude")
    ax.set_title(f"tag {tag} ({TAG_INFO.get(tag, '')}): {plane} FFT", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)


def plot_chromaticity(dp_values, qx_values, qy_values, ax_x, ax_y):
    """Plot tune vs dp with linear fit."""
    ax_x.plot(dp_values, qx_values, "bo", markersize=8, label="measured")
    ax_y.plot(dp_values, qy_values, "ro", markersize=8, label="measured")

    if len(dp_values) >= 2:
        cx = np.polyfit(dp_values, qx_values, 1)
        cy = np.polyfit(dp_values, qy_values, 1)
        dp_fit = np.linspace(min(dp_values), max(dp_values), 100)
        ax_x.plot(dp_fit, np.polyval(cx, dp_fit), "b--",
                  label=f"fit: DQx={cx[0]:.6f}\n(theory: DQx={DQX_CORRECTED:.6f})")
        ax_y.plot(dp_fit, np.polyval(cy, dp_fit), "r--",
                  label=f"fit: DQy={cy[0]:.6f}\n(theory: DQy={DQY_CORRECTED:.6f})")

    ax_x.axhline(QX_EXPECTED, color="k", linestyle=":", alpha=0.3,
                 label=f"Qx0={QX_EXPECTED:.4f}")
    ax_y.axhline(QY_EXPECTED, color="k", linestyle=":", alpha=0.3,
                 label=f"Qy0={QY_EXPECTED:.4f}")

    ax_x.set_xlabel(r"$\delta p$")
    ax_x.set_ylabel(r"$Q_x$")
    ax_x.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax_x.set_title(r"Chromaticity $Q_x(\delta p)$", fontsize=11)
    ax_x.legend(fontsize=9)
    ax_x.grid(True, alpha=0.3)

    ax_y.set_xlabel(r"$\delta p$")
    ax_y.set_ylabel(r"$Q_y$")
    ax_y.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax_y.set_title(r"Chromaticity $Q_y(\delta p)$", fontsize=11)
    ax_y.legend(fontsize=9)
    ax_y.grid(True, alpha=0.3)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    output_dir = find_latest_output(SCRIPT_DIR)
    if output_dir is None:
        print("[Error] No output directory found.")
        exit(1)

    print(f"[Analyse] Output: {output_dir}")

    # --- Read data ---
    pass_data = read_pass_tbt(output_dir)
    n_turns = len(next(iter(pass_data.values()))["turn"])
    print(f"[Analyse] Reading PASS TBT data ...")
    print(f"  Tags found: {sorted(pass_data.keys())}")
    print(f"  Turns: {n_turns}")

    stat_data = read_pass_stat(output_dir)

    # ============================================================
    # 1. Tune measurement
    # ============================================================
    print("\n" + "=" * 60)
    print("1. TUNE MEASUREMENT (FFT)")
    print("=" * 60)

    if 1 in pass_data:
        qx_meas, _ = measure_tune(pass_data[1]["x"], n_turns)
        print(f"  Qx:  expected={QX_EXPECTED:.4f}  measured={qx_meas:.6f}  "
              f"diff={qx_meas - QX_EXPECTED:+.6f}")

    if 2 in pass_data:
        qy_meas, _ = measure_tune(pass_data[2]["y"], n_turns)
        print(f"  Qy:  expected={QY_EXPECTED:.4f}  measured={qy_meas:.6f}  "
              f"diff={qy_meas - QY_EXPECTED:+.6f}")

    # ============================================================
    # 2. Courant-Snyder invariant
    # ============================================================
    print("\n" + "=" * 60)
    print("2. COURANT-SNYDER INVARIANT (should be constant per particle)")
    print("=" * 60)

    for tag in sorted(pass_data.keys()):
        d = pass_data[tag]
        if np.all(d["x"] == 0) and np.all(d["y"] == 0):
            continue

        # For particles with dp≠0, subtract dispersion to get betatron motion.
        # CS invariant is only conserved for the beta oscillation part:
        #   x_beta = x - Dx * dp,  px_beta = px - Dpx * dp
        dp_arr = d["dp"]
        x_beta = d["x"] - DX_EXPECTED * dp_arr
        px_beta = d["px"] - DPX_EXPECTED * dp_arr

        jx = cs_invariant(x_beta, px_beta, ALPHA_X, BETA_X)
        jy = cs_invariant(d["y"], d["py"], ALPHA_Y, BETA_Y)

        # Analytic: initial coords also contain dispersion → must subtract.
        #   x_beta_init = x_init - Dx * dp_init
        #   px_beta_init = px_init - Dpx * dp_init
        x0, px0, y0, py0, z0, dp0 = TEST_PARTICLES[tag - 1]
        x0_beta = x0 - DX_EXPECTED * dp0
        px0_beta = px0 - DPX_EXPECTED * dp0

        jx_analytic = cs_invariant(x0_beta, px0_beta, ALPHA_X, BETA_X)
        jy_analytic = cs_invariant(y0, py0, ALPHA_Y, BETA_Y)

        print(f"  tag {tag:2d}: Jx = {np.mean(jx):.6e} (std={np.std(jx):.2e}, "
              f"analytic={jx_analytic:.6e})  "
              f"Jy = {np.mean(jy):.6e} (std={np.std(jy):.2e}, "
              f"analytic={jy_analytic:.6e})")

    # ============================================================
    # 3. Chromaticity
    # ============================================================
    print("\n" + "=" * 60)
    print("3. CHROMATICITY (tune shift vs dp)")
    print("=" * 60)
    print(f"  Natural:   DQx={DQX_NATURAL:.6f}, DQy={DQY_NATURAL:.6f}")
    print(f"  Corrected: DQx={DQX_CORRECTED:.6f}, DQy={DQY_CORRECTED:.6f}")
    print(f"  (PASS should measure the corrected chromaticity)")

    dp_meas = []
    qx_pos_list = []
    qx_neg_list = []
    qy_pos_list = []
    qy_neg_list = []

    for tag_pos, tag_neg, dp_val in CHROM_PAIRS:
        if tag_pos not in pass_data or tag_neg not in pass_data:
            continue

        qx_pos, _ = measure_tune(pass_data[tag_pos]["x"], n_turns)
        qx_neg, _ = measure_tune(pass_data[tag_neg]["x"], n_turns)
        qy_pos, _ = measure_tune(pass_data[tag_pos]["y"], n_turns)
        qy_neg, _ = measure_tune(pass_data[tag_neg]["y"], n_turns)

        dp_meas.append(dp_val)
        qx_pos_list.append(qx_pos)
        qx_neg_list.append(qx_neg)
        qy_pos_list.append(qy_pos)
        qy_neg_list.append(qy_neg)

        dqx_pair = (qx_pos - qx_neg) / (2 * dp_val)
        dqy_pair = (qy_pos - qy_neg) / (2 * dp_val)

        print(f"  dp={dp_val:+.0e}: Qx+={qx_pos:.6f} Qx-={qx_neg:.6f} "
              f"-> DQx={dqx_pair:.6f}  |  "
              f"Qy+={qy_pos:.6f} Qy-={qy_neg:.6f} "
              f"-> DQy={dqy_pair:.6f}")

    if len(dp_meas) >= 1:
        dqx_pairs = [(qx_pos_list[i] - qx_neg_list[i]) / (2 * dp_meas[i])
                     for i in range(len(dp_meas))]
        dqy_pairs = [(qy_pos_list[i] - qy_neg_list[i]) / (2 * dp_meas[i])
                     for i in range(len(dp_meas))]

        print(f"\n  Per-pair average: DQx={np.mean(dqx_pairs):.6f} "
              f"(corrected {DQX_CORRECTED:.6f})")
        print(f"  Per-pair average: DQy={np.mean(dqy_pairs):.6f} "
              f"(corrected {DQY_CORRECTED:.6f})")

        all_dp = dp_meas + [-d for d in dp_meas]
        all_qx = qx_pos_list + qx_neg_list
        all_qy = qy_pos_list + qy_neg_list

        cx_fit = np.polyfit(all_dp, all_qx, 1)
        cy_fit = np.polyfit(all_dp, all_qy, 1)
        print(f"  Linear fit:       DQx={cx_fit[0]:.6f} "
              f"(corrected {DQX_CORRECTED:.6f})")
        print(f"  Linear fit:       DQy={cy_fit[0]:.6f} "
              f"(corrected {DQY_CORRECTED:.6f})")

    # ============================================================
    # 4. Dispersion
    # ============================================================
    print("\n" + "=" * 60)
    print("4. DISPERSION (from +/-dp particle pairs)")
    print("=" * 60)
    print(f"  Expected at s=0: Dx={DX_EXPECTED:.6f}, Dpx={DPX_EXPECTED:.6f}")

    for tag_pos, tag_neg, dp_val in CHROM_PAIRS:
        if tag_pos not in pass_data or tag_neg not in pass_data:
            continue

        d_pos = pass_data[tag_pos]
        d_neg = pass_data[tag_neg]

        x_avg_pos = np.mean(d_pos["x"])
        x_avg_neg = np.mean(d_neg["x"])
        px_avg_pos = np.mean(d_pos["px"])
        px_avg_neg = np.mean(d_neg["px"])

        dx_meas = (x_avg_pos - x_avg_neg) / (2 * dp_val)
        dpx_meas = (px_avg_pos - px_avg_neg) / (2 * dp_val)

        print(f"  dp={dp_val:+.0e}: Dx={dx_meas:.6f}  Dpx={dpx_meas:.6f}")

    # ============================================================
    # 5. Beam statistics
    # ============================================================
    if stat_data is not None:
        print("\n" + "=" * 60)
        print("5. BEAM STATISTICS (StatMonitor, from distribution particles)")
        print("=" * 60)

        for key, expected, label in [
            ("betax", BETA_X, "bx"),
            ("betay", BETA_Y, "by"),
            ("alphax", ALPHA_X, "ax"),
            ("alphay", ALPHA_Y, "ay"),
            ("xEmittance", EMIT_X, "ex"),
            ("yEmittance", EMIT_Y, "ey"),
            ("invariantx", 1.0, "Jx/ex"),
            ("invarianty", 1.0, "Jy/ey"),
        ]:
            if key in stat_data:
                vals = stat_data[key]
                print(f"  {label:8s}: mean={np.mean(vals):.6e}  "
                      f"std={np.std(vals):.2e}  expected={expected:.6e}")

    # ============================================================
    # Generate plots
    # ============================================================
    print("\n[Analyse] Generating plots ...")

    # --- Fig 1: Tune FFT ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle("Tune Measurement via FFT", fontsize=14, fontweight="bold")

    if 1 in pass_data:
        plot_tune_fft(pass_data, 1, "x", QX_EXPECTED, axes1[0, 0])
    if 2 in pass_data:
        plot_tune_fft(pass_data, 2, "y", QY_EXPECTED, axes1[0, 1])
    if 3 in pass_data:
        plot_tune_fft(pass_data, 3, "x", QX_EXPECTED, axes1[1, 0])
    axes1[1, 1].axis("off")

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # --- Fig 2: Chromaticity ---
    if len(dp_meas) >= 1:
        fig2, (ax_cx, ax_cy) = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle("Chromaticity Measurement", fontsize=14, fontweight="bold")

        all_dp = dp_meas + [-d for d in dp_meas]
        all_qx = qx_pos_list + qx_neg_list
        all_qy = qy_pos_list + qy_neg_list

        plot_chromaticity(all_dp, all_qx, all_qy, ax_cx, ax_cy)
        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    print("\n[Done] Plots displayed.")
