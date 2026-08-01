"""Analyze PASS element-by-element tracking output.

Reads TBT (turn-by-turn) particle monitor TFS files, extracts tunes via FFT,
fits chromaticity from dp scan, and compares with MADX reference values.

Usage:
    # Auto-detect latest output directory
    python analyze.py

    # Specify output directory
    python analyze.py --output-dir output/2026_0714/2127_48

    # Specify twiss file for reference values
    python analyze.py --twiss bring.tfs

    # Custom chromaticity dp list (must match generate_beam0.py)
    python analyze.py --dp-list 5e-5,1e-4,5e-4,1e-3
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import tfs

# ============================================================
# Constants: particle group definitions
# ============================================================

# Must match make_input.py make_test_particles() ordering
DEFAULT_DP_LIST = [5e-5, 1e-4, 5e-4, 1e-3]

# Group ranges are 1-indexed tag numbers [start, end] inclusive
GROUP_A_TUNE = (1, 2)  # tag 1-2:   x, y only
GROUP_B_CHROM = (3, 10)  # tag 3-10:  ±dp pairs (4 dp values × 2)
GROUP_C_LARGE_DP = (11, 12)  # tag 11-12: ±3e-3
GROUP_D_ADTS = (13, 16)  # tag 13-16: large amplitude
GROUP_E_COUPLING = (17, 17)  # tag 17:    x=y coupling

# ============================================================
# Step 1: Read TBT data
# ============================================================


def find_latest_output(script_dir: Path) -> Path | None:
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
            if time_dir.is_dir() and (time_dir / "particle").exists():
                return time_dir
    return None


def read_tbt_data(output_dir: Path, max_tag: int = 17) -> dict:
    """Read all particle monitor TFS files from output directory.

    Returns:
        {tag: {"x": array, "px": array, "y": array, "py": array,
               "z": array, "dp": array, "turn": array}}
    """
    particle_dir = output_dir / "particle"
    if not particle_dir.exists():
        raise FileNotFoundError(f"Particle directory not found: {particle_dir}")

    # Find all TFS files
    tfs_files = sorted(particle_dir.glob("*_beam*_tag*.tfs"))
    if not tfs_files:
        raise FileNotFoundError(f"No particle TFS files found in {particle_dir}")

    data = {}
    for f in tfs_files:
        # Extract tag from filename: ..._tagN.tfs or ..._tag_N.tfs
        tag_str = f.stem.split("_tag")[-1].lstrip("_")
        tag = int(tag_str)
        if tag > max_tag:
            continue

        df = tfs.read(str(f))
        data[tag] = {
            "turn": df["turn"].to_numpy(),
            "x": df["x"].to_numpy(),
            "px": df["px"].to_numpy(),
            "y": df["y"].to_numpy(),
            "py": df["py"].to_numpy(),
            "z": df["z"].to_numpy(),
            "dp": df["dp"].to_numpy(),
            "lost_turn": df["lostTurn"].to_numpy() if "lostTurn" in df.columns else None,
        }

    return data


# ============================================================
# Step 2: FFT tune extraction
# ============================================================


def extract_tune(signal: np.ndarray, n_fft: int = 65536) -> float:
    """Extract fractional tune from a TBT signal via FFT.

    Uses Hann window + zero-padding for spectral interpolation.
    Returns the fractional tune in [0, 0.5].
    """
    n = len(signal)

    # Skip particles that didn't oscillate (all zeros or constant)
    if np.std(signal) < 1e-15:
        return -1.0

    # Remove mean
    signal = signal - np.mean(signal)

    # Hann window
    window = np.hanning(n)
    windowed = signal * window

    # Zero-pad
    padded = np.zeros(n_fft)
    padded[:n] = windowed

    # FFT
    spectrum = np.abs(np.fft.rfft(padded))
    freqs = np.fft.rfftfreq(n_fft)  # in [0, 0.5]

    # Find peak (exclude DC region)
    spectrum[:2] = 0  # kill DC
    peak_idx = np.argmax(spectrum)
    tune_frac = freqs[peak_idx]

    # Refine with parabolic interpolation
    if 0 < peak_idx < len(spectrum) - 1:
        a = spectrum[peak_idx - 1]
        b = spectrum[peak_idx]
        c = spectrum[peak_idx + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-30:
            delta = 0.5 * (a - c) / denom
            tune_frac = (peak_idx + delta) / n_fft

    return tune_frac


def get_tunes_for_tag(tag_data: dict) -> tuple[float, float]:
    """Extract Qx and Qy fractional tunes for a single tag."""
    qx_frac = extract_tune(tag_data["x"])
    qy_frac = extract_tune(tag_data["y"])
    return qx_frac, qy_frac


# ============================================================
# Step 3: Chromaticity fit
# ============================================================


def fit_chromaticity(data: dict, dp_list: list[float]) -> dict:
    """Fit linear chromaticity from small-dp scan.

    Uses Group B particles (tags 3-10): pairs of ±dp for each dp value.
    All data points (with signed dp) are used directly in a linear fit:
        Q(dp) = Q0 + DQ * dp

    Fitting with signed ±dp preserves the linear (odd) term, which is
    the chromaticity.  (Averaging Q(+dp) and Q(-dp) would cancel it.)

    Returns dict with qx0, qy0, dq1, dq2, and raw data points.
    """
    # Reference tune from tag 1 (Qx) and tag 2 (Qy)
    qx0_frac = extract_tune(data[1]["x"])
    qy0_frac = extract_tune(data[2]["y"])

    # Collect all signed dp data points
    dp_vals = []
    qx_vals = []
    qy_vals = []

    tag = 3  # Group B starts at tag 3
    for dp in dp_list:
        # +dp particle
        qx_pos = extract_tune(data[tag]["x"])
        qy_pos = extract_tune(data[tag]["y"])
        tag += 1
        # -dp particle
        qx_neg = extract_tune(data[tag]["x"])
        qy_neg = extract_tune(data[tag]["y"])
        tag += 1

        dp_vals.append(+dp)
        qx_vals.append(qx_pos)
        qy_vals.append(qy_pos)
        dp_vals.append(-dp)
        qx_vals.append(qx_neg)
        qy_vals.append(qy_neg)

    dp_arr = np.array(dp_vals)
    qx_arr = np.array(qx_vals)
    qy_arr = np.array(qy_vals)

    # Linear fit: Q(dp) = Q0 + DQ * dp  (slope = DQ)
    qx_fit = np.polyfit(dp_arr, qx_arr, 1)  # [slope, intercept]
    qy_fit = np.polyfit(dp_arr, qy_arr, 1)

    return {
        "qx0_frac": qx0_frac,
        "qy0_frac": qy0_frac,
        "qx0_fit": qx_fit[1],
        "qy0_fit": qy_fit[1],
        "dq1_frac": qx_fit[0],
        "dq2_frac": qy_fit[0],
        "dp_vals": dp_arr,
        "qx_vals": qx_arr,
        "qy_vals": qy_arr,
    }


# ============================================================
# Step 4: Dispersion
# ============================================================

# Group B tag pairs: (tag+dp, tag-dp, delta_value)
DELTA_PAIRS = [
    (3, 4,  5e-5),
    (5, 6,  1e-4),
    (7, 8,  5e-4),
    (9, 10, 1e-3),
]


def compute_dispersion(data: dict) -> dict:
    """Compute Dx and Dpx at s=0 from symmetric +/-delta pairs (Group B).

    Dx = (x_mean(+d) - x_mean(-d)) / (2*d)
    Dpx = (px_mean(+d) - px_mean(-d)) / (2*d)

    Betatron oscillation cancels in the difference; the residual is
    the dispersion offset.  Averaging over turns removes the betatron
    contribution that does not cancel perfectly due to tune spread.
    """
    results = {"delta": [], "Dx": [], "Dpx": [], "Dx_std": [], "Dpx_std": []}

    skip = 10  # skip first few turns for transient

    for tag_p, tag_n, delta in DELTA_PAIRS:
        if tag_p not in data or tag_n not in data:
            continue

        xp = data[tag_p]["x"]
        xn = data[tag_n]["x"]
        pxp = data[tag_p]["px"]
        pxn = data[tag_n]["px"]

        dx_per_turn = (xp - xn) / (2 * delta)
        dpx_per_turn = (pxp - pxn) / (2 * delta)

        results["delta"].append(delta)
        results["Dx"].append(np.mean(dx_per_turn[skip:]))
        results["Dpx"].append(np.mean(dpx_per_turn[skip:]))
        results["Dx_std"].append(np.std(dx_per_turn[skip:]))
        results["Dpx_std"].append(np.std(dpx_per_turn[skip:]))

    return results


# ============================================================
# Step 5: ADTS analysis
# ============================================================


def analyze_adts(data: dict, adts_x: list[float], adts_y: list[float]) -> dict:
    """Analyze amplitude-dependent tune shift.

    Group D: tags 13-16 (single-plane: y=0 for x scan, x=0 for y scan)
        tag 13: x=5mm,  y=0    → Qx vs amplitude
        tag 14: x=10mm, y=0    → Qx vs amplitude
        tag 15: x=0,    y=5mm  → Qy vs amplitude
        tag 16: x=0,    y=10mm → Qy vs amplitude
    """
    # Qx vs x amplitude
    qx_adts = []
    for i, ax in enumerate(adts_x):
        tag = 13 + i  # tags 13, 14
        qx = extract_tune(data[tag]["x"])
        qx_adts.append((ax, qx))

    # Qy vs y amplitude
    qy_adts = []
    for i, ay in enumerate(adts_y):
        tag = 15 + i  # tags 15, 16
        qy = extract_tune(data[tag]["y"])
        qy_adts.append((ay, qy))

    return {"qx_adts": qx_adts, "qy_adts": qy_adts}


# ============================================================
# Step 6: Print results
# ============================================================


def print_separator(title: str, width: int = 60):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _fmt(val, spec: str) -> str:
    """Format value or return 'N/A' for None."""
    return f"{val:{spec}}" if val is not None else "N/A".rjust(12)


def _fmt_pct(val, ref, spec: str) -> str:
    """Format percentage (val-ref)/ref*100, or 'N/A'."""
    if val is not None and ref is not None and ref != 0:
        return f"{(val - ref) / ref * 100:{spec}}"
    return "N/A".rjust(12)


def print_results(data: dict, chrom: dict, disp: dict, adts: dict, madx_ref: dict):
    """Print formatted analysis results."""

    # --- Linear tune ---
    print_separator("A. Linear Tune (Group A)")
    qx_madx = madx_ref.get("q1", None)
    qy_madx = madx_ref.get("q2", None)
    qx_int = int(qx_madx) if qx_madx else 0
    qy_int = int(qy_madx) if qy_madx else 0

    qx_pass = qx_int + chrom["qx0_frac"]
    qy_pass = qy_int + chrom["qy0_frac"]

    print(f"  {'':20s} {'PASS':>12s} {'MADX PTC':>12s} {'ΔQ (%)':>12s}")
    print(f"  {'Qx':20s} {qx_pass:12.6f} {_fmt(qx_madx, '12.6f')} {_fmt_pct(qx_pass, qx_madx, '+12.6f')}")
    print(f"  {'Qy':20s} {qy_pass:12.6f} {_fmt(qy_madx, '12.6f')} {_fmt_pct(qy_pass, qy_madx, '+12.6f')}")

    # --- Chromaticity ---
    print_separator("B. Chromaticity (Group B, linear fit)")
    dq1_madx = madx_ref.get("dq1", None)
    dq2_madx = madx_ref.get("dq2", None)

    print(f"  {'':20s} {'PASS':>12s} {'MADX PTC':>12s} {'ΔDQ (%)':>12s}")
    print(f"  {'DQ1':20s} {chrom['dq1_frac']:12.4f} {_fmt(dq1_madx, '12.4f')} {_fmt_pct(chrom['dq1_frac'], dq1_madx, '+12.4f')}")
    print(f"  {'DQ2':20s} {chrom['dq2_frac']:12.4f} {_fmt(dq2_madx, '12.4f')} {_fmt_pct(chrom['dq2_frac'], dq2_madx, '+12.4f')}")

    print(f"\n  dp scan data points (signed):")
    print(f"  {'dp':>12s} {'Qx(frac)':>12s} {'Qy(frac)':>12s}")
    for i in range(len(chrom["dp_vals"])):
        print(f"  {chrom['dp_vals'][i]:+12.1e} {chrom['qx_vals'][i]:12.6f} {chrom['qy_vals'][i]:12.6f}")

    # --- Dispersion ---
    print_separator("B2. Dispersion at s=0 (Group B, +/-dp pairs)")
    dx_madx = madx_ref.get("dx", None)
    dpx_madx = madx_ref.get("dpx", None)
    dx_pass = np.mean(disp["Dx"]) if disp["Dx"] else None
    dpx_pass = np.mean(disp["Dpx"]) if disp["Dpx"] else None

    print(f"  {'delta':>12s} {'Dx (m)':>14s} {'Dx_std':>14s} {'Dpx':>14s} {'Dpx_std':>14s}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
    for i in range(len(disp["delta"])):
        print(f"  {disp['delta'][i]:12.1e} {disp['Dx'][i]:14.6f} {disp['Dx_std'][i]:14.6e} "
              f"{disp['Dpx'][i]:14.6f} {disp['Dpx_std'][i]:14.6e}")
    print(f"\n  {'':20s} {'PASS':>12s} {'MADX Twiss':>12s} {'Diff':>12s}")
    print(f"  {'Dx':20s} {_fmt(dx_pass, '12.6f')} {_fmt(dx_madx, '12.6f')} {_fmt(dx_pass - dx_madx if dx_pass is not None and dx_madx is not None else None, '+12.6f')}")
    print(f"  {'Dpx':20s} {_fmt(dpx_pass, '12.6f')} {_fmt(dpx_madx, '12.6f')} {_fmt(dpx_pass - dpx_madx if dpx_pass is not None and dpx_madx is not None else None, '+12.6f')}")

    # --- Large dp (nonlinear) ---
    print_separator("C. Large dp — Nonlinear Chromaticity (Group C)")
    tag_11 = 11  # +3e-3
    tag_12 = 12  # -3e-3
    if tag_11 in data and tag_12 in data:
        qx_large_pos = extract_tune(data[tag_11]["x"])
        qx_large_neg = extract_tune(data[tag_12]["x"])
        qy_large_pos = extract_tune(data[tag_11]["y"])
        qy_large_neg = extract_tune(data[tag_12]["y"])
        dp_large = 3e-3

        # Linear prediction
        qx_linear = chrom["qx0_fit"] + chrom["dq1_frac"] * dp_large
        qy_linear = chrom["qy0_fit"] + chrom["dq2_frac"] * dp_large

        print(f"  {'':20s} {'measured':>12s} {'linear pred':>12s} {'Δ':>12s}")
        print(f"  {'Qx(dp=+3e-3)':20s} {qx_large_pos:12.6f} {qx_linear:12.6f} {qx_large_pos - qx_linear:+12.6f}")
        print(f"  {'Qy(dp=+3e-3)':20s} {qy_large_pos:12.6f} {qy_linear:12.6f} {qy_large_pos - qy_linear:+12.6f}")

    # --- ADTS ---
    print_separator("D. Amplitude-Dependent Tune Shift (Group D)")
    print(f"  Qx vs x-amplitude:")
    print(f"  {'x (mm)':>10s} {'Qx(frac)':>12s} {'ΔQx':>12s}")
    qx_ref = chrom["qx0_frac"]
    for ax, qx in adts["qx_adts"]:
        print(f"  {ax*1e3:10.1f} {qx:12.6f} {qx - qx_ref:+12.6f}")

    print(f"\n  Qy vs y-amplitude:")
    print(f"  {'y (mm)':>10s} {'Qy(frac)':>12s} {'ΔQy':>12s}")
    qy_ref = chrom["qy0_frac"]
    for ay, qy in adts["qy_adts"]:
        print(f"  {ay*1e3:10.1f} {qy:12.6f} {qy - qy_ref:+12.6f}")

    # --- Coupling ---
    print_separator("E. Coupling (Group E)")
    tag_17 = 17
    if tag_17 in data:
        qx_coup = extract_tune(data[tag_17]["x"])
        qy_coup = extract_tune(data[tag_17]["y"])
        print(f"  x=y=3mm: Qx={qx_coup:.6f}, Qy={qy_coup:.6f}")
        print(f"  ΔQx from ref: {qx_coup - qx_ref:+.6f}")
        print(f"  ΔQy from ref: {qy_coup - qy_ref:+.6f}")

    # --- Summary ---
    print_separator("Summary")
    print(f"  Qx  = {qx_pass:.6f}  (MADX PTC: {_fmt(qx_madx, '.6f').strip()}, Δ = {qx_pass - qx_madx:+.6f}" if qx_madx else f"  Qx  = {qx_pass:.6f}  (no MADX reference)")
    print(f"  Qy  = {qy_pass:.6f}  (MADX PTC: {_fmt(qy_madx, '.6f').strip()}, Δ = {qy_pass - qy_madx:+.6f}" if qy_madx else f"  Qy  = {qy_pass:.6f}  (no MADX reference)")
    print(f"  DQ1 = {chrom['dq1_frac']:.6f}  (MADX PTC: {_fmt(dq1_madx, '.6f').strip()}, Δ = {chrom['dq1_frac'] - dq1_madx:+.6f}" if dq1_madx else f"  DQ1 = {chrom['dq1_frac']:.6f}  (no MADX reference)")
    print(f"  DQ2 = {chrom['dq2_frac']:.6f}  (MADX PTC: {_fmt(dq2_madx, '.6f').strip()}, Δ = {chrom['dq2_frac'] - dq2_madx:+.6f}" if dq2_madx else f"  DQ2 = {chrom['dq2_frac']:.6f}  (no MADX reference)")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PASS element-by-element tracking output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: auto-detect latest)")
    parser.add_argument("--twiss", type=str, default="fodo_ptc.tfs", help="Twiss TFS file for MADX reference (default: fodo_ptc.tfs)")
    parser.add_argument("--dp-list", type=str, default="5e-5,1e-4,5e-4,1e-3", help="Comma-separated dp values for chromaticity fit")
    parser.add_argument("--adts-x", type=str, default="5e-3,10e-3", help="Comma-separated x amplitudes for ADTS")
    parser.add_argument("--adts-y", type=str, default="5e-3,10e-3", help="Comma-separated y amplitudes for ADTS")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # Find output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = script_dir / output_dir
    else:
        output_dir = find_latest_output(script_dir)

    if output_dir is None or not output_dir.exists():
        print(f"Error: output directory not found")
        sys.exit(1)

    print(f"[Analyze] Output: {output_dir}")

    # Read MADX reference
    twiss_file = script_dir / args.twiss
    if twiss_file.exists():
        df = tfs.read(str(twiss_file))
        madx_ref = {
            "q1": df.headers["Q1"],
            "q2": df.headers["Q2"],
            "dq1": df.headers["DQ1"],
            "dq2": df.headers["DQ2"],
        }
        # Dispersion at s=0 from MADX Twiss (fodo.tfs, not fodo_ptc.tfs
        # which has DX=0 at s=0 due to PTC output convention)
        twiss_std = script_dir / "fodo.tfs"
        if twiss_std.exists():
            df_std = tfs.read(str(twiss_std))
            s_col = df_std["S"].to_numpy()
            idx_s0 = np.argmin(np.abs(s_col))
            madx_ref["dx"] = float(df_std.loc[idx_s0, "DX"])
            madx_ref["dpx"] = float(df_std.loc[idx_s0, "DPX"])
        else:
            madx_ref["dx"] = None
            madx_ref["dpx"] = None
        print(f"[Analyze] MADX reference: Q1={madx_ref['q1']:.6f}, Q2={madx_ref['q2']:.6f}, "
              f"DQ1={madx_ref['dq1']:.4f}, DQ2={madx_ref['dq2']:.4f}")
        if madx_ref["dx"] is not None:
            print(f"[Analyze] MADX dispersion at s=0: Dx={madx_ref['dx']:.6f}, Dpx={madx_ref['dpx']:.6f}")
    else:
        madx_ref = {}
        print(f"[Analyze] Warning: {args.twiss} not found, no MADX reference")

    # Parse dp list
    dp_list = [float(x) for x in args.dp_list.split(",")]
    adts_x = [float(x) for x in args.adts_x.split(",")]
    adts_y = [float(x) for x in args.adts_y.split(",")]

    # Read TBT data
    print(f"\n[Analyze] Reading TBT data ...")
    data = read_tbt_data(output_dir, max_tag=17)
    print(f"  Tags found: {sorted(data.keys())}")
    print(f"  Turns per tag: {len(data[list(data.keys())[0]]['turn'])}")

    # Analyze
    chrom = fit_chromaticity(data, dp_list)
    disp = compute_dispersion(data)
    adts = analyze_adts(data, adts_x, adts_y)

    # Print results
    print_results(data, chrom, disp, adts, madx_ref)


if __name__ == "__main__":
    main()
