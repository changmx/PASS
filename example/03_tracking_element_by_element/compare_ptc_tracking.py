"""Compare PASS element-by-element tracking with cpymad PTC tracking.

Reads PASS ParticleMonitor TBT data, runs PTC tracking with identical
initial coordinates and turn count, then produces three comparison
figures:

    1. Phase space scatter (all 1024 turns) — ellipse shape
    2. Relative difference vs turn (all 1024 turns) — divergence trend
    3. TBT trajectory (first 50 turns) — waveform + phase

Usage:
    python compare_ptc_tracking.py
    python compare_ptc_tracking.py --output-dir output/2026_0730/1149_06
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tfs
from cpymad.madx import Madx

# ============================================================
# Constants
# ============================================================

REPRESENTATIVE_TAGS = [1, 2, 3, 11, 13, 17]

# Tag metadata: which plane to plot for phase space
TAG_INFO = {
    1: dict(label="tag 1:  x=2mm", plane="x"),
    2: dict(label="tag 2:  y=2mm", plane="y"),
    3: dict(label="tag 3:  x=y=1mm, dp=+5e-5", plane="x"),
    11: dict(label="tag 11: x=y=1mm, dp=+3e-3", plane="x"),
    13: dict(label="tag 13: x=5mm, y=0", plane="x"),
    17: dict(label="tag 17: x=y=3mm", plane="x"),
}

# ============================================================
# Initial coordinates (must match make_input.py)
# ============================================================


def get_test_particles():
    """Return 17 test particles [x, px, y, py, z, dp]."""
    dp_list = [5e-5, 1e-4, 5e-4, 1e-3]
    adts_x = [5e-3, 10e-3]
    adts_y = [5e-3, 10e-3]

    particles = []
    # Group A: linear tune
    particles.append([2e-3, 0, 0, 0, 0, 0])  # tag 1
    particles.append([0, 0, 2e-3, 0, 0, 0])  # tag 2
    # Group B: chromaticity
    for dp in dp_list:
        particles.append([1e-3, 0, 1e-3, 0, 0, +dp])
        particles.append([1e-3, 0, 1e-3, 0, 0, -dp])
    # Group C: large dp
    particles.append([1e-3, 0, 1e-3, 0, 0, +3e-3])  # tag 11
    particles.append([1e-3, 0, 1e-3, 0, 0, -3e-3])  # tag 12
    # Group D: ADTS (single-plane: y=0 for x scan, x=0 for y scan)
    for ax in adts_x:
        particles.append([ax, 0, 0, 0, 0, 0])
    for ay in adts_y:
        particles.append([0, 0, ay, 0, 0, 0])
    # Group E: coupling
    particles.append([3e-3, 0, 3e-3, 0, 0, 0])  # tag 17
    return particles


# ============================================================
# PTC tracking
# ============================================================


def run_ptc_tracking(
    seq_path,
    particles,
    n_turns=1024,
    seq_name="RING",
    particle="Proton",
    energy="1000",
):
    """Run PTC tracking for multiple particles via cpymad.

    Returns:
        {tag: {"turn": array, "x": array, "px": array,
               "y": array, "py": array, "z": array, "dp": array}}
    """
    madx = Madx(stdout=False)
    madx.option(echo=False)

    madx.call(file=str(seq_path))
    madx.command.beam(sequence=seq_name, particle=particle, energy=energy)
    madx.use(sequence=seq_name)

    madx.command.ptc_create_universe(sector_nmul=10, sector_nmul_max=10)
    madx.command.ptc_create_layout(model=1, method=4, nst=5, exact=True)

    # Observe at sequence start (Injpoint marker at s=0)
    madx.command.ptc_observe(place="injpoint")

    # Define ALL particles before a single ptc_track call
    for x, px, y, py, z, dp in particles:
        madx.command.ptc_start(x=x, px=px, y=y, py=py, t=z, pt=dp)

    madx.command.ptc_track(
        icase=5,
        recloss=True,
        closed_orbit=False,
        onetable=True,
        dump=True,
        element_by_element=False,
        DELTAP=0,
        maxaper=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        turns=n_turns,
        ffile=1,
        file="ptc_track_temp",
        extension=".dat",
    )
    madx.command.ptc_track_end()

    df = madx.table["trackone"].dframe()

    # trackone records both #e (entry, end-of-turn) and injpoint (observe,
    # start-of-turn) rows per turn.  #e row for turn N = coordinates after
    # N turns, matching PASS convention.  Keep #e (the last row per turn).
    df = df.drop_duplicates(subset=["number", "turn"], keep="last")

    results = {}
    for tag_idx in range(1, len(particles) + 1):
        ptc_data = df[df["number"] == tag_idx].copy()
        results[tag_idx] = {
            "turn": ptc_data["turn"].to_numpy(),
            "x": ptc_data["x"].to_numpy(),
            "px": ptc_data["px"].to_numpy(),
            "y": ptc_data["y"].to_numpy(),
            "py": ptc_data["py"].to_numpy(),
            "z": ptc_data["t"].to_numpy() if "t" in ptc_data.columns else ptc_data["z"].to_numpy(),
            "dp": ptc_data["pt"].to_numpy() if "pt" in ptc_data.columns else ptc_data["dp"].to_numpy(),
        }
        print(f"  [PTC] tag {tag_idx:2d}: {len(ptc_data)} rows")

    madx.quit()
    return results


# ============================================================
# Read PASS TBT data
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
            if time_dir.is_dir() and (time_dir / "particle").exists():
                return time_dir
    return None


def read_pass_tbt(output_dir, max_tag=17):
    """Read PASS ParticleMonitor TFS files.

    Returns:
        {tag: {"turn": array, "x": array, "px": array,
               "y": array, "py": array, "z": array, "dp": array}}
    """
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

        df = tfs.read(str(f))
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


# ============================================================
# Plotting
# ============================================================


def _turn_mask(turns, turn_range):
    """Return boolean mask for turns within turn_range (inclusive).

    turn_range=None means all turns.
    """
    if turn_range is None:
        return np.ones(len(turns), dtype=bool)
    t_start, t_end = turn_range
    return (turns >= t_start) & (turns <= t_end)


def _range_label(turn_range):
    """Return a title suffix string for a turn range."""
    if turn_range is None:
        return "(all turns)"
    t_start, t_end = turn_range
    return f"(turns {t_start}-{t_end})"


def plot_phase_space(pass_data, ptc_data, tags, turn_range=None):
    """Figure 1: Transverse phase space scatter."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle(f"Phase Space: PASS vs PTC {_range_label(turn_range)}", fontsize=14, fontweight="bold")

    for idx, tag in enumerate(tags):
        ax = axes[idx // 2, idx % 2]
        plane = TAG_INFO[tag]["plane"]
        label = TAG_INFO[tag]["label"]

        px_col = "px" if plane == "x" else "py"

        mask_p = _turn_mask(pass_data[tag]["turn"], turn_range)
        mask_t = _turn_mask(ptc_data[tag]["turn"], turn_range)

        ax.scatter(pass_data[tag][plane][mask_p], pass_data[tag][px_col][mask_p], s=3, c="blue", alpha=0.3, label="PASS")
        ax.scatter(ptc_data[tag][plane][mask_t], ptc_data[tag][px_col][mask_t], s=3, c="red", alpha=0.3, label="PTC")

        ax.set_xlabel(f"{plane} (m)")
        ax.set_ylabel(f"{px_col} (rad)")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9, markerscale=4)
        ax.grid(True, alpha=0.3)

    fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.95, bottom=0.05)
    return fig


def plot_relative_difference(pass_data, ptc_data, tags, turn_range=None):
    """Figure 2: Relative difference (PASS - PTC) / amplitude vs turn.

    Normalized by half peak-to-peak amplitude of the PTC signal (computed
    over the selected turn range) to avoid division-by-zero at zero crossings.
    """
    n_tags = len(tags)
    fig, axes = plt.subplots(n_tags, 1, figsize=(14, 3.2 * n_tags), squeeze=False)
    fig.suptitle(f"Relative Difference: (PASS - PTC) / amplitude  {_range_label(turn_range)}", fontsize=14, fontweight="bold")

    for idx, tag in enumerate(tags):
        ax = axes[idx, 0]

        plane = TAG_INFO[tag]["plane"]
        col = plane                      # "x" or "y"
        pcol = "p" + plane               # "px" or "py"

        mask_p = _turn_mask(pass_data[tag]["turn"], turn_range)
        mask_t = _turn_mask(ptc_data[tag]["turn"], turn_range)

        turns = pass_data[tag]["turn"][mask_p]
        s_pass = pass_data[tag][col][mask_p]
        s_ptc = ptc_data[tag][col][mask_t]
        ps_pass = pass_data[tag][pcol][mask_p]
        ps_ptc = ptc_data[tag][pcol][mask_t]

        n = min(len(s_pass), len(s_ptc))
        turns = turns[:n]
        s_pass, s_ptc = s_pass[:n], s_ptc[:n]
        ps_pass, ps_ptc = ps_pass[:n], ps_ptc[:n]

        # Normalize by half peak-to-peak amplitude over selected range
        s_amp = (np.max(s_ptc) - np.min(s_ptc)) / 2
        ps_amp = (np.max(ps_ptc) - np.min(ps_ptc)) / 2

        rel_s = (s_pass - s_ptc) / s_amp * 100 if s_amp > 0 else np.zeros(n)
        rel_ps = (ps_pass - ps_ptc) / ps_amp * 100 if ps_amp > 0 else np.zeros(n)

        ax.plot(turns, rel_s, "b-", linewidth=0.5, label=f"Δ{col} / {col}_amp")
        ax.plot(turns, rel_ps, "r-", linewidth=0.5, label=f"Δ{pcol} / {pcol}_amp")
        if idx < n_tags - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("turn")
        ax.set_ylabel("relative diff (%)")
        ax.set_title(TAG_INFO[tag]["label"], fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.subplots_adjust(hspace=0.35, top=0.95, bottom=0.05)
    return fig


def plot_tbt_trajectory(pass_data, ptc_data, tags, turn_range=None):
    """Figure 3: TBT trajectory comparison."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle(f"TBT Trajectory: PASS vs PTC {_range_label(turn_range)}", fontsize=14, fontweight="bold")

    for idx, tag in enumerate(tags):
        ax = axes[idx // 2, idx % 2]
        plane = TAG_INFO[tag]["plane"]
        label = TAG_INFO[tag]["label"]

        mask_p = _turn_mask(pass_data[tag]["turn"], turn_range)
        mask_t = _turn_mask(ptc_data[tag]["turn"], turn_range)

        ax.plot(pass_data[tag]["turn"][mask_p], pass_data[tag][plane][mask_p], "b-", linewidth=1.2, label="PASS", alpha=0.8)
        ax.plot(ptc_data[tag]["turn"][mask_t], ptc_data[tag][plane][mask_t], "r--", linewidth=1.2, label="PTC", alpha=0.8)

        ax.set_xlabel("turn")
        ax.set_ylabel(f"{plane} (m)")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.95, bottom=0.05)
    return fig


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    seq_path = str(script_dir / "fodo.seq")
    n_turns = 1024

    # Turn range for all figures: (start, end) inclusive, or None for all turns
    turn_range = (0, 50)  # e.g. (0, 50) for first 50 turns
    # turn_range = (0, 50)
    # turn_range = (500, 1024)

    # --- Find PASS output ---
    output_dir = find_latest_output(script_dir)
    # output_dir = script_dir / "output" / "2026_0730" / "1149_06"
    if output_dir is None or not output_dir.exists():
        raise FileNotFoundError("No PASS output directory found")
    print(f"[Compare] PASS output: {output_dir}")

    # --- Read PASS data ---
    print("\n[Compare] Reading PASS TBT data ...")
    pass_data = read_pass_tbt(output_dir)
    print(f"  Tags found: {sorted(pass_data.keys())}")

    # --- Run PTC tracking ---
    print(f"\n[Compare] Running PTC tracking ({n_turns} turns) ...")
    particles = get_test_particles()
    ptc_data = run_ptc_tracking(seq_path, particles, n_turns=n_turns)
    print(f"  PTC tags done: {sorted(ptc_data.keys())}")

    # --- Plot ---
    print("\n[Compare] Generating plots ...")
    fig1 = plot_phase_space(pass_data, ptc_data, REPRESENTATIVE_TAGS, turn_range=turn_range)
    fig2 = plot_relative_difference(pass_data, ptc_data, REPRESENTATIVE_TAGS, turn_range=turn_range)
    fig3 = plot_tbt_trajectory(pass_data, ptc_data, REPRESENTATIVE_TAGS, turn_range=turn_range)

    output_path = script_dir / "output" / "comparison"
    output_path.mkdir(parents=True, exist_ok=True)

    fig1.savefig(str(output_path / "phase_space_comparison.png"), dpi=150)
    fig2.savefig(str(output_path / "relative_difference_comparison.png"), dpi=150)
    fig3.savefig(str(output_path / "tbt_trajectory_comparison.png"), dpi=150)
    print(f"\n[Done] Figures saved to: {output_path}")
    plt.show()
