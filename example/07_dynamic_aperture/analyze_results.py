"""Analyze one explicitly selected ParticleMonitor file."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PASS.analysis.dynamic_aperture import export_dynamic_aperture, read_dynamic_aperture
from PASS.plot.plot_dynamic_aperture import plot_dynamic_aperture, plot_dynamic_aperture_boundaries


def analyze(path, output):
    result = read_dynamic_aperture(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    export_dynamic_aperture(result, output / "dynamic_aperture.npz")
    export_dynamic_aperture(result, output / "dynamic_aperture.csv")
    figure, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    plot_dynamic_aperture_boundaries(result, ax=ax)
    figure.savefig(output / "dynamic_aperture_dp_boundaries.png", dpi=160)
    figure.savefig(output / "dynamic_aperture_dp_boundaries.pdf")
    plt.close(figure)
    for index, dp in enumerate(result["dp_values"]):
        for mode in ("status", "loss_turn"):
            figure, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
            plot_dynamic_aperture(result, dp=float(dp), mode=mode, boundary=True, ax=ax)
            figure.savefig(output / f"dynamic_aperture_dp_{index:02d}_{mode}.png", dpi=160)
            figure.savefig(output / f"dynamic_aperture_dp_{index:02d}_{mode}.pdf")
            plt.close(figure)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("monitor", type=Path, help="The *_particles.h5 file from the selected run")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("analysis"))
    args = parser.parse_args()
    analyze(args.monitor, args.output)
