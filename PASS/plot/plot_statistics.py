"""Nine-panel summaries of saved StatMonitor tables."""

import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy as np

from PASS.plot.style import create_figure, plotting_style, save_figure, style_axes


def plot_statistics(frame, output_stem, *, formats=("png", ), dpi=300):
    """Plot one monitor/bunch table using its saved turns and RMS statistics.

    The saved transverse momenta are normalized mechanical momenta, not
    slopes. Emittances are the saved projected RMS values in those coordinates;
    this function applies display units only, without relativistic normalization.
    """
    if len(frame) == 0:
        return {"status": "skipped", "reason": "No saved statistic samples", "files": []}

    headers = getattr(frame, "headers", {})
    z_coordinate = headers.get("ZCoordinate", "unspecified")
    z_symbol = r"z_{\rm fold}" if z_coordinate == "z_rel_folded_by_ring" else "z"
    panels = (
        (("xAverage", "yAverage"), 1e3, r"$\langle x\rangle,\langle y\rangle$ (mm)", ("$x$", "$y$")),
        (("sigmaX", "sigmaY"), 1e3, r"$\sigma_x,\sigma_y$ (mm)", ("$x$", "$y$")),
        (("xEmittance", "yEmittance"), 1e6, r"$\varepsilon_x,\varepsilon_y$ (mm mrad)", ("$x$", "$y$")),
        (("pxAverage", "pyAverage"), 1e3, r"$\langle P_x/P_0\rangle,\langle P_y/P_0\rangle$ ($10^{-3}$)", ("$x$", "$y$")),
        (("zAverage", ), 1.0, rf"$\langle {z_symbol}\rangle$ (m)", ()),
        (("sigmaZ", ), 1.0, rf"$\sigma_{{{z_symbol}}}$ (m)", ()),
        (("dpAverage", ), 1e3, r"$\langle\delta\rangle$ ($10^{-3}$)", ()),
        (("sigmadp", ), 1e3, r"$\sigma_\delta$ ($10^{-3}$)", ()),
        (("lossPercent", ), 1.0, "Loss (%)", ()),
    )
    required = {"turn"}
    required.update(name for names, _, _, _ in panels for name in names)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("StatMonitor table is missing columns: " + ", ".join(missing))

    turns = np.asarray(frame["turn"], dtype=np.float64)
    if not np.all(np.isfinite(turns)) or np.any(turns < 0) or np.any(turns != np.floor(turns)) or np.any(np.diff(turns) <= 0):
        raise ValueError("StatMonitor turns must be finite, strictly increasing nonnegative integers")

    has_live_particles = np.ones(len(frame), dtype=bool)
    if "numAlive" in frame:
        n_alive = np.asarray(frame["numAlive"], dtype=np.float64)
        has_live_particles = np.isfinite(n_alive) & (n_alive > 0)
        if "numInjected" in frame:
            n_injected = np.asarray(frame["numInjected"], dtype=np.float64)
            if np.all(n_alive == 0) and np.all(n_injected == 0):
                return {"status": "skipped", "reason": "No injected particles in this bunch", "files": []}

    with mpl.rc_context(plotting_style()):
        figure, axes = create_figure(3, 3, (13.5, 10.5))
        try:
            for axis, (names, scale, ylabel, labels) in zip(axes.flat, panels):
                for index, name in enumerate(names):
                    values = np.asarray(frame[name], dtype=np.float64) * scale
                    valid = np.isfinite(values)
                    if name != "lossPercent":
                        # Empty-bunch moment placeholders do not describe a physical centroid or width.
                        valid &= has_live_particles
                    plotted = np.where(valid, values, np.nan)
                    axis.plot(turns,
                              plotted,
                              color=("#0072B2", "#D55E00")[index],
                              linewidth=0.85,
                              marker="." if len(turns) == 1 else None,
                              markersize=3,
                              label=labels[index] if labels else None,
                              rasterized=True)
                axis.set(xlabel="Turn", ylabel=ylabel)
                style_axes(axis)
                axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
                if names == ("lossPercent", ):
                    finite = values[np.isfinite(values)]
                    if len(finite) and np.min(finite) >= 0 and np.max(finite) <= 100:
                        upper = max(1.0, min(100.0, 1.05 * float(np.max(finite))))
                        # Keep a zero-loss line visible above the bottom spine.
                        axis.set_ylim(-0.025 * upper, 1.025 * upper)
                if labels:
                    axis.legend(loc="best", ncols=2, handlelength=1.3, columnspacing=1.0)
            files = save_figure(figure, output_stem, formats=formats, dpi=dpi)
        finally:
            figure.clear()

    return {
        "status": "completed",
        "files": files,
        "sample_count": len(turns),
        "turn_range": [int(turns[0]), int(turns[-1])],
        "z_coordinate": str(z_coordinate),
        "emittance_definition": "Saved projected RMS in (x,Px/P0) and (y,Py/P0); display scale 1e6",
        "moment_sample_count": int(np.count_nonzero(has_live_particles)),
    }
