"""Shared, headless scientific plotting style and non-destructive exports."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


def plotting_style():
    """Return local rc settings without changing an application's backend."""
    return {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "path.simplify": False,
        "agg.path.chunksize": 0,
    }


def coordinate_spec():
    """Scales convert saved coordinates to the displayed units, without folding."""
    return {
        "x": (1e3, r"$x$ (mm)"),
        "y": (1e3, r"$y$ (mm)"),
        "z": (1.0, r"$z_{\rm rel}$ (m)"),
        "px": (1e3, r"$P_x/P_0$ ($10^{-3}$)"),
        "py": (1e3, r"$P_y/P_0$ ($10^{-3}$)"),
        "dp": (1e3, r"$\delta$ ($10^{-3}$)"),
    }


def create_figure(rows, cols, figsize):
    figure = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots(rows, cols, squeeze=False)


def style_axes(axis):
    axis.tick_params(direction="in", top=True, right=True, labelbottom=True)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.ticklabel_format(style="sci", scilimits=(-3, 4), useMathText=True, useOffset=False)
    axis.margins(x=0.025, y=0.05)


def available_path(path):
    """Keep previous exports; find an unused filename for a new rendering."""
    path = Path(path)
    candidate, index = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        index += 1
    return candidate


def save_figure(figure, output_stem, formats=("png", ), dpi=300):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    files = []
    try:
        for extension in formats:
            # Append rather than replace a suffix: monitor names can contain dots.
            target = available_path(Path(str(output_stem) + "." + extension))
            figure.savefig(target, dpi=dpi, format=extension, facecolor="white")
            files.append(str(target.resolve()))
    finally:
        figure.clear()
    return files
