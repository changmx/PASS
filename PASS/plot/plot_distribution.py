"""Six-panel distributions with shared coordinates and per-figure color scales."""

import matplotlib as mpl
from matplotlib.colors import Normalize
import numpy as np

from PASS.plot.style import coordinate_spec, create_figure, plotting_style, save_figure, style_axes


def projection_pairs():
    return (("x", "px"), ("y", "py"), ("z", "dp"), ("x", "y"), ("z", "x"), ("z", "y"))


def snapshot_coordinates(frame):
    required = (*coordinate_spec(), "tag")
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Distribution is missing columns: {sorted(missing)}")
    alive = np.asarray(frame["tag"]) > 0
    valid = alive.copy()
    for name in coordinate_spec():
        valid &= np.isfinite(np.asarray(frame[name]))
    values = {name: np.asarray(frame[name])[valid] * scale for name, (scale, _) in coordinate_spec().items()}
    return values, int(np.count_nonzero(alive & ~valid))


def expand_limits(low, high):
    padding = (high - low) * 0.025 if high > low else max(abs(low) * 0.025, 1e-9)
    return [float(low - padding), float(high + padding)]


def snapshot_histograms(values, limits, bins):
    return [np.histogram2d(values[x], values[y], bins=bins, range=(limits[x], limits[y])) for x, y in projection_pairs()]


def plot_distribution(frame, output_stem, *, limits, bins=80, formats=("png", ), dpi=300):
    """Use shared coordinate limits and a count scale local to this six-panel figure."""
    values, invalid_count = snapshot_coordinates(frame)
    if not len(values["x"]):
        return {"status": "skipped", "reason": "No finite live particles", "invalid_live_count": invalid_count}
    histograms = snapshot_histograms(values, limits, bins)
    count_max = max(1.0, *(float(counts.max()) for counts, _, _ in histograms))
    with mpl.rc_context(plotting_style()):
        figure, axes = create_figure(2, 3, (13.5, 8.0))
        try:
            spec = coordinate_spec()
            cmap = mpl.colormaps["jet"].with_extremes(bad="white")
            norm = Normalize(vmin=0, vmax=max(1, count_max))
            for axis, (x, y), (counts, x_edges, y_edges) in zip(axes.flat, projection_pairs(), histograms):
                mesh = axis.pcolormesh(x_edges, y_edges, np.ma.masked_equal(counts.T, 0), cmap=cmap, norm=norm, rasterized=True)
                axis.set(xlabel=spec[x][1], ylabel=spec[y][1], xlim=limits[x], ylim=limits[y])
                style_axes(axis)
            figure.colorbar(mesh, ax=list(axes.flat), fraction=0.025, pad=0.025, label=r"$N_{\rm bin}$")
            files = save_figure(figure, output_stem, formats=formats, dpi=dpi)
        finally:
            figure.clear()
    return {
        "files": files,
        "particle_count": len(values["x"]),
        "invalid_live_count": invalid_count,
        "limits": limits,
        "bins": bins,
        "count_max": float(count_max),
        "color_scale_scope": "figure",
        "colormap": "jet",
        "z_coordinate": "continuous z_rel",
        "selection": "tag > 0 and finite six-dimensional coordinates"
    }
