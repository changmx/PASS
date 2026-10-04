"""Plot sampled dynamic-aperture results without assuming a connected stable region."""

import numpy as np


def _draw_boundary(ax, x, y, status, fixed_coordinates, *, color="#273449", linestyle="--", label="Sampled aperture boundary", overlay=False):
    """Draw only transitions supported by fully classified Cartesian cells."""
    if not np.all(fixed_coordinates == fixed_coordinates[:1]):
        return False, "Boundary unavailable: px, py and z must be fixed within the dp group"
    unique_x, x_index = np.unique(x, return_inverse=True)
    unique_y, y_index = np.unique(y, return_inverse=True)
    cells = y_index * len(unique_x) + x_index
    complete = len(cells) == len(unique_x) * len(unique_y) and len(np.unique(cells)) == len(cells)
    if not complete or len(unique_x) < 2 or len(unique_y) < 2:
        return False, "Boundary unavailable: requires a complete, unique x-y grid"
    if not np.any(status == "survived"):
        return False, "No sampled surviving region; no aperture boundary can be drawn"
    if not np.any(status == "lost"):
        return False, "No sampled loss boundary; scan extent is not the DA boundary"

    stable = np.full((len(unique_y), len(unique_x)), np.nan)
    stable[y_index[status == "lost"], x_index[status == "lost"]] = 0.0
    stable[y_index[status == "survived"], x_index[status == "survived"]] = 1.0
    corners = (stable[:-1, :-1], stable[1:, :-1], stable[:-1, 1:], stable[1:, 1:])
    classified = np.logical_and.reduce([np.isfinite(corner) for corner in corners])
    mixed = (np.minimum.reduce(corners) < 0.5) & (np.maximum.reduce(corners) > 0.5)
    if not np.any(classified & mixed):
        return False, "Boundary unavailable: no fully classified cell brackets survival and loss"

    contour = ax.contour(unique_x,
                         unique_y,
                         np.ma.masked_invalid(stable),
                         levels=[0.5],
                         colors=[color],
                         linewidths=1.4,
                         linestyles=linestyle,
                         corner_mask=False)
    contour._pass_da_boundary = True
    contour._pass_da_overlay = overlay
    # Contour sticky bounds otherwise clip particles on the scan edge.
    contour.sticky_edges.x.clear()
    contour.sticky_edges.y.clear()
    ax.autoscale_view()
    line, = ax.plot([], [], color=color, linestyle=linestyle, linewidth=1.4, label=label)
    line._pass_da_boundary = True
    line._pass_da_overlay = overlay
    notes = []
    if np.any(stable[[0, -1], :] == 1.0) or np.any(stable[:, [0, -1]] == 1.0):
        notes.append("Survivors reach the scan edge; the unscanned region is not classified.")
    if np.any(~np.isfinite(stable)):
        notes.append("Unknown states are masked; boundaries may have gaps.")
    return True, "\n".join(notes)


def plot_dynamic_aperture(result, *, dp=None, mode="status", ax=None, boundary=True):
    """Plot initial x-y points for one exact initial dp group; return the Axes.

    ``mode`` is ``status`` or ``loss_turn``. Positions are displayed in mm and
    loss colors use absolute simulation turn. Dashed contours locate sampled
    survived/lost transitions on a complete Cartesian grid at fixed px, py and z.
    They are shown by default and can be hidden with ``boundary=False``. They do
    not replace the samples, close the scan edge, bridge unknown states, or
    suppress islands and holes. No contour is inferred when no fully classified
    cell brackets survival and loss, including all-surviving/all-lost scans.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if mode not in ("status", "loss_turn"):
        raise ValueError("mode must be 'status' or 'loss_turn'")
    values = np.asarray(result["dp_values"])
    if not len(values):
        raise ValueError("No valid initial coordinates are available for plotting")
    if dp is None:
        dp = float(values[0])
    if not np.any(values == dp):
        raise ValueError("dp must equal an available initial dp group")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    initial = np.asarray(result["initial_coordinates"])
    selection = np.asarray(result["initial_valid"]) & (initial[:, 5] == dp)
    x = initial[selection, 0] * 1e3
    y = initial[selection, 2] * 1e3
    status = np.asarray(result["status"])[selection]
    loss = np.asarray(result["lost_turn"])[selection]
    styles = {
        "survived": ("#208654", "o", "Survived"),
        "lost": ("#d4533c", "x", "Lost"),
        "invalid": ("#b13da0", "+", "Numerically invalid"),
        "incomplete": ("#e0a128", "s", "Incomplete coverage"),
        "unavailable": ("#80858a", ".", "Unavailable"),
    }
    for name, (color, marker, label) in styles.items():
        chosen = status == name
        if mode == "loss_turn" and name == "lost":
            chosen &= loss < 0
            label = "Lost (turn unavailable)"
        if np.any(chosen):
            artist = ax.scatter(x[chosen], y[chosen], c=color, marker=marker, s=18, linewidths=0.8, label=label, rasterized=len(x) > 10000)
            artist._pass_particle_ids = np.asarray(result["particle_id"])[selection][chosen]
    if mode == "loss_turn":
        chosen = (status == "lost") & (loss >= 0)
        if np.any(chosen):
            artist = ax.scatter(x[chosen], y[chosen], c=loss[chosen], cmap="viridis", s=20, linewidths=0, rasterized=len(x) > 10000)
            artist._pass_particle_ids = np.asarray(result["particle_id"])[selection][chosen]
            colorbar = ax.figure.colorbar(artist, ax=ax, label="Loss turn (simulation index)")
            colorbar.locator = MaxNLocator(integer=True)
            colorbar.update_ticks()
    note = ""
    if boundary:
        drawn, note = _draw_boundary(ax, x, y, status, initial[selection][:, [1, 3, 4]])
        if drawn:
            note = "Dashed: sampled survival/loss transitions; accuracy is limited by grid spacing." + ("\n" + note if note else "")
    metadata = result.get("metadata", {})
    turn = metadata.get("requested_turn", "?")
    monitor = metadata.get("Monitor", "monitor")
    ax.set_title(f"Initial dp = {dp:.8g}; turn {turn} at {monitor}")
    ax.set_xlabel("Initial x (mm)")
    ax.set_ylabel("Initial y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    if note:
        ax.text(0.0, -0.17, note, transform=ax.transAxes, fontsize=8, va="top", wrap=True)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(loc="best", fontsize=8)
        for handle in legend.get_lines():
            if handle.get_label() == "Sampled aperture boundary":
                handle._pass_da_boundary = True
    return ax


def plot_dynamic_aperture_boundaries(result, *, dp_values=None, ax=None):
    """Overlay boundaries of selected exact initial-dp groups; return the Axes.

    ``dp_values=None`` selects every group; otherwise pass a nonempty sequence
    of distinct values from ``result["dp_values"]``. Each group retains its
    color and line style when a subset is selected. Only contours are drawn:
    all holes, islands and open components use the same classified Cartesian
    cells as :func:`plot_dynamic_aperture`. Unknown states stay masked. Groups
    without a supported boundary remain identified in the legend and notes.
    """
    import matplotlib.pyplot as plt

    values = np.asarray(result["dp_values"], dtype=float)
    if not len(values):
        raise ValueError("No valid initial coordinates are available for plotting")
    try:
        selected = values if dp_values is None else np.asarray(dp_values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("dp_values must be a nonempty sequence of available initial dp groups") from error
    if selected.ndim != 1 or not len(selected) or not np.all(np.isfinite(selected)):
        raise ValueError("dp_values must be a nonempty sequence of finite initial dp groups")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("dp_values must not contain duplicate initial dp groups")
    if not np.all(np.isin(selected, values)):
        raise ValueError("dp_values must equal available initial dp groups")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    initial = np.asarray(result["initial_coordinates"])
    initial_valid = np.asarray(result["initial_valid"])
    status = np.asarray(result["status"])
    indices = np.flatnonzero(initial_valid)
    indices = indices[np.argsort(initial[indices, 5], kind="stable")]
    sorted_dp = initial[indices, 5]
    labels = [f"{dp:.8g}" for dp in values]
    _, label_indices, label_counts = np.unique(labels, return_inverse=True, return_counts=True)
    for index in np.flatnonzero(label_counts[label_indices] > 1):
        labels[index] = f"{values[index]:.17g}"
    colors = plt.get_cmap("tab10" if len(values) <= 10 else "turbo")
    linestyles = ("-", "--", "-.", ":")
    notes = ["Sampled survival/loss boundaries; accuracy is limited by grid spacing."]
    for dp in selected:
        index = int(np.flatnonzero(values == dp)[0])
        color = colors(index) if len(values) <= 10 else colors(0.1 + 0.8 * index / (len(values) - 1))
        linestyle = linestyles[index % len(linestyles)]
        start = np.searchsorted(sorted_dp, dp, side="left")
        end = np.searchsorted(sorted_dp, dp, side="right")
        selection = indices[start:end]
        x = initial[selection, 0] * 1e3
        y = initial[selection, 2] * 1e3
        # Keep every selected scan extent visible, even if no contour is supported.
        ax.update_datalim(np.column_stack((x, y)))
        label = f"dp0 = {labels[index]}"
        drawn, note = _draw_boundary(ax,
                                     x,
                                     y,
                                     status[selection],
                                     initial[selection][:, [1, 3, 4]],
                                     color=color,
                                     linestyle=linestyle,
                                     label=label,
                                     overlay=True)
        if not drawn:
            line, = ax.plot([], [], color=color, linestyle="none", marker="x", label=label + " (no boundary)")
            line._pass_da_boundary = True
            line._pass_da_overlay = True
        if note:
            note = " ".join(note.splitlines())
            notes.append(f"{label}: {note}")
    metadata = result.get("metadata", {})
    turn = metadata.get("requested_turn", "?")
    monitor = metadata.get("Monitor", "monitor")
    ax.set_title(f"Initial dp boundary comparison; turn {turn} at {monitor}")
    ax.set_xlabel("Initial x (mm)")
    ax.set_ylabel("Initial y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.autoscale_view()
    ax.text(0.0, -0.17, "\n".join(notes), transform=ax.transAxes, fontsize=8, va="top", wrap=True)
    legend = ax.legend(loc="best", fontsize=8)
    for handle in legend.get_lines():
        handle._pass_da_boundary = True
        handle._pass_da_overlay = True
    return ax
