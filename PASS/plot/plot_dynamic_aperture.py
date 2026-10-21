"""Plot sampled dynamic-aperture results without assuming a connected stable region."""

from collections import OrderedDict
import hashlib

import numpy as np


def _dynamic_aperture_signatures(result, *, include_particles):
    """Check GUI cache inputs once per redraw, including in-place array edits."""
    digest = hashlib.sha256()

    def update(name):
        values = np.asarray(result[name])
        digest.update(repr((name, values.dtype.str, values.shape)).encode("utf-8"))
        if values.dtype.hasobject:
            digest.update(repr(values.tolist()).encode("utf-8"))
        else:
            digest.update(np.ascontiguousarray(values).reshape(-1).view(np.uint8))

    for name in ("initial_coordinates", "initial_valid", "status"):
        update(name)
    cache_signature = digest.digest()
    update("dp_values")
    if include_particles:
        for name in ("lost_turn", "particle_id"):
            update(name)
    return cache_signature, digest.digest()


class _DynamicAperturePlotCache:
    """Page-owned display data for one validated version of an analysis result."""

    def __init__(self, result, max_bytes=32 * 1024 * 1024):
        self.result = result
        initial = np.asarray(result["initial_coordinates"])
        indices = np.flatnonzero(result["initial_valid"])
        self.indices = indices[np.argsort(initial[indices, 5], kind="stable")]
        self.sorted_dp = initial[self.indices, 5]
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._cached_bytes = 0

    def _get(self, key):
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached[0]
        return None

    def _store(self, key, value):

        def nbytes(item):
            if isinstance(item, np.ndarray):
                return item.nbytes
            if isinstance(item, (list, tuple)):
                return sum(nbytes(part) for part in item)
            return len(item) if isinstance(item, str) else 0

        size = nbytes(value)
        if size > self.max_bytes:
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._cached_bytes -= previous[1]
        while self._entries and (self._cached_bytes + size > self.max_bytes or len(self._entries) >= 128):
            _, (_, removed_size) = self._entries.popitem(last=False)
            self._cached_bytes -= removed_size
        self._entries[key] = value, size
        self._cached_bytes += size

    def _get_boundary(self, dp):
        return self._get(("boundary", dp))

    def _store_boundary(self, dp, value):
        self._store(("boundary", dp), value)

    def _get_group(self, dp):
        group = self._get(("group", dp))
        if group is None:
            start = np.searchsorted(self.sorted_dp, dp, side="left")
            end = np.searchsorted(self.sorted_dp, dp, side="right")
            selection = self.indices[start:end]
            initial = np.asarray(self.result["initial_coordinates"])
            group = (initial[selection, 0] * 1e3, initial[selection, 2] * 1e3, np.asarray(self.result["status"])[selection],
                     initial[selection][:, [1, 3, 4]], selection)
            self._store(("group", dp), group)
        return group


def _dp_labels(values):
    """Keep nearby exact groups distinguishable in every plot mode."""
    labels = [f"{dp:.8g}" for dp in values]
    _, label_indices, label_counts = np.unique(labels, return_inverse=True, return_counts=True)
    for index in np.flatnonzero(label_counts[label_indices] > 1):
        labels[index] = f"{values[index]:.17g}"
    return labels


def _prepare_boundary(x, y, status, fixed_coordinates):
    """Classify complete scan cells without discarding holes or islands."""
    if not np.all(fixed_coordinates == fixed_coordinates[:1]):
        return None, "Boundary unavailable: px, py and z must be fixed within the dp group"
    unique_x, x_index = np.unique(x, return_inverse=True)
    unique_y, y_index = np.unique(y, return_inverse=True)
    cells = y_index * len(unique_x) + x_index
    complete = len(cells) == len(unique_x) * len(unique_y) and len(np.unique(cells)) == len(cells)
    if not complete or len(unique_x) < 2 or len(unique_y) < 2:
        return None, "Boundary unavailable: requires a complete, unique x-y grid"
    if not np.any(status == "survived"):
        return None, "No sampled surviving region; no aperture boundary can be drawn"
    if not np.any(status == "lost"):
        return None, "No sampled loss boundary; scan extent is not the DA boundary"

    stable = np.full((len(unique_y), len(unique_x)), np.nan)
    stable[y_index[status == "lost"], x_index[status == "lost"]] = 0.0
    stable[y_index[status == "survived"], x_index[status == "survived"]] = 1.0
    corners = (stable[:-1, :-1], stable[1:, :-1], stable[:-1, 1:], stable[1:, 1:])
    classified = np.logical_and.reduce([np.isfinite(corner) for corner in corners])
    mixed = (np.minimum.reduce(corners) < 0.5) & (np.maximum.reduce(corners) > 0.5)
    if not np.any(classified & mixed):
        return None, "Boundary unavailable: no fully classified cell brackets survival and loss"
    notes = []
    if np.any(stable[[0, -1], :] == 1.0) or np.any(stable[:, [0, -1]] == 1.0):
        notes.append("Survivors reach the scan edge; the unscanned region is not classified.")
    if np.any(~np.isfinite(stable)):
        notes.append("Unknown states are masked; boundaries may have gaps.")
    return (unique_x, unique_y, stable), "\n".join(notes)


def _draw_boundary(ax,
                   x,
                   y,
                   status,
                   fixed_coordinates,
                   *,
                   color="#273449",
                   linestyle="--",
                   label="Sampled aperture boundary",
                   overlay=False,
                   _cache=None,
                   _dp=None):
    """Draw exact cached contour paths or classify the complete input scan."""
    cached = _cache._get_boundary(_dp) if _cache is not None else None
    if cached is None:
        grid, note = _prepare_boundary(x, y, status, fixed_coordinates)
        if grid is None:
            if _cache is not None:
                _cache._store_boundary(_dp, (None, None, note))
            return False, note
        unique_x, unique_y, stable = grid
        contour = ax.contour(unique_x,
                             unique_y,
                             np.ma.masked_invalid(stable),
                             levels=[0.5],
                             colors=[color],
                             linewidths=1.4,
                             linestyles=linestyle,
                             corner_mask=False)
        if _cache is not None:
            _cache._store_boundary(_dp, (contour.allsegs, contour.allkinds, note))
    else:
        segments, kinds, note = cached
        if segments is None:
            return False, note
        from matplotlib.contour import ContourSet

        contour = ContourSet(ax, [0.5], segments, kinds, colors=[color], linewidths=1.4, linestyles=linestyle)
    contour._pass_da_boundary = True
    contour._pass_da_overlay = overlay
    # Contour sticky bounds otherwise clip particles on the scan edge.
    contour.sticky_edges.x.clear()
    contour.sticky_edges.y.clear()
    ax.autoscale_view()
    line, = ax.plot([], [], color=color, linestyle=linestyle, linewidth=1.4, label=label)
    line._pass_da_boundary = True
    line._pass_da_overlay = overlay
    return True, note


def plot_dynamic_aperture(result, *, dp=None, mode="status", ax=None, boundary=True, _cache=None):
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
    if _cache is None:
        initial = np.asarray(result["initial_coordinates"])
        selection = np.asarray(result["initial_valid"]) & (initial[:, 5] == dp)
        x = initial[selection, 0] * 1e3
        y = initial[selection, 2] * 1e3
        status = np.asarray(result["status"])[selection]
        loss = np.asarray(result["lost_turn"])[selection]
        fixed_coordinates = initial[selection][:, [1, 3, 4]] if boundary else None
        particle_ids = np.asarray(result["particle_id"])[selection]
    else:
        x, y, status, fixed_coordinates, selection = _cache._get_group(dp)
        loss = np.asarray(result["lost_turn"])[selection]
        particle_ids = np.asarray(result["particle_id"])[selection]
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
            artist._pass_particle_ids = particle_ids[chosen]
    if mode == "loss_turn":
        chosen = (status == "lost") & (loss >= 0)
        if np.any(chosen):
            artist = ax.scatter(x[chosen], y[chosen], c=loss[chosen], cmap="viridis", s=20, linewidths=0, rasterized=len(x) > 10000)
            artist._pass_particle_ids = particle_ids[chosen]
            colorbar = ax.figure.colorbar(artist, ax=ax, label="Loss turn (simulation index)")
            colorbar.locator = MaxNLocator(integer=True)
            colorbar.update_ticks()
    note = ""
    if boundary:
        drawn, note = _draw_boundary(ax, x, y, status, fixed_coordinates, _cache=_cache, _dp=dp)
        if drawn:
            note = "Dashed: sampled survival/loss transitions; accuracy is limited by grid spacing." + ("\n" + note if note else "")
    metadata = result.get("metadata", {})
    turn = metadata.get("requested_turn", "?")
    monitor = metadata.get("Monitor", "monitor")
    label = _dp_labels(values)[int(np.flatnonzero(values == dp)[0])]
    ax.set_title(f"Initial dp = {label}; turn {turn} at {monitor}")
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


def plot_dynamic_aperture_boundaries(result, *, dp_values=None, ax=None, _cache=None):
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
    if _cache is None:
        _cache = _DynamicAperturePlotCache(result)
    labels = _dp_labels(values)
    colors = plt.get_cmap("tab10" if len(values) <= 10 else "turbo")
    linestyles = ("-", "--", "-.", ":")
    notes = ["Sampled survival/loss boundaries; accuracy is limited by grid spacing."]
    group_notes = {}
    for dp in selected:
        index = int(np.flatnonzero(values == dp)[0])
        color = colors(index) if len(values) <= 10 else colors(0.1 + 0.8 * index / (len(values) - 1))
        linestyle = linestyles[index % len(linestyles)]
        x, y, status, fixed_coordinates, _ = _cache._get_group(dp)
        # Keep every selected scan extent visible, even if no contour is supported.
        ax.update_datalim(np.column_stack((x, y)))
        label = f"dp0 = {labels[index]}"
        drawn, note = _draw_boundary(ax,
                                     x,
                                     y,
                                     status,
                                     fixed_coordinates,
                                     color=color,
                                     linestyle=linestyle,
                                     label=label,
                                     overlay=True,
                                     _cache=_cache,
                                     _dp=dp)
        if not drawn:
            line, = ax.plot([], [], color=color, linestyle="none", marker="x", label=label + " (no boundary)")
            line._pass_da_boundary = True
            line._pass_da_overlay = True
        if note:
            note = " ".join(note.splitlines())
            group_notes.setdefault(note, []).append(label)
    for note, groups in group_notes.items():
        prefix = "All selected dp groups" if len(groups) == len(selected) else ", ".join(groups)
        notes.append(f"{prefix}: {note}")
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
    legend_options = {"loc": "lower left", "bbox_to_anchor": (1.02, 0.0)} if len(selected) > 12 else {"loc": "best"}
    legend = ax.legend(fontsize=8, **legend_options)
    for handle in legend.get_lines():
        handle._pass_da_boundary = True
        handle._pass_da_overlay = True
    return ax
