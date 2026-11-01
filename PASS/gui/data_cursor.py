"""Full-data hit testing and transient crosshairs for Matplotlib canvases."""

from contextlib import contextmanager

import numpy as np


class CursorSeries:
    """Locate actual samples by nearest X or screen distance, with cached indexing."""

    def __init__(self, x, y, formatter=None, label="数据点", *, priority=0, hit_radius=14., lookup_mode="xy"):
        self.x, self.y = np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)
        if self.x.shape != self.y.shape:
            raise ValueError("Cursor coordinates must have matching shapes")
        self.formatter, self.label = formatter, label
        if lookup_mode not in {"xy", "x"}:
            raise ValueError("Cursor lookup mode must be 'xy' or 'x'")
        self.priority, self.hit_radius = priority, hit_radius
        self.lookup_mode = lookup_mode
        self._key = None

    def nearest(self, ax, pixel_x, pixel_y, radius=10.):
        key = (tuple(ax.get_xlim()), tuple(ax.get_ylim()), tuple(ax.bbox.bounds), ax.get_xscale(), ax.get_yscale(), radius)
        if key != self._key:
            pixels = ax.transData.transform(np.column_stack((self.x, self.y)))
            valid = np.isfinite(pixels).all(axis=1) & np.isfinite(self.x) & np.isfinite(self.y)
            valid &= (pixels[:, 0] >= ax.bbox.x0) & (pixels[:, 0] <= ax.bbox.x1)
            valid &= (pixels[:, 1] >= ax.bbox.y0) & (pixels[:, 1] <= ax.bbox.y1)
            indices = np.flatnonzero(valid)
            pixels = pixels[indices]
            if self.lookup_mode == "x":
                # Curve lookup needs only X ordering, not the scatter spatial grid.
                self._pixels, self._indices = pixels, indices
                self._x_order = np.argsort(pixels[:, 0], kind="stable")
                self._x_pixels = pixels[self._x_order, 0]
            else:
                self._origin = np.array([ax.bbox.x0 - 2 * radius, ax.bbox.y0 - 2 * radius])
                self._cell_size = radius
                self._stride = int(np.ceil((ax.bbox.width + 4 * radius) / radius)) + 1
                cells = np.floor((pixels - self._origin) / radius).astype(np.int64)
                codes = cells[:, 0] + cells[:, 1] * self._stride
                order = np.argsort(codes, kind="stable")
                self._codes, self._pixels, self._indices = codes[order], pixels[order], indices[order]
            self._key = key
        if self.lookup_mode == "x":
            if not len(self._indices):
                return None
            x_pixels = self._x_pixels
            offset = int(np.searchsorted(x_pixels, pixel_x))
            neighbors = np.array([max(0, offset - 1), min(len(x_pixels) - 1, offset)])
            nearest = int(neighbors[np.argmin(abs(x_pixels[neighbors] - pixel_x))])
            distance = float(abs(x_pixels[nearest] - pixel_x))
            # Equal-X samples are distinguished by their distance to the mouse Y.
            first, last = np.searchsorted(x_pixels, x_pixels[nearest], side="left"), np.searchsorted(x_pixels, x_pixels[nearest], side="right")
            candidates = self._x_order[first:last]
            if len(candidates) == 1:
                closest = int(candidates[0])
            else:
                distances = abs(self._pixels[candidates, 1] - pixel_y)
                tied = candidates[distances == distances.min()]
                # Preserve the spatial index's tie order without building its full grid.
                cells = np.floor((self._pixels[tied, 1] - (ax.bbox.y0 - 2 * radius)) / radius)
                closest = int(tied[np.argmin(cells)])
            index = int(self._indices[closest])
            if not ax.bbox.contains(*self._pixels[closest]):
                return None
            return float(np.hypot(distance, self._pixels[closest, 1] - pixel_y)), index
        cell = np.floor((np.array([pixel_x, pixel_y]) - self._origin) / self._cell_size).astype(np.int64)
        codes = np.array([cell[0] + dx + (cell[1] + dy) * self._stride for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
        starts, ends = np.searchsorted(self._codes, codes, side="left"), np.searchsorted(self._codes, codes, side="right")
        candidates = np.concatenate([np.arange(start, end) for start, end in zip(starts, ends)])
        if not candidates.size:
            return None
        distances = np.sum((self._pixels[candidates] - [pixel_x, pixel_y])**2, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] > radius**2:
            return None
        return float(np.sqrt(distances[nearest])), int(self._indices[candidates[nearest]])

    def describe(self, index):
        if self.formatter is not None:
            return str(self.formatter(index))
        return f"{self.label}\nx = {self.x[index]:.10g}\ny = {self.y[index]:.10g}"


def set_cursor_sources(ax, sources):
    """Register full-resolution source arrays, replacing any previous sources."""
    ax._pass_cursor_sources = list(sources)


class FigureDataCursor:
    """Follow the pointer and mark actual data separately, excluding saved figures."""

    def __init__(self, canvas, toolbar=None, *, add_action=True, controls_toolbar=None):
        self.canvas, self.toolbar = canvas, toolbar
        self.enabled = True
        self._suppressed = 0
        self._background = None
        self._artists = {}
        self._current = None
        self._highlight = None
        self._pressed_buttons = set()
        self.action = None
        action_toolbar = controls_toolbar if controls_toolbar is not None else toolbar
        if action_toolbar is not None and add_action:
            self.action = action_toolbar.addAction("十字线 / 数据提示")
            self.action.setCheckable(True)
            self.action.setChecked(True)
            self.action.setToolTip("十字线始终跟随鼠标；放大显示曲线对应横坐标附近的实际采样点或邻近散点，并在交点旁显示数据值。")
            self.action.toggled.connect(self.set_enabled)
        if hasattr(canvas, "setMouseTracking"):
            canvas.setMouseTracking(True)
        self._connections = [
            canvas.mpl_connect("motion_notify_event", self._move),
            canvas.mpl_connect("figure_leave_event", self._leave),
            canvas.mpl_connect("draw_event", self._draw),
            canvas.mpl_connect("button_press_event", self._press),
            canvas.mpl_connect("button_release_event", self._release)
        ]

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        if self.action is not None and self.action.isChecked() != self.enabled:
            self.action.setChecked(self.enabled)
        if not self.enabled:
            self._leave()

    def clear(self):
        self._current = None
        self._highlight = None
        for artists in self._artists.values():
            for artist in artists:
                try:
                    artist.remove()
                except (ValueError, NotImplementedError):
                    pass
        self._artists.clear()
        self._background = None

    def disconnect(self):
        self.clear()
        for connection in self._connections:
            self.canvas.mpl_disconnect(connection)
        self._connections.clear()

    @contextmanager
    def hidden(self):
        self._suppressed += 1
        self._leave()
        try:
            yield
        finally:
            self._suppressed -= 1
            self._background = None
            self.canvas.draw_idle()

    def _leave(self, event=None):
        self._current = None
        self._highlight = None
        self._paint()

    def _press(self, event):
        self._pressed_buttons.add(event.button)
        self._leave()

    def _release(self, event):
        from matplotlib.backend_bases import LocationEvent

        self._pressed_buttons.discard(event.button)
        self._background = None
        # Toolbar release handlers have changed the limits; recompute pointer coordinates.
        self._move(LocationEvent("motion_notify_event", self.canvas, event.x, event.y))

    def _draw(self, event):
        if self.canvas.is_saving():
            self._background = None
            return
        current_axes = set(self.canvas.figure.axes)
        for ax in tuple(self._artists):
            if ax not in current_axes:
                self._artists.pop(ax)
        if self._current is not None and self._current[0] not in current_axes:
            self._current = None
            self._highlight = None
        self._background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._paint()

    def _sources(self, ax):
        if hasattr(ax, "_pass_cursor_sources"):
            return ax._pass_cursor_sources
        sources = []
        for line in ax.lines:
            if line.get_visible():
                sources.append(CursorSeries(line.get_xdata(), line.get_ydata(), label=line.get_label().lstrip("_"), lookup_mode="x"))
        for collection in ax.collections:
            offsets = collection.get_offsets()
            if collection.get_visible() and offsets.ndim == 2 and offsets.shape[1] == 2 and len(offsets) > 1:
                sources.append(CursorSeries(offsets[:, 0], offsets[:, 1]))
        set_cursor_sources(ax, sources)
        return sources

    def _move(self, event):
        buttons = getattr(event, "buttons", None)
        if buttons is not None:
            self._pressed_buttons = set(buttons)
        ax = event.inaxes
        if (not self.enabled or self._suppressed or ax is None or not ax.axison or getattr(ax, "_colorbar", None) is not None or event.xdata is None
                or event.ydata is None or self._pressed_buttons):
            self._leave()
            return
        best = None
        pixel_ratio = float(getattr(self.canvas, "device_pixel_ratio", 1.))
        for source in self._sources(ax):
            # Matplotlib mouse events use physical pixels; thresholds are logical pixels.
            hit = source.nearest(ax, event.x, event.y, radius=source.hit_radius * pixel_ratio)
            if hit is not None:
                rank = (source.priority, -hit[0])
                if best is None or rank >= best[0]:
                    best = (rank, source, hit[1])
        x, y = event.xdata, event.ydata
        self._highlight = None
        if best is None:
            text = f"鼠标坐标\n{ax.get_xlabel() or 'x'} = {x:.10g}\n{ax.get_ylabel() or 'y'} = {y:.10g}"
        else:
            _, source, index = best
            # Highlight the true sample without changing the pointer crosshair.
            self._highlight = source.x[index], source.y[index]
            text = ("最近采样点 · " if source.lookup_mode == "x" else "") + source.describe(index)
        self._current = ax, x, y, text
        self._paint()

    def _paint(self):
        if self._background is None:
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self._background)
        if self._current is not None and self.enabled and not self._suppressed:
            from matplotlib.lines import Line2D

            ax, x, y, text = self._current
            if ax not in self._artists or any(artist not in ax.get_children() for artist in self._artists[ax]):
                vertical = Line2D([x, x], [0, 1], transform=ax.get_xaxis_transform(), linewidth=.65, alpha=.75, visible=False)
                horizontal = Line2D([0, 1], [y, y], transform=ax.get_yaxis_transform(), linewidth=.65, alpha=.75, visible=False)
                marker = Line2D([x], [y], transform=ax.transData, linestyle="none", marker="o", markersize=8., markeredgewidth=1.8, visible=False)
                ax.add_artist(vertical)
                ax.add_artist(horizontal)
                ax.add_artist(marker)
                annotation = ax.annotate("", (x, y),
                                         xytext=(10, 10),
                                         textcoords="offset points",
                                         fontsize=9,
                                         visible=False,
                                         bbox=dict(boxstyle="round,pad=.35", alpha=.95))
                for artist in (vertical, horizontal, annotation, marker):
                    artist.set_in_layout(False)
                    if artist is not annotation:
                        artist.set_clip_path(ax.patch)
                self._artists[ax] = vertical, horizontal, annotation, marker
            vertical, horizontal, annotation, marker = self._artists[ax]
            vertical.set_xdata([x, x])
            horizontal.set_ydata([y, y])
            if self._highlight is not None:
                marker.set_data([self._highlight[0]], [self._highlight[1]])
            annotation.xy = (x, y)
            annotation.set_text(text)
            annotation.set_fontproperties(ax.xaxis.label.get_fontproperties())
            annotation.set_fontfamily(list(dict.fromkeys([*ax.xaxis.label.get_fontfamily(), "Microsoft YaHei", "DejaVu Sans"])))
            annotation.set_fontsize(9)
            px, py = ax.transData.transform((x, y))
            right = px > (ax.bbox.x0 + ax.bbox.x1) / 2
            upper = py > (ax.bbox.y0 + ax.bbox.y1) / 2
            annotation.set_position((-10 if right else 10, -10 if upper else 10))
            annotation.set_ha("right" if right else "left")
            annotation.set_va("top" if upper else "bottom")
            dark = np.dot(ax.get_facecolor()[:3], [.2126, .7152, .0722]) < .5
            color = "#c4cbd2" if dark else "#697480"
            for artist in (vertical, horizontal):
                artist.set_color(color)
            marker.set_color("#ffcc66" if dark else "#a65000")
            annotation.set_color(ax.xaxis.label.get_color())
            marker.set_markerfacecolor(ax.get_facecolor())
            annotation.get_bbox_patch().set_facecolor(ax.get_facecolor())
            annotation.get_bbox_patch().set_edgecolor(color)
            self._position_tooltip(annotation, ax)
            # Visibility lasts only for this blit: all savefig paths omit the overlay.
            for artist in (vertical, horizontal, marker, annotation) if self._highlight is not None else (vertical, horizontal, annotation):
                artist.set_visible(True)
                ax.draw_artist(artist)
                artist.set_visible(False)
        self.canvas.blit(self.canvas.figure.bbox)

    def _position_tooltip(self, annotation, ax):
        renderer = self.canvas.get_renderer()
        annotation.update_positions(renderer)
        annotation.update_bbox_position_size(renderer)
        bounds = annotation.get_bbox_patch().get_window_extent(renderer)
        padding = renderer.points_to_pixels(2.)
        limits = ax.bbox
        if bounds.width + 2 * padding > limits.width or bounds.height + 2 * padding > limits.height:
            limits = self.canvas.figure.bbox
        left, right = limits.x0 + padding, limits.x1 - padding - bounds.width
        bottom, top = limits.y0 + padding, limits.y1 - padding - bounds.height
        x = np.clip(bounds.x0, left, right) if right >= left else left
        y = np.clip(bounds.y0, bottom, top) if top >= bottom else top
        # Offset points position the tooltip; the crosshair and data marker stay fixed.
        dx, dy = (np.array([x, y]) - bounds.p0) / renderer.points_to_pixels(1.)
        annotation.set_position(np.array(annotation.get_position()) + [dx, dy])
