"""Absolute-tune resonance diagrams for measured working points."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, gcd, isfinite
from pathlib import Path

import numpy as np


_ORDER_COLORS = (
    "#e41a1c",
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
)


def _ranges(qx_range, qy_range):
    values = tuple(float(v) for r in (qx_range, qy_range) for v in r)
    if len(values) != 4 or not all(isfinite(v) for v in values):
        raise ValueError("Tune ranges must contain two finite endpoints each.")
    x0, x1, y0, y1 = values
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Each tune range must be increasing.")
    return values


@dataclass(frozen=True)
class ResonanceLine:
    """Integer resonance m*Qx + n*Qy = l; geometry is deduplicated by key."""
    m: int
    n: int
    l: int

    def __post_init__(self):
        if any(type(v) is not int for v in (self.m, self.n, self.l)) or (self.m == self.n == 0):
            raise ValueError("m, n, l must be integers; m and n cannot both be zero.")

    @property
    def order(self):
        return abs(self.m) + abs(self.n)

    @property
    def key(self):
        divisor = gcd(gcd(abs(self.m), abs(self.n)), abs(self.l))
        sign = -1 if self.m < 0 or (self.m == 0 and self.n < 0) else 1
        return tuple(sign * v // divisor for v in (self.m, self.n, self.l))

    @property
    def kind(self):
        return "single" if not self.m or not self.n else "sum" if self.m * self.n > 0 else "diff"

    @property
    def label(self):
        return f"{self.m} Qx {'+' if self.n >= 0 else '-'} {abs(self.n)} Qy = {self.l}"

    def segment(self, qx_range, qy_range):
        x0, x1, y0, y1 = _ranges(qx_range, qy_range)
        points = []
        tolerance = 1e-12 * max(1., abs(x0), abs(x1), abs(y0), abs(y1))

        def add(x, y):
            if x0 - tolerance <= x <= x1 + tolerance and y0 - tolerance <= y <= y1 + tolerance:
                point = (max(x0, min(x1, x)), max(y0, min(y1, y)))
                if not any(abs(x - px) + abs(y - py) <= tolerance for px, py in points):
                    points.append(point)

        if self.n:
            add(x0, (self.l - self.m * x0) / self.n)
            add(x1, (self.l - self.m * x1) / self.n)
        if self.m:
            add((self.l - self.n * y0) / self.m, y0)
            add((self.l - self.n * y1) / self.m, y1)
        return tuple(points[:2]) if len(points) >= 2 else None


def resonance_lines(orders, qx_range, qy_range, kinds=("single", "sum", "diff")):
    """Visible lines, assigned to their lowest integer-coefficient order.

    Order is |m|+|n| after reducing the *triple* (m,n,l). This retains lines
    such as 2*Qx=1 while merging 2*Qx=2 with Qx=1. Work is bounded before
    enumerating pathological tune windows. Supported orders are 1 through 12.
    """
    x0, x1, y0, y1 = _ranges(qx_range, qy_range)
    orders = tuple(orders)
    if any(type(order) is not int or not 1 <= order <= 12 for order in orders):
        raise ValueError("Resonance orders must be integers from 1 to 12.")
    kinds = set(kinds)
    if not kinds <= {"single", "sum", "diff"}:
        raise ValueError("Unknown resonance kind.")
    lines, examined = {}, 0
    for order in sorted(set(orders)):
        for m in range(order + 1):
            for n in sorted({order - m, m - order}):
                if m == 0 and n <= 0:
                    continue
                direction = ResonanceLine(m, n, 0)
                if direction.kind not in kinds:
                    continue
                corners = (m*x0+n*y0, m*x0+n*y1, m*x1+n*y0, m*x1+n*y1)
                if not all(isfinite(v) for v in corners):
                    raise ValueError("Tune range is too large.")
                lo, hi = ceil(min(corners)), floor(max(corners))
                examined += max(0, hi - lo + 1)
                if examined > 25000:
                    raise ValueError("Too many resonance lines; reduce the tune range or selected orders.")
                for integer in range(lo, hi + 1):
                    line = ResonanceLine(m, n, integer)
                    key = line.key
                    if abs(key[0]) + abs(key[1]) != order:
                        continue
                    if key not in lines and line.segment(qx_range, qy_range) is not None:
                        lines[key] = line
    return list(lines.values())


def _farey(order: int) -> list[tuple[int, int]]:
    sequence = [(0, 1)]
    a, b, c, d = 0, 1, 1, order
    while c <= order:
        multiplier = (order + b) // d
        a, b, c, d = c, d, multiplier * c - a, multiplier * d - b
        sequence.append((a, b))
    return sequence


def _draw_line(
    ax: object,
    n: int,
    m: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    color: str,
    linewidth: float,
    drawn_lines: set[tuple[int, int, int]],
    kind: str,
) -> None:
    """Draw every visible integer line of ``n Qx + m Qy = p`` once."""
    if n == 0 and m == 0:
        return

    resonance_kind = "single" if n == 0 or m == 0 else "sum" if n * m > 0 else "diff"
    if kind == "sum" and resonance_kind == "diff":
        return
    if kind == "diff" and resonance_kind == "sum":
        return

    corners = (
        n * x_min + m * y_min,
        n * x_min + m * y_max,
        n * x_max + m * y_min,
        n * x_max + m * y_max,
    )
    for integer in range(int(np.floor(min(corners))), int(np.ceil(max(corners))) + 1):
        divisor = gcd(gcd(abs(n), abs(m)), abs(integer)) if integer else gcd(abs(n), abs(m))
        divisor = divisor or 1
        normalised = (n // divisor, m // divisor, integer // divisor)
        if normalised[0] < 0 or (normalised[0] == 0 and normalised[1] < 0):
            normalised = tuple(-value for value in normalised)
        if normalised in drawn_lines:
            continue
        drawn_lines.add(normalised)

        linestyle = "--" if resonance_kind == "diff" else "-"
        if n == 0:
            y_value = integer / m
            if y_min <= y_value <= y_max:
                ax.plot((x_min, x_max), (y_value, y_value), color=color, linewidth=linewidth, linestyle=linestyle)
        elif m == 0:
            x_value = integer / n
            if x_min <= x_value <= x_max:
                ax.plot((x_value, x_value), (y_min, y_max), color=color, linewidth=linewidth, linestyle=linestyle)
        else:
            x_values = np.array((x_min, x_max))
            y_values = (integer - n * x_values) / m
            if (y_values >= y_min).any() and (y_values <= y_max).any():
                y_values = np.clip(y_values, y_min, y_max)
                x_values = (integer - m * y_values) / n
                ax.plot(x_values, y_values, color=color, linewidth=linewidth, linestyle=linestyle)


def _draw_order(
    ax: object,
    order: int,
    color: str,
    drawn_lines: set[tuple[int, int, int]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    kind: str,
) -> None:
    """Generate the former Farey-sequence resonance directions for one order."""
    linewidth = max(0.4, 1.5 - order * 0.12)
    for h, k in _farey(order):
        for p, q in _farey(order):
            c, a, b = p * h, k * p, q - k * p
            if a > 0:
                _draw_line(ax, b, a, x_min, x_max, y_min, y_max, color, linewidth, drawn_lines, kind)
                _draw_line(ax, a, b, x_min, x_max, y_min, y_max, color, linewidth, drawn_lines, kind)
                _draw_line(ax, a - c, -(c + b), x_min, x_max, y_min, y_max, color, linewidth, drawn_lines, kind)
                _draw_line(ax, b, -a, x_min, x_max, y_min, y_max, color, linewidth, drawn_lines, kind)
                _draw_line(ax, a, -b, x_min, x_max, y_min, y_max, color, linewidth, drawn_lines, kind)
                _draw_line(ax, a - c, b - c, x_min, x_max, y_min, y_max, color, linewidth, drawn_lines, kind)
            if q == k and p == 1:
                break


def plot_tune_diagram(
    nat_tunes: tuple[float, float] | None = None,
    *,
    max_order: int = 4,
    qx_range: tuple[float, float] | None = None,
    qy_range: tuple[float, float] | None = None,
    kind: str = "all",
    ax: object | None = None,
    output_path: str | Path | None = None,
    show_legend: bool = True,
    legend_outside: bool = False,
) -> object:
    """Draw absolute-tune resonance lines and return the matplotlib axes.

    ``kind`` accepts ``"all"``, ``"sum"``, or ``"diff"``. The ``"sum"``
    and ``"diff"`` selections retain single-plane resonances; sum resonances
    use solid lines and difference resonances use dashed lines. Set
    ``legend_outside=True`` to place the legend to the right of the axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if max_order < 1:
        raise ValueError("max_order must be at least 1.")
    if kind not in {"all", "sum", "diff"}:
        raise ValueError("kind must be 'all', 'sum', or 'diff'.")

    qx, qy = nat_tunes or (0.0, 0.0)
    x_min, x_max = qx_range or (float(int(qx)), float(int(qx) + 1))
    y_min, y_max = qy_range or (float(int(qy)), float(int(qy) + 1))
    if not x_min < x_max or not y_min < y_max:
        raise ValueError("Each tune range must be increasing.")
    created_axes = ax is None
    if created_axes:
        _, ax = plt.subplots(figsize=(8, 8))

    drawn_lines: set[tuple[int, int, int]] = set()
    legend_handles = []
    for order in range(1, max_order + 1):
        color = _ORDER_COLORS[(order - 1) % len(_ORDER_COLORS)]
        _draw_order(ax, order, color, drawn_lines, x_min, x_max, y_min, y_max, kind)
        linestyle = "--" if kind == "diff" else "-"
        legend_handles.append(Line2D((0,), (0,), color=color, linewidth=1.2, linestyle=linestyle, label=f"{order}-order"))

    if nat_tunes is not None:
        ax.plot(qx, qy, ".", color="red", markersize=5, zorder=5, label="working point")
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), xlabel=r"$Q_x$", ylabel=r"$Q_y$", title="Tune Diagram")
    ax.set_aspect("equal")
    ax.grid(False)
    if show_legend:
        if legend_outside:
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=8,
            )
            if created_axes:
                ax.figure.subplots_adjust(right=0.78)
        else:
            ax.legend(handles=legend_handles, loc="best", fontsize=8)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(destination, dpi=300, bbox_inches="tight")
    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plot_tune_diagram(
        nat_tunes=(9.47, 9.43),
        max_order=5,
        qx_range=(9.0, 10.0),
        qy_range=(9.0, 10.0),
        legend_outside=True,
    )
    plt.show()
