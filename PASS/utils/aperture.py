"""Shared aperture geometry and CPU/GPU particle loss handling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np


def _array_module(*values):
    """Dispatch geometry predicates without importing CuPy during CPU runs."""
    if any(type(value).__module__.split(".")[0] == "cupy" for value in values):
        import cupy as cp
        return cp
    return np


def _value(values: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in values:
            return float(values[name])
    raise ValueError(f"aperture is missing one of {names}")


def _raw_value(values: Mapping[str, Any], *names: str):
    for name in names:
        if name in values:
            return values[name]
    raise ValueError(f"aperture is missing one of {names}")


def _parameter_list(aperture: Mapping[str, Any]) -> Sequence[Any] | None:
    for name in ("Aperture Value", "Value", "value", "values"):
        if name in aperture:
            value = aperture[name]
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise TypeError(f"aperture {name!r} must be a sequence")
            return value
    return None


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"aperture {name} must be positive and finite")
    return result


@dataclass(frozen=True)
class RectangleAperture:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if not self.x_min < self.x_max or not self.y_min < self.y_max:
            raise ValueError("rectangle aperture minima must be smaller than maxima")

    def mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        return ((x >= self.x_min) & (x <= self.x_max) & (y >= self.y_min) & (y <= self.y_max))

    def strict_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        return ((x > self.x_min) & (x < self.x_max) & (y > self.y_min) & (y < self.y_max))

    def distances(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
        xp = _array_module(x, y)
        return x - self.x_min, self.x_max - x, y - self.y_min, self.y_max - y


@dataclass(frozen=True)
class EllipticAperture:
    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a <= 0 or self.b <= 0:
            raise ValueError("elliptic aperture A and B must be positive")

    def _normalized_radius_squared(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        return (x / self.a)**2 + (y / self.b)**2

    def mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        return self._normalized_radius_squared(x, y) <= 1.0

    def strict_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        return self._normalized_radius_squared(x, y) < 1.0

    def distances(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
        xp = _array_module(x, y)
        x_wall = self.a * xp.sqrt(xp.maximum(0.0, 1.0 - (y / self.b)**2))
        y_wall = self.b * xp.sqrt(xp.maximum(0.0, 1.0 - (x / self.a)**2))
        return x + x_wall, x_wall - x, y + y_wall, y_wall - y


@dataclass(frozen=True)
class AllSpaceAperture:
    """No physical aperture; the outer FD grid remains the Dirichlet wall."""

    def mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        x, y = xp.broadcast_arrays(x, y)
        return xp.ones(x.shape, dtype=bool)

    def strict_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        return self.mask(x, y)


@dataclass(frozen=True)
class IntersectionAperture:
    apertures: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.apertures:
            raise ValueError("intersection aperture requires at least one component")

    def mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        result = xp.ones(xp.broadcast(x, y).shape, dtype=bool)
        for aperture in self.apertures:
            result &= aperture.mask(x, y)
        return result

    def strict_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        result = xp.ones(xp.broadcast(x, y).shape, dtype=bool)
        for aperture in self.apertures:
            result &= aperture.strict_mask(x, y)
        return result


@dataclass(frozen=True)
class RacetrackAperture:
    w: float
    h: float
    a: float
    b: float

    def __post_init__(self) -> None:
        for value, name in ((self.w, "W"), (self.h, "H"), (self.a, "A"), (self.b, "B")):
            _positive(value, name)

    def mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        x, y = xp.broadcast_arrays(xp.asarray(x), xp.asarray(y))
        ax = xp.abs(x)
        in_rectangle = (ax <= self.w) & (xp.abs(y) <= self.h)
        in_end = (ax > self.w) & (
            ((ax - self.w) / self.a) ** 2 + (y / self.b) ** 2 <= 1.0
        )
        return in_rectangle | in_end

    def strict_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        x, y = xp.broadcast_arrays(xp.asarray(x), xp.asarray(y))
        ax = xp.abs(x)
        in_rectangle = (ax < self.w) & (xp.abs(y) < self.h)
        in_end = (ax > self.w) & (
            ((ax - self.w) / self.a) ** 2 + (y / self.b) ** 2 < 1.0
        )
        seam = (ax == self.w) & (xp.abs(y) < min(self.h, self.b))
        return in_rectangle | in_end | seam


@dataclass(frozen=True)
class OctagonAperture:
    w: float
    h: float
    d: float

    def __post_init__(self) -> None:
        _positive(self.w, "W")
        _positive(self.h, "H")
        if not np.isfinite(self.d) or self.d < 0.0 or self.d > min(self.w, self.h):
            raise ValueError("octagon D must satisfy 0 <= D <= min(W, H)")

    def mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        ax, ay = xp.abs(x), xp.abs(y)
        return (ax <= self.w) & (ay <= self.h) & (ax + ay <= self.w + self.h - self.d)

    def strict_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        ax, ay = xp.abs(x), xp.abs(y)
        return (ax < self.w) & (ay < self.h) & (ax + ay < self.w + self.h - self.d)


@dataclass(frozen=True)
class PolygonAperture:
    vertices: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
            raise ValueError("polygon aperture requires at least three [x, y] vertices")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("polygon aperture vertices must be finite")
        x, y = vertices[:, 0], vertices[:, 1]
        area_twice = np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
        if abs(area_twice) <= 32 * np.finfo(float).eps * np.ptp(x) * np.ptp(y):
            raise ValueError("polygon aperture must have nonzero area")

    def _inside_and_boundary(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        xp = _array_module(x, y)
        x, y = xp.broadcast_arrays(xp.asarray(x, dtype=float), xp.asarray(y, dtype=float))
        # Vertices are configuration metadata; keep the short edge loop on CPU.
        vertices = np.asarray(self.vertices, dtype=float)
        inside = xp.zeros(x.shape, dtype=bool)
        on_boundary = xp.zeros(x.shape, dtype=bool)
        scale = max(1.0, float(np.max(np.abs(vertices))))
        tolerance = 64.0 * xp.finfo(float).eps * scale
        for index in range(vertices.shape[0]):
            x1, y1 = vertices[index]
            x2, y2 = vertices[(index + 1) % vertices.shape[0]]
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            if length == 0:
                continue
            cross_distance = xp.abs(dx * (y - y1) - dy * (x - x1)) / length
            within = (
                (x >= min(x1, x2) - tolerance)
                & (x <= max(x1, x2) + tolerance)
                & (y >= min(y1, y2) - tolerance)
                & (y <= max(y1, y2) + tolerance)
            )
            on_boundary |= (cross_distance <= tolerance) & within
            crossing = (y1 > y) != (y2 > y)
            if y2 != y1:
                x_intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                inside ^= crossing & (x < x_intersection)
        return inside, on_boundary

    def mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        inside, boundary = self._inside_and_boundary(x, y)
        return inside | boundary

    def strict_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xp = _array_module(x, y)
        inside, boundary = self._inside_and_boundary(x, y)
        return inside & ~boundary


def build_aperture(aperture: Mapping[str, Any]):
    if not isinstance(aperture, Mapping):
        raise TypeError("aperture geometry requires a mapping")
    kind_value = aperture.get("Type", aperture.get("type"))
    if kind_value is None:
        raise ValueError("aperture requires an explicit 'Type'")
    kind = str(kind_value).lower()
    parameters = _parameter_list(aperture)
    if kind == "off":
        return AllSpaceAperture()
    if kind == "default":
        return RectangleAperture(-1.0, 1.0, -1.0, 1.0)
    if kind in {"circle", "circular"}:
        radius = parameters[0] if parameters is not None else _raw_value(
            aperture, "R", "r", "Radius", "radius"
        )
        radius = _positive(radius, "radius")
        return EllipticAperture(radius, radius)
    if kind in {"ellipse", "elliptic"}:
        if parameters is not None:
            if len(parameters) != 2:
                raise ValueError("ellipse aperture value must be [A, B]")
            a, b = parameters
        else:
            a, b = _raw_value(aperture, "A", "a"), _raw_value(aperture, "B", "b")
        return EllipticAperture(_positive(a, "A"), _positive(b, "B"))
    if kind in {"rectangle", "rectangular"}:
        if parameters is not None:
            if len(parameters) != 2:
                raise ValueError("rectangle aperture value must be [half_width, half_height]")
            a, b = (_positive(parameters[0], "half-width"), _positive(parameters[1], "half-height"))
            return RectangleAperture(-a, a, -b, b)
        names = ("X min", "x_min", "X max", "x_max", "Y min", "y_min", "Y max", "y_max")
        if any(name in aperture for name in names):
            return RectangleAperture(
                _value(aperture, "X min", "x_min"),
                _value(aperture, "X max", "x_max"),
                _value(aperture, "Y min", "y_min"),
                _value(aperture, "Y max", "y_max"),
            )
        a, b = _value(aperture, "A", "a"), _value(aperture, "B", "b")
        return RectangleAperture(-a, a, -b, b)
    if kind == "rectcircle":
        if parameters is not None:
            if len(parameters) != 3:
                raise ValueError("rectcircle aperture value must be [W, H, R]")
            w, h, radius = parameters
        else:
            w = _raw_value(aperture, "W", "w")
            h = _raw_value(aperture, "H", "h")
            radius = _raw_value(aperture, "R", "r", "Radius", "radius")
        rectangle = RectangleAperture(-_positive(w, "W"), _positive(w, "W"), -_positive(h, "H"), _positive(h, "H"))
        radius = _positive(radius, "R")
        return IntersectionAperture((rectangle, EllipticAperture(radius, radius)))
    if kind == "rectellipse":
        if parameters is not None:
            if len(parameters) != 4:
                raise ValueError("rectellipse aperture value must be [W, H, A, B]")
            w, h, a, b = parameters
        else:
            w, h = _raw_value(aperture, "W", "w"), _raw_value(aperture, "H", "h")
            a, b = _raw_value(aperture, "A", "a"), _raw_value(aperture, "B", "b")
        w, h, a, b = (_positive(w, "W"), _positive(h, "H"), _positive(a, "A"), _positive(b, "B"))
        return IntersectionAperture((RectangleAperture(-w, w, -h, h), EllipticAperture(a, b)))
    if kind == "racetrack":
        if parameters is not None:
            if len(parameters) != 4:
                raise ValueError("racetrack aperture value must be [W, H, A, B]")
            w, h, a, b = parameters
        else:
            w, h = _raw_value(aperture, "W", "w"), _raw_value(aperture, "H", "h")
            a, b = _raw_value(aperture, "A", "a"), _raw_value(aperture, "B", "b")
        return RacetrackAperture(
            _positive(w, "W"), _positive(h, "H"), _positive(a, "A"), _positive(b, "B")
        )
    if kind == "octagon":
        if parameters is not None:
            if len(parameters) != 3:
                raise ValueError("octagon aperture value must be [W, H, D]")
            w, h, d = parameters
        else:
            w, h = _raw_value(aperture, "W", "w"), _raw_value(aperture, "H", "h")
            d = _raw_value(aperture, "D", "d")
        return OctagonAperture(_positive(w, "W"), _positive(h, "H"), float(d))
    if kind == "polygon":
        vertices = parameters if parameters is not None else _raw_value(
            aperture, "Vertices", "vertices"
        )
        return PolygonAperture(tuple(tuple(map(float, vertex)) for vertex in vertices))
    supported = "off, default, circle, rectangle, ellipse, rectcircle, rectellipse, racetrack, octagon, polygon"
    raise ValueError(f"unsupported aperture type {kind!r}; expected one of: {supported}")


def aperture_bounds(aperture) -> tuple[float, float, float, float] | None:
    """Exact bounding rectangle for supported geometries; None means unbounded."""
    if isinstance(aperture, AllSpaceAperture):
        return None
    if isinstance(aperture, RectangleAperture):
        return aperture.x_min, aperture.x_max, aperture.y_min, aperture.y_max
    if isinstance(aperture, EllipticAperture):
        return -aperture.a, aperture.a, -aperture.b, aperture.b
    if isinstance(aperture, RacetrackAperture):
        hx, hy = aperture.w + aperture.a, max(aperture.h, aperture.b)
        return -hx, hx, -hy, hy
    if isinstance(aperture, OctagonAperture):
        return -aperture.w, aperture.w, -aperture.h, aperture.h
    if isinstance(aperture, PolygonAperture):
        vertices = np.asarray(aperture.vertices)
        return (float(vertices[:, 0].min()), float(vertices[:, 0].max()),
                float(vertices[:, 1].min()), float(vertices[:, 1].max()))
    if isinstance(aperture, IntersectionAperture):
        # The supported intersections are centered rectangles and ellipses;
        # all reach their coordinate extrema on the common symmetry axes.
        bounds = [aperture_bounds(part) for part in aperture.apertures]
        return (max(b[0] for b in bounds), min(b[1] for b in bounds),
                max(b[2] for b in bounds), min(b[3] for b in bounds))
    raise TypeError(f"Unsupported aperture geometry: {type(aperture).__name__}")




from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PASS.core.beam import Beam
    from PASS.core.bunch import BunchInfo

import numpy as np
import logging

logger = logging.getLogger(__name__)

_VALID_TYPES = {"off", "default", "circle", "rectangle", "ellipse", "rectcircle", "rectellipse", "racetrack", "octagon", "polygon"}

# ------------------------------------------------------------------
# CPU
# ------------------------------------------------------------------


def _mark_lost_cpu(tag, lost_position, lost_turn, mask, s_position, turn):
    """Mark particles selected by mask as lost."""
    tag[mask] = -np.abs(tag[mask])
    lost_position[mask] = s_position
    lost_turn[mask] = turn


def check_aperture_cpu(beam: Beam, bunch: BunchInfo, aperture_type: str, aperture_value: list, s_position: float, turn: int):
    """Lose live particles outside the strict interior, including the wall.

    Geometry is shared with field solvers. ``off`` disables losses; the
    generic element ``default`` remains the +/-1 m rectangle. SpaceCharge
    resolves its grid-dependent default before calling this function.
    """

    if aperture_type.lower() == "off":
        return
    geometry = build_aperture({"Type": aperture_type, "Value": aperture_value})
    start, end = bunch.start_idx, bunch.end_idx
    p = beam.particles
    tag = p.tag[start:end]
    newly_lost = (tag > 0) & ~geometry.strict_mask(p.x[start:end], p.y[start:end])
    if np.any(newly_lost):
        _mark_lost_cpu(tag, p.lost_position[start:end], p.lost_turn[start:end],
                       newly_lost, s_position, turn)


# ------------------------------------------------------------------
# GPU
# ------------------------------------------------------------------

kernel_code = r'''
#ifndef PASS_USE_FLOAT
#define PASS_USE_FLOAT 0
#endif
#if PASS_USE_FLOAT
using pass_real_t = float;
#else
using pass_real_t = double;
#endif
using pass_loss_t = float;

extern "C" __global__
void check_aperture_rect(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    double half_width, double half_height, double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        if (fabs(x[i]) >= half_width || fabs(y[i]) >= half_height)
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}

extern "C" __global__
void check_aperture_circle(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    double radius, double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        if ((x[i] * x[i] + y[i] * y[i]) >= (radius * radius))
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}

extern "C" __global__
void check_aperture_ellipse(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    double a, double b, double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        double tx = x[i] / a;
        double ty = y[i] / b;
        if ((tx * tx + ty * ty) >= 1.0)
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}

extern "C" __global__
void check_aperture_rectcircle(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    double half_width, double half_height, double radius, double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        if (fabs(x[i]) >= half_width || fabs(y[i]) >= half_height ||
            (x[i] * x[i] + y[i] * y[i]) >= (radius * radius))
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}

extern "C" __global__
void check_aperture_rectellipse(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    double w, double h, double a, double b,
    double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        double tx = x[i] / a;
        double ty = y[i] / b;
        if (fabs(x[i]) >= w || fabs(y[i]) >= h || (tx * tx + ty * ty) >= 1.0)
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}

extern "C" __global__
void check_aperture_racetrack(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    double w, double h, double a, double b,
    double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        double ax = fabs(x[i]);
        double ay = fabs(y[i]);

        bool in_rect = (ax < w) && (ay < h);
        bool in_ellipse = false;
        if (ax > w)
        {
            double dx = (ax - w) / a;
            double ty = y[i] / b;
            in_ellipse = (dx * dx + ty * ty) < 1.0;
        }

        bool in_seam = (ax == w) && (ay < fmin(h, b));
        if (!in_rect && !in_ellipse && !in_seam)
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}

extern "C" __global__
void check_aperture_octagon(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    double w, double h, double d, double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        double ax = fabs(x[i]);
        double ay = fabs(y[i]);
        if (ax >= w || ay >= h || (ax + ay) >= (w + h - d))
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}

extern "C" __global__
void check_aperture_polygon(
    double* __restrict__ x, double* __restrict__ y,
    int* __restrict__ tag, double* __restrict__ lost_position, int* __restrict__ lost_turn,
    int start_index, int end_index,
    int nvert, const double* __restrict__ vertx, const double* __restrict__ verty,
    double s_position, int turn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    if (i >= end_index) return;
    if (tag[i] > 0)
    {
        bool inside = false;
        bool on_boundary = false;
        double scale = 1.0;
        for (int k = 0; k < nvert; k++)
            scale = fmax(scale, fmax(fabs(vertx[k]), fabs(verty[k])));
        double tolerance = 64.0 * 2.220446049250313e-16 * scale;
        for (int k = 0, j = nvert - 1; k < nvert; j = k++)
        {
            double dx = vertx[j] - vertx[k], dy = verty[j] - verty[k];
            double edge_length = hypot(dx, dy);
            if (edge_length == 0) continue;
            double distance = fabs(dx * (y[i] - verty[k]) - dy * (x[i] - vertx[k])) / edge_length;
            bool within = x[i] >= fmin(vertx[k], vertx[j]) - tolerance
                       && x[i] <= fmax(vertx[k], vertx[j]) + tolerance
                       && y[i] >= fmin(verty[k], verty[j]) - tolerance
                       && y[i] <= fmax(verty[k], verty[j]) + tolerance;
            on_boundary = on_boundary || (distance <= tolerance && within);
            if ((verty[k] > y[i]) != (verty[j] > y[i]))
            {
                double x_intersect = (vertx[j] - vertx[k]) * (y[i] - verty[k]) / (verty[j] - verty[k]) + vertx[k];
                if (x[i] < x_intersect)
                    inside = !inside;
            }
        }
        if (!inside || on_boundary)
        {
            tag[i] = -tag[i];
            lost_position[i] = s_position;
            lost_turn[i] = turn;
        }
    }
}
'''

_kernel_cache = {}


def _kernel_source(dtype):
    """Specialize aperture coordinates to the particle precision."""
    # Protect the type aliases in the preamble from the broad scalar-type
    # substitution applied to the kernel bodies.
    source = kernel_code.replace(
        "using pass_real_t = float;", "using pass_real_t = __PASS_FLOAT__;"
    ).replace(
        "using pass_real_t = double;", "using pass_real_t = __PASS_DOUBLE__;"
    )
    source = source.replace("double", "pass_real_t")
    source = source.replace("__PASS_FLOAT__", "float")
    source = source.replace("__PASS_DOUBLE__", "double")
    source = source.replace(
        "pass_real_t* __restrict__ lost_position",
        "pass_loss_t* __restrict__ lost_position",
    )
    return source


def _get_kernel(name, dtype):
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "GPU aperture checks require the optional 'cuda' dependencies "
            "(install PASS with the [cuda] extra)."
        ) from exc
    key = (name, np.dtype(dtype))
    if key not in _kernel_cache:
        use_float = np.dtype(dtype) == np.dtype(np.float32)
        _kernel_cache[key] = cp.RawKernel(
            _kernel_source(dtype), name,
            options=("--std=c++14", f"-DPASS_USE_FLOAT={int(use_float)}"),
        )
    return _kernel_cache[key]


def _launch_gpu(kernel, beam, bunch, *args):
    start = bunch.start_idx
    end = bunch.end_idx
    p = beam.particles
    N = end - start
    if N <= 0:
        return
    threads = 256
    blocks = (N + threads - 1) // threads
    kernel = _get_kernel(kernel, p.dtype)
    real = p.real
    args = tuple(
        real(arg) if isinstance(arg, (float, np.floating))
        else np.int32(arg) if isinstance(arg, (int, np.integer))
        else arg
        for arg in args
    )
    kernel((blocks, ), (threads, ), (p.x, p.y, p.tag, p.lost_position, p.lost_turn,
                                    np.int32(start), np.int32(end), *args))


def check_aperture_gpu(beam: Beam, bunch: BunchInfo, aperture_type: str, aperture_value: list, s_position: float, turn: int):
    """Lose wall/outside particles on GPU, preserving earlier loss records."""
    aperture_type = aperture_type.lower()

    if aperture_type == "off":
        return
    # Validate the same geometry as CPU before launching kernels. Scalar shape
    # dimensions must be passed as real values even when JSON supplied integers.
    build_aperture({"Type": aperture_type, "Value": aperture_value})
    if aperture_type not in {"polygon", "default"}:
        aperture_value = [float(value) for value in aperture_value]
    s_position = float(s_position)
    if aperture_type == "default":
        _launch_gpu("check_aperture_rect", beam, bunch, 1.0, 1.0, s_position, turn)
    elif aperture_type == "circle":
        _launch_gpu("check_aperture_circle", beam, bunch, aperture_value[0], s_position, turn)
    elif aperture_type == "rectangle":
        _launch_gpu("check_aperture_rect", beam, bunch, aperture_value[0], aperture_value[1], s_position, turn)
    elif aperture_type == "ellipse":
        _launch_gpu("check_aperture_ellipse", beam, bunch, aperture_value[0], aperture_value[1], s_position, turn)
    elif aperture_type == "rectcircle":
        _launch_gpu("check_aperture_rectcircle", beam, bunch, aperture_value[0], aperture_value[1], aperture_value[2], s_position, turn)
    elif aperture_type == "rectellipse":
        _launch_gpu("check_aperture_rectellipse", beam, bunch, aperture_value[0], aperture_value[1], aperture_value[2],
                    aperture_value[3], s_position, turn)
    elif aperture_type == "racetrack":
        _launch_gpu("check_aperture_racetrack", beam, bunch, aperture_value[0], aperture_value[1], aperture_value[2], aperture_value[3],
                    s_position, turn)
    elif aperture_type == "octagon":
        _launch_gpu("check_aperture_octagon", beam, bunch, aperture_value[0], aperture_value[1], aperture_value[2], s_position, turn)
    elif aperture_type == "polygon":
        try:
            import cupy as cp
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "GPU aperture checks require the optional 'cuda' dependencies "
                "(install PASS with the [cuda] extra)."
            ) from exc
        vertx = cp.asarray([v[0] for v in aperture_value], dtype=beam.particles.dtype)
        verty = cp.asarray([v[1] for v in aperture_value], dtype=beam.particles.dtype)
        nvert = len(aperture_value)
        _launch_gpu("check_aperture_polygon", beam, bunch, nvert, vertx, verty, s_position, turn)
    else:
        raise ValueError(f"Unknown aperture type: {aperture_type}. Must be one of {sorted(_VALID_TYPES)}")
