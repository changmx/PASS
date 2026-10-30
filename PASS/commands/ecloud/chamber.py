"""One conducting boundary for electron motion, emission and Poisson solves."""

import math

import numpy as np

from PASS.utils.aperture import aperture_bounds, build_aperture

from .pusher import first_circle_hit


class Chamber:
    """Analytic line/ellipse boundary pieces; particle arrays remain on their backend."""

    def __init__(self, configuration):
        kind, values = getattr(configuration, "aperture_type", "default"), getattr(configuration, "aperture_value", [])
        if kind in {"default", "off"}:
            kind, values = "circle", [configuration.buildup.chamber_radius]
        self.kind, self.values = kind, values
        self.specification = {"Type": kind, "Aperture Value": values}
        self.aperture = build_aperture(self.specification)
        self.bounds = aperture_bounds(self.aperture)
        self.scale = max(abs(value) for value in self.bounds)
        self.radius = values[0] if kind == "circle" else math.hypot(max(abs(self.bounds[0]), abs(self.bounds[1])),
                                                                    max(abs(self.bounds[2]), abs(self.bounds[3])))
        self.margin = 256 * np.finfo(float).eps * max(1., self.scale)
        self._segments, self._ellipses = [], []
        if kind in {"circle", "ellipse"}:
            a, b = (values[0], values[0]) if kind == "circle" else values
            self._ellipses.append((0., a, b, -np.pi, np.pi))
        elif kind == "rectangle":
            a, b = values
            self._polygon([[-a, -b], [a, -b], [a, b], [-a, b]])
        elif kind == "polygon":
            self._polygon(values)
        elif kind == "racetrack":
            w, h, a, b = values
            self._segments.extend([(-w, -h, w, -h, 0., 1.), (w, h, -w, h, 0., -1.)])
            self._ellipses.extend([(w, a, b, -np.pi / 2, np.pi / 2), (-w, a, b, np.pi / 2, 3 * np.pi / 2)])
            if h != b:
                for side in (-1., 1.):
                    for vertical in (-1., 1.):
                        self._segments.append((side * w, vertical * min(h, b), side * w, vertical * max(h, b), -side if h > b else side, 0.))
        else:
            raise ValueError("dynamic electron chambers support circle, ellipse, rectangle, racetrack and polygon")

    def _polygon(self, vertices):
        points = np.asarray(vertices, dtype=float)
        if np.unique(points, axis=0).shape != points.shape:
            raise ValueError("electron chamber polygon cannot have repeated vertices")
        area = np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))
        orientation = 1. if area > 0 else -1.
        for i in range(len(points)):
            first, second = points[i], points[(i + 1) % len(points)]
            for j in range(i + 1, len(points)):
                if j in {i, (i + 1) % len(points)} or (j + 1) % len(points) == i:
                    continue
                third, fourth = points[j], points[(j + 1) % len(points)]
                direction, other = second - first, fourth - third
                determinant = direction[0] * other[1] - direction[1] * other[0]
                offset = third - first
                if determinant != 0:
                    t = (offset[0] * other[1] - offset[1] * other[0]) / determinant
                    u = (offset[0] * direction[1] - offset[1] * direction[0]) / determinant
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        raise ValueError("electron chamber polygon must be simple, without crossing edges")
                elif offset[0] * direction[1] - offset[1] * direction[0] == 0:
                    projections = np.sort([np.dot(third - first, direction), np.dot(fourth - first, direction)]) / np.dot(direction, direction)
                    if projections[0] <= 1 and projections[1] >= 0:
                        raise ValueError("electron chamber polygon must be simple, without overlapping edges")
        for start, end in zip(points, np.roll(points, -1, axis=0)):
            dx, dy = end - start
            length = np.hypot(dx, dy)
            if length == 0:
                raise ValueError("electron chamber polygon cannot have repeated consecutive vertices")
            self._segments.append((*start, *end, -orientation * dy / length, orientation * dx / length))

    def contains(self, x, y, xp=np):
        return self.aperture.strict_mask(xp.asarray(x), xp.asarray(y))

    def contains_disk(self, x, y, radius):
        """Check the continuous source disk against every boundary component."""
        if not bool(self.contains(x, y)):
            return False
        distance = np.inf
        for x1, y1, x2, y2, _, _ in self._segments:
            dx, dy = x2 - x1, y2 - y1
            fraction = np.clip(((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy), 0., 1.)
            distance = min(distance, np.hypot(x - x1 - fraction * dx, y - y1 - fraction * dy))
        for center, a, b, low, high in self._ellipses:
            from scipy.optimize import minimize_scalar
            angles = np.linspace(low, high, 257)
            squared = (center + a * np.cos(angles) - x)**2 + (b * np.sin(angles) - y)**2
            candidates = [float(squared[0]), float(squared[-1])]
            for index in np.flatnonzero((squared[1:-1] <= squared[:-2]) & (squared[1:-1] <= squared[2:])) + 1:
                result = minimize_scalar(lambda theta: (center + a * np.cos(theta) - x)**2 + (b * np.sin(theta) - y)**2,
                                         bounds=(angles[index - 1], angles[index + 1]),
                                         method="bounded",
                                         options={"xatol": 1e-15})
                candidates.append(float(result.fun))
            distance = min(distance, np.sqrt(min(candidates)))
        return distance > radius + self.margin

    def first_hit(self, x, y, vx, vy, xp=np):
        if self.kind == "circle":
            return first_circle_hit(x, y, vx, vy, self.radius, xp)
        times = xp.full(x.shape, xp.inf)
        for x1, y1, x2, y2, nx, ny in self._segments:
            dx, dy = x2 - x1, y2 - y1
            denominator = vx * dy - vy * dx
            safe = xp.where(denominator != 0, denominator, 1.)
            t = ((x1 - x) * dy - (y1 - y) * dx) / safe
            fraction = ((x1 - x) * vy - (y1 - y) * vx) / safe
            valid = (denominator != 0) & (t >= 0) & (fraction >= -1e-13) & (fraction <= 1 + 1e-13) & (vx * nx + vy * ny < 0)
            times = xp.minimum(times, xp.where(valid, t, xp.inf))
        for center, a, b, low, high in self._ellipses:
            sx, sy, ux, uy = (x - center) / a, y / b, vx / a, vy / b
            aa = ux * ux + uy * uy
            bb, cc = sx * ux + sy * uy, sx * sx + sy * sy - 1
            discriminant = bb * bb - aa * cc
            root = xp.sqrt(xp.maximum(discriminant, 0))
            for sign in (-1., 1.):
                t = (-bb + sign * root) / xp.where(aa > 0, aa, 1.)
                hx, hy = x + vx * t, y + vy * t
                angle = xp.arctan2(hy / b, (hx - center) / a)
                angle = xp.where(angle < low - 1e-13, angle + 2 * np.pi, angle)
                outgoing = (hx - center) * vx / a**2 + hy * vy / b**2 > 0
                valid = (aa > 0) & (discriminant >= 0) & (t >= 0) & (angle <= high + 1e-13) & outgoing
                times = xp.minimum(times, xp.where(valid, t, xp.inf))
        return times

    def inward_normal(self, x, y, xp=np):
        if self.kind == "circle":
            radius = xp.hypot(x, y)
            return -x / radius, -y / radius
        distance = xp.full(x.shape, xp.inf)
        nx, ny = xp.zeros_like(x), xp.zeros_like(y)
        for x1, y1, x2, y2, normal_x, normal_y in self._segments:
            dx, dy = x2 - x1, y2 - y1
            fraction = xp.clip(((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy), 0, 1)
            candidate = xp.hypot(x - x1 - fraction * dx, y - y1 - fraction * dy)
            selected = candidate < distance - self.margin
            tied = xp.abs(candidate - distance) <= self.margin
            nx = xp.where(selected, normal_x, xp.where(tied, nx + normal_x, nx))
            ny = xp.where(selected, normal_y, xp.where(tied, ny + normal_y, ny))
            distance = xp.minimum(distance, candidate)
        for center, a, b, low, high in self._ellipses:
            angle = xp.arctan2(y / b, (x - center) / a)
            angle = xp.where(angle < low - 1e-13, angle + 2 * np.pi, angle)
            candidate = xp.where(angle <= high + 1e-13, xp.abs(xp.hypot((x - center) / a, y / b) - 1) * min(a, b), xp.inf)
            selected = candidate < distance - self.margin
            tied = xp.isfinite(candidate) & (xp.abs(candidate - distance) <= self.margin)
            gx, gy = -(x - center) / a**2, -y / b**2
            norm = xp.hypot(gx, gy)
            normal_x, normal_y = gx / xp.where(norm > 0, norm, 1), gy / xp.where(norm > 0, norm, 1)
            nx = xp.where(selected, normal_x, xp.where(tied, nx + normal_x, nx))
            ny = xp.where(selected, normal_y, xp.where(tied, ny + normal_y, ny))
            distance = xp.minimum(distance, candidate)
        norm = xp.hypot(nx, ny)
        if not bool(xp.all(norm > 0)):
            raise ValueError("electron chamber has an undefined inward normal at the impact")
        return nx / norm, ny / norm

    def move_inside(self, x, y, nx, ny, xp=np):
        result_x, result_y = x + self.margin * nx, y + self.margin * ny
        if not bool(xp.all(self.contains(result_x, result_y, xp))):
            raise ValueError("electron wall normal does not resolve an interior emission point")
        return result_x, result_y

    def sample_wall(self, count, generator, xp=np):
        pieces, lengths = [], []
        for segment in self._segments:
            pieces.append(("line", segment))
            lengths.append(np.hypot(segment[2] - segment[0], segment[3] - segment[1]))
        for ellipse in self._ellipses:
            center, a, b, low, high = ellipse
            angle = np.linspace(low, high, 4097)
            speed = np.hypot(a * np.sin(angle), b * np.cos(angle))
            cumulative = np.r_[0., np.cumsum((speed[1:] + speed[:-1]) / 2 * np.diff(angle))]
            pieces.append(("ellipse", (center, a, b, angle, cumulative)))
            lengths.append(cumulative[-1])
        cumulative = np.r_[0., np.cumsum(lengths)]
        selected = generator.random(count) * cumulative[-1]
        x, y, nx, ny = [np.empty(count) for _ in range(4)]
        for index, (kind, data) in enumerate(pieces):
            mask = (selected >= cumulative[index]) & (selected < cumulative[index + 1])
            offset = selected[mask] - cumulative[index]
            if kind == "line":
                x1, y1, x2, y2, normal_x, normal_y = data
                x[mask], y[mask] = x1 + offset / lengths[index] * (x2 - x1), y1 + offset / lengths[index] * (y2 - y1)
                nx[mask], ny[mask] = normal_x, normal_y
            else:
                center, a, b, angle, arc = data
                theta = np.interp(offset, arc, angle)
                x[mask], y[mask] = center + a * np.cos(theta), b * np.sin(theta)
                gx, gy = -np.cos(theta) / a, -np.sin(theta) / b
                norm = np.hypot(gx, gy)
                nx[mask], ny[mask] = gx / norm, gy / norm
        x, y = self.move_inside(x, y, nx, ny)
        return tuple(xp.asarray(values) for values in (x, y, nx, ny))

    def sample_volume(self, count, generator, xp=np):
        xmin, xmax, ymin, ymax = self.bounds
        accepted_x, accepted_y = [], []
        remaining = count
        for _ in range(10000):
            if remaining == 0:
                break
            x, y = generator.uniform(xmin, xmax, max(32, 2 * remaining)), generator.uniform(ymin, ymax, max(32, 2 * remaining))
            mask = self.contains(x, y)
            x, y = x[mask][:remaining], y[mask][:remaining]
            accepted_x.append(x)
            accepted_y.append(y)
            remaining -= x.size
        if remaining:
            raise ValueError("electron chamber volume sampling did not converge")
        return tuple(xp.asarray(np.concatenate(values) if values else np.empty(0)) for values in (accepted_x, accepted_y))
