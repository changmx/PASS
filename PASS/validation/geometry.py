"""Geometry checks that require neither a solver nor a particle allocation."""
import math


def validate_polygon(vertices):
    """Reject repeated vertices and non-adjacent edge intersections/touches."""
    n = len(vertices)
    if len({tuple(p) for p in vertices}) != n:
        raise ValueError("多边形顶点不能重复；不需要重复首点来闭合")
    scale = max(max(abs(x), abs(y)) for x, y in vertices)
    tolerance = 64 * math.ulp(scale if scale else 1.0) * scale

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on(a, b, c):
        return abs(cross(a, b, c)) <= tolerance and all(min(a[k], b[k]) <= c[k] <= max(a[k], b[k]) for k in (0, 1))

    for i, a in enumerate(vertices):
        b = vertices[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i + 1 or i == 0 and j == n - 1:
                continue
            c, d = vertices[j], vertices[(j + 1) % n]
            if (cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0 or on(a, b, c) or on(a, b, d) or on(c, d, a)
                    or on(c, d, b)):
                raise ValueError(f"多边形的第 {i + 1} 与第 {j + 1} 条边相交或接触")


def has_interior_node(geometry, kind, dimensions):
    """Scan grid nodes in bounded chunks; no matrix factorization or meshgrid."""
    import numpy as np
    from PASS.utils.aperture import build_aperture, aperture_bounds
    aperture = build_aperture({"Type": kind, "Value": dimensions})
    bounds = aperture_bounds(aperture)
    if bounds is None:
        return True
    x0, x1, y0, y1 = bounds
    ix0 = max(1, math.ceil((x0 - geometry.x_min) / geometry.dx))
    ix1 = min(geometry.nx - 2, math.floor((x1 - geometry.x_min) / geometry.dx))
    iy0 = max(1, math.ceil((y0 - geometry.y_min) / geometry.dy))
    iy1 = min(geometry.ny - 2, math.floor((y1 - geometry.y_min) / geometry.dy))
    for iy in range(iy0, iy1 + 1):
        y = geometry.y_min + iy * geometry.dy
        for ix in range(ix0, ix1 + 1, 65536):
            x = geometry.x_min + np.arange(ix, min(ix + 65536, ix1 + 1)) * geometry.dx
            if np.any(aperture.strict_mask(x, y)):
                return True
    return False
