"""Deterministic Cartesian initial coordinates for particle scans."""

import numpy as np


def _scan_axis(values, name):
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or not axis.size or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be a nonempty one-dimensional array of finite values")
    if np.unique(axis).size != axis.size:
        raise ValueError(f"{name} must not contain duplicate values")
    return axis


def generate_scan_grid(x_values, y_values, dp_values, *, px=0.0, py=0.0, z=0.0):
    """Return every x/y/dp combination in dp-major, y, x-fastest order.

    Coordinates are ``[x, px, y, py, z, dp]`` in the incoming reference.
    Positions and z are metres, px/py are mechanical momenta divided by P0,
    and dp is (P-P0)/P0. No dispersion or injection offsets are applied.
    ``indices`` contains the corresponding ``[dp_index, y_index, x_index]``.
    """
    x_values = _scan_axis(x_values, "x_values")
    y_values = _scan_axis(y_values, "y_values")
    dp_values = _scan_axis(dp_values, "dp_values")
    fixed = np.asarray([px, py, z], dtype=np.float64)
    if fixed.shape != (3, ) or not np.all(np.isfinite(fixed)):
        raise ValueError("px, py and z must be finite scalar values")
    if np.any(1.0 + dp_values <= np.hypot(px, py)):
        raise ValueError("Scan momenta require dp > -1 and positive real longitudinal momentum")
    shape = (dp_values.size, y_values.size, x_values.size)
    n_particles = int(dp_values.size) * int(y_values.size) * int(x_values.size)
    if n_particles > np.iinfo(np.int32).max:
        raise ValueError("Scan grid exceeds the int32 particle identity capacity")
    indices = np.indices(shape, dtype=np.int32).reshape(3, -1).T.copy()
    coordinates = np.empty((n_particles, 6), dtype=np.float64)
    coordinates[:, 0] = x_values[indices[:, 2]]
    coordinates[:, 1] = px
    coordinates[:, 2] = y_values[indices[:, 1]]
    coordinates[:, 3] = py
    coordinates[:, 4] = z
    coordinates[:, 5] = dp_values[indices[:, 0]]
    return dict(coordinates=coordinates, indices=indices, x_values=x_values.copy(), y_values=y_values.copy(), dp_values=dp_values.copy())
