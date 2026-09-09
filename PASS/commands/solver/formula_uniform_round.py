"""Reference field of a uniformly charged round transverse slice."""

from __future__ import annotations

import numpy as np

from PASS.utils.constants import const


def uniform_round_field(x, y, slice_charge, radius, *, epsilon_0=const.epsilon0):
    """Return the integrated field (V) inside and outside a round uniform slice."""
    try:
        radius = float(radius)
    except (TypeError, ValueError) as exc:
        raise ValueError("radius must be a scalar finite value") from exc
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive")
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    try:
        charge = float(slice_charge)
        epsilon = float(epsilon_0)
    except (TypeError, ValueError) as exc:
        raise ValueError("slice_charge and epsilon_0 must be scalar finite values") from exc
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.isfinite(charge) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("coordinates, slice_charge, and epsilon_0 must be finite; epsilon_0 must be positive")
    r2 = x * x + y * y
    inside = r2 <= radius**2
    inside_scale = charge / (2.0 * const.pi * epsilon * radius**2)
    outside_scale = np.divide(
        charge,
        2.0 * const.pi * epsilon * r2,
        out=np.zeros_like(r2),
        where=r2 > 0,
    )
    scale = np.where(inside, inside_scale, outside_scale)
    return scale * x, scale * y
