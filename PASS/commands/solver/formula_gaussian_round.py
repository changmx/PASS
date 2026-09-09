"""Reference transverse field for a round Gaussian beam."""

from __future__ import annotations

import numpy as np

from PASS.utils.constants import const


def gaussian_round_field(x, y, slice_charge, sigma, *, epsilon_0=const.epsilon0):
    """Integrated field of a round 2-D Gaussian charge slice (V)."""
    try:
        sigma = float(sigma)
    except (TypeError, ValueError) as exc:
        raise ValueError("sigma must be a scalar finite value") from exc
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive")
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    try:
        charge = float(slice_charge)
        epsilon = float(epsilon_0)
    except (TypeError, ValueError) as exc:
        raise ValueError("slice_charge and epsilon_0 must be scalar finite values") from exc
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.isfinite(charge) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("coordinates, slice_charge, and epsilon_0 must be finite; epsilon_0 must be positive")
    r2 = x * x + y * y
    factor = charge / (2.0 * const.pi * epsilon)
    enclosed_fraction = -np.expm1(-r2 / (2.0 * sigma**2))
    scale = np.divide(factor * enclosed_fraction, r2, out=np.zeros_like(r2), where=r2 > 0)
    return scale * x, scale * y
