"""Bassetti-Erskine reference field for an elliptic Gaussian beam."""

from __future__ import annotations

import numpy as np
from scipy.special import wofz

from PASS.utils.constants import const
from .formula_gaussian_round import gaussian_round_field


def gaussian_elliptic_field(x, y, slice_charge, sigma_x, sigma_y, *, epsilon_0=const.epsilon0):
    """Return the Bassetti-Erskine integrated field (V) for an elliptic Gaussian."""
    try:
        sigma_x = float(sigma_x)
        sigma_y = float(sigma_y)
    except (TypeError, ValueError) as exc:
        raise ValueError("sigma_x and sigma_y must be scalar finite values") from exc
    if not np.isfinite(sigma_x) or not np.isfinite(sigma_y) or sigma_x <= 0 or sigma_y <= 0:
        raise ValueError("sigma_x and sigma_y must be positive")
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    try:
        charge = float(slice_charge)
        epsilon = float(epsilon_0)
    except (TypeError, ValueError) as exc:
        raise ValueError("slice_charge and epsilon_0 must be scalar finite values") from exc
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.isfinite(charge) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("coordinates, slice_charge, and epsilon_0 must be finite; epsilon_0 must be positive")
    if np.isclose(sigma_x, sigma_y, rtol=const.eps, atol=0.0):
        return gaussian_round_field(x, y, charge, sigma_x, epsilon_0=epsilon)
    if sigma_x < sigma_y:
        ey, ex = gaussian_elliptic_field(y, x, charge, sigma_y, sigma_x, epsilon_0=epsilon)
        return ex, ey

    # Evaluate in the first quadrant: wofz grows exponentially in the lower
    # half-plane, making the subtraction below unstable (especially near a
    # round beam). Restore each component's odd parity after evaluation.
    abs_x, abs_y = np.abs(x), np.abs(y)
    difference = (sigma_x - sigma_y) * (sigma_x + sigma_y)
    denominator = np.sqrt(2.0 * difference)
    z1 = (abs_x + 1j * abs_y) / denominator
    z2 = (abs_x * sigma_y / sigma_x + 1j * abs_y * sigma_x / sigma_y) / denominator
    gaussian = np.exp(-x**2 / (2.0 * sigma_x**2) - y**2 / (2.0 * sigma_y**2))
    complex_field = (1j * charge / (2.0 * epsilon * np.sqrt(2.0 * const.pi * difference)) * (wofz(z1) - gaussian * wofz(z2)))
    ex = -complex_field.real * np.sign(x)
    ey = complex_field.imag * np.sign(y)
    # Close to the origin even upper-half-plane terms almost cancel. Use
    # the Gaussian field's Taylor expansion through cubic order there;
    # its relative remainder is O(((x/sigma_x)^2 + (y/sigma_y)^2)^2).
    near_center = (x / sigma_x)**2 + (y / sigma_y)**2 <= 1.0e-6
    widths_sum = sigma_x + sigma_y
    factor = charge / (2.0 * const.pi * epsilon * widths_sum)
    local_ex = factor * (x / sigma_x) * (1.0 - (x / sigma_x)**2 * (2.0 * sigma_x + sigma_y) / (6.0 * widths_sum) - (y / sigma_y)**2 * sigma_y /
                                         (2.0 * widths_sum))
    local_ey = factor * (y / sigma_y) * (1.0 - (y / sigma_y)**2 * (2.0 * sigma_y + sigma_x) / (6.0 * widths_sum) - (x / sigma_x)**2 * sigma_x /
                                         (2.0 * widths_sum))
    return np.where(near_center, local_ex, ex), np.where(near_center, local_ey, ey)
