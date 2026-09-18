"""Free-space field inside and outside a uniformly charged elliptic slice."""

from __future__ import annotations

import numpy as np

from PASS.utils.constants import const
from .formula_uniform_round import uniform_round_field


def uniform_elliptic_field(x, y, slice_charge, a, b, *, epsilon_0=const.epsilon0):
    """Return the integrated field (V) inside or outside a uniform ellipse."""
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError) as exc:
        raise ValueError("ellipse semi-axes must be scalar finite values") from exc
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        raise ValueError("ellipse semi-axes must be positive")
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    try:
        charge = float(slice_charge)
        epsilon = float(epsilon_0)
    except (TypeError, ValueError) as exc:
        raise ValueError("slice_charge and epsilon_0 must be scalar finite values") from exc
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.isfinite(charge) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("coordinates, slice_charge, and epsilon_0 must be finite; epsilon_0 must be positive")
    if np.isclose(a, b, rtol=const.eps, atol=0.0):
        return uniform_round_field(x, y, charge, a, epsilon_0=epsilon)

    # The uniform sheet density is charge/(pi*a*b).  For an exterior point,
    # lambda >= 0 is the confocal ellipse parameter satisfying
    # x^2/(a^2+lambda) + y^2/(b^2+lambda) = 1.  Clearing denominators gives
    # a quadratic in lambda.  Use a cancellation-resistant form of its
    # positive root so all exterior points are handled in vectorized NumPy.
    lambda_value = np.zeros_like(x)
    outside = (x / a)**2 + (y / b)**2 > 1.0
    if np.any(outside):
        x2, y2 = x[outside]**2, y[outside]**2
        aa, bb = a * a, b * b
        linear = aa + bb - x2 - y2
        constant = aa * bb - bb * x2 - aa * y2
        discriminant = np.maximum(0.0, linear * linear - 4.0 * constant)
        root = np.sqrt(discriminant)
        # Positive root is (-linear + root)/2.  The rationalized form avoids
        # cancellation when linear is positive; the direct form is stable
        # when linear is negative.
        positive_linear = linear >= 0.0
        positive_root = np.empty_like(root)
        positive_root[positive_linear] = (-2.0 * constant[positive_linear]) / (linear[positive_linear] + root[positive_linear])
        positive_root[~positive_linear] = 0.5 * (-linear[~positive_linear] + root[~positive_linear])
        lambda_value[outside] = np.maximum(0.0, positive_root)

    # Assume a >= b; swapping coordinates handles the opposite orientation.
    if a < b:
        ey, ex = uniform_elliptic_field(y, x, charge, b, a, epsilon_0=epsilon)
        return ex, ey

    # Rationalize (1 - B/A)/(a^2-b^2) = 1/(A*(A+B)), where
    # A^2=a^2+lambda and B^2=b^2+lambda. This avoids subtracting nearly
    # equal numbers when a quasi-frozen covariance is almost isotropic.
    axis_a = np.sqrt(a * a + lambda_value)
    axis_b = np.sqrt(b * b + lambda_value)
    coefficient = charge / (const.pi * epsilon * (axis_a + axis_b))
    ex = coefficient * x / axis_a
    ey = coefficient * y / axis_b
    return ex, ey
