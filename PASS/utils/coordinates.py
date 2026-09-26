"""Local coordinate projections; none of these operations changes particle state."""
import numpy as np


def delta_to_eta(delta, beta0, xp=np):
    """Canonical energy eta=(E-E0)/(beta0*P0*c), with stable subtraction."""
    beta0 = float(beta0)
    value = delta * (2 + delta)
    # Keep the rest-energy term when beta rounds to one in storage precision.
    # The positive sum also avoids cancellation for momentum ratios near zero.
    rest_fraction = (1 - beta0) * (1 + beta0)
    energy_ratio = xp.sqrt(rest_fraction + (beta0 * (1 + delta))**2)
    return value / (energy_ratio + 1)


def eta_to_delta(eta, beta0, xp=np):
    """Inverse canonical conversion; callers validate the positive-energy branch."""
    beta0 = float(beta0)
    value = 2 * eta + beta0 * beta0 * eta * eta
    rest_fraction = (1 - beta0) * (1 + beta0)
    # Positive eta uses a positive polynomial; large eta must not subtract
    # nearly equal squared terms for a nonrelativistic reference.
    squared_ratio = (1 + 2 * eta) + beta0 * beta0 * eta * eta
    if beta0 > .5:
        squared_ratio = xp.where(eta < 0, (1 + eta)**2 - rest_fraction * eta * eta, squared_ratio)
    momentum_ratio = xp.sqrt(squared_ratio)
    return value / (momentum_ratio + 1)


def resolve_slice_coordinate(coordinate=None, periodic=False):
    """Keep legacy Periodic=True as arrival phase, never geometric wrapping."""
    if not isinstance(periodic, (bool, np.bool_)):
        raise ValueError("Periodic must be a boolean")
    if coordinate is None:
        return "arrival_phase" if periodic else "z_rel"
    coordinate = str(coordinate).strip().lower().replace("-", "_")
    if coordinate not in {"z_rel", "z_periodic", "arrival_phase", "collision_z"}:
        raise ValueError("Coordinate must be z_rel, z_periodic, arrival_phase or collision_z")
    if periodic and coordinate != "arrival_phase":
        raise ValueError("Legacy Periodic=True requires Coordinate=arrival_phase")
    return coordinate


def ring_interval(lower, upper, circumference):
    if not np.all(np.isfinite([lower, upper, circumference])) or circumference <= 0 or not np.isclose(
            upper - lower, circumference, rtol=1e-13, atol=0.):
        raise ValueError('Arrival-phase interval must span one design circumference')
