"""Local coordinate projections; none of these operations changes particle state."""
import numpy as np


def resolve_slice_coordinate(coordinate=None, periodic=False):
    """Keep legacy Periodic=True as arrival phase, never geometric wrapping."""
    if not isinstance(periodic, (bool, np.bool_)):
        raise ValueError("Periodic must be a boolean")
    if coordinate is None:
        return "arrival_phase" if periodic else "z_rel"
    coordinate = str(coordinate).strip().lower().replace("-", "_")
    if coordinate not in {"z_rel", "z_periodic", "arrival_phase"}:
        raise ValueError("Coordinate must be z_rel, z_periodic or arrival_phase")
    if periodic and coordinate != "arrival_phase":
        raise ValueError("Legacy Periodic=True requires Coordinate=arrival_phase")
    return coordinate


def ring_interval(lower, upper, circumference):
    if not np.all(np.isfinite([lower, upper, circumference])) or circumference <= 0 or not np.isclose(
            upper - lower, circumference, rtol=1e-13, atol=0.):
        raise ValueError('Arrival-phase interval must span one design circumference')
