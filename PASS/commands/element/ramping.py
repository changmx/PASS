"""Load magnet programs and update current nominal strengths, without tracking."""

import numpy as np

from PASS.utils.magnet_program import load_magnet_ramp


def configure_magnet_ramping(element, kwargs, order=None):
    """Load once; the element owns its current strengths and their derived maps."""
    element.ramp = load_magnet_ramp(kwargs, length=element.length, order=order)
    element._magnet_order = order
    element._strength_signature = None


def _nominal_signature(element):
    order = element._magnet_order
    if order is None:
        normal, skew = np.asarray(element.knl, dtype=float), np.asarray(element.ksl, dtype=float)
        if normal.ndim != 1 or skew.ndim != 1 or not np.all(np.isfinite(normal)) or not np.all(np.isfinite(skew)):
            raise ValueError("Nominal magnet coefficients must be finite one-dimensional arrays")
        values = (normal.shape, normal.tobytes(), skew.shape, skew.tobytes())
    else:
        values = (getattr(element, f"k{order}l"), getattr(element, f"k{order}sl"))
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Nominal magnet coefficients must be finite")
    return (element.length, element.is_thick, *values)


def refresh_magnet_strengths(element):
    """Refresh element-owned derived values only when current nominal KL changes."""
    signature = _nominal_signature(element)
    if signature == element._strength_signature:
        return
    element._refresh_strengths()
    # A multipole can pad its nominal arrays while refreshing derived values.
    element._strength_signature = _nominal_signature(element)
    element.__dict__.pop("_multipole_coefficients", None)


def apply_magnet_ramp(element, values):
    """Assign absolute program values; omitted channels retain their current values."""
    if element._magnet_order is None:
        n = element.ramp.max_order + 1
        if len(element.knl) < n:
            element.knl = np.pad(element.knl, (0, n - len(element.knl)))
        if len(element.ksl) < n:
            element.ksl = np.pad(element.ksl, (0, n - len(element.ksl)))
        for (order, skew), value in zip(element.ramp.components, values, strict=True):
            (element.ksl if skew else element.knl)[order] = value
    else:
        for (order, skew), value in zip(element.ramp.components, values, strict=True):
            setattr(element, f"k{order}{'s' if skew else ''}l", float(value))
    refresh_magnet_strengths(element)


def update_magnet_strengths(element, reference_time, offset=0.):
    """Sample at reference_time + offset; tracking calls this once at bunch entry."""
    if element.ramp is not None:
        values = element.ramp.sample_values(reference_time, np.asarray([offset]))[0]
        apply_magnet_ramp(element, values)
