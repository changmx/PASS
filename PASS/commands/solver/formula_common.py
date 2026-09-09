"""Shared helpers for analytic transverse field formulas."""

from __future__ import annotations

import numpy as np

from PASS.utils.constants import const


def macro_charge_to_physical(real_particle_count, charge_number):
    """Convert represented real-particle counts to signed Coulombs."""
    return np.asarray(real_particle_count, dtype=float) * float(charge_number) * const.e
