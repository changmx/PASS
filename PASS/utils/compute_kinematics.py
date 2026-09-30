"""Relativistic particle kinematics in normalized momentum coordinates."""

import numpy as np

from PASS.utils.constants import const


def compute_particle_beta(momentum_ratio, reference_beta_gamma):
    """Return beta for P/P0 and the reference beta*gamma."""
    particle_beta_gamma = momentum_ratio * reference_beta_gamma
    return particle_beta_gamma / np.sqrt(1.0 + particle_beta_gamma**2)


def boost_proper_velocity(proper_velocity, beta, gamma, *, xp=np, inverse=False):
    """Boost p/m [m/s] along the last coordinate; inverse returns lab momenta.

    Light-cone components avoid subtracting two large lab-frame energies
    when particles nearly match a relativistic electron beam.
    """
    if not 0 <= beta < 1 or not np.isfinite(gamma) or gamma < 1:
        raise ValueError("A Lorentz boost requires a finite subluminal velocity")
    if not np.isclose(beta * beta + 1 / gamma**2, 1, rtol=0, atol=2e-14):
        raise ValueError("Lorentz boost beta and gamma are inconsistent")
    transverse_sq = xp.sum(proper_velocity[..., :2]**2, axis=-1)
    parallel = proper_velocity[..., 2]
    energy = xp.sqrt(const.c**2 + transverse_sq + parallel**2)
    positive = energy + parallel
    negative = energy - parallel
    # Rationalize the smaller component for either direction of motion.
    positive = xp.where(parallel >= 0, positive, (const.c**2 + transverse_sq) / xp.maximum(negative, np.finfo(float).tiny))
    negative = (const.c**2 + transverse_sq) / positive
    scale = gamma * (1 + beta)
    if inverse:
        scale = 1 / scale
    result = proper_velocity.copy()
    result[..., 2] = 0.5 * (positive / scale - negative * scale)
    return result
