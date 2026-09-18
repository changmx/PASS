"""Relativistic particle kinematics in normalized momentum coordinates."""

import numpy as np


def compute_particle_beta(momentum_ratio, reference_beta_gamma):
    """Return beta for P/P0 and the reference beta*gamma."""
    particle_beta_gamma = momentum_ratio * reference_beta_gamma
    return particle_beta_gamma / np.sqrt(1.0 + particle_beta_gamma**2)
