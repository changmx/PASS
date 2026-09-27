"""Explicit simplified true-secondary model; not the full Furman-Pivi process."""

import numpy as np

from .pusher import normalized_momentum


def true_secondary_yield(energy_ev, peak_yield, peak_energy_ev, shape, emission_energy_ev, xp=np):
    """Furman-Pivi Eq. 32 shape, capped by the weighted incident energy budget."""
    energy = xp.asarray(energy_ev, dtype=xp.float64)
    positive = energy > 0
    log_ratio = xp.log(xp.where(positive, energy, 1)) - np.log(peak_energy_ev)
    log_shape = np.log(shape) + log_ratio - xp.logaddexp(np.log(shape - 1), shape * log_ratio)
    raw_yield = xp.where(positive, peak_yield * xp.exp(log_shape), 0)
    return xp.minimum(raw_yield, energy / emission_energy_ev)


def cosine_emission(normal_x, normal_y, energy_ev, generator, xp=np):
    """Lambert 3D hemisphere, with the normal pointing into the vacuum."""
    shape = normal_x.shape
    mu = xp.asarray(np.sqrt(generator.random(shape)))
    azimuth = xp.asarray(2 * np.pi * generator.random(shape))
    tangential = xp.sqrt(xp.maximum(1 - mu * mu, 0))
    magnitude = normalized_momentum(energy_ev, xp)
    tangent_x, tangent_y = -normal_y, normal_x
    ux = magnitude * (mu * normal_x + tangential * xp.cos(azimuth) * tangent_x)
    uy = magnitude * (mu * normal_y + tangential * xp.cos(azimuth) * tangent_y)
    uz = magnitude * tangential * xp.sin(azimuth)
    return ux, uy, uz


def isotropic_emission(n_particles, energy_ev, generator, xp=np):
    """Uniform solid-angle initial distribution at the prescribed kinetic energy."""
    mu = xp.asarray(2 * generator.random(n_particles) - 1)
    azimuth = xp.asarray(2 * np.pi * generator.random(n_particles))
    magnitude = normalized_momentum(energy_ev, xp)
    transverse = xp.sqrt(xp.maximum(1 - mu * mu, 0))
    return magnitude * transverse * xp.cos(azimuth), magnitude * transverse * xp.sin(azimuth), magnitude * mu
