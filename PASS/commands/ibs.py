"""Position-local IBS commands, Gaussian coefficients and binary collisions.

The matrices follow the Gaussian phase-space derivation of Bjorken and
Mtingwa, Particle Accelerators 13 (1983) 115, and Eqs. (8)--(13) of
https://arxiv.org/abs/2310.03504. They are evaluated here by independent
numerical quadrature, without an external tracking implementation.

Gaussian coefficient times are laboratory seconds; binary collision times
are beam-rest-frame seconds. For the Gaussian closure, the dimensionless momentum order is
``(Px/P0, Py/P0, delta/gamma)``; transverse slopes equal these normalized
mechanical momenta only to the paraxial order assumed by this model.
The Gaussian closure uses uncoupled betatron optics, with dispersion in both planes.
Neither arbitrary transverse coupling nor non-Gaussian collision kinetics
is represented by this closure.

``covariance_derivative`` is the derivative of the full momentum covariance
at fixed position. It is twice the conventional BM kernel, an essential
factor when constructing a Langevin kick. The conditional thermal covariance
is the inverse BM auxiliary matrix. Full tensor friction is used rather
than omitting the off-diagonal friction terms.
"""

from collections.abc import Sequence
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
from numbers import Integral
from pathlib import Path
import re
import uuid

import numpy as np
from scipy.constants import c
from scipy.integrate import quad_vec

from PASS.commands.command import Command
from PASS.para.schema.ibs import IBSItem, load_intrabeam_scattering
from PASS.utils.constants import const


@dataclass(frozen=True)
class IBSBeamParameters:
    """Physical beam parameters; emittances and lengths are geometric SI.

    ``n_particles`` counts real particles, and ``classical_radius`` is
    ``q**2/(4*pi*epsilon_0*m*c**2)`` using the full species mass, including
    for ions. ``sigma_delta`` is RMS total relative momentum deviation.
    ``sigma_z`` is laboratory bunch length; coasting beams instead use
    ``circumference`` and a uniform longitudinal density. ``coulomb_log``
    is supplied explicitly; this module does not infer impact cutoffs.
    """

    beta: float
    gamma: float
    geometric_emittance_x: float
    geometric_emittance_y: float
    sigma_delta: float
    sigma_z: float
    n_particles: float
    classical_radius: float
    coulomb_log: float
    bunched: bool = True
    circumference: float | None = None

    def __post_init__(self):
        for name in ("beta", "gamma", "geometric_emittance_x", "geometric_emittance_y", "sigma_delta"):
            _validate_scalar(getattr(self, name), name, positive=True)
        for name in ("n_particles", "classical_radius", "coulomb_log"):
            _validate_scalar(getattr(self, name), name, positive=False)
        if not 0.0 < self.beta < 1.0 or self.gamma <= 1.0:
            raise ValueError("IBS requires 0 < beta < 1 and gamma > 1")
        if not math.isclose(self.beta**2 + self.gamma**-2, 1.0, rel_tol=1.e-10, abs_tol=1.e-12):
            raise ValueError("IBS beta and gamma must describe the same reference velocity")
        if not isinstance(self.bunched, (bool, np.bool_)):
            raise ValueError("IBS bunched must be a boolean")
        if self.bunched:
            _validate_scalar(self.sigma_z, "sigma_z", positive=True)
        else:
            _validate_scalar(self.circumference, "circumference", positive=True)


@dataclass(frozen=True)
class IBSOptics:
    """Local uncoupled optics; dispersion derivatives use normalized momenta."""

    beta_x: float
    alpha_x: float
    beta_y: float
    alpha_y: float
    dispersion_x: float = 0.0
    dispersion_px: float = 0.0
    dispersion_y: float = 0.0
    dispersion_py: float = 0.0

    def __post_init__(self):
        for name in ("beta_x", "beta_y"):
            _validate_scalar(getattr(self, name), name, positive=True)
        for name in ("alpha_x", "alpha_y", "dispersion_x", "dispersion_px", "dispersion_y", "dispersion_py"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"IBS {name} must be finite")


@dataclass(frozen=True)
class IBSGrowthRates:
    """Fractional instantaneous rates in inverse laboratory seconds.

    ``emittance_x/y`` mean ``d(log(epsilon_x/y))/dt``. ``sigma_delta``
    means ``d(log(sigma_delta))/dt`` at the collision kick, and
    ``variance_delta`` is twice that rate. No synchrotron phase averaging
    is included: RF and longitudinal transport determine the subsequent
    sharing of the longitudinal kick between bunch length and momentum.
    """

    emittance_x: float
    emittance_y: float
    sigma_delta: float
    variance_delta: float

    @property
    def amplitude_x(self):
        return self.emittance_x / 2.0

    @property
    def amplitude_y(self):
        return self.emittance_y / 2.0


@dataclass(frozen=True)
class IBSLocalCoefficients:
    """Coefficients averaged over a Gaussian bunch at one optics location.

    For thermal residual ``w = p - conditional_mean_matrix @ (x, y)``,
    use ``dw = -friction @ w * dt + sqrt(diffusion) @ dW``. Coordinates
    in that expression are centered on the bunch centroid. The diffusion
    is positive semidefinite, but its difference with friction can cool
    a hot degree of freedom. These are Gaussian spatial averages, not
    pointwise density-dependent coefficients.

    ``conditional_covariance`` equals ``inv(sum(auxiliary_matrices))``;
    ``covariance_derivative = diffusion - friction @ conditional_covariance
    - conditional_covariance @ friction.T``. Tensor entries have units
    of inverse seconds because their momentum coordinates are dimensionless.
    """

    friction: np.ndarray
    diffusion: np.ndarray
    covariance_derivative: np.ndarray
    conditional_covariance: np.ndarray
    conditional_mean_matrix: np.ndarray
    bm_kernel: np.ndarray
    auxiliary_matrices: tuple[np.ndarray, np.ndarray, np.ndarray]
    growth_rates: IBSGrowthRates


def _validate_scalar(value, name, *, positive):
    if value is None or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"IBS {name} must be finite")
    if (positive and value <= 0.0) or (not positive and value < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"IBS {name} must be {qualifier}")


def _auxiliary_matrices(beam, optics):
    matrices = []
    for index, beta, alpha, dispersion, derivative, emittance in (
        (0, optics.beta_x, optics.alpha_x, optics.dispersion_x, optics.dispersion_px, beam.geometric_emittance_x),
        (1, optics.beta_y, optics.alpha_y, optics.dispersion_y, optics.dispersion_py, beam.geometric_emittance_y),
    ):
        phi = derivative + alpha * dispersion / beta
        direction = np.zeros(3, dtype=np.float64)
        direction[index] = 1.0
        direction[2] = -beam.gamma * phi
        matrix = (beta / emittance) * np.outer(direction, direction)
        # H/epsilon, not H/beta: fixed-position Gaussian inverse covariance.
        matrix[2, 2] += beam.gamma**2 * dispersion**2 / (beta * emittance)
        matrices.append(matrix)
    longitudinal = np.zeros((3, 3), dtype=np.float64)
    longitudinal[2, 2] = (beam.gamma / beam.sigma_delta)**2
    matrices.append(longitudinal)
    return tuple(matrices)


def _conditional_mean_matrix(beam, optics):
    dispersion = np.array([optics.dispersion_x, optics.dispersion_y], dtype=np.float64)
    momentum_dispersion = np.array([optics.dispersion_px, optics.dispersion_py, 1.0 / beam.gamma], dtype=np.float64)
    position_covariance = np.diag([beam.geometric_emittance_x * optics.beta_x, beam.geometric_emittance_y * optics.beta_y])
    position_covariance += beam.sigma_delta**2 * np.outer(dispersion, dispersion)
    cross_covariance = beam.sigma_delta**2 * np.outer(momentum_dispersion, dispersion)
    cross_covariance[0, 0] -= beam.geometric_emittance_x * optics.alpha_x
    cross_covariance[1, 1] -= beam.geometric_emittance_y * optics.alpha_y
    return np.linalg.solve(position_covariance, cross_covariance.T).T


def _integrate_eigenvalues(eigenvalues, rtol):
    """Compute the three positive BM integrals after scaling to a finite interval."""
    scale = float(np.exp(np.mean(np.log(eigenvalues))))
    values = eigenvalues / scale

    def integrand(u):
        # lambda = scale*u**2/(1-u**2) removes both infinite tails and sqrt(0).
        denominators = values * (1.0 - u * u) + u * u
        return (2.0 * u * u / math.sqrt(float(np.prod(denominators)))) / denominators

    points = np.sqrt(values / (1.0 + values))
    result, error, info = quad_vec(integrand, 0.0, 1.0, epsabs=0.0, epsrel=rtol, points=points.tolist(), limit=1000, full_output=True)
    if not info.success or not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ArithmeticError(f"IBS integral failed to converge: {info.message}; estimated error={error:g}")
    return result / scale


def compute_local_coefficients(beam: IBSBeamParameters, optics: IBSOptics, *, rtol=1.e-9) -> IBSLocalCoefficients:
    """Evaluate local BM rates and a full Gaussian tensor kinetic closure.

    If J is the integral of ``sqrt(lambda)*(L+lambda*I)^-1 /
    sqrt(det(L+lambda*I))``, the conventional BM kernel is
    ``I_BM = a*(trace(J)*I - 3*J)``. The physical covariance derivative
    is ``2*I_BM``. Indeed equal-mass Rutherford scattering gives
    ``D_v = 4*pi*n*r0**2*c**4*logLambda * <(I-u*u/u**2)/|u|>``;
    averaging Gaussian relative velocities and spatial density, and
    transforming rest-frame time to lab time, yields the factor ``2*a``
    below. Here ``a`` is Eq. (8) of the cited paper.

    Coasting density uses ``sigma_z_eff = circumference/(2*sqrt(pi))``.
    No additional factor is applied to the physical longitudinal kick.
    """
    if not np.isfinite(rtol) or not 1.e-13 <= rtol < 0.1:
        raise ValueError("IBS quadrature rtol must be in [1e-13, 0.1)")
    matrices = _auxiliary_matrices(beam, optics)
    auxiliary = sum(matrices)
    eigenvalues, rotation = np.linalg.eigh(auxiliary)
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
        raise ValueError("IBS conditional momentum covariance is not numerically positive definite")
    covariance = (rotation / eigenvalues) @ rotation.T
    conditional_mean = _conditional_mean_matrix(beam, optics)
    if beam.n_particles == 0.0 or beam.classical_radius == 0.0 or beam.coulomb_log == 0.0:
        zero = np.zeros((3, 3), dtype=np.float64)
        return IBSLocalCoefficients(zero.copy(), zero.copy(), zero.copy(), covariance, conditional_mean, zero.copy(), matrices,
                                    IBSGrowthRates(0.0, 0.0, 0.0, 0.0))
    integral = _integrate_eigenvalues(eigenvalues, rtol)
    sigma_z = beam.sigma_z if beam.bunched else beam.circumference / (2.0 * math.sqrt(math.pi))
    prefactor = (
        c * beam.n_particles * beam.classical_radius**2 * beam.coulomb_log /
        (4.0 * math.pi * beam.beta**3 * beam.gamma**4 * beam.geometric_emittance_x * beam.geometric_emittance_y * sigma_z * beam.sigma_delta))
    friction = (rotation * (prefactor * integral * eigenvalues)) @ rotation.T
    diffusion = (rotation * (prefactor * (np.sum(integral) - integral))) @ rotation.T
    derivative_eigenvalues = np.sum(integral) - 3.0 * integral
    if np.ptp(eigenvalues) <= 32.0 * np.finfo(np.float64).eps * np.max(eigenvalues):
        derivative_eigenvalues[:] = 0.0
    derivative = (rotation * (prefactor * derivative_eigenvalues)) @ rotation.T
    kernel = derivative / 2.0
    rates = [float(np.sum(matrix * kernel)) for matrix in matrices]
    if not all(np.all(np.isfinite(matrix)) for matrix in (friction, diffusion, derivative)) or not np.all(np.isfinite(rates)):
        raise FloatingPointError("IBS coefficients exceed float64 range")
    return IBSLocalCoefficients(friction, diffusion, derivative, covariance, conditional_mean, kernel, matrices,
                                IBSGrowthRates(rates[0], rates[1], rates[2], 2.0 * rates[2]))


def ring_average_growth_rates(coefficients: Sequence[IBSLocalCoefficients], weights) -> IBSGrowthRates:
    """Average local fractional rates with explicit nonnegative quadrature weights.

    Use lattice segment lengths for fixed energy, or physical residence times
    when the reference velocity varies. Endpoints/periodicity belong to the
    caller's quadrature rule; this routine never invents a missing ring segment.
    """
    values = list(coefficients)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size != len(values) or not values:
        raise ValueError("IBS ring weights must match a nonempty sequence of local coefficients")
    total_weight = np.sum(weights)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("IBS ring weights must be finite, non-negative and have positive sum")
    rates = np.array(
        [[value.growth_rates.emittance_x, value.growth_rates.emittance_y, value.growth_rates.sigma_delta, value.growth_rates.variance_delta]
         for value in values],
        dtype=np.float64)
    average = (weights / total_weight) @ rates
    return IBSGrowthRates(*map(float, average))


compute_bjorken_mtingwa = compute_local_coefficients


def _check_positive(value, name, *, allow_zero=False):
    """Validate physical scalar inputs before consuming random numbers."""
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return value


def _get_grid(positions, grid_shape, bounds, xp):
    """Build an all-particle Cartesian mesh without discarding tails."""
    if bounds is None:
        lower = xp.min(positions, axis=0)
        upper = xp.max(positions, axis=0)
        if bool(xp.any(upper <= lower)):
            raise ValueError("IBS automatic bounds need nonzero extent in all three axes; supply explicit bounds for a degenerate distribution.")
        lower = xp.nextafter(lower, -xp.inf)
        upper = xp.nextafter(upper, xp.inf)
    else:
        bounds = xp.asarray(bounds, dtype=xp.float64)
        if bounds.shape != (2, 3) or not bool(xp.all(xp.isfinite(bounds))):
            raise ValueError("IBS bounds must be finite with shape (2, 3): [lower_xyz, upper_xyz].")
        lower, upper = bounds[0], bounds[1]
        if bool(xp.any(upper <= lower)):
            raise ValueError("IBS upper bounds must exceed lower bounds on every axis.")
        if bool(xp.any((positions < lower) | (positions > upper))):
            raise ValueError("IBS positions fall outside the explicit bounds; particles may not be silently excluded.")
    spacing = (upper - lower) / xp.asarray(grid_shape, dtype=xp.float64)
    cell_volume = float(xp.prod(spacing))
    if not math.isfinite(cell_volume) or cell_volume <= 0.0:
        raise ValueError("IBS collision cells need finite positive volume.")
    indices = xp.floor((positions - lower) / spacing).astype(xp.int64)
    # The upper face belongs to the last cell, including exact explicit edges.
    indices = xp.minimum(indices, xp.asarray(grid_shape, dtype=xp.int64) - 1)
    cell_indices = (indices[:, 0] * grid_shape[1] + indices[:, 1]) * grid_shape[2] + indices[:, 2]
    return cell_indices, cell_volume, xp.stack((lower, upper))


def _rotate_pairs(relative, variance, xp, rng):
    """Rotate pair-COM proper momentum with an isotropic azimuth."""
    n_pairs = relative.shape[0]
    magnitude = xp.linalg.norm(relative, axis=1)
    direction = relative / magnitude[:, None]
    axis = xp.eye(3, dtype=xp.float64)[xp.argmin(xp.abs(direction), axis=1)]
    first = xp.cross(direction, axis)
    first /= xp.linalg.norm(first, axis=1)[:, None]
    second = xp.cross(direction, first)
    azimuth = xp.asarray(rng.uniform(0.0, 2.0 * math.pi, size=n_pairs), dtype=xp.float64)
    delta = xp.asarray(rng.normal(size=n_pairs), dtype=xp.float64) * xp.sqrt(variance)
    sine = 2.0 * delta / (1.0 + delta * delta)
    cosine = (1.0 - delta * delta) / (1.0 + delta * delta)
    transverse = xp.cos(azimuth)[:, None] * first + xp.sin(azimuth)[:, None] * second
    rotated = cosine[:, None] * direction + sine[:, None] * transverse
    # Suppress accumulated norm drift over many angular substeps.
    return magnitude[:, None] * rotated / xp.linalg.norm(rotated, axis=1)[:, None]


def scatter_binary(positions,
                   velocities,
                   *,
                   xp,
                   rng,
                   macro_weight,
                   mass_kg,
                   charge_coulomb,
                   dt,
                   coulomb_log,
                   grid_shape=(8, 8, 8),
                   max_scattering=0.05,
                   max_substeps=1000,
                   bounds=None,
                   collision_steps=1):
    """Return new proper velocities and collision diagnostics on NumPy/CuPy.

    ``positions`` has shape (N, 3), in metres at equal beam-rest-frame time.
    ``velocities`` is specifically proper velocity p_rest/m (m/s), not dx/dt.
    It has the same shape. ``dt`` is elapsed beam-rest-frame time in seconds;
    ``mass_kg`` and ``charge_coulomb`` refer to a complete physical particle.
    Each macro particle represents the same ``macro_weight`` real particles.
    ``rng`` implements NumPy-like permutation, normal, and uniform methods;
    host NumPy draws may also drive a CuPy calculation reproducibly.

    Automatic bounds contain every position and are recomputed on each call.
    Fixed ``bounds=[lower_xyz, upper_xyz]`` are preferable for homogeneous or
    periodic-volume tests. Outside particles cause an error. Empty and singleton
    cells cannot collide and are reported. In an odd cell, a uniformly random
    particle sits out; multiplying paired exposure by N/(N-1) compensates its
    omission in expectation. The number density itself remains N*weight/volume.

    Rotations take place in the exact relativistic pair COM and preserve pair
    four-momentum. The nonrelativistic Rutherford frequency uses twice the
    individual pair-COM speed. All input and output proper speeds must be at
    most 0.05*c; this bound limits the thermal relativistic correction but does
    not make the collision operator relativistically exact.

    ``collision_steps`` divides dt into equal positive physical steps and
    resamples local partners at every step. Positions, grid bounds and cell
    memberships remain frozen for the call. Each selected pair's additional
    angular substeps keep Var[tan(theta/2)] at most ``max_scattering``.
    Exceeding ``max_substeps`` for any pair in any physical step raises
    ValueError instead of clipping its strength. Convergence in physical
    collision step size and in the caller's spatial sampling is still required.
    Neither input array is mutated, including on failure. The caller owns RNG
    state and must restore it if failed operations are retried deterministically.

    Diagnostics are JSON-compatible. ``n_pairs`` is the eligible pair count
    per physical step; ``n_pairings`` counts pair events across completed
    steps, including zero-relative-speed events. ``n_collision_steps`` counts
    completed physical steps; zero exposure or no eligible pairs gives zero.
    ``n_substeps`` is the largest angular subdivision count of any pair in
    any physical step, and ``max_scattering`` is the largest variance before
    angular subdivision. ``n_zero_relative_pairs`` accumulates over steps.
    ``n_unpaired_particles`` counts omissions per physical step, not unique
    particle identities. ``singleton_fraction`` is the fraction of particles
    in singleton cells. ``occupancy_histogram`` includes empty cells and has
    entries {"particles": occupancy, "cells": count}; it is empty when no
    grid was constructed. ``grid_bounds_m`` uses beam-rest-frame metres.
    ``proper_speed_max_over_c`` is the largest input or intermediate speed.

    The scattering law is independently derived from Rutherford deflection,
    following Takizuka--Abe, J. Comput. Phys. 25 (1977) 205--219,
    https://doi.org/10.1016/0021-9991(77)90099-7, and the discussion by Wang
    et al., J. Comput. Phys. 227 (2008) 4308--4329,
    https://www.osti.gov/servlets/purl/942017. For reduced mass mu=m/2,

        Var[tan(theta/2)] = q**4 n log(Lambda) dt / (8 pi epsilon0**2 mu**2 g**3).

    Integrating theta=2*q**2/(4*pi*epsilon0*mu*g**2*b) over encounter rate
    2*pi*n*g*b*db gives this small-angle variance. The approximation is the
    weakly coupled Landau operator, not hard scattering or Touschek losses.
    No external tracking implementation is imported, translated, or copied.
    """
    macro_weight = _check_positive(macro_weight, "macro_weight")
    mass_kg = _check_positive(mass_kg, "mass_kg")
    charge_coulomb = float(charge_coulomb)
    if not math.isfinite(charge_coulomb):
        raise ValueError("charge_coulomb must be finite.")
    dt = _check_positive(dt, "dt", allow_zero=True)
    coulomb_log = _check_positive(coulomb_log, "coulomb_log")
    max_scattering = _check_positive(max_scattering, "max_scattering")
    if max_scattering > 0.05:
        raise ValueError("max_scattering must be at most 0.05 for the small-angle model.")
    if isinstance(max_substeps, bool) or not isinstance(max_substeps, Integral) or max_substeps < 1:
        raise ValueError("max_substeps must be a positive integer.")
    if isinstance(collision_steps, bool) or not isinstance(collision_steps, Integral) or collision_steps < 1:
        raise ValueError("collision_steps must be a positive integer.")
    collision_steps = int(collision_steps)
    if len(grid_shape) != 3 or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in grid_shape):
        raise ValueError("grid_shape must contain three positive integers.")
    grid_shape = tuple(int(n) for n in grid_shape)
    n_grid_cells = math.prod(grid_shape)
    if n_grid_cells > np.iinfo(np.int64).max:
        raise ValueError("grid_shape exceeds the integer range of collision cell indices.")
    positions = xp.asarray(positions, dtype=xp.float64)
    velocities = xp.asarray(velocities, dtype=xp.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or velocities.shape != positions.shape:
        raise ValueError("IBS positions and velocities must have matching shape (N, 3).")
    if not bool(xp.all(xp.isfinite(positions))) or not bool(xp.all(xp.isfinite(velocities))):
        raise ValueError("IBS positions and velocities must be finite.")
    n_particles = positions.shape[0]
    result = velocities.copy()
    speed_over_c = float(xp.max(xp.linalg.norm(velocities, axis=1))) / const.c if n_particles else 0.0
    if speed_over_c > 0.05:
        raise ValueError("IBS binary collisions require beam-rest proper speeds <= 0.05*c.")
    diagnostics = {
        "n_particles": n_particles,
        "n_pairs": 0,
        "n_pairings": 0,
        "n_collision_steps": 0,
        "grid_shape": list(grid_shape),
        "grid_bounds_m": None,
        "n_grid_cells": n_grid_cells,
        "n_empty_cells": None,
        "occupancy_histogram": [],
        "singleton_fraction": 0.0,
        "n_occupied_cells": 0,
        "n_singleton_cells": 0,
        "n_odd_cells": 0,
        "n_unpaired_particles": n_particles,
        "n_zero_relative_pairs": 0,
        "cell_volume_m3": None,
        "density_min_m3": 0.0,
        "density_max_m3": 0.0,
        "max_scattering": 0.0,
        "n_substeps": 0,
        "proper_speed_max_over_c": speed_over_c,
    }
    if n_particles < 2 or dt == 0.0 or charge_coulomb == 0.0:
        return result, diagnostics
    step_dt = dt / collision_steps
    if step_dt <= 0.0:
        raise ValueError("IBS physical collision step is below floating-point resolution.")
    cell_indices, cell_volume, grid_bounds = _get_grid(positions, grid_shape, bounds, xp)
    order = xp.lexsort(xp.stack((xp.asarray(rng.permutation(n_particles)), cell_indices)))
    _, counts = xp.unique(cell_indices[order], return_counts=True)
    starts = xp.cumsum(counts) - counts
    counts_per_particle = xp.repeat(counts, counts)
    ranks = xp.arange(n_particles) - xp.repeat(starts, counts)
    left = xp.flatnonzero((ranks % 2 == 0) & (ranks < counts_per_particle - 1))
    pair_counts = counts_per_particle[left]
    n_pairs = left.size
    occupied_counts, cells_per_count = xp.unique(counts, return_counts=True)
    histogram = [{"particles": int(count), "cells": int(n_cells)} for count, n_cells in zip(occupied_counts.tolist(), cells_per_count.tolist())]
    n_empty_cells = n_grid_cells - counts.size
    if n_empty_cells:
        histogram.insert(0, {"particles": 0, "cells": n_empty_cells})
    n_singleton_cells = int(xp.count_nonzero(counts == 1))
    diagnostics.update({
        "n_pairs": n_pairs,
        "grid_bounds_m": grid_bounds.tolist(),
        "n_empty_cells": n_empty_cells,
        "occupancy_histogram": histogram,
        "singleton_fraction": n_singleton_cells / n_particles,
        "n_occupied_cells": counts.size,
        "n_singleton_cells": n_singleton_cells,
        "n_odd_cells": int(xp.count_nonzero(counts % 2)),
        "n_unpaired_particles": n_particles - 2 * n_pairs,
        "cell_volume_m3": cell_volume,
        "density_min_m3": float(xp.min(counts)) * macro_weight / cell_volume,
        "density_max_m3": float(xp.max(counts)) * macro_weight / cell_volume,
    })
    if n_pairs == 0:
        return result, diagnostics
    density = pair_counts * (macro_weight / cell_volume)
    exposure = xp.where(pair_counts % 2, pair_counts / (pair_counts - 1), 1.0)
    reduced_mass = 0.5 * mass_kg
    strength = charge_coulomb**4 * coulomb_log * step_dt / (8.0 * math.pi * const.epsilon0**2 * reduced_mass**2)
    for collision_step in range(collision_steps):
        # Each physical step uses new local partners, including a new odd tail.
        if collision_step:
            order = xp.lexsort(xp.stack((xp.asarray(rng.permutation(n_particles)), cell_indices)))
        first_indices, second_indices = order[left], order[left + 1]
        first, second = result[first_indices], result[second_indices]
        first_gamma = xp.sqrt(1.0 + xp.sum((first / const.c)**2, axis=1))
        second_gamma = xp.sqrt(1.0 + xp.sum((second / const.c)**2, axis=1))
        center = 0.5 * (first + second)
        beta_com = (first + second) / (const.c * (first_gamma + second_gamma))[:, None]
        gamma_com = 1.0 / xp.sqrt(1.0 - xp.sum(beta_com * beta_com, axis=1))
        boost_factor = gamma_com**2 / (gamma_com + 1.0)
        relative = 0.5 * (first - second)
        projection = xp.sum(relative * beta_com, axis=1)
        relative += (boost_factor * projection - 0.5 * gamma_com * const.c * (first_gamma - second_gamma))[:, None] * beta_com
        magnitude = xp.linalg.norm(relative, axis=1)
        relative_speed = 2.0 * magnitude / xp.sqrt(1.0 + (magnitude / const.c)**2)
        moving = relative_speed > 0.0
        variance = xp.zeros(n_pairs, dtype=xp.float64)
        variance[moving] = strength * density[moving] * exposure[moving] / relative_speed[moving]**3
        if not bool(xp.all(xp.isfinite(variance))):
            raise ValueError("IBS collision strength overflowed; check density, units, and the collision interval.")
        max_variance = float(xp.max(variance))
        required_substeps = max(1, math.ceil(max_variance / max_scattering))
        if required_substeps > max_substeps:
            raise ValueError(
                f"IBS binary collisions require {required_substeps} angular substeps, exceeding {max_substeps}; reduce the physical collision step.")
        substeps = xp.maximum(1, xp.ceil(variance / max_scattering).astype(xp.int64))
        for step in range(required_substeps):
            active = xp.flatnonzero(moving & (substeps > step))
            if active.size:
                relative[active] = _rotate_pairs(relative[active], variance[active] / substeps[active], xp, rng)
        projection = xp.sum(relative * beta_com, axis=1)
        boosted_relative = relative + (boost_factor * projection)[:, None] * beta_com
        result[first_indices] = center + boosted_relative
        result[second_indices] = center - boosted_relative
        # An identical-velocity pair has no resolvable relative motion to scatter.
        result[first_indices[~moving]] = first[~moving]
        result[second_indices[~moving]] = second[~moving]
        output_speed_over_c = float(xp.max(xp.linalg.norm(result, axis=1))) / const.c
        if not math.isfinite(output_speed_over_c) or output_speed_over_c > 0.05:
            raise ValueError("IBS scattering produced beam-rest proper speed above 0.05*c; the nonrelativistic thermal model is outside its domain.")
        diagnostics["n_collision_steps"] += 1
        diagnostics["n_pairings"] += n_pairs
        diagnostics["n_zero_relative_pairs"] += int(xp.count_nonzero(~moving))
        diagnostics["max_scattering"] = max(diagnostics["max_scattering"], max_variance)
        diagnostics["n_substeps"] = max(diagnostics["n_substeps"], required_substeps)
        diagnostics["proper_speed_max_over_c"] = max(diagnostics["proper_speed_max_over_c"], output_speed_over_c)
    return result, diagnostics


def _as_host(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


def _species_parameters(bunch):
    """Return whole-particle mass [kg], signed charge [C], and radius [m]."""
    mass_number = int(bunch.num_proton) + int(bunch.num_neutron)
    mass_energy = float(bunch.m0)
    if getattr(bunch, "particle_type", None) == "Ion":
        mass_energy *= mass_number
    mass_kg = mass_energy * const.e / const.c**2
    charge = int(bunch.num_charge) * const.e
    if not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("IBS requires a finite positive whole-particle mass")
    radius = charge**2 / (4 * np.pi * const.epsilon0 * mass_kg * const.c**2)
    return mass_kg, charge, radius


def _validate_reference(bunch):
    beta, gamma = float(bunch.beta), float(bunch.gamma)
    if not 0 < beta < 1 or not np.isfinite(gamma) or gamma < 1:
        raise ValueError("IBS requires a positive subluminal reference velocity")
    if not np.isclose(beta**2, 1 - 1 / gamma**2, rtol=1e-10, atol=1e-14):
        raise ValueError("IBS reference beta and gamma are inconsistent")
    if not np.isfinite(bunch.ratio) or bunch.ratio < 0:
        raise ValueError("IBS macroparticle weight must be finite and nonnegative")


def _validate_momenta(coordinates, xp):
    momentum_ratio = 1 + coordinates[:, 5]
    transverse_sq = coordinates[:, 1]**2 + coordinates[:, 3]**2
    if (not bool(xp.all(xp.isfinite(coordinates))) or bool(xp.any(momentum_ratio <= 0)) or bool(xp.any(momentum_ratio**2 <= transverse_sq))):
        raise ValueError("IBS requires finite coordinates and forward physical mechanical momenta")


def _gather_bunch(p, bunch):
    """Gather live rows in particle-identity order, independent of sorting."""
    xp = p.xp
    start, end = int(bunch.start_idx), int(bunch.end_idx)
    indices = xp.flatnonzero(p.tag[start:end] > 0) + start
    indices = indices[xp.argsort(p.tag[indices])]
    coordinates = xp.column_stack([getattr(p, name)[indices] for name in ("x", "px", "y", "py", "z", "dp")]).astype(xp.float64)
    _validate_momenta(coordinates, xp)
    return indices, coordinates


def _measure_parameters(coordinates, bunch, configuration, optics, xp):
    """Measure population moments after centroid and dispersion subtraction."""
    n_alive = len(coordinates)
    if n_alive < 4:
        raise ValueError("Gaussian IBS requires at least four live macroparticles with nonzero emittances and momentum spread")
    centered = coordinates - xp.mean(coordinates, axis=0)
    delta = centered[:, 5]
    emittances = []
    for position, momentum, dispersion, dispersion_p in (
        (0, 1, optics.dispersion_x, optics.dispersion_px),
        (2, 3, optics.dispersion_y, optics.dispersion_py),
    ):
        pair = xp.column_stack((centered[:, position] - dispersion * delta, centered[:, momentum] - dispersion_p * delta))
        covariance = _as_host(pair.T @ pair / n_alive)
        determinant = float(np.linalg.det(covariance))
        if determinant <= 0 or not np.isfinite(determinant):
            raise ValueError("Gaussian IBS requires strictly positive intrinsic transverse emittances")
        emittances.append(np.sqrt(determinant))
    sigma_delta = float(xp.sqrt(xp.mean(delta**2)))
    sigma_z = float(xp.sqrt(xp.mean(centered[:, 4]**2)))
    _, _, radius = _species_parameters(bunch)
    return IBSBeamParameters(
        beta=float(bunch.beta),
        gamma=float(bunch.gamma),
        geometric_emittance_x=emittances[0],
        geometric_emittance_y=emittances[1],
        sigma_delta=sigma_delta,
        sigma_z=sigma_z,
        n_particles=n_alive * float(bunch.ratio),
        classical_radius=radius,
        coulomb_log=configuration.coulomb_log,
        bunched=configuration.bunched,
        circumference=float(bunch.circum),
    )


def _measure_matching(coordinates, parameters, optics, tolerance, xp):
    """Compare centered normalized covariance with the uncoupled Gaussian model.

    The entrywise tolerance concerns measured second moments, not normality.
    Including longitudinal correlations for a bunch rejects an unsupported
    chirp that the Gaussian conditional collision covariance would omit.
    """
    centered = coordinates - xp.mean(coordinates, axis=0)
    delta = centered[:, 5]
    normalized = []
    for position, momentum, beta, alpha, dispersion, dispersion_p, emittance in (
        (0, 1, optics.beta_x, optics.alpha_x, optics.dispersion_x, optics.dispersion_px, parameters.geometric_emittance_x),
        (2, 3, optics.beta_y, optics.alpha_y, optics.dispersion_y, optics.dispersion_py, parameters.geometric_emittance_y),
    ):
        intrinsic_position = centered[:, position] - dispersion * delta
        intrinsic_momentum = centered[:, momentum] - dispersion_p * delta
        scale = np.sqrt(emittance * beta)
        normalized.extend((intrinsic_position / scale, (beta * intrinsic_momentum + alpha * intrinsic_position) / scale))
    normalized.append(delta / parameters.sigma_delta)
    if parameters.bunched:
        normalized.append(centered[:, 4] / parameters.sigma_z)
    normalized = xp.column_stack(normalized)
    covariance = _as_host(normalized.T @ normalized / len(coordinates))
    error = float(np.max(np.abs(covariance - np.eye(covariance.shape[0]))))
    return dict(normalized_covariance=covariance.tolist(),
                dimension=covariance.shape[0],
                max_abs_error=error,
                tolerance=float(tolerance),
                matched=bool(np.isfinite(error) and error <= tolerance))


def _require_matching(matching):
    if not matching["matched"]:
        raise ValueError(f"IBS kinetic requires matched uncoupled Gaussian optics and uncorrelated longitudinal coordinates: "
                         f"normalized covariance error={matching['max_abs_error']:.6g} exceeds Matching tolerance={matching['tolerance']:.6g}. "
                         "Check local optics, dispersion and coupling; use more macroparticles if sampling noise dominates. "
                         "For an evolving non-Gaussian or mismatched beam, use and converge the binary model.")


def _slice_density(coordinates, indices, bunch, configuration, parameters, xp):
    """Use saved membership and widths without modifying the user's SliceSet."""
    if configuration.slice_set is None:
        return xp.ones(len(coordinates), dtype=xp.float64)
    slice_set = getattr(bunch, "slice_sets", {}).get(configuration.slice_set)
    if slice_set is None or slice_set.slice_id is None or slice_set.slice_table is None:
        raise ValueError("IBS requires an executed Slicer; execute it again after regrouping")
    if getattr(slice_set, "purpose", "general") != "general":
        raise ValueError("IBS requires a general-purpose Slice set")
    expected = {"z_rel"} if configuration.bunched else {"z_rel", "z_periodic"}
    if slice_set.coordinate not in expected:
        raise ValueError(f"IBS Slice set coordinate must be one of {sorted(expected)}")
    slice_indices = xp.asarray(slice_set.slice_id)
    if slice_indices.ndim != 1 or len(slice_indices) != bunch.end_idx - bunch.start_idx or slice_indices.dtype.kind not in "iu":
        raise ValueError("IBS saved slice membership does not match the bunch")
    slice_indices = slice_indices[indices - bunch.start_idx]
    widths = xp.asarray(slice_set.slice_table["delta_z"], dtype=xp.float64)
    if widths.ndim != 1 or len(widths) == 0 or bool(xp.any(~xp.isfinite(widths))) or bool(xp.any(widths <= 0)):
        raise ValueError("IBS saved slice widths must be finite and positive")
    if bool(xp.any(slice_indices < 0)) or bool(xp.any(slice_indices >= len(widths))):
        raise ValueError("IBS live particles need valid saved memberships; execute Slicer after injection or regrouping")
    counts = xp.bincount(slice_indices, minlength=len(widths))
    normalized_density = counts[slice_indices] / (len(coordinates) * widths[slice_indices])
    effective_length = 2 * np.sqrt(np.pi) * parameters.sigma_z if configuration.bunched else parameters.circumference
    return effective_length * normalized_density


def _rest_momenta(coordinates, bunch, xp):
    """Lorentz-boost mechanical momenta into proper velocities p_rest/M."""
    momentum_ratio = 1 + coordinates[:, 5]
    transverse_sq = coordinates[:, 1]**2 + coordinates[:, 3]**2
    longitudinal_ratio = xp.sqrt(momentum_ratio**2 - transverse_sq)
    energy_ratio = xp.sqrt(1 / bunch.gamma**2 + bunch.beta**2 * momentum_ratio**2)
    # Rationalization avoids subtracting two O(gamma^2) laboratory terms.
    longitudinal = bunch.beta * const.c * (coordinates[:, 5] *
                                           (2 + coordinates[:, 5]) - bunch.gamma**2 * transverse_sq) / (longitudinal_ratio + energy_ratio)
    scale = bunch.beta * bunch.gamma * const.c
    return xp.column_stack((scale * coordinates[:, 1], scale * coordinates[:, 3], longitudinal))


def _check_thermal_limit(coordinates, bunch, xp):
    maximum = float(xp.max(xp.linalg.norm(_rest_momenta(coordinates, bunch, xp), axis=1))) / const.c
    if maximum > 0.05:
        raise ValueError(f"IBS small thermal-velocity model requires |p_rest|/(M c) <= 0.05; measured {maximum:.6g}")
    return maximum


def _apply_rest_momenta(coordinates, proper_velocities, bunch, xp):
    reference_beta_gamma = bunch.beta * bunch.gamma
    coordinates[:, 1] = proper_velocities[:, 0] / (reference_beta_gamma * const.c)
    coordinates[:, 3] = proper_velocities[:, 1] / (reference_beta_gamma * const.c)
    particle_gamma = xp.sqrt(1 + xp.sum((proper_velocities / const.c)**2, axis=1))
    longitudinal_ratio = particle_gamma + proper_velocities[:, 2] / (bunch.beta * const.c)
    if bool(xp.any(longitudinal_ratio <= 0)):
        raise ValueError("IBS produced backward laboratory motion outside PASS's forward-tracking domain")
    transverse_sq = coordinates[:, 1]**2 + coordinates[:, 3]**2
    momentum_ratio = xp.sqrt(transverse_sq + longitudinal_ratio**2)
    coordinates[:, 5] = (transverse_sq + (longitudinal_ratio - 1) * (longitudinal_ratio + 1)) / (momentum_ratio + 1)
    _validate_momenta(coordinates, xp)


def _advance_kinetic(coordinates, indices, bunch, configuration, optics, dt, rng, xp):
    """Integrate the conditional Gaussian Ornstein--Uhlenbeck collision step."""
    elapsed = 0.0
    n_substeps = 0
    initial_rates = None
    initial_parameters = None
    initial_matching = None
    max_matching_error = 0.0
    thermal_limit = _check_thermal_limit(coordinates, bunch, xp)
    while elapsed < dt:
        if n_substeps >= configuration.max_substeps:
            raise ValueError("IBS kinetic kick exceeded Max substeps; reduce Interaction length (m)")
        parameters = _measure_parameters(coordinates, bunch, configuration, optics, xp)
        matching = _measure_matching(coordinates, parameters, optics, configuration.matching_tolerance, xp)
        _require_matching(matching)
        max_matching_error = max(max_matching_error, matching["max_abs_error"])
        coefficients = compute_local_coefficients(parameters, optics)
        if initial_rates is None:
            initial_rates = asdict(coefficients.growth_rates)
            initial_parameters = asdict(parameters)
            initial_matching = matching
        density = _slice_density(coordinates, indices, bunch, configuration, parameters, xp)
        _, eigenvectors = np.linalg.eigh(coefficients.conditional_covariance)
        friction = np.diag(eigenvectors.T @ coefficients.friction @ eigenvectors).copy()
        diffusion = np.diag(eigenvectors.T @ coefficients.diffusion @ eigenvectors).copy()
        if np.any(friction < 0) or np.any(diffusion < 0):
            raise ArithmeticError("IBS collision operator has negative friction or diffusion eigenvalues")
        maximum_frequency = float(np.max(friction)) * float(xp.max(density))
        step = min(dt - elapsed, configuration.max_scattering / maximum_frequency) if maximum_frequency > 0 else dt - elapsed
        if step <= 0 or elapsed + step == elapsed:
            raise ArithmeticError("IBS collision time step is below floating-point resolution")
        momenta = coordinates[:, [1, 3, 5]].copy()
        momenta[:, 2] /= bunch.gamma
        mean = xp.mean(momenta, axis=0)
        positions = coordinates[:, [0, 2]]
        positions = positions - xp.mean(positions, axis=0)
        conditional_mean = mean + positions @ xp.asarray(coefficients.conditional_mean_matrix).T
        rotation = xp.asarray(eigenvectors)
        residual = (momenta - conditional_mean) @ rotation
        rates = xp.asarray(friction)
        diffusion_rates = xp.asarray(diffusion)
        exposure = step * density[:, None]
        exponent = -rates[None, :] * exposure
        decay = xp.exp(exponent)
        denominator = xp.where(rates > 0, 2 * rates, 1)
        variance = diffusion_rates[None, :] * (-xp.expm1(2 * exponent)) / denominator[None, :]
        variance = xp.where(rates[None, :] > 0, variance, diffusion_rates[None, :] * exposure)
        noise = xp.asarray(rng.standard_normal((len(coordinates), 3))) * xp.sqrt(variance)
        # Preserve the bunch centroid; compensate the removed noise variance.
        noise -= xp.mean(noise, axis=0)
        noise *= np.sqrt(len(coordinates) / (len(coordinates) - 1))
        updated = conditional_mean + (decay * residual + noise) @ rotation.T
        updated -= xp.mean(updated, axis=0) - mean
        coordinates[:, 1] = updated[:, 0]
        coordinates[:, 3] = updated[:, 1]
        coordinates[:, 5] = bunch.gamma * updated[:, 2]
        _validate_momenta(coordinates, xp)
        _check_thermal_limit(coordinates, bunch, xp)
        elapsed += step
        n_substeps += 1
    parameters = _measure_parameters(coordinates, bunch, configuration, optics, xp)
    final_matching = _measure_matching(coordinates, parameters, optics, configuration.matching_tolerance, xp)
    _require_matching(final_matching)
    max_matching_error = max(max_matching_error, final_matching["max_abs_error"])
    return dict(substeps=n_substeps,
                initial_parameters=initial_parameters,
                growth_rates=initial_rates,
                proper_velocity_max_over_c=thermal_limit,
                initial_matching=initial_matching,
                final_matching=final_matching,
                max_matching_error=max_matching_error)


def _advance_binary(coordinates, bunch, configuration, dt, rng, xp):
    mass_kg, charge, _ = _species_parameters(bunch)
    positions = coordinates[:, [0, 2, 4]].copy()
    bounds = np.array(configuration.grid_bounds, dtype=np.float64) if configuration.grid_bounds is not None else None
    if not configuration.bunched:
        positions[:, 2] = (positions[:, 2] + bunch.circum / 2) % bunch.circum - bunch.circum / 2
    positions[:, 2] *= bunch.gamma
    if not configuration.bunched and bounds is None:
        lower = _as_host(xp.min(positions, axis=0)).copy()
        upper = _as_host(xp.max(positions, axis=0)).copy()
        lower[2], upper[2] = -bunch.gamma * bunch.circum / 2, bunch.gamma * bunch.circum / 2
        bounds = np.stack((lower, upper))
    elif bounds is not None:
        if not configuration.bunched:
            if not np.allclose(bounds[:, 2], [-bunch.circum / 2, bunch.circum / 2], rtol=1.e-12, atol=0.0):
                raise ValueError("Coasting IBS Grid bounds (m) must span the full ring in z: [-circumference/2, circumference/2]")
            # Canonicalize accepted roundoff so the periodic lower face stays inside.
            bounds[:, 2] = [-bunch.circum / 2, bunch.circum / 2]
        bounds[:, 2] *= bunch.gamma
    proper_velocities = _rest_momenta(coordinates, bunch, xp)
    updated, diagnostics = scatter_binary(
        positions,
        proper_velocities,
        xp=xp,
        rng=rng,
        macro_weight=float(bunch.ratio),
        mass_kg=mass_kg,
        charge_coulomb=charge,
        dt=dt / bunch.gamma,
        coulomb_log=configuration.coulomb_log,
        grid_shape=configuration.grid_shape,
        max_scattering=configuration.max_scattering,
        max_substeps=configuration.max_substeps,
        collision_steps=configuration.collision_steps,
        bounds=bounds,
    )
    changed = xp.any(updated != proper_velocities, axis=1)
    if bool(xp.any(changed)):
        candidate = coordinates[changed].copy()
        _apply_rest_momenta(candidate, updated[changed], bunch, xp)
        coordinates[changed] = candidate
    return diagnostics


@Command.register("IBS")
class IBS(Command):
    """Apply one positive physical exposure without transporting the beam.

    bjorken_mtingwa reports local instantaneous rates only. kinetic evolves
    conditional Gaussian momenta, while binary rotates local particle pairs.
    State export covers this command's random streams, not the particles,
    reference state, or the executor's turn counter.
    """

    def __init__(self, beam_id, sim, **command_kwargs):
        kwargs = {str(k).lower(): v for k, v in command_kwargs.items()}
        self.cmd_name = str(kwargs.pop("name", "ibs"))
        kwargs.pop("command", None)
        self.cmd_type = "IBS"
        self.beam_id = int(beam_id)
        self.s = float(kwargs.get("s (m)", kwargs.get("s", 0.0)))
        self.length = 0.0
        self.last_diagnostics = None
        self.saved_diagnostics = []
        self._output_directory = None
        self._rngs = {}
        self._calls = 0
        self._sparse_warned = False
        self.parameters = None
        self.configuration = None
        self.optics = None
        settings = getattr(sim.cfg, "intrabeam_scattering", ())
        settings = settings[self.beam_id] if len(settings) > self.beam_id else load_intrabeam_scattering(sim.cfg.input_data[self.beam_id])
        self.is_enabled = bool(settings.enabled)
        if not self.is_enabled:
            return
        self.parameters = IBSItem.model_validate(kwargs)
        self.s = self.parameters.s
        self.is_enabled = self.parameters.is_enabled
        if not self.is_enabled:
            return
        if self.parameters.configuration not in settings.configurations:
            raise ValueError(f"IBS {self.cmd_name!r} references missing configuration {self.parameters.configuration!r}")
        self.configuration = settings.configurations[self.parameters.configuration]
        if self.configuration.method == "binary":
            if self.parameters.optics is not None:
                raise ValueError("IBS binary collisions do not use Optics")
        else:
            if self.parameters.optics is None:
                raise ValueError("IBS Gaussian methods require explicit local Optics")
            optics = self.parameters.optics
            self.optics = IBSOptics(
                beta_x=optics.beta_x,
                alpha_x=optics.alpha_x,
                beta_y=optics.beta_y,
                alpha_y=optics.alpha_y,
                dispersion_x=optics.dx,
                dispersion_px=optics.dpx,
                dispersion_y=optics.dy,
                dispersion_py=optics.dpy,
            )
        seed = self.configuration.random_seed
        self._entropy = int(np.random.SeedSequence().entropy) if seed is None else seed

    def print(self):
        logging.getLogger(__name__).info("S=%g, Command=IBS, Name=%s, Enabled=%s, Method=%s", self.s, self.cmd_name, self.is_enabled,
                                         self.configuration.method if self.configuration is not None else "disabled")

    def execute_cpu(self, sim):
        return self._execute(sim, "cpu")

    def execute_gpu(self, sim):
        return self._execute(sim, "gpu")

    def _candidate_rng(self, bunch_id):
        key = str(bunch_id)
        if key in self._rngs:
            rng = np.random.default_rng()
            rng.bit_generator.state = copy.deepcopy(self._rngs[key].bit_generator.state)
            return rng
        identity = f"{self.beam_id}:{self.cmd_name}:{bunch_id}".encode("utf-8")
        words = np.frombuffer(hashlib.sha256(identity).digest(), dtype="<u4").astype(np.uint32).tolist()
        return np.random.default_rng(np.random.SeedSequence([self._entropy, *words]))

    def _turn_selected(self, turn):
        if not self.parameters.save_turns:
            return True
        for item in self.parameters.save_turns:
            if len(item) == 1:
                if turn == item[0]:
                    return True
            elif item[0] <= turn <= item[1] and (turn - item[0]) % item[2] == 0:
                return True
        return False

    def _execute(self, sim, backend):
        if not self.is_enabled:
            return False
        beam = sim.beams[self.beam_id]
        if not self.configuration.bunched and len(beam.bunches) != 1:
            raise ValueError("Coasting IBS requires one bunch group representing the complete ring")
        p = beam.particles
        xp = p.xp
        if (backend == "cpu") != (xp is np):
            raise TypeError("IBS execution backend differs from the particle pool")
        staged = []
        rngs = {}
        records = []
        sparse_warning = None
        method = self.configuration.method
        save_diagnostics = self.parameters.save_diagnostics and self._turn_selected(int(sim.state.turn))
        for bunch in beam.bunches:
            _validate_reference(bunch)
            indices, coordinates = _gather_bunch(p, bunch)
            n_alive = len(indices)
            dt = self.parameters.interaction_length / (bunch.beta * const.c)
            record = dict(bunch_id=int(bunch.bunch_id),
                          n_alive=n_alive,
                          n_real=n_alive * float(bunch.ratio),
                          dt_lab_s=dt,
                          interaction_length_over_circumference=self.parameters.interaction_length / bunch.circum if bunch.circum > 0 else None)
            if n_alive == 0 or bunch.ratio == 0 or _species_parameters(bunch)[1] == 0:
                record["status"] = "zero_strength"
                records.append(record)
                continue
            if method == "bjorken_mtingwa":
                record["proper_velocity_max_over_c"] = _check_thermal_limit(coordinates, bunch, xp)
                parameters = _measure_parameters(coordinates, bunch, self.configuration, self.optics, xp)
                coefficients = compute_local_coefficients(parameters, self.optics)
                matching = _measure_matching(coordinates, parameters, self.optics, self.configuration.matching_tolerance, xp)
                record.update(parameters=asdict(parameters),
                              growth_rates=asdict(coefficients.growth_rates),
                              matching=matching,
                              model_moments_matched=matching["matched"],
                              status="diagnostic")
            elif dt == 0 or n_alive < 2:
                record["status"] = "zero_exposure" if dt == 0 else "no_pairs"
            else:
                rng = self._candidate_rng(bunch.bunch_id)
                if method == "kinetic":
                    result = _advance_kinetic(coordinates, indices, bunch, self.configuration, self.optics, dt, rng, xp)
                else:
                    result = _advance_binary(coordinates, bunch, self.configuration, dt, rng, xp)
                    if result["n_singleton_cells"] > 0 and not self._sparse_warned and sparse_warning is None:
                        sparse_warning = (result["n_singleton_cells"], bunch.bunch_id)
                stored = coordinates.astype(p.dtype)
                _validate_momenta(stored.astype(xp.float64), xp)
                staged.append((indices, stored[:, [1, 3, 5]]))
                rngs[str(bunch.bunch_id)] = rng
                record.update(result)
                record["status"] = "tracked"
            records.append(record)
        diagnostics = dict(
            format="PASS-ibs-diagnostics-1",
            command=self.cmd_name,
            beam_id=self.beam_id,
            turn=int(sim.state.turn),
            call=self._calls,
            s_m=self.s,
            interaction_length_m=self.parameters.interaction_length,
            method=method,
            coulomb_log=self.configuration.coulomb_log,
            coulomb_log_note=self.configuration.coulomb_log_note,
            rate_convention="instantaneous d(log(emittance_x, emittance_y, sigma_delta, variance_delta))/dt_lab",
            bunches=records,
        )
        if save_diagnostics:
            self._save_diagnostics(sim, diagnostics)
        for indices, momenta in staged:
            p.px[indices] = momenta[:, 0]
            p.py[indices] = momenta[:, 1]
            p.dp[indices] = momenta[:, 2]
        self._rngs.update(rngs)
        self._calls += 1
        self.last_diagnostics = diagnostics
        if sparse_warning is not None:
            logging.getLogger(__name__).warning(
                "IBS %s has %d singleton collision cells for bunch %d. "
                "Those particles cannot collide; increase the macroparticle count or use a coarser collision grid.", self.cmd_name, *sparse_warning)
            self._sparse_warned = True
        return bool(staged or method == "bjorken_mtingwa" or save_diagnostics)

    def _save_diagnostics(self, sim, diagnostics):
        payload = json.dumps(diagnostics, indent=2, allow_nan=False)
        if self._output_directory is None:
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.cmd_name).strip(".") or "ibs"
            self._output_directory = Path(sim.cfg.output_dir) / "ibs" / uuid.uuid4().hex / f"beam{self.beam_id}_{name}"
        self._output_directory.mkdir(parents=True, exist_ok=True)
        path = self._output_directory / f"turn_{int(sim.state.turn):08d}_call_{self._calls:08d}_{uuid.uuid4().hex[:8]}.json"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")
        self.saved_diagnostics.append(str(path))

    def _configuration_identity(self):
        values = dict(
            beam_id=self.beam_id,
            name=self.cmd_name,
            parameters=self.parameters.model_dump(mode="json", exclude={"save_diagnostics", "save_turns"}),
            configuration=self.configuration.model_dump(mode="json", exclude={"coulomb_log_note"}),
        )
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()

    def state_dict(self):
        """Export JSON-compatible RNG state; restore with matching beam state."""
        if not self.is_enabled:
            raise ValueError("Disabled IBS has no random state")
        return dict(
            format="PASS-ibs-state-1",
            configuration_sha256=self._configuration_identity(),
            entropy=self._entropy,
            calls=self._calls,
            generators={
                key: copy.deepcopy(rng.bit_generator.state)
                for key, rng in self._rngs.items()
            },
        )

    def load_state_dict(self, data):
        """Validate a candidate completely before replacing random streams."""
        if not self.is_enabled:
            raise ValueError("Disabled IBS has no random state")
        if data.get("format") != "PASS-ibs-state-1" or data.get("configuration_sha256") != self._configuration_identity():
            raise ValueError("IBS state does not match this command's physical configuration")
        for name in ("entropy", "calls"):
            if type(data.get(name)) is not int or data[name] < 0:
                raise ValueError(f"IBS state {name} must be a nonnegative integer")
        if not isinstance(data.get("generators"), dict):
            raise ValueError("IBS state generators must be a mapping")
        candidates = {}
        for key, state in data["generators"].items():
            if not isinstance(key, str) or not key.isdecimal():
                raise ValueError("IBS state generator keys must be nonnegative bunch IDs")
            rng = np.random.default_rng()
            rng.bit_generator.state = copy.deepcopy(state)
            candidates[key] = rng
        self._rngs = candidates
        self._entropy = data["entropy"]
        self._calls = data["calls"]
        self.last_diagnostics = None
