"""Incoherent electron cooling with a prescribed electron reservoir.

Collision kernels use electron-frame velocities [m/s], proper density [m^-3],
force [N] and momentum covariance per second [kg^2 m^2 s^-3]. Magnetic
finite-window coefficients require weak response; Parkhomchuk supplies drag
only. Derivations, references and validity limits are in element/electron_cooler.
"""

import copy
import hashlib
import json
import logging
import math
from numbers import Integral
from pathlib import Path
import re
import uuid

import numpy as np

from PASS.commands.command import Command
from PASS.para.schema.elements import ElectronCoolerItem
from PASS.utils.aperture import build_aperture
from PASS.utils.compute_kinematics import boost_proper_velocity
from PASS.utils.constants import const


def _cooler_scalar(value, name, *, positive=True):
    if not np.isscalar(value) or not np.isfinite(value) or (value <= 0 if positive else value < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"Electron cooler {name} must be finite and {qualifier}")
    return float(value)


def _cooler_order(value, name, minimum=4):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"Electron cooler {name} must be an integer >= {minimum}")
    return int(value)


def _cooler_arrays(velocity, density, xp):
    velocity = xp.asarray(velocity, dtype=xp.float64)
    if velocity.ndim != 2 or velocity.shape[1] != 3 or bool(xp.any(~xp.isfinite(velocity))):
        raise ValueError("Electron cooler velocity must be a finite (n_particles, 3) array")
    density = xp.broadcast_to(xp.asarray(density, dtype=xp.float64), (len(velocity), ))
    if bool(xp.any(~xp.isfinite(density))) or bool(xp.any(density < 0)):
        raise ValueError("Electron cooler proper electron density must be finite and nonnegative")
    return velocity, density


def gaussian_landau_coefficients(velocity, density, velocity_covariance, *, charge_number, ion_mass_kg, coulomb_log, quadrature_order=64, xp=np):
    """Return Gaussian Landau force and momentum diffusion in SI units.

    ``velocity_covariance`` is a positive-definite (3,3) or (N,3,3) matrix.
    The Coulomb logarithm is scalar or per ion, frozen outside the velocity
    integral. A velocity-dependent logarithm inside the microscopic integral
    is a different approximation and is not implemented by this function.
    Gaussian convolution of the two Rosenbluth potentials reduces to smooth
    one-dimensional integrals; no cylindrical symmetry is assumed.
    """
    velocity, density = _cooler_arrays(velocity, density, xp)
    ion_mass_kg = _cooler_scalar(ion_mass_kg, "ion_mass_kg")
    charge_number = _cooler_scalar(abs(charge_number), "absolute charge_number")
    quadrature_order = _cooler_order(quadrature_order, "quadrature_order", 8)
    covariance = xp.broadcast_to(xp.asarray(velocity_covariance, dtype=xp.float64), (len(velocity), 3, 3))
    if bool(xp.any(~xp.isfinite(covariance))):
        raise ValueError("Electron velocity covariance must be finite")
    scale_covariance = xp.max(xp.abs(covariance), axis=(1, 2))
    if bool(xp.any(xp.max(xp.abs(covariance - covariance.swapaxes(1, 2)), axis=(1, 2)) > 1.e-12 * scale_covariance)):
        raise ValueError("Electron velocity covariance must be symmetric")
    eigenvalues, rotation = xp.linalg.eigh(covariance)
    if bool(xp.any(eigenvalues <= 0)):
        raise ValueError("Electron velocity covariance must be positive definite")
    logarithm = xp.broadcast_to(xp.asarray(coulomb_log, dtype=xp.float64), (len(velocity), ))
    if bool(xp.any(~xp.isfinite(logarithm))) or bool(xp.any(logarithm < 0)):
        raise ValueError("Electron cooler Coulomb logarithm must be finite and nonnegative")
    rotated = xp.einsum("nji,nj->ni", rotation, velocity)
    scale = xp.mean(eigenvalues, axis=1) + xp.sum(rotated**2, axis=1)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    drag_integral = xp.zeros_like(velocity)
    diffusion_integral = xp.zeros((len(velocity), 3, 3), dtype=xp.float64)
    identity = xp.eye(3)
    log_range = xp.log(scale / eigenvalues[:, 0])
    n_intervals = max(1, math.ceil(float(xp.max(log_range)) / math.log(100))) if len(velocity) else 1
    thermal_scales = xp.sqrt(eigenvalues[:, :1] / scale[:, None]) * xp.exp(.5 * log_range[:, None] * xp.linspace(0, 1, n_intervals + 1)[None, :])
    boundaries = xp.column_stack((xp.zeros(len(velocity)), thermal_scales / (1 + thermal_scales), xp.ones(len(velocity))))
    for interval in range(n_intervals + 2):
        lower = boundaries[:, interval]
        span = boundaries[:, interval + 1] - lower
        for base_node, base_weight in zip((nodes + 1) / 2, weights / 2):
            node = lower + span * base_node
            s = scale * (node / (1 - node))**2
            jacobian = 2 * scale * node / (1 - node)**3
            inverse = 1 / (eigenvalues + s[:, None])
            argument = -.5 * xp.sum(rotated**2 * inverse, axis=1)
            factor = base_weight * span * jacobian * xp.exp(argument) * xp.sqrt(xp.prod(inverse, axis=1)) / math.sqrt(2 * math.pi)
            vector = rotated * inverse
            drag_integral += factor[:, None] * vector
            matrix = identity[None, :, :] * inverse[:, :, None] - vector[:, :, None] * vector[:, None, :]
            diffusion_integral += (factor * s)[:, None, None] * matrix
    interaction = charge_number * const.e**2 / (4 * math.pi * const.epsilon0)
    strength = 4 * math.pi * density * interaction**2 * logarithm
    force = -strength[:, None] * (1 / const.m_e_kg + 1 / ion_mass_kg) * xp.einsum("nij,nj->ni", rotation, drag_integral)
    diffusion = strength[:, None, None] * xp.einsum("nik,nkl,njl->nij", rotation, diffusion_integral, rotation)
    diffusion = (diffusion + diffusion.swapaxes(1, 2)) / 2
    if bool(xp.any(~xp.isfinite(force))) or bool(xp.any(~xp.isfinite(diffusion))):
        raise FloatingPointError("Gaussian Landau coefficients exceed floating-point range")
    diffusion_values = xp.linalg.eigvalsh(diffusion)
    if bool(xp.any(diffusion_values[:, 0] < -1.e-10 * xp.max(xp.abs(diffusion_values), axis=1))):
        raise ArithmeticError("Gaussian diffusion quadrature lost positive semidefiniteness; increase quadrature_order")
    return force, diffusion


def parkhomchuk_force(velocity,
                      density,
                      *,
                      charge_number,
                      magnetic_field_t,
                      longitudinal_rms_speed,
                      transverse_rms_speed,
                      effective_rms_speed=0.0,
                      interaction_time_s,
                      beam_radius_m,
                      coulomb_log=None,
                      maximum_impact_parameter_m=None,
                      xp=np):
    """Return empirical magnetized drag, without assigning a diffusion model.

    Transverse speed is the RMS of ONE Cartesian component, hence the mean
    squared gyroradius is 2*(m_e*sigma_perp/(e*B))**2. The extra effective
    speed represents field errors or unresolved drift, not transverse heat.
    The regularized cutoffs use u=sqrt(v**2+sigma_parallel**2+v_extra**2),
    b90=Z*e**2/(4*pi*epsilon_0*m_e*u**2), and
    bmax=min(beam_radius, u/omega_p, u*physical_interaction_time).
    """
    velocity, density = _cooler_arrays(velocity, density, xp)
    charge_number = _cooler_scalar(abs(charge_number), "absolute charge_number")
    magnetic_field_t = _cooler_scalar(abs(magnetic_field_t), "absolute magnetic_field_t")
    longitudinal_rms_speed = _cooler_scalar(longitudinal_rms_speed, "longitudinal_rms_speed")
    transverse_rms_speed = _cooler_scalar(transverse_rms_speed, "transverse_rms_speed", positive=False)
    effective_rms_speed = _cooler_scalar(effective_rms_speed, "effective_rms_speed", positive=False)
    interaction_time_s = _cooler_scalar(interaction_time_s, "interaction_time_s")
    beam_radius_m = _cooler_scalar(beam_radius_m, "beam_radius_m")
    speed_squared = xp.sum(velocity**2, axis=1) + longitudinal_rms_speed**2 + effective_rms_speed**2
    speed = xp.sqrt(speed_squared)
    interaction = charge_number * const.e**2 / (4 * math.pi * const.epsilon0)
    if coulomb_log is None:
        plasma_frequency = xp.sqrt(density * const.e**2 / (const.epsilon0 * const.m_e_kg))
        plasma_length = speed / xp.where(plasma_frequency > 0, plasma_frequency, 1)
        plasma_length = xp.where(plasma_frequency > 0, plasma_length, xp.inf)
        maximum = xp.minimum(xp.minimum(beam_radius_m, speed * interaction_time_s), plasma_length)
        if maximum_impact_parameter_m is not None:
            maximum = xp.minimum(maximum, _cooler_scalar(maximum_impact_parameter_m, "maximum_impact_parameter_m"))
        minimum = interaction / (const.m_e_kg * speed_squared)
        larmor = math.sqrt(2) * const.m_e_kg * transverse_rms_speed / (const.e * magnetic_field_t)
        logarithm = xp.log1p(maximum / (minimum + larmor))
    else:
        logarithm = xp.broadcast_to(xp.asarray(coulomb_log, dtype=xp.float64), (len(velocity), ))
        if bool(xp.any(~xp.isfinite(logarithm))) or bool(xp.any(logarithm < 0)):
            raise ValueError("Parkhomchuk Coulomb logarithm must be finite and nonnegative")
    return -4 * density[:, None] * interaction**2 / const.m_e_kg * logarithm[:, None] * velocity / speed_squared[:, None]**1.5


def _response_quadrature(velocity, density, *, interaction, ion_mass_kg, sigma_parallel, sigma_perp, omega, duration, b_min, b_max, orders, xp):
    radial_order, polar_order, azimuthal_order, time_order = orders
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(radial_order)
    polar_nodes, polar_weights = np.polynomial.legendre.leggauss(polar_order)
    time_nodes, time_weights = np.polynomial.legendre.leggauss(time_order)
    log_lower, log_upper = math.log(1 / b_max), math.log(1 / b_min)
    wave_number = np.exp(log_lower + (radial_nodes + 1) * (log_upper - log_lower) / 2)
    radial_weights = radial_weights * (log_upper - log_lower) / 2
    phi = (np.arange(azimuthal_order) + .5) * (2 * math.pi / azimuthal_order)
    radial, polar, azimuth = np.meshgrid(wave_number, polar_nodes, phi, indexing="ij")
    directions = np.stack((np.sqrt(1 - polar**2) * np.cos(azimuth), np.sqrt(1 - polar**2) * np.sin(azimuth), polar), axis=-1).reshape(-1, 3)
    radial = radial.ravel()
    weights = np.broadcast_to(radial_weights[:, None, None] * polar_weights[None, :, None], (radial_order, polar_order, azimuthal_order)).ravel()
    weights = weights * (2 * math.pi / azimuthal_order)
    force = xp.zeros_like(velocity)
    diffusion = xp.zeros((len(velocity), 3, 3), dtype=xp.float64)
    time = xp.asarray((time_nodes + 1) * duration / 2)
    time_weight = xp.asarray(time_weights * duration / 2) * (1 - time / duration)
    omega_time = omega * time
    transverse_displacement = time**2 * xp.sinc(omega_time / (2 * math.pi))**2
    transverse_response = time * xp.sinc(omega_time / math.pi)
    for start in range(0, len(radial), 64):
        end = min(start + 64, len(radial))
        direction = xp.asarray(directions[start:end])
        wave = xp.asarray(radial[start:end])
        weight = xp.asarray(weights[start:end])
        longitudinal_squared = direction[:, 2]**2
        thermal = xp.exp(-.5 * wave[:, None]**2 * (sigma_parallel**2 * longitudinal_squared[:, None] * time[None, :]**2 + sigma_perp**2 *
                                                   (1 - longitudinal_squared[:, None]) * transverse_displacement[None, :]))
        response = (longitudinal_squared[:, None] * time[None, :] +
                    (1 - longitudinal_squared[:, None]) * transverse_response[None, :]) / const.m_e_kg + time[None, :] / ion_mass_kg
        for particle_start in range(0, len(velocity), 32):
            particle_end = min(particle_start + 32, len(velocity))
            projection = velocity[particle_start:particle_end] @ direction.T
            phase = projection[:, :, None] * wave[None, :, None] * time[None, None, :]
            common = thermal * time_weight[None, :]
            cosine_integral = xp.sum(xp.cos(phase) * common[None, :, :], axis=2)
            sine_integral = xp.sum(xp.sin(phase) * (common * response)[None, :, :], axis=2)
            force[particle_start:particle_end] -= xp.einsum("nk,k,ki->ni", sine_integral, weight * wave**2, direction)
            diffusion[particle_start:particle_end] += 2 * xp.einsum("nk,k,ki,kj->nij", cosine_integral, weight * wave, direction, direction)
    strength = density * (2 * interaction**2 / math.pi)
    return strength[:, None] * force, strength[:, None, None] * diffusion


def magnetized_collision_coefficients(velocity,
                                      density,
                                      *,
                                      charge_number,
                                      ion_mass_kg,
                                      longitudinal_rms_speed,
                                      transverse_rms_speed,
                                      magnetic_field_t,
                                      interaction_time_s,
                                      minimum_impact_parameter_m,
                                      maximum_impact_parameter_m,
                                      radial_order=16,
                                      polar_order=12,
                                      azimuthal_order=16,
                                      time_order=64,
                                      quadrature_rtol=.02,
                                      max_refinements=2,
                                      xp=np):
    """Return finite-window magnetic response force, diffusion and diagnostics.

    Only the controlled weak-response regime is accepted; no extrapolation
    into collective screening or strong binary trapping is performed. The
    z axis is parallel to the field. The transverse RMS is per component.
    Cutoffs mean k_min=1/b_max and k_max=1/b_min in the Fourier convention.
    Increasing every quadrature order checks absolute tensor errors against
    the characteristic tensor/force scale, including zero-force symmetry.
    """
    velocity, density = _cooler_arrays(velocity, density, xp)
    charge_number = _cooler_scalar(abs(charge_number), "absolute charge_number")
    ion_mass_kg = _cooler_scalar(ion_mass_kg, "ion_mass_kg")
    sigma_parallel = _cooler_scalar(longitudinal_rms_speed, "longitudinal_rms_speed")
    sigma_perp = _cooler_scalar(transverse_rms_speed, "transverse_rms_speed")
    omega = const.e * _cooler_scalar(abs(magnetic_field_t), "absolute magnetic_field_t", positive=False) / const.m_e_kg
    duration = _cooler_scalar(interaction_time_s, "interaction_time_s")
    b_min = _cooler_scalar(minimum_impact_parameter_m, "minimum_impact_parameter_m")
    b_max = _cooler_scalar(maximum_impact_parameter_m, "maximum_impact_parameter_m")
    if b_max <= b_min:
        raise ValueError("Magnetized response requires maximum impact parameter > minimum impact parameter")
    if not np.isfinite(quadrature_rtol) or not 1.e-6 <= quadrature_rtol <= .1:
        raise ValueError("Magnetized quadrature_rtol must be between 1e-6 and 0.1")
    max_refinements = _cooler_order(max_refinements, "max_refinements", 1)
    orders = tuple(
        _cooler_order(value, name) for value, name in zip((radial_order, polar_order, azimuthal_order,
                                                           time_order), ("radial_order", "polar_order", "azimuthal_order", "time_order")))
    plasma_phase = float(xp.max(xp.sqrt(density * const.e**2 / (const.epsilon0 * const.m_e_kg)))) * duration if len(velocity) else 0.0
    if plasma_phase > .3:
        raise ValueError(f"Magnetized weak-response model requires omega_p * interaction_time <= 0.3; measured {plasma_phase:.6g}")
    ion_gyro_phase = charge_number * const.m_e_kg * omega * duration / ion_mass_kg
    if ion_gyro_phase > .1:
        raise ValueError(f"Magnetized response assumes negligible ion gyration: omega_ion * interaction_time <= 0.1; measured {ion_gyro_phase:.6g}")
    thermal_speed = max(sigma_parallel, sigma_perp)
    speed = xp.linalg.norm(velocity, axis=1)
    if thermal_speed > .05 * const.c or (len(velocity) and float(xp.max(speed)) > .05 * const.c):
        raise ValueError("Magnetized collision response requires nonrelativistic ion and electron thermal velocities (<0.05c)")
    interaction = charge_number * const.e**2 / (4 * math.pi * const.epsilon0)
    decorrelation_speed = xp.sqrt(speed**2 + sigma_parallel**2 + (2 * sigma_perp**2 if omega == 0 else 0))
    response_time = xp.minimum(duration, b_min / decorrelation_speed)
    deflection = interaction * response_time**2 / (const.m_e_kg * b_min**3)
    maximum_deflection = float(xp.max(deflection)) if len(velocity) else 0.0
    if maximum_deflection > .1:
        raise ValueError(f"Magnetized linear response requires weak UV-cutoff deflections <= 0.1; measured {maximum_deflection:.6g}")
    arguments = dict(interaction=interaction,
                     ion_mass_kg=ion_mass_kg,
                     sigma_parallel=sigma_parallel,
                     sigma_perp=sigma_perp,
                     omega=omega,
                     duration=duration,
                     b_min=b_min,
                     b_max=b_max,
                     xp=xp)
    previous_force, previous_diffusion = _response_quadrature(velocity, density, orders=orders, **arguments)
    error = math.inf
    for refinement in range(1, max_refinements + 1):
        orders = tuple(2 * value for value in orders)
        force, diffusion = _response_quadrature(velocity, density, orders=orders, **arguments)
        diffusion_scale = xp.linalg.norm(diffusion, axis=(1, 2))
        characteristic_force = diffusion_scale / (const.m_e_kg * max(sigma_parallel, sigma_perp))
        force_scale = xp.maximum(xp.linalg.norm(force, axis=1), characteristic_force * 1.e-8)
        force_error = xp.linalg.norm(force - previous_force, axis=1) / xp.where(force_scale > 0, force_scale, 1)
        diffusion_error = xp.linalg.norm(diffusion - previous_diffusion, axis=(1, 2)) / xp.where(diffusion_scale > 0, diffusion_scale, 1)
        error = max(float(xp.max(force_error)), float(xp.max(diffusion_error))) if len(velocity) else 0.0
        minimum_eigenvalue = xp.linalg.eigvalsh(diffusion)[:, 0]
        positive = bool(xp.all(minimum_eigenvalue >= -1.e-10 * diffusion_scale))
        phase_resolved = omega * duration <= 2 * orders[3]
        if error <= quadrature_rtol and positive and phase_resolved:
            return force, diffusion, dict(omega_p_time=plasma_phase,
                                          omega_c_time=omega * duration,
                                          omega_ion_time=ion_gyro_phase,
                                          weak_deflection=maximum_deflection,
                                          quadrature_relative_error=error,
                                          quadrature_orders=list(orders),
                                          refinements=refinement)
        previous_force, previous_diffusion = force, diffusion
    raise ArithmeticError(f"Magnetized collision quadrature did not converge: relative error={error:.6g}; orders={orders}; "
                          "increase quadrature orders/refinements or shorten the physical interaction window")


def _transport_coordinates(coordinates, bunch, length, ks, xp):
    """Uniform axial-field transport in PASS canonical coordinates."""
    if length == 0 or len(coordinates) == 0:
        return
    x, px, y, py, z, delta = coordinates.T
    mechanical_x = px + 0.5 * ks * y
    mechanical_y = py - 0.5 * ks * x
    transverse_sq = mechanical_x**2 + mechanical_y**2
    longitudinal_sq = (1 + delta)**2 - transverse_sq
    if not bool(xp.all(xp.isfinite(coordinates))) or bool(xp.any((delta <= -1) | (longitudinal_sq <= 0))):
        raise ValueError("ElectronCooler requires finite forward mechanical momenta")
    longitudinal = xp.sqrt(longitudinal_sq)
    angle = ks * length / longitudinal
    sine, cosine = xp.sin(angle), xp.cos(angle)
    first = length / longitudinal * xp.sinc(angle / np.pi)
    second = length / longitudinal * (0.5 * angle * xp.sinc(angle / (2 * np.pi))**2)
    new_x = x + first * mechanical_x + second * mechanical_y
    new_y = y + first * mechanical_y - second * mechanical_x
    energy_ratio = xp.sqrt(1 / bunch.gamma**2 + bunch.beta**2 * (1 + delta)**2)
    z += length * (delta * (2 + delta) / bunch.gamma**2 - transverse_sq) / (longitudinal * (longitudinal + energy_ratio))
    px[:] = cosine * mechanical_x + sine * mechanical_y - 0.5 * ks * new_y
    py[:] = cosine * mechanical_y - sine * mechanical_x + 0.5 * ks * new_x
    x[:], y[:] = new_x, new_y


def _lab_proper_velocity(coordinates, bunch, ks, xp):
    reference_beta_gamma = bunch.beta * bunch.gamma
    mechanical_x = coordinates[:, 1] + 0.5 * ks * coordinates[:, 2]
    mechanical_y = coordinates[:, 3] - 0.5 * ks * coordinates[:, 0]
    longitudinal_sq = (1 + coordinates[:, 5])**2 - mechanical_x**2 - mechanical_y**2
    if (not bool(xp.all(xp.isfinite(coordinates))) or bool(xp.any(coordinates[:, 5] <= -1)) or bool(xp.any(longitudinal_sq <= 0))):
        raise ValueError("ElectronCooler requires finite forward mechanical momenta")
    return reference_beta_gamma * const.c * xp.column_stack((mechanical_x, mechanical_y, xp.sqrt(longitudinal_sq)))


def _store_lab_proper_velocity(coordinates, proper_velocity, bunch, ks, xp):
    if not bool(xp.all(xp.isfinite(proper_velocity))) or bool(xp.any(proper_velocity[:, 2] <= 0)):
        raise ValueError("ElectronCooler produced invalid or backward lab momenta; reduce the step")
    normalized = proper_velocity / (bunch.beta * bunch.gamma * const.c)
    coordinates[:, 1] = normalized[:, 0] - 0.5 * ks * coordinates[:, 2]
    coordinates[:, 3] = normalized[:, 1] + 0.5 * ks * coordinates[:, 0]
    momentum_ratio = xp.sqrt(xp.sum(normalized**2, axis=1))
    coordinates[:, 5] = (xp.sum(normalized[:, :2]**2, axis=1) + (normalized[:, 2] - 1) * (normalized[:, 2] + 1)) / (momentum_ratio + 1)


def _transverse_electron_field(offset, line_charge, profile, sizes, order, xp):
    """Lab transverse electric field [V/m] of a locally uniform long beam."""
    square = xp.sum(offset**2, axis=1)
    if profile == "uniform_round":
        denominator = xp.maximum(square, sizes[0]**2)
        return line_charge[:, None] * offset / (2 * np.pi * const.epsilon0 * denominator[:, None])
    sigma_x, sigma_y = sizes
    if abs(sigma_x - sigma_y) <= 1e-12 * max(sigma_x, sigma_y):
        denominator = xp.where(square > 0, square, 1)
        factor = -xp.expm1(-square / (2 * sigma_x**2)) / denominator
        factor = xp.where(square > 0, factor, 1 / (2 * sigma_x**2))
        return line_charge[:, None] * offset * factor[:, None] / (2 * np.pi * const.epsilon0)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    scale = sigma_x * sigma_y + square
    result = xp.zeros_like(offset)
    for node, weight in zip(nodes, weights):
        parameter = scale * node / (1 - node)
        variance = xp.column_stack((sigma_x**2 + parameter, sigma_y**2 + parameter))
        factor = xp.exp(-0.5 * xp.sum(offset**2 / variance, axis=1))
        factor *= weight * scale / ((1 - node)**2 * xp.sqrt((sigma_x**2 + parameter) * (sigma_y**2 + parameter)))
        result += offset / variance * factor[:, None]
    return line_charge[:, None] * result / (4 * np.pi * const.epsilon0)


@Command.register("electroncooler")
class ElectronCooler(Command):
    """Track ions through a prescribed electron reservoir and axial solenoid."""

    def __init__(self, beam_id, sim, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}
        self.cmd_name = kwargs.pop("name")
        kwargs.pop("order", None)
        kwargs.setdefault("command", "ElectronCooler")
        self.parameters = ElectronCoolerItem.model_validate(kwargs)
        self.beam_id = beam_id
        self.cmd_type = self.__class__.__name__
        self.s = self.parameters.s
        self.length = self.parameters.length
        self.is_thick = self.length > 0
        self.is_enabled = True
        self._rngs = {}
        self._calls = 0
        self.last_diagnostics = None
        self.saved_diagnostics = []
        self._output_path = None
        seed = self.parameters.random_seed
        self._entropy = int(np.random.SeedSequence().entropy) if seed is None else seed
        electron = self.parameters.electron_beam
        self.electron_gamma = 1 + electron.kinetic_energy / const.m_e_eV
        self.electron_beta = np.sqrt((self.electron_gamma - 1) * (self.electron_gamma + 1)) / self.electron_gamma
        direction = np.array([np.tan(electron.angle_x), np.tan(electron.angle_y), 1.])
        direction /= np.linalg.norm(direction)
        first = np.cross([0., 1., 0.], direction)
        first /= np.linalg.norm(first)
        self.rotation = np.column_stack((first, np.cross(direction, first), direction))
        if electron.velocity_covariance is None:
            self.velocity_covariance = np.diag([electron.temperature_transverse, electron.temperature_transverse, electron.temperature_longitudinal
                                                ]) * const.e / const.m_e_kg
        else:
            self.velocity_covariance = np.asarray(electron.velocity_covariance, dtype=float)
        if np.sqrt(np.linalg.eigvalsh(self.velocity_covariance).max()) > 0.05 * const.c:
            raise ValueError("ElectronCooler requires nonrelativistic electron thermal speeds (rms <= 0.05 c)")
        self.velocity_gradient = np.zeros((3, 2)) if electron.velocity_gradient is None else np.asarray(electron.velocity_gradient)
        self.aperture = build_aperture({"Type": self.parameters.aperture_type, "Value": self.parameters.aperture_value})
        if self.parameters.model != "gaussian" and (electron.angle_x != 0 or electron.angle_y != 0):
            raise ValueError("Magnetized cooling requires the electron mean direction parallel to the axial solenoid")
        if self.parameters.mean_space_charge and electron.mode == "gaussian_bunch":
            maximum_size = max(value for value in (electron.radius, electron.radius_exit, electron.sigma_x, electron.sigma_x_exit, electron.sigma_y,
                                                   electron.sigma_y_exit) if value is not None)
            rest_length = self.electron_gamma * self.electron_beta * const.c * electron.sigma_time
            if rest_length < 10 * maximum_size:
                raise ValueError("The long-beam mean field requires the electron rest-frame RMS bunch length >= 10 transverse beam sizes")

    def print(self):
        logging.getLogger(__name__).info("S=%g, Command=ElectronCooler, Name=%s, Length=%g, Model=%s, Diffusion=%s", self.s, self.cmd_name,
                                         self.length, self.parameters.model, self.parameters.diffusion)

    def execute_cpu(self, sim):
        return self._execute(sim, "cpu")

    def execute_gpu(self, sim):
        return self._execute(sim, "gpu")

    def _candidate_rng(self, bunch_id):
        key = str(bunch_id)
        rng = np.random.default_rng()
        if key in self._rngs:
            rng.bit_generator.state = copy.deepcopy(self._rngs[key].bit_generator.state)
        else:
            identity = f"{self.beam_id}:{self.cmd_name}:{bunch_id}".encode("utf-8")
            words = np.frombuffer(hashlib.sha256(identity).digest(), dtype="<u4").tolist()
            rng = np.random.default_rng(np.random.SeedSequence([self._entropy, *words]))
        return rng

    def _sample_electrons(self, coordinates, bunch, position, xp):
        electron = self.parameters.electron_beam
        direction = self.rotation[:, 2]
        rotation = xp.asarray(self.rotation)
        displacement = xp.column_stack((coordinates[:, 0] - electron.center_x - position * direction[0] / direction[2],
                                        coordinates[:, 2] - electron.center_y - position * direction[1] / direction[2], xp.zeros(len(coordinates))))
        offset = (displacement @ rotation)[:, :2]
        fraction = position / self.length if self.length else 0.
        if electron.profile == "uniform_round":
            radius = electron.radius + fraction * ((electron.radius_exit or electron.radius) - electron.radius)
            sizes = (radius, radius)
            transverse = xp.where(xp.sum(offset**2, axis=1) < radius**2, 1 / (np.pi * radius**2), 0.)
        else:
            sigma_x = electron.sigma_x + fraction * ((electron.sigma_x_exit or electron.sigma_x) - electron.sigma_x)
            sigma_y = electron.sigma_y + fraction * ((electron.sigma_y_exit or electron.sigma_y) - electron.sigma_y)
            sizes = (sigma_x, sigma_y)
            transverse = xp.exp(-0.5 * (offset[:, 0]**2 / sigma_x**2 + offset[:, 1]**2 / sigma_y**2)) / (2 * np.pi * sigma_x * sigma_y)
        if electron.mode == "dc":
            line_number = xp.full(len(coordinates), electron.current / (const.e * self.electron_beta * const.c))
        else:
            local_time = -coordinates[:, 4] / (bunch.beta * const.c)
            center_time = electron.bunch_center_time + position / (self.electron_beta * const.c * direction[2])
            local_time += bunch.t0 + position / (bunch.beta * const.c) - center_time
            local_time -= displacement @ xp.asarray(direction) / (self.electron_beta * const.c)
            frequency = electron.repetition_frequency
            if frequency is None:
                temporal = xp.exp(-0.5 * (local_time / electron.sigma_time)**2) / (np.sqrt(2 * np.pi) * electron.sigma_time)
            elif electron.sigma_time * frequency < 0.15:
                phase = xp.remainder(local_time * frequency + 0.5, 1) - 0.5
                temporal = xp.zeros_like(phase)
                for neighbor in (-1, 0, 1):
                    temporal += xp.exp(-0.5 * ((phase + neighbor) / (electron.sigma_time * frequency))**2)
                temporal /= np.sqrt(2 * np.pi) * electron.sigma_time
            else:
                phase = xp.remainder(local_time * frequency, 1)
                temporal = xp.ones_like(phase)
                n_harmonics = int(np.ceil(np.sqrt(-2 * np.log(1e-15)) / (2 * np.pi * electron.sigma_time * frequency)))
                for harmonic in range(1, n_harmonics + 1):
                    temporal += 2 * np.exp(-2 * (np.pi * harmonic * electron.sigma_time * frequency)**2) * xp.cos(2 * np.pi * harmonic * phase)
                temporal *= frequency
            line_number = electron.bunch_charge * temporal / (const.e * self.electron_beta * const.c)
        density = line_number * transverse / self.electron_gamma
        local_mean = offset @ xp.asarray(self.velocity_gradient).T
        if bool(xp.any(xp.linalg.norm(local_mean, axis=1) > 0.05 * const.c)):
            raise ValueError("Electron velocity shear exceeds the nonrelativistic local-reservoir domain")
        return density, local_mean, offset, sizes, line_number

    def _collision_coefficients(self, velocity, density, mass_kg, charge_number, physical_time, radius, xp):
        parameters = self.parameters
        covariance = xp.asarray(self.velocity_covariance)
        if parameters.model == "parkhomchuk":
            force = parkhomchuk_force(velocity,
                                      density,
                                      charge_number=charge_number,
                                      magnetic_field_t=parameters.magnetic_field,
                                      longitudinal_rms_speed=np.sqrt(self.velocity_covariance[2, 2]),
                                      transverse_rms_speed=np.sqrt((self.velocity_covariance[0, 0] + self.velocity_covariance[1, 1]) / 2),
                                      effective_rms_speed=parameters.effective_velocity_spread,
                                      interaction_time_s=physical_time,
                                      beam_radius_m=radius,
                                      coulomb_log=parameters.coulomb_log,
                                      maximum_impact_parameter_m=parameters.max_impact_parameter,
                                      xp=xp)
            return force, xp.zeros((len(velocity), 3, 3)), {}
        if parameters.model == "magnetized_collision":
            return magnetized_collision_coefficients(velocity,
                                                     density,
                                                     charge_number=charge_number,
                                                     ion_mass_kg=mass_kg,
                                                     longitudinal_rms_speed=np.sqrt(self.velocity_covariance[2, 2]),
                                                     transverse_rms_speed=np.sqrt(self.velocity_covariance[0, 0]),
                                                     magnetic_field_t=parameters.magnetic_field,
                                                     interaction_time_s=physical_time,
                                                     minimum_impact_parameter_m=parameters.min_impact_parameter,
                                                     maximum_impact_parameter_m=parameters.max_impact_parameter,
                                                     radial_order=parameters.radial_order,
                                                     polar_order=parameters.polar_order,
                                                     azimuthal_order=parameters.azimuthal_order,
                                                     time_order=parameters.time_order,
                                                     quadrature_rtol=parameters.quadrature_rtol,
                                                     max_refinements=parameters.max_refinements,
                                                     xp=xp)
        if parameters.coulomb_log is None:
            speed = xp.sqrt(xp.sum(velocity**2, axis=1) + np.trace(self.velocity_covariance))
            reduced_mass = const.m_e_kg * mass_kg / (const.m_e_kg + mass_kg)
            classical = abs(charge_number) * const.e**2 / (4 * np.pi * const.epsilon0 * reduced_mass * speed**2)
            quantum = const.h_bar / (2 * reduced_mass * speed)
            minimum = xp.maximum(classical, quantum)
            plasma_frequency = xp.sqrt(density * const.e**2 / (const.epsilon0 * const.m_e_kg))
            maximum = xp.minimum(speed * physical_time, radius)
            maximum = xp.minimum(maximum, speed / xp.maximum(plasma_frequency, np.finfo(float).tiny))
            if parameters.max_impact_parameter is not None:
                maximum = xp.minimum(maximum, parameters.max_impact_parameter)
            coulomb_log = xp.log1p(maximum / minimum)
        else:
            coulomb_log = parameters.coulomb_log
        force, diffusion = gaussian_landau_coefficients(velocity,
                                                        density,
                                                        covariance,
                                                        charge_number=charge_number,
                                                        ion_mass_kg=mass_kg,
                                                        coulomb_log=coulomb_log,
                                                        quadrature_order=parameters.quadrature_order,
                                                        xp=xp)
        return force, diffusion, {
            "coulomb_log_min": float(xp.min(xp.asarray(coulomb_log))),
            "coulomb_log_max": float(xp.max(xp.asarray(coulomb_log)))
        }

    def _apply_collision(self, coordinates, bunch, position, length, ks, mass_kg, charge_number, rng, xp):
        density, local_mean, offset, sizes, line_number = self._sample_electrons(coordinates, bunch, position, xp)
        parameters = self.parameters
        if not bool(xp.any(line_number > 0)) or (not parameters.collisions and not parameters.mean_space_charge):
            return dict(n_substeps=0, noise_used=False, density_mean=float(xp.mean(density)), overlap_fraction=0., energy_change_j=0.)
        if not parameters.mean_space_charge and not bool(xp.any(density > 0)):
            return dict(n_substeps=0, noise_used=False, density_mean=0., overlap_fraction=0., energy_change_j=0.)
        rotation = xp.asarray(self.rotation)
        initial_velocity = _lab_proper_velocity(coordinates, bunch, ks, xp)
        lab_velocity = initial_velocity.copy()
        initial_gamma = xp.sqrt(1 + xp.sum((lab_velocity / const.c)**2, axis=1))
        reference_lab = xp.asarray([[0., 0., bunch.beta * bunch.gamma * const.c]]) @ rotation
        reference_rest = boost_proper_velocity(reference_lab, self.electron_beta, self.electron_gamma, xp=xp)
        reference_gamma_rest = float(xp.sqrt(1 + xp.sum((reference_rest / const.c)**2)))
        physical_time = self.length * reference_gamma_rest / (bunch.beta * bunch.gamma * const.c)
        mean_force = xp.zeros((len(coordinates), 3))
        if parameters.mean_space_charge:
            field = _transverse_electron_field(offset, -const.e * line_number, parameters.electron_beam.profile, sizes, parameters.quadrature_order,
                                               xp)
            mean_force[:, :2] = charge_number * const.e * field / self.electron_gamma
        elapsed = 0.
        n_substeps = 0
        noise_used = False
        detail = {}
        force_sum = xp.zeros(3)
        diffusion_sum = xp.zeros(3)
        thermal_ion_variance = const.m_e_kg / mass_kg * np.trace(self.velocity_covariance) / 3
        while elapsed < length:
            rest_velocity = boost_proper_velocity(lab_velocity @ rotation, self.electron_beta, self.electron_gamma, xp=xp)
            rest_gamma = xp.sqrt(1 + xp.sum((rest_velocity / const.c)**2, axis=1))
            relative_velocity = rest_velocity / rest_gamma[:, None] - local_mean
            if parameters.collisions and bool(xp.any(density > 0)):
                if bool(xp.any(xp.linalg.norm(relative_velocity[density > 0], axis=1) > 0.05 * const.c)):
                    raise ValueError("Electron-ion collision velocities exceed 0.05 c; the thermal collision model is nonrelativistic")
                force, diffusion, detail = self._collision_coefficients(relative_velocity, density, mass_kg, charge_number, physical_time, min(sizes),
                                                                        xp)
            else:
                force = xp.zeros_like(rest_velocity)
                diffusion = xp.zeros((len(coordinates), 3, 3))
            if not parameters.diffusion:
                diffusion = xp.zeros_like(diffusion)
            if not bool(xp.all(xp.isfinite(force))) or not bool(xp.all(xp.isfinite(diffusion))):
                raise ValueError("ElectronCooler collision coefficients are nonfinite")
            eigenvalues, eigenvectors = xp.linalg.eigh(0.5 * (diffusion + xp.swapaxes(diffusion, -1, -2)))
            spectral_scale = xp.max(xp.abs(eigenvalues), axis=1)
            if bool(xp.any(eigenvalues[:, 0] < -1e-10 * xp.maximum(spectral_scale, np.finfo(float).tiny))):
                raise ValueError("ElectronCooler diffusion tensor is not positive semidefinite")
            eigenvalues = xp.maximum(eigenvalues, 0)
            time_per_length = rest_gamma / lab_velocity[:, 2]
            relative_speed = xp.linalg.norm(relative_velocity, axis=1)
            speed_scale = xp.sqrt(relative_speed**2 + thermal_ion_variance)
            # Resolve linear drag even arbitrarily close to velocity matching.
            drag_scale = xp.maximum(relative_speed, 1e-14 * np.sqrt(thermal_ion_variance))
            collision_rate = xp.linalg.norm(force, axis=1) / (mass_kg * drag_scale)
            collision_rate += xp.linalg.norm(mean_force, axis=1) / (mass_kg * speed_scale)
            collision_rate += xp.sum(eigenvalues, axis=1) / (mass_kg**2 * speed_scale**2)
            maximum_rate = float(xp.max(collision_rate * time_per_length))
            step = min(length - elapsed, parameters.max_fractional_step / maximum_rate) if maximum_rate > 0 else length - elapsed
            if n_substeps >= parameters.max_substeps or not np.isfinite(step) or step <= 0 or elapsed + step == elapsed:
                raise ValueError("ElectronCooler exceeded Max substeps; increase Num slices or reduce interaction strength")
            dt = step * time_per_length
            delta_velocity = (force + mean_force) * (dt / mass_kg)[:, None]
            if bool(xp.any(eigenvalues > 0)):
                noise = xp.asarray(rng.standard_normal((len(coordinates), 3)))
                projected_noise = xp.einsum("nji,nj->ni", eigenvectors, noise)
                delta_velocity += xp.einsum("nij,nj->ni", eigenvectors, projected_noise * xp.sqrt(eigenvalues * dt[:, None])) / mass_kg
                noise_used = True
            changed = xp.any(delta_velocity != 0, axis=1)
            rest_velocity += delta_velocity
            if bool(xp.any(changed)):
                lab_velocity[changed] = boost_proper_velocity(rest_velocity[changed], self.electron_beta, self.electron_gamma, xp=xp,
                                                              inverse=True) @ rotation.T
            if bool(xp.any(lab_velocity[:, 2] <= 0)):
                raise ValueError("ElectronCooler produced backward motion; reduce interaction strength")
            force_sum += xp.mean(force, axis=0) * step
            diffusion_sum += xp.mean(xp.diagonal(diffusion, axis1=1, axis2=2), axis=0) * step
            elapsed += step
            n_substeps += 1
        changed = xp.any(lab_velocity != initial_velocity, axis=1)
        if bool(xp.any(changed)):
            updated = coordinates[changed].copy()
            _store_lab_proper_velocity(updated, lab_velocity[changed], bunch, ks, xp)
            coordinates[changed] = updated
        final_gamma = xp.sqrt(1 + xp.sum((lab_velocity / const.c)**2, axis=1))
        energy_change = xp.sum(xp.sum(
            (lab_velocity - initial_velocity) * (lab_velocity + initial_velocity), axis=1) / (final_gamma + initial_gamma)) * mass_kg
        return dict(n_substeps=n_substeps,
                    noise_used=noise_used,
                    density_mean=float(xp.mean(density)),
                    overlap_fraction=float(xp.mean(density > 0)),
                    energy_change_j=float(energy_change),
                    force_mean_n=[float(value) for value in force_sum / length],
                    diffusion_diagonal=[float(value) for value in diffusion_sum / length],
                    physical_interaction_time_s=physical_time,
                    coefficients=detail)

    def _track_bunch(self, coordinates, bunch, rng, xp):
        mass_number = int(bunch.num_proton) + int(bunch.num_neutron)
        mass_energy = float(bunch.m0) * (mass_number if bunch.particle_type == "Ion" else 1)
        mass_kg = mass_energy * const.e / const.c**2
        if not np.isfinite(mass_kg) or mass_kg <= 0 or not 0 < bunch.beta < 1 or not np.isfinite(bunch.gamma):
            raise ValueError("ElectronCooler requires a physical species and reference velocity")
        if not np.isclose(bunch.beta**2 + bunch.gamma**-2, 1, rtol=0, atol=2e-14):
            raise ValueError("ElectronCooler reference beta and gamma are inconsistent")
        charge_number = int(bunch.num_charge)
        ks = charge_number * const.e * self.parameters.magnetic_field / (mass_kg * bunch.beta * bunch.gamma * const.c)
        active = xp.ones(len(coordinates), dtype=bool)
        lost_position = xp.full(len(coordinates), xp.nan)
        length = self.length / self.parameters.num_slices
        energy_change = 0.
        n_substeps = 0
        noise_used = False
        last = {}
        _lab_proper_velocity(coordinates, bunch, 0., xp)
        inside = self.aperture.strict_mask(coordinates[:, 0], coordinates[:, 2])
        active[~inside] = False
        lost_position[~inside] = self.s - self.length
        for slice_index in range(self.parameters.num_slices):
            if self.length == 0:
                break
            indices = xp.flatnonzero(active)
            if len(indices) == 0:
                break
            working = coordinates[indices].copy()
            _transport_coordinates(working, bunch, 0.5 * length, ks, xp)
            at_center = (slice_index + 0.5) * length
            inside = self.aperture.strict_mask(working[:, 0], working[:, 2])
            newly_lost = indices[~inside]
            active[newly_lost] = False
            lost_position[newly_lost] = self.s - self.length + at_center
            moving = working[inside].copy()
            if charge_number != 0 and len(moving):
                record = self._apply_collision(moving, bunch, at_center, length, ks, mass_kg, charge_number, rng, xp)
                energy_change += record["energy_change_j"]
                n_substeps += record["n_substeps"]
                noise_used |= record["noise_used"]
                last = record
            _transport_coordinates(moving, bunch, 0.5 * length, ks, xp)
            working[inside] = moving
            coordinates[indices] = working
            inside = self.aperture.strict_mask(working[:, 0], working[:, 2])
            newly_lost = indices[active[indices] & ~inside]
            active[newly_lost] = False
            lost_position[newly_lost] = self.s - self.length + (slice_index + 1) * length
        _lab_proper_velocity(coordinates[active], bunch, 0., xp)
        last.update(n_alive_entry=len(coordinates),
                    n_alive_exit=int(xp.sum(active)),
                    n_substeps=n_substeps,
                    noise_used=noise_used,
                    energy_change_j=energy_change,
                    energy_change_real_j=energy_change * float(bunch.ratio))
        return active, lost_position, last

    def _turn_selected(self, turn):
        selectors = self.parameters.save_turns
        return not selectors or any(turn == item[0] if len(item) == 1 else item[0] <= turn <= item[1] and (turn - item[0]) % item[2] == 0
                                    for item in selectors)

    def _save_diagnostics(self, sim, diagnostics):
        payload = json.dumps(diagnostics, allow_nan=False) + "\n"
        if self._output_path is None:
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.cmd_name).strip(".") or "cooler"
            directory = Path(sim.cfg.output_dir) / "electron_cooler"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"beam{self.beam_id}_{name}_{uuid.uuid4().hex}.jsonl"
            with path.open("x", encoding="utf-8") as stream:
                stream.write(payload)
            self._output_path = path
            self.saved_diagnostics.append(str(path))
        else:
            with self._output_path.open("a", encoding="utf-8") as stream:
                stream.write(payload)

    def _execute(self, sim, backend):
        beam = sim.beams[self.beam_id]
        p = beam.particles
        xp = p.xp
        if (backend == "cpu") != (xp is np):
            raise TypeError("ElectronCooler execution backend differs from the particle pool")
        staged = []
        rngs = {}
        records = []
        for bunch in beam.bunches:
            indices = xp.flatnonzero(p.tag[bunch.start_idx:bunch.end_idx] > 0) + bunch.start_idx
            indices = indices[xp.argsort(p.tag[indices])]
            coordinates = xp.column_stack([getattr(p, name)[indices] for name in ("x", "px", "y", "py", "z", "dp")]).astype(xp.float64)
            rng = self._candidate_rng(bunch.bunch_id)
            active, loss, record = self._track_bunch(coordinates, bunch, rng, xp)
            stored = coordinates.astype(p.dtype)
            if not bool(xp.all(xp.isfinite(stored))):
                raise ValueError("ElectronCooler coordinates overflow particle storage precision")
            _lab_proper_velocity(stored[active].astype(xp.float64), bunch, 0., xp)
            record["bunch_id"] = int(bunch.bunch_id)
            records.append(record)
            if record["noise_used"]:
                rngs[str(bunch.bunch_id)] = rng
            staged.append((bunch, indices, stored, active, loss))
        diagnostics = dict(format="PASS-electron-cooler-diagnostics-1",
                           beam_id=self.beam_id,
                           command=self.cmd_name,
                           turn=int(sim.state.turn),
                           call=self._calls,
                           model=self.parameters.model,
                           diffusion=self.parameters.diffusion,
                           length_m=self.length,
                           force_frame="electron mean rest frame",
                           density_unit="electrons/m3 in electron rest frame",
                           bunches=records)
        if self.parameters.save_diagnostics and self._turn_selected(int(sim.state.turn)):
            self._save_diagnostics(sim, diagnostics)
        for bunch, indices, coordinates, active, loss in staged:
            for column, name in enumerate(("x", "px", "y", "py", "z", "dp")):
                getattr(p, name)[indices] = coordinates[:, column]
            lost = indices[~active]
            p.tag[lost] = -xp.abs(p.tag[lost])
            p.lost_position[lost] = loss[~active]
            p.lost_turn[lost] = int(sim.state.turn)
            bunch.t0 += self.length / (bunch.beta * const.c)
        self._rngs.update(rngs)
        self._calls += 1
        self.last_diagnostics = diagnostics
        return True

    def _configuration_identity(self):
        values = dict(beam_id=self.beam_id,
                      name=self.cmd_name,
                      parameters=self.parameters.model_dump(mode="json", exclude={"save_diagnostics", "save_turns"}))
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()

    def state_dict(self):
        return dict(format="PASS-electron-cooler-state-1",
                    configuration_sha256=self._configuration_identity(),
                    entropy=self._entropy,
                    calls=self._calls,
                    generators={
                        key: copy.deepcopy(rng.bit_generator.state)
                        for key, rng in self._rngs.items()
                    })

    def load_state_dict(self, data):
        if not isinstance(data, dict) or data.get("format") != "PASS-electron-cooler-state-1" or data.get(
                "configuration_sha256") != self._configuration_identity():
            raise ValueError("ElectronCooler state does not match its physical configuration")
        for name in ("entropy", "calls"):
            if type(data.get(name)) is not int or data[name] < 0:
                raise ValueError(f"ElectronCooler state {name} must be a nonnegative integer")
        if not isinstance(data.get("generators"), dict):
            raise ValueError("ElectronCooler state generators must be a mapping")
        candidates = {}
        for key, value in data["generators"].items():
            if not isinstance(key, str) or not key.isdecimal():
                raise ValueError("ElectronCooler random-stream keys must be nonnegative bunch IDs")
            rng = np.random.default_rng()
            rng.bit_generator.state = copy.deepcopy(value)
            candidates[key] = rng
        self._rngs = candidates
        self._entropy = data["entropy"]
        self._calls = data["calls"]
        self.last_diagnostics = None
