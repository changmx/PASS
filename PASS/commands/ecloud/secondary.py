"""Primary sources and legacy or empirical three-channel surface emission.

The three-channel model enforces a joint incident-energy budget; it is not the
complete Furman-Pivi model. GPU Philox draws use seeds from saved PCG64 state:
restores reproduce the same backend, while CPU and GPU individual samples differ.
"""

import math

import numpy as np
from scipy.special import gammainc, gammaincinv, gammaln, logsumexp

from .pusher import electron_energy_ev, normalized_momentum


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


def _host(value, xp):
    return np.asarray(value if xp is np else xp.asnumpy(value), dtype=float)


def _log_gamma_cdf(shape, value):
    if value <= 0:
        return -np.inf
    probability = gammainc(shape, value)
    if probability > 0:
        return float(np.log(probability))
    # The lower-tail power series avoids probability underflow.
    term, total = 1.0, 1.0
    for index in range(1, 10000):
        term *= value / (shape + index)
        total += term
        if term <= total * 2e-16:
            return shape * math.log(value) - value - gammaln(shape + 1) + math.log(total)
    raise ValueError("conditional secondary-energy distribution did not converge")


def _truncated_gamma(shape, scale, maximum, generator):
    if maximum <= 0:
        return 0.0
    return _invert_truncated_gamma(shape, scale, maximum, max(float(generator.random()), np.finfo(float).tiny))


def _invert_truncated_gamma(shape, scale, maximum, uniform):
    bound = maximum / scale
    probability = gammainc(shape, bound)
    if probability > np.finfo(float).tiny:
        return min(float(gammaincinv(shape, uniform * probability) * scale), maximum)
    target = _log_gamma_cdf(shape, bound) + math.log(uniform)
    lower, upper = 0.0, bound
    for _ in range(80):
        middle = (lower + upper) / 2
        if _log_gamma_cdf(shape, middle) < target:
            lower = middle
        else:
            upper = middle
    return min((lower + upper) * scale / 2, maximum)


def _true_yield(energy, cosine, model, xp=np):
    if model.true_yield_energy_ev:
        left, right = (0., 0.) if model.yield_table_outside == "zero" else (model.true_yield_table[0], model.true_yield_table[-1])
        shifted_energy = energy / (1 + model.peak_angle_coefficient * (1 - cosine))
        value = xp.interp(shifted_energy, xp.asarray(model.true_yield_energy_ev), xp.asarray(model.true_yield_table), left=left, right=right)
        return xp.where(energy > 0, value * xp.exp(model.yield_angle_coefficient * (1 - cosine)), 0.)
    peak = model.true_peak_energy_ev * (1 + model.peak_angle_coefficient * (1 - cosine))
    maximum = model.true_yield_max * xp.exp(model.yield_angle_coefficient * (1 - cosine))
    ratio = xp.log(xp.where(energy > 0, energy, 1)) - xp.log(peak)
    shape = model.true_shape
    value = maximum * xp.exp(np.log(shape) + ratio - xp.logaddexp(np.log(shape - 1), shape * ratio))
    return xp.where(energy > 0, value, 0)


def _poisson_conditioned(mean, maximum, generator):
    if maximum < 0 or mean == 0:
        return 0
    values = np.arange(maximum + 1)
    log_probabilities = values * np.log(mean) - gammaln(values + 1)
    probabilities = np.exp(log_probabilities - logsumexp(log_probabilities))
    return int(generator.choice(values, p=probabilities))


def emit_surface(incident_ux, incident_uy, incident_uz, incident_weight, normal_x, normal_y, model, generator, xp=np):
    """Return outgoing children and parent indices, without changing inputs.

    In stochastic_event mode reflection branches are exclusive. Otherwise a
    Poisson true-secondary multiplicity is drawn. Gamma child energies have the
    independent gamma joint law conditioned on their sum not exceeding the
    incident energy. Monoenergetic multiplicities use a conditioned Poisson law.
    weighted_channels keeps one representative of each nonzero channel; it is
    a separate weighted approximation and enforces its weighted energy budget.
    """
    if xp is not np:
        return _emit_surface_device(incident_ux, incident_uy, incident_uz, incident_weight, normal_x, normal_y, model, generator, xp)
    ux, uy, uz, weight, nx, ny = [_host(value, xp) for value in (incident_ux, incident_uy, incident_uz, incident_weight, normal_x, normal_y)]
    if any(value.ndim != 1 or value.shape != weight.shape or not np.all(np.isfinite(value)) for value in (ux, uy, uz, weight, nx, ny)):
        raise ValueError("surface emission requires equally sized finite vectors")
    if np.any(weight < 0) or not np.allclose(nx * nx + ny * ny, 1, rtol=1e-10, atol=1e-12):
        raise ValueError("surface emission requires nonnegative weights and unit inward normals")
    energy = electron_energy_ev(ux, uy, uz)
    magnitude = np.hypot(np.hypot(ux, uy), uz)
    if np.any(ux * nx + uy * ny > 1e-12 * np.maximum(magnitude, np.finfo(float).tiny)):
        raise ValueError("incident momenta must point towards the surface, opposite its inward normal")
    cosine = np.clip(-(ux * nx + uy * ny) / np.where(magnitude > 0, magnitude, 1), 0, 1)
    delta = _true_yield(energy, cosine, model)
    elastic = model.elastic_probability * np.exp(-energy / model.elastic_energy_scale_ev)
    rediffused = np.full(weight.shape, model.rediffused_probability)
    rows = []
    counts = {"true": 0.0, "elastic": 0.0, "rediffused": 0.0}
    for index in range(weight.size):
        if weight[index] == 0 or energy[index] <= 0:
            continue
        incident_energy = float(energy[index])
        reflected_energy = incident_energy * generator.random()**(1 / model.rediffused_energy_shape) if rediffused[index] > 0 else 0.0
        if model.sampling == "weighted_channels":
            children = []
            if elastic[index] > 0:
                children.append(("elastic", incident_energy, weight[index] * elastic[index]))
            if rediffused[index] > 0:
                children.append(("rediffused", reflected_energy, weight[index] * rediffused[index]))
            available = max(incident_energy * (1 - elastic[index]) - reflected_energy * rediffused[index], 0)
            if delta[index] > 0 and available > 0:
                upper = available / delta[index]
                if model.true_energy_distribution == "gamma":
                    emitted_energy = _truncated_gamma(model.true_energy_shape, model.true_energy_ev / model.true_energy_shape, upper, generator)
                    children.append(("true", emitted_energy, weight[index] * delta[index]))
                elif model.true_energy_ev <= upper:
                    children.append(("true", model.true_energy_ev, weight[index] * delta[index]))
        else:
            branch = generator.random()
            if branch < elastic[index]:
                children = [("elastic", incident_energy, weight[index])]
            elif branch < elastic[index] + rediffused[index]:
                children = [("rediffused", reflected_energy, weight[index])]
            else:
                probability = 1 - elastic[index] - rediffused[index]
                mean = float(delta[index] / probability) if probability > 0 else 0.0
                if not math.isfinite(mean):
                    raise ValueError("true-secondary multiplicity is not representable")
                if model.true_energy_distribution == "monoenergetic":
                    maximum = int(incident_energy / model.true_energy_ev)
                    if maximum > model.max_secondaries_per_impact:
                        number = int(generator.poisson(mean))
                        if number > maximum:
                            number = _poisson_conditioned(mean, maximum, generator)
                    else:
                        number = _poisson_conditioned(mean, maximum, generator)
                else:
                    number = int(generator.poisson(mean))
                if number > model.max_secondaries_per_impact:
                    raise ValueError("surface emission exceeds Max secondaries per impact; no offspring were discarded")
                if number == 0:
                    children = []
                elif model.true_energy_distribution == "monoenergetic":
                    children = [("true", model.true_energy_ev, weight[index])] * number
                else:
                    total = _truncated_gamma(number * model.true_energy_shape, model.true_energy_ev / model.true_energy_shape, incident_energy,
                                             generator)
                    fractions = generator.dirichlet(np.full(number, model.true_energy_shape))
                    children = [("true", total * fraction, weight[index]) for fraction in fractions]
        for channel, emitted_energy, child_weight in children:
            if channel == "elastic":
                projection = ux[index] * nx[index] + uy[index] * ny[index]
                velocity = (ux[index] - 2 * projection * nx[index], uy[index] - 2 * projection * ny[index], uz[index])
            else:
                velocity = tuple(
                    float(value[0]) for value in cosine_emission(np.array([nx[index]]), np.array([ny[index]]), emitted_energy, generator))
            rows.append((index, *velocity, child_weight, {"true": 0, "elastic": 1, "rediffused": 2}[channel]))
            counts[channel] += child_weight
    data = np.asarray(rows, dtype=float).reshape((-1, 6))
    outgoing_energy = electron_energy_ev(data[:, 1], data[:, 2], data[:, 3])
    incident_total = float(np.sum(weight * energy))
    emitted_total = float(np.sum(data[:, 4] * outgoing_energy))
    diagnostics = dict(incident_electrons=float(weight.sum()),
                       emitted_electrons=float(data[:, 4].sum()),
                       incident_energy_ev=incident_total,
                       emitted_energy_ev=emitted_total,
                       true_electrons=counts["true"],
                       elastic_electrons=counts["elastic"],
                       rediffused_electrons=counts["rediffused"],
                       wall_energy_ev=incident_total - emitted_total)
    result = {name: xp.asarray(data[:, column]) for name, column in (("ux", 1), ("uy", 2), ("uz", 3), ("weight", 4))}
    result["parent_indices"] = xp.asarray(data[:, 0], dtype=xp.int64)
    result["channel"] = xp.asarray(data[:, 5], dtype=xp.int64)
    result["diagnostics"] = diagnostics
    return result


def _truncated_gamma_device(shape, scale, maximum, device_generator, xp):
    from cupyx.scipy.special import gammainc as device_gammainc, gammaincinv as device_gammaincinv

    shape = xp.broadcast_to(xp.asarray(shape), maximum.shape)
    probability = device_gammainc(shape, maximum / scale)
    uniform = xp.maximum(device_generator.random(maximum.shape), np.finfo(float).tiny)
    result = xp.minimum(device_gammaincinv(shape, uniform * probability) * scale, maximum)
    rare = xp.flatnonzero(probability <= np.finfo(float).tiny)
    if rare.size:
        # Only underflowing conditional laws leave the device, in one batch.
        parameters = xp.asnumpy(xp.stack((shape[rare], maximum[rare], uniform[rare])))
        values = [_invert_truncated_gamma(a, scale, bound, u) for a, bound, u in parameters.T]
        result[rare] = xp.asarray(values)
    return result


def _conditioned_poisson_device(mean, maximum, device_generator, xp):
    from cupyx.scipy.special import gammaincc as device_gammaincc

    number = device_generator.poisson(mean)
    rejected = xp.flatnonzero(number > maximum)
    if rejected.size == 0:
        return number
    limits, rates = maximum[rejected], mean[rejected]
    cdf = device_gammaincc(limits + 1., rates)
    uniform = xp.maximum(device_generator.random(limits.shape), np.finfo(float).tiny)
    target = uniform * cdf
    lower, upper = xp.zeros_like(limits), limits.copy()
    # Bisection of the discrete conditional CDF avoids rejection at small caps.
    for _ in range(64):
        middle = lower + (upper - lower) // 2
        below = device_gammaincc(middle + 1., rates) < target
        lower = xp.where(below, middle + 1, lower)
        upper = xp.where(below, upper, middle)
    number[rejected] = lower
    rare = xp.flatnonzero(cdf <= np.finfo(float).tiny)
    if rare.size:
        parameters = xp.asnumpy(xp.stack((rates[rare], limits[rare], uniform[rare])))
        values = []
        for rate, limit, u in parameters.T:
            candidates = np.arange(int(limit) + 1)
            log_probability = candidates * np.log(rate) - gammaln(candidates + 1)
            cdf_host = np.cumsum(np.exp(log_probability - logsumexp(log_probability)))
            values.append(min(int(np.searchsorted(cdf_host, u)), int(limit)))
        number[rejected[rare]] = xp.asarray(values, dtype=xp.int64)
    return number


def _emit_surface_device(ux, uy, uz, weight, nx, ny, model, generator, xp):
    """Device-vectorized event sampling; host traffic is seeds and small ledgers."""
    arrays = [xp.asarray(value, dtype=xp.float64) for value in (ux, uy, uz, weight, nx, ny)]
    ux, uy, uz, weight, nx, ny = arrays
    if any(value.ndim != 1 or value.shape != weight.shape for value in arrays):
        raise ValueError("surface emission requires equally sized finite vectors")
    magnitude = xp.hypot(xp.hypot(ux, uy), uz)
    projection = ux * nx + uy * ny
    energy = electron_energy_ev(ux, uy, uz, xp)
    checks = xp.stack(
        (xp.all(xp.isfinite(xp.stack(arrays))) & xp.all(xp.isfinite(energy)), xp.all(weight >= 0) & xp.all(xp.abs(nx * nx + ny * ny - 1) <= 1.01e-10),
         xp.all(projection <= 1e-12 * xp.maximum(magnitude,
                                                 np.finfo(float).tiny))))
    valid = xp.asnumpy(checks)
    if not valid[0]:
        raise ValueError("surface emission requires equally sized finite vectors and representable kinetic energies")
    if not valid[1]:
        raise ValueError("surface emission requires nonnegative weights and unit inward normals")
    if not valid[2]:
        raise ValueError("incident momenta must point towards the surface, opposite its inward normal")
    # A fresh Philox stream derived from saved PCG64 makes clone/restore exact
    # within this backend, without adding persistent device RNG state.
    device_generator = xp.random.Generator(xp.random.Philox4x3210(int(generator.bit_generator.random_raw())))
    active = xp.flatnonzero((weight > 0) & (energy > 0))
    cosine = xp.clip(-projection / xp.where(magnitude > 0, magnitude, 1), 0, 1)
    delta = _true_yield(energy, cosine, model, xp)
    elastic = model.elastic_probability * xp.exp(-energy / model.elastic_energy_scale_ev)
    rediffused = model.rediffused_probability
    reflected = energy[active] * device_generator.random(active.size)**(1 / model.rediffused_energy_shape)
    if model.sampling == "weighted_channels":
        elastic_parents = active[elastic[active] > 0]
        rediffused_parents = active if rediffused > 0 else active[:0]
        rediffused_energy = reflected if rediffused > 0 else reflected[:0]
        available = xp.maximum(energy[active] * (1 - elastic[active]) - reflected * rediffused, 0)
        true_mask = (delta[active] > 0) & (available > 0)
        if model.true_energy_distribution == "monoenergetic":
            true_mask &= model.true_energy_ev * delta[active] <= available
        true_parents = active[true_mask]
        maximum = available[true_mask] / delta[true_parents]
        true_energy = (_truncated_gamma_device(model.true_energy_shape, model.true_energy_ev / model.true_energy_shape, maximum, device_generator, xp)
                       if model.true_energy_distribution == "gamma" else xp.full(true_parents.size, model.true_energy_ev))
        child_weight = xp.concatenate(
            (weight[elastic_parents] * elastic[elastic_parents], weight[rediffused_parents] * rediffused, weight[true_parents] * delta[true_parents]))
    else:
        branch = device_generator.random(active.size)
        elastic_mask = branch < elastic[active]
        rediffused_mask = (~elastic_mask) & (branch < elastic[active] + rediffused)
        elastic_parents, rediffused_parents = active[elastic_mask], active[rediffused_mask]
        rediffused_energy = reflected[rediffused_mask]
        candidates = active[~(elastic_mask | rediffused_mask)]
        mean = delta[candidates] / (1 - elastic[candidates] - rediffused)
        if not bool(xp.all(xp.isfinite(mean))):
            raise ValueError("true-secondary multiplicity is not representable")
        if model.true_energy_distribution == "monoenergetic":
            maximum = xp.minimum(xp.floor(energy[candidates] / model.true_energy_ev), np.iinfo(np.int64).max // 2).astype(xp.int64)
            number = _conditioned_poisson_device(mean, maximum, device_generator, xp)
        else:
            number = device_generator.poisson(mean)
        if bool(xp.any(number > model.max_secondaries_per_impact)):
            raise ValueError("surface emission exceeds Max secondaries per impact; no offspring were discarded")
        nonempty = number > 0
        candidates, number = candidates[nonempty], number[nonempty]
        true_parents = xp.repeat(candidates, number)
        if model.true_energy_distribution == "monoenergetic":
            true_energy = xp.full(true_parents.size, model.true_energy_ev)
        else:
            total = _truncated_gamma_device(number * model.true_energy_shape, model.true_energy_ev / model.true_energy_shape, energy[candidates],
                                            device_generator, xp)
            groups = xp.repeat(xp.arange(number.size), number)
            fractions = device_generator.gamma(model.true_energy_shape, size=true_parents.size)
            offsets = xp.r_[0, xp.cumsum(number)]
            normalization = xp.add.reduceat(fractions, offsets[:-1]) if number.size else xp.empty(0)
            fractions /= xp.where(normalization[groups] > 0, normalization[groups], 1)
            rare = xp.flatnonzero(normalization <= 0)
            if rare.size:
                intervals = xp.asnumpy(xp.stack((offsets[rare], offsets[rare + 1])))
                # Very small shape parameters can underflow every gamma draw.
                for start, end in intervals.T:
                    fractions[start:end] = xp.asarray(generator.dirichlet(np.full(end - start, model.true_energy_shape)))
            true_energy = total[groups] * fractions
        child_weight = weight[xp.concatenate((elastic_parents, rediffused_parents, true_parents))]
    parents = xp.concatenate((elastic_parents, rediffused_parents, true_parents))
    channels = xp.concatenate((xp.full(elastic_parents.size, 1, dtype=xp.int64), xp.full(rediffused_parents.size, 2,
                                                                                         dtype=xp.int64), xp.zeros(true_parents.size,
                                                                                                                   dtype=xp.int64)))
    diffuse_parents = xp.concatenate((rediffused_parents, true_parents))
    diffuse_energy = xp.concatenate((rediffused_energy, true_energy))
    mu = xp.sqrt(device_generator.random(diffuse_parents.size))
    azimuth = 2 * np.pi * device_generator.random(diffuse_parents.size)
    tangent = xp.sqrt(xp.maximum(1 - mu * mu, 0))
    momentum = normalized_momentum(diffuse_energy, xp)
    local_nx, local_ny = nx[diffuse_parents], ny[diffuse_parents]
    diffuse_ux = momentum * (mu * local_nx - tangent * xp.cos(azimuth) * local_ny)
    diffuse_uy = momentum * (mu * local_ny + tangent * xp.cos(azimuth) * local_nx)
    diffuse_uz = momentum * tangent * xp.sin(azimuth)
    child_ux = xp.concatenate((ux[elastic_parents] - 2 * projection[elastic_parents] * nx[elastic_parents], diffuse_ux))
    child_uy = xp.concatenate((uy[elastic_parents] - 2 * projection[elastic_parents] * ny[elastic_parents], diffuse_uy))
    child_uz = xp.concatenate((uz[elastic_parents], diffuse_uz))
    emitted_energy = electron_energy_ev(child_ux, child_uy, child_uz, xp)
    ledger = xp.asnumpy(
        xp.stack((weight.sum(), child_weight.sum(), xp.sum(weight * energy), xp.sum(child_weight * emitted_energy),
                  xp.sum(child_weight * (channels == 0)), xp.sum(child_weight * (channels == 1)), xp.sum(child_weight * (channels == 2)))))
    names = ("incident_electrons", "emitted_electrons", "incident_energy_ev", "emitted_energy_ev", "true_electrons", "elastic_electrons",
             "rediffused_electrons")
    diagnostics = dict(zip(names, map(float, ledger)))
    diagnostics["wall_energy_ev"] = float(ledger[2] - ledger[3])
    order = xp.argsort(parents)
    return dict(ux=child_ux[order],
                uy=child_uy[order],
                uz=child_uz[order],
                weight=child_weight[order],
                parent_indices=parents[order],
                channel=channels[order],
                diagnostics=diagnostics)


def _sample_beam_volume(count, geometry, generator, beam_sigma_x, beam_sigma_y, beam_center_x, beam_center_y, beam_x, beam_y, xp=np):
    if beam_x is not None or beam_y is not None:
        if beam_x is None or beam_y is None:
            raise ValueError("gas source requires both beam_x and beam_y")
        beam_x, beam_y = xp.asarray(beam_x), xp.asarray(beam_y)
        if beam_x.ndim != 1 or beam_x.size == 0 or beam_x.shape != beam_y.shape or not bool(xp.all(xp.isfinite(beam_x + beam_y))):
            raise ValueError("gas source beam coordinates must be nonempty paired finite arrays")
        if not bool(xp.all(geometry.contains(beam_x, beam_y, xp=xp))):
            raise ValueError("gas source beam particles must lie inside the chamber")
        selected = xp.asarray(generator.integers(0, beam_x.size, count))
        return beam_x[selected].copy(), beam_y[selected].copy()
    if not all(math.isfinite(value) for value in (beam_sigma_x, beam_sigma_y, beam_center_x, beam_center_y)) or min(beam_sigma_x, beam_sigma_y) <= 0:
        raise ValueError("gas source Gaussian widths must be finite and positive")
    x, y = np.empty(count), np.empty(count)
    pending = np.arange(count)
    for _ in range(1000):
        x[pending] = generator.normal(beam_center_x, beam_sigma_x, pending.size)
        y[pending] = generator.normal(beam_center_y, beam_sigma_y, pending.size)
        pending = pending[~np.asarray(geometry.contains(x[pending], y[pending], xp=np))]
        if pending.size == 0:
            return x, y
    raise ValueError("gas source Gaussian is insufficiently contained in chamber; provide actual beam coordinates")


def _sample_wall_sector(count, source, geometry, generator):
    width = source.wall_angle_max - source.wall_angle_min
    if width >= 2 * np.pi - 8 * np.finfo(float).eps:
        return geometry.sample_wall(count, generator, xp=np)
    arrays = [np.empty(count) for _ in range(4)]
    filled = 0
    for _ in range(1000):
        sampled = geometry.sample_wall(max(count - filled, 64), generator, xp=np)
        angle = (np.arctan2(sampled[1], sampled[0]) - source.wall_angle_min) % (2 * np.pi)
        selected = np.flatnonzero(angle < width)[:count - filled]
        for target, values in zip(arrays, sampled):
            target[filled:filled + selected.size] = values[selected]
        filled += selected.size
        if filled == count:
            return tuple(arrays)
    raise ValueError("illuminated wall sector is too narrow for rejection sampling")


def sample_primary_sources(sources,
                           n_real_particles,
                           source_length,
                           duration,
                           geometry,
                           generator,
                           xp=np,
                           beam_sigma=0.001,
                           beam_x=None,
                           beam_y=None,
                           beam_sigma_x=None,
                           beam_sigma_y=None,
                           beam_center_x=0.0,
                           beam_center_y=0.0):
    """Return newborn arrays and fractional birth times within an interval.

    n_real_particles counts particles traversing the station during duration.
    Gas yield is N_beam * pressure/(k_B*T) * sigma * source_length: beta is
    already accounted for in that integrated passage count. Photon input is a
    prescribed incident rate, not a synchrotron-radiation generation model.
    Wall sectors use polar angles about the chamber origin and uniform arc
    length conditional on that sector. All sampling uses the saved host RNG.
    """
    if not all(math.isfinite(value)
               for value in (n_real_particles, source_length, duration)) or n_real_particles < 0 or source_length <= 0 or duration < 0:
        raise ValueError("primary source requires finite nonnegative passage count/duration and positive source length")
    pieces = {name: [] for name in ("x", "y", "ux", "uy", "uz", "weight", "birth_fraction")}
    diagnostics = dict(primary_electrons=0.0,
                       primary_energy_ev=0.0,
                       primary_macroparticles=0,
                       gas_ionization_electrons=0.0,
                       prescribed_wall_electrons=0.0,
                       beam_loss_electrons=0.0,
                       photoelectric_electrons=0.0)
    for source in sources:
        if source.kind == "gas_ionization":
            number = n_real_particles * source_length * source.pressure_pa / (1.380649e-23 *
                                                                              source.temperature_k) * source.ionization_cross_section_m2
        elif source.kind == "prescribed_wall":
            number = n_real_particles * source_length * source.electrons_per_particle_per_m
        elif source.kind == "beam_loss":
            number = n_real_particles * source_length * source.loss_fraction_per_m * source.electrons_per_lost_particle
        else:
            number = source_length * duration * source.photon_rate_per_m_per_s * source.quantum_efficiency
        if not math.isfinite(number):
            raise ValueError("primary electron source population overflowed")
        if number == 0:
            continue
        count = source.macroparticles
        if source.energy_distribution == "monoenergetic" or source.energy_ev == 0:
            energy = np.full(count, source.energy_ev)
        else:
            energy = generator.gamma(source.energy_shape, source.energy_ev / source.energy_shape, count)
        if source.kind == "gas_ionization":
            x, y = _sample_beam_volume(count, geometry, generator, beam_sigma if beam_sigma_x is None else beam_sigma_x,
                                       beam_sigma if beam_sigma_y is None else beam_sigma_y, beam_center_x, beam_center_y, beam_x, beam_y, xp)
            ux, uy, uz = isotropic_emission(count, energy, generator)
        else:
            x, y, nx, ny = _sample_wall_sector(count, source, geometry, generator)
            ux, uy, uz = cosine_emission(nx, ny, energy, generator)
        birth_fraction = (np.arange(count) + generator.random(count)) / count
        weight = np.full(count, number / count)
        for name, value in zip(pieces, (x, y, ux, uy, uz, weight, birth_fraction)):
            pieces[name].append(value)
        diagnostics["primary_electrons"] += number
        diagnostics["primary_energy_ev"] += float(np.sum(weight * energy))
        diagnostics["primary_macroparticles"] += count
        diagnostics[source.kind + "_electrons"] += number
    result = {name: xp.concatenate([xp.asarray(value) for value in values]) if values else xp.empty(0) for name, values in pieces.items()}
    result["diagnostics"] = diagnostics
    return result
