"""Local conservative electron merging and exact deterministic weight splitting.

Grouping and conservative moment solves stay on the particles' backend. No
particles or charge are discarded when a requested target cannot be attained.
"""

import math

import numpy as np

from PASS.utils.constants import const

from .pusher import electron_energy_ev


def population_diagnostics(arrays, xp=np):
    weight = arrays["weight"]
    total = float(xp.sum(weight))
    scale = float(xp.max(weight)) if weight.size else 0.0
    effective = float(xp.sum(weight / scale)**2 / xp.sum((weight / scale)**2)) if scale > 0 else 0.0
    return dict(n_electrons=total,
                n_macroparticles=int(weight.size),
                effective_macroparticles=effective,
                max_weight_fraction=scale / total if total else 0.0)


def _moments(arrays):
    weight = arrays["weight"]
    coordinates = np.column_stack([arrays[name] for name in ("x", "y", "ux", "uy", "uz")])
    return (float(weight.sum()), np.sum(weight[:, None] * coordinates, axis=0),
            float(np.sum(weight * electron_energy_ev(arrays["ux"], arrays["uy"], arrays["uz"]))), np.sum(weight[:, None] * coordinates**2, axis=0))


def _merge_cluster(coordinates, weights):
    total = weights.sum()
    mean = np.sum(coordinates * (weights / total)[:, None], axis=0)
    energy = float(np.sum(weights / total * electron_energy_ev(*coordinates[:, 2:].T)))
    variance = np.sum((weights / total)[:, None] * (coordinates[:, 2:] - mean[2:])**2, axis=0)
    direction = np.zeros(3)
    direction[int(np.argmax(variance))] = 1.0
    kinetic_ratio = energy / const.m_e_eV
    # Solving (gamma(u+d*n)+gamma(u-d*n))/2 = 1+K gives this
    # closed form. K*(K+2) avoids cancellation in the cold-cloud limit.
    excess = max(kinetic_ratio * (kinetic_ratio + 2) - np.dot(mean[2:], mean[2:]), 0.)
    perpendicular_squared = np.sum((mean[2:] * (1 - direction))**2)
    distance = (1 + kinetic_ratio) * math.sqrt(excess / (1 + perpendicular_squared + excess))
    result = np.repeat(mean[None, :], 2, axis=0)
    result[0, 2:] += distance * direction
    result[1, 2:] -= distance * direction
    return result, np.full(2, total / 2)


def manage_population(arrays, configuration, generator=None, xp=np, geometry=None):
    """Return replacement arrays and conservation/error diagnostics.

    Merge only occupied five-dimensional cells, using two equal-weight
    representatives conserving total weight, first moments, and relativistic
    kinetic energy. Spatial and velocity second moments may change and their
    absolute errors are returned. Nonconvex-chamber centroid failures skip the
    merge. Identical-state splitting bounds weights without adding phase noise.
    """
    if xp is not np:
        return _manage_population_device(arrays, configuration, xp, geometry)
    names = ("x", "y", "ux", "uy", "uz", "weight")
    host = {name: np.array(arrays[name] if xp is np else xp.asnumpy(arrays[name]), dtype=float, copy=True) for name in names}
    shape = host["weight"].shape
    if len(shape) != 1 or any(value.shape != shape or not np.all(np.isfinite(value)) for value in host.values()) or np.any(host["weight"] < 0):
        raise ValueError("population arrays must be equally sized finite vectors with nonnegative weights")
    before = _moments(host)
    initial_count = shape[0]
    merged_groups, split_parents = 0, 0
    alive = host["weight"] > 0
    host = {name: value[alive] for name, value in host.items()}
    count = host["weight"].size
    if count > configuration.target_macroparticles:
        coordinates = np.column_stack([host[name] for name in names[:-1]])
        widths = np.array([configuration.spatial_cell_size] * 2 + [configuration.momentum_cell_size] * 3)
        cells = np.floor(coordinates / widths)
        if not np.all(np.isfinite(cells)):
            raise ValueError("population-control cells are not representable")
        _, inverse = np.unique(cells, axis=0, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        boundaries = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1, count]
        pieces, weight_pieces = [], []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            selected = order[start:end]
            if selected.size >= 3 and count > configuration.target_macroparticles:
                merged, weights = _merge_cluster(coordinates[selected], host["weight"][selected])
                if geometry is None or bool(np.all(geometry.contains(merged[:, 0], merged[:, 1], xp=np))):
                    pieces.append(merged)
                    weight_pieces.append(weights)
                    count -= selected.size - 2
                    merged_groups += 1
                    continue
            pieces.append(coordinates[selected])
            weight_pieces.append(host["weight"][selected])
        coordinates = np.concatenate(pieces)
        host = {name: coordinates[:, index] for index, name in enumerate(names[:-1])}
        host["weight"] = np.concatenate(weight_pieces)
    if 0 < count < configuration.target_macroparticles:
        mean_weight = host["weight"].sum() / count
        repeats = np.ones(count, dtype=int)
        budget = configuration.target_macroparticles - count
        for index in np.argsort(-host["weight"], kind="stable"):
            required = min(int(math.ceil(host["weight"][index] / (configuration.max_weight_ratio * mean_weight))), configuration.max_split_children)
            children = min(max(required, 1), budget + 1)
            repeats[index] = children
            budget -= children - 1
            split_parents += int(children > 1)
            if budget == 0:
                break
        host["weight"] = host["weight"] / repeats
        host = {name: np.repeat(value, repeats) for name, value in host.items()}
    after = _moments(host)
    diagnostics = population_diagnostics(host)
    diagnostics.update(population_before=initial_count,
                       merged_groups=merged_groups,
                       split_parents=split_parents,
                       target_reached=int(host["weight"].size <= configuration.target_macroparticles),
                       weight_error=after[0] - before[0],
                       kinetic_energy_error_ev=after[2] - before[2])
    for index, name in enumerate(names[:-1]):
        diagnostics[name + "_first_moment_error"] = float(after[1][index] - before[1][index])
        diagnostics[name + "_second_moment_error"] = float(after[3][index] - before[3][index])
    return {name: xp.asarray(value) for name, value in host.items()}, diagnostics


def _device_moments(coordinates, weight, xp):
    return xp.concatenate((weight.sum()[None], xp.sum(weight[:, None] * coordinates, axis=0),
                           xp.sum(weight * electron_energy_ev(*coordinates[:, 2:].T, xp))[None], xp.sum(weight[:, None] * coordinates**2, axis=0)))


def _manage_population_device(arrays, configuration, xp, geometry):
    names = ("x", "y", "ux", "uy", "uz", "weight")
    values = [xp.asarray(arrays[name], dtype=xp.float64) for name in names]
    shape = values[-1].shape
    if len(shape) != 1 or any(value.shape != shape for value in values):
        raise ValueError("population arrays must be equally sized finite vectors with nonnegative weights")
    coordinates, weight = xp.stack(values[:-1], axis=1), values[-1]
    scale = xp.max(weight) if weight.size else xp.asarray(0.)
    minimum = xp.min(weight) if weight.size else xp.asarray(0.)
    scaled = weight / xp.where(scale > 0, scale, 1.)
    initial = xp.asnumpy(
        xp.stack((xp.all(xp.isfinite(coordinates)) & xp.all(xp.isfinite(weight)) & xp.all(weight >= 0), weight.sum(), scale, minimum, xp.sum(scaled),
                  xp.sum(scaled**2))))
    if not initial[0]:
        raise ValueError("population arrays must be equally sized finite vectors with nonnegative weights")
    initial_count = weight.size
    total, scale, minimum, scaled_sum, scaled_squared = initial[1:]
    if initial_count == 0 or (minimum > 0 and initial_count <= configuration.target_macroparticles and
                              (initial_count == configuration.target_macroparticles
                               or scale <= configuration.max_weight_ratio * total / initial_count)):
        # Most substeps require neither merging nor splitting. Keep their whole
        # cloud resident without sorting, rebinning or computing moment changes.
        diagnostics = dict(n_electrons=float(total),
                           n_macroparticles=initial_count,
                           effective_macroparticles=float(scaled_sum**2 / scaled_squared) if scaled_squared else 0.,
                           max_weight_fraction=float(scale / total) if total else 0.,
                           population_before=initial_count,
                           merged_groups=0,
                           split_parents=0,
                           target_reached=1,
                           weight_error=0.,
                           kinetic_energy_error_ev=0.)
        for name in names[:-1]:
            diagnostics[name + "_first_moment_error"] = 0.
            diagnostics[name + "_second_moment_error"] = 0.
        return {name: value.copy() for name, value in zip(names, values)}, diagnostics
    before = _device_moments(coordinates, weight, xp)
    alive = weight > 0
    coordinates, weight = coordinates[alive], weight[alive]
    count = weight.size
    merged_groups = xp.asarray(0, dtype=xp.int64)
    split_parents = xp.asarray(0, dtype=xp.int64)
    if count > configuration.target_macroparticles:
        widths = xp.asarray([configuration.spatial_cell_size] * 2 + [configuration.momentum_cell_size] * 3)
        cells = xp.floor(coordinates / widths)
        if not bool(xp.all(xp.isfinite(cells))):
            raise ValueError("population-control cells are not representable")
        order = xp.lexsort(xp.ascontiguousarray(cells[:, ::-1].T))
        coordinates, weight, cells = coordinates[order], weight[order], cells[order]
        first = xp.r_[xp.asarray([True]), xp.any(cells[1:] != cells[:-1], axis=1)]
        starts = xp.flatnonzero(first)
        group = xp.cumsum(first) - 1
        sizes = xp.diff(xp.r_[starts, count])
        totals = xp.add.reduceat(weight, starts)
        means = xp.add.reduceat(weight[:, None] * coordinates, starts, axis=0) / totals[:, None]
        energies = xp.add.reduceat(weight * electron_energy_ev(*coordinates[:, 2:].T, xp), starts) / totals
        variance = xp.add.reduceat(weight[:, None] * (coordinates[:, 2:] - means[group, 2:])**2, starts, axis=0) / totals[:, None]
        directions = xp.argmax(variance, axis=1)
        kinetic_ratio = energies / const.m_e_eV
        excess = xp.maximum(kinetic_ratio * (kinetic_ratio + 2) - xp.sum(means[:, 2:]**2, axis=1), 0.)
        perpendicular_squared = xp.sum(xp.where(xp.arange(3)[None, :] != directions[:, None], means[:, 2:]**2, 0.), axis=1)
        distances = (1 + kinetic_ratio) * xp.sqrt(excess / (1 + perpendicular_squared + excess))
        allowed = sizes >= 3
        if geometry is not None:
            allowed &= geometry.contains(means[:, 0], means[:, 1], xp=xp)
        savings = xp.where(allowed, sizes - 2, 0)
        merge = allowed & (xp.cumsum(savings) - savings < count - configuration.target_macroparticles)
        merged_groups = merge.sum()
        merged = xp.repeat(means[merge], 2, axis=0)
        displacement = xp.repeat(distances[merge], 2) * xp.tile(xp.asarray([1., -1.]), int(merged.shape[0] // 2))
        merged[xp.arange(merged.shape[0]), xp.repeat(directions[merge], 2) + 2] += displacement
        keep = ~merge[group]
        # Stable group ordering matches the local-cell traversal on the CPU.
        labels = xp.concatenate((group[keep], xp.repeat(xp.flatnonzero(merge), 2)))
        order = xp.argsort(labels)
        coordinates = xp.concatenate((coordinates[keep], merged))[order]
        weight = xp.concatenate((weight[keep], xp.repeat(totals[merge] / 2, 2)))[order]
        count = weight.size
    if 0 < count < configuration.target_macroparticles:
        mean_weight = weight.sum() / count
        order = xp.argsort(-weight)
        required = xp.minimum(xp.ceil(weight[order] / (configuration.max_weight_ratio * mean_weight)),
                              configuration.max_split_children).astype(xp.int64)
        extra = xp.maximum(required - 1, 0)
        remaining = xp.maximum(configuration.target_macroparticles - count - (xp.cumsum(extra) - extra), 0)
        repeats = xp.empty(count, dtype=xp.int64)
        repeats[order] = xp.minimum(extra, remaining) + 1
        split_parents = xp.sum(repeats > 1)
        # One packed repeat avoids a device size synchronization per coordinate.
        packed = xp.repeat(xp.column_stack((coordinates, weight / repeats)), repeats, axis=0)
        coordinates, weight = packed[:, :5], packed[:, 5]
    after = _device_moments(coordinates, weight, xp)
    scale = xp.max(weight) if weight.size else xp.asarray(0.)
    scaled = weight / xp.where(scale > 0, scale, 1.)
    ledger = xp.asnumpy(
        xp.concatenate((after - before, xp.stack((weight.sum(), scale, xp.sum(scaled), xp.sum(scaled**2), merged_groups, split_parents)))))
    total, scale, scaled_sum, scaled_squared, merged_groups, split_parents = ledger[12:]
    diagnostics = dict(n_electrons=float(total),
                       n_macroparticles=weight.size,
                       effective_macroparticles=float(scaled_sum**2 / scaled_squared) if scaled_squared else 0.,
                       max_weight_fraction=float(scale / total) if total else 0.,
                       population_before=initial_count,
                       merged_groups=int(merged_groups),
                       split_parents=int(split_parents),
                       target_reached=int(weight.size <= configuration.target_macroparticles),
                       weight_error=float(ledger[0]),
                       kinetic_energy_error_ev=float(ledger[6]))
    for index, name in enumerate(names[:-1]):
        diagnostics[name + "_first_moment_error"] = float(ledger[index + 1])
        diagnostics[name + "_second_moment_error"] = float(ledger[index + 7])
    result = {name: coordinates[:, index].copy() for index, name in enumerate(names[:-1])}
    result["weight"] = weight.copy()
    return result, diagnostics
