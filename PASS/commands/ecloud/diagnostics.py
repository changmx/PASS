"""Weighted electron diagnostics evaluated only at requested observation times."""

from copy import deepcopy
from functools import lru_cache

import numpy as np

from PASS.utils.constants import const

from .pusher import electron_energy_ev


def population_statistics(state, core_radius=None):
    """Return SI observables for the represented source length, not the kick length."""
    weight = np.asarray(state.weight, dtype=np.float64)
    total = float(weight.sum())
    scale = float(weight.max(initial=0))
    normalized = weight / scale if scale else weight
    effective = float(normalized.sum()**2 / np.dot(normalized, normalized)) if scale else 0.0
    energy = electron_energy_ev(state.ux, state.uy, state.uz)
    result = dict(n_macroparticles=int(weight.size),
                  n_electrons=total,
                  effective_macroparticles=effective,
                  line_density=total / state.source_length,
                  maximum_weight=scale,
                  kinetic_energy_ev=float(np.dot(weight, energy)))
    result["mean_energy_ev"] = result["kinetic_energy_ev"] / total if total else 0.0
    for name in ("x", "y", "ux", "uy", "uz"):
        values = np.asarray(getattr(state, name))
        mean = float(np.dot(weight / total, values)) if total else 0.0
        variance = float(np.dot(weight / total, (values - mean)**2)) if total else 0.0
        result["mean_" + name] = mean
        result["rms_" + name] = float(np.sqrt(variance))
    if core_radius is not None:
        if not np.isfinite(core_radius) or core_radius <= 0:
            raise ValueError("core_radius must be finite and positive")
        core_number = float(weight[np.hypot(state.x, state.y) <= core_radius].sum())
        result["core_electrons"] = core_number
        result["core_density"] = core_number / (np.pi * core_radius**2 * state.source_length)
    return result


def diagnostic_histograms(state, chamber=None, bins=64):
    """Return number-weighted spectra; each spectrum integrates to total weight."""
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("diagnostic bins must be an integer of at least two")
    weight = np.asarray(state.weight)
    radial = np.hypot(state.x, state.y)
    energy = electron_energy_ev(state.ux, state.uy, state.uz)
    radius = max(float(radial.max(initial=0)), np.finfo(float).tiny)
    maximum_energy = max(float(energy.max(initial=0)), 1.0)
    definitions = (("energy", energy, (0.0, maximum_energy), "eV"), ("radius", radial, (0.0, radius), "m"), ("azimuth", np.arctan2(state.y, state.x),
                                                                                                             (-np.pi, np.pi), "rad"))
    result = {}
    for name, values, interval, unit in definitions:
        counts, edges = np.histogram(values, bins=bins, range=interval, weights=weight)
        result[name] = dict(edges=edges, centers=(edges[:-1] + edges[1:]) / 2, values=counts, axis_unit=unit, value_unit="electrons/bin")
    return result


def wall_power(incident_energy_ev, emitted_energy_ev, duration, source_length, wall_perimeter=None):
    """Net kinetic energy deposited in the wall, positive for absorption."""
    values = (incident_energy_ev, emitted_energy_ev, duration, source_length)
    if not all(np.isfinite(value) for value in values) or min(incident_energy_ev, emitted_energy_ev) < 0:
        raise ValueError("wall energy and normalization must be finite with nonnegative incident/emitted energies")
    if duration <= 0 or source_length <= 0:
        raise ValueError("wall power requires a positive observation duration and source length")
    energy = (incident_energy_ev - emitted_energy_ev) * const.e
    result = dict(wall_energy_j=energy, wall_power_per_m=energy / (duration * source_length))
    if wall_perimeter is not None:
        if not np.isfinite(wall_perimeter) or wall_perimeter <= 0:
            raise ValueError("wall perimeter must be finite and positive")
        result["mean_wall_power_per_area"] = result["wall_power_per_m"] / wall_perimeter
    return result


def write_diagnostic_histograms(group, histograms):
    """Write spectra with their own axes, without mixing different physical bins."""
    for name, histogram in histograms.items():
        target = group.create_group(name)
        for key in ("edges", "centers", "values"):
            target.create_dataset(key, data=histogram[key]).attrs["unit"] = histogram["value_unit" if key == "values" else "axis_unit"]


def record_wall_histograms(histograms, x, y, ux, uy, uz, weight, inward_nx, inward_ny, chamber, xp=np, emitted_energy_ev=None):
    """Return cumulative histograms without mutating an existing observation.

    emitted_energy_ev is the sum of weight*kinetic_energy for all children of
    each incident macro particle, in eV. It defaults to zero for full absorption.
    Wall position is polar angle about the chamber origin, not arclength. This
    coordinate remains explicit for noncircular or non-star-shaped chambers.
    Spectra are cumulative since the origin of the current component state.
    The last incident-energy bin includes every event at or above 1 MeV; its
    finite plotting edge is 10 MeV, with overflow semantics saved as metadata.
    """
    if xp is not np:
        return _record_wall_histograms_device(histograms, x, y, ux, uy, uz, weight, inward_nx, inward_ny, chamber, xp, emitted_energy_ev)
    arrays = [np.asarray(value, dtype=float) for value in (x, y, ux, uy, uz, weight, inward_nx, inward_ny)]
    x, y, ux, uy, uz, weight, nx, ny = arrays
    if any(value.ndim != 1 or value.shape != weight.shape or not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("wall observations require paired finite vectors")
    if np.any(weight < 0):
        raise ValueError("wall observation weights must be nonnegative")
    outgoing = np.zeros_like(weight) if emitted_energy_ev is None else np.asarray(emitted_energy_ev if xp is np else xp.asnumpy(emitted_energy_ev),
                                                                                  dtype=float)
    if outgoing.shape != weight.shape or not np.all(np.isfinite(outgoing)) or np.any(outgoing < 0):
        raise ValueError("outgoing wall energy must be a finite nonnegative weighted energy per incident macro particle")
    energy = electron_energy_ev(ux, uy, uz)
    magnitude = np.hypot(np.hypot(ux, uy), uz)
    cosine = np.clip(-(ux * nx + uy * ny) / np.where(magnitude > 0, magnitude, 1.), 0., 1.)
    incident_angle = np.arccos(cosine)
    wall_angle = np.arctan2(y, x)
    deposited_j = (weight * energy - outgoing) * const.e
    if not np.all(np.isfinite(deposited_j)):
        raise ValueError("wall deposited-energy observation overflowed")
    result = deepcopy(histograms or {})
    energy_edges = np.r_[0., np.geomspace(1e-3, 1e6, 73), 1e7]
    definitions = (
        ("wall_incident_energy", np.minimum(energy, np.nextafter(energy_edges[-1], 0.)), energy_edges, "eV", "Incident electron kinetic energy"),
        ("wall_incident_angle", incident_angle, np.linspace(0., np.pi / 2, 65), "rad", "Angle from inward normal with incident momentum reversed"),
        ("wall_position_angle", wall_angle, np.linspace(-np.pi, np.pi,
                                                        129), "rad", "Polar angle atan2(y,x) about chamber origin; not wall arclength"),
    )
    for name, coordinates, edges, unit, definition in definitions:
        for suffix, weights, value_unit in (("number", weight, "electrons/bin"), ("deposited_energy", deposited_j, "J/bin")):
            key = name + "_" + suffix
            counts = np.histogram(coordinates, bins=edges, weights=weights)[0]
            if key in result:
                previous = result[key]
                if not np.array_equal(previous["edges"], edges) or np.shape(previous["values"]) != counts.shape:
                    raise ValueError("wall histogram binning differs from the saved component observation")
                counts = np.asarray(previous["values"], dtype=float) + counts
            if not np.all(np.isfinite(counts)):
                raise ValueError("cumulative wall histogram overflowed")
            result[key] = dict(edges=edges.copy(),
                               centers=(edges[:-1] + edges[1:]) / 2,
                               values=counts,
                               axis_unit=unit,
                               value_unit=value_unit,
                               scope=result.get(key, {}).get("scope", "cumulative_since_state_origin"),
                               coordinate_definition=definition,
                               chamber_kind=chamber.kind,
                               last_bin_is_overflow=name == "wall_incident_energy")
    return result


@lru_cache(maxsize=1)
def _wall_histogram_kernel():
    import cupy as cp

    source = r'''
extern "C" __global__ void wall_histograms(
    const double* x,
    const double* y,
    const double* ux,
    const double* uy,
    const double* uz,
    const double* weight,
    const double* nx,
    const double* ny,
    const double* outgoing,
    const double* edges,
    const double rest_energy,
    const double elementary_charge,
    const long long count,
    double* result
) {
    __shared__ double local[532];
    for (int j = threadIdx.x; j < 532; j += blockDim.x) {
        local[j] = 0.0;
    }
    __syncthreads();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const bool finite = isfinite(x[i]) && isfinite(y[i]) && isfinite(ux[i]) && isfinite(uy[i]) && isfinite(uz[i]) && isfinite(weight[i]) &&
                            isfinite(nx[i]) && isfinite(ny[i]);
        const double magnitude = hypot(hypot(ux[i], uy[i]), uz[i]);
        const double energy = rest_energy * (magnitude / (hypot(1.0, magnitude) + 1.0)) * magnitude;
        const double deposited = (weight[i] * energy - outgoing[i]) * elementary_charge;
        if (!finite) {
            atomicAdd(result + 532, 1.0);
        } else if (weight[i] < 0.0) {
            atomicAdd(result + 533, 1.0);
        } else if (!isfinite(outgoing[i]) || outgoing[i] < 0.0) {
            atomicAdd(result + 534, 1.0);
        } else if (!isfinite(deposited)) {
            atomicAdd(result + 535, 1.0);
        } else {
            const double cosine = fmin(fmax(-(ux[i] * nx[i] + uy[i] * ny[i]) / (magnitude > 0.0 ? magnitude : 1.0), 0.0), 1.0);
            const double values[3] = {fmin(energy, nextafter(1e7, 0.0)), acos(cosine), atan2(y[i], x[i])};
            const int lengths[3] = {74, 64, 128};
            int edge_offset = 0;
            int bin_offset = 0;
            for (int axis = 0; axis < 3; ++axis) {
                const int length = lengths[axis];
                const double value = values[axis];
                int lower = 0;
                int upper = length + 1;
                while (lower < upper) {
                    const int middle = (lower + upper) / 2;
                    if (edges[edge_offset + middle] <= value) {
                        lower = middle + 1;
                    } else {
                        upper = middle;
                    }
                }
                const int bin = value == edges[edge_offset + length] ? length - 1 : lower - 1;
                if (bin >= 0 && bin < length) {
                    atomicAdd(local + bin_offset + bin, weight[i]);
                    atomicAdd(local + 266 + bin_offset + bin, deposited);
                }
                edge_offset += length + 1;
                bin_offset += length;
            }
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < 532; j += blockDim.x) {
        atomicAdd(result + j, local[j]);
    }
}
'''
    return cp.RawKernel(source, "wall_histograms")


def _record_wall_histograms_device(histograms, x, y, ux, uy, uz, weight, nx, ny, chamber, xp, emitted_energy_ev):
    arrays = [xp.ascontiguousarray(value, dtype=xp.float64) for value in (x, y, ux, uy, uz, weight, nx, ny)]
    weight = arrays[5]
    if any(value.ndim != 1 or value.shape != weight.shape for value in arrays):
        raise ValueError("wall observations require paired finite vectors")
    outgoing = xp.zeros_like(weight) if emitted_energy_ev is None else xp.ascontiguousarray(emitted_energy_ev, dtype=xp.float64)
    if outgoing.shape != weight.shape:
        raise ValueError("outgoing wall energy must be a finite nonnegative weighted energy per incident macro particle")
    definitions = (
        ("wall_incident_energy", np.r_[0., np.geomspace(1e-3, 1e6, 73), 1e7], "eV", "Incident electron kinetic energy"),
        ("wall_incident_angle", np.linspace(0., np.pi / 2, 65), "rad", "Angle from inward normal with incident momentum reversed"),
        ("wall_position_angle", np.linspace(-np.pi, np.pi, 129), "rad", "Polar angle atan2(y,x) about chamber origin; not wall arclength"),
    )
    ledger = xp.zeros(536, dtype=xp.float64)
    if weight.size:
        _wall_histogram_kernel()(((weight.size + 255) // 256, ), (256, ),
                                 (*arrays, outgoing, xp.asarray(np.concatenate([item[1] for item in definitions])), np.float64(
                                     const.m_e_eV), np.float64(const.e), np.int64(weight.size), ledger))
    # Six spectra plus validity flags are the only device-to-host transfer.
    ledger = xp.asnumpy(ledger)
    for flag, message in zip(ledger[532:], ("wall observations require paired finite vectors", "wall observation weights must be nonnegative",
                                            "outgoing wall energy must be a finite nonnegative weighted energy per incident macro particle",
                                            "wall deposited-energy observation overflowed")):
        if flag:
            raise ValueError(message)
    result = deepcopy(histograms or {})
    offset = 0
    for name, edges, unit, definition in definitions:
        bins = edges.size - 1
        for index, (suffix, value_unit) in enumerate((("number", "electrons/bin"), ("deposited_energy", "J/bin"))):
            key = name + "_" + suffix
            counts = ledger[index * 266 + offset:index * 266 + offset + bins].copy()
            if key in result:
                previous = result[key]
                if not np.array_equal(previous["edges"], edges) or np.shape(previous["values"]) != counts.shape:
                    raise ValueError("wall histogram binning differs from the saved component observation")
                counts += np.asarray(previous["values"], dtype=float)
            if not np.all(np.isfinite(counts)):
                raise ValueError("cumulative wall histogram overflowed")
            result[key] = dict(edges=edges.copy(),
                               centers=(edges[:-1] + edges[1:]) / 2,
                               values=counts,
                               axis_unit=unit,
                               value_unit=value_unit,
                               scope=result.get(key, {}).get("scope", "cumulative_since_state_origin"),
                               coordinate_definition=definition,
                               chamber_kind=chamber.kind,
                               last_bin_is_overflow=name == "wall_incident_energy")
        offset += bins
    return result
