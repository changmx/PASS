"""Derive the shared machine clock from an independent, ideal RF trajectory."""

from decimal import Decimal, localcontext
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from PASS.tool.particle_masses import tracking_mass_per_nucleon
from PASS.utils.constants import const
from PASS.utils.program import LinearProgram


def _lower_keys(value):
    if isinstance(value, dict):
        return {key.lower(): _lower_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_lower_keys(item) for item in value]
    return value


def _frequency(energy, mass, circumference):
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("Automatic reference trajectory requires positive finite kinetic energy")
    beta = np.sqrt(energy * (energy + 2 * mass)) / (energy + mass)
    return beta * const.c / circumference


def build_reference_program(data, base_dir=None):
    """Return a cached design clock, without allocating or tracking a particle pool.

    The ideal harmonic-ID-zero reference starts at S=0, t=0. Enabled RF kicks
    set its energy; fixed-circumference flights use the post-kick velocity.
    The machine frequency is linear between pre-kick passage nodes (including
    turn boundaries), with constant endpoint extrapolation. Collective forces,
    apertures, tracked bunch energies and measured centroids never enter this
    construction. RF phase remains physical phase modulation, not synchronous
    phase. The prescription is deterministic but cannot reproduce an arbitrary
    independently supplied frequency program.
    """
    data = _lower_keys(data)
    if data.get("reference clock") is not None or data.get("reference_clock") is not None:
        raise ValueError("Reference clock is no longer an input; the design clock is calculated automatically. "
                         "Regenerate legacy harmonic RF inputs or preserve an external RF waveform with Frequency (Hz).")
    sequence = data.get("sequence", {})
    injection = next((item for item in sequence.values() if item.get("command", "").lower() == "injection"), None)
    if injection is None:
        raise ValueError("Automatic reference clock requires Injection")
    initial = next(
        (item
         for key, item in injection.items() if key.startswith("bunch") and isinstance(item, dict) and item.get("harmonic id of this bunch", 0) == 0),
        None)
    if initial is None:
        raise ValueError("Automatic reference clock requires the harmonic-ID-zero injection group")
    energy = float(initial["kinetic energy per nucleon (ev/u)"])
    circumference = float(data["circumference (m)"])
    turns = data["number of turns"]
    if not np.isfinite(circumference) or circumference <= 0 or type(turns) is not int or turns < 1:
        raise ValueError("Automatic reference clock requires positive circumference and an integer positive turn count")
    protons, neutrons, charges = (int(data[key]) for key in ("number of protons", "number of neutrons", "number of charges"))
    mass = tracking_mass_per_nucleon(protons, neutrons, charges)
    charge = charges / (protons + neutrons) if protons + neutrons else float(charges)
    stations = []
    dependencies = []
    for name, item in sequence.items():
        if item.get("command", "").lower() != "rfcavity" or not item.get("is enabled", True):
            continue
        position = float(item["s (m)"])
        if not np.isfinite(position) or not 0 <= position <= circumference:
            raise ValueError(f"RFCavity {name}: S must lie within the ring for the automatic reference trajectory")
        components = item.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError(f"RFCavity {name}: Components must be a nonempty list")
        for component in components:
            if component.get("program file"):
                path = Path(component["program file"])
                if not path.is_absolute():
                    path = Path(base_dir or Path.cwd()) / path
                path = path.resolve()
                stat = path.stat()
                component["program file"] = str(path)
                dependencies.append((str(path), stat.st_mtime_ns, stat.st_size))
        stations.append((position, item.get("order"), name, components))
    # At one location every component sees the same physical time. Summing the
    # gains also avoids introducing a clock change between zero-length cavities.
    stations.sort(key=lambda item: (item[0], item[1] if item[1] is not None else 0))
    specification = json.dumps(stations, allow_nan=False, sort_keys=True)
    return _build_reference_program(energy, mass, circumference, charge, turns, specification, tuple(dependencies))


@lru_cache(maxsize=4)
def _build_reference_program(initial_energy, mass, circumference, charge, turns, specification, dependencies):
    # Delayed import: commands themselves import Beam during package setup.
    from PASS.commands.element.rfcavity import RFWaveform, _component_parameters

    initial_frequency = _frequency(initial_energy, mass, circumference)
    reference = LinearProgram(initial_frequency)
    stations = {}
    for position, order, name, components in json.loads(specification):
        for component in components:
            parameters = _component_parameters(component)
            waveform = RFWaveform(parameters, reference)
            # NumPy may copy read-only interpolation inputs on every scalar
            # call. These private, aligned work arrays are copied only once.
            samples = tuple(np.array(array, copy=True) for program in (waveform.voltage, waveform.phase) for array in (program.times, program.values))
            stations.setdefault(position, []).append((parameters.harmonic, waveform, samples))
    if not stations or all(not np.any(waveform.voltage.values) for components in stations.values() for harmonic, waveform, samples in components):
        return reference
    stations.setdefault(circumference, [])
    positions = sorted(stations)
    times, frequencies = [0.], [initial_frequency]
    energy, time, frequency, cycles = initial_energy, 0., initial_frequency, 0.
    with localcontext() as context:
        context.prec = 80
        accumulated = Decimal(0)
        # One extra RF-free passage closes interpolation after the final kick.
        for turn in range(turns + 1):
            previous_position = 0.
            for position in positions:
                distance = position - previous_position
                if distance:
                    next_frequency = _frequency(energy, mass, circumference)
                    beta = np.sqrt(energy * (energy + 2 * mass)) / (energy + mass)
                    next_time = time + distance / (beta * const.c)
                    if next_time <= time:
                        raise ValueError("Automatic reference passage spacing is below floating-point time resolution")
                    interval = Decimal.from_float(next_time) - Decimal.from_float(time)
                    mean = (Decimal.from_float(frequency) + Decimal.from_float(next_frequency)) / 2
                    accumulated = (accumulated + interval * mean) % 1
                    cycles = float(accumulated) % 1.
                    time, frequency = next_time, next_frequency
                    times.append(time)
                    frequencies.append(frequency)
                if turn == turns:
                    # Padding is outside the requested tracking interval. An
                    # unused kick must not stop an otherwise valid trajectory.
                    previous_position = position
                    continue
                gain = 0.
                for harmonic, waveform, samples in stations[position]:
                    if not waveform.time_min <= time <= waveform.time_max:
                        continue
                    if harmonic is None:
                        value = float(waveform.value(time))
                    else:
                        voltage = np.interp(time, samples[0], samples[1])
                        phase = np.interp(time, samples[2], samples[3])
                        value = voltage * np.sin(2 * np.pi * np.remainder(harmonic * cycles, 1.) + phase)
                    gain += value
                energy += charge * gain
                if not np.isfinite(energy) or energy <= 0:
                    raise ValueError(f"Automatic RF reference trajectory stopped at turn {turn}, S={position}")
                previous_position = position
    return LinearProgram(frequencies, times)
