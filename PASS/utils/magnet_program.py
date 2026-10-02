"""Immutable physical-time programs of normalized multipole strengths."""

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np
import pandas as pd
from tfs.reader import _read_metadata

from PASS.utils.constants import const
from PASS.utils.program import LinearProgram


def _ramping_sources(kwargs, *, order=None):
    """Select one unified table or distinct legacy tables without reading them."""
    values = {str(k).lower(): v for k, v in kwargs.items()}
    unified = values.get("ramping file", "")
    legacy_keys = ("kl ramping file", ) if order is None else (f"k{order}l ramping file", f"k{order}sl ramping file")
    legacy = [values.get(key, "") for key in legacy_keys if values.get(key, "")]
    if unified and legacy:
        raise ValueError("Use Ramping file or legacy component ramping files, not both")
    paths = [unified] if unified else legacy
    if not paths:
        raise ValueError("Enabled magnet ramping requires Ramping file or a legacy component ramping file")
    unique = []
    for value in paths:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Magnet ramping paths must be nonempty strings")
        path = Path(value).expanduser().resolve()
        if path not in unique:
            unique.append(path)
    return tuple(unique)


def _strength_component(name):
    match = re.fullmatch(r"K(0|[1-9][0-9]*)(S?)(L?)", str(name).upper())
    if match is None:
        raise ValueError(f"Invalid magnet ramping column {name!r}; use Kn, KnS, KnL or KnSL")
    return int(match[1]), bool(match[2]), bool(match[3])


def _real_samples(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"Magnet ramping {name} must contain a nonempty one-dimensional array of finite real numbers")
    return np.array(array, dtype=np.float64, copy=True)


def _strength_unit(order, integrated):
    power = order if integrated else order + 1
    return "1" if power == 0 else f"m^-{power}"


def _validate_samples(times, columns, headers=None, *, order=None):
    """Validate names, units and values without applying an element length."""
    times = _real_samples(times, "TIME")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Magnet ramping TIME must be strictly increasing")
    headers = {str(k).upper(): v for k, v in (headers or {}).items()}
    for key, expected in (("TIME_UNIT", "s"), ("STRENGTH_CONVENTION", "normalized")):
        if key in headers and headers[key] != expected:
            raise ValueError(f"Magnet ramping {key} must be {expected!r}")
    result, components = {}, set()
    for label, values in columns.items():
        name = str(label).upper()
        component_order, skew, integrated = _strength_component(name)
        if order is not None and component_order != order:
            raise ValueError(f"Magnet order {order} cannot use column {name}")
        component = component_order, skew
        if component in components:
            raise ValueError(f"Duplicate magnet ramping component {component}; K and KL forms cannot both be supplied")
        components.add(component)
        values = _real_samples(values, name)
        if len(values) != len(times):
            raise ValueError(f"Magnet ramping {name} and TIME must have equal lengths")
        unit_key = f"{name}_UNIT"
        expected = _strength_unit(component_order, integrated)
        power = component_order if integrated else component_order + 1
        accepted = {expected, f"1/m^{power}", f"m**-{power}"} if power else {"1", "rad", "dimensionless"}
        if unit_key in headers and headers[unit_key] not in accepted:
            raise ValueError(f"Magnet ramping {unit_key} must be {expected!r}")
        result[name] = values
    if not result:
        raise ValueError("Magnet ramping requires at least one strength column")
    return times, result


def _read_ramping_table(path):
    """Read TFS real columns once with round-trip float64 precision."""
    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        raise ValueError("Magnet ramping input must use TFS")
    try:
        metadata = _read_metadata(path)
    except UnboundLocalError as exc:
        raise ValueError("Magnet ramping file is empty") from exc
    if metadata.column_names is None or metadata.column_types is None:
        raise ValueError("Magnet ramping TFS requires column names and types")
    names = [str(name).upper() for name in metadata.column_names]
    if len(set(names)) != len(names):
        raise ValueError("Magnet ramping column names must be unique, ignoring case")
    time_names = [name for name in names if name in {"TIME", "TIME_S"}]
    if len(time_names) != 1:
        raise ValueError("Magnet ramping requires exactly one TIME column in seconds (legacy TIME_S is accepted)")
    selected = [name for name in names if name != "TURN"]
    for name, dtype in zip(names, metadata.column_types, strict=True):
        if name in selected and np.dtype(dtype).kind not in "iuf":
            raise ValueError(f"Magnet ramping {name} must contain real numbers")
    frame = pd.read_csv(path,
                        sep=r"\s+",
                        names=names,
                        usecols=selected,
                        dtype=np.float64,
                        engine="c",
                        skiprows=metadata.non_data_lines,
                        float_precision="round_trip",
                        na_values=["nil"])
    times = frame.pop(time_names[0]).to_numpy()
    return times, {name: frame[name].to_numpy() for name in frame}, metadata.headers


@dataclass(frozen=True)
class _RampGrid:
    """Read-only dynamic component tables that share one physical time grid."""

    indices: tuple[int, ...]
    times: np.ndarray
    values: np.ndarray
    slopes: np.ndarray


def _time_upper_bound(times, reference, offset):
    """Locate an interval without subtracting the epoch from the entire table."""
    start, end = 0, len(times)
    while start < end:
        middle = (start + end) // 2
        if times[middle] - reference <= offset:
            start = middle + 1
        else:
            end = middle
    return start


@dataclass(frozen=True)
class MagnetRamp:
    """Owned programs; supplied components replace static integrated strengths.

    Exact constant columns bypass interpolation. Dynamic columns with identical
    time grids share one interval search for a complete bunch's slice schedule.
    """

    _programs: tuple
    sources: tuple[Path, ...]
    _time_range: tuple[float, float] | None = None
    components: tuple[tuple[int, bool], ...] = field(init=False)
    is_constant: bool = field(init=False)
    _constant_values: np.ndarray = field(init=False, repr=False, compare=False)
    _grids: tuple[_RampGrid, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        programs = tuple(
            (component, LinearProgram(program.values, program.times)) for component, program in sorted(self._programs, key=lambda item: item[0]))
        components = tuple(component for component, _ in programs)
        if not components or len(set(components)) != len(components):
            raise ValueError("Magnet ramping requires distinct components")
        constant_values = np.array([program.values[0] for _, program in programs], dtype=np.float64)
        groups = {}
        for index, (_, program) in enumerate(programs):
            if np.all(program.values == program.values[0]):
                continue
            groups.setdefault(program.times.tobytes(), []).append(index)
        grids = []
        for indices in groups.values():
            times = programs[indices[0]][1].times.copy()
            samples = np.column_stack([programs[index][1].values for index in indices])
            slopes = np.column_stack([programs[index][1].slopes for index in indices])
            for array in (times, samples, slopes):
                array.setflags(write=False)
            grids.append(_RampGrid(tuple(indices), times, samples, slopes))
        constant_values.setflags(write=False)
        time_range = self._time_range
        if time_range is None:
            time_range = (min(program.times[0] for _, program in programs), max(program.times[-1] for _, program in programs))
        object.__setattr__(self, "_programs", programs)
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "_time_range", tuple(float(value) for value in time_range))
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "is_constant", not grids)
        object.__setattr__(self, "_constant_values", constant_values)
        object.__setattr__(self, "_grids", tuple(grids))

    @property
    def max_order(self):
        return self.components[-1][0]

    @property
    def time_range(self):
        """First and last file sample times, including constant source columns."""
        return self._time_range

    def summary(self):
        """Describe the normalized integrated coefficients and extrapolation."""
        columns = ",".join(f"K{order}{'S' if skew else ''}L" for order, skew in self.components)
        start, end = self.time_range
        mode = "constant" if self.is_constant else "dynamic"
        return f"{mode}; components={columns}; time=[{start:.12g}, {end:.12g}] s; endpoint hold"

    def sample_values(self, reference, offsets):
        """Return float64 integrated coefficients shaped (offsets, components).

        Offsets are a finite one-dimensional sequence of physical seconds.
        Subtract the reference from table knots before searching; adding offsets
        to a large absolute epoch would erase a short magnet's flight time.
        """
        reference_array = np.asarray(reference)
        offsets = np.asarray(offsets)
        if (reference_array.ndim != 0 or reference_array.dtype.kind not in "iuf" or not np.isfinite(reference_array) or offsets.ndim != 1
                or offsets.dtype.kind not in "iuf" or not np.all(np.isfinite(offsets))):
            raise ValueError("Magnet ramping requires a finite scalar reference and finite one-dimensional time offsets")
        reference = float(reference_array)
        offsets = offsets.astype(np.float64, copy=False)
        result = np.broadcast_to(self._constant_values, (len(offsets), len(self.components))).copy()
        if not len(offsets):
            return result
        bounds = None
        for grid in self._grids:
            if len(offsets) == 1:
                index = max(0, _time_upper_bound(grid.times, reference, offsets[0]) - 1)
                dx = (reference - grid.times[index]) + offsets[0]
                slopes = 0. if (reference - grid.times[0]) + offsets[0] < 0. else grid.slopes[index]
                result[0, grid.indices] = grid.values[index] + slopes * dx
                continue
            start, end = 0, len(grid.times)
            if end > 256:
                if bounds is None:
                    bounds = float(np.min(offsets)), float(np.max(offsets))
                # A short magnet usually crosses only a few knots of a long ramp.
                start = max(0, _time_upper_bound(grid.times, reference, bounds[0]) - 1)
                end = min(end, _time_upper_bound(grid.times, reference, bounds[1]) + 1)
            index = np.clip(np.searchsorted(grid.times[start:end] - reference, offsets, side="right") + start - 1, 0, len(grid.times) - 1)
            dx = (reference - grid.times[index]) + offsets
            before = (reference - grid.times[0]) + offsets < 0.
            slopes = np.where(before[:, None], 0., grid.slopes[index])
            result[:, grid.indices] = grid.values[index] + slopes * dx[:, None]
        return result

    def values_at(self, reference, offset=0.):
        """Return integrated strengths at a physical reference time and offset."""
        sampled = self.sample_values(reference, [offset])[0]
        return {component: float(value) for component, value in zip(self.components, sampled, strict=True)}


def load_magnet_ramp(kwargs, *, length, order=None):
    """Load enabled programs once; K columns require a finite positive length.

    Coefficients are normalized to the current bunch reference momentum. This
    describes prescribed optics, not a physical magnetic-field waveform.
    """
    values = {str(k).lower(): v for k, v in kwargs.items()}
    if not values.get("is ramping", False):
        return None
    if not np.isfinite(length) or length < 0:
        raise ValueError("Magnet ramping requires a finite nonnegative element length")
    if order is not None and (isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 0):
        raise ValueError("Magnet order must be a nonnegative integer")
    sources = _ramping_sources(values, order=order)
    programs = {}
    time_ranges = []
    for path in sources:
        times, columns, headers = _read_ramping_table(path)
        times, columns = _validate_samples(times, columns, headers, order=order)
        time_ranges.append((float(times[0]), float(times[-1])))
        for name, samples in columns.items():
            component_order, skew, integrated = _strength_component(name)
            component = component_order, skew
            if component in programs:
                raise ValueError(f"Duplicate magnet ramping source for component {component}")
            if not integrated:
                if length <= const.eps:
                    raise ValueError(f"Thin magnets require integrated strength columns; cannot use {name} with zero length")
                samples = samples * length
            programs[component] = LinearProgram(samples, times)
    time_range = (min(start for start, _ in time_ranges), max(end for _, end in time_ranges))
    return MagnetRamp(tuple(programs.items()), sources, time_range)
