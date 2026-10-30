"""Measured transverse cloud responses about an explicitly prescribed beam train.

The stored kick already contains source charge, interaction length and witness
reference normalization. It is not a translation-invariant impedance wake.
"""

from dataclasses import InitVar, asdict, dataclass, field
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from PASS.utils.constants import const


def _host(values):
    return np.asarray(values.get() if hasattr(values, "get") else values, dtype=np.float64)


def _digest(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _coordinates(x, y):
    x, y = _host(x).copy(), _host(y).copy()
    if x.ndim != 1 or x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Response coordinates require matching finite one-dimensional x/y arrays")
    return x, y


@dataclass(frozen=True)
class ResponseSlice:
    """One finite saved beam interval and its signed physical normalization.

    x/y are source coordinates; witness_x/y define a fixed independent sampling
    distribution. Source-only displacements never move those witnesses.
    """

    time_start: float
    time_end: float
    turn: int
    bunch_id: int
    slice_index: int
    x: object
    y: object
    charge_per_length: float
    n_real_particles: float
    beta: float
    brho: float
    num_charge: float
    interaction_length: float
    witness_x: object = None
    witness_y: object = None

    def __post_init__(self):
        x, y = _coordinates(self.x, self.y)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        if (self.witness_x is None) != (self.witness_y is None):
            raise ValueError("Specify both witness coordinates or neither")
        wx, wy = (x, y) if self.witness_x is None else _coordinates(self.witness_x, self.witness_y)
        if not wx.size:
            wx, wy = np.zeros(1), np.zeros(1)
        object.__setattr__(self, "witness_x", wx.copy())
        object.__setattr__(self, "witness_y", wy.copy())
        values = (self.time_start, self.time_end, self.charge_per_length, self.n_real_particles, self.beta, self.brho, self.num_charge,
                  self.interaction_length)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Response slice scalars must be finite")
        if self.time_end <= self.time_start or not 0 < self.beta <= 1 or self.brho <= 0 or self.num_charge == 0:
            raise ValueError("Response slices require positive duration, physical beta, positive rigidity and nonzero charge")
        if self.n_real_particles < 0 or self.interaction_length < 0:
            raise ValueError("Response populations and interaction lengths must be nonnegative")
        if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0
                for value in (self.turn, self.bunch_id, self.slice_index)):
            raise ValueError("Response turn, bunch and slice identifiers must be nonnegative integers")
        for name in ("turn", "bunch_id", "slice_index"):
            object.__setattr__(self, name, int(getattr(self, name)))
        expected_charge = self.n_real_particles * self.num_charge * const.e
        measured_charge = self.line_charge * self.beta * const.c * (self.time_end - self.time_start)
        if not np.isclose(measured_charge, expected_charge, rtol=2e-6, atol=1e-30):
            raise ValueError("Response source charge disagrees with its particle count and physical slice width")

    @property
    def line_charge(self):
        return float(self.charge_per_length * self.x.size)

    @property
    def kick_factor(self):
        return math.copysign(1., self.num_charge) * self.interaction_length / (self.beta * const.c * self.brho)

    @property
    def centroid(self):
        return np.array([self.x.mean(), self.y.mean()]) if self.x.size else np.zeros(2)

    @property
    def profile(self):
        if not self.x.size:
            return np.zeros((2, 2))
        centered = np.stack((self.x, self.y)) - self.centroid[:, None]
        return centered @ centered.T / self.x.size

    def metadata(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name not in {"x", "y", "witness_x", "witness_y"}}


def capture_response_slices(beam, slice_set_name, turn, s, interaction_length, *, xp=np):
    """Capture actual PASS saved slices without changing beam or cloud state."""
    from .driver import build_slice_intervals
    events = build_slice_intervals(beam, slice_set_name, turn, s, xp, include_particles=True)
    result = []
    for event in events:
        if event.time_end == event.time_start:
            raise ValueError("Response capture requires finite explicit Slicer intervals for empty bunch groups")
        bunch = beam.bunches[event.bunch_index]
        indices = event.particle_indices
        result.append(
            ResponseSlice(event.time_start, event.time_end, int(turn), event.bunch_id, event.slice_index, beam.particles.x[indices],
                          beam.particles.y[indices], event.charge_per_length, event.n_real_particles, event.beam_beta, float(bunch.brho),
                          float(bunch.num_charge), float(interaction_length)))
    if not result:
        raise ValueError("Response extraction requires at least one finite saved slice")
    return result


def _validate_timeline(slices):
    if not slices or not all(isinstance(event, ResponseSlice) for event in slices):
        raise ValueError("Response timeline requires ResponseSlice records")
    keys = [(event.turn, event.bunch_id, event.slice_index) for event in slices]
    if len(set(keys)) != len(keys):
        raise ValueError("Response timeline contains duplicate turn/bunch/slice identifiers")
    for previous, current in zip(slices, slices[1:]):
        if current.time_start < previous.time_end or current.turn < previous.turn:
            raise ValueError("Response intervals must be ordered, nonoverlapping and have nondecreasing turns")


def _advance_response_interval(candidate, event, offset, probe=None):
    """Advance an owned cloud through one source with independent witnesses."""
    if candidate.time is None:
        candidate.advance_to(event.time_start)
    if event.time_start > candidate.time:
        candidate.advance_gap(event.time_start)
    xp = candidate.xp
    x, y = xp.asarray(event.x + offset[0]), xp.asarray(event.y + offset[1])
    wx, wy = (event.witness_x, event.witness_y) if probe is None else _coordinates(*probe)
    witness = xp.asarray(wx), xp.asarray(wy)
    if not bool(xp.all(candidate.chamber.contains(x, y, xp))) or not bool(xp.all(candidate.chamber.contains(*witness, xp))):
        raise ValueError("Response source/probe lies outside the conducting chamber")
    return candidate.advance_slice(event.time_end,
                                   n_real_particles=event.n_real_particles,
                                   line_charge=event.line_charge,
                                   beam_beta=event.beta,
                                   witness=witness,
                                   beam_x=x,
                                   beam_y=y,
                                   charge_per_length=event.charge_per_length)


def run_response_train(cloud, slices, *, source_offsets=None, probes=None, collect_diagnostics=True):
    """Direct BuildUpCloud adapter: fixed witnesses, shifted sources, physical gaps.

    The supplied cloud is never advanced. The returned kicks are numpy arrays
    on both CPU and GPU; extraction runs clones serially to share factorization.
    """
    _validate_timeline(slices)
    if cloud.configuration.mode != "coupled" or cloud.pic is None:
        raise ValueError("Cloud response extraction requires coupled PIC mode")
    offsets = np.zeros((len(slices), 2)) if source_offsets is None else _host(source_offsets)
    if offsets.shape != (len(slices), 2) or not np.all(np.isfinite(offsets)):
        raise ValueError("Source offsets must be a finite (n_slices, 2) array")
    if probes is not None and len(probes) != len(slices):
        raise ValueError("One probe distribution is required per response slice")
    candidate = cloud.clone()
    samples, histories = [], []
    try:
        if candidate.time is None:
            candidate.advance_to(slices[0].time_start)
        if candidate.time > slices[0].time_start:
            raise ValueError("Response source snapshot is later than the first prescribed interval")
        for index, event in enumerate(slices):
            diagnostics = _advance_response_interval(candidate, event, offsets[index], None if probes is None else probes[index])
            samples.append(event.kick_factor * np.stack((_host(diagnostics["mean_ex"]), _host(diagnostics["mean_ey"])), axis=-1))
            if collect_diagnostics:
                histories.append({name: float(value) for name, value in diagnostics.items() if np.isscalar(value)})
        summary = {"history": histories, "final_cloud_sha256": _digest(candidate.state.state_dict())} if collect_diagnostics else {}
    finally:
        candidate.close()
    return samples, summary


@dataclass
class CloudResponse:
    """Finite-history G+K model; optional periodic use needs measured validation."""

    slices: list
    grid_x: object
    grid_y: object
    baseline: object
    kernel: object
    metadata: dict = field(default_factory=dict)
    _copy_arrays: InitVar[bool] = True

    def __post_init__(self, _copy_arrays):
        self.slices = list(self.slices)
        _validate_timeline(self.slices)
        for name in ("grid_x", "grid_y", "baseline", "kernel"):
            values = _host(getattr(self, name))
            setattr(self, name, values.copy() if _copy_arrays else values)
        n_slices = len(self.slices)
        if any(axis.ndim != 1 or axis.size < 2 or not np.all(np.diff(axis) > 0) for axis in (self.grid_x, self.grid_y)):
            raise ValueError("Response probe grid axes must be increasing with at least two points")
        if self.baseline.shape != (n_slices, self.grid_y.size, self.grid_x.size, 2) or self.kernel.shape != (n_slices, n_slices, 2, 2):
            raise ValueError("Response G/K shapes disagree with the saved probe grid and timeline")
        if not all(np.all(np.isfinite(values)) for values in (self.grid_x, self.grid_y, self.baseline)):
            raise ValueError("Response grid, baseline and kernel must be finite")
        # Avoid two quadratic integer index arrays and a second half-kernel copy.
        for index, row in enumerate(self.kernel):
            if not np.all(np.isfinite(row)):
                raise ValueError("Response grid, baseline and kernel must be finite")
            if np.any(row[index + 1:] != 0):
                raise ValueError("Response kernel has a noncausal source-to-earlier-witness entry")
        self.metadata = deepcopy(self.metadata)
        json.dumps(self.metadata, allow_nan=False)
        self._chamber = None
        if "configuration" in self.metadata:
            from PASS.para.schema.electron_cloud import ElectronCloudConfiguration
            from .chamber import Chamber
            self._chamber = Chamber(ElectronCloudConfiguration.model_validate(self.metadata["configuration"]))

    def _sample_baseline_arrays(self, values, x, y, gx, gy, xp):
        """Return an uncommitted interpolation and device-side validity flags."""
        x, y = xp.asarray(x), xp.asarray(y)
        if x.shape != y.shape:
            raise ValueError("Response witnesses require finite paired x/y coordinates")
        valid = xp.stack(
            (xp.all(xp.isfinite(x) & xp.isfinite(y)), xp.all(self._chamber.contains(x, y, xp)) if self._chamber is not None else xp.asarray(True),
             xp.all((x >= self.grid_x[0]) & (x <= self.grid_x[-1]) & (y >= self.grid_y[0]) & (y <= self.grid_y[-1]))))
        ix = xp.clip(xp.searchsorted(gx, x, side="right") - 1, 0, gx.size - 2)
        iy = xp.clip(xp.searchsorted(gy, y, side="right") - 1, 0, gy.size - 2)
        tx, ty = (x - gx[ix]) / (gx[ix + 1] - gx[ix]), (y - gy[iy]) / (gy[iy + 1] - gy[iy])
        kick = ((1 - tx) * (1 - ty))[..., None] * values[iy, ix] + (tx * (1 - ty))[..., None] * values[iy, ix + 1] + \
            ((1 - tx) * ty)[..., None] * values[iy + 1, ix] + (tx * ty)[..., None] * values[iy + 1, ix + 1]
        return kick, valid

    @staticmethod
    def _validate_probe_status(valid):
        if not valid[0]:
            raise ValueError("Response witnesses require finite paired x/y coordinates")
        if not valid[1]:
            raise ValueError("Response witness lies outside the conducting chamber")
        if not valid[2]:
            raise ValueError("Response witness lies outside the calibrated G probe grid")

    def sample_baseline(self, index, x, y, *, xp=np):
        """Bilinear G interpolation without an additional focusing kick."""
        kick, valid = self._sample_baseline_arrays(xp.asarray(self.baseline[index]), x, y, xp.asarray(self.grid_x), xp.asarray(self.grid_y), xp)
        self._validate_probe_status(_host(valid))
        return kick

    def replay(self, offsets, *, witness_positions=None, xp=np):
        """Evaluate the full finite timeline; all charge/length factors are stored."""
        offsets = xp.asarray(offsets, dtype=xp.float64)
        if offsets.shape != (len(self.slices), 2) or not bool(xp.all(xp.isfinite(offsets))):
            raise ValueError("Response replay offsets require shape (n_slices, 2)")
        limit = float(self.metadata.get("maximum_centroid_displacement", 0))
        if limit <= 0 or bool(xp.any(xp.linalg.norm(offsets, axis=1) > limit * (1 + 1e-12))):
            raise ValueError("Response replay displacement exceeds its calibrated linear domain")
        dynamic = xp.einsum("jiab,ib->ja", xp.asarray(self.kernel), offsets)
        positions = [(event.witness_x, event.witness_y) for event in self.slices] if witness_positions is None else witness_positions
        if len(positions) != len(self.slices):
            raise ValueError("Response replay needs one witness distribution per interval")
        gx, gy, baseline = xp.asarray(self.grid_x), xp.asarray(self.grid_y), xp.asarray(self.baseline)
        kicks, statuses = [], []
        for index, (x, y) in enumerate(positions):
            kick, valid = self._sample_baseline_arrays(baseline[index], x, y, gx, gy, xp)
            kicks.append(kick + dynamic[index])
            statuses.append(valid)
        for valid in _host(xp.stack(statuses)):
            self._validate_probe_status(valid)
        return kicks


def extract_response(cloud,
                     slices,
                     *,
                     epsilon,
                     grid_x=None,
                     grid_y=None,
                     grid_points=17,
                     linearity_tolerance=0.05,
                     derivative_atol=1e-12,
                     profile_tolerance=0.05):
    """Measure G and all four K components, with an epsilon/2 linearity check.

    epsilon is also the maximum permitted centroid displacement. Independent
    convergence in probe/PIC grids, time steps and macro counts is still needed.
    """
    slices = list(slices)
    _validate_timeline(slices)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Response epsilon must be positive and finite")
    if not all(math.isfinite(value) and value >= 0 for value in (linearity_tolerance, derivative_atol, profile_tolerance)):
        raise ValueError("Response tolerances must be finite and nonnegative")
    if (grid_x is None) != (grid_y is None):
        raise ValueError("Specify both response probe grid axes")
    if grid_x is None:
        if isinstance(grid_points, bool) or not isinstance(grid_points, int) or grid_points < 3:
            raise ValueError("Response grid_points must be an integer of at least three")
        all_x = np.concatenate([event.witness_x for event in slices])
        all_y = np.concatenate([event.witness_y for event in slices])
        width_x = max(float(np.max(np.abs(all_x))) + 2 * epsilon, cloud.radius / 10)
        width_y = max(float(np.max(np.abs(all_y))) + 2 * epsilon, cloud.radius / 10)
        grid_x, grid_y = np.linspace(-width_x, width_x, grid_points), np.linspace(-width_y, width_y, grid_points)
    grid_x, grid_y = _host(grid_x).copy(), _host(grid_y).copy()
    if any(axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0) for axis in (grid_x, grid_y)):
        raise ValueError("Response probe axes must be finite, one-dimensional and strictly increasing")
    gx, gy = np.meshgrid(grid_x, grid_y)
    if not np.all(cloud.chamber.contains(gx, gy)):
        raise ValueError("All response probe grid points must lie strictly inside the chamber")
    n_grid = gx.size
    probes = [(np.concatenate((gx.ravel(), event.witness_x)), np.concatenate((gy.ravel(), event.witness_y))) for event in slices]
    source_state = cloud.state.state_dict()
    base_samples, diagnostics = run_response_train(cloud, slices, probes=probes)
    baseline = np.stack([values[:n_grid].reshape(gy.shape + (2, )) for values in base_samples])
    fixed_baseline = np.stack([values[n_grid:].mean(axis=0) for values in base_samples])
    n_slices = len(slices)
    kernel = np.zeros((n_slices, n_slices, 2, 2))
    errors = np.zeros((n_slices, 2))
    even_errors = np.zeros((n_slices, 2))
    prefix = cloud.clone()
    try:
        for source in range(n_slices):
            remaining = slices[source:]
            for axis in range(2):
                derivatives = []
                for amplitude in (epsilon, epsilon / 2):
                    shifts = np.zeros((len(remaining), 2))
                    shifts[0, axis] = amplitude
                    plus, _ = run_response_train(prefix, remaining, source_offsets=shifts, collect_diagnostics=False)
                    minus, _ = run_response_train(prefix, remaining, source_offsets=-shifts, collect_diagnostics=False)
                    plus_mean = np.stack([values.mean(axis=0) for values in plus])
                    minus_mean = np.stack([values.mean(axis=0) for values in minus])
                    derivative = (plus_mean - minus_mean) / (2 * amplitude)
                    even_residual = (plus_mean + minus_mean) / 2 - fixed_baseline[source:]
                    even_error = np.linalg.norm(even_residual)
                    odd_scale = amplitude * np.linalg.norm(derivative)
                    even_errors[source, axis] = max(even_errors[source, axis], even_error / max(odd_scale, amplitude * derivative_atol,
                                                                                                np.finfo(float).tiny))
                    if even_error > amplitude * derivative_atol + linearity_tolerance * odd_scale:
                        raise ValueError(f"Response has a nonlinear even source response: source={source}, axis={axis}, amplitude={amplitude}, "
                                         f"even_residual={even_residual}, error={even_error}, "
                                         f"allowed={amplitude * derivative_atol + linearity_tolerance * odd_scale}")
                    derivatives.append(derivative)
                coarse, fine = derivatives
                scale, error = np.linalg.norm(fine), np.linalg.norm(fine - coarse)
                errors[source, axis] = error / max(scale, derivative_atol, np.finfo(float).tiny)
                if error > derivative_atol + linearity_tolerance * scale:
                    raise ValueError(
                        f"Response is outside its linear domain: source={source}, axis={axis}, coarse={coarse}, fine={fine}, error={error}, "
                        f"allowed={derivative_atol + linearity_tolerance * scale}")
                kernel[source:, source, :, axis] = fine
            if source + 1 < n_slices:
                # Advance one unperturbed prefix, retaining its physical state and RNG.
                # Perturbations cannot alter samples before their source interval.
                _advance_response_interval(prefix, slices[source], (0., 0.))
    finally:
        prefix.close()
    metadata = dict(format_version=1,
                    coordinate="continuous_z_rel_physical_time",
                    momentum="P_transverse/P0",
                    kernel_unit="1/m",
                    baseline_unit="1",
                    factors_included=["source_charge", "interaction_length", "signed_witness_charge_over_beta_c_p0"],
                    configuration=cloud.configuration.model_dump(mode="json"),
                    configuration_sha256=_digest(cloud.configuration.model_dump(mode="json")),
                    initial_cloud_sha256=_digest(source_state),
                    source_length=cloud.source_length,
                    initial_cloud_time=cloud.time,
                    initial_cloud_rng_sha256=_digest(source_state["rng_state"]),
                    backend=cloud.backend,
                    maximum_centroid_displacement=float(epsilon),
                    linearity_tolerance=float(linearity_tolerance),
                    derivative_atol=float(derivative_atol),
                    linearity_verified=True,
                    linearity_method="central_difference_and_even_residual_v1",
                    extraction_method="causal_prefix_reuse_v1",
                    linearity_relative_error=errors.tolist(),
                    linearity_even_relative_error=even_errors.tolist(),
                    profile_tolerance=float(profile_tolerance),
                    diagnostics=diagnostics,
                    initial_state_scope="local physical cloud snapshot, not a simulation checkpoint")
    return CloudResponse(slices, grid_x, grid_y, baseline, kernel, metadata, _copy_arrays=False)


def make_periodic_response(response, *, memory_periods, relative_tolerance=0.02, kick_atol=1e-15, kernel_atol=1e-12):
    """Authorize periodic replay only after comparing repeated measured periods.

    A period is one recorded turn. At least memory_periods+3 turns are needed;
    the final two complete histories and one omitted lag establish agreement.
    """
    descriptor = _periodic_descriptor(response,
                                      memory_periods=memory_periods,
                                      relative_tolerance=relative_tolerance,
                                      kick_atol=kick_atol,
                                      kernel_atol=kernel_atol)
    metadata = deepcopy(response.metadata)
    metadata["periodic"] = descriptor
    return CloudResponse(response.slices, response.grid_x, response.grid_y, response.baseline, response.kernel, metadata)


def _periodic_descriptor(response, *, memory_periods, relative_tolerance, kick_atol, kernel_atol):
    """Validate repeated periods without copying the full response kernel."""
    if isinstance(memory_periods, bool) or not isinstance(memory_periods, int) or memory_periods < 0:
        raise ValueError("memory_periods must be a nonnegative integer")
    if not all(math.isfinite(value) and value >= 0 for value in (relative_tolerance, kick_atol, kernel_atol)):
        raise ValueError("Periodic response tolerances must be finite and nonnegative")
    turns = sorted(set(event.turn for event in response.slices))
    rows = [[i for i, event in enumerate(response.slices) if event.turn == turn] for turn in turns]
    if len(turns) < memory_periods + 3 or turns != list(range(turns[0], turns[-1] + 1)) or len({len(row) for row in rows}) != 1:
        raise ValueError("Periodic calibration requires consecutive equal-size turns and memory_periods+3 measured periods")
    width = len(rows[0])
    period = response.slices[rows[-1][0]].time_start - response.slices[rows[-2][0]].time_start
    if period <= 0:
        raise ValueError("Periodic response requires a positive physical period")
    for previous, current in zip(rows, rows[1:]):
        for left, right in zip(previous, current):
            a, b = response.slices[left], response.slices[right]
            for name in ("bunch_id", "slice_index", "charge_per_length", "n_real_particles", "beta", "brho", "num_charge", "interaction_length"):
                if not np.isclose(getattr(a, name), getattr(b, name), rtol=1e-9, atol=0):
                    raise ValueError("Periodic response source/reference parameters differ between turns")
            if not np.allclose([b.time_start - a.time_start, b.time_end - a.time_end], period, rtol=1e-8, atol=0):
                raise ValueError("Periodic response physical timing differs between turns")
            if a.x.shape != b.x.shape or not np.array_equal(a.x, b.x) or not np.array_equal(a.y, b.y):
                raise ValueError("Periodic calibration requires the same unperturbed source distribution each turn")
            if not np.array_equal(a.witness_x, b.witness_x) or not np.array_equal(a.witness_y, b.witness_y):
                raise ValueError("Periodic calibration requires the same fixed witness distribution each turn")
    errors = {}
    base_a, base_b = response.baseline[rows[-2]], response.baseline[rows[-1]]
    error = np.linalg.norm(base_b - base_a)
    if error > kick_atol + relative_tolerance * np.linalg.norm(base_b):
        raise ValueError("Cloud baseline has not reached a periodic state")
    errors["baseline_absolute"] = float(error)
    for lag in range(memory_periods + 1):
        a = response.kernel[np.ix_(rows[-2], rows[-2 - lag])]
        b = response.kernel[np.ix_(rows[-1], rows[-1 - lag])]
        error = np.linalg.norm(b - a)
        if error > kernel_atol + relative_tolerance * np.linalg.norm(b):
            raise ValueError(f"Cloud response lag {lag} is not periodic within tolerance")
        errors[f"lag_{lag}_absolute"] = float(error)
    old = response.kernel[np.ix_(rows[-1], list(range(rows[-1 - memory_periods][0])))]
    kept = response.kernel[np.ix_(rows[-1], list(range(rows[-1 - memory_periods][0], len(response.slices))))]
    if np.linalg.norm(old) > kernel_atol + relative_tolerance * np.linalg.norm(kept):
        raise ValueError("Response memory truncation is not converged; retain more periods")
    return dict(period=period,
                events_per_period=width,
                template_start=rows[-1][0],
                memory_periods=memory_periods,
                errors=errors,
                omitted_kernel_norm=float(np.linalg.norm(old)),
                relative_tolerance=relative_tolerance,
                kick_atol=kick_atol,
                kernel_atol=kernel_atol,
                validated=True)


def write_response(path, response):
    """Create a portable response file; never overwrite an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as output:
        output.attrs["format"] = "PASS-electron-cloud-response-1"
        output.attrs["metadata_json"] = json.dumps(response.metadata, allow_nan=False, sort_keys=True)
        for name, unit in (("grid_x", "m"), ("grid_y", "m"), ("baseline", "1"), ("kernel", "1/m")):
            output.create_dataset(name, data=getattr(response, name), compression="gzip").attrs["unit"] = unit
        group = output.create_group("slices")
        for index, event in enumerate(response.slices):
            entry = group.create_group(str(index))
            entry.attrs["metadata_json"] = json.dumps(event.metadata(), allow_nan=False, sort_keys=True)
            for name in ("x", "y", "witness_x", "witness_y"):
                entry.create_dataset(name, data=getattr(event, name)).attrs["unit"] = "m"
    return path


def read_response(path, *, configuration=None):
    """Validate shape, units, physical metadata and optional cloud compatibility."""
    with h5py.File(path, "r") as source:
        if source.attrs.get("format") != "PASS-electron-cloud-response-1":
            raise ValueError("Unsupported electron-cloud response file format")
        metadata = json.loads(source.attrs["metadata_json"])
        if metadata.get("momentum") != "P_transverse/P0" or metadata.get("kernel_unit") != "1/m" or not metadata.get("linearity_verified"):
            raise ValueError("Response normalization or linearity certification is missing")
        if metadata.get("linearity_method") != "central_difference_and_even_residual_v1":
            raise ValueError("Response calibration must validate both odd and even source response")
        if metadata.get("coordinate") != "continuous_z_rel_physical_time":
            raise ValueError("Response physical-time convention is incompatible")
        expected_factors = ["source_charge", "interaction_length", "signed_witness_charge_over_beta_c_p0"]
        if metadata.get("factors_included") != expected_factors or metadata.get("configuration_sha256") != _digest(metadata.get("configuration")):
            raise ValueError("Response included factors/configuration digest is incompatible")
        if configuration is not None and _digest(configuration.model_dump(mode="json")) != metadata["configuration_sha256"]:
            raise ValueError("Response cloud configuration does not match its calibration")
        arrays = {}
        for name, unit in (("grid_x", "m"), ("grid_y", "m"), ("baseline", "1"), ("kernel", "1/m")):
            if source[name].attrs.get("unit") != unit:
                raise ValueError(f"Response {name} units are incompatible")
            arrays[name] = np.asarray(source[name])
        group = source["slices"]
        if set(group) != {str(index) for index in range(len(group))}:
            raise ValueError("Response slice dataset indices are not contiguous")
        slices = []
        for index in range(len(group)):
            entry = group[str(index)]
            values = json.loads(entry.attrs["metadata_json"])
            for name in ("x", "y", "witness_x", "witness_y"):
                if entry[name].attrs.get("unit") != "m":
                    raise ValueError("Response slice coordinates must use metres")
                values[name] = np.asarray(entry[name])
            slices.append(ResponseSlice(**values))
    response = CloudResponse(slices, metadata=metadata, _copy_arrays=False, **arrays)
    if metadata.get("format_version") != 1 or metadata.get("baseline_unit") != "1":
        raise ValueError("Response metadata version or baseline units are incompatible")
    limit = metadata.get("maximum_centroid_displacement", 0)
    if not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
        raise ValueError("Response file has no finite positive displacement domain")
    for name in ("profile_tolerance", "linearity_tolerance", "derivative_atol"):
        value = metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Response {name} must be finite and nonnegative")
    errors = np.asarray(metadata.get("linearity_relative_error"), dtype=float)
    if errors.shape != (len(slices), 2) or not np.all(np.isfinite(errors)) or np.any(errors < 0):
        raise ValueError("Response linearity diagnostics do not match its source timeline")
    even_errors = np.asarray(metadata.get("linearity_even_relative_error"), dtype=float)
    if even_errors.shape != (len(slices), 2) or not np.all(np.isfinite(even_errors)) or np.any(even_errors < 0):
        raise ValueError("Response even-linearity diagnostics do not match its source timeline")
    if "periodic" in metadata:
        descriptor = metadata["periodic"]
        verified = _periodic_descriptor(response,
                                        memory_periods=descriptor["memory_periods"],
                                        relative_tolerance=descriptor["relative_tolerance"],
                                        kick_atol=descriptor["kick_atol"],
                                        kernel_atol=descriptor["kernel_atol"])
        if verified != descriptor:
            raise ValueError("Response periodic descriptor disagrees with its measured arrays")
    return response


@dataclass(frozen=True)
class InstabilityEstimate:
    """Growth and frequency per unit of the supplied physical time or turn axis."""

    growth_rate: float
    growth_standard_error: float
    frequency: float
    frequency_standard_error: float
    log_amplitude_residual_rms: float
    phase_residual_rms: float
    fit_start: float
    fit_end: float
    sample_count: int


def _linear_fit(x, y):
    centered = x - x.mean()
    denominator = np.dot(centered, centered)
    if denominator <= 0:
        raise ValueError("Instability fit requires distinct times")
    mean_y = y.mean()
    slope = np.dot(centered, y - mean_y) / denominator
    residual = y - (mean_y + slope * centered)
    variance = np.dot(residual, residual) / (len(x) - 2)
    return float(slope), float(math.sqrt(max(variance, 0) / denominator)), float(np.sqrt(np.mean(residual**2)))


def analyze_instability(times, signal, *, fit_start=None, fit_end=None, minimum_amplitude=0.0, edge_fraction=0.1):
    """Fit log envelope and unwrapped phase over an explicit time window.

    Prefer a complex Courant-Snyder/modal signal. A real series is converted
    with a Hilbert transform and requires uniform sampling; its end fraction is
    discarded by default. Fit errors are regression errors, not seed uncertainty.
    """
    times, signal = np.asarray(times, dtype=float), np.asarray(signal)
    if times.ndim != 1 or signal.shape != times.shape or times.size < 8 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(signal)):
        raise ValueError("Instability analysis requires at least eight finite paired time/signal samples")
    if np.any(np.diff(times) <= 0) or not math.isfinite(minimum_amplitude) or minimum_amplitude < 0 or not 0 <= edge_fraction < .5:
        raise ValueError("Invalid instability time axis, amplitude threshold or edge fraction")
    if not np.iscomplexobj(signal):
        from scipy.signal import hilbert
        if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-8, atol=0):
            raise ValueError("Real-signal Hilbert analysis requires uniform sampling")
        signal = hilbert(signal.astype(float) - np.mean(signal))
        edge = int(math.ceil(edge_fraction * times.size))
    else:
        edge = 0
    phase, amplitude = np.unwrap(np.angle(signal)), np.abs(signal)
    mask = amplitude > max(minimum_amplitude, np.finfo(float).tiny)
    if edge:
        mask[:edge] = False
        mask[-edge:] = False
    if fit_start is not None:
        mask &= times >= fit_start
    if fit_end is not None:
        mask &= times <= fit_end
    if np.count_nonzero(mask) < 8:
        raise ValueError("Insufficient nonzero-amplitude samples in the instability fit window")
    x = times[mask]
    growth, growth_error, amplitude_residual = _linear_fit(x, np.log(amplitude[mask]))
    omega, omega_error, phase_residual = _linear_fit(x, phase[mask])
    return InstabilityEstimate(growth, growth_error, omega / (2 * np.pi), omega_error / (2 * np.pi), amplitude_residual, phase_residual, float(x[0]),
                               float(x[-1]), int(x.size))


def analyze_slice_modes(times, centroids, mode_shapes, *, slice_weights=None, **fit_kwargs):
    """Project arbitrary longitudinal mode shapes before measuring growth.

    Rows of centroids are times; columns are saved slices. Nonuniform physical
    slices need explicitly chosen shapes/weights, not an implicit slice-index FFT.
    """
    values, shapes = np.asarray(centroids), np.asarray(mode_shapes)
    if values.ndim != 2 or shapes.ndim != 2 or values.shape[1] != shapes.shape[1]:
        raise ValueError("Mode analysis needs (n_times,n_slices) centroids and (n_modes,n_slices) shapes")
    weights = np.ones(values.shape[1]) if slice_weights is None else np.asarray(slice_weights, dtype=float)
    if weights.shape != (values.shape[1], ) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Mode weights must be finite nonnegative values with nonzero total")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(shapes)):
        raise ValueError("Centroids and mode shapes must be finite")
    normalization = np.sum(weights * np.abs(shapes)**2, axis=1)
    if np.any(normalization <= 0):
        raise ValueError("Each longitudinal mode needs nonzero weighted norm")
    amplitudes = values @ (weights[None, :] * shapes.conj()).T / normalization[None, :]
    estimates = [analyze_instability(times, amplitudes[:, index], **fit_kwargs) for index in range(shapes.shape[0])]
    return amplitudes, estimates


def scan_instability(parameters, run_case, *, seeds, confidence=0.95, growth_threshold=0.0, fit_kwargs=None):
    """Run cases serially and combine seed scatter with fit uncertainty.

    run_case(parameter, seed) returns (times, signal), using either the actual
    coupled simulation or the registered ElectronCloudResponse sequence command.
    A scan never rescales a response calibrated at another beam/cloud setting.
    """
    parameters, seeds = list(parameters), list(seeds)
    if not parameters or not seeds or not all(math.isfinite(value) for value in parameters):
        raise ValueError("Instability scan requires finite parameters and at least one seed")
    if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Instability scan seeds must be distinct integers")
    if not 0 < confidence < 1 or not math.isfinite(growth_threshold):
        raise ValueError("Instability confidence and growth threshold are invalid")
    critical = None
    if len(seeds) > 1:
        from scipy.stats import t as student_t
        critical = float(student_t.ppf((1 + confidence) / 2, len(seeds) - 1))
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    rows = []
    for parameter in parameters:
        estimates = [analyze_instability(*run_case(parameter, seed), **fit_kwargs) for seed in seeds]
        rates = np.array([estimate.growth_rate for estimate in estimates])
        mean = float(rates.mean())
        if len(seeds) < 2:
            # One deterministic realization cannot establish stochastic coverage.
            lower, upper, status = None, None, "unresolved"
        else:
            variance = np.var(rates, ddof=1) / len(seeds)
            variance += sum(estimate.growth_standard_error**2 for estimate in estimates) / len(seeds)**2
            half_width = float(critical * np.sqrt(variance))
            lower, upper = mean - half_width, mean + half_width
            status = "unstable" if lower > growth_threshold else "stable" if upper <= growth_threshold else "unresolved"
        rows.append(
            dict(parameter=float(parameter),
                 growth_rate=mean,
                 confidence_low=lower,
                 confidence_high=upper,
                 confidence=confidence,
                 growth_threshold=growth_threshold,
                 status=status,
                 seeds=[int(seed) for seed in seeds],
                 estimates=[asdict(estimate) for estimate in estimates]))
    return rows


def bracket_instability_threshold(lower,
                                  upper,
                                  run_case,
                                  *,
                                  seeds,
                                  parameter_tolerance,
                                  maximum_iterations=20,
                                  confidence=0.95,
                                  growth_threshold=0.0,
                                  fit_kwargs=None):
    """Bracket a locally monotone stable-to-unstable transition with uncertainty.

    An unresolved midpoint returns the current bracket; it is never silently
    classified. The caller must establish monotonicity with a wider scan first.
    """
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper and math.isfinite(parameter_tolerance) and parameter_tolerance > 0):
        raise ValueError("Threshold search needs finite ordered bounds and positive parameter tolerance")
    if isinstance(maximum_iterations, bool) or not isinstance(maximum_iterations, int) or maximum_iterations < 1:
        raise ValueError("Threshold maximum_iterations must be a positive integer")
    seeds = list(seeds)
    options = dict(seeds=seeds, confidence=confidence, growth_threshold=growth_threshold, fit_kwargs=fit_kwargs)
    rows = scan_instability([lower, upper], run_case, **options)
    if [row["status"] for row in rows] != ["stable", "unstable"]:
        raise ValueError("Threshold bounds are not independently classified stable and unstable")
    reason = "iteration_limit"
    for _ in range(maximum_iterations):
        if upper - lower <= parameter_tolerance:
            reason = "parameter_tolerance"
            break
        midpoint = lower / 2 + upper / 2
        row = scan_instability([midpoint], run_case, **options)[0]
        rows.append(row)
        if row["status"] == "unresolved":
            reason = "statistical_uncertainty"
            break
        if row["status"] == "stable":
            lower = midpoint
        else:
            upper = midpoint
    return dict(lower=float(lower),
                upper=float(upper),
                stopping_reason=reason,
                samples=rows,
                assumption="locally monotone transition; each case independently calibrated")
