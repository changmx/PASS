"""Source preparation and causal six-dimensional slice-pair collisions.

All per-IP resources live on the coordinator instance. CPU and GPU follow the
same immutable pre-kick source snapshots; there is no full six-coordinate copy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from PASS.commands.collision.hourglass import AnalyticSource, PICSource
from PASS.commands.solver.analytic import PotentialQuadrature
from PASS.commands.solver.field_result import launch_gpu_kernel
from PASS.commands.solver.pic import GridGeometry, build_pic_resources, build_pic_resources_gpu
from PASS.utils.constants import const
from PASS.utils.coordinates import delta_to_eta, eta_to_delta


@dataclass
class CollisionResources:
    xp: object
    dtype: object
    pic: dict = field(default_factory=dict)
    quadrature: object = field(init=False)
    error_flags: object = field(init=False)

    def __post_init__(self):
        self.dtype = np.dtype(self.dtype)
        self.quadrature = PotentialQuadrature(self.xp, self.dtype)
        self.error_flags = self.xp.zeros((), dtype=self.xp.int32)

    def get_pic(self, beam_id, source, reference_length):
        grid = GridGeometry(source.nx, source.ny, -source.grid_half_width_x, source.grid_half_width_x, -source.grid_half_width_y,
                            source.grid_half_width_y)
        device = None if self.xp is np else self.xp.cuda.runtime.getDevice()
        stream = None if self.xp is np else self.xp.cuda.get_current_stream().ptr
        key = (beam_id, grid, self.dtype.str, reference_length, source.deposition_method, device, stream)
        if key not in self.pic:
            builder = build_pic_resources if self.xp is np else build_pic_resources_gpu
            self.pic[key] = builder(grid, field_solver="fft_free_space", dtype=self.dtype, potential_reference_length=reference_length)
        return self.pic[key]

    def begin(self):
        self.error_flags.fill(0)
        self.quadrature.error_flags.fill(0)

    def validate(self):
        if bool(self.error_flags):
            raise ValueError("BeamBeam grid coverage or finite-state validation failed")
        if bool(self.quadrature.error_flags):
            raise ValueError("BeamBeam covariance or analytic quadrature convergence check failed")

    def close(self):
        for resources in self.pic.values():
            if hasattr(resources, "close"):
                resources.close()
        self.pic.clear()


def _resources(sim, configuration, p):
    owner = getattr(sim, "collision", sim)
    if not hasattr(owner, "resources"):
        owner.resources = {}
    event = getattr(owner, "current_event", None)
    name = event[0] if event else id(configuration)
    device = None if p.xp is np else p.xp.cuda.runtime.getDevice()
    stream = None if p.xp is np else p.xp.cuda.get_current_stream().ptr
    key = ("interaction", name, np.dtype(p.x.dtype).str, device, stream)
    if key not in owner.resources:
        owner.resources[key] = CollisionResources(p.xp, p.x.dtype)
    return owner.resources[key]


def _common_sign(configuration, beam_id, bunch):
    return -1 if configuration.beams[1] == beam_id and not getattr(bunch, "collision_frame", None) else 1


def _members(sim, beam, bunch, source):
    """One compact initial schedule transfer; particle indices stay on device."""
    p, xp = beam.particles, beam.particles.xp
    start, end = int(bunch.start_idx), int(bunch.end_idx)
    slice_set = bunch.slice_sets[source.slice_set]
    if slice_set.slice_id is None or len(slice_set.slice_id) != end - start:
        raise ValueError("BeamBeam requires a current bunch-local explicit SliceSet")
    owner = getattr(sim, "collision", None)
    event = getattr(owner, "current_event", None)
    if event is not None:
        if slice_set.purpose != "beam_beam" or slice_set.configuration_id != event[0] or slice_set.valid_turn != event[1]:
            raise ValueError("BeamBeam requires its own current-turn beam_beam SliceSet")
        frame = getattr(bunch, "collision_frame", None)
        if slice_set.frame != ("collision" if frame else "lab") or slice_set.coordinate != ("collision_z" if frame else "z_rel"):
            raise ValueError("BeamBeam SliceSet frame does not match its current particle representation")
        positions = getattr(owner, "current_positions", {})
        if beam.beam_id in positions and not np.isclose(slice_set.valid_s, positions[beam.beam_id], rtol=0, atol=const.eps):
            raise ValueError("BeamBeam requires an explicit Slicer at this IP position")
        if slice_set.reference_time != bunch.t0 or slice_set.reference_beta != bunch.beta:
            raise ValueError("BeamBeam SliceSet reference changed after slicing")
        generation = getattr(slice_set, "observation_generation", 0)
        if getattr(slice_set, "_beam_beam_consumed_generation", None) == generation:
            raise ValueError("BeamBeam requires a new explicit Slicer execution for each encounter")
    sid = xp.asarray(slice_set.slice_id)
    alive = p.tag[start:end] > 0
    invalid = (sid < 0) | (sid >= slice_set.num_slices)
    for name in ("x", "px", "y", "py", "z", "dp"):
        invalid |= ~xp.isfinite(getattr(p, name)[start:end])
    momentum_ratio = 1 + p.dp[start:end]
    invalid |= (momentum_ratio <= 0) | (momentum_ratio**2 <= p.px[start:end]**2 + p.py[start:end]**2)
    if bool(xp.any(alive & invalid)):
        raise ValueError("Every live collision particle must have a valid SliceSet member and physical finite coordinates")
    local = xp.flatnonzero((p.tag[start:end] > 0) & (sid >= 0) & (sid < slice_set.num_slices))
    if not local.size:
        # CuPy bincount reduces max(indices) even with minlength specified.
        # An empty live set has no encounters and needs no device reduction.
        return [local] * slice_set.num_slices, np.zeros(slice_set.num_slices), np.full((slice_set.num_slices, 2), np.nan)
    order = local[xp.argsort(sid[local], kind="stable")]
    counts = xp.bincount(sid[local], minlength=slice_set.num_slices)
    sums = xp.bincount(sid[local], weights=p.z[start:end][local], minlength=slice_set.num_slices)
    table = slice_set.slice_table
    if table is None or any(name not in table for name in ("z_particle_min", "z_particle_max", "macro_count")):
        raise ValueError("BeamBeam requires a new Slicer result with actual particle endpoints")
    lower, upper = xp.asarray(table["z_particle_min"]), xp.asarray(table["z_particle_max"])
    saved_counts = xp.asarray(table["macro_count"])
    if lower.shape != (slice_set.num_slices, ) or upper.shape != lower.shape or saved_counts.shape != lower.shape:
        raise ValueError("BeamBeam actual particle endpoints do not match the SliceSet")
    schedule = xp.stack((counts, sums / xp.maximum(counts, 1), lower, upper, saved_counts), axis=-1)
    schedule = np.asarray(schedule) if xp is np else xp.asnumpy(schedule)
    if np.any(schedule[:, 0] != schedule[:, 4]):
        raise ValueError("BeamBeam live membership changed after slicing; rerun Slicer before the encounter")
    occupied = schedule[:, 0] > 0
    if np.any(~np.isfinite(schedule[occupied, 2:4])) or np.any(schedule[occupied, 3] < schedule[occupied, 2]):
        raise ValueError("BeamBeam requires finite ordered actual particle endpoints for each occupied slice")
    offsets = np.r_[0, np.cumsum(schedule[:, 0].astype(np.int64))]
    return [order[offsets[i]:offsets[i + 1]] + start for i in range(slice_set.num_slices)], schedule[:, 1], schedule[:, 2:4]


def _frozen_moments(sim, configuration, source, beam, bunch, slice_index, slice_center):
    """Prescribed Twiss closure at the ordinary IP, then common-frame rotation.

    Projected sizes include dispersion. Subtract its position covariance first.
    Conditional betatron slopes use rotated alpha/beta; independent angular
    noise uses rotated 1/beta. Add the prescribed energy-spread outer product
    once. At zero dispersion this rotates an uncoupled 4D Twiss distribution.
    Crossing uses the design-orbit linear conditional thin-slice approximation.
    """
    from PASS.commands.element.crab_cavity import resolve_ip_optics

    parameters = source.slice_parameters.get(slice_index, source.frozen_parameters)
    optics = resolve_ip_optics(sim, beam.beam_id, source.frozen_optics_reference) if source.frozen_optics_reference else None
    gaussian = source.solver.startswith("gaussian")
    factor = 1 if gaussian else np.sqrt(6) if source.solver.startswith("parabolic") else 2
    if "_round_" in source.solver:
        widths = np.array([parameters.sigma if gaussian else parameters.radius] * 2) / factor
    else:
        widths = np.array([parameters.sigma_x, parameters.sigma_y] if gaussian else [parameters.a, parameters.b]) / factor
    angle = parameters.angle
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dispersion = np.array([optics["dx"], optics["dpx"], optics["dy"], optics["dpy"]]) if optics else np.zeros(4)
    position_covariance = rotation @ np.diag(widths**2) @ rotation.T
    betatron = position_covariance - parameters.sigma_delta**2 * np.outer(dispersion[[0, 2]], dispersion[[0, 2]])
    if np.linalg.eigvalsh(betatron).min() <= 0:
        raise ValueError("Frozen projected dimensions must exceed their prescribed dispersion contribution")
    betas = np.array([optics["beta_x"], optics["beta_y"]]) if optics else np.ones(2)
    alphas = np.array([optics["alpha_x"], optics["alpha_y"]]) if optics else np.zeros(2)
    alpha_matrix = rotation @ np.diag(alphas / betas) @ rotation.T
    beta_matrix = rotation @ np.diag(1 / betas) @ rotation.T if optics else np.zeros((2, 2))
    qp = -betatron @ alpha_matrix
    pp = alpha_matrix @ betatron @ alpha_matrix + beta_matrix @ betatron @ beta_matrix
    covariance = np.empty((4, 4))
    covariance[np.ix_([0, 2], [0, 2])] = betatron
    covariance[np.ix_([0, 2], [1, 3])] = qp
    covariance[np.ix_([1, 3], [0, 2])] = qp.T
    covariance[np.ix_([1, 3], [1, 3])] = pp
    covariance += parameters.sigma_delta**2 * np.outer(dispersion, dispersion)
    mean = np.array([parameters.center_x, source.source_center_slopes[0], parameters.center_y, source.source_center_slopes[1]])
    side = configuration.beams.index(beam.beam_id)
    if getattr(bunch, "collision_frame", None):
        plane = configuration.crossing_plane * (-1 if side else 1)
        cosine, sine = np.cos(plane), np.sin(plane)
        rotation = np.array([[cosine, sine], [-sine, cosine]])
        transform = np.eye(4)
        transform[np.ix_([0, 2], [0, 2])] = rotation
        phi = configuration.full_crossing_angle / 2 * (-1 if side else 1)
        transform[np.ix_([1, 3], [1, 3])] = rotation / np.cos(phi)
        mean = transform @ mean
        mean[0] += slice_center * np.sin(phi)
        covariance = transform @ covariance @ transform.T
    if side:
        signs = np.array([-1, -1, 1, 1])
        mean *= signs
        covariance *= np.outer(signs, signs)
    return mean, covariance


def _prepare_analytic(sim, configuration, source, beam, bunch, indices, slice_index, center, resources):
    p, xp = beam.particles, resources.xp
    if source.method == "frozen":
        mean, covariance = _frozen_moments(sim, configuration, source, beam, bunch, slice_index, center)
    else:
        minimum = 2 if "_round_" in source.solver else 3
        if indices.size < minimum:
            raise ValueError(f"quasi-frozen slice {slice_index} requires at least {minimum} live source particles")
        sign = _common_sign(configuration, beam.beam_id, bunch)
        values = xp.stack((sign * p.x[indices], sign * p.px[indices], p.y[indices], p.py[indices]), axis=-1)
        values = values.astype(source.statistics_precision or resources.dtype, copy=False)
        mean = xp.mean(values, axis=0)
        centered = values - mean
        covariance = centered.T @ centered / values.shape[0]
    return AnalyticSource(xp.asarray(mean, dtype=resources.dtype), xp.asarray(covariance, dtype=resources.dtype),
                          indices.size * bunch.ratio * bunch.num_charge * const.e, source.solver, resources, configuration.potential_reference_length)


def _prepare_pic(configuration, source, beam, bunch, indices, resources):
    xp, dtype = resources.xp, resources.dtype
    p = beam.particles
    pic = resources.get_pic(beam.beam_id, source, configuration.potential_reference_length)
    grid = pic.geometry
    sign = _common_sign(configuration, beam.beam_id, bunch)
    # Own only the source transverse state, before either side receives a kick.
    coordinates = xp.stack((sign * p.x[indices], sign * p.px[indices], p.y[indices], p.py[indices]))
    return PICSource(coordinates, bunch.ratio * bunch.num_charge * const.e, grid, source.deposition_method, resources, pic)


def _apply_source(configuration, beam, bunch, indices, distances, source, resources, endpoints):
    p, xp = beam.particles, resources.xp
    if xp is not np:
        return _apply_source_gpu(configuration, beam, bunch, indices, distances, source, resources, endpoints)
    sign = _common_sign(configuration, beam.beam_id, bunch)
    x, px = sign * p.x[indices], sign * p.px[indices]
    y, py = p.y[indices], p.py[indices]
    frame = getattr(bunch, "collision_frame", None)
    reference_p0 = frame["p0"] if frame else bunch.p0
    eta = p.dp[indices] if frame else delta_to_eta(p.dp[indices], bunch.beta, xp=xp)
    kwargs = {"endpoints": endpoints} if isinstance(source, PICSource) else {}
    _, ex, ey, potential_derivative = source.evaluate(x + distances * px, y + distances * py, distances, **kwargs)
    # For ions PASS stores momentum per nucleon; qm_ratio is |Z|/A.
    coefficient = resources.dtype.type(np.sign(bunch.num_charge) * bunch.qm_ratio / reference_p0)
    dpx, dpy = coefficient * ex, coefficient * ey
    eta = eta - coefficient * potential_derivative / 2 + (2 * px * dpx + dpx * dpx + 2 * py * dpy + dpy * dpy) / 4
    new_px, new_py = px + dpx, py + dpy
    new_dp = eta if frame else eta_to_delta(eta, bunch.beta, xp=xp)
    p.x[indices] = sign * (x - distances * dpx)
    p.px[indices] = sign * new_px
    p.y[indices] = y - distances * dpy
    p.py[indices] = new_py
    p.dp[indices] = new_dp
    invalid = ~xp.isfinite(new_dp) | ~xp.isfinite(new_px) | ~xp.isfinite(new_py)
    invalid |= ~xp.isfinite(p.x[indices]) | ~xp.isfinite(p.y[indices])
    invalid |= (1 + (eta if frame else new_dp)) <= 0
    if not frame:
        invalid |= 1 + bunch.beta**2 * eta <= 0
    invalid |= (1 + (eta if frame else new_dp))**2 <= new_px**2 + new_py**2
    resources.error_flags[...] |= xp.any(invalid).astype(xp.int32)


def _apply_source_gpu(configuration, beam, bunch, indices, distances, source, resources, endpoints):
    """Fuse the target drift and kick, preserving the CPU canonical conversion."""
    p, xp, dtype = beam.particles, resources.xp, resources.dtype
    n_particles = indices.size
    indices = xp.ascontiguousarray(indices, dtype=xp.int64)
    distances = xp.ascontiguousarray(xp.broadcast_to(distances, (n_particles, )), dtype=dtype)
    sign = np.int32(_common_sign(configuration, beam.beam_id, bunch))
    x, y = xp.empty(n_particles, dtype=dtype), xp.empty(n_particles, dtype=dtype)
    launch_gpu_kernel(_SOURCE_MAP_CUDA, "collision_target_positions", n_particles,
                      (p.x, p.px, p.y, p.py, indices, distances, sign, x, y, np.int64(n_particles)), dtype)
    kwargs = {"endpoints": endpoints} if isinstance(source, PICSource) else {}
    _, ex, ey, potential_derivative = source.evaluate(x, y, distances, **kwargs)
    frame = getattr(bunch, "collision_frame", None)
    reference_p0 = frame["p0"] if frame else bunch.p0
    coefficient = dtype.type(np.sign(bunch.num_charge) * bunch.qm_ratio / reference_p0)
    launch_gpu_kernel(_SOURCE_MAP_CUDA, "collision_apply_kick", n_particles,
                      (p.x, p.px, p.y, p.py, p.dp, indices, distances, ex, ey, potential_derivative, sign, coefficient, np.float64(
                          bunch.beta), np.int32(bool(frame)), resources.error_flags, np.int64(n_particles)), dtype)


def collide_bunch_pair(sim, configuration, beam_a, bunch_a, beam_b, bunch_b, *, resources=None, luminosity=None):
    """Execute one ideal synchronous bunch encounter in the configured basis.

    The latest explicit Slicer owns membership. Entrance centroids determine
    event order. Both source snapshots precede either kick; the next event
    rebuilds any changed source. Reference energy and t0 remain unchanged.
    """
    if configuration.mode == "weak-weak" and luminosity is None:
        return {"slice_events": 0, "mode": "weak-weak"}
    if (beam_a.beam_id, beam_b.beam_id) != tuple(configuration.beams):
        raise ValueError("Beam order must match the configured common basis")
    if beam_a.particles.xp is not beam_b.particles.xp or beam_a.particles.x.dtype != beam_b.particles.x.dtype:
        raise ValueError("Both beams must use the same backend and particle precision")
    beams, bunches = (beam_a, beam_b), (bunch_a, bunch_b)
    sources = [configuration.sources[str(beam.beam_id)] for beam in beams]
    memberships = [_members(sim, beam, bunch, source) for beam, bunch, source in zip(beams, bunches, sources)]
    resources = resources or _resources(sim, configuration, beam_a.particles)
    resources.begin()
    overlap = resources.xp.zeros((), dtype=resources.xp.float64) if luminosity is not None else None
    events = [(i, j) for i, ia in enumerate(memberships[0][0]) if ia.size for j, ib in enumerate(memberships[1][0]) if ib.size]
    events.sort(key=lambda pair: -(memberships[0][1][pair[0]] + memberships[1][1][pair[1]]))
    # A charge-free source is an exact identity, including dp. Sending it
    # through delta/eta round trips adds artificial energy roundoff every turn.
    receive = [(configuration.mode == "strong-strong" or configuration.weak_beam == beam.beam_id)
               and bunches[1 - side].ratio * bunches[1 - side].num_charge != 0 for side, beam in enumerate(beams)]
    # Unkicked sources are immutable within this encounter, including their moments.
    source_cache = [{}, {}]
    for pair in events:
        indices = [memberships[side][0][pair[side]] for side in range(2)]
        sample_luminosity = luminosity is not None and bunch_a.ratio > 0 and bunch_b.ratio > 0
        distances = [
            (beams[side].particles.z[indices[side]] - resources.dtype.type(memberships[1 - side][1][pair[1 - side]])) / resources.dtype.type(2)
            for side in range(2)
        ]
        endpoints = [(memberships[side][2][pair[side]].astype(resources.dtype) - resources.dtype.type(memberships[1 - side][1][pair[1 - side]])) /
                     resources.dtype.type(2) for side in range(2)]
        snapshots = [None, None]
        coordinates = [None, None]
        for side in range(2):
            if not receive[1 - side] and not sample_luminosity:
                continue
            if not receive[side] and pair[side] in source_cache[side]:
                snapshots[side], coordinates[side] = source_cache[side][pair[side]]
                continue
            if sources[side].method is None:
                p = beams[side].particles
                sign = _common_sign(configuration, beams[side].beam_id, bunches[side])
                coordinates[side] = resources.xp.stack(
                    (sign * p.x[indices[side]], sign * p.px[indices[side]], p.y[indices[side]], p.py[indices[side]]))
            elif sources[side].method == "pic":
                snapshots[side] = _prepare_pic(configuration, sources[side], beams[side], bunches[side], indices[side], resources)
            else:
                snapshots[side] = _prepare_analytic(sim, configuration, sources[side], beams[side], bunches[side], indices[side], pair[side],
                                                    memberships[side][1][pair[side]], resources)
            if not receive[side]:
                source_cache[side][pair[side]] = snapshots[side], coordinates[side]
        if sample_luminosity:
            # Both densities describe the same collision plane before either kick.
            distance = resources.dtype.type((memberships[0][1][pair[0]] - memberships[1][1][pair[1]]) / 2)
            overlap += luminosity.overlap(snapshots[0],
                                          snapshots[1],
                                          distance,
                                          indices[0].size * bunches[0].ratio,
                                          indices[1].size * bunches[1].ratio,
                                          coordinates_a=coordinates[0],
                                          coordinates_b=coordinates[1],
                                          error_flags=resources.error_flags)
        for side in range(2):
            if receive[side]:
                _apply_source(configuration, beams[side], bunches[side], indices[side], distances[side], snapshots[1 - side], resources,
                              endpoints[side])
    resources.validate()
    for bunch, source in zip(bunches, sources):
        slice_set = bunch.slice_sets[source.slice_set]
        if hasattr(slice_set, "observation_generation"):
            slice_set._beam_beam_consumed_generation = slice_set.observation_generation
    result = {
        "slice_events": len(events),
        "mode": configuration.mode,
        "methods": [source.method for source in sources],
        "precision": resources.dtype.name,
        "potential_reference_length": configuration.potential_reference_length
    }
    if luminosity is not None:
        result["luminosity_overlap_m2"] = overlap
        result["luminosity_populations"] = tuple(
            sum(indices.size for indices in membership[0]) * bunch.ratio for membership, bunch in zip(memberships, bunches))
    return result


_SOURCE_MAP_CUDA = r"""
extern "C" __global__ void collision_target_positions(
    const T* x,
    const T* px,
    const T* y,
    const T* py,
    const long long* indices,
    const T* distances,
    int sign,
    T* collision_x,
    T* collision_y,
    long long n
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    long long j = indices[i];
    collision_x[i] = T(sign) * x[j] + distances[i] * (T(sign) * px[j]);
    collision_y[i] = y[j] + distances[i] * py[j];
}
extern "C" __global__ void collision_apply_kick(
    T* x,
    T* px,
    T* y,
    T* py,
    T* dp,
    const long long* indices,
    const T* distances,
    const T* ex,
    const T* ey,
    const T* potential_derivative,
    int sign,
    T coefficient,
    double reference_beta,
    int collision_frame,
    int* error_flags,
    long long n
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    long long j = indices[i];
    T old_x = T(sign) * x[j], old_px = T(sign) * px[j], old_y = y[j], old_py = py[j];
    T eta = dp[j], rest_fraction = T((1 - reference_beta) * (1 + reference_beta));
    T beta = T(reference_beta), beta_squared = T(reference_beta * reference_beta);
    if (!collision_frame) {
        T value = eta * (T(2) + eta), beta_momentum_ratio = beta * (T(1) + eta);
        T energy_ratio = sqrt(rest_fraction + beta_momentum_ratio * beta_momentum_ratio);
        eta = value / (energy_ratio + T(1));
    }
    T dpx = coefficient * ex[i], dpy = coefficient * ey[i];
    eta = eta - coefficient * potential_derivative[i] / T(2) + (T(2) * old_px * dpx + dpx * dpx + T(2) * old_py * dpy + dpy * dpy) / T(4);
    T new_px = old_px + dpx, new_py = old_py + dpy, new_dp = eta;
    if (!collision_frame) {
        T value = T(2) * eta + beta_squared * eta * eta;
        T squared_ratio = (T(1) + T(2) * eta) + beta_squared * eta * eta;
        if (reference_beta > .5 && eta < 0)
            squared_ratio = (T(1) + eta) * (T(1) + eta) - rest_fraction * eta * eta;
        new_dp = value / (sqrt(squared_ratio) + T(1));
    }
    T new_x = T(sign) * (old_x - distances[i] * dpx), new_y = old_y - distances[i] * dpy;
    x[j] = new_x;
    px[j] = T(sign) * new_px;
    y[j] = new_y;
    py[j] = new_py;
    dp[j] = new_dp;
    T momentum_ratio = T(1) + (collision_frame ? eta : new_dp);
    bool invalid = !isfinite(new_dp) || !isfinite(new_px) || !isfinite(new_py) || !isfinite(new_x) || !isfinite(new_y);
    invalid |= momentum_ratio <= 0;
    if (!collision_frame)
        invalid |= T(1) + beta_squared * eta <= 0;
    invalid |= momentum_ratio * momentum_ratio <= new_px * new_px + new_py * new_py;
    if (invalid)
        atomicOr(error_flags, 1);
}
"""
