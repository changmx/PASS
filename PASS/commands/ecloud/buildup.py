"""Circular 2D3V electron buildup with optional beam/cloud PIC coupling."""

from copy import copy, deepcopy
import math

import numpy as np

from PASS.utils.constants import const

from .fields import prescribed_beam_field, requires_pic
from .chamber import Chamber
from .pusher import boris_push, electron_energy_ev, electron_gamma, external_magnetic_field, first_circle_hit, round_gaussian_beam_field
from .secondary import cosine_emission, isotropic_emission, true_secondary_yield
from .state import DynamicElectronCloudState, _finite_scalar


def _record_diagnostics(target, values):
    """Accumulate event ledgers while replacing instantaneous population values."""
    for name, value in values.items():
        if name in {
                "n_electrons", "n_macroparticles", "effective_macroparticles", "max_weight_fraction", "target_reached", "population_before",
                "kinetic_energy_ev"
        }:
            target[name] = float(value)
        else:
            target[name] = target.get(name, 0.) + float(value)


def _beam_field_bounds(field, xp):
    if field is None:
        return 0., 0.
    values = xp.stack((xp.max(xp.hypot(field.ex, field.ey)), xp.max(xp.abs(field.density))))
    return tuple(map(float, values if xp is np else xp.asnumpy(values)))


def _sample_initial_positions(configuration, chamber, count, generator):
    """Sample the chamber-conditioned profile without changing its total charge."""
    parameters = configuration.buildup
    model = getattr(parameters, "initial_distribution", "uniform_disk")
    center_x, center_y = configuration.center_x, configuration.center_y
    if model == "uniform_disk":
        radial = configuration.radius * np.sqrt(generator.random(count))
        angle = 2 * np.pi * generator.random(count)
        return center_x + radial * np.cos(angle), center_y + radial * np.sin(angle)
    xmin, xmax, ymin, ymax = chamber.bounds
    if model == "uniform_rectangle":
        half_x, half_y = parameters.initial_position_spread
        xmin, xmax = max(xmin, center_x - half_x), min(xmax, center_x + half_x)
        ymin, ymax = max(ymin, center_y - half_y), min(ymax, center_y + half_y)
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("initial electron distribution has no area inside the chamber bounds")
    if model == "gaussian":
        axes = []
        for low, high, center, sigma in zip((xmin, ymin), (xmax, ymax), (center_x, center_y), parameters.initial_position_spread):
            peak = float(np.clip(center, low, high))
            width = (high - low) / sigma
            uniform = width <= 1. or width * (abs((peak - center) / sigma) + width / 2) <= 1.
            axes.append((low, high, center, sigma, peak, uniform))
        if not all(axis[-1] for axis in axes):
            from scipy.stats import truncnorm
    x, y = np.empty(count), np.empty(count)
    filled, drawn, limit = 0, 0, max(65536, 128 * count)
    while filled < count and drawn < limit:
        size = min(max(64, 2 * (count - filled)), 262144, limit - drawn)
        if model == "gaussian":
            samples, log_acceptance = [], np.zeros(size)
            for low, high, center, sigma, peak, uniform in axes:
                if uniform:
                    sample = generator.uniform(low, high, size)
                    # Physical-coordinate rejection avoids narrow-CDF collapse for broad Gaussians.
                    delta = (sample - peak) / sigma
                    log_acceptance -= .5 * delta * (delta + 2 * ((peak - center) / sigma))
                else:
                    sample = truncnorm.rvs((low - center) / sigma, (high - center) / sigma,
                                           loc=center,
                                           scale=sigma,
                                           size=size,
                                           random_state=generator)
                samples.append(sample)
            sample_x, sample_y = samples
            accepted = generator.random(size) < np.exp(log_acceptance)
        else:
            sample_x, sample_y = generator.uniform(xmin, xmax, size), generator.uniform(ymin, ymax, size)
            accepted = True
        inside = chamber.contains(sample_x, sample_y) & accepted
        selected = np.flatnonzero(inside)[:count - filled]
        end = filled + selected.size
        x[filled:end], y[filled:end] = sample_x[selected], sample_y[selected]
        filled, drawn = end, drawn + size
    if filled != count:
        raise ValueError("initial electron distribution has zero or insufficient overlap with the chamber; sampling limit reached")
    return x, y


def _sample_initial_momenta(parameters, count, generator):
    """Convert independent physical velocity components to mechanical p/(m_e*c)."""
    model = getattr(parameters, "initial_velocity_distribution", "monoenergetic")
    if model == "monoenergetic":
        return isotropic_emission(count, parameters.initial_energy_ev, generator)
    spread = np.asarray(parameters.initial_velocity_spread, dtype=float)
    velocity = np.empty((count, 3))
    filled, drawn, limit = 0, 0, max(65536, 128 * count)
    while filled < count and drawn < limit:
        size = min(count - filled, 262144, limit - drawn)
        sample = (generator.uniform(-spread, spread, (size, 3)) if model == "uniform" else generator.normal(0., spread, (size, 3)))
        beta = np.hypot(np.hypot(sample[:, 0] / const.c, sample[:, 1] / const.c), sample[:, 2] / const.c)
        sample = sample[beta < 1.]
        end = filled + sample.shape[0]
        velocity[filled:end] = sample
        filled, drawn = end, drawn + size
    if filled != count:
        raise ValueError("initial electron velocity distribution has insufficient subluminal probability; sampling limit reached")
    beta = np.hypot(np.hypot(velocity[:, 0] / const.c, velocity[:, 1] / const.c), velocity[:, 2] / const.c)
    momentum = velocity / const.c / np.sqrt((1. - beta) * (1. + beta))[:, None]
    return momentum[:, 0], momentum[:, 1], momentum[:, 2]


class _CloudResources:
    """Serially shared numerical workspace, independent of physical state."""

    def __init__(self, configuration, backend, dtype):
        self.users, self.pic, self.pusher = 1, None, None
        try:
            if requires_pic(configuration):
                from .fields import RoundPICFields
                self.pic = RoundPICFields(configuration, backend=backend, dtype=dtype)
            if backend == "gpu":
                from .pusher import _GpuElectronPusher
                self.pusher = _GpuElectronPusher()
        except Exception:
            self.close()
            raise

    def close(self):
        try:
            if self.pic is not None:
                self.pic.close()
        finally:
            if self.pusher is not None:
                self.pusher.close()


class BuildUpCloud:
    """A 2D3V conducting station with a saved PCG64 random state.

    The legacy circular dilute model remains unchanged by default. Optional
    material, primary, population and self-field models share this evolution.
    Coupled mode always includes self-fields and uses actual beam slices.
    """

    def __init__(self, configuration, backend="cpu", dtype="float64", state=None, *, record_wall_spectra=True):
        if backend not in {"cpu", "gpu"} or np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("BuildUpCloud requires a CPU/GPU backend and float32/float64 beam precision")
        self.configuration, self.parameters = configuration, configuration.buildup
        self._record_wall_spectra = bool(record_wall_spectra)
        self._resource_configuration = deepcopy(configuration)
        self.backend, self.dtype, self.xp = backend, np.dtype("float64"), np
        if backend == "gpu":
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("BuildUpCloud requires an available CUDA device")
            self.xp = cp
        expected_solver = {"build_up": "round_gaussian_beam", "coupled": "fd_dirichlet"}.get(configuration.mode)
        if expected_solver is None or configuration.solver != expected_solver:
            raise ValueError("BuildUpCloud requires build_up/round_gaussian_beam or coupled/fd_dirichlet")
        self.chamber = Chamber(configuration)
        self.radius = self.chamber.radius
        self._self_fields = configuration.mode == "coupled" or getattr(self.parameters, "self_fields", False)
        self.electron_density = float(configuration.electron_density)
        source_radius = float(configuration.radius)
        if getattr(self.parameters, "initial_distribution", "uniform_disk") == "uniform_disk":
            if self.chamber.kind == "circle" and math.hypot(configuration.center_x, configuration.center_y) + source_radius >= self.radius:
                raise ValueError("initial electron cloud disk must fit strictly inside the conducting chamber")
            if self.chamber.kind != "circle" and not self.chamber.contains_disk(configuration.center_x, configuration.center_y, source_radius):
                raise ValueError("initial electron cloud disk must fit strictly inside its chamber")
        if not math.isfinite(self.radius * self.radius) or self.radius * self.radius <= 0:
            raise ValueError("chamber radius squared must be representable")
        self._beam_normalization = 1.0
        if configuration.mode == "build_up" and not requires_pic(configuration):
            sigma_squared = float(self.parameters.beam_sigma) * float(self.parameters.beam_sigma)
            if not math.isfinite(sigma_squared) or sigma_squared <= 0:
                raise ValueError("beam sigma squared must be positive and representable")
            self._beam_normalization = -math.expm1(-self.radius**2 / (2 * sigma_squared))
            if not self._beam_normalization > 0:
                raise ValueError("truncated beam normalization must be positive and representable")
        self._magnetic_field = np.asarray(self.parameters.magnetic_field, dtype=float)
        if self._magnetic_field.shape != (3, ) or not np.all(np.isfinite(self._magnetic_field)):
            raise ValueError("magnetic field must contain three finite Tesla components")
        self._magnetic_gradient = _finite_scalar(getattr(self.parameters, "magnetic_gradient", 0.0), "magnetic gradient")
        self._magnetic_bound = float(np.linalg.norm(self._magnetic_field)) + abs(self._magnetic_gradient) * self.radius
        if not math.isfinite(self._magnetic_bound):
            raise ValueError("electron cloud magnetic-field bound is not representable")
        if state is None:
            seed = configuration.random_seed
            entropy = None if seed is None else 2 * int(seed) if seed >= 0 else -2 * int(seed) - 1
            generator = np.random.default_rng(entropy)
            total_electrons = getattr(self.parameters, "initial_line_density", None)
            if total_electrons is None:
                try:
                    total_electrons = self.electron_density * np.pi * source_radius**2 if self.electron_density > 0 else 0.
                except OverflowError as exc:
                    raise ValueError("initial electron line density is not representable") from exc
            if not math.isfinite(total_electrons) or total_electrons < 0:
                raise ValueError("initial electron line density must be finite and nonnegative")
            count = configuration.n_macroparticles if total_electrons > 0 else 0
            if count > self.parameters.max_macroparticles:
                raise ValueError("initial electron cloud exceeds max_macroparticles")
            if count and total_electrons / count <= 0:
                raise ValueError("initial electron macroparticle weight is not representable")
            x, y = _sample_initial_positions(configuration, self.chamber, count, generator)
            ux, uy, uz = _sample_initial_momenta(self.parameters, count, generator)
            weight = np.full(count, total_electrons / count if count else 0)
            if getattr(self.parameters, "initial_velocity_distribution", "monoenergetic") == "monoenergetic":
                initial_energy = total_electrons * self.parameters.initial_energy_ev
            else:
                initial_energy = float(np.sum(weight * electron_energy_ev(ux, uy, uz)))
            state = DynamicElectronCloudState(x=x,
                                              y=y,
                                              ux=ux,
                                              uy=uy,
                                              uz=uz,
                                              weight=weight,
                                              source_length=1.0,
                                              time=None,
                                              rng_state=generator.bit_generator.state,
                                              last_turn=None,
                                              counters={
                                                  "initial_electrons": total_electrons,
                                                  "initial_energy_ev": initial_energy,
                                              })
        self.state = state
        self._resources = _CloudResources(configuration, backend, dtype)
        self._closed = False

    @property
    def pic(self):
        return self._resources.pic

    @property
    def state(self):
        """Materialize a portable host snapshot only when explicitly requested."""
        if self._host_state is None:
            host = np.asarray if self.backend == "cpu" else self.xp.asnumpy
            self._host_state = DynamicElectronCloudState(**{
                name: host(values)
                for name, values in self._particles.items()
            },
                                                         source_length=self.source_length,
                                                         time=self.time,
                                                         last_turn=self.last_turn,
                                                         rng_state=self._rng_state,
                                                         counters=self._counters,
                                                         wall_histograms=self._wall_histograms)
        return self._host_state

    @state.setter
    def state(self, state):
        self._validate_state(state)
        arrays = {name: self.xp.array(getattr(state, name), dtype=self.xp.float64, copy=True) for name in ("x", "y", "ux", "uy", "uz", "weight")}
        self._particles = arrays
        self.source_length, self.time, self.last_turn = state.source_length, state.time, state.last_turn
        self._rng_state, self._counters = deepcopy(state.rng_state), deepcopy(state.counters)
        self._wall_histograms = deepcopy(state.wall_histograms) if self._record_wall_spectra else {}
        self.n_electrons = float(state.weight.sum())
        self._host_state = state if self._record_wall_spectra or not state.wall_histograms else None

    @property
    def n_macroparticles(self):
        return self._particles["x"].size

    def population_statistics(self):
        """Reduce observables on the backend; transfer only the scalar results."""
        xp, arrays = self.xp, self._particles
        weight, total = arrays["weight"], self.n_electrons
        maximum = xp.max(weight) if weight.size else xp.asarray(0.)
        normalized = weight / xp.where(maximum > 0, maximum, 1.)
        effective = xp.sum(normalized)**2 / xp.where(xp.sum(normalized**2) > 0, xp.sum(normalized**2), 1.)
        kinetic = xp.sum(weight * electron_energy_ev(*(arrays[name] for name in ("ux", "uy", "uz")), xp))
        names = ["maximum_weight", "effective_macroparticles", "kinetic_energy_ev", "mean_energy_ev"]
        values = [maximum, effective, kinetic, kinetic / total if total else xp.asarray(0.)]
        for name in ("x", "y", "ux", "uy", "uz"):
            mean = xp.sum(weight * arrays[name]) / total if total else xp.asarray(0.)
            variance = xp.sum(weight * (arrays[name] - mean)**2) / total if total else xp.asarray(0.)
            names.extend(("mean_" + name, "rms_" + name))
            values.extend((mean, xp.sqrt(variance)))
        values = xp.stack(values)
        host = values if xp is np else xp.asnumpy(values)
        result = dict(zip(names, map(float, host)))
        result.update(n_macroparticles=self.n_macroparticles, n_electrons=total, line_density=total / self.source_length)
        return result

    def _validate_state(self, state):
        if not isinstance(state, DynamicElectronCloudState):
            raise TypeError("build-up cloud requires DynamicElectronCloudState")
        if state.x.size > self.parameters.max_macroparticles:
            raise ValueError("restored electron cloud exceeds max_macroparticles")
        if not np.all(self.chamber.contains(state.x, state.y)):
            raise ValueError("dynamic electrons must lie strictly inside the conducting chamber")

    def clone(self):
        """Detach particle state while retaining the stationary field factorization."""
        if self._closed:
            raise RuntimeError("cannot clone a closed electron cloud")
        if self.configuration != self._resource_configuration:
            return type(self)(self.configuration,
                              backend=self.backend,
                              dtype=self.dtype,
                              state=self.state,
                              record_wall_spectra=self._record_wall_spectra)
        candidate = copy(self)
        candidate._particles = self._arrays()
        candidate._rng_state, candidate._counters = deepcopy(self._rng_state), deepcopy(self._counters)
        candidate._wall_histograms = deepcopy(self._wall_histograms)
        self._resources.users += 1
        return candidate

    def close(self):
        """Release numerical resources after the last staged state is closed."""
        if not self._closed:
            self._closed = True
            self._resources.users -= 1
            if self._resources.users == 0:
                self._resources.close()

    def _arrays(self):
        return {name: values.copy() for name, values in self._particles.items()}

    def _generator(self):
        generator = np.random.Generator(np.random.PCG64(0))
        generator.bit_generator.state = deepcopy(self._rng_state)
        return generator

    def _validate_particles(self, arrays):
        if self._resources.pusher is not None:
            inside = None if self.chamber.kind == "circle" else self.chamber.contains(arrays["x"], arrays["y"], self.xp)
            self._resources.pusher.validate(arrays, self.radius, inside=inside)
            return
        xp = self.xp
        if not all(bool(xp.all(xp.isfinite(values))) for values in arrays.values()):
            raise ValueError("electron cloud integration produced nonfinite particle state")
        if not bool(xp.all(self.chamber.contains(arrays["x"], arrays["y"], xp))):
            raise ValueError("dynamic electrons must lie strictly inside the conducting chamber")
        if bool(xp.any(arrays["weight"] < 0)):
            raise ValueError("dynamic electron cloud weights must be nonnegative")

    def _commit(self, arrays, generator, *, time, last_turn, counter_updates=None, n_electrons=None, validated=False, wall_histograms=None):
        counters = deepcopy(self._counters)
        for name, value in (counter_updates or {}).items():
            counters[name] = counters.get(name, 0.0) + value
        if not all(math.isfinite(value) for value in counters.values()):
            raise ValueError("electron cloud cumulative diagnostics overflowed; state and RNG are unchanged")
        if n_electrons is None:
            n_electrons = float(self.xp.sum(arrays["weight"]))
        if not math.isfinite(n_electrons):
            raise ValueError("electron cloud total weight overflowed; state and RNG are unchanged")
        if arrays["x"].size > self.parameters.max_macroparticles:
            raise ValueError("electron cloud exceeds max_macroparticles")
        if not validated:
            self._validate_particles(arrays)
        self._particles, self._rng_state, self._counters = arrays, generator.bit_generator.state, counters
        self.time, self.last_turn, self.n_electrons = time, last_turn, n_electrons
        if wall_histograms is not None:
            self._wall_histograms = wall_histograms
        self._host_state = None

    def set_last_turn(self, turn):
        if turn is not None:
            if isinstance(turn, (bool, np.bool_)) or not isinstance(turn, (int, np.integer)) or turn < 0:
                raise ValueError("dynamic electron cloud last_turn must be a nonnegative integer or None")
            if self.time is None:
                raise ValueError("dynamic electron cloud completed turns require a physical time")
            turn = int(turn)
        self.last_turn, self._host_state = turn, None

    def inject_primary(self, n_real_particles):
        """Emit the configured real-electron yield per beam particle per metre."""
        if not math.isfinite(n_real_particles) or n_real_particles < 0:
            raise ValueError("primary source requires a finite nonnegative real beam-particle count")
        number = n_real_particles * self.parameters.primary_electrons_per_particle_per_m * self.source_length
        if not math.isfinite(number):
            raise ValueError("primary electron count is not representable")
        if number == 0:
            return dict(primary_electrons=0.0, primary_macroparticles=0, primary_energy_ev=0.0)
        count = self.parameters.primary_macroparticles
        if self.n_macroparticles + count > self.parameters.max_macroparticles:
            raise ValueError("primary emission exceeds max_macroparticles; no electrons were discarded")
        xp, generator, arrays = self.xp, self._generator(), dict(self._particles)
        if self.chamber.kind == "circle":
            angle = xp.asarray(2 * np.pi * generator.random(count))
            nx, ny = -xp.cos(angle), -xp.sin(angle)
            radius_inside = self.radius * (1 - 64 * np.finfo(float).eps)
            x, y = -radius_inside * nx, -radius_inside * ny
        else:
            x, y, nx, ny = self.chamber.sample_wall(count, generator, xp)
        ux, uy, uz = cosine_emission(nx, ny, self.parameters.emission_energy_ev, generator, xp)
        new = dict(x=x, y=y, ux=ux, uy=uy, uz=uz, weight=xp.full(count, number / count))
        for name in arrays:
            arrays[name] = xp.concatenate((arrays[name], new[name]))
        self._commit(arrays,
                     generator,
                     time=self.time,
                     last_turn=self.last_turn,
                     counter_updates={
                         "primary_electrons": number,
                         "primary_energy_ev": number * self.parameters.emission_energy_ev,
                     })
        return dict(primary_electrons=number, primary_macroparticles=count, primary_energy_ev=number * self.parameters.emission_energy_ev)

    def advance_slice(self,
                      end_time,
                      *,
                      n_real_particles,
                      line_charge,
                      beam_beta,
                      beam_x=None,
                      beam_y=None,
                      charge_per_length=None,
                      witness=None,
                      _gap=False):
        """Advance one prescribed slice, including birth times and optional probes.

        A detached candidate makes source sampling, subinterval integration and
        random draws atomic even when called outside an ElectronCloud command.
        """
        candidate = self.clone()
        try:
            diagnostics = candidate._advance_slice(end_time,
                                                   n_real_particles=n_real_particles,
                                                   line_charge=line_charge,
                                                   beam_beta=beam_beta,
                                                   beam_x=beam_x,
                                                   beam_y=beam_y,
                                                   charge_per_length=charge_per_length,
                                                   witness=witness,
                                                   gap=_gap)
            for name in ("_particles", "_rng_state", "_counters", "_wall_histograms", "time", "last_turn", "n_electrons", "_host_state"):
                setattr(self, name, getattr(candidate, name))
        finally:
            candidate.close()
        return diagnostics

    def advance_gap(self, end_time, *, _owned=False):
        """Evolve an empty beam interval while retaining independent photon sources."""
        if not any(source.kind == "photoelectric" for source in getattr(self.parameters, "primary_sources", [])):
            return self.advance_to(end_time, _owned=_owned)
        if _owned:
            return self._advance_slice(end_time,
                                       n_real_particles=0.,
                                       line_charge=0.,
                                       beam_beta=0.,
                                       beam_x=None,
                                       beam_y=None,
                                       charge_per_length=None,
                                       witness=None,
                                       gap=True)
        return self.advance_slice(end_time, n_real_particles=0., line_charge=0., beam_beta=0., _gap=True)

    def _advance_slice(self, end_time, *, n_real_particles, line_charge, beam_beta, beam_x, beam_y, charge_per_length, witness, gap=False):
        if self.time is None or end_time <= self.time:
            raise ValueError("electron-cloud slice requires an initialized clock and positive duration")
        xp, start_time = self.xp, self.time
        duration = end_time - start_time
        beam_field = None
        if self.configuration.mode == "coupled" and not gap:
            empty_source = beam_x is None and beam_y is None and n_real_particles == 0 and line_charge == 0
            if not empty_source and (beam_x is None or beam_y is None or charge_per_length is None):
                raise ValueError("coupled slice requires actual transverse coordinates and charge per length")
            if not empty_source:
                beam_field = self.pic.solve(beam_x, beam_y, charge_per_length)
        elif self.pic is not None and not gap:
            beam_field = prescribed_beam_field(self, line_charge)
        sources = getattr(self.parameters, "primary_sources", [])
        if gap:
            sources = [source for source in sources if source.kind == "photoelectric"]
        primary_diagnostics = {}
        births = None
        if sources:
            from .secondary import sample_primary_sources
            generator = self._generator()
            driver = getattr(self.parameters, "beam_driver", None)
            if driver is not None and driver.model == "density_map" and any(source.kind == "gas_ionization" for source in sources):
                field = prescribed_beam_field(self, 1.)
                probability = np.asarray(field.density) if xp is np else xp.asnumpy(field.density)
                probability = probability.ravel() / probability.sum()
                sampled = generator.choice(probability.size, size=max(source.macroparticles for source in sources), p=probability)
                beam_x = xp.asarray(self.pic.geometry.x[sampled % self.pic.geometry.nx])
                beam_y = xp.asarray(self.pic.geometry.y[sampled // self.pic.geometry.nx])
            births = sample_primary_sources(sources,
                                            n_real_particles,
                                            self.source_length,
                                            duration,
                                            self.chamber,
                                            generator,
                                            xp,
                                            beam_sigma=self.parameters.beam_sigma,
                                            beam_x=beam_x,
                                            beam_y=beam_y,
                                            beam_sigma_x=None if driver is None else driver.sigma_x,
                                            beam_sigma_y=None if driver is None else driver.sigma_y,
                                            beam_center_x=0. if driver is None else driver.center_x,
                                            beam_center_y=0. if driver is None else driver.center_y)
            primary_diagnostics = births["diagnostics"]
            self._rng_state = deepcopy(generator.bit_generator.state)
            self._host_state = None
        else:
            primary_diagnostics = self.inject_primary(n_real_particles)
        diagnostics = dict(primary_diagnostics)
        beam_bounds = _beam_field_bounds(beam_field, xp)
        diagnostics["dt"] = duration
        if witness is not None:
            diagnostics["mean_ex"], diagnostics["mean_ey"] = [xp.zeros_like(value, dtype=xp.float64) for value in witness]

        def advance(time):
            fraction = (time - self.time) / duration
            if fraction <= 0:
                return
            result = self.advance_to(time, line_charge, beam_beta, beam_field=beam_field, witness=witness, _owned=True, _beam_bounds=beam_bounds)
            for name, value in result.items():
                if name in {"dt", "n_electrons", "n_macroparticles", "kinetic_energy_ev"}:
                    diagnostics[name] = value
                elif name in {"mean_ex", "mean_ey"}:
                    diagnostics[name] += fraction * value
                else:
                    _record_diagnostics(diagnostics, {name: value})
            if diagnostics.get("steps", 0) > self.parameters.max_steps:
                raise ValueError("electron cloud slice exceeds max_steps; source and RNG are unchanged")

        if births is not None and births["weight"].size:
            birth_fraction = births["birth_fraction"].get() if hasattr(births["birth_fraction"], "get") else np.asarray(births["birth_fraction"])
            order = np.argsort(birth_fraction, kind="stable")
            sorted_fraction = birth_fraction[order]
            for begin, finish in zip(np.r_[0, np.flatnonzero(np.diff(sorted_fraction)) + 1], np.r_[np.flatnonzero(np.diff(sorted_fraction)) + 1,
                                                                                                   order.size]):
                advance(start_time + float(sorted_fraction[begin]) * duration)
                indices = xp.asarray(order[begin:finish])
                arrays = {name: xp.concatenate((values, births[name][indices])) for name, values in self._particles.items()}
                generator = self._generator()
                population_counters = {}
                if getattr(self.parameters, "population_control", None) is not None:
                    from .population import manage_population
                    arrays, report = manage_population(arrays, self.parameters.population_control, generator, xp, geometry=self.chamber)
                    _record_diagnostics(diagnostics, report)
                    population_counters = {
                        name: value
                        for name, value in report.items() if name not in {
                            "n_electrons", "n_macroparticles", "kinetic_energy_ev", "effective_macroparticles", "max_weight_fraction",
                            "target_reached", "population_before"
                        }
                    }
                count = float(xp.sum(births["weight"][indices]))
                energy = float(xp.sum(births["weight"][indices] * electron_energy_ev(*(births[name][indices] for name in ("ux", "uy", "uz")), xp)))
                self._commit(arrays,
                             generator,
                             time=self.time,
                             last_turn=self.last_turn,
                             counter_updates={
                                 "primary_electrons": count,
                                 "primary_energy_ev": energy,
                                 **population_counters,
                             })
        advance(end_time)
        if sources:
            self._commit(self._particles,
                         self._generator(),
                         time=self.time,
                         last_turn=self.last_turn,
                         counter_updates={
                             name: value
                             for name, value in primary_diagnostics.items() if name not in {"primary_electrons", "primary_energy_ev"}
                         },
                         n_electrons=self.n_electrons,
                         validated=True)
        diagnostics["dt"] = duration
        for name in ("wall_hits", "incident_electrons", "emitted_electrons", "steps"):
            diagnostics.setdefault(name, 0.)
        return diagnostics

    def _record_wall_events(self, arrays, impacted, diagnostics, emitted_energy_ev=None):
        if "_wall_histograms" not in diagnostics:
            return
        from .diagnostics import record_wall_histograms
        nx, ny = self.chamber.inward_normal(arrays["x"][impacted], arrays["y"][impacted], self.xp)
        diagnostics["_wall_histograms"] = record_wall_histograms(diagnostics["_wall_histograms"],
                                                                 *(arrays[name][impacted] for name in ("x", "y", "ux", "uy", "uz", "weight")),
                                                                 nx,
                                                                 ny,
                                                                 self.chamber,
                                                                 self.xp,
                                                                 emitted_energy_ev=emitted_energy_ev)
        if not self._wall_histograms and self._counters.get("wall_hits", 0) > 0:
            for histogram in diagnostics["_wall_histograms"].values():
                histogram["scope"] = "cumulative_since_recording_restart"

    def _drift(self, arrays, duration, generator, diagnostics, hits=None):
        if self._resources.pusher is not None and self.chamber.kind == "circle" and getattr(self.parameters, "surface_emission", None) is None:
            return self._drift_gpu(arrays, duration, generator, diagnostics, hits)
        xp = self.xp
        if self._resources.pusher is not None:
            remaining, pending, has_zero = self._resources.pusher.prepare_chamber_drift(arrays, duration, self.chamber)
            if not pending and not has_zero:
                return hits
        else:
            remaining = xp.full(arrays["x"].size, duration)
        while bool(xp.any(remaining > 0)):
            selected = xp.flatnonzero(remaining > 0)
            ux, uy, uz = (arrays[name][selected] for name in ("ux", "uy", "uz"))
            gamma = electron_gamma(ux, uy, uz, xp)
            vx, vy = const.c * (ux / gamma), const.c * (uy / gamma)
            x, y = arrays["x"][selected], arrays["y"][selected]
            hit_time = self.chamber.first_hit(x, y, vx, vy, xp)
            crossing = hit_time <= remaining[selected]
            speed = xp.hypot(vx, vy)
            time_tolerance = 32 * np.finfo(float).eps * remaining[selected] + (32 * np.finfo(float).eps * self.chamber.scale) / xp.where(
                speed > 0, speed, 1.)
            elapsed = xp.minimum(hit_time, remaining[selected])
            end_x, end_y = x + vx * elapsed, y + vy * elapsed
            late = (~crossing) & xp.isfinite(hit_time) & (hit_time <= remaining[selected] + time_tolerance)
            # A roundoff allowance cannot move a still-interior slow particle
            # to a wall whose exact impact lies far in its future.
            moved = (end_x != x) | (end_y != y)
            crossing |= late & moved & ~self.chamber.contains(end_x, end_y, xp)
            arrays["x"][selected] = end_x
            arrays["y"][selected] = end_y
            remaining[selected] -= elapsed
            arrays["x"][selected[crossing]] = x[crossing] + vx[crossing] * hit_time[crossing]
            arrays["y"][selected[crossing]] = y[crossing] + vy[crossing] * hit_time[crossing]
            no_hit = selected[~crossing]
            remaining[no_hit] = 0
            impacted = selected[crossing]
            if impacted.size == 0:
                continue
            if hits is None:
                hits = xp.zeros(remaining.shape, dtype=xp.int64)
            hits[impacted] += 1
            if bool(xp.any(hits[impacted] > self.parameters.max_wall_hits_per_step)):
                raise ValueError("electron wall events exceed max_wall_hits_per_step; reduce max_time_step")
            if getattr(self.parameters, "surface_emission", None) is not None:
                from .secondary import emit_surface
                nx, ny = self.chamber.inward_normal(arrays["x"][impacted], arrays["y"][impacted], xp)
                emitted = emit_surface(*(arrays[name][impacted] for name in ("ux", "uy", "uz", "weight")), nx, ny, self.parameters.surface_emission,
                                       generator, xp)
                outgoing_energy = emitted["weight"] * electron_energy_ev(*(emitted[name] for name in ("ux", "uy", "uz")), xp)
                by_parent = (xp.bincount(emitted["parent_indices"], weights=outgoing_energy, minlength=impacted.size)
                             if emitted["parent_indices"].size else xp.zeros(impacted.size, dtype=xp.float64))
                self._record_wall_events(arrays, impacted, diagnostics, by_parent)
                parents = impacted[emitted["parent_indices"]]
                child_nx, child_ny = nx[emitted["parent_indices"]], ny[emitted["parent_indices"]]
                child_x, child_y = self.chamber.move_inside(arrays["x"][parents], arrays["y"][parents], child_nx, child_ny, xp)
                survivors = xp.ones(arrays["x"].size, dtype=bool)
                survivors[impacted] = False
                new_values = dict(x=child_x, y=child_y, **{name: emitted[name] for name in ("ux", "uy", "uz", "weight")})
                child_remaining, child_hits = remaining[parents], hits[parents]
                for name in arrays:
                    arrays[name] = xp.concatenate((arrays[name][survivors], new_values[name]))
                remaining = xp.concatenate((remaining[survivors], child_remaining))
                hits = xp.concatenate((hits[survivors], child_hits))
                diagnostics["wall_hits"] += int(impacted.size)
                for name, value in emitted["diagnostics"].items():
                    diagnostics[name] = diagnostics.get(name, 0.) + float(value)
                if arrays["x"].size > self.parameters.max_macroparticles and getattr(self.parameters, "population_control", None) is None:
                    raise ValueError("secondary emission exceeds max_macroparticles; reduce event weights or enable population control")
                continue
            incident_energy = electron_energy_ev(*(arrays[name][impacted] for name in ("ux", "uy", "uz")), xp)
            incident_weight = arrays["weight"][impacted]
            emitted_yield = true_secondary_yield(incident_energy, self.parameters.secondary_yield_max, self.parameters.secondary_peak_energy_ev,
                                                 self.parameters.secondary_shape, self.parameters.emission_energy_ev, xp)
            emitted_weight = incident_weight * emitted_yield
            emitted_energy = xp.minimum(incident_energy, self.parameters.emission_energy_ev)
            self._record_wall_events(arrays, impacted, diagnostics, emitted_weight * emitted_energy)
            diagnostics["wall_hits"] += int(impacted.size)
            diagnostics["incident_electrons"] += xp.sum(incident_weight)
            diagnostics["emitted_electrons"] += xp.sum(emitted_weight)
            diagnostics["incident_energy_ev"] += xp.sum(incident_weight * incident_energy)
            diagnostics["emitted_energy_ev"] += xp.sum(emitted_weight * emitted_energy)
            nx, ny = self.chamber.inward_normal(arrays["x"][impacted], arrays["y"][impacted], xp)
            ux, uy, uz = cosine_emission(nx, ny, emitted_energy, generator, xp)
            arrays["ux"][impacted], arrays["uy"][impacted], arrays["uz"][impacted] = ux, uy, uz
            if self.chamber.kind == "circle":
                radius_inside = self.radius * (1 - 64 * np.finfo(float).eps)
                arrays["x"][impacted], arrays["y"][impacted] = -radius_inside * nx, -radius_inside * ny
            else:
                arrays["x"][impacted], arrays["y"][impacted] = self.chamber.move_inside(arrays["x"][impacted], arrays["y"][impacted], nx, ny, xp)
            arrays["weight"][impacted] = emitted_weight
            remaining[impacted[emitted_weight == 0]] = 0
        if bool(xp.any(arrays["weight"] == 0)):
            alive = arrays["weight"] > 0
            for name in arrays:
                arrays[name] = arrays[name][alive]
            if hits is not None:
                hits = hits[alive]
        return hits

    def _drift_gpu(self, arrays, duration, generator, diagnostics, hits=None):
        """Process ordered collision rounds, preserving the host random stream."""
        xp, pusher = self.xp, self._resources.pusher
        remaining = (pusher.prepare_drift(arrays, duration, self.radius) if duration == 0 else xp.full(arrays["x"].size, duration, dtype=xp.float64))
        had_hits = hits is not None
        if hits is None:
            hits = xp.zeros(remaining.shape, dtype=xp.int64)
        has_zero = False
        while True:
            impacted, zero_in_round = pusher.drift_wall(arrays, remaining, hits, self.radius, self.parameters.max_wall_hits_per_step)
            has_zero = has_zero or zero_in_round
            if impacted is None:
                break
            had_hits = True
            incident_arrays = {name: values[impacted] for name, values in arrays.items()} if "_wall_histograms" in diagnostics else None
            # Draw the complete mu batch before azimuth, even for absorbed hits.
            mu = np.sqrt(generator.random(impacted.size))
            azimuth = 2 * np.pi * generator.random(impacted.size)
            ledgers = pusher.emit_wall(arrays, remaining, impacted, mu, azimuth, self.radius, self.parameters.secondary_yield_max,
                                       self.parameters.secondary_peak_energy_ev, self.parameters.secondary_shape, self.parameters.emission_energy_ev)
            if incident_arrays is not None:
                self._record_wall_events(incident_arrays, xp.arange(impacted.size), diagnostics, ledgers[3])
            # Keep each original reduction and host addition order; batch only D2H.
            totals = xp.stack([xp.sum(values) for values in ledgers])
            diagnostics["wall_hits"] += int(impacted.size)
            for name, value in zip(("incident_electrons", "emitted_electrons", "incident_energy_ev", "emitted_energy_ev"), totals):
                diagnostics[name] += value
        if has_zero:
            alive = xp.flatnonzero(arrays["weight"] > 0)
            for name in arrays:
                arrays[name] = arrays[name][alive]
            hits = hits[alive]
        return hits if had_hits else None

    def advance_to(self, end_time, line_charge=0.0, beam_beta=0.0, *, beam_field=None, witness=None, _owned=False, _beam_bounds=None):
        """Stage an interval; coupled witnesses receive its mean cloud E field.

        Beam charge and witness positions are held fixed within a saved slice.
        Cloud fields are rebuilt at each Boris kick; their midpoint quadrature
        provides the transverse slice kick without feeding beam self-fields back.
        Internal callers may reuse arrays only when they already own a detached
        candidate whose enclosing transaction discards every failure.
        """
        end_time = _finite_scalar(end_time, "time")
        if not all(math.isfinite(value) for value in (end_time, line_charge, beam_beta)) or not 0 <= beam_beta <= 1:
            raise ValueError("build-up times/line charge must be finite and beam_beta must be in [0,1]")
        if self.pic is None and (beam_field is not None or witness is not None):
            raise ValueError("PIC beam fields and witnesses require an enabled PIC solver")
        if self.pic is not None and line_charge != 0 and beam_field is None:
            if self.configuration.mode == "coupled":
                raise ValueError("coupled mode requires the actual beam slice field")
            beam_field = prescribed_beam_field(self, line_charge)
        start_time = self.time
        if start_time is not None and end_time < start_time:
            raise ValueError("electron cloud physical time cannot run backwards")
        duration = 0.0 if start_time is None else end_time - start_time
        diagnostics = dict(dt=duration,
                           steps=0,
                           wall_hits=0,
                           incident_electrons=0.0,
                           emitted_electrons=0.0,
                           incident_energy_ev=0.0,
                           emitted_energy_ev=0.0,
                           field_work_ev=0.0)
        if self._record_wall_spectra:
            diagnostics["_wall_histograms"] = deepcopy(self._wall_histograms)
        arrays, generator, xp = dict(self._particles) if _owned else self._arrays(), self._generator(), self.xp
        pusher = self._resources.pusher
        mean_field = None
        if witness is not None:
            mean_field = [xp.zeros_like(values, dtype=xp.float64) for values in witness]
        sigma = self.parameters.beam_sigma or min(self.chamber.bounds[1] - self.chamber.bounds[0], self.chamber.bounds[3] - self.chamber.bounds[2])
        line_strength = abs(line_charge) / (2 * const.pi * const.epsilon0 * self._beam_normalization) if self.pic is None else 0.0
        field_bound = line_strength * min(self.radius / (2 * sigma**2), 1 / (math.sqrt(2) * sigma)) if self.pic is None else 0.0
        beam_density_bound = 0.
        if beam_field is not None:
            field_bound, beam_density_bound = _beam_field_bounds(beam_field, xp) if _beam_bounds is None else _beam_bounds
        magnetic_bound = self._magnetic_bound + beam_beta * field_bound / const.c
        cyclotron = const.e * magnetic_bound / const.m_e_kg
        beam_frequency = math.sqrt(const.e * line_strength / (2 * sigma**2 * const.m_e_kg)) if self.pic is None else 0.0
        step_limit = float(self.parameters.max_time_step)
        if cyclotron > 0:
            step_limit = min(step_limit, .2 / cyclotron)
        if beam_frequency > 0:
            step_limit = min(step_limit, .2 / beam_frequency)
        if not math.isfinite(duration) or not math.isfinite(step_limit) or step_limit <= 0:
            raise ValueError("build-up interval or required integration step is not representable")
        clock_resolution = max(math.ulp(start_time or 0.0), math.ulp(end_time))
        relative_roundoff_limit = math.sqrt(math.ulp(1.0))
        elapsed = 0.0
        while elapsed < duration:
            if arrays["x"].size == 0:
                break
            if diagnostics["steps"] >= self.parameters.max_steps:
                raise ValueError("electron cloud interval exceeds max_steps; state and RNG are unchanged")
            remaining = duration - elapsed
            dt = min(step_limit, remaining)
            density_bound, electric_bound, transverse_speed = 0., 0., 0.
            bounds = []
            if self.pic is not None:
                if self._self_fields or mean_field is not None:
                    cloud_field = self.pic.solve(arrays["x"], arrays["y"], -const.e * arrays["weight"] / self.source_length)
                    bounds.extend((xp.max(xp.abs(cloud_field.density)), xp.max(xp.hypot(cloud_field.ex, cloud_field.ey))))
                else:
                    bounds.extend((xp.asarray(0.), xp.asarray(0.)))
            if line_charge != 0 or self.pic is not None:
                if pusher is None:
                    gamma = electron_gamma(*(arrays[name] for name in ("ux", "uy", "uz")), xp)
                    bounds.append(const.c * xp.max(xp.hypot(arrays["ux"], arrays["uy"]) / gamma))
                else:
                    bounds.append(xp.max(pusher.transverse_speed(arrays)))
            if bounds:
                reduced = xp.stack(bounds)
                host_bounds = reduced if xp is np else xp.asnumpy(reduced)
                if self.pic is not None:
                    density_bound, electric_bound, transverse_speed = map(float, host_bounds)
                else:
                    transverse_speed = float(host_bounds[0])
            if self.pic is not None:
                electric_bound += field_bound
                density_bound += beam_density_bound
                mesh_length = min(self.pic.geometry.dx, self.pic.geometry.dy)
                # Resolve plasma oscillations and acceleration across one cell.
                frequency_squared = const.e / const.m_e_kg * max(density_bound / const.epsilon0, electric_bound / mesh_length)
                if frequency_squared > 0:
                    dt = min(dt, .2 / math.sqrt(frequency_squared))
            if transverse_speed > 0:
                length_scale = mesh_length if self.pic is not None else min(sigma, self.radius)
                dt = min(dt, .2 * length_scale / transverse_speed)
            # Endpoint subtraction also carries the absolute clock's rounding.
            # Never let a poorly resolved clock merge a finite fraction of a step.
            roundoff = 8 * max(clock_resolution, math.ulp(duration), math.ulp(elapsed), math.ulp(dt))
            if remaining - dt <= min(roundoff, relative_roundoff_limit * dt):
                dt = remaining
            if not elapsed + dt > elapsed:
                raise ValueError("electron cloud time step is below floating-point resolution")
            hits = self._drift(arrays, dt / 2, generator, diagnostics)
            if self.pic is None:
                beam_ex, beam_ey = round_gaussian_beam_field(arrays["x"], arrays["y"], line_charge, sigma, self.radius, xp)
                ex, ey = 0.0, 0.0
            else:
                cloud_field = (self.pic.solve(arrays["x"], arrays["y"], -const.e * arrays["weight"] /
                                              self.source_length) if self._self_fields or mean_field is not None else None)
                if cloud_field is None:
                    ex, ey = 0., 0.
                    beam_ex, beam_ey = self.pic.sample(beam_field, arrays["x"], arrays["y"]) if beam_field is not None else (0., 0.)
                elif beam_field is None:
                    ex, ey = self.pic.sample(cloud_field, arrays["x"], arrays["y"])
                    beam_ex, beam_ey = 0.0, 0.0
                else:
                    (ex, ey), (beam_ex, beam_ey) = self.pic.sample_pair(cloud_field, beam_field, arrays["x"], arrays["y"])
                if mean_field is not None:
                    witness_ex, witness_ey = self.pic.sample(cloud_field, *witness)
                    mean_field[0] += witness_ex * (dt / duration)
                    mean_field[1] += witness_ey * (dt / duration)
                if not self._self_fields:
                    ex, ey = 0., 0.
            if pusher is None:
                bx, by, bz = external_magnetic_field(arrays["x"], arrays["y"], self._magnetic_field, self._magnetic_gradient)
                bx = bx - beam_beta * beam_ey / const.c
                by = by + beam_beta * beam_ex / const.c
                energy_before = electron_energy_ev(*(arrays[name] for name in ("ux", "uy", "uz")), xp)
                arrays["ux"], arrays["uy"], arrays["uz"] = boris_push(arrays["ux"], arrays["uy"], arrays["uz"], ex + beam_ex, ey + beam_ey, bx, by,
                                                                      bz, dt, xp)
                energy_after = electron_energy_ev(*(arrays[name] for name in ("ux", "uy", "uz")), xp)
                energy_delta = arrays["weight"] * (energy_after - energy_before)
            else:
                energy_delta = pusher.kick(arrays,
                                           ex,
                                           ey,
                                           beam_ex,
                                           beam_ey,
                                           self._magnetic_field,
                                           beam_beta,
                                           dt,
                                           magnetic_gradient=self._magnetic_gradient)
            diagnostics["field_work_ev"] += xp.sum(energy_delta)
            self._drift(arrays, dt / 2, generator, diagnostics, hits)
            if getattr(self.parameters, "population_control", None) is not None:
                from .population import manage_population
                arrays, population_diagnostics = manage_population(arrays, self.parameters.population_control, generator, xp, geometry=self.chamber)
                _record_diagnostics(diagnostics, population_diagnostics)
            self._validate_particles(arrays)
            elapsed += dt
            diagnostics["steps"] += 1
        diagnostics["wall_energy_ev"] = diagnostics["incident_energy_ev"] - diagnostics["emitted_energy_ev"]
        diagnostics["n_electrons"] = xp.sum(arrays["weight"])
        diagnostics["n_macroparticles"] = int(arrays["x"].size)
        weighted_energy = (arrays["weight"] *
                           electron_energy_ev(*(arrays[name]
                                                for name in ("ux", "uy", "uz")), xp) if pusher is None else pusher.weighted_energy(arrays))
        diagnostics["kinetic_energy_ev"] = xp.sum(weighted_energy)
        wall_histograms = diagnostics.pop("_wall_histograms", None)
        if xp is not np:
            names = [name for name, value in diagnostics.items() if isinstance(value, xp.ndarray)]
            values = xp.asnumpy(xp.stack([diagnostics[name] for name in names]))
            diagnostics.update(zip(names, map(float, values)))
        if not all(math.isfinite(value) for value in diagnostics.values()):
            raise ValueError("electron cloud diagnostics overflowed; state and RNG are unchanged")
        counter_updates = {
            name: value
            for name, value in diagnostics.items() if name not in {
                "dt", "n_electrons", "n_macroparticles", "kinetic_energy_ev", "effective_macroparticles", "max_weight_fraction", "target_reached",
                "population_before"
            }
        }
        counter_updates["integrated_time"] = duration
        if mean_field is not None and not all(bool(xp.all(xp.isfinite(values))) for values in mean_field):
            raise ValueError("electron cloud mean witness field is not finite")
        self._commit(arrays,
                     generator,
                     time=end_time,
                     last_turn=self.last_turn,
                     counter_updates=counter_updates,
                     n_electrons=diagnostics["n_electrons"],
                     validated=True,
                     wall_histograms=wall_histograms)
        if mean_field is not None:
            diagnostics["mean_ex"], diagnostics["mean_ey"] = mean_field
        return diagnostics
