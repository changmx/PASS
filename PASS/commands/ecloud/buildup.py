"""Circular 2D3V electron buildup with optional beam/cloud PIC coupling."""

from copy import copy, deepcopy
import math

import numpy as np

from PASS.utils.constants import const

from .pusher import boris_push, electron_energy_ev, electron_gamma, external_magnetic_field, first_circle_hit, round_gaussian_beam_field
from .secondary import cosine_emission, isotropic_emission, true_secondary_yield
from .state import DynamicElectronCloudState, _finite_scalar


class _CloudResources:
    """Serially shared numerical workspace, independent of physical state."""

    def __init__(self, configuration, backend, dtype):
        self.users, self.pic, self.pusher = 1, None, None
        try:
            if configuration.mode == "coupled":
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
    """A circular 2D3V station; all random draws use a saved host PCG64 state.

    Each incident macro electron produces at most one reweighted true-secondary
    macro electron. There is no elastic/reflected component. Only coupled
    mode includes electron space charge and samples the actual beam slices.
    """

    def __init__(self, configuration, backend="cpu", dtype="float64", state=None):
        if backend not in {"cpu", "gpu"} or np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("BuildUpCloud requires a CPU/GPU backend and float32/float64 beam precision")
        self.configuration, self.parameters = configuration, configuration.buildup
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
        self.radius = float(self.parameters.chamber_radius)
        self.electron_density = float(configuration.electron_density)
        source_radius = float(configuration.radius)
        if math.hypot(configuration.center_x, configuration.center_y) + source_radius >= self.radius:
            raise ValueError("initial electron cloud disk must fit strictly inside the round chamber")
        if not math.isfinite(self.radius * self.radius) or self.radius * self.radius <= 0:
            raise ValueError("chamber radius squared must be representable")
        self._beam_normalization = 1.0
        if configuration.mode == "build_up":
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
            count = configuration.n_macroparticles if self.electron_density > 0 else 0
            if count > self.parameters.max_macroparticles:
                raise ValueError("initial electron cloud exceeds max_macroparticles")
            radial = source_radius * np.sqrt(generator.random(count))
            angle = 2 * np.pi * generator.random(count)
            ux, uy, uz = isotropic_emission(count, self.parameters.initial_energy_ev, generator)
            total_electrons = self.electron_density * np.pi * source_radius**2
            state = DynamicElectronCloudState(x=configuration.center_x + radial * np.cos(angle),
                                              y=configuration.center_y + radial * np.sin(angle),
                                              ux=ux,
                                              uy=uy,
                                              uz=uz,
                                              weight=np.full(count, total_electrons / count if count else 0),
                                              source_length=1.0,
                                              time=None,
                                              rng_state=generator.bit_generator.state,
                                              last_turn=None,
                                              counters={
                                                  "initial_electrons": total_electrons,
                                                  "initial_energy_ev": total_electrons * self.parameters.initial_energy_ev,
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
                                                         counters=self._counters)
        return self._host_state

    @state.setter
    def state(self, state):
        self._validate_state(state)
        arrays = {name: self.xp.array(getattr(state, name), dtype=self.xp.float64, copy=True) for name in ("x", "y", "ux", "uy", "uz", "weight")}
        self._particles = arrays
        self.source_length, self.time, self.last_turn = state.source_length, state.time, state.last_turn
        self._rng_state, self._counters = deepcopy(state.rng_state), deepcopy(state.counters)
        self.n_electrons = float(state.weight.sum())
        self._host_state = state

    @property
    def n_macroparticles(self):
        return self._particles["x"].size

    def _validate_state(self, state):
        if not isinstance(state, DynamicElectronCloudState):
            raise TypeError("build-up cloud requires DynamicElectronCloudState")
        if state.x.size > self.parameters.max_macroparticles:
            raise ValueError("restored electron cloud exceeds max_macroparticles")
        if np.any(np.hypot(state.x, state.y) >= self.radius):
            raise ValueError("dynamic electrons must lie strictly inside the round chamber")

    def clone(self):
        """Detach particle state while retaining the stationary field factorization."""
        if self._closed:
            raise RuntimeError("cannot clone a closed electron cloud")
        if self.configuration != self._resource_configuration:
            return type(self)(self.configuration, backend=self.backend, dtype=self.dtype, state=self.state)
        candidate = copy(self)
        candidate._particles = self._arrays()
        candidate._rng_state, candidate._counters = deepcopy(self._rng_state), deepcopy(self._counters)
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
            self._resources.pusher.validate(arrays, self.radius)
            return
        xp = self.xp
        if not all(bool(xp.all(xp.isfinite(values))) for values in arrays.values()):
            raise ValueError("electron cloud integration produced nonfinite particle state")
        if bool(xp.any(xp.hypot(arrays["x"], arrays["y"]) >= self.radius)):
            raise ValueError("dynamic electrons must lie strictly inside the round chamber")
        if bool(xp.any(arrays["weight"] < 0)):
            raise ValueError("dynamic electron cloud weights must be nonnegative")

    def _commit(self, arrays, generator, *, time, last_turn, counter_updates=None, n_electrons=None, validated=False):
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
            return dict(primary_electrons=0.0, primary_macroparticles=0)
        count = self.parameters.primary_macroparticles
        if self.n_macroparticles + count > self.parameters.max_macroparticles:
            raise ValueError("primary emission exceeds max_macroparticles; no electrons were discarded")
        xp, generator, arrays = self.xp, self._generator(), dict(self._particles)
        angle = xp.asarray(2 * np.pi * generator.random(count))
        nx, ny = -xp.cos(angle), -xp.sin(angle)
        ux, uy, uz = cosine_emission(nx, ny, self.parameters.emission_energy_ev, generator, xp)
        radius_inside = self.radius * (1 - 64 * np.finfo(float).eps)
        new = dict(x=-radius_inside * nx, y=-radius_inside * ny, ux=ux, uy=uy, uz=uz, weight=xp.full(count, number / count))
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
        return dict(primary_electrons=number, primary_macroparticles=count)

    def _drift(self, arrays, duration, generator, diagnostics, hits=None):
        if self._resources.pusher is not None:
            return self._drift_gpu(arrays, duration, generator, diagnostics, hits)
        xp = self.xp
        remaining = xp.full(arrays["x"].size, duration)
        while bool(xp.any(remaining > 0)):
            selected = xp.flatnonzero(remaining > 0)
            ux, uy, uz = (arrays[name][selected] for name in ("ux", "uy", "uz"))
            gamma = electron_gamma(ux, uy, uz, xp)
            vx, vy = const.c * (ux / gamma), const.c * (uy / gamma)
            x, y = arrays["x"][selected], arrays["y"][selected]
            hit_time = first_circle_hit(x, y, vx, vy, self.radius, xp)
            crossing = hit_time <= remaining[selected]
            elapsed = xp.minimum(hit_time, remaining[selected])
            arrays["x"][selected] = x + vx * elapsed
            arrays["y"][selected] = y + vy * elapsed
            remaining[selected] -= elapsed
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
            incident_energy = electron_energy_ev(*(arrays[name][impacted] for name in ("ux", "uy", "uz")), xp)
            incident_weight = arrays["weight"][impacted]
            emitted_yield = true_secondary_yield(incident_energy, self.parameters.secondary_yield_max, self.parameters.secondary_peak_energy_ev,
                                                 self.parameters.secondary_shape, self.parameters.emission_energy_ev, xp)
            emitted_weight = incident_weight * emitted_yield
            emitted_energy = xp.minimum(incident_energy, self.parameters.emission_energy_ev)
            diagnostics["wall_hits"] += int(impacted.size)
            diagnostics["incident_electrons"] += float(xp.sum(incident_weight))
            diagnostics["emitted_electrons"] += float(xp.sum(emitted_weight))
            diagnostics["incident_energy_ev"] += float(xp.sum(incident_weight * incident_energy))
            diagnostics["emitted_energy_ev"] += float(xp.sum(emitted_weight * emitted_energy))
            radius = xp.hypot(arrays["x"][impacted], arrays["y"][impacted])
            nx, ny = -arrays["x"][impacted] / radius, -arrays["y"][impacted] / radius
            ux, uy, uz = cosine_emission(nx, ny, emitted_energy, generator, xp)
            arrays["ux"][impacted], arrays["uy"][impacted], arrays["uz"][impacted] = ux, uy, uz
            radius_inside = self.radius * (1 - 64 * np.finfo(float).eps)
            arrays["x"][impacted], arrays["y"][impacted] = -radius_inside * nx, -radius_inside * ny
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
            # Draw the complete mu batch before azimuth, even for absorbed hits.
            mu = np.sqrt(generator.random(impacted.size))
            azimuth = 2 * np.pi * generator.random(impacted.size)
            ledgers = pusher.emit_wall(arrays, remaining, impacted, mu, azimuth, self.radius, self.parameters.secondary_yield_max,
                                       self.parameters.secondary_peak_energy_ev, self.parameters.secondary_shape, self.parameters.emission_energy_ev)
            # Keep each original reduction and host addition order; batch only D2H.
            totals = xp.stack([xp.sum(values) for values in ledgers]).get()
            diagnostics["wall_hits"] += int(impacted.size)
            for name, value in zip(("incident_electrons", "emitted_electrons", "incident_energy_ev", "emitted_energy_ev"), totals):
                diagnostics[name] += float(value)
        if has_zero:
            alive = xp.flatnonzero(arrays["weight"] > 0)
            for name in arrays:
                arrays[name] = arrays[name][alive]
            hits = hits[alive]
        return hits if had_hits else None

    def advance_to(self, end_time, line_charge=0.0, beam_beta=0.0, *, beam_field=None, witness=None):
        """Stage an interval; coupled witnesses receive its mean cloud E field.

        Beam charge and witness positions are held fixed within a saved slice.
        Cloud fields are rebuilt at each Boris kick; their midpoint quadrature
        provides the transverse slice kick without feeding beam self-fields back.
        """
        end_time = _finite_scalar(end_time, "time")
        if not all(math.isfinite(value) for value in (end_time, line_charge, beam_beta)) or not 0 <= beam_beta <= 1:
            raise ValueError("build-up times/line charge must be finite and beam_beta must be in [0,1]")
        if self.pic is None and (beam_field is not None or witness is not None):
            raise ValueError("PIC beam fields and witnesses require coupled mode")
        if self.pic is not None and line_charge != 0 and beam_field is None:
            raise ValueError("coupled mode requires the actual beam slice field")
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
        arrays, generator, xp = self._arrays(), self._generator(), self.xp
        pusher = self._resources.pusher
        mean_field = None
        if witness is not None:
            mean_field = [xp.zeros_like(values, dtype=xp.float64) for values in witness]
        sigma = self.parameters.beam_sigma
        line_strength = abs(line_charge) / (2 * const.pi * const.epsilon0 * self._beam_normalization) if self.pic is None else 0.0
        field_bound = line_strength * min(self.radius / (2 * sigma**2), 1 / (math.sqrt(2) * sigma)) if self.pic is None else 0.0
        if beam_field is not None:
            field_bound = float(xp.max(xp.hypot(beam_field.ex, beam_field.ey)))
        beam_density_bound = float(xp.max(xp.abs(beam_field.density))) if beam_field is not None else 0.0
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
        elapsed = 0.0
        while elapsed < duration:
            if arrays["x"].size == 0:
                break
            if diagnostics["steps"] >= self.parameters.max_steps:
                raise ValueError("electron cloud interval exceeds max_steps; state and RNG are unchanged")
            remaining = duration - elapsed
            dt = min(step_limit, remaining)
            if self.pic is not None:
                cloud_field = self.pic.solve(arrays["x"], arrays["y"], -const.e * arrays["weight"] / self.source_length)
                density_bound = float(xp.max(xp.abs(cloud_field.density)))
                electric_bound = float(xp.max(xp.hypot(cloud_field.ex, cloud_field.ey))) + field_bound
                density_bound += beam_density_bound
                mesh_length = min(self.pic.geometry.dx, self.pic.geometry.dy)
                # Resolve plasma oscillations and acceleration across one cell.
                frequency_squared = const.e / const.m_e_kg * max(density_bound / const.epsilon0, electric_bound / mesh_length)
                if frequency_squared > 0:
                    dt = min(dt, .2 / math.sqrt(frequency_squared))
            if (line_charge != 0 or self.pic is not None) and arrays["x"].size:
                if pusher is None:
                    gamma = electron_gamma(*(arrays[name] for name in ("ux", "uy", "uz")), xp)
                    transverse_speed = const.c * float(xp.max(xp.hypot(arrays["ux"], arrays["uy"]) / gamma))
                else:
                    transverse_speed = float(xp.max(pusher.transverse_speed(arrays)))
                if transverse_speed > 0:
                    length_scale = mesh_length if self.pic is not None else min(sigma, self.radius)
                    dt = min(dt, .2 * length_scale / transverse_speed)
            # Integrate a roundoff-sized final remainder as part of this step,
            # rather than taking a spurious extra step at an exact step budget.
            if remaining - dt <= 8 * max(math.ulp(duration), math.ulp(elapsed), math.ulp(dt)):
                dt = remaining
            if not elapsed + dt > elapsed:
                raise ValueError("electron cloud time step is below floating-point resolution")
            hits = self._drift(arrays, dt / 2, generator, diagnostics)
            if self.pic is None:
                beam_ex, beam_ey = round_gaussian_beam_field(arrays["x"], arrays["y"], line_charge, sigma, self.radius, xp)
                ex, ey = 0.0, 0.0
            else:
                cloud_field = self.pic.solve(arrays["x"], arrays["y"], -const.e * arrays["weight"] / self.source_length)
                if beam_field is None:
                    ex, ey = self.pic.sample(cloud_field, arrays["x"], arrays["y"])
                    beam_ex, beam_ey = 0.0, 0.0
                else:
                    (ex, ey), (beam_ex, beam_ey) = self.pic.sample_pair(cloud_field, beam_field, arrays["x"], arrays["y"])
                if mean_field is not None:
                    witness_ex, witness_ey = self.pic.sample(cloud_field, *witness)
                    mean_field[0] += witness_ex * (dt / duration)
                    mean_field[1] += witness_ey * (dt / duration)
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
            diagnostics["field_work_ev"] += float(xp.sum(energy_delta))
            self._drift(arrays, dt / 2, generator, diagnostics, hits)
            self._validate_particles(arrays)
            elapsed += dt
            diagnostics["steps"] += 1
        diagnostics["wall_energy_ev"] = diagnostics["incident_energy_ev"] - diagnostics["emitted_energy_ev"]
        diagnostics["n_electrons"] = float(xp.sum(arrays["weight"]))
        diagnostics["n_macroparticles"] = int(arrays["x"].size)
        weighted_energy = (arrays["weight"] *
                           electron_energy_ev(*(arrays[name]
                                                for name in ("ux", "uy", "uz")), xp) if pusher is None else pusher.weighted_energy(arrays))
        diagnostics["kinetic_energy_ev"] = float(xp.sum(weighted_energy))
        if not all(math.isfinite(value) for value in diagnostics.values()):
            raise ValueError("electron cloud diagnostics overflowed; state and RNG are unchanged")
        counter_updates = {
            name: value
            for name, value in diagnostics.items() if name not in {"dt", "n_electrons", "n_macroparticles", "kinetic_energy_ev"}
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
                     validated=True)
        if mean_field is not None:
            diagnostics["mean_ex"], diagnostics["mean_ey"] = mean_field
        return diagnostics
