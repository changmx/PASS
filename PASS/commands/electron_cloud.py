"""Position-local frozen, prescribed-beam and coupled electron clouds."""

import hashlib
import logging
import math
from pathlib import Path
import re
import uuid

import numpy as np

from PASS.commands.command import Command
from PASS.commands.ecloud.driver import build_slice_intervals
from PASS.commands.ecloud.fields import FrozenCloudFields
from PASS.commands.ecloud.interaction import apply_cloud_kicks, stage_frozen_cloud
from PASS.commands.ecloud.response import _host, read_response
from PASS.para.schema.electron_cloud import ElectronCloudItem, ElectronCloudResponseItem, load_electron_cloud


@Command.register("ElectronCloud")
class ElectronCloud(Command):
    """Apply a frozen-cloud kick or evolve a local dynamic electron cloud.

    The effective interaction length is not a transport length. Dynamic modes
    consume saved SliceSets; only coupled mode feeds their cloud fields back.
    """

    def __init__(self, beam_id, sim, **command_kwargs):
        kwargs = {str(k).lower(): v for k, v in command_kwargs.items()}
        self.cmd_name = str(kwargs.pop("name", "electron_cloud"))
        kwargs.pop("command", None)
        self.beam_id = int(beam_id)
        self.cmd_type = "ElectronCloud"
        self.s = float(kwargs.get("s (m)", kwargs.get("s", 0.0)))
        self.length = 0.0
        self.last_diagnostics = None
        self.saved_fields = []
        self._output_directory = None
        self._output_calls = 0
        self.fields = None
        self.configuration = None
        self.parameters = None
        self.backend = "gpu" if getattr(sim.cfg, "use_gpu", False) else "cpu"
        self.dtype = getattr(sim.cfg, "particle_precision", "float64")
        configurations = getattr(sim.cfg, "electron_cloud", None)
        settings = (configurations[self.beam_id]
                    if configurations is not None and len(configurations) > self.beam_id else load_electron_cloud(sim.cfg.input_data[self.beam_id]))
        self.is_enabled = bool(settings.enabled)
        if not self.is_enabled:
            return
        self.parameters = ElectronCloudItem.model_validate(kwargs)
        self.s = self.parameters.s
        self.is_enabled = self.parameters.is_enabled
        if not self.is_enabled:
            return
        self.configuration_name = self.parameters.configuration
        if self.configuration_name not in settings.configurations:
            raise ValueError(f"ElectronCloud {self.cmd_name!r} references missing configuration {self.configuration_name!r}")
        self.configuration = settings.configurations[self.configuration_name]
        if self.configuration.mode in {"build_up", "coupled"}:
            from PASS.commands.ecloud.buildup import BuildUpCloud
            if self.parameters.slice_set is None:
                raise ValueError("ElectronCloud dynamic modes require a named Slice set")
            self.fields = BuildUpCloud(self.configuration, backend=self.backend, dtype=self.dtype, record_wall_spectra=self.parameters.save_fields)
        else:
            self.fields = FrozenCloudFields(self.configuration, backend=self.backend, dtype=self.dtype)

    def execute_cpu(self, sim):
        return self._execute(sim, "cpu")

    def execute_gpu(self, sim):
        return self._execute(sim, "gpu")

    def _execute(self, sim, backend):
        if not self.is_enabled:
            return False
        if self.backend != backend:
            raise TypeError("ElectronCloud execution backend differs from its configured field resources")
        if self.configuration.mode in {"build_up", "coupled"}:
            return self._execute_buildup(sim)
        turn = int(sim.state.turn)
        beam = sim.beams[self.beam_id]
        diagnostics, staged = stage_frozen_cloud(beam, self.fields, self.parameters.interaction_length)
        diagnostics["turn"] = turn
        wrote_fields = self.parameters.save_fields and self._turn_selected(turn)
        if wrote_fields:
            from PASS.commands.ecloud.io import write_field_snapshot
            directory = self._get_output_directory(sim.cfg)
            path = directory / f"turn_{turn:08d}_call_{self._output_calls:06d}.h5"
            # A failed write may leave a partial diagnostic file. Reserve a
            # new name on retry while keeping all physical state unchanged.
            self._output_calls += 1
            metadata = dict(command=self.cmd_name,
                            beam_id=self.beam_id,
                            turn=turn,
                            s=self.s,
                            interaction_length=self.parameters.interaction_length,
                            configuration=self.configuration.model_dump(mode="json"),
                            diagnostics=diagnostics)
            write_field_snapshot(path, self.fields, metadata)
            self.saved_fields.append(str(path))
        apply_cloud_kicks(beam.particles, staged)
        self.last_diagnostics = diagnostics
        return bool(wrote_fields or diagnostics["max_kick"] > 0)

    def _execute_buildup(self, sim):
        from PASS.commands.ecloud.driver import build_slice_intervals, evolve_buildup
        from PASS.commands.ecloud.io import write_buildup_snapshot
        turn = int(sim.state.turn)
        beam = sim.beams[self.beam_id]
        events = build_slice_intervals(beam,
                                       self.parameters.slice_set,
                                       turn,
                                       self.s,
                                       self.fields.xp,
                                       include_particles=self.configuration.mode == "coupled")
        candidate, diagnostics, staged = evolve_buildup(self.fields, events, turn, beam=beam, interaction_length=self.parameters.interaction_length)
        try:
            if self.parameters.save_fields and self._turn_selected(turn):
                directory = self._get_output_directory(sim.cfg)
                path = directory / f"turn_{turn:08d}_call_{self._output_calls:06d}.h5"
                self._output_calls += 1
                metadata = dict(command=self.cmd_name,
                                beam_id=self.beam_id,
                                turn=turn,
                                s=self.s,
                                interaction_length=self.parameters.interaction_length,
                                configuration=self.configuration.model_dump(mode="json"),
                                diagnostics={
                                    key: value
                                    for key, value in diagnostics.items() if key != "history"
                                })
                write_buildup_snapshot(path, candidate, metadata, diagnostics["history"])
                self.saved_fields.append(str(path))
        except Exception:
            candidate.close()
            raise
        apply_cloud_kicks(beam.particles, staged)
        previous, self.fields = self.fields, candidate
        self.last_diagnostics = diagnostics
        previous.close()
        return True

    def _turn_selected(self, turn):
        for selection in self.parameters.save_turns:
            if len(selection) == 1:
                if turn == selection[0]:
                    return True
            else:
                start, end, step = selection
                if start <= turn <= end and (turn - start) % step == 0:
                    return True
        return False

    def _get_output_directory(self, cfg):
        if self._output_directory is None:
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.cmd_name).strip(".") or "electron_cloud"
            run = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(getattr(cfg, "output_hms", "run")))
            # A fresh invocation never overwrites snapshots from an earlier run.
            token = uuid.uuid4().hex[:12]
            self._output_directory = Path(cfg.output_dir) / "electron_cloud" / f"{run}_{token}" / f"beam{self.beam_id}_{name}"
        return self._output_directory

    def _configuration_identity(self):
        # Preserve the first frozen-state format's identity when adding optional
        # dynamic-only schema fields. Existing frozen sources remain restorable.
        frozen = self.configuration.mode == "frozen"
        config_exclude = {"buildup": True} if frozen else {}
        if not frozen:
            parameters = self.configuration.buildup
            excluded = {
                name
                for name in ("self_fields", "beam_driver", "surface_emission", "primary_sources", "population_control")
                if not getattr(parameters, name)
            }
            if parameters.magnetic_gradient == 0:
                excluded.add("magnetic_gradient")
            config_exclude["buildup"] = excluded
        item_exclude = {"save_fields", "save_turns"} | ({"slice_set"} if frozen else set())
        values = self.configuration.model_dump_json(exclude=config_exclude) + self.parameters.model_dump_json(exclude=item_exclude)
        return hashlib.sha256(values.encode("utf-8")).hexdigest()

    def state_dict(self):
        """Export only this prescribed cloud, not a complete tracking checkpoint."""
        if not self.is_enabled:
            raise ValueError("A disabled ElectronCloud has no cloud state")
        return {
            "format": "PASS-electron-cloud-1",
            "configuration_sha256": self._configuration_identity(),
            "cloud": None if self.fields.state is None else self.fields.state.state_dict(),
        }

    def load_state_dict(self, data):
        """Validate a complete candidate before replacing the current cloud."""
        candidate = self._fields_from_state(data)
        previous, self.fields = self.fields, candidate
        self.last_diagnostics = None
        previous.close()

    def _fields_from_state(self, data):
        """Construct detached resources for a component snapshot restore."""
        if not self.is_enabled:
            raise ValueError("A disabled ElectronCloud has no cloud state")
        if not isinstance(
                data, dict) or data.get("format") != "PASS-electron-cloud-1" or data.get("configuration_sha256") != self._configuration_identity():
            raise ValueError("ElectronCloud state does not match this configuration and interaction point")
        if "cloud" not in data:
            raise ValueError("ElectronCloud state is missing its prescribed source")
        if self.configuration.mode in {"build_up", "coupled"}:
            from PASS.commands.ecloud.buildup import BuildUpCloud
            from PASS.commands.ecloud.state import DynamicElectronCloudState
            state = DynamicElectronCloudState.from_state_dict(data["cloud"])
            return BuildUpCloud(self.configuration,
                                backend=self.backend,
                                dtype=self.dtype,
                                state=state,
                                record_wall_spectra=self.parameters.save_fields)
        analytic = self.configuration.solver == "uniform_round_free_space"
        if analytic != (data["cloud"] is None):
            raise ValueError("ElectronCloud state source does not match its solver")
        state = None
        if not analytic:
            from PASS.commands.ecloud.state import ElectronCloudState
            state = ElectronCloudState.from_state_dict(data["cloud"])
        return FrozenCloudFields(self.configuration, backend=self.backend, dtype=self.dtype, state=state)

    def save_state(self, path):
        """Write a new HDF5 file containing this cloud's reproducible source."""
        from PASS.commands.ecloud.io import write_cloud_state
        write_cloud_state(path, self.state_dict())

    def load_state(self, path):
        from PASS.commands.ecloud.io import read_cloud_state
        self.load_state_dict(read_cloud_state(path))

    def print(self):
        logging.getLogger(__name__).info(
            "S=%.6g, Command=ElectronCloud, Name=%s, Enabled=%s%s", self.s, self.cmd_name, self.is_enabled,
            f", Solver={self.configuration.solver}, InteractionLength={self.parameters.interaction_length:.6g} m" if self.is_enabled else "")


@Command.register("ElectronCloudResponse")
class ElectronCloudResponse(Command):
    """Transverse reduced model with an explicit calibration and finite memory.

    A new command starts with no perturbation history. An artifact contains the
    unperturbed physical cloud response, never a whole-simulation checkpoint.
    """

    def __init__(self, beam_id, sim, **command_kwargs):
        kwargs = {str(k).lower(): v for k, v in command_kwargs.items()}
        self.cmd_name = str(kwargs.pop("name", "electron_cloud_response"))
        kwargs.pop("command", None)
        self.parameters = ElectronCloudResponseItem.model_validate(kwargs)
        self.beam_id, self.cmd_type = int(beam_id), "ElectronCloudResponse"
        self.s, self.length = self.parameters.s, 0.0
        self.is_enabled = self.parameters.is_enabled
        self.backend = "gpu" if getattr(sim.cfg, "use_gpu", False) else "cpu"
        self.response = None
        self.last_turn, self.last_diagnostics = None, None
        self._history, self._finite_next, self._origin = {}, 0, None
        if not self.is_enabled:
            return
        self.response = read_response(self.parameters.response_file)
        metadata = self.response.metadata
        self.displacement_limit = metadata["maximum_centroid_displacement"]
        self.profile_tolerance = metadata["profile_tolerance"]
        for name, attribute in (("maximum_centroid_displacement", "displacement_limit"), ("profile_tolerance", "profile_tolerance")):
            requested = getattr(self.parameters, name)
            if requested is not None:
                if requested > getattr(self, attribute):
                    raise ValueError("Response command cannot relax the calibration's validity limits")
                setattr(self, attribute, requested)
        self.periodic = metadata.get("periodic") if self.parameters.timeline_mode == "periodic" else None
        if self.parameters.timeline_mode == "periodic" and self.periodic is None:
            raise ValueError("Periodic response replay requires a measured and validated periodic artifact")
        lengths = {event.interaction_length for event in self.response.slices}
        if len(lengths) != 1:
            raise ValueError("A response sequence station requires one calibrated interaction length")
        self._interaction_length = lengths.pop()
        self._reference_centroids = np.stack([event.centroid for event in self.response.slices])
        self._reference_profiles = np.stack([event.profile for event in self.response.slices])
        self._profile_scales = np.maximum(np.linalg.norm(self._reference_profiles, axis=(1, 2)), 1e-24)
        self._replay_arrays, self._array_range = None, None

    def _prepare_replay_arrays(self, first, end, xp):
        """Keep only the reusable periodic block, or the current finite rows."""
        if self._array_range != (first, end):
            source_start = 0 if self.periodic is None else first - self.periodic["memory_periods"] * (end - first)
            response = self.response
            self._replay_arrays = (xp.asarray(response.grid_x), xp.asarray(response.grid_y), xp.asarray(response.baseline[first:end]),
                                   xp.asarray(response.kernel[first:end, source_start:end]))
            self._array_range = first, end
        return self._replay_arrays

    def execute_cpu(self, sim):
        return self._execute(sim, "cpu")

    def execute_gpu(self, sim):
        return self._execute(sim, "gpu")

    def _execute(self, sim, backend):
        if not self.is_enabled:
            return False
        if backend != self.backend:
            raise TypeError("ElectronCloudResponse execution backend differs from its configuration")
        xp = np
        if backend == "gpu":
            import cupy as cp
            xp = cp
        turn = int(sim.state.turn)
        if self.last_turn is None and turn != 0:
            raise ValueError("ElectronCloudResponse follows the fresh-start workflow and must begin at turn zero")
        if self.last_turn is not None and turn != self.last_turn + 1:
            raise ValueError("ElectronCloudResponse requires exactly one ordered invocation per turn")
        beam = sim.beams[self.beam_id]
        intervals = build_slice_intervals(beam, self.parameters.slice_set, turn, self.s, xp, include_particles=True)
        if any(event.time_end <= event.time_start for event in intervals):
            raise ValueError("Response capture requires finite explicit Slicer intervals for empty bunch groups")
        n_slices = len(intervals)
        response = self.response
        if self.periodic is None:
            indices = list(range(self._finite_next, self._finite_next + n_slices))
            if indices[-1] >= len(response.slices):
                raise ValueError("Finite response calibration timeline is exhausted; periodic reuse requires independent validation")
        else:
            if n_slices != self.periodic["events_per_period"]:
                raise ValueError("Periodic response slice count differs from calibration")
            indices = list(range(self.periodic["template_start"], self.periodic["template_start"] + n_slices))
        # Reduce on the particle backend; only five moments per slice leave the GPU.
        p = beam.particles
        moments = []
        for event in intervals:
            ids = event.particle_indices
            if not ids.size:
                moments.append(xp.zeros(5, dtype=xp.float64))
                continue
            x, y = p.x[ids].astype(xp.float64), p.y[ids].astype(xp.float64)
            mean_x, mean_y = xp.mean(x), xp.mean(y)
            x, y = x - mean_x, y - mean_y
            moments.append(xp.stack((mean_x, mean_y, xp.mean(x * x), xp.mean(x * y), xp.mean(y * y))))
        moments = _host(xp.stack(moments))
        if not np.all(np.isfinite(moments)):
            raise ValueError("Response witnesses require finite paired x/y coordinates and moments")
        offsets = moments[:, :2] - self._reference_centroids[indices]
        profiles = moments[:, [2, 3, 3, 4]].reshape(n_slices, 2, 2)
        history = dict(self._history)
        origin = intervals[0].time_start if self._origin is None else self._origin
        for position, (event, index) in enumerate(zip(intervals, indices)):
            reference = response.slices[index]
            bunch = beam.bunches[event.bunch_index]
            if (event.bunch_id, event.slice_index) != (reference.bunch_id, reference.slice_index):
                raise ValueError("Response bunch/slice identity differs from its calibration")
            actual = dict(n_real_particles=event.n_real_particles,
                          charge_per_length=event.charge_per_length,
                          beta=event.beam_beta,
                          brho=float(bunch.brho),
                          num_charge=float(bunch.num_charge))
            for name, value in actual.items():
                if not np.isclose(value, getattr(reference, name), rtol=1e-9, atol=0):
                    raise ValueError(f"Response calibration mismatch in {name}; recompute the response")
            if event.particle_indices.size != reference.x.size:
                raise ValueError("Response source macro-particle count differs from calibration")
            duration = reference.time_end - reference.time_start
            if self.periodic is None:
                expected = np.array([reference.time_start, reference.time_end])
                if turn != reference.turn:
                    raise ValueError("Finite response turn differs from its calibration timeline")
            else:
                base = response.slices[self.periodic["template_start"]].time_start
                expected = origin + turn * self.periodic["period"] + np.array([reference.time_start - base, reference.time_end - base])
            measured = np.array([event.time_start, event.time_end])
            tolerance = max(1e-6 * duration, 8 * max(math.ulp(float(value)) for value in measured))
            if np.max(np.abs(measured - expected)) > tolerance:
                raise ValueError("Response arrival times/gaps differ from the calibrated physical timeline")
            difference = np.linalg.norm(profiles[position] - self._reference_profiles[index])
            # Backend reductions can differ by rounding even for identical sources.
            roundoff = 64 * np.finfo(float).eps * self._profile_scales[index]
            roundoff += (64 * np.finfo(float).eps * np.linalg.norm(self._reference_centroids[index]))**2
            if difference > self.profile_tolerance * self._profile_scales[index] + roundoff:
                raise ValueError("Response transverse source profile left the calibrated covariance domain")
            offset = offsets[position]
            if np.linalg.norm(offset) > self.displacement_limit * (1 + 1e-12):
                raise ValueError("Response centroid displacement left the calibrated linear domain")
            history[(turn, position) if self.periodic is not None else index] = offset
        gx, gy, baseline, kernel = self._prepare_replay_arrays(indices[0], indices[-1] + 1, xp)
        if self.periodic is None:
            source_offsets = np.stack([history[source] for source in range(indices[-1] + 1)])
        else:
            source_offsets = np.zeros((self.periodic["memory_periods"] + 1, n_slices, 2))
            for lag in range(min(turn, self.periodic["memory_periods"]) + 1):
                source_offsets[self.periodic["memory_periods"] - lag] = [history[(turn - lag, source)] for source in range(n_slices)]
            source_offsets = source_offsets.reshape(-1, 2)
        dynamic = xp.einsum("jiab,ib->ja", kernel, xp.asarray(source_offsets))
        staged, statuses = [], []
        for position, interval in enumerate(intervals):
            ids = interval.particle_indices
            if not ids.size:
                continue
            kick, valid = response._sample_baseline_arrays(baseline[position], p.x[ids], p.y[ids], gx, gy, xp)
            kick += dynamic[position]
            previous_px, previous_py = p.px[ids].astype(xp.float64), p.py[ids].astype(xp.float64)
            px = xp.asarray(previous_px + kick[:, 0], dtype=p.px.dtype)
            py = xp.asarray(previous_py + kick[:, 1], dtype=p.py.dtype)
            ratio = 1 + p.dp[ids].astype(xp.float64)
            valid_momentum = xp.all(
                xp.isfinite(px) & xp.isfinite(py) & xp.isfinite(ratio) & (ratio > 0)
                & (previous_px**2 + previous_py**2 < ratio**2)
                & (px.astype(xp.float64)**2 + py.astype(xp.float64)**2 < ratio**2))
            statuses.append(xp.concatenate((valid.astype(xp.float64), xp.stack((valid_momentum, xp.max(xp.hypot(kick[:, 0], kick[:, 1])))))))
            staged.append((ids, px, py))
        max_kick = 0.0
        if statuses:
            # One synchronization validates every slice before any beam/history commit.
            for status in _host(xp.stack(statuses)):
                response._validate_probe_status(status[:3])
                if not status[3] or not math.isfinite(status[4]):
                    raise ValueError("ElectronCloudResponse kick exceeds the forward thin-lens momentum domain")
                max_kick = max(max_kick, status[4])
        apply_cloud_kicks(beam.particles, staged)
        if self.periodic is not None:
            history = {key: value for key, value in history.items() if key[0] >= turn - self.periodic["memory_periods"]}
        self._history, self._origin, self.last_turn = history, origin, turn
        self._finite_next += n_slices
        self.last_diagnostics = dict(turn=turn,
                                     max_kick=max_kick,
                                     centroid_offsets=np.asarray(offsets).tolist(),
                                     timeline_mode=self.parameters.timeline_mode,
                                     calibrated_response_file=self.parameters.response_file)
        return bool(staged)

    def print(self):
        logging.getLogger(__name__).info("ElectronCloudResponse %s at s=%g m, file=%s, timeline=%s", self.cmd_name, self.s,
                                         self.parameters.response_file, self.parameters.timeline_mode)
