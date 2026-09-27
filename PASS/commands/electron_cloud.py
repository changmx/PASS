"""Position-local frozen, dilute and coupled electron clouds."""

import hashlib
import logging
from pathlib import Path
import re
import uuid

from PASS.commands.command import Command
from PASS.commands.ecloud.fields import FrozenCloudFields
from PASS.commands.ecloud.interaction import apply_cloud_kicks, stage_frozen_cloud
from PASS.para.schema.electron_cloud import ElectronCloudItem, load_electron_cloud


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
            self.fields = BuildUpCloud(self.configuration, backend=self.backend, dtype=self.dtype)
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
        if not frozen and self.configuration.buildup.magnetic_gradient == 0:
            config_exclude["buildup"] = {"magnetic_gradient"}
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
        """Construct detached resources for a command or joint checkpoint restore."""
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
            return BuildUpCloud(self.configuration, backend=self.backend, dtype=self.dtype, state=state)
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
