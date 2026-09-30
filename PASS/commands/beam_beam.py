"""Beam-beam command and simulation-owned, same-turn collision coordination."""

import copy
from dataclasses import asdict
from enum import Enum
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import re

import numpy as np

from PASS.commands.command import Command
from PASS.utils.constants import const

logger = logging.getLogger(__name__)


class CommandResult(Enum):
    """WAITING retains a cursor; SKIPPED advances without counting timed work."""

    DONE = "done"
    SKIPPED = "skipped"
    WAITING = "waiting"


class CollisionCoordinator:
    """Meet both beams once per IP occurrence without adjusting physical time.

    Both sequences complete a common turn before either starts the next one.
    State export is limited to that boundary; cached field data are disposable.
    """

    def __init__(self, sim):
        self.sim = sim
        self.configurations = getattr(sim.cfg, "beam_beam_configurations", {})
        self.resources = {}
        self.current_event = None
        self.current_positions = None
        self.event_counts = {}
        self.pair_counts = {}
        self.last_diagnostics = {}
        self._events = {}
        self._completed = set()
        self._pairs = {}
        self._positions = {}
        self._bunch_ids = {}
        self._prepared = False
        self._at_turn_boundary = True
        self._luminosity_recorders = {}
        self._luminosity_calculators = {}
        self._luminosity_pending = {}
        self.next_turn = int(getattr(sim.state, "next_turn", 0))

    def is_enabled(self, configuration_id):
        if not getattr(self.sim.cfg, "beam_beam_enabled", False):
            return False
        if configuration_id not in self.configurations:
            raise ValueError(f"Unknown Beam beam configuration {configuration_id!r}")
        configuration = self.configurations[configuration_id]
        luminosity = getattr(configuration, "luminosity", None)
        return configuration.mode != "weak-weak" or bool(luminosity and luminosity.enabled)

    def prepare(self, sequences):
        """Bind stable pairs and occurrences; reject cyclic/cross-turn IP plans."""
        if self._events or not self._at_turn_boundary:
            raise RuntimeError("Cannot compile collision sequences during an unfinished turn")
        self._pairs.clear()
        self._positions.clear()
        self._bunch_ids.clear()
        plans = {}
        for sequence in sequences:
            beam_id = int(sequence.beam_id)
            if beam_id in plans:
                raise ValueError(f"Beam {beam_id} has more than one execution sequence")
            occurrences = {}
            plans[beam_id] = []
            for command in sequence.cmds:
                if command.cmd_type != "BeamBeam" or not self.is_enabled(command.configuration_id):
                    continue
                configuration_id = command.configuration_id
                if beam_id not in self.configurations[configuration_id].beams:
                    raise ValueError(f"Beam {beam_id} is not a participant in {configuration_id!r}")
                occurrence = occurrences.get(configuration_id, 0)
                command.occurrence = occurrence
                occurrences[configuration_id] = occurrence + 1
                plans[beam_id].append((configuration_id, occurrence))
                self._positions.setdefault((configuration_id, occurrence), {})[beam_id] = float(command.s)
        active = {name for plan in plans.values() for name, occurrence in plan}
        if active:
            if len(self.sim.beams) != 2 or set(plans) != {0, 1}:
                raise ValueError("Beam-beam tracking requires one sequence for each of two beams")
            if plans[0] != plans[1]:
                raise ValueError("Beam-beam IP/occurrence order must agree within each common turn; "
                                 f"cross-turn or cyclic rendezvous is unsupported: beam 0={plans[0]}, beam 1={plans[1]}")
            for configuration_id in active:
                self._bind_pairs(configuration_id)
            if len({tuple(pairs) for pairs in self._pairs.values()}) != 1:
                raise ValueError("Every active IP must use the same fixed bunch-pair bijection")
        if self.next_turn:
            expected = {
                key
                for key in self._positions
                if getattr(self.configurations[key[0]], "luminosity", None) and self.configurations[key[0]].luminosity.enabled
            }
            if set(self._luminosity_recorders) != expected:
                raise ValueError("Collision checkpoint luminosity histories do not match the active IP occurrences")
            for (name, occurrence), recorder in self._luminosity_recorders.items():
                if set(recorder.references) != {f"{a}:{b}" for a, b in self._pairs[name]}:
                    raise ValueError("Collision checkpoint luminosity histories do not match the fixed bunch pairs")
        self._prepared = True

    def _bind_pairs(self, configuration_id):
        configuration = self.configurations[configuration_id]
        if sorted(configuration.beams) != [0, 1]:
            raise ValueError(f"{configuration_id!r} must reference beams 0 and 1 exactly once")
        identities = []
        for beam_id in configuration.beams:
            ids = [int(bunch.bunch_id) for bunch in self.sim.beams[beam_id].bunches]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Beam {beam_id} has duplicate bunch identities")
            identities.append(set(ids))
            self._bunch_ids[beam_id] = frozenset(ids)
        supplied = configuration.bunch_pairs
        pairs = sorted((int(a), int(b)) for a, b in supplied) if supplied is not None else [(value, value) for value in sorted(identities[0])]
        first_ids = {a for a, b in pairs}
        second_ids = {b for a, b in pairs}
        if (len(identities[0]) != len(identities[1]) or len(pairs) != len(identities[0]) or first_ids != identities[0]
                or second_ids != identities[1]):
            raise ValueError(f"{configuration_id!r} bunch pairs must be a complete bijection of stable bunch IDs")
        self._pairs[configuration_id] = sorted(pairs if configuration.beams[0] == 0 else [(b, a) for a, b in pairs])

    def begin_turn(self, turn):
        if self._events or not self._at_turn_boundary:
            raise RuntimeError("The previous collision turn has not reached its common boundary")
        if turn != self.next_turn:
            raise ValueError(f"Collision scheduler expected turn {self.next_turn}, got {turn}")
        self._at_turn_boundary = False
        self._completed.clear()

    def finish_turn(self, turn):
        if self._events:
            raise RuntimeError(f"Unconsumed collision events at turn boundary: {list(self._events)}")
        if any(getattr(bunch, "collision_frame", None) is not None for beam in self.sim.beams for bunch in beam.bunches):
            raise RuntimeError("A collision frame is still active at the common turn boundary")
        for key, values in self._luminosity_pending.items():
            rows = self._luminosity_recorders[key].record(turn, **values)
            # The final row is the IP total, or its sole bunch-pair record.
            row = rows[-1]
            logger.info("BeamBeam %s/IP%d turn %d: luminosity=%.8e cm^-2 s^-1, factor=%.8g, loss=%.8g", key[0], key[1], turn, row[8], row[10],
                        row[11])
        self._luminosity_pending.clear()
        self._at_turn_boundary = True
        self.next_turn = int(turn) + 1
        self._completed.clear()

    def check_command_frame(self, command):
        """Block ordinary commands from reading temporary collision coordinates."""
        frames = [getattr(bunch, "collision_frame", None) for bunch in self.sim.beams[command.beam_id].bunches]
        frames = [frame for frame in frames if frame is not None]
        if not frames:
            return
        configuration_id = getattr(command, "configuration_id", getattr(command, "configuration", None))
        allowed = command.cmd_type in {"CrossingAngle", "BeamBeam"}
        if command.cmd_type == "Slicer":
            allowed = getattr(command, "purpose", None) == "beam_beam"
        if not allowed or any(frame.get("frame") != "collision" or frame.get("configuration") != configuration_id for frame in frames):
            raise RuntimeError(f"{command.cmd_type} {getattr(command, 'cmd_name', '')!r} cannot execute in the active collision frame")

    def arrive(self, configuration_id, beam_id, occurrence=0):
        """Execute the physical event once, then consume completion on both sides."""
        if not self.is_enabled(configuration_id):
            return CommandResult.SKIPPED
        if not self._prepared:
            raise RuntimeError("CollisionCoordinator.prepare must run before the first BeamBeam event")
        configuration = self.configurations[configuration_id]
        if beam_id not in configuration.beams:
            raise ValueError(f"Beam {beam_id} does not participate in {configuration_id!r}")
        event_id = (configuration_id, int(self.sim.state.turn), int(occurrence))
        if event_id in self._completed:
            raise RuntimeError(f"Collision event {event_id} has already been consumed by both beams")
        event = self._events.setdefault(event_id, {"ready": set(), "consumed": set(), "complete": False})
        if event["complete"]:
            if beam_id in event["consumed"]:
                raise RuntimeError(f"Beam {beam_id} consumed collision event {event_id} twice")
            self._events.pop(event_id)
            self._completed.add(event_id)
            return CommandResult.SKIPPED
        event["ready"].add(beam_id)
        if event["ready"] != set(configuration.beams):
            return CommandResult.WAITING
        self.current_event = event_id
        self.current_positions = self._positions[(configuration_id, int(occurrence))]
        try:
            self._collide(configuration_id, configuration)
        finally:
            self.current_event = None
            self.current_positions = None
        event["complete"] = True
        event["consumed"].add(beam_id)
        self.event_counts[configuration_id] = self.event_counts.get(configuration_id, 0) + 1
        return CommandResult.DONE

    def _collide(self, configuration_id, configuration):
        from PASS.commands.collision.interaction import collide_bunch_pair

        beams = self.sim.beams
        bunches = [{int(bunch.bunch_id): bunch for bunch in beam.bunches} for beam in beams]
        for beam_id in configuration.beams:
            if frozenset(bunches[beam_id]) != self._bunch_ids[beam_id]:
                raise RuntimeError("Bunch identities changed while fixed beam-beam pairing was active")
        for a, b in self._pairs[configuration_id]:
            self._check_reference_times(bunches[0][a], bunches[1][b], configuration_id)
        calculator = None
        key = (configuration_id, self.current_event[2])
        luminosity = getattr(configuration, "luminosity", None)
        if luminosity and luminosity.enabled:
            recorder = self._get_luminosity_recorder(key)
            turn = int(self.sim.state.turn)
            if recorder.should_sample(turn) or turn == self.sim.cfg.num_turn - 1:
                from PASS.commands.collision.luminosity import LuminosityCalculator

                if key not in self._luminosity_calculators:
                    p = beams[0].particles
                    self._luminosity_calculators[key] = LuminosityCalculator(p.xp, p.x.dtype)
                calculator = self._luminosity_calculators[key]
        times, frequencies, populations, overlaps = [], [], [], []
        diagnostics = []
        for a, b in self._pairs[configuration_id]:
            first, second = configuration.beams
            paired = {0: bunches[0][a], 1: bunches[1][b]}
            frequency = self._luminosity_frequency(luminosity, paired[0], paired[1]) if calculator is not None else None
            kwargs = {"luminosity": calculator} if calculator is not None else {}
            result = collide_bunch_pair(self.sim, configuration, beams[first], paired[first], beams[second], paired[second], **kwargs)
            if calculator is not None:
                overlaps.append(result.pop("luminosity_overlap_m2"))
                numbers = result.pop("luminosity_populations")
                populations.append(numbers if first == 0 else numbers[::-1])
                times.append(float(paired[0].t0))
                frequencies.append(frequency)
            diagnostics.append({"bunch_pair": (a, b), "result": result})
        self.pair_counts[configuration_id] = self.pair_counts.get(configuration_id, 0) + len(diagnostics)
        self.last_diagnostics[configuration_id] = diagnostics
        if calculator is not None:
            xp = beams[0].particles.xp
            self._luminosity_pending[key] = dict(pair_ids=self._pairs[configuration_id],
                                                 overlaps=xp.stack(overlaps),
                                                 times=times,
                                                 frequencies=frequencies,
                                                 populations=populations,
                                                 xp=xp)

    def _get_luminosity_recorder(self, key, *, path=None):
        from PASS.commands.collision.luminosity import LuminosityRecorder
        from PASS.utils.table_io import table_path

        if key not in self._luminosity_recorders:
            name, occurrence = key
            configuration = self.configurations[name]
            luminosity = configuration.luminosity
            if path is None:
                slug = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:48]
                digest = hashlib.sha256(name.encode()).hexdigest()[:10]
                path = table_path(
                    Path(self.sim.cfg.output_dir) / "luminosity" / f"{self.sim.cfg.output_hms}_{slug}_{digest}_ip{occurrence}.tfs",
                    luminosity.output_format)
            metadata = {
                "Configuration": name,
                "Occurrence": occurrence,
                "Mode": configuration.mode,
                "SourceMethods": ",".join(str(configuration.sources[str(beam)].method) for beam in configuration.beams),
                "SourceSolvers": ",".join(str(configuration.sources[str(beam)].solver) for beam in configuration.beams),
                "Geometry": "ultrarelativistic common-frame thin-slice overlap; pre-current-kick densities",
                "Frequency": "prescribed" if luminosity.collision_frequency is not None else "reference beta*c/circumference"
            }
            self._luminosity_recorders[key] = LuminosityRecorder(path,
                                                                 interval=luminosity.sample_interval_turns,
                                                                 reference_luminosity=luminosity.reference_luminosity,
                                                                 metadata=metadata,
                                                                 output_format=luminosity.output_format)
        return self._luminosity_recorders[key]

    @staticmethod
    def _luminosity_frequency(luminosity, bunch_a, bunch_b):
        if luminosity.collision_frequency is not None:
            return luminosity.collision_frequency
        frequencies = [float(bunch.beta) * const.c / float(bunch.circum) for bunch in (bunch_a, bunch_b)]
        if not all(math.isfinite(value) and value > 0 for value in frequencies) or not np.isclose(*frequencies, rtol=1e-12, atol=0):
            raise ValueError("Luminosity needs matching reference revolution frequencies or an explicit Collision frequency (Hz)")
        return frequencies[0]

    def finalize(self):
        for calculator in self._luminosity_calculators.values():
            calculator.close()
        self._luminosity_calculators.clear()
        for recorder in self._luminosity_recorders.values():
            recorder.close()

    @staticmethod
    def _check_reference_times(bunch_a, bunch_b, configuration_id):
        times = [float(bunch_a.t0), float(bunch_b.t0)]
        if not all(math.isfinite(value) for value in times):
            raise ValueError(f"{configuration_id!r} reference arrival times must be finite")
        flight = max(float(bunch.circum) / (float(bunch.beta) * const.c) for bunch in (bunch_a, bunch_b))
        tolerance = max(1e-15, 256 * math.ulp(max(abs(times[0]), abs(times[1]), flight)))
        if abs(times[0] - times[1]) > tolerance:
            raise ValueError(f"{configuration_id!r} assumes synchronous reference encounters: "
                             f"t0_a={times[0]:.17g} s, t0_b={times[1]:.17g} s, "
                             f"difference={times[1]-times[0]:.6g} s exceeds {tolerance:.6g} s; WAITING cannot correct physical timing")

    def _configuration_digest(self):
        values = {
            "enabled": bool(getattr(self.sim.cfg, "beam_beam_enabled", False)),
            "configurations": {
                name: configuration.model_dump(mode="json", by_alias=True)
                for name, configuration in self.configurations.items()
            }
        }
        return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def state_dict(self):
        """Export scheduler state only; particles/references must be saved with it."""
        if not self._at_turn_boundary or self._events or self.current_event is not None or self._luminosity_pending:
            raise RuntimeError("Collision checkpoint requires a common completed-turn boundary")
        if any(getattr(bunch, "collision_frame", None) is not None for beam in self.sim.beams for bunch in beam.bunches):
            raise RuntimeError("Collision checkpoint requires ordinary PASS particle frames")
        data = {
            "format": "PASS-collision-1",
            "configuration_sha256": self._configuration_digest(),
            "next_turn": self.next_turn,
            "event_counts": dict(self.event_counts),
            "pair_counts": dict(self.pair_counts)
        }
        if self._luminosity_recorders:
            data["luminosity"] = [
                dict(configuration=name, occurrence=occurrence, path=str(recorder.path.resolve()), state=recorder.state_dict())
                for (name, occurrence), recorder in sorted(self._luminosity_recorders.items())
            ]
        return data

    def load_state_dict(self, data):
        if not self._at_turn_boundary or self._events:
            raise RuntimeError("Cannot restore collision state into an unfinished turn")
        if data.get("format") != "PASS-collision-1" or data.get("configuration_sha256") != self._configuration_digest():
            raise ValueError("Collision checkpoint does not match this configuration")
        next_turn = data.get("next_turn")
        if isinstance(next_turn, bool) or not isinstance(next_turn, int) or next_turn < 0:
            raise ValueError("Collision checkpoint has an invalid next turn")
        counts = {}
        for kind in ("event_counts", "pair_counts"):
            values = data.get(kind, {})
            if not isinstance(values, dict) or any(name not in self.configurations or type(value) is not int or value < 0
                                                   for name, value in values.items()):
                raise ValueError(f"Collision checkpoint has invalid {kind}")
            counts[kind] = dict(values)
        if self._luminosity_recorders:
            raise RuntimeError("Restore luminosity into an unused collision coordinator")
        try:
            for saved in data.get("luminosity", []):
                name, occurrence = saved["configuration"], saved["occurrence"]
                if (name not in self.configurations or type(occurrence) is not int or occurrence < 0 or not self.configurations[name].luminosity
                        or not self.configurations[name].luminosity.enabled or (name, occurrence) in self._luminosity_recorders):
                    raise ValueError("Collision checkpoint contains invalid luminosity IP identities")
                recorder = self._get_luminosity_recorder((name, occurrence), path=saved["path"])
                recorder.load_state_dict(saved["state"], next_turn=next_turn)
        except BaseException:
            self._luminosity_recorders.clear()
            raise
        self.next_turn = next_turn
        self.event_counts, self.pair_counts = counts["event_counts"], counts["pair_counts"]
        self.resources.clear()
        self.last_diagnostics.clear()


def get_collision_coordinator(sim):
    """Lazily create a simulation-owned coordinator without particle allocation."""
    coordinator = getattr(sim, "collision", None)
    if coordinator is None:
        coordinator = CollisionCoordinator(sim)
        sim.collision = coordinator
    return coordinator


@Command.register("beambeam")
class BeamBeam(Command):
    """The explicit IP rendezvous node on one beam's sequence."""

    def __init__(self, beam_id, sim, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}
        self.beam_id = int(beam_id)
        self.s = float(kwargs.pop("s (m)"))
        self.cmd_type = "BeamBeam"
        self.cmd_name = kwargs.pop("name", "BeamBeam")
        self.configuration_id = kwargs.pop("configuration")
        self.length = 0.0
        self.occurrence = 0

    def execute_cpu(self, sim):
        return get_collision_coordinator(sim).arrive(self.configuration_id, self.beam_id, self.occurrence)

    def execute_gpu(self, sim):
        return get_collision_coordinator(sim).arrive(self.configuration_id, self.beam_id, self.occurrence)

    def finalize(self, sim):
        get_collision_coordinator(sim).finalize()

    def print(self):
        logger.info("S=%.4f, Command=BeamBeam, Name=%s, Configuration=%s", self.s, self.cmd_name, self.configuration_id)


def _legacy_electron_cloud_inputs(inputs, *, frozen_defaults=True):
    """Normalize a zero magnetic gradient and optional legacy frozen fields."""
    inputs = copy.deepcopy(inputs)

    def value(mapping, name, default):
        return next((item for key, item in mapping.items() if str(key).casefold() == name), default)

    for data in inputs:
        block = value(data, "electron cloud", {})
        configurations = value(block, "configurations", {})
        frozen = set()
        for name, configuration in configurations.items():
            buildup = value(configuration, "build up", value(configuration, "buildup", None))
            if isinstance(buildup, dict):
                for key in list(buildup):
                    if str(key).casefold() in {"magnetic gradient (t/m)", "magnetic_gradient"} and buildup[key] == 0:
                        buildup.pop(key)
            if not frozen_defaults or value(configuration, "mode", "frozen") != "frozen":
                continue
            frozen.add(name)
            for key in list(configuration):
                if str(key).casefold() in {"build up", "buildup"} and configuration[key] is None:
                    configuration.pop(key)
        for command in value(data, "sequence", {}).values():
            if not isinstance(command, dict) or str(value(command, "command", "")).casefold() != "electroncloud":
                continue
            if value(command, "configuration", None) in frozen:
                for key in list(command):
                    if str(key).casefold() in {"slice set", "slice_set"} and command[key] is None:
                        command.pop(key)
    return inputs


def _checkpoint_input_digest(sim, sequences, *, legacy_electron_cloud=False):
    sequence = [[(command.cmd_name, command.cmd_type, float(command.s), getattr(command, "order", None)) for command in item.cmds]
                for item in sequences]
    loaded_programs = []
    for item in sequences:
        for command in item.cmds:
            if command.cmd_type == "RFCavity":
                components = []
                for component in command.components:
                    programs = {
                        name: {
                            "origin": getattr(component, name).origin,
                            "times": getattr(component, name).times.tolist(),
                            "values": getattr(component, name).values.tolist()
                        }
                        for name in ("frequency", "voltage", "phase")
                    }
                    components.append({"harmonic": component.harmonic, "programs": programs})
                loaded_programs.append((item.beam_id, command.cmd_name, components))
            elif command.cmd_type == "Bump":
                loaded_programs.append((item.beam_id, command.cmd_name, command.waveform.tolist(), command.waveform_bounds.tolist()))
    inputs = _legacy_electron_cloud_inputs(sim.cfg.input_data, frozen_defaults=legacy_electron_cloud)
    values = {"inputs": inputs, "sequence": sequence, "precision": sim.cfg.particle_precision, "loaded_programs": loaded_programs}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _host_checkpoint_array(value):
    return np.array(value.get() if hasattr(value, "get") else value, copy=True)


def _capture_command_state(command):
    kind = command.cmd_type
    if kind == "WakeField":
        return command.state_dict()
    if kind in {"ElectronCloud", "IBS", "ElectronCooler"}:
        return command.state_dict() if command.is_enabled else None
    if kind == "Injection":
        if not command._finished or any(source.Np_injected != source.planned_count for source in command.inj_bunchs):
            raise ValueError("Collision checkpoints require every Injection batch to be complete")
        return {
            "rng":
            command.rng.getstate(),
            "executed":
            sorted(command._executed),
            "completed_batches":
            sorted(command._completed_batches),
            "sources": [{
                "bunch_id": source.bunch_id,
                "Np_injected": source.Np_injected,
                "Np_inj_curTurn": source.Np_inj_curTurn,
                "saved_init_dist": source._saved_init_dist
            } for source in command.inj_bunchs]
        }
    if kind == "StatMonitor":
        if command._output_failed or command._gpu_pending or any(command._pending_rows.values()) or command._tfs_dirty:
            raise ValueError("Flush StatMonitor output successfully before capturing a collision checkpoint")
    if kind == "ParticleMonitor":
        return {"buffer": _host_checkpoint_array(command.buffer), "recorded_end": command._recorded_end}
    if kind == "PhaseAdvanceMonitor":
        return {
            "written": [window.written for window in command.windows],
            "active": {
                index: {
                    name: _host_checkpoint_array(value)
                    for name, value in vars(runtime).items()
                }
                for index, runtime in command._active_windows.items()
            }
        }
    return None


def capture_collision_state(sim, sequences):
    """Capture an owned Python checkpoint at a common completed-turn boundary.

    No files or GPU caches are serialized. All injection must have completed;
    monitor output buffers must have been flushed by normal finalization.
    Particle and tune-monitor accumulation needed for continuation is retained.
    """
    coordinator = get_collision_coordinator(sim)
    collision = coordinator.state_dict()
    next_turn = int(getattr(sim.state, "next_turn", 0))
    if next_turn <= 0 or next_turn != coordinator.next_turn or sim.state.turn != next_turn - 1:
        raise ValueError("Capture requires both beams to have completed the same full turn")
    if sim.cfg.use_gpu:
        sim.beams[0].particles.xp.cuda.runtime.deviceSynchronize()
    beams = []
    for beam in sim.beams:
        p = beam.particles
        arrays = {name: _host_checkpoint_array(getattr(p, name)) for name in ("x", "px", "y", "py", "z", "dp", "tag", "lost_turn", "lost_position")}
        injection = getattr(beam, "injection_state", None)
        if injection is not None and (injection.remaining != 0 or injection.reserved_ids is not None or np.any(arrays["tag"] == 0)):
            raise ValueError("Collision checkpoints do not support pending injection reservations")
        references = []
        for bunch in beam.bunches:
            names = ("bunch_id", "harmonic_id", "harmonic_number", "start_idx", "end_idx", "Np", "Nrp", "Np_sur", "t0", "Ek", "gamma", "beta", "p0",
                     "p0_kg", "brho", "m0", "qm_ratio", "num_charge", "ratio", "circum")
            references.append({name: getattr(bunch, name) for name in names})
        program = beam.reference_program
        beams.append({
            "beam_id": beam.beam_id,
            "particles": arrays,
            "bunches": references,
            "clock": {
                "origin": program.origin,
                "times": program.times.copy(),
                "values": program.values.copy()
            },
            "injection_batches": None if injection is None else [asdict(batch) for batch in injection.batches]
        })
    commands = [{
        "identity": (sequence.beam_id, command.cmd_name, command.cmd_type),
        "state": _capture_command_state(command)
    } for sequence in sequences for command in sequence.cmds]
    return {
        "format": "PASS-beam-beam-state-1",
        "input_sha256": _checkpoint_input_digest(sim, sequences),
        "collision": collision,
        "simulation": {
            name: getattr(sim.state, name)
            for name in ("turn", "next_turn", "time", "revolution", "Ek")
        },
        "beams": beams,
        "commands": commands
    }


def _stage_command_state(command, data, next_turn, xp):
    """Validate all command state using detached objects before the commit."""
    kind = command.cmd_type
    if kind in {"IBS", "ElectronCooler"}:
        if not command.is_enabled:
            if data is not None:
                raise ValueError(f"Disabled {kind} cannot restore active random state")
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Enabled {kind} requires checkpoint random state")
        candidate = copy.copy(command)
        candidate.load_state_dict(copy.deepcopy(data))
        return {name: getattr(candidate, name) for name in ("_rngs", "_entropy", "_calls", "last_diagnostics")}
    if kind == "ElectronCloud":
        if not command.is_enabled:
            if data is not None:
                raise ValueError("Disabled ElectronCloud cannot restore an active cloud source")
            return {}
        fields = command._fields_from_state(data)
        if command.configuration.mode in {"build_up", "coupled"} and fields.state.last_turn is not None and fields.state.last_turn >= next_turn:
            fields.close()
            raise ValueError("ElectronCloud build-up state is not earlier than the checkpoint's next turn")
        return {"fields": fields, "last_diagnostics": None}
    if kind == "StatMonitor":
        _capture_command_state(command)
    if kind == "WakeField":
        candidate = copy.copy(command)
        candidate.load_state_dict(copy.deepcopy(data))
        if any(state.last_turn is not None and state.last_turn >= next_turn for state in candidate.group_states):
            raise ValueError("Wake history extends beyond the checkpoint boundary")
        for config, components, state in zip(candidate.configuration.groups, candidate.component_groups, candidate.group_states):
            if config.history == "state":
                expected_modes = set(range(len(components))) if state.last_time is not None else set()
                if set(state.mode_amplitudes) != expected_modes:
                    raise ValueError("Wake checkpoint is missing or has inconsistent persistent mode vectors")
        # WakeField normally validates convolution buffers lazily on its next
        # execution. A joint restore must finish that validation before commit.
        plans = {}
        if any(state.convolution is not None for state in candidate.group_states):
            from PASS.commands.wake.execution import GroupExecution
            from PASS.commands.wake.convolution import ConvolutionState
            from PASS.commands.wake.time_convolution import TimeConvolutionState
            backend = "cpu" if xp is np else "gpu"
            plans[backend] = [
                GroupExecution(config, components, backend) for config, components in zip(candidate.configuration.groups, candidate.component_groups)
            ]
            for state, plan in zip(candidate.group_states, plans[backend]):
                if state.convolution is None:
                    continue
                state_type = TimeConvolutionState if plan.config.solver == "time_fft" else ConvolutionState
                state.convolution = state_type.restore(state.convolution, plan.convolution)
                if state.last_turn != state.convolution.start_turn + state.convolution.count - 1:
                    raise ValueError("Wake convolution history count does not match its last turn")
        return {
            "group_states": candidate.group_states,
            "_execution_plans": plans,
            "last_coefficients": None,
            "last_sources": None,
            "last_diagnostics": None
        }
    if kind == "Injection":
        rng = random.Random()
        rng.setstate(data["rng"])
        if len(data["sources"]) != len(command.inj_bunchs):
            raise ValueError("Injection source count differs from the checkpoint")
        for source, saved in zip(command.inj_bunchs, data["sources"]):
            if saved["bunch_id"] != source.bunch_id or saved["Np_injected"] != source.planned_count:
                raise ValueError("Checkpoint contains incomplete or mismatched injection")
            if type(saved["Np_inj_curTurn"]) is not int or not 0 <= saved["Np_inj_curTurn"] <= source.planned_count:
                raise ValueError("Checkpoint contains an invalid injection batch count")
        executed = set(data["executed"])
        completed = {tuple(value) for value in data["completed_batches"]}
        if any(type(turn) is not int or not 0 <= turn < next_turn for turn in executed):
            raise ValueError("Checkpoint contains invalid injection execution turns")
        if any(
                len(value) != 2 or type(value[0]) is not int or type(value[1]) is not int or value[0] < 0 or not 0 <= value[1] < next_turn
                for value in completed):
            raise ValueError("Checkpoint contains invalid completed injection batches")
        return {
            "rng": rng,
            "_executed": executed,
            "_completed_batches": completed,
            "_finished": True,
            "_checkpoint_injection_sources": copy.deepcopy(data["sources"])
        }
    if kind == "ParticleMonitor":
        buffer = np.asarray(data["buffer"])
        recorded_end = data["recorded_end"]
        if buffer.shape != command.buffer.shape or buffer.dtype != command.buffer.dtype:
            raise ValueError("ParticleMonitor checkpoint buffer does not match")
        expected_end = (max(command.start_turn, min(next_turn, command.end_turn))
                        if command.max_tag >= 1 and command.num_record_turn > 0 else command.start_turn)
        if type(recorded_end) is not int or recorded_end != expected_end:
            raise ValueError("ParticleMonitor checkpoint has invalid recorded turns")
        return {"buffer": xp.asarray(buffer.copy()), "_recorded_end": recorded_end, "_tables_written": False}
    if kind == "PhaseAdvanceMonitor":
        from PASS.commands.monitor.phase_advance import _WindowRuntime
        if len(data["written"]) != len(command.windows) or any(type(value) is not bool for value in data["written"]):
            raise ValueError("PhaseAdvanceMonitor checkpoint windows do not match")
        expected_written = [bool(command.enable and window.end <= next_turn) for window in command.windows]
        expected_active = {index for index, window in enumerate(command.windows) if command.enable and window.start < next_turn < window.end}
        if data["written"] != expected_written or set(data["active"]) != expected_active:
            raise ValueError("PhaseAdvanceMonitor checkpoint is missing or misclassifies observation windows")
        active = {}
        for index, values in data["active"].items():
            if type(index) is not int or not 0 <= index < len(command.windows):
                raise ValueError("PhaseAdvanceMonitor checkpoint window index is invalid")
            spec = command.windows[index]
            if not spec.start < next_turn < spec.end or data["written"][index]:
                raise ValueError("PhaseAdvanceMonitor checkpoint window is inconsistent with the turn")
            template = command._allocate_window_runtime()
            if set(values) != set(vars(template)):
                raise ValueError("PhaseAdvanceMonitor checkpoint arrays do not match")
            arrays = {}
            for name, value in values.items():
                value, target = np.asarray(value), getattr(template, name)
                if value.shape != target.shape or value.dtype != target.dtype or not np.all(np.isfinite(value)):
                    raise ValueError(f"Invalid PhaseAdvanceMonitor checkpoint array {name}")
                arrays[name] = xp.asarray(value.copy())
            active[index] = _WindowRuntime(**arrays)
        return {"_active_windows": active, "_checkpoint_window_written": list(data["written"])}
    if data is not None:
        raise ValueError(f"Unexpected checkpoint state for command {kind}")
    return {}


def restore_collision_state(sim, sequences, data):
    """Validate the complete checkpoint, then restore both beams together.

    Input validation and device allocation finish before any live arrays or
    references change. Device failure during the final commit is not rollbackable.
    The next source must come from another explicit Slicer execution.
    """
    from PASS.commands.injection import InjectionBatch, InjectionState
    from PASS.core.bunch import set_reference_energy

    coordinator = getattr(sim, "collision", None)
    if coordinator is not None:
        coordinator.state_dict()
    input_matches = data.get("input_sha256") == _checkpoint_input_digest(sim, sequences)
    if not input_matches:
        input_matches = data.get("input_sha256") == _checkpoint_input_digest(sim, sequences, legacy_electron_cloud=True)
    if data.get("format") != "PASS-beam-beam-state-1" or not input_matches:
        raise ValueError("Collision checkpoint inputs or execution sequence do not match")
    state = data["simulation"]
    if set(state) != {"turn", "next_turn", "time", "revolution", "Ek"}:
        raise ValueError("Collision checkpoint simulation fields do not match")
    next_turn = state["next_turn"]
    if type(next_turn) is not int or not 0 < next_turn <= sim.cfg.num_turn or state["turn"] != next_turn - 1:
        raise ValueError("Collision checkpoint has an invalid common turn boundary")
    if not all(np.isfinite(value) for value in state.values()):
        raise ValueError("Collision checkpoint has nonfinite simulation state")
    candidate_coordinator = CollisionCoordinator(sim)
    candidate_coordinator.load_state_dict(data["collision"])
    if candidate_coordinator.next_turn != next_turn:
        raise ValueError("Collision and simulation checkpoint turns do not match")
    detached_sequences = []
    for sequence in sequences:
        detached = copy.copy(sequence)
        detached.cmds = [copy.copy(command) for command in sequence.cmds]
        detached_sequences.append(detached)
    candidate_coordinator.prepare(detached_sequences)
    expected_events = {}
    for name, occurrence in candidate_coordinator._positions:
        expected_events[name] = expected_events.get(name, 0) + next_turn
    expected_pairs = {name: count * len(candidate_coordinator._pairs[name]) for name, count in expected_events.items()}
    if candidate_coordinator.event_counts != expected_events or candidate_coordinator.pair_counts != expected_pairs:
        raise ValueError("Collision checkpoint event/pair counters do not match its completed turns and sequence")
    if len(data["beams"]) != len(sim.beams):
        raise ValueError("Collision checkpoint beam count does not match")
    staged_beams = []
    for beam, saved in zip(sim.beams, data["beams"]):
        if saved["beam_id"] != beam.beam_id:
            raise ValueError("Collision checkpoint beam identity does not match")
        p, xp = beam.particles, beam.particles.xp
        fields = {"x", "px", "y", "py", "z", "dp", "tag", "lost_turn", "lost_position"}
        if set(saved["particles"]) != fields:
            raise ValueError("Collision checkpoint particle fields do not match")
        arrays = {}
        for name, value in saved["particles"].items():
            value, target = np.asarray(value), getattr(p, name)
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"Collision checkpoint particle array {name} shape/dtype does not match")
            arrays[name] = value.copy()
        tags = arrays["tag"]
        ids = np.abs(tags.astype(np.int64))
        if len(np.unique(ids[ids > 0])) != np.count_nonzero(ids) or np.any(ids > len(tags)):
            raise ValueError("Collision checkpoint particle identities are not unique pool identities")
        live = tags > 0
        if any(not np.all(np.isfinite(arrays[name][live])) for name in p.real_fields):
            raise ValueError("Collision checkpoint contains nonfinite live coordinates")
        delta, px, py = (arrays[name][live].astype(float) for name in ("dp", "px", "py"))
        if np.any(delta <= -1) or np.any((1 + delta)**2 <= px * px + py * py):
            raise ValueError("Collision checkpoint has nonphysical live momentum")
        current = {bunch.bunch_id: bunch for bunch in beam.bunches}
        references = saved["bunches"]
        reference_fields = {
            "bunch_id", "harmonic_id", "harmonic_number", "start_idx", "end_idx", "Np", "Nrp", "Np_sur", "t0", "Ek", "gamma", "beta", "p0", "p0_kg",
            "brho", "m0", "qm_ratio", "num_charge", "ratio", "circum"
        }
        if any(set(reference) != reference_fields for reference in references):
            raise ValueError("Collision checkpoint bunch reference fields do not match")
        if len(references) != len(current) or {value["bunch_id"] for value in references} != set(current):
            raise ValueError("Collision checkpoint bunch identities do not match")
        expected_start = 0
        for reference in sorted(references, key=lambda value: (value["start_idx"], value["end_idx"])):
            bunch = current[reference["bunch_id"]]
            if not all(np.isfinite(value) for value in reference.values()):
                raise ValueError("Collision checkpoint has nonfinite bunch reference state")
            for name in ("start_idx", "end_idx", "Np", "bunch_id", "harmonic_id", "harmonic_number", "Nrp", "Np_sur"):
                if type(reference[name]) is not int:
                    raise ValueError("Collision checkpoint bunch ranges and identities must be integers")
            if reference["start_idx"] != expected_start or not expected_start <= reference["end_idx"] <= len(tags):
                raise ValueError("Collision checkpoint bunch ranges must partition the pool")
            if reference["Np"] != reference["end_idx"] - reference["start_idx"]:
                raise ValueError("Collision checkpoint bunch count differs from its range")
            # Np_sur is legacy metadata and is not refreshed by regrouping.
            if reference["Np_sur"] < 0 or reference["Nrp"] < 0:
                raise ValueError("Collision checkpoint bunch population is invalid")
            expected_start = reference["end_idx"]
            for name in ("m0", "qm_ratio", "num_charge", "ratio", "circum", "harmonic_id", "harmonic_number"):
                if reference[name] != getattr(bunch, name):
                    raise ValueError(f"Collision checkpoint bunch {name} does not match the configured reference")
            candidate = copy.copy(bunch)
            set_reference_energy(candidate, reference["Ek"] + bunch.m0)
            for name in ("gamma", "beta", "p0", "p0_kg", "brho"):
                if not np.isclose(reference[name], getattr(candidate, name), rtol=64 * np.finfo(float).eps, atol=0):
                    raise ValueError(f"Collision checkpoint contains inconsistent reference {name}")
        if expected_start != len(tags):
            raise ValueError("Collision checkpoint bunch ranges do not cover the pool")
        clock, program = saved["clock"], beam.reference_program
        if clock["origin"] != program.origin or not np.array_equal(clock["times"], program.times) or not np.array_equal(
                clock["values"], program.values):
            raise ValueError("Collision checkpoint prescribed clock does not match")
        injection = None
        if saved["injection_batches"] is not None:
            if np.any(tags == 0):
                raise ValueError("Collision checkpoint contains pending injection particles")
            batches = []
            for value in saved["injection_batches"]:
                if set(value) != {"first_id", "count", "turn", "index"} or any(type(item) is not int for item in value.values()):
                    raise ValueError("Collision checkpoint has invalid injection batch metadata")
                if value["first_id"] < 1 or value["count"] <= 0 or value["index"] < 0 or value["first_id"] + value["count"] > len(
                        tags) + 1 or not 0 <= value["turn"] < next_turn:
                    raise ValueError("Collision checkpoint injection batch is outside the pool or time boundary")
                batches.append(InjectionBatch(**value))
            covered = 1
            for batch in sorted(batches, key=lambda value: value.first_id):
                if batch.first_id != covered:
                    raise ValueError("Collision checkpoint injection batches must partition particle identities")
                covered += batch.count
            if covered != len(tags) + 1:
                raise ValueError("Collision checkpoint injection batches do not cover the pool")
            injection = InjectionState.__new__(InjectionState)
            injection.remaining, injection.reserved_ids, injection.batches = 0, None, batches
        elif getattr(beam, "injection_state", None) is not None:
            raise ValueError("Collision checkpoint is missing Injection metadata")
        staged_beams.append((beam, {name: xp.asarray(value) for name, value in arrays.items()}, copy.deepcopy(references), injection))
    commands = {(sequence.beam_id, command.cmd_name, command.cmd_type): command for sequence in sequences for command in sequence.cmds}
    if len(data["commands"]) != len(commands) or {tuple(value["identity"]) for value in data["commands"]} != set(commands):
        raise ValueError("Collision checkpoint command identities do not match")
    staged_commands = []
    for saved in data["commands"]:
        command = commands[tuple(saved["identity"])]
        xp = sim.beams[command.beam_id].particles.xp
        staged_commands.append((command, _stage_command_state(command, saved["state"], next_turn, xp)))
    if sim.cfg.use_gpu:
        sim.beams[0].particles.xp.cuda.runtime.deviceSynchronize()
    # All user-controlled data and required allocations have passed validation.
    for beam, arrays, references, injection in staged_beams:
        for name, value in arrays.items():
            getattr(beam.particles, name)[...] = value
        current = {bunch.bunch_id: bunch for bunch in beam.bunches}
        for reference in references:
            bunch = current[reference["bunch_id"]]
            # The reference API validated consistency above. Preserve its saved
            # rounding exactly so normalized particle momenta do not change.
            bunch.__dict__.update(reference)
            bunch.collision_frame = None
            for slices in bunch.slice_sets.values():
                slices.invalidate()
        if injection is not None:
            beam.injection_state = injection
    for command, values in staged_commands:
        sources = values.pop("_checkpoint_injection_sources", None)
        written = values.pop("_checkpoint_window_written", None)
        previous_cloud_fields = command.fields if command.cmd_type == "ElectronCloud" and "fields" in values else None
        command.__dict__.update(values)
        if previous_cloud_fields is not None:
            previous_cloud_fields.close()
        if sources is not None:
            for source, saved in zip(command.inj_bunchs, sources):
                source.Np_injected, source.Np_inj_curTurn, source._saved_init_dist = saved["Np_injected"], saved["Np_inj_curTurn"], saved[
                    "saved_init_dist"]
        if written is not None:
            for window, value in zip(command.windows, written):
                window.written = value
    sim.state.__dict__.update(state)
    sim.collision = candidate_coordinator
    sim._collision_state_restored = True
