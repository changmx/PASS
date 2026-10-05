"""Beam-beam command and simulation-owned, same-turn collision coordination."""

from enum import Enum
import hashlib
import logging
import math
from pathlib import Path
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
    Each simulation starts at turn zero; cached field data are disposable.
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
        self.next_turn = 0

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
        if self.next_turn != 0:
            raise RuntimeError("BeamBeam tracking must start from turn 0 with a new simulation")
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

    def _get_luminosity_recorder(self, key):
        from PASS.commands.collision.luminosity import LuminosityRecorder
        from PASS.utils.table_io import table_path

        if key not in self._luminosity_recorders:
            name, occurrence = key
            configuration = self.configurations[name]
            luminosity = configuration.luminosity
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
