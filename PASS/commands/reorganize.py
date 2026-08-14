"""ReorganizeBunch: switch the ring to a new harmonic (bucket grid).

All particles are sorted by longitudinal position and reassigned to the
nearest center of the NEW bucket grid (C/new_harmonic), the beam harmonic
number is updated, and the bunch structure is rebuilt as one bunch per new
bucket.  The switch is permanent and takes effect on ``start_turn``.
"""
from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.utils.logger import set_simple_logging, set_normal_logging

from PASS.commands.sort_bunch import regroup_particles

import logging

logger = logging.getLogger(__name__)


@Command.register("reorganizebunch")
class ReorganizeBunch(Command):
    """Reorganize bunch command: switch to a new harmonic (bucket grid).

    Parameters (JSON keys, case-insensitive)
    ----------------------------------------
    name : str
        Command name (auto-filled by the sequence loader).
    s (m) : float
        S-position of the command in the ring [m].
    start turn : int
        Turn at which the harmonic switch takes effect (inclusive, 0-indexed).
    new harmonic number : int
        Harmonic number (bucket count) after reorganization.
    """

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = 0.0
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]

        self.start_turn: int = int(kwargs.get("start turn", 0))
        if kwargs.get("new harmonic number") is None:
            raise ValueError(
                f"ReorganizeBunch {self.cmd_name}: 'New harmonic number' "
                f"is required"
            )
        self.new_harmonic: int = int(kwargs["new harmonic number"])
        if self.new_harmonic < 1:
            raise ValueError(
                f"new harmonic number must be >= 1 in {self.cmd_name}"
            )

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(
            f"S={self.s:.4f}, Command={self.cmd_type:s}, "
            f"Name={self.cmd_name:s}, StartTurn={self.start_turn}, "
            f"NewHarmonic={self.new_harmonic}"
        )
        set_normal_logging()

    def execute_cpu(self, sim: Simulation):
        beam = sim.beams[self.beam_id]
        turn = sim.state.turn
        if turn != self.start_turn:
            return
        set_simple_logging()
        logger.info(
            f"[ReorganizeBunch] {self.cmd_name}: switching harmonic "
            f"{beam.harmonic_number} -> {self.new_harmonic}"
        )
        regroup_particles(beam, new_harmonic=self.new_harmonic)
        for k, b in enumerate(beam.bunches):
            logger.info(
                f"[ReorganizeBunch] bunch{k}: harmonic_id={b.harmonic_id}, "
                f"z_center={b.z_center:.3f}, start_idx={b.start_idx}, "
                f"end_idx={b.end_idx}, Np={b.Np}"
            )
        set_normal_logging()

    def execute_gpu(self, sim: Simulation):
        self.execute_cpu(sim)
