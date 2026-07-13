from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const

import numpy as np
import copy
import logging

logger = logging.getLogger(__name__)


@Command.register("reorganizebunch")
class ReorganizeBunch(Command):
    """Reorganize bunch command: redistribute particle indices among bunches.

    This command changes how particles are grouped into bunches by updating
    ``start_idx`` and ``end_idx`` of each :class:`BunchInfo`. It does **not**
    modify any particle coordinates (x, px, y, py, z, dp, tag, ...).

    Note: this command only reassigns bunch index ranges. Actual debunching
    (bunch lengthening due to RF adjustments) and rebunching (bunch compression)
    are achieved by tuning RF parameters, not by this command.

    Two modes are supported:

    - ``mode = "merge"``: merge multiple bunches into fewer bunches.
      The total particle array is re-divided as evenly as possible.
    - ``mode = "split"``: split existing bunches into more bunches.
      Each bunch is divided into equal sub-bunches.

    Parameters (JSON keys, case-insensitive)
    ----------------------------------------
    name : str
        Command name (auto-filled by the sequence loader).
    s (m) : float
        S-position of the command in the ring [m].
    mode : str
        ``"merge"`` or ``"split"``.
    start turn : int
        Turn from which the reorganization takes effect (inclusive, 0-indexed).
    end turn : int
        Turn until which the reorganization is active (exclusive).
        After ``end turn``, the original bunch structure is restored.
    new num bunch : int
        New number of bunches after reorganization.
    """

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = 0.0
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]

        self.mode: str = kwargs.get("mode", "merge").lower()
        if self.mode not in ("merge", "split"):
            raise ValueError(
                f"ReorganizeBunch mode must be 'merge' or 'split', got '{self.mode}' "
                f"in {self.cmd_name}"
            )

        self.start_turn: int = int(kwargs.get("start turn", 0))
        self.end_turn: int = int(kwargs.get("end turn", -1))  # -1 means no upper limit
        self.new_num_bunch: int = int(kwargs.get("new num bunch", 1))

        if self.new_num_bunch < 1:
            raise ValueError(
                f"new num bunch must be >= 1 in {self.cmd_name}, "
                f"got {self.new_num_bunch}"
            )

        # Store original bunch structure for restoration after end_turn
        self._original_ranges: list[tuple[int, int]] | None = None

        super().__init__()

    def _is_active(self, turn: int) -> bool:
        """Check if reorganization is active for the given turn."""
        if turn < self.start_turn:
            return False
        if self.end_turn >= 0 and turn >= self.end_turn:
            return False
        return True

    def _apply_reorganize(self, beam: Beam):
        """Redistribute particle indices among bunches.

        The total particle array is divided as evenly as possible into
        ``new_num_bunch`` groups. Only ``start_idx`` and ``end_idx`` are
        modified — no particle coordinates are touched.
        """
        Np_total = beam.Np_total
        old_num_bunch = beam.num_bunch
        new_num_bunch = self.new_num_bunch

        if new_num_bunch == old_num_bunch:
            logger.info(
                f"[ReorganizeBunch] new_num_bunch ({new_num_bunch}) == old_num_bunch, "
                f"nothing to do"
            )
            return

        # Save original ranges on first call
        if self._original_ranges is None:
            self._original_ranges = [
                (b.start_idx, b.end_idx) for b in beam.bunches
            ]

        logger.info(
            f"[ReorganizeBunch] {self.cmd_name}: {old_num_bunch} bunches -> "
            f"{new_num_bunch} bunches (Np_total={Np_total})"
        )

        # Evenly divide total particles into new_num_bunch groups
        base = Np_total // new_num_bunch
        remainder = Np_total % new_num_bunch

        if new_num_bunch > old_num_bunch:
            # Split: create additional BunchInfo objects
            template_bunch = beam.bunches[-1]
            while len(beam.bunches) < new_num_bunch:
                new_bunch = copy.deepcopy(template_bunch)
                beam.bunches.append(new_bunch)

        # Redistribute indices
        idx = 0
        for k in range(new_num_bunch):
            count = base + (1 if k < remainder else 0)
            beam.bunches[k].start_idx = idx
            beam.bunches[k].end_idx = idx + count
            beam.bunches[k].Np = count
            idx += count

        # If merging (fewer bunches), keep extra bunch objects but mark them inactive
        if new_num_bunch < old_num_bunch:
            for k in range(new_num_bunch, old_num_bunch):
                beam.bunches[k].start_idx = Np_total
                beam.bunches[k].end_idx = Np_total
                beam.bunches[k].Np = 0

        beam.num_bunch = new_num_bunch

        for k in range(new_num_bunch):
            b = beam.bunches[k]
            logger.info(
                f"[ReorganizeBunch] bunch{k}: start_idx={b.start_idx}, "
                f"end_idx={b.end_idx}, Np={b.Np}"
            )

    def _restore_original(self, beam: Beam):
        """Restore the original bunch structure after end_turn."""
        if self._original_ranges is None:
            return

        logger.info(f"[ReorganizeBunch] {self.cmd_name}: restoring original bunch structure")

        orig_num = len(self._original_ranges)

        for k, (s, e) in enumerate(self._original_ranges):
            if k < len(beam.bunches):
                beam.bunches[k].start_idx = s
                beam.bunches[k].end_idx = e
                beam.bunches[k].Np = e - s
            else:
                new_bunch = copy.deepcopy(beam.bunches[0])
                new_bunch.start_idx = s
                new_bunch.end_idx = e
                new_bunch.Np = e - s
                beam.bunches.append(new_bunch)

        while len(beam.bunches) > orig_num:
            beam.bunches.pop()

        beam.num_bunch = orig_num
        self._original_ranges = None

    def execute_cpu(self, sim: Simulation):
        beam = sim.beams[self.beam_id]
        turn = sim.state.turn

        if not self._is_active(turn):
            if self.end_turn >= 0 and turn == self.end_turn:
                self._restore_original(beam)
            return

        if turn == self.start_turn:
            self._apply_reorganize(beam)

    def execute_gpu(self, sim: Simulation):
        self.execute_cpu(sim)

    def print(self):
        set_simple_logging()
        logger.info(
            f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, "
            f"Mode={self.mode}, NewNumBunch={self.new_num_bunch}, "
            f"StartTurn={self.start_turn}, EndTurn={self.end_turn}"
        )
        set_normal_logging()
