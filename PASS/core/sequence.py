from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.utils.helper import convert_keys_to_lower
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const

import json
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

# Priority mapping for sorting commands with same s-position
COMMAND_PRIORITY = {
    "Injection": 0,
    "SortBunch": 100,
    "ReorganizeBunch": 150,
    "Twiss": 200,
    "Marker": 300,
    "Drift": 300,
    "SBend": 300,
    "RBend": 300,
    "Quadrupole": 300,
    "Sextupole": 300,
    "Octupole": 300,
    "Multipole": 300,
    "Kicker": 300,
    "RF": 300,
    "ElSeparator": 300,
    "Exciter": 300,
    "SpaceCharge": 400,
    "WakeField": 500,
    "BeamBeam": 600,
    "ElectronCloud": 700,
    "LumiMonitor": 800,
    "PhaseMonitor": 800,
    "DistMonitor": 800,
    "StatMonitor": 800,
    "ParticleMonitor": 800,
    "Other": 999,
}


def _get_priority(cmd) -> int:
    """Get sort priority for a command based on its type name."""
    return COMMAND_PRIORITY.get(cmd.cmd_type, COMMAND_PRIORITY["Other"])


class CommandSequence:

    def __init__(self, input_file: str, beam_id: int, sim: Simulation):

        self.beam_id = beam_id
        self.cmds = []
        self._load_input(input_file, sim)

    def _load_input(self, input_file: str, sim: Simulation):
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = convert_keys_to_lower(data)

        if "sequence" not in data:
            raise KeyError("JSON root must contain a 'sequence' key")

        for cmd_name, cmd_def in data["sequence"].items():
            cmd_def_with_name = cmd_def.copy()
            cmd_def_with_name["name"] = cmd_name
            cmd = Command.create(self.beam_id, cmd_def_with_name, sim)
            self.cmds.append(cmd)

        self.num_cmd = len(self.cmds)

    def sort(self, eps: float = None) -> None:
        """Sort commands in-place, first by s-position ascending, then by command type priority.

        For commands whose s-values differ by less than *eps*, they are considered
        to share the same s and are ordered by their priority (see COMMAND_PRIORITY).

        Args:
            eps: Tolerance for comparing s-positions. Defaults to const.eps.
        """
        if eps is None:
            eps = const.eps

        def sort_key(cmd):
            # Primary: s-value (rounded to bins of size eps)
            s_bin = round(cmd.s / eps) if eps > 0 else cmd.s
            # Secondary: priority
            return (s_bin, _get_priority(cmd))

        self.cmds.sort(key=sort_key)

    def print(self):
        set_simple_logging()

        logger.info("")
        logger.info(center_string(s=f" Sequence{self.beam_id} "))

        logger.info(f"Sequence ID: {self.beam_id}")
        logger.info(f"Number of Commands: {self.num_cmd}")
        logger.info("")

        set_normal_logging()

        for cmd in self.cmds:
            cmd.print()
