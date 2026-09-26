from typing import List
import logging

from PASS.commands import Command
from PASS.core.simulation import Simulation
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.command_order import sort_commands

logger = logging.getLogger(__name__)


class CommandSequence:

    def __init__(self, input_data: dict, beam_id: int, sim: Simulation):

        self.beam_id = beam_id
        self.cmds = []
        self._load_input(input_data, sim)

    def _load_input(self, data: dict, sim: Simulation):
        if "sequence" not in data:
            raise KeyError("JSON root must contain a 'sequence' key")

        for cmd_name, cmd_def in data["sequence"].items():
            cmd_def_with_name = cmd_def.copy()
            cmd_def_with_name["name"] = cmd_name
            cmd = Command.create(self.beam_id, cmd_def_with_name, sim)
            self.cmds.append(cmd)

        self.num_cmd = len(self.cmds)

    def sort(self, eps: float = None) -> None:
        """Apply the shared position/Order/legacy-priority ordering contract."""
        self.cmds[:] = sort_commands(self.cmds, eps=eps)

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
