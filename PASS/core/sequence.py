from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.utils.helper import convert_keys_to_lower
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string

import json
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


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
