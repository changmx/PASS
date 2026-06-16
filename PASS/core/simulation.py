from PASS.core.config import Config
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.state import State
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS import __version__

import logging
import json
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class Simulation:

    def __init__(self, cfg: Config, beams: list[Beam], state: State):

        self.cfg = cfg
        self.beams = beams
        self.state = state

    def print(self):
        print_copyright()
        self.cfg.print()
        for beam in self.beams:
            beam.print()


def print_copyright():
    set_simple_logging()

    logger.info(center_string(f" PASS v{__version__} ", fillchar='-'))
    logger.info(f"Copyright (C) 2025-{datetime.now().year} Institute of Modern Physics, Chinese Academy of Sciences")
    logger.info("Website: https://github.com/changmx/PASS")
    logger.info(center_string("-", fillchar='-'))
    
    set_normal_logging()
