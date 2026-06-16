from PASS.core.particle import ParticlePool
from PASS.core.bunch import BunchInfo
from PASS.core.beam import Beam
from PASS.core.config import Config
from PASS.core.executor import Executor
from PASS.core.simulation import Simulation
from PASS.core.state import State
from PASS.commands import Command
from PASS.core.sequence import CommandSequence

from PASS.utils.logger import setup_logging, set_simple_logging, set_normal_logging

from PASS.utils import helper

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def main(beam0_path: str, beam1_path: str | None = None, is_cal_phase: bool = True):

    cfg = Config()
    cfg.load_input(beam0_path, beam1_path)

    setup_logging(log_file=cfg.get_log_path())

    try:
        beams = []
        for i in range(cfg.num_beam):
            beams.append(Beam(cfg.input_path[i], cfg, is_cal_phase))

        state = State()

        sim = Simulation(cfg, beams, state)
        sim.print()

        seqs = []
        for i in range(cfg.num_beam):
            seqs.append(CommandSequence(cfg.input_path[i], i))

        for seq in seqs:
            seq.print()

        executor = Executor()
        executor.run(sim, seqs)

    except KeyboardInterrupt:
        logger.info("Interrupted by the user")

    except Exception as e:
        logger.exception("Error occurred")

    finally:
        pass


if __name__ == "__main__":
    main(r"C:\Users\changmx\Documents\PASS\input\beam0.json")
