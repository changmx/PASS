from PASS.core.simulation import Simulation
from PASS.core.bunch import BunchInfo
from PASS.core.beam import Beam
from PASS.core.config import Config
from PASS.commands import Command
from PASS.core.sequence import CommandSequence
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.plot.plot_main import plot_main

import logging

logger = logging.getLogger(__name__)


class Executor:

    def __init__(self):
        pass

    def run(self, sim: Simulation, seqs: list[CommandSequence]):

        cfg = sim.cfg
        state = sim.state
        beams = sim.beams

        total_turns = cfg.num_turn

        set_simple_logging()
        logger.info("")
        logger.info(center_string(" Start Simulation "))
        set_normal_logging()

        for turn in range(total_turns):
            logger.info(f"Turn: {turn}/{total_turns}")
            state.turn = turn

            for seq in seqs:
                for cmd in seq.cmds:
                    if cfg.use_cpu:
                        cmd.execute_cpu(sim)
                    elif cfg.use_gpu:
                        cmd.execute_gpu(sim)
                    else:
                        raise ValueError(f"unknown backend {cfg.backend}")

        set_simple_logging()
        logger.info("")
        logger.info(center_string(" Simulation Completed "))
        set_normal_logging()

        if cfg.is_plot:
            set_simple_logging()
            logger.info("")
            logger.info(center_string(" Start Plotting "))
            set_normal_logging()

            plot_main(sim)

            set_simple_logging()
            logger.info("")
            logger.info(center_string(" Plotting Completed "))
            set_normal_logging()
