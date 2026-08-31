from PASS.core.simulation import Simulation
from PASS.core.bunch import BunchInfo
from PASS.core.beam import Beam
from PASS.core.config import Config
from PASS.commands import Command
from PASS.core.sequence import CommandSequence
from PASS.core.timing import ExecutionProfiler
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.plot.plot_main import plot_main

import logging

logger = logging.getLogger(__name__)


class Executor:

    def __init__(self):
        pass

    @staticmethod
    def _command_executed(result) -> bool:
        """Normalize command return values during the bool-protocol migration.

        Legacy commands commonly omit ``return`` and therefore produce
        ``None``; those invocations retain the historical counted behavior.
        New commands should return an explicit bool, where only ``False``
        denotes a no-op.
        """
        if result is None:
            return True
        if isinstance(result, bool):
            return result
        raise TypeError(
            "Command execute_cpu/execute_gpu must return bool or None, "
            f"got {type(result).__name__}"
        )

    def run(self, sim: Simulation, seqs: list[CommandSequence]):

        cfg = sim.cfg
        state = sim.state
        beams = sim.beams
        total_turns = cfg.num_turn
        profiler = ExecutionProfiler(sim)
        profiler.start_run()

        set_simple_logging()
        logger.info("")
        logger.info(center_string(" Start Simulation "))
        set_normal_logging()

        current_turn = None
        try:
            for turn in range(total_turns):
                current_turn = turn
                state.turn = turn
                profiler.start_turn(turn)

                for seq in seqs:
                    for cmd in seq.cmds:
                        profiler.start_command(cmd, sim)
                        executed = True
                        try:
                            if cfg.use_cpu:
                                result = cmd.execute_cpu(sim)
                            elif cfg.use_gpu:
                                result = cmd.execute_gpu(sim)
                            else:
                                raise ValueError(f"unknown backend {cfg.backend}")
                            # Existing commands return None.  Only an
                            # explicit False means that this invocation did
                            # no work and should be omitted from command
                            # timing statistics.
                            executed = self._command_executed(result)
                        finally:
                            profiler.stop_command(cmd, sim, turn, executed=executed)

                profiler.finish_turn(turn)
                if profiler.should_log_turn(turn):
                    if profiler.mode == "off":
                        logger.info(f"Turn: {turn}/{total_turns}")
                    else:
                        logger.info(profiler.format_progress(turn))
        finally:
            # Preserve timing for a turn interrupted by an exception when
            # possible, then always print the partial or complete summary.
            if (
                current_turn is not None
                and profiler.mode != "off"
                and current_turn not in profiler.turn_seconds
            ):
                profiler.finish_turn(current_turn)
            profiler.print_summary()

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
