import logging
import sys

from PASS.core.simulation import Simulation
from PASS.core.bunch import BunchInfo
from PASS.core.beam import Beam
from PASS.core.config import Config
from PASS.commands import Command
from PASS.core.sequence import CommandSequence
from PASS.core.timing import ExecutionProfiler
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.plot.plot_main import plot_main

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
        raise TypeError("Command execute_cpu/execute_gpu must return bool or None, "
                        f"got {type(result).__name__}")

    def run(self, sim: Simulation, seqs: list[CommandSequence], *, stop_requested=None):
        """Run complete turns, honoring an optional stop request between turns."""

        cfg = sim.cfg
        state = sim.state
        total_turns = cfg.num_turn
        from PASS.utils.sc_coverage import validate_sc_coverage
        validate_sc_coverage(sim, seqs)
        from PASS.commands.wake.wake_timing import prepare_wake_tracking
        prepare_wake_tracking(sim, seqs)
        profiler = ExecutionProfiler(sim)
        profiler.start_run()

        set_simple_logging()
        logger.info("")
        logger.info(center_string(" Start Simulation "))
        set_normal_logging()

        current_turn = None
        stopped = False
        try:
            for turn in range(total_turns):
                if stop_requested is not None and stop_requested():
                    stopped = True
                    logger.info("Stop requested at turn boundary before turn %d", turn)
                    break
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
            pending_error = sys.exception()
            output_error = None
            try:
                if stopped and cfg.use_gpu:
                    try:
                        profiler._synchronize_gpu(force=True)
                    except Exception as exc:
                        output_error = exc
                        logger.exception("Failed to synchronize GPU before finalizing stopped run")
                for seq in seqs:
                    for cmd in seq.cmds:
                        finalize = getattr(cmd, "finalize", None)
                        if finalize is not None:
                            try:
                                finalize(sim)
                            except Exception as exc:
                                if output_error is None:
                                    output_error = exc
                                logger.exception("Failed to finalize command output")
                # Preserve timing for a turn interrupted by an exception when
                # possible, then print the partial or complete summary.
                if (current_turn is not None and profiler.mode != "off" and current_turn not in profiler.turn_seconds):
                    profiler.finish_turn(current_turn)
                profiler.print_summary()
            except KeyboardInterrupt as interruption:
                # A cleanup interruption must not erase an earlier failure.
                if isinstance(pending_error, Exception):
                    raise pending_error
                if output_error is not None:
                    raise output_error from interruption
                raise
            # An output failure must not be reported as a clean interruption.
            # Preserve the original tracking error when one already exists.
            if output_error is not None and (pending_error is None or isinstance(pending_error, KeyboardInterrupt)):
                raise output_error from pending_error

        set_simple_logging()
        logger.info("")
        logger.info(center_string(" Simulation Stopped " if stopped else " Simulation Completed "))
        set_normal_logging()

        if cfg.is_plot and not stopped:
            set_simple_logging()
            logger.info("")
            logger.info(center_string(" Start Plotting "))
            set_normal_logging()

            plot_main(sim)

            set_simple_logging()
            logger.info("")
            logger.info(center_string(" Plotting Completed "))
            set_normal_logging()
        return False if stopped else None
