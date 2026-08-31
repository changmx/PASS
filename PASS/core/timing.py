"""Execution timing and progress reporting for PASS simulations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import logging
import time

from PASS.utils.logger import center_string, set_normal_logging, set_simple_logging

logger = logging.getLogger(__name__)


TIMING_MODES = {"off", "turn", "command", "synchronized-command"}


@dataclass
class CommandTiming:
    """Accumulated timing for one command class."""

    calls: int = 0
    total_seconds: float = 0.0
    calls_by_turn: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    seconds_by_turn: dict[int, float] = field(default_factory=lambda: defaultdict(float))


@dataclass
class _GpuEvents:
    start: object
    end: object


class ExecutionProfiler:
    """Collect command timings, turn timings, and ETA for an Executor run.

    ``command`` mode measures GPU device time with CUDA events and synchronizes
    once at the end of each turn. ``synchronized-command`` measures command
    wall time and synchronizes before and after every GPU command, so command
    rows include Python and I/O work. ``turn`` measures only turn wall time,
    while ``off`` disables all timing work.
    """

    def __init__(self, sim):
        timing = getattr(sim.cfg, "timing", {}) or {}
        self.mode = str(timing.get("mode", "command")).lower()
        self.log_interval = int(timing.get("log_interval", 10))
        self.warmup_turns = int(timing.get("warmup_turns", 1))
        self.include_io = bool(timing.get("include_io", True))

        self.total_turns = int(getattr(sim.cfg, "num_turn", 0))
        self.use_gpu = bool(getattr(sim.cfg, "use_gpu", False))
        self.command_timings: dict[str, CommandTiming] = {}
        self.turn_seconds: dict[int, float] = {}
        self.turn_calls: dict[int, int] = defaultdict(int)
        self._turn_start: float | None = None
        self._run_start: float | None = None
        self._last_command_start: float | None = None
        self._gpu_events: dict[int, _GpuEvents] = {}
        self._pending_event_records: list[tuple[CommandTiming, int, _GpuEvents]] = []
        self._cupy = None
        self._gpu_synchronized = False

        if self.mode not in TIMING_MODES:
            logger.warning(
                "Unknown timing mode %r; using 'command'. Supported modes: %s",
                self.mode,
                ", ".join(sorted(TIMING_MODES)),
            )
            self.mode = "command"
        if self.log_interval < 1:
            logger.warning("Timing log interval must be >= 1; using 10.")
            self.log_interval = 10
        if self.warmup_turns < 0:
            logger.warning("Timing warmup turns must be >= 0; using 0.")
            self.warmup_turns = 0

        if self.mode == "command" and self.use_gpu:
            self._load_cupy()

    def _load_cupy(self):
        if self._cupy is not None:
            return self._cupy
        try:
            import cupy as cp
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "GPU command timing requires CuPy; use timing mode 'turn' "
                "or install the optional CUDA dependencies."
            ) from exc
        self._cupy = cp
        return cp

    def start_run(self):
        if self.mode != "off":
            self._run_start = time.perf_counter()

    def start_turn(self, turn: int):
        if self.mode == "off":
            return
        if self._run_start is None:
            self.start_run()
        self._turn_start = time.perf_counter()
        self._gpu_synchronized = False

    def start_command(self, cmd, sim):
        if self.mode == "off" or self.mode == "turn":
            return

        if self.mode == "synchronized-command" and self.use_gpu:
            self._synchronize_gpu(force=True)
            self._last_command_start = time.perf_counter()
            return

        if self.mode == "command" and self.use_gpu:
            cp = self._load_cupy()
            events = self._gpu_events.get(id(cmd))
            if events is None:
                events = _GpuEvents(cp.cuda.Event(), cp.cuda.Event())
                self._gpu_events[id(cmd)] = events
            events.start.record()
            return

        self._last_command_start = time.perf_counter()

    def stop_command(self, cmd, sim, turn: int, executed: bool = True):
        if self.mode == "off" or self.mode == "turn":
            return

        # Commands are called from the executor on every turn, but some of
        # them intentionally do no work on most turns (for example,
        # Injection between its configured injection turns).  A command may
        # return ``False`` to exclude such a no-op invocation from command
        # counts and elapsed time while preserving the existing ``None``
        # return convention used by other commands.
        if executed is False:
            self._last_command_start = None
            return

        record = self.command_timings.setdefault(cmd.cmd_type, CommandTiming())
        record.calls += 1
        record.calls_by_turn[turn] += 1
        self.turn_calls[turn] += 1

        if self.mode == "command" and self.use_gpu:
            events = self._gpu_events[id(cmd)]
            events.end.record()
            # The elapsed time is read after finish_turn synchronizes the
            # stream. Store the event pair until then.
            self._pending_event_records.append((record, turn, events))
            return

        if self.mode == "synchronized-command" and self.use_gpu:
            self._synchronize_gpu(force=True)
            elapsed = time.perf_counter() - self._last_command_start
        else:
            if self._last_command_start is None:
                return
            elapsed = time.perf_counter() - self._last_command_start

        record.total_seconds += elapsed
        record.seconds_by_turn[turn] += elapsed

    def finish_turn(self, turn: int):
        if self.mode == "off":
            return

        # Turn wall time and ETA must include queued GPU work even when
        # per-command timing is disabled.
        if self.use_gpu and self.mode in {"turn", "command"}:
            self._synchronize_gpu()

        if self.mode == "command" and self.use_gpu:
            for record, record_turn, events in self._pending_event_records:
                elapsed = self._cupy.cuda.get_elapsed_time(events.start, events.end) / 1000.0
                record.total_seconds += elapsed
                record.seconds_by_turn[record_turn] += elapsed
            self._pending_event_records.clear()

        if self._turn_start is not None:
            self.turn_seconds[turn] = time.perf_counter() - self._turn_start

    def _synchronize_gpu(self, force: bool = False):
        if not self.use_gpu or (self._gpu_synchronized and not force):
            return
        cp = self._load_cupy()
        cp.cuda.runtime.deviceSynchronize()
        self._gpu_synchronized = True

    def should_log_turn(self, turn: int) -> bool:
        if self.mode == "off":
            return True
        return ((turn + 1) % self.log_interval == 0 or turn == self.total_turns - 1)

    def format_progress(self, turn: int) -> str:
        turn_time = self.turn_seconds.get(turn, 0.0)
        completed = turn + 1
        elapsed = sum(self.turn_seconds.values())
        usable = [value for index, value in self.turn_seconds.items() if index >= self.warmup_turns]
        average = sum(usable) / len(usable) if usable else 0.0
        remaining = max(self.total_turns - completed, 0)
        eta = remaining * average
        return (
            f"Turn: {turn}/{self.total_turns} | turn: {self._format_seconds(turn_time)} | "
            f"avg: {self._format_seconds(average)}/turn | "
            f"elapsed: {self._format_long_duration(elapsed)} | "
            f"ETA: {self._format_long_duration(eta)}"
        )

    def print_summary(self):
        if self.mode == "off":
            return

        total_turn = sum(self.turn_seconds.values())
        turn_values = list(self.turn_seconds.values())
        average_turn = total_turn / len(turn_values) if turn_values else 0.0

        set_simple_logging()
        logger.info("")
        logger.info(center_string(" Timing Summary "))
        logger.info(f"Timing mode: {self.mode}")
        logger.info(f"Total tracking time: {self._format_long_duration(total_turn)}")
        logger.info(f"Average turn time: {self._format_seconds(average_turn)}")
        if turn_values:
            logger.info(f"Minimum turn time: {self._format_seconds(min(turn_values))}")
            logger.info(f"Maximum turn time: {self._format_seconds(max(turn_values))}")

        if self.command_timings:
            denominator = sum(item.total_seconds for item in self.command_timings.values())
            logger.info("")
            timing_kind = "GPU device time" if self.use_gpu and self.mode == "command" else "command wall time"
            logger.info(f"Command timing ({timing_kind}):")
            turns = max(len(self.turn_seconds), 1)
            names = sorted(self.command_timings)
            command_width = max(len("Command"), *(len(name) for name in names), len("Total"))
            headers = (
                f"{'Command':<{command_width}} | {'Calls':>8} | {'Calls/Turn':>10} | "
                f"{'Total Time':>12} | {'Avg/Turn':>12} | {'Avg/Call':>12} | {'Time Percentage':>16}"
            )
            logger.info(headers)
            logger.info("-" * len(headers))

            def format_row(name: str, calls: int, total_seconds: float, percentage: float) -> str:
                avg_turn = total_seconds / turns
                avg_call = total_seconds / calls if calls else 0.0
                return (
                    f"{name:<{command_width}} | {calls:>8d} | {calls / turns:>10.2f} | "
                    f"{self._format_seconds(total_seconds):>12} | "
                    f"{self._format_seconds(avg_turn):>12} | "
                    f"{self._format_seconds(avg_call):>12} | {percentage:>15.2f}%"
                )

            for name in names:
                item = self.command_timings[name]
                percentage = (item.total_seconds / denominator * 100.0) if denominator else 0.0
                logger.info(format_row(name, item.calls, item.total_seconds, percentage))

            total_calls = sum(item.calls for item in self.command_timings.values())
            logger.info("-" * len(headers))
            logger.info(format_row("Total", total_calls, denominator, 100.0 if denominator else 0.0))
        set_normal_logging()

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        """Format a per-turn or per-call duration using ms or s."""
        seconds = max(0.0, float(seconds))
        if seconds < 1.0:
            return f"{seconds * 1000.0:.3f} ms"
        return f"{seconds:.3f} s"

    @staticmethod
    def _format_long_duration(seconds: float) -> str:
        """Format elapsed/ETA duration using ms, s, min, or h."""
        seconds = max(0.0, float(seconds))
        if seconds < 1.0:
            return f"{seconds * 1000.0:.3f} ms"
        if seconds < 60.0:
            return f"{seconds:.3f} s"
        if seconds < 3600.0:
            return f"{seconds / 60.0:.2f} min"
        return f"{seconds / 3600.0:.2f} h"
