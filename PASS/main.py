import logging
import json
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from PASS.core.particle import ParticlePool
from PASS.core.bunch import BunchInfo
from PASS.core.beam import Beam
from PASS.core.config import Config
from PASS.core.executor import Executor
from PASS.core.simulation import Simulation
from PASS.core.state import SimulationState
from PASS.commands import Command
from PASS.core.sequence import CommandSequence
from PASS.utils.logger import setup_logging, set_simple_logging, set_normal_logging
from PASS.utils import helper

logger = logging.getLogger(__name__)


def main(beam0_path: str, beam1_path: str | None = None, *, stop_requested=None, on_initialized=None, flat_output=False, raise_errors: bool = False):
    from PASS.validation import validate_files
    report = validate_files([beam0_path] + ([beam1_path] if beam1_path is not None else []))
    if not report.ok:
        raise ValueError("JSON preflight failed before initialization:\n" + report.text())
    for issue in report.warnings:
        logger.warning("JSON preflight: %s", issue)
    cfg = Config()
    cfg.load_input(beam0_path, beam1_path, flat_output=flat_output)

    setup_logging(log_file=cfg.get_log_path())

    try:
        if cfg.use_gpu:
            cfg.select_gpu_device()
        if on_initialized is not None:
            on_initialized(cfg)

        beams = []
        for i in range(cfg.num_beam):
            beams.append(Beam(cfg.input_path[i], cfg))

        state = SimulationState()

        sim = Simulation(cfg, beams, state)
        sim.print()

        from PASS.commands.space_charge import initialize_space_charge_resources
        initialize_space_charge_resources(sim)

        seqs = []
        for i in range(cfg.num_beam):
            seqs.append(CommandSequence(cfg.input_data[i], i, sim))

        for seq in seqs:
            seq.sort()
            seq.print()

        executor = Executor()
        return executor.run(sim, seqs, stop_requested=stop_requested)

    except KeyboardInterrupt:
        if raise_errors:
            raise
        logger.info("Interrupted by the user")
        return False

    except Exception:
        if raise_errors:
            raise
        logger.exception("Error occurred")

    finally:
        pass
