from datetime import datetime, timezone
import logging
from pathlib import Path
import platform
import sys
from uuid import uuid4

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
from PASS.utils.input_snapshot import archive_input_documents, atomic_write, json_bytes, resolve_output_base

logger = logging.getLogger(__name__)


class _SnapshotStopped(Exception):
    """Input copying stopped before a simulation was constructed."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _prepare_cli_inputs(paths, report, *, flat_output, stop_requested):
    from PASS import __version__
    from PASS.validation import parse_json
    from PASS.validation.rules import validate_documents

    documents = []
    for value in paths:
        path = Path(value).resolve()
        data, parsed = parse_json(path.read_bytes())
        if not parsed.ok:
            raise ValueError(parsed.text())
        documents.append((str(path), data, path.parent))
    values = {str(key).casefold(): value for key, value in documents[0][1].items()}
    output = resolve_output_base(values.get("output directory"), paths[0])
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:10]
    # Flat output is contractually free of subdirectories, including snapshots.
    snapshot_root = (output.parent if flat_output else output) / "input_snapshots"
    if flat_output and snapshot_root == output:
        snapshot_root = output.parent / "input_snapshots_archive"
    snapshot = snapshot_root / run_id
    snapshot.mkdir(parents=True, exist_ok=False)
    record_path = snapshot / "run.json"
    record = {
        "format_version": 1,
        "id": run_id,
        "status": "preparing",
        "created_at": _utc_now(),
        "started_at": None,
        "ended_at": None,
        "exit_code": None,
        "pass_version": __version__,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "launcher": "cli",
        "backend": values.get("backend (gpu/cpu)", "cpu"),
        "particle_precision": values.get("particle precision", "float64"),
        "configured_device_ids": values.get("device id", []),
        "output_root": str(output),
        "output_directory": str(output),
        "results_directory": str(output),
        "inputs": [],
        "dependencies": [],
        "random_seeds": [],
        "source_run": None,
        "warnings": [str(issue) for issue in report.warnings],
    }
    atomic_write(record_path, json_bytes(record))

    def check():
        if stop_requested is not None and stop_requested():
            raise _SnapshotStopped()

    try:
        check()
        archived = archive_input_documents(documents, snapshot, output, check=check, record=record)
        checked = validate_documents([(str(path), parse_json(path.read_bytes())[0], path.parent) for path in archived])
        check()
        if not checked.ok:
            raise ValueError("Snapshot preflight failed before initialization:\n" + checked.text())
        record["status"] = "ready"
    except BaseException as exc:
        if isinstance(exc, _SnapshotStopped):
            status, exit_code = "stopped", 3
        elif isinstance(exc, KeyboardInterrupt):
            status, exit_code = "interrupted", 130
        else:
            status, exit_code = "preparation_failed", 1
        record.update(status=status, exit_code=exit_code, ended_at=_utc_now(), error=str(exc))
        raise
    finally:
        atomic_write(record_path, json_bytes(record))
    return [str(path) for path in archived], record_path, record


def main(beam0_path: str,
         beam1_path: str | None = None,
         *,
         stop_requested=None,
         on_initialized=None,
         flat_output=False,
         raise_errors: bool = False,
         archive_inputs: bool = True):
    """Run from archived input dependencies by default, preserving output layout.

    GUI-managed runs pass archive_inputs=False because their input files already
    belong to a verified snapshot. Low-level Config.load_input does not archive dependencies.
    """
    from PASS.validation import validate_files
    paths = [beam0_path] + ([beam1_path] if beam1_path is not None else [])
    report = validate_files(paths)
    if not report.ok:
        raise ValueError("JSON preflight failed before initialization:\n" + report.text())
    for issue in report.warnings:
        logger.warning("JSON preflight: %s", issue)
    record_path, record = None, None
    if archive_inputs:
        try:
            paths, record_path, record = _prepare_cli_inputs(paths, report, flat_output=flat_output, stop_requested=stop_requested)
        except _SnapshotStopped:
            return False
    cfg = Config()
    try:
        cfg.load_input(paths[0], paths[1] if len(paths) > 1 else None, flat_output=flat_output)
    except BaseException as exc:
        if record is not None:
            record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          exit_code=130 if isinstance(exc, KeyboardInterrupt) else 1,
                          ended_at=_utc_now(),
                          error=str(exc))
            atomic_write(record_path, json_bytes(record))
        raise

    status, exit_code = "failed", 1
    try:
        setup_logging(log_file=cfg.get_log_path())
        if record is not None:
            cfg.input_snapshot_path = str(record_path)
            record.update(status="running",
                          started_at=_utc_now(),
                          results_directory=str(Path(cfg.output_dir).resolve()),
                          backend=cfg.backend,
                          particle_precision=cfg.particle_precision,
                          configured_device_ids=cfg.gpu_id)
            atomic_write(record_path, json_bytes(record))
            logger.info("Input snapshot: %s", record_path)
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
        completed = executor.run(sim, seqs, stop_requested=stop_requested)
        status, exit_code = ("stopped", 3) if completed is False else ("completed", 0)
        return completed
    except KeyboardInterrupt:
        status, exit_code = "interrupted", 130
        if raise_errors:
            raise
        logger.info("Interrupted by the user")
        return False
    except Exception as exc:
        if record is not None:
            record["error"] = str(exc)
        if raise_errors:
            raise
        logger.exception("Error occurred")
    finally:
        if record is not None:
            record.update(status=status, exit_code=exit_code, ended_at=_utc_now())
            atomic_write(record_path, json_bytes(record))
