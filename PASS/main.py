import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import IntEnum
import logging
from pathlib import Path
import platform
import sys
from uuid import uuid4

from PASS.utils.input_snapshot import archive_input_documents, atomic_write, create_run_directory, json_bytes, resolve_output_base

logger = logging.getLogger(__name__)


class RunExitCode(IntEnum):
    """Process outcomes; argparse reserves 2 for invalid command arguments."""
    COMPLETED = 0
    FAILED = 1
    STOPPED = 3
    INTERRUPTED = 130


class _SnapshotStopped(Exception):
    """Input copying stopped before a simulation was constructed."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _load_run_inputs(paths, *, output_dir=None):
    """Keep project assets alive until the selected inputs have been archived."""
    from PASS.validation import parse_json

    # Command-line overrides belong to the invoking directory, unlike saved paths.
    output = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if Path(paths[0]).suffix.lower() == ".passproj":
        from PASS.gui.project import Project

        if len(paths) != 1:
            raise ValueError("A .passproj project must be run on its own; it already selects its beam inputs")
        project = Project.open(Path(paths[0]).expanduser())
        try:
            settings = project.run_settings
            beam0 = settings.get("beam0")
            identifiers = [project.active_config_id if beam0 in (None, "") else beam0]
            beam1 = settings.get("beam1")
            if beam1 not in (None, ""):
                identifiers.append(beam1)
            if any(not isinstance(identifier, str) or identifier not in project.configs for identifier in identifiers):
                raise ValueError("Project run settings refer to an unknown input; select the beam inputs in the GUI and save the project")
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("Project Beam 0 and Beam 1 must select different inputs")
            documents = [(project.configs[identifier].name, project.configs[identifier].data, project.config_base) for identifier in identifiers]
            if output is None:
                value = settings.get("output_directory", "output")
                if value in (None, ""):
                    value = "output"
                if not isinstance(value, str):
                    raise ValueError("Project output_directory must be a path string")
                output = Path(value).expanduser()
                if not output.is_absolute():
                    output = project.path.parent / output
            yield documents, output.resolve(), project.path
        finally:
            project.close()
        return

    if any(Path(value).suffix.lower() == ".passproj" for value in paths[1:]):
        raise ValueError("A .passproj project cannot be combined with JSON inputs")
    documents = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        data, parsed = parse_json(path.read_bytes())
        if not parsed.ok:
            raise ValueError(f"{path}: {parsed.text()}")
        documents.append((str(path), data, path.parent))
    if output is None:
        values = {str(key).casefold(): value for key, value in documents[0][1].items()}
        output = resolve_output_base(values.get("output directory"), documents[0][0])
    yield documents, output, None


def _prepare_run_inputs(documents, report, output, *, flat_output, stop_requested, project_path):
    from PASS import __version__
    from PASS.validation import parse_json
    from PASS.validation.rules import validate_documents

    values = {str(key).casefold(): value for key, value in documents[0][1].items()}
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:10]
    if flat_output:
        # Explicit flat verification runs keep their existing external archive.
        results = output
        snapshot_root = output.parent / "input_snapshots"
        if snapshot_root == output:
            snapshot_root = output.parent / "input_snapshots_archive"
        snapshot = snapshot_root / run_id
    else:
        results = create_run_directory(output)
        snapshot = results / "input"
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
        "results_directory": str(results),
        "snapshot_layout": "flat_external" if flat_output else "results/input",
        "inputs": [],
        "dependencies": [],
        "random_seeds": [],
        "source_run": None,
        "warnings": [str(issue) for issue in report.warnings],
    }
    if project_path is not None:
        record["source_project"] = str(project_path)
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
         on_completed=None,
         flat_output=False,
         raise_errors: bool = False,
         archive_inputs: bool = True,
         output_dir: str | Path | None = None,
         _run_directory: str | Path | None = None):
    """Run one/two JSON inputs or the saved selection in a .passproj project.

    GUI-managed runs pass archive_inputs=False because their input files already
    belong to a verified snapshot. Low-level Config.load_input does not archive dependencies.
    Projects require archiving. JSON and project runs share the same dated result
    layout, with an input/ archive inside each result directory. Explicit flat_output
    keeps results directly in that root and stores snapshots beside it.
    output_dir overrides the output root for this run, relative to the caller's
    working directory. It requires archiving and never changes the original inputs.
    on_completed(sim) runs after successful tracking, while the Simulation is still
    available. Callback failures mark the run failed; stopped runs skip this callback.
    """
    from PASS.validation.rules import validate_documents

    if _run_directory is not None and (archive_inputs or flat_output):
        raise ValueError("_run_directory is reserved for already archived, dated runs")
    if output_dir is not None:
        if not str(output_dir).strip():
            raise ValueError("output_dir must not be empty")
        if not archive_inputs:
            raise ValueError("output_dir requires archive_inputs=True; existing snapshots keep their recorded output paths")
    paths = [beam0_path] + ([beam1_path] if beam1_path is not None else [])
    record_path, record = None, None
    run_directory = _run_directory
    with _load_run_inputs(paths, output_dir=output_dir) as (documents, output, project_path):
        if project_path is not None and not archive_inputs:
            raise ValueError("Project runs require archive_inputs=True to preserve their input dependencies")
        validation_documents = []
        for index, (name, data, base) in enumerate(documents):
            runtime_data = data
            if output_dir is not None or project_path is not None or index > 0:
                runtime_data = dict(data)
                output_key = next((key for key in runtime_data if key.casefold() == "output directory"), "Output directory")
                runtime_data[output_key] = str(output)
            validation_documents.append((name, runtime_data, base))
        report = validate_documents(validation_documents)
        if not report.ok:
            raise ValueError("JSON preflight failed before initialization:\n" + report.text())
        for issue in report.warnings:
            logger.warning("JSON preflight: %s", issue)
        if archive_inputs:
            try:
                paths, record_path, record = _prepare_run_inputs(documents,
                                                                 report,
                                                                 output,
                                                                 flat_output=flat_output,
                                                                 stop_requested=stop_requested,
                                                                 project_path=project_path)
            except _SnapshotStopped:
                return False
            if not flat_output:
                run_directory = record["results_directory"]
        else:
            paths = [name for name, _data, _base in documents]
    try:
        from PASS.core.config import Config

        cfg = Config()
        cfg.load_input(paths[0], paths[1] if len(paths) > 1 else None, flat_output=flat_output, _run_directory=run_directory)
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
        from PASS.core.beam import Beam
        from PASS.core.executor import Executor
        from PASS.core.simulation import Simulation
        from PASS.core.state import SimulationState
        from PASS.core.sequence import CommandSequence
        from PASS.utils.logger import setup_logging

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
        if completed is not False and on_completed is not None:
            on_completed(sim)
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


def cli_main(argv: list[str] | None = None, *, on_completed=None) -> int:
    """Run JSON inputs or a project's saved selection without loading Qt."""
    from PASS import __version__

    parser = argparse.ArgumentParser(
        prog="pass-run",
        description="Run one or two PASS JSON inputs, or the saved beam selection in a .passproj project, without loading Qt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  pass-run beam0.json
  pass-run beam0.json beam1.json
  pass-run project.passproj
  pass-run --beam0 beam0.json
  pass-run --beam0 beam0.json --beam1 beam1.json
  pass-run --passproj project.passproj --output ./output
  pass-run --version

Choose positional inputs or named input options; do not mix them.
--beam1 requires --beam0. --passproj uses the project's saved beam selection.
The default output root is output beside the first JSON input or project;
saved output settings or --output can change it. Both input formats write
results to <output>/YYYY_MMDD/HHMM_SS, with input snapshots in its input/ folder.
Relative --output paths use the current working directory.
""")
    parser.add_argument("--version", action="version", version=f"PASS {__version__}", help="Show the PASS version and exit")
    parser.add_argument("input", nargs="?", metavar="INPUT", help="Beam 0 JSON input or .passproj project")
    parser.add_argument("second_input", nargs="?", metavar="BEAM1", help="Optional Beam 1 JSON input (positional JSON mode only)")
    named = parser.add_argument_group("named inputs", "Use these options instead of positional inputs")
    modes = named.add_mutually_exclusive_group()
    modes.add_argument("--beam0", metavar="JSON", help="Beam 0 JSON input; optionally combine with --beam1")
    modes.add_argument("--passproj", metavar="FILE", help="Run a .passproj project using its saved beam selection; cannot combine with --beam1")
    named.add_argument("--beam1", metavar="JSON", help="Beam 1 JSON input; requires --beam0")
    parser.add_argument("--output", metavar="DIR", help="Override the output root for this run; relative paths use the current working directory")
    parser.add_argument("--stop-file", help="Stop before initialization or at a turn boundary when this file exists")
    args = parser.parse_args(argv)
    if args.output is not None and not args.output.strip():
        parser.error("--output must not be empty")
    if any(value is not None for value in (args.beam0, args.beam1, args.passproj)):
        if args.input is not None or args.second_input is not None:
            parser.error("Use positional inputs or --beam0/--beam1/--passproj, not both")
        if args.passproj is not None:
            if args.beam1 is not None:
                parser.error("--passproj cannot be combined with --beam1; use the project's saved beam selection")
            if Path(args.passproj).suffix.lower() != ".passproj":
                parser.error("--passproj requires a .passproj project")
            paths = [args.passproj]
        else:
            if args.beam0 is None:
                parser.error("--beam1 requires --beam0")
            paths = [args.beam0] + ([args.beam1] if args.beam1 is not None else [])
            if any(Path(path).suffix.lower() != ".json" for path in paths):
                parser.error("--beam0 and --beam1 require .json inputs")
    else:
        if args.input is None:
            parser.error("Provide a JSON input or .passproj project, or use --beam0 or --passproj")
        paths = [args.input] + ([args.second_input] if args.second_input is not None else [])
    if any(Path(path).suffix.lower() not in {".json", ".passproj"} for path in paths):
        parser.error("Expected a .json input or a .passproj project")
    if len(paths) == 2 and any(Path(path).suffix.lower() == ".passproj" for path in paths):
        parser.error("A .passproj project must be run on its own; use its saved beam selection")
    stop_path = Path(args.stop_file).expanduser() if args.stop_file else None
    try:
        completed = main(paths[0],
                         paths[1] if len(paths) > 1 else None,
                         stop_requested=stop_path.is_file if stop_path else None,
                         on_completed=on_completed,
                         raise_errors=True,
                         output_dir=args.output)
    except KeyboardInterrupt:
        logger.warning("Run interrupted; the current turn may be incomplete")
        return RunExitCode.INTERRUPTED
    except Exception:
        logger.exception("Simulation initialization, tracking or output failed")
        return RunExitCode.FAILED
    return RunExitCode.STOPPED if completed is False else RunExitCode.COMPLETED


if __name__ == "__main__":
    raise SystemExit(cli_main())
