import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import IntEnum
import logging
from pathlib import Path
import platform
import sys
from uuid import uuid4

from PASS.utils.input_snapshot import (archive_input_documents, atomic_write, canonicalize_task_name, create_run_directory, json_bytes,
                                       normalize_task_name, read_task_name, resolve_output_base)

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


def _is_project_input(path):
    """Project paths accept an entry file or a directory resolved by Project.open."""
    path = Path(path).expanduser()
    return path.is_dir() or path.suffix.lower() == ".passproj"


@contextmanager
def _load_run_inputs(paths, *, output_dir=None, task_name=None, comparison_documents=None, input_overrides=None):
    """Keep project assets alive until the selected runtime inputs are archived.

    The optional comparison_documents list receives editable project values;
    runtime documents keep referencing immutable cached dependencies.
    """
    from PASS.validation import parse_json

    # Command-line overrides belong to the invoking directory, unlike saved paths.
    output = Path(output_dir).expanduser().resolve() if output_dir is not None else None

    def override_documents(documents):
        nonlocal output, task_name
        if comparison_documents is not None and not comparison_documents:
            comparison_documents.extend(data for _label, data, _base in documents)
        if not input_overrides:
            return documents
        from PASS.utils.cli_overrides import apply_overrides
        documents, shared = apply_overrides(documents, input_overrides)
        if "Task name" in shared:
            if task_name is not None:
                raise ValueError("Task name was specified more than once")
            task_name = normalize_task_name(shared["Task name"])
        if "Output directory" in shared:
            if output_dir is not None:
                raise ValueError("Output directory was specified more than once")
            output = Path(shared["Output directory"])
        return documents

    if _is_project_input(paths[0]):
        from PASS.gui.project import Project

        if len(paths) != 1:
            raise ValueError("A project must be run on its own; it already selects its beam inputs")
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
            if comparison_documents is not None:
                comparison_documents.extend(project.editable_input(identifier)[0] for identifier in identifiers)
            documents = override_documents(documents)
            if output is None:
                value = settings.get("output_directory", "output")
                if value in (None, ""):
                    value = "output"
                if not isinstance(value, str):
                    raise ValueError("Project output_directory must be a path string")
                output = Path(value).expanduser()
                if not output.is_absolute():
                    output = project.path.parent / output
            name = task_name
            if name is None:
                name = read_task_name(settings, settings=True, default=read_task_name(documents[0][1]))
            yield documents, output.resolve(), project.path, normalize_task_name(name)
        finally:
            project.close()
        return

    if any(_is_project_input(value) for value in paths[1:]):
        raise ValueError("A project cannot be combined with JSON inputs")
    documents = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        data, parsed = parse_json(path.read_bytes())
        if not parsed.ok:
            raise ValueError(f"{path}: {parsed.text()}")
        documents.append((str(path), data, path.parent))
    documents = override_documents(documents)
    values = {str(key).casefold(): value for key, value in documents[0][1].items()}
    if output is None:
        output = resolve_output_base(values.get("output directory"), documents[0][0])
    name = read_task_name(documents[0][1]) if task_name is None else task_name
    yield documents, output, None, normalize_task_name(name)


def _prepare_run_inputs(documents, report, output, *, flat_output, stop_requested, project_path, task_name=None, comparison_documents=None):
    from PASS import __version__
    from PASS.validation import parse_json
    from PASS.validation.rules import validate_documents

    values = {str(key).casefold(): value for key, value in documents[0][1].items()}
    task_name = normalize_task_name(read_task_name(documents[0][1]) if task_name is None else task_name)
    if flat_output and task_name:
        raise ValueError("Task name cannot be combined with flat_output")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:10]
    if flat_output:
        # Explicit flat verification runs keep their existing external archive.
        results = output
        snapshot_root = output.parent / "input_snapshots"
        if snapshot_root == output:
            snapshot_root = output.parent / "input_snapshots_archive"
        snapshot = snapshot_root / run_id
    else:
        results = create_run_directory(output, task_name)
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
        "task_name": task_name,
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
        archived = archive_input_documents(documents,
                                           snapshot,
                                           output,
                                           task_name=task_name,
                                           comparison_documents=comparison_documents,
                                           check=check,
                                           record=record)
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
         task_name: str | None = None,
         output_subdirectory: str | None = None,
         output_name: str | None = None,
         input_overrides=None,
         _run_directory: str | Path | None = None):
    """Run one/two JSON inputs or the saved selection in a project file/directory.

    GUI-managed runs pass archive_inputs=False because their input files already
    belong to a verified snapshot. Low-level Config.load_input does not archive dependencies.
    Projects require archiving. JSON and project runs share a single task-folder
    layout, with an input/ archive inside each result directory. Explicit flat_output
    keeps results directly in that root and stores snapshots beside it.
    output_dir overrides the output root for this run, relative to the caller's
    working directory. It requires archiving and never changes the original inputs.
    task_name overrides the result-folder name; an empty string or 'default'
    (case-insensitive) generates YYYY_MM_DD_HH_MM using local time. None uses the
    saved setting (Beam 0 for JSON inputs). Both names and timestamps receive
    _1, _2, ... suffixes when occupied. Internal subdirectories and timestamp
    filenames are preserved. A custom name cannot combine with flat_output.
    Legacy output_subdirectory/output_name keywords remain accepted, but only one
    name option may be supplied. New snapshots use only the Task name field.
    on_completed(sim) runs after successful tracking, while the Simulation is still
    available. Callback failures mark the run failed; stopped runs skip this callback.
    """
    from PASS.validation.rules import validate_documents

    if input_overrides and not archive_inputs:
        raise ValueError("Input overrides require archive_inputs=True to record the effective configuration")

    names = [value for value in (task_name, output_subdirectory, output_name) if value is not None]
    if len(names) > 1:
        raise ValueError("Use task_name only once; do not combine it with legacy output-name keywords")
    task_name = names[0] if names else None
    if _run_directory is not None and (archive_inputs or flat_output):
        raise ValueError("_run_directory is reserved for already archived runs")
    if output_dir is not None:
        if not str(output_dir).strip():
            raise ValueError("output_dir must not be empty")
        if not archive_inputs:
            raise ValueError("output_dir requires archive_inputs=True; existing snapshots keep their recorded output paths")
    if task_name is not None:
        task_name = normalize_task_name(task_name)
        if not archive_inputs:
            raise ValueError("task_name requires archive_inputs=True; existing snapshots keep their recorded output settings")
    paths = [beam0_path] + ([beam1_path] if beam1_path is not None else [])
    record_path, record = None, None
    run_directory = _run_directory
    comparison_documents = []
    with _load_run_inputs(paths,
                          output_dir=output_dir,
                          task_name=task_name,
                          comparison_documents=comparison_documents,
                          input_overrides=input_overrides) as (documents, output, project_path, name):
        if project_path is not None and not archive_inputs:
            raise ValueError("Project runs require archive_inputs=True to preserve their input dependencies")
        if flat_output and name:
            raise ValueError("Task name cannot be combined with flat_output")
        validation_documents = []
        for index, (label, data, base) in enumerate(documents):
            runtime_data = canonicalize_task_name(data)
            runtime_data["Task name"] = name
            if output_dir is not None or input_overrides or project_path is not None or index > 0:
                output_key = next((key for key in runtime_data if key.casefold() == "output directory"), "Output directory")
                runtime_data[output_key] = str(output)
            validation_documents.append((label, runtime_data, base))
        report = validate_documents(validation_documents)
        if not report.ok:
            raise ValueError("JSON preflight failed before initialization:\n" + report.text())
        documents = validation_documents
        for issue in report.warnings:
            logger.warning("JSON preflight: %s", issue)
        if archive_inputs:
            try:
                paths, record_path, record = _prepare_run_inputs(documents,
                                                                 report,
                                                                 output,
                                                                 flat_output=flat_output,
                                                                 stop_requested=stop_requested,
                                                                 project_path=project_path,
                                                                 task_name=name,
                                                                 comparison_documents=comparison_documents or None)
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

    parser = argparse.ArgumentParser(prog="pass-run",
                                     allow_abbrev=False,
                                     description="Run one or two PASS JSON inputs, or a .passproj file/project directory, without loading Qt.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""Examples:
  pass-run beam0.json
  pass-run beam0.json beam1.json
  pass-run project.passproj
  pass-run path/to/project
  pass-run .
  pass-run --beam0 beam0.json
  pass-run --beam0 beam0.json --beam1 beam1.json
  pass-run --passproj project.passproj --output ./output
  pass-run --passproj path/to/project
  pass-run --version

Choose positional inputs or named input options; do not mix them.
--beam1 requires --beam0. --passproj uses the project's saved beam selection.
Directories prefer project.passproj, otherwise they require exactly one .passproj file.
If multiple project files exist without project.passproj, specify the file explicitly.
Project files may be editable JSON descriptors or legacy ZIP projects (read only).
The default output root is output beside the first JSON input or project;
saved output settings or --output can change it. Both input formats write
results to <output>/YYYY_MM_DD_HH_MM by default, or <output>/<name> when
Task name or --task-name specifies a custom name. Empty or 'default'
(case-insensitive, whitespace trimmed) generates the local date/minute timestamp.
Input snapshots live in the result directory's input/ folder.
Existing task folders receive _1, _2, ... suffixes; result filenames stay unchanged.
Relative --output paths use the current working directory.

JSON overrides: --number-of-turns=100 --backend=gpu --device-id=1
Use JSON keys with spaces replaced by hyphens and parenthesized units removed.
Nested fields: --timing.log-interval=20
Named elements: '--sequence[QF1_s1.000].length=0.5'
Array entries: '--device-id[0]=1'
Use beam0. or beam1. prefixes to select an input; the default is beam0.
Task name and output directory are shared run settings (beam0 only).
Exact paths: --set '["Sequence","QF1_s1.000","Length (m)"]' 0.5
Values use JSON types; plain strings are accepted for string fields.
Overrides are validated and saved in the run snapshot; source files stay unchanged.
Relative file paths supplied as overrides use the current working directory.
""")
    parser.add_argument("--version", action="version", version=f"PASS {__version__}", help="Show the PASS version and exit")
    parser.add_argument("input", nargs="?", metavar="INPUT", help="Beam 0 JSON input, .passproj file or project directory")
    parser.add_argument("second_input", nargs="?", metavar="BEAM1", help="Optional Beam 1 JSON input (positional JSON mode only)")
    named = parser.add_argument_group("named inputs", "Use these options instead of positional inputs")
    modes = named.add_mutually_exclusive_group()
    modes.add_argument("--beam0", metavar="JSON", help="Beam 0 JSON input; optionally combine with --beam1")
    modes.add_argument("--passproj",
                       metavar="PATH",
                       help="Run a .passproj file or project directory using its saved beam selection; cannot combine with --beam1")
    named.add_argument("--beam1", metavar="JSON", help="Beam 1 JSON input; requires --beam0")
    parser.add_argument("--output", metavar="DIR", help="Override the output root for this run; relative paths use the current working directory")
    parser.add_argument("--output-directory", metavar="DIR", help="JSON-key spelling of --output")
    parser.add_argument("--set", nargs=2, metavar=("KEY_PATH", "VALUE"), help="Override an exact JSON key-path array")
    output_options = parser.add_mutually_exclusive_group()
    output_options.add_argument("--task-name",
                                metavar="NAME",
                                help="Name the task folder beneath the output root; empty or 'default' generates YYYY_MM_DD_HH_MM")
    output_options.add_argument("--output-subdirectory", dest="task_name", metavar="NAME", help=argparse.SUPPRESS)
    output_options.add_argument("--output-name", dest="task_name", metavar="NAME", help=argparse.SUPPRESS)
    parser.add_argument("--stop-file", help="Stop before initialization or at a turn boundary when this file exists")
    from PASS.utils.cli_overrides import extract_overrides
    try:
        remaining, input_overrides = extract_overrides(list(sys.argv[1:] if argv is None else argv), parser._option_string_actions)
    except ValueError as exc:
        parser.error(str(exc))
    args = parser.parse_args(remaining)
    if args.output is not None and not args.output.strip():
        parser.error("--output must not be empty")
    if args.task_name is not None:
        try:
            args.task_name = normalize_task_name(args.task_name)
        except ValueError as exc:
            parser.error(str(exc))
    if any(value is not None for value in (args.beam0, args.beam1, args.passproj)):
        if args.input is not None or args.second_input is not None:
            parser.error("Use positional inputs or --beam0/--beam1/--passproj, not both")
        if args.passproj is not None:
            if args.beam1 is not None:
                parser.error("--passproj cannot be combined with --beam1; use the project's saved beam selection")
            if not _is_project_input(args.passproj):
                if not Path(args.passproj).expanduser().exists():
                    parser.error(f"Project path does not exist: {args.passproj}")
                parser.error("--passproj requires a .passproj file or project directory")
            paths = [args.passproj]
        else:
            if args.beam0 is None:
                parser.error("--beam1 requires --beam0")
            paths = [args.beam0] + ([args.beam1] if args.beam1 is not None else [])
            if any(Path(path).suffix.lower() != ".json" or Path(path).expanduser().is_dir() for path in paths):
                parser.error("--beam0 and --beam1 require .json input files")
    else:
        if args.input is None:
            parser.error("Provide a JSON input, .passproj file or project directory, or use --beam0 or --passproj")
        paths = [args.input] + ([args.second_input] if args.second_input is not None else [])
    for path in paths:
        if not _is_project_input(path) and Path(path).suffix.lower() != ".json":
            if not Path(path).expanduser().exists():
                parser.error(f"Input path does not exist: {path}")
            parser.error("Expected a .json input file, .passproj file or project directory")
    if len(paths) == 2 and any(_is_project_input(path) for path in paths):
        parser.error("A project must be run on its own; use its saved beam selection")
    stop_path = Path(args.stop_file).expanduser() if args.stop_file else None
    try:
        completed = main(paths[0],
                         paths[1] if len(paths) > 1 else None,
                         stop_requested=stop_path.is_file if stop_path else None,
                         on_completed=on_completed,
                         raise_errors=True,
                         output_dir=args.output,
                         task_name=args.task_name,
                         input_overrides=input_overrides)
    except KeyboardInterrupt:
        logger.warning("Run interrupted; the current turn may be incomplete")
        return RunExitCode.INTERRUPTED
    except Exception:
        logger.exception("Simulation initialization, tracking or output failed")
        return RunExitCode.FAILED
    return RunExitCode.STOPPED if completed is False else RunExitCode.COMPLETED


if __name__ == "__main__":
    raise SystemExit(cli_main())
