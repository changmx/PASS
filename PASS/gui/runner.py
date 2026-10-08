"""Process entry point with a reliable exit status for GUI and exported bundles."""
from __future__ import annotations

import logging
from pathlib import Path

from PASS.main import RunExitCode


def run_inputs(beam0: str, beam1: str | None = None, *, stop_file: str | None = None, record_path: str | None = None) -> int:
    from PASS.main import main

    logger = logging.getLogger(__name__)
    stop_path = Path(stop_file) if stop_file else None

    def initialized(cfg):
        if record_path:
            from PASS.gui.project import read_json
            from PASS.utils.input_snapshot import atomic_write, json_bytes
            path = Path(record_path)
            record = read_json(path.read_bytes())
            record["results_directory"] = str(Path(cfg.output_dir).resolve())
            record["backend"] = cfg.backend
            record["particle_precision"] = cfg.particle_precision
            record["configured_device_ids"] = cfg.gpu_id
            record["observed_gpu"] = None
            if cfg.use_gpu:
                import cupy as cp
                device_id = cp.cuda.Device().id
                properties = cp.cuda.runtime.getDeviceProperties(device_id)
                name = properties["name"]
                record["observed_gpu"] = {
                    "device_id": device_id,
                    "name": name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name),
                    "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
                    "driver_version": cp.cuda.runtime.driverGetVersion(),
                }
            atomic_write(path, json_bytes(record))

    try:
        run_directory = None
        if record_path:
            from PASS.gui.project import read_json
            record = read_json(Path(record_path).read_bytes())
            if record.get("snapshot_layout") == "results/input":
                run_directory = record["results_directory"]
        completed = main(beam0,
                         beam1,
                         stop_requested=stop_path.is_file if stop_path else None,
                         on_initialized=initialized,
                         archive_inputs=not bool(record_path),
                         _run_directory=run_directory,
                         raise_errors=True)
    except KeyboardInterrupt:
        logger.warning("Run interrupted; the current turn may be incomplete")
        return RunExitCode.INTERRUPTED
    except Exception:
        logger.exception("Simulation initialization, tracking or output failed")
        return RunExitCode.FAILED
    return RunExitCode.STOPPED if completed is False else RunExitCode.COMPLETED


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("beam0")
    parser.add_argument("beam1", nargs="?")
    parser.add_argument("--stop-file")
    parser.add_argument("--record")
    args = parser.parse_args()
    raise SystemExit(run_inputs(args.beam0, args.beam1, stop_file=args.stop_file, record_path=args.record))
