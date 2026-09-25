"""Process entry point with a reliable exit status for GUI and exported bundles."""
from __future__ import annotations

from enum import IntEnum
import logging
from pathlib import Path


class RunExitCode(IntEnum):
    """Distinct process outcomes; 2 remains reserved for CLI argument errors."""
    COMPLETED = 0
    FAILED = 1
    STOPPED = 3
    INTERRUPTED = 130


def run_inputs(beam0: str, beam1: str | None = None, *, stop_file: str | None = None, record_path: str | None = None) -> int:
    from PASS.main import main

    logger = logging.getLogger(__name__)
    stop_path = Path(stop_file) if stop_file else None

    def initialized(cfg):
        if record_path:
            from PASS.gui.project import atomic_write, json_bytes, read_json
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
        completed = main(beam0,
                         beam1,
                         stop_requested=stop_path.is_file if stop_path else None,
                         on_initialized=initialized,
                         flat_output=bool(record_path),
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
