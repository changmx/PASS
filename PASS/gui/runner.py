"""Process entry point with a reliable exit status for GUI and exported bundles."""
from __future__ import annotations

import logging
import sys


def run_inputs(beam0: str, beam1: str | None = None) -> int:
    from PASS.main import main

    class FailureStatus(logging.Handler):
        failed = False

        def emit(self, record):
            if record.levelno >= logging.ERROR:
                self.failed = True

    # PASS.main currently logs tracking exceptions rather than re-raising.
    # Attach to that logger itself, so setup_logging's root reconfiguration
    # cannot erase the status observer and falsely report a successful run.
    status = FailureStatus()
    logger = logging.getLogger("PASS.main")
    logger.addHandler(status)
    try:
        main(beam0, beam1)
    except Exception:
        logger.exception("Input initialization failed")
        return 1
    finally:
        logger.removeHandler(status)
    return 1 if status.failed else 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: python -m PASS.gui.runner beam0.json [beam1.json]")
    raise SystemExit(run_inputs(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None))
