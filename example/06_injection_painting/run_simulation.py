"""Use the pass-run CLI and also save completed.json with final particle counts."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PASS.main import cli_main as pass_cli_main, main as pass_main
from PASS.utils.input_snapshot import atomic_write, json_bytes


def _save_completed(sim, started):
    cfg = sim.cfg
    p = sim.beams[0].particles
    result = {
        "completed_turns": cfg.num_turn,
        "seconds": time.perf_counter() - started,
        "alive": int((p.tag > 0).sum()),
        "lost": int((p.tag < 0).sum()),
        "pending": int((p.tag == 0).sum()),
        "output_directory": cfg.output_dir
    }
    atomic_write(Path(cfg.output_dir) / "completed.json", json_bytes(result))
    print(json.dumps(result), flush=True)
    return result


def run(path):
    """Run one input through the shared pipeline and return its completion report."""
    started = time.perf_counter()
    result = {}

    def on_completed(sim):
        result.update(_save_completed(sim, started))

    pass_main(str(path), raise_errors=True, on_completed=on_completed)
    return result


def cli_main(argv=None):
    started = time.perf_counter()
    return pass_cli_main(argv, on_completed=lambda sim: _save_completed(sim, started))


if __name__ == "__main__":
    raise SystemExit(cli_main())
