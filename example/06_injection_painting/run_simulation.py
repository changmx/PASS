"""Run a validated input; exceptions propagate and partial runs are not success."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PASS.core.config import Config
from PASS.core.beam import Beam
from PASS.core.simulation import Simulation
from PASS.core.state import SimulationState
from PASS.core.sequence import CommandSequence
from PASS.core.executor import Executor
from PASS.commands.space_charge import initialize_space_charge_resources
from PASS.validation import validate_file
from PASS.utils.logger import setup_logging


def run(path):
    report = validate_file(path)
    if not report.ok:
        raise ValueError(report.text())
    cfg = Config()
    cfg.load_input(str(path))
    setup_logging(log_file=cfg.get_log_path())
    if cfg.use_gpu:
        cfg.select_gpu_device()
    sim = Simulation(cfg, [Beam(cfg.input_path[0], cfg)], SimulationState())
    initialize_space_charge_resources(sim)
    seq = CommandSequence(cfg.input_data[0], 0, sim)
    seq.sort()
    started = time.perf_counter()
    Executor().run(sim, [seq])
    p = sim.beams[0].particles
    result = {
        "completed_turns": cfg.num_turn,
        "seconds": time.perf_counter() - started,
        "alive": int((p.tag > 0).sum()),
        "lost": int((p.tag < 0).sum()),
        "pending": int((p.tag == 0).sum()),
        "output_directory": cfg.output_dir
    }
    (Path(cfg.output_dir) / "completed.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    run(parser.parse_args().input)
