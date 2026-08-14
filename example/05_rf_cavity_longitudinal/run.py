"""Run PASS simulation for one Example-05 case.

Usage:
    python run.py          (uncomment the case to run in __main__)

Each case reads beam0_<name>.json and writes output/<name>/YYYY_MMDD/HHMM_SS/.
"""

from pathlib import Path

from PASS.main import main as pass_main


def run_case(name: str, script_dir: Path) -> None:
    beam0 = script_dir / f"beam0_{name}.json"
    if not beam0.exists():
        raise FileNotFoundError(f"Missing input file: {beam0} (run make_input.py first)")
    print(f"[run] {beam0}")
    pass_main(str(beam0))


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    run_case("twiss_h1_fixed", script_dir)
    run_case("twiss_h2_fixed", script_dir)
    run_case("twiss_h1_ramping", script_dir)
    run_case("element_h1_fixed", script_dir)
