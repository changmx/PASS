"""Run the explicit input selected by the user."""

import argparse
from pathlib import Path

from PASS.main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("beam0.json"))
    args = parser.parse_args()
    main(str(args.input), raise_errors=True)
