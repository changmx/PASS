"""Write PASS engine-compatible JSON from schema objects.

The JSON structure consumed by the engine:

    {
        "Beam Name": "proton",
        "Number of Protons": 8,
        ...
        "Sequence": {
            "injection": {"S (m)": 0, "Command": "Injection", "bunch0": {...}},
            "qd1": {"S (m)": 1.0, "Command": "Quadrupole", ...},
            "stat1": {"S (m)": 0, "Command": "StatMonitor"},
            ...
        }
    }

The Sequence dict is sorted by (s, command priority) before writing.
"""

import json
from pathlib import Path

from PASS.para.schema.main import MainConfig
from PASS.para.schema.sequence import Sequence
from PASS.para.schema.space_charge import SpaceChargeConfig


def write_input_json(
    main: MainConfig,
    sequence: Sequence,
    output_path: str,
    space_charge: SpaceChargeConfig | None = None,
    extra_modules: dict | None = None,
) -> str:
    """Generate a PASS engine-compatible JSON input file.

    Args:
        main: MainConfig with global simulation parameters.
        sequence: Sequence container with all sequence items.
        output_path: path for the output JSON file.
        space_charge: optional SpaceChargeConfig.
        extra_modules: optional dict of additional top-level modules
                       (e.g. beam-beam parameters).

    Returns:
        The output file path.
    """
    result = main.model_dump(by_alias=True)

    if space_charge is not None:
        sc_dict = space_charge.model_dump(by_alias=True)
        # Wrap in top-level key matching engine expectation
        result["Space-charge simulation parameters"] = sc_dict
        result["Is space charge"] = space_charge.is_enabled

    if extra_modules:
        result.update(extra_modules)

    result["Sequence"] = sequence.to_dict()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"[JSON Writer] Input file written to: {path}")
    return str(path)


def load_input_json(path: str) -> tuple[MainConfig, dict]:
    """Load an existing JSON input file → (MainConfig, raw_sequence_dict).

    This is the reverse of write_input_json. Useful for modifying
    an existing input file and re-writing it.

    Args:
        path: path to the JSON input file.

    Returns:
        (MainConfig, raw_sequence_dict) where raw_sequence_dict is the
        unsorted Sequence dict from the file.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract main config (everything except Sequence and module blocks)
    sequence_data = data.pop("Sequence", {})
    data.pop("Space-charge simulation parameters", None)

    main = MainConfig.model_validate(data)
    return main, sequence_data
