"""High-level API for the PASS parameter system.

Usage::

    from PASS.para.api import generate_input
    from PASS.para.schema.main import MainConfig
    from PASS.para.schema.bunch import BunchConfig, InjectionItem
    from PASS.para.schema.sequence import Sequence
    from PASS.para.readers.smooth_approx import generate_smooth_twiss

    main = MainConfig(beam_name="proton", num_turns=1000)
    bunch = BunchConfig(kinetic_energy=33.2e6, num_real_particles=int(1e11),
                        num_macro_particles=int(1e5))

    items, circum = generate_smooth_twiss(569.1, 9.47, 9.43, 100)
    main.circumference = circum

    seq = Sequence()
    seq.add("injection", InjectionItem(s=0.0, bunches=[bunch]))
    for i, item in enumerate(items):
        seq.add(f"twiss_{i}", item)

    generate_input(main, seq, "beam0.json")
"""

from PASS.para.schema.main import MainConfig
from PASS.para.schema.sequence import Sequence
from PASS.para.schema.space_charge import SpaceChargeConfig
from PASS.para.writers.json_writer import write_input_json, load_input_json


def generate_input(
    main: MainConfig,
    sequence: Sequence,
    output_path: str,
    space_charge: SpaceChargeConfig | None = None,
    extra_modules: dict | None = None,
) -> str:
    """Generate a PASS input JSON file from schema objects.

    Args:
        main: global simulation parameters.
        sequence: ordered sequence of items.
        output_path: output JSON file path.
        space_charge: optional space-charge configuration.
        extra_modules: optional additional top-level JSON blocks.

    Returns:
        The output file path.
    """
    return write_input_json(
        main, sequence, output_path,
        space_charge=space_charge,
        extra_modules=extra_modules,
    )


def load_input(path: str) -> tuple[MainConfig, dict]:
    """Load an existing PASS input JSON file.

    Args:
        path: path to the JSON file.

    Returns:
        (MainConfig, raw_sequence_dict)
    """
    return load_input_json(path)


__all__ = [
    "generate_input",
    "load_input",
    "MainConfig",
    "Sequence",
    "SpaceChargeConfig",
]
