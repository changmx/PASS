"""Generate deterministic one-turn inputs for the Slicer integration tests."""

from __future__ import annotations

from pathlib import Path

from PASS.para.api import generate_input
from PASS.para.schema.bunch import BunchConfig, InjectionItem
from PASS.para.schema.main import MainConfig
from PASS.para.schema.monitors import DistMonitor
from PASS.para.schema.slicer import Slicer
from PASS.para.schema.sequence import Sequence


CASES = {
    "auto_equal_length": {"slice_model": "equal_length", "num_slices": 16},
    "auto_equal_particle": {"slice_model": "equal_particle", "num_slices": 16},
}


def create_input(path: str | Path, case: str, backend: str = "cpu", output_dir: str | Path | None = None) -> Path:
    if case not in CASES:
        raise ValueError(f"unknown Slicer case {case!r}; expected one of {sorted(CASES)}")
    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be 'cpu' or 'gpu'")

    bunch = BunchConfig(
        kinetic_energy=33.2e6,
        num_real_particles=10_000,
        num_macro_particles=10_000,
        beta_x=1.0,
        beta_y=1.0,
        emit_x=1.0e-6,
        emit_y=1.0e-6,
        sigma_z=0.25,
        dp=1.0e-3,
        dist_trans="gaussian",
        dist_longi="gaussian",
        save_init_dist=False,
    )
    main = MainConfig(
        beam_name="slicer-test-proton",
        num_turns=1,
        backend=backend,
        particle_precision="float64",
        circumference=100.0,
        output_dir=str(output_dir) if output_dir is not None else "./output",
        is_plot=False,
    )
    injection = InjectionItem(
        harmonic_number=1,
        random_seed=20260901,
        bunches=[bunch],
    )
    settings = CASES[case]
    sequence = Sequence()
    sequence.add("injection", injection)
    sequence.add(
        "slicer",
        Slicer(
            s=0.0,
            slice_set="longitudinal",
            slice_model=settings["slice_model"],
            num_slices=settings["num_slices"],
            z_range_mode="auto",
            save_turns=[[0]],
        ),
    )
    sequence.add("distmonitor", DistMonitor(s=0.0, save_turns=[[0]]))
    generate_input(main, sequence, str(path))
    return Path(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("generated"))
    args = parser.parse_args()
    selected = CASES if args.case == "all" else {args.case: CASES[args.case]}
    for name in selected:
        create_input(args.output / f"beam0_{name}_{args.backend}.json", name, args.backend)
