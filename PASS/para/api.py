"""High-level API for the PASS parameter system.

Usage (simple — smooth twiss)::

    from PASS.para.api import generate_input
    from PASS.para.schema.main import MainConfig
    from PASS.para.schema.bunch import BunchConfig, InjectionItem
    from PASS.para.schema.sequence import Sequence
    from PASS.para.smooth import generate_smooth_twiss

    main = MainConfig(beam_name="proton", num_turns=1000)
    bunch = BunchConfig(kinetic_energy=33.2e6, num_real_particles=int(1e11),
                        num_macro_particles=int(1e5))

    items, names, circum = generate_smooth_twiss(569.1, 9.47, 9.43, 100)
    main.circumference = circum

    seq = Sequence()
    seq.add("injection", InjectionItem(s=0.0, bunches=[bunch]))
    for name, item in zip(names, items):
        seq.add(name, item)

    generate_input(main, seq, "beam0.json")

Usage (from MADX TFS — one function)::

    from PASS.para.api import generate_from_tfs

    generate_from_tfs(
        twiss_file="bring.tfs",
        output_path="beam0.json",
        main=dict(beam_name="proton", num_turns=1024),
        bunches=[dict(
            kinetic_energy=99e9, num_macro_particles=17,
            dist_trans="kv", dist_longi="coasting",
            insert_particle=[[2e-3,0,0,0,0,0], [0,0,2e-3,0,0,0]],
        )],
        element_settings=dict(quad_slices=5, bend_slices=5),
        monitors=[dict(type="ParticleMonitor", s=0, max_tag=17)],
    )
"""

import json
from pathlib import Path

from PASS.para.schema.main import MainConfig
from PASS.para.schema.bunch import BunchConfig, InjectionItem
from PASS.para.schema.sequence import Sequence
from PASS.para.schema.space_charge import SpaceChargeConfig

# ============================================================
# Low-level: schema objects → JSON
# ============================================================


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
    result = main.model_dump(by_alias=True)

    if space_charge is not None:
        sc_dict = space_charge.model_dump(by_alias=True)
        result["Space-charge simulation parameters"] = sc_dict
        result["Is space charge"] = space_charge.is_enabled

    if extra_modules:
        result.update(extra_modules)

    result["Sequence"] = sequence.to_dict()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"[PASS] Input file written to: {path}")
    return str(path)


def load_input(path: str) -> tuple[MainConfig, dict]:
    """Load an existing PASS input JSON file.

    Args:
        path: path to the JSON file.

    Returns:
        (MainConfig, raw_sequence_dict)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sequence_data = data.pop("Sequence", {})
    data.pop("Space-charge simulation parameters", None)

    main = MainConfig.model_validate(data)
    return main, sequence_data


# ============================================================
# Mid-level: assemble sequence from items + names
# ============================================================


def build_sequence(
    items: list,
    names: list[str],
    bunches: list,
    monitors: list | None = None,
) -> Sequence:
    """Assemble a Sequence from lattice items + bunches + monitors.

    Args:
        items: list of Element or TwissPoint schema objects.
        names: list of string names (same length as items).
        bunches: list of BunchConfig objects (injected at s=0).
        monitors: optional list of monitor objects (StatMonitor,
            ParticleMonitor, etc.).

    Returns:
        A Sequence ready for generate_input().
    """
    seq = Sequence()

    if not bunches:
        raise ValueError("at least one bunch must be declared")

    prepared_bunches = []
    for harmonic_id, bunch in enumerate(bunches):
        # The common API path defines groups by list order. Preserve an
        # explicitly supplied harmonic_id, which is validated by Beam later.
        if "harmonic_id" not in bunch.model_fields_set:
            bunch = bunch.model_copy(update={"harmonic_id": harmonic_id})
        prepared_bunches.append(bunch)

    # One bunch per bucket: the harmonic number equals the number of
    # declared bunches (declare empty 0-particle bunches for unfilled
    # buckets).
    seq.add("injection", InjectionItem(s=0.0, harmonic_number=len(prepared_bunches), bunches=prepared_bunches))

    for name, item in zip(names, items):
        seq.add(name, item)

    if monitors:
        for i, mon in enumerate(monitors):
            cmd = mon.model_dump(by_alias=True).get("Command", "Monitor")
            mon_name = f"{cmd.lower()}_{i + 1}"
            seq.add(mon_name, mon)

    return seq


# ============================================================
# High-level: MADX TFS → JSON in one call
# ============================================================


def generate_from_tfs(
    twiss_file: str,
    output_path: str,
    main: dict,
    bunches: list[dict],
    element_settings: dict | None = None,
    monitors: list[dict] | None = None,
    is_merge_drift: bool = True,
    error_file: str = "",
    is_field_error: bool = False,
) -> str:
    """Generate a PASS input JSON from a MADX twiss TFS file.

    Automatically:
    - Reads TFS headers -> gamma_t, circumference, kinetic_energy.
    - Reads elements + preserves MADX names + s-position suffix.
    - Patches SBend K0L from ANGLE column.
    - Applies per-type element settings (slices, integrators).
    - Assembles Sequence + writes JSON.

    Args:
        twiss_file: path to MADX twiss TFS file.
        output_path: output JSON file path.
        main: dict of MainConfig fields (beam_name, num_turns, ...).
            'circumference' and 'gamma_t' are auto-filled from TFS
            if not provided.
        bunches: list of dicts, each passed to BunchConfig(**dict).
            'kinetic_energy' is auto-filled from TFS if not provided.
        element_settings: optional dict with keys:
            quad_slices, bend_slices, sext_slices, oct_slices,
            bend_model, quad_model,
            quad_integrator, bend_integrator, sext_integrator, oct_integrator.
        monitors: optional list of dicts, each with 'type' key
            (e.g. "ParticleMonitor") plus the monitor's fields.
        is_merge_drift: merge consecutive drift elements (default True).
        error_file: path to MADX error TFS file.
        is_field_error: attach field errors to matching elements.

    Returns:
        The output file path.
    """
    from PASS.para.madx import read_madx_elements, _read_tfs_headers

    # --- Read TFS headers for auto-fill ---
    headers = _read_tfs_headers(twiss_file)
    circum = headers["circumference"]
    gamma_tr = headers["gamma_tr"]
    ek_per_nucleon = (headers["energy"] - headers["mass"]) * 1e9  # GeV -> eV/u

    # GAMMATR=0 -> division by zero in eta; use large value
    if abs(gamma_tr) < 1e-10:
        gamma_tr = 1e6

    print(f"[PASS] TFS: C={circum:.4f}, gamma={headers['gamma']:.6f}, "
          f"gamma_tr={gamma_tr}, Q1={headers['q1']:.6f}, Q2={headers['q2']:.6f}")
    print(f"[PASS] Ekin={ek_per_nucleon:.6e} eV/u")

    # --- Auto-fill main ---
    main = dict(main)  # copy
    main.setdefault("circumference", circum)
    main.setdefault("gamma_t", gamma_tr)
    main.setdefault("beam_name", "proton")
    main.setdefault("num_proton", 1)
    main.setdefault("num_neutron", 0)
    main.setdefault("num_electron", 1)
    main_cfg = MainConfig(**main)

    # --- Auto-fill bunches ---
    bunch_objs = []
    for bdict in bunches:
        bdict = dict(bdict)  # copy
        bdict.setdefault("kinetic_energy", ek_per_nucleon)
        bunch_objs.append(BunchConfig(**bdict))

    # --- Read lattice elements ---
    items, names, _ = read_madx_elements(
        twiss_file,
        error_file=error_file,
        is_merge_drift=is_merge_drift,
        is_field_error=is_field_error,
    )

    # --- Apply element settings ---
    if element_settings:
        _apply_element_settings(items, **element_settings)

    # --- Build monitors ---
    monitor_objs = _build_monitors(monitors) if monitors else None

    # --- Assemble + write ---
    seq = build_sequence(items, names, bunches=bunch_objs, monitors=monitor_objs)
    return generate_input(main_cfg, seq, output_path)


def _apply_element_settings(
    items: list,
    quad_slices: int = 0,
    bend_slices: int = 0,
    sext_slices: int = 0,
    oct_slices: int = 0,
    bend_model: str = "",
    quad_model: str = "",
    quad_integrator: str = "",
    bend_integrator: str = "",
    sext_integrator: str = "",
    oct_integrator: str = "",
) -> None:
    """Apply slice counts, models, and integrators to lattice elements by type.

    Modifies items in-place. 0 / empty string means skip.
    """
    from collections import Counter
    counts = Counter()

    for item in items:
        cmd = item.__class__.__name__
        counts[cmd] += 1

        if cmd == "QuadrupoleElement":
            if quad_slices > 0:
                item.num_slices = quad_slices
            if quad_model:
                item.model = quad_model
            if quad_integrator:
                item.integrator = quad_integrator
        elif cmd == "SBendElement":
            if bend_slices > 0:
                item.num_slices = bend_slices
            if bend_model:
                item.model = bend_model
            if bend_integrator:
                item.integrator = bend_integrator
        elif cmd == "SextupoleElement":
            if sext_slices > 0:
                item.num_slices = sext_slices
            if sext_integrator:
                item.integrator = sext_integrator
        elif cmd == "OctupoleElement":
            if oct_slices > 0:
                item.num_slices = oct_slices
            if oct_integrator:
                item.integrator = oct_integrator

    print(f"[Lattice] {len(items)} elements")
    for t, c in sorted(counts.items()):
        print(f"  {t}: {c}")


def _build_monitors(monitors: list[dict]) -> list:
    """Convert monitor dicts to schema objects."""
    from PASS.para.schema.monitors import (
        StatMonitor,
        DistMonitor,
        PhaseMonitor,
        ParticleMonitor,
    )

    _monitor_map = {
        "statmonitor": StatMonitor,
        "stat": StatMonitor,
        "distmonitor": DistMonitor,
        "dist": DistMonitor,
        "phasemonitor": PhaseMonitor,
        "phase": PhaseMonitor,
        "particlemonitor": ParticleMonitor,
        "particle": ParticleMonitor,
    }

    result = []
    for m in monitors:
        mtype = m.pop("type").lower()
        cls = _monitor_map.get(mtype)
        if cls is None:
            raise ValueError(f"Unknown monitor type: {mtype}")
        result.append(cls(**m))
    return result


__all__ = [
    "generate_input",
    "load_input",
    "generate_from_tfs",
    "build_sequence",
    "MainConfig",
    "Sequence",
    "SpaceChargeConfig",
]
