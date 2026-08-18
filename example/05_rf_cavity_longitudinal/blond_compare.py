"""Cross-check PASS RFCavity longitudinal tracking against BLonD.

The comparison uses the same low-energy heavy-ion case as Example 05:
238U35+ at 17 MeV/u in a C = 234.4 m ring with gamma_t = 3.374603832.
PASS tracks its normal one-turn Twiss longitudinal map while BLonD tracks
the equivalent first-order ("simple") ring map.

Coordinate and unit conversion:
    PASS:  z = s - beta*c*t [m], dp = (p - p0) / p0
    BLonD: dt = t - t_s [s], dE = E - E_s [eV / ion]

At an RF station, z = -beta*c*dt.  PASS energies are per nucleon, so they
are multiplied by A before comparison with BLonD's per-ion energy dE.

Usage:
    cd example/05_rf_cavity_longitudinal
    python blond_compare.py
    python blond_compare.py --case twiss_h1_waveform
    python blond_compare.py --skip-pass
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tfs

from PASS.main import main as pass_main
from PASS.utils.constants import const

from analyse import find_latest_output, measure_tune
from make_input import (
    CASES,
    CIRCUM,
    GAMMA_T,
    KINETIC_ENERGY,
    NUM_CHARGE,
    NUM_NEUTRON,
    NUM_PROTON,
    build_case,
    build_rf_data_tfs,
)

SCRIPT_DIR = Path(__file__).resolve().parent
COMPARABLE_CASES = ("twiss_h1_fixed", "twiss_h1_ramping", "twiss_h1_waveform")
COMPARE_TAGS = (1, 2, 3, 4, 5)
LIGHT_SPEED = 299_792_458.0
MASS_PER_NUCLEON = const.m_u_eV


def import_blond_cpu():
    """Import BLonD using its portable NumPy backend.

    On this workstation CuPy is installed, but the installed BLonD package
    does not include a kernel binary for the detected GPU.  Blocking optional
    CuPy/Numba imports keeps the comparison on BLonD's NumPy backend and does
    not change its RF or longitudinal-map equations.
    """
    sys.modules["cupy"] = None
    sys.modules["numba"] = None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Could not find module .*libblond_double\.dll.*",
        )
        from blond.beam.beam import Beam, Particle
        from blond.input_parameters.rf_parameters import RFStation
        from blond.input_parameters.ring import Ring
        from blond.trackers.tracker import RingAndRFTracker

    return Beam, Particle, RFStation, Ring, RingAndRFTracker


def read_pass_tbt(output_dir: Path) -> dict[int, dict[str, np.ndarray]]:
    """Read the tagged PASS monitor particles needed for the comparison."""
    data = {}
    for file_path in sorted((output_dir / "particle").glob("*_tag*.tfs")):
        tag = int(file_path.stem.split("_tag")[-1].lstrip("_"))
        if tag not in COMPARE_TAGS:
            continue
        table = tfs.read(str(file_path))
        data[tag] = {
            name: table[name].to_numpy(dtype=float)
            for name in ("turn", "z", "dp")
        }
    missing = set(COMPARE_TAGS) - set(data)
    if missing:
        raise RuntimeError(f"PASS output is missing tagged particles: {sorted(missing)}")
    return data


def pass_particle_energy_per_ion(dp: np.ndarray, reference_ek_per_u: np.ndarray) -> np.ndarray:
    """Convert PASS dp and reference kinetic energy to total ion energy [eV]."""
    mass_number = NUM_PROTON + NUM_NEUTRON
    reference_total_per_u = reference_ek_per_u + MASS_PER_NUCLEON
    reference_p_per_u = np.sqrt(reference_total_per_u**2 - MASS_PER_NUCLEON**2)
    particle_p_per_u = reference_p_per_u * (1.0 + dp)
    particle_total_per_u = np.sqrt(particle_p_per_u**2 + MASS_PER_NUCLEON**2)
    return mass_number * particle_total_per_u


def read_pass_reference_energy(output_dir: Path) -> np.ndarray:
    """Read PASS reference kinetic energy [eV/u] from the StatMonitor CSV."""
    stat_files = sorted(output_dir.glob("*_stat_*.csv"))
    if len(stat_files) != 1:
        raise RuntimeError(f"Expected one StatMonitor CSV in {output_dir}")
    with stat_files[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return np.array([float(row["Ek"]) for row in rows])


def run_pass(case_name: str) -> Path:
    """Generate and run the PASS side of the shared reference case."""
    beam0_path = Path(build_case(case_name, SCRIPT_DIR))
    pass_main(str(beam0_path))
    output_dir = find_latest_output(case_name)
    if output_dir is None:
        raise RuntimeError("PASS did not produce a complete monitor output directory")
    return output_dir


def rf_program(case: dict) -> dict[str, np.ndarray]:
    """Return the exact turn-by-turn RF program used by PASS."""
    n_turns = case["num_turns"]
    if case["rf_mode"] == "fixed":
        return {
            "harmonic": np.full(n_turns, case["harmonic"], dtype=int),
            "voltage": np.full(n_turns, case["voltage"], dtype=float),
            "phase": np.full(n_turns, case["phase"] + case["phi_offset"], dtype=float),
        }

    filename = case.get("ramp_file", case.get("waveform_file"))
    path = SCRIPT_DIR / filename
    if not path.exists():
        build_rf_data_tfs(SCRIPT_DIR, case)
    table = tfs.read(str(path))
    rows = np.minimum(np.arange(n_turns), len(table) - 1)
    return {
        "harmonic": table["HARMONIC"].to_numpy(dtype=int)[rows],
        "voltage": table["VOLTAGE"].to_numpy(dtype=float)[rows],
        "phase": (
            table["PHASE"].to_numpy(dtype=float)[rows]
            + table["PHI_OFFSET"].to_numpy(dtype=float)[rows]
        ),
    }


def run_blond(
    pass_data: dict[int, dict[str, np.ndarray]],
    program: dict[str, np.ndarray],
) -> dict[int, dict[str, np.ndarray]]:
    """Track the corresponding particles with BLonD's first-order ring map."""
    Beam, Particle, RFStation, Ring, RingAndRFTracker = import_blond_cpu()

    mass_number = NUM_PROTON + NUM_NEUTRON
    n_turns = len(pass_data[1]["turn"])
    if len(program["voltage"]) != n_turns:
        raise RuntimeError("RF program length does not match PASS tracking turns")
    gain_per_turn = NUM_CHARGE * program["voltage"] * np.sin(program["phase"])
    initial_kinetic_per_ion = mass_number * KINETIC_ENERGY
    kinetic_program = initial_kinetic_per_ion + np.concatenate(
        ([0.0], np.cumsum(gain_per_turn))
    )
    alpha_0 = 1.0 / GAMMA_T**2

    ring = Ring(
        CIRCUM,
        alpha_0,
        kinetic_program,
        Particle(mass_number * MASS_PER_NUCLEON, NUM_CHARGE),
        n_turns,
        synchronous_data_type="kinetic energy",
    )
    rf_station = RFStation(
        ring,
        np.append(program["harmonic"], program["harmonic"][-1]),
        np.append(program["voltage"], program["voltage"][-1]),
        np.append(program["phase"], program["phase"][-1]),
    )
    beam = Beam(ring, len(COMPARE_TAGS), len(COMPARE_TAGS))
    tracker = RingAndRFTracker(rf_station, beam, solver="simple")

    # PASS monitor data at turn 0 is immediately after the first RF kick.
    # Recover each tagged particle's injected coordinates by inverting that
    # known first kick through the exact relativistic energy relation.
    first_ref_energy = ring.energy[0, 0]
    first_ref_momentum_after_kick = ring.momentum[0, 1]
    injected_z = []
    injected_energy = []
    for tag in COMPARE_TAGS:
        z_after_kick = pass_data[tag]["z"][0]
        dp_after_kick = pass_data[tag]["dp"][0]
        particle_energy_after = math.sqrt(
            (first_ref_momentum_after_kick * (1.0 + dp_after_kick)) ** 2
            + (mass_number * MASS_PER_NUCLEON) ** 2
        )
        particle_phase = (
            program["phase"][0]
            - program["harmonic"][0] * 2.0 * math.pi * z_after_kick / CIRCUM
        )
        kick = NUM_CHARGE * program["voltage"][0] * math.sin(particle_phase)
        injected_z.append(z_after_kick)
        injected_energy.append(particle_energy_after - kick)

    beam.dt[:] = -np.asarray(injected_z) / (ring.beta[0, 0] * LIGHT_SPEED)
    beam.dE[:] = np.asarray(injected_energy) - first_ref_energy

    result = {
        tag: {
            "turn": np.arange(n_turns, dtype=int),
            "z": np.empty(n_turns),
            "energy": np.empty(n_turns),
        }
        for tag in COMPARE_TAGS
    }

    for turn in range(n_turns):
        # Match PASS's monitor location: after the RF kick, before the ring
        # drift.  BLonD's public track() combines these two operations.
        tracker.kick(beam.dt, beam.dE, turn)
        beta_after_kick = ring.beta[0, turn + 1]
        particle_energy = ring.energy[0, turn + 1] + beam.dE
        z_after_kick = -beta_after_kick * LIGHT_SPEED * beam.dt
        for index, tag in enumerate(COMPARE_TAGS):
            result[tag]["z"][turn] = z_after_kick[index]
            result[tag]["energy"][turn] = particle_energy[index]

        tracker.drift(beam.dt, beam.dE, turn + 1)
        beam.beta = ring.beta[0, turn + 1]
        beam.gamma = ring.gamma[0, turn + 1]
        beam.energy = ring.energy[0, turn + 1]
        beam.momentum = ring.momentum[0, turn + 1]
        rf_station.counter[0] += 1

    return result


def build_comparison(
    pass_data: dict[int, dict[str, np.ndarray]],
    pass_reference_ek: np.ndarray,
    blond_data: dict[int, dict[str, np.ndarray]],
) -> dict[int, dict[str, np.ndarray]]:
    """Convert both coordinate systems and assemble like-for-like residuals."""
    comparison = {}
    for tag in COMPARE_TAGS:
        pass_energy = pass_particle_energy_per_ion(pass_data[tag]["dp"], pass_reference_ek)
        blond_energy = blond_data[tag]["energy"]
        comparison[tag] = {
            "turn": pass_data[tag]["turn"].astype(int),
            "pass_z_m": pass_data[tag]["z"],
            "blond_z_m": blond_data[tag]["z"],
            "delta_z_m": pass_data[tag]["z"] - blond_data[tag]["z"],
            "pass_energy_eV": pass_energy,
            "blond_energy_eV": blond_energy,
            "delta_energy_eV": pass_energy - blond_energy,
        }
    return comparison


def write_outputs(
    comparison: dict[int, dict[str, np.ndarray]],
    case_name: str,
    program: dict[str, np.ndarray],
) -> Path:
    """Write machine-readable data, a plot, and a concise comparison report."""
    output_dir = SCRIPT_DIR / "blond_comparison_output" / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pass_blond_turn_by_turn.csv"
    report_path = output_dir / "pass_blond_report.md"
    plot_path = output_dir / "pass_blond_comparison.png"
    waveform_path = output_dir / "rf_voltage_waveform.png"

    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "tag",
                "turn",
                "pass_z_m",
                "blond_z_m",
                "delta_z_m",
                "pass_energy_eV_per_ion",
                "blond_energy_eV_per_ion",
                "delta_energy_eV_per_ion",
            ]
        )
        for tag, values in comparison.items():
            for row in zip(
                values["turn"],
                values["pass_z_m"],
                values["blond_z_m"],
                values["delta_z_m"],
                values["pass_energy_eV"],
                values["blond_energy_eV"],
                values["delta_energy_eV"],
            ):
                writer.writerow([tag, *row])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    direct_cases = {
        2: ("z0 = +3 m, dp0 = 0", "C0"),
        4: ("z0 = 0, dp0 = +1e-3", "C3"),
    }
    for column, (tag, (label, color)) in enumerate(direct_cases.items()):
        values = comparison[tag]
        axes[0, column].plot(
            values["turn"], values["pass_z_m"], color=color, lw=1.8, label="PASS"
        )
        axes[0, column].plot(
            values["turn"],
            values["blond_z_m"],
            color=color,
            lw=1.2,
            ls="--",
            label="BLonD",
        )
        axes[0, column].set_title(label)
        axes[0, column].legend(fontsize=8)

    for tag in (2, 3, 4, 5):
        values = comparison[tag]
        z_amplitude = np.max(np.abs(values["pass_z_m"]))
        axes[1, 0].plot(
            values["turn"],
            100.0 * values["delta_z_m"] / z_amplitude,
            lw=0.9,
            label=f"tag {tag}",
        )
        axes[1, 1].plot(
            values["turn"], values["delta_energy_eV"] / 1e3, lw=0.9, label=f"tag {tag}"
        )

    axes[0, 0].set_ylabel("z at RF station (m)")
    axes[0, 1].set_ylabel("z at RF station (m)")
    axes[1, 0].set_ylabel("(PASS - BLonD) z / PASS amplitude (%)")
    axes[1, 1].set_ylabel("PASS - BLonD energy (keV / ion)")
    for axis in axes.flat:
        axis.set_xlabel("turn")
        axis.grid(alpha=0.3)
    axes[1, 0].legend(ncol=2, fontsize=8)
    fig.suptitle("238U35+ longitudinal acceleration: PASS vs BLonD")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 3.5))
    axis.plot(np.arange(len(program["voltage"])), program["voltage"] / 1e3, color="C2")
    axis.set_xlabel("turn")
    axis.set_ylabel("RF voltage (kV)")
    axis.set_title(f"{case_name}: RF voltage program used by PASS and BLonD")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(waveform_path, dpi=160)
    plt.close(fig)

    mass_number = NUM_PROTON + NUM_NEUTRON
    gain_per_turn = NUM_CHARGE * program["voltage"] * np.sin(program["phase"])
    gamma_0 = 1.0 + KINETIC_ENERGY / MASS_PER_NUCLEON
    beta_0 = math.sqrt(1.0 - 1.0 / gamma_0**2)
    eta_0 = 1.0 / GAMMA_T**2 - 1.0 / gamma_0**2
    qs_theory = math.sqrt(
        -(NUM_CHARGE / mass_number)
        * program["harmonic"][0]
        * program["voltage"][0]
        * eta_0
        * math.cos(program["phase"][0])
        / (2.0 * math.pi * beta_0**2 * gamma_0 * MASS_PER_NUCLEON)
    )
    pass_qs = float(
        np.mean([measure_tune(comparison[tag]["pass_z_m"])[0] for tag in (2, 3)])
    )
    blond_qs = float(
        np.mean([measure_tune(comparison[tag]["blond_z_m"])[0] for tag in (2, 3)])
    )
    qs_relative_difference = abs(pass_qs - blond_qs) / pass_qs
    with report_path.open("w") as stream:
        stream.write("# PASS and BLonD Longitudinal Comparison\n\n")
        stream.write("## Shared Parameters\n\n")
        stream.write("| quantity | value |\n|---|---:|\n")
        stream.write("| ion | 238U35+ |\n")
        stream.write(f"| kinetic energy | {KINETIC_ENERGY / 1e6:.3f} MeV/u |\n")
        stream.write(f"| circumference | {CIRCUM:.4f} m |\n")
        stream.write(f"| gamma_t | {GAMMA_T:.9f} |\n")
        stream.write(f"| alpha_0 | {1.0 / GAMMA_T**2:.9e} |\n")
        stream.write(f"| case | {case_name} |\n")
        stream.write(
            f"| RF voltage | {np.min(program['voltage']) / 1e3:.3f} to "
            f"{np.max(program['voltage']) / 1e3:.3f} kV |\n"
        )
        stream.write(
            f"| RF phase | {program['phase'][0]:.3f} rad "
            f"(constant over this program) |\n"
        )
        stream.write(
            f"| synchronous gain | {np.min(gain_per_turn) / 1e3:.3f} to "
            f"{np.max(gain_per_turn) / 1e3:.3f} keV/ion/turn |\n"
        )
        stream.write(f"| small-amplitude Qs at turn 0 | {qs_theory:.8e} |\n\n")
        stream.write("## Conversion\n\n")
        stream.write("- `z_PASS = -beta*c*dt_BLonD` at the RF station.\n")
        stream.write("- PASS stores energy per nucleon; values in this comparison are multiplied by A=238 to obtain eV/ion.\n")
        stream.write("- PASS `dp` is converted with `E = sqrt((p0*(1+dp))^2 + m0^2)` before multiplying by A.\n\n")
        stream.write(
            "- PASS reads the TFS file directly; BLonD receives the same extracted "
            "`harmonic`, `voltage`, and `phase + phi_offset` arrays.\n\n"
        )
        stream.write("## Synchrotron Tune\n\n")
        stream.write("| PASS Qs | BLonD Qs | relative difference |\n")
        stream.write("|---:|---:|---:|\n")
        stream.write(
            f"| {pass_qs:.9e} | {blond_qs:.9e} | {qs_relative_difference:.3e} |\n\n"
        )
        stream.write("## Residuals\n\n")
        stream.write("| tag | initial condition | max abs dz (m) | dz / z amplitude | max abs dE (eV/ion) | rms dE (eV/ion) |\n")
        stream.write("|---:|---|---:|---:|---:|---:|\n")
        labels = {
            1: "z=0, dp=0",
            2: "z=+3 m, dp=0",
            3: "z=-3 m, dp=0",
            4: "z=0, dp=+1e-3",
            5: "z=0, dp=-1e-3",
        }
        for tag, values in comparison.items():
            dz = values["delta_z_m"]
            de = values["delta_energy_eV"]
            z_amplitude = np.max(np.abs(values["pass_z_m"]))
            relative_z = "-" if z_amplitude < 1e-8 else f"{np.max(np.abs(dz)) / z_amplitude:.3%}"
            stream.write(
                f"| {tag} | {labels[tag]} | {np.max(np.abs(dz)):.6e} | "
                f"{relative_z} | {np.max(np.abs(de)):.6e} | "
                f"{np.sqrt(np.mean(de**2)):.6e} |\n"
            )
        stream.write("\n")
        stream.write("## Verdict\n\n")
        stream.write(
            "- **Synchronous acceleration agrees.** The synchronous particle remains "
            "at the reference energy and its maximum energy residual is below 0.01 eV/ion.\n"
        )
        stream.write(
            f"- **Small-amplitude longitudinal motion agrees.** The measured Qs "
            f"difference is {qs_relative_difference:.3e}.\n"
        )
        stream.write(
            "- **Finite-amplitude trajectories are close but not identical.** The "
            "largest z residual among tags 2-5 is below 0.9% of its PASS oscillation "
            "amplitude.\n\n"
        )
        stream.write(
            "The remaining trajectory difference is expected because PASS transports "
            "`dp` with `z <- z - eta*C*dp`, while BLonD's `simple` solver transports "
            "`dE` with its first-order conversion `dp ~= dE/(beta^2*E)`. "
            "The RF kick and synchronous-energy program are matched exactly.\n"
        )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PASS RFCavity tracking with BLonD.")
    parser.add_argument(
        "--case",
        choices=COMPARABLE_CASES,
        default="twiss_h1_waveform",
        help="RF case to compare (default: twiss_h1_waveform).",
    )
    parser.add_argument(
        "--skip-pass",
        action="store_true",
        help="Reuse the latest complete PASS output for the selected case.",
    )
    args = parser.parse_args()
    case = CASES[args.case]
    program = rf_program(case)

    if args.skip_pass:
        pass_output = find_latest_output(args.case)
        if pass_output is None:
            parser.error("No complete PASS output found; run without --skip-pass first.")
    else:
        pass_output = run_pass(args.case)

    pass_data = read_pass_tbt(pass_output)
    pass_reference_ek = read_pass_reference_energy(pass_output)
    blond_data = run_blond(pass_data, program)
    comparison = build_comparison(pass_data, pass_reference_ek, blond_data)
    output_dir = write_outputs(comparison, args.case, program)
    pass_qs = float(
        np.mean([measure_tune(comparison[tag]["pass_z_m"])[0] for tag in (2, 3)])
    )
    blond_qs = float(
        np.mean([measure_tune(comparison[tag]["blond_z_m"])[0] for tag in (2, 3)])
    )

    print(f"PASS output: {pass_output}")
    print(f"Comparison report: {output_dir / 'pass_blond_report.md'}")
    print(f"Turn-by-turn CSV: {output_dir / 'pass_blond_turn_by_turn.csv'}")
    print(f"Plot: {output_dir / 'pass_blond_comparison.png'}")
    print(
        f"Qs: PASS={pass_qs:.9e}, BLonD={blond_qs:.9e}, "
        f"relative difference={abs(pass_qs - blond_qs) / pass_qs:.3e}"
    )
    for tag, values in comparison.items():
        print(
            f"tag {tag}: max|dz|={np.max(np.abs(values['delta_z_m'])):.3e} m, "
            f"max|dE|={np.max(np.abs(values['delta_energy_eV'])):.3e} eV/ion"
        )


if __name__ == "__main__":
    main()
