"""Generate PASS input for RF cavity longitudinal tracking (Example 05).

Four test cases, all driven by the single CASES dictionary below.
analyse.py imports CASES from this module so that lattice / RF / particle
parameters and theory expectations are never duplicated.

    twiss_h1_fixed    - one-turn Twiss map (longitudinal_transfer="drift") + RFCavity, h=1
    twiss_h2_fixed    - same lattice, h=2 (RF-period symmetry)
    twiss_h1_ramping  - h=1, RF parameters from a TFS ramping file (one row per turn)
    element_h1_fixed  - element-by-element drift ring + RFCavity (exact longitudinal drift)

Beam: low-energy heavy ion 238U35+ at 17 MeV/u, in the example-03/04 FODO
ring (fodo.tfs headers: C = 234.4 m, gamma_t = 3.3746, proton 2 TeV).
    q/A = 35/238 = 0.147059,  gamma = 1.018250,  beta = 0.188473
    C = 234.4 m,  gamma_t = 3.374603832
    eta = 1/gamma_t^2 - 1/gamma^2 = -0.876666  (< 0, below transition)

No transition crossing: gamma = 1.018 is far below gamma_t = 3.375
(crossing would require ~2.2 GeV/u); after 2048 turns gamma grows only
to ~1.019.  The fodo K-values are calibrated for the 2 TeV proton, so
the element case keeps a pure-drift ring (longitudinal physics is
independent of transverse focusing); the twiss case uses only the
gamma_t / circumference headers.

Theory (first-order, small-amplitude; same convention as PASS RFCavity kick
dE = (q/A) V sin(phase - h z/R), synchronous particle at z = 0):
    dE_syn = (q/A) V sin(phi_s)                               [eV/u per turn]
    Qs     = sqrt( -(q/A) h V eta cos(phi_s) / (2 pi beta^2 E) )
    dpmax  = sqrt( -(q/A) V (2 cos(phi_s) - (pi - 2 phi_s) sin(phi_s))
                   / (pi beta^2 E h eta) )
    zmax   = R (pi - 2 phi_s) / h                             [m]

Usage:
    python make_input.py
    python make_input.py --case twiss_h1_fixed
    python make_input.py --case all
"""

import math
from pathlib import Path

import numpy as np
import tfs as tfs_lib

from PASS.para.api import generate_input, build_sequence
from PASS.para.schema.main import MainConfig
from PASS.para.schema.bunch import BunchConfig, OffsetConfig
from PASS.para.schema.monitors import StatMonitor, ParticleMonitor
from PASS.para.schema.elements import RFCavityElement
from PASS.para.schema.twiss import TwissPoint

SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================================
# Beam parameters (low-energy heavy ion 238U35+)
# ============================================================

BEAM_NAME = "uranium-238-35+"
NUM_PROTON = 92
NUM_NEUTRON = 146
NUM_CHARGE = 35
QM_RATIO = NUM_CHARGE / (NUM_PROTON + NUM_NEUTRON)   # 0.147059

KINETIC_ENERGY = 17.0e6      # eV/u
M0 = 931.494e6               # eV/c^2 per nucleon (u)
GAMMA_0 = 1.0 + KINETIC_ENERGY / M0
BETA_0 = math.sqrt(1.0 - 1.0 / GAMMA_0**2)
E_TOTAL_0 = GAMMA_0 * M0     # eV per nucleon

CIRCUM = 234.4              # m (example-03/04 FODO, fodo.tfs LENGTH)
GAMMA_T = 3.374603832       # fodo.tfs GAMMATR
RADIUS = CIRCUM / (2.0 * math.pi)
ETA_0 = 1.0 / GAMMA_T**2 - 1.0 / GAMMA_0**2

# ============================================================
# RF parameters (shared by all cases, per-case overrides in CASES)
# ============================================================

RF_VOLTAGE = 20.0e3          # V (cavity voltage)
RF_HARMONIC = 1
RF_PHASE = 0.1               # rad, synchronous phase (eta < 0 -> 0 < phi_s < pi/2)
RF_PHI_OFFSET = 0.0          # rad
NUM_TURNS = 2048
NUM_DIST = 5000

# Distribution (matched to the RF bucket; dp spread is the binding constraint)
SIGMA_Z = 5.0                # m
SIGMA_DP = 1.0e-3


def get_eta(gamma: float) -> float:
    """Slip factor eta = 1/gamma_t^2 - 1/gamma^2."""
    return 1.0 / GAMMA_T**2 - 1.0 / gamma**2


def calc_theory(voltage: float, harmonic: int, phase: float) -> dict:
    """Theory expectations with the initial (gamma_0, beta_0) reference.

    Returns a dict with dE_syn [eV/u/turn], Qs, dpmax, zmax [m],
    and the dp acceptance value used for the RFCavity dp aperture.
    """
    dE_syn = QM_RATIO * voltage * math.sin(phase)

    beta2 = BETA_0**2
    qs = math.sqrt(
        -(QM_RATIO * harmonic * voltage * ETA_0 * math.cos(phase))
        / (2.0 * math.pi * beta2 * E_TOTAL_0)
    )
    bracket = (2.0 * math.cos(phase)
               - (math.pi - 2.0 * phase) * math.sin(phase))
    temp = -(QM_RATIO * voltage * bracket) / (math.pi * beta2 * E_TOTAL_0 * harmonic * ETA_0)
    dpmax = math.sqrt(temp) if temp > 0.0 else 0.0
    zmax = RADIUS * (math.pi - 2.0 * phase) / harmonic

    return {
        "dE_syn": dE_syn,
        "Qs": qs,
        "dpmax": dpmax,
        "zmax": zmax,
        "dp_aperture": 1.08 * dpmax,   # dp acceptance [lower, upper] = +- this
    }


def make_test_particles(harmonic: int, dpmax: float) -> list:
    """13 tagged test particles (+2 for RF-period symmetry).

    The stored coordinate is bunch-relative, so the bunch-center particle
    always has z_rel = 0.  For this one-bunch example z_center = 0 as well,
    hence z_rel = 0 receives the synchronous RF gain.  The two additional
    h=2 particles are separated by one RF period C/h = C/2.

    tag  1: z=z_sync, dp=0         -> synchronous particle (energy gain)
    tag  2-3: z=z_sync +- 3 m      -> Qs from z oscillation
    tag  4-5: dp=+-1e-3 (z=z_sync) -> Qs from dp oscillation
    tag  6-11: dp=+-{0.5,0.8,1.0}*dpmax -> bucket boundary scan
    tag 12: dp=+1.2*dpmax          -> outside bucket (loss via dp aperture)
    tag 13: x=3mm, px=1e-4         -> adiabatic damping (bunch-level check)
    tag 14-15 (h=2 case): z=+-C/2 -> one-period RF symmetry
    """
    z_sync = 0.0   # bunch-relative: the bunch center is always z_rel = 0
    dp_frac = [0.5, 0.8, 1.0]
    particles = [
        [0.0, 0.0, 0.0, 0.0, z_sync, 0.0],          # tag  1: synchronous
        [0.0, 0.0, 0.0, 0.0, z_sync + 3.0, 0.0],    # tag  2: z +
        [0.0, 0.0, 0.0, 0.0, z_sync - 3.0, 0.0],    # tag  3: z -
        [0.0, 0.0, 0.0, 0.0, z_sync, +1.0e-3],      # tag  4: dp +
        [0.0, 0.0, 0.0, 0.0, z_sync, -1.0e-3],      # tag  5: dp -
    ]
    for frac in dp_frac:
        particles.append([0.0, 0.0, 0.0, 0.0, z_sync, +frac * dpmax])
        particles.append([0.0, 0.0, 0.0, 0.0, z_sync, -frac * dpmax])
    particles.append([0.0, 0.0, 0.0, 0.0, z_sync, +1.2 * dpmax])   # tag 12
    particles.append([3.0e-3, 1.0e-4, 0.0, 0.0, z_sync, 0.0])      # tag 13

    if harmonic == 2:
        # For h=2, +/-C/2 differ from z=0 by one RF period C/h.
        # Both therefore receive the same kick without coordinate folding
        # or a parity-dependent phase correction.
        particles.append([0.0, 0.0, 0.0, 0.0, +CIRCUM / 2.0, 0.0])   # tag 14
        particles.append([0.0, 0.0, 0.0, 0.0, -CIRCUM / 2.0, 0.0])   # tag 15

    return particles


# ============================================================
# Case configuration (single source of truth)
# ============================================================

CASES = {
    "twiss_h1_fixed": dict(
        lattice="twiss",
        rf_mode="fixed",
        voltage=RF_VOLTAGE,
        harmonic=1,
        phase=RF_PHASE,
        phi_offset=RF_PHI_OFFSET,
        num_turns=NUM_TURNS,
        checks=["energy_gain", "qs_fft", "bucket_scan", "bucket_plot",
                "damping", "loss"],
    ),
    "twiss_h2_fixed": dict(
        lattice="twiss",
        rf_mode="fixed",
        voltage=RF_VOLTAGE,
        harmonic=2,
        phase=RF_PHASE,
        phi_offset=RF_PHI_OFFSET,
        num_turns=NUM_TURNS,
        checks=["energy_gain", "qs_fft", "h2_symmetry", "bucket_scan"],
    ),
    "twiss_h1_ramping": dict(
        lattice="twiss",
        rf_mode="file",
        voltage=RF_VOLTAGE,
        harmonic=1,
        phase=RF_PHASE,
        phi_offset=RF_PHI_OFFSET,
        num_turns=200,
        ramp_file="rf_ramp.tfs",
        ramp_slope=0.02,          # V(n) = V0 * (1 + slope * n)
        ramp_rows=50,
        checks=["ramping_gain", "ramping_clamp"],
    ),
    "element_h1_fixed": dict(
        lattice="element",
        rf_mode="fixed",
        voltage=RF_VOLTAGE,
        harmonic=1,
        phase=RF_PHASE,
        phi_offset=RF_PHI_OFFSET,
        num_turns=NUM_TURNS,
        checks=["energy_gain", "qs_fft", "bucket_scan", "loss"],
    ),
}

# Cross-case comparison (twiss vs element first-order drift error)
CASE_PAIRS = [("twiss_h1_fixed", "element_h1_fixed")]


def input_path(case_name: str) -> Path:
    """Return the generated JSON path for a named case."""
    return SCRIPT_DIR / f"beam0_{case_name}.json"


def selected_cases(case_name: str) -> list[str]:
    """Expand the all shortcut and validate a user-provided case name."""
    if case_name == "all":
        return list(CASES)
    return [case_name]


def build_ramp_tfs(script_dir: Path, case: dict) -> str:
    """Write the RF ramping TFS file: one row per turn (V ramps linearly)."""
    n_rows = case["ramp_rows"]
    slope = case["ramp_slope"]
    turns = np.arange(n_rows)
    voltage = case["voltage"] * (1.0 + slope * turns)

    df = tfs_lib.TfsDataFrame(
        {"HARMONIC": np.full(n_rows, case["harmonic"], dtype=np.int64),
         "VOLTAGE": voltage,
         "PHASE": np.full(n_rows, case["phase"]),
         "PHI_OFFSET": np.full(n_rows, case["phi_offset"])}
    )
    path = script_dir / case["ramp_file"]
    tfs_lib.write(str(path), df)
    return str(path)


def build_items(case: dict, script_dir: Path):
    """Return (items, names) for the sequence, excluding injection/monitors."""
    if case["lattice"] == "twiss":
        # One Twiss point covering the whole ring (s_prev=0 -> s=C),
        # longitudinal_transfer="drift": z += -eta * C * dp per turn.
        # Horizontal tune Qx=0.2, Qy=0.15 for a well-defined linear map.
        twiss = TwissPoint(
            s=CIRCUM, s_previous=0.0,
            alpha_x=0.0, alpha_y=0.0,
            beta_x=10.0, beta_y=10.0,
            mu_x=0.2, mu_y=0.15, mu_z=0.0,
            dx=0.0, dpx=0.0,
            alpha_x_previous=0.0, alpha_y_previous=0.0,
            beta_x_previous=10.0, beta_y_previous=10.0,
            mu_x_previous=0.0, mu_y_previous=0.0, mu_z_previous=0.0,
            dx_previous=0.0, dpx_previous=0.0,
            dqx=0.0, dqy=0.0,
            longitudinal_transfer="drift",
        )
        items = [twiss]
        names = [f"twiss_ring_s{CIRCUM:.3f}"]
    else:
        # Real FODO ring from fodo.tfs (example 03/04 lattice).  K-values are
        # normalised (divided by beam rigidity) so they are energy-independent
        # optics parameters in PASS, exactly as in MADX; no B-rho rescaling.
        # Momentum compaction (gamma_t = 3.3746) emerges from the dipole
        # longitudinal mapping — this is what the Qs comparison verifies.
        from PASS.para.madx import read_madx_elements
        items, names, _ = read_madx_elements(
            str(script_dir / "fodo.tfs"), is_merge_drift=True)

    return items, names


def build_case(name: str, script_dir: Path) -> str:
    """Build beam0_<name>.json for one case."""
    case = CASES[name]
    theory = calc_theory(case["voltage"], case["harmonic"], case["phase"])

    # --- RF data file (ramping mode) ---
    rf_file = None
    if case["rf_mode"] == "file":
        rf_file = build_ramp_tfs(script_dir, case)
        rf_voltage = case["voltage"]          # first-row value
    else:
        rf_voltage = case["voltage"]

    # --- test particles ---
    test_particles = make_test_particles(case["harmonic"], theory["dpmax"])
    n_test = len(test_particles)
    n_total = NUM_DIST + n_test

    # --- main config ---
    main = MainConfig(
        beam_name=BEAM_NAME,
        num_proton=NUM_PROTON,
        num_neutron=NUM_NEUTRON,
        num_electron=NUM_CHARGE,
        gamma_t=GAMMA_T,
        circumference=CIRCUM,
        num_turns=case["num_turns"],
        backend="cpu",
        num_gpu=1,
        gpu_id=[0],
        output_dir=str(script_dir / "output" / name),
        is_plot=False,
        is_space_charge=False,
        is_beambeam=False,
    )

    # --- bunch (RF parameters of the injection distribution MUST match the
    #     RFCavity element: sigma_z / sigma_dp sit inside the bucket) ---
    bunch = BunchConfig(
        kinetic_energy=KINETIC_ENERGY,
        num_real_particles=n_total,
        num_macro_particles=n_total,
        is_load_from_file=False,
        file_path="",
        injection_turns=1,
        injection_interval=1,
        alpha_x=0.0,
        alpha_y=0.0,
        beta_x=10.0,
        beta_y=10.0,
        emit_x=200e-6,
        emit_y=100e-6,
        dx=0.0,
        dpx=0.0,
        sigma_z=SIGMA_Z,
        dp=SIGMA_DP,
        dist_trans="kv",
        dist_longi="gaussian",
        rf_voltage=rf_voltage,
        rf_phase=case["phase"],
        rf_s_position=0.0,
        momentum_offset_dp=0.0,
        kinetic_energy_offset=0.0,
        save_init_dist=False,
        insert_particle=test_particles,
        offset_x=OffsetConfig(),
        offset_y=OffsetConfig(),
    )

    # --- monitors ---
    monitors = [
        StatMonitor(s=0.0),
        ParticleMonitor(s=0.0, max_tag=n_test, start_turn=0, end_turn=-1),
    ]

    # --- RF cavity (fixed mode or file mode) ---
    items, names = build_items(case, script_dir)
    rfcavity = RFCavityElement(
        s=0.0,
        voltage=case["voltage"],
        harmonic=case["harmonic"],
        phase=case["phase"],
        phi_offset=case["phi_offset"],
        rf_data_file=rf_file,
        is_enabled=True,
        dp_aperture=[-theory["dp_aperture"], theory["dp_aperture"]],
    )
    items.insert(0, rfcavity)
    names.insert(0, f"rfcavity_h{case['harmonic']}_s0.000")

    # One bunch per bucket: declare empty 0-particle bunches for the
    # unfilled buckets (harmonic number = number of declared bunches).
    bunches = [bunch]
    for i in range(1, case["harmonic"]):
        empty = bunch.model_copy(deep=True)
        empty.num_real_particles = 0
        empty.num_macro_particles = 0
        empty.harmonic_id = i
        bunches.append(empty)

    seq = build_sequence(items=items, names=names, bunches=bunches,
                         monitors=monitors)

    output_path = str(input_path(name))
    generate_input(main, seq, output_path)

    print(f"[{name}]")
    print(f"  lattice={case['lattice']}, rf_mode={case['rf_mode']}, h={case['harmonic']}, "
          f"V={case['voltage']/1e3:.1f} kV, phi_s={case['phase']:.3f} rad")
    print(f"  theory: dE_syn={theory['dE_syn']:.3f} eV/u, Qs={theory['Qs']:.6e}, "
          f"dpmax={theory['dpmax']:.4e}, zmax={theory['zmax']:.2f} m, "
          f"dp_aperture=+-{theory['dp_aperture']:.4e}")
    print(f"  particles: {n_test} tagged + {NUM_DIST} distribution = {n_total}, "
          f"turns={case['num_turns']}")
    print(f"  -> {output_path}\n")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Example 05 inputs.")
    parser.add_argument(
        "--case",
        choices=["all", *CASES],
        default="all",
        help="Input case to generate (default: all).",
    )
    args = parser.parse_args()

    for case_name in selected_cases(args.case):
        build_case(case_name, SCRIPT_DIR)
