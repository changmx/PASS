"""Generate beam0.json for twiss-by-twiss tracking with sextupole thin lenses.

Reads two MADX TFS files:
    fodo_natural.tfs  — K2=0, natural chromaticity in headers
    fodo.tfs          — K2≠0, corrected optics + sextupole K2L values

The natural chromaticity (DQ1, DQ2) is distributed to each TwissPoint
proportional to phase advance (standard first-order approximation).

Thin-lens sextupoles (length=0) are inserted at their s-positions from
fodo.tfs. The Sequence sorts by (s, priority): Twiss (200) before
Sextupole (300) at the same s, so the sextupole kick is applied after
the linear twiss transport — physically correct.

The total chromaticity seen by particles will be:
    DQ_total = DQ_natural (from TwissPoint dqx/dqy) + DQ_sextupole (from kicks)
             = DQ_natural + (DQ_corrected - DQ_natural) = DQ_corrected

Test particles (12 total):
    tag  1: x=2mm                    -> Qx
    tag  2: y=2mm                    -> Qy
    tag  3-8: x=1mm, y=1mm, dp=±{1e-4, 5e-4, 1e-3}  -> chromaticity
    tag  9: z=0.1m                   -> longitudinal (if enabled)
    tag 10: origin                   -> reference
    tag 11: x=5mm                    -> large amplitude x
    tag 12: y=5mm                    -> large amplitude y

Usage:
    python make_input.py
"""

from pathlib import Path

import tfs as tfs_lib

from PASS.para.api import generate_input, build_sequence
from PASS.para.schema.main import MainConfig
from PASS.para.schema.bunch import BunchConfig, OffsetConfig
from PASS.para.schema.monitors import StatMonitor, ParticleMonitor
from PASS.para.madx import read_madx_twiss


# ============================================================
# Parameters
# ============================================================

NUM_TURNS = 1024
NUM_DIST = 10000

# Longitudinal transfer: "off" keeps dp fixed → chromaticity measurable via FFT.
# See 02_oneturn_map README for the Qs/chromaticity FFT incompatibility.
LONGI_TRANSFER = "off"

# Beam
EMIT_X = 200e-6            # m·rad
EMIT_Y = 100e-6            # m·rad

# Sextupole name patterns for thin-lens insertion
SEXTUPOLE_PATTERNS = ["SF1", "SD1"]


def make_test_particles():
    """Return 12 test particles [x, px, y, py, z, dp]."""

    dp_list = [1e-5, 5e-5, 1e-4]

    particles = [
        [2e-3, 0, 0, 0, 0, 0],          # tag 1: Qx
        [0, 0, 2e-3, 0, 0, 0],          # tag 2: Qy
    ]

    # Chromaticity pairs: symmetric +/-dp
    for dp in dp_list:
        particles.append([1e-3, 0, 1e-3, 0, 0, +dp])   # odd tags
        particles.append([1e-3, 0, 1e-3, 0, 0, -dp])   # even tags

    particles.append([0, 0, 0, 0, 0.1, 0])           # tag 9: longitudinal
    particles.append([0, 0, 0, 0, 0, 0])             # tag 10: reference
    particles.append([5e-3, 0, 0, 0, 0, 0])          # tag 11: large amp x
    particles.append([0, 0, 5e-3, 0, 0, 0])          # tag 12: large amp y

    return particles


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    natural_tfs = str(script_dir / "fodo_natural.tfs")
    corrected_tfs = str(script_dir / "fodo.tfs")
    output_path = str(script_dir / "beam0.json")

    # --- Read natural chromaticity ---
    natural = tfs_lib.read(natural_tfs)
    dq1_nat = natural.headers["DQ1"]
    dq2_nat = natural.headers["DQ2"]
    circum = natural.headers["LENGTH"]
    gamma_tr = natural.headers["GAMMATR"]
    ek_per_nucleon = (natural.headers["ENERGY"] - natural.headers["MASS"]) * 1e9  # GeV -> eV/u

    print(f"[Natural] Q1={natural.headers['Q1']:.6f}, Q2={natural.headers['Q2']:.6f}")
    print(f"[Natural] DQ1={dq1_nat:.6f}, DQ2={dq2_nat:.6f}")
    print(f"[Natural] C={circum:.4f}, gamma_tr={gamma_tr:.4f}, Ekin={ek_per_nucleon:.6e} eV/u")

    # --- Read corrected TFS headers for verification ---
    corrected = tfs_lib.read(corrected_tfs)
    print(f"[Corrected] DQ1={corrected.headers['DQ1']:.6f}, DQ2={corrected.headers['DQ2']:.6f}")
    print(f"[Expected sextupole contribution] dDQ1={corrected.headers['DQ1'] - dq1_nat:.6f}, "
          f"dDQ2={corrected.headers['DQ2'] - dq2_nat:.6f}")

    # --- Build twiss sequence with natural chromaticity + sextupole thin lenses ---
    items, names, _ = read_madx_twiss(
        twiss_file=corrected_tfs,
        dqx=dq1_nat,             # natural chromaticity, distributed by phase advance
        dqy=dq2_nat,
        insert_patterns=SEXTUPOLE_PATTERNS,  # thin-lens sextupoles from corrected TFS
        longitudinal_transfer=LONGI_TRANSFER,
    )

    print(f"[Sequence] {len(items)} items ({sum(1 for i in items if i.command == 'Twiss')} twiss + "
          f"{sum(1 for i in items if i.command == 'Sextupole')} sextupole)")

    # --- Test particles ---
    test_particles = make_test_particles()
    n_test = len(test_particles)
    n_total = NUM_DIST + n_test

    # --- Main config ---
    main = MainConfig(
        beam_name="proton",
        num_proton=1,
        num_neutron=0,
        num_electron=1,
        gamma_t=gamma_tr,
        circumference=circum,
        num_turns=NUM_TURNS,
        backend="cpu",
        num_gpu=1,
        gpu_id=[0],
        output_dir=str(script_dir / "output"),
        is_plot=False,
        is_space_charge=False,
        is_beambeam=False,
    )

    # --- Bunch ---
    bunch = BunchConfig(
        kinetic_energy=ek_per_nucleon,
        num_real_particles=n_total,
        num_macro_particles=n_total,
        is_load_from_file=False,
        file_path="",
        injection_turns=1,
        injection_interval=1,
        alpha_x=natural.iloc[0]["ALFX"],
        alpha_y=natural.iloc[0]["ALFY"],
        beta_x=natural.iloc[0]["BETX"],
        beta_y=natural.iloc[0]["BETY"],
        emit_x=EMIT_X,
        emit_y=EMIT_Y,
        dx=natural.iloc[0]["DX"],
        dpx=natural.iloc[0]["DPX"],
        sigma_z=1.0,
        dp=1e-3,
        dist_trans="kv",
        dist_longi="gaussian",
        rf_voltage=0.0,
        rf_phase=0.0,
        harmonic_number=1,
        harmonic_id=0,
        rf_s_position=0.0,
        momentum_offset_dp=0.0,
        kinetic_energy_offset=0.0,
        save_init_dist=False,
        insert_particle=test_particles,
        offset_x=OffsetConfig(),
        offset_y=OffsetConfig(),
    )

    # --- Monitors ---
    monitors = [
        StatMonitor(s=0.0),
        ParticleMonitor(s=0.0, max_tag=n_test, start_turn=0, end_turn=-1),
    ]

    # --- Build sequence ---
    seq = build_sequence(
        items=items,
        names=names,
        bunches=[bunch],
        monitors=monitors,
    )

    # --- Write ---
    generate_input(main, seq, output_path)

    print(f"\n[Done] {output_path}")
    print(f"  {n_test} test particles + {NUM_DIST} distribution particles = {n_total} total")
    print(f"  {NUM_TURNS} turns")
    print(f"  Natural DQ1={dq1_nat:.4f}, DQ2={dq2_nat:.4f}")
    print(f"  Corrected DQ1={corrected.headers['DQ1']:.4f}, DQ2={corrected.headers['DQ2']:.4f}")
    print(f"  Longitudinal: {LONGI_TRANSFER}")
