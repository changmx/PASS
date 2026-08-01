"""Generate beam0.json for single-turn Twiss map tracking.

The lattice is a single Twiss point at s=C (one-turn map).
Optical parameters are periodic (beta_prev = beta, alpha_prev = alpha)
so the map is a pure rotation in normalized phase space.

Test particles (12 total, tag 1-12):
    tag  1: x=2mm                    -> Qx measurement
    tag  2: y=2mm                    -> Qy measurement
    tag  3: x=1mm, y=1mm, dp=+1e-4  -> chromaticity
    tag  4: x=1mm, y=1mm, dp=-1e-4  -> chromaticity (symmetric pair)
    tag  5: x=1mm, y=1mm, dp=+5e-4  -> chromaticity
    tag  6: x=1mm, y=1mm, dp=-5e-4  -> chromaticity (symmetric pair)
    tag  7: x=1mm, y=1mm, dp=+1e-3  -> chromaticity (large)
    tag  8: x=1mm, y=1mm, dp=-1e-3  -> chromaticity (large, symmetric pair)
    tag  9: z=0.1m, dp=0             -> Qs measurement (longitudinal)
    tag 10: x=0, y=0, z=0, dp=0      -> reference particle (origin)
    tag 11: x=5mm, y=0               -> large amplitude x
    tag 12: x=0, y=5mm               -> large amplitude y

Plus 10000 KV-distributed particles for beam statistics (tag=0).

Usage:
    python make_input.py
"""

from pathlib import Path

from PASS.para.api import generate_input, build_sequence
from PASS.para.schema.main import MainConfig
from PASS.para.schema.bunch import BunchConfig, OffsetConfig
from PASS.para.schema.twiss import TwissPoint
from PASS.para.schema.monitors import StatMonitor, ParticleMonitor


# ============================================================
# Lattice parameters (periodic one-turn map)
# ============================================================

CIRCUM = 251.327          # m
GAMMA_T = 4.8
NUM_TURNS = 1024

# Transverse twiss (periodic: prev = curr)
ALPHA_X = -2.614303952
ALPHA_Y = 1.57442348
BETA_X = 0.5              # m
BETA_Y = 0.5              # m
MU_X = 0.47               # tune = 0.47
MU_Y = 0.43               # tune = 0.43

# Dispersion (zero for this test)
DX = 0.0
DPX = 0.0

# Chromaticity (non-zero for chromaticity measurement)
DQX = -2.0
DQY = -2.0

# Longitudinal transport: "off" keeps dp constant (identity matrix).
# With "matrix", dp oscillates → tune is phase-modulated → FFT cannot
# measure chromaticity. Use "off" so dp stays fixed and the chromatic
# tune shift Qx(dp) = Qx + DQx*dp is a constant, directly measurable.
# Qs is already verified by analytic matrix comparison (machine precision).
LONGI_TRANSFER = "off"
MU_Z = 0.01              # synchrotron tune (not used when transfer="off")
SIGMA_Z = 90.0            # m
SIGMA_DP = 0.02           # sigma_dp/p

# Beam
KINETIC_ENERGY = 45e6     # eV/u
NUM_REAL = int(1e11)
EMIT_X = 200e-6           # m'rad
EMIT_Y = 100e-6           # m'rad

# Distribution particles for statistics
NUM_DIST = 10000


def make_test_particles():
    """Return 12 test particles [x, px, y, py, z, dp]."""

    dp_list = [1e-4, 5e-4, 1e-3]

    particles = [
        [2e-3, 0, 0, 0, 0, 0],          # tag 1: Qx
        [0, 0, 2e-3, 0, 0, 0],          # tag 2: Qy
    ]

    # Chromaticity pairs: symmetric +/-dp
    for dp in dp_list:
        particles.append([1e-3, 0, 1e-3, 0, 0, +dp])   # odd tags
        particles.append([1e-3, 0, 1e-3, 0, 0, -dp])   # even tags

    particles.append([0, 0, 0, 0, 0.1, 0])           # tag 9: Qs
    particles.append([0, 0, 0, 0, 0, 0])             # tag 10: reference
    particles.append([5e-3, 0, 0, 0, 0, 0])          # tag 11: large amp x
    particles.append([0, 0, 5e-3, 0, 0, 0])          # tag 12: large amp y

    return particles


if __name__ == "__main__":
    # Longitudinal transfer mode determines which measurements are possible:
    #   "off"    → dp fixed → chromaticity measurable via FFT (current setting)
    #   "matrix" → dp oscillates → Qs measurable via FFT, but chromaticity hidden in sidebands
    # Qs and chromaticity cannot be measured by FFT simultaneously. The analytic
    # matrix comparison in analyse.py verifies both regardless of this setting.
    script_dir = Path(__file__).resolve().parent
    output_path = str(script_dir / "beam0.json")

    test_particles = make_test_particles()
    n_test = len(test_particles)
    n_total = NUM_DIST + n_test

    # --- main config ---
    main = MainConfig(
        beam_name="proton",
        num_proton=1,
        num_neutron=0,
        num_electron=1,
        gamma_t=GAMMA_T,
        circumference=CIRCUM,
        num_turns=NUM_TURNS,
        backend="cpu",
        num_gpu=1,
        gpu_id=[0],
        output_dir=str(script_dir / "output"),
        is_plot=False,
        is_space_charge=False,
        is_beambeam=False,
    )

    # --- bunch ---
    bunch = BunchConfig(
        kinetic_energy=KINETIC_ENERGY,
        num_real_particles=NUM_REAL,
        num_macro_particles=n_total,
        is_load_from_file=False,
        file_path="",
        injection_turns=1,
        injection_interval=1,
        alpha_x=ALPHA_X,
        alpha_y=ALPHA_Y,
        beta_x=BETA_X,
        beta_y=BETA_Y,
        emit_x=EMIT_X,
        emit_y=EMIT_Y,
        dx=DX,
        dpx=DPX,
        sigma_z=SIGMA_Z,
        dp=SIGMA_DP,
        dist_trans="kv",
        dist_longi="gaussian",
        rf_voltage=100e3,
        rf_phase=0.5235987755982988,
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

    # --- twiss point (one-turn map at s=C) ---
    twiss = TwissPoint(
        s=CIRCUM,
        s_previous=0.0,
        alpha_x=ALPHA_X,
        alpha_y=ALPHA_Y,
        beta_x=BETA_X,
        beta_y=BETA_Y,
        mu_x=MU_X,
        mu_y=MU_Y,
        mu_z=MU_Z,
        dx=DX,
        dpx=DPX,
        alpha_x_previous=ALPHA_X,
        alpha_y_previous=ALPHA_Y,
        beta_x_previous=BETA_X,
        beta_y_previous=BETA_Y,
        mu_x_previous=0.0,
        mu_y_previous=0.0,
        mu_z_previous=0.0,
        dx_previous=0.0,
        dpx_previous=0.0,
        dqx=DQX,
        dqy=DQY,
        longitudinal_transfer=LONGI_TRANSFER,
    )

    # --- monitors ---
    monitors = [
        StatMonitor(s=0.0),
        ParticleMonitor(s=0.0, max_tag=n_test, start_turn=0, end_turn=-1),
    ]

    # --- build sequence ---
    seq = build_sequence(
        items=[twiss],
        names=["twiss1"],
        bunches=[bunch],
        monitors=monitors,
    )

    # --- write ---
    generate_input(main, seq, output_path)

    print(f"\n[Done] {output_path}")
    print(f"  {n_test} test particles + {NUM_DIST} distribution particles = {n_total} total")
    print(f"  {NUM_TURNS} turns")
    print(f"  Qx={MU_X}, Qy={MU_Y}, Qs={MU_Z}")
    print(f"  DQx={DQX}, DQy={DQY}")
    print(f"  beta_x={BETA_X} m, beta_y={BETA_Y} m")
    print(f"  alpha_x={ALPHA_X}, alpha_y={ALPHA_Y}")
    print(f"  longitudinal: {LONGI_TRANSFER}, mu_z={MU_Z}")
