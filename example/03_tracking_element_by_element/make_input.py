"""Generate PASS input JSON for full-ring element-by-element tracking.

Reads a MADX Twiss TFS file and produces a beam0.json input with test particles
organised in groups for tune, chromaticity, ADTS and coupling measurement.

Test particles (17 total):
    Group A -- linear tune (2):
        tag  1: x=2mm                    -> Qx
        tag  2: y=2mm                    -> Qy

    Group B -- chromaticity, small dp (8):
        tag  3-10: x=1mm, y=1mm, dp = +/-5e-5, +/-1e-4, +/-5e-4, +/-1e-3
        Symmetric +/-dp pairs -> linear fit gives DQ1, DQ2

    Group C -- large dp, nonlinear chromaticity (2):
        tag 11: x=1mm, y=1mm, dp=+3e-3
        tag 12: x=1mm, y=1mm, dp=-3e-3

    Group D -- amplitude-dependent tune shift (4):
        tag 13: x=5mm,  y=0       -> Qx(large), pure single-plane
        tag 14: x=10mm, y=0       -> Qx(larger)
        tag 15: x=0,    y=5mm     -> Qy(large), pure single-plane
        tag 16: x=0,    y=10mm    -> Qy(larger)

    Group E -- coupling (1):
        tag 17: x=3mm, y=3mm

Usage:
    python generate_beam0.py
"""

from pathlib import Path

from PASS.para.api import generate_from_tfs


def make_test_particles():
    """Generate 17 test particles for tune/chromaticity/ADTS/coupling."""

    dp_list = [5e-5, 1e-4, 5e-4, 1e-3]
    adts_x = [5e-3, 10e-3]
    adts_y = [5e-3, 10e-3]

    particles = []

    # Group A: linear tune
    particles.append([2e-3, 0, 0, 0, 0, 0])  # tag 1: Qx
    particles.append([0, 0, 2e-3, 0, 0, 0])  # tag 2: Qy

    # Group B: chromaticity (symmetric +/-dp pairs)
    for dp in dp_list:
        particles.append([1e-3, 0, 1e-3, 0, 0, +dp])
        particles.append([1e-3, 0, 1e-3, 0, 0, -dp])

    # Group C: large dp (nonlinear chromaticity)
    particles.append([1e-3, 0, 1e-3, 0, 0, +3e-3])
    particles.append([1e-3, 0, 1e-3, 0, 0, -3e-3])

    # Group D: amplitude-dependent tune shift (single-plane: y=0 for x scan, x=0 for y scan)
    for ax in adts_x:
        particles.append([ax, 0, 0, 0, 0, 0])
    for ay in adts_y:
        particles.append([0, 0, ay, 0, 0, 0])

    # Group E: coupling
    particles.append([3e-3, 0, 3e-3, 0, 0, 0])

    return particles


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    twiss_file = str(script_dir / "fodo.tfs")
    output_path = str(script_dir / "beam0.json")

    insert_particles = make_test_particles()
    num_particles = len(insert_particles)

    generate_from_tfs(
        twiss_file=twiss_file,
        output_path=output_path,
        main=dict(
            beam_name="proton",
            num_proton=1,
            num_neutron=0,
            num_electron=1,
            gamma_t=7.635,  # auto-filled from TFS if omitted
            circumference=569.098,  # auto-filled from TFS if omitted
            num_turns=1024,
            backend="cpu",
            num_gpu=1,
            gpu_id=[0],
            output_dir=str(script_dir / "output"),
            is_plot=False,
            is_space_charge=False,
            is_beambeam=False,
        ),
        bunches=[
            dict(
                kinetic_energy=50e3,  # eV/u, auto-filled from TFS if omitted
                num_real_particles=num_particles,
                num_macro_particles=num_particles,
                is_load_from_file=False,
                file_path="",
                injection_turns=1,
                injection_interval=1,
                alpha_x=-2.614303952,
                alpha_y=1.57442348,
                beta_x=17.56341783,
                beta_y=8.624482365,
                emit_x=200e-6,
                emit_y=100e-6,
                dx=0.0,
                dpx=0.0,
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
                insert_particle=insert_particles,
                offset_x=dict(
                    is_offset=False,
                    is_load_from_file=False,
                    file_path="",
                    file_time_kind="turn",
                    offset_position=0.0,
                    offset_momentum=0.0,
                ),
                offset_y=dict(
                    is_offset=False,
                    is_load_from_file=False,
                    file_path="",
                    file_time_kind="turn",
                    offset_position=0.0,
                    offset_momentum=0.0,
                ),
            )
        ],
        element_settings=dict(
            # quadrupole
            quad_slices=5,
            quad_model="drift-kick-drift-exact",
            quad_integrator="yoshida4",
            # sbend
            bend_slices=1,
            bend_model="rot-kick-rot",
            bend_integrator="yoshida4",
            # sextupole
            sext_slices=1,
            sext_integrator="yoshida4",
            # octupole
            oct_slices=1,
            oct_integrator="yoshida4",
        ),
        monitors=[
            dict(
                type="StatMonitor",
                s=0.0,
            ),
            dict(
                type="ParticleMonitor",
                s=0.0,
                max_tag=num_particles,
                start_turn=0,
                end_turn=-1,
            ),
        ],
        is_merge_drift=True,
        error_file="",
        is_field_error=False,
    )

    print(f"\n[Done] {output_path}")
    print(f"  {num_particles} particles, 1024 turns")
