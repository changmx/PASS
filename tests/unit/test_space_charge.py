from types import SimpleNamespace

import numpy as np
import h5py

from PASS.commands.space_charge import SpaceCharge
from PASS.para.schema.space_charge import SpaceChargeConfig, SpaceChargeResourceConfig


def _simulation(*, ratio=1.0e9, num_charge=1, delta_z=0.01, sc_length=0.1, output_dir=None):
    particles = SimpleNamespace(
        x=np.array([-2.0e-3, 2.0e-3, 0.0]),
        y=np.array([0.0, 0.0, 0.0]),
        px=np.zeros(3),
        py=np.zeros(3),
        z=np.array([1.0, 2.0, 3.0]),
        dp=np.array([0.1, 0.2, 0.3]),
        tag=np.array([1, 1, -1], dtype=np.int32),
    )
    slice_set = SimpleNamespace(
        slice_id=np.array([0, 0, -1], dtype=np.int32),
        slice_table={"delta_z": np.array([delta_z])},
    )
    bunch = SimpleNamespace(
        bunch_id=0,
        start_idx=0,
        end_idx=3,
        slice_sets={"space_charge": slice_set},
        ratio=ratio,
        num_charge=num_charge,
        beta=0.5,
        gamma=1.1547005383792517,
        brho=3.0,
    )
    beam = SimpleNamespace(particles=particles, bunches=[bunch])
    cfg = SimpleNamespace(
        output_dir=output_dir or ".",
        output_dir_space_charge=str(output_dir / "space_charge") if output_dir else ".",
        output_hms="test",
        backend="cpu",
        particle_precision="float64",
        input_data=[{
            "sequence": {
                "slicer": {
                    "command": "slicer",
                    "slice set": "space_charge",
                },
                "space_charge": {
                    "command": "spacecharge",
                    "configuration": "test",
                }
            }
        }],
        space_charge=[SpaceChargeConfig(
            enabled=True,
            configurations={
                "test": SpaceChargeResourceConfig(
                    slice_set="space_charge",
                    nx=33,
                    ny=33,
                    grid_width_x=0.02,
                    grid_width_y=0.02,
                    solver='fd_dirichlet',
                    deposition_method="CIC",
                )
            },
        )],
        space_charge_configuration_counts=[1],
    )
    return SimpleNamespace(beams=[beam], state=SimpleNamespace(turn=0), cfg=cfg)


def _command(sim, **extra):
    values = {
        "S (m)": 0.0,
        "Configuration": "test",
        "SC length (m)": 0.1,
    }
    values.update(extra)
    return SpaceCharge(0, sim, **values)


def test_space_charge_applies_outward_kick_without_particle_side_effects():
    sim = _simulation()
    p = sim.beams[0].particles
    before = {name: getattr(p, name).copy() for name in ("x", "y", "z", "dp", "tag")}
    assert _command(sim).execute_cpu(sim) is True
    assert p.px[0] < 0.0 and p.px[1] > 0.0
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(p, name), value)
    assert p.px[2] == 0.0 and p.py[2] == 0.0


def test_delta_z_and_sc_length_scale_only_the_kick():
    sim_a = _simulation(delta_z=0.01, sc_length=0.1)
    sim_b = _simulation(delta_z=0.02, sc_length=0.2)
    _command(sim_a, **{"SC length (m)": 0.1}).execute_cpu(sim_a)
    _command(sim_b, **{"SC length (m)": 0.2}).execute_cpu(sim_b)
    # Doubling both normalization width and integration length cancels in this
    # comparison, while the source density and all particle coordinates match.
    np.testing.assert_allclose(sim_b.beams[0].particles.px, sim_a.beams[0].particles.px)


def test_resources_are_reused_between_turns():
    sim = _simulation()
    command = _command(sim)
    resources = command._resources
    command.execute_cpu(sim)
    sim.state.turn = 1
    command.execute_cpu(sim)
    assert command._resources is resources


def test_space_charge_saves_selected_hdf5_snapshot(tmp_path):
    sim = _simulation(output_dir=tmp_path)
    command = _command(sim, **{"Save field": True, "Save density": True, "Save potential": True, "Save turns": [0]})
    assert command.execute_cpu(sim) is True
    files = list(tmp_path.glob("space_charge/*/turn_000000/*.h5"))
    assert len(files) == 1
    with h5py.File(files[0], "r") as handle:
        assert handle["integrated_Ex"].shape == (1, 33, 33)
        assert handle["integrated_Ey"].shape == (1, 33, 33)
        assert handle["charge_density"].shape == (1, 33, 33)
        assert handle["potential"].shape == (1, 33, 33)
        assert handle["delta_z"].shape == (1,)
        assert handle.attrs["turn"] == 0
        assert handle.attrs["potential_gauge"] == "boundary_zero"


def test_space_charge_does_not_save_unselected_turn(tmp_path):
    sim = _simulation(output_dir=tmp_path)
    command = _command(sim, **{"Save field": True, "Save turns": [[1]]})
    command.execute_cpu(sim)
    assert not list(tmp_path.glob("space_charge/**/*.h5"))


def test_enabled_save_flags_with_empty_turns_do_not_save(tmp_path):
    sim = _simulation(output_dir=tmp_path)
    command = _command(
        sim,
        **{
            "Save field": True,
            "Save density": True,
            "Save potential": True,
            "Save turns": [],
        },
    )
    assert command.execute_cpu(sim) is True
    assert not list(tmp_path.glob("space_charge/**/*.h5"))


def test_commands_with_same_configuration_share_resources():
    sim = _simulation()
    first = _command(sim)
    second = _command(sim)
    assert first._resources is second._resources
