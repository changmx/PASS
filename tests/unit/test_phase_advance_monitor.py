import math
from types import SimpleNamespace

import numpy as np
import pytest

from PASS.commands.monitor.phase_advance import PhaseAdvanceMonitor
from PASS.core.particle import ParticlePool
from PASS.para.schema.monitors import PhaseAdvanceMonitor as PhaseAdvanceSchema


def _make_monitor(tmp_path, n_turn=5, n_particles=2, xp=np, **fields):
    particles = ParticlePool(n_particles, xp, dtype=np.float64)
    bunch = SimpleNamespace(
        start_idx=0, end_idx=n_particles, bunch_id=0, harmonic_id=0,
        harmonic_number=1, Np=n_particles, circum=1.0, z_center=0.0,
    )
    beam = SimpleNamespace(particles=particles, bunches=[bunch],
                           Np_total=n_particles, beam_name="test")
    cfg = SimpleNamespace(num_turn=n_turn, output_dir_tuneSpread=str(tmp_path), output_hms="run")
    sim = SimpleNamespace(cfg=cfg, beams=[beam], state=SimpleNamespace(turn=0))
    defaults = {
        "name": "phase", "S (m)": 0.0, "Beta x (m)": 1.0,
        "Beta y (m)": 1.0, "Alpha x": 0.0, "Alpha y": 0.0,
        "Turn ranges": [[0, n_turn]], "Min action": 0.0,
    }
    defaults.update(fields)
    return PhaseAdvanceMonitor(0, sim, **defaults), sim


def test_known_rotation_accumulates_fractional_tune(tmp_path):
    monitor, sim = _make_monitor(tmp_path)
    particles = sim.beams[0].particles
    qx, qy = 0.23, 0.46
    for turn in range(5):
        sim.state.turn = turn
        particles.x[0] = math.cos(-2 * math.pi * qx * turn)
        particles.px[0] = math.sin(-2 * math.pi * qx * turn)
        particles.y[0] = math.cos(-2 * math.pi * qy * turn)
        particles.py[0] = math.sin(-2 * math.pi * qy * turn)
        monitor.execute_cpu(sim)

    import tfs

    output = tfs.read(next(tmp_path.glob("*.tfs")))
    assert output.loc[0, "intervalCountX"] == 4
    assert output.loc[0, "intervalCountY"] == 4
    assert output.loc[0, "tuneXFractional"] == pytest.approx(qx)
    assert output.loc[0, "tuneYFractional"] == pytest.approx(qy)
    assert monitor.windows[0].written
    assert monitor._active_windows == {}


def test_tag_index_survives_reordering(tmp_path):
    monitor, sim = _make_monitor(tmp_path, n_particles=2)
    particles = sim.beams[0].particles
    particles.tag[:] = [2, 1]
    particles.x[:] = [0.0, 1.0]
    particles.px[:] = [1.0, 0.0]
    monitor.execute_cpu(sim)
    particles.x[:] = [1.0, 0.0]
    particles.px[:] = [0.0, -1.0]
    sim.state.turn = 1
    monitor.execute_cpu(sim)
    state = monitor._active_windows[0]
    assert state.interval_x_count.tolist() == [1, 1]


def test_turn_ranges_and_disabled_monitor(tmp_path):
    monitor, _ = _make_monitor(tmp_path, **{"Turn ranges": [[2, 3]]})
    assert monitor.windows == []
    monitor, _ = _make_monitor(tmp_path, **{"Turn ranges": 0})
    assert monitor.windows == []
    monitor, _ = _make_monitor(tmp_path, **{"Enable": False})
    assert monitor.execute_cpu(SimpleNamespace(state=SimpleNamespace(turn=0))) is False


def test_lost_particle_has_no_reported_tune(tmp_path):
    monitor, sim = _make_monitor(tmp_path, n_turn=2, n_particles=1)
    particles = sim.beams[0].particles
    particles.x[0] = 1.0
    monitor.execute_cpu(sim)

    sim.state.turn = 1
    particles.tag[0] = -1
    particles.lost_turn[0] = 1
    monitor.execute_cpu(sim)

    output = next(tmp_path.glob("*.tfs"))
    import tfs

    data = tfs.read(output)
    assert math.isnan(data.loc[0, "tuneXFractional"])
    assert not data.loc[0, "validX"]
    assert not data.loc[0, "completeX"]


def test_invalid_tags_are_retained_without_indexing_accumulators(tmp_path):
    monitor, sim = _make_monitor(tmp_path, n_turn=2, n_particles=2)
    particles = sim.beams[0].particles
    particles.tag[:] = [0, 3]
    monitor.execute_cpu(sim)
    sim.state.turn = 1
    monitor.execute_cpu(sim)

    import tfs

    data = tfs.read(next(tmp_path.glob("*.tfs")))
    assert len(data) == 2
    assert data["intervalCountX"].tolist() == [0, 0]
    assert data["tuneXFractional"].isna().all()


def test_schema_defaults_and_new_command_name():
    dumped = PhaseAdvanceSchema(s=0.0, beta_x=1.0, beta_y=1.0,
                                alpha_x=0.0, alpha_y=0.0).model_dump(by_alias=True)
    assert dumped["Command"] == "PhaseAdvanceMonitor"
    assert dumped["Enable"] is True
    assert dumped["Dx (m)"] == 0.0
    assert dumped["Dpx"] == 0.0
    assert dumped["X CO (m)"] == 0.0


def test_gpu_matches_cpu_when_cuda_is_available(tmp_path):
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device unavailable")

    monitor_cpu, sim_cpu = _make_monitor(tmp_path / "cpu", n_turn=3, n_particles=1)
    particles_cpu = sim_cpu.beams[0].particles
    monitor_gpu, sim_gpu = _make_monitor(tmp_path / "gpu", n_turn=3, n_particles=1, xp=cp)
    particles_gpu = sim_gpu.beams[0].particles

    qx, qy = 0.23, 0.46
    for turn in range(3):
        values = (
            math.cos(-2 * math.pi * qx * turn),
            math.sin(-2 * math.pi * qx * turn),
            math.cos(-2 * math.pi * qy * turn),
            math.sin(-2 * math.pi * qy * turn),
        )
        particles_cpu.x[0], particles_cpu.px[0], particles_cpu.y[0], particles_cpu.py[0] = values
        particles_gpu.x[0], particles_gpu.px[0], particles_gpu.y[0], particles_gpu.py[0] = values
        sim_cpu.state.turn = sim_gpu.state.turn = turn
        monitor_cpu.execute_cpu(sim_cpu)
        monitor_gpu.execute_gpu(sim_gpu)

    import tfs

    cpu_data = tfs.read(next((tmp_path / "cpu").glob("*.tfs")))
    gpu_data = tfs.read(next((tmp_path / "gpu").glob("*.tfs")))
    assert gpu_data.loc[0, "tuneXFractional"] == pytest.approx(cpu_data.loc[0, "tuneXFractional"])
    assert gpu_data.loc[0, "tuneYFractional"] == pytest.approx(cpu_data.loc[0, "tuneYFractional"])
