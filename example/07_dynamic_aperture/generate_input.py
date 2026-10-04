"""Generate a Cartesian x-y-dp scan with a user-editable nonlinear sequence."""

import argparse
from pathlib import Path

from PASS.para.api import generate_input
from PASS.para.schema.bunch import BunchConfig, InjectionItem, ScanGridConfig
from PASS.para.schema.elements import MarkerItem, SextupoleItem
from PASS.para.schema.main import MainConfig
from PASS.para.schema.monitors import ParticleMonitorItem
from PASS.para.schema.sequence import Sequence
from PASS.para.schema.twiss import TwissItem


def build_input(path, *, backend="cpu", turns=256, points=31, k2l=10.0):
    """Write an illustrative nonlinear map, not a calibrated machine lattice."""
    path = Path(path).resolve()
    grid = ScanGridConfig(x_range=[-0.02, 0.02], y_range=[-0.02, 0.02], num_x=points, num_y=points, dp_values=[-0.003, 0.0, 0.003])
    n_particles = points * points * 3
    main = MainConfig(beam_name="proton",
                      num_proton=1,
                      num_neutron=0,
                      num_electron=1,
                      circumference=100.0,
                      gamma_t=5.0,
                      num_turns=turns,
                      backend=backend,
                      output_dir=str(path.parent / "output"),
                      is_plot=False)
    bunch = BunchConfig(kinetic_energy=100e6, num_real_particles=1000000, num_macro_particles=n_particles, scan_grid=grid, save_init_dist=True)
    sequence = Sequence()
    sequence.add("injection", InjectionItem(bunches=[bunch], random_seed=1))
    sequence.add(
        "rotation",
        TwissItem(s=100.0,
                  s_previous=0.0,
                  order=10,
                  alpha_x=0.0,
                  alpha_y=0.0,
                  beta_x=10.0,
                  beta_y=10.0,
                  mu_x=0.28,
                  mu_y=0.31,
                  dx=0.0,
                  dpx=0.0,
                  alpha_x_previous=0.0,
                  alpha_y_previous=0.0,
                  beta_x_previous=10.0,
                  beta_y_previous=10.0,
                  mu_x_previous=0.0,
                  mu_y_previous=0.0,
                  dqx=-2.0,
                  dqy=-1.0,
                  longitudinal_transfer="off"))
    sequence.add("nonlinear_kick", SextupoleItem(s=100.0, order=20, length=0.0, k2l=k2l))
    sequence.add("escape_aperture", MarkerItem(s=100.0, order=15, aperture_type="rectangle", aperture_value=[0.08, 0.08]))
    sequence.add("da", ParticleMonitorItem(s=100.0, order=40, max_tag=n_particles, output_format="hdf5", write_interval_turns=32))
    return generate_input(main, sequence, str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("beam0.json"))
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--turns", type=int, default=256)
    parser.add_argument("--points", type=int, default=31)
    parser.add_argument("--k2l", type=float, default=10.0)
    args = parser.parse_args()
    build_input(args.output, backend=args.backend, turns=args.turns, points=args.points, k2l=args.k2l)
