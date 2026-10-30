"""Run chamber examples with separate time-step and population-control comparisons."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PASS.commands.ecloud.buildup import BuildUpCloud
from PASS.commands.ecloud.diagnostics import population_statistics
from PASS.commands.ecloud.io import write_buildup_snapshot
from PASS.para.api import (ElectronCloudBeamDriver, ElectronCloudBuildUpConfiguration, ElectronCloudConfiguration, PopulationControlConfiguration,
                           PrimarySourceConfiguration, SurfaceEmissionConfiguration)
from PASS.utils.constants import const


def make_configuration(kind, time_step=5e-11, population_control="off"):
    shapes = dict(circle=[.02],
                  ellipse=[.025, .02],
                  rectangle=[.025, .02],
                  racetrack=[.01, .015, .02, .015],
                  polygon=[[-.02, -.02], [.025, -.015], [.02, .02], [-.025, .015]])
    driver = ElectronCloudBeamDriver(sigma_x=.003, sigma_y=.002, center_x=.0005)
    if kind == "rectangle":
        axis = np.linspace(-.03, .03, 33)
        x, y = np.meshgrid(axis, axis)
        density = np.exp(-.5 * (((x - .0005) / .003)**2 + (y / .002)**2))
        driver = ElectronCloudBeamDriver(model="density_map", x=axis.tolist(), y=axis.tolist(), density=density.tolist())
    sources = [
        PrimarySourceConfiguration(kind="gas_ionization", pressure_pa=1e-6, ionization_cross_section_m2=2e-22, macroparticles=8),
        PrimarySourceConfiguration(kind="prescribed_wall", electrons_per_particle_per_m=1e-5, macroparticles=8),
        PrimarySourceConfiguration(kind="beam_loss", loss_fraction_per_m=1e-7, electrons_per_lost_particle=10., macroparticles=8),
        PrimarySourceConfiguration(kind="photoelectric", photon_rate_per_m_per_s=1e11, quantum_efficiency=.1, macroparticles=8)
    ]
    if population_control not in {"off", "coarse", "fine"}:
        raise ValueError("population_control must be off, coarse or fine")
    population = None
    if population_control != "off":
        population = PopulationControlConfiguration(target_macroparticles=100,
                                                    spatial_cell_size=.002 if population_control == "coarse" else .001,
                                                    momentum_cell_size=.01 if population_control == "coarse" else .003)
    parameters = ElectronCloudBuildUpConfiguration(beam_sigma=.002,
                                                   beam_driver=driver,
                                                   self_fields=True,
                                                   max_time_step=time_step,
                                                   initial_energy_ev=20.,
                                                   primary_sources=sources,
                                                   surface_emission=SurfaceEmissionConfiguration(true_yield_max=1.3,
                                                                                                 elastic_probability=.2,
                                                                                                 rediffused_probability=.1),
                                                   population_control=population,
                                                   max_macroparticles=5000)
    return ElectronCloudConfiguration(mode="build_up",
                                      solver="round_gaussian_beam",
                                      electron_density=2e9,
                                      radius=.003,
                                      n_macroparticles=128,
                                      random_seed=123,
                                      nx=33,
                                      ny=33,
                                      grid_width_x=.07,
                                      grid_width_y=.07,
                                      aperture_type=kind,
                                      aperture_value=shapes[kind],
                                      buildup=parameters)


def run_case(folder, name, configuration, backend):
    cloud = BuildUpCloud(configuration, backend=backend)
    history = []
    try:
        cloud.advance_to(0.)
        for start in (0., 4e-9):
            if start > cloud.time:
                before = cloud.time
                row = cloud.advance_gap(start)
                history.append(dict(time_start=before, time_end=start, phase="gap", **row))
            duration, population, beta = 5e-10, 1e8, .8
            row = cloud.advance_slice(start + duration,
                                      n_real_particles=population,
                                      line_charge=population * const.e / (beta * const.c * duration),
                                      beam_beta=beta)
            history.append(dict(time_start=start, time_end=start + duration, phase="slice", **row))
        before = cloud.time
        row = cloud.advance_gap(8e-9)
        history.append(dict(time_start=before, time_end=8e-9, phase="gap", **row))
        result = dict(case=name, backend=backend, **population_statistics(cloud.state, core_radius=.001))
        population = configuration.buildup.population_control
        result.update(population_control_enabled=population is not None,
                      population_target=population.target_macroparticles if population is not None else None,
                      population_target_reached=result["n_macroparticles"] <= population.target_macroparticles if population is not None else None,
                      merged_groups=int(cloud.state.counters.get("merged_groups", 0)),
                      split_parents=int(cloud.state.counters.get("split_parents", 0)))
        if not all(np.isfinite(value) for value in result.values() if isinstance(value, (int, float))):
            raise RuntimeError(f"Nonfinite diagnostic in {name}: {result}")
        metadata = dict(configuration=configuration.model_dump(mode="json"), observation_time=cloud.time, population=result)
        write_buildup_snapshot(folder / f"{name}.h5", cloud, metadata, history)
        (folder / f"{name}_configuration.json").write_text(json.dumps(configuration.model_dump(by_alias=True), indent=2), encoding="utf-8")
        return result, history
    finally:
        cloud.close()


def run(output, backend="cpu"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases, histories = [], []
    for kind in ("circle", "ellipse", "rectangle", "racetrack", "polygon"):
        result, history = run_case(output, kind, make_configuration(kind), backend)
        cases.append(result)
        histories.append(history)
    fine, fine_history = run_case(output, "ellipse_half_step", make_configuration("ellipse", time_step=2.5e-11), backend)
    cases.append(fine)
    histories.append(fine_history)
    comparison = {}
    quantities = ("n_electrons", "mean_energy_ev", "rms_x", "rms_y", "effective_macroparticles")
    for quantity in quantities:
        coarse_value, fine_value = cases[1][quantity], fine[quantity]
        comparison[quantity] = dict(coarse=coarse_value,
                                    fine=fine_value,
                                    relative_change=abs(coarse_value - fine_value) / max(abs(fine_value),
                                                                                         np.finfo(float).tiny))
    population_comparison = {}
    for control in ("coarse", "fine"):
        name = "ellipse_population_control" if control == "coarse" else "ellipse_fine_population_control"
        controlled, history = run_case(output, name, make_configuration("ellipse", time_step=2.5e-11, population_control=control), backend)
        cases.append(controlled)
        histories.append(history)
        population_comparison[control] = dict(baseline="ellipse_half_step",
                                              controlled_case=name,
                                              maximum_time_step_s=2.5e-11,
                                              population_target=controlled["population_target"],
                                              population_target_reached=controlled["population_target_reached"],
                                              n_macroparticles=controlled["n_macroparticles"],
                                              effective_macroparticles=controlled["effective_macroparticles"],
                                              merged_groups=controlled["merged_groups"],
                                              quantities={
                                                  quantity:
                                                  dict(baseline=fine[quantity],
                                                       controlled=controlled[quantity],
                                                       relative_difference=(controlled[quantity] - fine[quantity]) /
                                                       max(abs(fine[quantity]),
                                                           np.finfo(float).tiny))
                                                  for quantity in quantities
                                              })
    report = dict(
        cases=cases,
        time_step_comparison=comparison,
        time_step_population_control="disabled for both calculations",
        population_control_comparison=population_comparison,
        scope="Prescribed-beam 2D3V examples. Single-seed step and merge comparisons do not establish equilibrium or statistical convergence.")
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 5), layout="constrained")
    labels = dict(ellipse_half_step="ellipse dt/2",
                  ellipse_population_control="ellipse coarse merge",
                  ellipse_fine_population_control="ellipse fine merge")
    for case, history in zip(cases, histories):
        axis.plot([row["time_end"] * 1e9 for row in history], [row["n_electrons"] for row in history],
                  marker="o",
                  label=labels.get(case["case"], case["case"]))
    axis.set(xlabel="Physical time [ns]", ylabel="Represented electron number")
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2, fontsize=8)
    figure.savefig(output / "population_histories.png", dpi=180)
    plt.close(figure)
    print(json.dumps(dict(output=str(output.resolve()), time_step_comparison=comparison), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output",
                        type=Path,
                        default=Path(__file__).resolve().parents[2] / "tests" / "codex" / "ecloud_examples" /
                        ("models_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    arguments = parser.parse_args()
    run(arguments.output, arguments.backend)
