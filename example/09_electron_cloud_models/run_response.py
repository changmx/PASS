"""Run a real coupled-PIC response extraction and independent reconstruction.

Run from the repository root. Output is a new directory; existing artifacts are
never replaced. This example measures a finite train, not a validated long-turn
periodic response or an instability threshold.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PASS.commands.ecloud.buildup import BuildUpCloud
from PASS.commands.ecloud.response import ResponseSlice, extract_response, run_response_train, write_response
from PASS.para.schema.electron_cloud import ElectronCloudBuildUpConfiguration, ElectronCloudConfiguration
from PASS.utils.constants import const


def run_demo(output_directory, backend="cpu"):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    configuration = ElectronCloudConfiguration(mode="coupled",
                                               solver="fd_dirichlet",
                                               electron_density=2e9,
                                               radius=.004,
                                               n_macroparticles=64,
                                               random_seed=123,
                                               nx=33,
                                               ny=33,
                                               grid_width_x=.04,
                                               grid_width_y=.04,
                                               buildup=ElectronCloudBuildUpConfiguration(chamber_radius=.02,
                                                                                         beam_sigma=.001,
                                                                                         max_time_step=5e-11,
                                                                                         primary_electrons_per_particle_per_m=0.,
                                                                                         secondary_yield_max=0.))
    cloud = BuildUpCloud(configuration, backend=backend)
    x, y = np.array([-.0004, -.0004, .0004, .0004]), np.array([-.0003, .0003, -.0003, .0003])
    slices = []
    for turn in range(2):
        for bunch_id in range(2):
            start = turn * 7e-10 + bunch_id * 1.2e-10
            end = start + 1.2e-10
            slices.append(ResponseSlice(start, end, turn, bunch_id, 0, x, y, 1e7 * const.e / (4 * .8 * const.c * (end - start)), 1e7, .8, 3.5, 1.,
                                        1.))
    offsets = np.array([[2e-6, -1e-6], [-3e-6, 2e-6], [1e-6, 1e-6], [0., 0.]])
    try:
        response = extract_response(cloud, slices, epsilon=1e-5, grid_x=np.linspace(-.001, .001, 21), grid_y=np.linspace(-.001, .001, 21))
        write_response(output_directory / "finite_response.h5", response)
        shifted = [
            replace(event, x=event.x + dx, y=event.y + dy, witness_x=event.witness_x + dx, witness_y=event.witness_y + dy)
            for event, (dx, dy) in zip(slices, offsets)
        ]
        actual, _ = run_response_train(cloud, shifted)
        predicted = response.replay(offsets, witness_positions=[(event.witness_x, event.witness_y) for event in shifted])
        base, _ = run_response_train(cloud, slices)
        fixed, _ = run_response_train(cloud, slices, source_offsets=offsets)
        actual_dynamic = np.stack([(new - old).mean(axis=0) for new, old in zip(fixed, base)])
        predicted_dynamic = np.einsum("jiab,ib->ja", response.kernel, offsets)
        error = float(np.linalg.norm(np.concatenate(predicted) - np.concatenate(actual)) / np.linalg.norm(np.concatenate(actual)))
        dynamic_error = float(np.linalg.norm(predicted_dynamic - actual_dynamic) / np.linalg.norm(actual_dynamic))
        report = dict(backend=backend,
                      full_kick_relative_l2_error=error,
                      source_response_relative_l2_error=dynamic_error,
                      measured_full_kicks=[values.tolist() for values in actual],
                      predicted_full_kicks=[values.tolist() for values in predicted],
                      measured_dynamic_kicks=actual_dynamic.tolist(),
                      predicted_dynamic_kicks=predicted_dynamic.tolist(),
                      same_slice_kernel_norm=float(np.linalg.norm(response.kernel[0, 0])),
                      cross_turn_kernel_norm=float(np.linalg.norm(response.kernel[2, 0])),
                      maximum_linearity_error=float(np.max(response.metadata["linearity_relative_error"])),
                      acceptance={
                          "full_kick_relative_l2_error": .01,
                          "source_response_relative_l2_error": .03
                      },
                      passed=error < .01 and dynamic_error < .03,
                      scope="Real coupled PIC; finite four-interval/two-turn train. Not a machine instability-threshold validation.")
        (output_directory / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if not key.startswith(("measured_", "predicted_"))}, indent=2))
        if not report["passed"]:
            raise RuntimeError("Coupled response reconstruction failed its declared tolerances")
        return report
    finally:
        cloud.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output",
                        type=Path,
                        default=Path(__file__).resolve().parents[2] / "tests" / "codex" / "ecloud_examples" /
                        ("response_demo_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    args = parser.parse_args()
    run_demo(args.output, args.backend)
