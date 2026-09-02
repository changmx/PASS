"""Independent checks for Slicer and DistMonitor snapshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tfs


def _one(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern))
    if len(files) != 1:
        raise AssertionError(f"expected one {pattern!r} under {path}, found {len(files)}")
    return files[0]


def load_outputs(output_dir: str | Path):
    output_dir = Path(output_dir)
    dist = tfs.read(str(_one(output_dir / "distribution", "*.tfs")))
    particles = tfs.read(str(_one(output_dir / "slice", "*_particles.tfs")))
    summary = tfs.read(str(_one(output_dir / "slice", "*_summary.tfs")))
    return dist, particles, summary


def expected_slice_ids(z: np.ndarray, tag: np.ndarray, model: str, num_slices: int) -> np.ndarray:
    alive = tag > 0
    result = np.full(z.size, -1, dtype=np.int32)
    active_z = z[alive]
    z_min, z_max = float(active_z.min()), float(active_z.max())
    width = (z_max - z_min) / num_slices
    if model == "equal_particle":
        order = np.argsort(active_z, kind="stable")
        ranks = np.empty(active_z.size, dtype=np.int64)
        ranks[order] = np.arange(active_z.size, dtype=np.int64)
        result[alive] = num_slices - 1 - np.minimum(
            (ranks * num_slices) // active_z.size, num_slices - 1
        )
    else:
        result[alive] = num_slices - 1 - np.clip(
            np.floor((active_z - z_min) / width), 0, num_slices - 1
        ).astype(np.int32)
    return result


def assert_outputs(output_dir: str | Path, model: str, num_slices: int = 16) -> None:
    print(f"\n=== Slicer analysis: model={model}, output={Path(output_dir).resolve()} ===")
    dist, particles, summary = load_outputs(output_dir)

    def check(name: str, passed: bool, detail: str) -> None:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    dist_tags = dist["tag"].to_numpy()
    particle_tags = particles["tag"].to_numpy()
    tags_equal = np.array_equal(particle_tags, dist_tags)
    check("tag alignment", tags_equal, f"rows={len(dist_tags)}, mismatches={np.count_nonzero(particle_tags != dist_tags)}")

    dist_z = dist["z"].to_numpy(dtype=float)
    particle_z = particles["z"].to_numpy(dtype=float)
    z_error = float(np.max(np.abs(particle_z - dist_z)))
    check("z alignment", z_error <= 1e-14, f"max_abs_error={z_error:.3e}, tolerance=1.000e-14")

    check("turn header", dist.headers["Turn"] == 0, f"measured={dist.headers['Turn']}, expected=0")
    check("coordinate header", particles.headers["ZCoordinate"] == "z_rel", f"measured={particles.headers['ZCoordinate']!r}, expected='z_rel'")
    check("alive particle count", int(dist.headers["NumAlive"]) == 10_000, f"measured={dist.headers['NumAlive']}, expected=10000")

    expected = expected_slice_ids(
        dist_z, dist_tags, model, num_slices
    )
    actual_ids = particles["slice_id"].to_numpy()
    id_mismatches = int(np.count_nonzero(actual_ids != expected))
    check("slice_id", id_mismatches == 0, f"mismatches={id_mismatches}, model={model}")

    counts = np.bincount(expected[expected >= 0], minlength=num_slices)
    actual_counts = summary["macro_count"].to_numpy(dtype=int)
    check("summary count vector", np.array_equal(actual_counts, counts), f"expected_total={int(counts.sum())}, actual_total={int(actual_counts.sum())}")
    for slice_id, (expected_count, actual_count) in enumerate(zip(counts, actual_counts)):
        check(
            f"slice[{slice_id:02d}] macro_count",
            int(expected_count) == int(actual_count),
            f"expected={int(expected_count)}, actual={int(actual_count)}, delta={int(actual_count - expected_count)}",
        )

    check("summary alive count", int(summary["num_alive"].iloc[0]) == int(counts.sum()) == 10_000, f"measured={int(summary['num_alive'].iloc[0])}, expected=10000")
    check("effective slice count", int(summary["effective_num_slices"].iloc[0]) == num_slices, f"measured={int(summary['effective_num_slices'].iloc[0])}, expected={num_slices}")
    check("summary name", summary.headers["Name"] == "PASS Slicer Summary", f"measured={summary.headers['Name']!r}")
    check("summary data type", summary.headers["DataType"] == "per_slice", f"measured={summary.headers['DataType']!r}, expected='per_slice'")
    check("summary turn", int(summary.headers["Turn"]) == 0, f"measured={summary.headers['Turn']}, expected=0")

    z_min = summary["z_min"].to_numpy(dtype=float)
    z_max = summary["z_max"].to_numpy(dtype=float)
    delta_z = summary["delta_z"].to_numpy(dtype=float)
    boundaries_decreasing = bool(np.all(np.diff(z_min) <= 0))
    check("slice boundaries", boundaries_decreasing, f"z_min_first={z_min[0]:.6g}, z_max_last={z_max[-1]:.6g}")
    geometry_error = float(np.max(np.abs(delta_z - (z_max - z_min))))
    check("delta_z geometry", geometry_error <= 2e-12, f"max_abs_error={geometry_error:.3e}, tolerance=2.000e-12")

    if model == "equal_particle":
        spread = int(counts.max() - counts.min())
        check("equal-particle balance", spread <= 1, f"count_spread={spread}, tolerance=1")
