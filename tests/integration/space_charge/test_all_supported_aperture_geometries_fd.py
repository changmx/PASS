"""Validate arbitrary-aperture FD solutions and their grid convergence.

The contained Gaussian and KV source envelopes deliberately fill a substantial
fraction of every aperture without representing particle interception. The
same integrated source charge is used for every aperture and resolution, so
potential changes come from the boundary model rather than source charge.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from PASS.commands.solver.fd_arbitrary import build_aperture
from PASS.commands.solver.pic import GridGeometry, build_pic_resources
from PASS.utils.aperture import check_aperture_cpu
from PASS.utils.constants import const


OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "output"
    / "gaussian_and_kv_uniform_all_aperture_geometries_fd"
)
GRID_BOUNDS_M = (-0.080, 0.080, -0.065, 0.065)
GRID_SIZES = (65, 129, 257)
REFERENCE_GRID_SIZE = GRID_SIZES[-1]
TOTAL_SOURCE_CHARGE_C = 1.0e-9

# A 4-sigma Gaussian envelope and the exact KV support use the same ellipse.
# Dense boundary sampling verifies below that this ellipse is inside every
# physical aperture. Gaussian tails outside the envelope are negligible but
# are still truncated and renormalized consistently inside each FD domain.
SOURCE_ENVELOPE_SEMI_AXES_M = (0.044, 0.033)
GAUSSIAN_ENVELOPE_SIGMAS = 4.0
GAUSSIAN_SIGMA_M = tuple(
    value / GAUSSIAN_ENVELOPE_SIGMAS for value in SOURCE_ENVELOPE_SEMI_AXES_M
)
KV_SEMI_AXES_M = SOURCE_ENVELOPE_SEMI_AXES_M

POLYGON_VERTICES = [
    [0.0573, 0.0011],
    [0.0317, 0.0432],
    [-0.0331, 0.0414],
    [-0.0562, -0.0017],
    [-0.0298, -0.0421],
    [0.0324, -0.0403],
]
APERTURE_CASES = {
    "off": [],
    "default": [],
    "circle": [0.0523],
    "rectangle": [0.0541, 0.0413],
    "ellipse": [0.0582, 0.0387],
    "rectcircle": [0.0507, 0.0431, 0.0574],
    "rectellipse": [0.0527, 0.0409, 0.0591, 0.0463],
    "racetrack": [0.0314, 0.0357, 0.0238, 0.0357],
    "octagon": [0.0561, 0.0447, 0.0132],
    "polygon": POLYGON_VERTICES,
}

SOURCE_LABELS = {
    "gaussian": "Gaussian (contained 4-sigma envelope)",
    "kv_uniform_projection": "KV uniform projection (contained support)",
}


def _geometry(grid_size: int) -> GridGeometry:
    return GridGeometry(grid_size, grid_size, *GRID_BOUNDS_M)


def _tracking_mask(geometry: GridGeometry, kind: str, values: list) -> np.ndarray:
    xx, yy = np.meshgrid(geometry.x, geometry.y)
    count = xx.size
    particles = SimpleNamespace(
        x=xx.ravel().copy(),
        y=yy.ravel().copy(),
        tag=np.ones(count, dtype=np.int32),
        lost_position=np.full(count, np.nan),
        lost_turn=np.full(count, -1, dtype=np.int32),
    )
    beam = SimpleNamespace(particles=particles)
    bunch = SimpleNamespace(start_idx=0, end_idx=count)
    check_aperture_cpu(beam, bunch, kind, values, s_position=1.25, turn=3)
    return (particles.tag > 0).reshape(xx.shape)


def _source_shapes(geometry: GridGeometry) -> dict[str, np.ndarray]:
    xx, yy = np.meshgrid(geometry.x, geometry.y)
    sigma_x, sigma_y = GAUSSIAN_SIGMA_M
    kv_a, kv_b = KV_SEMI_AXES_M
    return {
        "gaussian": np.exp(-0.5 * ((xx / sigma_x) ** 2 + (yy / sigma_y) ** 2)),
        # A 4-D KV distribution has a uniform projected x-y density in its ellipse.
        "kv_uniform_projection": np.where(
            (xx / kv_a) ** 2 + (yy / kv_b) ** 2 <= 1.0,
            1.0,
            0.0,
        ),
    }


def _normalized_density(
    raw_density: np.ndarray,
    interior: np.ndarray,
    geometry: GridGeometry,
) -> tuple[np.ndarray, float]:
    retained = np.where(interior, raw_density, 0.0)
    retained_sum = float(retained.sum())
    full_sum = float(raw_density.sum())
    assert retained_sum > 0.0
    density = retained * (
        TOTAL_SOURCE_CHARGE_C / (retained_sum * geometry.dx * geometry.dy)
    )
    return density, retained_sum / full_sum


def _assert_source_envelope_is_contained() -> dict[str, bool]:
    angle = np.linspace(0.0, 2.0 * np.pi, 32769)
    a, b = SOURCE_ENVELOPE_SEMI_AXES_M
    envelope_x = a * np.cos(angle)
    envelope_y = b * np.sin(angle)
    containment = {}
    for kind, values in APERTURE_CASES.items():
        if kind in {"off", "default"}:
            inside = (
                (envelope_x > GRID_BOUNDS_M[0])
                & (envelope_x < GRID_BOUNDS_M[1])
                & (envelope_y > GRID_BOUNDS_M[2])
                & (envelope_y < GRID_BOUNDS_M[3])
            )
        else:
            inside = build_aperture(
                {"Type": kind, "Aperture Value": values}
            ).strict_mask(envelope_x, envelope_y)
        containment[kind] = bool(np.all(inside))
        assert containment[kind], f"source envelope intersects {kind} aperture"
    return containment


def _plot_continuous_boundary(ax: plt.Axes, kind: str, values: list) -> None:
    if kind in {"off", "default"}:
        x_min, x_max, y_min, y_max = GRID_BOUNDS_M
        ax.plot(
            np.array([x_min, x_max, x_max, x_min, x_min]) * 1.0e3,
            np.array([y_min, y_min, y_max, y_max, y_min]) * 1.0e3,
            color="black",
            linewidth=1.0,
            zorder=2.0,
        )
        return

    # Evaluate the same continuous predicate used by Shortley-Weller on a
    # display-only dense mesh, independently of the coarse FD nodal mask.
    x_dense = np.linspace(GRID_BOUNDS_M[0], GRID_BOUNDS_M[1], 1001)
    y_dense = np.linspace(GRID_BOUNDS_M[2], GRID_BOUNDS_M[3], 813)
    xx_dense, yy_dense = np.meshgrid(x_dense, y_dense)
    continuous_mask = build_aperture(
        {"Type": kind, "Aperture Value": values}
    ).mask(xx_dense, yy_dense)
    ax.contour(
        x_dense * 1.0e3,
        y_dense * 1.0e3,
        continuous_mask.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=1.0,
        zorder=2.0,
    )


def _plot_results(
    source_name: str,
    grid_size: int,
    results: dict[str, dict],
    shared_potential_max: float,
) -> None:
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    geometry = _geometry(grid_size)
    fig, axes = plt.subplots(2, 5, figsize=(17.0, 7.4), constrained_layout=True)
    extent = [
        geometry.x_min * 1.0e3,
        geometry.x_max * 1.0e3,
        geometry.y_min * 1.0e3,
        geometry.y_max * 1.0e3,
    ]
    norm = Normalize(vmin=0.0, vmax=shared_potential_max)
    image = None
    for ax, (kind, result) in zip(axes.ravel(), results.items()):
        potential = result["potential"]
        image = ax.imshow(
            potential,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            norm=norm,
            interpolation="nearest",
        )
        positive_max = float(potential.max())
        levels = np.linspace(0.15 * positive_max, 0.9 * positive_max, 6)
        ax.contour(
            geometry.x * 1.0e3,
            geometry.y * 1.0e3,
            potential,
            levels=levels,
            colors="white",
            linewidths=0.45,
        )
        ax.contour(
            geometry.x * 1.0e3,
            geometry.y * 1.0e3,
            result["interior_mask"].astype(float),
            levels=[0.5],
            colors="red",
            linestyles="--",
            linewidths=0.65,
            zorder=3.0,
        )
        _plot_continuous_boundary(ax, kind, APERTURE_CASES[kind])
        ax.set(title=kind, xlabel="x (mm)", ylabel="y (mm)")

    assert image is not None
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.82,
        pad=0.012,
        label="potential (V m); common scale for all grids and apertures",
    )
    fig.legend(
        handles=[
            Line2D(
                [0], [0], color="black", linewidth=1.0,
                label="continuous conductor boundary"
            ),
            Line2D(
                [0], [0], color="red", linestyle="--", linewidth=0.65,
                label="discrete FD active-node boundary"
            ),
            Line2D(
                [0], [0], color="white", linewidth=0.8,
                label="potential contours"
            ),
        ],
        loc="lower center",
        ncol=3,
        fontsize=8,
        framealpha=0.95,
    )
    fig.suptitle(
        f"FD potential: {SOURCE_LABELS[source_name]} across all aperture geometries\n"
        f"{grid_size} x {grid_size} nodes, dx={geometry.dx * 1.0e3:.3f} mm, "
        f"dy={geometry.dy * 1.0e3:.3f} mm"
    )
    output_path = (
        figures_dir / f"{source_name}_fd_aperture_potential_grid_{grid_size}.png"
    )
    fig.savefig(output_path, dpi=170)
    if grid_size == REFERENCE_GRID_SIZE:
        # Keep the established filename current while explicitly named files
        # make the three-resolution sequence unambiguous.
        fig.savefig(
            figures_dir / f"{source_name}_fd_aperture_potential_matrix.png",
            dpi=170,
        )
    plt.close(fig)


def _relative_l2(measured: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    measured_values = measured[mask]
    reference_values = reference[mask]
    denominator = float(np.linalg.norm(reference_values))
    assert denominator > 0.0
    return float(np.linalg.norm(measured_values - reference_values) / denominator)


def _convergence_rows(results_by_grid: dict[int, dict]) -> list[dict]:
    reference_results = results_by_grid[REFERENCE_GRID_SIZE]
    rows = []
    for grid_size in GRID_SIZES:
        geometry = _geometry(grid_size)
        stride = (REFERENCE_GRID_SIZE - 1) // (grid_size - 1)
        assert stride * (grid_size - 1) == REFERENCE_GRID_SIZE - 1
        for source_name in SOURCE_LABELS:
            for kind in APERTURE_CASES:
                measured = results_by_grid[grid_size][source_name][kind]
                reference = reference_results[source_name][kind]
                mask = measured["interior_mask"]
                sampled_reference = {
                    key: reference[key][::stride, ::stride]
                    for key in ("potential", "integrated_ex", "integrated_ey")
                }
                potential_error = _relative_l2(
                    measured["potential"], sampled_reference["potential"], mask
                )
                ex_error = _relative_l2(
                    measured["integrated_ex"], sampled_reference["integrated_ex"], mask
                )
                ey_error = _relative_l2(
                    measured["integrated_ey"], sampled_reference["integrated_ey"], mask
                )
                measured_field = np.stack(
                    (measured["integrated_ex"][mask], measured["integrated_ey"][mask])
                )
                reference_field = np.stack(
                    (
                        sampled_reference["integrated_ex"][mask],
                        sampled_reference["integrated_ey"][mask],
                    )
                )
                field_error = float(
                    np.linalg.norm(measured_field - reference_field)
                    / np.linalg.norm(reference_field)
                )
                rows.append(
                    {
                        "source": source_name,
                        "aperture": kind,
                        "grid_size": grid_size,
                        "dx_m": geometry.dx,
                        "dy_m": geometry.dy,
                        "reference_grid_size": REFERENCE_GRID_SIZE,
                        "potential_relative_l2_vs_257": potential_error,
                        "ex_relative_l2_vs_257": ex_error,
                        "ey_relative_l2_vs_257": ey_error,
                        "field_vector_relative_l2_vs_257": field_error,
                    }
                )
    return rows


def _write_convergence_csv(rows: list[dict]) -> Path:
    analysis_dir = OUTPUT_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    path = analysis_dir / "grid_convergence.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _plot_convergence(rows: list[dict]) -> None:
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    error_columns = {
        "potential_relative_l2_vs_257": "potential",
        "ex_relative_l2_vs_257": "Ex",
        "ey_relative_l2_vs_257": "Ey",
        "field_vector_relative_l2_vs_257": "E vector",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, source_name in zip(axes, SOURCE_LABELS):
        source_rows = [row for row in rows if row["source"] == source_name]
        for column, label in error_columns.items():
            worst_by_grid = []
            for grid_size in GRID_SIZES[:-1]:
                values = [
                    row[column]
                    for row in source_rows
                    if row["grid_size"] == grid_size
                ]
                worst_by_grid.append(max(values))
            ax.semilogy(
                GRID_SIZES[:-1],
                worst_by_grid,
                "o-",
                linewidth=1.0,
                markersize=4.0,
                label=label,
            )
        ax.text(
            0.98,
            0.04,
            "257 x 257 is the reference;\nits self-error is zero and is not plotted",
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=7,
            color="0.35",
        )
        ax.set(
            xlabel="grid nodes per axis",
            ylabel="worst relative L2 error across apertures",
            title=SOURCE_LABELS[source_name],
            xticks=GRID_SIZES[:-1],
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Shortley-Weller FD grid convergence against the 257 x 257 result"
    )
    fig.savefig(
        figures_dir / "fd_all_apertures_grid_convergence_relative_l2.png",
        dpi=170,
    )
    plt.close(fig)


def test_gaussian_and_kv_uniform_sources_support_every_tracking_aperture_geometry():
    containment = _assert_source_envelope_is_contained()
    results_by_grid: dict[int, dict] = {}
    summary = {
        "grid_sizes": list(GRID_SIZES),
        "reference_grid_size": REFERENCE_GRID_SIZE,
        "grid_bounds_m": list(GRID_BOUNDS_M),
        "total_source_charge_c": TOTAL_SOURCE_CHARGE_C,
        "source_envelope_semi_axes_m": list(SOURCE_ENVELOPE_SEMI_AXES_M),
        "gaussian_sigma_m": list(GAUSSIAN_SIGMA_M),
        "gaussian_envelope_sigmas": GAUSSIAN_ENVELOPE_SIGMAS,
        "kv_semi_axes_m": list(KV_SEMI_AXES_M),
        "source_envelope_inside_every_aperture": containment,
        "resolutions": {},
    }

    for grid_size in GRID_SIZES:
        geometry = _geometry(grid_size)
        source_shapes = _source_shapes(geometry)
        results_by_grid[grid_size] = {
            source_name: {} for source_name in source_shapes
        }
        resolution_summary = {
            "grid_points": [geometry.nx, geometry.ny],
            "dx_m": geometry.dx,
            "dy_m": geometry.dy,
            "sources": {
                source_name: {"apertures": {}} for source_name in source_shapes
            },
        }
        summary["resolutions"][str(grid_size)] = resolution_summary

        for kind, values in APERTURE_CASES.items():
            aperture = {"Type": kind, "Aperture Value": values}
            # Factor the geometry matrix once and solve both sources as a batch.
            resources = build_pic_resources(
                geometry, aperture=aperture, field_solver="fd"
            )
            expected_mask = _tracking_mask(geometry, kind, values)
            np.testing.assert_array_equal(
                resources.aperture_mask,
                expected_mask,
                err_msg=(
                    f"FD and tracking aperture masks differ for {kind} on {grid_size}"
                ),
            )
            interior = resources.field_solver.interior_mask
            assert int(interior.sum()) > 100, (
                f"{kind} aperture has too few active FD nodes on {grid_size}"
            )

            densities = []
            retained_fractions = {}
            for source_name, raw_density in source_shapes.items():
                density, retained_fraction = _normalized_density(
                    raw_density, interior, geometry
                )
                densities.append(density)
                retained_fractions[source_name] = retained_fraction
            density_stack = np.stack(densities)
            solved = resources.field_solver.solve(density_stack)
            inactive = ~interior
            indices = resources.field_solver.interior_indices

            for source_index, source_name in enumerate(source_shapes):
                density = density_stack[source_index]
                potential = solved.potential[source_index]
                integrated_ex = solved.integrated_ex[source_index]
                integrated_ey = solved.integrated_ey[source_index]
                outside_max = float(
                    max(
                        np.max(np.abs(potential[inactive])),
                        np.max(np.abs(integrated_ex[inactive])),
                        np.max(np.abs(integrated_ey[inactive])),
                    )
                )
                assert outside_max == 0.0, (
                    f"{source_name}/{kind}/{grid_size} FD solution is nonzero outside "
                    f"its active domain: measured={outside_max:.8g}, theory=0, "
                    "tolerance=0"
                )
                potential_vector = potential.ravel()[indices]
                rhs = density.ravel()[indices] / const.epsilon0
                algebraic_residual = float(
                    np.linalg.norm(
                        resources.field_solver.matrix @ potential_vector - rhs
                    )
                    / np.linalg.norm(rhs)
                )
                assert algebraic_residual < 2.0e-11, (
                    f"{source_name}/{kind}/{grid_size} FD residual failed: "
                    f"measured={algebraic_residual:.8g}, theory=0, tolerance=2e-11"
                )
                integrated_charge = float(
                    density.sum() * geometry.dx * geometry.dy
                )
                np.testing.assert_allclose(
                    integrated_charge,
                    TOTAL_SOURCE_CHARGE_C,
                    rtol=5.0e-14,
                    atol=0.0,
                    err_msg=(
                        "source charge normalization failed for "
                        f"{source_name}/{kind}/{grid_size}"
                    ),
                )
                results_by_grid[grid_size][source_name][kind] = {
                    "interior_mask": interior,
                    "potential": potential,
                    "integrated_ex": integrated_ex,
                    "integrated_ey": integrated_ey,
                }
                resolution_summary["sources"][source_name]["apertures"][kind] = {
                    "aperture_value": values,
                    "mask_matches_tracking": True,
                    "source_envelope_inside_aperture": containment[kind],
                    "active_node_count": int(interior.sum()),
                    "retained_raw_source_fraction_before_charge_normalization": (
                        retained_fractions[source_name]
                    ),
                    "integrated_source_charge_c": integrated_charge,
                    "outside_max_abs": outside_max,
                    "linear_system_relative_l2_residual": algebraic_residual,
                    "peak_potential_v_m": float(potential.max()),
                    "peak_field_v": float(
                        np.hypot(integrated_ex, integrated_ey).max()
                    ),
                }

    convergence_rows = _convergence_rows(results_by_grid)
    indexed_rows = {
        (row["source"], row["aperture"], row["grid_size"]): row
        for row in convergence_rows
    }
    for source_name in SOURCE_LABELS:
        for kind in APERTURE_CASES:
            coarse = indexed_rows[(source_name, kind, 65)]
            medium = indexed_rows[(source_name, kind, 129)]
            assert (
                medium["potential_relative_l2_vs_257"]
                < coarse["potential_relative_l2_vs_257"]
            ), (
                f"potential did not converge monotonically for {source_name}/{kind}: "
                f"65={coarse['potential_relative_l2_vs_257']:.8g}, "
                f"129={medium['potential_relative_l2_vs_257']:.8g}"
            )

    summary["convergence"] = convergence_rows
    analysis_dir = OUTPUT_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_convergence_csv(convergence_rows)

    shared_limits = {
        source_name: max(
            float(
                results_by_grid[grid_size][source_name][kind]["potential"].max()
            )
            for grid_size in GRID_SIZES
            for kind in APERTURE_CASES
        )
        for source_name in SOURCE_LABELS
    }
    for grid_size in GRID_SIZES:
        for source_name in SOURCE_LABELS:
            _plot_results(
                source_name,
                grid_size,
                results_by_grid[grid_size][source_name],
                shared_limits[source_name],
            )
    _plot_convergence(convergence_rows)
