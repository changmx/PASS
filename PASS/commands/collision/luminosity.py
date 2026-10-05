"""Collision-plane density overlap and small, append-only luminosity tables.

Overlap uses real particle counts, independently of their electric charge.
The caller owns the collision clock, sampling schedule and source snapshots.
"""

from pathlib import Path

import numpy as np

from PASS.commands.solver.pic import GridGeometry, build_pic_resources, build_pic_resources_gpu, deposit_particles, deposit_particles_gpu
from PASS.utils.table_io import append_table, normalize_output_format, write_table


class LuminosityCalculator:
    """Estimate pre-kick thin-slice overlap in the common collision frame.

    The result is a single-encounter overlap in m^-2, without frequency.
    Gaussian sources use their configured moment closure. Particle/analytic
    pairs use particle samples, and particle pairs use common-grid density.
    Non-Gaussian analytic pairs use deterministic polar density quadrature.
    """

    def __init__(self, xp=np, dtype=np.float64, *, quadrature_order=128):
        if isinstance(quadrature_order, bool) or int(quadrature_order) != quadrature_order or quadrature_order < 16:
            raise ValueError("Luminosity quadrature order must be an integer >=16")
        self.xp = xp
        self.dtype = np.dtype(dtype)
        self.quadrature_order = int(quadrature_order)
        self._resources = {}
        self._quadrature = None
        self._angles = None

    def _validate(self, invalid, error_flags):
        invalid = self.xp.any(invalid)
        if error_flags is None:
            if bool(invalid):
                raise ValueError("Luminosity requires finite positive source covariance and fully covered particle stencils")
        else:
            error_flags[...] |= invalid.astype(self.xp.int32)

    def _moments(self, source, distance, error_flags):
        center, covariance = source.moments(distance)
        center = self.xp.asarray(center, dtype=self.xp.float64)
        covariance = self.xp.asarray(covariance, dtype=self.xp.float64)
        xx, xy, yy = covariance
        determinant = xx * yy - xy * xy
        self._validate(~self.xp.all(self.xp.isfinite(center)) | ~self.xp.all(self.xp.isfinite(covariance)) | (xx <= 0)
                       | (determinant <= 0), error_flags)
        return center, covariance

    def _density(self, x, y, center, covariance, solver):
        xp = self.xp
        xx, xy, yy = covariance
        determinant = xx * yy - xy * xy
        dx, dy = x - center[0], y - center[1]
        radius_squared = (yy * dx * dx - 2 * xy * dx * dy + xx * dy * dy) / determinant
        scale = np.pi * xp.sqrt(determinant)
        if solver.startswith("gaussian_"):
            return xp.exp(-radius_squared / 2) / (2 * scale)
        if solver.startswith("uniform_"):
            return xp.where(radius_squared < 4, 1 / (4 * scale), 0)
        if solver.startswith("parabolic_"):
            return xp.maximum(1 - radius_squared / 6, 0) / (3 * scale)
        raise ValueError(f"Unsupported luminosity density profile {solver!r}")

    def _integrate_density(self, sampling, evaluated):
        xp = self.xp
        if self._quadrature is None:
            nodes, weights = np.polynomial.legendre.leggauss(self.quadrature_order)
            angle = np.arange(4 * self.quadrature_order) * (2 * np.pi / (4 * self.quadrature_order))
            self._quadrature = tuple(
                xp.asarray(value, dtype=xp.float64) for value in ((nodes + 1) / 2, weights / (2 * angle.size), np.cos(angle), np.sin(angle)))
        probability, weights, cosine, sine = self._quadrature
        center, covariance, solver = sampling
        if solver.startswith("gaussian_"):
            radius = xp.sqrt(-2 * xp.log(probability))
        elif solver.startswith("uniform_"):
            radius = 2 * xp.sqrt(probability)
        else:
            radius = xp.sqrt(6 * probability / (1 + xp.sqrt(1 - probability)))
        xx, xy, yy = covariance
        a = xp.sqrt(xx)
        b = xy / a
        c = xp.sqrt(yy - b * b)
        u, v = radius[:, None] * cosine[None, :], radius[:, None] * sine[None, :]
        density = self._density(center[0] + a * u, center[1] + b * u + c * v, *evaluated)
        return xp.sum(density * weights[:, None], dtype=xp.float64)

    def _analytic_overlap(self, source_a, source_b, distance, error_flags):
        xp = self.xp
        center_a, covariance_a = self._moments(source_a, -distance, error_flags)
        center_b, covariance_b = self._moments(source_b, distance, error_flags)
        if source_a.solver.startswith("gaussian_") and source_b.solver.startswith("gaussian_"):
            covariance = covariance_a + covariance_b
            return self._density(center_a[0], center_a[1], center_b, covariance, "gaussian_")
        a = center_a, covariance_a, source_a.solver
        b = center_b, covariance_b, source_b.solver
        # Integrating a bounded profile against a Gaussian avoids its hard edge
        # in the integrand. Two bounded profiles use symmetric quadrature.
        if source_a.solver.startswith("gaussian_"):
            return self._integrate_density(b, a)
        if source_b.solver.startswith("gaussian_"):
            return self._integrate_density(a, b)
        return (self._integrate_bounded_density(a, b) + self._integrate_bounded_density(b, a)) / xp.float64(2)

    def _integrate_bounded_density(self, sampling, evaluated):
        """Clip each radial ray to both ellipse supports and integrate it exactly."""
        xp = self.xp
        if self._angles is None:
            angle = (np.arange(8 * self.quadrature_order) + .5) * (2 * np.pi / (8 * self.quadrature_order))
            self._angles = tuple(xp.asarray(value, dtype=xp.float64) for value in (np.cos(angle), np.sin(angle)))
        cosine, sine = self._angles
        center_a, covariance_a, solver_a = sampling
        center_b, covariance_b, solver_b = evaluated
        parabolic_a, parabolic_b = solver_a.startswith("parabolic_"), solver_b.startswith("parabolic_")
        factor_a, factor_b = (6 if parabolic_a else 4), (6 if parabolic_b else 4)
        xx_a, xy_a, yy_a = covariance_a * factor_a
        xx_b, xy_b, yy_b = covariance_b * factor_b
        determinant_b = xx_b * yy_b - xy_b * xy_b
        a = xp.sqrt(xx_a)
        b = xy_a / a
        dx, dy = a * cosine, b * cosine + xp.sqrt(yy_a - b * b) * sine
        vx, vy = center_a - center_b
        # q_B(r)=a*r^2+b*r+c; its roots bound the part inside ellipse B.
        quadratic = (yy_b * dx * dx - 2 * xy_b * dx * dy + xx_b * dy * dy) / determinant_b
        linear = 2 * (yy_b * vx * dx - xy_b * (vx * dy + vy * dx) + xx_b * vy * dy) / determinant_b
        constant = (yy_b * vx * vx - 2 * xy_b * vx * vy + xx_b * vy * vy) / determinant_b
        discriminant = linear * linear - 4 * quadratic * (constant - 1)
        root = xp.sqrt(xp.maximum(discriminant, 0))
        lower = xp.clip((-linear - root) / (2 * quadratic), 0, 1)
        upper = xp.clip((-linear + root) / (2 * quadratic), 0, 1)

        def primitive(radius):
            if parabolic_b:
                value = (1 - constant) * radius**2 / 2 - linear * radius**3 / 3 - quadratic * radius**4 / 4
                if parabolic_a:
                    value -= (1 - constant) * radius**4 / 4 - linear * radius**5 / 5 - quadratic * radius**6 / 6
                return value
            return radius**2 / 2 - (radius**4 / 4 if parabolic_a else 0)

        radial = xp.where((discriminant > 0) & (upper > lower), xp.maximum(primitive(upper) - primitive(lower), 0), 0)
        normalization = (2 if parabolic_a else 1) * (2 if parabolic_b else 1) / (np.pi**2 * xp.sqrt(determinant_b))
        return normalization * xp.sum(radial, dtype=xp.float64) * (2 * np.pi / cosine.size)

    def _common_grid(self, source_a, source_b):
        grids = [source.grid for source in (source_a, source_b) if source is not None and hasattr(source, "coordinates")]
        if not grids:
            raise ValueError("Particle-particle luminosity requires at least one configured PIC source grid")
        x_min, x_max = min(grid.x_min for grid in grids), max(grid.x_max for grid in grids)
        y_min, y_max = min(grid.y_min for grid in grids), max(grid.y_max for grid in grids)
        dx, dy = min(grid.dx for grid in grids), min(grid.dy for grid in grids)
        nx, ny = int(np.ceil((x_max - x_min) / dx)) + 1, int(np.ceil((y_max - y_min) / dy)) + 1
        return GridGeometry(nx, ny, x_min, x_max, y_min, y_max)

    def _particle_overlap(self, source_a, source_b, coordinates_a, coordinates_b, distance, number_a, number_b, error_flags):
        xp = self.xp
        grid = self._common_grid(source_a, source_b)
        method = "TSC" if any(source.method == "TSC" for source in (source_a, source_b)
                              if source is not None and hasattr(source, "coordinates")) else "CIC"
        x_a, y_a = coordinates_a[0] + distance * coordinates_a[1], coordinates_a[2] + distance * coordinates_a[3]
        x_b, y_b = coordinates_b[0] - distance * coordinates_b[1], coordinates_b[2] - distance * coordinates_b[3]
        x = xp.concatenate((x_a, x_b)).astype(xp.float64, copy=False)
        y = xp.concatenate((y_a, y_b)).astype(xp.float64, copy=False)
        margin = .5 if method == "TSC" else 0
        self._validate(
            ~xp.isfinite(x) | ~xp.isfinite(y) | (x < grid.x_min + margin * grid.dx) | (x > grid.x_max - margin * grid.dx)
            | (y < grid.y_min + margin * grid.dy) | (y > grid.y_max - margin * grid.dy), error_flags)
        n_a, n_b = coordinates_a.shape[1], coordinates_b.shape[1]
        indices = xp.concatenate((xp.zeros(n_a, dtype=xp.int64), xp.ones(n_b, dtype=xp.int64)))
        weights = xp.concatenate((xp.full(n_a, number_a / n_a, dtype=xp.float64), xp.full(n_b, number_b / n_b, dtype=xp.float64)))
        if grid not in self._resources:
            build_resources = build_pic_resources if xp is np else build_pic_resources_gpu
            self._resources[grid] = build_resources(grid, field_solver="fft_free_space", dtype=np.float64)
        if xp is np:
            density = deposit_particles({
                "x": x,
                "y": y
            },
                                        indices,
                                        grid,
                                        resources=self._resources[grid],
                                        method=method,
                                        charge_per_macro=weights,
                                        num_slices=2,
                                        dtype=np.float64).density
        else:
            density = deposit_particles_gpu({
                "x": x,
                "y": y
            },
                                            indices,
                                            grid,
                                            resources=self._resources[grid],
                                            method=method,
                                            charge_per_macro=weights,
                                            num_slices=2,
                                            validate=False,
                                            copy=False).density
        return xp.sum(density[0] * density[1], dtype=xp.float64) * (grid.dx * grid.dy)

    def overlap(self, source_a, source_b, distance, number_a, number_b, *, coordinates_a=None, coordinates_b=None, error_flags=None):
        """Sum one slice pair, with side A drifting +S and side B -S.

        Analytic sources expose ``moments(distance)`` using source drift -S.
        Particle arrays have shape (4,N) in common transverse coordinates.
        Counts are true particle numbers, never charge/e for multiply charged
        ions. An analytic moment closure is not a non-Gaussian PIC diagnostic.
        """
        xp = self.xp
        if not np.isfinite(number_a) or not np.isfinite(number_b) or min(number_a, number_b) < 0 or not np.isfinite(distance):
            raise ValueError("Luminosity particle counts and collision distance must be finite, with nonnegative counts")
        if number_a == 0 or number_b == 0:
            return xp.asarray(0, dtype=xp.float64)
        analytic_a = source_a is not None and hasattr(source_a, "moments")
        analytic_b = source_b is not None and hasattr(source_b, "moments")
        if analytic_a and analytic_b:
            value = self._analytic_overlap(source_a, source_b, distance, error_flags) * (number_a * number_b)
        else:
            if not analytic_a:
                coordinates_a = getattr(source_a, "coordinates", coordinates_a)
            if not analytic_b:
                coordinates_b = getattr(source_b, "coordinates", coordinates_b)
            for analytic, coordinates in ((analytic_a, coordinates_a), (analytic_b, coordinates_b)):
                if not analytic and (coordinates is None or coordinates.ndim != 2 or coordinates.shape[0] != 4 or coordinates.shape[1] == 0):
                    raise ValueError("Particle luminosity sources require a nonempty (4,N) common-frame snapshot")
            if analytic_a or analytic_b:
                source = source_a if analytic_a else source_b
                coordinates = coordinates_b if analytic_a else coordinates_a
                drift = -distance if analytic_a else distance
                center, covariance = self._moments(source, drift, error_flags)
                x, y = coordinates[0] + drift * coordinates[1], coordinates[2] + drift * coordinates[3]
                values = self._density(x.astype(xp.float64), y.astype(xp.float64), center, covariance, source.solver)
                value = xp.sum(values, dtype=xp.float64) * (number_a * number_b / coordinates.shape[1])
            else:
                value = self._particle_overlap(source_a, source_b, coordinates_a, coordinates_b, distance, number_a, number_b, error_flags)
        self._validate(~xp.isfinite(value) | (value < 0), error_flags)
        return value

    def close(self):
        for resources in self._resources.values():
            if hasattr(resources, "close"):
                resources.close()
        self._resources.clear()


class LuminosityRecorder:
    """Record sampled encounters, refusing implicit overwrite or rollback.

    Appends are one numeric batch; a failed batch disables this instance
    instead of risking duplicate or partial output on retry.
    """

    def __init__(self, path, *, interval=100, reference_luminosity=None, metadata=None, output_format="tfs"):
        if isinstance(interval, bool) or int(interval) != interval or interval < 1:
            raise ValueError("Luminosity interval must be a positive integer")
        if reference_luminosity is not None and (not np.isfinite(reference_luminosity) or reference_luminosity <= 0):
            raise ValueError("Explicit luminosity reference must be finite and positive")
        self.path = Path(path)
        self.output_format = normalize_output_format(output_format)
        suffixes = (".tfs", ) if self.output_format == "tfs" else (".h5", ".hdf5")
        if self.path.suffix.lower() not in suffixes:
            raise ValueError("Luminosity output format must match its table file extension")
        self.interval = int(interval)
        self.reference_luminosity = reference_luminosity
        self.metadata = dict(metadata or {})
        self.references = {}
        self.last_turn = -1
        self._failed = False
        self.columns = ("TURN", "TIME", "BUNCH_A", "BUNCH_B", "FREQUENCY_HZ", "N_A", "N_B", "OVERLAP_M2", "LUMINOSITY", "L_REFERENCE", "FACTOR",
                        "LOSS")

    def should_sample(self, turn):
        return self.last_turn < 0 or (int(turn) + 1) % self.interval == 0

    def _headers(self):
        return {
            **self.metadata, "Name": "PASS BeamBeam Luminosity",
            "Reference": "initial_collision" if self.reference_luminosity is None else "explicit_per_pair",
            "LuminosityUnit": "cm^-2 s^-1",
            "OverlapUnit": "m^-2",
            "TimeUnit": "s",
            "Sampling": "sampled_turn",
            "Interval": self.interval,
            "TotalRow": "BUNCH_A=BUNCH_B=-1; sums distinct bunch pairs at this IP",
            "AccumulationPrecision": "float64"
        }

    def record(self, turn, pair_ids, overlaps, times, frequencies, *, populations=None, xp=np):
        if self._failed:
            raise RuntimeError("Luminosity output previously failed; this batch cannot be retried")
        if isinstance(turn, bool) or int(turn) != turn or turn <= self.last_turn or turn < 0:
            raise ValueError("Luminosity turn must increase strictly without duplicate records")
        pairs = [tuple(pair) for pair in pair_ids]
        if not pairs or len(set(pairs)) != len(pairs) or any(
                len(pair) != 2 or any(isinstance(value, bool) or int(value) != value or value < 0 for value in pair) for pair in pairs):
            raise ValueError("Luminosity requires unique nonnegative integer bunch pairs")
        n_pairs = len(pairs)
        values = xp.asarray(overlaps, dtype=xp.float64).reshape(-1)
        if values.size != n_pairs:
            raise ValueError("Luminosity overlaps must have one value per bunch pair")
        times = np.broadcast_to(np.asarray(times, dtype=float), (n_pairs, ))
        frequencies = np.broadcast_to(np.asarray(frequencies, dtype=float), (n_pairs, ))
        populations = np.full((n_pairs, 2), np.nan) if populations is None else np.asarray(populations, dtype=float)
        if populations.shape != (n_pairs, 2) or (not np.isnan(populations).all() and (not np.isfinite(populations).all() or np.any(populations < 0))):
            raise ValueError("Luminosity populations must be finite nonnegative (pairs,2) values")
        if not np.isfinite(times).all() or not np.isfinite(frequencies).all() or np.any(frequencies <= 0):
            raise ValueError("Luminosity times and positive frequencies must be finite")
        # One compact device-to-host transfer for all pairs of this encounter.
        values = np.asarray(values) if xp is np else xp.asnumpy(values)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Luminosity overlaps must be finite and nonnegative")
        luminosities = values * frequencies * 1e-4
        if not np.isfinite(luminosities).all():
            raise ValueError("Luminosity frequency normalization overflowed")
        keys = [f"{int(a)}:{int(b)}" for a, b in pairs]
        if self.references and set(keys) != set(self.references):
            raise ValueError("Luminosity bunch pairing changed after the reference encounter")
        references = dict(self.references) or dict(
            zip(keys, luminosities if self.reference_luminosity is None else np.full(n_pairs, self.reference_luminosity)))
        rows = []
        for index, (pair, key) in enumerate(zip(pairs, keys)):
            reference = references[key]
            factor = luminosities[index] / reference if reference > 0 else np.nan
            rows.append((int(turn), times[index], *map(int, pair), frequencies[index], *populations[index], values[index], luminosities[index],
                         reference, factor, 1 - factor))
        if n_pairs > 1:
            luminosity, reference = float(luminosities.sum()), float(sum(references.values()))
            factor = luminosity / reference if reference > 0 else np.nan
            rows.append((int(turn), float(times.max()), -1, -1, np.nan, *populations.sum(axis=0), float(values.sum()), luminosity, reference, factor,
                         1 - factor))
        frame = {
            name: np.asarray([row[index] for row in rows], dtype=np.int64 if name in {"TURN", "BUNCH_A", "BUNCH_B"} else np.float64)
            for index, name in enumerate(self.columns)
        }
        if self.last_turn < 0 and self.path.exists():
            raise FileExistsError(f"Refusing to overwrite existing luminosity output: {self.path}")
        if self.last_turn >= 0 and not self.path.is_file():
            raise FileNotFoundError(f"Luminosity history disappeared: {self.path}")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.output_format != "tfs":
                append_table(self.path, frame, self._headers(), chunk_rows=max(128, len(rows)), output_format=self.output_format)
            elif self.last_turn < 0:
                write_table(self.path, frame, self._headers(), output_format="tfs", colwidth=24, headerswidth=24)
            else:
                with self.path.open("a", encoding="utf-8", newline="") as stream:
                    for row in rows:
                        stream.write(" ".join(str(value) if index in (0, 2, 3) else format(float(value), ".17g")
                                              for index, value in enumerate(row)) + "\n")
        except BaseException:
            self._failed = True
            raise
        self.references = {key: float(value) for key, value in references.items()}
        self.last_turn = int(turn)
        return rows

    def close(self):
        """Writes complete at record time; no unsaved device or text buffer."""
