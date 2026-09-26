"""Internal source propagation for beam-beam hourglass calculations.

Prepared common-frame sources drift along -S. Their potential derivatives hold
collision-plane x and y fixed; interaction.py owns the target map and kick.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from PASS.commands.solver.analytic import evaluate_potential_jet
from PASS.commands.solver.field_result import launch_gpu_kernel
from PASS.commands.solver.pic import deposit_particles, deposit_particles_gpu, gather_cubic_potential_jet


@dataclass
class AnalyticSource:
    mean: object
    covariance: object
    charge: float
    solver: str
    resources: object
    reference_length: float = 1.0

    def _propagate_moments(self, distance):
        xp, dtype = self.resources.xp, self.resources.dtype
        distance = xp.asarray(distance, dtype=dtype)
        if xp is not np:
            center = xp.empty(distance.shape + (2, ), dtype=dtype)
            covariance = xp.empty(distance.shape + (3, ), dtype=dtype)
            center_derivative = xp.empty(2, dtype=dtype)
            derivative = xp.empty_like(covariance)
            launch_gpu_kernel(
                _PROPAGATION_CUDA, "collision_propagate_moments", max(distance.size, 1),
                (xp.ascontiguousarray(distance), xp.ascontiguousarray(self.mean, dtype=dtype), xp.ascontiguousarray(self.covariance, dtype=dtype),
                 np.int32("_round_" in self.solver), center, covariance, center_derivative, derivative, np.int64(distance.size)), dtype)
            return center, covariance, center_derivative, derivative
        mean, c = self.mean, self.covariance
        center = xp.stack((mean[0] - distance * mean[1], mean[2] - distance * mean[3]), axis=-1)
        xx = c[0, 0] - 2 * distance * c[0, 1] + distance**2 * c[1, 1]
        xy = c[0, 2] - distance * (c[0, 3] + c[1, 2]) + distance**2 * c[1, 3]
        yy = c[2, 2] - 2 * distance * c[2, 3] + distance**2 * c[3, 3]
        dxx = -2 * c[0, 1] + 2 * distance * c[1, 1]
        dxy = -(c[0, 3] + c[1, 2]) + 2 * distance * c[1, 3]
        dyy = -2 * c[2, 3] + 2 * distance * c[3, 3]
        if "_round_" in self.solver:
            xx = yy = (xx + yy) / 2
            dxx = dyy = (dxx + dyy) / 2
            xy, dxy = xp.zeros_like(xx), xp.zeros_like(xx)
        covariance = xp.stack((xx, xy, yy), axis=-1)
        derivative = xp.stack((dxx, dxy, dyy), axis=-1)
        center_derivative = xp.stack((-mean[1], -mean[3]))
        return center, covariance, center_derivative, derivative

    def moments(self, distance):
        """Return center and packed covariance (xx, xy, yy) after a -S drift."""
        center, covariance, _, _ = self._propagate_moments(distance)
        return center, covariance

    def evaluate(self, x, y, distance):
        center, covariance, center_derivative, derivative = self._propagate_moments(distance)
        return evaluate_potential_jet(x, y, self.charge, center, covariance, center_derivative, derivative, self.solver, self.resources.quadrature,
                                      self.reference_length)


_PROPAGATION_CUDA = r"""
extern "C" __global__ void collision_propagate_moments(
    const T* distance,
    const T* mean,
    const T* c,
    int round_source,
    T* center,
    T* covariance,
    T* center_derivative,
    T* derivative,
    long long n
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        center_derivative[0] = -mean[1];
        center_derivative[1] = -mean[3];
    }
    if (i >= n)
        return;
    T s = distance[i], squared = s * s;
    center[2 * i] = mean[0] - s * mean[1];
    center[2 * i + 1] = mean[2] - s * mean[3];
    T xx = c[0] - T(2) * s * c[1] + squared * c[5];
    T xy = c[2] - s * (c[3] + c[6]) + squared * c[7];
    T yy = c[10] - T(2) * s * c[11] + squared * c[15];
    T dxx = -T(2) * c[1] + T(2) * s * c[5];
    T dxy = -(c[3] + c[6]) + T(2) * s * c[7];
    T dyy = -T(2) * c[11] + T(2) * s * c[15];
    if (round_source) {
        xx = yy = (xx + yy) / T(2);
        dxx = dyy = (dxx + dyy) / T(2);
        xy = dxy = 0;
    }
    covariance[3 * i] = xx;
    covariance[3 * i + 1] = xy;
    covariance[3 * i + 2] = yy;
    derivative[3 * i] = dxx;
    derivative[3 * i + 1] = dxy;
    derivative[3 * i + 2] = dyy;
}
"""


def _validate_propagation_step(prescribed_step, resources):
    """Require a manually prescribed S spacing representable in tracking precision."""
    if prescribed_step is None:
        raise ValueError("PIC requires an explicit positive Propagation step (m); automatic step selection is not supported")
    with np.errstate(over="ignore", under="ignore"):
        step = resources.dtype.type(prescribed_step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"Propagation step (m) must remain finite and positive in {resources.dtype.name}")
    return resources.xp.asarray(step, dtype=resources.dtype)


@dataclass
class PICSource:
    coordinates: object
    charge_per_macro: float
    step: object
    grid: object
    method: str
    resources: object
    pic: object

    def _check_coverage(self, distances):
        xp, scalar = self.resources.xp, self.resources.dtype.type
        normalized = distances / self.step
        limit = scalar(2**(20 if self.resources.dtype.itemsize == 4 else 50))
        invalid = xp.any(~xp.isfinite(normalized) | (xp.abs(normalized) > limit))
        self.resources.error_flags[...] |= invalid.astype(xp.int32)
        normalized = xp.nan_to_num(normalized, nan=0., posinf=float(limit), neginf=-float(limit))
        normalized = xp.clip(normalized, -limit, limit)
        intervals = xp.floor(normalized).astype(xp.int64)
        endpoints = xp.stack((intervals.min() - 1, intervals.max() + 2)).astype(self.resources.dtype) * self.step
        x, px, y, py = self.coordinates
        x = x[None, :] - endpoints[:, None] * px[None, :]
        y = y[None, :] - endpoints[:, None] * py[None, :]
        margin = .5 if self.method == "TSC" else 0
        outside = xp.any((x < self.grid.x_min + margin * self.grid.dx) | (x > self.grid.x_max - margin * self.grid.dx)
                         | (y < self.grid.y_min + margin * self.grid.dy) | (y > self.grid.y_max - margin * self.grid.dy)
                         | ~xp.isfinite(x) | ~xp.isfinite(y))
        if xp is np and bool(outside | invalid):
            raise ValueError("BeamBeam propagation grid does not cover complete source stencils or resolvable distances")
        self.resources.error_flags[...] |= outside.astype(xp.int32)
        return intervals, normalized - intervals.astype(self.resources.dtype)

    def _solve_interval(self, interval):
        xp, dtype = self.resources.xp, self.resources.dtype
        # Every interval uses the same absolute S lattice, including its halo.
        nodes = xp.asarray([interval, interval - 1, interval + 1, interval + 2], dtype=dtype) * self.step
        x, px, y, py = self.coordinates
        kwargs = dict(resources=self.pic, method=self.method, charge_per_macro=self.charge_per_macro)
        if xp is np:
            # A plane at a time bounds CPU stencil temporaries without changing
            # the accumulation order within any physical density plane.
            sid = np.zeros(self.coordinates.shape[1], dtype=np.int64)
            density = np.stack([
                deposit_particles({
                    "x": x - node * px,
                    "y": y - node * py
                }, sid, self.grid, dtype=dtype, num_slices=1, **kwargs).density[0] for node in nodes
            ])
        else:
            x = x[None, :] - nodes[:, None] * px[None, :]
            y = y[None, :] - nodes[:, None] * py[None, :]
            sid = xp.repeat(xp.arange(4, dtype=xp.int64), self.coordinates.shape[1])
            density = deposit_particles_gpu({
                "x": x.ravel(),
                "y": y.ravel()
            }, sid, self.grid, num_slices=4, validate=False, copy=False, **kwargs).density
        # Difference densities remove the large common potential before FFT.
        density[1:] -= density[:1]
        if xp is np:
            return self.pic.field_solver.solve(density, compute_fields=False).potential
        # All gathers finish on this stream before the next interval reuses phi.
        return self.pic.field_solver.solve(density, compute_fields=False, validate=False, copy=False).potential

    def evaluate(self, x, y, distance):
        xp, dtype = self.resources.xp, self.resources.dtype
        x, y, distance = xp.broadcast_arrays(xp.asarray(x, dtype=dtype), xp.asarray(y, dtype=dtype), xp.asarray(distance, dtype=dtype))
        shape = x.shape
        x, y, distance = x.ravel(), y.ravel(), distance.ravel()
        if not x.size:
            return tuple(xp.empty(shape, dtype=dtype) for _ in range(4))
        intervals, weight = self._check_coverage(distance)
        occupied = xp.unique(intervals)
        occupied = occupied if xp is np else xp.asnumpy(occupied)
        outputs = tuple(xp.empty(x.size, dtype=dtype) for _ in range(4))
        kwargs = dict(method=self.method, xp=xp, error_flags=self.resources.error_flags)
        for interval in occupied:
            selected = slice(None) if occupied.size == 1 else xp.flatnonzero(intervals == interval)
            potential = self._solve_interval(int(interval))
            values = gather_cubic_potential_jet(potential, x[selected], y[selected], weight[selected], self.step, self.grid, **kwargs)
            for output, value in zip(outputs, values):
                output[selected] = value
        return tuple(output.reshape(shape) for output in outputs)
