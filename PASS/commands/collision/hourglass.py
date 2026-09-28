"""Internal source propagation for beam-beam hourglass calculations.

Prepared common-frame sources drift along -S. Their potential derivatives hold
collision-plane x and y fixed; interaction.py owns the target map and kick.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from PASS.commands.solver.analytic import evaluate_potential_jet
from PASS.commands.solver.field_result import launch_gpu_kernel
from PASS.commands.solver.pic import deposit_source_density_jet, deposit_source_pair, gather_linear_potential_jet


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


@dataclass
class PICSource:
    coordinates: object
    charge_per_macro: float
    grid: object
    method: str
    resources: object
    pic: object

    def _solve_endpoints(self, endpoints):
        xp = self.resources.xp
        x, px, y, py = self.coordinates
        if endpoints[0] == endpoints[1]:
            # The derivative of a zero-width source interval need not vanish.
            x = xp.asarray(x, dtype=xp.float64) - float(endpoints[0]) * xp.asarray(px, dtype=xp.float64)
            y = xp.asarray(y, dtype=xp.float64) - float(endpoints[0]) * xp.asarray(py, dtype=xp.float64)
            density = deposit_source_density_jet(x,
                                                 y,
                                                 -px,
                                                 -py,
                                                 self.grid,
                                                 charge_per_macro=self.charge_per_macro,
                                                 method=self.method,
                                                 xp=xp,
                                                 error_flags=self.resources.error_flags)
        else:
            # Form each particle's endpoint difference before grid accumulation.
            density = deposit_source_pair(x,
                                          px,
                                          y,
                                          py,
                                          float(endpoints[0]),
                                          float(endpoints[1]),
                                          self.grid,
                                          charge_per_macro=self.charge_per_macro,
                                          method=self.method,
                                          xp=xp,
                                          error_flags=self.resources.error_flags)
        if xp is np:
            return self.pic.field_solver.solve(density, compute_fields=False).potential
        return self.pic.field_solver.solve(density, compute_fields=False, validate=False, copy=False).potential

    def evaluate(self, x, y, distance, *, endpoints):
        xp, dtype = self.resources.xp, self.resources.dtype
        x, y, distance = xp.broadcast_arrays(xp.asarray(x, dtype=dtype), xp.asarray(y, dtype=dtype), xp.asarray(distance, dtype=dtype))
        shape = x.shape
        if not x.size:
            return None, *(xp.empty(shape, dtype=dtype) for _ in range(3))
        # Bounds come from one compact SliceSet schedule, never a GPU reduction
        # for each slice pair. Cast before subtracting to match tracked S.
        endpoints = np.asarray(endpoints, dtype=dtype)
        if endpoints.shape != (2, ) or not np.all(np.isfinite(endpoints)) or endpoints[1] < endpoints[0]:
            raise ValueError("BeamBeam requires finite ordered actual slice endpoints")
        width = dtype.type(endpoints[1] - endpoints[0])
        if not np.isfinite(width):
            raise ValueError("BeamBeam actual slice span is not representable")
        if width == 0:
            invalid = xp.any(~xp.isfinite(distance) | (distance != endpoints[0]))
            if xp is np and bool(invalid):
                raise ValueError("BeamBeam target distances exceed its zero-width slice")
            self.resources.error_flags[...] |= invalid.astype(xp.int32)
            weight, denominator = xp.zeros(shape, dtype=dtype), dtype.type(1)
        else:
            weight, denominator = (distance - endpoints[0]) / width, width
        potential = self._solve_endpoints(endpoints)
        return gather_linear_potential_jet(potential,
                                           x,
                                           y,
                                           weight,
                                           denominator,
                                           self.grid,
                                           method=self.method,
                                           xp=xp,
                                           error_flags=self.resources.error_flags,
                                           compute_potential=False)


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
